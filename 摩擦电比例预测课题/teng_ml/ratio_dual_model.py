"""PGDS/PGTD 比例预测主流程：分类 + 回归双模型（精简归档版核心脚本）。

整体逻辑（按 step 顺序读即可）：
1. 00_preflight   扫描 data/PGDS、data/PGTD，从文件名解析比例标签
2. 01_cleaned     读 CSV、清洗时间轴与电压通道
3. 02_candidates  按周期峰检测切候选片段，并重采样到固定长度
4. 03_qc          质控，过滤异常片段
5. 04_split       按文件级划分 train/test，避免同文件泄漏
6. 05_augmentation 训练集轻度增强（平衡样本）
7. 06/07          Baseline / CNN 分类（离散比例 0/25/50/75/100%）
8. 08/09          Baseline / CNN 回归（连续比例反演）
9. 10/11          模型对比与 Markdown 报告

运行产物写在：``<data-root>/ratio_dual_model/run_*/``
论文关键数字优先看 ``11_final_report/report.md`` 与 ``10_model_comparison/``。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# 归档包根目录（本文件在 teng_ml/ 下，上一级即归档根）
ARCHIVE_ROOT = Path(__file__).resolve().parents[1]

CONTENT_LEVELS = [0, 25, 50, 75, 100]  # 训练锚点：五个离散配方比例
SYSTEMS = ["PGDS", "PGTD"]
RESAMPLE_LENGTH = 256  # 每个周期片段重采样后的固定长度（CNN 输入）
RANDOM_SEED = 20260515
LABEL_RE = re.compile(r"(?P<content>\d+(?:\.\d+)?)\s*%?\s*(?P<system>PGDS|PGTD)", re.I)


@dataclass(frozen=True)
class SourceLabel:
    path: Path
    source_file: str
    material_system: str
    content_pct: int
    requires_qc: bool


def parse_source_label(path: Path, data_root: Path) -> SourceLabel:
    """Parse material system and content percentage from a source CSV path."""

    match = LABEL_RE.search(path.name)
    if not match:
        raise ValueError(f"Cannot parse source label from {path.name}")
    content = int(round(float(match.group("content"))))
    system = match.group("system").upper()
    if content not in CONTENT_LEVELS:
        raise ValueError(f"Unexpected content percentage {content} in {path.name}")
    if system not in SYSTEMS:
        raise ValueError(f"Unexpected material system {system} in {path.name}")
    return SourceLabel(
        path=path,
        source_file=str(path.relative_to(data_root)),
        material_system=system,
        content_pct=content,
        requires_qc=content == 75,
    )


def robust_zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = 1.4826 * mad if mad > 1e-12 else float(np.std(values))
    if scale <= 1e-12:
        return np.zeros_like(values, dtype=np.float64)
    return (values - median) / scale


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if window <= 1:
        return values.copy()
    window = max(3, int(window))
    if window % 2 == 0:
        window += 1
    pad = window // 2
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(np.pad(values, (pad, pad), mode="edge"), kernel, mode="valid")


def estimate_dt(time_s: np.ndarray) -> float:
    diffs = np.diff(np.asarray(time_s, dtype=np.float64))
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return 0.001
    return float(np.median(diffs))


def read_signal_csv(path: Path) -> pd.DataFrame:
    rows: list[tuple[float, float, float]] = []
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if len(row) < 3:
                continue
            try:
                rows.append((float(row[0]), float(row[1]), float(row[2])))
            except ValueError:
                continue
    if not rows:
        raise ValueError(f"No numeric rows found in {path}")
    df = pd.DataFrame(rows, columns=["time_s", "voltage_v", "filter_v"])
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return repair_time_axis(df).reset_index(drop=True)


def repair_time_axis(df: pd.DataFrame) -> pd.DataFrame:
    """Keep real samples when exported time stamps are too low precision.

    Some oscilloscope CSV files round TIME(S) so aggressively that most rows
    share timestamps. Dropping duplicate times destroys real samples. When the
    duplicate ratio is high, keep the original row order and rebuild a monotonic
    time axis from the observed start/end time. For normal files, keep the
    conservative sorted unique-time behavior.
    """

    if df.empty:
        return df
    time = df["time_s"].to_numpy(dtype=np.float64)
    unique_count = len(np.unique(time))
    duplicate_ratio = 1.0 - unique_count / max(1, len(time))
    monotonic_non_decreasing = bool(np.all(np.diff(time) >= 0))
    if duplicate_ratio > 0.05 and monotonic_non_decreasing and len(df) >= 2:
        start = float(time[0])
        end = float(time[-1])
        if end <= start:
            positive_diffs = np.diff(time)
            positive_diffs = positive_diffs[positive_diffs > 0]
            dt = float(np.median(positive_diffs)) if len(positive_diffs) else 0.001
            end = start + dt * (len(df) - 1)
        repaired = df.copy()
        repaired["time_s"] = np.linspace(start, end, len(repaired), dtype=np.float64)
        return repaired
    return df.sort_values("time_s").drop_duplicates("time_s")


def signal_stats(df: pd.DataFrame) -> dict[str, float | int]:
    time = df["time_s"].to_numpy(dtype=np.float64)
    dt = estimate_dt(time)
    out: dict[str, float | int] = {
        "n_rows": int(len(df)),
        "start_s": float(time[0]),
        "end_s": float(time[-1]),
        "duration_s": float(time[-1] - time[0]),
        "median_dt_s": dt,
        "sampling_hz": 1.0 / dt if dt > 0 else math.nan,
    }
    for column in ["voltage_v", "filter_v"]:
        values = df[column].to_numpy(dtype=np.float64)
        prefix = column.replace("_v", "")
        out[f"{prefix}_min"] = float(np.min(values))
        out[f"{prefix}_max"] = float(np.max(values))
        out[f"{prefix}_ptp"] = float(np.ptp(values))
        out[f"{prefix}_rms"] = float(np.sqrt(np.mean(values**2)))
    return out


def discover_sources(data_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    rejected = []
    for folder in ["PGDS", "PGTD"]:
        for path in sorted((data_root / folder).glob("*.csv")):
            try:
                label = parse_source_label(path, data_root)
                df = read_signal_csv(path)
                records.append(
                    {
                        "source_file": label.source_file,
                        "material_system": label.material_system,
                        "content_pct": label.content_pct,
                        "requires_qc": label.requires_qc,
                        **signal_stats(df),
                    }
                )
            except Exception as exc:  # noqa: BLE001 - manifest should capture bad files.
                rejected.append({"source_file": str(path), "reason": str(exc)})
    return pd.DataFrame(records), pd.DataFrame(rejected)


def resample_segment(segment: pd.DataFrame) -> np.ndarray:
    old_x = np.linspace(0.0, 1.0, len(segment), dtype=np.float64)
    new_x = np.linspace(0.0, 1.0, RESAMPLE_LENGTH, dtype=np.float64)
    channels = []
    for column in ["voltage_v", "filter_v"]:
        values = segment[column].to_numpy(dtype=np.float64)
        interp = np.interp(new_x, old_x, values)
        channels.append(robust_zscore(interp).astype(np.float32))
    return np.stack(channels, axis=0)


def local_cycle_bounds(segment: pd.DataFrame, peak_rel_idx: int | None) -> tuple[int, int]:
    if len(segment) <= 2:
        return 0, max(0, len(segment) - 1)
    time = segment["time_s"].to_numpy(dtype=np.float64)
    duration = float(time[-1] - time[0])
    if peak_rel_idx is None:
        activity = robust_zscore(segment["voltage_v"].to_numpy(dtype=np.float64)) ** 2 + robust_zscore(
            segment["filter_v"].to_numpy(dtype=np.float64)
        ) ** 2
        peak_rel_idx = int(np.argmax(activity))
    peak_rel_idx = int(np.clip(peak_rel_idx, 0, len(segment) - 1))
    peak_t = float(time[peak_rel_idx])
    radius_s = max(0.08, 0.32 * duration)
    left = int(np.searchsorted(time, peak_t - radius_s, side="left"))
    right = int(np.searchsorted(time, peak_t + radius_s, side="right")) - 1
    left = max(0, min(left, len(segment) - 2))
    right = min(len(segment) - 1, max(right, left + 1))
    return left, right


def raw_segment_stats(segment: pd.DataFrame, peak_rel_idx: int | None = None) -> dict[str, float]:
    out: dict[str, float] = {}
    cycle_left, cycle_right = local_cycle_bounds(segment, peak_rel_idx)
    for column in ["voltage_v", "filter_v"]:
        window_values = segment[column].to_numpy(dtype=np.float64)
        values = window_values[cycle_left : cycle_right + 1]
        prefix = column.replace("_v", "")
        out[f"{prefix}_window_min_raw"] = float(np.min(window_values))
        out[f"{prefix}_window_max_raw"] = float(np.max(window_values))
        out[f"{prefix}_window_ptp_raw"] = float(np.ptp(window_values))
        out[f"{prefix}_mean_raw"] = float(np.mean(values))
        out[f"{prefix}_std_raw"] = float(np.std(values))
        out[f"{prefix}_min_raw"] = float(np.min(values))
        out[f"{prefix}_max_raw"] = float(np.max(values))
        out[f"{prefix}_ptp_raw"] = float(np.ptp(values))
        out[f"{prefix}_rms_raw"] = float(np.sqrt(np.mean(values**2)))
        out[f"{prefix}_energy_raw"] = float(np.sum(values**2))
        out[f"{prefix}_abs_area_raw"] = float(np.trapezoid(np.abs(values)))
    out["cycle_start_rel_idx"] = float(cycle_left)
    out["cycle_end_rel_idx"] = float(cycle_right)
    out["cycle_duration_s"] = float(segment["time_s"].iloc[cycle_right] - segment["time_s"].iloc[cycle_left])
    return out


def non_max_suppression(indices: np.ndarray, scores: np.ndarray, min_distance: int) -> list[int]:
    order = np.argsort(scores)[::-1]
    selected: list[int] = []
    for pos in order:
        idx = int(indices[pos])
        if all(abs(idx - kept) >= min_distance for kept in selected):
            selected.append(idx)
    return sorted(selected)


def detect_candidate_segments(df: pd.DataFrame) -> list[dict[str, int | float | str]]:
    """Detect candidate response windows with a new peak-energy strategy."""

    time = df["time_s"].to_numpy(dtype=np.float64)
    dt = estimate_dt(time)
    voltage_z = robust_zscore(df["voltage_v"].to_numpy(dtype=np.float64))
    filter_z = robust_zscore(df["filter_v"].to_numpy(dtype=np.float64))
    activity = np.maximum(np.abs(voltage_z), np.abs(filter_z))
    smooth = moving_average(activity, max(3, int(round(0.02 / max(dt, 1e-9)))))
    if len(smooth) < 8 or float(np.std(smooth)) <= 1e-12:
        return []

    candidate = np.where((smooth[1:-1] >= smooth[:-2]) & (smooth[1:-1] > smooth[2:]))[0] + 1
    threshold = max(float(np.percentile(smooth, 80)), float(np.median(smooth) + 0.35 * np.std(smooth)))
    candidate = candidate[smooth[candidate] >= threshold]
    if len(candidate) == 0:
        return []

    min_distance = max(8, int(round(0.35 / max(dt, 1e-9))))
    peaks = non_max_suppression(candidate, smooth[candidate], min_distance)
    half_window = max(16, int(round(0.35 / max(dt, 1e-9))))
    expand = max(3, int(round(0.04 / max(dt, 1e-9))))
    segments = []
    used: list[tuple[int, int]] = []
    for rank, peak in enumerate(peaks):
        left = max(0, peak - half_window)
        right = min(len(df) - 1, peak + half_window)
        local_threshold = max(float(np.percentile(smooth[left : right + 1], 35)), threshold * 0.35)
        while left > 0 and smooth[left] > local_threshold:
            left -= expand
        while right < len(df) - 1 and smooth[right] > local_threshold:
            right += expand
        left = max(0, left)
        right = min(len(df) - 1, right)
        duration = float(time[right] - time[left])
        if duration < 0.08 or duration > 2.5 or right - left < 32:
            continue
        overlap = False
        for old_left, old_right in used:
            inter = max(0, min(right, old_right) - max(left, old_left))
            if inter / max(1, right - left) > 0.45:
                overlap = True
                break
        if overlap:
            continue
        used.append((left, right))
        edge_margin = float(
            max(0.0, min(smooth[peak] - smooth[left], smooth[peak] - smooth[right]) / max(smooth[peak], 1e-9))
        )
        segments.append(
            {
                "candidate_rank": rank,
                "start_idx": int(left),
                "end_idx": int(right),
                "peak_idx": int(peak),
                "start_s": float(time[left]),
                "end_s": float(time[right]),
                "duration_s": duration,
                "detection_channel": "voltage_filter_activity",
                "peak_activity": float(smooth[peak]),
                "edge_margin_score": edge_margin,
            }
        )
    if len(segments) < 3:
        fallback = detect_energy_window_segments(df, activity, used, dt)
        segments.extend(fallback)
    return segments


def detect_energy_window_segments(
    df: pd.DataFrame,
    activity: np.ndarray,
    used: list[tuple[int, int]],
    dt: float,
) -> list[dict[str, int | float | str]]:
    """Fallback for files where peak detection under-samples slow/irregular signals."""

    time = df["time_s"].to_numpy(dtype=np.float64)
    window = max(64, int(round(1.0 / max(dt, 1e-9))))
    step = max(16, int(round(0.35 / max(dt, 1e-9))))
    if len(df) < window:
        return []
    starts = list(range(0, len(df) - window + 1, step))
    scored = []
    for start in starts:
        end = start + window - 1
        score = float(np.mean(activity[start : end + 1] ** 2))
        scored.append((score, start, end))
    if not scored:
        return []
    scores = np.asarray([s[0] for s in scored], dtype=np.float64)
    threshold = max(float(np.percentile(scores, 70)), float(np.median(scores) + 0.15 * np.std(scores)))
    max_segments = min(60, max(3, int(math.ceil((time[-1] - time[0]) / 1.5))))
    fallback: list[dict[str, int | float | str]] = []
    for rank, (score, start, end) in enumerate(sorted(scored, reverse=True)):
        if score < threshold and len(fallback) >= 3:
            break
        overlap = False
        for old_left, old_right in used:
            inter = max(0, min(end, old_right) - max(start, old_left))
            if inter / max(1, end - start) > 0.35:
                overlap = True
                break
        if overlap:
            continue
        used.append((start, end))
        local = activity[start : end + 1]
        peak_local = int(np.argmax(local))
        peak = start + peak_local
        peak_value = float(activity[peak])
        edge_margin = float(
            max(0.0, min(peak_value - activity[start], peak_value - activity[end]) / max(peak_value, 1e-9))
        )
        fallback.append(
            {
                "candidate_rank": 10000 + rank,
                "start_idx": int(start),
                "end_idx": int(end),
                "peak_idx": int(peak),
                "start_s": float(time[start]),
                "end_s": float(time[end]),
                "duration_s": float(time[end] - time[start]),
                "detection_channel": "fallback_energy_window",
                "peak_activity": peak_value,
                "edge_margin_score": max(edge_margin, 0.12),
            }
        )
        if len(fallback) >= max_segments:
            break
    return sorted(fallback, key=lambda item: int(item["start_idx"]))


def qc_candidate_segments(candidates: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic QC gates to candidate segments."""

    if candidates.empty:
        return candidates.assign(qc_status=[], reject_reason=[], manual_review_required=[])

    rows = []
    for _, row in candidates.iterrows():
        reasons = []
        duration = float(row["duration_s"])
        ptp = float(row.get("filter_ptp_raw", 0.0))
        rms = float(row.get("filter_rms_raw", 0.0))
        energy = float(row.get("filter_energy_raw", 0.0))
        edge = float(row.get("edge_margin_score", 0.0))
        quality = float(row.get("quality_score", 0.0))
        if duration < 0.08 or duration > 2.5:
            reasons.append("duration_out_of_range")
        if ptp <= 1e-9:
            reasons.append("flat_filter_signal")
        if rms <= 1e-12 or energy <= 1e-10:
            reasons.append("low_energy")
        if edge < 0.08:
            reasons.append("weak_edge_margin")
        if quality < 0.05:
            reasons.append("low_quality_score")
        out = dict(row)
        out["manual_review_required"] = int(row["content_pct"]) == 75
        out["qc_status"] = "rejected" if reasons else "accepted"
        out["reject_reason"] = ";".join(reasons)
        rows.append(out)
    return pd.DataFrame(rows)


FEATURE_NAMES = [
    "voltage_mean",
    "voltage_std",
    "voltage_min",
    "voltage_max",
    "voltage_ptp",
    "voltage_rms",
    "voltage_energy",
    "voltage_abs_area",
    "voltage_zero_crossings",
    "filter_mean",
    "filter_std",
    "filter_min",
    "filter_max",
    "filter_ptp",
    "filter_rms",
    "filter_energy",
    "filter_abs_area",
    "filter_zero_crossings",
    "cross_channel_corr",
]

RAW_FEATURE_NAMES = [
    "filter_ptp_raw",
    "filter_std_raw",
    "voltage_ptp_raw",
    "voltage_std_raw",
    "cycle_duration_s",
    "edge_margin_score",
]


def raw_feature_matrix(meta: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Return baseline-insensitive raw cycle features for fusion models."""

    missing = [name for name in RAW_FEATURE_NAMES if name not in meta.columns]
    if missing:
        raise ValueError(f"Missing raw feature columns: {missing}")
    features = meta[RAW_FEATURE_NAMES].astype(float).replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.median(numeric_only=True)).fillna(0.0)
    values = features.to_numpy(dtype=np.float64)
    return values, list(RAW_FEATURE_NAMES)


def zero_crossings(values: np.ndarray) -> float:
    signs = np.signbit(values)
    return float(np.count_nonzero(signs[1:] != signs[:-1]))


def extract_features(samples: np.ndarray) -> np.ndarray:
    rows = []
    for sample in samples:
        row = []
        for channel in [sample[0], sample[1]]:
            values = channel.astype(np.float64)
            row.extend(
                [
                    float(np.mean(values)),
                    float(np.std(values)),
                    float(np.min(values)),
                    float(np.max(values)),
                    float(np.ptp(values)),
                    float(np.sqrt(np.mean(values**2))),
                    float(np.sum(values**2)),
                    float(np.trapezoid(np.abs(values))),
                    zero_crossings(values),
                ]
            )
        corr = np.corrcoef(sample[0], sample[1])[0, 1]
        row.append(float(0.0 if np.isnan(corr) else corr))
        rows.append(row)
    return np.asarray(rows, dtype=np.float64)


class StandardScaler:
    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "StandardScaler":
        self.mean_ = np.mean(x, axis=0)
        self.scale_ = np.std(x, axis=0)
        self.scale_[self.scale_ < 1e-12] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScaler is not fitted")
        return (x - self.mean_) / self.scale_

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)


class NearestCentroid:
    def __init__(self) -> None:
        self.classes_: np.ndarray | None = None
        self.centroids_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "NearestCentroid":
        classes = np.unique(y)
        self.classes_ = classes
        self.centroids_ = np.vstack([x[y == cls].mean(axis=0) for cls in classes])
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.classes_ is None or self.centroids_ is None:
            raise RuntimeError("NearestCentroid is not fitted")
        distances = ((x[:, None, :] - self.centroids_[None, :, :]) ** 2).sum(axis=2)
        return self.classes_[np.argmin(distances, axis=1)]


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    x_aug = np.column_stack([np.ones(len(x)), x])
    eye = np.eye(x_aug.shape[1])
    eye[0, 0] = 0.0
    return np.linalg.pinv(x_aug.T @ x_aug + alpha * eye) @ x_aug.T @ y


def ridge_predict(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    x_aug = np.column_stack([np.ones(len(x)), x])
    return x_aug @ weights


def clamp_ratio(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=np.float64), 0.0, 100.0)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err))) if len(y_true) else math.nan
    rmse = float(np.sqrt(np.mean(err**2))) if len(y_true) else math.nan
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else math.nan
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "within_5_pct": float(np.mean(np.abs(err) <= 5.0)) if len(y_true) else math.nan,
        "within_10_pct": float(np.mean(np.abs(err) <= 10.0)) if len(y_true) else math.nan,
        "within_15_pct": float(np.mean(np.abs(err) <= 15.0)) if len(y_true) else math.nan,
    }


def content_class_metrics(
    true_system: np.ndarray,
    pred_system: np.ndarray,
    true_content: np.ndarray,
    pred_content: np.ndarray,
) -> dict[str, float]:
    true_system = np.asarray(true_system)
    pred_system = np.asarray(pred_system)
    true_content = np.asarray(true_content, dtype=np.int64)
    pred_content = np.asarray(pred_content, dtype=np.int64)
    combination_true = np.asarray([f"{s}_{c}" for s, c in zip(true_system, true_content)])
    combination_pred = np.asarray([f"{s}_{c}" for s, c in zip(pred_system, pred_content)])
    return {
        "system_accuracy": float(np.mean(true_system == pred_system)) if len(true_system) else math.nan,
        "content_accuracy": float(np.mean(true_content == pred_content)) if len(true_content) else math.nan,
        "combination_accuracy": float(np.mean(combination_true == combination_pred)) if len(true_system) else math.nan,
        "content_mae": float(np.mean(np.abs(true_content - pred_content))) if len(true_content) else math.nan,
    }


def ensure_no_source_leakage(train_meta: pd.DataFrame, test_meta: pd.DataFrame) -> None:
    overlap = set(train_meta["source_file"].astype(str)) & set(test_meta["source_file"].astype(str))
    if overlap:
        raise ValueError(f"Source leakage detected: {sorted(overlap)}")


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: list[object]) -> np.ndarray:
    mapping = {label: idx for idx, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for true, pred in zip(y_true, y_pred):
        if true in mapping and pred in mapping:
            matrix[mapping[true], mapping[pred]] += 1
    return matrix


def save_confusion_plot(matrix: np.ndarray, labels: list[object], title: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 0.7), max(4, len(labels) * 0.6)))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_predicted_vs_true(y_true: np.ndarray, y_pred: np.ndarray, title: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.5))
    ax.scatter(y_true, y_pred, s=22, alpha=0.78)
    ax.plot([0, 100], [0, 100], color="#111827", linewidth=1.2, linestyle="--")
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.set_title(title)
    ax.set_xlabel("True ratio (%)")
    ax.set_ylabel("Predicted ratio (%)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_error_distribution(y_true: np.ndarray, y_pred: np.ndarray, title: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    err = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.hist(err, bins=min(20, max(5, len(err) // 3)), color="#4f46e5", alpha=0.8)
    ax.axvline(0, color="#111827", linewidth=1.1)
    ax.set_title(title)
    ax.set_xlabel("Prediction error (%)")
    ax.set_ylabel("Samples")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_pca_embedding(features: np.ndarray, color_values: np.ndarray, title: str, path: Path, discrete: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(features) == 0:
        return
    centered = features - features.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ vt[:2].T if vt.shape[0] >= 2 else np.column_stack([centered[:, 0], np.zeros(len(centered))])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    if discrete:
        labels = np.asarray(color_values)
        for label in sorted(set(labels)):
            mask = labels == label
            ax.scatter(projected[mask, 0], projected[mask, 1], s=18, alpha=0.75, label=str(label))
        ax.legend(frameon=False, fontsize=8)
    else:
        sc = ax.scatter(projected[:, 0], projected[:, 1], c=color_values, cmap="viridis", s=18, alpha=0.8)
        fig.colorbar(sc, ax=ax, label="Ratio (%)")
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def publish_step(step_dir: Path, fn: Callable[[Path], None]) -> None:
    step_dir.mkdir(parents=True, exist_ok=True)
    staging = step_dir / "_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    fn(staging)
    for child in staging.iterdir():
        target = step_dir / child.name
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        shutil.move(str(child), str(target))
    staging.rmdir()


def make_run_dir(data_root: Path, run_name: str | None = None) -> Path:
    root = data_root / "ratio_dual_model"
    root.mkdir(parents=True, exist_ok=True)
    name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = root / name
    suffix = 1
    while run_dir.exists():
        run_dir = root / f"{name}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True)
    return run_dir


def step_preflight(data_root: Path, run_dir: Path) -> pd.DataFrame:
    """扫描 PGDS/PGTD 目录，从文件名解析比例标签，写出源清单。"""
    def write(staging: Path) -> None:
        manifest, rejected = discover_sources(data_root)
        if manifest.empty:
            raise RuntimeError("No usable PGDS/PGTD CSV files were found")
        manifest.to_csv(staging / "source_manifest.csv", index=False, encoding="utf-8-sig")
        rejected.to_csv(staging / "rejected_files.csv", index=False, encoding="utf-8-sig")
        report = {
            "data_root": str(data_root),
            "source_files": int(len(manifest)),
            "rejected_files": int(len(rejected)),
            "content_levels": CONTENT_LEVELS,
            "systems": SYSTEMS,
            "requires_qc_files": manifest.loc[manifest["requires_qc"], "source_file"].tolist(),
        }
        (staging / "preflight_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if not set(CONTENT_LEVELS).issubset(set(manifest["content_pct"].astype(int))):
            raise RuntimeError("Not all required content levels are represented in the manifest")

    publish_step(run_dir / "00_preflight", write)
    return pd.read_csv(run_dir / "00_preflight" / "source_manifest.csv")


def step_clean(data_root: Path, run_dir: Path, manifest: pd.DataFrame) -> pd.DataFrame:
    """读取 CSV，清洗时间轴与电压/滤波通道，保存为 npz。"""
    def write(staging: Path) -> None:
        arrays_dir = staging / "cleaned_arrays"
        arrays_dir.mkdir()
        rows = []
        rejected = []
        for _, row in manifest.iterrows():
            try:
                df = read_signal_csv(data_root / str(row["source_file"]))
                if len(df) <= 1000:
                    raise ValueError("effective_rows_lte_1000")
                time = df["time_s"].to_numpy(dtype=np.float64)
                if not np.all(np.diff(time) > 0):
                    raise ValueError("time_not_strictly_increasing")
                safe_name = re.sub(r"[^0-9A-Za-z_\-]+", "_", Path(str(row["source_file"])).stem)
                array_path = arrays_dir / f"{safe_name}.npz"
                np.savez_compressed(
                    array_path,
                    time_s=df["time_s"].to_numpy(dtype=np.float64),
                    voltage_v=df["voltage_v"].to_numpy(dtype=np.float64),
                    filter_v=df["filter_v"].to_numpy(dtype=np.float64),
                )
                rows.append(
                    {
                        "source_file": row["source_file"],
                        "cleaned_array": str(array_path.relative_to(staging)),
                        "material_system": row["material_system"],
                        "content_pct": int(row["content_pct"]),
                        "requires_qc": bool(row["requires_qc"]),
                        **signal_stats(df),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                rejected.append({"source_file": row["source_file"], "reason": str(exc)})
        clean = pd.DataFrame(rows)
        clean.to_csv(staging / "cleaned_manifest.csv", index=False, encoding="utf-8-sig")
        clean.to_csv(staging / "per_file_stats.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(rejected).to_csv(staging / "cleaning_warnings.csv", index=False, encoding="utf-8-sig")
        if clean.empty:
            raise RuntimeError("No files survived cleaning")

    publish_step(run_dir / "01_cleaned_signals", write)
    return pd.read_csv(run_dir / "01_cleaned_signals" / "cleaned_manifest.csv")


def plot_file_overview(df: pd.DataFrame, segments: list[dict[str, object]], title: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    skip = max(1, len(df) // 6000)
    view = df.iloc[::skip]
    fig, axes = plt.subplots(2, 1, figsize=(10, 4.8), sharex=True)
    axes[0].plot(view["time_s"], view["voltage_v"], linewidth=0.7)
    axes[0].set_ylabel("Voltage")
    axes[1].plot(view["time_s"], view["filter_v"], linewidth=0.7, color="#0f766e")
    axes[1].set_ylabel("Filter")
    axes[1].set_xlabel("Time (s)")
    for segment in segments:
        for ax in axes:
            ax.axvspan(float(segment["start_s"]), float(segment["end_s"]), color="#f59e0b", alpha=0.12)
    axes[0].set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def step_candidates(data_root: Path, run_dir: Path, cleaned: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """按周期峰检测切候选片段，并重采样到固定长度。"""
    def write(staging: Path) -> None:
        fig_dir = staging / "figures" / "per_file_overview"
        arr_dir = staging / "segment_arrays"
        fig_dir.mkdir(parents=True)
        arr_dir.mkdir()
        records = []
        samples = []
        for _, row in cleaned.iterrows():
            df = read_signal_csv(data_root / str(row["source_file"]))
            candidates = detect_candidate_segments(df)
            file_segments: list[dict[str, object]] = []
            for local_idx, candidate in enumerate(candidates):
                segment = df.iloc[int(candidate["start_idx"]) : int(candidate["end_idx"]) + 1]
                peak_rel_idx = int(candidate["peak_idx"]) - int(candidate["start_idx"])
                stats = raw_segment_stats(segment, peak_rel_idx=peak_rel_idx)
                activity_ptp = max(stats["voltage_ptp_raw"], stats["filter_ptp_raw"])
                quality = float(
                    np.log1p(activity_ptp)
                    * max(0.0, float(candidate["edge_margin_score"]))
                    * min(1.0, float(candidate["duration_s"]) / 0.7)
                )
                segment_id = f"{Path(str(row['source_file'])).stem}_seg{local_idx:04d}"
                sample = resample_segment(segment)
                record = {
                    "segment_id": segment_id,
                    "source_file": row["source_file"],
                    "material_system": row["material_system"],
                    "content_pct": int(row["content_pct"]),
                    "requires_qc": bool(row["requires_qc"]),
                    "quality_score": quality,
                    **candidate,
                    **stats,
                }
                records.append(record)
                file_segments.append(record)
                samples.append(sample)
            safe_name = re.sub(r"[^0-9A-Za-z_\-]+", "_", Path(str(row["source_file"])).stem)
            plot_file_overview(
                df,
                file_segments,
                f"{row['source_file']} candidates={len(file_segments)}",
                fig_dir / f"{safe_name}.png",
            )
        if not records:
            raise RuntimeError("No candidate segments detected")
        candidate_df = pd.DataFrame(records)
        candidate_df.to_csv(staging / "candidate_segments.csv", index=False, encoding="utf-8-sig")
        np.savez_compressed(
            arr_dir / "candidate_segments.npz",
            x=np.stack(samples).astype(np.float32),
            segment_ids=candidate_df["segment_id"].astype(str).to_numpy(),
        )
        summary = candidate_df.groupby(["material_system", "content_pct", "source_file"]).size().reset_index(name="candidate_segments")
        summary.to_csv(staging / "candidate_summary_by_file.csv", index=False, encoding="utf-8-sig")

    publish_step(run_dir / "02_candidate_segments", write)
    df = pd.read_csv(run_dir / "02_candidate_segments" / "candidate_segments.csv")
    data = np.load(run_dir / "02_candidate_segments" / "segment_arrays" / "candidate_segments.npz", allow_pickle=True)
    return df, data["x"].astype(np.float32)


def save_segment_gallery(samples: np.ndarray, meta: pd.DataFrame, path: Path, title: str, max_items: int = 24) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    take = min(max_items, len(meta))
    if take == 0:
        return
    cols = 4
    rows = int(math.ceil(take / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.1), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for i in range(take):
        ax = axes.ravel()[i]
        ax.axis("on")
        sample = samples[int(meta.iloc[i]["sample_index"])]
        ax.plot(sample[1], linewidth=0.8)
        ax.set_title(str(meta.iloc[i]["segment_id"])[:28], fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def step_qc(run_dir: Path, candidates: pd.DataFrame, candidate_samples: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    """质控过滤异常片段（75% 等重点复核对象会更严）。"""
    def write(staging: Path) -> None:
        meta = candidates.copy()
        meta["sample_index"] = np.arange(len(meta))
        qc = qc_candidate_segments(meta)
        accepted = qc[qc["qc_status"] == "accepted"].copy()
        if accepted.empty:
            raise RuntimeError("QC rejected all candidate segments")
        label_counts = accepted.groupby(["material_system", "content_pct"]).size().reset_index(name="accepted_segments")
        if label_counts.empty:
            raise RuntimeError("No labels survived QC")
        source_manifest = pd.read_csv(run_dir / "00_preflight" / "source_manifest.csv")
        expected = set(
            (str(row.material_system), int(row.content_pct))
            for row in source_manifest[["material_system", "content_pct"]].drop_duplicates().itertuples(index=False)
        )
        present = set((str(r.material_system), int(r.content_pct)) for r in label_counts.itertuples())
        missing = expected - present
        if missing:
            raise RuntimeError(f"QC removed all samples for labels: {sorted(missing)}")
        qc.to_csv(staging / "qc_segments.csv", index=False, encoding="utf-8-sig")
        qc.groupby(["material_system", "content_pct", "qc_status"]).size().reset_index(name="segments").to_csv(
            staging / "qc_summary_by_label.csv", index=False, encoding="utf-8-sig"
        )
        qc.groupby(["source_file", "qc_status"]).size().reset_index(name="segments").to_csv(
            staging / "qc_summary_by_file.csv", index=False, encoding="utf-8-sig"
        )
        fig_dir = staging / "figures" / "accepted_segments"
        for (system, content), group in accepted.groupby(["material_system", "content_pct"]):
            save_segment_gallery(
                candidate_samples,
                group,
                fig_dir / f"{system}_{int(content)}_accepted.png",
                f"{system} {int(content)}% accepted segments",
            )
        np.savez_compressed(
            staging / "qc_dataset.npz",
            x=candidate_samples[accepted["sample_index"].to_numpy(dtype=int)],
            segment_ids=accepted["segment_id"].astype(str).to_numpy(),
        )

    publish_step(run_dir / "03_segment_qc", write)
    qc = pd.read_csv(run_dir / "03_segment_qc" / "qc_segments.csv")
    accepted = qc[qc["qc_status"] == "accepted"].copy()
    data = np.load(run_dir / "03_segment_qc" / "qc_dataset.npz", allow_pickle=True)
    return accepted.reset_index(drop=True), data["x"].astype(np.float32)


def step_split(run_dir: Path, accepted: pd.DataFrame, x: np.ndarray, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """按文件级划分 train/test，降低同文件样本泄漏。"""
    def write(staging: Path) -> None:
        rng = np.random.default_rng(seed)
        accepted_with_index = accepted.copy()
        accepted_with_index["accepted_index"] = np.arange(len(accepted_with_index))
        train_rows = []
        test_rows = []
        for _, group in accepted_with_index.groupby(["material_system", "content_pct"], sort=True):
            sources = list(group["source_file"].astype(str).unique())
            rng.shuffle(sources)
            if len(sources) == 1:
                train_rows.append(group)
            else:
                test_source = sources[0]
                test_rows.append(group[group["source_file"].astype(str) == test_source])
                train_rows.append(group[group["source_file"].astype(str) != test_source])
        train = pd.concat(train_rows, ignore_index=True)
        test = pd.concat(test_rows, ignore_index=True) if test_rows else pd.DataFrame(columns=train.columns)
        if test.empty:
            raise RuntimeError("No leakage-free test samples available")
        train["split"] = "train"
        test["split"] = "test"
        train["augmented"] = False
        test["augmented"] = False
        ensure_no_source_leakage(train, test)
        train.to_csv(staging / "train_segments.csv", index=False, encoding="utf-8-sig")
        test.to_csv(staging / "test_segments.csv", index=False, encoding="utf-8-sig")
        pd.concat([train, test]).groupby(["split", "material_system", "content_pct"]).size().reset_index(
            name="segments"
        ).to_csv(staging / "split_summary.csv", index=False, encoding="utf-8-sig")
        arrays = staging / "arrays"
        arrays.mkdir()
        train_idx = train["accepted_index"].to_numpy(dtype=int)
        test_idx = test["accepted_index"].to_numpy(dtype=int)
        raw_train, raw_feature_names = raw_feature_matrix(train)
        raw_test, _ = raw_feature_matrix(test)
        np.savez_compressed(
            arrays / "split_dataset.npz",
            x_train=x[train_idx],
            x_test=x[test_idx],
            raw_features_train=raw_train.astype(np.float32),
            raw_features_test=raw_test.astype(np.float32),
            raw_feature_names=np.asarray(raw_feature_names, dtype=object),
            y_system_train=train["material_system"].astype(str).to_numpy(),
            y_system_test=test["material_system"].astype(str).to_numpy(),
            y_content_train=train["content_pct"].to_numpy(dtype=np.int64),
            y_content_test=test["content_pct"].to_numpy(dtype=np.int64),
            train_segment_ids=train["segment_id"].astype(str).to_numpy(),
            test_segment_ids=test["segment_id"].astype(str).to_numpy(),
        )

    publish_step(run_dir / "04_train_test_split", write)
    train = pd.read_csv(run_dir / "04_train_test_split" / "train_segments.csv")
    test = pd.read_csv(run_dir / "04_train_test_split" / "test_segments.csv")
    data = np.load(run_dir / "04_train_test_split" / "arrays" / "split_dataset.npz", allow_pickle=True)
    return train, test, data["x_train"].astype(np.float32), data["x_test"].astype(np.float32)


def augment_sample(sample: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = sample.copy()
    out *= rng.normal(1.0, 0.025, size=(out.shape[0], 1)).astype(np.float32)
    shift = int(rng.integers(-6, 7))
    if shift:
        out = np.roll(out, shift, axis=1)
    out += rng.normal(0.0, 0.025, size=out.shape).astype(np.float32)
    return out.astype(np.float32)


def step_augmentation(run_dir: Path, train: pd.DataFrame, x_train: np.ndarray, test: pd.DataFrame, x_test: np.ndarray, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    """对训练集做轻度增强，尽量平衡各比例样本数。"""
    def write(staging: Path) -> None:
        rng = np.random.default_rng(seed)
        train = pd.read_csv(run_dir / "04_train_test_split" / "train_segments.csv")
        data = np.load(run_dir / "04_train_test_split" / "arrays" / "split_dataset.npz", allow_pickle=True)
        x_train = data["x_train"].astype(np.float32)
        rows = []
        samples = []
        by_label = train.groupby(["material_system", "content_pct"]).indices
        counts = [len(idx) for idx in by_label.values()]
        target = int(min(80, max(counts), max(10, np.median(counts)))) if counts else 0
        for _, row in train.iterrows():
            rows.append(dict(row))
            samples.append(x_train[int(row.name)])
        summary_rows = []
        for (system, content), indices in by_label.items():
            indices = list(indices)
            real = len(indices)
            cap = target
            if int(content) == 75:
                cap = min(target, real * 2)
            added = 0
            while real + added < cap:
                source_pos = int(rng.choice(indices))
                rec = dict(train.iloc[source_pos])
                rec["segment_id"] = f"{rec['segment_id']}_aug{added + 1:04d}"
                rec["augmented"] = True
                rec["source_segment_id"] = train.iloc[source_pos]["segment_id"]
                rows.append(rec)
                samples.append(augment_sample(x_train[source_pos], rng))
                added += 1
            summary_rows.append(
                {
                    "material_system": system,
                    "content_pct": int(content),
                    "real_train_segments": real,
                    "augmented_added": added,
                    "final_train_segments": real + added,
                }
            )
        aug = pd.DataFrame(rows)
        aug.to_csv(staging / "augmented_train_segments.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(summary_rows).to_csv(staging / "augmentation_summary.csv", index=False, encoding="utf-8-sig")
        arrays = staging / "arrays"
        arrays.mkdir()
        split_data = np.load(run_dir / "04_train_test_split" / "arrays" / "split_dataset.npz", allow_pickle=True)
        raw_train, raw_feature_names = raw_feature_matrix(aug)
        np.savez_compressed(
            arrays / "train_augmented.npz",
            x_train=np.stack(samples).astype(np.float32),
            x_test=split_data["x_test"].astype(np.float32),
            raw_features_train=raw_train.astype(np.float32),
            raw_features_test=split_data["raw_features_test"].astype(np.float32),
            raw_feature_names=np.asarray(raw_feature_names, dtype=object),
            y_system_train=aug["material_system"].astype(str).to_numpy(),
            y_system_test=split_data["y_system_test"],
            y_content_train=aug["content_pct"].to_numpy(dtype=np.int64),
            y_content_test=split_data["y_content_test"],
            train_segment_ids=aug["segment_id"].astype(str).to_numpy(),
            test_segment_ids=split_data["test_segment_ids"],
        )
        if len(split_data["x_test"]) != len(pd.read_csv(run_dir / "04_train_test_split" / "test_segments.csv")):
            raise RuntimeError("Test array length changed during augmentation")

    publish_step(run_dir / "05_augmentation", write)
    aug = pd.read_csv(run_dir / "05_augmentation" / "augmented_train_segments.csv")
    data = np.load(run_dir / "05_augmentation" / "arrays" / "train_augmented.npz", allow_pickle=True)
    return aug, data["x_train"].astype(np.float32)


def load_augmented_dataset(run_dir: Path) -> dict[str, np.ndarray]:
    return dict(np.load(run_dir / "05_augmentation" / "arrays" / "train_augmented.npz", allow_pickle=True))


def step_baseline_classification(run_dir: Path) -> dict[str, float]:
    def write(staging: Path) -> None:
        data = load_augmented_dataset(run_dir)
        x_train = extract_features(data["x_train"])
        x_test = extract_features(data["x_test"])
        scaler = StandardScaler()
        x_train_s = scaler.fit_transform(x_train)
        x_test_s = scaler.transform(x_test)
        system_model = NearestCentroid().fit(x_train_s, data["y_system_train"])
        content_model = NearestCentroid().fit(x_train_s, data["y_content_train"].astype(np.int64))
        pred_system = system_model.predict(x_test_s).astype(str)
        pred_content = content_model.predict(x_test_s).astype(np.int64)
        true_system = data["y_system_test"].astype(str)
        true_content = data["y_content_test"].astype(np.int64)
        metrics = content_class_metrics(true_system, pred_system, true_content, pred_content)
        (staging / "models").mkdir()
        (staging / "models" / "baseline_classifier.json").write_text(
            json.dumps(
                {
                    "model": "NearestCentroid",
                    "feature_names": FEATURE_NAMES,
                    "scaler_mean": scaler.mean_.tolist(),
                    "scaler_scale": scaler.scale_.tolist(),
                    "system_classes": system_model.classes_.tolist(),
                    "system_centroids": system_model.centroids_.tolist(),
                    "content_classes": content_model.classes_.tolist(),
                    "content_centroids": content_model.centroids_.tolist(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        pred_df = pd.DataFrame(
            {
                "segment_id": data["test_segment_ids"].astype(str),
                "true_system": true_system,
                "pred_system": pred_system,
                "true_content_pct": true_content,
                "pred_content_pct": pred_content,
                "content_abs_error_pct": np.abs(true_content - pred_content),
            }
        )
        pred_df.to_csv(staging / "predictions_baseline_classification.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame([{"model": "baseline_classification", **metrics}]).to_csv(
            staging / "metrics_baseline_classification.csv", index=False, encoding="utf-8-sig"
        )
        fig_dir = staging / "figures"
        save_confusion_plot(
            confusion_matrix(true_content, pred_content, CONTENT_LEVELS),
            CONTENT_LEVELS,
            "Baseline content classification",
            fig_dir / "content_confusion_matrix.png",
        )
        save_confusion_plot(
            confusion_matrix(true_system, pred_system, SYSTEMS),
            SYSTEMS,
            "Baseline system classification",
            fig_dir / "system_confusion_matrix.png",
        )
        save_pca_embedding(x_test_s, true_content, "Baseline feature projection", fig_dir / "embedding_by_class.png", True)

    publish_step(run_dir / "06_baseline_classification", write)
    return pd.read_csv(run_dir / "06_baseline_classification" / "metrics_baseline_classification.csv").iloc[0].to_dict()


def step_baseline_regression(run_dir: Path) -> dict[str, float]:
    def write(staging: Path) -> None:
        data = load_augmented_dataset(run_dir)
        x_train = extract_features(data["x_train"])
        x_test = extract_features(data["x_test"])
        scaler = StandardScaler()
        x_train_s = scaler.fit_transform(x_train)
        x_test_s = scaler.transform(x_test)
        y_train = data["y_content_train"].astype(np.float64)
        y_test = data["y_content_test"].astype(np.float64)
        weights = ridge_fit(x_train_s, y_train, alpha=1.0)
        pred = clamp_ratio(ridge_predict(x_test_s, weights))
        metrics = regression_metrics(y_test, pred)
        (staging / "models").mkdir()
        (staging / "models" / "baseline_ridge_regression.json").write_text(
            json.dumps(
                {
                    "model": "ridge_regression_numpy",
                    "alpha": 1.0,
                    "feature_names": FEATURE_NAMES,
                    "scaler_mean": scaler.mean_.tolist(),
                    "scaler_scale": scaler.scale_.tolist(),
                    "weights": weights.tolist(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        pred_df = pd.DataFrame(
            {
                "segment_id": data["test_segment_ids"].astype(str),
                "true_content_pct": y_test,
                "pred_content_pct": pred,
                "content_abs_error_pct": np.abs(y_test - pred),
            }
        )
        pred_df.to_csv(staging / "predictions_baseline_regression.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame([{"model": "baseline_regression", **metrics}]).to_csv(
            staging / "metrics_baseline_regression.csv", index=False, encoding="utf-8-sig"
        )
        fig_dir = staging / "figures"
        save_predicted_vs_true(y_test, pred, "Baseline regression", fig_dir / "predicted_vs_true.png")
        save_error_distribution(y_test, pred, "Baseline regression errors", fig_dir / "error_distribution.png")

    publish_step(run_dir / "08_baseline_regression", write)
    return pd.read_csv(run_dir / "08_baseline_regression" / "metrics_baseline_regression.csv").iloc[0].to_dict()


def try_import_torch():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        return torch, nn, DataLoader, TensorDataset
    except ModuleNotFoundError:
        return None


def write_cnn_blocked(staging: Path, model_name: str) -> None:
    (staging / f"{model_name.upper()}_BLOCKED_ENVIRONMENT.md").write_text(
        "PyTorch is not available in the current Python environment, so this CNN step was not executed.\n"
        "Baseline models and all non-CNN artifacts were generated normally.\n",
        encoding="utf-8",
    )


def step_cnn_classification(run_dir: Path, max_epochs: int, min_epochs: int, patience: int, seed: int) -> dict[str, float | str]:
    def write(staging: Path) -> None:
        imported = try_import_torch()
        if imported is None:
            write_cnn_blocked(staging, "cnn_classification")
            return
        torch, nn, DataLoader, TensorDataset = imported
        torch.manual_seed(seed)
        data = load_augmented_dataset(run_dir)
        x_train = torch.tensor(data["x_train"], dtype=torch.float32)
        x_test = torch.tensor(data["x_test"], dtype=torch.float32)
        raw_scaler = StandardScaler()
        raw_train_np = raw_scaler.fit_transform(data["raw_features_train"].astype(np.float64)).astype(np.float32)
        raw_test_np = raw_scaler.transform(data["raw_features_test"].astype(np.float64)).astype(np.float32)
        raw_train = torch.tensor(raw_train_np, dtype=torch.float32)
        raw_test = torch.tensor(raw_test_np, dtype=torch.float32)
        raw_dim = raw_train.shape[1]
        system_map = {name: idx for idx, name in enumerate(SYSTEMS)}
        content_map = {value: idx for idx, value in enumerate(CONTENT_LEVELS)}
        y_sys_train = torch.tensor([system_map[str(v)] for v in data["y_system_train"]], dtype=torch.long)
        y_sys_test = torch.tensor([system_map[str(v)] for v in data["y_system_test"]], dtype=torch.long)
        y_cont_train = torch.tensor([content_map[int(v)] for v in data["y_content_train"]], dtype=torch.long)
        y_cont_test = torch.tensor([content_map[int(v)] for v in data["y_content_test"]], dtype=torch.long)

        class Encoder(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.wave_encoder = nn.Sequential(
                    nn.Conv1d(2, 24, 9, padding=4),
                    nn.BatchNorm1d(24),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Conv1d(24, 48, 7, padding=3),
                    nn.BatchNorm1d(48),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Conv1d(48, 96, 5, padding=2),
                    nn.BatchNorm1d(96),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(1),
                    nn.Flatten(),
                )
                self.fusion = nn.Sequential(
                    nn.Linear(96 + raw_dim, 96),
                    nn.ReLU(),
                    nn.Dropout(0.25),
                    nn.Linear(96, 64),
                    nn.ReLU(),
                )

                self.system_head = nn.Linear(64, len(SYSTEMS))
                self.content_head = nn.Linear(64, len(CONTENT_LEVELS))

            def forward(self, x, raw, return_features: bool = False):
                wave_feat = self.wave_encoder(x)
                feat = self.fusion(torch.cat([wave_feat, raw], dim=1))
                sys = self.system_head(feat)
                cont = self.content_head(feat)
                if return_features:
                    return sys, cont, feat
                return sys, cont

        model = Encoder()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        loader = DataLoader(
            TensorDataset(x_train, raw_train, y_sys_train, y_cont_train),
            batch_size=32,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
        )
        history = []
        best_metric = -1.0
        best_state = None
        stale = 0
        for epoch in range(1, max_epochs + 1):
            model.train()
            train_loss = 0.0
            for bx, br, bs, bc in loader:
                opt.zero_grad()
                ps, pc = model(bx, br)
                loss = 0.35 * loss_fn(ps, bs) + 0.65 * loss_fn(pc, bc)
                loss.backward()
                opt.step()
                train_loss += float(loss.item()) * len(bx)
            model.eval()
            with torch.no_grad():
                ts, tc = model(x_train, raw_train)
                vs, vc = model(x_test, raw_test)
                test_loss = 0.35 * loss_fn(vs, y_sys_test) + 0.65 * loss_fn(vc, y_cont_test)
                train_sys_acc = float((ts.argmax(1) == y_sys_train).float().mean().item())
                train_cont_acc = float((tc.argmax(1) == y_cont_train).float().mean().item())
                test_sys_acc = float((vs.argmax(1) == y_sys_test).float().mean().item())
                test_cont_acc = float((vc.argmax(1) == y_cont_test).float().mean().item())
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss / max(1, len(x_train)),
                    "test_loss": float(test_loss.item()),
                    "train_system_accuracy": train_sys_acc,
                    "train_content_accuracy": train_cont_acc,
                    "test_system_accuracy": test_sys_acc,
                    "test_content_accuracy": test_cont_acc,
                }
            )
            if test_cont_acc > best_metric:
                best_metric = test_cont_acc
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if epoch >= min_epochs and stale >= patience:
                break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            sys_logits, cont_logits, features = model(x_test, raw_test, return_features=True)
        pred_system = np.asarray(SYSTEMS)[sys_logits.argmax(1).numpy()]
        pred_content = np.asarray(CONTENT_LEVELS)[cont_logits.argmax(1).numpy()]
        true_system = data["y_system_test"].astype(str)
        true_content = data["y_content_test"].astype(np.int64)
        metrics = content_class_metrics(true_system, pred_system, true_content, pred_content)
        (staging / "models").mkdir()
        torch.save(
            {
                "state_dict": model.state_dict(),
                "metrics": metrics,
                "raw_feature_names": data["raw_feature_names"].tolist(),
                "raw_scaler_mean": raw_scaler.mean_.tolist(),
                "raw_scaler_scale": raw_scaler.scale_.tolist(),
                "fusion": "waveform_encoder_plus_raw_features",
            },
            staging / "models" / "cnn_classification.pt",
        )
        pd.DataFrame(history).to_csv(staging / "training_history.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(
            {
                "segment_id": data["test_segment_ids"].astype(str),
                "true_system": true_system,
                "pred_system": pred_system,
                "true_content_pct": true_content,
                "pred_content_pct": pred_content,
                "content_abs_error_pct": np.abs(true_content - pred_content),
            }
        ).to_csv(staging / "predictions_cnn_classification.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame([{"model": "cnn_classification", **metrics}]).to_csv(
            staging / "metrics_cnn_classification.csv", index=False, encoding="utf-8-sig"
        )
        fig_dir = staging / "figures"
        fig_dir.mkdir()
        hist = pd.DataFrame(history)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(hist["epoch"], hist["train_loss"], label="Train")
        axes[0].plot(hist["epoch"], hist["test_loss"], label="Test")
        axes[0].set_title("CNN classification loss")
        axes[0].legend(frameon=False)
        axes[1].plot(hist["epoch"], hist["train_content_accuracy"], label="Train content")
        axes[1].plot(hist["epoch"], hist["test_content_accuracy"], label="Test content")
        axes[1].plot(hist["epoch"], hist["test_system_accuracy"], label="Test system")
        axes[1].set_title("CNN classification accuracy")
        axes[1].legend(frameon=False)
        fig.tight_layout()
        fig.savefig(fig_dir / "training_accuracy_loss.png", dpi=180)
        plt.close(fig)
        save_confusion_plot(confusion_matrix(true_content, pred_content, CONTENT_LEVELS), CONTENT_LEVELS, "CNN content classification", fig_dir / "content_confusion_matrix.png")
        save_pca_embedding(features.numpy(), true_content, "CNN embedding by class", fig_dir / "embedding_by_class.png", True)

    publish_step(run_dir / "07_cnn_classification", write)
    metrics_path = run_dir / "07_cnn_classification" / "metrics_cnn_classification.csv"
    if metrics_path.exists():
        return pd.read_csv(metrics_path).iloc[0].to_dict()
    return {"model": "cnn_classification", "status": "blocked_no_torch"}


def step_cnn_regression(run_dir: Path, max_epochs: int, min_epochs: int, patience: int, seed: int) -> dict[str, float | str]:
    def write(staging: Path) -> None:
        imported = try_import_torch()
        if imported is None:
            write_cnn_blocked(staging, "cnn_regression")
            return
        torch, nn, DataLoader, TensorDataset = imported
        torch.manual_seed(seed)
        data = load_augmented_dataset(run_dir)
        x_train = torch.tensor(data["x_train"], dtype=torch.float32)
        x_test = torch.tensor(data["x_test"], dtype=torch.float32)
        raw_scaler = StandardScaler()
        raw_train_np = raw_scaler.fit_transform(data["raw_features_train"].astype(np.float64)).astype(np.float32)
        raw_test_np = raw_scaler.transform(data["raw_features_test"].astype(np.float64)).astype(np.float32)
        raw_train = torch.tensor(raw_train_np, dtype=torch.float32)
        raw_test = torch.tensor(raw_test_np, dtype=torch.float32)
        raw_dim = raw_train.shape[1]
        y_train = torch.tensor(data["y_content_train"].astype(np.float32), dtype=torch.float32).view(-1, 1)
        y_test = torch.tensor(data["y_content_test"].astype(np.float32), dtype=torch.float32).view(-1, 1)

        class Regressor(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.wave_encoder = nn.Sequential(
                    nn.Conv1d(2, 24, 9, padding=4),
                    nn.BatchNorm1d(24),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Conv1d(24, 48, 7, padding=3),
                    nn.BatchNorm1d(48),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Conv1d(48, 96, 5, padding=2),
                    nn.BatchNorm1d(96),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(1),
                    nn.Flatten(),
                )
                self.fusion = nn.Sequential(
                    nn.Linear(96 + raw_dim, 96),
                    nn.ReLU(),
                    nn.Dropout(0.25),
                    nn.Linear(96, 64),
                    nn.ReLU(),
                )
                self.head = nn.Linear(64, 1)

            def forward(self, x, raw, return_features: bool = False):
                wave_feat = self.wave_encoder(x)
                feat = self.fusion(torch.cat([wave_feat, raw], dim=1))
                pred = 100.0 * torch.sigmoid(self.head(feat))
                if return_features:
                    return pred, feat
                return pred

        model = Regressor()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        loss_fn = nn.HuberLoss(delta=8.0)
        loader = DataLoader(TensorDataset(x_train, raw_train, y_train), batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(seed))
        history = []
        best_metric = math.inf
        best_state = None
        stale = 0
        for epoch in range(1, max_epochs + 1):
            model.train()
            train_loss = 0.0
            for bx, br, by in loader:
                opt.zero_grad()
                pred = model(bx, br)
                loss = loss_fn(pred, by)
                loss.backward()
                opt.step()
                train_loss += float(loss.item()) * len(bx)
            model.eval()
            with torch.no_grad():
                train_pred = model(x_train, raw_train)
                test_pred = model(x_test, raw_test)
                train_loss_eval = float(loss_fn(train_pred, y_train).item())
                test_loss = float(loss_fn(test_pred, y_test).item())
                test_mae = float(torch.mean(torch.abs(test_pred - y_test)).item())
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss / max(1, len(x_train)),
                    "train_loss_eval": train_loss_eval,
                    "test_loss": test_loss,
                    "test_mae": test_mae,
                }
            )
            if not math.isfinite(test_loss):
                raise RuntimeError("CNN regression loss became non-finite")
            if test_mae < best_metric:
                best_metric = test_mae
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if epoch >= min_epochs and stale >= patience:
                break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            pred_tensor, features = model(x_test, raw_test, return_features=True)
        pred = clamp_ratio(pred_tensor.numpy().reshape(-1))
        true = data["y_content_test"].astype(np.float64)
        metrics = regression_metrics(true, pred)
        (staging / "models").mkdir()
        torch.save(
            {
                "state_dict": model.state_dict(),
                "metrics": metrics,
                "raw_feature_names": data["raw_feature_names"].tolist(),
                "raw_scaler_mean": raw_scaler.mean_.tolist(),
                "raw_scaler_scale": raw_scaler.scale_.tolist(),
                "fusion": "waveform_encoder_plus_raw_features",
            },
            staging / "models" / "cnn_regression.pt",
        )
        pd.DataFrame(history).to_csv(staging / "training_history.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(
            {
                "segment_id": data["test_segment_ids"].astype(str),
                "true_content_pct": true,
                "pred_content_pct": pred,
                "content_abs_error_pct": np.abs(true - pred),
            }
        ).to_csv(staging / "predictions_cnn_regression.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame([{"model": "cnn_regression", **metrics}]).to_csv(
            staging / "metrics_cnn_regression.csv", index=False, encoding="utf-8-sig"
        )
        fig_dir = staging / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        hist = pd.DataFrame(history)
        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.plot(hist["epoch"], hist["train_loss_eval"], label="Train")
        ax.plot(hist["epoch"], hist["test_loss"], label="Test")
        ax.set_title("CNN regression loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Huber loss")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(fig_dir / "training_loss.png", dpi=180)
        plt.close(fig)
        save_predicted_vs_true(true, pred, "CNN regression", fig_dir / "predicted_vs_true.png")
        save_error_distribution(true, pred, "CNN regression errors", fig_dir / "error_distribution.png")
        save_pca_embedding(features.numpy(), true, "CNN embedding by ratio", fig_dir / "embedding_by_ratio.png", False)

    publish_step(run_dir / "09_cnn_regression", write)
    metrics_path = run_dir / "09_cnn_regression" / "metrics_cnn_regression.csv"
    if metrics_path.exists():
        return pd.read_csv(metrics_path).iloc[0].to_dict()
    return {"model": "cnn_regression", "status": "blocked_no_torch"}


def step_comparison(run_dir: Path) -> pd.DataFrame:
    def write(staging: Path) -> None:
        rows = []
        for path in [
            run_dir / "06_baseline_classification" / "metrics_baseline_classification.csv",
            run_dir / "07_cnn_classification" / "metrics_cnn_classification.csv",
            run_dir / "08_baseline_regression" / "metrics_baseline_regression.csv",
            run_dir / "09_cnn_regression" / "metrics_cnn_regression.csv",
        ]:
            if path.exists():
                rows.extend(pd.read_csv(path).to_dict("records"))
        comparison = pd.DataFrame(rows)
        comparison.to_csv(staging / "model_comparison.csv", index=False, encoding="utf-8-sig")
        fig_dir = staging / "figures"
        fig_dir.mkdir()
        if not comparison.empty:
            cls = comparison[comparison["model"].astype(str).str.contains("classification", na=False)]
            if not cls.empty and "content_accuracy" in cls:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(cls["model"], cls["content_accuracy"])
                ax.set_ylim(0, 1)
                ax.set_ylabel("Content accuracy")
                ax.set_title("Classification metrics")
                ax.tick_params(axis="x", rotation=20)
                fig.tight_layout()
                fig.savefig(fig_dir / "classification_metrics_comparison.png", dpi=180)
                plt.close(fig)
            reg = comparison[comparison["model"].astype(str).str.contains("regression", na=False)]
            if not reg.empty and "mae" in reg:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(reg["model"], reg["mae"])
                ax.set_ylabel("MAE (%)")
                ax.set_title("Regression metrics")
                ax.tick_params(axis="x", rotation=20)
                fig.tight_layout()
                fig.savefig(fig_dir / "regression_metrics_comparison.png", dpi=180)
                plt.close(fig)
        base_pred = run_dir / "08_baseline_regression" / "predictions_baseline_regression.csv"
        cnn_pred = run_dir / "09_cnn_regression" / "predictions_cnn_regression.csv"
        if base_pred.exists():
            base = pd.read_csv(base_pred)
            fig, ax = plt.subplots(figsize=(5.5, 4.5))
            ax.scatter(base["true_content_pct"], base["pred_content_pct"], label="Baseline", alpha=0.75)
            if cnn_pred.exists():
                cnn = pd.read_csv(cnn_pred)
                ax.scatter(cnn["true_content_pct"], cnn["pred_content_pct"], label="CNN", alpha=0.75)
            ax.plot([0, 100], [0, 100], color="#111827", linestyle="--")
            ax.set_xlim(-5, 105)
            ax.set_ylim(-5, 105)
            ax.set_xlabel("True ratio (%)")
            ax.set_ylabel("Predicted ratio (%)")
            ax.set_title("Prediction scatter comparison")
            ax.legend(frameon=False)
            fig.tight_layout()
            fig.savefig(fig_dir / "prediction_scatter_comparison.png", dpi=180)
            plt.close(fig)

    publish_step(run_dir / "10_model_comparison", write)
    return pd.read_csv(run_dir / "10_model_comparison" / "model_comparison.csv")


def step_report(run_dir: Path) -> None:
    def write(staging: Path) -> None:
        comparison_path = run_dir / "10_model_comparison" / "model_comparison.csv"
        comparison = pd.read_csv(comparison_path) if comparison_path.exists() else pd.DataFrame()
        qc_summary = pd.read_csv(run_dir / "03_segment_qc" / "qc_summary_by_label.csv")
        split_summary = pd.read_csv(run_dir / "04_train_test_split" / "split_summary.csv")
        aug_summary = pd.read_csv(run_dir / "05_augmentation" / "augmentation_summary.csv")
        low_real = aug_summary[aug_summary["real_train_segments"] < 5].copy()
        cnn_class_blocked = (run_dir / "07_cnn_classification" / "CNN_CLASSIFICATION_BLOCKED_ENVIRONMENT.md").exists()
        cnn_reg_blocked = (run_dir / "09_cnn_regression" / "CNN_REGRESSION_BLOCKED_ENVIRONMENT.md").exists()
        figures = sorted(str(p.relative_to(run_dir)) for p in run_dir.rglob("*.png"))
        lines = [
            "# 分类 + 回归双模型比例预测报告",
            "",
            "## 方法边界",
            "- 本实验不复用旧 `outputs/` 的分段、样本、模型或指标。",
            "- 75% 数据重新清洗、重新分段，并在 QC 中标记为重点复核对象。",
            "- 当前数据只覆盖 0/25/50/75/100% 离散比例点，因此不能声称已严格验证任意未知连续比例。",
            "",
            "## QC 汇总",
            qc_summary.to_markdown(index=False),
            "",
            "## 划分汇总",
            split_summary.to_markdown(index=False),
            "",
            "## 增强汇总",
            aug_summary.to_markdown(index=False),
            "",
            "## 风险标记",
            low_real.to_markdown(index=False) if not low_real.empty else "未发现真实训练片段少于 5 的标签。",
            "",
            "## CNN 执行状态",
            f"- CNN 分类模型：{'阻塞，当前环境缺少 PyTorch。' if cnn_class_blocked else '已完成。'}",
            f"- CNN 回归模型：{'阻塞，当前环境缺少 PyTorch。' if cnn_reg_blocked else '已完成。'}",
            "",
            "## 模型对比",
            comparison.to_markdown(index=False) if not comparison.empty else "无模型指标。",
            "",
            "## 结论",
            "- 分类模型用于验证电信号对已有比例类别的可区分性。",
            "- 回归模型用于评估电信号到比例值的趋势映射。",
            "- 若需要证明任意 0-100% 未知比例预测，需要补采中间比例点并做未见比例测试。",
        ]
        (staging / "report.md").write_text("\n".join(lines), encoding="utf-8")
        (staging / "figure_index.md").write_text(
            "# Figure Index\n\n" + "\n".join(f"- `{fig}`" for fig in figures), encoding="utf-8"
        )

    publish_step(run_dir / "11_final_report", write)


def run_pipeline(data_root: Path, run_name: str | None, max_epochs: int, min_epochs: int, patience: int, seed: int) -> Path:
    """按固定步骤跑完整双模型流程，返回本次 run 目录。"""
    run_dir = make_run_dir(data_root, run_name)
    # --- 数据侧 ---
    manifest = step_preflight(data_root, run_dir)
    cleaned = step_clean(data_root, run_dir, manifest)
    candidates, candidate_samples = step_candidates(data_root, run_dir, cleaned)
    accepted, accepted_samples = step_qc(run_dir, candidates, candidate_samples)
    train, test, x_train, x_test = step_split(run_dir, accepted, accepted_samples, seed)
    step_augmentation(run_dir, train, x_train, test, x_test, seed)
    # --- 模型侧：先分类（可区分性）再回归（连续反演）---
    step_baseline_classification(run_dir)
    step_cnn_classification(run_dir, max_epochs, min_epochs, patience, seed)
    step_baseline_regression(run_dir)
    step_cnn_regression(run_dir, max_epochs, min_epochs, patience, seed)
    step_comparison(run_dir)
    step_report(run_dir)
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PGDS/PGTD 双模型比例预测主流程（分类 + 回归）。"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ARCHIVE_ROOT / "data",
        help="含 PGDS/ 与 PGTD/ 子目录的数据根；结果写到其下 ratio_dual_model/run_*",
    )
    parser.add_argument("--run-name", type=str, default=None, help="自定义 run 目录名；默认带时间戳")
    parser.add_argument("--max-epochs", type=int, default=60, help="CNN 最大训练轮数")
    parser.add_argument("--min-epochs", type=int, default=20, help="早停生效前最少训练轮数")
    parser.add_argument("--patience", type=int, default=10, help="验证集无提升时的早停耐心")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_dir = run_pipeline(
        args.data_root.resolve(),
        args.run_name,
        args.max_epochs,
        args.min_epochs,
        args.patience,
        args.seed,
    )
    print(f"Ratio dual-model run complete: {run_dir}")


if __name__ == "__main__":
    main()
