"""外部实验点迭代验证故事（精简归档版）。

论文叙事对应关系：
1. 先用原始五个锚点（0/25/50/75/100%）训练的回归模型，预测新加的 45%
2. 把 45% 片段并入训练集做校准，再预测剩余的 10% 与 60%
3. 输出分阶段 MAE/散点图，供外部验证讨论（不要与原始测试集 R2 混算）

依赖：
- --base-run：主流程 ratio_dual_model 某次 run（需要其中 05_augmentation/arrays/train_augmented.npz）
- --pgds-external-root：含 PGDS/ 子目录的外部数据根（归档默认 data/external）
- --pgtd-external-root：PGTD 外部文件目录（归档默认 data/external_PGTD）
- --vpp-points：可选，用于画电信号趋势支撑图（归档默认 data/vpp_points.csv）
"""

from __future__ import annotations

import argparse
import math
import re
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from teng_ml.ratio_dual_model import (
    RANDOM_SEED,
    StandardScaler,
    clamp_ratio,
    detect_candidate_segments,
    raw_feature_matrix,
    raw_segment_stats,
    repair_time_axis,
    resample_segment,
    try_import_torch,
)


ARCHIVE_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class TrainedRegressor:
    model: object
    raw_scaler: StandardScaler
    history: pd.DataFrame


def col_to_idx(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    idx = 0
    for ch in letters:
        idx = idx * 26 + ord(ch.upper()) - 64
    return idx - 1


def read_xlsx_first_sheet(path: Path) -> pd.DataFrame:
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as zf:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            xml = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in xml.findall("a:si", ns):
                shared.append("".join(t.text or "" for t in si.findall(".//a:t", ns)))
        sheet = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))
        rows: list[list[str | None]] = []
        for row in sheet.findall(".//a:sheetData/a:row", ns):
            values: list[str | None] = []
            for cell in row.findall("a:c", ns):
                idx = col_to_idx(cell.attrib.get("r", "A1"))
                while len(values) <= idx:
                    values.append(None)
                value_node = cell.find("a:v", ns)
                value = None if value_node is None else value_node.text
                if cell.attrib.get("t") == "s" and value is not None:
                    value = shared[int(value)]
                values[idx] = value
            rows.append(values)
    width = max(len(row) for row in rows)
    rows = [row + [None] * (width - len(row)) for row in rows]
    headers = [str(value).strip() if value is not None else f"col{i}" for i, value in enumerate(rows[0])]
    return pd.DataFrame(rows[1:], columns=headers)


def read_signal_file(path: Path) -> tuple[pd.DataFrame, str]:
    if path.suffix.lower() == ".xlsx":
        raw = read_xlsx_first_sheet(path)
    else:
        raw = pd.read_csv(path, encoding="utf-8-sig")
    columns = {str(col).strip().upper(): col for col in raw.columns}
    time_col = columns.get("TIME(S)") or list(raw.columns)[0]
    voltage_col = columns.get("VOLTAGE(V)") or list(raw.columns)[1]
    filter_col = columns.get("FILTER(V)")
    time = pd.to_numeric(raw[time_col], errors="coerce")
    voltage = pd.to_numeric(raw[voltage_col], errors="coerce")
    if filter_col is None:
        filt = pd.Series(np.nan, index=raw.index)
    else:
        filt = pd.to_numeric(raw[filter_col], errors="coerce")
    valid = time.notna() & voltage.notna()
    time_np = time[valid].to_numpy(dtype=np.float64)
    voltage_np = voltage[valid].to_numpy(dtype=np.float64)
    filter_np = filt[valid].to_numpy(dtype=np.float64)
    if len(filter_np) == 0 or np.all(~np.isfinite(filter_np)) or np.nanstd(filter_np) < 1e-12:
        filter_np = voltage_np.copy()
        filter_mode = "mirrored_voltage"
    else:
        filter_np = np.where(np.isfinite(filter_np), filter_np, voltage_np)
        filter_mode = "provided"
    frame = pd.DataFrame({"time_s": time_np, "voltage_v": voltage_np, "filter_v": filter_np})
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    return repair_time_axis(frame).reset_index(drop=True), filter_mode


def parse_ratio(path: Path) -> int:
    match = re.search(r"(10|45|60)", path.name)
    if match is None:
        raise ValueError(f"Cannot parse ratio from {path}")
    return int(match.group(1))


def build_external_segments(pgds_root: Path, pgtd_root: Path) -> tuple[pd.DataFrame, np.ndarray]:
    records: list[dict[str, object]] = []
    samples: list[np.ndarray] = []
    paths: list[tuple[Path, str, str]] = []
    paths.extend((path, "PGDS", "origin_faxiaotian") for path in sorted((pgds_root / "PGDS").glob("*.csv")))
    paths.extend((path, "PGTD", "fa_yuanli_latest") for path in sorted(pgtd_root.glob("*")) if path.is_file())
    for path, system, batch in paths:
        ratio = parse_ratio(path)
        df, filter_mode = read_signal_file(path)
        candidates = detect_candidate_segments(df)
        for idx, candidate in enumerate(candidates):
            segment = df.iloc[int(candidate["start_idx"]) : int(candidate["end_idx"]) + 1]
            peak_rel = int(candidate["peak_idx"]) - int(candidate["start_idx"])
            stats = raw_segment_stats(segment, peak_rel_idx=peak_rel)
            segment_id = f"{batch}_{system}_{ratio}_{idx:04d}"
            records.append(
                {
                    "segment_id": segment_id,
                    "source_file": str(path),
                    "material_system": system,
                    "content_pct": ratio,
                    "source_batch": batch,
                    "filter_mode": filter_mode,
                    "duration_s": candidate["duration_s"],
                    "edge_margin_score": candidate["edge_margin_score"],
                    **stats,
                }
            )
            samples.append(resample_segment(segment))
    if not records:
        raise RuntimeError("No external segments detected")
    return pd.DataFrame(records), np.stack(samples).astype(np.float32)


def load_base_train(base_run: Path) -> dict[str, np.ndarray]:
    data = np.load(base_run / "05_augmentation" / "arrays" / "train_augmented.npz", allow_pickle=True)
    return {key: data[key] for key in data.files}


def merge_train(base: dict[str, np.ndarray], meta: pd.DataFrame, x: np.ndarray) -> dict[str, np.ndarray]:
    raw, names = raw_feature_matrix(meta)
    if list(names) != list(base["raw_feature_names"]):
        raise RuntimeError(f"Raw feature mismatch: {names} vs {base['raw_feature_names']}")
    return {
        "x": np.concatenate([base["x_train"].astype(np.float32), x.astype(np.float32)], axis=0),
        "raw": np.concatenate([base["raw_features_train"].astype(np.float32), raw.astype(np.float32)], axis=0),
        "y": np.concatenate([base["y_content_train"].astype(np.float32), meta["content_pct"].to_numpy(dtype=np.float32)]),
        "ids": np.concatenate([base["train_segment_ids"].astype(str), meta["segment_id"].astype(str).to_numpy()]),
        "raw_feature_names": np.asarray(names, dtype=object),
    }


def base_train_only(base: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "x": base["x_train"].astype(np.float32),
        "raw": base["raw_features_train"].astype(np.float32),
        "y": base["y_content_train"].astype(np.float32),
        "ids": base["train_segment_ids"].astype(str),
        "raw_feature_names": base["raw_feature_names"],
    }


def make_test(meta: pd.DataFrame, x: np.ndarray) -> dict[str, np.ndarray]:
    raw, names = raw_feature_matrix(meta)
    return {
        "x": x.astype(np.float32),
        "raw": raw.astype(np.float32),
        "y": meta["content_pct"].to_numpy(dtype=np.float32),
        "ids": meta["segment_id"].astype(str).to_numpy(),
        "raw_feature_names": np.asarray(names, dtype=object),
    }


def train_regressor(train: dict[str, np.ndarray], validation: dict[str, np.ndarray], max_epochs: int, seed: int) -> TrainedRegressor:
    imported = try_import_torch()
    if imported is None:
        raise RuntimeError("PyTorch is required")
    torch, nn, DataLoader, TensorDataset = imported
    torch.manual_seed(seed)
    raw_scaler = StandardScaler()
    raw_train_np = raw_scaler.fit_transform(train["raw"].astype(np.float64)).astype(np.float32)
    raw_val_np = raw_scaler.transform(validation["raw"].astype(np.float64)).astype(np.float32)
    x_train = torch.tensor(train["x"], dtype=torch.float32)
    raw_train = torch.tensor(raw_train_np, dtype=torch.float32)
    y_train = torch.tensor(train["y"], dtype=torch.float32).view(-1, 1)
    x_val = torch.tensor(validation["x"], dtype=torch.float32)
    raw_val = torch.tensor(raw_val_np, dtype=torch.float32)
    y_val = torch.tensor(validation["y"], dtype=torch.float32).view(-1, 1)
    raw_dim = raw_train.shape[1]

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

        def forward(self, x, raw):
            feat = self.fusion(torch.cat([self.wave_encoder(x), raw], dim=1))
            return 100.0 * torch.sigmoid(self.head(feat))

    model = Regressor()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    loss_fn = nn.HuberLoss(delta=8.0)
    loader = DataLoader(
        TensorDataset(x_train, raw_train, y_train),
        batch_size=32,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    history: list[dict[str, float]] = []
    best_mae = math.inf
    best_state = None
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_x, batch_raw, batch_y in loader:
            optimizer.zero_grad()
            pred = model(batch_x, batch_raw)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * len(batch_x)
        model.eval()
        with torch.no_grad():
            train_pred = model(x_train, raw_train)
            val_pred = model(x_val, raw_val)
            train_loss_eval = float(loss_fn(train_pred, y_train).item())
            val_loss = float(loss_fn(val_pred, y_val).item())
            val_mae = float(torch.mean(torch.abs(val_pred - y_val)).item())
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss / max(1, len(x_train)),
                "train_loss_eval": train_loss_eval,
                "validation_loss": val_loss,
                "validation_mae": val_mae,
            }
        )
        if val_mae < best_mae:
            best_mae = val_mae
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return TrainedRegressor(model=model, raw_scaler=raw_scaler, history=pd.DataFrame(history))


def predict(model_pack: TrainedRegressor, test: dict[str, np.ndarray]) -> np.ndarray:
    imported = try_import_torch()
    if imported is None:
        raise RuntimeError("PyTorch is required")
    torch = imported[0]
    raw = model_pack.raw_scaler.transform(test["raw"].astype(np.float64)).astype(np.float32)
    model_pack.model.eval()
    with torch.no_grad():
        pred = model_pack.model(torch.tensor(test["x"], dtype=torch.float32), torch.tensor(raw, dtype=torch.float32))
    return clamp_ratio(pred.numpy().reshape(-1))


def prediction_frame(meta: pd.DataFrame, pred: np.ndarray, stage: str) -> pd.DataFrame:
    out = meta[["segment_id", "material_system", "content_pct", "source_batch", "source_file"]].copy()
    out["stage"] = stage
    out["true_ratio_pct"] = out["content_pct"].astype(float)
    out["pred_ratio_pct"] = pred
    out["abs_error_pct"] = np.abs(out["pred_ratio_pct"] - out["true_ratio_pct"])
    return out


def summarize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    return (
        predictions.groupby(["stage", "material_system", "true_ratio_pct"])
        .agg(
            n=("segment_id", "size"),
            pred_mean_pct=("pred_ratio_pct", "mean"),
            pred_sd_pct=("pred_ratio_pct", "std"),
            mae_pct=("abs_error_pct", "mean"),
        )
        .reset_index()
    )


def draw_story_figures(
    run_dir: Path,
    external_meta: pd.DataFrame,
    predictions: pd.DataFrame,
    summary: pd.DataFrame,
    vpp_points: Path | None = None,
) -> None:
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    # 电信号趋势支撑图：用已汇总的局部峰峰值，不依赖缺失的「综合数据分析」目录
    latest_points_path = vpp_points or (ARCHIVE_ROOT / "data" / "vpp_points.csv")
    if latest_points_path.exists():
        points = pd.read_csv(latest_points_path)
        fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
        for ax, system in zip(axes, ["PGDS", "PGTD"]):
            g = points[points["system"].eq(system)].sort_values("ratio_pct")
            x = g["ratio_pct"].to_numpy(float)
            y = g["measured_vpp_mean_v"].to_numpy(float)
            ax.errorbar(x, y, yerr=g["measured_vpp_sd_v"].fillna(0), fmt="o", color="#111827", capsize=3, label="Data points")
            ax.plot(x, y, color="#2563eb", linewidth=1.8, label="Observed trend")
            coeff = np.polyfit(x, y, 1)
            xx = np.linspace(max(0, x.min() - 3), min(100, x.max() + 3), 200)
            pred_line = np.polyval(coeff, x)
            r2 = 1 - np.sum((y - pred_line) ** 2) / np.sum((y - y.mean()) ** 2)
            ax.plot(xx, np.polyval(coeff, xx), "--", color="#dc2626", linewidth=1.5, label=f"Linear fit R2={r2:.3f}")
            ax.scatter([45], [g[g["ratio_pct"].eq(45)]["measured_vpp_mean_v"].mean()], s=95, facecolors="none", edgecolors="#f97316", linewidths=2.0, label="Calibration point")
            ax.set_title(system)
            ax.set_xlabel("Content ratio (%)")
            ax.set_ylabel("Local-cycle Vpp (V)")
            ax.grid(alpha=0.16)
            ax.legend(frameon=False, fontsize=8)
        fig.suptitle("Electrical trend supporting iterative ratio prediction", y=1.02)
        fig.tight_layout()
        fig.savefig(fig_dir / "panel_vpp_trend_with_calibration_point.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    def plot_stage_bar(stage: str, title: str, path: Path) -> None:
        g = summary[summary["stage"].eq(stage)].sort_values(["material_system", "true_ratio_pct"])
        labels = [f"{row.material_system}\n{int(row.true_ratio_pct)}%" for row in g.itertuples(index=False)]
        x = np.arange(len(g))
        fig, ax = plt.subplots(figsize=(max(6.2, len(g) * 1.0), 4.6))
        ax.bar(x - 0.18, g["true_ratio_pct"], width=0.36, color="#9ca3af", label="True ratio")
        ax.bar(x + 0.18, g["pred_mean_pct"], width=0.36, yerr=g["pred_sd_pct"].fillna(0), capsize=3, color="#7c3aed", label="Predicted ratio")
        for i, row in enumerate(g.itertuples(index=False)):
            ax.text(i, max(row.true_ratio_pct, row.pred_mean_pct) + 3, f"MAE {row.mae_pct:.1f}%", ha="center", fontsize=8)
        ax.set_xticks(x, labels)
        ax.set_ylim(0, 105)
        ax.set_ylabel("Ratio (%)")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.16)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(path, dpi=220)
        plt.close(fig)

    plot_stage_bar("initial_predict_45", "Initial five-point model predicts 45% calibration point", fig_dir / "panel_initial_prediction_45.png")
    plot_stage_bar("calibrated_predict_10_60", "Calibrated model predicts remaining 10% and 60% points", fig_dir / "panel_calibrated_prediction_10_60.png")

    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    for stage, marker, color, label in [
        ("initial_predict_45", "o", "#f97316", "Initial prediction: 45%"),
        ("calibrated_predict_10_60", "s", "#7c3aed", "After calibration: 10%/60%"),
    ]:
        g = predictions[predictions["stage"].eq(stage)]
        ax.scatter(g["true_ratio_pct"], g["pred_ratio_pct"], s=22, alpha=0.72, marker=marker, color=color, label=label)
    ax.plot([0, 100], [0, 100], "--", color="#111827", linewidth=1.2)
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)
    ax.set_xlabel("True ratio (%)")
    ax.set_ylabel("Predicted ratio (%)")
    ax.set_title("Iterative ratio prediction performance")
    ax.grid(alpha=0.16)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "panel_iterative_prediction_scatter.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5))
    for ax, stage, title in zip(
        axes,
        ["initial_predict_45", "calibrated_predict_10_60"],
        ["Initial model", "Calibrated model"],
    ):
        g = summary[summary["stage"].eq(stage)].sort_values(["material_system", "true_ratio_pct"])
        for system, color in [("PGDS", "#2563eb"), ("PGTD", "#dc2626")]:
            gs = g[g["material_system"].eq(system)]
            ax.errorbar(gs["true_ratio_pct"], gs["pred_mean_pct"], yerr=gs["pred_sd_pct"].fillna(0), fmt="o-", color=color, capsize=3, label=system)
        ax.plot([0, 100], [0, 100], "--", color="#111827", linewidth=1.0)
        ax.set_xlim(0, 70)
        ax.set_ylim(0, 70)
        ax.set_title(title)
        ax.set_xlabel("True ratio (%)")
        ax.set_ylabel("Predicted ratio (%)")
        ax.grid(alpha=0.16)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "panel_iterative_stage_summary.png", dpi=220)
    plt.close(fig)


def run_story(
    base_run: Path,
    pgds_external_root: Path,
    pgtd_external_root: Path,
    output_root: Path,
    run_name: str,
    max_epochs: int,
    seed: int,
    vpp_points: Path | None = None,
) -> Path:
    """跑通「初始五锚点 → 预测45% → 校准 → 预测10/60%」故事并出图。"""
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    external_meta, external_x = build_external_segments(pgds_external_root, pgtd_external_root)
    base = load_base_train(base_run)
    train_initial = base_train_only(base)
    calibration_mask = external_meta["content_pct"].eq(45).to_numpy()
    validation_mask = external_meta["content_pct"].isin([10, 60]).to_numpy()
    meta_45 = external_meta.loc[calibration_mask].copy()
    x_45 = external_x[calibration_mask]
    meta_10_60 = external_meta.loc[validation_mask].copy()
    x_10_60 = external_x[validation_mask]
    test_45 = make_test(meta_45, x_45)
    test_10_60 = make_test(meta_10_60, x_10_60)
    train_calibrated = merge_train(base, meta_45, x_45)

    # 阶段1：仅原始锚点 → 预测 45%
    initial_model = train_regressor(train_initial, test_45, max_epochs=max_epochs, seed=seed)
    initial_pred_45 = predict(initial_model, test_45)
    # 阶段2：并入 45% 校准后 → 预测 10%/60%
    calibrated_model = train_regressor(train_calibrated, test_10_60, max_epochs=max_epochs, seed=seed)
    calibrated_pred_10_60 = predict(calibrated_model, test_10_60)

    pred_df = pd.concat(
        [
            prediction_frame(meta_45, initial_pred_45, "initial_predict_45"),
            prediction_frame(meta_10_60, calibrated_pred_10_60, "calibrated_predict_10_60"),
        ],
        ignore_index=True,
    )
    summary = summarize_predictions(pred_df)

    (run_dir / "tables").mkdir()
    external_meta.to_csv(run_dir / "tables" / "external_segments.csv", index=False, encoding="utf-8-sig")
    pred_df.to_csv(run_dir / "tables" / "iterative_predictions.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(run_dir / "tables" / "iterative_prediction_summary.csv", index=False, encoding="utf-8-sig")
    initial_model.history.to_csv(run_dir / "tables" / "initial_model_training_history.csv", index=False, encoding="utf-8-sig")
    calibrated_model.history.to_csv(run_dir / "tables" / "calibrated_model_training_history.csv", index=False, encoding="utf-8-sig")

    draw_story_figures(run_dir, external_meta, pred_df, summary, vpp_points=vpp_points)
    report = [
        "# Iterative Ratio Prediction Story",
        "",
        "Initial model: original five ratio anchors only.",
        "Calibration point: 45% PGDS and 45% PGTD external segments.",
        "Final validation points: 10% and 60% PGDS/PGTD external segments.",
        "",
        "## Prediction Summary",
        "",
        summary.round(4).to_csv(index=False),
    ]
    (run_dir / "README.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Iterative story run complete: {run_dir}")
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成外部点迭代验证故事图。")
    parser.add_argument(
        "--base-run",
        type=Path,
        required=True,
        help="主流程某次 run 目录，例如 data/ratio_dual_model/run_xxx",
    )
    parser.add_argument(
        "--pgds-external-root",
        type=Path,
        default=ARCHIVE_ROOT / "data" / "external",
        help="含 PGDS/ 子目录的外部数据根",
    )
    parser.add_argument(
        "--pgtd-external-root",
        type=Path,
        default=ARCHIVE_ROOT / "data" / "external_PGTD",
        help="PGTD 外部文件目录（可含 csv / xlsx）",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ARCHIVE_ROOT / "data" / "ratio_dual_model",
        help="故事 run 的父目录",
    )
    parser.add_argument("--run-name", type=str, default="run_iterative_story")
    parser.add_argument(
        "--vpp-points",
        type=Path,
        default=ARCHIVE_ROOT / "data" / "vpp_points.csv",
        help="局部峰峰值汇总表，用于趋势支撑图",
    )
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_story(
        args.base_run.resolve(),
        args.pgds_external_root.resolve(),
        args.pgtd_external_root.resolve(),
        args.output_root.resolve(),
        args.run_name,
        args.max_epochs,
        args.seed,
        vpp_points=args.vpp_points.resolve(),
    )


if __name__ == "__main__":
    main()
