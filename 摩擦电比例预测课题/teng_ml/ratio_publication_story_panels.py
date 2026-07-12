"""电信号 Vpp 趋势 / 线性证明相关论文面板（精简归档版）。

这类图回答的是「比例改变是否带来可检测、近似线性的电压响应」，
不是 CNN 的 R2。写论文时不要与深度学习回归 R2 混用。

输入：
- data/vpp_points.csv              各比例局部峰峰值均值/标准差
- data/external_vpp_comparison.csv 外部点上「模型预测 Vpp vs 实测」对比（可选面板）

输出默认写到：outputs/vpp_story_panels/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ARCHIVE_ROOT = Path(__file__).resolve().parents[1]

SYSTEM_COLORS = {"PGDS": "#2563eb", "PGTD": "#dc2626"}
POINT_COLOR = "#111827"
FIT_COLOR = "#f97316"
CAL_COLOR = "#7c3aed"


def linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else np.nan
    return float(slope), float(intercept), float(r2)


def load_points(points_csv: Path) -> pd.DataFrame:
    points = pd.read_csv(points_csv)
    points = points.sort_values(["system", "ratio_pct"]).reset_index(drop=True)
    return points


def build_calibrated_inverse_summary(points: pd.DataFrame) -> pd.DataFrame:
    """用锚点 + 45% 做 Vpp→比例 的简单反演，再验证 10%/60%。"""
    rows: list[dict[str, float | str]] = []
    for system, group in points.groupby("system"):
        train_mask = group["source_batch"].isin(["original_anchor", "boundary_anchor"]) | group[
            "ratio_pct"
        ].eq(45)
        train = group[train_mask].copy()
        test = group[group["ratio_pct"].isin([10, 60]) & group["is_external_experiment"]].copy()
        slope, intercept, r2 = linear_fit(
            train["measured_vpp_mean_v"].to_numpy(float),
            train["ratio_pct"].to_numpy(float),
        )
        for _, row in test.iterrows():
            pred_ratio = slope * float(row["measured_vpp_mean_v"]) + intercept
            rows.append(
                {
                    "system": system,
                    "stage": "after_45pct_calibration",
                    "true_ratio_pct": float(row["ratio_pct"]),
                    "measured_vpp_mean_v": float(row["measured_vpp_mean_v"]),
                    "measured_vpp_sd_v": float(row["measured_vpp_sd_v"]),
                    "pred_ratio_pct": float(np.clip(pred_ratio, 0, 100)),
                    "abs_error_pct": abs(float(np.clip(pred_ratio, 0, 100)) - float(row["ratio_pct"])),
                    "calibration_fit_r2": r2,
                }
            )
    return pd.DataFrame(rows)


def draw_dual_material_linearity(points: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.1), dpi=240)
    for ax, system in zip(axes, ["PGDS", "PGTD"]):
        group = points[points["system"].eq(system)].sort_values("ratio_pct")
        x = group["ratio_pct"].to_numpy(float)
        y = group["measured_vpp_mean_v"].to_numpy(float)
        err = group["measured_vpp_sd_v"].fillna(0).to_numpy(float)
        slope, intercept, r2 = linear_fit(x, y)
        x_fit = np.linspace(x.min(), x.max(), 200)
        y_fit = slope * x_fit + intercept

        ax.plot(x, y, color=SYSTEM_COLORS[system], lw=2.8, marker="o", ms=5.2, label="Actual trend")
        ax.plot(x_fit, y_fit, color=FIT_COLOR, lw=2.4, ls="--", label=f"Linear fit, R2={r2:.3f}")
        ax.errorbar(
            x,
            y,
            yerr=err,
            fmt="none",
            ecolor="#111827",
            elinewidth=1.3,
            capsize=3.5,
            alpha=0.9,
        )
        ax.scatter(x, y, s=40, color=POINT_COLOR, zorder=4, label="Data points")
        cal = group[group["ratio_pct"].eq(45)]
        if not cal.empty:
            ax.scatter(
                cal["ratio_pct"],
                cal["measured_vpp_mean_v"],
                s=170,
                facecolor="white",
                edgecolor=CAL_COLOR,
                lw=2.6,
                zorder=5,
                label="45% calibration",
            )
        ax.set_title(system, fontsize=16, fontweight="bold")
        ax.set_xlabel("Content ratio (%)", fontsize=12)
        ax.set_ylabel("Local-cycle Vpp (V)", fontsize=12)
        ax.grid(True, alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=9.5, loc="best")
    fig.suptitle("Dual-material electrical response for ratio prediction", fontsize=17, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def draw_prediction_validation(summary: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8), dpi=240, sharey=True)
    for ax, system in zip(axes, ["PGDS", "PGTD"]):
        group = summary[summary["system"].eq(system)].sort_values("true_ratio_pct")
        x = np.arange(len(group))
        width = 0.34
        ax.bar(
            x - width / 2,
            group["true_ratio_pct"],
            width,
            color="#111827",
            alpha=0.88,
            label="Experimental ratio",
        )
        ax.bar(
            x + width / 2,
            group["pred_ratio_pct"],
            width,
            color=SYSTEM_COLORS[system],
            alpha=0.85,
            label="Predicted ratio",
        )
        for xi, (_, row) in zip(x, group.iterrows()):
            ax.text(
                xi,
                max(row["true_ratio_pct"], row["pred_ratio_pct"]) + 3,
                f"MAE {row['abs_error_pct']:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9.5,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(v)}%" for v in group["true_ratio_pct"]])
        ax.set_ylim(0, 78)
        ax.set_title(system, fontsize=16, fontweight="bold")
        ax.set_xlabel("Validation formulation", fontsize=12)
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    axes[0].set_ylabel("Ratio (%)", fontsize=12)
    fig.suptitle("After adding 45% calibration: inverse ratio validation", fontsize=17, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def draw_vpp_prediction_comparison(external_csv: Path, out_path: Path) -> None:
    external = pd.read_csv(external_csv)
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8), dpi=240, sharey=False)
    for ax, system in zip(axes, ["PGDS", "PGTD"]):
        group = external[external["system"].eq(system)].sort_values("ratio_pct")
        x = np.arange(len(group))
        width = 0.34
        ax.bar(
            x - width / 2,
            group["model_pred_vpp_v"],
            width,
            color="#9ca3af",
            label="Predicted Vpp",
        )
        ax.bar(
            x + width / 2,
            group["measured_vpp_mean_v"],
            width,
            yerr=group["measured_vpp_sd_v"],
            color=SYSTEM_COLORS[system],
            alpha=0.88,
            capsize=3,
            label="Measured Vpp",
        )
        for xi, (_, row) in zip(x, group.iterrows()):
            ax.text(
                xi,
                max(row["model_pred_vpp_v"], row["measured_vpp_mean_v"]) + 0.65,
                f"{row['error_vs_model_pct']:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9.5,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(v)}%" for v in group["ratio_pct"]])
        ax.set_title(system, fontsize=16, fontweight="bold")
        ax.set_xlabel("Target ratio", fontsize=12)
        ax.set_ylabel("Local-cycle Vpp (V)", fontsize=12)
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=9.5)
    fig.suptitle("Predicted electrical target vs experimental measurement", fontsize=17, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def write_readme(out_dir: Path, inverse: pd.DataFrame) -> None:
    readme = out_dir / "README.md"
    # 用 CSV 文本代替 to_markdown，避免依赖可选包 tabulate
    table_text = inverse.round(4).to_csv(index=False)
    readme.write_text(
        "# Publication story panels\n\n"
        "These figures are generated from local-cycle Vpp summary tables. "
        "They support the electrical-response part of the story and should not be "
        "confused with CNN regression R2.\n\n"
        "## Calibrated inverse summary\n\n"
        "```csv\n"
        + table_text
        + "```\n",
        encoding="utf-8",
    )


def run(points_csv: Path, external_csv: Path | None, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    points = load_points(points_csv)
    inverse = build_calibrated_inverse_summary(points)
    inverse.to_csv(output_dir / "calibrated_inverse_prediction_summary.csv", index=False)
    draw_dual_material_linearity(points, output_dir / "main_panel_dual_material_vpp_linearity.png")
    if external_csv is not None and external_csv.exists():
        draw_vpp_prediction_comparison(external_csv, output_dir / "main_panel_predicted_vs_measured_vpp.png")
    draw_prediction_validation(inverse, output_dir / "main_panel_calibrated_inverse_validation.png")
    write_readme(output_dir, inverse)
    print(f"Vpp story panels written to: {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--points-csv",
        type=Path,
        default=ARCHIVE_ROOT / "data" / "vpp_points.csv",
    )
    parser.add_argument(
        "--external-csv",
        type=Path,
        default=ARCHIVE_ROOT / "data" / "external_vpp_comparison.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ARCHIVE_ROOT / "outputs" / "vpp_story_panels",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args.points_csv.resolve(), args.external_csv.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
