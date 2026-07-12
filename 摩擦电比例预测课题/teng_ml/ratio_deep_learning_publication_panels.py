"""由主流程 run 结果生成深度学习论文面板图（精简归档版）。

读入某次 ratio_dual_model run 目录中的指标/预测 CSV，画出：
- 分类：训练曲线 + 混淆矩阵
- 回归：预测-真实散点（可叠加外部 10/45/60% 点）
- Baseline vs CNN 对比
- 外部验证分阶段面板

注意：这是「出图脚本」，不重新训练。必须先跑完主流程；
若要画外部点，再提供 iterative_story 的 prediction summary。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ARCHIVE_ROOT = Path(__file__).resolve().parents[1]

CONTENT_LABELS = [0, 25, 50, 75, 100]
SYSTEM_COLORS = {"PGDS": "#2563eb", "PGTD": "#dc2626"}
CNN_COLOR = "#2563eb"
BASELINE_COLOR = "#9ca3af"
EXTERNAL_EDGE = "#7c3aed"


def smooth(values: pd.Series, window: int = 7) -> pd.Series:
    if len(values) < window:
        return values
    return values.rolling(window=window, center=True, min_periods=1).mean()


def load_external_predictions(external_summary: Path | None) -> pd.DataFrame:
    if external_summary is None or not external_summary.exists():
        return pd.DataFrame(
            columns=[
                "stage",
                "material_system",
                "true_ratio_pct",
                "n",
                "pred_mean_pct",
                "pred_sd_pct",
                "mae_pct",
            ]
        )
    external = pd.read_csv(external_summary)
    external = external.rename(
        columns={
            "true_ratio_pct": "true_content_pct",
            "pred_mean_pct": "pred_content_pct",
            "pred_sd_pct": "pred_content_sd_pct",
            "mae_pct": "content_abs_error_pct",
        }
    )
    return external


def load_run(run_dir: Path, external_summary: Path | None) -> dict[str, pd.DataFrame]:
    frames = {
        "clf_history": pd.read_csv(run_dir / "07_cnn_classification/training_history.csv"),
        "reg_history": pd.read_csv(run_dir / "09_cnn_regression/training_history.csv"),
        "clf_pred": pd.read_csv(run_dir / "07_cnn_classification/predictions_cnn_classification.csv"),
        "reg_pred": pd.read_csv(run_dir / "09_cnn_regression/predictions_cnn_regression.csv"),
        "clf_metrics": pd.read_csv(run_dir / "07_cnn_classification/metrics_cnn_classification.csv"),
        "reg_metrics": pd.read_csv(run_dir / "09_cnn_regression/metrics_cnn_regression.csv"),
        "model_comparison": pd.read_csv(run_dir / "10_model_comparison/model_comparison.csv"),
        "test_segments": pd.read_csv(run_dir / "04_train_test_split/test_segments.csv"),
    }
    frames["reg_pred"] = frames["reg_pred"].merge(
        frames["test_segments"][["segment_id", "material_system"]],
        on="segment_id",
        how="left",
    )
    frames["external_reg_summary"] = load_external_predictions(external_summary)
    return frames


def confusion_matrix_from_predictions(pred: pd.DataFrame) -> np.ndarray:
    matrix = np.zeros((len(CONTENT_LABELS), len(CONTENT_LABELS)), dtype=int)
    label_to_idx = {label: idx for idx, label in enumerate(CONTENT_LABELS)}
    for _, row in pred.iterrows():
        true = int(row["true_content_pct"])
        predicted = int(row["pred_content_pct"])
        if true in label_to_idx and predicted in label_to_idx:
            matrix[label_to_idx[true], label_to_idx[predicted]] += 1
    return matrix


def draw_classification_panel(data: dict[str, pd.DataFrame], out_path: Path) -> None:
    hist = data["clf_history"]
    pred = data["clf_pred"]
    metrics = data["clf_metrics"].iloc[0]
    matrix = confusion_matrix_from_predictions(pred)

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.7), dpi=240)
    ax = axes[0]
    ax.plot(hist["epoch"], hist["train_content_accuracy"], color="#60a5fa", lw=2.0, label="Train content acc")
    ax.plot(hist["epoch"], hist["test_content_accuracy"], color=CNN_COLOR, lw=2.6, label="Test content acc")
    ax.plot(hist["epoch"], hist["train_system_accuracy"], color="#fca5a5", lw=1.8, ls="--", label="Train system acc")
    ax.plot(hist["epoch"], hist["test_system_accuracy"], color="#dc2626", lw=2.2, ls="--", label="Test system acc")
    ax.set_title("Classification accuracy", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.3, 1.03)
    ax.grid(True, alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8.7)

    ax = axes[1]
    ax.plot(hist["epoch"], smooth(hist["train_loss"]), color="#60a5fa", lw=2.0, label="Train loss")
    ax.plot(hist["epoch"], smooth(hist["test_loss"]), color=CNN_COLOR, lw=2.6, label="Test loss")
    ax.set_title("Classification loss", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross entropy")
    ax.grid(True, alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8.7)

    ax = axes[2]
    im = ax.imshow(matrix, cmap="Blues")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                color="white" if value > matrix.max() * 0.55 else "#111827",
                fontsize=9.3,
                fontweight="bold" if value else "normal",
            )
    ax.set_title(
        f"Content confusion matrix\naccuracy={metrics['content_accuracy']:.3f}",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Predicted ratio (%)")
    ax.set_ylabel("True ratio (%)")
    ax.set_xticks(range(len(CONTENT_LABELS)), CONTENT_LABELS)
    ax.set_yticks(range(len(CONTENT_LABELS)), CONTENT_LABELS)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Count", rotation=270, labelpad=13)

    fig.suptitle("CNN classification model for discrete formulation ratios", fontsize=17, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def draw_regression_panel(data: dict[str, pd.DataFrame], out_path: Path) -> None:
    hist = data["reg_history"]
    pred = data["reg_pred"].copy()
    external = data["external_reg_summary"].copy()
    metrics = data["reg_metrics"].iloc[0]

    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.8), dpi=240)
    ax = axes[0]
    ax.plot(hist["epoch"], smooth(hist["train_loss_eval"]), color="#60a5fa", lw=2.0, label="Train loss")
    ax.plot(hist["epoch"], smooth(hist["test_loss"]), color=CNN_COLOR, lw=2.6, label="Test loss")
    ax2 = ax.twinx()
    ax2.plot(hist["epoch"], smooth(hist["test_mae"]), color="#f97316", lw=2.0, label="Test MAE")
    ax.set_title("Regression training", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Huber loss")
    ax2.set_ylabel("MAE (%)")
    ax.grid(True, alpha=0.22)
    ax.spines[["top"]].set_visible(False)
    ax2.spines[["top"]].set_visible(False)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, frameon=False, fontsize=8.7)

    ax = axes[1]
    for system, group in pred.groupby("material_system", dropna=False):
        color = SYSTEM_COLORS.get(system, "#111827")
        ax.scatter(
            group["true_content_pct"],
            group["pred_content_pct"],
            s=24,
            alpha=0.78,
            color=color,
            edgecolor="white",
            linewidth=0.35,
            label=str(system),
        )
    if not external.empty:
        for system, group in external.groupby("material_system"):
            ax.errorbar(
                group["true_content_pct"],
                group["pred_content_pct"],
                yerr=group["pred_content_sd_pct"],
                fmt="D",
                ms=7.5,
                mfc="white",
                mec=EXTERNAL_EDGE,
                mew=2.0,
                ecolor=EXTERNAL_EDGE,
                elinewidth=1.2,
                capsize=3,
                alpha=0.95,
                label=f"{system} external",
            )
    ax.plot([0, 100], [0, 100], color="#111827", lw=1.8, ls="--", label="Ideal")
    ax.set_title(f"Predicted vs true\nanchor-test R2={metrics['r2']:.4f}", fontsize=14, fontweight="bold")
    ax.set_xlabel("True ratio (%)")
    ax.set_ylabel("Predicted ratio (%)")
    ax.set_xlim(-4, 104)
    ax.set_ylim(-4, 104)
    ax.grid(True, alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8.7, loc="upper left")

    ax = axes[2]
    bins = np.linspace(0, max(16, pred["content_abs_error_pct"].max() + 1), 20)
    ax.hist(pred["content_abs_error_pct"], bins=bins, color=CNN_COLOR, alpha=0.84, edgecolor="white")
    ax.axvline(metrics["mae"], color="#f97316", lw=2.3, label=f"MAE={metrics['mae']:.2f}%")
    ax.axvline(5, color="#111827", lw=1.5, ls="--", label="5% error")
    ax.set_title("Absolute error distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Absolute error (%)")
    ax.set_ylabel("Segment count")
    ax.grid(axis="y", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8.7)

    fig.suptitle("CNN regression model for continuous ratio prediction", fontsize=17, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def draw_external_validation_panel(data: dict[str, pd.DataFrame], out_path: Path) -> None:
    external = data["external_reg_summary"].copy()
    if external.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8), dpi=240, sharey=True)
    for ax, system in zip(axes, ["PGDS", "PGTD"]):
        group = external[external["material_system"].eq(system)].sort_values("true_content_pct")
        x = np.arange(len(group))
        width = 0.34
        ax.bar(
            x - width / 2,
            group["true_content_pct"],
            width,
            color="#111827",
            alpha=0.88,
            label="Experimental ratio",
        )
        ax.bar(
            x + width / 2,
            group["pred_content_pct"],
            width,
            yerr=group["pred_content_sd_pct"],
            color=SYSTEM_COLORS.get(system, CNN_COLOR),
            alpha=0.86,
            capsize=3,
            label="CNN predicted ratio",
        )
        for xi, (_, row) in zip(x, group.iterrows()):
            ax.text(
                xi,
                max(row["true_content_pct"], row["pred_content_pct"]) + 4,
                f"{row['content_abs_error_pct']:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9.2,
            )
        ax.set_title(system, fontsize=15, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(v)}%" for v in group["true_content_pct"]])
        ax.set_xlabel("Later experimental point")
        ax.grid(axis="y", alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8.8, loc="upper left")
    axes[0].set_ylabel("Ratio (%)")
    fig.suptitle("Later 10%/45%/60% experiments included as external validation", fontsize=16.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def draw_model_comparison_panel(data: dict[str, pd.DataFrame], out_path: Path) -> None:
    comparison = data["model_comparison"]
    clf = comparison[comparison["model"].isin(["baseline_classification", "cnn_classification"])].copy()
    reg = comparison[comparison["model"].isin(["baseline_regression", "cnn_regression"])].copy()

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.9), dpi=240)

    ax = axes[0]
    labels = ["Content acc", "System acc", "Combination acc"]
    cols = ["content_accuracy", "system_accuracy", "combination_accuracy"]
    x = np.arange(len(labels))
    width = 0.34
    base = clf[clf["model"].eq("baseline_classification")].iloc[0]
    cnn = clf[clf["model"].eq("cnn_classification")].iloc[0]
    ax.bar(x - width / 2, [base[c] for c in cols], width, color=BASELINE_COLOR, label="Baseline")
    ax.bar(x + width / 2, [cnn[c] for c in cols], width, color=CNN_COLOR, label="CNN")
    ax.set_xticks(x, labels, rotation=12)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Accuracy")
    ax.set_title("Classification comparison", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)

    ax = axes[1]
    labels = ["MAE", "RMSE", "R2", "<=5%"]
    base = reg[reg["model"].eq("baseline_regression")].iloc[0]
    cnn = reg[reg["model"].eq("cnn_regression")].iloc[0]
    values_base = [base["mae"], base["rmse"], base["r2"] * 20, base["within_5_pct"] * 20]
    values_cnn = [cnn["mae"], cnn["rmse"], cnn["r2"] * 20, cnn["within_5_pct"] * 20]
    x = np.arange(len(labels))
    ax.bar(x - width / 2, values_base, width, color=BASELINE_COLOR, label="Baseline")
    ax.bar(x + width / 2, values_cnn, width, color=CNN_COLOR, label="CNN")
    ax.set_xticks(x, labels)
    ax.set_ylabel("MAE/RMSE (%) or scaled score")
    ax.set_title("Regression comparison", fontsize=14, fontweight="bold")
    ax.text(2, max(values_base[2], values_cnn[2]) + 1, "R2 x20", ha="center", fontsize=8.8)
    ax.text(3, max(values_base[3], values_cnn[3]) + 1, "<=5% x20", ha="center", fontsize=8.8)
    ax.grid(axis="y", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)

    fig.suptitle("Baseline vs CNN model performance", fontsize=17, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def write_summary(data: dict[str, pd.DataFrame], out_dir: Path) -> None:
    clf = data["clf_metrics"].iloc[0]
    reg = data["reg_metrics"].iloc[0]
    summary = pd.DataFrame(
        [
            {
                "panel": "classification",
                "primary_metric": "content_accuracy",
                "value": clf["content_accuracy"],
                "secondary": f"system_accuracy={clf['system_accuracy']:.6f}; combination_accuracy={clf['combination_accuracy']:.6f}",
            },
            {
                "panel": "regression",
                "primary_metric": "r2",
                "value": reg["r2"],
                "secondary": f"MAE={reg['mae']:.6f}; RMSE={reg['rmse']:.6f}; within_5_pct={reg['within_5_pct']:.6f}",
            },
        ]
    )
    summary.to_csv(out_dir / "deep_learning_panel_summary.csv", index=False)
    external = data["external_reg_summary"]
    if not external.empty:
        external.to_csv(out_dir / "external_10_45_60_regression_summary.csv", index=False)
    (out_dir / "README.md").write_text(
        "# Deep learning publication panels\n\n"
        "These panels use the verified 200-epoch run and keep classification and regression separate. "
        "The regression R2 here is the CNN model prediction R2, not the electrical Vpp-linearity R2.\n\n"
        "Later 10%/45%/60% experimental points are included as external validation markers when available. "
        "They are not added to the classification confusion matrix because the classification head is defined "
        "only for the original discrete classes 0/25/50/75/100%.\n\n"
        + summary.to_markdown(index=False)
        + "\n",
        encoding="utf-8",
    )


def run(run_dir: Path, output_dir: Path, external_summary: Path | None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_run(run_dir, external_summary)
    draw_classification_panel(data, output_dir / "main_panel_cnn_classification.png")
    draw_regression_panel(data, output_dir / "main_panel_cnn_regression.png")
    draw_external_validation_panel(data, output_dir / "main_panel_external_10_45_60_validation.png")
    draw_model_comparison_panel(data, output_dir / "main_panel_baseline_vs_cnn.png")
    write_summary(data, output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="由主流程 run 结果生成 DL 论文面板图。")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="主流程 run 目录，例如 data/ratio_dual_model/run_xxx",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="出图目录；默认写到 <run-dir>/publication_panels",
    )
    parser.add_argument(
        "--external-summary",
        type=Path,
        default=None,
        help="可选：iterative_story 的 iterative_prediction_summary.csv，用于叠加外部点",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = (args.output_dir or (run_dir / "publication_panels")).resolve()
    external = args.external_summary.resolve() if args.external_summary else None
    run(run_dir, output_dir, external)


if __name__ == "__main__":
    main()
