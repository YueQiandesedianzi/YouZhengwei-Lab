"""单独绘制「双材料 Vpp-比例线性关系」证明图（精简归档版）。

用途：证明 PGDS / PGTD 的局部峰峰值随配方比例近似线性变化。
这是材料电信号证据图，不是深度学习模型性能图。

输入默认：data/vpp_points.csv
输出默认：outputs/linearity_proof/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ARCHIVE_ROOT = Path(__file__).resolve().parents[1]
COLORS = {"PGDS": "#2563eb", "PGTD": "#dc2626"}


def fit_linearity(x: np.ndarray, y: np.ndarray) -> dict[str, float | np.ndarray]:
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    residual = y - y_hat
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else np.nan
    rmse = float(np.sqrt(np.mean(residual**2)))
    pearson = float(np.corrcoef(x, y)[0, 1])
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(r2),
        "rmse": rmse,
        "pearson_r": pearson,
        "y_hat": y_hat,
    }


def prediction_band(x: np.ndarray, y: np.ndarray, x_grid: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    n = len(x)
    if n <= 2:
        return np.zeros_like(x_grid)
    y_hat = slope * x + intercept
    residual_sd = np.sqrt(np.sum((y - y_hat) ** 2) / (n - 2))
    x_mean = x.mean()
    sxx = np.sum((x - x_mean) ** 2)
    if sxx == 0:
        return np.zeros_like(x_grid)
    # 约 95% 置信带；点数约 8 时 t≈2.45
    t_value = 2.45
    se = residual_sd * np.sqrt(1 / n + (x_grid - x_mean) ** 2 / sxx)
    return t_value * se


def build_panel(points_csv: Path, out_fig: Path, out_metrics: Path) -> None:
    points = pd.read_csv(points_csv)
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.1), dpi=300)
    metric_rows: list[dict[str, float | str]] = []
    handles, labels = [], []

    for ax, system in zip(axes, ["PGDS", "PGTD"]):
        group = points[points["system"].eq(system)].sort_values("ratio_pct")
        x = group["ratio_pct"].to_numpy(float)
        y = group["measured_vpp_mean_v"].to_numpy(float)
        yerr = group["measured_vpp_sd_v"].fillna(0).to_numpy(float)
        fit = fit_linearity(x, y)
        x_grid = np.linspace(x.min(), x.max(), 240)
        y_grid = fit["slope"] * x_grid + fit["intercept"]
        band = prediction_band(x, y, x_grid, float(fit["slope"]), float(fit["intercept"]))

        ax.fill_between(
            x_grid,
            y_grid - band,
            y_grid + band,
            color=COLORS[system],
            alpha=0.12,
            linewidth=0,
            label="95% fit band",
        )
        ax.plot(x_grid, y_grid, color=COLORS[system], lw=2.8, label="Linear fit")
        ax.plot(x, y, color="#111827", lw=1.4, alpha=0.45, label="_nolegend_")
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="o",
            ms=6.2,
            mfc="white",
            mec=COLORS[system],
            mew=2.0,
            ecolor="#374151",
            elinewidth=1.25,
            capsize=3.5,
            label="Data points",
            zorder=4,
        )

        equation = f"Vpp = {fit['slope']:.3f}x {fit['intercept']:+.3f}"
        if float(fit["slope"]) < 0:
            text_x, text_ha = 0.96, "right"
        else:
            text_x, text_ha = 0.04, "left"
        ax.text(
            text_x,
            0.94,
            f"{equation}\nR2 = {fit['r2']:.3f}\nr = {fit['pearson_r']:.3f}",
            transform=ax.transAxes,
            ha=text_ha,
            va="top",
            fontsize=10.8,
            bbox={"boxstyle": "round,pad=0.32", "facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.88},
        )
        ax.set_title(f"{system} formulation", fontsize=16, fontweight="bold")
        ax.set_xlabel("Content ratio (%)", fontsize=12.5)
        ax.set_ylabel("Local-cycle Vpp (V)", fontsize=12.5)
        ax.grid(True, alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
        if system == "PGDS":
            handles, labels = ax.get_legend_handles_labels()

        metric_rows.append(
            {
                "system": system,
                "n_points": len(group),
                "slope_v_per_pct": fit["slope"],
                "intercept_v": fit["intercept"],
                "r2": fit["r2"],
                "pearson_r": fit["pearson_r"],
                "rmse_v": fit["rmse"],
            }
        )

    fig.suptitle("Linearity of dual-material electrical response", fontsize=18, fontweight="bold")
    fig.legend(handles, labels, frameon=False, fontsize=10, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.01))
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(metric_rows).to_csv(out_metrics, index=False)
    print(f"Linearity panel: {out_fig}")
    print(f"Linearity metrics: {out_metrics}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points-csv", type=Path, default=ARCHIVE_ROOT / "data" / "vpp_points.csv")
    parser.add_argument(
        "--out-fig",
        type=Path,
        default=ARCHIVE_ROOT / "outputs" / "linearity_proof" / "dual_formula_linearity_proof.png",
    )
    parser.add_argument(
        "--out-metrics",
        type=Path,
        default=ARCHIVE_ROOT / "outputs" / "linearity_proof" / "dual_formula_linearity_metrics.csv",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    build_panel(args.points_csv.resolve(), args.out_fig.resolve(), args.out_metrics.resolve())


if __name__ == "__main__":
    main()
