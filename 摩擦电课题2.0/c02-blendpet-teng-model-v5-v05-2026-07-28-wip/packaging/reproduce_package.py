from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BLUE = "#2F5D8C"
BLUE_LIGHT = "#AFC8DF"
ORANGE = "#D9893D"
ORANGE_LIGHT = "#F2C89F"
INK = "#25313C"
GREY = "#D8DEE4"
GREY_DARK = "#66717D"


def _read(root: Path, relative: str) -> pd.DataFrame:
    return pd.read_csv(root / relative)


def compute_metrics(root: Path) -> dict:
    candidates = _read(root, "data/classification/candidate-summary.csv")
    selected = candidates[(candidates["model"] == "rbf_svm") & (candidates["augmentation"] == "none")].iloc[0]
    macroblocks = _read(root, "data/features/formula-macroblocks.csv")
    regression = _read(root, "data/regression/calibration-predictions.csv")

    truth = regression["z_true"].to_numpy(dtype=float)
    prediction = regression["z_pred"].to_numpy(dtype=float)
    residual = prediction - truth
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((truth - truth.mean()) ** 2))

    family_metrics = {}
    for family, frame in regression.groupby("family", sort=True):
        family_truth = frame["z_true"].to_numpy(dtype=float)
        family_prediction = frame["z_pred"].to_numpy(dtype=float)
        family_residual = family_prediction - family_truth
        family_ss_tot = float(np.sum((family_truth - family_truth.mean()) ** 2))
        family_metrics[str(family)] = {
            "n": int(len(frame)),
            "mae_pp": float(np.mean(np.abs(family_residual))),
            "r2": float(1.0 - np.sum(family_residual**2) / family_ss_tot),
            "mean_signed_error_pp": float(np.mean(family_residual)),
        }

    return {
        "classification": {
            "accuracy": float(selected["held_out_accuracy_mean"]),
            "balanced_accuracy": float(selected["held_out_balanced_accuracy_mean"]),
            "macro_f1": float(selected["held_out_macro_f1_mean"]),
            "macro_f1_sd_across_folds": float(selected["macro_f1_sd_across_folds"]),
            "formula_count": int(macroblocks["canonical_formula"].nunique()),
            "macroblock_count": int(len(macroblocks)),
        },
        "regression": {
            "mae_pp": float(np.mean(np.abs(residual))),
            "rmse_pp": float(np.sqrt(np.mean(residual**2))),
            "r2": float(1.0 - ss_res / ss_tot),
            "mean_signed_error_pp": float(np.mean(residual)),
            "mixture_formula_count": int(len(regression)),
            "by_family": family_metrics,
        },
    }


def _save_figure(fig: plt.Figure, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        kwargs = {"dpi": 220} if suffix == "png" else {}
        fig.savefig(directory / f"{stem}.{suffix}", bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(fig)


def _base_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.edgecolor": GREY_DARK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_workflow(root: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 3.5))
    ax.set_axis_off()
    labels = [
        "TENG waveform",
        "Cycle QC",
        "10 physical\nfeatures",
        "12-cycle robust\naggregation",
        "RBF feature\nspace",
        "15-formulation\nclassification",
    ]
    xs = np.linspace(0.07, 0.93, len(labels))
    for index, (x, label) in enumerate(zip(xs, labels)):
        color = ORANGE_LIGHT if index == len(labels) - 1 else BLUE_LIGHT
        edge = ORANGE if index == len(labels) - 1 else BLUE
        ax.add_patch(
            plt.Rectangle(
                (x - 0.07, 0.44), 0.14, 0.28, transform=ax.transAxes,
                facecolor=color, edgecolor=edge, linewidth=1.5
            )
        )
        ax.text(x, 0.58, label, ha="center", va="center", transform=ax.transAxes, fontsize=10, weight="bold")
        if index < len(labels) - 1:
            ax.annotate(
                "", xy=(xs[index + 1] - 0.075, 0.58), xytext=(x + 0.075, 0.58),
                xycoords=ax.transAxes, textcoords=ax.transAxes,
                arrowprops={"arrowstyle": "->", "color": GREY_DARK, "lw": 1.5}
            )
    ax.add_patch(
        plt.Rectangle((0.76, 0.02), 0.17, 0.20, transform=ax.transAxes,
                      facecolor="white", edgecolor=ORANGE, linewidth=1.5)
    )
    ax.annotate(
        "", xy=(0.845, 0.23), xytext=(0.758, 0.43),
        xycoords=ax.transAxes, textcoords=ax.transAxes,
        arrowprops={"arrowstyle": "->", "color": GREY_DARK, "lw": 1.3}
    )
    ax.text(0.70, 0.28, "Known family", transform=ax.transAxes, ha="center", va="center", fontsize=9)
    ax.text(0.845, 0.12, "Family-conditioned\nratio estimation", ha="center", va="center",
            transform=ax.transAxes, fontsize=9, weight="bold")
    ax.set_title("Physics-informed AI workflow for BlendPET-TENG", loc="left", pad=12, weight="bold")
    ax.text(0.0, 0.90, "Primary endpoint: discrete formulation identification; secondary endpoint: semiquantitative ratio estimation",
            transform=ax.transAxes, color=GREY_DARK)
    _save_figure(fig, root / "figures/main", "01-ai-workflow")


def plot_feature_response(root: Path) -> None:
    data = _read(root, "data/features/formula-feature-summary.csv")
    pgd = data[data["canonical_formula"] == "100PGD"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, family, color, light in zip(axes, ["PGS", "PGT"], [BLUE, ORANGE], [BLUE_LIGHT, ORANGE_LIGHT]):
        frame = pd.concat([pgd, data[data["system_label"] == family]], ignore_index=True).sort_values("z")
        ax.plot(frame["z"], frame["ptp_v_median"], marker="o", color=color, linewidth=2, markersize=6)
        ax.fill_between(frame["z"], frame["ptp_v_q25"], frame["ptp_v_q75"], color=light, alpha=0.45, linewidth=0)
        for row in frame.itertuples():
            ax.annotate(f"{int(row.z)}", (row.z, row.ptp_v_median), xytext=(0, 7), textcoords="offset points",
                        ha="center", fontsize=8, color=INK)
        ax.set_title(f"{family} family")
        ax.set_xlabel(f"{family} fraction (%)")
        ax.grid(axis="y", color=GREY, linewidth=0.7)
        ax.set_xlim(-3, 103)
    axes[0].set_ylabel("Peak-to-peak voltage median (V)")
    fig.suptitle("Composition-dependent waveform response", x=0.08, ha="left", fontsize=13, weight="bold")
    fig.text(0.08, 0.91, "Points are formulation-level medians; ribbons show macroblock interquartile ranges", color=GREY_DARK)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    _save_figure(fig, root / "figures/main", "02-feature-response")


def plot_model_comparison(root: Path) -> None:
    data = _read(root, "data/classification/candidate-summary.csv")
    data = data[data["augmentation"] == "none"].copy()
    order = ["ridge", "hierarchical_extra_trees", "extra_trees", "rbf_svm"]
    names = {"ridge": "Ridge", "hierarchical_extra_trees": "Hierarchical Extra Trees", "extra_trees": "Extra Trees", "rbf_svm": "RBF-SVM"}
    data["rank"] = data["model"].map({name: i for i, name in enumerate(order)})
    data = data.sort_values("rank")
    colors = [BLUE_LIGHT, BLUE_LIGHT, BLUE, ORANGE]
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    bars = ax.barh([names[x] for x in data["model"]], data["held_out_macro_f1_mean"], color=colors, edgecolor=INK, linewidth=0.6)
    ax.errorbar(data["held_out_macro_f1_mean"], np.arange(len(data)), xerr=data["macro_f1_sd_across_folds"],
                fmt="none", ecolor=INK, capsize=3, linewidth=1)
    for bar, value in zip(bars, data["held_out_macro_f1_mean"]):
        ax.text(value + 0.012, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Held-out Macro-F1")
    ax.grid(axis="x", color=GREY, linewidth=0.7)
    fig.suptitle("Classification model comparison", x=0.12, y=0.98, ha="left", fontsize=13, weight="bold")
    fig.text(0.12, 0.91, "No synthetic augmentation; bars show fold mean and SD", color=GREY_DARK)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    _save_figure(fig, root / "figures/main", "03-model-comparison")


def plot_confusion(root: Path) -> None:
    matrix = _read(root, "data/classification/selected-confusion-matrix.csv")
    labels, values = pivot_confusion(matrix)
    fig, ax = plt.subplots(figsize=(10.5, 9.5))
    image = ax.imshow(values, cmap="Blues", vmin=0, vmax=max(1, values.max()))
    threshold = values.max() * 0.55
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if values[i, j] > 0:
                ax.text(j, i, f"{int(values[i, j])}", ha="center", va="center",
                        color="white" if values[i, j] > threshold else INK, fontsize=8)
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=55, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(labels)), labels=labels, fontsize=8)
    ax.set_xlabel("Predicted formulation")
    ax.set_ylabel("True formulation")
    metrics = compute_metrics(root)["classification"]
    fig.suptitle("RBF-SVM classification confusion matrix", x=0.12, y=0.99, ha="left", fontsize=13, weight="bold")
    fig.text(0.12, 0.955, f"15 classes | Macro-F1 = {metrics['macro_f1']:.3f} | Balanced accuracy = {metrics['balanced_accuracy']:.3f}",
             color=GREY_DARK)
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02, label="Macroblocks")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save_figure(fig, root / "figures/main", "04-rbf-svm-classification")


def plot_feature_ablation(root: Path) -> None:
    data = _read(root, "data/classification/feature-robustness-summary-metrics.csv")
    data = data[data["model"] == "rbf_svm"].copy()
    order = ["peak_area_ratio", "amplitude", "shape_only", "full"]
    labels = {"peak_area_ratio": "Peak/area ratios", "amplitude": "Amplitude only", "shape_only": "Shape only", "full": "Full 10-feature set"}
    data["rank"] = data["feature_set"].map({name: i for i, name in enumerate(order)})
    data = data.sort_values("rank")
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    colors = [BLUE_LIGHT, BLUE_LIGHT, BLUE, ORANGE]
    bars = ax.barh([labels[x] for x in data["feature_set"]], data["held_out_macro_f1_mean"], color=colors, edgecolor=INK, linewidth=0.6)
    for bar, value in zip(bars, data["held_out_macro_f1_mean"]):
        ax.text(value + 0.015, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Held-out Macro-F1")
    ax.grid(axis="x", color=GREY, linewidth=0.7)
    fig.suptitle("Feature-set ablation for RBF-SVM", x=0.10, y=0.98, ha="left", fontsize=13, weight="bold")
    fig.text(0.10, 0.91, "The full waveform fingerprint provides the most complete separation", color=GREY_DARK)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    _save_figure(fig, root / "figures/main", "05-feature-ablation")


def pivot_confusion(matrix: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    labels = list(dict.fromkeys(matrix["true_formula"].astype(str).tolist()))
    wide = matrix.pivot(index="true_formula", columns="predicted_formula", values="count")
    wide = wide.reindex(index=labels, columns=labels, fill_value=0).fillna(0)
    return labels, wide.to_numpy(dtype=float)


def plot_regression(root: Path) -> None:
    data = _read(root, "data/regression/calibration-predictions.csv")
    data["signed_error_pp"] = data["z_pred"] - data["z_true"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), gridspec_kw={"height_ratios": [1.05, 1]})
    for col, (family, color) in enumerate([("PGS", BLUE), ("PGT", ORANGE)]):
        frame = data[data["family"] == family].sort_values("z_true")
        ax = axes[0, col]
        ax.plot([0, 100], [0, 100], linestyle="--", color=GREY_DARK, linewidth=1)
        ax.scatter(frame["z_true"], frame["z_pred"], s=58, color=color, edgecolor=INK, linewidth=0.6, zorder=3)
        for row in frame.itertuples():
            ax.annotate(f"{int(row.z_true)}", (row.z_true, row.z_pred), xytext=(4, 5), textcoords="offset points", fontsize=8)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_xlabel("True fraction (%)")
        ax.set_ylabel("Predicted fraction (%)")
        ax.set_title(f"{family}: predicted vs true")
        ax.grid(color=GREY, linewidth=0.7)

        ax = axes[1, col]
        bar_colors = [color if value >= 0 else (BLUE_LIGHT if family == "PGS" else ORANGE_LIGHT) for value in frame["signed_error_pp"]]
        ax.bar(frame["z_true"].astype(int).astype(str), frame["signed_error_pp"], color=bar_colors, edgecolor=INK, linewidth=0.5)
        ax.axhline(0, color=INK, linewidth=1)
        ax.set_xlabel(f"True {family} fraction (%)")
        ax.set_ylabel("Prediction error (percentage points)")
        ax.grid(axis="y", color=GREY, linewidth=0.7)
    metrics = compute_metrics(root)["regression"]
    fig.suptitle("Family-conditioned blend-ratio regression", x=0.07, ha="left", fontsize=13, weight="bold")
    fig.text(0.07, 0.94, f"12 held-out mixture formulations | MAE = {metrics['mae_pp']:.2f} pp | R² = {metrics['r2']:.3f}", color=GREY_DARK)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    _save_figure(fig, root / "figures/diagnostic", "06-family-regression")


def plot_experiment_priority(root: Path) -> None:
    data = _read(root, "data/regression/regression-experiment-priority.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, (family, color) in zip(axes, [("PGS", BLUE), ("PGT", ORANGE)]):
        frame = data[data["family"] == family].sort_values(["action", "fraction_pct"])
        y = frame["priority_score"]
        marker = ["o" if action == "repeat" else "s" for action in frame["action"]]
        for row, mark in zip(frame.itertuples(), marker):
            ax.scatter(row.fraction_pct, row.priority_score, s=95, marker=mark, color=color,
                       edgecolor=INK, linewidth=0.7)
            ax.annotate(f"{int(row.fraction_pct)}%", (row.fraction_pct, row.priority_score),
                        xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 3.5)
        ax.set_xlabel(f"{family} fraction (%)")
        ax.set_title(f"{family} follow-up points")
        ax.grid(axis="y", color=GREY, linewidth=0.7)
        ax.text(0.02, 0.04, "○ repeat existing point\n□ add calibration point", transform=ax.transAxes, fontsize=8, color=GREY_DARK)
    axes[0].set_ylabel("Experimental priority (3 = highest)")
    fig.suptitle("Targeted experiments for quantitative regression", x=0.07, ha="left", fontsize=13, weight="bold")
    fig.text(0.07, 0.91, "Priorities target response compression and large concentration-specific residuals", color=GREY_DARK)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    _save_figure(fig, root / "figures/diagnostic", "07-regression-experiment-priority")


def generate_figures(root: Path) -> None:
    _base_style()
    plot_workflow(root)
    plot_feature_response(root)
    plot_model_comparison(root)
    plot_confusion(root)
    plot_feature_ablation(root)
    plot_regression(root)
    plot_experiment_priority(root)


def _markdown_image_targets(readme: Path) -> set[str]:
    text = readme.read_text(encoding="utf-8")
    return set(re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text))


def validate_package(root: Path) -> list[str]:
    errors: list[str] = []
    required = [
        "README.md",
        "METHODS_AI.md",
        "RESULTS_AI.md",
        "REVIEW_CHECKLIST.md",
        "data/classification/candidate-summary.csv",
        "data/classification/selected-confusion-matrix.csv",
        "data/regression/calibration-predictions.csv",
        "data/features/formula-macroblocks.csv",
        "code/reproduce.py",
    ]
    for relative in required:
        if not (root / relative).exists():
            errors.append(f"missing required file: {relative}")

    readme = root / "README.md"
    if readme.exists():
        targets = _markdown_image_targets(readme)
        for target in sorted(targets):
            if target.startswith(("http://", "https://", "/")):
                errors.append(f"non-portable README image target: {target}")
            elif not (root / target).exists():
                errors.append(f"broken README image target: {target}")
        for png in sorted((root / "figures").rglob("*.png")):
            relative = png.relative_to(root).as_posix()
            if relative not in targets:
                errors.append(f"PNG not embedded in README: {relative}")

    forbidden_patterns = ["/" + "Users" + "/", "C:" + "\\\\" + "Users" + "\\\\"]
    placeholder_pattern = r"\b(?:" + "TO" + "DO|" + "TB" + "D)\\b|" + "待" + "补充"
    for path in sorted(root.rglob("*")) if root.exists() else []:
        if path.is_file() and path.suffix.lower() in {".md", ".py", ".json", ".csv"}:
            text = path.read_text(encoding="utf-8-sig", errors="replace")
            for pattern in forbidden_patterns:
                if pattern in text:
                    errors.append(f"absolute local path in {path.relative_to(root)}")
                    break
            if re.search(placeholder_pattern, text, flags=re.IGNORECASE):
                errors.append(f"unresolved placeholder in {path.relative_to(root)}")
    return errors


def build_manifest(root: Path) -> pd.DataFrame:
    manifest_path = root / "data/manifests/package-manifest.csv"
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path == manifest_path:
            continue
        relative = path.relative_to(root).as_posix()
        if relative.startswith("figures/main/"):
            role = "main_figure"
        elif relative.startswith("figures/supplementary/"):
            role = "supplementary_figure"
        elif relative.startswith("figures/diagnostic/"):
            role = "diagnostic_figure"
        elif relative.startswith("figures/archive/"):
            role = "archived_figure"
        elif relative.startswith("data/"):
            role = "derived_data"
        elif relative.startswith("code/"):
            role = "reproduction_code"
        else:
            role = "documentation"
        rows.append(
            {
                "relative_path": relative,
                "role": role,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    manifest = pd.DataFrame(rows)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce BlendPET-TENG AI metrics and figures")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    metrics = compute_metrics(root)
    (root / "data/manifests").mkdir(parents=True, exist_ok=True)
    (root / "data/manifests/reproduction-metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    generate_figures(root)
    build_manifest(root)
    errors = validate_package(root) if args.validate else []
    print(json.dumps({"metrics": metrics, "validation_errors": errors}, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
