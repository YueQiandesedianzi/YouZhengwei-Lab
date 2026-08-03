from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd


SOURCE_ROOT = Path(__file__).resolve().parents[1]
TARGET_ROOT = SOURCE_ROOT.parent / "blendpet-teng-ai-manuscript-package-v1-review"


DATA_FILES = {
    "artifacts/formula-macroblocks-v5-v05.csv": "data/features/formula-macroblocks.csv",
    "qa/formula-repeatability-v5-v05.csv": "data/features/formula-repeatability.csv",
    "independent-review/formula-classification/candidate-summary.csv": "data/classification/candidate-summary.csv",
    "independent-review/formula-classification/selected-confusion-matrix.csv": "data/classification/selected-confusion-matrix.csv",
    "independent-review/formula-classification/selected-per-class-recall.csv": "data/classification/selected-per-class-recall.csv",
    "independent-review/formula-classification/direct-summary-metrics.csv": "data/classification/direct-summary-metrics.csv",
    "independent-review/formula-classification/direct-fold-metrics.csv": "data/classification/direct-fold-metrics.csv",
    "independent-review/formula-classification/paired-augmentation-effects.csv": "data/classification/paired-augmentation-effects.csv",
    "independent-review/formula-classification/feature-robustness-summary-metrics.csv": "data/classification/feature-robustness-summary-metrics.csv",
    "independent-review/alternative-calibration-predictions.csv": "data/regression/calibration-predictions.csv",
    "independent-review/alternative-calibration-metrics.csv": "data/regression/calibration-metrics.csv",
    "independent-review/family-regression-quick/family-regression-predictions.csv": "data/regression/family-regression-predictions.csv",
    "independent-review/family-regression-quick/family-regression-summary.csv": "data/regression/family-regression-summary.csv",
    "independent-review/family-regression-quick/family-regression-decision.json": "data/regression/family-regression-decision.json",
}


def copy_file(source: Path, target: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def build_formula_summary(target: Path) -> None:
    data = pd.read_csv(target / "data/features/formula-macroblocks.csv")
    feature_columns = [column for column in data.columns if column.endswith("_median")]
    grouped = data.groupby(["canonical_formula", "system_label", "z"], as_index=False)
    rows = []
    for keys, frame in grouped:
        formula, family, fraction = keys
        row = {
            "canonical_formula": formula,
            "system_label": family,
            "z": float(fraction),
            "macroblock_count": int(len(frame)),
        }
        for feature in feature_columns:
            short = feature.removesuffix("_median")
            row[f"{short}_median"] = float(frame[feature].median())
            row[f"{short}_q25"] = float(frame[feature].quantile(0.25))
            row[f"{short}_q75"] = float(frame[feature].quantile(0.75))
        rows.append(row)
    pd.DataFrame(rows).sort_values(["system_label", "z", "canonical_formula"]).to_csv(
        target / "data/features/formula-feature-summary.csv", index=False, encoding="utf-8-sig"
    )


def build_priority_table(target: Path) -> None:
    rows = [
        {"family": "PGS", "fraction_pct": 40, "action": "repeat", "priority_score": 2, "reason": "local underprediction"},
        {"family": "PGS", "fraction_pct": 75, "action": "repeat", "priority_score": 2, "reason": "high-region response compression"},
        {"family": "PGS", "fraction_pct": 90, "action": "repeat", "priority_score": 3, "reason": "largest PGS residual"},
        {"family": "PGS", "fraction_pct": 60, "action": "add", "priority_score": 1, "reason": "bridge 55 to 75 percent"},
        {"family": "PGS", "fraction_pct": 65, "action": "add", "priority_score": 1, "reason": "bridge 55 to 75 percent"},
        {"family": "PGS", "fraction_pct": 80, "action": "add", "priority_score": 2, "reason": "resolve high-region plateau"},
        {"family": "PGS", "fraction_pct": 85, "action": "add", "priority_score": 2, "reason": "resolve high-region plateau"},
        {"family": "PGT", "fraction_pct": 45, "action": "repeat", "priority_score": 3, "reason": "large negative residual"},
        {"family": "PGT", "fraction_pct": 50, "action": "repeat", "priority_score": 3, "reason": "local positive residual"},
        {"family": "PGT", "fraction_pct": 60, "action": "repeat", "priority_score": 3, "reason": "non-monotonic response"},
        {"family": "PGT", "fraction_pct": 75, "action": "repeat", "priority_score": 3, "reason": "largest PGT residual"},
        {"family": "PGT", "fraction_pct": 35, "action": "add", "priority_score": 2, "reason": "bridge 25 to 45 percent"},
        {"family": "PGT", "fraction_pct": 55, "action": "add", "priority_score": 2, "reason": "resolve mid-range ordering"},
        {"family": "PGT", "fraction_pct": 65, "action": "add", "priority_score": 2, "reason": "resolve mid-high plateau"},
        {"family": "PGT", "fraction_pct": 85, "action": "add", "priority_score": 2, "reason": "connect 75 percent to endpoint"},
    ]
    pd.DataFrame(rows).to_csv(
        target / "data/regression/regression-experiment-priority.csv", index=False, encoding="utf-8-sig"
    )


def build_source_summary(target: Path) -> None:
    macroblocks = pd.read_csv(target / "data/features/formula-macroblocks.csv")
    repeatability = pd.read_csv(target / "data/features/formula-repeatability.csv")
    coverage = (
        macroblocks.groupby(["canonical_formula", "system_label", "z"], as_index=False)
        .agg(macroblock_count=("macroblock_id", "size"), cycle_count=("cycle_count", "sum"))
        .merge(
            repeatability[["canonical_formula", "sample_count", "source_file_count", "cycle_count_retained"]],
            on="canonical_formula",
            how="left",
        )
        .sort_values(["system_label", "z", "canonical_formula"])
    )
    manifest_directory = target / "data/manifests"
    manifest_directory.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(manifest_directory / "analysis-unit-summary.csv", index=False, encoding="utf-8-sig")


def copy_figures(target: Path) -> None:
    source_figures = SOURCE_ROOT / "figures"
    for path in sorted(source_figures.iterdir()):
        if path.is_file() and path.suffix.lower() in {".png", ".pdf", ".svg"}:
            copy_file(path, target / "figures/archive" / f"legacy-{path.name}")

    final_classification = SOURCE_ROOT / "independent-review/formula-classification/figures"
    for path in sorted(final_classification.iterdir()):
        if path.is_file() and path.suffix.lower() in {".png", ".pdf"}:
            copy_file(path, target / "figures/supplementary" / f"original-{path.name}")

    cnn_sets = {
        "raw-waveform-cnn-results": "cnn-original",
        "raw-waveform-cnn-results-v1-early-stop": "cnn-early-stop",
    }
    base = SOURCE_ROOT / "independent-review/formula-classification"
    for directory, prefix in cnn_sets.items():
        for path in sorted((base / directory).glob("*.png")):
            copy_file(path, target / "figures/archive" / f"{prefix}-{path.name}")


def build_package() -> Path:
    target = TARGET_ROOT
    target.mkdir(parents=True, exist_ok=True)
    for source_relative, target_relative in DATA_FILES.items():
        copy_file(SOURCE_ROOT / source_relative, target / target_relative)

    build_formula_summary(target)
    build_priority_table(target)
    build_source_summary(target)
    copy_figures(target)
    copy_file(SOURCE_ROOT / "packaging/reproduce_package.py", target / "code/reproduce.py")

    requirements = "matplotlib>=3.5\nnumpy>=1.22\npandas>=1.5\n"
    (target / "code/requirements.txt").write_text(requirements, encoding="utf-8")
    scope = {
        "package": target.name,
        "raw_waveforms_included": False,
        "reproduction_scope": "reported metrics and package-owned figures from derived tables",
        "primary_endpoint": "15-formulation classification",
        "secondary_endpoint": "known-family semiquantitative blend-ratio regression",
    }
    (target / "data/manifests/package-scope.json").write_text(
        json.dumps(scope, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return target


if __name__ == "__main__":
    print(build_package())
