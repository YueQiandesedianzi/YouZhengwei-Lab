"""Path and column conventions for Chapter-4 inheritance package."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
UPSTREAM = DATA / "upstream"
RAW_LIT = UPSTREAM / "00_raw_literature"
MERGE_DIR = UPSTREAM / "01_merge_intermediate"
CLEAN_DIR = UPSTREAM / "02_cleaned_domain"
SMILES_DICT_DIR = UPSTREAM / "03_smiles_dicts"
STAGE02 = DATA / "stage02"
STAGE03 = DATA / "stage03"
LOCKED = DATA / "locked"
REF = ROOT / "results_reference"
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

# Flat copies kept for script compatibility
MODELING_CSV = STAGE03 / "task03_final_features_for_modeling.csv"
STEP2_MERGED = UPSTREAM / "final_merged_dataset.csv"
STEP4_MATRIX = UPSTREAM / "feature_matrix_step4.csv"
PARAM_V1 = REF / "task05" / "task05_best_params_by_target_model.json"
PARAM_V2 = REF / "task05" / "task05_v2_best_params_by_target_model.json"
FALLBACK = REF / "task05" / "task05_v2_selected_best_with_fallback.csv"
TASK04_METRICS = REF / "task04" / "task04_metrics_by_target_model.csv"
FINAL_LOCK = LOCKED / "task07_final_locked_result.csv"
FINAL_FEAT_DIR = LOCKED / "task07_final_feature_sets"

META_COLS = [
    "全局编号", "文献来源", "动态键类型", "dk_type_label", "硬段1", "软段1", "动态键组分",
    "论文中样品编号", "data_flag", "_idx_step4", "usable_for_weighted_smiles",
]
TARGET_COLS = [
    "sigma_max_MPa",
    "epsilon_break_pct",
    "E_MPa",
    "sigma_healed_MPa",
    "epsilon_healed_pct",
    "E_healed_MPa",
    "Tg",
]
