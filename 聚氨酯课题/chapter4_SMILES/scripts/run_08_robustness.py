from __future__ import annotations

# Adapted for inheritance package: relative paths, no figure export.
from config import (
    ROOT, DATA, STAGE03, LOCKED, REF, OUT,
    MODELING_CSV, PARAM_V1, PARAM_V2, FALLBACK, TASK04_METRICS,
    FINAL_LOCK, FINAL_FEAT_DIR, META_COLS, TARGET_COLS,
)

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

TASK7 = OUT / 'task07'
TASK8 = OUT / 'task08'
TASK8.mkdir(parents=True, exist_ok=True)



TARGETS = [
    'sigma_max_MPa', 'epsilon_break_pct', 'E_MPa',
    'sigma_healed_MPa', 'epsilon_healed_pct', 'E_healed_MPa', 'Tg'
]
DISPLAY_NAME_MAP = {
    'sigma_max_MPa': 'Tensile Strength',
    'sigma_healed_MPa': 'Tensile Strength Healed',
    'epsilon_break_pct': 'Elongation at Break',
    'epsilon_healed_pct': 'Elongation at Break Healed',
    'E_MPa': 'Toughness',
    'E_healed_MPa': 'Toughness Healed',
    'Tg': 'Tg',
}
META = [
    '全局编号', '文献来源', '动态键类型', 'dk_type_label', '硬段1', '软段1', '动态键组分',
    '论文中样品编号', 'data_flag', '_idx_step4', 'usable_for_weighted_smiles'
]


def discover_data_root() -> Path:
    required = [MODELING_CSV, FINAL_LOCK, FINAL_FEAT_DIR]
    if all(p.exists() for p in required):
        return LOCKED
    return TASK7


def build_base_model(name: str) -> Pipeline:
    if name == 'RandomForest':
        return Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', RandomForestRegressor(random_state=42, n_jobs=-1)),
        ])
    if name == 'ExtraTrees':
        return Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', ExtraTreesRegressor(random_state=42, n_jobs=-1)),
        ])
    if name == 'HistGBR':
        return Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', HistGradientBoostingRegressor(random_state=42)),
        ])
    if name == 'SVR_RBF':
        return Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', SVR(kernel='rbf')),
        ])
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', ElasticNet(max_iter=30000, random_state=42)),
    ])


def load_param_store() -> tuple[Dict, Dict]:
    with open(PARAM_V1, 'r', encoding='utf-8') as f:
        p1 = json.load(f)
    with open(PARAM_V2, 'r', encoding='utf-8') as f:
        p2 = json.load(f)
    return p1, p2


def build_selected_model(target: str, selected_stage: str, selected_model: str, p1: Dict, p2: Dict) -> Pipeline:
    model = build_base_model(selected_model)
    key = f'{target}__{selected_model}'

    best_params = {}
    if selected_stage == 'v2':
        best_params = p2.get(key, {}).get('v2_best_params', {})
    elif selected_stage == 'v1':
        best_params = p1.get(key, {})

    if not best_params:
        best_params = p2.get(key, {}).get('v1_best_params', {})
    if not best_params:
        best_params = p1.get(key, {})

    if best_params:
        model.set_params(**best_params)
    return model


def bootstrap_ci_r2(y_true: np.ndarray, y_pred: np.ndarray, n_boot: int = 1200, random_state: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y_true[idx]
        pb = y_pred[idx]
        if np.isclose(np.var(yb), 0.0):
            continue
        vals.append(r2_score(yb, pb))
    if not vals:
        return float('nan'), float('nan')
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return float(lo), float(hi)



def main() -> None:
    data_root = discover_data_root()
    in_file = MODELING_CSV
    final_lock = FINAL_LOCK
    final_feat_dir = FINAL_FEAT_DIR

    tag = data_root.name
    out_root = TASK8 / f'task08_robustness_outputs_{tag}'
    out_fig = out_root / 'residual_diagnostics'
    out_fig.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_file)
    final_df = pd.read_csv(final_lock)
    p1, p2 = load_param_store()

    feature_cols = [c for c in df.columns if c not in set(TARGETS + [m for m in META if m in df.columns])]
    x_all = df[feature_cols].apply(pd.to_numeric, errors='coerce')

    rows: List[Dict] = []
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    for target in TARGETS:
        row = final_df[final_df['target'] == target]
        if len(row) == 0:
            continue

        selected_stage = str(row['selected_stage'].iloc[0])
        selected_model = str(row['selected_model'].iloc[0])

        feat_file = final_feat_dir / f'{target}_selected_features.csv'
        if not feat_file.exists():
            continue
        feat_df = pd.read_csv(feat_file)
        keep_cols = [c for c in feat_df['feature'].tolist() if c in x_all.columns]
        if not keep_cols:
            continue

        y = pd.to_numeric(df[target], errors='coerce')
        mask = y.notna()
        x_t = x_all.loc[mask, keep_cols].reset_index(drop=True)
        y_t = y.loc[mask].reset_index(drop=True)
        if len(y_t) < 30:
            continue

        model = build_selected_model(target, selected_stage, selected_model, p1, p2)
        pred_oof = cross_val_predict(clone(model), x_t, y_t, cv=cv, n_jobs=-1)

        y_np = y_t.to_numpy(dtype=float)
        p_np = np.asarray(pred_oof, dtype=float)
        residual = p_np - y_np

        q95 = float(np.quantile(y_np, 0.95))
        q05 = float(np.quantile(y_np, 0.05))
        span = max(q95 - q05, 1e-8)
        band_01 = 0.10 * span
        band_02 = 0.20 * span

        cov_01 = float(np.mean(np.abs(residual) <= band_01))
        cov_02 = float(np.mean(np.abs(residual) <= band_02))

        r2 = float(r2_score(y_np, p_np))
        rmse = float(root_mean_squared_error(y_np, p_np))
        mae = float(mean_absolute_error(y_np, p_np))
        medae = float(median_absolute_error(y_np, p_np))

        r2_lo, r2_hi = bootstrap_ci_r2(y_np, p_np, n_boot=1200, random_state=42)

        resid_mean = float(np.mean(residual))
        resid_std = float(np.std(residual, ddof=1))
        resid_q05 = float(np.quantile(residual, 0.05))
        resid_q95 = float(np.quantile(residual, 0.95))

        diag_file = out_fig / f'{target}_residual_diag.png'
        plot_residual_diag(target, y_np, p_np, diag_file)

        rows.append({
            'target': target,
            'display_name': DISPLAY_NAME_MAP.get(target, target),
            'selected_stage': selected_stage,
            'selected_model': selected_model,
            'n_samples': int(len(y_np)),
            'n_features': int(len(keep_cols)),
            'oof_r2': r2,
            'oof_r2_ci95_low': r2_lo,
            'oof_r2_ci95_high': r2_hi,
            'oof_rmse': rmse,
            'oof_mae': mae,
            'oof_medae': medae,
            'coverage_q95q05_band_0p1': cov_01,
            'coverage_q95q05_band_0p2': cov_02,
            'band_0p1_abs_width': band_01,
            'band_0p2_abs_width': band_02,
            'residual_mean': resid_mean,
            'residual_std': resid_std,
            'residual_q05': resid_q05,
            'residual_q95': resid_q95,
            'diag_plot': str(diag_file),
        })

        print(f'[Task08-Robust] done target={target}, r2={r2:.4f}, cov0.1={cov_01:.3f}')

    out_df = pd.DataFrame(rows).sort_values('target').reset_index(drop=True)
    out_df.to_csv(out_root / 'task08_robustness_summary.csv', index=False)

    print(f'Task08 robustness finished. source={data_root}')
    print(f'Output dir: {out_root}')


if __name__ == '__main__':
    main()
