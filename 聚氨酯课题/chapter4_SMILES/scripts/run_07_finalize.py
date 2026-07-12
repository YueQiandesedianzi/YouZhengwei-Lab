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
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import make_scorer, root_mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

TASK_OUT = OUT / 'task07'
TASK_OUT.mkdir(parents=True, exist_ok=True)
(TASK_OUT / 'task07_pruned_feature_sets').mkdir(parents=True, exist_ok=True)
(TASK_OUT / 'task07_final_feature_sets').mkdir(parents=True, exist_ok=True)

IN_FILE = MODELING_CSV


TARGETS = [
    'sigma_max_MPa', 'epsilon_break_pct', 'E_MPa',
    'sigma_healed_MPa', 'epsilon_healed_pct', 'E_healed_MPa', 'Tg'
]
META = [
    '全局编号', '文献来源', '动态键类型', 'dk_type_label', '硬段1', '软段1', '动态键组分',
    '论文中样品编号', 'data_flag', '_idx_step4', 'usable_for_weighted_smiles'
]

SCORING = {
    'r2': 'r2',
    'rmse': make_scorer(root_mean_squared_error, greater_is_better=False),
    'mae': make_scorer(mean_absolute_error, greater_is_better=False),
}


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


def evaluate_cv(model: Pipeline, x: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    out = cross_validate(clone(model), x, y, cv=cv, scoring=SCORING, n_jobs=-1)
    return {
        'r2': float(np.mean(out['test_r2'])),
        'rmse': float(-np.mean(out['test_rmse'])),
        'mae': float(-np.mean(out['test_mae'])),
    }


def norm_series_high_better(series: pd.Series) -> pd.Series:
    vmin = float(series.min())
    vmax = float(series.max())
    if np.isclose(vmax, vmin):
        return pd.Series(np.ones(len(series)), index=series.index)
    return (series - vmin) / (vmax - vmin)


def norm_series_low_better(series: pd.Series) -> pd.Series:
    vmin = float(series.min())
    vmax = float(series.max())
    if np.isclose(vmax, vmin):
        return pd.Series(np.ones(len(series)), index=series.index)
    return (vmax - series) / (vmax - vmin)


def pick_final_by_error_priority(curve: pd.DataFrame) -> pd.Series:
    d = curve.copy()
    d['score_r2'] = norm_series_high_better(d['r2'])
    d['score_rmse'] = norm_series_low_better(d['rmse'])
    d['score_mae'] = norm_series_low_better(d['mae'])
    d['score_final'] = 0.20 * d['score_r2'] + 0.40 * d['score_rmse'] + 0.40 * d['score_mae']
    return d.sort_values(['score_final', 'r2'], ascending=[False, False]).iloc[0]


def run_target_ablation(target: str, x: pd.DataFrame, y: pd.Series, model: Pipeline) -> tuple[pd.DataFrame, Dict]:
    imputed = SimpleImputer(strategy='median').fit_transform(x)
    mi = mutual_info_regression(imputed, y, random_state=42)
    rank = pd.Series(mi, index=x.columns).sort_values(ascending=False)

    keep_fracs = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.33, 0.25, 0.2, 0.15]
    rows = []
    best = None

    for frac in keep_fracs:
        k = max(8, int(round(frac * len(rank))))
        k = min(k, len(rank))
        cols = list(rank.index[:k])
        met = evaluate_cv(model, x[cols], y)
        row = {
            'target': target,
            'retain_frac': frac,
            'n_features': len(cols),
            'r2': met['r2'],
            'rmse': met['rmse'],
            'mae': met['mae'],
        }
        rows.append(row)
        if (best is None) or (row['r2'] > best['r2']):
            best = row | {'selected_features': cols}

    curve = pd.DataFrame(rows)
    return curve, best



def main() -> None:
    df = pd.read_csv(IN_FILE)
    fb = pd.read_csv(FALLBACK)
    p1, p2 = load_param_store()

    feature_cols = [c for c in df.columns if c not in set(TARGETS + [m for m in META if m in df.columns])]
    x_all = df[feature_cols].apply(pd.to_numeric, errors='coerce')

    summary_rows = []
    curve_list: List[pd.DataFrame] = []

    for target in TARGETS:
        if target not in df.columns:
            continue

        y = pd.to_numeric(df[target], errors='coerce')
        mask = y.notna()
        x_t = x_all.loc[mask].reset_index(drop=True)
        y_t = y.loc[mask].reset_index(drop=True)

        if len(y_t) < 40:
            continue

        selected = fb[fb['target'] == target]
        selected_stage = selected['selected_stage'].iloc[0] if len(selected) else 'v1'
        selected_model = selected['selected_model'].iloc[0] if len(selected) else 'ExtraTrees'

        model = build_selected_model(target, selected_stage, selected_model, p1, p2)

        full_met = evaluate_cv(model, x_t, y_t)
        curve, best = run_target_ablation(target, x_t, y_t, model)
        final_row = pick_final_by_error_priority(curve)

        final_n = int(final_row['n_features'])
        imputed = SimpleImputer(strategy='median').fit_transform(x_t)
        mi = mutual_info_regression(imputed, y_t, random_state=42)
        rank = pd.Series(mi, index=x_t.columns).sort_values(ascending=False)
        final_features = list(rank.index[:final_n])

        summary_rows.append({
            'target': target,
            'selected_stage': selected_stage,
            'selected_model': selected_model,
            'n_features_full': int(x_t.shape[1]),
            'r2_full': full_met['r2'],
            'rmse_full': full_met['rmse'],
            'mae_full': full_met['mae'],
            'n_features_best_r2': int(best['n_features']),
            'r2_best_r2': float(best['r2']),
            'rmse_best_r2': float(best['rmse']),
            'mae_best_r2': float(best['mae']),
            'retain_frac_best_r2': float(best['retain_frac']),
            'n_features_final': final_n,
            'retain_frac_final': float(final_row['retain_frac']),
            'r2_final': float(final_row['r2']),
            'rmse_final': float(final_row['rmse']),
            'mae_final': float(final_row['mae']),
            'delta_r2_final_minus_full': float(final_row['r2'] - full_met['r2']),
            'delta_rmse_final_minus_full': float(final_row['rmse'] - full_met['rmse']),
            'delta_mae_final_minus_full': float(final_row['mae'] - full_met['mae']),
            'selection_rule': '0.2*R2 + 0.4*RMSE + 0.4*MAE (normalized, RMSE/MAE lower better)',
        })
        curve_list.append(curve)

        feat_df = pd.DataFrame({
            'target': target,
            'selected_rank': np.arange(1, len(best['selected_features']) + 1),
            'feature': best['selected_features'],
        })
        feat_df.to_csv(TASK_OUT / 'task07_pruned_feature_sets' / f'{target}_selected_features.csv', index=False)

        final_feat_df = pd.DataFrame({
            'target': target,
            'selected_rank': np.arange(1, len(final_features) + 1),
            'feature': final_features,
        })
        final_feat_df.to_csv(TASK_OUT / 'task07_final_feature_sets' / f'{target}_selected_features.csv', index=False)

    summary = pd.DataFrame(summary_rows).sort_values('target').reset_index(drop=True)
    curves = pd.concat(curve_list, ignore_index=True) if curve_list else pd.DataFrame()

    summary.to_csv(TASK_OUT / 'task07_full_vs_pruned_summary.csv', index=False)
    summary.to_csv(TASK_OUT / 'task07_final_locked_result.csv', index=False)
    curves.to_csv(TASK_OUT / 'task07_ablation_curve_all.csv', index=False)

    if len(summary) > 0:
        plot_outputs(summary)

    print('Task07 final current-pipeline ablation (R2/RMSE/MAE) finished.')
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
