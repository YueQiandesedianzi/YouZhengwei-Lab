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
import shap
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
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
    """Always use the bundled locked snapshot for reproducible inheritance runs."""
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


def transform_with_pipeline(pipeline: Pipeline, x: pd.DataFrame) -> np.ndarray:
    xt = x
    if 'imputer' in pipeline.named_steps:
        xt = pipeline.named_steps['imputer'].transform(xt)
    if 'scaler' in pipeline.named_steps:
        xt = pipeline.named_steps['scaler'].transform(xt)
    return np.asarray(xt, dtype=float)


def normalize_shap_values(shap_values: object) -> np.ndarray:
    if hasattr(shap_values, 'values'):
        arr = np.asarray(shap_values.values)
    else:
        arr = np.asarray(shap_values)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr


def compute_shap_values(selected_model: str, estimator: object, x_trans: np.ndarray, random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    n = x_trans.shape[0]
    eval_n = min(220, n)
    eval_idx = np.sort(rng.choice(n, size=eval_n, replace=False)) if eval_n < n else np.arange(n)
    x_eval = x_trans[eval_idx]

    if selected_model in {'RandomForest', 'ExtraTrees', 'HistGBR'}:
        explainer = shap.TreeExplainer(estimator)
        try:
            sv = explainer.shap_values(x_eval, check_additivity=False)
        except TypeError:
            sv = explainer.shap_values(x_eval)
        return normalize_shap_values(sv), eval_idx

    if selected_model == 'ElasticNet':
        background_n = min(300, n)
        bg_idx = np.sort(rng.choice(n, size=background_n, replace=False)) if background_n < n else np.arange(n)
        explainer = shap.LinearExplainer(estimator, x_trans[bg_idx])
        sv = explainer.shap_values(x_eval)
        return normalize_shap_values(sv), eval_idx

    background_n = min(80, n)
    bg_idx = np.sort(rng.choice(n, size=background_n, replace=False)) if background_n < n else np.arange(n)
    explainer = shap.KernelExplainer(estimator.predict, x_trans[bg_idx])
    sv = explainer.shap_values(x_eval, nsamples=120)
    return normalize_shap_values(sv), eval_idx




def main() -> None:
    data_root = discover_data_root()
    in_file = MODELING_CSV
    final_lock = FINAL_LOCK
    final_feat_dir = FINAL_FEAT_DIR
    tag = data_root.name
    out_root = TASK8 / f'task08_shap_outputs_{tag}'
    out_global = out_root / 'global_importance_plots'
    out_local = out_root / 'local_explanations_plots'
    out_table = out_root / 'tables'
    out_global.mkdir(parents=True, exist_ok=True)
    out_local.mkdir(parents=True, exist_ok=True)
    out_table.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_file)
    final_df = pd.read_csv(final_lock)
    p1, p2 = load_param_store()

    feature_cols = [c for c in df.columns if c not in set(TARGETS + [m for m in META if m in df.columns])]
    x_all = df[feature_cols].apply(pd.to_numeric, errors='coerce')

    summary_rows: List[Dict] = []
    top_rows: List[Dict] = []

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
        model.fit(x_t, y_t)
        x_trans = transform_with_pipeline(model, x_t)
        estimator = model.named_steps['model']

        shap_values, eval_idx = compute_shap_values(selected_model, estimator, x_trans)
        x_eval = x_t.iloc[eval_idx].reset_index(drop=True)
        y_eval = y_t.iloc[eval_idx].reset_index(drop=True)
        pred_eval = pd.Series(model.predict(x_eval), index=y_eval.index)

        mean_abs = np.mean(np.abs(shap_values), axis=0)
        imp_df = pd.DataFrame({
            'target': target,
            'feature': keep_cols,
            'mean_abs_shap': mean_abs,
        }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

        imp_df.to_csv(out_table / f'{target}_shap_global_importance.csv', index=False)
        plot_global_bar(target, imp_df, out_global / f'{target}_shap_global_top15.png', top_n=15)

        for rank, rec in enumerate(imp_df.head(10).itertuples(index=False), start=1):
            top_rows.append({
                'target': target,
                'rank': rank,
                'feature': rec.feature,
                'mean_abs_shap': float(rec.mean_abs_shap),
                'selected_model': selected_model,
            })

        residual = pred_eval - y_eval
        case_map = {
            'best_fit': int(np.argmin(np.abs(residual.to_numpy()))),
            'over_predict': int(np.argmax(residual.to_numpy())),
            'under_predict': int(np.argmin(residual.to_numpy())),
        }

        for case_name, local_i in case_map.items():
            sv_row = shap_values[local_i]
            local_df = pd.DataFrame({
                'feature': keep_cols,
                'feature_value': x_eval.iloc[local_i].to_numpy(),
                'shap_value': sv_row,
                'abs_shap': np.abs(sv_row),
            }).sort_values('abs_shap', ascending=False).reset_index(drop=True)
            local_df.to_csv(out_table / f'{target}_local_{case_name}_contrib.csv', index=False)
            plot_local_case(target, case_name, local_df, out_local / f'{target}_local_{case_name}_top10.png', top_n=10)

        summary_rows.append({
            'target': target,
            'display_name': DISPLAY_NAME_MAP.get(target, target),
            'selected_stage': selected_stage,
            'selected_model': selected_model,
            'n_samples': int(len(y_t)),
            'n_features': int(len(keep_cols)),
            'n_eval_for_shap': int(len(eval_idx)),
            'global_plot': str(out_global / f'{target}_shap_global_top15.png'),
        })

        print(f'[Task08] done target={target}, model={selected_model}, n={len(y_t)}, f={len(keep_cols)}')

    summary_df = pd.DataFrame(summary_rows).sort_values('target').reset_index(drop=True)
    top_df = pd.DataFrame(top_rows).sort_values(['target', 'rank']).reset_index(drop=True)

    summary_df.to_csv(out_root / 'task08_shap_summary.csv', index=False)
    top_df.to_csv(out_root / 'task08_shap_top10_features_all_targets.csv', index=False)

    print(f'Task08 SHAP finished. source={data_root}')
    print(f'Output dir: {out_root}')


if __name__ == '__main__':
    main()
