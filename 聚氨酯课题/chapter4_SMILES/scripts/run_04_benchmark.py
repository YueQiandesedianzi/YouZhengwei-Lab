from __future__ import annotations

# Adapted for inheritance package: relative paths, no figure export.
from config import (
    ROOT, DATA, STAGE03, LOCKED, REF, OUT,
    MODELING_CSV, PARAM_V1, PARAM_V2, FALLBACK, TASK04_METRICS,
    FINAL_LOCK, FINAL_FEAT_DIR, META_COLS, TARGET_COLS,
)

import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    median_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.dummy import DummyRegressor

IN_FILE = MODELING_CSV
OUT_DIR = OUT / 'task04'
OUT_DIR.mkdir(parents=True, exist_ok=True)



META_COLS = [
    '全局编号', '文献来源', '动态键类型', 'dk_type_label', '硬段1', '软段1', '动态键组分',
    '论文中样品编号', 'data_flag', '_idx_step4', 'usable_for_weighted_smiles'
]
TARGET_COLS = [
    'sigma_max_MPa',
    'epsilon_break_pct',
    'E_MPa',
    'sigma_healed_MPa',
    'epsilon_healed_pct',
    'E_healed_MPa',
    'Tg',
]


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred)
    denom = np.where(denom == 0, 1e-8, denom)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom))


def build_models(n_features: int) -> Dict[str, Pipeline]:
    pls_components = max(2, min(6, n_features))

    return {
        'Dummy': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', DummyRegressor(strategy='mean')),
        ]),
        'ElasticNet': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', ElasticNet(alpha=0.02, l1_ratio=0.5, max_iter=20000, random_state=42)),
        ]),
        'PLS': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', PLSRegression(n_components=pls_components)),
        ]),
        'SVR_RBF': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', SVR(C=10.0, epsilon=0.1, gamma='scale')),
        ]),
        'RandomForest': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', RandomForestRegressor(
                n_estimators=500,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            )),
        ]),
        'ExtraTrees': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', ExtraTreesRegressor(
                n_estimators=700,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            )),
        ]),
        'HistGBR': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_depth=6,
                max_iter=400,
                random_state=42,
            )),
        ]),
    }


def evaluate_single_target(
    x: pd.DataFrame,
    y: pd.Series,
    target_name: str,
    models: Dict[str, Pipeline],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
    fold_rows: List[dict] = []

    for model_name, model in models.items():
        for fold_idx, (train_idx, test_idx) in enumerate(rkf.split(x), start=1):
            x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            est = clone(model)
            t0 = time.time()
            est.fit(x_train, y_train)
            pred = est.predict(x_test)
            fit_seconds = time.time() - t0

            y_true = y_test.to_numpy(dtype=float)
            y_pred = np.asarray(pred, dtype=float)

            mape_val = float(mean_absolute_percentage_error(y_true, y_pred))
            if not np.isfinite(mape_val):
                mape_val = np.nan

            fold_rows.append({
                'target': target_name,
                'model': model_name,
                'fold': fold_idx,
                'n_train': len(train_idx),
                'n_test': len(test_idx),
                'r2': float(r2_score(y_true, y_pred)),
                'rmse': float(root_mean_squared_error(y_true, y_pred)),
                'mae': float(mean_absolute_error(y_true, y_pred)),
                'medae': float(median_absolute_error(y_true, y_pred)),
                'mape': mape_val,
                'smape': float(smape(y_true, y_pred)),
                'evs': float(explained_variance_score(y_true, y_pred)),
                'fit_seconds': fit_seconds,
            })

    fold_df = pd.DataFrame(fold_rows)

    agg = fold_df.groupby(['target', 'model'], as_index=False).agg(
        n_folds=('fold', 'count'),
        r2_mean=('r2', 'mean'),
        r2_std=('r2', 'std'),
        rmse_mean=('rmse', 'mean'),
        rmse_std=('rmse', 'std'),
        mae_mean=('mae', 'mean'),
        mae_std=('mae', 'std'),
        medae_mean=('medae', 'mean'),
        medae_std=('medae', 'std'),
        mape_mean=('mape', 'mean'),
        mape_std=('mape', 'std'),
        smape_mean=('smape', 'mean'),
        smape_std=('smape', 'std'),
        evs_mean=('evs', 'mean'),
        evs_std=('evs', 'std'),
        fit_seconds_mean=('fit_seconds', 'mean'),
    )

    return agg, fold_df



def main() -> None:
    if not IN_FILE.exists():
        raise FileNotFoundError(f'Input file not found: {IN_FILE}')

    df = pd.read_csv(IN_FILE)

    target_cols = [c for c in TARGET_COLS if c in df.columns]
    meta_cols = [c for c in META_COLS if c in df.columns]

    feature_cols = [
        c for c in df.columns
        if c not in set(target_cols + meta_cols)
    ]

    x_all = df[feature_cols].apply(pd.to_numeric, errors='coerce')

    metrics_list = []
    fold_list = []
    sample_rows = []

    for target in target_cols:
        y_raw = pd.to_numeric(df[target], errors='coerce')
        mask = y_raw.notna()

        x_t = x_all.loc[mask].reset_index(drop=True)
        y_t = y_raw.loc[mask].reset_index(drop=True)

        if len(y_t) < 40:
            continue

        models = build_models(n_features=x_t.shape[1])
        agg_df, fold_df = evaluate_single_target(x_t, y_t, target, models)

        metrics_list.append(agg_df)
        fold_list.append(fold_df)
        sample_rows.append({
            'target': target,
            'n_samples_for_target': int(len(y_t)),
            'n_features': int(x_t.shape[1]),
            'cv_splits': 5,
            'cv_repeats': 3,
        })

    metrics_df = pd.concat(metrics_list, ignore_index=True)
    fold_df = pd.concat(fold_list, ignore_index=True)
    sample_df = pd.DataFrame(sample_rows)

    metrics_df['rank_by_r2_within_target'] = metrics_df.groupby('target')['r2_mean'].rank(ascending=False, method='dense').astype(int)

    best_df = (
        metrics_df.sort_values(['target', 'r2_mean'], ascending=[True, False])
        .groupby('target', as_index=False)
        .first()[['target', 'model', 'r2_mean', 'rmse_mean', 'mae_mean', 'evs_mean']]
        .rename(columns={'model': 'best_model', 'r2_mean': 'best_r2_mean'})
    )

    overall_df = metrics_df.groupby('model', as_index=False).agg(
        r2_mean_overall=('r2_mean', 'mean'),
        rmse_mean_overall=('rmse_mean', 'mean'),
        mae_mean_overall=('mae_mean', 'mean'),
        evs_mean_overall=('evs_mean', 'mean'),
        avg_rank=('rank_by_r2_within_target', 'mean'),
    )

    metrics_df.to_csv(OUT_DIR / 'task04_metrics_by_target_model.csv', index=False)
    fold_df.to_csv(OUT_DIR / 'task04_metrics_fold_level.csv', index=False)
    best_df.to_csv(OUT_DIR / 'task04_target_best_model_summary.csv', index=False)
    overall_df.to_csv(OUT_DIR / 'task04_model_overall_summary.csv', index=False)
    sample_df.to_csv(OUT_DIR / 'task04_target_sample_overview.csv', index=False)

    print('Task04 model benchmark finished.')
    print('Input:', IN_FILE)
    print('Output dir:', OUT_DIR)
    print('\nBest model by target:')
    print(best_df.to_string(index=False))


if __name__ == '__main__':
    main()
