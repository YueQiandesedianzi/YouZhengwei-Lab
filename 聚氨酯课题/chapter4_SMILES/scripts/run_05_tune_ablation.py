from __future__ import annotations

# Adapted for inheritance package: relative paths, no figure export.
from config import (
    ROOT, DATA, STAGE03, LOCKED, REF, OUT,
    MODELING_CSV, PARAM_V1, PARAM_V2, FALLBACK, TASK04_METRICS,
    FINAL_LOCK, FINAL_FEAT_DIR, META_COLS, TARGET_COLS,
)

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet
from sklearn.metrics import make_scorer, r2_score, root_mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

IN_FILE = MODELING_CSV
OUT_DIR = OUT / 'task05'
OUT_DIR.mkdir(parents=True, exist_ok=True)


TARGET_COLS = [
    'sigma_max_MPa',
    'epsilon_break_pct',
    'E_MPa',
    'sigma_healed_MPa',
    'epsilon_healed_pct',
    'E_healed_MPa',
    'Tg',
]
META_COLS = [
    '全局编号', '文献来源', '动态键类型', 'dk_type_label', '硬段1', '软段1', '动态键组分',
    '论文中样品编号', 'data_flag', '_idx_step4', 'usable_for_weighted_smiles'
]

SCORING = {
    'r2': 'r2',
    'rmse': make_scorer(root_mean_squared_error, greater_is_better=False),
    'mae': make_scorer(mean_absolute_error, greater_is_better=False),
}


def build_model_and_space(model_name: str, n_features: int) -> Tuple[Pipeline, Dict[str, List]]:
    if model_name == 'Dummy':
        model = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', DummyRegressor()),
        ])
        space = {
            'model__strategy': ['mean', 'median', 'quantile'],
            'model__quantile': [0.25, 0.5, 0.75],
        }
        return model, space

    if model_name == 'ElasticNet':
        model = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', ElasticNet(max_iter=30000, random_state=42)),
        ])
        space = {
            'model__alpha': np.logspace(-3, 1, 40).tolist(),
            'model__l1_ratio': np.linspace(0.05, 0.95, 19).tolist(),
        }
        return model, space

    if model_name == 'PLS':
        n_max = max(2, min(12, n_features))
        model = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', PLSRegression()),
        ])
        space = {
            'model__n_components': list(range(2, n_max + 1)),
            'model__scale': [True, False],
        }
        return model, space

    if model_name == 'SVR_RBF':
        model = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', SVR(kernel='rbf')),
        ])
        space = {
            'model__C': np.logspace(-1, 2, 25).tolist(),
            'model__epsilon': np.logspace(-3, -0.3, 20).tolist(),
            'model__gamma': ['scale', 'auto'] + np.logspace(-3, 0, 12).tolist(),
        }
        return model, space

    if model_name == 'RandomForest':
        model = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', RandomForestRegressor(random_state=42, n_jobs=-1)),
        ])
        space = {
            'model__n_estimators': [300, 500, 700, 900],
            'model__max_depth': [None, 4, 6, 8, 12, 16],
            'model__min_samples_split': [2, 4, 6, 8, 12],
            'model__min_samples_leaf': [1, 2, 3, 4, 6],
            'model__max_features': ['sqrt', 'log2', 0.4, 0.6, 0.8],
        }
        return model, space

    if model_name == 'ExtraTrees':
        model = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', ExtraTreesRegressor(random_state=42, n_jobs=-1)),
        ])
        space = {
            'model__n_estimators': [400, 700, 1000],
            'model__max_depth': [None, 4, 6, 8, 12, 16],
            'model__min_samples_split': [2, 4, 6, 8, 12],
            'model__min_samples_leaf': [1, 2, 3, 4, 6],
            'model__max_features': ['sqrt', 'log2', 0.4, 0.6, 0.8],
        }
        return model, space

    if model_name == 'HistGBR':
        model = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', HistGradientBoostingRegressor(random_state=42)),
        ])
        space = {
            'model__learning_rate': np.logspace(-2, -0.5, 15).tolist(),
            'model__max_depth': [None, 3, 4, 5, 6, 8, 10],
            'model__max_iter': [200, 300, 400, 600, 800],
            'model__min_samples_leaf': [10, 20, 30, 40, 60],
            'model__l2_regularization': np.logspace(-6, -1, 10).tolist(),
        }
        return model, space

    raise ValueError(f'Unknown model: {model_name}')


def tune_one_model(
    model_name: str,
    x: pd.DataFrame,
    y: pd.Series,
    cv: KFold,
) -> Tuple[Pipeline, dict, float]:
    model, space = build_model_and_space(model_name, n_features=x.shape[1])
    n_space = int(np.prod([len(v) for v in space.values()]))
    n_iter = min(20, n_space)

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=space,
        n_iter=n_iter,
        scoring='r2',
        cv=cv,
        random_state=42,
        n_jobs=-1,
        refit=True,
    )
    search.fit(x, y)
    return search.best_estimator_, search.best_params_, float(search.best_score_)


def evaluate_model_cv(model: Pipeline, x: pd.DataFrame, y: pd.Series, cv: KFold) -> dict:
    cv_res = cross_validate(model, x, y, cv=cv, scoring=SCORING, n_jobs=-1)
    return {
        'r2_mean': float(np.mean(cv_res['test_r2'])),
        'r2_std': float(np.std(cv_res['test_r2'], ddof=1)),
        'rmse_mean': float(-np.mean(cv_res['test_rmse'])),
        'mae_mean': float(-np.mean(cv_res['test_mae'])),
    }


def get_feature_importance(model: Pipeline, x: pd.DataFrame, y: pd.Series, feature_cols: List[str]) -> pd.Series:
    est = clone(model)
    est.fit(x, y)
    final_model = est.named_steps['model']

    if hasattr(final_model, 'feature_importances_'):
        imp = np.asarray(final_model.feature_importances_, dtype=float)
        return pd.Series(imp, index=feature_cols)

    if hasattr(final_model, 'coef_'):
        coef = np.asarray(final_model.coef_, dtype=float)
        if coef.ndim > 1:
            coef = coef.ravel()
        return pd.Series(np.abs(coef), index=feature_cols)

    perm = permutation_importance(
        est,
        x,
        y,
        n_repeats=12,
        random_state=42,
        scoring='r2',
        n_jobs=-1,
    )
    return pd.Series(np.maximum(perm.importances_mean, 0.0), index=feature_cols)


def run_ablation(
    tuned_model: Pipeline,
    x: pd.DataFrame,
    y: pd.Series,
    target: str,
    feature_cols: List[str],
    cv: KFold,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    importance = get_feature_importance(tuned_model, x, y, feature_cols)
    importance = importance.sort_values(ascending=False)

    keep_fracs = [1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2]
    rows = []
    best_row = None

    for frac in keep_fracs:
        k = max(8, int(round(frac * len(feature_cols))))
        keep_cols = list(importance.index[:k])
        model_eval = evaluate_model_cv(clone(tuned_model), x[keep_cols], y, cv)

        row = {
            'target': target,
            'retain_frac': frac,
            'n_features': k,
            **model_eval,
        }
        rows.append(row)

        if (best_row is None) or (row['r2_mean'] > best_row['r2_mean']):
            best_row = row | {'selected_features': keep_cols}

    curve_df = pd.DataFrame(rows)

    selected = best_row['selected_features']
    selected_df = pd.DataFrame({
        'target': target,
        'feature': selected,
        'rank_by_importance': np.arange(1, len(selected) + 1),
        'importance': importance.loc[selected].values,
    })

    summary_df = pd.DataFrame([{
        'target': target,
        'recommended_retain_frac': best_row['retain_frac'],
        'recommended_n_features': best_row['n_features'],
        'recommended_r2_mean': best_row['r2_mean'],
        'recommended_r2_std': best_row['r2_std'],
    }])

    return curve_df, selected_df, summary_df



def main() -> None:
    df = pd.read_csv(IN_FILE)
    task04_df = pd.read_csv(TASK04_METRICS)

    target_cols = [c for c in TARGET_COLS if c in df.columns]
    meta_cols = [c for c in META_COLS if c in df.columns]
    feature_cols = [c for c in df.columns if c not in set(target_cols + meta_cols)]

    x_all = df[feature_cols].apply(pd.to_numeric, errors='coerce')

    model_names = ['Dummy', 'ElasticNet', 'PLS', 'SVR_RBF', 'RandomForest', 'ExtraTrees', 'HistGBR']
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    tuned_rows = []
    param_store = {}
    best_target_rows = []
    ablation_curve_rows = []
    ablation_best_rows = []

    for target in target_cols:
        y = pd.to_numeric(df[target], errors='coerce')
        mask = y.notna()
        x_t = x_all.loc[mask].reset_index(drop=True)
        y_t = y.loc[mask].reset_index(drop=True)

        if len(y_t) < 40:
            continue

        target_model_results = []
        tuned_model_cache = {}

        for model_name in model_names:
            best_est, best_params, best_cv_r2 = tune_one_model(model_name, x_t, y_t, cv)
            eval_res = evaluate_model_cv(best_est, x_t, y_t, cv)

            row = {
                'target': target,
                'model': model_name,
                'tuned_r2_cv': best_cv_r2,
                'r2_mean': eval_res['r2_mean'],
                'r2_std': eval_res['r2_std'],
                'rmse_mean': eval_res['rmse_mean'],
                'mae_mean': eval_res['mae_mean'],
            }
            tuned_rows.append(row)
            target_model_results.append(row)
            tuned_model_cache[model_name] = best_est
            param_store[f'{target}__{model_name}'] = best_params

        target_ranked = sorted(target_model_results, key=lambda d: d['r2_mean'], reverse=True)
        winner = target_ranked[0]
        best_target_rows.append({
            'target': target,
            'best_model_tuned': winner['model'],
            'best_r2_after_tuned': winner['r2_mean'],
            'best_r2_after_tuned_std': winner['r2_std'],
            'best_rmse_after_tuned': winner['rmse_mean'],
            'best_mae_after_tuned': winner['mae_mean'],
        })

        curve_df, selected_df, summary_df = run_ablation(
            tuned_model=tuned_model_cache[winner['model']],
            x=x_t,
            y=y_t,
            target=target,
            feature_cols=feature_cols,
            cv=cv,
        )

        ablation_curve_rows.append(curve_df)
        ablation_best_rows.append(summary_df)
        safe_target = target.replace('/', '_').replace(' ', '_')
        selected_df.to_csv(OUT_DIR / f'task05_selected_features_{safe_target}.csv', index=False)

    tuned_df = pd.DataFrame(tuned_rows)
    best_target_df = pd.DataFrame(best_target_rows)
    ablation_curve_df = pd.concat(ablation_curve_rows, ignore_index=True)
    ablation_best_df = pd.concat(ablation_best_rows, ignore_index=True)

    baseline = (
        task04_df.sort_values(['target', 'r2_mean'], ascending=[True, False])
        .groupby('target', as_index=False)
        .first()[['target', 'model', 'r2_mean']]
        .rename(columns={'model': 'best_model_before', 'r2_mean': 'best_r2_before'})
    )

    before_after = baseline.merge(best_target_df, on='target', how='inner')
    before_after['delta_r2'] = before_after['best_r2_after_tuned'] - before_after['best_r2_before']

    tuned_df.to_csv(OUT_DIR / 'task05_tuned_model_results.csv', index=False)
    best_target_df.to_csv(OUT_DIR / 'task05_best_model_per_target_tuned.csv', index=False)
    ablation_curve_df.to_csv(OUT_DIR / 'task05_ablation_curve.csv', index=False)
    ablation_best_df.to_csv(OUT_DIR / 'task05_ablation_best_subset_summary.csv', index=False)
    before_after.to_csv(OUT_DIR / 'task05_r2_before_after_tuning.csv', index=False)

    with open(OUT_DIR / 'task05_best_params_by_target_model.json', 'w', encoding='utf-8') as f:
        json.dump(param_store, f, ensure_ascii=False, indent=2)

    plot_outputs(tuned_df, before_after, ablation_curve_df, ablation_best_df)

    print('Task05 tuning + ablation finished.')
    print('Output dir:', OUT_DIR)
    print('\nBest tuned model per target:')
    print(best_target_df.to_string(index=False))
    print('\nR2 before vs after tuning:')
    print(before_after[['target', 'best_model_before', 'best_model_tuned', 'best_r2_before', 'best_r2_after_tuned', 'delta_r2']].to_string(index=False))


if __name__ == '__main__':
    main()
