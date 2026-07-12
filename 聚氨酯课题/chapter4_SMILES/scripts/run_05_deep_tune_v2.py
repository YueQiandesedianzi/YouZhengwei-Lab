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
from sklearn.linear_model import ElasticNet
from sklearn.metrics import make_scorer, mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, RepeatedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

TASK05 = OUT / 'task05'
IN_FILE = MODELING_CSV


TARGET_COLS = [
    'sigma_max_MPa', 'epsilon_break_pct', 'E_MPa',
    'sigma_healed_MPa', 'epsilon_healed_pct', 'E_healed_MPa', 'Tg'
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


def build_space(model_name: str, n_features: int):
    if model_name == 'ElasticNet':
        pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', ElasticNet(max_iter=50000, random_state=42)),
        ])
        space = {
            'model__alpha': np.logspace(-4, 1, 70).tolist(),
            'model__l1_ratio': np.linspace(0.01, 0.99, 60).tolist(),
        }
        n_iter = 60
        return pipe, space, n_iter

    if model_name == 'SVR_RBF':
        pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', SVR(kernel='rbf')),
        ])
        space = {
            'model__C': np.logspace(-2, 3, 80).tolist(),
            'model__epsilon': np.logspace(-4, -0.1, 60).tolist(),
            'model__gamma': ['scale', 'auto'] + np.logspace(-4, 0.5, 50).tolist(),
        }
        n_iter = 70
        return pipe, space, n_iter

    if model_name == 'RandomForest':
        pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', RandomForestRegressor(random_state=42, n_jobs=-1)),
        ])
        space = {
            'model__n_estimators': [400, 700, 1000, 1300],
            'model__max_depth': [None, 6, 8, 12, 16, 22],
            'model__min_samples_split': [2, 3, 4, 6, 8, 12],
            'model__min_samples_leaf': [1, 2, 3, 4, 6],
            'model__max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7, 0.9],
            'model__bootstrap': [True, False],
        }
        n_iter = 60
        return pipe, space, n_iter

    if model_name == 'ExtraTrees':
        pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', ExtraTreesRegressor(random_state=42, n_jobs=-1)),
        ])
        space = {
            'model__n_estimators': [500, 800, 1200, 1500],
            'model__max_depth': [None, 6, 8, 12, 16, 22],
            'model__min_samples_split': [2, 3, 4, 6, 8, 12],
            'model__min_samples_leaf': [1, 2, 3, 4, 6],
            'model__max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7, 0.9],
            'model__bootstrap': [False, True],
        }
        n_iter = 70
        return pipe, space, n_iter

    if model_name == 'HistGBR':
        pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', HistGradientBoostingRegressor(random_state=42)),
        ])
        space = {
            'model__learning_rate': np.logspace(-2.3, -0.2, 60).tolist(),
            'model__max_depth': [None, 3, 4, 5, 6, 8, 10, 12],
            'model__max_iter': [250, 400, 600, 800, 1000],
            'model__min_samples_leaf': [8, 12, 20, 30, 40, 60],
            'model__l2_regularization': np.logspace(-7, -0.2, 30).tolist(),
            'model__max_bins': [128, 192, 255],
        }
        n_iter = 70
        return pipe, space, n_iter

    if model_name == 'PLS':
        nmax = max(2, min(16, n_features))
        pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', PLSRegression()),
        ])
        space = {
            'model__n_components': list(range(2, nmax + 1)),
            'model__scale': [True, False],
        }
        n_iter = min(30, len(space['model__n_components']) * 2)
        return pipe, space, n_iter

    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', DummyRegressor()),
    ])
    space = {
        'model__strategy': ['mean', 'median', 'quantile'],
        'model__quantile': [0.2, 0.5, 0.8],
    }
    return pipe, space, 9


def evaluate(model: Pipeline, x: pd.DataFrame, y: pd.Series) -> dict:
    cv_eval = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
    res = cross_validate(model, x, y, cv=cv_eval, scoring=SCORING, n_jobs=-1)
    return {
        'r2_mean': float(np.mean(res['test_r2'])),
        'r2_std': float(np.std(res['test_r2'], ddof=1)),
        'rmse_mean': float(-np.mean(res['test_rmse'])),
        'mae_mean': float(-np.mean(res['test_mae'])),
    }


def main():
    TASK05.mkdir(parents=True, exist_ok=True)
    src05 = TASK05 if (TASK05 / 'task05_tuned_model_results.csv').exists() else (REF / 'task05')
    df = pd.read_csv(IN_FILE)
    tuned_v1 = pd.read_csv(src05 / 'task05_tuned_model_results.csv')
    with open(src05 / 'task05_best_params_by_target_model.json', 'r', encoding='utf-8') as f:
        old_params = json.load(f)

    feature_cols = [c for c in df.columns if c not in set(TARGET_COLS + [c for c in META_COLS if c in df.columns])]
    x_all = df[feature_cols].apply(pd.to_numeric, errors='coerce')

    deep_rows = []
    deep_best = []
    new_params = {}

    cv_search = RepeatedKFold(n_splits=5, n_repeats=2, random_state=42)

    for target in TARGET_COLS:
        if target not in df.columns:
            continue
        y = pd.to_numeric(df[target], errors='coerce')
        mask = y.notna()
        x_t = x_all.loc[mask].reset_index(drop=True)
        y_t = y.loc[mask].reset_index(drop=True)

        # 使用消融推荐特征子集
        sel_file = src05 / f'task05_selected_features_{target}.csv'
        if not sel_file.exists():
            sel_file = src05 / 'selected_features' / f'task05_selected_features_{target}.csv'
        if sel_file.exists():
            sel = pd.read_csv(sel_file)
            keep = [c for c in sel['feature'].tolist() if c in x_t.columns]
            if len(keep) >= 8:
                x_t = x_t[keep]

        top_models = (
            tuned_v1[tuned_v1['target'] == target]
            .sort_values('r2_mean', ascending=False)
            .head(3)['model']
            .tolist()
        )
        if not top_models:
            continue

        target_rows = []
        for model_name in top_models:
            model, space, n_iter = build_space(model_name, x_t.shape[1])

            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=space,
                n_iter=n_iter,
                scoring='r2',
                cv=cv_search,
                random_state=42,
                n_jobs=-1,
                refit=True,
            )
            search.fit(x_t, y_t)
            best_model = search.best_estimator_
            ev = evaluate(best_model, x_t, y_t)

            row = {
                'target': target,
                'model': model_name,
                'deep_cv_search_best_r2': float(search.best_score_),
                **ev,
                'n_features_used': int(x_t.shape[1]),
            }
            deep_rows.append(row)
            target_rows.append(row)
            new_params[f'{target}__{model_name}'] = {
                'v1_best_params': old_params.get(f'{target}__{model_name}', {}),
                'v2_best_params': search.best_params_,
            }

        best_row = sorted(target_rows, key=lambda d: d['r2_mean'], reverse=True)[0]
        deep_best.append({
            'target': target,
            'best_model_v2': best_row['model'],
            'best_r2_v2': best_row['r2_mean'],
            'best_r2_std_v2': best_row['r2_std'],
            'n_features_used': best_row['n_features_used'],
        })

    deep_df = pd.DataFrame(deep_rows)
    best_df = pd.DataFrame(deep_best)

    v1_best = (
        tuned_v1.sort_values(['target', 'r2_mean'], ascending=[True, False])
        .groupby('target', as_index=False)
        .first()[['target', 'model', 'r2_mean']]
        .rename(columns={'model': 'best_model_v1', 'r2_mean': 'best_r2_v1'})
    )

    gain = v1_best.merge(best_df, on='target', how='inner')
    gain['delta_r2_v2_minus_v1'] = gain['best_r2_v2'] - gain['best_r2_v1']

    deep_df.to_csv(TASK05 / 'task05_v2_deep_tuned_results.csv', index=False)
    best_df.to_csv(TASK05 / 'task05_v2_best_per_target.csv', index=False)
    gain.to_csv(TASK05 / 'task05_v2_gain_vs_v1.csv', index=False)
    with open(TASK05 / 'task05_v2_best_params_by_target_model.json', 'w', encoding='utf-8') as f:
        json.dump(new_params, f, ensure_ascii=False, indent=2)

    # 同步一份到Task04用于“04/05参数调优追踪”
    gain.to_csv((OUT / 'task04' / 'task04_v2_gain_from_task05.csv'), index=False)

    x = np.arange(len(gain))
    ax.set_xticks(x)
    ax.set_xticklabels(gain['target'], rotation=20, ha='right')
    ax.set_ylabel('Best CV mean R2')
    ax.set_title('Task05 v2 Deep Tuning Gain by Target')
    ax.legend(frameon=False)

    # heatmap
    piv = deep_df.pivot(index='target', columns='model', values='r2_mean')
    ax.set_xticks(np.arange(len(piv.columns)))
    ax.set_yticks(np.arange(len(piv.index)))
    ax.set_xticklabels(piv.columns, rotation=30, ha='right')
    ax.set_yticklabels(piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=12)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('Deep-tuned CV R2')
    ax.set_title('Task05 v2 Deep-Tuned R2 Heatmap (Top-3 Models per Target)')

    print('Task05 v2 deep tuning finished.')
    print(gain[['target', 'best_model_v1', 'best_model_v2', 'best_r2_v1', 'best_r2_v2', 'delta_r2_v2_minus_v1']].to_string(index=False))


if __name__ == '__main__':
    main()
