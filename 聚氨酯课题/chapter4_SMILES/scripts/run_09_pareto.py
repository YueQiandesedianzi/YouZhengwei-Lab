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
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

TASK7 = OUT / 'task07'
TASK8 = OUT / 'task08'
TASK9 = OUT / 'task09'
TASK9.mkdir(parents=True, exist_ok=True)


TARGETS = [
    'sigma_max_MPa', 'epsilon_break_pct', 'E_MPa',
    'sigma_healed_MPa', 'epsilon_healed_pct', 'E_healed_MPa', 'Tg'
]
META = [
    '全局编号', '文献来源', '动态键类型', 'dk_type_label', '硬段1', '软段1', '动态键组分',
    '论文中样品编号', 'data_flag', '_idx_step4', 'usable_for_weighted_smiles'
]

OBJ_A = 'sigma_max_MPa'
OBJ_B = 'epsilon_break_pct'

POP = 180
GEN = 90
ETA_C = 15
ETA_M = 20
SEED = 42


def discover_snapshot_root() -> Path:
    # Inheritance package: locked snapshot is pre-bundled under data/locked
    required = [MODELING_CSV, FINAL_LOCK, FINAL_FEAT_DIR]
    if all(p.exists() for p in required):
        return LOCKED
    raise FileNotFoundError('Locked snapshot incomplete under data/locked')



def build_base_model(name: str) -> Pipeline:
    if name == 'RandomForest':
        return Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', RandomForestRegressor(random_state=SEED, n_jobs=-1)),
        ])
    if name == 'ExtraTrees':
        return Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', ExtraTreesRegressor(random_state=SEED, n_jobs=-1)),
        ])
    if name == 'HistGBR':
        return Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', HistGradientBoostingRegressor(random_state=SEED)),
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
        ('model', ElasticNet(max_iter=30000, random_state=SEED)),
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


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a <= b) and np.any(a < b))


def fast_non_dominated_sort(F: np.ndarray) -> List[List[int]]:
    n = len(F)
    dominates_set: List[List[int]] = [[] for _ in range(n)]
    dominated_count = [0] * n
    fronts: List[List[int]] = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates(F[p], F[q]):
                dominates_set[p].append(q)
            elif dominates(F[q], F[p]):
                dominated_count[p] += 1
        if dominated_count[p] == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front: List[int] = []
        for p in fronts[i]:
            for q in dominates_set[p]:
                dominated_count[q] -= 1
                if dominated_count[q] == 0:
                    next_front.append(q)
        fronts.append(next_front)
        i += 1
    return fronts[:-1]


def crowding_distance(F: np.ndarray, front: List[int]) -> Dict[int, float]:
    if len(front) <= 2:
        return {i: float('inf') for i in front}

    dist = {i: 0.0 for i in front}
    n_obj = F.shape[1]
    for obj in range(n_obj):
        values = sorted([(F[i, obj], i) for i in front])
        dist[values[0][1]] = float('inf')
        dist[values[-1][1]] = float('inf')
        value_range = values[-1][0] - values[0][0] + 1e-12
        for k in range(1, len(values) - 1):
            dist[values[k][1]] += (values[k + 1][0] - values[k - 1][0]) / value_range
    return dist


def sbx_crossover(parents: np.ndarray, xl: np.ndarray, xu: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    pop, n_var = parents.shape
    children = np.empty_like(parents)
    for i in range(0, pop, 2):
        p1 = parents[i]
        p2 = parents[min(i + 1, pop - 1)]
        u = rng.random(n_var)
        beta = np.where(
            u < 0.5,
            (2 * u + 1e-12) ** (1 / (ETA_C + 1)),
            (1 / (2 * (1 - u) + 1e-12)) ** (1 / (ETA_C + 1)),
        )
        children[i] = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
        if i + 1 < pop:
            children[i + 1] = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
    return np.clip(children, xl, xu)


def polynomial_mutation(x: np.ndarray, xl: np.ndarray, xu: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    pop, n_var = x.shape
    pm = 1.0 / n_var
    mask = rng.random((pop, n_var)) < pm
    u = rng.random((pop, n_var))
    delta = np.where(
        u < 0.5,
        (2 * u) ** (1 / (ETA_M + 1)) - 1,
        1 - (2 * (1 - u)) ** (1 / (ETA_M + 1)),
    )
    out = np.where(mask, x + delta * (xu - xl), x)
    return np.clip(out, xl, xu)


def snap_discrete_columns(x: np.ndarray, discrete_values: Dict[int, np.ndarray]) -> np.ndarray:
    out = x.copy()
    for col_i, values in discrete_values.items():
        if len(values) <= 1:
            continue
        v = values.reshape(1, -1)
        c = out[:, col_i].reshape(-1, 1)
        nearest = np.argmin(np.abs(c - v), axis=1)
        out[:, col_i] = values[nearest]
    return out


def objective_values(x_df: pd.DataFrame, model_a: Pipeline, cols_a: List[str], model_b: Pipeline, cols_b: List[str]) -> np.ndarray:
    pred_a = np.asarray(model_a.predict(x_df[cols_a]), dtype=float)
    pred_b = np.asarray(model_b.predict(x_df[cols_b]), dtype=float)
    pred_a = np.clip(pred_a, 0, None)
    pred_b = np.clip(pred_b, 0, None)
    return np.column_stack([-pred_a, -pred_b])


def select_representatives(df: pd.DataFrame, col_a: str, col_b: str) -> pd.DataFrame:
    work = df.copy()
    eps = 1e-12
    work['_na'] = (work[col_a] - work[col_a].min()) / (work[col_a].max() - work[col_a].min() + eps)
    work['_nb'] = (work[col_b] - work[col_b].min()) / (work[col_b].max() - work[col_b].min() + eps)
    work['_balanced'] = work['_na'] + work['_nb']
    work['_knee_dist'] = np.sqrt((1 - work['_na']) ** 2 + (1 - work['_nb']) ** 2)

    rows = []
    rows.append(('max_obj_a', work.loc[work[col_a].idxmax()]))
    rows.append(('max_obj_b', work.loc[work[col_b].idxmax()]))
    rows.append(('balanced_max_sum', work.loc[work['_balanced'].idxmax()]))
    rows.append(('knee_closest_ideal', work.loc[work['_knee_dist'].idxmin()]))

    out = []
    for tag, row in rows:
        item = row.drop(labels=['_na', '_nb', '_balanced', '_knee_dist']).to_dict()
        item['representative_type'] = tag
        out.append(item)
    return pd.DataFrame(out)


def main() -> None:
    snapshot = discover_snapshot_root()
    in_file = MODELING_CSV
    final_lock = FINAL_LOCK
    final_feat_dir = FINAL_FEAT_DIR

    out_root = TASK9 / f'task09_pareto_outputs_{snapshot.name}'
    out_root.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_file)
    final_df = pd.read_csv(final_lock)
    p1, p2 = load_param_store()

    feature_cols = [c for c in df.columns if c not in set(TARGETS + [m for m in META if m in df.columns])]
    x_all = df[feature_cols].apply(pd.to_numeric, errors='coerce')

    row_a = final_df[final_df['target'] == OBJ_A].iloc[0]
    row_b = final_df[final_df['target'] == OBJ_B].iloc[0]

    cols_a = pd.read_csv(final_feat_dir / f'{OBJ_A}_selected_features.csv')['feature'].tolist()
    cols_b = pd.read_csv(final_feat_dir / f'{OBJ_B}_selected_features.csv')['feature'].tolist()
    cols_a = [c for c in cols_a if c in x_all.columns]
    cols_b = [c for c in cols_b if c in x_all.columns]

    union_cols = sorted(set(cols_a) | set(cols_b))

    y_a = pd.to_numeric(df[OBJ_A], errors='coerce')
    y_b = pd.to_numeric(df[OBJ_B], errors='coerce')
    mask_ab = y_a.notna() & y_b.notna()
    x_train = x_all.loc[mask_ab, union_cols].reset_index(drop=True)
    y_train_a = y_a.loc[mask_ab].reset_index(drop=True)
    y_train_b = y_b.loc[mask_ab].reset_index(drop=True)

    model_a = build_selected_model(OBJ_A, str(row_a['selected_stage']), str(row_a['selected_model']), p1, p2)
    model_b = build_selected_model(OBJ_B, str(row_b['selected_stage']), str(row_b['selected_model']), p1, p2)
    model_a.fit(x_train[cols_a], y_train_a)
    model_b.fit(x_train[cols_b], y_train_b)

    col_stats = {}
    discrete_values: Dict[int, np.ndarray] = {}
    for i, c in enumerate(union_cols):
        series = pd.to_numeric(x_train[c], errors='coerce').dropna()
        q01 = float(series.quantile(0.01))
        q99 = float(series.quantile(0.99))
        if np.isclose(q01, q99):
            q01 = float(series.min())
            q99 = float(series.max())
        col_stats[c] = {'q01': q01, 'q99': q99}

        uniq = np.sort(series.unique())
        if len(uniq) <= 6:
            discrete_values[i] = uniq.astype(float)

    xl = np.array([col_stats[c]['q01'] for c in union_cols], dtype=float)
    xu = np.array([col_stats[c]['q99'] for c in union_cols], dtype=float)

    rng = np.random.default_rng(SEED)
    x_pop = rng.uniform(xl, xu, size=(POP, len(union_cols)))
    x_pop = snap_discrete_columns(x_pop, discrete_values)
    f_pop = objective_values(pd.DataFrame(x_pop, columns=union_cols), model_a, cols_a, model_b, cols_b)

    for gen in range(GEN):
        i1 = rng.integers(0, POP, POP)
        i2 = rng.integers(0, POP, POP)
        better = np.where(f_pop[i1, 0] <= f_pop[i2, 0], i1, i2)
        parents = x_pop[better]

        children = sbx_crossover(parents, xl, xu, rng)
        children = polynomial_mutation(children, xl, xu, rng)
        children = snap_discrete_columns(children, discrete_values)
        f_children = objective_values(pd.DataFrame(children, columns=union_cols), model_a, cols_a, model_b, cols_b)

        x_all_gen = np.vstack([x_pop, children])
        f_all_gen = np.vstack([f_pop, f_children])

        fronts = fast_non_dominated_sort(f_all_gen)
        selected: List[int] = []
        for front in fronts:
            if len(selected) + len(front) <= POP:
                selected.extend(front)
            else:
                cd = crowding_distance(f_all_gen, front)
                remain = sorted(front, key=lambda idx: -cd[idx])
                selected.extend(remain[: POP - len(selected)])
                break

        x_pop = x_all_gen[selected]
        f_pop = f_all_gen[selected]

        if (gen + 1) % 30 == 0:
            n_front = len(fronts[0]) if fronts else 0
            print(f'gen={gen + 1}, pareto_size={n_front}')

    final_front = fast_non_dominated_sort(f_pop)[0]
    x_p = x_pop[final_front]

    df_p = pd.DataFrame(x_p, columns=union_cols)
    pred_a = np.clip(model_a.predict(df_p[cols_a]), 0, None)
    pred_b = np.clip(model_b.predict(df_p[cols_b]), 0, None)

    df_p[f'pred_{OBJ_A}'] = pred_a
    df_p[f'pred_{OBJ_B}'] = pred_b

    a_min, a_max = float(np.min(pred_a)), float(np.max(pred_a))
    b_min, b_max = float(np.min(pred_b)), float(np.max(pred_b))
    df_p['norm_obj_a'] = (df_p[f'pred_{OBJ_A}'] - a_min) / (a_max - a_min + 1e-12)
    df_p['norm_obj_b'] = (df_p[f'pred_{OBJ_B}'] - b_min) / (b_max - b_min + 1e-12)
    df_p['balanced_score'] = 0.5 * df_p['norm_obj_a'] + 0.5 * df_p['norm_obj_b']

    df_p = df_p.sort_values(['balanced_score', f'pred_{OBJ_A}', f'pred_{OBJ_B}'], ascending=[False, False, False]).reset_index(drop=True)

    reps = select_representatives(df_p, f'pred_{OBJ_A}', f'pred_{OBJ_B}')

    summary = pd.DataFrame([
        {
            'snapshot': snapshot.name,
            'objective_a': OBJ_A,
            'objective_b': OBJ_B,
            'model_a': row_a['selected_model'],
            'model_b': row_b['selected_model'],
            'n_train_joint': int(len(x_train)),
            'n_features_union': int(len(union_cols)),
            'pop': POP,
            'gen': GEN,
            'pareto_size': int(len(df_p)),
            'pred_a_min': a_min,
            'pred_a_max': a_max,
            'pred_b_min': b_min,
            'pred_b_max': b_max,
        }
    ])

    out_front = out_root / 'task09_pareto_front_solutions.csv'
    out_reps = out_root / 'task09_pareto_representative_solutions.csv'
    out_summary = out_root / 'task09_pareto_run_summary.csv'
    out_plot = out_root / 'task09_pareto_front_plot.png'

    df_p.to_csv(out_front, index=False)
    reps.to_csv(out_reps, index=False)
    summary.to_csv(out_summary, index=False)


    print(f'Pareto finished. snapshot={snapshot.name}')
    print(f'Output dir: {out_root}')
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
