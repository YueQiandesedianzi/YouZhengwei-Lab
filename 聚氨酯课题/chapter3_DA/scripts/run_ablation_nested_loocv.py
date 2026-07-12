#!/usr/bin/env python3
"""
12 vs 14 formulation-feature ablation.

Why two kinds of numbers?
-------------------------
1) nested_per_tier: for EVERY feature tier, runs nested LOOCV (inner search on each
   training fold). This is a different estimand and usually does NOT match the
   manuscript headline R² (0.76 / 0.66 / 0.81).

2) manuscript_ref14HP_loocv: matches alghorithm_multi_1.9_tensile.ipynb Figure 3 style:
   tune ONCE on the FULL 14-mech (or 16+process for healing) set → majority hyperparams maj;
   then for EACH tier use LeaveOneOut + cross_val_predict with the SAME fixed maj
   (only the feature columns change). This aligns with evaluate_loocv() in the notebook.

3) manuscript_separateHP_loocv: same as (2) but 14-ladder uses maj tuned on full-14,
   12-ladder uses maj tuned on full-12 (each feature count gets its own HP).

4) manuscript_notebook_defaults_loocv: no nested search — uses build_model_by_name()
   hard-coded SVR / XGBoost / GPR from the same notebook cell (fallback path).
"""
import argparse
import os
import pickle
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    LeaveOneOut,
    ParameterGrid,
    RandomizedSearchCV,
    cross_val_predict,
)
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")
CSV = os.path.join(DATA, "Final_Features_Ultimate_all_deletion.csv")
RESULTS = os.path.join(os.path.dirname(HERE), "results")

features_mech = [
    "X1_Soft_Content",
    "X2_DA_Content",
    "X3_Hard_Content",
    "X4_R_Ratio",
    "X5_Add_Crosslink",
    "DA_Linker_Type",
    "Network_Topology",
    "Polyol_Class",
    "Iso_Class",
    "Soft_Mw",
    "Constraint_Factor",
    "Hard_Symmetry",
    "Soft_Cryst",
    "Synergy_Feature",
]


def get_clean_data(target_col, feature_cols, df):
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"missing columns: {missing_cols}")
    data = df.dropna(subset=[target_col] + feature_cols).copy()
    return data[feature_cols].values, data[target_col].values


def nested_loocv_tune_eval(model_name, base_estimator, param_space, X, y, grid_threshold=60, random_iter=50):
    loo = LeaveOneOut()
    X_arr = np.asarray(X)
    y = np.asarray(y).ravel()
    y_pred = np.zeros(len(y))
    inner_rmse = []
    fold_params = []
    route = "NestedLOOCV"
    for tr, te in loo.split(X_arr):
        X_tr, X_te = X_arr[tr], X_arr[te]
        y_tr = y[tr]
        nn = len(tr)
        ns = min(5, nn)
        inner = KFold(n_splits=ns, shuffle=True, random_state=42) if ns >= 2 else None
        est = clone(base_estimator)
        if not param_space:
            est.fit(X_tr, y_tr)
            y_pred[te[0]] = float(est.predict(X_te)[0])
            fold_params.append({})
            continue
        if inner is None:
            est.fit(X_tr, y_tr)
            y_pred[te[0]] = float(est.predict(X_te)[0])
            fold_params.append({})
            continue
        n_comb = len(list(ParameterGrid(param_space)))
        if n_comb <= grid_threshold:
            search = GridSearchCV(
                est, param_space, cv=inner, scoring="neg_mean_squared_error", n_jobs=-1, verbose=0
            )
            route = f"NestedGrid({n_comb})"
        else:
            ni = min(random_iter, n_comb)
            search = RandomizedSearchCV(
                est,
                param_distributions=param_space,
                n_iter=ni,
                cv=inner,
                scoring="neg_mean_squared_error",
                random_state=42,
                n_jobs=-1,
                verbose=0,
            )
            route = f"NestedRandom({ni}/{n_comb})"
        search.fit(X_tr, y_tr)
        y_pred[te[0]] = float(search.predict(X_te)[0])
        inner_rmse.append(float(np.sqrt(-search.best_score_)))
        fold_params.append(search.best_params_)
    r2 = r2_score(y, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
    tune_rmse = float(np.mean(inner_rmse)) if inner_rmse else float("nan")
    pickled = [pickle.dumps(p, protocol=4) for p in fold_params if p]
    maj = pickle.loads(Counter(pickled).most_common(1)[0][0]) if pickled else {}
    final_m = clone(base_estimator)
    if maj:
        final_m.set_params(**maj)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        final_m.fit(X_arr, y)
    return final_m, route, maj, tune_rmse, r2, rmse, y_pred


def evaluate_loocv_fixed(estimator, X, y):
    """Same as notebook evaluate_loocv: fixed HP, LOOCV."""
    y_pred = cross_val_predict(estimator, X, y, cv=LeaveOneOut(), n_jobs=-1)
    r2 = r2_score(y, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
    return r2, rmse, y_pred


def build_model_by_name(model_name: str):
    """Mirror alghorithm_multi_1.9_tensile.ipynb cell 18 fallback."""
    name = str(model_name).strip()
    if name == "MLR":
        from sklearn.linear_model import LinearRegression

        return LinearRegression()
    if name == "Ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=42))
    if name in ["LASSO", "Lasso"]:
        return make_pipeline(StandardScaler(), Lasso(alpha=0.01, random_state=42, max_iter=20000))
    if name == "SVR":
        return make_pipeline(StandardScaler(), SVR(C=500, gamma=1.0, epsilon=0.1, kernel="rbf"))
    if name in ["Random Forest", "RF"]:
        return RandomForestRegressor(
            n_estimators=400, max_depth=10, min_samples_split=5, random_state=42, n_jobs=-1
        )
    if name == "XGBoost":
        return XGBRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
        )
    if name == "GPR":
        kernel = 1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5) + WhiteKernel(
            noise_level=0.1, noise_level_bounds=(1e-5, 1e1)
        )
        return make_pipeline(
            StandardScaler(),
            GaussianProcessRegressor(
                kernel=kernel, normalize_y=True, random_state=42, n_restarts_optimizer=5
            ),
        )
    if name in ["MLP", "ANN"]:
        return make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(64,),
                activation="tanh",
                solver="lbfgs",
                alpha=0.01,
                max_iter=3000,
                random_state=42,
            ),
        )
    raise ValueError(f"未知模型名: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip nested_tune_per_tier (very slow); run manuscript-style protocols only.",
    )
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    df = pd.read_csv(CSV)
    lvl1_ab = features_mech[:5]
    lvl2_ab = features_mech[5:9]
    lvl3_14 = features_mech[9:]
    lvl3_12 = ["Soft_Mw", "Hard_Symmetry", "Soft_Cryst"]
    assert len(lvl1_ab + lvl2_ab + lvl3_12) == 12

    _ablation_base = {
        "SVR": make_pipeline(StandardScaler(), SVR(kernel="rbf")),
        "XGBoost": XGBRegressor(random_state=42, n_jobs=-1),
        "GPR": make_pipeline(
            StandardScaler(),
            GaussianProcessRegressor(
                kernel=1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
                + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1)),
                random_state=42,
            ),
        ),
    }
    _ablation_spaces = {
        "SVR": {
            "svr__C": [1, 10, 50, 100, 300, 500, 1000],
            "svr__gamma": ["scale", 0.001, 0.01, 0.1, 0.5, 1.0],
            "svr__epsilon": [0.01, 0.05, 0.1, 0.2, 0.5],
        },
        "XGBoost": {
            "n_estimators": [100, 200, 300, 500],
            "learning_rate": [0.01, 0.02, 0.05, 0.1],
            "max_depth": [3, 4, 5, 6, 8],
            "subsample": [0.6, 0.7, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "reg_alpha": [0.0, 0.1, 1.0],
            "reg_lambda": [1.0, 2.0, 5.0],
        },
        "GPR": {
            "gaussianprocessregressor__kernel": [
                1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5)
                + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1)),
                1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
                + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1)),
                1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
                + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1)),
            ],
            "gaussianprocessregressor__alpha": [1e-10, 1e-6, 1e-4],
            "gaussianprocessregressor__normalize_y": [True, False],
            "gaussianprocessregressor__n_restarts_optimizer": [0, 5, 10],
        },
    }

    TASK_MODEL_AB = {
        "tensile_strength": "SVR",
        "elongation": "XGBoost",
        "healing_eff": "GPR",
    }
    _proc = ["healing_temperature", "healing_time"]

    full_14_mech = lvl1_ab + lvl2_ab + lvl3_14
    full_12_mech = lvl1_ab + lvl2_ab + lvl3_12

    def tiers_for_set(tag):
        l3 = lvl3_14 if tag == "14_mech" else lvl3_12
        mech_tiers = [
            ("level1", lvl1_ab),
            ("level1+level2", lvl1_ab + lvl2_ab),
            ("level1+level2+level3", lvl1_ab + lvl2_ab + l3),
        ]
        heal_tiers = [
            ("level1", lvl1_ab),
            ("level1+level2", lvl1_ab + lvl2_ab),
            ("level1+level2+level3+process", lvl1_ab + lvl2_ab + l3 + _proc),
        ]
        return mech_tiers, heal_tiers

    all_rows = []

    # ----- A) nested per tier (original script) -----
    if args.fast:
        print("=== Skipping nested_tune_per_tier (--fast) ===")
    else:
        print("=== Protocol: nested_tune_per_tier (NOT the manuscript headline R²) ===")
    if not args.fast:
        for fset in ("14_mech", "12_mech"):
            mech_tiers, heal_tiers = tiers_for_set(fset)
            for target_col, tier_list in [
                ("tensile_strength", mech_tiers),
                ("elongation", mech_tiers),
                ("healing_eff", heal_tiers),
            ]:
                mname = TASK_MODEL_AB[target_col]
                base = _ablation_base[mname]
                pspace = _ablation_spaces[mname]
                for level_label, feat_list in tier_list:
                    X, y = get_clean_data(target_col, feat_list, df)
                    _, route, _, tune_rmse, r2, rmse, _ = nested_loocv_tune_eval(
                        mname, base, pspace, X, y
                    )
                    all_rows.append(
                        {
                            "Protocol": "nested_tune_per_tier",
                            "TargetCol": target_col,
                            "Level": level_label,
                            "FeatureSet": fset,
                            "NumFeatures": len(feat_list),
                            "Model": mname,
                            "Route": route,
                            "R2": r2,
                            "RMSE": rmse,
                            "MeanInnerCV_RMSE": tune_rmse,
                        }
                    )
                    print(f"[nested] [{fset}] {target_col} | {level_label} | p={len(feat_list)} | R2={r2:.4f}")

    # ----- B) manuscript: tune once on full-14, fixed HP LOOCV all tiers (both 12 & 14 ladders) -----
    print("\n=== Protocol: manuscript_ref14HP_loocv (matches Figure 3 style) ===")
    maj_ref14 = {}
    nested_r2_on_full14 = {}
    for target_col, mname in TASK_MODEL_AB.items():
        base = _ablation_base[mname]
        pspace = _ablation_spaces[mname]
        if target_col == "healing_eff":
            feat_full = full_14_mech + _proc
        else:
            feat_full = full_14_mech
        Xf, yf = get_clean_data(target_col, feat_full, df)
        _, route, maj, _, r2_nest, _, _ = nested_loocv_tune_eval(mname, base, pspace, Xf, yf)
        maj_ref14[target_col] = maj
        nested_r2_on_full14[target_col] = r2_nest
        print(f"  [tune-once full-14] {target_col} | {mname} | nested_R2_full={r2_nest:.4f} | route={route}")

    for fset in ("14_mech", "12_mech"):
        mech_tiers, heal_tiers = tiers_for_set(fset)
        for target_col, tier_list in [
            ("tensile_strength", mech_tiers),
            ("elongation", mech_tiers),
            ("healing_eff", heal_tiers),
        ]:
            mname = TASK_MODEL_AB[target_col]
            base = _ablation_base[mname]
            maj = maj_ref14[target_col]
            for level_label, feat_list in tier_list:
                X, y = get_clean_data(target_col, feat_list, df)
                est = clone(base)
                if maj:
                    est.set_params(**maj)
                r2, rmse, _ = evaluate_loocv_fixed(est, X, y)
                all_rows.append(
                    {
                        "Protocol": "manuscript_ref14HP_loocv",
                        "TargetCol": target_col,
                        "Level": level_label,
                        "FeatureSet": fset,
                        "NumFeatures": len(feat_list),
                        "Model": mname,
                        "Route": "fixed_MAJ_from_full14_nested_majority",
                        "R2": r2,
                        "RMSE": rmse,
                        "MeanInnerCV_RMSE": np.nan,
                    }
                )
                print(f"[ref14HP] [{fset}] {target_col} | {level_label} | p={len(feat_list)} | R2={r2:.4f}")

    # ----- C) manuscript: separate maj for 14-ladder vs 12-ladder -----
    print("\n=== Protocol: manuscript_separateHP_loocv ===")
    maj_ref12 = {}
    for target_col, mname in TASK_MODEL_AB.items():
        base = _ablation_base[mname]
        pspace = _ablation_spaces[mname]
        if target_col == "healing_eff":
            feat_full = full_12_mech + _proc
        else:
            feat_full = full_12_mech
        Xf, yf = get_clean_data(target_col, feat_full, df)
        _, route, maj, _, r2_nest, _, _ = nested_loocv_tune_eval(mname, base, pspace, Xf, yf)
        maj_ref12[target_col] = maj
        print(f"  [tune-once full-12] {target_col} | nested_R2_full={r2_nest:.4f} | route={route}")

    for fset in ("14_mech", "12_mech"):
        mech_tiers, heal_tiers = tiers_for_set(fset)
        maj_map = maj_ref14 if fset == "14_mech" else maj_ref12
        for target_col, tier_list in [
            ("tensile_strength", mech_tiers),
            ("elongation", mech_tiers),
            ("healing_eff", heal_tiers),
        ]:
            mname = TASK_MODEL_AB[target_col]
            base = _ablation_base[mname]
            maj = maj_map[target_col]
            for level_label, feat_list in tier_list:
                X, y = get_clean_data(target_col, feat_list, df)
                est = clone(base)
                if maj:
                    est.set_params(**maj)
                r2, rmse, _ = evaluate_loocv_fixed(est, X, y)
                all_rows.append(
                    {
                        "Protocol": "manuscript_separateHP_loocv",
                        "TargetCol": target_col,
                        "Level": level_label,
                        "FeatureSet": fset,
                        "NumFeatures": len(feat_list),
                        "Model": mname,
                        "Route": f"fixed_MAJ_from_full_{'14' if fset == '14_mech' else '12'}",
                        "R2": r2,
                        "RMSE": rmse,
                        "MeanInnerCV_RMSE": np.nan,
                    }
                )
                print(f"[sepHP] [{fset}] {target_col} | {level_label} | R2={r2:.4f}")

    # ----- D) notebook hard-coded defaults, no nested search -----
    print("\n=== Protocol: manuscript_notebook_defaults_loocv (build_model_by_name) ===")
    for fset in ("14_mech", "12_mech"):
        mech_tiers, heal_tiers = tiers_for_set(fset)
        for target_col, tier_list in [
            ("tensile_strength", mech_tiers),
            ("elongation", mech_tiers),
            ("healing_eff", heal_tiers),
        ]:
            mname = TASK_MODEL_AB[target_col]
            est = build_model_by_name(mname)
            for level_label, feat_list in tier_list:
                X, y = get_clean_data(target_col, feat_list, df)
                r2, rmse, _ = evaluate_loocv_fixed(clone(est), X, y)
                all_rows.append(
                    {
                        "Protocol": "manuscript_notebook_defaults_loocv",
                        "TargetCol": target_col,
                        "Level": level_label,
                        "FeatureSet": fset,
                        "NumFeatures": len(feat_list),
                        "Model": mname,
                        "Route": "notebook_hardcoded_HP",
                        "R2": r2,
                        "RMSE": rmse,
                        "MeanInnerCV_RMSE": np.nan,
                    }
                )
                print(f"[defaults] [{fset}] {target_col} | {level_label} | R2={r2:.4f}")

    ablation_df = pd.DataFrame(all_rows)

    def write_delta(proto: str, path_suffix: str):
        sub = ablation_df[ablation_df["Protocol"] == proto]
        if sub.empty:
            print(f"  (skip delta {proto}: no rows)")
            return pd.DataFrame()
        b14 = sub[sub["FeatureSet"] == "14_mech"][["TargetCol", "Level", "R2", "RMSE"]].rename(
            columns={"R2": "R2_14", "RMSE": "RMSE_14"}
        )
        b12 = sub[sub["FeatureSet"] == "12_mech"][["TargetCol", "Level", "R2", "RMSE"]].rename(
            columns={"R2": "R2_12", "RMSE": "RMSE_12"}
        )
        cmp_ = b14.merge(b12, on=["TargetCol", "Level"], how="inner")
        cmp_["Delta_R2"] = cmp_["R2_12"] - cmp_["R2_14"]
        cmp_["Delta_RMSE"] = cmp_["RMSE_12"] - cmp_["RMSE_14"]
        cmp_.to_csv(os.path.join(RESULTS, path_suffix), index=False)
        return cmp_

    out_all = os.path.join(RESULTS, "ablation_12vs14_all_protocols.csv")
    ablation_df.to_csv(out_all, index=False)

    cmp_nested = write_delta("nested_tune_per_tier", "ablation_12vs14_delta_nested_per_tier.csv")
    cmp_ref14 = write_delta("manuscript_ref14HP_loocv", "ablation_12vs14_delta_manuscript_ref14HP.csv")
    cmp_sep = write_delta("manuscript_separateHP_loocv", "ablation_12vs14_delta_manuscript_separateHP.csv")
    cmp_def = write_delta("manuscript_notebook_defaults_loocv", "ablation_12vs14_delta_notebook_defaults.csv")

    print("\n========== Delta (12 − 14), manuscript_ref14HP_loocv ==========")
    print(cmp_ref14.to_string(index=False))
    print(f"\nWrote {out_all} and delta CSVs per protocol.")

    # Optional: single-feature nested (slow) — keep commented or behind env
    if os.environ.get("ABLATION_RUN_SINGLE_DROP_NESTED") == "1":
        lvl3_minus_c = [c for c in lvl3_14 if c != "Constraint_Factor"]
        lvl3_minus_s = [c for c in lvl3_14 if c != "Synergy_Feature"]
        extra_drop_tiers = [
            ("level1+level2+level3_minus_Constraint", lvl1_ab + lvl2_ab + lvl3_minus_c),
            ("level1+level2+level3_minus_Synergy", lvl1_ab + lvl2_ab + lvl3_minus_s),
        ]
        drop_rows = []
        for target_col in ("tensile_strength", "elongation"):
            mname = TASK_MODEL_AB[target_col]
            base = _ablation_base[mname]
            pspace = _ablation_spaces[mname]
            for level_label, feat_list in extra_drop_tiers:
                X, y = get_clean_data(target_col, feat_list, df)
                _, route, _, tune_rmse, r2, rmse, _ = nested_loocv_tune_eval(mname, base, pspace, X, y)
                drop_rows.append(
                    {
                        "Protocol": "nested_tune_per_tier",
                        "TargetCol": target_col,
                        "Level": level_label,
                        "FeatureSet": "14_mech_single_drop",
                        "NumFeatures": len(feat_list),
                        "Model": mname,
                        "Route": route,
                        "R2": r2,
                        "RMSE": rmse,
                        "MeanInnerCV_RMSE": tune_rmse,
                    }
                )
        pd.concat([ablation_df, pd.DataFrame(drop_rows)], ignore_index=True).to_csv(out_all, index=False)


if __name__ == "__main__":
    main()
