"""Rebuild Task-03 modeling matrix from Stage-02 molecular+domain table.

Logic mirrors the thesis Chapter-4 pipeline (STAR Task-03):
1) molecular columns = molw_* + mfpw_*
2) rows with usable_for_weighted_smiles==0 -> molecular NaN
3) KNNImputer on molecular channel
4) Pearson |r|>0.95 redundancy drop
5) StandardScaler + PCA to 95% cumulative variance
6) concat domain features + mol_pca_*

Output overwrites nothing by default; writes under outputs/task03_rebuild/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

from config import STAGE02, META_COLS, TARGET_COLS, OUT

IN_FILE = STAGE02 / "task02_domain_plus_molecular_full314.csv"
OUT_DIR = OUT / "task03_rebuild"
PEARSON_THR = 0.95
PCA_VAR = 0.95
RANDOM_STATE = 42


def pearson_drop(x: pd.DataFrame, thr: float = PEARSON_THR) -> list[str]:
    corr = x.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > thr)]
    return [c for c in x.columns if c not in set(to_drop)]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(IN_FILE)

    mol_cols = [c for c in df.columns if c.startswith("molw_") or c.startswith("mfpw_")]
    if not mol_cols:
        raise RuntimeError("No molecular columns (molw_/mfpw_) found in stage02 table.")

    meta = [c for c in META_COLS if c in df.columns]
    targets = [c for c in TARGET_COLS if c in df.columns]
    # keep healing process vars if present as features
    extra_targets = [c for c in ("healing_efficiency", "healing_temp_C", "healing_time_h") if c in df.columns]
    reserved = set(meta + targets + extra_targets + mol_cols)
    domain_cols = [c for c in df.columns if c not in reserved]

    mol = df[mol_cols].apply(pd.to_numeric, errors="coerce")
    if "usable_for_weighted_smiles" in df.columns:
        usable = pd.to_numeric(df["usable_for_weighted_smiles"], errors="coerce").fillna(0).astype(int)
        mol.loc[usable == 0, :] = np.nan

    imputer = KNNImputer(n_neighbors=5, weights="distance")
    mol_imp = pd.DataFrame(imputer.fit_transform(mol), columns=mol_cols, index=df.index)

    kept = pearson_drop(mol_imp)
    dropped = [c for c in mol_cols if c not in set(kept)]
    mol_kept = mol_imp[kept]

    scaler = StandardScaler()
    z = scaler.fit_transform(mol_kept)
    pca = PCA(n_components=PCA_VAR, svd_solver="full", random_state=RANDOM_STATE)
    pcs = pca.fit_transform(z)
    pc_cols = [f"mol_pca_{i+1:03d}" for i in range(pcs.shape[1])]
    pc_df = pd.DataFrame(pcs, columns=pc_cols, index=df.index)

    out = pd.concat(
        [
            df[meta] if meta else pd.DataFrame(index=df.index),
            df[domain_cols].apply(pd.to_numeric, errors="coerce") if domain_cols else pd.DataFrame(index=df.index),
            pc_df,
            df[targets + extra_targets] if (targets or extra_targets) else pd.DataFrame(index=df.index),
        ],
        axis=1,
    )

    out.to_csv(OUT_DIR / "task03_final_features_for_modeling.rebuild.csv", index=False)
    pd.DataFrame({"molecular_kept_col": kept}).to_csv(OUT_DIR / "molecular_kept_after_pearson.rebuild.csv", index=False)
    pd.DataFrame({"molecular_dropped_col": dropped}).to_csv(OUT_DIR / "molecular_dropped_by_pearson.rebuild.csv", index=False)
    pd.DataFrame(
        {
            "pc": pc_cols,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_variance": np.cumsum(pca.explained_variance_ratio_),
        }
    ).to_csv(OUT_DIR / "pca_variance_table.rebuild.csv", index=False)

    summary = pd.DataFrame(
        [
            {"metric": "rows_total", "value": len(df)},
            {"metric": "molecular_total_before", "value": len(mol_cols)},
            {"metric": "molecular_after_pearson", "value": len(kept)},
            {"metric": "pearson_dropped", "value": len(dropped)},
            {"metric": "pca_n_components_95var", "value": pcs.shape[1]},
            {"metric": "final_feature_dim_domain", "value": len(domain_cols)},
            {"metric": "final_feature_dim_pca", "value": pcs.shape[1]},
        ]
    )
    summary.to_csv(OUT_DIR / "summary_metrics.rebuild.csv", index=False)
    print("Wrote rebuild outputs to", OUT_DIR)
    print(summary.to_string(index=False))
    print("NOTE: Prefer data/stage03/task03_final_features_for_modeling.csv for thesis-locked modeling.")


if __name__ == "__main__":
    main()
