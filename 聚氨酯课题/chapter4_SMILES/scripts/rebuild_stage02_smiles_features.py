"""Optional Stage-02 rebuild: weighted RDKit descriptors + Morgan fingerprints.

The thesis-locked tables already live in data/stage02/. This script is a
self-contained reconstruction for teaching/extension. Bit-exact equality with
the locked CSVs is NOT guaranteed across RDKit versions.

Align rule (historical): healing_temp_C + healing_time_h + sigma_max_MPa
rounded to 4 decimals, matching Step4 rows to Step2 SMILES/weight fields.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import UPSTREAM, OUT

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem
except ImportError as exc:  # pragma: no cover
    raise SystemExit("RDKit is required for stage02 rebuild. pip install rdkit") from exc

OUT_DIR = OUT / "task02_rebuild"
STEP2 = UPSTREAM / "final_merged_dataset.csv"
STEP4 = UPSTREAM / "feature_matrix_step4.csv"

SMILES_WEIGHT_PAIRS = [
    ("硬段1_SMILES", "f_hs1"),
    ("硬段2_SMILES", "f_hs2"),
    ("硬段3_SMILES", "f_hs3"),
    ("软段1_SMILES", "f_ss1"),
    ("软段2_SMILES", "f_ss2"),
    ("动态键组分_SMILES", "f_dk"),
]

DESC_FUNCS = [
    ("MolWt", Descriptors.MolWt),
    ("MolLogP", Descriptors.MolLogP),
    ("TPSA", Descriptors.TPSA),
    ("NumHDonors", Descriptors.NumHDonors),
    ("NumHAcceptors", Descriptors.NumHAcceptors),
    ("NumRotatableBonds", Descriptors.NumRotatableBonds),
    ("RingCount", Descriptors.RingCount),
    ("FractionCSP3", Descriptors.FractionCSP3),
    ("HeavyAtomCount", Descriptors.HeavyAtomCount),
    ("NHOHCount", Descriptors.NHOHCount),
    ("NOCount", Descriptors.NOCount),
    ("NumAliphaticRings", Descriptors.NumAliphaticRings),
    ("NumAromaticRings", Descriptors.NumAromaticRings),
    ("NumSaturatedRings", Descriptors.NumSaturatedRings),
    ("NumValenceElectrons", Descriptors.NumValenceElectrons),
]
FP_BITS = 512
FP_RADIUS = 2


def round_key(df: pd.DataFrame) -> pd.Series:
    t = pd.to_numeric(df["healing_temp_C"], errors="coerce").round(4)
    h = pd.to_numeric(df["healing_time_h"], errors="coerce").round(4)
    s = pd.to_numeric(df["sigma_max_MPa"], errors="coerce").round(4)
    return t.astype(str) + "|" + h.astype(str) + "|" + s.astype(str)


def mol_features(smiles: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not isinstance(smiles, str) or not smiles.strip():
        return None, None
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return None, None
    desc = np.array([float(fn(mol)) for _, fn in DESC_FUNCS], dtype=float)
    fp = np.zeros((FP_BITS,), dtype=float)
    bitvect = AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, nBits=FP_BITS)
    AllChem.DataStructs.ConvertToNumpyArray(bitvect, fp)
    return desc, fp


def weighted_row(row: pd.Series) -> tuple[np.ndarray, np.ndarray, int, int]:
    desc_acc = np.zeros(len(DESC_FUNCS), dtype=float)
    fp_acc = np.zeros(FP_BITS, dtype=float)
    w_sum = 0.0
    n_ok = 0
    n_fail = 0
    for smiles_col, w_col in SMILES_WEIGHT_PAIRS:
        smiles = row.get(smiles_col, "")
        w = pd.to_numeric(row.get(w_col, np.nan), errors="coerce")
        if not np.isfinite(w) or w <= 0:
            continue
        desc, fp = mol_features(smiles)
        if desc is None:
            n_fail += 1
            continue
        desc_acc += w * desc
        fp_acc += w * fp
        w_sum += float(w)
        n_ok += 1
    if w_sum <= 0:
        return desc_acc * np.nan, fp_acc * np.nan, n_ok, n_fail
    return desc_acc, fp_acc, n_ok, n_fail


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    step2 = pd.read_csv(STEP2)
    step4 = pd.read_csv(STEP4)
    step2 = step2.copy()
    step4 = step4.copy()
    step2["_k"] = round_key(step2)
    step4["_k"] = round_key(step4)

    keep_s2 = [c for c, _ in SMILES_WEIGHT_PAIRS] + [w for _, w in SMILES_WEIGHT_PAIRS] + ["_k", "全局编号"]
    keep_s2 = [c for c in keep_s2 if c in step2.columns]
    merged = step4.merge(step2[keep_s2], on="_k", how="left", suffixes=("", "_s2"))

    desc_cols = [f"molw_{name}" for name, _ in DESC_FUNCS]
    fp_cols = [f"mfpw_{i}" for i in range(FP_BITS)]
    desc_mat = np.full((len(merged), len(DESC_FUNCS)), np.nan)
    fp_mat = np.full((len(merged), FP_BITS), np.nan)
    usable = np.zeros(len(merged), dtype=int)
    invalid_pairs = []

    for i, row in merged.iterrows():
        desc, fp, n_ok, n_fail = weighted_row(row)
        desc_mat[merged.index.get_loc(i)] = desc
        fp_mat[merged.index.get_loc(i)] = fp
        usable[merged.index.get_loc(i)] = int(n_ok > 0 and np.isfinite(desc).all())
        invalid_pairs.append(n_fail)

    mol_df = pd.DataFrame(desc_mat, columns=desc_cols)
    mol_df = pd.concat([mol_df, pd.DataFrame(fp_mat, columns=fp_cols)], axis=1)
    mol_df.insert(0, "usable_for_weighted_smiles", usable)
    mol_df["invalid_smiles_pairs"] = invalid_pairs

    # attach identifiers if present
    for col in ("全局编号", "文献来源", "动态键类型"):
        if col in merged.columns:
            mol_df.insert(0, col, merged[col].values)

    full = pd.concat([merged.reset_index(drop=True), mol_df.reset_index(drop=True)], axis=1)
    # drop duplicated helper
    if "_k" in full.columns:
        full = full.drop(columns=["_k"])

    mol_df.to_csv(OUT_DIR / "task02_weighted_molecular_features_full.rebuild.csv", index=False)
    full.to_csv(OUT_DIR / "task02_domain_plus_molecular_full.rebuild.csv", index=False)
    full.loc[full["usable_for_weighted_smiles"] == 1].to_csv(
        OUT_DIR / "task02_domain_plus_molecular_usable.rebuild.csv", index=False
    )
    pd.DataFrame({"molecular_desc_col": desc_cols}).to_csv(OUT_DIR / "descriptor_columns.rebuild.csv", index=False)
    pd.DataFrame({"molecular_fp_col": fp_cols}).to_csv(OUT_DIR / "fingerprint_columns.rebuild.csv", index=False)
    print("Wrote", OUT_DIR)
    print("usable rows:", int(usable.sum()), "/", len(usable))
    print("Prefer locked CSVs in data/stage02/ for thesis-consistent downstream modeling.")


if __name__ == "__main__":
    main()
