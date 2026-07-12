#!/usr/bin/env python3
"""Thin wrapper: run legacy feature construction against package data/ layout.

The full historical builder lived as data_washing_3.8.py and wrote
Final_Features_3.8.csv. Thesis Chapter-3 modeling uses the locked files:

  data/Final_Features_Ultimate.csv              (92)
  data/Final_Features_Ultimate_all_deletion.csv (84)

This script only validates that the locked tables exist and prints counts.
For regenerating features from the raw integration table, see the original
logic archived notes in docs/数据清洗与样本集方法学说明.md.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def main() -> None:
    raw = DATA / "DA-PU数据整合-new_furan_cleaned.csv"
    ult = DATA / "Final_Features_Ultimate.csv"
    final = DATA / "Final_Features_Ultimate_all_deletion.csv"
    for p in (raw, ult, final):
        if not p.exists():
            raise FileNotFoundError(p)
    print("raw rows:", len(pd.read_csv(raw)))
    print("ultimate rows:", len(pd.read_csv(ult)))
    print("final(all_deletion) rows:", len(pd.read_csv(final)))
    print("OK — locked Chapter-3 tables are present under data/.")


if __name__ == "__main__":
    main()
