#!/usr/bin/env python3
"""复现整合表、Ultimate、Final 的样本量、sample_id 差集及愈合 complete-case 统计（标准库 csv）。"""
from __future__ import annotations

import csv
from pathlib import Path


def parse_float(x: str | None) -> float | None:
    if x is None or str(x).strip() == "":
        return None
    try:
        return float(str(x).strip())
    except ValueError:
        return None


def is_seq(cell: str) -> bool:
    if not cell or not str(cell).strip():
        return False
    try:
        float(str(cell).strip())
        return True
    except ValueError:
        return False


def main() -> None:
    base = Path(__file__).resolve().parents[1] / "data"
    final_path = base / "Final_Features_Ultimate_all_deletion.csv"
    ultimate_path = base / "Final_Features_Ultimate.csv"
    raw_path = base / "DA-PU数据整合-new_furan_cleaned.csv"

    with open(final_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    print("=== Final_Features_Ultimate_all_deletion.csv ===")
    print(f"len(rows) = {len(rows)}")

    mech_ok = heal_ok = 0
    missing_mech: list[tuple[str | None, str, str]] = []
    for row in rows:
        ts, el = parse_float(row.get("tensile_strength")), parse_float(row.get("elongation"))
        if ts is not None and el is not None:
            mech_ok += 1
        else:
            missing_mech.append((row.get("sample_id"), row.get("tensile_strength") or "", row.get("elongation") or ""))
        if parse_float(row.get("healing_eff")) is not None:
            heal_ok += 1

    print(f"non-missing (tensile_strength AND elongation) = {mech_ok}")
    print(f"non-missing healing_eff (complete-case for healing model) = {heal_ok}")
    print(f"missing healing_eff (listwise excluded, NOT outlier removal) = {len(rows) - heal_ok}")
    if missing_mech:
        print("missing tensile/elongation rows:", missing_mech)

    final_ids = {
        row["sample_id"].strip()
        for row in rows
        if row.get("sample_id") and str(row["sample_id"]).strip()
    }

    with open(ultimate_path, newline="", encoding="utf-8-sig") as f:
        urows = list(csv.DictReader(f))
    ultimate_ids = {
        r["sample_id"].strip()
        for r in urows
        if r.get("sample_id") and str(r["sample_id"]).strip()
    }
    print("\n=== Final_Features_Ultimate.csv vs all_deletion ===")
    print(f"Ultimate len(rows) = {len(urows)}")
    only_ult = sorted(ultimate_ids - final_ids)
    print(f"only in Ultimate, not in all_deletion ({len(only_ult)}): {only_ult}")

    with open(raw_path, newline="", encoding="utf-8-sig") as f:
        lines = list(csv.reader(f))

    raw_ids: list[str] = []
    max_seq = None
    for line in lines[1:]:
        if len(line) < 2:
            continue
        if is_seq(line[0]):
            v = float(str(line[0]).strip())
            max_seq = v if max_seq is None else max(max_seq, v)
            sid = line[1].strip()
            if sid:
                raw_ids.append(sid)
    raw_set = set(raw_ids)

    only_raw = sorted(raw_set - final_ids)
    only_final = sorted(final_ids - raw_set)

    print("\n=== Raw (配方首行) vs Final sample_id ===")
    print(f"raw unique formulations = {len(raw_set)} (max 序号 ≈ {int(max_seq) if max_seq is not None else 'n/a'})")
    print(f"final unique = {len(final_ids)}")
    print(f"only in raw ({len(only_raw)}): {only_raw}")
    print(f"only in final ({len(only_final)}): {only_final}")


if __name__ == "__main__":
    main()
