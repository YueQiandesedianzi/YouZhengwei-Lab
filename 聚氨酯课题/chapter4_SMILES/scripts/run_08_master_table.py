from __future__ import annotations

# Adapted for inheritance package: relative paths, no figure export.
from config import (
    ROOT, DATA, STAGE03, LOCKED, REF, OUT,
    MODELING_CSV, PARAM_V1, PARAM_V2, FALLBACK, TASK04_METRICS,
    FINAL_LOCK, FINAL_FEAT_DIR, META_COLS, TARGET_COLS,
)

from pathlib import Path
from typing import List

import pandas as pd

TASK8 = OUT / 'task08'

DISPLAY_NAME_MAP = {
    'sigma_max_MPa': 'Tensile Strength',
    'sigma_healed_MPa': 'Tensile Strength Healed',
    'epsilon_break_pct': 'Elongation at Break',
    'epsilon_healed_pct': 'Elongation at Break Healed',
    'E_MPa': 'Toughness',
    'E_healed_MPa': 'Toughness Healed',
    'Tg': 'Tg',
}


def latest_output_dir(prefix: str) -> Path:
    cands = sorted(TASK8.glob(f'{prefix}_snapshot_*'))
    if not cands:
        raise FileNotFoundError(f'Cannot find output dir with prefix: {prefix}')
    return cands[-1]


def pick_top_features(top_df: pd.DataFrame, target: str, top_n: int = 3) -> str:
    sub = top_df[top_df['target'] == target].sort_values('rank').head(top_n)
    feats: List[str] = sub['feature'].tolist()
    return ', '.join(feats)


def build_markdown_table(df: pd.DataFrame) -> str:
    cols = [
        'target_display', 'selected_model', 'n_samples', 'n_features',
        'oof_r2', 'oof_r2_ci95', 'oof_rmse', 'oof_mae',
        'coverage_q95q05_band_0p1', 'top3_shap_features'
    ]
    d = df[cols].copy()
    d = d.rename(columns={
        'target_display': 'Target',
        'selected_model': 'Model',
        'n_samples': 'N',
        'n_features': 'Feat',
        'oof_r2': 'OOF R2',
        'oof_r2_ci95': 'R2 95% CI',
        'oof_rmse': 'OOF RMSE',
        'oof_mae': 'OOF MAE',
        'coverage_q95q05_band_0p1': 'Coverage(0.1·Q95-Q05)',
        'top3_shap_features': 'Top3 SHAP Features',
    })
    for col in ['OOF R2', 'OOF RMSE', 'OOF MAE', 'Coverage(0.1·Q95-Q05)']:
        d[col] = d[col].map(lambda x: f'{x:.4f}')

    header = '| ' + ' | '.join(d.columns) + ' |'
    sep = '| ' + ' | '.join(['---'] * len(d.columns)) + ' |'
    rows = ['| ' + ' | '.join(map(str, row)) + ' |' for row in d.to_numpy()]
    return '\n'.join([header, sep] + rows)


def main() -> None:
    shap_dir = latest_output_dir('task08_shap_outputs')
    robust_dir = latest_output_dir('task08_robustness_outputs')

    shap_summary = pd.read_csv(shap_dir / 'task08_shap_summary.csv')
    shap_top = pd.read_csv(shap_dir / 'task08_shap_top10_features_all_targets.csv')
    robust = pd.read_csv(robust_dir / 'task08_robustness_summary.csv')

    merged = robust.merge(
        shap_summary[['target', 'global_plot']],
        on='target',
        how='left',
    )
    merged['target_display'] = merged['target'].map(DISPLAY_NAME_MAP).fillna(merged['target'])
    merged['oof_r2_ci95'] = merged.apply(
        lambda r: f"[{r['oof_r2_ci95_low']:.4f}, {r['oof_r2_ci95_high']:.4f}]", axis=1
    )
    merged['top3_shap_features'] = merged['target'].map(lambda t: pick_top_features(shap_top, t, top_n=3))

    out_cols = [
        'target', 'target_display', 'selected_model', 'n_samples', 'n_features',
        'oof_r2', 'oof_r2_ci95_low', 'oof_r2_ci95_high', 'oof_r2_ci95',
        'oof_rmse', 'oof_mae', 'oof_medae',
        'coverage_q95q05_band_0p1', 'coverage_q95q05_band_0p2',
        'band_0p1_abs_width', 'band_0p2_abs_width',
        'top3_shap_features', 'global_plot', 'diag_plot'
    ]
    out_df = merged[out_cols].sort_values('target').reset_index(drop=True)

    snap_tag = shap_dir.name.replace('task08_shap_outputs_', '')
    out_root = TASK8 / f'task08_paper_master_outputs_{snap_tag}'
    out_root.mkdir(parents=True, exist_ok=True)

    out_csv = out_root / 'task08_paper_master_results.csv'
    out_md = out_root / 'task08_paper_master_results.md'
    out_fig_index = out_root / 'task08_figure_index.csv'

    out_df.to_csv(out_csv, index=False)

    md_table = build_markdown_table(out_df)
    md_text = (
        '# Task08 论文主结果总表（快照版）\n\n'
        f'- 数据来源：`{snap_tag}`\n'
        f'- SHAP目录：`{shap_dir}`\n'
        f'- 稳健性目录：`{robust_dir}`\n\n'
        '## 主结果表\n\n'
        + md_table
        + '\n'
    )
    out_md.write_text(md_text, encoding='utf-8')

    fig_index = out_df[[
        'target', 'target_display', 'selected_model', 'global_plot', 'diag_plot'
    ]].copy()
    fig_index['task07_regression_plot'] = ''  # figures intentionally omitted in inheritance package
    fig_index.to_csv(out_fig_index, index=False)

    print(f'Master table built: {out_root}')
    print(out_df[['target', 'selected_model', 'oof_r2', 'coverage_q95q05_band_0p1', 'top3_shap_features']].to_string(index=False))


if __name__ == '__main__':
    main()
