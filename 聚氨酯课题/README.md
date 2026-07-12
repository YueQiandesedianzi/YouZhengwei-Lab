# 课题组传承包：机器学习数据与代码（精简版）

对应学位论文：《机器学习辅助动态聚氨酯配方设计与多目标优化研究》（上传版）。

本目录是从原始工作区精选后的**可传承数据包**：只保留代码与表格数据，不含图片、PPT、venv、拉力机原始曲线、探索分支与中间备份。

## 1. 先看什么（5 分钟）

| 优先级 | 文件 | 作用 |
|---|---|---|
| 1 | `DATA_MANIFEST.md` | **完整数据清单与复现路径（必读）** |
| 2 | `00_论文对照.md` | 论文章节 ↔ 数据/脚本对照 |
| 3 | `chapter3_DA/data/Final_Features_Ultimate_all_deletion.csv` | 第三章定稿特征表（84 条） |
| 4 | `chapter4_SMILES/data/upstream/00_raw_literature/` | 第四章文献原始搜集三表 |
| 5 | `chapter4_SMILES/data/stage03/task03_final_features_for_modeling.csv` | 第四章统一建模输入（锁定） |
| 6 | `chapter4_SMILES/data/locked/` + `results_reference/` | 定版结果与超参 |

日常使用建议：

- 读论文数字 / 复述结论：优先看 `results_reference/` 与 `data/locked/`。
- 继续建模 / 改特征：从 `task03_final_features_for_modeling.csv` 起步。
- 重新跑全流程：按下面「推荐运行顺序」。

## 2. 目录结构

```
课题组传承_机器学习数据与代码/
├── README.md                      ← 本文件
├── 00_论文对照.md
├── 03_不纳入说明.md
├── requirements.txt
├── chapter3_DA/                   ← 论文第三章（DA 小样本）
│   ├── data/                      原始整合表 + 92/84 特征定稿
│   ├── scripts/                   清洗/消融/校验
│   ├── results/                   算法对比与消融结果表
│   └── docs/                      方法学说明
├── DATA_MANIFEST.md               ← 数据层清单（复现必读）
└── chapter4_SMILES/               ← 论文第四章（跨文献 SMILES）
    ├── data/
    │   ├── upstream/
    │   │   ├── 00_raw_literature/     文献原始三表
    │   │   ├── 01_merge_intermediate/ 合并中间表
    │   │   ├── 02_cleaned_domain/     Step3/4/8 清洗表
    │   │   └── 03_smiles_dicts/       SMILES 字典
    │   ├── stage02/               SMILES 加权特征产物
    │   ├── stage03/               Pearson+PCA 后建模输入
    │   └── locked/                论文锁定快照（特征子集+定版结果）
    ├── scripts/                   相对路径可运行脚本（无出图）
    ├── results_reference/         历史定版结果（勿轻易改）
    └── outputs/                   你重新运行时的新输出（自动创建）
```

## 3. 环境

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

说明：

- 第三章消融脚本依赖 `xgboost`。
- 第四章 SHAP / Pareto 依赖 `shap`；可选重建 Stage-02 需要 `rdkit`。

## 4. 推荐使用路径

### A. 只核对论文结果（最快）

1. 第三章：`chapter3_DA/results/` + `docs/数据清洗与样本集方法学说明.md`
2. 第四章：`chapter4_SMILES/results_reference/task07|task08|task09/`
3. 对照 `00_论文对照.md` 中的表号/口径

### B. 从定版特征重跑解释 / Pareto

```bash
cd chapter4_SMILES/scripts
python run_08_robustness.py
python run_08_shap.py          # 较慢
python run_09_pareto.py        # 依赖 pymoo 或自实现部分，见脚本
```

输入已固定为：

- `data/stage03/task03_final_features_for_modeling.csv`
- `data/locked/task07_final_locked_result.csv`
- `data/locked/task07_final_feature_sets/`
- `results_reference/task05/*params*.json`

### C. 从建模矩阵重跑基线→调参→定版（耗时）

```bash
cd chapter4_SMILES/scripts
python run_04_benchmark.py
python run_05_tune_ablation.py
python run_05_deep_tune_v2.py
python run_07_finalize.py
```

新结果写入 `chapter4_SMILES/outputs/`，不会覆盖 `results_reference/`。

### D. 可选：重建 Stage-02 / Stage-03

```bash
python rebuild_stage02_smiles_features.py   # 需 RDKit；不保证与锁定表 bit-exact
python rebuild_stage03_pca.py               # 从 stage02 重建；默认写入 outputs/
```

论文复现请始终优先使用 `data/stage02` 与 `data/stage03` 中的锁定 CSV。

### E. 第三章

```bash
cd chapter3_DA/scripts
python verify_dataset_counts.py
python build_features_from_raw.py
python run_ablation_nested_loocv.py   # 可选重跑消融
```

## 5. 数据流（第四章主链）

```
00_raw_literature（三类动态键原始搜集表）
        ↓
01_merge_intermediate（综合数据 → final_merged_dataset）
        ↓
02_cleaned_domain（Step3/4/8，清洗后约 314 行）
        ↓
stage02（SMILES加权 + domain合并）
        ↓
stage03（Pearson去冗余 + PCA95% → 建模矩阵）
        ↓
task04 多模型横比 → task05 调参消融 → task07 定版
        ↓
task08 SHAP/稳健性 → task09 NSGA-II Pareto
```

中间探索阶段 task06（旧特征对照）未纳入，因终稿未以其为选择依据。  
各层文件细节见 `DATA_MANIFEST.md`。

## 6. 注意事项

1. **口径锁定**：论文数字以 `results_reference` + `data/locked` 为准。
2. **不要改锁定文件**：新实验请写到 `outputs/`。
3. **绝对路径已去除**：脚本通过 `config.py` 使用相对路径。
4. **无图**：出图逻辑已剥离；需要图请自行在结果 CSV 上绘制。
5. **原始工作区仍保留**：本包是拷贝精选，未删除原 `多文献课题_2` / `小论文/实验过程`。
