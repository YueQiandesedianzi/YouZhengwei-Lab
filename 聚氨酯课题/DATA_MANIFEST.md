# 数据清单（复现用）

本文件说明传承包内每一层数据“从哪来、干什么、下一步读谁”。  
**不含图片与拉力机原始曲线。**

---

## 第三章 `chapter3_DA/data/`

| 文件 | 角色 | 行数口径 |
|---|---|---|
| `DA-PU数据整合-new_furan_cleaned.csv` | 文献手工提取后的整合原始表 | 配方级 |
| `Final_Features_Ultimate.csv` | 特征工程后、离群剔除前 | 92 |
| `Final_Features_Ultimate_all_deletion.csv` | **论文定稿建模表** | 84（愈合子集 37） |

结果表在 `chapter3_DA/results/`（算法对比、消融），不参与特征重建，但可核对论文数字。

---

## 第四章数据链（按复现顺序读）

### A. 文献原始搜集表  
`chapter4_SMILES/data/upstream/00_raw_literature/`

| 文件 | 内容 |
|---|---|
| `DA动态键.csv` | DA 体系文献提取原始表 |
| `二硫动态键.csv` | 二硫键体系原始表 |
| `硼酸酯键.csv` | 硼酸酯体系原始表 |

这是第四章真正的“搜集到的原始数据”起点。

### B. 合并中间表  
`chapter4_SMILES/data/upstream/01_merge_intermediate/`

| 文件 | 角色 |
|---|---|
| `综合数据文件.csv` | 三体系初步合并 |
| `综合数据_step1.csv` / `综合数据_step2.csv` | 字段统一与清洗中间版 |
| `merged_all_raw.csv` | Step2 全量合并原始 |
| `merged_clean.csv` | Step2 清洗版 |
| `final_merged_dataset.csv` | **含 SMILES + 质量分数的关键主表**（Stage02 对齐用） |
| `feature_matrix_full_aligned.csv` | 对齐后的特征矩阵 |

### C. Domain 清洗与 ML 就绪  
`chapter4_SMILES/data/upstream/02_cleaned_domain/`

| 文件 | 角色 |
|---|---|
| `feature_matrix_step3_v2.csv` | 模块化特征定稿前 |
| `feature_matrix_step4.csv` | **清洗后 domain 主表（314 行口径）** |
| `outlier_log.csv` / `dropped_rows_detail.csv` | 清洗剔除记录 |
| `feature_matrix_ml_full.csv` | ML 就绪全表（可行性核对用） |
| `cv_strategy.csv` | 交叉验证策略记录 |

### D. SMILES 字典  
`chapter4_SMILES/data/upstream/03_smiles_dicts/`

| 文件 | 角色 |
|---|---|
| `组分SMILES字典.csv` | 组分名→SMILES 初版 |
| `组分SMILES字典_validated.csv` | 校验后字典 |
| `组分SMILES汇总表.csv` | 汇总 |

### E. Stage02 / Stage03（第四章主建模链）

| 目录 | 关键文件 | 角色 |
|---|---|---|
| `data/stage02/` | `task02_domain_plus_molecular_full314.csv` | SMILES 加权 + domain 合并 |
| `data/stage02/` | `task02_domain_plus_molecular_usable_valid199.csv` | RDKit 可解析子集 |
| `data/stage03/` | `task03_final_features_for_modeling.csv` | **统一建模输入（推荐起点）** |

### F. 论文锁定定版

| 路径 | 角色 |
|---|---|
| `data/locked/task07_final_locked_result.csv` | 七目标最终模型与 OOF 指标 |
| `data/locked/task07_final_feature_sets/` | 每目标入模特征列 |
| `data/locked/task07_pruned_feature_sets/` | R² 最优子集（对照用） |
| `results_reference/task04~09/` | 已跑通的结果表与超参 JSON |

---

## 兼容扁平副本（脚本直接读）

为兼容现有脚本，以下文件在 `data/upstream/` 根目录也保留一份同名副本：

- `final_merged_dataset.csv`
- `feature_matrix_step4.csv`
- `综合数据文件.csv`
- `组分SMILES字典_validated.csv`
- `feasibility_*.csv`

---

## 最小复现三档

1. **只复现论文数字**：读 `results_reference/` + `data/locked/`  
2. **从建模矩阵重跑 ML**：读 `stage03/task03_final_features_for_modeling.csv` + `results_reference/task05/*params*`  
3. **从原始文献表重建全链**：  
   `00_raw_literature` → `01_merge_intermediate` → `02_cleaned_domain` → `stage02` → `stage03` → `run_04…09`  
   （Stage02/03 可用包内锁定 CSV，或运行 `rebuild_stage02/03` 教学重建）
