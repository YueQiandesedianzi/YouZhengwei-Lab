# 附录 A：特征空间马氏距离离群筛查（可复现）

本文档由同目录脚本 `mahalanobis_feature_outliers.py --write-appendix` 自动生成，
便于与正文「马氏距离 + χ² 阈值」表述及人工定稿名单交叉核对。

## 1. 数据与特征

- **输入文件**：`Final_Features_Ultimate.csv`（当前 **92** 条配方）。
- **特征列（p = 20）**：组成 / 拓扑 / 工艺编码，**不含** `tensile_strength`、`elongation`、`healing_eff`。
- **预处理**：列内 z-score；样本协方差加岭 **λ = 1e-03 · I** 后求逆。

## 2. 判据

- 马氏距离平方 \(d_i^2\)（基于上述标准化特征与岭协方差）。
- 参照分布：多元正态近似下 \(d_i^2 \sim \chi^2_p\)；取 **α = 0.05**，临界值 **χ²_{p,0.95} ≈ 31.4017**（Wilson–Hilferty 近似）。

## 3. 超过 χ² 阈值的配方（算法标记）

| sample_id | d² |
|-----------|-----|
| FAm-DA-PU | 85.2022 |
| HTK-PU（凝胶，没有软段） | 80.9303 |
| PU-BDO-UP | 59.9585 |
| BQDU-CPU | 54.0856 |
| 5%Cu-BQDU-CPU | 53.1891 |
| TCPUA-B-20% | 36.1239 |
| TCPUA-E-15% | 35.8644 |
| ITB-PU | 35.3493 |
| DCPU-WYP | 34.3005 |
| DA-PU | 32.0755 |

## 4. 与定稿表 `Final_Features_Ultimate_all_deletion.csv` 的对比

从 Ultimate 中**人工剔除**并进入 all_deletion 所**保留**的补集，对应剔除集合大小为 **8**（见方法学说明正文）。

- **算法标记数**：10
- **与人工剔除名单的交集**（4）：`5%Cu-BQDU-CPU, DCPU-WYP, HTK-PU（凝胶，没有软段）, ITB-PU`
- **仅人工剔除、算法未标记**（4）：`LPU-WYP-2, PBA-PU-3(热熔胶，自愈合无用), PU-DA-PPG-R2.0-hard1, SPU3`
- **仅算法标记、人工仍保留**（6）：`BQDU-CPU, DA-PU, FAm-DA-PU, PU-BDO-UP, TCPUA-B-20%, TCPUA-E-15%`

**说明**：马氏距离依赖多元正态与协方差估计，小样本下与「残差筛查 + 领域知识」定稿名单不必完全一致；
正文可写「马氏距离复核」或先调整 α、岭或稳健协方差后再与定稿对齐。
