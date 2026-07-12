# 摩擦电比例预测课题

> 本目录是游正伟课题组资料库（`YouZhengwei-Lab`）下的课题传承包。
> 面向师兄写论文：可顺着读代码逻辑，也可自行重跑出图。

给不懂代码也能顺着读、并能自己重跑出图的归档包。  
**不含**成品图合集（你那边已有），**不含**旧探索代码与中间训练缓存。

---

## 1. 这个包在做什么（一句话）

用 PGDS / PGTD 两种材料的摩擦电电压波形，训练模型：

1. **分类**：识别离散比例 `0 / 25 / 50 / 75 / 100%`
2. **回归**：从波形反推连续比例
3. **外部验证**：用后来补测的 `10 / 45 / 60%` 讲「校准 → 再验证」的故事
4. **电信号证据**：用局部峰峰值 Vpp 证明比例变化确实带来近似线性的电压响应

写论文时务必分清两类 R²：

| 数字来源 | 含义 | 能不能当成「模型预测精度」 |
|---------|------|---------------------------|
| CNN 回归 R² / MAE | 测试集上电信号→比例的模型表现 | 可以 |
| Vpp–比例线性拟合 R² | 材料电信号趋势本身是否近似线性 | **不可以**当成模型精度 |

---

## 2. 文件夹怎么看

```text
摩擦电比例预测课题/
├── README.md                 ← 你正在读的说明
├── requirements.txt
├── data/
│   ├── PGDS/                 ← 训练用原始 CSV（五个锚点）
│   ├── PGTD/
│   ├── external/PGDS/        ← 外部 10/45/60%（Origin 批次）
│   ├── external_PGTD/        ← 外部 10/45/60%（发袁李最新批次）
│   ├── vpp_points.csv        ← 已汇总的局部峰峰值（画趋势/线性图）
│   └── external_vpp_comparison.csv
└── teng_ml/                  ← 仅保留可出图的核心代码
    ├── ratio_dual_model.py                      ★ 主流程
    ├── ratio_deep_learning_publication_panels.py ★ DL 面板出图
    ├── ratio_iterative_story.py                 ★ 外部点迭代故事
    ├── ratio_publication_story_panels.py        ★ Vpp 趋势面板
    └── build_linearity_proof_panel.py           ★ 线性证明图
```

跑完主流程后，结果会出现在：

`data/ratio_dual_model/run_时间戳/`

里面按 `00_…` 到 `11_…` 编号，对应处理步骤。

---

## 3. 代码动向逻辑（按阅读顺序）

```text
原始 CSV
   │
   ▼
① ratio_dual_model.py
   清洗 → 周期分段 → 质控 → 划分训练/测试
   → Baseline/CNN 分类
   → Baseline/CNN 回归
   → 对比表 + 报告
   │
   ├──────────────────────────────┐
   ▼                              ▼
② DL 面板出图                   ③ 外部迭代故事
   publication_panels.py          iterative_story.py
   （读 run 目录里的 CSV 画图）    （45% 校准后再看 10/60%）
                                  │
                                  ▼
                               把 summary 再喂给 ②，可在回归图上叠外部点

④ / ⑤ 不依赖重训 CNN：
   直接读 data/vpp_points.csv
   → 画双材料 Vpp 趋势 / 线性证明
```

**师兄若只想先改「电信号趋势图」样式**：从第 ④/⑤ 步开始即可，几秒出图。  
**若要改「深度学习分类/回归图」**：必须先跑 ①，再跑 ②。

---

## 4. 环境与启动

在本归档文件夹根目录执行：

```bash
cd 摩擦电比例预测课题
python -m pip install -r requirements.txt
```

之后所有命令都在**本目录**下运行（保证能找到 `teng_ml` 包）。

---

## 5. 推荐运行顺序（可复制）

### 步骤 A：不训练，先出电信号图（最快）

```bash
# 双材料 Vpp 趋势 + 简单反演验证
python -m teng_ml.ratio_publication_story_panels

# 线性证明图（方程 / R2 / Pearson r）
python -m teng_ml.build_linearity_proof_panel
```

输出在：

- `outputs/vpp_story_panels/`
- `outputs/linearity_proof/`

### 步骤 B：训练主模型（耗时，需要 PyTorch）

```bash
# 正式出图建议提高 epoch；快速试跑可保持默认 60
python -m teng_ml.ratio_dual_model --run-name my_run --max-epochs 200
```

完成后看：

- `data/ratio_dual_model/my_run/11_final_report/report.md` ← 论文数字总览
- `data/ratio_dual_model/my_run/10_model_comparison/` ← Baseline vs CNN

### 步骤 C：画深度学习面板

把下面的 `RUN` 换成你的实际目录名：

```bash
RUN=data/ratio_dual_model/my_run

python -m teng_ml.ratio_deep_learning_publication_panels \
  --run-dir "$RUN"
```

图在：`$RUN/publication_panels/`

### 步骤 D：外部 10/45/60% 迭代故事（可选）

```bash
python -m teng_ml.ratio_iterative_story \
  --base-run data/ratio_dual_model/my_run \
  --run-name run_iterative_story

# 再把外部点叠到 DL 回归面板上
python -m teng_ml.ratio_deep_learning_publication_panels \
  --run-dir data/ratio_dual_model/my_run \
  --external-summary data/ratio_dual_model/run_iterative_story/tables/iterative_prediction_summary.csv \
  --output-dir data/ratio_dual_model/my_run/publication_panels_with_external
```

---

## 6. 想改图时改哪里

| 想改什么 | 主要改哪个文件 | 说明 |
|---------|----------------|------|
| 分类/回归主图样式、颜色、标题 | `ratio_deep_learning_publication_panels.py` | 函数名带 `draw_` 的都是画图 |
| 外部验证故事图 | `ratio_iterative_story.py` 里的 `draw_story_figures` | 改完重跑步骤 D |
| Vpp 趋势线、误差棒 | `ratio_publication_story_panels.py` | 不需重训 |
| 线性拟合证明图 | `build_linearity_proof_panel.py` | 不需重训 |
| 分段/质控/模型结构 | `ratio_dual_model.py` | 改完必须重跑步骤 B |

改图风格时，通常只动 `figsize`、`color`、`title`、图例文字即可；不必先读懂全部训练代码。

---

## 7. 论文写作可直接用的「故事线」

推荐叙述顺序（与之前最终图合集 A–G 一致）：

1. **先讲电信号**：比例变化 → Vpp 有稳定趋势（PGDS 正相关，PGTD 负相关；线性 R² 约 0.90 / 0.95）
2. **再讲分类**：CNN 能区分 0/25/50/75/100%（混淆矩阵接近对角）
3. **再讲回归**：同体系测试集上可做连续比例反演（历史结果约 R²=0.994，MAE≈1.2%；你重跑后以新 `report.md` 为准）
4. **再讲为何用 CNN**：Baseline 手工特征明显更弱
5. **最后讲外部点**：45% 作校准，10%/60% 作验证；要诚实写批次偏差，**不要**把外部点并进原始测试集 R²

---

## 8. 数据说明（为什么带原始 CSV）

没有原始波形，主流程无法重跑，也就无法按师兄自己的审美重画 DL 图。因此本包保留：

- 训练锚点：`data/PGDS`、`data/PGTD`
- 外部补测：`data/external/PGDS`、`data/external_PGTD`
- 已算好的 Vpp 汇总表：即使不重训，也能立刻重画趋势/线性图

文件名里的 `25%PGDS`、`75PGTD` 等会被代码自动解析成标签；新增 CSV 时请保持类似命名。

---

## 9. 常见注意点

- 必须在归档根目录运行 `python -m teng_ml.xxx`，不要先 `cd teng_ml`。
- `ratio_iterative_story` 依赖主流程产物里的 `05_augmentation/arrays/train_augmented.npz`，所以要先跑步骤 B。
- 外部 PGTD 使用「发袁李」最新批次，不是 Origin 文件夹里的旧 PGTD；这是有意选择，写作时可注明批次来源。
- 第一次完整训练可能较慢；只调图时可反复跑步骤 A / C，不必每次重训。

如需把某张图改成中文轴标题、或换成期刊双栏宽度，把目标图告诉合作者/AI，指向上表对应脚本即可。
