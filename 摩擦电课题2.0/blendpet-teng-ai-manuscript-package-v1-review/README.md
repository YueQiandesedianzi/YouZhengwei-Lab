# BlendPET-TENG：配方智能识别与比例估算的AI证据包

## 技术结论

当前最稳妥的数据故事不是用一个复杂网络直接从波形回归任意配比，而是先把TENG波形转换为具有物理含义的多维“配方指纹”，再完成两个层次的任务：**RBF-SVM识别15种预设配方是主要结果；已知材料体系后的配比回归是辅助性的半定量结果。**

在136个宏块、15种已观测配方上，无合成增强的RBF-SVM获得平均准确率0.969、平衡准确率0.971和Macro-F1 0.966。汇总混淆矩阵中132/136个宏块被正确识别，剩余4个错误集中在相邻PGS配方和同一25%比例的PGS/PGT体系之间。相关数字见[候选模型汇总](data/classification/candidate-summary.csv)和[混淆矩阵源数据](data/classification/selected-confusion-matrix.csv)。

分体系回归在12个混合配方上的总体MAE为8.07个百分点、R²为0.818。这个总体数字掩盖了局部偏差：PGS主要受高比例响应压缩影响，PGT在45%—75%区间出现更明显的平台和非单调变化。因此，现阶段回归应表述为“已知体系下的比例趋势或半定量估算”；要升级为全范围定量预测，需要针对问题区间重新制样并补充浓度节点。逐配方结果见[回归预测表](data/regression/calibration-predictions.csv)。

## 文章中的AI贡献应该怎样表述

这项工作的算法贡献不是提出全新的机器学习架构，而是建立一条与传感机理相匹配、可以解释和复核的分析路径：

1. 将原始TENG波形切分为周期，并通过质量控制排除明显异常周期；
2. 从每个周期提取峰峰值、正负峰、RMS、正负面积、峰值不对称性、波瓣面积平衡、峰间相位和波峰因子等10个特征；
3. 对连续12个周期进行稳健聚合，降低单周期波动对判断的影响；
4. 在完整物理特征空间中使用RBF-SVM识别15种预设配方；
5. 当PGS或PGT体系已知时，单独建立配比校准模型，输出辅助性的比例估计。

![Physics-informed AI workflow](figures/main/01-ai-workflow.png)

[PNG](figures/main/01-ai-workflow.png) · [PDF](figures/main/01-ai-workflow.pdf) · [SVG](figures/main/01-ai-workflow.svg)

这张图建议作为AI部分的第一幅子图。它把“波形—物理特征—配方分类—条件回归”的关系一次讲清楚，也避免把文章包装成以深度神经网络为核心的工作。

## 数据范围和指标定义

本包只包含复现结果和图件所需的派生数据，不包含完整原始波形。核心分析单位为宏块：每个宏块由连续周期的特征中位数构成；当前共有136个宏块，覆盖15种配方。分类使用每种配方内部的连续宏块窗口进行五折评估，每个测试折均覆盖15类。Macro-F1用于平衡评价各配方，平衡准确率是15个类别召回率的平均值。

这里的分类是**封闭配方库识别**：输入样本被判断为当前15种配方之一。它不等同于识别从未出现过的新配方，也不直接证明跨新批次、新器件或新实验条件的性能。文章应把应用边界明确写成“预设配方筛查与质量控制”。数据覆盖情况见[分析单位汇总](data/manifests/analysis-unit-summary.csv)。

## 数据故事一：组成变化首先改变的是多维波形指纹

峰峰值随配比总体发生变化，但不是所有区间都保持简单线性。PGS从0%到75%下降明显，75%—100%进入压缩区；PGT在45%—75%之间出现平台和局部反转。这说明单独依赖一个电压幅值难以同时完成所有配方的稳定识别和连续定量。

![Composition-dependent waveform response](figures/main/02-feature-response.png)

[PNG](figures/main/02-feature-response.png) · [PDF](figures/main/02-feature-response.pdf) · [SVG](figures/main/02-feature-response.svg) · [源数据](data/features/formula-feature-summary.csv)

文章中的关键转折应写成：**虽然单一幅值与配比之间并非处处线性，但不同材料体系和配比仍在峰值、面积、相位和形状等多个维度形成可区分的联合模式。** 因而，分类明显比全范围连续回归更符合当前传感响应的真实结构。

## 数据故事二：非线性边界是15配方分类成功的关键

线性Ridge分类器的Macro-F1只有0.657，而RBF-SVM达到0.966，Extra Trees也达到0.961。两类非线性模型均显著优于线性基线，说明15种配方在物理特征空间中具有可分结构，但边界不是一个简单线性关系。

![Classification model comparison](figures/main/03-model-comparison.png)

[PNG](figures/main/03-model-comparison.png) · [PDF](figures/main/03-model-comparison.pdf) · [SVG](figures/main/03-model-comparison.svg) · [源数据](data/classification/candidate-summary.csv)

正文推荐RBF-SVM而不是将合成增强后的最高单点结果作为主结果，原因是RBF-SVM在不增加虚拟样本的条件下已经稳定达到较高水平，决策边界清楚，模型复杂度适合当前数据规模。Extra Trees可作为非线性模型对照，证明结论并不依赖单一算法。

## 数据故事三：分类错误数量少，而且具有明确结构

RBF-SVM的汇总混淆矩阵中只有三种错误方向：45PGD+55PGS与50PGD+50PGS各有1个宏块相互混淆；75PGD+25PGT有2个宏块被判断为75PGD+25PGS。前一组是相邻配比，后一组保持了25%的比例信息但混淆了材料体系。因此，模型的大部分剩余错误不是随机散布，而是集中在物理响应相近的局部区域。

![RBF-SVM confusion matrix](figures/main/04-rbf-svm-classification.png)

[PNG](figures/main/04-rbf-svm-classification.png) · [PDF](figures/main/04-rbf-svm-classification.pdf) · [SVG](figures/main/04-rbf-svm-classification.svg) · [混淆矩阵](data/classification/selected-confusion-matrix.csv) · [逐类召回率](data/classification/selected-per-class-recall.csv)

文章可以将其解释为：TENG平台能够同时编码“材料体系”和“混合比例”两类信息，但在少数响应接近的邻域中，两类信息的可分性并不完全一致。

## 数据故事四：完整特征组合优于单一幅值或少量比值

特征消融进一步支持“多维指纹”而不是“单一电压标定”的叙事。完整10特征的Macro-F1为0.966，仅使用波形形状特征为0.887，仅使用幅值特征为0.867，仅使用少量峰值/面积比值则降至0.574。

![Feature-set ablation](figures/main/05-feature-ablation.png)

[PNG](figures/main/05-feature-ablation.png) · [PDF](figures/main/05-feature-ablation.pdf) · [SVG](figures/main/05-feature-ablation.svg) · [源数据](data/classification/feature-robustness-summary-metrics.csv)

这组结果适合作为算法部分的解释性证据：幅值变化提供主要组成信息，波形形态补充材料体系和局部配比信息，两者结合才形成完整的配方指纹。

## 数据故事五：回归的总体R²可用，但偏差集中在响应压缩区

回归不能只报告总体R²。PGS的MAE为6.87个百分点，主要问题是90%被预测为71.97%；PGT的MAE为9.27个百分点，45%、50%、60%和75%之间出现交错偏差。尤其75% PGT被预测为58.15%，说明模型倾向于把高比例结果拉回中间区域。

![Family-conditioned regression](figures/diagnostic/06-family-regression.png)

[PNG](figures/diagnostic/06-family-regression.png) · [PDF](figures/diagnostic/06-family-regression.pdf) · [SVG](figures/diagnostic/06-family-regression.svg) · [逐配方预测](data/regression/calibration-predictions.csv) · [汇总指标](data/regression/calibration-metrics.csv)

这种误差不是一个可以通过整体加减常数消除的系统偏移。PGS高比例区和PGT中高比例区的特征响应本身发生压缩或非单调变化，不同回归模型也在相近位置出现较大误差。因此，继续增加模型复杂度的收益有限；更重要的是确认这些区间的响应形状是否能够在独立制备样品上重复。

## 回归要升级为定量结果，实验应该补在哪里

建议分两步进行。第一步只复测现有问题点：PGS的40%、75%、90%，PGT的45%、50%、60%、75%，并同时测量0%和100%端点。每个比例至少使用3个独立制备样品。第二步只有在响应趋势可重复后才增加浓度节点：PGS优先补60%、65%、80%、85%，PGT优先补35%、55%、65%、85%。

![Targeted regression experiments](figures/diagnostic/07-regression-experiment-priority.png)

[PNG](figures/diagnostic/07-regression-experiment-priority.png) · [PDF](figures/diagnostic/07-regression-experiment-priority.pdf) · [SVG](figures/diagnostic/07-regression-experiment-priority.svg) · [建议点位表](data/regression/regression-experiment-priority.csv)

判断是否值得继续回归的标准不是R²是否再次上升，而是以下两个问题：同一比例的独立样品能否获得相近的特征分布；相邻比例的组间差异是否持续大于组内波动。如果答案为是，可以建立分段、单调或非线性的校准模型；如果答案为否，文章应保留半定量结论并明确有效区间。

## 建议的AI小节结构

论文中AI内容不需要占据过多篇幅，可以组织成一个主图和一个补充图组：

1. **智能分析流程。** 介绍周期质量控制、10特征提取、12周期稳健聚合，以及分类与条件回归两个输出；
2. **15种预设配方识别。** 报告模型对比、RBF-SVM的Macro-F1/平衡准确率和混淆矩阵；
3. **特征解释与稳健性。** 用消融结果说明多维物理指纹的必要性，并把增强实验放入补充材料；
4. **已知体系下的配比估算。** 分开报告PGS和PGT，展示真实值—预测值及残差，避免只给总体R²；
5. **应用边界。** 将当前能力定义为预设配方筛查与质量控制；补实验完成后再决定是否升级为全范围定量监测。

可直接使用的中文方法和结果草稿分别位于[METHODS_AI.md](METHODS_AI.md)和[RESULTS_AI.md](RESULTS_AI.md)。

## 补充图：原始分类组合图

下面保留现有RBF-SVM组合图，适合放入补充材料或作为最终排版参考。

![Original RBF-SVM classification figure](figures/supplementary/original-rbf-svm-15-formula-classification.png)

[PNG](figures/supplementary/original-rbf-svm-15-formula-classification.png) · [PDF](figures/supplementary/original-rbf-svm-15-formula-classification.pdf)

## 历史与诊断图图库

以下图片全部保留并直接展示，便于审核时决定是否使用。它们不应与上述推荐主图等权处理：旧CNN架构、训练过程和原始CNN结果属于路线比较或历史记录；旧回归和性能改进图需要以当前最终结论重新审阅后再进入稿件。

<details>
<summary>旧版七张主图</summary>

### 旧版图1：CNN architecture

![Legacy CNN architecture](figures/archive/legacy-figure-01-cnn-architecture.png)

[PNG](figures/archive/legacy-figure-01-cnn-architecture.png) · [PDF](figures/archive/legacy-figure-01-cnn-architecture.pdf) · [SVG](figures/archive/legacy-figure-01-cnn-architecture.svg)

### 旧版图2：Training process

![Legacy training process](figures/archive/legacy-figure-02-training-process.png)

[PNG](figures/archive/legacy-figure-02-training-process.png) · [PDF](figures/archive/legacy-figure-02-training-process.pdf) · [SVG](figures/archive/legacy-figure-02-training-process.svg)

### 旧版图3：Classification performance

![Legacy classification performance](figures/archive/legacy-figure-03-classification-performance.png)

[PNG](figures/archive/legacy-figure-03-classification-performance.png) · [PDF](figures/archive/legacy-figure-03-classification-performance.pdf) · [SVG](figures/archive/legacy-figure-03-classification-performance.svg)

### 旧版图4：Regression performance

![Legacy regression performance](figures/archive/legacy-figure-04-regression-performance.png)

[PNG](figures/archive/legacy-figure-04-regression-performance.png) · [PDF](figures/archive/legacy-figure-04-regression-performance.pdf) · [SVG](figures/archive/legacy-figure-04-regression-performance.svg)

### 旧版图5：Performance improvement

![Legacy performance improvement](figures/archive/legacy-figure-05-performance-improvement.png)

[PNG](figures/archive/legacy-figure-05-performance-improvement.png) · [PDF](figures/archive/legacy-figure-05-performance-improvement.pdf) · [SVG](figures/archive/legacy-figure-05-performance-improvement.svg)

### 旧版图6：Quality control

![Legacy quality control](figures/archive/legacy-figure-06-quality-control.png)

[PNG](figures/archive/legacy-figure-06-quality-control.png) · [PDF](figures/archive/legacy-figure-06-quality-control.pdf) · [SVG](figures/archive/legacy-figure-06-quality-control.svg)

### 旧版图7：Complexity and generalization

![Legacy complexity and generalization](figures/archive/legacy-figure-07-complexity-generalization.png)

[PNG](figures/archive/legacy-figure-07-complexity-generalization.png) · [PDF](figures/archive/legacy-figure-07-complexity-generalization.pdf) · [SVG](figures/archive/legacy-figure-07-complexity-generalization.svg)

</details>

<details>
<summary>原始波形CNN：早停版本</summary>

![CNN early stop condition comparison](figures/archive/cnn-early-stop-raw-waveform-cnn-condition-comparison.png)

![CNN early stop paired delta](figures/archive/cnn-early-stop-raw-waveform-cnn-paired-delta.png)

![CNN early stop QC](figures/archive/cnn-early-stop-raw-waveform-cnn-qc.png)

![CNN early stop confusion](figures/archive/cnn-early-stop-raw-waveform-cnn-selected-confusion.png)

![CNN early stop per-class recall](figures/archive/cnn-early-stop-raw-waveform-cnn-selected-per-class-recall.png)

</details>

<details>
<summary>原始波形CNN：原始训练版本</summary>

![CNN original condition comparison](figures/archive/cnn-original-raw-waveform-cnn-condition-comparison.png)

![CNN original paired delta](figures/archive/cnn-original-raw-waveform-cnn-paired-delta.png)

![CNN original QC](figures/archive/cnn-original-raw-waveform-cnn-qc.png)

![CNN original confusion](figures/archive/cnn-original-raw-waveform-cnn-selected-confusion.png)

![CNN original per-class recall](figures/archive/cnn-original-raw-waveform-cnn-selected-per-class-recall.png)

</details>

## 如何复现本包数字和新图件

本包的复现范围是：从已保存的派生CSV重新计算本文引用的分类与回归指标，并重新生成`figures/main/`和`figures/diagnostic/`中的7张新图。完整原始波形不在本包内，因此这里不负责从电压文件重新训练全部候选模型。

在文件夹根目录运行：

```bash
python3 -m pip install -r code/requirements.txt
python3 code/reproduce.py --validate
```

成功运行后会刷新[data/manifests/reproduction-metrics.json](data/manifests/reproduction-metrics.json)、7张新图和[文件校验清单](data/manifests/package-manifest.csv)。

## 文件导航

- `data/classification/`：模型对比、混淆矩阵、逐类召回率、增强和特征消融；
- `data/regression/`：逐配方回归预测、汇总指标、PLS/GPR对照和建议实验点；
- `data/features/`：136个宏块特征、配方级特征摘要和重复性统计；
- `figures/main/`：建议进入论文AI主图的5张图；
- `figures/diagnostic/`：回归偏差和补实验设计图；
- `figures/supplementary/`：可以进入补充材料的现有组合图；
- `figures/archive/`：全部历史和CNN诊断图；
- `code/reproduce.py`：从包内CSV重算指标并重绘7张新图；
- `REVIEW_CHECKLIST.md`：上传前需要人工确认的项目。

## 当前结论的边界

分类结果支持当前15种已观测配方之间的封闭式识别，不应扩展为任意未知配方识别。回归结果支持已知PGS/PGT体系下的半定量估计，但尚不足以支持0%—100%全区间的高精度连续预测。最值得补充的证据不是更复杂的网络，而是问题浓度点的独立重复和中间浓度校准。

这个边界并不削弱文章故事。相反，它使AI部分与TENG平台的实际能力保持一致：**平台产生配方相关的多维信号，机器学习把这些信号转化为可操作的配方识别结果，并在可分辨区间内提供比例信息。**
