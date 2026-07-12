"""PGDS/PGTD 摩擦电信号 → 配方比例预测（精简归档版）。

读代码建议顺序：
1. ratio_dual_model.py          主流程（清洗→分段→训练→指标）
2. ratio_deep_learning_publication_panels.py  由主流程结果出 DL 面板图
3. ratio_iterative_story.py     外部 10/45/60% 迭代验证故事
4. ratio_publication_story_panels.py / build_linearity_proof_panel.py
   电信号 Vpp 趋势与线性证明图（不依赖 CNN 重训）
"""

__all__ = ["__version__"]

__version__ = "1.0.0-archive"
