# Source and scope notes

Audience: technical collaborators preparing the AI subsection of a TENG formulation-sensing manuscript.

Decision supported: select a defensible primary AI claim, explain the secondary regression result, and identify the smallest experiment that could improve quantitative calibration.

The canonical report surface is the repository `README.md`, as requested for GitHub review. The README follows the technical-report sequence: conclusion, evidence, definitions, methodology, limitations, next experiment, and reproducibility. The separate `METHODS_AI.md` and `RESULTS_AI.md` files are reusable manuscript drafts rather than parallel reports.

The evidence package contains derived macroblock and formulation tables only. It reproduces the reported metrics and package-owned figures but does not retrain every candidate model from raw voltage traces. Archived figures are preserved for human review and are not treated as equal-strength evidence in the recommended manuscript story.

The five folds are the uncertainty units for the conservative no-augmentation RBF-SVM result. The displayed confusion matrix is pooled across the 136 held-out macroblocks, while the headline accuracy, balanced accuracy, and Macro-F1 are means across evaluation folds. These grains are related but should not be treated as numerically identical summaries.

Regression is evaluated at the held-out mixture-formulation level with the corresponding pure endpoints retained as calibration anchors. PGS and PGT are analyzed separately. The experiment-priority figure is a decision aid derived from the observed residuals and response curves, not an additional model-performance result.
