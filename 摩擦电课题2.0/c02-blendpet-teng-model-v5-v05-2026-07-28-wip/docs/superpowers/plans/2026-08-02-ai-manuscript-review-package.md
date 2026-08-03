# BlendPET-TENG AI Manuscript Review Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local, GitHub-ready review folder whose README explains the final AI manuscript story, embeds every usable figure, and reproduces core classification and regression outputs from compact derived data.

**Architecture:** A tested Python reproduction script will consume only packaged CSV files and regenerate package-owned metrics and figures with relative paths. A separate package builder will copy the reviewed derived tables and legacy figures into role-based folders, write a checksum manifest, and leave the folder local for user review.

**Tech Stack:** Python 3, pandas, NumPy, scikit-learn metrics, Matplotlib, Markdown, CSV, SHA-256.

## Global Constraints

- Do not include the complete raw waveform archive.
- Treat fifteen-formulation classification as the primary result and family-conditioned regression as secondary semiquantitative evidence.
- Embed every usable PNG in `README.md`; link PDF and SVG counterparts when available.
- Keep main, supplementary, diagnostic, and archived figures visibly separated.
- Do not use absolute machine paths inside distributable files.
- Do not clone, commit, push, release, or otherwise change GitHub before explicit user approval.

---

### Task 1: Define and test the portable analysis contract

**Files:**
- Create: `packaging/tests/test_reproduce_package.py`
- Create: `packaging/reproduce_package.py`

**Interfaces:**
- Consumes: packaged CSV files below `data/`.
- Produces: `compute_metrics(root: Path) -> dict`, `validate_package(root: Path) -> list[str]`, and publication-ready figures below `figures/main/` and `figures/diagnostic/`.

- [ ] Write a failing test that requires exact conservative RBF-SVM metrics, current endpoint-calibrated regression metrics, complete README image targets, and absence of absolute local paths.
- [ ] Run `python3 -m unittest discover -s packaging/tests -p 'test_*.py' -v` and verify failure because `packaging/reproduce_package.py` does not exist.
- [ ] Implement the minimum metric and validation functions needed by the test.
- [ ] Re-run the test and verify it passes.

### Task 2: Build the compact evidence package

**Files:**
- Create: `packaging/build_review_package.py`
- Create: `blendpet-teng-ai-manuscript-package-v1-review/data/**`
- Create: `blendpet-teng-ai-manuscript-package-v1-review/code/**`

**Interfaces:**
- Consumes: reviewed V5-v05 macroblock features, classification ledgers, regression ledgers, figure source data, and selected scripts.
- Produces: a self-contained derived-data package with portable relative paths.

- [ ] Extend the failing test with required compact data files and expected formula, macroblock, and regression row counts.
- [ ] Run the focused unit test and verify failure because the package has not been built.
- [ ] Implement the package builder with an explicit source-to-destination manifest and no wildcard copying of raw data.
- [ ] Build the package and re-run the focused test.

### Task 3: Generate the final AI figure set

**Files:**
- Create: `blendpet-teng-ai-manuscript-package-v1-review/figures/main/*.{png,pdf,svg}`
- Create: `blendpet-teng-ai-manuscript-package-v1-review/figures/diagnostic/*.{png,pdf,svg}`
- Populate: `blendpet-teng-ai-manuscript-package-v1-review/figures/supplementary/`
- Populate: `blendpet-teng-ai-manuscript-package-v1-review/figures/archive/`

**Interfaces:**
- Consumes: compact data files copied in Task 2.
- Produces: workflow, feature-response, model-comparison, confusion-matrix, feature-ablation, regression-diagnostic, and experiment-priority figures.

- [ ] Extend the test to require every package-owned figure in PNG, PDF, and SVG and require non-empty files.
- [ ] Run the test and verify it fails because figures are missing.
- [ ] Implement neutral, publication-ready Matplotlib renderers using a restrained blue/orange palette, direct labels, honest axes, and visible sample definitions.
- [ ] Generate the figures and verify the focused test passes.
- [ ] Inspect a contact sheet and revise any clipping, collisions, or misleading scales.

### Task 4: Write the manuscript narrative and visual gallery

**Files:**
- Create: `blendpet-teng-ai-manuscript-package-v1-review/README.md`
- Create: `blendpet-teng-ai-manuscript-package-v1-review/METHODS_AI.md`
- Create: `blendpet-teng-ai-manuscript-package-v1-review/RESULTS_AI.md`
- Create: `blendpet-teng-ai-manuscript-package-v1-review/REVIEW_CHECKLIST.md`

**Interfaces:**
- Consumes: verified metrics, figure roles, and diagnostic evidence.
- Produces: an answer-first Chinese README, reusable manuscript sections, and an upload gate checklist.

- [ ] Write the README around the primary classification claim, feature-based explanation, secondary regression evidence, and targeted follow-up experiment.
- [ ] Embed every usable PNG through relative links and attach each claim to a packaged CSV.
- [ ] Write concise Methods and Results drafts with language that matches the evidence boundary.
- [ ] Add a checklist covering numerical consistency, figure roles, relative paths, omitted raw waveforms, and explicit upload approval.

### Task 5: Create manifest and run final verification

**Files:**
- Create: `blendpet-teng-ai-manuscript-package-v1-review/data/manifests/package-manifest.csv`
- Create: `blendpet-teng-ai-manuscript-package-v1-review/data/manifests/reproduction-metrics.json`

**Interfaces:**
- Consumes: every distributable file except the generated manifest itself.
- Produces: SHA-256 checksums and a machine-readable metric receipt.

- [ ] Generate the manifest with relative path, role, byte size, and SHA-256 for every packaged file.
- [ ] Run `python3 -m unittest discover -s packaging/tests -p 'test_*.py' -v` and require zero failures.
- [ ] Run the packaged reproduction script from the package root and compare regenerated metrics to documented values.
- [ ] Search distributable text files for absolute local paths and unresolved placeholders.
- [ ] Inspect the README and all figure thumbnails, then hand off the local folder without any GitHub operation.
