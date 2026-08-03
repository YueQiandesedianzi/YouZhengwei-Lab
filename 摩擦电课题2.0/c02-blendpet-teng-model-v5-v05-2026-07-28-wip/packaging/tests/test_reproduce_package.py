from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd


SOURCE_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = SOURCE_ROOT.parent / "blendpet-teng-ai-manuscript-package-v1-review"
MODULE_PATH = SOURCE_ROOT / "packaging" / "reproduce_package.py"


def load_reproduction_module():
    if not MODULE_PATH.exists():
        raise AssertionError(f"Missing reproduction module: {MODULE_PATH}")
    spec = importlib.util.spec_from_file_location("reproduce_package", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ReproductionContractTest(unittest.TestCase):
    def test_long_form_confusion_table_is_pivoted_in_label_order(self):
        module = load_reproduction_module()
        long_form = pd.DataFrame(
            [
                {"true_formula": "A", "predicted_formula": "A", "count": 3},
                {"true_formula": "A", "predicted_formula": "B", "count": 1},
                {"true_formula": "B", "predicted_formula": "A", "count": 0},
                {"true_formula": "B", "predicted_formula": "B", "count": 4},
            ]
        )

        labels, values = module.pivot_confusion(long_form)

        self.assertEqual(labels, ["A", "B"])
        self.assertEqual(values.tolist(), [[3, 1], [0, 4]])

    def test_core_metrics_match_hand_checked_results(self):
        module = load_reproduction_module()
        metrics = module.compute_metrics(PACKAGE_ROOT)

        self.assertAlmostEqual(metrics["classification"]["macro_f1"], 0.9659682539682539, places=12)
        self.assertAlmostEqual(metrics["classification"]["balanced_accuracy"], 0.9711111111111110, places=12)
        self.assertAlmostEqual(metrics["classification"]["accuracy"], 0.9692307692307691, places=12)
        self.assertEqual(metrics["classification"]["formula_count"], 15)
        self.assertEqual(metrics["classification"]["macroblock_count"], 136)

        self.assertAlmostEqual(metrics["regression"]["mae_pp"], 8.07119326209488, places=12)
        self.assertAlmostEqual(metrics["regression"]["r2"], 0.8180976391071065, places=12)
        self.assertEqual(metrics["regression"]["mixture_formula_count"], 12)

    def test_package_has_required_portable_artifacts(self):
        module = load_reproduction_module()
        expected_triplets = [
            "01-ai-workflow",
            "02-feature-response",
            "03-model-comparison",
            "04-rbf-svm-classification",
            "05-feature-ablation",
            "06-family-regression",
            "07-regression-experiment-priority",
        ]
        for stem in expected_triplets:
            folder = "main" if int(stem[:2]) <= 5 else "diagnostic"
            for suffix in ("png", "pdf", "svg"):
                path = PACKAGE_ROOT / "figures" / folder / f"{stem}.{suffix}"
                self.assertTrue(path.exists(), f"Missing figure: {path}")
                self.assertGreater(path.stat().st_size, 1000, f"Empty or truncated figure: {path}")

        errors = module.validate_package(PACKAGE_ROOT)
        self.assertEqual(errors, [], "\n".join(errors))

        receipt = PACKAGE_ROOT / "data" / "manifests" / "reproduction-metrics.json"
        self.assertTrue(receipt.exists())
        recorded = json.loads(receipt.read_text(encoding="utf-8"))
        self.assertEqual(recorded, module.compute_metrics(PACKAGE_ROOT))


if __name__ == "__main__":
    unittest.main()
