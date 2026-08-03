from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "build_review_package.py"


def load_builder_module():
    spec = importlib.util.spec_from_file_location("build_review_package", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PackageBuilderTest(unittest.TestCase):
    def test_source_summary_creates_manifest_parent_directory(self):
        module = load_builder_module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "data/features").mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "macroblock_id": "m1",
                        "canonical_formula": "100PGD",
                        "system_label": "PGD",
                        "z": 0.0,
                        "cycle_count": 12,
                    }
                ]
            ).to_csv(root / "data/features/formula-macroblocks.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "canonical_formula": "100PGD",
                        "sample_count": 1,
                        "source_file_count": 1,
                        "cycle_count_retained": 12,
                    }
                ]
            ).to_csv(root / "data/features/formula-repeatability.csv", index=False)

            module.build_source_summary(root)

            self.assertTrue((root / "data/manifests/analysis-unit-summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
