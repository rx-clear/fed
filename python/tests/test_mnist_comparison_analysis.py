"""Check that comparison statistics use seeds as the replication unit."""

import csv
import importlib.util
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


SCRIPT = Path(__file__).parents[1] / "examples/federate/simulation/sp_fedavg_mnist_lr_example/analyze_comparison.py"
SPEC = importlib.util.spec_from_file_location("mnist_comparison_analysis_under_test", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class MNISTComparisonAnalysisTest(unittest.TestCase):
    def _write_rows(self, path, missing_seed=False):
        fields = ["method", "seed", "round", "test_acc", "test_loss",
                  "matrix_id", "backend", "backend_secure"]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for method, backend in (("FedAvg", "plaintext"),
                                    ("FedRep+FixedPointReference", "chunked-reference")):
                for seed in (0, 1):
                    if missing_seed and method != "FedAvg" and seed == 1:
                        continue
                    for round_idx in (0, 1):
                        writer.writerow({
                            "method": method, "seed": seed, "round": round_idx,
                            "test_acc": 0.50 + 0.10 * seed + 0.05 * round_idx
                            + (0.02 if method != "FedAvg" else 0),
                            "test_loss": 1.0 - 0.10 * seed - 0.05 * round_idx,
                            "matrix_id": "matrix", "backend": backend,
                            "backend_secure": False,
                        })

    def test_seed_level_paired_differences_and_plots(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "metrics.csv"
            self._write_rows(input_path)
            runs = MODULE._read_runs(input_path)
            round_rows, seed_rows, paired, paired_summary = MODULE.summarize(runs)
            self.assertEqual(len(runs), 4)
            self.assertEqual(len(seed_rows), 4)
            self.assertEqual(len(paired), 2)
            self.assertEqual(paired_summary[0]["n_paired_seeds"], 2)
            self.assertAlmostEqual(paired_summary[0]["mean_delta_final_test_acc"], 0.02)
            self.assertAlmostEqual(
                next(row for row in round_rows if row["method"] == "FedAvg" and row["round"] == 1)["test_acc_sd"],
                0.1 / (2 ** 0.5),
            )
            self.assertEqual(MODULE.main(["--input", str(input_path), "--output-dir", str(root / "analysis")]), 0)
            for name in ("round_summary.csv", "seed_summary.csv", "paired_differences.csv", "paired_summary.csv"):
                self.assertTrue((root / "analysis" / name).exists())
            self.assertTrue((root / "analysis/plots/test_acc.png").exists())
            self.assertTrue((root / "analysis/plots/test_loss.png").exists())

    def test_unmatched_seeds_are_rejected(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.csv"
            self._write_rows(path, missing_seed=True)
            with self.assertRaisesRegex(ValueError, "unmatched seeds"):
                MODULE._read_runs(path)


if __name__ == "__main__":
    unittest.main()
