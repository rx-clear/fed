import sys
import unittest
from pathlib import Path

EXAMPLE_DIR = Path(__file__).parents[1] / "examples" / "federate" / "simulation" / "sp_coded_ip_fedmf_example"
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

from baselines import run_baseline_comparison  # noqa: E402
from movielens import make_synthetic_ratings  # noqa: E402
from protocol_validation import run_diagnostics  # noqa: E402


class CodedIPFedMFBaselineTest(unittest.TestCase):
    def test_coded_matches_centralized_without_dropout(self):
        ratings = make_synthetic_ratings(4, 8, 2, 4, seed=33)
        rows = run_baseline_comparison(
            ratings, rank=2, rounds=2, dropout_rate=0.0,
            ldp_noise_multiplier=0.0, seed=33,
        )
        self.assertLessEqual(
            max(row["coded_vs_central_max_error"] for row in rows), 1e-8
        )
        self.assertTrue(
            all(
                abs(row["coded_rmse"] - row["centralized_rmse"]) <= 1e-8
                for row in rows
            )
        )

    def test_coded_matches_centralized_with_dropout(self):
        ratings = make_synthetic_ratings(5, 9, 2, 4, seed=34)
        rows = run_baseline_comparison(
            ratings, rank=2, rounds=2, dropout_rate=0.4,
            ldp_noise_multiplier=0.0, seed=34,
        )
        self.assertLessEqual(
            max(row["coded_vs_central_max_error"] for row in rows), 1e-8
        )
        self.assertEqual(rows[-1]["online_clients"], 3)

    def test_invalid_experiment_parameters_fail(self):
        ratings = make_synthetic_ratings(4, 8, 2, 4, seed=35)
        with self.assertRaises(ValueError):
            run_baseline_comparison(ratings, dropout_rate=1.0)
        with self.assertRaises(ValueError):
            run_baseline_comparison(ratings, rounds=0)

    def test_protocol_diagnostics_expose_model_limits(self):
        diagnostics = run_diagnostics(4, 2, 4)
        self.assertFalse(diagnostics["upload_dropout"]["pre_upload_dropout_modelled"])
        self.assertTrue(diagnostics["upload_dropout"]["share_upload_dropout_modelled"])
        self.assertLessEqual(diagnostics["upload_dropout"]["share_upload_max_error"], 1e-6)
        self.assertTrue(diagnostics["aligned_poisoning"]["passes_threshold_0_5"])
        self.assertTrue(diagnostics["conditioning"]["maximum_subset_condition"] > 0)


if __name__ == "__main__":
    unittest.main()
