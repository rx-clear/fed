import sys
import unittest
from pathlib import Path


EXAMPLE_DIR = (
    Path(__file__).parents[1]
    / "examples"
    / "federate"
    / "simulation"
    / "sp_coded_ip_fedmf_example"
)
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

from dmcfe_pfedmf import run_experiment  # noqa: E402
from movielens import make_synthetic_ratings  # noqa: E402


class DMCFEPFedMFTest(unittest.TestCase):
    def test_personalized_recommendation_smoke(self):
        ratings = make_synthetic_ratings(
            client_num=4,
            item_num=10,
            rank=3,
            ratings_per_client=5,
            seed=21,
        )
        rows = run_experiment(
            ratings,
            rank=3,
            rounds=2,
            dropout_rate=0.25,
            top_k=3,
            seed=21,
        )
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(0.0 <= row["recall_at_k"] <= 1.0 for row in rows))
        self.assertTrue(all(0.0 <= row["ndcg_at_k"] <= 1.0 for row in rows))
        self.assertTrue(all(row["backend_secure"] is False for row in rows))
        self.assertTrue(all(row["online_clients"] == 3 for row in rows))


if __name__ == "__main__":
    unittest.main()
