import importlib.util
import sys
import unittest
from pathlib import Path

import torch


ADAPTER_PATH = (
    Path(__file__).parents[1]
    / "fedml"
    / "simulation"
    / "sp"
    / "fedavg"
    / "dmcfe_ip_adapter.py"
)
SPEC = importlib.util.spec_from_file_location("dmcfe_ip_adapter_under_test", ADAPTER_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
DMCFEIPAggregator = MODULE.DMCFEIPAggregator


class DMCFEIPAdapterTest(unittest.TestCase):
    def test_weighted_delta_aggregation_and_integer_state(self):
        global_params = {
            "weight": torch.tensor([1.0, 2.0]),
            "counter": torch.tensor(3, dtype=torch.int64),
        }
        local_models = [
            (2, {"weight": torch.tensor([2.0, 2.0]), "counter": torch.tensor(3)}),
            (1, {"weight": torch.tensor([1.0, 5.0]), "counter": torch.tensor(3)}),
        ]
        result = DMCFEIPAggregator(scale=10000).aggregate(
            global_params, local_models, round_idx=0
        )
        self.assertTrue(
            torch.allclose(result["weight"], torch.tensor([1.6667, 3.0]), atol=2e-4)
        )
        self.assertEqual(result["counter"].item(), 3)

    def test_invalid_global_update_is_rejected(self):
        aggregator = DMCFEIPAggregator()
        with self.assertRaises(ValueError):
            aggregator.aggregate(
                {"weight": torch.tensor([float("nan")])},
                [(1, {"weight": torch.tensor([1.0])})],
                round_idx=0,
            )

    def test_integer_state_mismatch_is_rejected(self):
        aggregator = DMCFEIPAggregator()
        with self.assertRaises(ValueError):
            aggregator.aggregate(
                {"counter": torch.tensor(1, dtype=torch.int64)},
                [
                    (1, {"counter": torch.tensor(1, dtype=torch.int64)}),
                    (1, {"counter": torch.tensor(2, dtype=torch.int64)}),
                ],
                round_idx=0,
            )

    def test_fractional_sample_count_is_rejected(self):
        with self.assertRaises(ValueError):
            DMCFEIPAggregator().aggregate(
                {"weight": torch.tensor([0.0])},
                [(1.5, {"weight": torch.tensor([1.0])})],
                round_idx=0,
            )


if __name__ == "__main__":
    unittest.main()
