import unittest

import torch

from fedml.simulation.sp.fedavg.optimized_dmcfe_ip import (
    OptimizedDMCFEIPAggregator,
    ToyMCFEBackend,
)


class OptimizedDMCFEIPTest(unittest.TestCase):
    def test_chunked_weighted_sum_matches_plain_average(self):
        local = [
            (2, {"weight": torch.tensor([0.125, -0.20])}),
            (3, {"weight": torch.tensor([0.50, 0.10])}),
        ]
        result = OptimizedDMCFEIPAggregator(
            scale=1 << 14, chunk_size=1
        ).aggregate_deltas(local, round_idx=4)
        expected = torch.tensor([0.35, -0.02])
        self.assertTrue(torch.allclose(result.deltas["weight"], expected, atol=1e-4))
        self.assertEqual(result.diagnostics.chunks, 2)
        self.assertEqual(result.diagnostics.coordinates, 2)
        self.assertLessEqual(result.diagnostics.max_abs_error, 1.0 / (1 << 14))
        self.assertFalse(result.diagnostics.backend_secure)
        self.assertGreaterEqual(result.diagnostics.encode_seconds, 0.0)
        self.assertGreaterEqual(result.diagnostics.aggregate_seconds, 0.0)
        self.assertGreaterEqual(result.diagnostics.total_aggregation_seconds,
                                result.diagnostics.aggregate_seconds)

    def test_full_state_and_integer_buffer_validation(self):
        local = [
            (1, {"weight": torch.tensor([1.0]), "counter": torch.tensor(2)}),
            (1, {"weight": torch.tensor([3.0]), "counter": torch.tensor(2)}),
        ]
        result = OptimizedDMCFEIPAggregator().aggregate_deltas(local)
        self.assertEqual(result.deltas["counter"].item(), 2)
        self.assertAlmostEqual(result.deltas["weight"].item(), 2.0, places=4)

        with self.assertRaises(ValueError):
            OptimizedDMCFEIPAggregator().aggregate_deltas(
                [(1, {"weight": torch.tensor([1.0])}),
                 (1, {"other": torch.tensor([1.0])})]
            )

    def test_clipping_is_reported(self):
        result = OptimizedDMCFEIPAggregator(
            clip_bound=0.5, scale=1000
        ).aggregate_vectors(
            [torch.tensor([1.0, -0.25])], [1]
        )
        self.assertAlmostEqual(result[0][0].item(), 0.5, places=3)
        self.assertEqual(result[1].clipped_fraction, 0.5)

    def test_mean_error_weights_parameter_coordinates(self):
        result = OptimizedDMCFEIPAggregator(scale=2).aggregate_deltas([
            (1, {"short": torch.tensor([0.25], dtype=torch.float64),
                 "long": torch.zeros(3, dtype=torch.float64)})
        ])
        self.assertAlmostEqual(result.diagnostics.mean_abs_error, 0.25 / 4)

    def test_toy_mcfe_backend_matches_reference_on_small_vector(self):
        result = OptimizedDMCFEIPAggregator(
            scale=32,
            chunk_size=2,
            backend=ToyMCFEBackend(modulus_bits=96),
        ).aggregate_vectors(
            [torch.tensor([0.25, -0.5]), torch.tensor([0.75, 0.25])],
            [2, 3],
            round_idx=2,
        )
        expected = torch.tensor([0.55, -0.05], dtype=torch.float64)
        self.assertTrue(torch.allclose(result[0], expected, atol=0.05))
        self.assertFalse(result[1].backend == "chunked-reference")


if __name__ == "__main__":
    unittest.main()
