import importlib.util
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from fedml.simulation.sp.fedavg.coded_ip_fedmf import (
    CodedIPFedMFAggregator,
    MDSCode,
    dequantize,
    normalize_and_quantize,
)
from fedml.simulation.sp.fedavg.quantization import quantize as fedml_quantize


class CodedIPFedMFTest(unittest.TestCase):
    def test_mds_recovers_from_any_threshold_shares(self):
        code = MDSCode(n=7, k=4)
        source = torch.arange(20, dtype=torch.float64).reshape(4, 5)
        encoded = code.encode(source)
        for online in ((0, 1, 2, 3), (1, 3, 5, 6), (0, 2, 4, 6)):
            self.assertTrue(torch.allclose(code.decode(encoded[list(online)], online), source))
            self.assertTrue(torch.allclose(code.decode_sum(encoded[list(online)], online), source.sum(0)))

    def test_coded_aggregate_survives_dropouts(self):
        updates = {i: torch.tensor([float(i), -float(i + 1)]) for i in range(6)}
        aggregator = CodedIPFedMFAggregator(6, 3, seed=3)
        coded = aggregator.encode_client_updates(updates)
        result = aggregator.aggregate(coded, [0, 2, 4])
        expected = torch.stack(list(updates.values())).sum(0)
        self.assertTrue(torch.allclose(result.aggregate, expected, atol=1e-6))

    def test_collect_uploaded_shares_models_transport_dropout(self):
        updates = {i: torch.tensor([float(i), 1.0]) for i in range(5)}
        aggregator = CodedIPFedMFAggregator(5, 3, seed=4)
        coded = aggregator.encode_client_updates(updates)
        received = aggregator.collect_uploaded_shares(coded, [4, 1, 3])
        result = aggregator.aggregate(received, [4, 1, 3])
        expected = sum(updates.values(), torch.zeros(2))
        self.assertTrue(torch.allclose(result.aggregate, expected, atol=1e-6))
        with self.assertRaises(ValueError):
            aggregator.aggregate(received, [0, 1, 3])

    def test_insufficient_online_clients_fail(self):
        aggregator = CodedIPFedMFAggregator(5, 3)
        coded = aggregator.encode_shards(torch.ones(3, 2))
        with self.assertRaises(ValueError):
            aggregator.aggregate(coded, [0, 2])

    def test_quantization_and_cosine_filter(self):
        value = torch.tensor([0.125, -0.25, 0.5])
        restored = dequantize(normalize_and_quantize(value, scale=10000), scale=10000)
        self.assertTrue(torch.allclose(value.to(torch.float64), restored, atol=1e-4))
        updates = {0: value, 1: -value, 2: torch.zeros_like(value)}
        scores = CodedIPFedMFAggregator.score_updates(updates, value)
        self.assertAlmostEqual(scores[0], 1.0, places=6)
        self.assertAlmostEqual(scores[1], -1.0, places=6)
        self.assertEqual(CodedIPFedMFAggregator.valid_clients(scores, 0.0), (0, 2))

    def test_update_mapping_requires_all_client_ids(self):
        aggregator = CodedIPFedMFAggregator(3, 2)
        with self.assertRaises(ValueError):
            aggregator.encode_client_updates({0: torch.ones(2), 2: torch.ones(2)})

    def test_update_shape_mismatch_is_value_error(self):
        aggregator = CodedIPFedMFAggregator(2, 1)
        with self.assertRaises(ValueError):
            aggregator.encode_client_updates([torch.ones(2), torch.ones(3)])

    def test_float32_mds_recovery_weights(self):
        code = MDSCode(n=6, k=4, dtype=torch.float32)
        weights = code.recovery_weights([0, 1, 2, 3])
        self.assertEqual(weights.dtype, torch.float32)

    def test_weighted_aggregation(self):
        updates = {i: torch.tensor([float(i + 1), 1.0]) for i in range(4)}
        weights = [0.5, 1.0, 1.5, 2.0]
        aggregator = CodedIPFedMFAggregator(4, 2, seed=5)
        result = aggregator.aggregate_weighted_updates(updates, weights, [1, 3])
        expected = sum(
            (updates[i] * weights[i] for i in range(4)), torch.zeros(2)
        )
        self.assertTrue(torch.allclose(result.aggregate, expected, atol=1e-6))

    def test_reference_byzantine_filter(self):
        reference = torch.tensor([1.0, 0.0])
        updates = {
            0: torch.tensor([1.0, 0.0]),
            1: torch.tensor([0.9, 0.1]),
            2: torch.tensor([-1.0, 0.0]),
            3: torch.tensor([1.0, 0.0]),
        }
        aggregator = CodedIPFedMFAggregator(4, 2, seed=6)
        result = aggregator.aggregate_with_byzantine_filter(
            updates, [0, 1], reference, 0.5
        )
        self.assertEqual(result.accepted_client_ids, (0, 1, 3))
        expected = updates[0] + updates[1] + updates[3]
        self.assertTrue(torch.allclose(result.aggregate, expected, atol=1e-6))

    def test_fedml_quantization_rejects_non_finite_and_overflow(self):
        with self.assertRaises(ValueError):
            fedml_quantize(torch.tensor([float("nan")]))
        with self.assertRaises(OverflowError):
            fedml_quantize(torch.tensor([1.0e20], dtype=torch.float64), precision=62)

    def test_movielens_dat_keeps_first_record(self):
        module_path = (
            Path(__file__).parents[1]
            / "examples" / "federate" / "simulation"
            / "sp_coded_ip_fedmf_example" / "movielens.py"
        )
        spec = importlib.util.spec_from_file_location("coded_ip_fedmf_movielens", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with TemporaryDirectory() as directory:
            path = Path(directory) / "ratings.dat"
            path.write_text("1::10::5::1\n2::11::4::2\n", encoding="utf-8")
            ratings = module.load_ratings(str(path), client_num=2)
        self.assertEqual(sum(len(ids) for ids, _ in ratings), 2)

    def test_movielens_csv_supports_header_and_headerless_files(self):
        module_path = (
            Path(__file__).parents[1]
            / "examples" / "federate" / "simulation"
            / "sp_coded_ip_fedmf_example" / "movielens.py"
        )
        spec = importlib.util.spec_from_file_location("coded_ip_fedmf_movielens_csv", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with TemporaryDirectory() as directory:
            header_path = Path(directory) / "ratings_header.csv"
            header_path.write_text(
                "userId,movieId,rating,timestamp\n1,10,5,1\n2,11,4,2\n",
                encoding="utf-8",
            )
            raw_path = Path(directory) / "ratings_raw.csv"
            raw_path.write_text("1,10,5\n2,11,4\n", encoding="utf-8")
            self.assertEqual(
                sum(len(ids) for ids, _ in module.load_ratings(str(header_path), 2)), 2
            )
            self.assertEqual(
                sum(len(ids) for ids, _ in module.load_ratings(str(raw_path), 2)), 2
            )


if __name__ == "__main__":
    unittest.main()
