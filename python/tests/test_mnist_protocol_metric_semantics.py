"""Keep numerical reference aggregation separate from recovery telemetry."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from fedml.simulation.sp.fedavg.fedavg_api import FedAvgAPI


def make_api(*, protocol_active):
    api = object.__new__(FedAvgAPI)
    api.args = SimpleNamespace(
        enable_dmcfe=True,
        privacy_aggregation=None,
        experiment_method="FedAvg+FixedPointReference",
        experiment_run_id="reference_seed0_1r",
        experiment_config_hash="config",
        experiment_matrix_id="matrix",
        random_seed=0,
    )
    api.experiment_recorder = Mock()
    api.current_dmcfe_max_error = 1e-5
    api.current_dmcfe_mean_error = 2e-6
    api.current_online_clients = 3
    api.current_online_client_ids = "1,2,3"
    api.current_participation_rate = 0.003
    api.current_protocol_recovery_active = protocol_active
    api.current_recovery_threshold = 2 if protocol_active else None
    api.current_dropout_rate = 1 / 3 if protocol_active else None
    api.current_dmcfe_weight_signatures_verified = 3
    api.current_aggregation_diagnostics = None if protocol_active else SimpleNamespace(
        backend="chunked-reference", backend_secure=False, coordinates=4,
        chunks=1, chunk_size=8, scale=16384, clip_bound=8.0,
        clipped_fraction=0.0, encode_seconds=0.01, aggregate_seconds=0.02,
        total_aggregation_seconds=0.04,
    )
    return api


class MNISTProtocolMetricSemanticsTest(unittest.TestCase):
    def test_optimized_reference_leaves_recovery_metrics_blank(self):
        api = make_api(protocol_active=False)
        api._record_metrics(0, test_acc=0.5, test_loss=1.0)
        values = api.experiment_recorder.append.call_args.kwargs
        self.assertEqual(values["backend"], "chunked-reference")
        self.assertIsNone(values["recovery_error"])
        self.assertIsNone(values["recovery_threshold"])
        self.assertIsNone(values["coded_communication_rounds"])
        self.assertIsNone(values["protocol_dropout_rate"])
        self.assertEqual(values["participation_rate"], 0.003)
        self.assertEqual(values["dmcfe_fedavg_max_error"], 1e-5)

    def test_smoke_recovery_reports_protocol_metrics(self):
        api = make_api(protocol_active=True)
        api._record_metrics(0, test_acc=0.5, test_loss=1.0)
        values = api.experiment_recorder.append.call_args.kwargs
        self.assertEqual(values["backend"], "smoke-reference")
        self.assertEqual(values["recovery_error"], 1e-5)
        self.assertEqual(values["recovery_threshold"], 2)
        self.assertEqual(values["coded_communication_rounds"], 1)
        self.assertAlmostEqual(values["protocol_dropout_rate"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
