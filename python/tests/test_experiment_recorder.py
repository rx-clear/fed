import csv
import importlib.util
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


RECORDER_PATH = (
    Path(__file__).parents[1]
    / "fedml"
    / "simulation"
    / "sp"
    / "fedavg"
    / "experiment_recorder.py"
)
SPEC = importlib.util.spec_from_file_location("experiment_recorder_under_test", RECORDER_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
ExperimentRecorder = MODULE.ExperimentRecorder


class ExperimentRecorderTest(unittest.TestCase):
    def test_new_file_records_one_row_with_protocol_metrics(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.csv"
            recorder = ExperimentRecorder(str(path))
            recorder.append(
                round_idx=2,
                train_acc=0.8,
                train_loss=0.4,
                test_acc=0.75,
                test_loss=0.5,
                dropout_rate=0.25,
                online_clients=3,
                online_client_ids="1,2,4",
                recovery_threshold=3,
                coded_communication_rounds=1,
                recovery_error=1e-9,
                dmcfe_fedrep_max_error=2e-5,
                dmcfe_fedrep_mean_error=1e-5,
                run_id="fedrep_reference_seed0_50r",
                method="FedRep+FixedPointReference",
                seed=0,
                config_hash="abc123",
                backend="chunked-reference",
                backend_secure=False,
                participation_rate=0.003,
                protocol_dropout_rate=0.0,
                clipped_fraction=0.125,
                encode_seconds=0.01,
            )
            with path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["online_clients"], "3")
            self.assertEqual(rows[0]["recovery_threshold"], "3")
            self.assertEqual(rows[0]["online_client_ids"], "1,2,4")
            self.assertEqual(rows[0]["dmcfe_fedavg_mean_error"], "")
            self.assertEqual(rows[0]["dmcfe_fedrep_max_error"], "2e-05")
            self.assertEqual(rows[0]["backend_secure"], "False")
            self.assertEqual(rows[0]["participation_rate"], "0.003")
            self.assertEqual(rows[0]["clipped_fraction"], "0.125")

    def test_legacy_header_is_preserved(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.csv"
            path.write_text(
                "round,train_acc,train_loss,test_acc,test_loss,"
                "dmcfe_fedavg_max_error,dmcfe_fedavg_mean_error\n",
                encoding="utf-8",
            )
            recorder = ExperimentRecorder(str(path))
            recorder.append(0, None, None, 0.2, 1.0, recovery_error=0.0)
            with path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual(set(rows[0]), set(recorder.fieldnames))
            self.assertEqual(rows[0]["train_acc"], "")

    def test_non_finite_metric_is_rejected(self):
        with TemporaryDirectory() as directory:
            recorder = ExperimentRecorder(str(Path(directory) / "metrics.csv"))
            with self.assertRaises(ValueError):
                recorder.append(0, float("nan"), 0.0, 0.0, 0.0)


if __name__ == "__main__":
    unittest.main()
