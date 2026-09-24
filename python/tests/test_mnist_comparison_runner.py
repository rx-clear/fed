"""Focused checks for resume, validation, and preservation of MNIST runs."""

import csv
import importlib.util
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml


SCRIPT = Path(__file__).parents[1] / "examples/federate/simulation/sp_fedavg_mnist_lr_example/run_comparison.py"
SPEC = importlib.util.spec_from_file_location("mnist_comparison_runner_under_test", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def fake_launch(root, config_path):
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    common = config["common_args"]
    train = config["train_args"]
    backend = "chunked-reference" if train["enable_dmcfe"] else str(train.get("privacy_aggregation") or "plaintext")
    path = Path(train["experiment_result_file"])
    fields = ["run_id", "method", "seed", "round", "test_acc", "test_loss",
              "config_hash", "matrix_id", "backend", "backend_secure", "coordinates",
              "chunks", "clipped_fraction", "scale", "clip_bound", "encode_seconds", "aggregate_seconds"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for round_idx in range(train["comm_round"]):
            writer.writerow({
                "run_id": common["experiment_run_id"], "method": common["experiment_method"],
                "seed": common["random_seed"], "round": round_idx,
                "test_acc": 0.5 + round_idx * 0.01, "test_loss": 1.0,
                "config_hash": common["experiment_config_hash"],
                "matrix_id": common["experiment_matrix_id"], "backend": backend,
                "backend_secure": False, "coordinates": 4 if train["enable_dmcfe"] else "",
                "chunks": 1 if train["enable_dmcfe"] else "",
                "clipped_fraction": 0 if train["enable_dmcfe"] else "",
                "scale": 16384 if train["enable_dmcfe"] else "",
                "clip_bound": 8 if train["enable_dmcfe"] else "",
                "encode_seconds": 0.01 if train["enable_dmcfe"] else "",
                "aggregate_seconds": 0.02 if train["enable_dmcfe"] else "",
            })
    return 0


class MNISTComparisonRunnerTest(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.result_dir = self.root / "comparison"
        self.base = {"common_args": {}, "train_args": {}, "validation_args": {"frequency_of_the_test": 50}}

    def run_matrix(self, methods, seeds, **kwargs):
        MODULE.run_matrix(self.root, self.base, methods, seeds, 2, self.result_dir,
                          launch=kwargs.pop("launch", fake_launch), **kwargs)

    def test_config_resume_and_deterministic_merge(self):
        self.run_matrix(["FedAvg", "FedRep", "Ours"], [0, 1])
        path = self.result_dir / "comparison_metrics.csv"
        initial = path.read_bytes()
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 12)
        self.assertEqual({row["method"] for row in rows},
                         {"FedAvg", "FedRep", "FedRep+FixedPointReference"})
        self.assertEqual(len({row["matrix_id"] for row in rows}), 1)
        snapshot = yaml.safe_load((self.result_dir / "configs/fedrep_reference_seed0_2r.yaml").read_text(encoding="utf-8"))
        self.assertEqual(snapshot["validation_args"]["frequency_of_the_test"], 1)
        self.assertEqual(snapshot["train_args"]["dmcfe_backend"], "chunked_reference")
        self.assertEqual(snapshot["common_args"]["mlops_run_name"], "fedrep_reference_seed0_2r")
        self.assertEqual(snapshot["tracking_args"]["run_name"], "fedrep_reference_seed0_2r")

        def unexpected_launch(root, config_path):
            self.fail("resume should skip completed runs")

        self.run_matrix(["FedAvg", "FedRep", "Ours"], [0, 1], resume=True,
                        launch=unexpected_launch)
        self.assertEqual(path.read_bytes(), initial)

    def test_failed_force_preserves_completed_output(self):
        self.run_matrix(["FedAvg"], [0])
        output = self.result_dir / "fedavg_seed0_2r.csv"
        initial = output.read_bytes()

        def failed_launch(root, config_path):
            fake_launch(root, config_path)
            return 7

        with self.assertRaises(RuntimeError):
            self.run_matrix(["FedAvg"], [0], force=True, launch=failed_launch)
        self.assertEqual(output.read_bytes(), initial)
        with (self.result_dir / "comparison_manifest.csv").open(newline="", encoding="utf-8") as handle:
            manifest = list(csv.DictReader(handle))
        self.assertEqual(manifest[0]["status"], "failed")
        self.assertEqual(manifest[0]["exit_code"], "7")
        self.assertTrue(Path(manifest[0]["partial_file"]).exists())
        self.run_matrix(["FedAvg"], [0], resume=True,
                        launch=lambda *_: self.fail("valid old output should be adopted"))
        with (self.result_dir / "comparison_manifest.csv").open(newline="", encoding="utf-8") as handle:
            self.assertEqual(list(csv.DictReader(handle))[0]["status"], "completed")

    def test_invalid_rows_never_replace_result(self):
        def invalid_launch(root, config_path):
            fake_launch(root, config_path)
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            path = Path(config["train_args"]["experiment_result_file"])
            fields, rows = MODULE._read_csv(path)
            rows[1]["round"] = "0"
            MODULE._atomic_csv(path, fields, rows)
            return 0

        with self.assertRaises(ValueError):
            self.run_matrix(["FedAvg"], [0], launch=invalid_launch)
        self.assertFalse((self.result_dir / "fedavg_seed0_2r.csv").exists())

    def test_legacy_merged_csv_is_preserved(self):
        self.result_dir.mkdir()
        old = self.result_dir / "comparison_metrics.csv"
        old.write_text("legacy metrics\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "no resumable manifest"):
            self.run_matrix(["FedAvg"], [0], resume=True)
        self.assertEqual(old.read_text(encoding="utf-8"), "legacy metrics\n")

    def test_changed_base_config_cannot_mix_with_completed_matrix(self):
        self.run_matrix(["FedAvg"], [0])
        self.base["train_args"]["learning_rate"] = 0.1
        with self.assertRaisesRegex(ValueError, "another experiment matrix"):
            self.run_matrix(["FedRep"], [0], resume=True)

    def test_formal_configs_match_mnist_protocol(self):
        base = yaml.safe_load((SCRIPT.parent / "fedml_config.yaml").read_text(encoding="utf-8"))
        for method in ("FedAvg", "FedRep", "Ours"):
            for seed in (0, 1, 2):
                run_id = f"{MODULE.METHODS[method]['slug']}_seed{seed}_50r"
                config = MODULE._config(base, method, seed, 50,
                                        self.result_dir / f"{run_id}.csv", run_id, "matrix")
                self.assertEqual(config["data_args"]["dataset"], "mnist")
                self.assertEqual(config["model_args"]["model"], "cnn")
                self.assertEqual(config["common_args"]["random_seed"], seed)
                self.assertEqual(config["train_args"]["comm_round"], 50)
                self.assertEqual(config["train_args"]["client_num_in_total"], 1000)
                self.assertEqual(config["train_args"]["client_num_per_round"], 3)
                self.assertEqual(config["validation_args"]["frequency_of_the_test"], 1)
                self.assertEqual(config["train_args"]["enable_dmcfe"], method == "Ours")
                self.assertEqual(config["common_args"]["mlops_run_name"], run_id)


if __name__ == "__main__":
    unittest.main()
