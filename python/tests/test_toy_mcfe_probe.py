"""Correctness and accounting checks for the bounded toy MCFE probe."""

import importlib.util
import statistics
import sys
import unittest
from pathlib import Path

from fedml.simulation.sp.fedavg.dmcfe.mcfe import MCFE


SCRIPT = Path(__file__).parents[1] / "examples/federate/simulation/sp_fedavg_mnist_lr_example/benchmark_toy_mcfe.py"
SPEC = importlib.util.spec_from_file_location("toy_mcfe_probe_under_test", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ToyMCFEProbeTest(unittest.TestCase):
    def test_signed_integer_payload_roundtrips_at_boundaries(self):
        for value in (0, 127, 128, -128, -129, 1 << 80, -(1 << 80)):
            encoded = MODULE.signed_integer_bytes(value)
            self.assertEqual(int.from_bytes(encoded, "big", signed=True), value)
            if len(encoded) > 1:
                with self.assertRaises(OverflowError):
                    value.to_bytes(len(encoded) - 1, "big", signed=True)

    def test_trial_recovers_weighted_sum_and_counts_ciphertext_payload(self):
        mcfe = MCFE(modulus_bits=96)
        keys = [mcfe.keygen(100), mcfe.keygen(100)]
        result = MODULE.measure_trial(
            mcfe, keys, [2, 3], [[1, -2], [4, 5]], label="probe-test:one",
        )
        width = (mcfe.N2.bit_length() + 7) // 8
        self.assertEqual(result["max_abs_error"], 0)
        self.assertEqual(result["ciphertext_payload_bytes"], 4 * width)
        self.assertGreaterEqual(result["derived_key_payload_bytes"], 1)
        self.assertGreaterEqual(result["round_seconds"], result["decryption_seconds"])

    def test_summary_uses_seed_medians_instead_of_all_repeats(self):
        trials = []
        for seed, values in ((0, (1.0, 2.0, 100.0)), (1, (3.0, 4.0, 100.0))):
            for trial, seconds in enumerate(values):
                row = {metric: 0.0 for metric in MODULE.TIMING_FIELDS}
                row.update({metric: 10 for metric in MODULE.PAYLOAD_FIELDS})
                row.update(seed=seed, dimension=2, trial=trial, round_seconds=seconds)
                trials.append(row)
        setups = [
            {"seed": seed, "dimension": 2, "setup_seconds": 0.1, "keygen_seconds": 0.01}
            for seed in (0, 1)
        ]
        rows = MODULE.summarize(trials, setups, clients=2, modulus_bits=96, repeats=3)
        total = next(row for row in rows if row["metric"] == "round_seconds")
        self.assertEqual(total["mean_seed_median"], 3.0)
        self.assertAlmostEqual(total["sd_seed_median"], statistics.stdev((2.0, 4.0)))
        self.assertEqual(total["seeds"], 2)


if __name__ == "__main__":
    unittest.main()
