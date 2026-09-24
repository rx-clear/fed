# Task Plan: MNIST formal comparison runs

## Goal
Run a reproducible 3-method, 3-seed, 50-round MNIST comparison with honest protocol metrics and validated analysis outputs.

## Actions
- [x] Review the current comparison code, config, manuscript experiment settings, and existing outputs.
- [x] Correct protocol-only recorder fields for the optimized reference path; add focused verification.
- [x] Freeze method-specific run labels and a separate formal output directory; verify the generated configs.
- [x] Execute FedAvg, FedRep, and FedRep+FixedPointReference for seeds 0, 1, and 2 with resume support. Started at 2026-09-18 17:36 Asia/Shanghai; launcher PID 32676; all 9 jobs exited successfully.
- [x] Validate 9 manifest entries, 450 unique metric rows, per-run snapshots, and analysis tables/plots.
- [x] Record exact run environment, evidence, and remaining limits.

## Decisions
- Use the existing `fedml_config.yaml` as the base; the runner overrides method, seed, rounds, result path, and per-round evaluation.
- Write formal results to `results/comparison_formal_50r` so the pre-existing `results/comparison/comparison_manifest.csv` remains untouched.
- Preserve the earlier one-round smoke outputs and the two-point 50-round local output.
- Treat `chunked-reference` as an insecure fixed-point numerical reference.

## Status
Complete: 9/9 runs, 450/450 unique round rows, 9 YAML and 9 environment snapshots, 4 analysis CSVs, and 2 inspected plots. Matrix ID `aa8854cd2e503503`.
