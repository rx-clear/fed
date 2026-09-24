# Notes: Privacy-Preserving Personalized FL Optimization

## Evidence
- `python/fedml/simulation/sp/fedavg/fedavg_api.py` imports `wandb` unconditionally, while the root `requirements.txt` no longer installs it.
- `FedAvgAPI._aggregate` accepts empty input, zero/negative sample counts, and mismatched parameter keys; it initializes outputs with Python integer `0`.
- `privacy_baselines.py` imports `sympy` and `dmcfe/weight_signatures.py` imports `cryptography`, but these dependencies are not declared in the root requirements or `python/setup.py`.
- The active example config contains non-empty MLOps/W&B credential fields and enables remote MLOps by default.
- Existing documentation correctly states that the current DMCFE backends are non-secure reference implementations.

## Intended Changes
- Make W&B optional and fail clearly only when explicitly enabled without the package.
- Validate and clone weighted aggregation inputs without mutating caller state.
- Declare cryptography and sympy as dependencies used by the research path.
- Replace embedded credentials with empty local defaults and disable remote tracking in the example config.

## Applied
- `FedAvgAPI._wandb_enabled()` now keeps W&B optional and emits an actionable error only when enabled without the package.
- Plain weighted aggregation now rejects empty updates, non-positive/non-integral sample counts, inconsistent keys/shapes, and non-tensor values; it clones output tensors before accumulation.
- `requirements.txt` and `python/setup.py` now declare `cryptography` and `sympy` for the imported research modules.
- `fedml_config.yaml` now uses empty credentials and `using_mlops: false` by default.

## Additional Research Reproducibility Gap
- The custom MNIST entry point always constructs an MLP even when the YAML says `model: cnn`; this makes results depend on which entry script is used.
- `run_comparison.py` only runs one seed and writes no seed column, which is insufficient for multi-seed paper tables.

## Phase 5 Results
- The custom MNIST entry point now uses YAML-selected models; `mlp`/`fedrep_mlp` selects the custom MLP and `cnn` delegates to FedML's model hub.
- MNIST CNN creation now uses ten output classes for MNIST while retaining the 62-class path for FEMNIST.
- `run_comparison.py --seeds 0,1,2` now produces per-seed result files, a `seed` column in merged metrics, and seed-aware manifest rows.
- Full-environment targeted tests pass: 20 tests. A real 1-round GPU MNIST/FedRep+DMCFE smoke run also completed and wrote `results/dmcfe_fedrep_smoke.csv`.
- The seed-merge helper was verified with a temporary CSV; an earlier shell assertion used escaped newlines incorrectly and was rerun successfully.
