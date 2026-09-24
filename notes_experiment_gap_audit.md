# Notes: Experiment Completion Gap Audit

## Confirmed Existing Work
- Chapter 4.3 fixed-point sensitivity sweep is complete: 300 rows covering 3 scales, 4 clipping bounds, 5 chunk sizes, and 5 seeds.
- A 50-round MNIST FedRep + reference aggregation run exists, but it contains only two evaluation rows (rounds 0 and 49).
- MovieLens 20/50-round runs exist, but the recorded Recall@10 and NDCG@10 are both zero under current defaults.
- The active high-dimensional `chunked-reference` backend reports `is_secure=False` and does not execute DMCFE-IP encryption.

## Confirmed Gaps So Far
- `results/comparison/comparison_metrics.csv` is missing.
- `results/comparison/plots/` is missing.
- `results/comparison/comparison_manifest.csv` contains only two unavailable-method rows and uses an older header without a seed column.
- `ExperimentRecorder` does not persist clipping fraction, backend identity/security, per-stage timing, serialized bytes, or peak memory.

## Pending Checks
- Focused verification passed: 38 tests across optimized aggregation, adapters, paper features, personalized MF, coded MF, baselines, FedRep, and experiment recording.
- The optimized high-dimensional path and complete dropout recovery remain separate. Dropout IDs are consumed only by the non-optimized smoke path; optimized mode calls `_aggregate_dmcfe_optimized` directly.
- The MNIST comparison runner supports method/seed matrices, but the saved formal comparison output has not been generated. The current manifest is stale and contains only two unavailable methods.
- The runner is not interruption-safe: it deletes merged output at startup, writes no `running`/`failed` status, stores no generated config snapshot, and offers no resume mode.
- Client sampling is seeded by `round_idx` alone. This deliberately pairs client selections across methods and seeds, but the run artifact should record this policy explicitly.
- The classification recorder does not persist `clipped_fraction`, backend/security identity, chunk/coordinate counts, stage timings, communication bytes, peak memory, run ID, seed, or config hash.
- The comparison plotter averages seeds but draws no variability band and produces no seed-level statistics table.
- The personalized recommendation CLI exposes rank/rounds/dropout/top-k but not learning rate, regularization, or seed. Its current MovieLens results have zero Recall@10/NDCG@10; the paper-bundle runner evaluates a different RMSE-oriented coded MF path and does not repair this ranking experiment.

## Priority Classification

### P0: Required for a defensible completed MNIST experiment
- Extend the result schema and FedAvg/FedRep plumbing with real per-round diagnostics.
- Make the comparison runner resumable, provenance-preserving, and strict about 50 unique rows per method/seed.
- Add seed-level aggregation/statistics and uncertainty-aware plots.
- Run FedAvg, FedRep, and the accurately named reference-aggregation variant for seeds 0, 1, and 2.
- Decide the claim boundary: either implement a reviewed secure DMCFE-IP backend or explicitly avoid calling the reference variant a cryptographic DMCFE implementation.

### P1: Required only if dropout and system-overhead claims stay central
- Compose the high-dimensional training path with threshold dropout recovery.
- Add per-stage timing, serialized payload bytes, and peak-memory measurement for the concrete backend.
- Add integration tests for early dropout, late dropout, threshold abort, and high-dimensional aggregation.

### P2: Required only if MovieLens/recommendation remains a reported contribution
- Add seed, learning rate, regularization, negative-sampling/evaluation controls, and hyperparameter sweep support.
- Add a matched plaintext personalized-MF baseline and multi-seed ranking statistics.
- Re-run until the protocol is validated and metrics are non-degenerate; do not tune on the held-out test item.

## Verification
- Command: `D:\anaconda\envs\fedmlnew\python.exe -m pytest -q -p no:cacheprovider ...`
- Result: `38 passed, 7 warnings in 11.14s`.
- `py_compile` passed for the comparison runner, plotter, recorder, FedAvg API, and personalized MF script.
- `run_comparison.py --help` exits successfully.
