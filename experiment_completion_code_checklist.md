# Experiment Completion Code Checklist

## Recommended Scope

Complete the MNIST experiment first. Treat the existing `chunked-reference` variant as a fixed-point reference aggregation method unless a reviewed secure DMCFE-IP backend is actually integrated. Keep the MovieLens branch out of the main results unless recommendation is an explicit thesis contribution.

## P0: Complete Before the Final Experimental Runs

### 1. Persist complete per-round diagnostics

Files:
- `python/fedml/simulation/sp/fedavg/experiment_recorder.py`
- `python/fedml/simulation/sp/fedavg/fedavg_api.py`
- `python/fedml/simulation/sp/fedavg/optimized_dmcfe_ip.py`

Add these fields:
- `run_id`, `method`, `seed`, `config_hash`, `backend`, `backend_secure`
- `clipped_fraction`, `coordinates`, `chunks`, `scale`, `clip_bound`, `chunk_size`
- `encode_seconds`, `aggregate_seconds`, and `total_aggregation_seconds` for the reference path
- real cryptographic phase timings only when a concrete secure backend exists
- `serialized_upload_bytes`, `serialized_key_bytes`, `recovery_bytes`, and `peak_memory_bytes` when those payloads are actually materialized

Correct the meaning of `dropout_rate`: keep sampled participation rate separate from injected protocol-dropout rate.

Acceptance checks:
- One row per `(run_id, round)`.
- Exactly 50 unique round rows for each completed 50-round run.
- Every DMCFE/reference row records backend identity and `backend_secure`.
- Clipping diagnostics are non-empty on every optimized aggregation round.

### 2. Harden the unified comparison runner

File:
- `python/examples/federate/simulation/sp_fedavg_mnist_lr_example/run_comparison.py`

Add:
- `--resume` and `--force` behavior.
- Atomic per-run output followed by merge; do not delete valid completed outputs on startup.
- Manifest states `pending`, `running`, `completed`, and `failed`, including exit code and error text.
- A copied YAML snapshot and environment metadata for every method/seed.
- Validation for rounds `0..49`, no duplicates, finite metrics, expected method semantics, and expected seed.
- Accurate method naming, such as `FedRep+FixedPointReference`, while `backend_secure=False`.

Add focused tests for config generation, resume behavior, failed-run preservation, row validation, and deterministic merging.

### 3. Add formal comparison analysis

Suggested new file:
- `python/examples/federate/simulation/sp_fedavg_mnist_lr_example/analyze_comparison.py`

Generate:
- Per-method/per-round mean and standard deviation over seeds.
- Final-round and convergence summaries at the seed level.
- Paired method differences using matched seeds; never treat 50 rounds as 50 independent replicates.
- Curves with uncertainty bands and a machine-readable summary CSV.
- A claim-boundary field distinguishing reference aggregation from secure functional encryption.

### 4. Execute the complete matrix

Required runs:
- Methods: FedAvg, FedRep, FedRep + fixed-point reference aggregation.
- Seeds: 0, 1, 2.
- Rounds: 50, evaluation every round.

Expected minimum output:
- 9 completed runs.
- 450 unique metric rows before aggregation.
- One manifest entry and one config snapshot per run.
- Summary tables and convergence plots generated from the merged CSV.

## P1: Only If the Thesis Claims Dropout-Integrated DMCFE or Real Overhead

### 5. Integrate high-dimensional aggregation with dropout recovery

Current limitation: the complete recovery code is used by the smoke path, while optimized high-dimensional aggregation bypasses it.

Implement one round interface that consumes the key-sharing, ciphertext-upload, and recovery-response client sets; applies threshold checks; and returns the same diagnostics contract used by training. Test early dropout, late dropout, threshold abort, signature failure, and a multi-chunk vector.

### 6. Implement or bind a reviewed DMCFE-IP backend

This is mandatory before claiming functional-encryption confidentiality or cryptographic overhead. The backend contract should expose setup, client encryption, partial functional-key generation, key combination, decryption, serialization, security parameters, and measured phase diagnostics. Keep `chunked-reference` as the numerical oracle.

If this backend is not implemented, revise the method/result label and limit conclusions to personalization flow, fixed-point numerical consistency, and independent recovery-path behavior.

## P2: Only If Recommendation Is Part of the Final Contribution

Files:
- `python/examples/federate/simulation/sp_coded_ip_fedmf_example/dmcfe_pfedmf.py`
- a new ranking sweep/analysis script or an extension of the existing paper runner

Add CLI/config support for seed, learning rate, regularization, local update steps, ranking evaluation protocol, and sweep ranges. Add a matched plaintext personalized-MF baseline and report multi-seed Recall@K/NDCG@K. The current zero ranking metrics are not publishable evidence.

## Minimal Development Order

1. Recorder schema and semantic fixes.
2. Runner resume/provenance/validation.
3. One-round smoke matrix for all three MNIST methods.
4. Formal 9-run matrix.
5. Statistical analysis and plots.
6. Secure backend and integrated dropout only if those claims are retained.
7. Recommendation branch only if it remains in the thesis scope.
