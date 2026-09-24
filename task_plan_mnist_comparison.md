# Task Plan: MNIST comparison experiment readiness

## Goal
Make the FedAvg/FedRep/reference-aggregation comparison safe to run, reproducible, and analyzable without overwriting completed results.

## Actions
- [x] Inspect the current recorder, aggregation diagnostics, runner, plotting code, configs, and saved outputs.
- [x] Record run provenance and complete per-round numerical diagnostics; keep the reference backend clearly labeled.
- [x] Add validated, resumable comparison runs with config/environment snapshots and safe result merging.
- [x] Add seed-level summaries and uncertainty plots; document the exact run commands.
- [x] Run focused tests and a small integration smoke test; inspect the diff and outputs.

## Decisions
- Work only in the MNIST comparison path and its directly shared recorder/aggregator code.
- Preserve existing result CSV files and the user's unrelated worktree changes.
- Do not call the chunked reference backend cryptographically secure.
- Keep one-round smoke outputs separate from the formal 50-round matrix.

## Errors Encountered
- A full-file `apply_patch` with both Delete and Add for the same path was rejected before changing files. The runner was then backed up and replaced in separate operations.
- The first real FedAvg smoke run finished training but could not write `C:\Users\A7R\.fedml\fedml_trace` inside the sandbox. The approved external rerun exited successfully.

## Status
Complete for experiment readiness. The formal 3-method × 3-seed × 50-round matrix has not been started.
