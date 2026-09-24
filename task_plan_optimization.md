# Task Plan: Privacy-Preserving Personalized FL Optimization

## Goal
Make the existing FedRep + DMCFE research prototype reproducible and safer to run for the paper "Privacy-Preserving Personalized Federated Learning Based on Functional Encryption".

## Phases
- [x] Phase 1: Inspect current FedML, FedRep, DMCFE, configuration, and worktree state
- [x] Phase 2: Fix dependency and credential boundaries
- [x] Phase 3: Harden weighted aggregation and optional telemetry behavior
- [x] Phase 4: Run static/tests and document exact limits
- [x] Phase 5: Align experiment entry points and add multi-seed comparison support

## Key Decisions
- Preserve unrelated user changes in the dirty worktree.
- Keep the DMCFE reference backend explicitly non-secure; improve reproducibility rather than making unsupported security claims.
- Default the example to local execution with no embedded credentials.

## Risks / Limits
- The current chunked and toy MCFE backends remain correctness/reference implementations, not production cryptography.
- A broad legacy test invocation has duplicate `run_test.py` module names; focused research tests pass.

## Status
**Complete** - Model construction, multi-seed comparison support, full-environment tests, and a real MNIST smoke run are complete.
