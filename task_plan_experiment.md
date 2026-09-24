# Task Plan: FedML MNIST/FedRep and Personalized Recommendation Runs

## Goal
Verify the existing DMCFE-FedRep implementation for 50 MNIST rounds and run the personalized DMCFE-IP recommendation experiment for 20 and 50 rounds in the available FedML environment.

## Phases
- [x] Phase 1: Verify MNIST 50-round artifacts and process completion
- [x] Phase 2: Run the 20-round recommendation smoke experiment
- [x] Phase 3: Run the 50-round recommendation experiment if Phase 2 is stable
- [x] Phase 4: Validate CSV schemas, row counts, metrics, and document limits

## Decisions
- Use `D:\anaconda\envs\fedmlnew\python.exe`, the verified FedML environment.
- Use the deterministic MovieLens-shaped fallback because no local MovieLens ratings file is present.
- Preserve existing user modifications and write new experiment outputs under the example's `results/` directory.

## Status
Complete. All requested round counts finished with exit code 0 and validated CSV outputs.
