# Notes: MNIST comparison readiness

- Existing `run_comparison.py` unlinks the merged CSV and each per-run CSV, with no resume or output validation.
- Existing 50-round single-method CSV has only rounds 0 and 49 because its evaluation frequency is 50.
- The comparison base config evaluates every round; a complete 3-method x 3-seed x 50-round matrix should have 450 metric rows.
- `chunked_reference` reports `is_secure=False`; any paper label must say reference or fixed-point aggregation.
- The verified local interpreter is `D:\anaconda\envs\fedmlnew\python.exe`; CUDA is available and MNIST is cached.
- The recorder now accepts run identity, backend identity, participation/protocol-dropout rates, and optimized aggregation diagnostics. Non-evaluation rounds write one row with blank accuracy/loss.
- Final focused test suite: 19 tests passed, covering recorder, optimized aggregator, runner, analyzer, and FedRep.
- Real one-round smoke outputs: `python/examples/federate/simulation/sp_fedavg_mnist_lr_example/results/comparison_smoke/`.
- The manifest has three completed runs (FedAvg, FedRep, FedRep+FixedPointReference), each with one validated row. The reference run records 1,198,592 shared coordinates, 23 chunks, clipped fraction 0, and `backend_secure=False`.
- `plot_comparison.py` produced four analysis CSV files and two PNGs from the smoke metrics. One seed is insufficient for an uncertainty band or performance conclusion.
- Formal 50-round, three-seed comparison remains unrun. Use a fresh `results/comparison/` matrix; source/config hashes prevent mixing implementations.
