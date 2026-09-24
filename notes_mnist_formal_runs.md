# Notes: MNIST formal comparison

- Chapter 4.1 specifies 50 rounds, 1000 pre-partitioned clients, 3 sampled clients per round, CNN, batch size 10, SGD at learning rate 0.03, and seeds 0/1/2.
- FedML initializes NumPy and PyTorch with `random_seed`; the MNIST loader uses pre-partitioned data and a fixed internal NumPy seed, while client sampling uses the round index so methods see the same clients per round.
- The optimized reference path does not invoke dropout recovery. The recorder was corrected to leave `recovery_error`, `recovery_threshold`, and `coded_communication_rounds` blank for formal reference runs.
- The existing comparison directory contains only an older two-row manifest for unimplemented methods. A separate formal output directory avoids altering it.
- The optimized reference recorder now leaves recovery threshold/error, protocol dropout, and coded communication rounds blank. A focused test distinguishes it from the smoke recovery path.
- The runner writes each method/seed into its own MLOps and tracking run name and captures CUDA device metadata in the environment snapshot.
- Chapter 4.1 wording was updated to distinguish old CSV semantics from the new recorder and to limit timing claims to the non-secure reference backend.
- Preflight: `results/comparison_formal_50r` does not exist, D: has about 235 GB free, and the RTX 3060 has about 4.6 GB free GPU memory.
- Launched the 9-job matrix at 2026-09-18 17:36 Asia/Shanghai. Process ID 32676; stdout/stderr and launcher metadata are under `results/comparison_formal_50r/`. Initial manifest lists 1 running and 8 pending jobs.
- All three FedAvg jobs completed with exit code 0 and 50 unique rounds each; `fedrep_seed0_50r` started automatically.
- All six FedAvg/FedRep jobs completed with 50 rows each (300 merged rows). FedAvg/FedRep seed 0 used identical client IDs each round. The fixed-point reference seed 0 job started next.
- Final manifest: 9 completed jobs, exit code 0 and 50 rows each. `comparison_metrics.csv` contains 450 unique `(run_id, round)` pairs covering rounds 0–49. All three methods have matching sampled client sequences within each seed. Matrix ID: `aa8854cd2e503503`.
- Snapshots: 9 YAML configs and 9 environment JSON files. The recorded interpreter is `D:\anaconda\envs\fedmlnew\python.exe` (Python 3.9.25), PyTorch 2.8.0+cu129, CUDA 12.9, NVIDIA GeForce RTX 3060 Laptop GPU, and git HEAD `03e11dfee69a458a9820ec4e05b531a5f935eb2b`. Each environment snapshot includes source SHA-256 hashes.
- Reference rows: all 150 have `backend=chunked-reference` and `backend_secure=False`, have blank recovery/protocol dropout fields, and have `participation_rate=0.003`. The largest recorded fixed-point FedRep error is `3.0481253293900382e-05`.
- Final test accuracy across 3 seeds (mean ± sample SD): FedAvg `0.606521 ± 0.019334`, FedRep `0.531090 ± 0.054689`, FedRep+FixedPointReference `0.568173 ± 0.048684`. Paired final-accuracy differences versus FedAvg: FedRep `-0.075431`, reference `-0.038348` on average. These descriptive results do not establish statistical significance with 3 seeds.
- Outputs: `results/comparison_formal_50r/validation_summary.json`, four tables in `analysis/`, and `plots/test_acc.png` plus `plots/test_loss.png`. Both plots were visually inspected.
- Limits: the fixed-point path is a numerical, insecure reference; it does not measure an actual secure protocol, dropout recovery, cryptographic traffic, or memory overhead. Do not interpret its timing as secure-system overhead.
