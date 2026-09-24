# Experiment Notes: FedML MNIST/FedRep and DMCFE-IP MovieLens

## Environment
- Interpreter: `D:\anaconda\envs\fedmlnew\python.exe`
- Verified imports include `torch`, `fedml`, `yaml`, `sympy`, and `cryptography`.
- Source compile check passed with `python -m compileall`.

## Runs
- MNIST/FedRep: 50 training rounds using `fedml_config_50_local.yaml`.
  - Output: `python/examples/federate/simulation/sp_fedavg_mnist_lr_example/results/dmcfe_fedrep_50_local.csv`
  - Validation rows: round 0 and round 49 (evaluation frequency is 50).
  - Final test accuracy: `0.7851037851037851`.
  - Final DMCFE-FedRep max error: `3.0220671760616824e-05`.
- MovieLens small: `D:\temp\fedml_movielens\ml-latest-small\ratings.csv`, partitioned into 12 clients.
  - 20 rounds: `python/examples/federate/simulation/sp_coded_ip_fedmf_example/results/dmcfe_pfedmf_movielens_20.csv`.
  - 50 rounds: `python/examples/federate/simulation/sp_coded_ip_fedmf_example/results/dmcfe_pfedmf_movielens_50.csv`.
  - Both runs used dropout rate 0.2, 10 online clients per round, 3 aggregation chunks, and zero clipping.
  - 20-round max DMCFE error: `2.25289950941484e-05`.
  - 50-round max DMCFE error: `2.32001302553289e-05`.
  - Recall@10 and NDCG@10 remained 0.0 under the current default MF hyperparameters.

## Limits
- The optimized DMCFE backend is a chunked reference implementation with `backend_secure=False`; these runs validate integration, numerical error, dropout handling, and metrics plumbing, not a cryptographic security theorem.
- Existing 20/50-round synthetic fallback outputs remain available, but the reported MovieLens outputs use the real MovieLens small ratings file.
