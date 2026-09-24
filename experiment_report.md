# FedML Experiment Report

## Conclusion

The requested runs completed successfully in the verified FedML environment: 50 rounds of MNIST/FedRep, plus 20 and 50 rounds of personalized DMCFE-IP matrix factorization on MovieLens small.

## Outputs

| Experiment | Input | Output | Rows | Final/maximum metric |
|---|---|---|---:|---|
| MNIST/FedRep, 50 rounds | `python/examples/federate/simulation/sp_fedavg_mnist_lr_example/fedml_config_50_local.yaml` | `python/examples/federate/simulation/sp_fedavg_mnist_lr_example/results/dmcfe_fedrep_50_local.csv` | 2 evaluation rows (round 0, 49) | round-49 test accuracy `0.7851`; max error `3.02e-5` |
| MovieLens small, 20 rounds | `D:\temp\fedml_movielens\ml-latest-small\ratings.csv` | `python/examples/federate/simulation/sp_coded_ip_fedmf_example/results/dmcfe_pfedmf_movielens_20.csv` | 20 | max error `2.25e-5`; Recall@10 `0.0` |
| MovieLens small, 50 rounds | `D:\temp\fedml_movielens\ml-latest-small\ratings.csv` | `python/examples/federate/simulation/sp_coded_ip_fedmf_example/results/dmcfe_pfedmf_movielens_50.csv` | 50 | max error `2.32e-5`; Recall@10 `0.0` |

All commands exited with code 0. The recommendation CSVs contain 10 online clients per round, three aggregation chunks, zero clipping, and `backend_secure=False`.

## Interpretation

The integration and numerical plumbing are working. The zero MovieLens ranking metrics mean the current default factorization setup is not yet a publishable recommendation result; tune rank, learning rate, initialization, and evaluation protocol before drawing accuracy conclusions. The reference backend must also be replaced or paired with a reviewed secure DMCFE implementation before claiming cryptographic privacy.
