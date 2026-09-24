# Optimization Report

This report records the scoped reliability and reproducibility changes made to the FedRep + DMCFE experiment prototype.

The implementation remains a research reference: the chunked backend and toy MCFE backend do not provide production cryptographic security. Results should therefore be reported with the stated threat-model limitation.

## Changed Files

- `python/fedml/simulation/sp/fedavg/fedavg_api.py`: optional W&B import with an early actionable error when explicitly enabled, plus strict, non-mutating plaintext aggregation validation.
- `python/setup.py`: declared `cryptography` and `sympy` dependencies used by the privacy research path.
- `requirements.txt`: declared the same two dependencies for direct environment setup.
- `python/examples/federate/simulation/sp_fedavg_mnist_lr_example/fedml_config.yaml`: removed embedded credentials and disabled remote MLOps by default.
- `python/examples/federate/simulation/sp_fedavg_mnist_lr_example/torch_fedavg_mnist_lr_custum_data_and_model_example.py`: respects the configured model instead of silently forcing MLP.
- `python/fedml/model/model_hub.py`: fixes MNIST CNN output width to 10 classes.
- `python/examples/federate/simulation/sp_fedavg_mnist_lr_example/run_comparison.py`: adds deterministic multi-seed execution and seed-aware merged metrics.

## Verification

- `D:\anaconda\python.exe -c "...ast.parse(...)..."`: `AST OK 5`.
- `D:\anaconda\python.exe -m compileall -q ...`: `compileall OK`.
- PyYAML load of the active config: `YAML OK`; `using_mlops=False`, MLOps/W&B key fields empty.
- `git diff --check`: passed (only line-ending warnings from pre-existing modified files).
- Targeted pytest collection was attempted but is blocked by `ModuleNotFoundError: No module named 'torch'` in the available environments.
- With `D:\anaconda\envs\fedmlnew\python.exe`, the focused FedRep/DMCFE suite passes: `20 passed`.
- A real GPU 1-round MNIST/FedRep+DMCFE run completed; output is `python/examples/federate/simulation/sp_fedavg_mnist_lr_example/results/dmcfe_fedrep_smoke.csv`.
- After model-selection fixes, a real GPU 1-round CNN MNIST/FedRep+DMCFE run also completed; the model exposed 10 output classes and wrote `python/examples/federate/simulation/sp_fedavg_mnist_lr_example/results/dmcfe_fedrep_cnn_smoke.csv`.
- A broader mixed test invocation still has an existing pytest module-name collision between `test_fedml_mlops_log/run_test.py` and `test_model_cli/run_test.py`; the focused suite is unaffected.
- The seed-aware CSV merge helper was verified with a temporary source/target pair and produced `method,seed,round,test_acc` as expected.

Verification commands and their exact outcomes are recorded in the final handoff.
