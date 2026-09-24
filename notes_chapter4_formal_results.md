# Notes: Chapter 4 formal MNIST results

## Sources
- Formal matrix: `python/examples/federate/simulation/sp_fedavg_mnist_lr_example/results/comparison_formal_50r/`.
- `comparison_manifest.csv`: 9 completed runs, each with 50 rows and exit code 0.
- `comparison_metrics.csv`: 450 unique `(run_id, round)` rows, rounds 0–49.
- `analysis/seed_summary.csv` and `analysis/paired_summary.csv`: seed-level summaries and paired differences.
- `validation_summary.json`: same sampled client sequence within each seed; 9 config and 9 environment snapshots; all reference recovery fields blank.

## Claim boundary
- The third method is `FedRep+FixedPointReference`, executed with `chunked-reference` and `backend_secure=False`.
- Its coordinate-level error measures the fixed-point reference against plaintext weighted aggregation; it does not measure recovery error or cryptographic security.
- The separate dropout-recovery smoke result remains independent of the 50-round training comparison.
- Three seed-level runs support descriptive means and sample SD, but no strong statistical ranking or security/overhead claim.

## Recomputed values (seed-level final round, n=3)
- FedAvg: accuracy `60.6521 ± 1.9334` percentage points; loss `1.123792 ± 0.074133`.
- FedRep: accuracy `53.1090 ± 5.4689` percentage points; loss `1.295988 ± 0.154881`.
- FedRep+FixedPointReference: accuracy `56.8173 ± 4.8684` percentage points; loss `1.166102 ± 0.099086`.
- Paired accuracy difference versus FedAvg: FedRep `-7.5431 ± 6.4379` percentage points; reference `-3.8348 ± 4.8187` percentage points.
- All 150 reference rounds: 1,198,592 coordinates, 23 chunks, `q=16384`, `B=8`, `L=65536`, clipped fraction `0`; maximum coordinate error `3.0481253293900382e-05`.
- `FedAvgAPI._local_test_on_all_clients` loops over all clients with a local test set and aggregates correct predictions and losses by sample count. The plot shows per-round means across 3 seeds, with bands of ± sample SD.
- `_client_sampling` seeds NumPy from the communication-round index. The CSV confirms that all 9 runs use the same client IDs at each of the 50 rounds; seed SD therefore excludes variation from client sampling sequences.

## Allowed interpretation
- Report the observed final-round method differences in this specific setup. Do not claim significance, general superiority, or an accuracy gain caused by the fixed-point method.
- The 20-round FedAvg numerical regression and older 50-round single run remain separate preliminary artifacts; do not pool them with the formal matrix.
- Add the formal accuracy and loss plots as Figures 4-1 and 4-2; renumber Section 4.3's existing plots to Figures 4-3 and 4-4.

## Legacy CSV correction
- `results/fedavg_baseline.csv` has 40 rows: 20 rounds duplicated exactly.
- `results/dmcfe_fedavg.csv` has 46 rows: its first 40 rows form the paired 20-round trajectory, then 6 rows append a separate 3-round trajectory. Global deduplication would yield 23 distinct rows, so Table 4-7 explicitly selects the first complete segment before deduplication.
- Recomputed from the selected 20 rounds: final loss difference `2.3255757699569557e-06`, maximum accuracy difference `0.054266720933388024` percentage points, mean accuracy difference `0.011531678198344608` percentage points, maximum loss difference `1.3889653148080328e-05`, maximum coordinate error `3.0214207947163482e-05`; these match Table 4-7 after rounding.
- Initial manuscript validation script failed on Chinese caption literals when piped through PowerShell to Python. Replacing the literals with Unicode escapes resolved the check; data and image paths were unaffected.
- The final validation script initially rejected intentional two-space Markdown hard breaks in Section 4.1. Allowing those existing hard breaks while checking other trailing whitespace resolved the false positive.

## Final checks
- Recomputed all new Table 4-8 accuracy/loss means and sample SDs and paired accuracy differences from the 9 seed summaries.
- Verified 450 formal metric rows, 150 reference rows with three verified weight signatures and zero clipped fraction, and the legacy 40/46-row segment boundary.
- Verified four existing image links, Figures 4-1 through 4-4, Tables 4-7 through 4-11, and no accidental trailing whitespace.
- No training configuration, runner source, analysis CSV, or previous result file was edited in this manuscript pass.
