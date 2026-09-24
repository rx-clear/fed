# Notes: Chapter 4 Section 3

## Scope
Parameter sensitivity and computational overhead analysis for fixed-point aggregation.

## Findings
- Core implementation: `python/fedml/simulation/sp/fedavg/optimized_dmcfe_ip.py`.
- Default `ChunkedReferenceBackend` performs CPU int64 chunked weighted sums; it has `is_secure=False` and does not encrypt.
- `DMCFEAggregationDiagnostics` reports backend, clients, coordinates, chunks, scale, clip bound, clipped fraction, max absolute error, and mean absolute error.
- Existing unit tests (`python/tests/test_optimized_dmcfe_ip.py`) cover weighted-sum correctness, integer-buffer validation, clipping reporting, and toy backend matching, but not a parameter-sensitivity grid or timing protocol.
- Existing chapter 4.2 explicitly blocks interpreting reference-backend timing as production functional-encryption overhead.
- Reproducibility run: `python/tests/run_dmcfe_sensitivity_ablation.py` generated 300 rows (3 scales × 4 clipping bounds × 5 chunk sizes × 5 seeds); each timing row is a seven-repeat post-warmup median.
- Analysis bundle: `analysis-output/chapter4_section3/` contains raw summaries, two vector PDF figures, a claim-boundary report, and a statistics appendix. At the principal configuration $q=2^{14}$, $B=8$, $L=65536$, mean maximum error is $2.440e-05$ and mean reference time is 6.735 ms (both with seed-level SD in Table 4-10).
- Timing environment: Windows 11; Python 3.12.14; PyTorch 2.14.0+cpu; 8 PyTorch CPU threads. This is separate from the Chapter 4.1 GPU training environment.
