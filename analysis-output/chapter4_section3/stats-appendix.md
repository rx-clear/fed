# Statistical Appendix: Fixed-Point Aggregation Sensitivity

## Validity and statistical scope

The raw file has 300 complete rows, each tagged `backend=chunked-reference`; no missing values were found. Every configuration is evaluated on the same five synthetic seed IDs (0–4), so factor-level numerical comparisons are paired by seed. The per-row elapsed-time field is the median of seven post-warmup calls in a single process. It is not an independent timing replicate and it excludes encryption, functional-key work, serialization, network transfer, and peak-memory measurement.

Timing environment: Windows 11; Python 3.12.14; PyTorch 2.14.0+cpu; 8 PyTorch CPU threads. This CPU microbenchmark is separate from the Chapter 4.1 GPU training environment.

Because there are only five synthetic seed-level numerical observations per factor level and the timing observations are benchmark medians, this analysis deliberately makes no null-hypothesis claim, normality test, p-value, or multiple-comparison decision. The analysis reports all levels, sample SD, and 95% t intervals for descriptive uncertainty; it does not treat the five seeds as five independent federated-training runs.

## Exact numeric summary (one factor varied; remaining parameters fixed at q=2^14, B=8, chunk=65,536)

| Factor | Level | Max abs. error (mean ± SD) | Mean abs. error (mean ± SD) | Clipped fraction (mean ± SD) | Reference elapsed time, ms (mean ± SD) |
|---|---:|---:|---:|---:|---:|
| q | 1,024 | 3.870e-04 ± 1.653e-05 | 8.349e-05 ± 1.826e-07 | 0.000 ± 0.000 | 6.241 ± 0.333 |
| q | 16,384 | 2.440e-05 ± 8.325e-07 | 5.220e-06 ± 1.398e-08 | 0.000 ± 0.000 | 6.735 ± 0.450 |
| q | 262,144 | 1.475e-06 ± 6.107e-08 | 3.268e-07 ± 9.895e-10 | 0.000 ± 0.000 | 6.438 ± 0.453 |
| B | 0.5 | 1.655e+00 ± 1.253e-01 | 1.423e-01 ± 2.865e-04 | 0.509 ± 0.001 | 7.657 ± 2.394 |
| B | 1.0 | 1.486e+00 ± 1.411e-01 | 7.299e-02 ± 2.850e-04 | 0.190 ± 0.001 | 6.892 ± 0.655 |
| B | 2.0 | 1.151e+00 ± 1.520e-01 | 2.117e-02 ± 1.893e-04 | 0.017 ± 0.000 | 6.886 ± 0.337 |
| B | 8.0 | 2.440e-05 ± 8.325e-07 | 5.220e-06 ± 1.398e-08 | 0.000 ± 0.000 | 6.735 ± 0.450 |
| chunk | 256 | 2.440e-05 ± 8.325e-07 | 5.220e-06 ± 1.398e-08 | 0.000 ± 0.000 | 17.018 ± 1.500 |
| chunk | 1,024 | 2.440e-05 ± 8.325e-07 | 5.220e-06 ± 1.398e-08 | 0.000 ± 0.000 | 8.773 ± 0.588 |
| chunk | 4,096 | 2.440e-05 ± 8.325e-07 | 5.220e-06 ± 1.398e-08 | 0.000 ± 0.000 | 6.764 ± 0.290 |
| chunk | 16,384 | 2.440e-05 ± 8.325e-07 | 5.220e-06 ± 1.398e-08 | 0.000 ± 0.000 | 6.510 ± 0.532 |
| chunk | 65,536 | 2.440e-05 ± 8.325e-07 | 5.220e-06 ± 1.398e-08 | 0.000 ± 0.000 | 6.735 ± 0.450 |


## 95% t interval half-widths

The summary CSV contains the 95% t interval half-width for every metric and factor level (`summary.csv`, $t_{0.975,4}=2.776$). These intervals characterize variation across the five generated update sets only; they do not establish population performance, efficiency, or cryptographic cost.

## Blocked claims

- No claim about functional-encryption latency, throughput, communication volume, or memory footprint is valid from this benchmark.
- No claim about end-to-end training accuracy, convergence, or optimal parameter values is valid from these synthetic-vector experiments.
- No cross-method superiority claim or significance claim is valid.
