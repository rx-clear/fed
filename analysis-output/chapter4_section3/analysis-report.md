# Analysis Report: Fixed-Point Aggregation Sensitivity

## Analysis question

For the non-secure `chunked-reference` backend, how do fixed-point scale $q$, coordinate clipping bound $B$, and chunk size affect numerical deviation from an **unclipped floating-point weighted average**, clipping incidence, and reference implementation elapsed time?

## Evidence and comparison unit

- Input: `python/examples/federate/simulation/sp_fedavg_mnist_lr_example/results/dmcfe_sensitivity_ablation.csv` (300 rows).
- Design: 3 scale levels × 4 clipping bounds × 5 chunk sizes × 5 seeded synthetic update sets; 8 clients, 32,768 coordinates, and fixed integer client weights per seed.
- Unit for numerical summaries: one seed-level aggregation result ($n=5$ per plotted factor level after holding the other two factors at $q=2^{14}$, $B=8$, chunk size 65,536).
- Unit for timing summaries: one seed-level median of 7 warm timing repetitions, measured within one Python process.
- Timing environment: Windows 11; Python 3.12.14; PyTorch 2.14.0+cpu; 8 PyTorch CPU threads. It is distinct from the GPU training environment reported in Chapter 4.1.
- Uncertainty: mean ± sample SD; 95% t intervals are listed in the appendix only. No significance test is reported: $n=5$ is small, timings are same-process measurements, and synthetic seeds are sensitivity probes rather than independent training runs.

## Key findings

1. With $B=8$ and chunk size 65,536, increasing $q$ from $2^{10}$ to $2^{18}$ changes the mean maximum absolute error from 3.870e-04 to 1.475e-06; this is the expected reduction in rounding resolution error under an inactive clipping bound.
2. With $q=2^{14}$ and chunk size 65,536, reducing $B$ from 8 to 0.5 changes the mean maximum absolute error from 2.440e-05 to 1.655e+00. The associated clipped-coordinate fraction is reported directly in the summary table; this comparison conflates clipping bias and quantization error by design because the error reference is unclipped.
3. At the baseline configuration, the numerical summary is maximum error 2.440e-05 ± 8.325e-07, mean error 5.220e-06 ± 1.398e-08, clipped fraction 0.000 ± 0.000, and reference-backend elapsed time 6.735 ± 0.450 ms.
4. Among the tested chunk sizes, the fastest mean reference-backend elapsed time occurs at 16,384 coordinates/chunk (6.510 ms); the slowest occurs at 256 (17.018 ms). This is a property of this CPU loop and allocator/runtime state, not a cryptographic benchmark.

## Claim Candidates

- Claim: Increasing the scale factor reduces the measured deviation when clipping is inactive in this synthetic fixed-point probe.
  - Source evidence: `figure-01-error-sensitivity.pdf`, factor-held-constant rows in the raw CSV.
  - Allowed wording: “在该定点参考聚合探针中，较大的缩放因子对应更小的数值偏差。”
  - Forbidden stronger wording: “更大的缩放因子必然提升端到端训练精度。”
  - Uncertainty: only five synthetic update sets; no training-accuracy outcome is measured.
  - Next check: repeat on saved per-round model updates from multi-seed training.
  - Decision: keep.

- Claim: A smaller clipping bound produces greater deviation from the unclipped reference when more coordinates are clipped.
  - Source evidence: raw clipping fractions and `figure-01-error-sensitivity.pdf`.
  - Allowed wording: “较小裁剪阈值在本探针中引入更明显的相对无裁剪参考偏差。”
  - Forbidden stronger wording: “该阈值损害最终模型性能。”
  - Uncertainty: no learning trajectory or generalization metric is included.
  - Next check: attach the same diagnostic to model-training rounds.
  - Decision: keep.

- Claim: Chunk size changes `chunked-reference` elapsed time under a fixed CPU workload.
  - Source evidence: `figure-02-reference-timing.pdf`.
  - Allowed wording: “分块长度影响参考实现的本地聚合耗时。”
  - Forbidden stronger wording: “分块长度决定功能加密计算开销” or “方案密码计算效率为 X ms.”
  - Uncertainty: no encryption, key generation, ciphertext serialization, communication, or peak memory is measured.
  - Next check: benchmark a production-reviewed DMCFE-IP backend by protocol stage.
  - Decision: keep with boundary.
