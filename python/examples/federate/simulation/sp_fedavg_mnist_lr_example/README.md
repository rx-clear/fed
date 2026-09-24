# FedRep + DMCFE on MNIST

This example uses a two-part MNIST representation model. `linear_2` is the
client-specific prediction head. All other floating-point state belongs to the
shared representation and is aggregated through DMCFE when `enable_dmcfe` is
enabled.

Each client keeps its own head across communication rounds. Local training
first adapts that head for `fedrep_head_epochs`, then freezes it and trains the
representation for `fedrep_representation_epochs`.

## Install FedML
```
pip install fedml
```

# Run the example (one line API)
```
python torch_fedavg_mnist_lr_one_line_example.py --cf fedml_config.yaml
```

# Run the example (step by step APIs)
```
python torch_fedavg_mnist_lr_step_by_step_example.py --cf fedml_config.yaml
```

# Run the example (custom dataset and model)
```
python torch_fedavg_mnist_lr_custum_data_and_model_example.py --cf fedml_config.yaml
```

When `enable_dmcfe: true`, the run writes protocol metrics to
`results/dmcfe_fedrep.csv`. `dmcfe_validation_dim` controls how many flattened
representation coordinates are checked by the real quantized DMCFE smoke test
(the remaining shared coordinates continue through the plaintext weighted-
average path). Personalized head coordinates are never sent to server
aggregation. The reference backend is a correctness adapter, not evidence of
cryptographic security.

Set `dmcfe_aggregation_mode: optimized` to use the full shared-state,
chunked fixed-point aggregator. It reports clipping and quantization error and
does not use the prefix-only smoke-test fallback. The backend is still a
reference implementation until a reviewed DMCFE-IP primitive is supplied.
The optimized reference path does not implement dropout recovery; it rejects
non-empty failure-injection lists rather than treating them as a tested
recovery run. For small protocol checks, set `dmcfe_backend: toy_mcfe`; this invokes the
cached toy `MCFE.encrypt/derive_key/decrypt` path and is not a security claim.

The DMCFE/plaintext equivalence errors are recorded in the
`dmcfe_fedrep_max_error` and `dmcfe_fedrep_mean_error` columns.

The implementation also includes three paper-derived building blocks:

* `dmcfe_enable_weight_signatures` enables Ed25519 signatures over the round,
  participant set, client id, and sample weight. The server rejects missing,
  replayed, or modified proposals before DMCFE key derivation and records the
  number verified in `dmcfe_weight_signatures_verified`.
* `dmcfe.plaintext_packing` provides signed fixed-point slot packing and
  aggregate decoding. It is opt-in because the current 256-bit correctness
  backend is intentionally too small for production-scale packed plaintexts.
* `MCFE.prepare_encryption` / `prepare_decryption` cache the repeated
  `H(label)^s` terms described in the paper's modular-exponentiation
  optimization section.

Round-3 dropout recovery is implemented in the smoke-test DMCFE path. That path
constructs the `U2/U3/U4` protocol sets; when a client is absent before
ciphertext upload, its `sk_i^(1)` is reconstructed from Shamir shares to remove
the unmatched pairwise mask. When it is absent before the recovery response,
its `eta_i` is reconstructed and removed as well. For a local failure-injection
test, set for example:

```yaml
dmcfe_dropout_before_encryption_ids: [2]
dmcfe_dropout_before_recovery_ids: [1]
dmcfe_recovery_threshold: 2
```

The round aborts if fewer than the configured threshold of recovery responders
remain. Empty dropout lists still execute the same protocol path and reduce to
the normal no-dropout case.

## 50-round comparison

Run a one-round smoke matrix in a separate output directory first:

```powershell
python run_comparison.py --methods FedAvg,FedRep,Ours --seeds 0 --rounds 1 --output-dir results/comparison_smoke
```

If that directory already contains a different matrix ID after code or config
changes, choose a fresh `--output-dir` for the new smoke run.

For the 50-round comparison, run three matched seeds. `--resume` validates and
skips finished method/seed runs after an interruption:

```powershell
python run_comparison.py --methods FedAvg,FedRep,Ours --seeds 0,1,2 --rounds 50 --output-dir results/comparison_formal_50r --resume
```

The runner writes a validated per-run CSV and YAML/environment snapshot before
marking a run completed in `results/comparison_formal_50r/comparison_manifest.csv`. It
builds `results/comparison_formal_50r/comparison_metrics.csv` from completed runs only.
The matrix ID includes the base config, round count, and relevant source-file
hashes, so a changed implementation requires a new output directory. Use
`--force` only when intentionally replacing selected completed runs; the
old per-run CSV remains in place until the replacement passes validation.
With the default three methods, three seeds, and 50 rounds, expect 9 completed
manifest entries and 450 metric rows. The `Ours` CLI alias appears in results
as `FedRep+FixedPointReference`, because its backend is not cryptographically
secure. BatchCrypt and
Masking are implemented as local simulation baselines in
`fedavg/privacy_baselines.py`. Use `--methods all` to also write the requested
`HybridAlpha` and `CryptoFE` entries to `comparison_manifest.csv`. Those two
entries remain `not_implemented`: their MIFE/FE constructions require
scheme-specific key setup and neither paper provides a drop-in implementation
for this repository. The manifest includes the paper references, and they are
never silently substituted with FedAvg or DMCFE.

After the runs finish, generate mean curves with sample-standard-deviation
bands and seed-level summaries with:

```powershell
python plot_comparison.py --input results/comparison_formal_50r/comparison_metrics.csv --output-dir results/comparison_formal_50r/plots
```

This writes `results/comparison_formal_50r/plots/test_acc.png` and
`results/comparison_formal_50r/plots/test_loss.png`, plus CSV files under
`results/comparison_formal_50r/analysis/`. The `dropout_rate` column is populated
only when the smoke protocol actually runs recovery; it is blank for the
optimized reference method. `participation_rate` records sampled clients
divided by total clients.
Paired differences use the same seed for each
method and FedAvg; rounds are not treated as independent replicates. A single
seed has no standard-deviation band, and these summaries make no significance
claim.

These additions do not claim the paper's 2048-bit security level: the local
correctness backend still uses a reduced modulus and the debug PRG remains a
test implementation. Use a production DCR/PRG implementation before treating
the run as a secure deployment.

## Isolated toy MCFE timing probe

The repository has no reviewed DMCFE-IP backend for the 1,198,592-coordinate
FedRep representation. To inspect the existing toy modular-arithmetic path at
small dimensions, run this separate probe from the example directory:

```powershell
python benchmark_toy_mcfe.py --dimensions 1,8,32 --seeds 0,1,2,3,4 --clients 3 --repeats 7 --warmups 2 --modulus-bits 256 --output-dir results/toy_mcfe_probe_v1
```

Choose a fresh output directory for a rerun; the script refuses to replace a
non-empty directory. It checks the exact sample-weighted integer sum on every
trial. `setup.csv` records one-time modulus setup and key generation;
`trials.csv` separates context preparation, encryption, key derivation,
decryption, and integer-payload encoding times. `summary.csv` first takes the
median of the seven timed repetitions for each seed, then reports the mean and
sample standard deviation across seeds. Warmup trials are excluded.
The `round_seconds` wall time also includes the local exact-result check and
payload encoding; use the separate phase columns when inspecting arithmetic
costs. The seed fixes generated integer messages; modulus setup and secret
keys are freshly sampled, so repeated invocations need not reproduce timings.

The payload counts use this probe's fixed-width unsigned encoding for toy
ciphertext integers and minimal signed encoding for the derived-key integer.
They exclude metadata, network framing, key exchange, and transport. The
primitive reuses one label across coordinates as the current toy backend does;
its reduced parameters and key distribution are not security-reviewed. These
local Python timings and payload counts are **not** secure DMCFE-IP or
end-to-end federated-training overhead and are not included in Chapter 4.3's
functional-encryption cost claims.

The completed small-dimension run and its audit inputs are documented in
[`results/toy_mcfe_probe_v1/report.md`](results/toy_mcfe_probe_v1/report.md).
