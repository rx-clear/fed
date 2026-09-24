# Coded-IP-FedMF simulation

This example exercises the protocol described in the accompanying paper:

1. `K` source shards are encoded into `N` Vandermonde/MDS shares.
2. Each share is independently wrapped by the reference inner-product backend.
3. The server uses the online client set to derive recovery weights and obtains only the aggregate gradient.

Run it from this directory with:

```bash
python coded_ip_fedmf_demo.py --client-num 8 --recovery-threshold 5 --dropout 2
```

The printed `max_recovery_error` should be close to machine precision. The
backend is intentionally marked `reference_backend_secure: false`; it models
the protocol interface and must be replaced with a reviewed MCFE/IPFE library
for production security experiments.

## Experiment sweep

Run the paper-style dropout sweep (0%, 10%, 20%, and 30%; three repeats):

```bash
python run_experiments.py --output results/coded_ip_fedmf_sweep.csv
```

The CSV contains coded recovery error, plaintext-online baseline error,
communication rounds, redundancy ratio, and aggregation latency. It uses
synthetic matrix-factorisation update vectors; replace `updates` in
`run_sweep` with MovieLens client gradients for the full dataset experiment.

## Matrix-factorisation training loop

The MF experiment creates client-local ratings, computes item-factor SGD
gradients, and evaluates RMSE after every communication round:

```bash
python run_mf_experiment.py --rounds 8 --dropout-rate 0.2 \
  --output results/coded_ip_fedmf_rmse.csv
```

The data generator can be replaced by a MovieLens parser while retaining the
same client gradient and coded aggregation interfaces.

## Full comparison

Use a local MovieLens `ratings.dat`/`ratings.csv` file, or omit `--ratings` for
the deterministic synthetic fallback:

```bash
python full_experiment.py --rounds 20 --dropout-rate 0.2 \
  --output results/full_fedmf_comparison.csv
python full_experiment.py --ratings path/to/ratings.dat \
  --client-num 50 --output results/movielens_fedmf.csv
```

The comparison records RMSE, MAE, recovery error, online client count, and
communication rounds for Coded-IP-FedMF and the plaintext online baseline.
The experiment encodes all updates centrally and then simulates dropouts at
aggregation time; it is a protocol reference, not a client-upload network
benchmark.

Run all four dropout rates and generate a summary/plot:

```bash
python run_full_sweep.py --rounds 20 --repeats 3 \
  --output results/full_fedmf_sweep.csv
python summarize_results.py results/full_fedmf_sweep.csv \
  --output results/full_fedmf_summary.csv
python plot_results.py results/full_fedmf_sweep.csv \
  --output results/rmse_curves.png
```

Compute repeat statistics and scalability measurements:

```bash
python statistical_report.py results/full_fedmf_sweep.csv \
  --output results/full_fedmf_confidence_intervals.csv
python communication_benchmark.py \
  --output results/communication_benchmark.csv
```

The statistical report includes mean, standard deviation, and 95% confidence
intervals per dropout rate and round. The communication benchmark scans client
count and update dimension while recording encoding/aggregation latency and
the MDS redundancy ratio. It reports value-level payload ratios only; it does
not pretend that toy ciphertext sizes are production network costs.

## Paper claim validation

Run the executable mathematical checks and print an implementation audit:

```bash
python paper_validation.py --client-num 8 --recovery-threshold 5
```

For numerical-conditioning and threat-model diagnostics, run:

```bash
python protocol_validation.py --client-num 8 --recovery-threshold 5
```

The report checks arbitrary MDS dropout subsets, dynamic weighted recovery,
share-upload loss, the reference cosine-score filter, fixed-point error, and
transparent toy SecAgg/DP components. It also labels the claims that cannot be
verified by this repository: IND-CPA/sta-IND security, production SecAgg/DP
security, and a client disappearing before it constructs its source shard.
The `ReferenceInnerProduct` backend remains intentionally insecure and must
not be used as evidence for the paper's cryptographic security theorem.

## Paired accuracy baselines

To compare the same initialization and ratings across centralized, coded,
online-plaintext and noisy local (LDP-style) training:

```bash
python run_baselines.py --rounds 20 --repeats 3 \
  --ldp-noise-multiplier 1.0 \
  --output results/baseline_comparison.csv
```

For a single archive containing all checks, sweeps, baselines, statistics and
communication measurements, run:

```bash
python run_paper_experiments.py --output-dir results/paper --plot
```

The generated manifest records every output path and parameter.  The Paillier
and production-secure aggregation baselines are intentionally not emulated by
plaintext code; those require independently reviewed cryptographic libraries.

The baseline and full-sweep CSV files include `online_client_ids`, so an
accuracy point can be reproduced with the exact arbitrary online subset rather
than an implicit prefix of client IDs.  FedML's `ExperimentRecorder` preserves
legacy CSV headers and records optional dropout/recovery fields for new runs.

## Personalized DMCFE-IP matrix factorisation

`dmcfe_pfedmf.py` is the personalized recommendation experiment for the
paper. Each client updates a private user embedding locally; only the global
item-gradient is aggregated through the optimized, chunked DMCFE-IP interface.
The default backend is still a correctness reference (`backend_secure=false`),
so the output is suitable for convergence and overhead studies, not a
cryptographic security theorem.

Run the deterministic smoke experiment:

```bash
python dmcfe_pfedmf.py --rounds 20 --client-num 12 --dropout-rate 0.2 \
  --output results/dmcfe_pfedmf.csv
```

The output records Recall@K, NDCG@K, clipping/quantization error, online
client IDs, and the number of aggregation chunks. Supply `--ratings` with a
MovieLens `ratings.dat` or `ratings.csv` file for a real-data run.
