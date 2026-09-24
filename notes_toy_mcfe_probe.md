# Notes: Toy MCFE overhead probe

## Findings
- `OptimizedDMCFEIPAggregator` defaults to `ChunkedReferenceBackend`, which performs no encryption.
- `ToyMCFEBackend` invokes `MCFE.derive_key`, `MCFE.encrypt`, and `MCFE.decrypt` coordinate by coordinate; it has `is_secure=False` and is intended for small correctness checks.
- `MCFE` uses reduced toy modulus parameters and a simple signed-integer key distribution. It does not constitute an audited production DMCFE-IP backend.
- The 50-round formal reference matrix has 1,198,592 shared coordinates. It must remain untouched and cannot be extrapolated from this small probe.

## Measurement boundary
- A trial uses one fresh label and a fixed set of generated integer messages and positive sample weights.
- Stage times are local Python wall-clock durations. Payload counts include fixed-width ciphertext integers and a signed derived-key integer only; they exclude metadata, transport framing, network time, and key exchange.
- Repeats under one seed are technical timing repetitions; report seed-level medians before any cross-seed summary.

## Completed run
- Command: `benchmark_toy_mcfe.py --dimensions 1,8,32 --seeds 0,1,2,3,4 --clients 3 --repeats 7 --warmups 2 --modulus-bits 256 --output-dir results/toy_mcfe_probe_v1`.
- Output directory: `python/examples/federate/simulation/sp_fedavg_mnist_lr_example/results/toy_mcfe_probe_v1/`.
- `trials.csv`: 105 measured repetitions, all with exact weighted-sum recovery. `setup.csv`: 15 one-time setup/key-generation rows. `summary.csv`: 33 dimension/metric rows. `environment.json` records the source hash and claim boundary.
- Mean of per-seed median local round times: 1 coordinate `0.2270 ms`, 8 coordinates `0.4845 ms`, 32 coordinates `1.1320 ms` (5 seeds each). These are descriptive toy-probe timings, not production secure overhead.
- `round_seconds` includes the exact-result check and payload encoding after decryption; separate phase columns isolate context preparation, encryption, derived-key computation, and decryption.
- Seeds fix generated integer messages, while toy modulus setup and secret keys use fresh randomness; runtime values are not deterministic across invocations.
- Ciphertext integer payloads for 3 clients: 192, 1,536, and 6,144 bytes at dimensions 1, 8, and 32, respectively. Derived-key integer payloads vary by key value. No transport/framing, real serialization, or network bytes were measured.
- Validation checked the complete seed/dimension/trial grid, all zero errors, payload-width formula, source SHA-256, and seed-median summary calculation. Three focused unit tests and syntax compilation passed.
- The run record at `results/toy_mcfe_probe_v1/report.md` gives the primary question, five-seed descriptive table, source artifacts, and the boundary before any paper or secure-overhead claim.
