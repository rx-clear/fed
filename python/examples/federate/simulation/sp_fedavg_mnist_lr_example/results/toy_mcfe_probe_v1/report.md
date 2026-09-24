# Toy MCFE small-dimension probe — run record

## Question and result

Can this repository's **toy** MCFE primitive recover a three-client, integer-weighted sum at small vector dimensions, and what local Python timings and encoded integer payload sizes does this isolated path produce?

All 105 timed trials recovered every weighted-sum coordinate exactly (`max_abs_error = 0`). The mean of five seed-level median round times was 0.2270 ms at dimension 1, 0.4845 ms at dimension 8, and 1.1320 ms at dimension 32. These numbers describe this local toy probe only. They are **not** measurements of a secure DMCFE-IP backend, FedML communication, or full-model training.

## Protocol and provenance

- Run: 2026-09-18, Windows, Python 3.9.25, SymPy 1.14.0. The exact platform string and SHA-256 hashes of the MCFE primitive and benchmark script are in [environment.json](environment.json).
- Primitive: `fedml.simulation.sp.fedavg.dmcfe.mcfe.MCFE`, with a 256-bit toy modulus. Three clients have weights `[32, 39, 46]`; each message coordinate is an integer in `[-256, 256]`.
- Dimensions: 1, 8, 32. For each dimension, seeds 0–4 generate messages. Each seed/dimension pair has fresh randomized modulus and secret keys, two excluded warmups, and seven timed trials. A trial uses a fresh label shared by its coordinates.
- Timing: `time.perf_counter()` around local Python operations. `round_seconds` starts before encryption-context preparation and ends after exact-result checking and payload encoding. One-time modulus setup and key generation are recorded separately.
- Summary: the median of seven trials is one timing observation per seed. Reported `mean ± SD` is the mean and sample standard deviation of those five seed medians (`n = 5`), converted from seconds to milliseconds. These are descriptive technical measurements; no significance test or confidence interval is claimed.
- Command, from the example directory: `python benchmark_toy_mcfe.py --dimensions 1,8,32 --seeds 0,1,2,3,4 --clients 3 --repeats 7 --warmups 2 --modulus-bits 256 --output-dir results/toy_mcfe_probe_v1`. Use a new directory to rerun because the script refuses to replace a nonempty output directory.

## Observations

| Coordinates | Total local trial (ms) | Encryption context (ms) | Encrypt (ms) | Decrypt (ms) | Ciphertext integer payload (bytes) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.2270 ± 0.0442 | 0.1148 ± 0.0430 | 0.00486 ± 0.00040 | 0.02278 ± 0.00107 | 192 |
| 8 | 0.4845 ± 0.0318 | 0.1709 ± 0.0382 | 0.02972 ± 0.00145 | 0.19070 ± 0.00791 | 1,536 |
| 32 | 1.1320 ± 0.1101 | 0.1834 ± 0.0122 | 0.10854 ± 0.00582 | 0.72836 ± 0.07128 | 6,144 |

The total also includes derived-key computation, decryption-context preparation, exact-result checking, and integer-payload encoding. Phase medians are summarized independently, so their displayed means need not add up to the total. The full stage timings, including setup and key generation, are in [summary.csv](summary.csv). Ciphertext byte counts use this probe's fixed-width unsigned encoding and equal 3 clients × 64 bytes × coordinates. The derived-key integer payload averages 3.8, 3.8, and 4.0 bytes respectively; it depends on the randomized key value.

## Interpretation boundary

The current primitive and toy backend are marked non-secure. The primitive uses reduced parameters and one label across coordinates; its key distribution and security have not been reviewed for deployment. Payload sizes exclude metadata, framing, key exchange, and network transport. No network, memory, full-model, or training-end-to-end cost was measured. Modulus and keys are freshly random, so another run will not reproduce timing values exactly. Extrapolating from 32 coordinates to the formal FedRep run's 1,198,592 shared coordinates would be unjustified.

This record remains separate from the formal 50-round comparison and Chapter 4.3 overhead claims. A defensible secure-overhead experiment first needs a reviewed DMCFE-IP backend implementing the existing weighted-inner-product interface, explicit secure parameters and serialization, then stage and end-to-end measurements on the intended model and hardware.

## Verification

Checked the complete 3-dimension × 5-seed × 7-trial grid, all 105 zero-error results, the 15 setup records, and the 33 summary rows. Recomputed the displayed timing and byte values from `summary.csv`; the benchmark source SHA-256 matches `environment.json`. With the repository's `python` directory on `PYTHONPATH`, `test_toy_mcfe_probe.py` passed 3 tests, `test_optimized_dmcfe_ip.py` passed 5, and `test_dmcfe_paper_features.py` passed 5 using the local `fedmlnew` Python environment.

## Audit inputs

- [trials.csv](trials.csv): 105 timed trials with per-stage durations, byte counts, and exact-result errors.
- [setup.csv](setup.csv): 15 modulus setup/key-generation records.
- [summary.csv](summary.csv): 33 dimension/metric summaries.
- [environment.json](environment.json): run settings, software versions, and source hashes.
