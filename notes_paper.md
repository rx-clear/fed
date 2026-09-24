# Notes: Chapter 3 Section 1 System Model

## Repository Evidence
- The implementation uses single-process FedRep with shared representation parameters and client-specific prediction heads.
- The optimized DMCFE path aggregates floating-point shared deltas after clipping and fixed-point quantization; integer buffers are copied only when consistent.
- Aggregation weights are local sample counts and can be authenticated with Ed25519 round-bound proposals.
- The implementation supports dropout recovery parameters and records online clients, recovery threshold, clipping fraction, and quantization error.
- `chunked_reference` and `toy_mcfe` are explicitly marked non-secure reference backends.

## System-Model Decisions
- Entities: a coordinating server, K clients, and a functional-key authority/trusted setup component in the target protocol.
- Data remain on clients and are non-IID across clients.
- Model state is partitioned as shared representation theta^s and personalized head theta_i^p.
- The server should learn only the weighted aggregate of shared updates, not each individual update or personalized head.
- The current code validates numerical/protocol behavior; it does not establish a cryptographic security theorem.

## Draft Output
- `chapter3_section1_system_model.md` contains the Chinese draft with system entities, non-IID data, shared/personalized model decomposition, round protocol, FE aggregation abstraction, dropout threshold, weight signatures, threat model, security goals, and implementation boundary.

## Section 3.2 Method Evidence
- `FedRepClient.train` alternates two phases: train the personalized head with the representation frozen, then train the representation with the head frozen.
- `FedRepAPI` aggregates only `shared_state_keys` and persists personalized state by client ID.
- `OptimizedDMCFEIPAggregator` clips finite floating-point updates, applies integer fixed-point encoding, computes a sample-weighted inner product, and reports clipping and numerical errors.
- Round-bound signed weight proposals cover the round index, participant set, client ID, and positive sample count.
- The complete dropout prototype distinguishes participant/key-sharing set `U2`, ciphertext-upload set `U3`, and recovery-response set `U4`; early and late dropout secrets are reconstructed only when the threshold is met.
- The optimized full-dimensional reference backend and complete dropout protocol are currently separate validation paths; the manuscript must not describe their composition as a production deployment.

## Section 3.2 Claim-Evidence Map
- Claim: personalized heads remain client-local. Evidence: `FedRepAPI` partitions state and `FedRepClient` persists heads by client ID. Status: supported by implementation and tests.
- Claim: full shared floating-point state can be aggregated after clipping and fixed-point encoding. Evidence: `OptimizedDMCFEIPAggregator` and the GPU smoke runs. Status: supported for the reference backend.
- Claim: signed weights bind the round and participant set. Evidence: `weight_signatures.py` and paper-feature tests. Status: supported by implementation.
- Claim: early and late dropout masks can be recovered at a threshold. Evidence: `real_delta_smoke.py` and dropout tests. Status: supported in the independent protocol-validation path.
- Claim: the server learns only the authorized weighted function. Evidence required: a production DMCFE-IP construction and formal security proof. Status: conditional, explicitly bounded in the draft.
- Claim: runtime and communication efficiency. Evidence required: multi-seed timing, ciphertext-size, and recovery-traffic experiments. Status: not claimed; only complexity formulas are provided.

## Section 3.2 Draft Output
- `chapter3_section2_pfl_fe_method.md` contains nine subsections, 22 display equations, Algorithm 3-1, a quantization error bound, dropout recovery, complexity analysis, and an implementation-boundary statement.
- Section 3.1 was adjusted so the trusted initializer publishes public parameters and registers identities, while clients independently hold keys and combine partial functional keys.

## Section 3.3 Claim-Evidence Map
- Claim: personalized parameters are not exposed through server aggregation. Evidence: `FedRepClient` separates the two training phases; `FedRepAPI` aggregates only `shared_state_keys` and persists client heads. Status: supported by implementation and tests, with the boundary that shared representations may still leak information.
- Claim: a secure DMCFE-IP instance can limit the server to the authorized weighted sum. Evidence required: concrete construction, security definition, parameters, and proof. Status: conditional; not established by `chunked_reference` or `toy_mcfe`.
- Claim: fixed-point aggregation has bounded rounding error. Evidence: vectorized clipping/encoding in `OptimizedDMCFEIPAggregator`, error metrics, and numerical tests. Status: supported for the reference aggregation path; clipping bias must be reported separately.
- Claim: signed sample weights resist tampering and cross-round replay. Evidence: `dmcfe/weight_signatures.py` binds round, participant set, client ID, and positive sample count. Status: supported for message integrity; signatures do not detect malicious model updates.
- Claim: dropout recovery is threshold-gated. Evidence: independent dropout smoke tests distinguish `U2`, `U3`, and `U4`, reconstruct early/late dropout material, and abort below threshold. Status: supported for protocol behavior; not a formal anti-collusion proof.
- Claim: the complete high-dimensional FE aggregation and dropout recovery are production-secure as one composed system. Evidence: unavailable because the current paths are separate and reference backends are explicitly non-secure. Status: must not be claimed.

## Section 3.3 Draft Output
- `chapter3_section3_security_privacy_analysis.md` contains eight subsections covering assumptions, personalization isolation, functional-encryption confidentiality, fixed-point correctness, weight integrity, dropout recovery, residual privacy risks, and implementation limits.

## Section 3.4 Draft Output
- `chapter3_section4_summary.md` summarizes the system model, personalized DMCFE-IP aggregation, security boundaries, and the experiment dimensions that should be evaluated next.
- The summary introduces no new experimental result, citation, or unconditional cryptographic claim.

## Section 4.1 Experiment-Setup Evidence
- Dataset: the MNIST loader reads the repository's pre-partitioned JSON benchmark rather than running a Dirichlet partition. The dataset documentation specifies 1000 clients, 61,664 training samples, 7,371 test samples, two digit classes per client, and a power-law client sample-count distribution.
- Configuration caveat: partition_method=hetero and partition_alpha=0.5 remain in YAML, but load_partition_data_mnist does not consume either value. The manuscript must not describe alpha=0.5 as the operative heterogeneity mechanism.
- Model: CNN_DropOut(True) has 1,199,882 parameters. The personalized linear_2 head has 1,290 parameters; the shared representation has 1,198,592 parameters.
- Training: 50 communication rounds, 3 sampled clients per round, batch size 10, SGD, learning rate 0.03, one head epoch and one representation epoch. Cross-entropy is used. The SGD branch does not pass the configured weight_decay=0.001, so effective weight decay is zero.
- Main comparison roles: FedAvg tests the global baseline; FedRep isolates personalization; FedRep+DMCFE tests personalization plus fixed-point functional aggregation. BatchCrypt and pairwise Masking exist as FedAvg-based protocol prototypes and should be interpreted separately rather than as matched personalized baselines.
- DMCFE settings: optimized full shared-state mode, scale 16,384, clip bound 8.0, chunk size 65,536, signed sample weights enabled, configured recovery threshold 2, and no injected dropout in the default run.
- Backend boundary: chunked_reference performs no encryption and is explicitly insecure. Modulus bits and validation dimension are not operative for this optimized reference path; cryptographic timing and communication claims require a real backend.
- Environment verified on 2026-09-07: Windows 11 build 22631, AMD Ryzen 7 5800H (8 cores/16 threads), 15.9 GiB RAM, RTX 3060 Laptop GPU (6,144 MiB), driver 576.52, Python 3.9.25, FedML 0.8.30, PyTorch 2.8.0+cu129, and CUDA 12.9.
- Existing CSV metrics include train/test accuracy and loss, DMCFE maximum/mean error, online-client fields, recovery threshold/error, and verified weight-signature count. Clipping fraction, phase timings, and serialized communication bytes are not persisted in the current CSV schema.
- Metric caveat: in the optimized training path, dropout_rate is computed as one minus sampled clients divided by total clients (0.997 for 3 of 1000), so it is a non-participation rate rather than an injected protocol-dropout rate. The same path records recovery_threshold as the sampled-client count. Dropout analysis must use the independent U2/U3/U4 recovery path.
- Reproducibility protocol: use seeds 0, 1, and 2 for final comparison and report mean plus standard deviation. Existing completed MNIST evidence is only a seed-0 integration run; it is not sufficient for multi-seed statistical claims.

## Section 4.1 Draft Output
- `chapter4_section1_experimental_setup.md` contains eight subsections covering research questions, the pre-partitioned non-IID MNIST benchmark, CNN and personalized parameter partitioning, comparison methods, effective training settings, DMCFE-IP and dropout parameters, the verified execution environment, and evaluation/statistical procedures.
- The draft contains six tables and three display-equation groups. It distinguishes configured values from effective runtime behavior, especially for `partition_alpha`, `weight_decay`, `dmcfe_modulus_bits`, `dmcfe_validation_dim`, and the recorded `dropout_rate` field.
- Twenty-two concrete configuration and implementation markers were checked automatically. Model parameter counts and package versions were also recomputed with `D:\anaconda\envs\fedmlnew\python.exe`.
- Thirteen focused tests cover FedRep parameter partitioning and persistence, optimized fixed-point aggregation, clipping diagnostics, toy-backend equivalence, BatchCrypt and masking numerical behavior, dropout recovery, packing, and round-bound weight signatures.
- No citation was added in this section. Dataset claims come from the repository MNIST documentation and loader, while method and runtime claims are limited to code and generated artifacts already present in the project.

## Section 4.2 Result-Evidence Audit
- The formal comparison output `results/comparison/comparison_metrics.csv` does not exist. The manifest contains only `HybridAlpha` and `CryptoFE` entries marked `not_implemented`; therefore no three-method, three-seed accuracy comparison or significance test is currently valid.
- `fedavg_baseline.csv` and the first complete 20-round segment of `dmcfe_fedavg.csv` each contain two identical records per communication round. Exact duplicates were collapsed by round, leaving 20 paired round-level observations. Communication rounds are a trajectory, not independent experimental replicates, so they are used only for descriptive numerical comparison.
- At round 19, the baseline and DMCFE-FedAvg test accuracies are both 0.6567629901. Their final test-loss difference is `2.32557577e-06`. Across 20 rounds, the maximum and mean absolute accuracy differences are `5.42667209e-04` and `1.15316782e-04`, respectively; the maximum test-loss difference is `1.38896531e-05`.
- Within the 20-round DMCFE-FedAvg run, the largest recorded coordinate-wise aggregation error is `3.02142079e-05`; the largest recorded mean absolute aggregation error is `1.31626996e-05`, and its round average is `9.61289851e-06`.
- `dmcfe_fedrep_50_local.csv` contains evaluation records only at rounds 0 and 49. Test accuracy rises from 0.2771672772 to 0.7851037851, an increase of 50.7937 percentage points. Test loss falls from 2.2067349103 to 0.8523546953, a reduction of 1.3543802150 or 61.3748% relative to round 0.
- The same 50-round endpoint file reports a maximum coordinate-wise aggregation error of `3.02274657e-05` across its two checkpoints and a maximum mean absolute error of `1.05963372e-05`. Each recorded checkpoint has three online clients and three verified sample-weight signatures.
- The independent dropout smoke scenario samples clients 10, 11, 12, and 13; client 13 drops before encryption and client 12 drops before recovery. With threshold 2, the active encrypted set is 10, 11, and 12 and the recovered weighted integer sum has maximum and mean error 0. With only one recovery responder under threshold 2, the protocol raises the expected threshold-not-met abort.

## Section 4.2 Claim Ledger
- Claim: fixed-point DMCFE aggregation tracks plaintext FedAvg closely in the available 20-round paired trajectory. Evidence: paired deduplicated CSV rows and recorded aggregation errors. Allowed wording: numerical consistency under the recorded configuration. Forbidden wording: cryptographic security, statistical equivalence, or universal accuracy preservation.
- Claim: the 50-round FedRep+DMCFE run converges substantially relative to its own initial checkpoint. Evidence: round-0 and round-49 accuracy/loss values. Allowed wording: within-run improvement. Forbidden wording: superiority over FedAvg or plain FedRep, statistical significance, or robustness across seeds.
- Claim: the recovery prototype handles one early and one late dropout and aborts below threshold. Evidence: deterministic smoke execution and focused tests. Allowed wording: protocol-path correctness for the constructed case. Forbidden wording: production availability, arbitrary-dropout tolerance, or formal anti-collusion security.
- Claim: signed sample weights are verified for all three selected clients at the recorded checkpoints. Evidence: `dmcfe_weight_signatures_verified=3`. Allowed wording: all submitted proposals in those checkpoints passed verification. Forbidden wording: malicious-update detection or Byzantine robustness.

## Section 4.2 Draft Output
- `chapter4_section2_experimental_results_analysis.md` contains five subsections and three tables: evidence screening, 20-round FedAvg numerical consistency, the seed-0 FedRep+DMCFE endpoint comparison, dropout and signature validation, and an RQ-by-RQ evidence boundary.
- Nineteen formatted numerical statements were recomputed from `fedavg_baseline.csv`, `dmcfe_fedavg.csv`, and `dmcfe_fedrep_50_local.csv` and matched against the manuscript text.
- The Markdown structure check found 78 content lines, five Section 4.2 subsections, three tables, balanced inline math delimiters, no control characters, and no citation or result placeholders.
- Focused verification completed with `13 passed` for FedRep state partitioning and persistence, optimized aggregation, quantization and clipping, signature binding, packing, protocol dropout recovery, BatchCrypt, and pairwise masking. Seven warnings come from third-party dependency deprecations.
- No source code, YAML configuration, CSV result, or plot was modified while drafting Section 4.2. The formal three-method, three-seed experiment remains the smallest missing evidence needed before making comparative accuracy or significance claims.
