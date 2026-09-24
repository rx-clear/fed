# Notes: Personalized DMCFE-IP and Recommendation Example

## Current Findings
- The active FedRep example uses a two-layer MLP (`linear_1` representation and `linear_2` local head), despite the config naming the model CNN.
- The existing DMCFE path checks only a configurable prefix of the flattened update and keeps the remaining suffix on plaintext FedAvg.
- `ReferenceInnerProductBackend` explicitly reports `is_secure = False`; it is suitable for equivalence tests, not security claims.
- Existing FedRep infrastructure already preserves personalized state per client and exposes shared parameter keys.

## Design Direction
- Add a vectorized, chunked weighted fixed-point aggregator behind a backend protocol.
- Make full shared-state coverage the default for the new path and record quantization/clipping diagnostics.
- Add a compact implicit-feedback matrix-factorization example using client-local user embeddings and a globally aggregated item embedding.
- Report Recall@K, NDCG@K, communication size, and aggregation diagnostics.

## Implemented
- `optimized_dmcfe_ip.py` adds a chunked reference backend and complete shared-state fixed-point aggregation with clipping/quantization diagnostics.
- `FedAvgAPI` accepts `dmcfe_aggregation_mode: optimized`; the legacy prefix-only smoke path remains selectable as `smoke`.
- `dmcfe_pfedmf.py` keeps user embeddings local and aggregates item gradients, with Recall@K/NDCG@K and dropout metrics.
- `ToyMCFEBackend` exercises the existing cached MCFE encrypt/derive/decrypt API for small protocol checks; it is explicitly non-secure.
- Direct PyTorch smoke execution completed for the optimized aggregator and the two-round recommender experiment.

## Verification Limits
- `compileall` passes for all modified Python files.
- Standard unittest collection is blocked in the available environments by missing FedML runtime dependencies (`GPUtil`, `click`, and related packages); no dependency files were changed to hide this.
