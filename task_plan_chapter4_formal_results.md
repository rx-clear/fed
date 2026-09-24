# Task Plan: Integrate formal MNIST comparison into Chapter 4

## Goal
Replace stale preliminary-only claims with the completed 3-method, 3-seed, 50-round evidence while preserving the fixed-point reference and protocol boundaries.

## Actions
- [x] Read the formal manifest, seed and paired summaries, validation record, and current Chapter 4 sections.
- [x] Recompute the manuscript values from the formal CSVs and record allowed claims.
- [x] Update Sections 4.1, 4.2, and 4.4; renumber Section 4.3's figures to follow the new formal-result plots.
- [x] Check every new number and table/figure reference against the artifacts, then review the edited manuscript files.

## Decisions
- Keep the existing table numbers: Table 4-7 for the earlier numerical regression, Table 4-8 for the formal multi-seed comparison, and Table 4-9 for independent dropout recovery.
- Treat `FedRep+FixedPointReference` as a numerical reference with `backend_secure=False`; do not call it a secure DMCFE-IP result.
- Use seed-level final-round values for mean and sample SD. With three seeds, report descriptive paired differences without significance claims.

## Status
Complete: final-round values, paired differences, the legacy 20-round segment, four image links, figure/table numbering, and protocol field semantics were checked against local artifacts.
