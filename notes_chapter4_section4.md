# Notes: Chapter 4 Section 4

## Evidence ledger

- RQ1: one 50-round FedRep+DMCFE run improved from 27.7167% to 78.5104% test accuracy, but lacks a matched plaintext FedRep control and multi-seed comparison; no cross-method superiority claim.
- RQ2: 20 paired points and one 50-round run support fixed-point numerical consistency at roughly $10^{-5}$ coordinate error for the configured reference path.
- RQ3: a constructed threshold-recovery scenario succeeds with two responses at threshold two and aborts with one response; this is not an end-to-end high-dimensional recovery claim.
- RQ4: a 300-row synthetic-vector sensitivity study reports only CPU `chunked-reference` timing. It excludes encryption, functional-key work, serialization, network transfer, and peak memory.

## Writing constraints

- No new citation is required because this section summarizes local experimental artifacts.
- Use “reference backend” consistently and do not call its timing functional-encryption overhead.
