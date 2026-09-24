# DMCFE-IP + Personalized Federated Learning Design

This document records the implementation contract for the research prototype.

## Scope

The code will demonstrate protocol-compatible weighted inner-product aggregation and personalized model partitioning. The current backend remains a non-secure reference backend; security claims require a separate production DMCFE implementation and threat-model evaluation.

## Contributions Represented in Code

1. Chunked/vectorized fixed-point aggregation with clipping and error diagnostics.
2. Explicit shared/personalized parameter partitioning for FedRep-style training.
3. An implicit-feedback matrix-factorization example with private user vectors and globally shared item vectors.

The aggregation layer accepts either the fast chunked reference backend or the
small cached toy-MCFE backend for protocol-level checks. Both are marked
non-secure until a reviewed production DMCFE-IP implementation is supplied.

## Required Evaluation

- FedAvg, FedRep, and DMCFE-PFed variants.
- At least three random seeds and multiple non-IID/dropout settings.
- Recall@K and NDCG@K for recommendation, plus communication and aggregation overhead.
- Equivalence error and clipping rate for every run.
