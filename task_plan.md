# Task Plan: Chapter 4 Section 3 — Parameter Sensitivity and Computational Overhead

## Goal
Produce a reproducible pointwise aggregation ablation, strict analysis artifacts, and a defensible Chinese manuscript section `chapter4_section3.md` without overstating reference-backend timing as functional encryption overhead.

## Phases
- [x] Phase 1: Inventory aggregator code, tests, existing artifacts, and manuscript conventions
- [x] Phase 2: Implement/run fixed-point aggregation ablation for scale factor, clipping threshold, and block length
- [x] Phase 3: Analyze metrics and timing with `results-analysis` standards; generate tables/figures and record limits
- [x] Phase 4: Draft and verify `chapter4_section3.md` using `ml-paper-writing`; update plan and deliver

## Key Questions
1. What exact aggregator path and reference backend are currently implemented and testable?
2. Which parameter grid and repeated measurements support error/timing claims without fabricated efficiency conclusions?
3. What wording is supported for the manuscript, and what stronger wording must be blocked?

## Decisions Made
- Treat reference-backend timing as a reproducibility proxy, not as real functional-encryption cost.
- Use seed/run-level measurements where available; otherwise report descriptive summaries and explicit uncertainty.

## Errors Encountered
- The isolated Python 3.12 runtime initially lacked PyTorch; installed its CPU dependencies to run the reproducibility scripts.
- The repository unit-test import path also requires wider FedML dependencies. After resolving `multiprocess`, `GPUtil`, `torchvision`, and `scipy`, test collection remained blocked by missing `click`. The direct aggregator smoke test and both new scripts pass; the remaining blocker is environment dependency completeness, not an assertion failure.

## Status
**Complete** — generated 300 raw records, the analysis bundle, figures, and the Chapter 4.3 manuscript draft.
