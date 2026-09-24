# Task Plan: Isolated toy MCFE overhead probe

## Goal
Measure the repository's existing toy MCFE arithmetic at small dimensions without presenting it as a secure or full-model DMCFE-IP benchmark.

## Actions
- [x] Inspect the optimized/reference and toy MCFE backends, API stages, tests, and current manuscript boundary.
- [x] Implement a standalone CLI that measures setup, key generation, context preparation, encryption, derived-key computation, decryption, correctness, and payload byte counts.
- [x] Add focused tests for the stage accounting, exact weighted-sum result, and output schema.
- [x] Run a bounded multi-seed probe in a new result directory and validate raw/summary outputs.
- [x] Document the command, environment, and strict limitations in the example README and a probe note.
- [x] Add a self-contained run record beside the raw results with descriptive timings, provenance, and claim limits.

## Decisions
- Use the existing `MCFE` primitive at its documented toy parameter size, with fresh labels for each trial and no changes to the 50-round matrix or its source files.
- Describe ciphertext and key sizes as encoded integer payload bytes, not network traffic or production serialization.
- Keep timing observations out of Chapter 4.3's functional-encryption claims; the probe is an engineering artifact for future backend integration.

## Status
Complete: 105 exact trials, 15 setup records, 33 summary rows, a self-contained run record, 13 related tests, syntax compilation, and artifact validation passed.
