# Task Plan: Remediate Chapter 3/4 Priority Findings

## Goal

Remove the duplicated Chapter 3/4 content from the Section 3.1 source, repair the retained Section 3.1 manuscript integration, and distinguish completed reference-aggregation experiments from the unmeasured target DMCFE-IP backend.

## Phases

- [x] Phase 1: Lock the audit-backed scope and preserve separate chapter-section source files.
- [x] Phase 2: Repair the standalone Section 3.1 file and make its system figure portable.
- [x] Phase 3: Revise Chapter 4 terminology and experimental-scope wording.
- [x] Phase 4: Verify headings, math delimiters, image path, terminology, and diff scope.

## Decisions Made

- Do not add citations in this pass because no verified bibliography workflow or paper template is present in the workspace.
- Use “定点参考聚合” for the executed `chunked_reference` configuration and reserve “DMCFE-IP” for the target protocol or a future audited backend.
- Keep the existing system-model image but copy it into the repository as a relative asset instead of retaining an absolute local link.

## Status

**Complete** — the standalone Section 3.1 source, protocol terminology, and audit checks are complete.
