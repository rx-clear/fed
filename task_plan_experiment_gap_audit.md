# Task Plan: Experiment Completion Gap Audit

## Goal
Identify the minimum remaining code and runs needed to turn the current prototype outputs into a complete, reproducible thesis experiment.

## Phases
- [x] Phase 1: Read the existing experiment plans, manuscript claims, scripts, and result inventory
- [x] Phase 2: Audit the unified runner, result schema, protocol integration, and statistical outputs
- [x] Phase 3: Run focused read-only verification and classify gaps as P0, P1, or P2
- [x] Phase 4: Deliver an implementation-ready checklist without modifying experiment code

## Key Questions
1. Which experiment claims are already supported by runnable code and saved artifacts?
2. Which missing code blocks publication-level comparison, overhead, or security claims?
3. What is the smallest implementation order that avoids rerunning expensive jobs twice?

## Decisions Made
- Treat the user's request as a diagnostic audit; do not modify experiment implementation.
- Preserve existing planning files and use scoped audit files because `task_plan.md` and `notes.md` already belong to the completed Chapter 4.3 task.

## Errors Encountered
- A read-only PowerShell inspection command failed to parse because `$p:` and Bash-style brace expansion are invalid in this PowerShell context. No command body ran and no files changed; continue with explicit paths and `${p}` interpolation.

## Status
**Complete** - the gap audit and prioritized implementation checklist are ready; experiment implementation code was not modified.
