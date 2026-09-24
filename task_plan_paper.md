# Task Plan: Chapters 3-4 Paper Draft

## Goal
Draft Chinese, publication-ready method and experiment sections for the paper "Privacy-Preserving Personalized Federated Learning Based on Functional Encryption".

## Phases
- [x] Inspect repository design, implementation boundaries, and available evidence
- [x] Define notation and system assumptions
- [x] Draft Section 3.1 with equations and protocol flow
- [x] Check claim-evidence alignment and deliver the draft
- [x] Inspect the FedRep, fixed-point aggregation, DMCFE-IP, signature, and dropout-recovery implementations
- [x] Draft Section 3.2 personalized federated learning and functional-encryption aggregation method
- [x] Check equations, correctness claims, and implementation boundaries
- [x] Draft Section 3.3 security analysis and privacy discussion
- [x] Check Section 3.3 claim-evidence alignment, notation, and implementation boundaries
- [x] Draft Section 3.4 chapter summary
- [x] Check that Section 3.4 summarizes only supported claims and introduces no new results
- [x] Inspect experiment configs, scripts, result schemas, and recorded environment evidence for Section 4.1
- [x] Define research questions, comparison groups, parameters, and evaluation metrics from repository evidence
- [x] Draft Section 4.1 experimental setup
- [x] Verify every concrete setting against code or mark it as pending confirmation
- [x] Inventory and validate the available Section 4.2 result artifacts
- [x] Compute defensible descriptive statistics from comparable runs
- [x] Draft Section 4.2 experimental results and analysis
- [x] Verify every numerical statement against its source artifact

## Constraints
- Do not invent experimental results or citations.
- Distinguish the intended FE protocol from the current non-secure reference backend.
- Keep notation consistent with FedRep and weighted DMCFE aggregation in the code.

## Errors Encountered
- The first Section 3.3 write interpreted inline LaTeX backslashes as control characters. The new file was rewritten with literal backslashes before validation; Sections 3.1 and 3.2 were untouched.
- The first PowerShell settings check placed a pipeline directly after a `foreach` block and failed with `Empty pipe element`; assigning the loop output to `$out` before piping resolved the syntax error. No files were changed by the failed command.
- The first model-parameter check tried to import `CNN_DropOut` from the example entry script, but that script creates the configured CNN through `fedml.model.create`. Importing `CNN_DropOut` from `fedml.model.cv.cnn` produced the expected counts. No files were changed by the failed check.
- The inline Python document check first used conflicting PowerShell and Python quote delimiters, then lost its embedded double quotes during Windows native-argument parsing; both attempts failed with `SyntaxError` and changed no files. The assertions were moved to PowerShell with .NET strict UTF-8 decoding to remove the quoting ambiguity.
- The first Section 4.2 CSV inventory repeated the same PowerShell `foreach`-pipeline syntax mistake. Assigning the loop output to `$out` produced the intended inventory; the failed read-only command changed no files.
- The first Section 4.2 numeric-string verifier searched for Python `e-05` notation while the manuscript uses equivalent LaTeX `\\times10^{-5}` notation. The six reported misses were verifier-format mismatches, not numerical discrepancies; the check was updated to generate the manuscript notation.

## Status
**Complete for Section 4.2** - The results section reports only validated descriptive evidence, explicitly blocks unsupported multi-method and security claims, and passes numerical, structural, and focused test verification.
