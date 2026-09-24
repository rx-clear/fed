# Notes: Chapter 3/4 Remediation Evidence

## Confirmed scope

- `chapter3_section1_system_model.md` contains the intended Section 3.1 at lines 1–120, then repeats Sections 3.2–3.4 and Chapter 4.1–4.2 from line 126 onward.
- The duplicated tail contains malformed `\\[$...\\]` display-math delimiters; the retained Section 3.1 also uses that malformed delimiter style.
- The executed configuration value is `chunked_reference`; its backend reports `name = "chunked-reference"` and `is_secure = False`.

## Boundaries

- The edit will not alter the separate source files for Sections 3.2–3.4 or 4.3–4.4.
- The raw experiments, analysis bundle, code, and statistical claims remain unchanged.

## Completed edits

- Restored `chapter3_section1_system_model.md` to the Section 3.1 material only, repaired all retained display-math delimiters, and made the shared-update notation consistently use $\Delta^s_{i,t}$.
- Copied the system diagram into `assets/chapter3/system-model.png`; the new caption identifies it as a target-protocol architecture and states that the current high-dimensional reference backend does not execute its cryptographic stages.
- Renamed executed Chapter 4 variants to “定点参考聚合” and reserved “目标 DMCFE-IP” for the unmeasured audited backend.

## Verification

- The Section 3.1 file has one chapter heading, one Section 3.1 heading, and five Section 3.1 subsections; it contains no later Chapter 3 or Chapter 4 section headings.
- No malformed `\\[$...\\]` delimiters, absolute image paths, `DMCFE-FedAvg`, or `FedRep+DMCFE` labels remain in the reviewed section files.
- `assets/chapter3/system-model.png` exists in the repository.

## Errors encountered

- An initial verification command passed PowerShell wildcard paths directly to `rg`, which Windows rejected. The corrected command supplied an explicit file list and completed successfully.
