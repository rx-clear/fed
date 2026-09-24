# Notes: Chapter 4.3/4.4 Revision Evidence

## Audit evidence retained

- Raw sensitivity artifact: 300 complete `chunked-reference` records, covering 3 scales, 4 clipping bounds, 5 chunk sizes, and seeds 0–4.
- The timing unit remains a per-seed median of seven post-warmup calls in one Python process; it is not an independent cryptographic or end-to-end timing measurement.
- For the 32,768-coordinate probe, chunk counts are 128, 32, 8, 2, and 1 for chunk sizes 256, 1,024, 4,096, 16,384, and 65,536 respectively.

## Revision boundaries

- Remove informal statistical language such as “显著增加/显著扩大”; retain the explicit statement that no significance test is performed.
- Do not infer functional-encryption latency, communication cost, memory use, end-to-end efficiency, or cross-method superiority.
- Correct malformed LaTeX notation in the Section 4.3 protocol before reusing the text.

## Execution and verification

- `python/tests/analyze_dmcfe_sensitivity_ablation.py` passed `py_compile` and regenerated two one-page vector PDFs plus two 600-DPI PNG previews.
- Both PDFs were rendered at 180 DPI and visually inspected. The clipping-bound labels in Figure 4-1 are separate and legible; neither figure has clipped labels or overlapping elements.
- Text checks confirmed the corrected `\\in` and set notation, no remaining “显著增加/显著扩大” wording, both Table 4-10 and Table 4-11 labels, and the 300-row raw-record count.

## Errors encountered

- An initial patch for the generated figure catalogue escaped Markdown backticks incorrectly and did not apply. A targeted follow-up patch applied the intended catalogue labels without altering the analysis logic.
