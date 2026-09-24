# Figure Catalog: Fixed-Point Aggregation Sensitivity

## figure-01-error-sensitivity.pdf / figure-01-error-sensitivity.png

- Purpose: isolate numerical sensitivity to scale factor and clipping bound while holding the other two parameters fixed.
- Data source: five seed-level rows per point from `dmcfe_sensitivity_ablation.csv`.
- File roles: the PDF is the vector manuscript figure; the 600-DPI PNG is a Markdown preview of the same plot.
- Caption requirements: state the fixed settings, error reference (unclipped floating-point weighted average), and that error bars are sample SD over five synthetic update sets.
- Key observation: scale affects rounding resolution under a loose clipping bound; smaller clipping bounds produce larger error relative to the unclipped reference.
- Interpretation checklist: distinguish clipping bias from quantization error; do not generalize to training accuracy.

## figure-02-reference-timing.pdf / figure-02-reference-timing.png

- Purpose: show the CPU execution-time sensitivity of the non-secure reference implementation and the clipping incidence under the same probe.
- Data source: timing medians from seven post-warmup calls per seed, summarized over five seeds.
- File roles: the PDF is the vector manuscript figure; the 600-DPI PNG is a Markdown preview of the same plot.
- Caption requirements: state that error bars are sample SD and timing is not functional-encryption cost.
- Key observation: chunk size changes reference-loop time; clipping incidence decreases as the clipping bound grows.
- Interpretation checklist: do not infer encryption, communication, memory, or deployed-system overhead.
