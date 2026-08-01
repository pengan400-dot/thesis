# DRAFT: v0.2.1 BIT structural-feasibility amendment

Status: DRAFT. Must be independently reviewed, committed, released, archived, and registered before any BIT model output.

## Classification

This is a **post-BIT-structure-only, pre-model-evaluation amendment**. It is triggered solely by archive completeness and physical-cycle landmark feasibility. No BIT capacity values, SOH values, EOL events, predictions, RMSE values, model rankings, or error plots were viewed.

## Structural finding

The supplied version-3 archive contains 73 cell directories: 55 arbitrary-use and 18 fixed-profile. Four fixed-profile directories listed by official metadata are absent (#10, #13, #16, #19), and #2 contains only its first20 workbook.

The structure report identifies local `Cycle_Index` ranges of 1–21 in first20 workbooks and 1–101 in later workbooks. All 145 XLSX members are marked as not representing physical cycle 190. Therefore the original cycle-190 primary landmark is not evaluable.

## Amendment

1. The primary BIT landmark is changed from physical cycle 190 to physical cycle 70.
2. Cycle 70 was already prespecified in v0.2.0 as an exploratory cutoff; it is not selected using model performance.
3. Cycles 130 and 190 are reported as structurally unavailable and receive no model evaluation.
4. The arbitrary-use cohort remains primary.
5. The fixed-profile cohort remains key secondary.
6. #2 is structurally excluded because it lacks the later workbook.
7. #10, #13, #16, and #19 are recorded as absent from the supplied archive.
8. Model weights, architecture, scalers, conformal quantiles, SOH formula, EOL threshold, future-information prohibitions, H1→H2 sequence, cell-first weighting, sign-flip test, bootstrap settings, and non-pooling rule remain unchanged.
9. The parser source must be written and hashed but not executed on BIT capacity values before the new immutable freeze.
10. The amended freeze must be recorded with a new commit, GitHub release, Zenodo version DOI, and immutable OSF registration before any BIT model output.

## Permitted wording

“The BIT protocol was structurally amended after archive-schema inspection but before any BIT model evaluation because the prespecified cycle-190 landmark was not representable in the supplied archive. The amended primary landmark, cycle 70, was selected from the previously specified cutoff set on feasibility grounds only.”
