# Version notes: paper-submission-v1.0

DOI: 10.5281/zenodo.21758262

This is the paper-submission release for the protocol-first external
validation study.

## Included evidence

- Evaluator frozen before HUST calibration access
- Twenty-cell parser and uncertainty calibration arm
- Fifty-seven-cell one-shot confirmation arm
- Zero confirmation exclusions
- Cell-first inference with random seeds aggregated within cell
- Calibration, interval-width, and EOL-compatibility diagnostics
- Registered implementation amendments and integrity audits
- Independent recomputation from archived cell-level records

## Main findings

The frozen full K5 model did not outperform the target-adaptive local
linear trend. Local linear trend had lower future-SOH RMSE in all 57
confirmation cells.

The full multiscale K5 model did not significantly outperform the
single-scale K5 model.

Nominal simultaneous coverage was obtained only with intervals that
were too wide for operational interpretation.

## Registered implementation corrections

The release retains:

1. CSV round-trip parser amendments registered before confirmation
   access.
2. A post-confirmation Matplotlib keyword compatibility amendment
   translating `labels` to `tick_labels`.

These corrections did not change models, weights, input data,
landmarks, exclusions, metrics, hypotheses, statistical tests,
calibration quantiles, or confirmation outputs.

## Excluded material

No raw HUST, XJTU, BIT, NASA, CALCE, MATR, or HNEI data are included.

The TRAIL-RUL development roadmap is not included.
