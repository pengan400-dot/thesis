# HUST final one-shot evaluation and paper submission release

Version: `paper-submission-v1.0`

DOI: `10.5281/zenodo.21758262`

GitHub tag: `paper-submission-v1.0`

## Scope

This release supports the manuscript:

**External Validity Before Model Complexity: A Protocol-First,
Cell-Level Evaluation of Recursive Battery RUL Forecasting**

It archives the frozen HUST evaluator, the 20-cell calibration outputs,
the one-shot 57-cell confirmation results, independent recomputation
materials, and SHA-256 integrity records.

## Confirmatory result

- Assigned confirmation cells: 57
- Evaluable confirmation cells: 57
- Excluded confirmation cells: 0
- Local linear trend had lower future-SOH RMSE in 57/57 cells
- H1 complex-model superiority claim: not supported
- H2 multiscale superiority claim: not supported
- No post-outcome model tuning, landmark change, cell deletion,
  or subgroup reselection was performed

## Version–DOI map

| Version | Scope | DOI |
|---|---|---|
| v0.1.0-freeze | XJTU pre-unblinding freeze | 10.5281/zenodo.21570017 |
| v0.1.1-amendment | XJTU observation-semantics amendment | 10.5281/zenodo.21571142 |
| v0.2.0-SOH | SOH development freeze | 10.5281/zenodo.21729888 |
| v0.2.1-BIT | BIT structural-feasibility amendment | 10.5281/zenodo.21734477 |
| paper-submission-v1.0 | HUST final one-shot evaluation and paper release | 10.5281/zenodo.21758262 |

## Contents

- `evaluator/`: frozen evaluator archive, manifest, receipt, and runtime correction
- `calibration/`: calibration freeze, registered amendment, and 20-cell derived outputs
- `results/`: 57-cell one-shot result package, sidecar, and hard-audit logs
- `scripts/recompute_hust_confirmation.py`: independent recomputation from archived records
- `scripts/audit_release_stage.py`: release-content and hash gate
- `SHA256SUMS.txt`: generated after the release tree is final

## Independent recomputation

The recomputation script reads the archived cell-level records and protocol
directly from the final result ZIP. It independently reproduces the model
summary, H1/H2 paired effects, Wilcoxon tests, bootstrap intervals,
calibration quantiles, and coverage-width summaries.

```bash
python scripts/recompute_hust_confirmation.py \
  results/MSTT_RUL_v0.3.0_HUST_confirmatory_one_shot_results.zip \
  --output recomputed
```

## Data and licensing

No third-party raw battery datasets are redistributed.

The MIT License applies only to author-created source code,
documentation, and derived result artifacts. Original datasets remain
subject to their provider licenses and access conditions.

## Reuse boundary

The 57 HUST confirmation cells are permanently unsealed for this
research program. They must not be presented as an untouched
confirmation cohort for a successor model.
