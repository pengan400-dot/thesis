# v0.2.0-BIT-SOH confirmatory protocol

Status: scientifically specified; must be paired with an immutable OSF
Registration, an exact GitHub release, and a Zenodo specific-version DOI
before the first BIT model output is generated.

Author: Peng Xue An

## Evidence roles

- XJTU Batch-1 and Batch-3 are the only development data. They may determine
  epoch selection, final SOH-model weights, feature scalers, and cross-fitted
  conformal half-widths.
- HNEI is an exploratory cross-institutional replication and pipeline stress
  test. HNEI performance cannot change v0.2.0. If it changes any model,
  feature, numerical hyperparameter, or decision rule, the revision becomes
  v0.2.1 and HNEI is development-exposed for that revision.
- BIT Version 3 (DOI `10.17632/kw34hhw7xg.3`) is the confirmatory external
  dataset. Its arbitrary-use cohort is primary and its fixed-profile cohort is
  key secondary.

## SOH and BOL reference

For physical cell \(i\) at physical aging cycle \(t\),

\[
SOH_{i,t} =
\frac{C_{i,t}}
{\operatorname{median}(C_{i,\text{first 5 valid}})}.
\]

A valid BOL-reference capacity is finite, explicitly in Ah, and within
0.5–1.5 times the dataset-declared nominal capacity. The denominator uses the
chronologically first five valid observations. A value inside the prespecified
physical range is not replaced because it appears unusual after viewing the
complete life. Fewer than five valid values triggers the data-quality gate.
The first-five median must also lie within 0.8–1.2 times the declared nominal
capacity; this rejects already-normalized inputs mislabeled as Ah. The EOL
threshold is \(SOH\le0.8\).

Primary BOL is the first-five median. The first valid observation and the
first-three median are sensitivity analyses only.

## SOH-native model identity

The full model retains 74,405 trainable parameters, 12 causal multiscale
branches, two RoPE encoder blocks, and the trend-residual head. The weights are
new SOH weights. The legacy absolute-Ah checkpoint is a sensitivity comparator
only and is never described as the same frozen model.

Additive Ah-scale quantities are converted once using the prespecified 2.0-Ah
XJTU development reference:

| Quantity | Legacy Ah | Frozen SOH |
|---|---:|---:|
| residual bound | 0.08 | 0.04 |
| Huber delta | 0.02 | 0.01 |
| EOL-weight band | 0.08 | 0.04 |
| upward-jump tolerance | 0.002 | 0.001 |
| prediction clip | [0.4, 2.4] | [0.2, 1.2] |

No HNEI or BIT performance selects these values. The feature scaler is fitted
only on the appropriate XJTU training cells. XJTU model capacity is recomputed
from raw Ah with the frozen seven-cycle causal trailing mean; any pre-existing
smoothed-capacity column is ignored.

## Development and calibration

- Seeds: 42, 2024, and 3407.
- The 16 B1/B3 physical-cell IDs, batch assignments, and cycle counts must
  match `manifests/xjtu_expected_reference.csv`; actual input and normalized
  curve hashes are frozen.
- Epoch selection: Batch-1 trains and Batch-3 validates.
- Final refit: Batch-1 plus Batch-3 for the selected number of epochs.
- Neural arms: full K5 and single-scale K5, with identical cell–target-cycle
  support.
- Local-linear comparator: deterministic 20-cycle lookback.
- Conformal: four-fold, batch-stratified, physical-cell outer cross-fitting
  using XJTU Batch-1 plus Batch-3 only. Each fold holds out two B1 and two B3
  cells. Remaining B1 cells train; remaining B3 cells select epochs.
- One conformity score is produced per held-out physical cell and cutoff:
  the maximum absolute SOH error across evaluable observed future points after
  pointwise averaging the three seed trajectories. This yields a simultaneous
  trajectory half-width. Nominal levels are 80% and 90%. BIT never updates the
  quantiles.

## Frozen deployment task

Task title:

> fixed-cutoff recursive SOH forecasting under unknown future operating
> conditions

The primary landmark is physical aging-cycle 190, not row number, diagnostic
test number, or arbitrary record index. Cutoffs 70 and 130 are exploratory.
The neural input contains 16 model steps plus four physical-cycle warm-up
steps for the rolling features, so the minimum pre-cutoff history is 20
physical cycles.

All models are prohibited from using the post-cutoff actual current schedule,
post-cutoff charge-rate labels, future sequence length, true termination
position, total lifetime, future EOL, future missingness pattern, or any
complete-trajectory statistic. A known-future-current oracle, if run, is
separate and exploratory.

## Landmark flow and endpoint-specific analysis sets

The flow reports:

1. total files;
2. data-quality pass;
3. EOL before cycle 190;
4. right-censored before cycle 190;
5. at risk and evaluable at cycle 190;
6. RMSE set with at least five valid post-cutoff SOH observations;
7. timing set with an observed or interval-determined EOL.

RMSE uses genuinely observed post-cutoff SOH points through the first observed
EOL upper bound, or through the right-censoring boundary when EOL is not
observed, and never beyond the 700-cycle forecast horizon. Seed trajectories
are averaged pointwise first; then one RMSE is calculated for each physical
cell. Physical cells receive equal inferential weight.

Timing uses interval-aware event scoring. A predicted EOL inside the observed
crossing interval has zero timing distance. A truth trajectory without an
observed crossing remains right-censored. A prediction without a crossing
remains prediction-right-censored; the evaluation does not replace it with the
window end and does not report a fabricated ordinary RUL MAE.

## Confirmatory hypotheses and fixed-sequence inference

Primary cohort: arbitrary-use BIT cells that remain at risk and evaluable at
the physical cycle-190 landmark.

Primary metric: cell-level future SOH RMSE.

\[
\Delta_1 =
RMSE_{\text{single-scale K5}}-RMSE_{\text{full K5}}.
\]

H1 tests full K5 against single-scale K5. Only if the two-sided H1 paired
sign-flip test has \(p<0.05\) is H2 tested confirmatorily:

\[
\Delta_2 =
RMSE_{\text{local linear}}-RMSE_{\text{full K5}}.
\]

Positive differences favor full K5. Inference uses a two-sided paired
sign-flip randomization test, exact when there are no more than 20 nonzero
pairs and otherwise 100,000 Monte Carlo sign flips with seed 20260728.
Effect intervals use 10,000 paired physical-cell bootstrap samples with seed
20260727. Zeros remain in descriptive and bootstrap estimates and do not
affect randomized signs. Missing model pairs are removed only as complete
pairs for the applicable contrast, with every cell ID and reason reported;
H1 and H2 may therefore have different reported paired sample sizes. No
significance-based stopping is allowed.

No Holm adjustment is applied to H1 and gated H2. If H1 fails, H2 is
descriptive/exploratory:

| Outcome | Permitted conclusion |
|---|---|
| H1 and H2 pass | Multiscale contribution and superiority to local linear are confirmed. |
| H1 passes; H2 does not | Multiscale structural contribution is confirmed, but superiority to local linear is not established. |
| H1 does not pass | BIT did not replicate the XJTU late-stage multiscale advantage. |

## Cohorts and non-pooling rule

The arbitrary-use cohort is primary. The fixed-profile cohort is key
secondary and reports RMSE, interval-aware timing, hit rate, conformal
coverage, interval width, paired effects, and cell-bootstrap intervals
separately. There is no ordinary pooled paired p value across the expected
55/22 cohorts. Any overall supplement uses a cohort-stratified bootstrap and
reports cohort heterogeneity.

## Structural preflight and immutable freeze

Before freezing, BIT access is limited to archive hashes, file names and
counts, schemas, cycle-index semantics, units, missingness, cohort identifiers,
and cycle-190 availability. The structure-only program imports no model code
and generates no predictions, RMSE, rankings, or model-error plots.
Numeric capacity and current ranges are not summarized during this preflight.
After schema resolution, the BIT parser source is written and hashed but is
not executed on BIT capacity values before the immutable freeze.

After HNEI, an explicit version-decision receipt is required. If any HNEI
performance result changed v0.2.0, the receipt blocks BIT and requires a new
XJTU-only v0.2.1 development freeze.

The exact archive inventory, resolved schema mapping, parser, protocol,
weights, scalers, conformal quantiles, and all SHA-256 values are included in
`v0.2.0-BIT-SOH-freeze`. The first BIT model output is blocked until an OSF
Registration, GitHub release/commit, and Zenodo specific-version DOI have been
recorded.
