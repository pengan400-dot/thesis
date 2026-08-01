# HUST v0.3.0 preregistration record

## Scope

This record governs a study-specific, structure-first evaluation on the public HUST personalized-discharge dataset, Mendeley Data version 2, DOI `10.17632/nsc7hnsg4s.2`. It must be frozen before any HUST pickle is unpickled or any HUST capacity, SOH, EOL, prediction, error, or model ranking is viewed.

HUST is a historical public dataset. “Prospectively specified” refers only to the analysis protocol and study-specific unblinding sequence, not to prospective data collection.

The machine-readable source of truth is `configs/hust_confirmatory_protocol_v0.3.0.json`. If this narrative differs from that JSON, the JSON and its registered SHA-256 control.

## Dataset and structural gate

- Expected archive: `hust_data.zip` from the registered official record.
- Expected physical cells: 77 pickle members under `our_data/`.
- Structure-only inventory may read ZIP metadata and bytes required for CRC testing but may not unpickle a member or compute any outcome value.
- The official-source archive SHA-256, inventory SHA-256, and split-manifest SHA-256 are recorded before model evaluation.
- Ordinary Python pickle loading is allowed only after this binding and explicit operator acknowledgement that the archive is trusted.

## Deterministic split

For every physical source cell ID, compute SHA-256 over:

```text
UTF8("MSTT_RUL_v0.3.0_HUST_split_20260801") || NUL || UTF8(cell_id)
```

Sort by digest and then cell ID. The first 20 cells form the parser/UQ calibration arm. The remaining 57 form the sealed confirmation arm. Performance-stratified reassignment is prohibited.

## Frozen model lineage

The only neural candidates are the exact normalized-SOH models from `v0.2.0-SOH`:

- `mstt_full_K5`, expected 74,405 trainable parameters;
- `mstt_single_scale_K5`, expected 69,789 trainable parameters.

Each is trained with seeds 42, 2024, and 3407 on XJTU Batches 1 and 3 only under `configs/soh_development_protocol_v0.2.0.json`. External retraining, fine-tuning, scaler updating, architecture selection, hyperparameter selection, or seed selection is prohibited. The pointwise mean of the three seed trajectories is formed before cell-level metrics and threshold crossing.

The transparent primary comparator is a local linear trend using the final 20 physical-cycle grid values at the landmark, with slope clipped to `[-0.02, -1e-6]` SOH per cycle.

## Capacity, SOH, and event definitions

Per-cycle HUST discharge capacity is computed from the registered time series as:

```text
sum(max(-I_i_A, 0) * max(t_i - t_{i-1}, 0) / 3600), i = 1..n-1
```

Required columns are `Current (mA)`, `Time (s)`, and `Voltage (V)`. For source cell `7-5`, the first two sorted raw cycles are excluded exactly as specified before HUST outcome access. Remaining physical cycle numbers are preserved.

A cycle containing any non-finite current, time, or voltage sample is excluded from capacity integration and retained in the cycle-level failure audit. No interpolation or integration across the missing sample is allowed. The cell remains eligible only if all other frozen history, landmark, and future-support gates are met.

For each cell:

```text
BOL = median(first 5 chronologically valid capacity observations)
raw SOH(t) = capacity(t) / BOL
EOL = first genuinely observed raw SOH(t) <= 0.80
```

The model input target is a seven-cycle causal trailing mean after causal last-observation-carried-forward on the physical-cycle grid. No backward fill, future interpolation, future schedule, future length, true EOL, or future missingness information may enter a forecast.

## Causal SOH-anchored landmark

For a cell, the landmark is the first genuinely observed physical cycle satisfying all of the following:

1. raw SOH is greater than 0.80 and no greater than 0.90;
2. no earlier genuinely observed raw SOH is at or below 0.80;
3. at least 20 genuinely observed capacity measurements exist at or before the landmark;
4. at least five genuinely observed post-landmark raw-SOH values exist within the registered metric support.

Metric support ends at the first observed EOL upper bound or right-censoring boundary and is capped at landmark plus 700 physical cycles.

A cell without an eligible landmark or sufficient future support remains in the flow and exclusion tables. The landmark, threshold, minimum-history gate, minimum-future gate, or exclusion rule must not change after HUST capacity access.

## Calibration arm

Only the 20 assigned calibration cells may be opened after the evaluator source, model weights, protocol, source inventory, and split are frozen and independently registered.

Calibration may be used only to validate the already-frozen parser and compute full-K5 simultaneous trajectory-band widths. The per-cell nonconformity score is the maximum absolute raw-SOH error over registered future support. For nominal coverage `c`, the quantile is the `ceil((n+1)c)`-th ordered score, capped at `n`.

Nominal levels are 0.80 and 0.90. At least 15 eligible calibration cells are required for UQ estimation. If this minimum is not met, UQ is labelled not estimable; the frozen point-prediction confirmation proceeds without a substituted calibration sample.

No calibration result may change the model, features, weights, seeds, scaler, parser rule, landmark, exclusion rule, primary comparator, endpoint, or inferential procedure. A necessary parser correction discovered here creates a newly frozen version and requires a new evaluator registration before the confirmation arm is opened.

## Confirmation arm and one-shot state

After the calibration artifacts are frozen and independently registered, the program writes a hash-bound `one_shot_state.json` before reading any of the 57 confirmation members. From that point the confirmation arm is considered opened.

One completed run is allowed. An interrupted run may resume only with identical protocol, source archive, evaluator archive, calibration archive, code, models, and output root. Deleting the state receipt, changing an input, or starting a new output root invalidates confirmatory status.

All parse failures, data-quality exclusions, non-crossings, and right-censored cells are retained.

## Estimand and statistics

The primary endpoint is cell-level future raw-SOH RMSE. The statistical unit is the physical cell.

H1 compares `mstt_full_K5` with `local_linear_trend`. Define paired effect as comparator RMSE minus full-K5 RMSE; positive favors full K5. H1 is evaluated with a two-sided cell-level Wilcoxon signed-rank test at alpha 0.05 and a 10,000-replicate paired-cell bootstrap using seed 20260801.

H1 meets the registered confirmatory-and-practical success criterion only if:

1. two-sided p < 0.05;
2. paired-bootstrap 95% interval lower bound > 0;
3. mean paired improvement is at least 0.005 SOH RMSE.

H2 compares `mstt_full_K5` with `mstt_single_scale_K5`. Its confirmatory Wilcoxon p value is computed only if the H1 p-value gate opens. If not, the H2 effect and bootstrap interval are descriptive only.

At least 40 confirmation cells with complete RMSE pairs for all three models are required. If fewer are available, H1 and H2 are not estimable; only feasibility and descriptive results are reported.

Secondary descriptive endpoints are future SOH MAE, censor-aware timing score, threshold-hit rate, simultaneous 80%/90% trajectory coverage, mean interval width, and EOL-band compatibility.

## Governance and reporting

Required order:

1. structure-only HUST inventory and deterministic split;
2. XJTU-only model training or validation of exact formal weights;
3. evaluator archive freeze and external timestamped registration;
4. calibration-arm opening and UQ computation;
5. calibration archive freeze and external timestamped registration;
6. one-shot state receipt, then confirmation-arm opening;
7. deterministic confirmation evaluation;
8. cell-first aggregation and complete result archive.

Any post-confirmation change creates an exploratory successor and cannot replace v0.3.0. Results must be reported regardless of direction, including failure to meet practical improvement, simple-baseline superiority, poor coverage, empty/insufficient analysis sets, and non-estimability.
