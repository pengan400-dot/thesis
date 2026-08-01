# BIT v0.2.1 evaluator implementation note

Governance label: **post-registration, pre-capacity-access evaluator implementation**.

This implementation note records the evaluation software written after the
immutable v0.2.1 protocol registration and before any BIT capacity values were
opened. It is an implementation of the registered protocol, not a new protocol
amendment.

## Frozen evaluation identity

- Registered protocol commit: `780a057d26cd4a0fc2dc50a80161d66c936155b5`
- Primary landmark: physical cycle 70
- Structurally unavailable landmarks: physical cycles 130 and 190
- Included structural cell set: 72 cells
- Structural exclusion: `BIT_#2`
- Primary cohort: 55 arbitrary-use cells
- Key secondary cohort: 17 fixed-profile cells
- Frozen neural models: `mstt_full_K5` and `mstt_single_scale_K5`
- Frozen transparent baseline: `local_linear_trend`
- Frozen seeds: 42, 2024, and 3407

## Registered data handling

1. The already frozen parser extracts raw discharge capacity in Ah using the
   span of cumulative capacity over negative-current rows within each mapped
   physical cycle.
2. SOH is calculated as raw discharge capacity divided by the median of the
   chronologically first five valid raw-Ah observations.
3. Missing physical cycles used for recursive model input are filled only by
   causal last observation carried forward. No backward fill or future
   interpolation is allowed.
4. The model target is the registered seven-cycle causal trailing mean of SOH.
5. Post-cutoff measured current, charge-rate labels, future length, termination
   position, lifetime, future EOL, future missingness, and any complete-future
   statistic are prohibited as model inputs.

## Evaluation and inference

- Three frozen-seed neural trajectories are averaged pointwise before one
  cell-level metric is calculated.
- Future-capacity RMSE is calculated only with at least five genuinely observed
  post-cutoff SOH points on the registered support.
- Threshold-hit rate, hit-conditional interval distance, and the censor-aware
  finite-horizon timing score are reported separately.
- Physical cells, not future cycles, are the inferential units.
- H1 compares `mstt_full_K5` with `mstt_single_scale_K5` in the arbitrary-use
  cohort.
- H2 compares `mstt_full_K5` with `local_linear_trend` and is confirmatory only
  when H1 passes the frozen two-sided alpha threshold.
- No ordinary pooled paired p value across the two cohorts is calculated.
- The fixed-profile cohort is reported as the registered key secondary cohort.

The evaluator source must be committed, pushed, statically verified, and packed
into its own pre-capacity source freeze before the parser can be invoked.
