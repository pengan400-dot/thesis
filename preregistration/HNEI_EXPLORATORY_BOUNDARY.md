# HNEI v0.2.0 evidence boundary

HNEI is an exploratory cross-institutional replication and pipeline stress
test. Its archive schema and selected capacity/EOL locations were exposed
before this revision, so it is not an untouched confirmatory dataset.

The v0.2.0 SOH weights, scaler, additive SOH hyperparameters, cutoffs, EOL
definition, censoring rules, and conformal quantiles are frozen using XJTU
Batch-1 and Batch-3 only. HNEI may reveal parsing or data-contract failures,
but its performance cannot select or change v0.2.0. Any performance-driven
change creates v0.2.1, and HNEI is then development-exposed for v0.2.1.

After the HNEI result package is created, a machine-readable decision receipt
must record either `unchanged` or `modified`. Only `unchanged` unlocks BIT
structure preflight for v0.2.0. A `modified` decision blocks v0.2.0 and
requires a new XJTU-only development and freeze cycle.
