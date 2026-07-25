# v0.1.1 Post-unblinding, pre-model-evaluation amendment

## Classification

This is a **post-unblinding, pre-model-evaluation amendment**. It must
not be described as part of the untouched `v0.1.0-freeze`
preregistration.

## Trigger

The immutable `v0.1.0-freeze` workflow opened the external archives
only after a valid Git commit, release tag, Zenodo version DOI, and
freeze receipt existed. The original external preflight then failed
because it required Batch-4 reference capacity to be observed at every
MATLAB cycle record.

## Observed Batch-4 structure

A read-only audit found, for all eight Batch-4 cells:

- the first genuine reference-capacity test occurs at MATLAB record 1;
- subsequent reference-capacity tests occur at records 7, 13, 19, ...;
- every reference-test gap is exactly six records;
- the final reference-capacity test equals `cycle_life_metadata`;
- each six-record block contains one reference-capacity test and five
  additional full charge/discharge records at other rates.

Therefore, Batch-4 reference capacity is not observed at every physical
charge/discharge record. It is observed once per deterministic
six-record block.

## Original outcome retained

The original `v0.1.0-freeze` preflight result remains **FAIL**. It is
not overwritten or relabeled as PASS.

## Amendment scope

Only the following item changes:

- Batch-4 external observation semantics and its preflight gate change
  from “reference capacity every record” to “exact reference-capacity
  pattern every six MATLAB cycle records”.

The following remain unchanged:

- frozen model weights;
- random seeds;
- model architecture;
- training data;
- hyperparameters;
- cutoffs 70, 130, and 190;
- 700-cycle forecast horizon;
- causal LOCF sparse-input handling;
- fixed 1.6-Ah threshold;
- interval-censored EOL semantics;
- calibration quantiles;
- cell-level statistical plan;
- all exclusions and failure records.

## Information seen before amendment

Before this amendment:

- external file structure and reference-test descriptions were parsed;
- the count of fixed-threshold crossings was available;
- no external model predictions, errors, rankings, or statistical tests
  had been produced.

## Claim limitation

Results produced after this amendment are **amended external
validation** and not untouched preregistered confirmation.
