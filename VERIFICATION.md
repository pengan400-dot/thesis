# Verification record

Date: 2026-07-25

## Real archive gate

- 10 selected 7-Zip archives opened successfully.
- 24 expected MATLAB filenames and uncompressed byte sizes matched.
- All selected outer SHA-256 values matched the frozen inventory.
- The integrity test passed for every archive.
- No MATLAB payload, `summary`, capacity, cycle life, or outcome value was parsed.

## Code checks

- Python byte compilation: PASS.
- Shell syntax check for `manager.sh`: PASS.
- Ruff static analysis: PASS.
- Core unit tests: PASS.
  - K=1 and K=5 common target support.
  - finite-sample conformal order statistic.
  - sparse EOL interval semantics.
  - right-censored EOL retention and one-sided timing score.
  - development curve content hashing.

## Execution checks

- CPU quick training and frozen checkpoint creation: PASS.
- Nested cell-level cross-fit calibration quick run: PASS.
- Synthetic 24-cell end-to-end evaluation: PASS.
- Synthetic cell-first aggregation, bootstrap, Wilcoxon and Holm family: PASS.
- Synthetic run retained a deliberately right-censored external cell.
- Development 31-cell 8/15/8 CSV validation and ZIP builder: PASS.
- Full 16-cell nested cross-fit calibration on synthetic development curves: PASS.
- Positive freeze-package gate with 10-archive/24-file audit, model/scaler hashes,
  16-cell calibration and immutable `FREEZE_READY.json`: PASS.
- Freeze-receipt registration with a syntactically valid 40-character commit and
  Zenodo version DOI: PASS.
- Final results packaging, audit-hash consistency and ZIP CRC validation: PASS.

## Deliberate non-test

The real Batch-4/5/6 MATLAB parser and external evaluator were not run. They are
blocked until a complete freeze package has been committed, released as
`v0.1.0-freeze`, assigned a Zenodo version DOI, and registered by the receipt
gate.

## Required input not bundled

The repository does not redistribute XJTU data. The server must provide the ten
selected Batch-4/5/6 archives and the previously prepared 31-cell
`xjtu_q2_prepared_csv.zip`. If only the 31 source CSV files remain,
`manager.sh make_dev_zip` validates and packages them.
