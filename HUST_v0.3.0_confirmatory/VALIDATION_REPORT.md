# Package validation report

Validation date: 2026-08-01

Package: `MSTT_RUL_v0.3.0_HUST_confirmatory_20260801`

## Outcome

The code package passed the available pre-data validation gates. No HUST pickle, capacity trajectory, SOH, EOL, prediction, error, or model ranking was accessed during package construction or testing.

## Checks performed

1. Python source compilation passed for every file under `src/` and `tests/`.
2. `manager.sh` passed Bash syntax validation.
3. The bundled XJTU archive contains only the 16 registered Batch 1+3 CSV files.
4. Each bundled XJTU CSV matches the frozen per-file SHA-256 and cycle count in `manifests/xjtu_expected_reference.csv`.
5. Every Python file under `src/mstt_soh/` is byte-for-byte identical to the corresponding `v0.2.0-SOH` frozen source file.
6. Twenty-five unit tests passed or were appropriately skipped in the build environment:
   - 23 passed;
   - 2 PyTorch topology/forward tests were skipped because PyTorch was not installed in the build runtime. They are enabled automatically after `manager.sh install` in the execution environment.
7. New synthetic tests covered:
   - structure-only 77-cell ZIP inventory without unpickling;
   - deterministic 20/57 split reproducibility;
   - official-form negative-current capacity integration;
   - exact `7-5` first-two-cycle rule;
   - causal SOH landmark and future-support logic;
   - strict JSON serialization of non-finite statistical values;
   - rejection of critical machine-readable protocol mutation;
   - finite-sample conformal order statistics;
   - the 40-cell minimum confirmation gate;
   - fixed-sequence H1/H2 gatekeeping;
   - zero-evaluable-cell aggregation as “not estimable,” including placeholder figures rather than a crash.
8. The package manager enforces evaluator registration before calibration access, calibration registration before confirmation access, project/model/source hash validation, non-overwriting freeze artifacts, and one-shot confirmation state.

## Deliberately not executed

- Formal two-model × three-seed training was not rerun in the build environment because PyTorch was unavailable there. The exact previously frozen training/model source and XJTU B1+3 inputs are included for the execution environment.
- The real HUST pipeline was not executed because the official raw HUST archive is intentionally not bundled or opened before the user creates the registered structure receipt and external freezes.
- No HUST performance claim can be made from this validation report.

## Execution-time gates still required

The user must run `python scripts/verify_code_package.py`, install dependencies, obtain the official HUST version-2 archive, and follow `README_CN.md` in order. A successful code-package test does not replace the two externally timestamped freeze receipts or the final one-shot result archive.
