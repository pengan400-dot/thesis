#!/usr/bin/env python3
"""Audit the paper-submission release staging directory."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import py_compile

DOI = "10.5281/zenodo.21758262"
VERSION = "paper-submission-v1.0"
EXPECTED = {
    "evaluator/MSTT_RUL_v0.3.0_HUST_evaluator_pre_calibration_freeze.zip":
        "408faf046cd199ff7f1a2329a939ad0bc18358c10dbd2bbf4b5803cfc67af18c",
    "calibration/MSTT_RUL_v0.3.0_HUST_calibration_pre_confirmation_freeze.zip":
        "485fa7bf9eb9665d1b73f3ed1da84949e0a9aeda65486dbebfb323d04f69cd83",
    "results/MSTT_RUL_v0.3.0_HUST_confirmatory_one_shot_results.zip":
        "b694230e3ed124f6a6d9f1aaf9dbf14715d64605a11bb33fb6de1ecce5cadf9c",
}
REQUIRED = [
    "README.md",
    "VERSION_NOTES.md",
    "LICENSE",
    "RELEASE_METADATA.json",
    "requirements-recompute.txt",
    "scripts/recompute_hust_confirmation.py",
    "scripts/audit_release_stage.py",
    "calibration/results/calibration_audit.json",
    "calibration/results/calibration_quantiles.json",
    "calibration/results/calibration_scores.csv",
    "calibration/results/calibration_seed_records.csv",
    "results/MSTT_RUL_v0.3.0_HUST_confirmatory_one_shot_results.zip.sha256",
]
FORBIDDEN_SUFFIXES = {".pkl", ".pickle", ".mat", ".h5", ".hdf5", ".pt", ".pth"}
FORBIDDEN_NAMES = {"hust_data.zip"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type=Path)
    args = parser.parse_args()
    stage = args.stage.resolve()

    checks: dict[str, bool] = {"stage_exists": stage.is_dir()}
    for rel in REQUIRED:
        checks[f"required:{rel}"] = (stage / rel).is_file()
    for rel, expected in EXPECTED.items():
        path = stage / rel
        checks[f"hash:{rel}"] = path.is_file() and sha256(path) == expected

    readme = (stage / "README.md").read_text(encoding="utf-8", errors="replace")
    notes = (stage / "VERSION_NOTES.md").read_text(encoding="utf-8", errors="replace")
    checks["README_DOI"] = DOI in readme
    checks["README_version"] = VERSION in readme
    checks["README_BIT_DOI_preserved"] = "10.5281/zenodo.21734477" in readme
    checks["VERSION_NOTES_DOI"] = DOI in notes
    checks["VERSION_NOTES_version"] = VERSION in notes

    forbidden = []
    for path in stage.rglob("*"):
        if not path.is_file():
            continue
        if path.name.lower() in FORBIDDEN_NAMES or path.suffix.lower() in FORBIDDEN_SUFFIXES:
            forbidden.append(path.relative_to(stage).as_posix())
    checks["no_forbidden_raw_or_model_files"] = not forbidden

    compile_failures = []
    for rel in ("scripts/recompute_hust_confirmation.py", "scripts/audit_release_stage.py"):
        script = stage / rel
        try:
            py_compile.compile(str(script), doraise=True)
        except Exception as exc:
            compile_failures.append(f"{rel}: {exc}")
    checks["scripts_compile"] = not compile_failures

    for name, passed in checks.items():
        print(f"{name}={'PASS' if passed else 'FAIL'}")
    print(f"forbidden_files={forbidden}")
    print(f"compile_failures={compile_failures}")

    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        print(f"[STOP] failed_checks={failed}")
        raise SystemExit(1)
    print("[PASS] paper-submission staging directory gate passed")


if __name__ == "__main__":
    main()
