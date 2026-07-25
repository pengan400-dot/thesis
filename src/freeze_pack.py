#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

from common import json_sha256, load_protocol, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-zip", type=Path, required=True)
    return parser.parse_args()


def require_pass(path: Path, external_false: bool = False) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS":
        raise RuntimeError(f"Audit is not PASS: {path}")
    if external_false and payload.get("external_data_loaded") is not False:
        raise RuntimeError(f"External-data gate is not false: {path}")
    return payload


def main() -> None:
    args = parse_args()
    source = args.source_root.resolve()
    run_root = args.run_root.resolve()
    protocol = load_protocol(source / "configs" / "confirmatory_protocol.json")
    config_hash = json_sha256(protocol)
    archive_audit = require_pass(run_root / "inventory" / "archive_gate.json")
    model_audit = require_pass(
        run_root / "frozen_models" / "frozen_model_manifest.json",
        external_false=True,
    )
    calibration_audit = require_pass(
        run_root / "calibration" / "calibration_quantiles.json",
        external_false=True,
    )
    if model_audit.get("quick_test") or calibration_audit.get("quick_test"):
        raise RuntimeError("Quick-test outputs cannot be frozen")
    if (
        archive_audit.get("archives_expected") != 3
        or archive_audit.get("mat_entries_expected") != 24
        or archive_audit.get("mat_entries_observed") != 24
        or archive_audit.get("mat_payload_parsed") is not False
    ):
        raise RuntimeError(
            "Archive inventory is not the frozen 3-archive/24-file gate"
        )
    if (
        model_audit.get("config_sha256") != config_hash
        or calibration_audit.get("config_sha256") != config_hash
    ):
        raise RuntimeError("Model/calibration config hash differs from source protocol")
    if not model_audit.get("development_data_sha256") or model_audit.get(
        "development_data_sha256"
    ) != calibration_audit.get("development_data_sha256"):
        raise RuntimeError("Model/calibration development-data hashes differ")
    expected_jobs = {
        (str(spec["id"]), int(seed))
        for spec in protocol["models"]
        if str(spec["id"]).startswith("mstt_")
        for seed in protocol["training"]["seeds"]
    }
    pass_jobs = [
        row for row in model_audit.get("jobs", []) if row.get("status") == "PASS"
    ]
    observed_jobs = {
        (str(row.get("model_id")), int(row.get("seed"))) for row in pass_jobs
    }
    if observed_jobs != expected_jobs or len(pass_jobs) != len(expected_jobs):
        raise RuntimeError(
            f"Frozen-model job set mismatch: expected={sorted(expected_jobs)}, "
            f"observed={sorted(observed_jobs)}"
        )
    for model_id, seed in sorted(expected_jobs):
        job_dir = run_root / "frozen_models" / "models" / model_id / f"seed_{seed}"
        audit_path = job_dir / "job_audit.json"
        model_path = job_dir / "model.pt"
        scaler_path = job_dir / "scaler.joblib"
        audit = require_pass(audit_path, external_false=True)
        if (
            audit.get("config_sha256") != config_hash
            or audit.get("development_data_sha256")
            != model_audit.get("development_data_sha256")
            or sha256_file(model_path) != audit.get("model_sha256")
            or sha256_file(scaler_path) != audit.get("scaler_sha256")
        ):
            raise RuntimeError(f"Frozen job hash gate failed: {job_dir}")
    if (
        calibration_audit.get("n_physical_cells") != 16
        or calibration_audit.get("nested_epoch_selection") is not True
        or calibration_audit.get("outer_held_out_cells_used_for_epoch_selection")
        is not False
    ):
        raise RuntimeError("Calibration is not the frozen 16-cell nested cross-fit")

    artifact_root = source / "freeze_artifacts"
    if artifact_root.exists():
        shutil.rmtree(artifact_root)
    artifact_root.mkdir(parents=True)
    copies = {
        run_root / "inventory": artifact_root / "inventory",
        run_root / "frozen_models": artifact_root / "frozen_models",
        run_root / "calibration": artifact_root / "calibration",
    }
    for origin, destination in copies.items():
        shutil.copytree(origin, destination)
    development = run_root / "development_prepared"
    if not development.is_dir():
        raise FileNotFoundError("Missing audited development preparation directory")
    shutil.copytree(
        development,
        artifact_root / "development_audit",
        ignore=shutil.ignore_patterns("curves"),
    )
    environment = run_root / "environment"
    if environment.is_dir():
        shutil.copytree(environment, artifact_root / "environment")

    manifest_rows = []
    for path in sorted(artifact_root.rglob("*")):
        if path.is_file():
            manifest_rows.append(
                {
                    "path": path.relative_to(source).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    ready = {
        "status": "PASS",
        "release_tag": "v0.1.0-freeze",
        "external_mat_payload_loaded": False,
        "archive_audit": archive_audit,
        "model_config_sha256": model_audit["config_sha256"],
        "common_support_sha256": model_audit["support_sha256"],
        "development_data_sha256": model_audit["development_data_sha256"],
        "calibration_cells": calibration_audit["n_physical_cells"],
        "calibration_quantiles_Ah": calibration_audit["coverage_quantiles_Ah"],
        "files": manifest_rows,
        "next_action": (
            "Commit this exact source tree, publish v0.1.0-freeze, obtain a "
            "Zenodo version DOI, then register the receipt before external extraction."
        ),
    }
    ready_path = artifact_root / "FREEZE_READY.json"
    ready_path.write_text(
        json.dumps(ready, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    output = args.output_zip.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    excluded_parts = {".venv", "__pycache__", ".git", "runs"}
    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for path in sorted(source.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(source)
            if any(part in excluded_parts for part in relative.parts):
                continue
            if path.resolve() == output:
                continue
            archive.write(path, (source.name / relative).as_posix())
    with zipfile.ZipFile(output) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"Freeze ZIP CRC failed at {bad}")
    digest = sha256_file(output)
    sha_path = output.with_suffix(output.suffix + ".sha256")
    sha_path.write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    print("[PASS] freeze package created")
    print(f"[ZIP] {output}")
    print(f"[SHA256] {digest}")
    print("[NEXT] Commit, release v0.1.0-freeze, and obtain a Zenodo version DOI")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        raise
