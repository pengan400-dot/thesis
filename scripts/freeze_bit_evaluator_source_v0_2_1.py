#!/usr/bin/env python3
"""Freeze the committed BIT v0.2.1 evaluator source before capacity access."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


REGISTERED_COMMIT = "780a057d26cd4a0fc2dc50a80161d66c936155b5"
EXPECTED_VERIFY_STATUS = "PASS_BIT_V0_2_1_EVALUATOR_SOURCE_VERIFY"
OUTPUT_STATUS = "PASS_BIT_V0_2_1_EVALUATOR_SOURCE_FREEZE"
SOURCE_FILES = (
    "manager.sh",
    "scripts/bit_v021_evaluator.py",
    "scripts/verify_bit_evaluator_source_v0_2_1.py",
    "scripts/freeze_bit_evaluator_source_v0_2_1.py",
    "scripts/parse_bit_capacity_v0_2_1.py",
    "tests/test_bit_v021_evaluator_contract.py",
    "preregistration/IMPLEMENTATION_NOTE_v0.2.1_BIT_EVALUATOR_PRE_CAPACITY.md",
    "configs/soh_development_protocol_v0.2.1_BIT_amendment.json",
    "configs/soh_development_protocol_v0.2.0.json",
    "src/mstt_soh/pipeline.py",
    "src/mstt_soh/model_variants.py",
    "src/mstt_soh/soh.py",
    "src/mstt_soh/statistics.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def deterministic_write(archive: zipfile.ZipFile, name: str, payload: bytes) -> None:
    info = zipfile.ZipInfo(name)
    info.date_time = (1980, 1, 1, 0, 0, 0)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    archive.writestr(info, payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--verify-receipt", type=Path, required=True)
    parser.add_argument("--registration-receipt", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--output-zip", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    args = parser.parse_args()

    project = args.project_root.resolve()
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=project, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--short"], cwd=project, text=True
    ).strip()
    if status:
        raise RuntimeError("Working tree must be clean before evaluator-source freeze")
    verify = read_json(args.verify_receipt)
    if verify.get("status") != EXPECTED_VERIFY_STATUS:
        raise RuntimeError("Evaluator source verify receipt is not PASS")
    if verify.get("evaluator_source_commit") != head:
        raise RuntimeError("Verify receipt does not match current commit")
    if verify.get("registered_protocol_commit") != REGISTERED_COMMIT:
        raise RuntimeError("Verify receipt references wrong protocol commit")
    for key in ("capacity_opened", "model_weights_loaded", "model_output_generated"):
        if verify.get(key) is not False:
            raise RuntimeError(f"Pre-source-freeze boundary failed: {key}")
    if args.evaluation_dir.exists() and any(args.evaluation_dir.rglob("*")):
        raise RuntimeError("BIT evaluation directory is no longer empty")

    source_hashes: dict[str, str] = {}
    for relative in SOURCE_FILES:
        path = project / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        source_hashes[relative] = sha256(path)
    if verify.get("source_files") != {
        key: source_hashes[key]
        for key in verify.get("source_files", {})
    }:
        raise RuntimeError("Evaluator source hashes changed after verification")

    manifest = {
        "status": OUTPUT_STATUS,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "governance_label": (
            "post-registration, pre-capacity-access evaluator implementation"
        ),
        "registered_protocol_commit": REGISTERED_COMMIT,
        "evaluator_source_commit": head,
        "primary_cutoff": 70,
        "structurally_unavailable_cutoffs": [130, 190],
        "registration_receipt_sha256": sha256(args.registration_receipt),
        "bit_freeze_manifest_sha256": sha256(args.freeze_manifest),
        "source_verify_receipt_sha256": sha256(args.verify_receipt),
        "capacity_opened_before_source_freeze": False,
        "model_weights_loaded_before_source_freeze": False,
        "model_output_generated_before_source_freeze": False,
        "source_files": source_hashes,
    }
    args.output_zip.parent.mkdir(parents=True, exist_ok=True)
    if args.output_zip.exists():
        raise FileExistsError(args.output_zip)
    with zipfile.ZipFile(
        args.output_zip, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for relative in sorted(SOURCE_FILES):
            deterministic_write(
                archive,
                f"code/{relative}",
                (project / relative).read_bytes(),
            )
        deterministic_write(
            archive,
            "evidence/evaluator_source_verify_receipt.json",
            args.verify_receipt.read_bytes(),
        )
        deterministic_write(
            archive,
            "evidence/registration_receipt.json",
            args.registration_receipt.read_bytes(),
        )
        deterministic_write(
            archive,
            "evidence/bit_freeze_manifest.json",
            args.freeze_manifest.read_bytes(),
        )
        deterministic_write(
            archive,
            "BIT_EVALUATOR_SOURCE_FREEZE.json",
            json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
            + b"\n",
        )
    with zipfile.ZipFile(args.output_zip) as archive:
        bad = archive.testzip()
    if bad is not None:
        raise RuntimeError(f"Evaluator source ZIP failed: {bad}")
    digest = sha256(args.output_zip)
    manifest["source_freeze_zip_sha256"] = digest
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.output_zip.with_suffix(args.output_zip.suffix + ".sha256").write_text(
        f"{digest}  {args.output_zip.name}\n",
        encoding="utf-8",
    )
    receipt = {
        **manifest,
        "source_freeze_zip": args.output_zip.name,
        "source_freeze_manifest_sha256": sha256(args.output_manifest),
        "bit_evaluation_unlocked_by_registration": True,
        "runtime_requires_this_source_receipt": True,
    }
    args.output_receipt.parent.mkdir(parents=True, exist_ok=True)
    args.output_receipt.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[PASS] evaluator source freeze: {args.output_zip}")
    print(f"SHA256={digest}")
    print(f"RECEIPT={args.output_receipt}")


if __name__ == "__main__":
    main()
