#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

from common import json_sha256, load_protocol, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-zip", type=Path, required=True)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    protocol = load_protocol(
        args.source_root / "configs" / "confirmatory_protocol.json"
    )
    config_hash = json_sha256(protocol)
    required = [
        run_root / "freeze_receipt.json",
        run_root / "external_prepared" / "external_preflight.json",
        run_root / "external_evaluation" / "evaluation_audit.json",
        run_root / "aggregate" / "aggregation_audit.json",
    ]
    audits = {}
    for path in required[1:]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "PASS":
            raise RuntimeError(f"Audit is not PASS: {path}")
        if payload.get("config_sha256") != config_hash:
            raise RuntimeError(f"Audit config hash mismatch: {path}")
        audits[path.name] = payload
    receipt = json.loads(required[0].read_text(encoding="utf-8"))
    if receipt.get("status") != "PASS":
        raise RuntimeError("Freeze receipt is not PASS")
    preflight = audits["external_preflight.json"]
    embedded_receipt = preflight.get("freeze_receipt", {})
    for key in (
        "commit",
        "zenodo_version_doi",
        "freeze_zip_sha256",
    ):
        if embedded_receipt.get(key) != receipt.get(key):
            raise RuntimeError(f"External preflight receipt mismatch for {key}")
    evaluation = audits["evaluation_audit.json"]
    aggregation = audits["aggregation_audit.json"]
    if evaluation.get("external_data_sha256") != aggregation.get(
        "external_data_sha256"
    ):
        raise RuntimeError("Evaluation/aggregation external-data hashes differ")
    output = args.output_zip.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    include_roots = [
        run_root / "inventory",
        run_root / "frozen_models",
        run_root / "calibration",
        run_root / "external_prepared",
        run_root / "external_evaluation",
        run_root / "aggregate",
    ]
    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        archive.write(required[0], f"results/{required[0].name}")
        for root in include_roots:
            for path in sorted(root.rglob("*")):
                if not path.is_file():
                    continue
                if root.name == "external_prepared" and "curves" in path.parts:
                    continue
                relative = path.relative_to(run_root)
                archive.write(path, f"results/{relative.as_posix()}")
        for relative in (
            "configs/confirmatory_protocol.json",
            "manifests/library_archive_inventory_20260725.csv",
            "preregistration/xjtu_b456_preregistration_v0.1.md",
        ):
            path = args.source_root / relative
            archive.write(path, f"source/{relative}")
    with zipfile.ZipFile(output) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"Result ZIP CRC failure at {bad}")
    digest = sha256_file(output)
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{digest}  {output.name}\n", encoding="utf-8"
    )
    print(f"[PASS] {output}")
    print(f"[SHA256] {digest}")


if __name__ == "__main__":
    main()
