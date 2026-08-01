#!/usr/bin/env python3
"""Static, capacity-free verification for the v0.2.1 BIT amendment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import yaml

EVIDENCE_COMMIT = "44294e192debaa19df7840658aa5242b787cf658"
MAPPING_JSON_SHA256 = "43a4091db89e796215b69c09cca86663f934476e84fe417ac9dd3ce8febd555f"
MAPPING_MANIFEST_SHA256 = "34b08a5f401aee9cde564bc49e542768517722b682ee5b927109e8ed5e0fd3b4"
PAIRING_MANIFEST_SHA256 = "fd873e12acbf2039504a8a9f27f6bcade223e609a55292a8373491ee5dc38b43"
ARCHIVE_SHA256 = "9701ff850b8b738f9b15b5e02d1ea6fe2e3630ef704336f19acc918f133b18d7"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected object: {path}")
    return value


def git(project_root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=project_root,
        text=True,
    ).strip()


def resolve_relative(mapping_path: Path, value: object) -> Path:
    candidate = Path(str(value))
    return candidate if candidate.is_absolute() else (mapping_path.parent / candidate).resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--schema-mapping", type=Path, required=True)
    parser.add_argument("--mapping-resolution", type=Path, required=True)
    parser.add_argument("--mapping-receipt", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if "torch" in sys.modules:
        raise RuntimeError("torch must not be imported by v0.2.1 verification")

    failures: list[str] = []
    branch = git(args.project_root, "branch", "--show-current")
    commit = git(args.project_root, "rev-parse", "HEAD")
    working_tree = git(args.project_root, "status", "--short")

    if branch != "amendment/v0.2.1-BIT-structural-feasibility":
        failures.append(f"Unexpected branch: {branch}")
    if working_tree:
        failures.append("Working tree must be clean before verification")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", EVIDENCE_COMMIT, commit],
        cwd=args.project_root,
        check=False,
    )
    if ancestor.returncode != 0:
        failures.append("Mapping-evidence commit is not an ancestor")

    config = read_json(args.config)
    resolution = read_json(args.mapping_resolution)
    mapping = yaml.safe_load(args.schema_mapping.read_text(encoding="utf-8"))

    if sha256(args.mapping_resolution) != MAPPING_JSON_SHA256:
        failures.append("Mapping-resolution JSON hash changed")
    if resolution.get("status") != "PASS_OFFSET20_FILENAME_SEMANTIC_MAPPING":
        failures.append("Mapping resolution is not PASS")
    if resolution.get("capacity_read") is not False:
        failures.append("Mapping resolution accessed capacity")
    if resolution.get("model_output_generated") is not False:
        failures.append("Mapping resolution generated model output")

    if config["task"]["primary_cutoff"] != 70:
        failures.append("Task primary cutoff is not 70")
    if config["bit"]["primary_cutoff"] != 70:
        failures.append("BIT primary cutoff is not 70")
    if sorted(config["bit"]["structurally_unavailable_cutoffs"]) != [130, 190]:
        failures.append("Unavailable cutoffs are not [130, 190]")
    if config["amendment"]["mapping_evidence_commit"] != EVIDENCE_COMMIT:
        failures.append("Mapping evidence commit changed")

    if mapping["dataset"]["archive_sha256"] != ARCHIVE_SHA256:
        failures.append("BIT archive SHA changed")
    if mapping["approval"]["mapping_state"] != "READY_FOR_IMMUTABLE_FREEZE":
        failures.append("Schema mapping is not ready")
    if mapping["landmark"]["primary_physical_cycle"] != 70:
        failures.append("Schema primary landmark is not 70")
    if sorted(mapping["landmark"]["structurally_unavailable_cycles"]) != [130, 190]:
        failures.append("Schema unavailable landmarks changed")

    mapping_manifest = resolve_relative(
        args.schema_mapping,
        mapping["resolved_structure"]["evaluation_manifest_csv"],
    )
    pairing_manifest = resolve_relative(
        args.schema_mapping,
        mapping["resolved_structure"]["workbook_pairing_csv"],
    )
    parser_source = resolve_relative(
        args.schema_mapping,
        mapping["parser"]["source_file"],
    )

    if sha256(mapping_manifest) != MAPPING_MANIFEST_SHA256:
        failures.append("Mapping manifest hash changed")
    if sha256(pairing_manifest) != PAIRING_MANIFEST_SHA256:
        failures.append("Pairing manifest hash changed")
    if sha256(parser_source) != mapping["parser"]["source_sha256"]:
        failures.append("Parser source hash changed")

    parser_text = parser_source.read_text(encoding="utf-8")
    lower = parser_text.lower()
    if "import torch" in lower or "from torch" in lower:
        failures.append("Parser imports torch")
    guard_position = parser_text.find(
        "validate_registration_receipt(args.registration_receipt)"
    )
    open_position = parser_text.find("zipfile.ZipFile(args.archive")
    if guard_position < 0 or open_position < 0 or guard_position > open_position:
        failures.append("Parser archive open precedes registration guard")

    receipt_text = args.mapping_receipt.read_text(encoding="utf-8")
    for expected in (
        f"local_commit={EVIDENCE_COMMIT}",
        f"remote_commit={EVIDENCE_COMMIT}",
        f"mapping_json_sha256={MAPPING_JSON_SHA256}",
        f"mapping_manifest_sha256={MAPPING_MANIFEST_SHA256}",
    ):
        if expected not in receipt_text:
            failures.append(f"Mapping receipt lacks: {expected}")

    status = (
        "PASS_V0_2_1_PRE_FREEZE_VERIFY"
        if not failures
        else "FAIL_V0_2_1_PRE_FREEZE_VERIFY"
    )
    result = {
        "status": status,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "branch": branch,
        "verified_commit": commit,
        "mapping_evidence_commit": EVIDENCE_COMMIT,
        "protocol_config_sha256": sha256(args.config),
        "schema_mapping_sha256": sha256(args.schema_mapping),
        "mapping_resolution_sha256": sha256(args.mapping_resolution),
        "mapping_manifest_sha256": sha256(mapping_manifest),
        "workbook_pairing_sha256": sha256(pairing_manifest),
        "parser_source_sha256": sha256(parser_source),
        "capacity_opened": False,
        "model_imported": False,
        "model_output_generated": False,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(6)


if __name__ == "__main__":
    main()
