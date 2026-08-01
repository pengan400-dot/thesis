#!/usr/bin/env python3
"""Create the immutable v0.2.1 pre-BIT model-evaluation freeze."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import zipfile

import yaml

PLACEHOLDER = re.compile(r"REPLACE|UNRESOLVED|TODO|TBD", re.IGNORECASE)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected object: {path}")
    return value


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def parse_key_value_receipt(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key.strip()] = value.strip()
    return result


def resolve_relative(mapping_path: Path, value: object) -> Path:
    candidate = Path(str(value))
    return candidate if candidate.is_absolute() else (mapping_path.parent / candidate).resolve()


def current_commit(project_root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        text=True,
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--schema-mapping", type=Path, required=True)
    parser.add_argument("--preflight-dir", type=Path, required=True)
    parser.add_argument("--development-freeze", type=Path, required=True)
    parser.add_argument("--development-manifest", type=Path, required=True)
    parser.add_argument("--hnei-decision", type=Path, required=True)
    parser.add_argument("--mapping-resolution", type=Path, required=True)
    parser.add_argument("--mapping-receipt", type=Path, required=True)
    parser.add_argument("--verify-receipt", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-zip", type=Path, required=True)
    args = parser.parse_args()

    if "torch" in sys.modules:
        raise RuntimeError("torch must not be imported while freezing BIT")

    config = read_json(args.config)
    mapping_text = args.schema_mapping.read_text(encoding="utf-8")
    if PLACEHOLDER.search(mapping_text):
        raise RuntimeError("BIT v0.2.1 schema mapping contains placeholders")
    mapping = yaml.safe_load(mapping_text)
    preflight_path = args.preflight_dir / "bit_structural_preflight.json"
    preflight = read_json(preflight_path)
    development = read_json(args.development_manifest)
    hnei_decision = read_json(args.hnei_decision)
    resolution = read_json(args.mapping_resolution)
    verification = read_json(args.verify_receipt)
    mapping_receipt = parse_key_value_receipt(args.mapping_receipt)

    if config["version_labels"]["bit_freeze"] != "v0.2.1-BIT-structural-feasibility-amendment":
        raise RuntimeError("Wrong v0.2.1 BIT freeze version label")
    if config["task"]["primary_cutoff"] != 70 or config["bit"]["primary_cutoff"] != 70:
        raise RuntimeError("Primary BIT cutoff must be physical cycle 70")
    if sorted(config["bit"]["structurally_unavailable_cutoffs"]) != [130, 190]:
        raise RuntimeError("BIT 130/190 structural unavailability changed")
    if config["bit"]["physical_cycle_mapping"] != {
        "first20_local_1_to_20": "physical 1 to 20",
        "first20_local_21": "excluded terminal workbook bucket",
        "later_formula": "physical_cycle = 20 + local_cycle",
    }:
        raise RuntimeError("Offset-20 physical-cycle mapping changed")

    if (
        preflight.get("status") != "STRUCTURE_ONLY_PASS"
        or preflight.get("model_imported") is not False
        or preflight.get("model_output_generated") is not False
        or preflight.get("rmse_computed") is not False
        or preflight.get("soh_computed") is not False
        or preflight.get("eol_computed") is not False
    ):
        raise RuntimeError("BIT structure-only preflight gate failed")

    if (
        development.get("status") != "PASS"
        or development.get("version") != config["version_labels"]["development"]
        or development.get("xjtu_only_development") is not True
        or development.get("hnei_performance_seen_by_workflow") is not False
        or development.get("bit_performance_seen_by_workflow") is not False
        or development.get("legacy_absolute_Ah_weights_included") is not False
    ):
        raise RuntimeError("XJTU-only SOH development freeze gate failed")

    if (
        hnei_decision.get("status") != "PASS_V0_2_0_UNCHANGED"
        or hnei_decision.get("decision") != "unchanged"
        or hnei_decision.get("bit_v0_2_0_structure_preflight_unlocked") is not True
    ):
        raise RuntimeError("HNEI unchanged receipt gate failed")

    if resolution.get("status") != "PASS_OFFSET20_FILENAME_SEMANTIC_MAPPING":
        raise RuntimeError("Offset-20 mapping resolution is not PASS")
    if resolution.get("capacity_read") is not False or resolution.get("model_output_generated") is not False:
        raise RuntimeError("Mapping resolution touched capacity or model output")
    if resolution.get("included_cells") != 72 or resolution.get("excluded_cell_ids") != ["BIT_#2"]:
        raise RuntimeError("Frozen BIT inclusion set changed")
    if resolution.get("cycle70_representable_for_all_included_cells") is not True:
        raise RuntimeError("Cycle 70 is not structurally representable")
    if resolution.get("cycle130_representable_for_any_included_cell") is not False:
        raise RuntimeError("Cycle 130 unexpectedly became representable")
    if resolution.get("cycle190_representable_for_any_included_cell") is not False:
        raise RuntimeError("Cycle 190 unexpectedly became representable")

    evidence_commit = config["amendment"]["mapping_evidence_commit"]
    if mapping_receipt.get("local_commit") != evidence_commit:
        raise RuntimeError("Mapping receipt local commit changed")
    if mapping_receipt.get("remote_commit") != evidence_commit:
        raise RuntimeError("Mapping receipt remote commit changed")
    if mapping_receipt.get("mapping_json_sha256") != sha256(args.mapping_resolution):
        raise RuntimeError("Mapping receipt JSON hash mismatch")

    if verification.get("status") != "PASS_V0_2_1_PRE_FREEZE_VERIFY":
        raise RuntimeError("v0.2.1 pre-freeze verification receipt is absent")
    if verification.get("verified_commit") != current_commit(args.project_root):
        raise RuntimeError("Verification receipt does not match current commit")
    if verification.get("capacity_opened") is not False:
        raise RuntimeError("Verification receipt indicates capacity access")
    if verification.get("model_output_generated") is not False:
        raise RuntimeError("Verification receipt indicates model output")

    if str(mapping["dataset"]["archive_sha256"]).lower() != str(preflight["source_sha256"]).lower():
        raise RuntimeError("BIT archive hash differs between mapping and preflight")
    if mapping["dataset"]["doi"] != "10.17632/kw34hhw7xg.3":
        raise RuntimeError("BIT dataset DOI/version changed")
    if mapping["approval"]["mapping_state"] != "READY_FOR_IMMUTABLE_FREEZE":
        raise RuntimeError("Schema mapping is not ready for immutable freeze")
    if mapping["landmark"]["primary_physical_cycle"] != 70:
        raise RuntimeError("Schema mapping primary landmark is not 70")
    if sorted(mapping["landmark"]["structurally_unavailable_cycles"]) != [130, 190]:
        raise RuntimeError("Schema mapping unavailable landmarks changed")
    if mapping["cell_schema"]["capacity_unit"] != "Ah":
        raise RuntimeError("BIT capacity unit must be Ah")
    if mapping["cell_schema"]["physical_aging_cycle_field"] != "derived_physical_cycle_v0_2_1_offset20":
        raise RuntimeError("Physical-cycle semantics changed")
    if mapping["normalization"]["formula"] != "SOH_i_t = C_i_t / median(C_i_first_5_valid)":
        raise RuntimeError("SOH formula changed")
    if mapping["future_information"] != {
        "post_cutoff_actual_current_schedule_visible": False,
        "post_cutoff_charge_rate_labels_visible": False,
        "true_data_termination_visible": False,
        "total_lifetime_visible": False,
        "future_eol_visible": False,
        "future_missingness_pattern_visible": False,
    }:
        raise RuntimeError("Unknown-future-condition prohibitions changed")

    mapping_manifest = resolve_relative(args.schema_mapping, mapping["resolved_structure"]["evaluation_manifest_csv"])
    pairing_manifest = resolve_relative(args.schema_mapping, mapping["resolved_structure"]["workbook_pairing_csv"])
    parser_source = resolve_relative(args.schema_mapping, mapping["parser"]["source_file"])
    for required in (mapping_manifest, pairing_manifest, parser_source):
        if not required.is_file():
            raise FileNotFoundError(required)

    if sha256(mapping_manifest) != mapping["resolved_structure"]["evaluation_manifest_sha256"]:
        raise RuntimeError("Evaluation manifest hash changed")
    if sha256(pairing_manifest) != mapping["resolved_structure"]["workbook_pairing_sha256"]:
        raise RuntimeError("Workbook-pairing manifest hash changed")
    if sha256(parser_source) != mapping["parser"]["source_sha256"]:
        raise RuntimeError("Frozen parser source hash changed")
    if mapping["parser"]["executed_on_capacity_values_before_freeze"] is not False:
        raise RuntimeError("Parser was marked as executed before freeze")
    if mapping["parser"]["imports_model_code"] is not False or mapping["parser"]["generates_model_output"] is not False:
        raise RuntimeError("Parser model gate failed")

    parser_text = parser_source.read_text(encoding="utf-8").lower()
    for forbidden in ("import torch", "from torch", "recursive_forecast", "model_variants"):
        if forbidden in parser_text:
            raise RuntimeError(f"Parser contains prohibited term: {forbidden}")
    guard_position = parser_text.find(
        "validate_registration_receipt(args.registration_receipt)"
    )
    archive_position = parser_text.find("zipfile.zipfile(args.archive")
    if (
        guard_position < 0
        or archive_position < 0
        or guard_position > archive_position
    ):
        raise RuntimeError("Parser authorization gate appears after archive open")

    rows = read_csv(mapping_manifest)
    if len(rows) != 73:
        raise RuntimeError("Evaluation manifest must contain 73 archive cells")
    included = [row for row in rows if as_bool(row["include_in_amended_BIT_evaluation"])]
    excluded = [row for row in rows if not as_bool(row["include_in_amended_BIT_evaluation"])]
    if len(included) != 72 or [row["cell_id"] for row in excluded] != ["BIT_#2"]:
        raise RuntimeError("Frozen inclusion/exclusion set changed")
    cohort_counts = {
        "arbitrary_use": sum(row["cohort"] == "arbitrary_use" for row in included),
        "fixed_profile": sum(row["cohort"] == "fixed_profile" for row in included),
    }
    if cohort_counts != {"arbitrary_use": 55, "fixed_profile": 17}:
        raise RuntimeError(f"Included cohort counts changed: {cohort_counts}")
    if not all(as_bool(row["cycle70_representable"]) for row in included):
        raise RuntimeError("Not every included cell represents cycle 70")
    if any(as_bool(row["cycle130_representable"]) for row in included):
        raise RuntimeError("Cycle 130 unexpectedly represented")
    if any(as_bool(row["cycle190_representable"]) for row in included):
        raise RuntimeError("Cycle 190 unexpectedly represented")

    if args.output_zip.exists():
        raise FileExistsError(f"BIT freeze already exists: {args.output_zip}")

    project_files = [
        path for path in sorted(args.project_root.rglob("*"))
        if path.is_file()
        and ".venv" not in path.parts
        and "__pycache__" not in path.parts
        and "results" not in path.parts
        and ".git" not in path.parts
        and ".v021_patch_backup" not in path.parts
    ]
    preflight_files = [path for path in sorted(args.preflight_dir.rglob("*")) if path.is_file()]

    manifest = {
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "version": config["version_labels"]["bit_freeze"],
        "governance_label": "post-BIT-structure-only, pre-model-evaluation amendment",
        "wording": config["registration"]["allowed_wording"],
        "dataset_doi": mapping["dataset"]["doi"],
        "dataset_version": int(mapping["dataset"]["version"]),
        "bit_archive_sha256": preflight["source_sha256"],
        "archive_structure_counts": {"total": 73, "arbitrary_use": 55, "fixed_profile": 18},
        "evaluation_inclusion_counts": {"total": 72, **cohort_counts},
        "excluded_cell_ids": ["BIT_#2"],
        "missing_official_cell_ids": [10, 13, 16, 19],
        "physical_cycle_mapping": config["bit"]["physical_cycle_mapping"],
        "primary_landmark": 70,
        "structurally_unavailable_landmarks": [130, 190],
        "minimum_future_observations_for_rmse": 5,
        "soh_formula": config["soh_definition"]["formula"],
        "eol_threshold_soh": 0.8,
        "unknown_future_conditions": True,
        "H1_then_H2_fixed_sequence": True,
        "ordinary_pooled_p_value_prohibited": True,
        "mapping_evidence_commit": evidence_commit,
        "frozen_commit": current_commit(args.project_root),
        "development_freeze_file": args.development_freeze.name,
        "development_freeze_sha256": sha256(args.development_freeze),
        "development_manifest_sha256": sha256(args.development_manifest),
        "hnei_decision_sha256": sha256(args.hnei_decision),
        "protocol_config_sha256": json_sha256(config),
        "schema_mapping_sha256": sha256(args.schema_mapping),
        "parser_source_sha256": sha256(parser_source),
        "evaluation_manifest_sha256": sha256(mapping_manifest),
        "workbook_pairing_sha256": sha256(pairing_manifest),
        "mapping_resolution_sha256": sha256(args.mapping_resolution),
        "mapping_receipt_sha256": sha256(args.mapping_receipt),
        "verification_receipt_sha256": sha256(args.verify_receipt),
        "structure_preflight_sha256": sha256(preflight_path),
        "bit_capacity_opened_before_freeze": False,
        "bit_model_imported_before_freeze": False,
        "bit_model_output_generated_before_freeze": False,
        "source_files": {
            str(path.relative_to(args.project_root)): sha256(path)
            for path in project_files
        },
        "preflight_files": {
            str(path.relative_to(args.preflight_dir)): sha256(path)
            for path in preflight_files
        },
    }

    manifest_path = args.output_zip.with_suffix(".manifest.json")
    write_json(manifest_path, manifest)
    args.output_zip.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output_zip.with_suffix(".zip.partial")
    with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.write(args.development_freeze, Path("development_freeze") / args.development_freeze.name)
        archive.write(args.development_manifest, "development_freeze/development_manifest.json")
        archive.write(args.hnei_decision, "hnei/hnei_version_decision.json")
        archive.write(args.schema_mapping, "bit/bit_schema_mapping_v0.2.1.yaml")
        archive.write(parser_source, "bit/frozen_bit_capacity_parser_v0.2.1.py")
        archive.write(mapping_manifest, "bit/bit_evaluation_manifest_v0.2.1.csv")
        archive.write(pairing_manifest, "bit/bit_workbook_pairing.csv")
        archive.write(args.mapping_resolution, "bit/mapping_resolution.json")
        archive.write(args.mapping_receipt, "bit/mapping_commit_receipt.txt")
        archive.write(args.verify_receipt, "bit/pre_freeze_verify_receipt.json")
        for path in preflight_files:
            archive.write(path, Path("bit/structure_preflight") / path.relative_to(args.preflight_dir))
        for path in project_files:
            archive.write(path, Path("code") / path.relative_to(args.project_root))
        archive.write(manifest_path, "PRE_BIT_MODEL_EVALUATION_FREEZE.json")

    partial.replace(args.output_zip)
    with zipfile.ZipFile(args.output_zip) as archive:
        bad = archive.testzip()
    if bad is not None:
        raise RuntimeError(f"BIT freeze ZIP CRC failed: {bad}")
    digest = sha256(args.output_zip)
    Path(str(args.output_zip) + ".sha256").write_text(
        f"{digest}  {args.output_zip.name}\n",
        encoding="utf-8",
    )
    print(f"[PASS] {config['version_labels']['bit_freeze']}")
    print(f"ZIP={args.output_zip}")
    print(f"SHA256={digest}")


if __name__ == "__main__":
    main()
