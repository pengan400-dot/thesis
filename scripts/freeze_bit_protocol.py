#!/usr/bin/env python3
"""Create the immutable pre-BIT model-evaluation freeze.

This script packages an already frozen XJTU-only SOH development artifact with
the structure-only BIT inventory and resolved schema. It imports no torch and
performs no prediction.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
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


def read_cell_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "cell_id",
        "cohort",
        "source_member",
        "structure_quality_pass",
        "physical_cycle_190_representable",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError(
            f"Cell manifest must contain {sorted(required)}: {path}"
        )
    if len({row["cell_id"] for row in rows}) != len(rows):
        raise ValueError("BIT structure-only cell manifest has duplicate cell IDs")
    cohorts = {row["cohort"] for row in rows}
    if not cohorts.issubset({"fixed_profile", "arbitrary_use"}):
        raise ValueError(f"Unresolved BIT cohorts: {sorted(cohorts)}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze BIT protocol before any BIT model output"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--schema-mapping", type=Path, required=True)
    parser.add_argument("--preflight-dir", type=Path, required=True)
    parser.add_argument("--development-freeze", type=Path, required=True)
    parser.add_argument("--development-manifest", type=Path, required=True)
    parser.add_argument("--hnei-decision", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-zip", type=Path, required=True)
    args = parser.parse_args()
    if "torch" in __import__("sys").modules:
        raise RuntimeError("torch must not be imported while freezing BIT")
    config = read_json(args.config)
    mapping_text = args.schema_mapping.read_text(encoding="utf-8")
    if PLACEHOLDER.search(mapping_text):
        raise RuntimeError(
            "BIT schema mapping is unresolved; run structure preflight and "
            "resolve every placeholder before freezing"
        )
    mapping = yaml.safe_load(mapping_text)
    preflight_path = args.preflight_dir / "bit_structural_preflight.json"
    preflight = read_json(preflight_path)
    development = read_json(args.development_manifest)
    hnei_decision = read_json(args.hnei_decision)
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
        or development.get("version")
        != config["version_labels"]["development"]
        or development.get("xjtu_only_development") is not True
        or development.get("hnei_performance_seen_by_workflow") is not False
        or development.get("bit_performance_seen_by_workflow") is not False
        or development.get("legacy_absolute_Ah_weights_included") is not False
    ):
        raise RuntimeError("XJTU-only SOH development freeze gate failed")
    if (
        hnei_decision.get("status") != "PASS_V0_2_0_UNCHANGED"
        or hnei_decision.get("decision") != "unchanged"
        or hnei_decision.get(
            "bit_v0_2_0_structure_preflight_unlocked"
        )
        is not True
    ):
        raise RuntimeError(
            "HNEI changed v0.2.0 or its decision receipt is missing; "
            "freeze a new XJTU-only v0.2.1 before BIT"
        )
    if (
        str(mapping["dataset"]["archive_sha256"]).lower()
        != str(preflight["source_sha256"]).lower()
    ):
        raise RuntimeError("BIT archive hash differs between mapping and preflight")
    if mapping["dataset"]["doi"] != "10.17632/kw34hhw7xg.3":
        raise RuntimeError("BIT dataset DOI/version changed")
    if mapping["cell_schema"]["capacity_unit"] != "Ah":
        raise RuntimeError("BIT capacity unit must be explicitly resolved as Ah")
    if (
        mapping["cell_schema"]["physical_aging_cycle_field"]
        in {None, ""}
        or mapping["landmark"]["index_semantics"]
        != "physical_aging_cycle_number"
    ):
        raise RuntimeError("Physical aging-cycle semantics are unresolved")
    if mapping["normalization"]["formula"] != (
        "SOH_i_t = C_i_t / median(C_i_first_5_valid)"
    ):
        raise RuntimeError("SOH formula changed in BIT mapping")
    if list(
        map(
            float,
            mapping["normalization"][
                "bol_median_quality_range_relative_to_declared_nominal"
            ],
        )
    ) != [0.8, 1.2]:
        raise RuntimeError("BOL median quality gate changed")
    if mapping["future_information"] != {
        "post_cutoff_actual_current_schedule_visible": False,
        "post_cutoff_charge_rate_labels_visible": False,
        "true_data_termination_visible": False,
        "total_lifetime_visible": False,
        "future_eol_visible": False,
        "future_missingness_pattern_visible": False,
    }:
        raise RuntimeError("Unknown-future-condition prohibitions changed")
    cell_manifest_value = Path(
        str(mapping["resolved_structure"]["cell_manifest_csv"])
    )
    cell_manifest = (
        cell_manifest_value
        if cell_manifest_value.is_absolute()
        else args.schema_mapping.parent / cell_manifest_value
    )
    parser_value = Path(str(mapping["parser"]["source_file"]))
    parser_source = (
        parser_value
        if parser_value.is_absolute()
        else args.schema_mapping.parent / parser_value
    )
    if (
        not parser_source.is_file()
        or str(mapping["parser"]["source_sha256"]).lower()
        != sha256(parser_source)
        or mapping["parser"][
            "executed_on_capacity_values_before_freeze"
        ]
        is not False
        or mapping["parser"]["imports_model_code"] is not False
        or mapping["parser"]["generates_model_output"] is not False
    ):
        raise RuntimeError("Frozen, unexecuted BIT parser gate failed")
    parser_text = parser_source.read_text(
        encoding="utf-8",
        errors="strict",
    ).lower()
    if any(
        forbidden in parser_text
        for forbidden in (
            "import torch",
            "from torch",
            "model_variants",
            "recursive_forecast",
            "rmse",
        )
    ):
        raise RuntimeError(
            "BIT parser contains model/evaluation terms and cannot be frozen"
        )
    cells = read_cell_manifest(cell_manifest)
    counts = {
        "total": len(cells),
        "fixed_profile": sum(
            row["cohort"] == "fixed_profile" for row in cells
        ),
        "arbitrary_use": sum(
            row["cohort"] == "arbitrary_use" for row in cells
        ),
    }
    declared = {
        "total": int(mapping["resolved_structure"]["actual_total_cells"]),
        "fixed_profile": int(
            mapping["resolved_structure"]["actual_fixed_profile_cells"]
        ),
        "arbitrary_use": int(
            mapping["resolved_structure"]["actual_arbitrary_use_cells"]
        ),
    }
    if counts != declared:
        raise RuntimeError(
            f"Resolved BIT counts differ: manifest={counts}, mapping={declared}"
        )
    if args.output_zip.exists():
        raise FileExistsError(f"BIT freeze already exists: {args.output_zip}")
    project_files = [
        path
        for path in sorted(args.project_root.rglob("*"))
        if path.is_file()
        and ".venv" not in path.parts
        and "__pycache__" not in path.parts
        and "results" not in path.parts
        and ".git" not in path.parts
    ]
    preflight_files = [
        path
        for path in sorted(args.preflight_dir.rglob("*"))
        if path.is_file()
    ]
    manifest = {
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "version": config["version_labels"]["bit_freeze"],
        "wording": config["registration"]["allowed_wording"],
        "dataset_doi": mapping["dataset"]["doi"],
        "dataset_version": int(mapping["dataset"]["version"]),
        "bit_archive_sha256": preflight["source_sha256"],
        "actual_structure_counts": counts,
        "physical_cycle_field": mapping["cell_schema"][
            "physical_aging_cycle_field"
        ],
        "capacity_field": mapping["cell_schema"]["capacity_field"],
        "capacity_unit": "Ah",
        "primary_cohort": "arbitrary_use",
        "key_secondary_cohort": "fixed_profile",
        "primary_landmark": 190,
        "minimum_future_observations_for_rmse": 5,
        "soh_formula": config["soh_definition"]["formula"],
        "eol_threshold_soh": 0.8,
        "unknown_future_conditions": True,
        "H1_then_H2_fixed_sequence": True,
        "ordinary_pooled_p_value_prohibited": True,
        "development_freeze_file": args.development_freeze.name,
        "development_freeze_sha256": sha256(args.development_freeze),
        "development_manifest_sha256": sha256(args.development_manifest),
        "hnei_decision_sha256": sha256(args.hnei_decision),
        "protocol_config_sha256": json_sha256(config),
        "schema_mapping_sha256": sha256(args.schema_mapping),
        "parser_source_sha256": sha256(parser_source),
        "cell_manifest_sha256": sha256(cell_manifest),
        "structure_preflight_sha256": sha256(preflight_path),
        "bit_model_imported_before_freeze": False,
        "bit_model_output_generated_before_freeze": False,
        "bit_rmse_computed_before_freeze": False,
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
    with zipfile.ZipFile(
        partial,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        archive.write(
            args.development_freeze,
            Path("development_freeze") / args.development_freeze.name,
        )
        archive.write(
            args.development_manifest,
            "development_freeze/development_manifest.json",
        )
        archive.write(args.hnei_decision, "hnei/hnei_version_decision.json")
        archive.write(args.schema_mapping, "bit/bit_schema_mapping.yaml")
        archive.write(parser_source, "bit/frozen_bit_parser.py")
        archive.write(cell_manifest, "bit/bit_structure_cell_manifest.csv")
        for path in preflight_files:
            archive.write(
                path,
                Path("bit/structure_preflight")
                / path.relative_to(args.preflight_dir),
            )
        for path in project_files:
            archive.write(
                path,
                Path("code") / path.relative_to(args.project_root),
            )
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
