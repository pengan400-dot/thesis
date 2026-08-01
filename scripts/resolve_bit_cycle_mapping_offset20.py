#!/usr/bin/env python3
"""Resolve the BIT first20-to-later physical-cycle mapping from frozen structure evidence.

This script reads only previously generated structure-only audit outputs and the
archive naming/README audit text. It does not open XLSX workbooks, read capacity,
compute SOH/EOL, import model code, or generate model outputs.

Governance mapping adopted by this amendment:
- A workbook whose basename contains "first20cycle" or "first20cycles" is
  interpreted according to its explicit filename semantics as containing the
  first 20 completed physical cycles.
- In that workbook, local Cycle_Index 1..20 map to physical cycles 1..20.
- Local Cycle_Index 21 is retained in the audit trail as an unmapped terminal
  workbook bucket and is excluded from physical-landmark extraction.
- In the chronological later workbook, local Cycle_Index k maps to physical
  cycle 20 + k. Thus local 1 maps to physical 21.
- This is a transparent post-structure-only, pre-model-evaluation governance
  choice. It is not represented as recovered hidden ground truth.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys


FIRST20_RE = re.compile(
    r"^LR1865SZ_first20cycles?\d{6}_.+\.xlsx$",
    re.IGNORECASE,
)
LATER_RE = re.compile(
    r"^LR1865SZ_cycles?\d{6}_.+\.xlsx$",
    re.IGNORECASE,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual.lower() != expected.lower():
        raise RuntimeError(
            f"{label} SHA-256 mismatch: expected={expected}, actual={actual}"
        )
    return actual


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_number(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        result = float(text)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def as_int(value, label: str) -> int:
    number = as_number(value)
    if number is None or not float(number).is_integer():
        raise ValueError(f"{label} must be an integer, got {value!r}")
    return int(number)


def parse_datetime(value, label: str) -> datetime:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{label} is empty")
    for candidate in (text, text.replace("/", "-")):
        try:
            return datetime.fromisoformat(candidate)
        except ValueError:
            pass
    raise ValueError(f"{label} is not ISO-like: {value!r}")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workbook-inventory", type=Path, required=True)
    parser.add_argument("--cell-inventory", type=Path, required=True)
    parser.add_argument("--naming-audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-workbook-sha256", required=True)
    parser.add_argument("--expected-cell-sha256", required=True)
    parser.add_argument("--expected-naming-sha256", required=True)
    args = parser.parse_args()

    if "torch" in sys.modules:
        raise RuntimeError("torch must not be imported by this resolver")

    for path in (
        args.workbook_inventory,
        args.cell_inventory,
        args.naming_audit,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    input_hashes = {
        "workbook_inventory_sha256": require_sha(
            args.workbook_inventory,
            args.expected_workbook_sha256,
            "workbook inventory",
        ),
        "cell_inventory_sha256": require_sha(
            args.cell_inventory,
            args.expected_cell_sha256,
            "cell inventory",
        ),
        "naming_audit_sha256": require_sha(
            args.naming_audit,
            args.expected_naming_sha256,
            "naming audit",
        ),
    }

    workbooks = read_csv(args.workbook_inventory)
    cells = read_csv(args.cell_inventory)
    naming_text = args.naming_audit.read_text(
        encoding="utf-8", errors="replace"
    )

    failures: list[str] = []
    warnings: list[str] = []

    if len(workbooks) != 145:
        failures.append(f"Expected 145 workbook rows, found {len(workbooks)}")
    if len(cells) != 73:
        failures.append(f"Expected 73 cell rows, found {len(cells)}")

    workbook_by_key = {
        (row["cell_id"], row["file_role"]): row
        for row in workbooks
    }

    cell_ids = {row["cell_id"] for row in cells}
    if "BIT_#2" not in cell_ids:
        failures.append("Expected incomplete cell BIT_#2 is absent")

    mapping_rows: list[dict] = []
    complete_count = 0
    incomplete_count = 0

    for cell in sorted(cells, key=lambda row: int(row["cell_number"])):
        cell_id = cell["cell_id"]
        cell_number = int(cell["cell_number"])
        first20 = workbook_by_key.get((cell_id, "first20"))
        later = workbook_by_key.get((cell_id, "later"))

        if first20 is None or later is None:
            incomplete_count += 1
            if cell_id != "BIT_#2":
                failures.append(
                    f"Unexpected incomplete cell: {cell_id}"
                )
            mapping_rows.append(
                {
                    "cell_id": cell_id,
                    "cell_number": cell_number,
                    "cohort": cell["cohort"],
                    "include_in_amended_BIT_evaluation": False,
                    "exclusion_reason": "missing_later_workbook",
                    "first20_local_retained": "1..20",
                    "first20_local_excluded_terminal_bucket": "21",
                    "later_mapping_formula": "physical_cycle=20+local_cycle",
                    "later_local_min": "",
                    "later_local_max": "",
                    "mapped_physical_min": "",
                    "mapped_physical_max": "",
                    "cycle70_representable": False,
                    "cycle130_representable": False,
                    "cycle190_representable": False,
                }
            )
            continue

        complete_count += 1

        first_name = PurePosixPath(first20["member"]).name
        later_name = PurePosixPath(later["member"]).name

        if not FIRST20_RE.match(first_name):
            failures.append(
                f"{cell_id}: unexpected first20 basename {first_name}"
            )
        if not LATER_RE.match(later_name):
            failures.append(
                f"{cell_id}: unexpected later basename {later_name}"
            )

        if first_name not in naming_text:
            failures.append(
                f"{cell_id}: first20 basename absent from naming audit"
            )
        if later_name not in naming_text:
            failures.append(
                f"{cell_id}: later basename absent from naming audit"
            )

        first_min = as_int(
            first20["local_cycle_min"],
            f"{cell_id} first20 local minimum",
        )
        first_max = as_int(
            first20["local_cycle_max"],
            f"{cell_id} first20 local maximum",
        )
        later_min = as_int(
            later["local_cycle_min"],
            f"{cell_id} later local minimum",
        )
        later_max = as_int(
            later["local_cycle_max"],
            f"{cell_id} later local maximum",
        )

        if (first_min, first_max) != (1, 21):
            failures.append(
                f"{cell_id}: first20 local range is "
                f"{first_min}..{first_max}, expected 1..21"
            )
        if (later_min, later_max) != (1, 101):
            failures.append(
                f"{cell_id}: later local range is "
                f"{later_min}..{later_max}, expected 1..101"
            )

        first_end = parse_datetime(
            first20["datetime_max"],
            f"{cell_id} first20 datetime_max",
        )
        later_start = parse_datetime(
            later["datetime_min"],
            f"{cell_id} later datetime_min",
        )
        if later_start <= first_end:
            failures.append(
                f"{cell_id}: later workbook is not strictly chronological"
            )

        later_dp_min = as_number(later.get("data_point_min"))
        if later_dp_min != 1:
            failures.append(
                f"{cell_id}: later Data_Point minimum is "
                f"{later_dp_min}, expected reset to 1"
            )

        mapped_min = 20 + later_min
        mapped_max = 20 + later_max

        mapping_rows.append(
            {
                "cell_id": cell_id,
                "cell_number": cell_number,
                "cohort": cell["cohort"],
                "include_in_amended_BIT_evaluation": True,
                "exclusion_reason": "",
                "first20_local_retained": "1..20",
                "first20_local_excluded_terminal_bucket": "21",
                "later_mapping_formula": "physical_cycle=20+local_cycle",
                "later_local_min": later_min,
                "later_local_max": later_max,
                "mapped_physical_min": mapped_min,
                "mapped_physical_max": mapped_max,
                "cycle70_representable": mapped_min <= 70 <= mapped_max,
                "cycle130_representable": mapped_min <= 130 <= mapped_max,
                "cycle190_representable": mapped_min <= 190 <= mapped_max,
            }
        )

    if complete_count != 72:
        failures.append(
            f"Expected 72 complete cells, found {complete_count}"
        )
    if incomplete_count != 1:
        failures.append(
            f"Expected 1 incomplete cell, found {incomplete_count}"
        )

    included = [
        row
        for row in mapping_rows
        if row["include_in_amended_BIT_evaluation"] is True
    ]
    all_cycle70 = bool(
        included
        and all(row["cycle70_representable"] for row in included)
    )
    any_cycle130 = any(
        row["cycle130_representable"] for row in included
    )
    any_cycle190 = any(
        row["cycle190_representable"] for row in included
    )

    if not all_cycle70:
        failures.append(
            "Cycle 70 is not representable for every included cell"
        )
    if any_cycle130:
        failures.append(
            "Cycle 130 unexpectedly became representable"
        )
    if any_cycle190:
        failures.append(
            "Cycle 190 unexpectedly became representable"
        )

    cohort_counts: dict[str, int] = {}
    for row in included:
        cohort_counts[row["cohort"]] = (
            cohort_counts.get(row["cohort"], 0) + 1
        )

    if cohort_counts != {
        "arbitrary_use": 55,
        "fixed_profile": 17,
    }:
        failures.append(
            f"Unexpected included cohort counts: {cohort_counts}"
        )

    status = (
        "PASS_OFFSET20_FILENAME_SEMANTIC_MAPPING"
        if not failures
        else "BLOCKED_MAPPING_RESOLUTION"
    )

    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.out / "bit_cycle_mapping_manifest_v0.2.1.csv",
        mapping_rows,
    )

    audit = {
        "status": status,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "governance_label": (
            "post-BIT-structure-only, pre-model-evaluation amendment"
        ),
        "mapping_is_governance_choice_not_hidden_ground_truth": True,
        "mapping_rule": {
            "first20_filename_semantics": (
                "first 20 completed physical cycles"
            ),
            "first20_local_retained": "1..20 -> physical 1..20",
            "first20_local_21": (
                "excluded terminal workbook bucket; not used for "
                "physical-landmark extraction"
            ),
            "later_formula": "physical_cycle = 20 + local_cycle",
            "later_local_1_maps_to": 21,
            "maximum_mapped_physical_cycle": 121,
        },
        "input_hashes": input_hashes,
        "workbook_rows": len(workbooks),
        "physical_cell_rows": len(cells),
        "included_cells": len(included),
        "excluded_cells": 1,
        "excluded_cell_ids": ["BIT_#2"],
        "included_cohort_counts": cohort_counts,
        "cycle70_representable_for_all_included_cells": all_cycle70,
        "cycle130_representable_for_any_included_cell": any_cycle130,
        "cycle190_representable_for_any_included_cell": any_cycle190,
        "xlsx_opened": False,
        "capacity_read": False,
        "soh_computed": False,
        "eol_computed": False,
        "model_code_imported": False,
        "model_output_generated": False,
        "warnings": warnings,
        "failures": failures,
        "next_step": (
            "If PASS, review and commit this audit. Then update the "
            "v0.2.1 protocol/config/freeze gate and register the amendment "
            "before any BIT capacity parsing or model evaluation."
        ),
    }

    (args.out / "bit_cycle_mapping_resolution_v0.2.1.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(audit, ensure_ascii=False, indent=2))

    if status != "PASS_OFFSET20_FILENAME_SEMANTIC_MAPPING":
        raise SystemExit(5)


if __name__ == "__main__":
    main()
