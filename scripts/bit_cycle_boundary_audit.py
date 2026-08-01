#!/usr/bin/env python3
"""BIT workbook boundary audit using structure-only fields.

Allowed workbook fields:
- Data_Point
- Date_Time
- Cycle_Index

The script does not read Capacity(Ah), Current(A), Voltage(V), Temperature(℃),
or any model output. It imports no model code and computes no SOH, EOL, RMSE,
ranking, or prediction quantity.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import re
import statistics
import sys
import xml.etree.ElementTree as ET
import zipfile

from openpyxl.styles.numbers import is_date_format
from openpyxl.utils.datetime import from_excel


MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
DOC_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"

TARGET_SHEET = "记录表"
ALLOWED_HEADERS = ("Data_Point", "Date_Time", "Cycle_Index")
CELL_REF_RE = re.compile(r"([A-Z]+)([0-9]+)$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def excel_col_index(ref: str) -> int:
    match = CELL_REF_RE.match(ref)
    if not match:
        raise ValueError(f"Invalid cell reference: {ref}")
    letters = match.group(1)
    value = 0
    for letter in letters:
        value = value * 26 + ord(letter) - ord("A") + 1
    return value


def normalize_sheet_target(target: str) -> str:
    target = target.replace("\\", "/")
    if target.startswith("/"):
        return target.lstrip("/")
    if target.startswith("xl/"):
        return target
    return "xl/" + target


def read_shared_strings(inner: zipfile.ZipFile) -> list[str]:
    path = "xl/sharedStrings.xml"
    if path not in inner.namelist():
        return []
    root = ET.fromstring(inner.read(path))
    result: list[str] = []
    for item in root.findall(f"{{{MAIN_NS}}}si"):
        result.append(
            "".join(node.text or "" for node in item.iter(f"{{{MAIN_NS}}}t"))
        )
    return result


def read_date_style_flags(inner: zipfile.ZipFile) -> dict[int, bool]:
    path = "xl/styles.xml"
    if path not in inner.namelist():
        return {}

    root = ET.fromstring(inner.read(path))
    custom_formats: dict[int, str] = {}

    num_fmts = root.find(f"{{{MAIN_NS}}}numFmts")
    if num_fmts is not None:
        for item in num_fmts.findall(f"{{{MAIN_NS}}}numFmt"):
            custom_formats[int(item.attrib["numFmtId"])] = item.attrib.get(
                "formatCode", ""
            )

    cell_xfs = root.find(f"{{{MAIN_NS}}}cellXfs")
    if cell_xfs is None:
        return {}

    built_in_date_ids = set(range(14, 23)) | {45, 46, 47}
    flags: dict[int, bool] = {}

    for index, xf in enumerate(cell_xfs.findall(f"{{{MAIN_NS}}}xf")):
        num_fmt_id = int(xf.attrib.get("numFmtId", "0"))
        format_code = custom_formats.get(num_fmt_id, "")
        flags[index] = (
            num_fmt_id in built_in_date_ids
            or (bool(format_code) and is_date_format(format_code))
        )

    return flags


def workbook_sheet_metadata(inner: zipfile.ZipFile) -> tuple[str, bool]:
    workbook = ET.fromstring(inner.read("xl/workbook.xml"))
    workbook_pr = workbook.find(f"{{{MAIN_NS}}}workbookPr")
    date1904 = bool(
        workbook_pr is not None
        and workbook_pr.attrib.get("date1904", "").lower() in {"1", "true"}
    )

    relationship_id = None
    for sheet in workbook.findall(f".//{{{MAIN_NS}}}sheet"):
        if sheet.attrib.get("name") == TARGET_SHEET:
            relationship_id = sheet.attrib.get(f"{{{DOC_REL_NS}}}id")
            break

    if not relationship_id:
        raise ValueError(f"Worksheet not found: {TARGET_SHEET}")

    relationships = ET.fromstring(
        inner.read("xl/_rels/workbook.xml.rels")
    )
    target = None
    for rel in relationships.findall(f"{{{PKG_REL_NS}}}Relationship"):
        if rel.attrib.get("Id") == relationship_id:
            target = rel.attrib.get("Target")
            break

    if not target:
        raise ValueError(
            f"Worksheet relationship not found: {relationship_id}"
        )

    return normalize_sheet_target(target), date1904


def decode_cell(
    cell: ET.Element,
    shared_strings: list[str],
    date_flags: dict[int, bool],
    date1904: bool,
):
    cell_type = cell.attrib.get("t")
    style_index = int(cell.attrib.get("s", "0"))
    value_node = cell.find(f"{{{MAIN_NS}}}v")

    if cell_type == "inlineStr":
        return "".join(
            node.text or "" for node in cell.iter(f"{{{MAIN_NS}}}t")
        )

    if value_node is None or value_node.text is None:
        return None

    raw = value_node.text

    if cell_type == "s":
        return shared_strings[int(raw)]
    if cell_type in {"str", "e"}:
        return raw
    if cell_type == "b":
        return raw == "1"

    try:
        number = float(raw)
    except ValueError:
        return raw

    if date_flags.get(style_index, False):
        epoch = datetime(1904, 1, 1) if date1904 else datetime(1899, 12, 30)
        try:
            return from_excel(number, epoch=epoch)
        except Exception:
            return number

    return int(number) if number.is_integer() else number


def finite_number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def parse_datetime(value):
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value

    text = str(value).strip()
    candidates = (text, text.replace("/", "-"))
    for candidate in candidates:
        try:
            return datetime.fromisoformat(candidate)
        except ValueError:
            pass
    return None


def datetime_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def cell_number_from_member(member: str) -> int:
    match = re.search(r"/#(\d+)/", member)
    if not match:
        raise ValueError(f"Cell directory token not found: {member}")
    return int(match.group(1))


def cohort_from_member(member: str) -> str:
    if "/Cycled with Arbitrary Uses Profiles/" in member:
        return "arbitrary_use"
    if "/Cycled with Fixed Current Profiles/" in member:
        return "fixed_profile"
    return "unresolved"


def role_from_member(member: str) -> str:
    return "first20" if "first20" in member.lower() else "later"


def safe_median(values):
    numbers = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return statistics.median(numbers) if numbers else None


def inspect_member(task: tuple[str, str]) -> dict:
    archive_text, member = task
    archive = Path(archive_text)

    with zipfile.ZipFile(archive) as outer:
        workbook_bytes = outer.read(member)

    with zipfile.ZipFile(BytesIO(workbook_bytes)) as inner:
        sheet_path, date1904 = workbook_sheet_metadata(inner)
        shared_strings = read_shared_strings(inner)
        date_flags = read_date_style_flags(inner)

        header_row = None
        selected_columns: dict[str, int] = {}
        cycle_stats: dict[int, dict] = {}
        rows_seen = 0
        selected_rows = 0
        overall_datetime_min = None
        overall_datetime_max = None
        data_point_min = None
        data_point_max = None

        with inner.open(sheet_path) as sheet_handle:
            for _, element in ET.iterparse(sheet_handle, events=("end",)):
                if element.tag != f"{{{MAIN_NS}}}row":
                    continue

                row_number = int(element.attrib.get("r", "0"))
                rows_seen += 1

                if header_row is None:
                    decoded: dict[int, object] = {}
                    for cell in element.findall(f"{{{MAIN_NS}}}c"):
                        column = excel_col_index(cell.attrib.get("r", ""))
                        decoded[column] = decode_cell(
                            cell, shared_strings, date_flags, date1904
                        )

                    header_map = {
                        str(value).strip(): column
                        for column, value in decoded.items()
                        if value is not None
                    }

                    if all(name in header_map for name in ALLOWED_HEADERS):
                        header_row = row_number
                        selected_columns = {
                            name: header_map[name]
                            for name in ALLOWED_HEADERS
                        }

                    element.clear()

                    if row_number >= 30 and header_row is None:
                        raise ValueError(
                            "Required allowed headers were not found in "
                            f"the first 30 rows: {member}"
                        )
                    continue

                selected_values: dict[str, object] = {}
                for cell in element.findall(f"{{{MAIN_NS}}}c"):
                    column = excel_col_index(cell.attrib.get("r", ""))
                    for field_name, wanted_column in selected_columns.items():
                        if column == wanted_column:
                            selected_values[field_name] = decode_cell(
                                cell, shared_strings, date_flags, date1904
                            )
                            break

                element.clear()

                cycle_value = finite_number(
                    selected_values.get("Cycle_Index")
                )
                if cycle_value is None:
                    continue

                cycle_number = int(round(cycle_value))
                selected_rows += 1

                data_point = finite_number(
                    selected_values.get("Data_Point")
                )
                date_time = parse_datetime(
                    selected_values.get("Date_Time")
                )

                stats = cycle_stats.setdefault(
                    cycle_number,
                    {
                        "rows": 0,
                        "data_point_min": None,
                        "data_point_max": None,
                        "datetime_min": None,
                        "datetime_max": None,
                    },
                )
                stats["rows"] += 1

                if data_point is not None:
                    stats["data_point_min"] = (
                        data_point
                        if stats["data_point_min"] is None
                        else min(stats["data_point_min"], data_point)
                    )
                    stats["data_point_max"] = (
                        data_point
                        if stats["data_point_max"] is None
                        else max(stats["data_point_max"], data_point)
                    )
                    data_point_min = (
                        data_point
                        if data_point_min is None
                        else min(data_point_min, data_point)
                    )
                    data_point_max = (
                        data_point
                        if data_point_max is None
                        else max(data_point_max, data_point)
                    )

                if date_time is not None:
                    stats["datetime_min"] = (
                        date_time
                        if stats["datetime_min"] is None
                        else min(stats["datetime_min"], date_time)
                    )
                    stats["datetime_max"] = (
                        date_time
                        if stats["datetime_max"] is None
                        else max(stats["datetime_max"], date_time)
                    )
                    overall_datetime_min = (
                        date_time
                        if overall_datetime_min is None
                        else min(overall_datetime_min, date_time)
                    )
                    overall_datetime_max = (
                        date_time
                        if overall_datetime_max is None
                        else max(overall_datetime_max, date_time)
                    )

    if header_row is None:
        raise ValueError(f"Header row unresolved: {member}")
    if not cycle_stats:
        raise ValueError(f"No allowed cycle rows found: {member}")

    cycles = sorted(cycle_stats)
    first_cycle = cycles[0]
    last_cycle = cycles[-1]

    prior_row_counts = [
        cycle_stats[cycle]["rows"] for cycle in cycles[:-1]
    ]
    middle_row_counts = [
        cycle_stats[cycle]["rows"] for cycle in cycles[1:-1]
    ]
    if not middle_row_counts:
        middle_row_counts = [
            cycle_stats[cycle]["rows"] for cycle in cycles
        ]

    prior_median = safe_median(prior_row_counts)
    middle_median = safe_median(middle_row_counts)
    first_rows = cycle_stats[first_cycle]["rows"]
    last_rows = cycle_stats[last_cycle]["rows"]

    return {
        "member": member,
        "member_sha256": hashlib.sha256(workbook_bytes).hexdigest(),
        "cell_number": cell_number_from_member(member),
        "cell_id": f"BIT_#{cell_number_from_member(member)}",
        "cohort": cohort_from_member(member),
        "file_role": role_from_member(member),
        "worksheet": TARGET_SHEET,
        "header_row": header_row,
        "rows_seen": rows_seen,
        "selected_rows": selected_rows,
        "unique_local_cycles": len(cycles),
        "local_cycle_min": first_cycle,
        "local_cycle_max": last_cycle,
        "first_cycle_rows": first_rows,
        "last_cycle_rows": last_rows,
        "median_prior_cycle_rows": prior_median,
        "median_middle_cycle_rows": middle_median,
        "last_cycle_fraction_of_prior_median": (
            last_rows / prior_median if prior_median else None
        ),
        "first_cycle_fraction_of_middle_median": (
            first_rows / middle_median if middle_median else None
        ),
        "data_point_min": data_point_min,
        "data_point_max": data_point_max,
        "datetime_min": datetime_text(overall_datetime_min),
        "datetime_max": datetime_text(overall_datetime_max),
        "first_cycle_datetime_min": datetime_text(
            cycle_stats[first_cycle]["datetime_min"]
        ),
        "first_cycle_datetime_max": datetime_text(
            cycle_stats[first_cycle]["datetime_max"]
        ),
        "last_cycle_datetime_min": datetime_text(
            cycle_stats[last_cycle]["datetime_min"]
        ),
        "last_cycle_datetime_max": datetime_text(
            cycle_stats[last_cycle]["datetime_max"]
        ),
        "allowed_fields_only": True,
        "outcome_fields_read": False,
        "model_code_imported": False,
        "model_output_generated": False,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    field_names = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--file-inventory", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    if "torch" in sys.modules:
        raise RuntimeError("torch must not be imported during this audit")
    if not args.archive.is_file():
        raise FileNotFoundError(args.archive)
    if not args.file_inventory.is_file():
        raise FileNotFoundError(args.file_inventory)
    if not 1 <= args.workers <= 8:
        raise ValueError("--workers must be between 1 and 8")

    with args.file_inventory.open(
        newline="", encoding="utf-8"
    ) as handle:
        inventory_rows = list(csv.DictReader(handle))

    members = sorted(
        row["member"]
        for row in inventory_rows
        if row.get("suffix", "").lower() == ".xlsx"
    )

    if len(members) != 145:
        raise RuntimeError(
            f"Expected 145 XLSX members, found {len(members)}"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    partial_jsonl = (
        args.out / "workbook_boundary_inventory.partial.jsonl"
    )

    results: list[dict] = []
    failures: list[dict] = []

    with partial_jsonl.open("w", encoding="utf-8") as progress:
        with ProcessPoolExecutor(
            max_workers=args.workers
        ) as executor:
            futures = {
                executor.submit(
                    inspect_member, (str(args.archive), member)
                ): member
                for member in members
            }

            for completed, future in enumerate(
                as_completed(futures), start=1
            ):
                member = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    progress.write(
                        json.dumps(
                            result, ensure_ascii=False
                        ) + "\n"
                    )
                    progress.flush()
                except Exception as error:
                    failures.append(
                        {
                            "member": member,
                            "error_type": type(error).__name__,
                            "error": str(error),
                        }
                    )

                print(
                    f"[PROGRESS] {completed}/{len(members)} "
                    f"failures={len(failures)}",
                    flush=True,
                )

    results.sort(
        key=lambda row: (
            int(row["cell_number"]),
            row["file_role"],
        )
    )
    write_csv(
        args.out / "workbook_boundary_inventory.csv",
        results,
    )
    write_csv(
        args.out / "boundary_failures.csv",
        failures,
    )

    grouped: dict[int, list[dict]] = {}
    for row in results:
        grouped.setdefault(
            int(row["cell_number"]), []
        ).append(row)

    cell_rows: list[dict] = []
    decision_counts = {
        "boundary_fragment_continuation": 0,
        "append_after_last_cycle": 0,
        "unresolved": 0,
        "incomplete": 0,
    }

    for cell_number in sorted(grouped):
        group = grouped[cell_number]
        first20 = next(
            (
                row
                for row in group
                if row["file_role"] == "first20"
            ),
            None,
        )
        later = next(
            (
                row
                for row in group
                if row["file_role"] == "later"
            ),
            None,
        )

        if first20 is None or later is None:
            decision = "incomplete"
            decision_counts[decision] += 1
            cell_rows.append(
                {
                    "cell_id": f"BIT_#{cell_number}",
                    "cell_number": cell_number,
                    "cohort": group[0]["cohort"],
                    "first20_member": (
                        first20["member"] if first20 else ""
                    ),
                    "later_member": (
                        later["member"] if later else ""
                    ),
                    "boundary_time_order_pass": False,
                    "boundary_gap_seconds": "",
                    "first20_last_cycle": (
                        first20["local_cycle_max"]
                        if first20 else ""
                    ),
                    "first20_last_cycle_rows": (
                        first20["last_cycle_rows"]
                        if first20 else ""
                    ),
                    "first20_last_cycle_fraction_of_prior_median": (
                        first20[
                            "last_cycle_fraction_of_prior_median"
                        ]
                        if first20 else ""
                    ),
                    "later_first_cycle": (
                        later["local_cycle_min"]
                        if later else ""
                    ),
                    "later_first_cycle_rows": (
                        later["first_cycle_rows"]
                        if later else ""
                    ),
                    "later_first_cycle_fraction_of_middle_median": (
                        later[
                            "first_cycle_fraction_of_middle_median"
                        ]
                        if later else ""
                    ),
                    "mapping_decision": decision,
                    "later_cycle_offset": "",
                    "maximum_mapped_physical_cycle": "",
                    "physical_cycle_70_representable": False,
                    "physical_cycle_130_representable": False,
                    "physical_cycle_190_representable": False,
                    "decision_rule": (
                        "missing_first20_or_later_workbook"
                    ),
                }
            )
            continue

        first20_end = parse_datetime(
            first20["datetime_max"]
        )
        later_start = parse_datetime(
            later["datetime_min"]
        )
        time_order_pass = bool(
            first20_end is not None
            and later_start is not None
            and later_start >= first20_end
        )
        gap_seconds = (
            (later_start - first20_end).total_seconds()
            if time_order_pass
            and first20_end is not None
            and later_start is not None
            else None
        )

        terminal_fraction = first20[
            "last_cycle_fraction_of_prior_median"
        ]
        initial_fraction = later[
            "first_cycle_fraction_of_middle_median"
        ]

        if (
            time_order_pass
            and terminal_fraction is not None
            and initial_fraction is not None
            and terminal_fraction <= 0.25
            and initial_fraction >= 0.50
        ):
            decision = "boundary_fragment_continuation"
            offset = int(first20["local_cycle_max"]) - 1
            rule = (
                "first20 terminal cycle <=25% of prior median "
                "rows and later first cycle >=50% of later "
                "middle-cycle median rows"
            )
        elif (
            time_order_pass
            and terminal_fraction is not None
            and terminal_fraction >= 0.75
        ):
            decision = "append_after_last_cycle"
            offset = int(first20["local_cycle_max"])
            rule = (
                "first20 terminal cycle >=75% of prior median rows"
            )
        else:
            decision = "unresolved"
            offset = None
            rule = (
                "boundary evidence did not satisfy the "
                "deterministic structural rule"
            )

        decision_counts[decision] += 1
        maximum_physical_cycle = (
            offset + int(later["local_cycle_max"])
            if offset is not None
            else None
        )

        cell_rows.append(
            {
                "cell_id": f"BIT_#{cell_number}",
                "cell_number": cell_number,
                "cohort": first20["cohort"],
                "first20_member": first20["member"],
                "later_member": later["member"],
                "boundary_time_order_pass": time_order_pass,
                "boundary_gap_seconds": (
                    gap_seconds
                    if gap_seconds is not None
                    else ""
                ),
                "first20_last_cycle": (
                    first20["local_cycle_max"]
                ),
                "first20_last_cycle_rows": (
                    first20["last_cycle_rows"]
                ),
                "first20_last_cycle_fraction_of_prior_median": (
                    terminal_fraction
                ),
                "later_first_cycle": later["local_cycle_min"],
                "later_first_cycle_rows": (
                    later["first_cycle_rows"]
                ),
                "later_first_cycle_fraction_of_middle_median": (
                    initial_fraction
                ),
                "mapping_decision": decision,
                "later_cycle_offset": (
                    offset if offset is not None else ""
                ),
                "maximum_mapped_physical_cycle": (
                    maximum_physical_cycle
                    if maximum_physical_cycle is not None
                    else ""
                ),
                "physical_cycle_70_representable": bool(
                    maximum_physical_cycle is not None
                    and maximum_physical_cycle >= 70
                ),
                "physical_cycle_130_representable": bool(
                    maximum_physical_cycle is not None
                    and maximum_physical_cycle >= 130
                ),
                "physical_cycle_190_representable": bool(
                    maximum_physical_cycle is not None
                    and maximum_physical_cycle >= 190
                ),
                "decision_rule": rule,
            }
        )

    write_csv(
        args.out / "cell_boundary_inventory.csv",
        cell_rows,
    )

    resolved = [
        row
        for row in cell_rows
        if row["mapping_decision"]
        in {
            "boundary_fragment_continuation",
            "append_after_last_cycle",
        }
    ]
    resolved_offsets = sorted(
        {
            int(row["later_cycle_offset"])
            for row in resolved
        }
    )

    all_complete_cells_resolved = bool(
        len(cell_rows) == 73
        and decision_counts["incomplete"] == 1
        and decision_counts["unresolved"] == 0
        and len(resolved) == 72
    )
    cycle70_for_all_resolved = bool(
        resolved
        and all(
            bool(row["physical_cycle_70_representable"])
            for row in resolved
        )
    )

    status = (
        "PASS_CYCLE70_STRUCTURALLY_RESOLVED"
        if all_complete_cells_resolved
        and cycle70_for_all_resolved
        and not failures
        else "REVIEW_REQUIRED"
    )

    audit = {
        "status": status,
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "archive": str(args.archive),
        "archive_sha256": sha256_file(args.archive),
        "file_inventory": str(args.file_inventory),
        "file_inventory_sha256": sha256_file(
            args.file_inventory
        ),
        "workers": args.workers,
        "xlsx_members_expected": 145,
        "xlsx_members_completed": len(results),
        "failures": failures,
        "physical_cells": len(cell_rows),
        "mapping_decision_counts": decision_counts,
        "resolved_offsets": resolved_offsets,
        "all_complete_cells_mapping_resolved": (
            all_complete_cells_resolved
        ),
        "cycle70_representable_for_all_resolved_complete_cells": (
            cycle70_for_all_resolved
        ),
        "cycle130_representable_for_any_resolved_cell": any(
            bool(row["physical_cycle_130_representable"])
            for row in resolved
        ),
        "cycle190_representable_for_any_resolved_cell": any(
            bool(row["physical_cycle_190_representable"])
            for row in resolved
        ),
        "allowed_fields": list(ALLOWED_HEADERS),
        "outcome_fields_read": False,
        "model_code_imported": False,
        "model_output_generated": False,
        "next_step": (
            "Review cell_boundary_inventory.csv. Do not freeze "
            "or evaluate a model unless the status is "
            "PASS_CYCLE70_STRUCTURALLY_RESOLVED and the mapping "
            "rule is accepted in a separately registered amendment."
        ),
    }

    (args.out / "cycle_boundary_audit.json").write_text(
        json.dumps(
            audit,
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    partial_jsonl.replace(
        args.out / "workbook_boundary_inventory.jsonl"
    )

    print(
        json.dumps(
            audit,
            ensure_ascii=False,
            indent=2,
        )
    )

    if status != "PASS_CYCLE70_STRUCTURALLY_RESOLVED":
        raise SystemExit(4)


if __name__ == "__main__":
    main()
