#!/usr/bin/env python3
"""Structure-only BIT inventory.

This file deliberately has no torch or MSTT import. It inventories container
members, schemas, field shapes, missingness, cycle-index candidates, unit hints,
cohort hints, and whether the declared physical-cycle field can represent 190.
It cannot generate a model prediction or an error metric.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys
import tarfile
from collections.abc import Iterator
from typing import Any
import zipfile

import numpy as np


DOI = "10.17632/kw34hhw7xg.3"
MODEL_TERMS = frozenset(
    {
        "prediction",
        "predicted",
        "rmse",
        "mae",
        "model_error",
        "forecast_output",
    }
)
CYCLE_TERMS = ("cycle", "cyc", "aging_index", "aging_cycle")
CAPACITY_TERMS = (
    "capacity",
    "cap_ah",
    "discharge_capacity",
    "q_discharge",
)
CURRENT_TERMS = ("current", "charge_rate", "c_rate", "crate")
COHORT_TERMS = ("profile", "group", "cohort", "protocol", "use_type")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_name(name: str) -> bool:
    path = PurePosixPath(name.replace("\\", "/"))
    return not path.is_absolute() and ".." not in path.parts


def candidate_role(name: str) -> str:
    lowered = name.lower()
    if any(term in lowered for term in CYCLE_TERMS):
        return "physical_cycle_candidate"
    if any(term in lowered for term in CAPACITY_TERMS):
        return "capacity_candidate"
    if any(term in lowered for term in CURRENT_TERMS):
        return "current_or_rate_candidate"
    if any(term in lowered for term in COHORT_TERMS):
        return "cohort_candidate"
    return "other"


def cohort_hint(text: str) -> str:
    lowered = text.lower()
    if re.search(r"arbitrary|random|variable|varying", lowered):
        return "arbitrary_use"
    if re.search(r"fixed|constant", lowered):
        return "fixed_profile"
    return "unresolved"


def scalar_summary(value: Any) -> dict[str, Any]:
    array = np.asarray(value)
    result: dict[str, Any] = {
        "dtype": str(array.dtype),
        "shape": "x".join(str(part) for part in array.shape),
        "size": int(array.size),
        "missing_or_nonfinite": None,
        "numeric_min": None,
        "numeric_max": None,
    }
    if array.dtype.kind in "biufc" and array.size:
        numeric = np.asarray(array, dtype=float).reshape(-1)
        finite = np.isfinite(numeric)
        result["missing_or_nonfinite"] = int((~finite).sum())
        if finite.any():
            result["numeric_min"] = float(numeric[finite].min())
            result["numeric_max"] = float(numeric[finite].max())
    elif array.dtype.kind in "OSU":
        flattened = array.reshape(-1)
        result["missing_or_nonfinite"] = int(
            sum(value is None or str(value).strip() == "" for value in flattened)
        )
    return result


def field_summary(name: str, value: Any) -> dict[str, Any]:
    role = candidate_role(name)
    result = scalar_summary(value)
    if role != "physical_cycle_candidate":
        result["numeric_min"] = None
        result["numeric_max"] = None
    return {
        "field_path": name,
        **result,
        "candidate_role": role,
    }


def walk_object(
    value: Any,
    prefix: str,
    rows: list[dict[str, Any]],
    *,
    depth: int = 0,
    max_depth: int = 8,
) -> None:
    if depth > max_depth:
        rows.append(
            {
                "field_path": prefix,
                "dtype": "depth_limit",
                "shape": "",
                "size": 0,
                "missing_or_nonfinite": None,
                "numeric_min": None,
                "numeric_max": None,
                "candidate_role": candidate_role(prefix),
            }
        )
        return
    if isinstance(value, dict):
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            if str(key).startswith("__"):
                continue
            child = f"{prefix}.{key}" if prefix else str(key)
            walk_object(item, child, rows, depth=depth + 1)
        return
    if isinstance(value, (list, tuple)) and value and any(
        isinstance(item, (dict, list, tuple)) for item in value[:20]
    ):
        rows.append(
            {
                "field_path": prefix,
                "dtype": type(value).__name__,
                "shape": str(len(value)),
                "size": len(value),
                "missing_or_nonfinite": None,
                "numeric_min": None,
                "numeric_max": None,
                "candidate_role": candidate_role(prefix),
            }
        )
        for index, item in enumerate(value[:20]):
            walk_object(
                item,
                f"{prefix}[{index}]",
                rows,
                depth=depth + 1,
            )
        return
    rows.append(field_summary(prefix, value))


def inspect_csv(data: bytes, suffix: str) -> tuple[str, list[dict[str, Any]]]:
    import pandas as pd

    separator = "\t" if suffix in {".tsv", ".txt"} else ","
    frame = pd.read_csv(io.BytesIO(data), sep=separator)
    rows: list[dict[str, Any]] = []
    for column in frame.columns:
        rows.append(
            field_summary(str(column), frame[column].to_numpy())
        )
    return f"pandas_{suffix.lstrip('.')}", rows


def inspect_excel(data: bytes) -> tuple[str, list[dict[str, Any]]]:
    import pandas as pd

    workbook = pd.ExcelFile(io.BytesIO(data))
    rows: list[dict[str, Any]] = []
    for sheet in workbook.sheet_names:
        frame = pd.read_excel(workbook, sheet_name=sheet)
        for column in frame.columns:
            rows.append(
                field_summary(
                    f"{sheet}.{column}",
                    frame[column].to_numpy(),
                )
            )
    return "pandas_excel", rows


def inspect_mat(data: bytes) -> tuple[str, list[dict[str, Any]]]:
    import scipy.io

    rows: list[dict[str, Any]] = []
    try:
        payload = scipy.io.loadmat(
            io.BytesIO(data),
            simplify_cells=True,
        )
        walk_object(payload, "", rows)
        return "scipy_loadmat_simplify_cells", rows
    except (NotImplementedError, ValueError, OSError):
        try:
            import h5py
        except ImportError as exc:
            raise RuntimeError(
                "MATLAB v7.3/HDF5 file requires h5py"
            ) from exc
        with h5py.File(io.BytesIO(data), "r") as handle:
            def visitor(name: str, item: Any) -> None:
                if isinstance(item, h5py.Dataset):
                    rows.append(field_summary(name, item[()]))

            handle.visititems(visitor)
        return "h5py_mat73", rows


def inspect_json(data: bytes) -> tuple[str, list[dict[str, Any]]]:
    payload = json.loads(data.decode("utf-8"))
    rows: list[dict[str, Any]] = []
    walk_object(payload, "", rows)
    return "json", rows


def inspect_member(
    name: str,
    data: bytes,
) -> tuple[str, list[dict[str, Any]]]:
    suffix = Path(name).suffix.lower()
    if suffix in {".csv", ".tsv", ".txt"}:
        return inspect_csv(data, suffix)
    if suffix in {".xlsx", ".xls"}:
        return inspect_excel(data)
    if suffix == ".mat":
        return inspect_mat(data)
    if suffix == ".json":
        return inspect_json(data)
    return "binary_inventory_only", []


def source_sha256(source: Path) -> str:
    if source.is_dir():
        digest = hashlib.sha256()
        for path in sorted(source.rglob("*")):
            if not path.is_file():
                continue
            name = str(path.relative_to(source)).replace("\\", "/")
            digest.update(name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(bytes.fromhex(sha256_file(path)))
        return digest.hexdigest()
    return sha256_file(source)


def iter_members(source: Path) -> Iterator[tuple[str, bytes]]:
    if source.is_dir():
        for path in sorted(source.rglob("*")):
            if path.is_file():
                yield (
                    str(path.relative_to(source)).replace("\\", "/"),
                    path.read_bytes(),
                )
        return
    suffixes = "".join(source.suffixes).lower()
    if source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            if archive.testzip() is not None:
                raise RuntimeError("ZIP CRC test failed")
            for info in archive.infolist():
                if info.is_dir():
                    continue
                if not safe_name(info.filename):
                    raise RuntimeError(f"Unsafe archive member: {info.filename}")
                if info.flag_bits & 1:
                    raise RuntimeError(
                        f"Encrypted archive member: {info.filename}"
                    )
                data = archive.read(info)
                if Path(info.filename).suffix.lower() == ".zip":
                    with zipfile.ZipFile(io.BytesIO(data)) as nested:
                        if nested.testzip() is not None:
                            raise RuntimeError(
                                f"Nested ZIP CRC failed: {info.filename}"
                            )
                        for child in nested.infolist():
                            if child.is_dir():
                                continue
                            if (
                                not safe_name(child.filename)
                                or child.flag_bits & 1
                            ):
                                raise RuntimeError(
                                    "Unsafe/encrypted nested member: "
                                    f"{info.filename}!/{child.filename}"
                                )
                            yield (
                                f"{info.filename}!/{child.filename}",
                                nested.read(child),
                            )
                else:
                    yield info.filename, data
        return
    if suffixes.endswith((".tar", ".tar.gz", ".tgz")):
        with tarfile.open(source, "r:*") as archive:
            for info in archive.getmembers():
                if not info.isfile():
                    continue
                if not safe_name(info.name):
                    raise RuntimeError(f"Unsafe archive member: {info.name}")
                handle = archive.extractfile(info)
                if handle is None:
                    raise RuntimeError(f"Cannot read archive member: {info.name}")
                yield info.name, handle.read()
        return
    yield source.name, source.read_bytes()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BIT structure-only preflight; never generates model output"
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if "torch" in sys.modules:
        raise RuntimeError("torch must not be imported in BIT structural preflight")
    source_hash = source_sha256(args.source)
    file_rows: list[dict[str, Any]] = []
    field_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    member_count = 0
    for name, data in iter_members(args.source):
        member_count += 1
        if not safe_name(name):
            failures.append({"member": name, "error": "unsafe_path"})
            continue
        lowered_name = name.lower()
        source_member_output_like = any(
            term in lowered_name for term in MODEL_TERMS
        )
        try:
            parser_name, fields = inspect_member(name, data)
            cycle_fields = [
                row for row in fields
                if row["candidate_role"] == "physical_cycle_candidate"
            ]
            capacity_fields = [
                row for row in fields
                if row["candidate_role"] == "capacity_candidate"
            ]
            cycle_190 = "unknown"
            if len(cycle_fields) == 1:
                low = cycle_fields[0]["numeric_min"]
                high = cycle_fields[0]["numeric_max"]
                if low is not None and high is not None:
                    cycle_190 = str(float(low) <= 190 <= float(high)).lower()
            file_rows.append(
                {
                    "member": name,
                    "bytes": len(data),
                    "sha256": sha256_bytes(data),
                    "suffix": Path(name).suffix.lower(),
                    "parser": parser_name,
                    "field_count": len(fields),
                    "cycle_candidate_count": len(cycle_fields),
                    "capacity_candidate_count": len(capacity_fields),
                    "cohort_hint": cohort_hint(name),
                    "physical_cycle_190_representable": cycle_190,
                    "source_member_name_output_like": str(
                        source_member_output_like
                    ).lower(),
                    "model_output_generated": "no",
                }
            )
            for row in fields:
                field_rows.append({"member": name, **row})
        except Exception as exc:  # noqa: BLE001
            failures.append({"member": name, "error": str(exc)})
    write_csv(args.out / "bit_file_inventory.csv", file_rows)
    write_csv(args.out / "bit_field_inventory.csv", field_rows)
    recognized = [
        row for row in file_rows
        if int(row["capacity_candidate_count"]) > 0
        and int(row["cycle_candidate_count"]) > 0
    ]
    cohort_counts: dict[str, int] = {}
    for row in recognized:
        key = str(row["cohort_hint"])
        cohort_counts[key] = cohort_counts.get(key, 0) + 1
    status = "STRUCTURE_ONLY_PASS" if not failures else "NEEDS_REVIEW"
    report = {
        "status": status,
        "dataset_doi": DOI,
        "dataset_version": 3,
        "source": str(args.source.resolve()),
        "source_sha256": source_hash,
        "members": member_count,
        "schema_recognized_members": len(recognized),
        "cohort_hint_counts": cohort_counts,
        "official_expected_counts": {
            "total": 77,
            "fixed_profile": 22,
            "arbitrary_use": 55,
        },
        "actual_cell_counts_frozen": False,
        "physical_cycle_field_frozen": False,
        "capacity_unit_frozen": False,
        "structural_fields_read": True,
        "capacity_normalized": False,
        "numeric_capacity_values_summarized": False,
        "soh_computed": False,
        "eol_computed": False,
        "model_imported": False,
        "model_output_generated": False,
        "rmse_computed": False,
        "model_error_plot_generated": False,
        "failures": failures,
        "next_step": (
            "Resolve one physical-cycle field, one Ah capacity field, one "
            "physical-cell ID, and one fixed/arbitrary cohort field in "
            "bit_schema_mapping.yaml; do not run a model."
        ),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "bit_structural_preflight.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
