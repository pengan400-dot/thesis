from __future__ import annotations

import io
import json
import math
import pickle
import re
import zipfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from mstt_soh.pipeline import causal_smooth
from mstt_soh.soh import bol_reference_from_values

from .common import (
    assert_hash,
    json_sha256,
    load_hust_protocol,
    normalized_zip_name,
    read_json,
    sha256_bytes,
    sha256_file,
    stable_cell_order,
    utc_now,
    write_json,
)


CELL_ID_PATTERN = re.compile(r"^[0-9]+-[0-9]+$")


def _pickle_members(archive: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    normalized_seen: set[str] = set()
    with zipfile.ZipFile(archive, "r") as handle:
        corrupt = handle.testzip()
        if corrupt is not None:
            raise RuntimeError(f"ZIP CRC gate failed at {corrupt}")
        for info in handle.infolist():
            if info.is_dir():
                continue
            normalized = normalized_zip_name(info.filename)
            if normalized in normalized_seen:
                raise ValueError(f"Duplicate normalized ZIP member: {normalized}")
            normalized_seen.add(normalized)
            if not normalized.lower().endswith(".pkl"):
                continue
            if Path(normalized).parent.as_posix() != "our_data":
                raise ValueError(
                    "Unexpected HUST pickle location; expected our_data/*.pkl, "
                    f"found {normalized}"
                )
            cell_id = Path(normalized).stem
            if not CELL_ID_PATTERN.fullmatch(cell_id):
                raise ValueError(
                    f"Unexpected HUST pickle cell ID {cell_id!r} in {normalized}"
                )
            rows.append(
                {
                    "cell_id": cell_id,
                    "archive_member": info.filename,
                    "normalized_member": normalized,
                    "compressed_size_bytes": int(info.compress_size),
                    "uncompressed_size_bytes": int(info.file_size),
                    "crc32": f"{int(info.CRC):08x}",
                }
            )
    if len({str(row["cell_id"]) for row in rows}) != len(rows):
        raise ValueError("Duplicate HUST physical-cell ID")
    return sorted(rows, key=lambda row: str(row["cell_id"]))


def inventory_source(archive: Path, config: Path, output: Path) -> None:
    protocol = load_hust_protocol(config)
    archive = Path(archive).resolve()
    if not archive.is_file():
        raise FileNotFoundError(archive)
    output = Path(output)
    if (output / "source_receipt.json").exists():
        raise RuntimeError(
            "HUST structure inventory is already frozen in this run root; "
            "reuse it instead of overwriting it"
        )
    output.mkdir(parents=True, exist_ok=True)
    members = _pickle_members(archive)
    expected = int(protocol["dataset"]["expected_physical_cells"])
    if len(members) != expected:
        raise RuntimeError(f"Expected {expected} HUST pickle cells, found {len(members)}")
    salt = str(protocol["split"]["sha256_salt"])
    ordered = stable_cell_order(
        [str(row["cell_id"]) for row in members],
        salt,
    )
    calibration_count = int(protocol["split"]["calibration_cells"])
    role_by_cell: dict[str, dict[str, object]] = {}
    for rank, (cell_id, split_hash) in enumerate(ordered, start=1):
        role_by_cell[cell_id] = {
            "split_rank": rank,
            "split_sha256": split_hash,
            "role": "calibration" if rank <= calibration_count else "confirmation",
        }
    rows = []
    for row in members:
        rows.append({**row, **role_by_cell[str(row["cell_id"])]})
    frame = pd.DataFrame(rows).sort_values("split_rank").reset_index(drop=True)
    counts = frame["role"].value_counts().to_dict()
    expected_counts = {
        "calibration": int(protocol["split"]["calibration_cells"]),
        "confirmation": int(protocol["split"]["confirmation_cells"]),
    }
    if counts != expected_counts:
        raise RuntimeError(f"Split count gate failed: {counts}")
    inventory_path = output / "HUST_source_inventory.csv"
    split_path = output / "HUST_split_manifest.csv"
    frame.drop(columns=["role", "split_rank", "split_sha256"]).sort_values(
        "cell_id"
    ).to_csv(inventory_path, index=False)
    frame[
        [
            "cell_id",
            "role",
            "split_rank",
            "split_sha256",
            "archive_member",
            "normalized_member",
        ]
    ].to_csv(split_path, index=False)
    receipt = {
        "status": "PASS",
        "created_utc": utc_now(),
        "analysis_stage": "structure_only_no_pickle_unpickled",
        "config_sha256": json_sha256(protocol),
        "archive_path_recorded": str(archive),
        "archive_name": archive.name,
        "archive_sha256": sha256_file(archive),
        "archive_size_bytes": archive.stat().st_size,
        "physical_cells": len(frame),
        "role_counts": expected_counts,
        "inventory_sha256": sha256_file(inventory_path),
        "split_manifest_sha256": sha256_file(split_path),
        "pickle_unpickled": False,
        "capacity_value_read": False,
        "soh_computed": False,
        "eol_computed": False,
        "model_output_computed": False,
        "source_security_note": protocol["dataset"]["pickle_security_note"],
    }
    write_json(output / "source_receipt.json", receipt)
    print(
        "[PASS] HUST structure inventory: "
        f"cells={len(frame)} calibration=20 confirmation=57 "
        f"archive_sha256={receipt['archive_sha256']}"
    )


def load_source_gate(
    archive: Path,
    config: Path,
    inventory_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    protocol = load_hust_protocol(config)
    receipt = read_json(Path(inventory_dir) / "source_receipt.json")
    split_path = Path(inventory_dir) / "HUST_split_manifest.csv"
    inventory_path = Path(inventory_dir) / "HUST_source_inventory.csv"
    if (
        receipt.get("status") != "PASS"
        or receipt.get("analysis_stage") != "structure_only_no_pickle_unpickled"
        or receipt.get("config_sha256") != json_sha256(protocol)
        or receipt.get("pickle_unpickled") is not False
        or receipt.get("capacity_value_read") is not False
    ):
        raise RuntimeError("HUST structure-only receipt gate failed")
    assert_hash(Path(archive), str(receipt["archive_sha256"]), "HUST source archive")
    assert_hash(split_path, str(receipt["split_manifest_sha256"]), "HUST split")
    assert_hash(inventory_path, str(receipt["inventory_sha256"]), "HUST inventory")
    split = pd.read_csv(split_path, dtype={"cell_id": str})
    if split["cell_id"].nunique() != 77:
        raise RuntimeError("HUST split no longer contains 77 unique cells")
    counts = split["role"].value_counts().to_dict()
    if counts != {"confirmation": 57, "calibration": 20}:
        raise RuntimeError(f"HUST split counts changed: {counts}")
    return protocol, receipt, split


def extract_xjtu_b1_b3(source_zip: Path, output: Path) -> None:
    source_zip = Path(source_zip)
    output = Path(output)
    curves = output / "curves"
    curves.mkdir(parents=True, exist_ok=True)
    written: list[dict[str, object]] = []
    with zipfile.ZipFile(source_zip, "r") as handle:
        for info in handle.infolist():
            if info.is_dir():
                continue
            normalized = normalized_zip_name(info.filename)
            basename = Path(normalized).name
            if not basename.lower().endswith(".csv"):
                continue
            if not re.match(r"^B[13]_", basename, flags=re.IGNORECASE):
                continue
            data = handle.read(info)
            destination = curves / basename
            destination.write_bytes(data)
            written.append(
                {
                    "file": basename,
                    "size_bytes": len(data),
                    "sha256": sha256_bytes(data),
                }
            )
    if len(written) != 16:
        raise RuntimeError(f"Expected 16 XJTU B1+B3 CSVs, extracted {len(written)}")
    write_json(
        output / "xjtu_input_receipt.json",
        {
            "status": "PASS",
            "created_utc": utc_now(),
            "source_zip": source_zip.name,
            "source_zip_sha256": sha256_file(source_zip),
            "files": sorted(written, key=lambda row: str(row["file"])),
            "external_hust_data_loaded": False,
        },
    )
    print(f"[PASS] extracted {len(written)} frozen XJTU B1+B3 development CSVs")


def _column(frame: pd.DataFrame, exact: str) -> str:
    if exact in frame.columns:
        return exact
    lowered = {str(column).strip().lower(): str(column) for column in frame.columns}
    candidate = lowered.get(exact.strip().lower())
    if candidate is None:
        raise ValueError(f"Missing required HUST timeseries column: {exact}")
    return candidate


def discharge_capacity_Ah(frame: pd.DataFrame) -> float:
    current = pd.to_numeric(frame[_column(frame, "Current (mA)")], errors="coerce")
    times = pd.to_numeric(frame[_column(frame, "Time (s)")], errors="coerce")
    voltage = pd.to_numeric(frame[_column(frame, "Voltage (V)")], errors="coerce")
    if not (np.isfinite(current).all() and np.isfinite(times).all() and np.isfinite(voltage).all()):
        raise ValueError("HUST cycle contains a non-finite current, time, or voltage sample")
    current_A = current.to_numpy(float) / 1000.0
    time_s = times.to_numpy(float)
    if len(time_s) < 2:
        raise ValueError("HUST cycle has fewer than two finite samples")
    delta = np.diff(time_s)
    if np.any(delta < -1e-9):
        raise ValueError("HUST cycle time is not nondecreasing")
    increments = np.maximum(-current_A[1:], 0.0) * np.maximum(delta, 0.0) / 3600.0
    capacity = float(np.sum(increments))
    if not math.isfinite(capacity) or capacity <= 0:
        raise ValueError(f"Invalid integrated discharge capacity: {capacity}")
    return capacity


def _cell_record(payload: object, cell_id: str) -> Mapping[str, object]:
    if not isinstance(payload, Mapping):
        raise ValueError("HUST pickle top level is not a mapping")
    if cell_id in payload:
        record = payload[cell_id]
    elif len(payload) == 1:
        record = next(iter(payload.values()))
    else:
        raise ValueError(f"HUST pickle does not contain expected key {cell_id}")
    if not isinstance(record, Mapping) or "data" not in record:
        raise ValueError("HUST cell record does not contain a data mapping")
    return record


def _cycle_items(data: object) -> list[tuple[int, pd.DataFrame]]:
    if isinstance(data, Mapping):
        rows = []
        for key, value in data.items():
            try:
                cycle = int(key)
            except (TypeError, ValueError) as error:
                raise ValueError(f"Non-integer HUST cycle key: {key!r}") from error
            if not isinstance(value, pd.DataFrame):
                raise ValueError(f"HUST cycle {cycle} is not a pandas DataFrame")
            rows.append((cycle, value))
        return sorted(rows)
    if isinstance(data, (list, tuple)):
        if not all(isinstance(value, pd.DataFrame) for value in data):
            raise ValueError("HUST cycle list contains a non-DataFrame item")
        return [(index, value) for index, value in enumerate(data, start=1)]
    raise ValueError("HUST data field is neither a mapping nor a sequence")


def find_dynamic_landmark(frame: pd.DataFrame, protocol: Mapping[str, Any]) -> int | None:
    specification = protocol["dynamic_landmark"]
    lower = float(specification["lower_open_raw_SOH"])
    upper = float(specification["upper_closed_raw_SOH"])
    minimum_prefix = int(specification["minimum_observed_prefix_cycles"])
    observed = frame[frame["measurement_observed"].eq(1)].sort_values("cycle")
    previous_eol = False
    for prefix_count, row in enumerate(observed.itertuples(index=False), start=1):
        raw_soh = float(row.raw_capacity)
        if (
            prefix_count >= minimum_prefix
            and not previous_eol
            and raw_soh > lower
            and raw_soh <= upper
        ):
            return int(row.cycle)
        if raw_soh <= lower:
            previous_eol = True
    return None


def _future_support_count(
    frame: pd.DataFrame,
    landmark: int,
    threshold: float,
    horizon: int,
) -> int:
    observed = frame[frame["measurement_observed"].eq(1)].sort_values("cycle")
    future = observed[observed["cycle"].gt(landmark)]
    if future.empty:
        return 0
    hits = future[future["raw_capacity"].le(threshold)]
    support_end = int(hits.iloc[0]["cycle"]) if not hits.empty else int(future["cycle"].max())
    return int(
        future[
            future["cycle"].le(min(int(landmark + horizon), support_end))
        ].shape[0]
    )


def parse_hust_cell(data: bytes, cell_id: str, protocol: Mapping[str, Any]) -> tuple[pd.DataFrame, dict[str, object]]:
    # This is intentionally ordinary pickle loading. Governance requires that
    # the official-source archive hash be bound before this function is called.
    payload = pickle.loads(data)
    record = _cell_record(payload, cell_id)
    cycles = _cycle_items(record["data"])
    raw_cycle_count = len(cycles)
    if cell_id == "7-5":
        cycles = cycles[2:]
    rows: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    for cycle, frame in cycles:
        try:
            capacity = discharge_capacity_Ah(frame)
        except Exception as error:  # retained as a cell-level audit row
            errors.append(
                {
                    "cycle": int(cycle),
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            continue
        rows.append({"cycle": int(cycle), "raw_capacity_Ah": capacity})
    if len(rows) < int(protocol["soh_definition"]["reference_observations"]):
        raise ValueError("Fewer than five valid HUST capacity cycles")
    observed = pd.DataFrame(rows).sort_values("cycle").drop_duplicates("cycle")
    nominal = float(protocol["dataset"]["nominal_capacity_Ah"])
    reference, bol_values = bol_reference_from_values(
        observed["raw_capacity_Ah"].tolist(),
        nominal_capacity_Ah=nominal,
        count=int(protocol["soh_definition"]["reference_observations"]),
    )
    observed["raw_capacity"] = observed["raw_capacity_Ah"] / reference
    trajectory_upper = 1.5 * nominal
    invalid = (
        ~np.isfinite(observed["raw_capacity_Ah"])
        | observed["raw_capacity_Ah"].le(0)
        | observed["raw_capacity_Ah"].gt(trajectory_upper)
    )
    observed.loc[invalid, ["raw_capacity_Ah", "raw_capacity"]] = np.nan
    first_cycle = int(observed.dropna(subset=["raw_capacity"])["cycle"].min())
    last_cycle = int(observed["cycle"].max())
    grid = pd.DataFrame({"cycle": np.arange(first_cycle, last_cycle + 1, dtype=int)})
    grid = grid.merge(observed, on="cycle", how="left")
    grid["measurement_observed"] = grid["raw_capacity"].notna().astype(int)
    causal_input = grid["raw_capacity"].ffill()
    if causal_input.isna().any():
        raise ValueError("HUST curve has an unfilled causal prefix")
    grid["capacity"] = causal_smooth(
        causal_input.to_numpy(float),
        int(protocol["task"]["causal_smoothing_window_cycles"]),
    )
    grid["battery_id"] = f"HUST_{cell_id}"
    grid["source_cell_id"] = cell_id
    grid["batch"] = 8
    grid["cohort"] = "HUST_v0.3.0"
    grid["initial_capacity_Ah"] = 1.0
    grid["bol_reference_capacity_Ah"] = reference
    threshold = float(protocol["soh_definition"]["eol_threshold_raw_SOH"])
    landmark = find_dynamic_landmark(grid, protocol)
    observed_soh = grid[grid["measurement_observed"].eq(1)]
    eol_hits = observed_soh[observed_soh["raw_capacity"].le(threshold)]
    eol_cycle = None if eol_hits.empty else int(eol_hits.iloc[0]["cycle"])
    future_points = (
        0
        if landmark is None
        else _future_support_count(
            grid,
            landmark,
            threshold,
            int(protocol["task"]["forecast_horizon_cycles"]),
        )
    )
    audit = {
        "cell_id": f"HUST_{cell_id}",
        "source_cell_id": cell_id,
        "status": "PASS",
        "pickle_sha256": sha256_bytes(data),
        "raw_cycles_in_pickle": raw_cycle_count,
        "cycles_after_problem_cell_rule": len(cycles),
        "valid_capacity_cycles": int(grid["measurement_observed"].sum()),
        "cycle_integration_failures": len(errors),
        "cycle_integration_failure_detail": json.dumps(errors, ensure_ascii=False),
        "first_physical_cycle": first_cycle,
        "last_physical_cycle": last_cycle,
        "bol_reference_capacity_Ah": reference,
        "bol_first_five_valid_Ah": json.dumps(bol_values),
        "soh_formula": protocol["soh_definition"]["formula"],
        "landmark_cycle": landmark,
        "landmark_raw_soh": (
            None
            if landmark is None
            else float(grid.loc[grid["cycle"].eq(landmark), "raw_capacity"].iloc[0])
        ),
        "observed_eol_cycle": eol_cycle,
        "truth_event_type": "observed" if eol_cycle is not None else "right_censored",
        "future_metric_support_points": future_points,
        "problem_cell_first_two_cycles_excluded": cell_id == "7-5",
    }
    return grid, audit


def _prepared_curves_sha256(curves: Mapping[str, pd.DataFrame]) -> str:
    import hashlib

    digest = hashlib.sha256()
    columns = [
        "cycle",
        "raw_capacity_Ah",
        "raw_capacity",
        "capacity",
        "measurement_observed",
        "battery_id",
        "source_cell_id",
        "bol_reference_capacity_Ah",
    ]
    for cell_id in sorted(curves):
        digest.update((cell_id + "\n").encode())
        digest.update(
            curves[cell_id][columns].to_csv(
                index=False,
                na_rep="NA",
                float_format="%.17g",
                lineterminator="\n",
            ).encode()
        )
    return digest.hexdigest()


def prepare_role(
    archive: Path,
    config: Path,
    inventory_dir: Path,
    output: Path,
    role: str,
    *,
    trust_official_pickle: bool,
) -> dict[str, Any]:
    if role not in {"calibration", "confirmation"}:
        raise ValueError(role)
    if not trust_official_pickle:
        raise RuntimeError(
            "Refusing to unpickle HUST data without --trust-official-pickle. "
            "Use only the official archive whose SHA-256 passed inventory."
        )
    protocol, source_receipt, split = load_source_gate(
        archive,
        config,
        inventory_dir,
    )
    selected = split[split["role"].eq(role)].copy().sort_values("split_rank")
    expected = int(protocol["split"][f"{role}_cells"])
    if len(selected) != expected:
        raise RuntimeError(f"Expected {expected} {role} cells, got {len(selected)}")
    output = Path(output)
    preflight_path = output / f"{role}_preflight.json"
    if preflight_path.exists():
        raise RuntimeError(
            f"HUST {role} arm is already prepared in this run root; "
            "overwriting a revealed arm is prohibited"
        )
    curves_dir = output / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)
    audits: list[dict[str, object]] = []
    exclusions: list[dict[str, object]] = []
    curves: dict[str, pd.DataFrame] = {}
    minimum_future = int(
        protocol["dynamic_landmark"]["minimum_future_observations_for_RMSE"]
    )
    with zipfile.ZipFile(archive, "r") as handle:
        names = set(handle.namelist())
        for row in selected.itertuples(index=False):
            member = str(row.archive_member)
            source_cell_id = str(row.cell_id)
            if member not in names:
                raise RuntimeError(f"Frozen HUST member disappeared: {member}")
            try:
                data = handle.read(member)
                frame, audit = parse_hust_cell(data, source_cell_id, protocol)
                audit.update({"role": role, "split_rank": int(row.split_rank)})
                reason = None
                if audit["landmark_cycle"] is None:
                    reason = "no_preregistered_dynamic_landmark"
                elif int(audit["future_metric_support_points"]) < minimum_future:
                    reason = "insufficient_future_observed_support"
                if reason is not None:
                    audit["status"] = "EXCLUDED"
                    audit["exclusion_reason"] = reason
                    exclusions.append(audit.copy())
                else:
                    audit["exclusion_reason"] = None
                    cell_id = str(audit["cell_id"])
                    path = curves_dir / f"{cell_id}.csv"
                    frame.to_csv(path, index=False)
                    audit["prepared_curve_sha256"] = sha256_file(path)
                    curves[cell_id] = frame
                audits.append(audit)
            except Exception as error:
                failed = {
                    "cell_id": f"HUST_{source_cell_id}",
                    "source_cell_id": source_cell_id,
                    "role": role,
                    "split_rank": int(row.split_rank),
                    "status": "EXCLUDED",
                    "exclusion_reason": "parse_or_quality_gate_failure",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                audits.append(failed)
                exclusions.append(failed.copy())
    audit_frame = pd.DataFrame(audits).sort_values("split_rank")
    exclusion_frame = pd.DataFrame(exclusions)
    if exclusion_frame.empty:
        exclusion_frame = pd.DataFrame(
            columns=[
                "cell_id",
                "source_cell_id",
                "role",
                "split_rank",
                "status",
                "exclusion_reason",
            ]
        )
    audit_path = output / f"{role}_cell_audit.csv"
    exclusion_path = output / f"{role}_exclusions.csv"
    audit_frame.to_csv(audit_path, index=False)
    exclusion_frame.to_csv(exclusion_path, index=False)
    gate = {
        "status": "PASS",
        "created_utc": utc_now(),
        "role": role,
        "config_sha256": json_sha256(protocol),
        "source_archive_sha256": source_receipt["archive_sha256"],
        "split_manifest_sha256": source_receipt["split_manifest_sha256"],
        "assigned_cells": expected,
        "evaluable_cells": len(curves),
        "excluded_cells": len(exclusions),
        "minimum_future_observations": minimum_future,
        "curves_sha256": _prepared_curves_sha256(curves),
        "cell_audit_sha256": sha256_file(audit_path),
        "exclusions_sha256": sha256_file(exclusion_path),
        "pickle_unpickled": True,
        "capacity_value_read": True,
        "model_output_computed": False,
        "confirmation_members_unpickled": role == "confirmation",
    }
    write_json(preflight_path, gate)
    print(
        f"[PASS] HUST {role} preparation: assigned={expected} "
        f"evaluable={len(curves)} excluded={len(exclusions)}"
    )
    return gate


def load_prepared_role(
    path: Path,
    protocol: Mapping[str, Any],
    role: str,
) -> dict[str, pd.DataFrame]:
    gate = read_json(Path(path) / f"{role}_preflight.json")
    if (
        gate.get("status") != "PASS"
        or gate.get("role") != role
        or gate.get("config_sha256") != json_sha256(protocol)
    ):
        raise RuntimeError(f"Prepared HUST {role} gate failed")
    curves: dict[str, pd.DataFrame] = {}
    for csv_path in sorted((Path(path) / "curves").glob("HUST_*.csv")):
        frame = pd.read_csv(csv_path).sort_values("cycle").reset_index(drop=True)
        cell_id = str(frame["battery_id"].iloc[0])
        curves[cell_id] = frame
    if _prepared_curves_sha256(curves) != gate.get("curves_sha256"):
        raise RuntimeError(f"Prepared HUST {role} curve hash changed")
    return curves
