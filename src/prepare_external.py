#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

from common import (
    causal_smooth,
    eol_interval,
    json_sha256,
    load_protocol,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--shared-root", type=Path, required=True)
    parser.add_argument("--archive-inventory", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--freeze-zip", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_receipt(receipt_path: Path, freeze_zip: Path) -> dict:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "PASS" or not receipt.get(
        "external_opening_authorized"
    ):
        raise RuntimeError("External opening is not authorized by a freeze receipt")
    observed = sha256_file(freeze_zip)
    if observed != receipt.get("freeze_zip_sha256"):
        raise RuntimeError("Freeze ZIP hash no longer matches the registered receipt")
    return receipt


def extract_archives(
    shared_root: Path,
    inventory_rows: list[dict[str, str]],
    raw_dir: Path,
) -> None:
    try:
        import py7zr
    except ImportError as exc:
        raise RuntimeError("py7zr is required; run manager.sh setup") from exc
    archives = sorted({row["archive_name"] for row in inventory_rows})
    raw_dir.mkdir(parents=True, exist_ok=True)
    for archive_name in archives:
        row = next(
            item for item in inventory_rows if item["archive_name"] == archive_name
        )
        path = (
            Path(row["archive_path"])
            if row.get("archive_path")
            else shared_root / archive_name
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        expected_hash = row["archive_sha256"]
        if sha256_file(path) != expected_hash:
            raise RuntimeError(f"{archive_name}: SHA-256 mismatch at opening")
        with py7zr.SevenZipFile(path, mode="r") as archive:
            archive.extractall(path=raw_dir)
        print(f"[EXTRACTED] {archive_name}")


def matlab_text(value: object) -> str:
    array = np.asarray(value)
    if array.size == 0:
        return ""
    flat = array.reshape(-1)
    if flat.dtype.kind in {"U", "S"}:
        return " ".join(str(item) for item in flat).strip()
    pieces = []
    for item in flat:
        if isinstance(item, np.ndarray):
            pieces.append(matlab_text(item))
        else:
            pieces.append(str(item))
    return " ".join(piece for piece in pieces if piece).strip()


def summary_capacity_life(mat: dict, path: Path) -> tuple[np.ndarray, int]:
    if "summary" not in mat:
        raise KeyError(f"{path}: missing summary")
    summary = mat["summary"]
    capacity = None
    cycle_life = None
    try:
        record = summary[0][0]
        capacity = np.asarray(record[1], dtype=float).reshape(-1)
        life_values = np.asarray(record[8], dtype=float).reshape(-1)
        if len(life_values):
            cycle_life = round(float(life_values[0]))
    except (IndexError, TypeError, ValueError):
        capacity = None
        cycle_life = None
    if capacity is None and getattr(summary.dtype, "names", None):
        names = {str(name).lower(): name for name in summary.dtype.names or ()}
        cap_field = next((names[name] for name in names if "capacity" in name), None)
        life_field = next(
            (names[name] for name in names if "cycle" in name and "life" in name),
            None,
        )
        if cap_field is not None:
            capacity = np.asarray(summary[cap_field][0][0], dtype=float).reshape(-1)
        if life_field is not None:
            values = np.asarray(summary[life_field][0][0], dtype=float).reshape(-1)
            if len(values):
                cycle_life = round(float(values[0]))
    if capacity is None or cycle_life is None:
        raise ValueError(f"{path}: cannot locate summary capacity/cycle life")
    if len(capacity) < cycle_life:
        raise ValueError(
            f"{path}: capacity length {len(capacity)} < cycle life {cycle_life}"
        )
    return capacity[:cycle_life], cycle_life


def cycle_descriptions(mat: dict, cycle_life: int, path: Path) -> list[str]:
    if "data" not in mat:
        raise KeyError(f"{path}: missing data")
    data = mat["data"]
    if data.ndim < 2 or data.shape[1] < cycle_life:
        raise ValueError(f"{path}: data cycles do not cover cycle life")
    descriptions = []
    for index in range(cycle_life):
        record = data[0][index]
        try:
            descriptions.append(matlab_text(record[7]))
        except Exception as exc:
            raise ValueError(
                f"{path}: cannot read cycle {index + 1} description"
            ) from exc
    return descriptions


def batch_from_filename(name: str) -> int:
    if name.startswith("R3_battery-"):
        return 4
    if name.startswith("RW_battery-"):
        return 5
    if name.startswith("Sim_satellite_battery-"):
        return 6
    raise ValueError(f"Unexpected external filename: {name}")


def prepare_one(
    path: Path, batch: int, threshold: float, smoothing_window: int
) -> tuple[pd.DataFrame, dict[str, object]]:
    mat = loadmat(path)
    capacity, cycle_life = summary_capacity_life(mat, path)
    descriptions = cycle_descriptions(mat, cycle_life, path)
    reference_cycles = [
        index + 1
        for index, description in enumerate(descriptions)
        if "test capacity" in description.lower()
    ]
    if not reference_cycles:
        raise ValueError(f"{path}: no 'test capacity' cycles found")
    values = capacity[np.asarray(reference_cycles, dtype=int) - 1]
    valid = np.isfinite(values) & (values > 0.2) & (values < 3.0)
    reference_cycles = np.asarray(reference_cycles, dtype=int)[valid].tolist()
    values = values[valid].astype(float)
    if len(reference_cycles) < 3:
        raise ValueError(f"{path}: fewer than 3 valid reference-capacity tests")
    first_reference = int(reference_cycles[0])
    grid = pd.DataFrame(
        {"cycle": np.arange(first_reference, cycle_life + 1, dtype=int)}
    )
    observations = pd.DataFrame({"cycle": reference_cycles, "raw_capacity": values})
    grid = grid.merge(observations, on="cycle", how="left")
    grid["measurement_observed"] = grid["raw_capacity"].notna().astype(int)
    grid["capacity_loco"] = grid["raw_capacity"].ffill()
    if grid["capacity_loco"].isna().any():
        raise RuntimeError(f"{path}: causal input has an unfilled prefix")
    q0 = float(values[0])
    grid["capacity"] = causal_smooth(
        grid["capacity_loco"].to_numpy(float), smoothing_window
    )
    grid["soh_raw"] = grid["raw_capacity"] / q0
    grid["soh_model"] = grid["capacity"] / q0
    cell_id = f"B{batch}_{path.stem}"
    grid["battery_id"] = cell_id
    grid["batch"] = batch
    grid["initial_capacity_Ah"] = q0
    interval = eol_interval(grid, threshold)
    gaps = np.diff(np.asarray(reference_cycles, dtype=int))
    audit = {
        "battery_id": cell_id,
        "batch": batch,
        "source_file": path.name,
        "source_sha256": sha256_file(path),
        "cycle_life_metadata": cycle_life,
        "first_reference_cycle": first_reference,
        "last_reference_cycle": int(reference_cycles[-1]),
        "reference_capacity_tests": len(reference_cycles),
        "reference_coverage_fraction": len(reference_cycles) / cycle_life,
        "max_reference_gap_cycles": int(gaps.max()) if len(gaps) else 0,
        "initial_reference_capacity_Ah": q0,
        "observed_eol_interval_lower": interval[0] if interval else None,
        "observed_eol_interval_upper": interval[1] if interval else None,
        "has_observed_eol_crossing": int(interval is not None),
        "future_interpolation_used": 0,
        "model_input_fill": "causal_LOCF_then_one_sided_rolling_mean",
    }
    return grid, audit


def main() -> None:
    args = parse_args()
    protocol = load_protocol(args.config)
    config_hash = json_sha256(protocol)
    threshold = float(protocol["task"]["eol_threshold_Ah"])
    smoothing_window = int(protocol["task"]["causal_smoothing_window_cycles"])
    validate_receipt(args.receipt, args.freeze_zip)
    output = args.output_dir.resolve()
    preflight = output / "external_preflight.json"
    if preflight.is_file() and not args.overwrite:
        old = json.loads(preflight.read_text(encoding="utf-8"))
        if old.get("status") == "PASS":
            print(f"[SKIP] external preparation already passed: {output}")
            return
        raise RuntimeError(
            "Existing external preparation is incomplete; use --overwrite"
        )
    if args.overwrite:
        if args.raw_dir.exists():
            shutil.rmtree(args.raw_dir)
        if output.exists():
            shutil.rmtree(output)
    inventory_rows = list(
        csv.DictReader(args.archive_inventory.open(encoding="utf-8", newline=""))
    )
    extract_archives(args.shared_root.resolve(), inventory_rows, args.raw_dir.resolve())

    expected_paths = {
        Path(row["inner_path"]).name: {
            "batch": int(row["batch"].split("-")[1]),
            "size": int(row["uncompressed_bytes"]),
        }
        for row in inventory_rows
    }
    extracted = {}
    for path in args.raw_dir.rglob("*.mat"):
        if path.name in expected_paths:
            if path.name in extracted:
                raise RuntimeError(f"Duplicate extracted file: {path.name}")
            extracted[path.name] = path
    if set(extracted) != set(expected_paths):
        raise RuntimeError(
            f"Extracted filename mismatch; missing={sorted(set(expected_paths) - set(extracted))}, "
            f"extra={sorted(set(extracted) - set(expected_paths))}"
        )
    for name, path in extracted.items():
        if path.stat().st_size != expected_paths[name]["size"]:
            raise RuntimeError(f"{name}: extracted byte size mismatch")

    curves_dir = output / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)
    audit_rows = []
    failures = []
    for name in sorted(extracted):
        path = extracted[name]
        batch = batch_from_filename(name)
        try:
            frame, audit = prepare_one(path, batch, threshold, smoothing_window)
            frame.to_csv(curves_dir / f"{audit['battery_id']}.csv", index=False)
            audit_rows.append(audit)
            print(
                f"[PARSED] {name}: references={audit['reference_capacity_tests']} "
                f"max_gap={audit['max_reference_gap_cycles']}"
            )
        except Exception as exc:  # noqa: BLE001 - retain every parse failure
            failures.append({"file": name, "error": str(exc)})

    audit = pd.DataFrame(audit_rows)
    if not audit.empty:
        audit = audit.sort_values(["batch", "battery_id"])
        audit.to_csv(output / "external_curve_audit.csv", index=False)
    batch_counts = {
        batch: int((audit["batch"] == batch).sum()) if not audit.empty else 0
        for batch in (4, 5, 6)
    }
    batch4 = audit[audit["batch"].eq(4)] if not audit.empty else audit
    batch4_every_cycle = bool(
        len(batch4) == 8
        and batch4["first_reference_cycle"].eq(1).all()
        and batch4["max_reference_gap_cycles"].le(1).all()
        and batch4["reference_capacity_tests"].eq(batch4["cycle_life_metadata"]).all()
    )
    observed_eol_count = (
        int(audit["has_observed_eol_crossing"].sum()) if not audit.empty else 0
    )
    status = (
        "PASS"
        if not failures and batch_counts == {4: 8, 5: 8, 6: 8} and batch4_every_cycle
        else "FAIL"
    )
    gate = {
        "status": status,
        "config_sha256": config_hash,
        "freeze_receipt": json.loads(args.receipt.read_text(encoding="utf-8")),
        "batch_counts": batch_counts,
        "batch4_reference_capacity_observed_every_cycle": batch4_every_cycle,
        "observed_fixed_threshold_eol_crossings": observed_eol_count,
        "right_censored_at_last_reference_observation": 24 - observed_eol_count,
        "cells_without_observed_fixed_threshold_crossing_retained": True,
        "sparse_input_fill": "causal_LOCF",
        "future_interpolation_used": False,
        "failures": failures,
    }
    output.mkdir(parents=True, exist_ok=True)
    preflight.write_text(
        json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if status != "PASS":
        print(json.dumps(gate, ensure_ascii=False, indent=2), file=sys.stderr)
        raise SystemExit(2)
    print("[PASS] External curves prepared under the frozen protocol")
    print(f"[OUT] {output}")


if __name__ == "__main__":
    main()
