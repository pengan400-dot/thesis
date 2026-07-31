#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import pickle
import pickletools
import random
import re
import time
import zipfile
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .model_variants import PaperMSTTVariant, identity_record
from .soh import bol_reference_from_values


FEATURE_NAMES = (
    "cycle_scaled",
    "soh",
    "health_loss",
    "soh_diff",
    "soh_rolling_mean",
    "soh_rolling_std",
    "degradation_rate",
)

FORBIDDEN_PICKLE_OPCODES = frozenset(
    {
        "GLOBAL",
        "STACK_GLOBAL",
        "REDUCE",
        "BUILD",
        "INST",
        "OBJ",
        "NEWOBJ",
        "NEWOBJ_EX",
        "EXT1",
        "EXT2",
        "EXT4",
        "PERSID",
        "BINPERSID",
    }
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def load_protocol(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "2.0":
        raise ValueError("Unsupported protocol schema")
    if payload.get("analysis_class") != (
        "xjtu_only_soh_development_hnei_exploration_"
        "pre_bit_confirmation"
    ):
        raise ValueError("Unexpected SOH/BIT analysis classification")
    conversion = payload["ah_to_soh_conversion"]
    locked = {
        "residual_scale_SOH": 0.04,
        "huber_delta_SOH": 0.01,
        "threshold_weight_band_SOH": 0.04,
        "upward_tolerance_SOH": 0.001,
    }
    for key, expected in locked.items():
        observed = float(
            payload["training"].get(key, conversion.get(key))
        )
        if not math.isclose(observed, expected, rel_tol=0, abs_tol=1e-12):
            raise ValueError(
                f"Frozen SOH hyperparameter changed: {key}={observed}"
            )
        if not math.isclose(
            float(conversion[key]),
            expected,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"Frozen Ah-to-SOH conversion changed: {key}"
            )
    if list(map(float, payload["task"]["prediction_clip_SOH"])) != [
        0.2,
        1.2,
    ]:
        raise ValueError("Frozen SOH physical clipping range changed")
    if list(map(float, conversion["prediction_clip_SOH"])) != [0.2, 1.2]:
        raise ValueError("Frozen converted SOH clipping range changed")
    if int(payload["task"]["minimum_pre_cutoff_physical_cycles"]) != 20:
        raise ValueError("The 16-step window plus warm-up requires 20 cycles")
    if int(payload["task"]["minimum_future_observations_for_RMSE"]) != 5:
        raise ValueError("Frozen minimum future RMSE support changed")
    if payload["features"]["scaler"] != (
        "StandardScaler fitted only on the relevant XJTU training cells "
        "for each selection, refit, or outer cross-fitting fold."
    ):
        raise ValueError("Frozen XJTU-only scaler rule changed")
    if payload["soh_definition"]["formula"] != (
        "SOH_i_t = C_i_t / median(C_i_first_5_valid)"
    ):
        raise ValueError("SOH formula direction or reference rule changed")
    return payload


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def causal_smooth(values: Sequence[float], window: int) -> np.ndarray:
    return (
        pd.Series(np.asarray(values, dtype=float))
        .rolling(window=max(1, int(window)), min_periods=1)
        .mean()
        .to_numpy(float)
    )


def compute_features(
    cycles: Sequence[int],
    capacities: Sequence[float],
    q0: float = 1.0,
) -> np.ndarray:
    del q0
    cycle = np.asarray(cycles, dtype=float)
    soh = np.asarray(capacities, dtype=float)
    if len(cycle) != len(soh) or not len(soh):
        raise ValueError("Cycle and SOH arrays must have equal non-zero length")
    difference = np.r_[0.0, np.diff(soh)]
    rolling = pd.Series(soh).rolling(5, min_periods=1)
    rolling_mean = rolling.mean().to_numpy(float)
    rolling_std = rolling.std(ddof=0).fillna(0.0).to_numpy(float)
    degradation = np.zeros_like(soh)
    for index in range(1, len(soh)):
        previous = max(0, index - 5)
        cycle_delta = max(1.0, cycle[index] - cycle[previous])
        degradation[index] = (
            soh[index] - soh[previous]
        ) / cycle_delta
    if len(soh) > 1:
        degradation[0] = degradation[1]
    return np.column_stack(
        [
            cycle / 1000.0,
            soh,
            1.0 - soh,
            difference,
            rolling_mean,
            rolling_std,
            degradation,
        ]
    ).astype(np.float32)


def curves_sha256(curves: Mapping[str, pd.DataFrame]) -> str:
    digest = hashlib.sha256()
    preferred = (
        "cycle",
        "raw_capacity",
        "capacity",
        "measurement_observed",
        "battery_id",
        "batch",
        "initial_capacity_Ah",
    )
    for cell_id in sorted(curves):
        frame = curves[cell_id].sort_values("cycle").reset_index(drop=True)
        columns = [column for column in preferred if column in frame.columns]
        digest.update(f"{cell_id}\n".encode())
        digest.update(
            frame[columns]
            .to_csv(
                index=False,
                na_rep="NA",
                float_format="%.17g",
                lineterminator="\n",
            )
            .encode()
        )
    return digest.hexdigest()


def scan_pickle_bytes(data: bytes) -> dict[str, object]:
    protocol = None
    counts: dict[str, int] = {}
    forbidden: dict[str, int] = {}
    for opcode, argument, _ in pickletools.genops(data):
        counts[opcode.name] = counts.get(opcode.name, 0) + 1
        if opcode.name == "PROTO":
            protocol = int(argument)
        if opcode.name in FORBIDDEN_PICKLE_OPCODES:
            forbidden[opcode.name] = forbidden.get(opcode.name, 0) + 1
    return {
        "pickle_protocol": protocol,
        "forbidden_opcode_counts": forbidden,
        "opcode_count": int(sum(counts.values())),
    }


def command_inventory(args: argparse.Namespace) -> None:
    protocol = load_protocol(args.config)
    expected = set(protocol["hnei"]["expected_members"])
    actual_hash = sha256_file(args.archive)
    expected_hash = str(protocol["hnei"]["archive_sha256"])
    failures: list[str] = []
    if actual_hash != expected_hash:
        failures.append(
            f"archive SHA-256 mismatch: expected {expected_hash}, got {actual_hash}"
        )
    rows: list[dict[str, object]] = []
    with zipfile.ZipFile(args.archive) as archive:
        if archive.testzip() is not None:
            failures.append("ZIP CRC test failed")
        actual = {
            info.filename
            for info in archive.infolist()
            if not info.is_dir()
        }
        if actual != expected:
            failures.append(
                f"member mismatch: missing={sorted(expected - actual)}, "
                f"extra={sorted(actual - expected)}"
            )
        total_uncompressed = sum(info.file_size for info in archive.infolist())
        if total_uncompressed > 1_000_000_000:
            failures.append("uncompressed archive exceeds 1 GB safety limit")
        for info in archive.infolist():
            path = PurePosixPath(info.filename)
            if path.is_absolute() or ".." in path.parts:
                failures.append(f"unsafe archive path: {info.filename}")
            if info.flag_bits & 1:
                failures.append(f"encrypted member is prohibited: {info.filename}")
            if info.is_dir():
                continue
            if info.file_size > 100_000_000:
                failures.append(f"oversized member: {info.filename}")
            data = archive.read(info)
            scan = scan_pickle_bytes(data)
            if scan["forbidden_opcode_counts"]:
                failures.append(
                    f"{info.filename}: forbidden pickle opcodes "
                    f"{scan['forbidden_opcode_counts']}"
                )
            rows.append(
                {
                    "member": info.filename,
                    "file_size": info.file_size,
                    "compressed_size": info.compress_size,
                    "crc32": f"{info.CRC:08x}",
                    "member_sha256": hashlib.sha256(data).hexdigest(),
                    **scan,
                }
            )
    payload = {
        "status": "PASS" if not failures else "FAIL",
        "created_utc": utc_now(),
        "config_sha256": json_sha256(protocol),
        "archive": str(args.archive.resolve()),
        "archive_sha256": actual_hash,
        "expected_member_count": len(expected),
        "observed_member_count": len(rows),
        "pickle_loaded": False,
        "outcome_values_parsed": False,
        "members": rows,
        "failures": failures,
    }
    write_json(args.output / "inventory.json", payload)
    if failures:
        raise RuntimeError("\n".join(failures))
    print(
        f"[PASS] inventory: cells={len(rows)} sha256={actual_hash} "
        "primitive-only pickle gate=PASS"
    )


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        raise pickle.UnpicklingError(f"forbidden global: {module}.{name}")

    def persistent_load(self, persistent_id):
        raise pickle.UnpicklingError(
            f"persistent pickle IDs are forbidden: {persistent_id!r}"
        )


def restricted_load(data: bytes) -> object:
    scan = scan_pickle_bytes(data)
    if scan["forbidden_opcode_counts"]:
        raise pickle.UnpicklingError(
            f"forbidden pickle opcodes: {scan['forbidden_opcode_counts']}"
        )
    return RestrictedUnpickler(io.BytesIO(data)).load()


def finite_values(values: object) -> list[float]:
    if not isinstance(values, list):
        return []
    return [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    ]


def eol_interval(
    frame: pd.DataFrame,
    threshold: float,
) -> tuple[int, int] | None:
    measured = frame[frame["measurement_observed"].eq(1)].copy()
    measured = measured.dropna(subset=["raw_capacity"]).sort_values("cycle")
    hits = measured[measured["raw_capacity"] <= threshold]
    if hits.empty:
        return None
    upper = int(hits.iloc[0]["cycle"])
    earlier = measured[measured["cycle"] < upper]
    lower = 1 if earlier.empty else int(earlier.iloc[-1]["cycle"]) + 1
    return lower, upper


def parse_hnei_cell(
    member_name: str,
    data: bytes,
    protocol: dict,
) -> tuple[pd.DataFrame, dict[str, object]]:
    root = restricted_load(data)
    if not isinstance(root, dict):
        raise ValueError(f"{member_name}: root is not a dict")
    required_root = {
        "cell_id",
        "cycle_data",
        "nominal_capacity_in_Ah",
        "form_factor",
        "anode_material",
        "cathode_material",
    }
    if not required_root.issubset(root):
        raise ValueError(f"{member_name}: missing root fields")
    cycle_data = root["cycle_data"]
    if not isinstance(cycle_data, list) or not cycle_data:
        raise ValueError(f"{member_name}: cycle_data is empty")
    nominal = float(root["nominal_capacity_in_Ah"])
    expected_nominal = float(protocol["hnei"]["declared_nominal_capacity_Ah"])
    if not math.isclose(nominal, expected_nominal, rel_tol=0, abs_tol=1e-9):
        raise ValueError(
            f"{member_name}: expected nominal {expected_nominal}, got {nominal}"
        )
    lower_rate, upper_rate = map(
        float, protocol["hnei"]["accepted_discharge_rate_C"]
    )
    observations: list[dict[str, float | int]] = []
    rejected_rates: list[float] = []
    sequence_lengths: list[int] = []
    for record in cycle_data:
        if not isinstance(record, dict):
            raise ValueError(f"{member_name}: non-dict cycle record")
        cycle = int(record["cycle_number"])
        currents = finite_values(record.get("current_in_A"))
        capacities = finite_values(record.get("discharge_capacity_in_Ah"))
        if not currents or not capacities:
            continue
        negative = [-value for value in currents if value < -0.05]
        if not negative:
            continue
        discharge_rate = max(negative) / nominal
        if not lower_rate <= discharge_rate <= upper_rate:
            rejected_rates.append(discharge_rate)
            continue
        capacity = max(capacities)
        if not 0.0 < capacity <= nominal * 1.5:
            raise ValueError(
                f"{member_name} cycle {cycle}: invalid capacity {capacity}"
            )
        sequence_lengths.append(len(currents))
        observations.append(
            {
                "cycle": cycle,
                "original_discharge_capacity_Ah": capacity,
                "discharge_rate_C": discharge_rate,
            }
        )
    observed = pd.DataFrame(observations).sort_values("cycle").reset_index(drop=True)
    if len(observed) < 200:
        raise ValueError(f"{member_name}: fewer than 200 accepted-rate cycles")
    if observed["cycle"].duplicated().any():
        raise ValueError(f"{member_name}: duplicate cycle numbers")
    cycles = observed["cycle"].to_numpy(int)
    if np.any(np.diff(cycles) <= 0) or cycles[0] != 1:
        raise ValueError(f"{member_name}: cycle numbers must increase from 1")
    initial_capacity, bol_values = bol_reference_from_values(
        observed["original_discharge_capacity_Ah"].tolist(),
        nominal_capacity_Ah=nominal,
        count=int(protocol["soh_definition"]["reference_observations"]),
    )
    observed["raw_capacity"] = (
        observed["original_discharge_capacity_Ah"] / initial_capacity
    )
    last_cycle = int(observed["cycle"].max())
    grid = pd.DataFrame({"cycle": np.arange(1, last_cycle + 1, dtype=int)})
    grid = grid.merge(observed, on="cycle", how="left")
    grid["measurement_observed"] = grid["raw_capacity"].notna().astype(int)
    grid["capacity_loco"] = grid["raw_capacity"].ffill()
    if grid["capacity_loco"].isna().any():
        raise RuntimeError(f"{member_name}: causal input has an unfilled prefix")
    smoothing_window = int(
        protocol["task"]["causal_smoothing_window_cycles"]
    )
    grid["capacity"] = causal_smooth(
        grid["capacity_loco"].to_numpy(float),
        smoothing_window,
    )
    grid["soh_raw"] = grid["raw_capacity"]
    grid["soh_model"] = grid["capacity"]
    suffix = str(root["cell_id"]).rsplit("_", 1)[-1]
    cell_id = f"B7_HNEI_{suffix}"
    grid["battery_id"] = cell_id
    grid["batch"] = 7
    grid["initial_capacity_Ah"] = 1.0
    grid["original_initial_capacity_Ah"] = initial_capacity
    grid["nominal_capacity_Ah"] = nominal
    threshold = float(protocol["task"]["eol_threshold_SOH"])
    interval = eol_interval(grid, threshold)
    observed_gaps = np.diff(cycles)
    audit = {
        "battery_id": cell_id,
        "source_member": member_name,
        "source_cell_id": root["cell_id"],
        "form_factor": root["form_factor"],
        "anode_material": root["anode_material"],
        "cathode_material": root["cathode_material"],
        "nominal_capacity_Ah": nominal,
        "bol_reference_capacity_Ah": initial_capacity,
        "bol_first_five_valid_Ah": bol_values,
        "bol_rule": "median_first_5_valid",
        "stored_cycle_records": len(cycle_data),
        "accepted_rate_observations": len(observed),
        "rejected_rate_observations": len(rejected_rates),
        "first_cycle": int(cycles[0]),
        "last_cycle": int(cycles[-1]),
        "missing_physical_cycles": int(last_cycle - len(observed)),
        "maximum_observation_gap_cycles": int(observed_gaps.max()),
        "minimum_sequence_length": int(min(sequence_lengths)),
        "maximum_sequence_length": int(max(sequence_lengths)),
        "minimum_discharge_rate_C": float(observed["discharge_rate_C"].min()),
        "maximum_discharge_rate_C": float(observed["discharge_rate_C"].max()),
        "eol_event_observed": int(interval is not None),
        "eol_interval_lower_cycle": interval[0] if interval else None,
        "eol_interval_upper_cycle": interval[1] if interval else None,
        "future_interpolation_used": False,
        "missing_cycle_fill": "causal_LOCF_then_one_sided_rolling_mean",
    }
    return grid, audit


def validate_receipt(
    receipt_path: Path,
    freeze_zip: Path,
    protocol: dict,
) -> dict:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("status") != "PASS"
        or receipt.get("analysis_class")
        != protocol["analysis_class"]
        or receipt.get("config_sha256") != json_sha256(protocol)
        or receipt.get("freeze_zip_sha256") != sha256_file(freeze_zip)
    ):
        raise RuntimeError("Pre-evaluation freeze receipt is invalid")
    return receipt


def command_prepare_hnei(args: argparse.Namespace) -> None:
    protocol = load_protocol(args.config)
    receipt = validate_receipt(args.receipt, args.freeze_zip, protocol)
    inventory = json.loads(
        (args.inventory / "inventory.json").read_text(encoding="utf-8")
    )
    if (
        inventory.get("status") != "PASS"
        or inventory.get("archive_sha256")
        != protocol["hnei"]["archive_sha256"]
        or inventory.get("outcome_values_parsed") is not False
    ):
        raise RuntimeError("Primitive-only archive inventory gate failed")
    current_archive_hash = sha256_file(args.archive)
    if current_archive_hash != inventory.get("archive_sha256"):
        raise RuntimeError(
            "BatteryLife.zip changed after the primitive-only inventory"
        )
    curves_dir = args.output / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)
    audits: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    expected = set(protocol["hnei"]["expected_members"])
    with zipfile.ZipFile(args.archive) as archive:
        for member in sorted(expected):
            try:
                frame, audit = parse_hnei_cell(
                    member,
                    archive.read(member),
                    protocol,
                )
                frame.to_csv(
                    curves_dir / f"{audit['battery_id']}.csv",
                    index=False,
                )
                audits.append(audit)
                print(
                    f"[PARSED] {audit['battery_id']}: "
                    f"observations={audit['accepted_rate_observations']} "
                    f"EOL={audit['eol_interval_lower_cycle']}-"
                    f"{audit['eol_interval_upper_cycle']}"
                )
            except Exception as exc:  # noqa: BLE001
                failures.append({"member": member, "error": str(exc)})
    audit_frame = pd.DataFrame(audits)
    if not audit_frame.empty:
        audit_frame = audit_frame.sort_values("battery_id")
        audit_frame.to_csv(args.output / "hnei_curve_audit.csv", index=False)
    expected_cells = int(protocol["hnei"]["expected_cells"])
    status = (
        "PASS"
        if (
            not failures
            and len(audits) == expected_cells
            and int(audit_frame["eol_event_observed"].sum()) == expected_cells
            and int(audit_frame["rejected_rate_observations"].sum()) == 0
        )
        else "FAIL"
    )
    curves = {
        path.stem: pd.read_csv(path)
        for path in sorted(curves_dir.glob("*.csv"))
    }
    gate = {
        "status": status,
        "created_utc": utc_now(),
        "analysis_class": protocol["analysis_class"],
        "claim_limit": protocol["claim_limit"],
        "config_sha256": json_sha256(protocol),
        "archive_sha256": sha256_file(args.archive),
        "freeze_receipt": receipt,
        "physical_cells": len(audits),
        "observed_eol_cells": (
            int(audit_frame["eol_event_observed"].sum())
            if not audit_frame.empty
            else 0
        ),
        "accepted_rate_observations": (
            int(audit_frame["accepted_rate_observations"].sum())
            if not audit_frame.empty
            else 0
        ),
        "rejected_rate_observations": (
            int(audit_frame["rejected_rate_observations"].sum())
            if not audit_frame.empty
            else 0
        ),
        "curves_sha256": curves_sha256(curves) if curves else None,
        "future_interpolation_used": False,
        "hnei_model_predictions_generated": False,
        "failures": failures,
    }
    write_json(args.output / "external_preflight.json", gate)
    if status != "PASS":
        raise RuntimeError(f"HNEI preparation failed: {failures}")
    print(
        f"[PASS] HNEI prepared: cells={len(audits)} "
        f"curve_hash={gate['curves_sha256']}"
    )


def command_prepare_development(args: argparse.Namespace) -> None:
    protocol = load_protocol(args.config)
    source_curves = (
        args.source / "curves"
        if (args.source / "curves").is_dir()
        else args.source
    )
    paths = sorted(source_curves.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No development CSV files in {source_curves}")
    output_curves = args.output / "curves"
    output_curves.mkdir(parents=True, exist_ok=True)
    audits: list[dict[str, object]] = []
    curves: dict[str, pd.DataFrame] = {}
    for path in paths:
        source = pd.read_csv(path)
        lower_columns = {
            str(column).strip().lower(): str(column)
            for column in source.columns
        }
        cycle_column = next(
            (
                lower_columns[name]
                for name in ("cycle", "cycle_index", "cycle_number")
                if name in lower_columns
            ),
            None,
        )
        if cycle_column is None:
            continue
        source_smoothed_capacity_ignored = False
        if {"raw_capacity", "battery_id", "batch"}.issubset(
            source.columns
        ):
            raw_Ah = pd.to_numeric(
                source["raw_capacity"],
                errors="coerce",
            )
            smoothed_Ah = pd.Series(
                causal_smooth(
                    raw_Ah.to_numpy(float),
                    int(
                        protocol["task"][
                            "causal_smoothing_window_cycles"
                        ]
                    ),
                )
            )
            source_smoothed_capacity_ignored = "capacity" in source.columns
            cell_id = str(source["battery_id"].iloc[0])
            batch = int(source["batch"].iloc[0])
        else:
            capacity_column = next(
                (
                    lower_columns[name]
                    for name in (
                        "capacity_ah",
                        "capacity",
                        "discharge_capacity_ah",
                    )
                    if name in lower_columns
                ),
                None,
            )
            batch_match = re.search(
                r"(?:^|[_-])B([13])(?:[_-]|$)",
                path.stem,
                flags=re.IGNORECASE,
            )
            if capacity_column is None or batch_match is None:
                continue
            batch = int(batch_match.group(1))
            cell_id = path.stem
            raw_Ah = pd.to_numeric(
                source[capacity_column],
                errors="coerce",
            )
            smoothed_Ah = pd.Series(
                causal_smooth(
                    raw_Ah.to_numpy(float),
                    int(
                        protocol["task"][
                            "causal_smoothing_window_cycles"
                        ]
                    ),
                )
            )
        if batch not in (1, 3):
            continue
        frame = pd.DataFrame(
            {
                "cycle": pd.to_numeric(
                    source[cycle_column],
                    errors="raise",
                ),
                "raw_capacity_Ah": raw_Ah,
                "smoothed_capacity_Ah": smoothed_Ah,
            }
        ).sort_values("cycle").reset_index(drop=True)
        cycles_float = frame["cycle"].to_numpy(float)
        if not np.allclose(cycles_float, np.round(cycles_float)):
            raise ValueError(f"{path}: cycles are not integer-valued")
        cycles = np.round(cycles_float).astype(int)
        if not np.array_equal(cycles, np.arange(1, len(cycles) + 1)):
            raise ValueError(f"{path}: development cycles are not contiguous")
        if frame["raw_capacity_Ah"].isna().any():
            raise ValueError(
                f"{path}: XJTU development capacity contains missing values"
            )
        q0, bol_values = bol_reference_from_values(
            frame["raw_capacity_Ah"].tolist(),
            nominal_capacity_Ah=float(
                protocol["development_data"][
                    "declared_nominal_capacity_Ah"
                ]
            ),
            count=int(protocol["soh_definition"]["reference_observations"]),
        )
        normalized = pd.DataFrame(
            {
                "cycle": cycles,
                "original_raw_capacity_Ah": frame["raw_capacity_Ah"],
                "original_model_capacity_Ah": frame[
                    "smoothed_capacity_Ah"
                ],
            }
        )
        normalized["raw_capacity"] = (
            normalized["original_raw_capacity_Ah"] / q0
        )
        normalized["capacity"] = (
            normalized["original_model_capacity_Ah"] / q0
        )
        normalized["measurement_observed"] = (
            normalized["raw_capacity"].notna().astype(int)
        )
        normalized["initial_capacity_Ah"] = 1.0
        normalized["bol_reference_capacity_Ah"] = q0
        normalized["original_initial_capacity_Ah"] = q0
        normalized["battery_id"] = cell_id
        normalized["batch"] = batch
        normalized.to_csv(output_curves / f"{cell_id}.csv", index=False)
        curves[cell_id] = normalized
        audits.append(
            {
                "battery_id": cell_id,
                "batch": batch,
                "source_file": str(path.resolve()),
                "source_sha256": sha256_file(path),
                "cycles": len(normalized),
                "bol_rule": "median_first_5_valid",
                "bol_reference_capacity_Ah": q0,
                "bol_first_five_valid_Ah": json.dumps(bol_values),
                "normalized_first_observation": float(
                    normalized["raw_capacity"].iloc[0]
                ),
                "soh_formula_direction": "capacity_divided_by_BOL_reference",
                "source_smoothed_capacity_ignored": (
                    source_smoothed_capacity_ignored
                ),
                "smoothing_recomputed": (
                    "seven-cycle causal trailing mean from raw Ah"
                ),
            }
        )
    counts = {
        batch: sum(int(item["batch"]) == batch for item in audits)
        for batch in (1, 3)
    }
    expected_count = int(
        protocol["development_data"]["expected_cells_per_batch"]
    )
    expected_counts = {1: expected_count, 3: expected_count}
    if counts != expected_counts:
        raise RuntimeError(
            f"Expected normalized development counts {expected_counts}, got {counts}"
        )
    reference_path = (
        args.config.resolve().parents[1]
        / "manifests"
        / "xjtu_expected_reference.csv"
    )
    reference = pd.read_csv(reference_path)
    expected_structure = {
        str(row.cell_id): {
            "batch": int(row.batch),
            "n_cycles": int(row.n_cycles),
            "source_sha256": str(row.source_sha256),
        }
        for row in reference.itertuples(index=False)
    }
    observed_structure = {
        str(row["battery_id"]): {
            "batch": int(row["batch"]),
            "n_cycles": int(row["cycles"]),
            "source_sha256": str(row["source_sha256"]),
        }
        for row in audits
    }
    if set(observed_structure) != set(expected_structure):
        raise RuntimeError(
            "XJTU B1+B3 physical-cell IDs differ from the frozen reference: "
            f"missing={sorted(set(expected_structure) - set(observed_structure))}, "
            f"extra={sorted(set(observed_structure) - set(expected_structure))}"
        )
    structural_mismatches = {
        cell_id: {
            "expected": {
                key: expected_structure[cell_id][key]
                for key in ("batch", "n_cycles")
            },
            "observed": {
                key: observed_structure[cell_id][key]
                for key in ("batch", "n_cycles")
            },
        }
        for cell_id in expected_structure
        if any(
            observed_structure[cell_id][key]
            != expected_structure[cell_id][key]
            for key in ("batch", "n_cycles")
        )
    }
    if structural_mismatches:
        raise RuntimeError(
            "XJTU B1+B3 cycle structure differs from the frozen reference: "
            f"{structural_mismatches}"
        )
    exact_source_hash_matches = sum(
        observed_structure[cell_id]["source_sha256"]
        == expected_structure[cell_id]["source_sha256"]
        for cell_id in expected_structure
    )
    audit_frame = pd.DataFrame(audits).sort_values(["batch", "battery_id"])
    audit_frame.to_csv(args.output / "development_soh_audit.csv", index=False)
    written_curves = {
        str(frame["battery_id"].iloc[0]): frame
        for frame in (
            pd.read_csv(path)
            for path in sorted(output_curves.glob("*.csv"))
        )
    }
    gate = {
        "status": "PASS",
        "created_utc": utc_now(),
        "config_sha256": json_sha256(protocol),
        "counts": counts,
        "physical_cells": len(written_curves),
        "normalization": protocol["development_data"]["normalization"],
        "reference_manifest_sha256": sha256_file(reference_path),
        "reference_structural_identity_pass": True,
        "exact_source_file_hash_matches": int(exact_source_hash_matches),
        "exact_source_file_hash_total": len(expected_structure),
        "source_hash_note": (
            "A mismatch may reflect an existing prepared CSV with extra "
            "columns; the exact source hash and normalized curve hash remain "
            "frozen. Cell IDs, batches, and cycle counts must match."
        ),
        "external_data_loaded": False,
        "hnei_data_loaded": False,
        "bit_data_loaded": False,
        "curves_sha256": curves_sha256(written_curves),
    }
    write_json(args.output / "development_preflight.json", gate)
    print(
        f"[PASS] normalized development curves: counts={counts} "
        f"hash={gate['curves_sha256']}"
    )


def load_development_curves(path: Path, protocol: dict) -> dict[str, pd.DataFrame]:
    gate = json.loads(
        (path / "development_preflight.json").read_text(encoding="utf-8")
    )
    if (
        gate.get("status") != "PASS"
        or gate.get("config_sha256") != json_sha256(protocol)
        or gate.get("external_data_loaded") is not False
        or gate.get("hnei_data_loaded") is not False
        or gate.get("bit_data_loaded") is not False
    ):
        raise RuntimeError("Normalized development gate failed")
    curves: dict[str, pd.DataFrame] = {}
    for csv_path in sorted((path / "curves").glob("*.csv")):
        frame = pd.read_csv(csv_path).sort_values("cycle").reset_index(drop=True)
        cell_id = str(frame["battery_id"].iloc[0])
        curves[cell_id] = frame
    counts = {
        batch: sum(int(frame["batch"].iloc[0]) == batch for frame in curves.values())
        for batch in (1, 3)
    }
    expected = int(protocol["development_data"]["expected_cells_per_batch"])
    if counts != {1: expected, 3: expected}:
        raise RuntimeError(f"Normalized development counts changed: {counts}")
    if curves_sha256(curves) != gate.get("curves_sha256"):
        raise RuntimeError("Normalized development content hash changed")
    return curves


def observed_training_end(frame: pd.DataFrame, threshold: float) -> int:
    hits = np.flatnonzero(frame["raw_capacity"].to_numpy(float) <= threshold)
    return int(hits[0]) if len(hits) else len(frame) - 1


@dataclass
class ArraySet:
    X: np.ndarray
    capacities: np.ndarray
    cycles: np.ndarray
    q0: np.ndarray
    targets: np.ndarray
    cells: np.ndarray
    target_cycles: np.ndarray


@dataclass
class WindowSets:
    train: ArraySet
    validation: ArraySet | None
    scaler: StandardScaler


def windows_for_cells(
    curves: Mapping[str, pd.DataFrame],
    cell_ids: Sequence[str],
    window: int,
    support_horizon: int,
    threshold: float,
    rolling_warmup: int,
) -> ArraySet:
    parts: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "X",
            "capacities",
            "cycles",
            "q0",
            "targets",
            "cells",
            "target_cycles",
        )
    }
    for cell_id in cell_ids:
        frame = curves[cell_id]
        q0 = 1.0
        support_end = observed_training_end(frame, threshold)
        use = frame.iloc[: support_end + 1].reset_index(drop=True)
        cycles = use["cycle"].to_numpy(int)
        capacities = use["capacity"].to_numpy(float)
        features = compute_features(cycles, capacities, q0)
        stop = len(use) - support_horizon + 1
        rows = list(range(window + rolling_warmup, stop))
        if len(rows) < 10:
            raise ValueError(f"{cell_id}: only {len(rows)} training windows")
        parts["X"].append(
            np.asarray(
                [features[index - window : index] for index in rows],
                dtype=np.float32,
            )
        )
        parts["capacities"].append(
            np.asarray(
                [capacities[index - window : index] for index in rows],
                dtype=np.float32,
            )
        )
        parts["cycles"].append(
            np.asarray(
                [cycles[index - window : index] for index in rows],
                dtype=np.float32,
            )
        )
        parts["q0"].append(np.asarray([q0] * len(rows), dtype=np.float32))
        parts["targets"].append(
            np.asarray(
                [
                    capacities[index : index + support_horizon]
                    for index in rows
                ],
                dtype=np.float32,
            )
        )
        parts["cells"].append(
            np.asarray([cell_id] * len(rows), dtype=object)
        )
        parts["target_cycles"].append(
            np.asarray([cycles[index] for index in rows], dtype=np.int32)
        )
    return ArraySet(
        X=np.concatenate(parts["X"]),
        capacities=np.concatenate(parts["capacities"]),
        cycles=np.concatenate(parts["cycles"]),
        q0=np.concatenate(parts["q0"]),
        targets=np.concatenate(parts["targets"]),
        cells=np.concatenate(parts["cells"]),
        target_cycles=np.concatenate(parts["target_cycles"]),
    )


def transform_array_set(data: ArraySet, scaler: StandardScaler) -> ArraySet:
    shape = data.X.shape
    transformed = (
        scaler.transform(data.X.reshape(-1, shape[-1]))
        .reshape(shape)
        .astype(np.float32)
    )
    return ArraySet(
        X=transformed,
        capacities=data.capacities,
        cycles=data.cycles,
        q0=data.q0,
        targets=data.targets,
        cells=data.cells,
        target_cycles=data.target_cycles,
    )


def build_window_sets(
    curves: Mapping[str, pd.DataFrame],
    train_cells: Sequence[str],
    validation_cells: Sequence[str] | None,
    window: int,
    support_horizon: int,
    threshold: float,
    rolling_warmup: int,
) -> WindowSets:
    raw_train = windows_for_cells(
        curves,
        train_cells,
        window,
        support_horizon,
        threshold,
        rolling_warmup,
    )
    scaler = StandardScaler().fit(
        raw_train.X.reshape(-1, raw_train.X.shape[-1])
    )
    validation = None
    if validation_cells:
        validation = transform_array_set(
            windows_for_cells(
                curves,
                validation_cells,
                window,
                support_horizon,
                threshold,
                rolling_warmup,
            ),
            scaler,
        )
    return WindowSets(
        train=transform_array_set(raw_train, scaler),
        validation=validation,
        scaler=scaler,
    )


class RolloutDataset(Dataset):
    def __init__(self, arrays: ArraySet):
        self.X = torch.from_numpy(arrays.X)
        self.capacities = torch.from_numpy(arrays.capacities)
        self.cycles = torch.from_numpy(arrays.cycles)
        self.q0 = torch.from_numpy(arrays.q0)
        self.targets = torch.from_numpy(arrays.targets)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        return (
            self.X[index],
            self.capacities[index],
            self.cycles[index],
            self.q0[index],
            self.targets[index],
        )


def append_predicted_step(
    feature_window: torch.Tensor,
    capacity_window: torch.Tensor,
    cycle_window: torch.Tensor,
    q0: torch.Tensor,
    prediction: torch.Tensor,
    scaler_mean: torch.Tensor,
    scaler_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del q0
    next_cycle = cycle_window[:, -1] + 1.0
    last_capacity = capacity_window[:, -1]
    recent = torch.cat([capacity_window[:, -4:], prediction[:, None]], dim=1)
    rolling_mean = recent.mean(dim=1)
    rolling_std = recent.std(dim=1, unbiased=False)
    lag_index = max(0, capacity_window.shape[1] - 5)
    cycle_delta = (next_cycle - cycle_window[:, lag_index]).clamp(min=1.0)
    degradation = (
        prediction - capacity_window[:, lag_index]
    ) / cycle_delta
    raw = torch.stack(
        [
            next_cycle / 1000.0,
            prediction,
            1.0 - prediction,
            prediction - last_capacity,
            rolling_mean,
            rolling_std,
            degradation,
        ],
        dim=1,
    )
    scaled = (raw - scaler_mean) / scaler_scale
    return (
        torch.cat([feature_window[:, 1:, :], scaled[:, None, :]], dim=1),
        torch.cat([capacity_window[:, 1:], prediction[:, None]], dim=1),
        torch.cat([cycle_window[:, 1:], next_cycle[:, None]], dim=1),
    )


def rollout_loss(
    model: nn.Module,
    X: torch.Tensor,
    capacities: torch.Tensor,
    cycles: torch.Tensor,
    q0: torch.Tensor,
    targets: torch.Tensor,
    steps: int,
    scaler_mean: torch.Tensor,
    scaler_scale: torch.Tensor,
    threshold: float,
    include_penalty: bool,
    loss_options: Mapping[str, object],
) -> torch.Tensor:
    if steps > targets.shape[1]:
        raise ValueError("Loss horizon exceeds common target support")
    huber = nn.HuberLoss(
        reduction="none",
        delta=float(loss_options["huber_delta_SOH"]),
    )
    feature_window = X
    capacity_window = capacities
    cycle_window = cycles
    weights = torch.pow(
        torch.tensor(
            float(loss_options["gamma"]),
            device=X.device,
            dtype=X.dtype,
        ),
        torch.arange(steps, device=X.device, dtype=X.dtype),
    )
    weights = weights / weights.sum()
    losses = []
    for step in range(steps):
        prediction = model(feature_window, capacity_window)
        target = targets[:, step]
        eol_weight = 1.0 + float(
            loss_options["threshold_weight_strength"]
        ) * torch.exp(
            -torch.abs(target - threshold)
            / float(loss_options["threshold_weight_band_SOH"])
        )
        loss = huber(prediction, target) * eol_weight
        if include_penalty:
            loss = loss + float(
                loss_options["upward_penalty"]
            ) * torch.relu(
                prediction
                - capacity_window[:, -1]
                - float(loss_options["upward_tolerance_SOH"])
            ).square()
        losses.append(loss.mean() * weights[step])
        feature_window, capacity_window, cycle_window = append_predicted_step(
            feature_window,
            capacity_window,
            cycle_window,
            q0,
            prediction,
            scaler_mean,
            scaler_scale,
        )
    return torch.stack(losses).sum()


def data_loader(
    arrays: ArraySet,
    batch_size: int,
    shuffle: bool,
    workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        RolloutDataset(arrays),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )


def train_selection(
    windows: WindowSets,
    ablation: str,
    steps: int,
    seed: int,
    max_epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    threshold: float,
    device: torch.device,
    workers: int,
    model_options: Mapping[str, object],
) -> tuple[int, pd.DataFrame]:
    if windows.validation is None:
        raise ValueError("Validation windows are required")
    set_seed(seed)
    model = PaperMSTTVariant(
        len(FEATURE_NAMES),
        ablation=ablation,
        n_heads=int(model_options["attention_heads"]),
        dropout=float(model_options["dropout"]),
        residual_scale=float(model_options["residual_scale_SOH"]),
        rope_base=float(model_options["rope_base"]),
        trend_delta_window=int(model_options["trend_delta_window"]),
        trend_delta_clip_soh=tuple(
            float(value)
            for value in model_options["trend_delta_clip_SOH"]
        ),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    train_loader = data_loader(
        windows.train,
        batch_size,
        True,
        workers,
        device,
    )
    validation_loader = data_loader(
        windows.validation,
        batch_size,
        False,
        workers,
        device,
    )
    mean = torch.tensor(
        windows.scaler.mean_,
        dtype=torch.float32,
        device=device,
    )
    scale = torch.tensor(
        windows.scaler.scale_,
        dtype=torch.float32,
        device=device,
    )
    best_epoch = 1
    best_loss = float("inf")
    stale = 0
    rows: list[dict[str, float | int]] = []
    autocast_context = nullcontext
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_values = []
        for X, capacities, cycles, q0, targets in train_loader:
            X = X.to(device)
            capacities = capacities.to(device)
            cycles = cycles.to(device)
            q0 = q0.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context():
                loss = rollout_loss(
                    model,
                    X,
                    capacities,
                    cycles,
                    q0,
                    targets,
                    steps,
                    mean,
                    scale,
                    threshold,
                    include_penalty=True,
                    loss_options=model_options,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_values.append(float(loss.detach().cpu()))
        model.eval()
        validation_values = []
        with torch.no_grad():
            for X, capacities, cycles, q0, targets in validation_loader:
                loss = rollout_loss(
                    model,
                    X.to(device),
                    capacities.to(device),
                    cycles.to(device),
                    q0.to(device),
                    targets.to(device),
                    steps,
                    mean,
                    scale,
                    threshold,
                    include_penalty=False,
                    loss_options=model_options,
                )
                validation_values.append(float(loss.cpu()))
        train_loss = float(np.mean(train_values))
        validation_loss = float(np.mean(validation_values))
        rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_loss - 1e-8:
            best_epoch = epoch
            best_loss = validation_loss
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    return best_epoch, pd.DataFrame(rows)


def train_final(
    windows: WindowSets,
    ablation: str,
    steps: int,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    threshold: float,
    device: torch.device,
    workers: int,
    model_options: Mapping[str, object],
) -> tuple[nn.Module, pd.DataFrame]:
    set_seed(seed)
    model = PaperMSTTVariant(
        len(FEATURE_NAMES),
        ablation=ablation,
        n_heads=int(model_options["attention_heads"]),
        dropout=float(model_options["dropout"]),
        residual_scale=float(model_options["residual_scale_SOH"]),
        rope_base=float(model_options["rope_base"]),
        trend_delta_window=int(model_options["trend_delta_window"]),
        trend_delta_clip_soh=tuple(
            float(value)
            for value in model_options["trend_delta_clip_SOH"]
        ),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    loader = data_loader(
        windows.train,
        batch_size,
        True,
        workers,
        device,
    )
    mean = torch.tensor(
        windows.scaler.mean_,
        dtype=torch.float32,
        device=device,
    )
    scale = torch.tensor(
        windows.scaler.scale_,
        dtype=torch.float32,
        device=device,
    )
    rows: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        values = []
        for X, capacities, cycles, q0, targets in loader:
            X = X.to(device)
            capacities = capacities.to(device)
            cycles = cycles.to(device)
            q0 = q0.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = rollout_loss(
                model,
                X,
                capacities,
                cycles,
                q0,
                targets,
                steps,
                mean,
                scale,
                threshold,
                include_penalty=True,
                loss_options=model_options,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            values.append(float(loss.detach().cpu()))
        rows.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(values)),
            }
        )
    model.eval()
    return model, pd.DataFrame(rows)


def support_sha256(arrays: ArraySet) -> str:
    frame = pd.DataFrame(
        {
            "cell_id": arrays.cells.astype(str),
            "target_cycle": arrays.target_cycles.astype(int),
        }
    ).sort_values(["cell_id", "target_cycle"])
    return hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()


def command_train(args: argparse.Namespace) -> None:
    protocol = load_protocol(args.config)
    curves = load_development_curves(args.development, protocol)
    training = protocol["training"]
    task = protocol["task"]
    train_batch = int(protocol["development_data"]["selection_train_batch"])
    validation_batch = int(
        protocol["development_data"]["selection_validation_batch"]
    )
    train_cells = sorted(
        cell_id
        for cell_id, frame in curves.items()
        if int(frame["batch"].iloc[0]) == train_batch
    )
    validation_cells = sorted(
        cell_id
        for cell_id, frame in curves.items()
        if int(frame["batch"].iloc[0]) == validation_batch
    )
    all_cells = sorted(curves)
    model_specs = [
        specification
        for specification in protocol["models"]
        if str(specification["id"]).startswith("mstt_")
    ]
    seeds = [int(seed) for seed in training["seeds"]]
    max_epochs = int(training["max_epochs"])
    patience = int(training["patience"])
    if args.quick_test:
        train_cells = train_cells[:1]
        validation_cells = validation_cells[:1]
        all_cells = train_cells + validation_cells
        model_specs = model_specs[:1]
        seeds = seeds[:1]
        max_epochs = 2
        patience = 1
    support_horizon = max(
        int(specification["common_target_support_horizon"])
        for specification in model_specs
    )
    window = int(task["input_window_cycles"])
    rolling_warmup = int(task["rolling_warmup_cycles"])
    threshold = float(task["eol_threshold_SOH"])
    selection = build_window_sets(
        curves,
        train_cells,
        validation_cells,
        window,
        support_horizon,
        threshold,
        rolling_warmup,
    )
    final = build_window_sets(
        curves,
        all_cells,
        None,
        window,
        support_horizon,
        threshold,
        rolling_warmup,
    )
    common_support_hash = support_sha256(selection.train)
    device = choose_device(args.device)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    batch_size = int(training["batch_size"])
    learning_rate = float(training["learning_rate"])
    weight_decay = float(training["weight_decay"])
    model_options = {
        "attention_heads": training["attention_heads"],
        "dropout": training["dropout"],
        "residual_scale_SOH": training["residual_scale_SOH"],
        "rope_base": training["rope_base"],
        "trend_delta_window": training["trend_delta_window"],
        "trend_delta_clip_SOH": protocol[
            "ah_to_soh_conversion"
        ]["trend_delta_clip_SOH"],
        "gamma": training["gamma"],
        "huber_delta_SOH": training["huber_delta_SOH"],
        "threshold_weight_strength": training[
            "threshold_weight_strength"
        ],
        "threshold_weight_band_SOH": training[
            "threshold_weight_band_SOH"
        ],
        "upward_tolerance_SOH": training["upward_tolerance_SOH"],
        "upward_penalty": training["upward_penalty"],
    }
    config_hash = json_sha256(protocol)
    development_hash = curves_sha256(curves)
    jobs: list[dict[str, object]] = []
    args.output.mkdir(parents=True, exist_ok=True)
    for specification in model_specs:
        model_id = str(specification["id"])
        ablation = str(specification["ablation"])
        steps = int(specification["loss_rollout_steps"])
        for seed in seeds:
            job_dir = args.output / "models" / model_id / f"seed_{seed}"
            audit_path = job_dir / "job_audit.json"
            model_path = job_dir / "model.pt"
            scaler_path = job_dir / "scaler.joblib"
            scaler_json_path = job_dir / "scaler.json"
            if (
                audit_path.is_file()
                and model_path.is_file()
                and scaler_path.is_file()
                and scaler_json_path.is_file()
            ):
                audit = json.loads(audit_path.read_text(encoding="utf-8"))
                if (
                    audit.get("status") == "PASS"
                    and audit.get("config_sha256") == config_hash
                    and audit.get("development_data_sha256") == development_hash
                    and audit.get("model_sha256") == sha256_file(model_path)
                    and audit.get("scaler_sha256") == sha256_file(scaler_path)
                    and audit.get("scaler_json_sha256")
                    == sha256_file(scaler_json_path)
                ):
                    print(f"[SKIP] {model_id} seed={seed}")
                    jobs.append(audit)
                    continue
                raise RuntimeError(
                    f"Existing job is inconsistent; use a new output root: {job_dir}"
                )
            job_dir.mkdir(parents=True, exist_ok=True)
            started = time.time()
            selected_epoch, selection_history = train_selection(
                selection,
                ablation,
                steps,
                seed,
                max_epochs,
                patience,
                batch_size,
                learning_rate,
                weight_decay,
                threshold,
                device,
                args.num_workers,
                model_options,
            )
            model, final_history = train_final(
                final,
                ablation,
                steps,
                seed,
                selected_epoch,
                batch_size,
                learning_rate,
                weight_decay,
                threshold,
                device,
                args.num_workers,
                model_options,
            )
            selection_history.to_csv(
                job_dir / "epoch_selection.csv",
                index=False,
            )
            final_history.to_csv(
                job_dir / "final_training.csv",
                index=False,
            )
            joblib.dump(final.scaler, scaler_path)
            write_json(
                scaler_json_path,
                {
                    "feature_names": list(FEATURE_NAMES),
                    "mean": final.scaler.mean_.tolist(),
                    "scale": final.scaler.scale_.tolist(),
                    "variance": final.scaler.var_.tolist(),
                    "n_features": int(final.scaler.n_features_in_),
                    "n_samples_seen": int(final.scaler.n_samples_seen_),
                    "fit_data": "XJTU_B1_plus_B3_final_refit_only",
                },
            )
            identity = identity_record(model)
            if ablation == "full" and int(
                identity["trainable_parameters"]
            ) != int(
                protocol["source_implementation"][
                    "full_model_trainable_parameters"
                ]
            ):
                raise RuntimeError(
                    f"{model_id}: unexpected parameter count "
                    f"{identity['trainable_parameters']}"
                )
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "identity": identity,
                    "model_id": model_id,
                    "ablation": ablation,
                    "loss_rollout_steps": steps,
                    "seed": seed,
                    "selected_epoch": selected_epoch,
                    "config_sha256": config_hash,
                    "development_data_sha256": development_hash,
                    "common_support_sha256": common_support_hash,
                },
                model_path,
            )
            audit = {
                "status": "PASS",
                "model_id": model_id,
                "seed": seed,
                "ablation": ablation,
                "loss_rollout_steps": steps,
                "selected_epoch": selected_epoch,
                "train_cells": train_cells,
                "validation_cells": validation_cells,
                "final_fit_cells": all_cells,
                "external_data_loaded": False,
                "target_unit": "SOH_fraction",
                "batch_size": batch_size,
                "amp": False,
                "device": str(device),
                "identity": identity,
                "config_sha256": config_hash,
                "development_data_sha256": development_hash,
                "common_support_sha256": common_support_hash,
                "model_sha256": sha256_file(model_path),
                "scaler_sha256": sha256_file(scaler_path),
                "scaler_json_sha256": sha256_file(scaler_json_path),
                "runtime_seconds": time.time() - started,
                "quick_test": bool(args.quick_test),
            }
            write_json(audit_path, audit)
            jobs.append(audit)
            print(
                f"[PASS] {model_id} seed={seed} epoch={selected_epoch} "
                f"runtime={audit['runtime_seconds'] / 60:.1f} min"
            )
    manifest = {
        "status": "PASS",
        "created_utc": utc_now(),
        "config_sha256": config_hash,
        "development_data_sha256": development_hash,
        "common_support_sha256": common_support_hash,
        "target_unit": "SOH_fraction",
        "external_data_loaded": False,
        "quick_test": bool(args.quick_test),
        "jobs": jobs,
    }
    write_json(args.output / "frozen_model_manifest.json", manifest)
    print(f"[PASS] frozen normalized-SOH models: {args.output}")


def command_freeze(args: argparse.Namespace) -> None:
    protocol = load_protocol(args.config)
    config_hash = json_sha256(protocol)
    inventory = json.loads(
        (args.inventory / "inventory.json").read_text(encoding="utf-8")
    )
    development_gate = json.loads(
        (args.development / "development_preflight.json").read_text(
            encoding="utf-8"
        )
    )
    model_manifest = json.loads(
        (args.models / "frozen_model_manifest.json").read_text(encoding="utf-8")
    )
    if (
        inventory.get("status") != "PASS"
        or inventory.get("config_sha256") != config_hash
        or development_gate.get("status") != "PASS"
        or development_gate.get("config_sha256") != config_hash
        or model_manifest.get("status") != "PASS"
        or model_manifest.get("config_sha256") != config_hash
        or model_manifest.get("external_data_loaded") is not False
        or model_manifest.get("quick_test") is not False
    ):
        raise RuntimeError("One or more pre-evaluation freeze gates failed")
    expected_jobs = {
        (str(specification["id"]), int(seed))
        for specification in protocol["models"]
        if str(specification["id"]).startswith("mstt_")
        for seed in protocol["training"]["seeds"]
    }
    observed_jobs = {
        (str(job.get("model_id")), int(job.get("seed")))
        for job in model_manifest.get("jobs", [])
    }
    if observed_jobs != expected_jobs:
        raise RuntimeError(
            f"Frozen job set is incomplete: expected={sorted(expected_jobs)}, "
            f"observed={sorted(observed_jobs)}"
        )
    for model_id, seed in sorted(expected_jobs):
        job_dir = args.models / "models" / model_id / f"seed_{seed}"
        audit = json.loads(
            (job_dir / "job_audit.json").read_text(encoding="utf-8")
        )
        model_path = job_dir / "model.pt"
        scaler_path = job_dir / "scaler.joblib"
        scaler_json_path = job_dir / "scaler.json"
        if (
            audit.get("status") != "PASS"
            or audit.get("quick_test") is not False
            or audit.get("external_data_loaded") is not False
            or audit.get("config_sha256") != config_hash
            or audit.get("development_data_sha256")
            != model_manifest.get("development_data_sha256")
            or audit.get("common_support_sha256")
            != model_manifest.get("common_support_sha256")
            or audit.get("model_sha256") != sha256_file(model_path)
            or audit.get("scaler_sha256") != sha256_file(scaler_path)
            or audit.get("scaler_json_sha256")
            != sha256_file(scaler_json_path)
        ):
            raise RuntimeError(f"Frozen job audit failed: {job_dir}")
    if args.evaluation.exists() and any(args.evaluation.rglob("*")):
        raise RuntimeError(
            "Evaluation output already exists; cannot create a pre-evaluation freeze"
        )
    if args.output_zip.exists():
        raise FileExistsError(
            f"Freeze ZIP already exists; choose a new path: {args.output_zip}"
        )
    source_files = [
        path
        for path in sorted(args.project_root.rglob("*"))
        if path.is_file()
        and ".venv" not in path.parts
        and "__pycache__" not in path.parts
        and ".git" not in path.parts
    ]
    model_files = [
        path for path in sorted(args.models.rglob("*")) if path.is_file()
    ]
    manifest = {
        "status": "PASS",
        "created_utc": utc_now(),
        "analysis_class": protocol["analysis_class"],
        "claim_limit": protocol["claim_limit"],
        "config_sha256": config_hash,
        "archive_sha256": inventory["archive_sha256"],
        "archive_outcome_values_parsed_by_inventory": inventory[
            "outcome_values_parsed"
        ],
        "exposure_declaration": {
            "hnei_structure_and_selected_capacity_values_seen": True,
            "hnei_model_rankings_seen_before_this_freeze": False,
            "bit_structure_seen_before_this_freeze": False,
            "bit_model_outputs_seen_before_this_freeze": False,
        },
        "workflow_evaluation_outputs_present_before_freeze": False,
        "model_predictions_generated_by_this_workflow_before_freeze": False,
        "development_data_sha256": model_manifest[
            "development_data_sha256"
        ],
        "common_support_sha256": model_manifest["common_support_sha256"],
        "source_files": {
            str(path.relative_to(args.project_root)): sha256_file(path)
            for path in source_files
        },
        "model_files": {
            str(path.relative_to(args.models)): sha256_file(path)
            for path in model_files
        },
    }
    manifest_path = args.output_zip.with_suffix(".manifest.json")
    write_json(manifest_path, manifest)
    args.output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        args.output_zip,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        for path in source_files:
            archive.write(
                path,
                Path("code") / path.relative_to(args.project_root),
            )
        for path in model_files:
            archive.write(
                path,
                Path("frozen_models") / path.relative_to(args.models),
            )
        archive.write(
            args.inventory / "inventory.json",
            "gates/inventory.json",
        )
        archive.write(
            args.development / "development_preflight.json",
            "gates/development_preflight.json",
        )
        archive.write(manifest_path, "PRE_EVALUATION_FREEZE.json")
    bad_member = None
    with zipfile.ZipFile(args.output_zip) as archive:
        bad_member = archive.testzip()
    if bad_member is not None:
        raise RuntimeError(f"Freeze ZIP CRC failure: {bad_member}")
    freeze_hash = sha256_file(args.output_zip)
    sha_path = Path(str(args.output_zip) + ".sha256")
    sha_path.write_text(
        f"{freeze_hash}  {args.output_zip.name}\n",
        encoding="utf-8",
    )
    print(f"[PASS] pre-evaluation freeze: {args.output_zip}")
    print(f"SHA256={freeze_hash}")


COMMIT_PATTERN = re.compile(r"^[0-9a-fA-F]{40}$")
DOI_PATTERN = re.compile(
    r"^10\.\d{4,9}/zenodo\.\d+$",
    flags=re.IGNORECASE,
)
GITHUB_RELEASE_PATTERN = re.compile(
    r"^https://github\.com/[^/]+/[^/]+/releases/tag/[^/]+$",
    flags=re.IGNORECASE,
)


def command_register(args: argparse.Namespace) -> None:
    protocol = load_protocol(args.config)
    commit = args.commit.strip()
    github_release = args.github_release.strip()
    doi = (
        args.doi.strip()
        .removeprefix("https://doi.org/")
        .removeprefix("http://doi.org/")
    )
    if not COMMIT_PATTERN.fullmatch(commit):
        raise ValueError("Commit must be a complete 40-character hexadecimal hash")
    if not GITHUB_RELEASE_PATTERN.fullmatch(github_release):
        raise ValueError("A concrete GitHub release-tag URL is required")
    if not DOI_PATTERN.fullmatch(doi):
        raise ValueError("DOI must look like 10.xxxx/zenodo.xxxxxxxx")
    if args.evaluation.exists() and any(args.evaluation.rglob("*")):
        raise RuntimeError(
            "Evaluation already exists; receipt must precede model evaluation"
        )
    manifest = json.loads(args.freeze_manifest.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "PASS"
        or manifest.get("config_sha256") != json_sha256(protocol)
        or (
            manifest.get("hnei_model_predictions_generated_before_freeze")
            is not False
            and manifest.get(
                "model_predictions_generated_by_this_workflow_before_freeze"
            )
            is not False
        )
    ):
        raise RuntimeError("Freeze manifest is invalid")
    with zipfile.ZipFile(args.freeze_zip) as archive:
        members = set(archive.namelist())
        manifest_member = (
            "SOH_DEVELOPMENT_FREEZE.json"
            if "SOH_DEVELOPMENT_FREEZE.json" in members
            else "PRE_EVALUATION_FREEZE.json"
        )
        archived_manifest = json.loads(
            archive.read(manifest_member).decode("utf-8")
        )
    if archived_manifest != manifest:
        raise RuntimeError(
            "External freeze manifest does not match the manifest inside the ZIP"
        )
    receipt = {
        "status": "PASS",
        "registered_utc": utc_now(),
        "analysis_class": protocol["analysis_class"],
        "claim_limit": protocol["claim_limit"],
        "commit": commit.lower(),
        "github_release": github_release,
        "zenodo_version_doi": doi.lower(),
        "config_sha256": json_sha256(protocol),
        "freeze_zip": args.freeze_zip.name,
        "freeze_zip_sha256": sha256_file(args.freeze_zip),
        "freeze_manifest_sha256": sha256_file(args.freeze_manifest),
        "external_model_evaluation_existed_before_receipt": False,
        "registered_artifact_role": "xjtu_only_soh_development_freeze",
        "untouched_preregistration_claim_allowed": False,
    }
    write_json(args.output, receipt)
    print(f"[PASS] pre-evaluation receipt: {args.output}")


def load_external_curves(
    path: Path,
    protocol: dict,
) -> dict[str, pd.DataFrame]:
    gate = json.loads(
        (path / "external_preflight.json").read_text(encoding="utf-8")
    )
    if (
        gate.get("status") != "PASS"
        or gate.get("config_sha256") != json_sha256(protocol)
        or gate.get("physical_cells")
        != int(protocol["hnei"]["expected_cells"])
    ):
        raise RuntimeError("HNEI external preflight gate failed")
    curves: dict[str, pd.DataFrame] = {}
    for csv_path in sorted((path / "curves").glob("B7_HNEI_*.csv")):
        frame = pd.read_csv(csv_path)
        required = {
            "cycle",
            "raw_capacity",
            "capacity",
            "measurement_observed",
            "battery_id",
            "batch",
            "initial_capacity_Ah",
        }
        if not required.issubset(frame.columns):
            raise ValueError(f"{csv_path}: missing external columns")
        cell_id = str(frame["battery_id"].iloc[0])
        curves[cell_id] = frame
    if len(curves) != int(protocol["hnei"]["expected_cells"]):
        raise RuntimeError(f"Expected 14 HNEI curves, got {len(curves)}")
    if curves_sha256(curves) != gate.get("curves_sha256"):
        raise RuntimeError("Prepared HNEI curve hash changed")
    return curves


def load_frozen_models(
    protocol: dict,
    frozen_root: Path,
    device: torch.device,
) -> dict[str, dict[int, tuple[nn.Module, StandardScaler]]]:
    manifest = json.loads(
        (frozen_root / "frozen_model_manifest.json").read_text(encoding="utf-8")
    )
    config_hash = json_sha256(protocol)
    if (
        manifest.get("status") != "PASS"
        or manifest.get("config_sha256") != config_hash
        or manifest.get("external_data_loaded") is not False
        or manifest.get("quick_test") is not False
    ):
        raise RuntimeError("Frozen model manifest failed")
    seeds = [int(seed) for seed in protocol["training"]["seeds"]]
    result: dict[str, dict[int, tuple[nn.Module, StandardScaler]]] = {}
    for specification in protocol["models"]:
        model_id = str(specification["id"])
        if not model_id.startswith("mstt_"):
            continue
        result[model_id] = {}
        for seed in seeds:
            job_dir = frozen_root / "models" / model_id / f"seed_{seed}"
            audit = json.loads(
                (job_dir / "job_audit.json").read_text(encoding="utf-8")
            )
            model_path = job_dir / "model.pt"
            scaler_path = job_dir / "scaler.joblib"
            scaler_json_path = job_dir / "scaler.json"
            if (
                audit.get("status") != "PASS"
                or audit.get("config_sha256") != config_hash
                or audit.get("external_data_loaded") is not False
                or audit.get("model_sha256") != sha256_file(model_path)
                or audit.get("scaler_sha256") != sha256_file(scaler_path)
                or audit.get("scaler_json_sha256")
                != sha256_file(scaler_json_path)
            ):
                raise RuntimeError(f"Frozen checkpoint gate failed: {job_dir}")
            model = PaperMSTTVariant(
                len(FEATURE_NAMES),
                ablation=str(specification["ablation"]),
                n_heads=int(protocol["training"]["attention_heads"]),
                dropout=float(protocol["training"]["dropout"]),
                residual_scale=float(
                    protocol["training"]["residual_scale_SOH"]
                ),
                rope_base=float(protocol["training"]["rope_base"]),
                trend_delta_window=int(
                    protocol["training"]["trend_delta_window"]
                ),
                trend_delta_clip_soh=tuple(
                    float(value)
                    for value in protocol["ah_to_soh_conversion"][
                        "trend_delta_clip_SOH"
                    ]
                ),
            ).to(device)
            checkpoint = torch.load(
                model_path,
                map_location=device,
                weights_only=False,
            )
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            if identity_record(model) != checkpoint["identity"]:
                raise RuntimeError(f"Model identity changed: {model_path}")
            model.eval()
            result[model_id][seed] = (model, joblib.load(scaler_path))
    return result


def local_linear_next(
    capacities: Sequence[float],
    lookback: int = 20,
    slope_clip: tuple[float, float] = (-0.02, -1e-6),
) -> float:
    values = np.asarray(capacities[-lookback:], dtype=float)
    if len(values) < 2:
        return float(values[-1])
    slope, intercept = np.polyfit(
        np.arange(len(values), dtype=float),
        values,
        1,
    )
    slope = float(np.clip(slope, *slope_clip))
    return float(intercept + slope * len(values))


def predict_neural(
    model: nn.Module,
    scaler: StandardScaler,
    cycles: Sequence[int],
    capacities: Sequence[float],
    window: int,
    device: torch.device,
) -> float:
    features = compute_features(cycles, capacities, 1.0)[-window:]
    scaled = scaler.transform(features).astype(np.float32)
    with torch.no_grad():
        X = torch.from_numpy(scaled[None]).to(device)
        capacity_tensor = torch.from_numpy(
            np.asarray(capacities[-window:], dtype=np.float32)[None]
        ).to(device)
        return float(model(X, capacity_tensor).detach().cpu().item())


def recursive_forecast(
    model_name: str,
    frame: pd.DataFrame,
    cutoff: int,
    horizon: int,
    threshold: float,
    window: int,
    clip_bounds: tuple[float, float],
    model: nn.Module | None = None,
    scaler: StandardScaler | None = None,
    device: torch.device | None = None,
    minimum_history: int = 20,
    local_linear_lookback: int = 20,
    local_linear_slope_clip: tuple[float, float] = (-0.02, -1e-6),
) -> tuple[pd.DataFrame, int | None]:
    observed = frame[frame["cycle"] <= cutoff]
    cycles = observed["cycle"].astype(int).tolist()
    capacities = observed["capacity"].astype(float).tolist()
    if len(cycles) < max(window, minimum_history):
        raise ValueError(
            f"cutoff {cutoff} has fewer than "
            f"{max(window, minimum_history)} physical cycles"
        )
    predicted_eol = None
    rows = []
    for step in range(1, horizon + 1):
        cycle = cutoff + step
        if model_name == "persistence":
            prediction = float(capacities[-1])
        elif model_name == "local_linear_trend":
            prediction = local_linear_next(
                capacities,
                lookback=local_linear_lookback,
                slope_clip=local_linear_slope_clip,
            )
        else:
            if model is None or scaler is None or device is None:
                raise ValueError("Neural forecast requires model/scaler/device")
            prediction = predict_neural(
                model,
                scaler,
                cycles,
                capacities,
                window,
                device,
            )
        prediction = float(np.clip(prediction, *clip_bounds))
        cycles.append(cycle)
        capacities.append(prediction)
        if predicted_eol is None and prediction <= threshold:
            predicted_eol = cycle
        rows.append(
            {
                "cycle": cycle,
                "predicted_capacity": prediction,
            }
        )
    return pd.DataFrame(rows), predicted_eol


def interval_distance(value: int, interval: tuple[int, int]) -> float:
    lower, upper = interval
    if value < lower:
        return float(lower - value)
    if value > upper:
        return float(value - upper)
    return 0.0


def censor_aware_timing_score(
    predicted_eol: int | None,
    cutoff: int,
    horizon: int,
    truth_interval: tuple[int, int] | None,
    right_censor_cycle: int,
) -> float:
    if truth_interval is not None:
        if predicted_eol is not None:
            return interval_distance(predicted_eol, truth_interval)
        prediction_lower_bound = cutoff + horizon + 1
        return float(
            max(0, prediction_lower_bound - int(truth_interval[1]))
        )
    if predicted_eol is None or predicted_eol > right_censor_cycle:
        return 0.0
    return float(right_censor_cycle + 1 - predicted_eol)


def evaluate_forecast(
    frame: pd.DataFrame,
    forecast: pd.DataFrame,
    cutoff: int,
    predicted_eol: int | None,
    threshold: float,
    horizon: int,
    minimum_future_observations: int = 5,
) -> dict[str, object]:
    truth_interval = eol_interval(frame, threshold)
    observed_truth = frame[
        frame["measurement_observed"].eq(1)
    ].dropna(subset=["raw_capacity"])
    right_censor_cycle = int(observed_truth["cycle"].max())
    support_end = (
        int(truth_interval[1])
        if truth_interval is not None
        else right_censor_cycle
    )
    measured = frame[
        frame["measurement_observed"].eq(1)
        & frame["cycle"].gt(cutoff)
        & frame["cycle"].le(min(cutoff + horizon, support_end))
    ][["cycle", "raw_capacity"]]
    aligned = measured.merge(forecast, on="cycle", how="inner")
    if len(aligned) < minimum_future_observations:
        raise ValueError(
            "Insufficient genuine future HNEI support: "
            f"{len(aligned)} < {minimum_future_observations}"
        )
    residual = (
        aligned["predicted_capacity"].to_numpy(float)
        - aligned["raw_capacity"].to_numpy(float)
    )
    result: dict[str, object] = {
        "truth_event_type": (
            "interval_observed"
            if truth_interval is not None
            else "right_censored"
        ),
        "truth_event_observed": int(truth_interval is not None),
        "right_censor_cycle": (
            np.nan if truth_interval is not None else right_censor_cycle
        ),
        "pred_eol_cycle": (
            np.nan if predicted_eol is None else int(predicted_eol)
        ),
        "prediction_censor_cycle": (
            cutoff + horizon if predicted_eol is None else np.nan
        ),
        "censor_aware_timing_score_cycles": censor_aware_timing_score(
            predicted_eol,
            cutoff,
            horizon,
            truth_interval,
            right_censor_cycle,
        ),
        "future_capacity_MAE_SOH": float(np.mean(np.abs(residual))),
        "future_capacity_RMSE_SOH": float(np.sqrt(np.mean(residual**2))),
        "future_observed_points": len(aligned),
        "max_abs_capacity_error_SOH": float(np.max(np.abs(residual))),
    }
    if truth_interval is None:
        result.update(
            {
                "true_eol_interval_lower": np.nan,
                "true_eol_interval_upper": np.nan,
            }
        )
    else:
        result.update(
            {
                "true_eol_interval_lower": truth_interval[0],
                "true_eol_interval_upper": truth_interval[1],
            }
        )
    return result


def merge_seed_forecasts(
    seed_forecasts: Mapping[int, pd.DataFrame],
) -> pd.DataFrame:
    merged = None
    for seed, frame in sorted(seed_forecasts.items()):
        current = frame.rename(
            columns={
                "predicted_capacity": f"predicted_capacity_seed_{seed}"
            }
        )
        merged = (
            current
            if merged is None
            else merged.merge(current, on="cycle", how="inner")
        )
    if merged is None:
        raise ValueError("No seed forecasts")
    seed_columns = [
        column
        for column in merged
        if column.startswith("predicted_capacity_seed_")
    ]
    merged["predicted_capacity"] = merged[seed_columns].mean(axis=1)
    return merged


def first_predicted_eol(
    forecast: pd.DataFrame,
    threshold: float,
) -> int | None:
    hits = forecast[forecast["predicted_capacity"] <= threshold]
    return None if hits.empty else int(hits.iloc[0]["cycle"])


def command_evaluate(args: argparse.Namespace) -> None:
    protocol = load_protocol(args.config)
    validate_receipt(args.receipt, args.freeze_zip, protocol)
    curves = load_external_curves(args.external, protocol)
    device = choose_device(args.device)
    models = load_frozen_models(protocol, args.models, device)
    threshold = float(protocol["task"]["eol_threshold_SOH"])
    window = int(protocol["task"]["input_window_cycles"])
    horizon = int(protocol["task"]["forecast_horizon_cycles"])
    cutoffs = [int(value) for value in protocol["task"]["cutoffs"]]
    clip_bounds = tuple(
        map(float, protocol["task"]["prediction_clip_SOH"])
    )
    local_linear_specification = next(
        specification
        for specification in protocol["models"]
        if specification["id"] == "local_linear_trend"
    )
    local_linear_lookback = int(
        local_linear_specification["lookback_physical_cycles"]
    )
    local_linear_slope_clip = tuple(
        map(
            float,
            local_linear_specification["slope_clip_SOH_per_cycle"],
        )
    )
    minimum_history = int(
        protocol["task"]["minimum_pre_cutoff_physical_cycles"]
    )
    trajectories = args.output / "trajectories"
    trajectories.mkdir(parents=True, exist_ok=True)
    ensemble_records: list[dict[str, object]] = []
    seed_records: list[dict[str, object]] = []
    exclusions: list[dict[str, object]] = []
    for cell_id, frame in sorted(curves.items()):
        truth_interval = eol_interval(frame, threshold)
        last_observation = int(
            frame.loc[
                frame["measurement_observed"].eq(1),
                "cycle",
            ].max()
        )
        for cutoff in cutoffs:
            if cutoff >= last_observation:
                exclusions.append(
                    {
                        "cell_id": cell_id,
                        "cutoff": cutoff,
                        "reason": "cutoff_at_or_after_last_observation",
                    }
                )
                continue
            if truth_interval is not None and cutoff >= truth_interval[1]:
                exclusions.append(
                    {
                        "cell_id": cell_id,
                        "cutoff": cutoff,
                        "reason": "cutoff_at_or_after_eol_upper_bound",
                    }
                )
                continue
            for model_id, seed_models in models.items():
                forecasts: dict[int, pd.DataFrame] = {}
                for seed, (model, scaler) in seed_models.items():
                    forecast, seed_eol = recursive_forecast(
                        model_id,
                        frame,
                        cutoff,
                        horizon,
                        threshold,
                        window,
                        clip_bounds,
                        model=model,
                        scaler=scaler,
                        device=device,
                        minimum_history=minimum_history,
                    )
                    forecasts[seed] = forecast
                    metrics = evaluate_forecast(
                        frame,
                        forecast,
                        cutoff,
                        seed_eol,
                        threshold,
                        horizon,
                    )
                    seed_records.append(
                        {
                            "batch": 7,
                            "cell_id": cell_id,
                            "cutoff": cutoff,
                            "model": model_id,
                            "seed": seed,
                            **metrics,
                        }
                    )
                ensemble = merge_seed_forecasts(forecasts)
                ensemble_eol = first_predicted_eol(ensemble, threshold)
                metrics = evaluate_forecast(
                    frame,
                    ensemble,
                    cutoff,
                    ensemble_eol,
                    threshold,
                    horizon,
                )
                ensemble_records.append(
                    {
                        "batch": 7,
                        "cell_id": cell_id,
                        "cutoff": cutoff,
                        "model": model_id,
                        "aggregation": "mean_capacity_trajectory_across_seeds",
                        **metrics,
                    }
                )
                ensemble.insert(0, "model", model_id)
                ensemble.insert(0, "cutoff", cutoff)
                ensemble.insert(0, "cell_id", cell_id)
                ensemble.to_csv(
                    trajectories / f"{cell_id}_c{cutoff}_{model_id}.csv",
                    index=False,
                )
            baseline_ids = [
                str(specification["id"])
                for specification in protocol["models"]
                if not str(specification["id"]).startswith("mstt_")
            ]
            for baseline in baseline_ids:
                forecast, baseline_eol = recursive_forecast(
                    baseline,
                    frame,
                    cutoff,
                    horizon,
                    threshold,
                    window,
                    clip_bounds,
                    minimum_history=minimum_history,
                    local_linear_lookback=local_linear_lookback,
                    local_linear_slope_clip=local_linear_slope_clip,
                )
                metrics = evaluate_forecast(
                    frame,
                    forecast,
                    cutoff,
                    baseline_eol,
                    threshold,
                    horizon,
                )
                ensemble_records.append(
                    {
                        "batch": 7,
                        "cell_id": cell_id,
                        "cutoff": cutoff,
                        "model": baseline,
                        "aggregation": "deterministic",
                        **metrics,
                    }
                )
                forecast.insert(0, "model", baseline)
                forecast.insert(0, "cutoff", cutoff)
                forecast.insert(0, "cell_id", cell_id)
                forecast.to_csv(
                    trajectories / f"{cell_id}_c{cutoff}_{baseline}.csv",
                    index=False,
                )
        print(f"[EVALUATED] {cell_id}")
    ensemble_frame = pd.DataFrame(ensemble_records)
    seed_frame = pd.DataFrame(seed_records)
    exclusion_frame = pd.DataFrame(exclusions)
    ensemble_frame.to_csv(
        args.output / "ensemble_cell_records.csv",
        index=False,
    )
    seed_frame.to_csv(
        args.output / "seed_level_records.csv",
        index=False,
    )
    exclusion_frame.to_csv(
        args.output / "exclusions.csv",
        index=False,
    )
    expected_models = {
        str(specification["id"])
        for specification in protocol["models"]
    }
    expected_records = (
        int(protocol["hnei"]["expected_cells"])
        * len(cutoffs)
        * len(expected_models)
    )
    if (
        set(ensemble_frame["model"]) != expected_models
        or len(ensemble_frame) != expected_records
        or ensemble_frame.duplicated(
            ["cell_id", "cutoff", "model"]
        ).any()
    ):
        raise RuntimeError("HNEI evaluation record gate failed")
    audit = {
        "status": "PASS",
        "created_utc": utc_now(),
        "analysis_class": protocol["analysis_class"],
        "claim_limit": protocol["claim_limit"],
        "config_sha256": json_sha256(protocol),
        "external_data_sha256": curves_sha256(curves),
        "physical_cells": ensemble_frame["cell_id"].nunique(),
        "ensemble_records": len(ensemble_frame),
        "seed_records": len(seed_frame),
        "exclusions": len(exclusion_frame),
        "models": sorted(expected_models),
        "cutoffs": cutoffs,
        "test_selection_or_retraining": False,
        "uncertainty_analysis_included": False,
    }
    write_json(args.output / "evaluation_audit.json", audit)
    print(
        f"[PASS] HNEI evaluation: ensemble={len(ensemble_frame)} "
        f"seed={len(seed_frame)}"
    )


def paired_bootstrap_interval(
    differences: np.ndarray,
    repetitions: int,
    seed: int,
) -> tuple[float, float]:
    generator = np.random.default_rng(seed)
    indices = generator.integers(
        0,
        len(differences),
        size=(repetitions, len(differences)),
    )
    means = differences[indices].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)


def command_aggregate(args: argparse.Namespace) -> None:
    protocol = load_protocol(args.config)
    evaluation_audit = json.loads(
        (args.evaluation / "evaluation_audit.json").read_text(encoding="utf-8")
    )
    if (
        evaluation_audit.get("status") != "PASS"
        or evaluation_audit.get("config_sha256") != json_sha256(protocol)
    ):
        raise RuntimeError("Evaluation audit failed")
    records = pd.read_csv(args.evaluation / "ensemble_cell_records.csv")
    args.output.mkdir(parents=True, exist_ok=True)
    metrics = [
        "future_capacity_MAE_SOH",
        "future_capacity_RMSE_SOH",
        "censor_aware_timing_score_cycles",
    ]
    summary_rows = []
    for (cutoff, model), group in records.groupby(["cutoff", "model"]):
        row: dict[str, object] = {
            "cutoff": int(cutoff),
            "model": model,
            "physical_cells": group["cell_id"].nunique(),
        }
        for metric in metrics:
            values = group[metric].to_numpy(float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))
            row[f"{metric}_median"] = float(np.median(values))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values(["cutoff", "model"])
    summary.to_csv(args.output / "hnei_model_summary.csv", index=False)
    family = protocol["statistics"]["hnei_exploratory_family"]
    cutoff = int(family["cutoff"])
    reference = str(family["reference_model"])
    comparisons = [str(item) for item in family["comparators"]]
    family_metrics = [str(item) for item in family["metrics"]]
    primary = records[records["cutoff"].eq(cutoff)].copy()
    rows: list[dict[str, object]] = []
    repetitions = int(protocol["statistics"]["bootstrap_repetitions"])
    bootstrap_seed = int(protocol["statistics"]["bootstrap_seed"])
    for comparator_index, comparator in enumerate(comparisons):
        for metric_index, metric in enumerate(family_metrics):
            pivot = primary.pivot(
                index="cell_id",
                columns="model",
                values=metric,
            )
            pair = pivot[[reference, comparator]].dropna()
            difference = (
                pair[reference].to_numpy(float)
                - pair[comparator].to_numpy(float)
            )
            if np.allclose(difference, 0.0):
                statistic = 0.0
                p_value = 1.0
            else:
                result = wilcoxon(
                    difference,
                    zero_method="wilcox",
                    alternative="two-sided",
                    method="auto",
                )
                statistic = float(result.statistic)
                p_value = float(result.pvalue)
            lower, upper = paired_bootstrap_interval(
                difference,
                repetitions,
                bootstrap_seed
                + comparator_index * 100
                + metric_index,
            )
            rows.append(
                {
                    "cutoff": cutoff,
                    "metric": metric,
                    "reference_model": reference,
                    "comparator_model": comparator,
                    "physical_cells": len(pair),
                    "mean_paired_effect_reference_minus_comparator": float(
                        difference.mean()
                    ),
                    "median_paired_effect_reference_minus_comparator": float(
                        np.median(difference)
                    ),
                    "bootstrap_95ci_lower": lower,
                    "bootstrap_95ci_upper": upper,
                    "wilcoxon_statistic": statistic,
                    "wilcoxon_p_raw": p_value,
                }
            )
            rows[-1]["inference_label"] = family["inference_label"]
    for row in rows:
        row["negative_effect_favors_reference"] = True
    primary_effects = pd.DataFrame(rows)
    primary_effects.to_csv(
        args.output / "hnei_exploratory_paired_effects.csv",
        index=False,
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    display_models = [
        str(specification["id"])
        for specification in protocol["models"]
    ]
    colors = {
        "mstt_full_K5": "#2b6cb0",
        "mstt_single_scale_K5": "#805ad5",
        "local_linear_trend": "#dd6b20",
    }
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
    for model in display_models:
        group = summary[summary["model"].eq(model)].sort_values("cutoff")
        axes[0].errorbar(
            group["cutoff"],
            group["future_capacity_RMSE_SOH_mean"],
            yerr=group["future_capacity_RMSE_SOH_std"],
            marker="o",
            capsize=3,
            label=model,
            color=colors[model],
        )
        axes[1].errorbar(
            group["cutoff"],
            group["censor_aware_timing_score_cycles_mean"],
            yerr=group["censor_aware_timing_score_cycles_std"],
            marker="o",
            capsize=3,
            label=model,
            color=colors[model],
        )
    axes[0].set_title("HNEI future-capacity RMSE")
    axes[0].set_xlabel("Fixed cutoff cycle")
    axes[0].set_ylabel("RMSE (SOH fraction)")
    axes[1].set_title("HNEI censor-aware EOL timing error")
    axes[1].set_xlabel("Fixed cutoff cycle")
    axes[1].set_ylabel("Timing score (cycles)")
    axes[1].legend(fontsize=7, frameon=False)
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.savefig(
        args.output / "hnei_performance_by_cutoff.png",
        dpi=300,
    )
    plt.close(figure)
    audit = {
        "status": "PASS",
        "created_utc": utc_now(),
        "analysis_class": protocol["analysis_class"],
        "claim_limit": protocol["claim_limit"],
        "config_sha256": json_sha256(protocol),
        "physical_cells": records["cell_id"].nunique(),
        "primary_cutoff": cutoff,
        "hnei_exploratory_tests": len(rows),
        "confirmatory_multiplicity_adjustment_applied": False,
        "inference_label": family["inference_label"],
        "negative_effect_favors_reference": True,
    }
    write_json(args.output / "aggregate_audit.json", audit)
    print(f"[PASS] HNEI aggregation: primary tests={len(rows)}")


def command_pack(args: argparse.Namespace) -> None:
    required = [
        args.inventory / "inventory.json",
        args.development / "development_preflight.json",
        args.models / "frozen_model_manifest.json",
        args.external / "external_preflight.json",
        args.evaluation / "evaluation_audit.json",
        args.aggregate / "aggregate_audit.json",
        args.receipt,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing result gates: {missing}")
    if args.output_zip.exists():
        raise FileExistsError(
            f"Result ZIP already exists; choose a new path: {args.output_zip}"
        )
    sources = {
        "01_inventory": args.inventory,
        "02_development_gate": args.development,
        "03_model_audits": args.models,
        "04_hnei_prepared": args.external,
        "05_evaluation": args.evaluation,
        "06_aggregate": args.aggregate,
    }
    args.output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        args.output_zip,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        archive.write(args.receipt, "pre_evaluation_receipt.json")
        for prefix, directory in sources.items():
            for path in sorted(directory.rglob("*")):
                if not path.is_file():
                    continue
                if prefix == "02_development_gate" and path.parent.name == "curves":
                    continue
                if prefix == "03_model_audits" and path.suffix in {
                    ".pt",
                    ".joblib",
                }:
                    continue
                if prefix == "04_hnei_prepared" and path.parent.name == "curves":
                    continue
                archive.write(path, Path(prefix) / path.relative_to(directory))
    with zipfile.ZipFile(args.output_zip) as archive:
        bad_member = archive.testzip()
    if bad_member is not None:
        raise RuntimeError(f"Result ZIP CRC failure: {bad_member}")
    digest = sha256_file(args.output_zip)
    sha_path = Path(str(args.output_zip) + ".sha256")
    sha_path.write_text(
        f"{digest}  {args.output_zip.name}\n",
        encoding="utf-8",
    )
    print(f"[PASS] result package: {args.output_zip}")
    print(f"SHA256={digest}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HNEI normalized-SOH external transfer pipeline"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory = subparsers.add_parser("inventory")
    inventory.add_argument("--config", type=Path, required=True)
    inventory.add_argument("--archive", type=Path, required=True)
    inventory.add_argument("--output", type=Path, required=True)
    inventory.set_defaults(function=command_inventory)

    development = subparsers.add_parser("prepare-development")
    development.add_argument("--config", type=Path, required=True)
    development.add_argument("--source", type=Path, required=True)
    development.add_argument("--output", type=Path, required=True)
    development.set_defaults(function=command_prepare_development)

    train = subparsers.add_parser("train")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--development", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--device", default="auto")
    train.add_argument("--num-workers", type=int, default=0)
    train.add_argument("--quick-test", action="store_true")
    train.set_defaults(function=command_train)

    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--config", type=Path, required=True)
    freeze.add_argument("--project-root", type=Path, required=True)
    freeze.add_argument("--inventory", type=Path, required=True)
    freeze.add_argument("--development", type=Path, required=True)
    freeze.add_argument("--models", type=Path, required=True)
    freeze.add_argument("--evaluation", type=Path, required=True)
    freeze.add_argument("--output-zip", type=Path, required=True)
    freeze.set_defaults(function=command_freeze)

    register = subparsers.add_parser("register")
    register.add_argument("--config", type=Path, required=True)
    register.add_argument("--freeze-zip", type=Path, required=True)
    register.add_argument("--freeze-manifest", type=Path, required=True)
    register.add_argument("--evaluation", type=Path, required=True)
    register.add_argument("--commit", required=True)
    register.add_argument("--github-release", required=True)
    register.add_argument("--doi", required=True)
    register.add_argument("--output", type=Path, required=True)
    register.set_defaults(function=command_register)

    prepare_hnei = subparsers.add_parser("prepare-hnei")
    prepare_hnei.add_argument("--config", type=Path, required=True)
    prepare_hnei.add_argument("--archive", type=Path, required=True)
    prepare_hnei.add_argument("--inventory", type=Path, required=True)
    prepare_hnei.add_argument("--freeze-zip", type=Path, required=True)
    prepare_hnei.add_argument("--receipt", type=Path, required=True)
    prepare_hnei.add_argument("--output", type=Path, required=True)
    prepare_hnei.set_defaults(function=command_prepare_hnei)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--config", type=Path, required=True)
    evaluate.add_argument("--external", type=Path, required=True)
    evaluate.add_argument("--models", type=Path, required=True)
    evaluate.add_argument("--freeze-zip", type=Path, required=True)
    evaluate.add_argument("--receipt", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--device", default="auto")
    evaluate.set_defaults(function=command_evaluate)

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--config", type=Path, required=True)
    aggregate.add_argument("--evaluation", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    aggregate.set_defaults(function=command_aggregate)

    pack = subparsers.add_parser("pack")
    pack.add_argument("--inventory", type=Path, required=True)
    pack.add_argument("--development", type=Path, required=True)
    pack.add_argument("--models", type=Path, required=True)
    pack.add_argument("--external", type=Path, required=True)
    pack.add_argument("--evaluation", type=Path, required=True)
    pack.add_argument("--aggregate", type=Path, required=True)
    pack.add_argument("--receipt", type=Path, required=True)
    pack.add_argument("--output-zip", type=Path, required=True)
    pack.set_defaults(function=command_pack)
    return parser


def main() -> None:
    parser = build_parser()
    arguments = parser.parse_args()
    arguments.function(arguments)


if __name__ == "__main__":
    main()
