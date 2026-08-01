#!/usr/bin/env python3
"""Registered BIT v0.2.1 evaluator.

This implementation is written after immutable protocol registration and before
BIT capacity access. Every runtime command validates both the immutable
registration receipt and a separately frozen evaluator-source receipt.

The evaluator uses only the registered physical-cycle-70 task. Physical cycles
130 and 190 are structurally unavailable and are never evaluated.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile
from typing import Iterable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd


REGISTERED_PROTOCOL_COMMIT = "780a057d26cd4a0fc2dc50a80161d66c936155b5"
REGISTERED_VERSION = "v0.2.1-BIT-structural-feasibility-amendment"
REGISTERED_FREEZE_SHA256 = (
    "d66a7ff79b76c497f73e842b9accd6c18035f4169720e796366ea6f7a7f48aed"
)
REGISTERED_GITHUB_RELEASE = (
    "https://github.com/pengan400-dot/thesis/releases/tag/"
    "v0.2.1-BIT-structural-feasibility-amendment"
)
REGISTERED_OSF = "https://osf.io/gr6xj/"
REGISTERED_ZENODO_DOI = "10.5281/zenodo.21734477"
BIT_ARCHIVE_SHA256 = (
    "9701ff850b8b738f9b15b5e02d1ea6fe2e3630ef704336f19acc918f133b18d7"
)
PRIMARY_CUTOFF = 70
STRUCTURALLY_UNAVAILABLE_CUTOFFS = [130, 190]
EXPECTED_INCLUDED_CELLS = 72
EXPECTED_COHORT_COUNTS = {"arbitrary_use": 55, "fixed_profile": 17}
EXCLUDED_CELL_IDS = ["BIT_#2"]
NOMINAL_CAPACITY_AH = 2.4
MODEL_IDS = ["mstt_full_K5", "mstt_single_scale_K5", "local_linear_trend"]
NEURAL_MODEL_IDS = ["mstt_full_K5", "mstt_single_scale_K5"]
SEEDS = [42, 2024, 3407]
FORBIDDEN_FUTURE_INPUTS = [
    "post-cutoff actual current schedule",
    "post-cutoff charge-rate labels",
    "future sequence length",
    "true data termination position",
    "total lifetime",
    "future EOL",
    "future missingness pattern",
    "statistics computed from the complete future trajectory",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def normalize_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def validate_registration_receipt(path: Path) -> dict:
    receipt = read_json(path)
    expected = {
        "status": "PASS",
        "version": REGISTERED_VERSION,
        "commit": REGISTERED_PROTOCOL_COMMIT,
        "github_release": REGISTERED_GITHUB_RELEASE,
        "osf_registration": REGISTERED_OSF,
        "zenodo_specific_version_doi": REGISTERED_ZENODO_DOI,
        "freeze_zip_sha256": REGISTERED_FREEZE_SHA256,
        "primary_landmark": PRIMARY_CUTOFF,
        "structurally_unavailable_landmarks": STRUCTURALLY_UNAVAILABLE_CUTOFFS,
        "bit_capacity_opened_before_receipt": False,
        "bit_model_output_existed_before_receipt": False,
        "bit_evaluation_unlocked": True,
    }
    failures = {
        key: {"expected": value, "observed": receipt.get(key)}
        for key, value in expected.items()
        if receipt.get(key) != value
    }
    if failures:
        raise RuntimeError(f"Immutable registration receipt gate failed: {failures}")
    return receipt


def validate_source_receipt(
    receipt_path: Path,
    source_freeze_zip: Path,
    project_root: Path,
) -> dict:
    receipt = read_json(receipt_path)
    if receipt.get("status") != "PASS_BIT_V0_2_1_EVALUATOR_SOURCE_FREEZE":
        raise RuntimeError("Evaluator source-freeze receipt is not PASS")
    if receipt.get("registered_protocol_commit") != REGISTERED_PROTOCOL_COMMIT:
        raise RuntimeError("Evaluator receipt references the wrong protocol commit")
    if receipt.get("capacity_opened_before_source_freeze") is not False:
        raise RuntimeError("Evaluator source was not frozen before capacity access")
    if receipt.get("model_weights_loaded_before_source_freeze") is not False:
        raise RuntimeError("Model weights were loaded before evaluator source freeze")
    if receipt.get("model_output_generated_before_source_freeze") is not False:
        raise RuntimeError("Model output existed before evaluator source freeze")
    require_file(source_freeze_zip)
    if sha256_file(source_freeze_zip) != receipt.get("source_freeze_zip_sha256"):
        raise RuntimeError("Evaluator source-freeze ZIP hash mismatch")
    current_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        text=True,
    ).strip()
    if current_commit != receipt.get("evaluator_source_commit"):
        raise RuntimeError(
            "Current source commit differs from the frozen evaluator source commit"
        )
    for relative, expected_hash in receipt.get("source_files", {}).items():
        path = project_root / relative
        require_file(path)
        if sha256_file(path) != expected_hash:
            raise RuntimeError(f"Evaluator source changed after freeze: {relative}")
    return receipt


def validate_freeze_manifest(path: Path) -> dict:
    manifest = read_json(path)
    expected = {
        "status": "PASS",
        "version": REGISTERED_VERSION,
        "frozen_commit": REGISTERED_PROTOCOL_COMMIT,
        "primary_landmark": PRIMARY_CUTOFF,
        "structurally_unavailable_landmarks": STRUCTURALLY_UNAVAILABLE_CUTOFFS,
        "excluded_cell_ids": EXCLUDED_CELL_IDS,
        "bit_capacity_opened_before_freeze": False,
        "bit_model_output_generated_before_freeze": False,
    }
    failures = {
        key: {"expected": value, "observed": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    counts = manifest.get("evaluation_inclusion_counts", {})
    if counts != {
        "total": 72,
        "arbitrary_use": 55,
        "fixed_profile": 17,
    }:
        failures["evaluation_inclusion_counts"] = counts
    if failures:
        raise RuntimeError(f"BIT freeze manifest gate failed: {failures}")
    return manifest


def validate_protocols(
    bit_config_path: Path,
    development_config_path: Path,
) -> tuple[dict, dict]:
    bit = read_json(bit_config_path)
    development = read_json(development_config_path)
    if bit.get("version_labels", {}).get("bit_freeze") != REGISTERED_VERSION:
        raise RuntimeError("Wrong BIT protocol version")
    task = bit.get("task", {})
    if task.get("cutoffs") != [70] or task.get("primary_cutoff") != 70:
        raise RuntimeError("Only physical-cycle 70 is allowed")
    if task.get("structurally_unavailable_cutoffs") != [130, 190]:
        raise RuntimeError("Structural-unavailability contract changed")
    bit_section = bit.get("bit", {})
    if bit_section.get("evaluation_included_cells") != 72:
        raise RuntimeError("Frozen BIT cell count changed")
    if bit_section.get("excluded_cell_ids") != ["BIT_#2"]:
        raise RuntimeError("Frozen BIT exclusion changed")
    unchanged_sections = (
        "models",
        "training",
        "features",
        "soh_definition",
        "ah_to_soh_conversion",
    )
    changed = [name for name in unchanged_sections if bit.get(name) != development.get(name)]
    if changed:
        raise RuntimeError(f"Registered amendment changed frozen model sections: {changed}")
    amendment = bit.get("amendment", {})
    for key in (
        "model_weights_changed",
        "scaler_changed",
        "soh_definition_changed",
        "eol_threshold_changed",
        "statistics_changed",
        "H1_then_H2_order_changed",
        "cohort_roles_changed",
    ):
        if amendment.get(key) is not False:
            raise RuntimeError(f"Amendment invariant failed: {key}")
    return bit, development


def runtime_gate(args: argparse.Namespace) -> tuple[dict, dict, dict, dict]:
    project_root = args.project_root.resolve()
    registration = validate_registration_receipt(args.registration_receipt)
    source = validate_source_receipt(
        args.source_receipt,
        args.source_freeze_zip,
        project_root,
    )
    freeze_manifest = validate_freeze_manifest(args.freeze_manifest)
    bit, development = validate_protocols(
        args.bit_config,
        args.development_config,
    )
    return registration, source, freeze_manifest, {
        "bit": bit,
        "development": development,
    }


def run_parser(args: argparse.Namespace) -> None:
    runtime_gate(args)
    require_file(args.archive)
    if sha256_file(args.archive) != BIT_ARCHIVE_SHA256:
        raise RuntimeError("BIT V3 archive SHA-256 mismatch")
    if args.output.exists():
        raise FileExistsError(
            f"Refusing to overwrite registered raw-capacity output: {args.output}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(sys.executable),
        str(args.parser),
        "--archive",
        str(args.archive),
        "--pairing-manifest",
        str(args.pairing_manifest),
        "--mapping-manifest",
        str(args.mapping_manifest),
        "--registration-receipt",
        str(args.registration_receipt),
        "--output",
        str(args.output),
    ]
    subprocess.run(command, check=True)
    print(f"[PASS] registered BIT raw-capacity parse complete: {args.output}")


def first_valid_bol_reference(
    cycles: Sequence[int],
    capacities_ah: Sequence[float],
    *,
    nominal_capacity_ah: float = NOMINAL_CAPACITY_AH,
    count: int = 5,
) -> tuple[float, list[int], list[float]]:
    lower = 0.5 * nominal_capacity_ah
    upper = 1.5 * nominal_capacity_ah
    selected_cycles: list[int] = []
    selected_values: list[float] = []
    for cycle, value in zip(cycles, capacities_ah, strict=True):
        numeric = float(value)
        if math.isfinite(numeric) and lower <= numeric <= upper:
            selected_cycles.append(int(cycle))
            selected_values.append(numeric)
            if len(selected_values) == count:
                break
    if len(selected_values) != count:
        raise ValueError(
            f"BOL gate requires {count} valid Ah observations; got {len(selected_values)}"
        )
    reference = float(np.median(selected_values))
    if not 0.8 * nominal_capacity_ah <= reference <= 1.2 * nominal_capacity_ah:
        raise ValueError(
            f"BOL median {reference:g} Ah outside registered quality range"
        )
    return reference, selected_cycles, selected_values


def prepare_cell_frame(
    raw_group: pd.DataFrame,
    *,
    smoothing_window: int = 7,
    nominal_capacity_ah: float = NOMINAL_CAPACITY_AH,
) -> tuple[pd.DataFrame, dict[str, object]]:
    group = raw_group.copy()
    group["physical_cycle"] = pd.to_numeric(
        group["physical_cycle"], errors="raise"
    ).astype(int)
    if group["physical_cycle"].duplicated().any():
        raise ValueError("Duplicate physical cycle in raw parser output")
    group = group.sort_values("physical_cycle").reset_index(drop=True)
    numeric = pd.to_numeric(
        group["raw_discharge_capacity_Ah"], errors="coerce"
    )
    parser_quality = group["capacity_quality_pass"].map(normalize_bool)
    trajectory_quality = (
        parser_quality
        & numeric.notna()
        & numeric.gt(0.0)
        & numeric.le(1.5 * nominal_capacity_ah)
    )
    observed_ah = numeric.where(trajectory_quality)
    reference, bol_cycles, bol_values = first_valid_bol_reference(
        group["physical_cycle"].tolist(),
        observed_ah.tolist(),
        nominal_capacity_ah=nominal_capacity_ah,
        count=5,
    )
    max_cycle = int(group["physical_cycle"].max())
    valid_cycles = group.loc[trajectory_quality, "physical_cycle"]
    if valid_cycles.empty:
        raise ValueError("No valid raw-capacity observation")
    first_observed_cycle = int(valid_cycles.min())
    grid = pd.DataFrame(
        {"cycle": np.arange(first_observed_cycle, max_cycle + 1, dtype=int)}
    )
    source = pd.DataFrame(
        {
            "cycle": group["physical_cycle"],
            "raw_capacity_Ah": observed_ah,
        }
    )
    frame = grid.merge(source, on="cycle", how="left")
    frame["measurement_observed"] = frame["raw_capacity_Ah"].notna().astype(int)
    frame["raw_capacity"] = frame["raw_capacity_Ah"] / reference
    causal_filled = frame["raw_capacity"].ffill()
    if causal_filled.iloc[0] != causal_filled.iloc[0]:
        raise RuntimeError("Internal causal-grid construction failed")
    frame["capacity"] = (
        causal_filled.rolling(max(1, int(smoothing_window)), min_periods=1).mean()
    )
    cell_id = str(group["cell_id"].iloc[0])
    cohort = str(group["cohort"].iloc[0])
    frame["battery_id"] = cell_id
    frame["cohort"] = cohort
    frame["dataset"] = "BIT_V3"
    frame["bol_reference_capacity_Ah"] = reference
    frame["initial_capacity_Ah"] = 1.0
    observed = frame[frame["measurement_observed"].eq(1)]
    audit = {
        "cell_id": cell_id,
        "cohort": cohort,
        "status": "PASS",
        "first_observed_physical_cycle": first_observed_cycle,
        "physical_cycles_in_grid": len(frame),
        "observed_capacity_cycles": len(observed),
        "bol_reference_capacity_Ah": reference,
        "bol_first_five_cycles": bol_cycles,
        "bol_first_five_values_Ah": bol_values,
        "causal_missing_cycle_fill": "last_observation_carried_forward",
        "backward_fill_used": False,
        "future_interpolation_used": False,
        "causal_smoothing_window": int(smoothing_window),
        "last_observed_cycle": int(observed["cycle"].max()),
    }
    return frame, audit


def eol_interval(frame: pd.DataFrame, threshold: float = 0.8) -> tuple[int, int] | None:
    measured = frame[frame["measurement_observed"].eq(1)].dropna(
        subset=["raw_capacity"]
    ).sort_values("cycle")
    hits = measured[measured["raw_capacity"] <= threshold]
    if hits.empty:
        return None
    upper = int(hits.iloc[0]["cycle"])
    earlier = measured[measured["cycle"] < upper]
    lower = 1 if earlier.empty else int(earlier.iloc[-1]["cycle"]) + 1
    return lower, upper


def prepare_soh(args: argparse.Namespace) -> None:
    _, source, freeze_manifest, protocols = runtime_gate(args)
    require_file(args.raw_capacity_csv)
    if args.output.exists() and any(args.output.rglob("*")):
        raise FileExistsError(f"Prepared output is not empty: {args.output}")
    raw = pd.read_csv(args.raw_capacity_csv)
    required = {
        "cell_id",
        "cell_number",
        "cohort",
        "physical_cycle",
        "raw_discharge_capacity_Ah",
        "capacity_quality_pass",
    }
    if not required.issubset(raw.columns):
        raise ValueError(f"Raw parser output lacks columns: {sorted(required - set(raw.columns))}")
    cells = sorted(raw["cell_id"].astype(str).unique())
    if len(cells) != EXPECTED_INCLUDED_CELLS:
        raise RuntimeError(f"Expected 72 frozen cells, got {len(cells)}")
    cohort_counts = (
        raw[["cell_id", "cohort"]].drop_duplicates()["cohort"].value_counts().to_dict()
    )
    if cohort_counts != EXPECTED_COHORT_COUNTS:
        raise RuntimeError(f"Frozen cohort counts changed: {cohort_counts}")
    bit = protocols["bit"]
    smoothing_window = int(bit["task"]["causal_smoothing_window_cycles"])
    threshold = float(bit["task"]["eol_threshold_SOH"])
    horizon = int(bit["task"]["forecast_horizon_cycles"])
    minimum_future = int(bit["task"]["minimum_future_observations_for_RMSE"])
    curves_dir = args.output / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)
    audits: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    curves: dict[str, pd.DataFrame] = {}
    for cell_id, group in raw.groupby("cell_id", sort=True):
        try:
            frame, audit = prepare_cell_frame(
                group,
                smoothing_window=smoothing_window,
                nominal_capacity_ah=NOMINAL_CAPACITY_AH,
            )
            interval = eol_interval(frame, threshold)
            observed = frame[frame["measurement_observed"].eq(1)]
            last_observed = int(observed["cycle"].max())
            support_end = interval[1] if interval is not None else last_observed
            future_points = int(
                (
                    observed["cycle"].gt(PRIMARY_CUTOFF)
                    & observed["cycle"].le(
                        min(PRIMARY_CUTOFF + horizon, support_end)
                    )
                ).sum()
            )
            if interval is not None and interval[1] <= PRIMARY_CUTOFF:
                risk_status = "EXCLUDED_EOL_AT_OR_BEFORE_CYCLE_70"
            elif last_observed <= PRIMARY_CUTOFF:
                risk_status = "EXCLUDED_RIGHT_CENSORED_AT_OR_BEFORE_CYCLE_70"
            else:
                risk_status = "AT_RISK_AT_CYCLE_70"
            audit.update(
                {
                    "true_eol_interval_lower": None if interval is None else interval[0],
                    "true_eol_interval_upper": None if interval is None else interval[1],
                    "truth_event_type": "right_censored" if interval is None else "interval_observed",
                    "risk_status_cycle_70": risk_status,
                    "future_observed_points_on_registered_support": future_points,
                    "rmse_evaluable_cycle_70": bool(
                        risk_status == "AT_RISK_AT_CYCLE_70" and future_points >= minimum_future
                    ),
                    "timing_evaluable_cycle_70": bool(risk_status == "AT_RISK_AT_CYCLE_70"),
                }
            )
            frame.to_csv(curves_dir / f"{cell_id}.csv", index=False)
            curves[str(cell_id)] = frame
            audits.append(audit)
            print(f"[PREPARED] {cell_id}")
        except Exception as exc:  # noqa: BLE001
            failures.append(
                {
                    "cell_id": str(cell_id),
                    "status": "DATA_QUALITY_EXCLUSION",
                    "reason": str(exc),
                }
            )
            print(f"[EXCLUDED] {cell_id}: {exc}")
    audit_frame = pd.DataFrame(audits)
    failure_frame = pd.DataFrame(failures)
    audit_frame.to_csv(args.output / "preparation_cell_audit.csv", index=False)
    failure_frame.to_csv(args.output / "data_quality_exclusions.csv", index=False)
    prepared_cohorts = audit_frame["cohort"].value_counts().to_dict() if not audit_frame.empty else {}
    gate = {
        "status": "PASS",
        "created_utc": utc_now(),
        "registered_protocol_commit": REGISTERED_PROTOCOL_COMMIT,
        "evaluator_source_commit": source["evaluator_source_commit"],
        "raw_capacity_csv_sha256": sha256_file(args.raw_capacity_csv),
        "prepared_curves_sha256": curves_digest(curves),
        "frozen_cells_in_raw_parser_output": len(cells),
        "prepared_cells": len(curves),
        "data_quality_exclusions": len(failures),
        "prepared_cohort_counts": prepared_cohorts,
        "risk_flow_cycle_70": audit_frame["risk_status_cycle_70"].value_counts().to_dict()
        if not audit_frame.empty
        else {},
        "rmse_evaluable_cells_cycle_70": int(audit_frame["rmse_evaluable_cycle_70"].sum())
        if not audit_frame.empty
        else 0,
        "timing_evaluable_cells_cycle_70": int(audit_frame["timing_evaluable_cycle_70"].sum())
        if not audit_frame.empty
        else 0,
        "primary_cutoff": PRIMARY_CUTOFF,
        "structurally_unavailable_cutoffs": STRUCTURALLY_UNAVAILABLE_CUTOFFS,
        "normalization": "raw_Ah / median(first_5_valid_raw_Ah)",
        "smoothing": "7-cycle causal trailing mean after causal LOCF on physical-cycle grid",
        "future_interpolation_used": False,
        "backward_fill_used": False,
        "post_cutoff_operating_conditions_loaded": False,
        "freeze_manifest_sha256": sha256_file(args.freeze_manifest),
        "freeze_manifest_version": freeze_manifest["version"],
        "failures": [],
    }
    write_json(args.output / "preparation_audit.json", gate)
    print(
        f"[PASS] BIT SOH preparation: prepared={len(curves)}, "
        f"quality_excluded={len(failures)}"
    )


def curves_digest(curves: Mapping[str, pd.DataFrame]) -> str:
    digest = hashlib.sha256()
    columns = (
        "cycle",
        "raw_capacity_Ah",
        "raw_capacity",
        "capacity",
        "measurement_observed",
        "battery_id",
        "cohort",
        "bol_reference_capacity_Ah",
    )
    for cell_id in sorted(curves):
        digest.update(cell_id.encode("utf-8"))
        frame = curves[cell_id]
        for column in columns:
            digest.update(column.encode("utf-8"))
            values = frame[column].to_numpy()
            if values.dtype.kind in "fiu":
                digest.update(np.asarray(values, dtype=np.float64).tobytes())
            else:
                digest.update("\n".join(map(str, values)).encode("utf-8"))
    return digest.hexdigest()


def load_prepared_curves(path: Path) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict]:
    gate = read_json(path / "preparation_audit.json")
    if gate.get("status") != "PASS" or gate.get("primary_cutoff") != 70:
        raise RuntimeError("Prepared BIT gate failed")
    audit = pd.read_csv(path / "preparation_cell_audit.csv")
    curves: dict[str, pd.DataFrame] = {}
    for csv_path in sorted((path / "curves").glob("BIT_*.csv")):
        frame = pd.read_csv(csv_path).sort_values("cycle").reset_index(drop=True)
        required = {
            "cycle",
            "raw_capacity",
            "capacity",
            "measurement_observed",
            "battery_id",
            "cohort",
        }
        if not required.issubset(frame.columns):
            raise ValueError(f"Prepared curve columns changed: {csv_path}")
        cell_id = str(frame["battery_id"].iloc[0])
        curves[cell_id] = frame
    if curves_digest(curves) != gate.get("prepared_curves_sha256"):
        raise RuntimeError("Prepared BIT curves changed after audit")
    return curves, audit, gate


def metric_record(
    frame: pd.DataFrame,
    forecast: pd.DataFrame,
    *,
    cutoff: int,
    predicted_eol: int | None,
    threshold: float,
    horizon: int,
    minimum_future: int,
) -> dict[str, object]:
    interval = eol_interval(frame, threshold)
    observed = frame[frame["measurement_observed"].eq(1)].dropna(
        subset=["raw_capacity"]
    )
    right_censor_cycle = int(observed["cycle"].max())
    support_end = interval[1] if interval is not None else right_censor_cycle
    measured = frame[
        frame["measurement_observed"].eq(1)
        & frame["cycle"].gt(cutoff)
        & frame["cycle"].le(min(cutoff + horizon, support_end))
    ][["cycle", "raw_capacity"]]
    aligned = measured.merge(forecast, on="cycle", how="inner")
    rmse_evaluable = len(aligned) >= minimum_future
    if rmse_evaluable:
        residual = aligned["predicted_capacity"].to_numpy(float) - aligned[
            "raw_capacity"
        ].to_numpy(float)
        mae = float(np.mean(np.abs(residual)))
        rmse = float(np.sqrt(np.mean(residual**2)))
        max_error = float(np.max(np.abs(residual)))
    else:
        mae = math.nan
        rmse = math.nan
        max_error = math.nan
    hit = predicted_eol is not None and predicted_eol <= cutoff + horizon
    if interval is not None:
        if hit:
            if predicted_eol < interval[0]:
                interval_distance = float(interval[0] - predicted_eol)
            elif predicted_eol > interval[1]:
                interval_distance = float(predicted_eol - interval[1])
            else:
                interval_distance = 0.0
            censor_score = interval_distance
        else:
            interval_distance = math.nan
            censor_score = float(max(0, cutoff + horizon + 1 - interval[1]))
    else:
        interval_distance = math.nan
        if not hit or predicted_eol > right_censor_cycle:
            censor_score = 0.0
        else:
            censor_score = float(right_censor_cycle + 1 - predicted_eol)
    return {
        "truth_event_type": "interval_observed" if interval is not None else "right_censored",
        "truth_event_observed": int(interval is not None),
        "true_eol_interval_lower": math.nan if interval is None else int(interval[0]),
        "true_eol_interval_upper": math.nan if interval is None else int(interval[1]),
        "right_censor_cycle": math.nan if interval is not None else right_censor_cycle,
        "predicted_eol_cycle": math.nan if predicted_eol is None else int(predicted_eol),
        "threshold_hit_within_horizon": int(hit),
        "hit_conditional_interval_distance_cycles": interval_distance,
        "censor_aware_timing_score_cycles": censor_score,
        "future_observed_points": len(aligned),
        "rmse_evaluable": bool(rmse_evaluable),
        "future_capacity_MAE_SOH": mae,
        "future_capacity_RMSE_SOH": rmse,
        "max_abs_capacity_error_SOH": max_error,
    }


def evaluate(args: argparse.Namespace) -> None:
    _, source, _, protocols = runtime_gate(args)
    curves, cell_audit, prepare_gate = load_prepared_curves(args.prepared)
    if args.output.exists() and any(args.output.rglob("*")):
        raise FileExistsError(f"Evaluation output is not empty: {args.output}")
    bit = protocols["bit"]
    development = protocols["development"]
    sys.path.insert(0, str((args.project_root / "src").resolve()))
    from mstt_soh.pipeline import (  # noqa: PLC0415
        choose_device,
        first_predicted_eol,
        load_frozen_models,
        merge_seed_forecasts,
        recursive_forecast,
    )

    device = choose_device(args.device)
    models = load_frozen_models(development, args.models, device)
    if sorted(models) != sorted(NEURAL_MODEL_IDS):
        raise RuntimeError(f"Frozen neural model IDs changed: {sorted(models)}")
    threshold = float(bit["task"]["eol_threshold_SOH"])
    window = int(bit["task"]["input_window_cycles"])
    horizon = int(bit["task"]["forecast_horizon_cycles"])
    minimum_history = int(bit["task"]["minimum_pre_cutoff_physical_cycles"])
    minimum_future = int(bit["task"]["minimum_future_observations_for_RMSE"])
    clip_bounds = tuple(map(float, bit["task"]["prediction_clip_SOH"]))
    local_spec = next(item for item in bit["models"] if item["id"] == "local_linear_trend")
    local_lookback = int(local_spec["lookback_physical_cycles"])
    local_slope = tuple(map(float, local_spec["slope_clip_SOH_per_cycle"]))
    trajectories = args.output / "trajectories"
    trajectories.mkdir(parents=True, exist_ok=True)
    ensemble_rows: list[dict[str, object]] = []
    seed_rows: list[dict[str, object]] = []
    runtime_exclusions: list[dict[str, object]] = []
    risk_ids = set(
        cell_audit.loc[
            cell_audit["risk_status_cycle_70"].eq("AT_RISK_AT_CYCLE_70"),
            "cell_id",
        ].astype(str)
    )
    for cell_id in sorted(curves):
        if cell_id not in risk_ids:
            continue
        frame = curves[cell_id]
        cohort = str(frame["cohort"].iloc[0])
        try:
            for model_id, seed_models in sorted(models.items()):
                forecasts: dict[int, pd.DataFrame] = {}
                for seed, (model, scaler) in sorted(seed_models.items()):
                    forecast, seed_eol = recursive_forecast(
                        model_id,
                        frame,
                        PRIMARY_CUTOFF,
                        horizon,
                        threshold,
                        window,
                        clip_bounds,
                        model=model,
                        scaler=scaler,
                        device=device,
                        minimum_history=minimum_history,
                    )
                    forecasts[int(seed)] = forecast
                    seed_rows.append(
                        {
                            "cell_id": cell_id,
                            "cohort": cohort,
                            "cutoff": PRIMARY_CUTOFF,
                            "model": model_id,
                            "seed": int(seed),
                            **metric_record(
                                frame,
                                forecast,
                                cutoff=PRIMARY_CUTOFF,
                                predicted_eol=seed_eol,
                                threshold=threshold,
                                horizon=horizon,
                                minimum_future=minimum_future,
                            ),
                        }
                    )
                ensemble = merge_seed_forecasts(forecasts)
                ensemble_eol = first_predicted_eol(ensemble, threshold)
                ensemble_rows.append(
                    {
                        "cell_id": cell_id,
                        "cohort": cohort,
                        "cutoff": PRIMARY_CUTOFF,
                        "model": model_id,
                        "aggregation": "pointwise_mean_across_frozen_seeds",
                        **metric_record(
                            frame,
                            ensemble,
                            cutoff=PRIMARY_CUTOFF,
                            predicted_eol=ensemble_eol,
                            threshold=threshold,
                            horizon=horizon,
                            minimum_future=minimum_future,
                        ),
                    }
                )
                output = ensemble.merge(
                    frame[["cycle", "raw_capacity"]].rename(
                        columns={"raw_capacity": "observed_raw_soh"}
                    ),
                    on="cycle",
                    how="left",
                )
                output.insert(0, "model", model_id)
                output.insert(0, "cutoff", PRIMARY_CUTOFF)
                output.insert(0, "cohort", cohort)
                output.insert(0, "cell_id", cell_id)
                output.to_csv(
                    trajectories / f"{cell_id}_c70_{model_id}.csv",
                    index=False,
                )
            baseline, baseline_eol = recursive_forecast(
                "local_linear_trend",
                frame,
                PRIMARY_CUTOFF,
                horizon,
                threshold,
                window,
                clip_bounds,
                minimum_history=minimum_history,
                local_linear_lookback=local_lookback,
                local_linear_slope_clip=local_slope,
            )
            ensemble_rows.append(
                {
                    "cell_id": cell_id,
                    "cohort": cohort,
                    "cutoff": PRIMARY_CUTOFF,
                    "model": "local_linear_trend",
                    "aggregation": "deterministic",
                    **metric_record(
                        frame,
                        baseline,
                        cutoff=PRIMARY_CUTOFF,
                        predicted_eol=baseline_eol,
                        threshold=threshold,
                        horizon=horizon,
                        minimum_future=minimum_future,
                    ),
                }
            )
            baseline_output = baseline.merge(
                frame[["cycle", "raw_capacity"]].rename(
                    columns={"raw_capacity": "observed_raw_soh"}
                ),
                on="cycle",
                how="left",
            )
            baseline_output.insert(0, "model", "local_linear_trend")
            baseline_output.insert(0, "cutoff", PRIMARY_CUTOFF)
            baseline_output.insert(0, "cohort", cohort)
            baseline_output.insert(0, "cell_id", cell_id)
            baseline_output.to_csv(
                trajectories / f"{cell_id}_c70_local_linear_trend.csv",
                index=False,
            )
            print(f"[EVALUATED] {cell_id}")
        except Exception as exc:  # noqa: BLE001
            runtime_exclusions.append(
                {"cell_id": cell_id, "cohort": cohort, "reason": str(exc)}
            )
            print(f"[RUNTIME-EXCLUDED] {cell_id}: {exc}")
    ensemble = pd.DataFrame(ensemble_rows)
    seed = pd.DataFrame(seed_rows)
    excluded = pd.DataFrame(runtime_exclusions)
    ensemble.to_csv(args.output / "ensemble_cell_records.csv", index=False)
    seed.to_csv(args.output / "seed_level_records.csv", index=False)
    excluded.to_csv(args.output / "runtime_exclusions.csv", index=False)
    if runtime_exclusions:
        raise RuntimeError(
            "Runtime evaluation exclusions occurred after preparation; "
            "do not silently continue"
        )
    if ensemble.empty:
        raise RuntimeError("No BIT evaluation records generated")
    if set(ensemble["model"]) != set(MODEL_IDS):
        raise RuntimeError("Frozen model set changed")
    if ensemble.duplicated(["cell_id", "cutoff", "model"]).any():
        raise RuntimeError("Duplicate cell-level BIT records")
    expected_records = len(risk_ids) * len(MODEL_IDS)
    if len(ensemble) != expected_records:
        raise RuntimeError(
            f"Expected {expected_records} cell-model records, got {len(ensemble)}"
        )
    audit = {
        "status": "PASS",
        "created_utc": utc_now(),
        "registered_protocol_commit": REGISTERED_PROTOCOL_COMMIT,
        "evaluator_source_commit": source["evaluator_source_commit"],
        "primary_cutoff": PRIMARY_CUTOFF,
        "structurally_unavailable_cutoffs": STRUCTURALLY_UNAVAILABLE_CUTOFFS,
        "models": MODEL_IDS,
        "seeds": SEEDS,
        "risk_cells": len(risk_ids),
        "risk_cohort_counts": ensemble[["cell_id", "cohort"]].drop_duplicates()[
            "cohort"
        ].value_counts().to_dict(),
        "ensemble_records": len(ensemble),
        "seed_records": len(seed),
        "rmse_evaluable_records": int(ensemble["rmse_evaluable"].sum()),
        "prepared_curves_sha256": prepare_gate["prepared_curves_sha256"],
        "frozen_model_manifest_sha256": sha256_file(
            args.models / "frozen_model_manifest.json"
        ),
        "test_selection_or_retraining": False,
        "external_scaler_update": False,
        "post_cutoff_actual_current_schedule_loaded": False,
        "post_cutoff_charge_rate_labels_loaded": False,
        "future_sequence_length_loaded": False,
        "true_termination_position_loaded": False,
        "future_eol_loaded_as_input": False,
        "future_missingness_pattern_loaded_as_input": False,
        "model_outputs_generated": True,
        "failures": [],
    }
    write_json(args.output / "evaluation_audit.json", audit)
    print(
        f"[PASS] BIT evaluation: risk_cells={len(risk_ids)}, "
        f"records={len(ensemble)}"
    )


def complete_pair(
    records: pd.DataFrame,
    cohort: str,
    comparator: str,
) -> tuple[np.ndarray, list[str], list[str]]:
    subset = records[
        records["cohort"].eq(cohort)
        & records["rmse_evaluable"].map(normalize_bool)
    ]
    pivot = subset.pivot(index="cell_id", columns="model", values="future_capacity_RMSE_SOH")
    eligible = pivot[["mstt_full_K5", comparator]].dropna()
    differences = (
        eligible[comparator].to_numpy(float)
        - eligible["mstt_full_K5"].to_numpy(float)
    )
    all_cells = sorted(
        records.loc[records["cohort"].eq(cohort), "cell_id"].astype(str).unique()
    )
    paired_cells = sorted(eligible.index.astype(str))
    missing = sorted(set(all_cells) - set(paired_cells))
    return differences, paired_cells, missing


def descriptive_effect(
    records: pd.DataFrame,
    cohort: str,
    comparator: str,
) -> dict[str, object]:
    differences, cells, missing = complete_pair(records, cohort, comparator)
    if not len(differences):
        return {
            "cohort": cohort,
            "comparator": comparator,
            "n_pairs": 0,
            "mean_comparator_minus_full_RMSE": None,
            "median_comparator_minus_full_RMSE": None,
            "paired_cells": [],
            "missing_pair_cells": missing,
            "inference": "descriptive_only",
        }
    return {
        "cohort": cohort,
        "comparator": comparator,
        "n_pairs": len(differences),
        "mean_comparator_minus_full_RMSE": float(differences.mean()),
        "median_comparator_minus_full_RMSE": float(np.median(differences)),
        "paired_cells": cells,
        "missing_pair_cells": missing,
        "inference": "descriptive_only",
    }


def aggregate(args: argparse.Namespace) -> None:
    _, source, _, protocols = runtime_gate(args)
    evaluation_audit = read_json(args.evaluation / "evaluation_audit.json")
    if evaluation_audit.get("status") != "PASS":
        raise RuntimeError("BIT evaluation audit is not PASS")
    if args.output.exists() and any(args.output.rglob("*")):
        raise FileExistsError(f"Aggregate output is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    records = pd.read_csv(args.evaluation / "ensemble_cell_records.csv")
    required_models = set(MODEL_IDS)
    if set(records["model"]) != required_models:
        raise RuntimeError("Unexpected model set in BIT records")
    summary_rows: list[dict[str, object]] = []
    for (cohort, model), group in records.groupby(["cohort", "model"], sort=True):
        rmse = pd.to_numeric(group["future_capacity_RMSE_SOH"], errors="coerce").dropna()
        timing = pd.to_numeric(
            group["censor_aware_timing_score_cycles"], errors="coerce"
        ).dropna()
        hits = pd.to_numeric(group["threshold_hit_within_horizon"], errors="coerce")
        hit_distance = pd.to_numeric(
            group["hit_conditional_interval_distance_cycles"], errors="coerce"
        ).dropna()
        summary_rows.append(
            {
                "cohort": cohort,
                "model": model,
                "risk_cells": group["cell_id"].nunique(),
                "rmse_evaluable_cells": len(rmse),
                "future_capacity_RMSE_SOH_mean": float(rmse.mean()) if len(rmse) else math.nan,
                "future_capacity_RMSE_SOH_std": float(rmse.std(ddof=1)) if len(rmse) > 1 else math.nan,
                "future_capacity_RMSE_SOH_median": float(rmse.median()) if len(rmse) else math.nan,
                "threshold_hit_rate": float(hits.mean()),
                "censor_aware_timing_score_cycles_mean": float(timing.mean()),
                "censor_aware_timing_score_cycles_median": float(timing.median()),
                "hit_conditional_interval_distance_cycles_mean": float(hit_distance.mean())
                if len(hit_distance)
                else math.nan,
                "hit_conditional_cells": len(hit_distance),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output / "cohort_model_summary.csv", index=False)
    bit = protocols["bit"]
    statistics = bit["statistics"]
    h1, h1_cells, h1_missing = complete_pair(
        records, "arbitrary_use", "mstt_single_scale_K5"
    )
    h2, h2_cells, h2_missing = complete_pair(
        records, "arbitrary_use", "local_linear_trend"
    )
    if not len(h1) or not len(h2):
        raise RuntimeError("H1 and H2 require non-empty complete-pair sets")
    sys.path.insert(0, str((args.project_root / "src").resolve()))
    from mstt_soh.statistics import fixed_sequence_gatekeeping  # noqa: PLC0415

    fixed = fixed_sequence_gatekeeping(
        h1,
        h2,
        alpha=float(statistics["alpha_two_sided"]),
        bootstrap_repetitions=int(statistics["bootstrap_repetitions"]),
        bootstrap_seed=int(statistics["bootstrap_seed"]),
        permutation_repetitions=int(statistics["permutation_repetitions_if_not_exact"]),
        permutation_seed=int(statistics["permutation_seed"]),
        exact_max_nonzero_pairs=int(statistics["exact_permutation_max_nonzero_pairs"]),
    )
    fixed["primary_cohort"] = "arbitrary_use"
    fixed["primary_cutoff"] = 70
    fixed["H1_paired_cells"] = h1_cells
    fixed["H2_paired_cells"] = h2_cells
    fixed["H1_missing_pair_cells"] = h1_missing
    fixed["H2_missing_pair_cells"] = h2_missing
    fixed["ordinary_pooled_p_value_computed"] = False
    write_json(args.output / "confirmatory_fixed_sequence.json", fixed)
    secondary = {
        "status": "DESCRIPTIVE_ONLY",
        "cohort": "fixed_profile",
        "effects": [
            descriptive_effect(records, "fixed_profile", "mstt_single_scale_K5"),
            descriptive_effect(records, "fixed_profile", "local_linear_trend"),
        ],
        "ordinary_pooled_p_value_computed": False,
    }
    write_json(args.output / "fixed_profile_descriptive_effects.json", secondary)
    missing_rows = [
        *(
            {"hypothesis": "H1", "cell_id": cell, "reason": "incomplete_RMSE_pair"}
            for cell in h1_missing
        ),
        *(
            {"hypothesis": "H2", "cell_id": cell, "reason": "incomplete_RMSE_pair"}
            for cell in h2_missing
        ),
    ]
    pd.DataFrame(missing_rows).to_csv(args.output / "missing_pairs.csv", index=False)
    flow = pd.read_csv(args.prepared / "preparation_cell_audit.csv")
    flow_counts = [
        {"stage": "frozen_included_cells", "count": EXPECTED_INCLUDED_CELLS},
        {"stage": "data_quality_pass", "count": len(flow)},
        {
            "stage": "EOL_at_or_before_cycle_70",
            "count": int(flow["risk_status_cycle_70"].eq("EXCLUDED_EOL_AT_OR_BEFORE_CYCLE_70").sum()),
        },
        {
            "stage": "right_censored_at_or_before_cycle_70",
            "count": int(
                flow["risk_status_cycle_70"].eq(
                    "EXCLUDED_RIGHT_CENSORED_AT_OR_BEFORE_CYCLE_70"
                ).sum()
            ),
        },
        {
            "stage": "at_risk_cycle_70",
            "count": int(flow["risk_status_cycle_70"].eq("AT_RISK_AT_CYCLE_70").sum()),
        },
        {
            "stage": "RMSE_analysis_set",
            "count": int(flow["rmse_evaluable_cycle_70"].map(normalize_bool).sum()),
        },
        {
            "stage": "timing_analysis_set",
            "count": int(flow["timing_evaluable_cycle_70"].map(normalize_bool).sum()),
        },
    ]
    pd.DataFrame(flow_counts).to_csv(args.output / "analysis_flow_counts.csv", index=False)
    no_pooling = {
        "status": "PASS",
        "ordinary_pooled_p_value_computed": False,
        "primary_inference_cohort": "arbitrary_use",
        "secondary_cohort_role": "fixed_profile_descriptive_only",
        "overall_summary_role": "descriptive_only",
        "H1_then_H2_fixed_sequence": True,
        "holm_applied_to_H1_H2": False,
    }
    write_json(args.output / "no_pooled_p_value_receipt.json", no_pooling)
    make_plots(summary, flow_counts, args.output)
    audit = {
        "status": "PASS",
        "created_utc": utc_now(),
        "registered_protocol_commit": REGISTERED_PROTOCOL_COMMIT,
        "evaluator_source_commit": source["evaluator_source_commit"],
        "primary_cutoff": 70,
        "primary_cohort": "arbitrary_use",
        "H1_then_H2_fixed_sequence": True,
        "H1_p_value": fixed["H1_full_vs_single_scale"]["two_sided_p"],
        "H1_pass": fixed["H1_full_vs_single_scale"]["passes_alpha"],
        "H2_confirmatory_test_performed": fixed["H2_full_vs_local_linear"][
            "confirmatory_test_performed"
        ],
        "H2_p_value": fixed["H2_full_vs_local_linear"]["two_sided_p"],
        "H2_pass_under_gate": fixed["H2_full_vs_local_linear"][
            "passes_alpha_under_gate"
        ],
        "ordinary_pooled_p_value_computed": False,
        "confirmatory_multiplicity_adjustment": "fixed_sequence_gatekeeping_no_Holm",
        "failures": [],
    }
    write_json(args.output / "aggregate_audit.json", audit)
    print("[PASS] BIT cell-first aggregation and fixed-sequence inference complete")


def make_plots(summary: pd.DataFrame, flow_counts: list[dict[str, object]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for cohort in ("arbitrary_use", "fixed_profile"):
        group = summary[summary["cohort"].eq(cohort)].copy()
        if group.empty:
            continue
        figure, axis = plt.subplots(figsize=(7.2, 4.2))
        positions = np.arange(len(group))
        axis.bar(positions, group["future_capacity_RMSE_SOH_mean"])
        axis.errorbar(
            positions,
            group["future_capacity_RMSE_SOH_mean"],
            yerr=group["future_capacity_RMSE_SOH_std"].fillna(0.0),
            fmt="none",
            capsize=3,
        )
        axis.set_xticks(positions, group["model"], rotation=20, ha="right")
        axis.set_ylabel("Cell-level future SOH RMSE")
        axis.set_title(f"BIT cycle-70 RMSE: {cohort}")
        figure.tight_layout()
        figure.savefig(output / f"cycle70_rmse_{cohort}.png", dpi=300)
        plt.close(figure)
    flow = pd.DataFrame(flow_counts)
    figure, axis = plt.subplots(figsize=(8.0, 4.2))
    axis.bar(np.arange(len(flow)), flow["count"])
    axis.set_xticks(np.arange(len(flow)), flow["stage"], rotation=25, ha="right")
    axis.set_ylabel("Physical cells")
    axis.set_title("BIT cycle-70 analysis flow")
    figure.tight_layout()
    figure.savefig(output / "cycle70_analysis_flow.png", dpi=300)
    plt.close(figure)


def deterministic_zip(output_zip: Path, root: Path, files: Iterable[Path], manifest: dict) -> str:
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    if output_zip.exists():
        raise FileExistsError(output_zip)
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(files), key=lambda item: item.as_posix()):
            relative = path.relative_to(root).as_posix()
            info = zipfile.ZipInfo(relative)
            info.date_time = (1980, 1, 1, 0, 0, 0)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())
        info = zipfile.ZipInfo("BIT_RESULT_PACKAGE.json")
        info.date_time = (1980, 1, 1, 0, 0, 0)
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o100644 << 16
        archive.writestr(
            info,
            json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8") + b"\n",
        )
    bad = zipfile.ZipFile(output_zip).testzip()
    if bad is not None:
        raise RuntimeError(f"Result ZIP failed integrity test: {bad}")
    return sha256_file(output_zip)


def pack(args: argparse.Namespace) -> None:
    _, source, freeze_manifest, _ = runtime_gate(args)
    required = [
        args.prepared / "preparation_audit.json",
        args.evaluation / "evaluation_audit.json",
        args.aggregate / "aggregate_audit.json",
        args.aggregate / "confirmatory_fixed_sequence.json",
        args.aggregate / "no_pooled_p_value_receipt.json",
    ]
    for path in required:
        require_file(path)
        gate = read_json(path)
        if path.name.endswith("audit.json") and gate.get("status") != "PASS":
            raise RuntimeError(f"Result gate is not PASS: {path}")
    roots = [args.prepared, args.evaluation, args.aggregate]
    files: list[Path] = []
    file_hashes: dict[str, str] = {}
    package_root = args.package_root.resolve()
    package_root.mkdir(parents=True, exist_ok=True)
    staging = package_root / "staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    for label, root in zip(("prepared", "evaluation", "aggregate"), roots, strict=True):
        for source_path in sorted(root.rglob("*")):
            if not source_path.is_file():
                continue
            destination = staging / label / source_path.relative_to(root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination)
            files.append(destination)
            file_hashes[destination.relative_to(staging).as_posix()] = sha256_file(destination)
    evidence = {
        "registration_receipt.json": args.registration_receipt,
        "evaluator_source_freeze_receipt.json": args.source_receipt,
        "bit_freeze_manifest.json": args.freeze_manifest,
        "bit_config.json": args.bit_config,
        "development_config.json": args.development_config,
        "frozen_model_manifest.json": args.models / "frozen_model_manifest.json",
    }
    for name, source_path in evidence.items():
        require_file(source_path)
        destination = staging / "evidence" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)
        files.append(destination)
        file_hashes[destination.relative_to(staging).as_posix()] = sha256_file(destination)
    manifest = {
        "status": "PASS",
        "created_utc": utc_now(),
        "version": REGISTERED_VERSION,
        "governance_label": "post-BIT-structure-only, pre-model-evaluation amendment",
        "registered_protocol_commit": REGISTERED_PROTOCOL_COMMIT,
        "evaluator_source_commit": source["evaluator_source_commit"],
        "primary_cutoff": 70,
        "structurally_unavailable_cutoffs": [130, 190],
        "models": MODEL_IDS,
        "seeds": SEEDS,
        "cell_first_inference": True,
        "H1_then_H2_fixed_sequence": True,
        "ordinary_pooled_p_value_computed": False,
        "test_selection_or_retraining": False,
        "bit_freeze_manifest_sha256": sha256_file(args.freeze_manifest),
        "bit_archive_sha256": freeze_manifest.get("bit_archive_sha256"),
        "registered_protocol_freeze_zip_sha256": REGISTERED_FREEZE_SHA256,
        "source_freeze_zip_sha256": source["source_freeze_zip_sha256"],
        "files": file_hashes,
    }
    digest = deterministic_zip(args.output_zip, staging, files, manifest)
    external_manifest = args.output_zip.with_suffix(".manifest.json")
    write_json(external_manifest, {**manifest, "result_zip_sha256": digest})
    args.output_zip.with_suffix(args.output_zip.suffix + ".sha256").write_text(
        f"{digest}  {args.output_zip.name}\n",
        encoding="utf-8",
    )
    print(f"[PASS] BIT result package: {args.output_zip}")
    print(f"SHA256={digest}")


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--bit-config", type=Path, required=True)
    parser.add_argument("--development-config", type=Path, required=True)
    parser.add_argument("--registration-receipt", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--source-freeze-zip", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Registered BIT v0.2.1 evaluator")
    sub = parser.add_subparsers(dest="command", required=True)

    parse = sub.add_parser("parse")
    add_common(parse)
    parse.add_argument("--archive", type=Path, required=True)
    parse.add_argument("--parser", type=Path, required=True)
    parse.add_argument("--pairing-manifest", type=Path, required=True)
    parse.add_argument("--mapping-manifest", type=Path, required=True)
    parse.add_argument("--output", type=Path, required=True)
    parse.set_defaults(function=run_parser)

    prepare = sub.add_parser("prepare")
    add_common(prepare)
    prepare.add_argument("--raw-capacity-csv", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.set_defaults(function=prepare_soh)

    evaluation = sub.add_parser("evaluate")
    add_common(evaluation)
    evaluation.add_argument("--prepared", type=Path, required=True)
    evaluation.add_argument("--models", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument("--device", default="auto")
    evaluation.set_defaults(function=evaluate)

    aggregation = sub.add_parser("aggregate")
    add_common(aggregation)
    aggregation.add_argument("--prepared", type=Path, required=True)
    aggregation.add_argument("--evaluation", type=Path, required=True)
    aggregation.add_argument("--output", type=Path, required=True)
    aggregation.set_defaults(function=aggregate)

    package = sub.add_parser("pack")
    add_common(package)
    package.add_argument("--prepared", type=Path, required=True)
    package.add_argument("--evaluation", type=Path, required=True)
    package.add_argument("--aggregate", type=Path, required=True)
    package.add_argument("--models", type=Path, required=True)
    package.add_argument("--package-root", type=Path, required=True)
    package.add_argument("--output-zip", type=Path, required=True)
    package.set_defaults(function=pack)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    arguments.function(arguments)


if __name__ == "__main__":
    main()
