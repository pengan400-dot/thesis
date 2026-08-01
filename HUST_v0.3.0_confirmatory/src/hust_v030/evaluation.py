from __future__ import annotations

import json
import math
import os
import uuid
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

from mstt_soh.pipeline import (
    choose_device,
    eol_interval,
    evaluate_forecast,
    first_predicted_eol,
    load_frozen_models,
    load_protocol as load_dev_protocol,
    merge_seed_forecasts,
    recursive_forecast,
)

from .common import (
    assert_hash,
    json_sha256,
    load_hust_protocol,
    read_json,
    sha256_file,
    utc_now,
    write_json,
)
from .freeze import (
    _file_rows,
    _walk,
    _write_deterministic_zip,
    validate_calibration_receipt,
    validate_evaluator_receipt,
)
from .source import (
    find_dynamic_landmark,
    load_prepared_role,
    load_source_gate,
    prepare_role,
)
from .statistics import (
    conformal_order_statistic,
    fixed_sequence_confirmation,
    model_summary,
)


CORE_RECORD_COLUMNS = [
    "cell_id",
    "source_cell_id",
    "model",
    "landmark_cycle",
    "landmark_raw_soh",
    "bol_reference_capacity_Ah",
    "truth_event_type",
    "truth_event_observed",
    "right_censor_cycle",
    "pred_eol_cycle",
    "prediction_censor_cycle",
    "censor_aware_timing_score_cycles",
    "future_capacity_MAE_SOH",
    "future_capacity_RMSE_SOH",
    "future_observed_points",
    "max_abs_capacity_error_SOH",
    "true_eol_interval_lower",
    "true_eol_interval_upper",
    "threshold_hit",
]


def _records_frame(rows: list[dict[str, object]], *, seed_level: bool) -> pd.DataFrame:
    if rows:
        order = ["cell_id", "model", "seed"] if seed_level else ["cell_id", "model"]
        return pd.DataFrame(rows).sort_values(order).reset_index(drop=True)
    columns = list(CORE_RECORD_COLUMNS)
    if seed_level:
        columns.append("seed")
    else:
        for coverage in (80, 90):
            columns.extend(
                [
                    f"coverage_{coverage}_quantile_SOH",
                    f"coverage_{coverage}_simultaneous_trajectory_covered",
                    f"coverage_{coverage}_pointwise_coverage",
                    f"coverage_{coverage}_mean_interval_width_SOH",
                    f"coverage_{coverage}_eol_band_truth_compatible",
                    f"coverage_{coverage}_eol_band_lower_cycle",
                    f"coverage_{coverage}_eol_band_upper_cycle",
                ]
            )
    return pd.DataFrame(columns=columns)


def _model_options(protocol: Mapping[str, Any]) -> tuple[int, int, float, tuple[float, float], int, tuple[float, float]]:
    task = protocol["task"]
    local = next(
        item
        for item in protocol["model_lineage"]["models"]
        if item["id"] == "local_linear_trend"
    )
    return (
        int(task["input_window_cycles"]),
        int(task["forecast_horizon_cycles"]),
        float(protocol["soh_definition"]["eol_threshold_raw_SOH"]),
        tuple(map(float, task["prediction_clip_SOH"])),
        int(local["lookback_physical_cycles"]),
        tuple(map(float, local["slope_clip_SOH_per_cycle"])),
    )


def _load_models(
    dev_config: Path,
    model_root: Path,
    requested_device: str,
):
    dev_protocol = load_dev_protocol(dev_config)
    device = choose_device(requested_device)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    models = load_frozen_models(dev_protocol, model_root, device)
    return models, device


def _landmark(frame: pd.DataFrame, protocol: Mapping[str, Any]) -> int:
    value = find_dynamic_landmark(frame, protocol)
    if value is None:
        raise ValueError("Prepared curve no longer has the preregistered landmark")
    return int(value)


def _seed_ensemble_forecast(
    model_id: str,
    seed_models: Mapping[int, tuple[torch.nn.Module, object]],
    frame: pd.DataFrame,
    landmark: int,
    protocol: Mapping[str, Any],
    device: torch.device,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    window, horizon, threshold, clip, lookback, slope_clip = _model_options(protocol)
    forecasts: dict[int, pd.DataFrame] = {}
    seed_metrics: list[dict[str, object]] = []
    minimum_future = int(
        protocol["dynamic_landmark"]["minimum_future_observations_for_RMSE"]
    )
    for seed, (model, scaler) in sorted(seed_models.items()):
        forecast, predicted_eol = recursive_forecast(
            model_id,
            frame,
            landmark,
            horizon,
            threshold,
            window,
            clip,
            model=model,
            scaler=scaler,
            device=device,
            minimum_history=int(
                protocol["dynamic_landmark"]["minimum_observed_prefix_cycles"]
            ),
            local_linear_lookback=lookback,
            local_linear_slope_clip=slope_clip,
        )
        metrics = evaluate_forecast(
            frame,
            forecast,
            landmark,
            predicted_eol,
            threshold,
            horizon,
            minimum_future_observations=minimum_future,
        )
        metrics.update(
            {
                "cell_id": str(frame["battery_id"].iloc[0]),
                "model": model_id,
                "seed": int(seed),
                "landmark_cycle": landmark,
                "threshold_hit": int(predicted_eol is not None),
            }
        )
        seed_metrics.append(metrics)
        forecasts[int(seed)] = forecast
    return merge_seed_forecasts(forecasts), seed_metrics


def _ensemble_metrics(
    model_id: str,
    frame: pd.DataFrame,
    forecast: pd.DataFrame,
    landmark: int,
    protocol: Mapping[str, Any],
) -> dict[str, object]:
    _, horizon, threshold, _, _, _ = _model_options(protocol)
    predicted_eol = first_predicted_eol(forecast, threshold)
    result = evaluate_forecast(
        frame,
        forecast,
        landmark,
        predicted_eol,
        threshold,
        horizon,
        minimum_future_observations=int(
            protocol["dynamic_landmark"]["minimum_future_observations_for_RMSE"]
        ),
    )
    raw_at_landmark = float(
        frame.loc[frame["cycle"].eq(landmark), "raw_capacity"].iloc[0]
    )
    result.update(
        {
            "cell_id": str(frame["battery_id"].iloc[0]),
            "source_cell_id": str(frame["source_cell_id"].iloc[0]),
            "model": model_id,
            "landmark_cycle": landmark,
            "landmark_raw_soh": raw_at_landmark,
            "bol_reference_capacity_Ah": float(
                frame["bol_reference_capacity_Ah"].iloc[0]
            ),
            "threshold_hit": int(predicted_eol is not None),
        }
    )
    return result


def _local_forecast(
    frame: pd.DataFrame,
    landmark: int,
    protocol: Mapping[str, Any],
) -> pd.DataFrame:
    window, horizon, threshold, clip, lookback, slope_clip = _model_options(protocol)
    forecast, _ = recursive_forecast(
        "local_linear_trend",
        frame,
        landmark,
        horizon,
        threshold,
        window,
        clip,
        minimum_history=int(
            protocol["dynamic_landmark"]["minimum_observed_prefix_cycles"]
        ),
        local_linear_lookback=lookback,
        local_linear_slope_clip=slope_clip,
    )
    return forecast


def _trajectory_output(
    model_id: str,
    frame: pd.DataFrame,
    forecast: pd.DataFrame,
    landmark: int,
) -> pd.DataFrame:
    observed = frame[["cycle", "raw_capacity", "measurement_observed"]].rename(
        columns={"raw_capacity": "observed_raw_soh"}
    )
    output = forecast.merge(observed, on="cycle", how="left")
    output.insert(0, "model", model_id)
    output.insert(0, "cell_id", str(frame["battery_id"].iloc[0]))
    output.insert(2, "landmark_cycle", landmark)
    return output


def _eol_band_compatibility(
    frame: pd.DataFrame,
    forecast: pd.DataFrame,
    threshold: float,
    q: float,
) -> dict[str, object]:
    lower_hits = forecast[forecast["predicted_capacity"].sub(q).le(threshold)]
    upper_hits = forecast[forecast["predicted_capacity"].add(q).le(threshold)]
    lower = None if lower_hits.empty else int(lower_hits.iloc[0]["cycle"])
    upper = None if upper_hits.empty else int(upper_hits.iloc[0]["cycle"])
    truth = eol_interval(frame, threshold)
    last_observed = int(
        frame.loc[frame["measurement_observed"].eq(1), "cycle"].max()
    )
    if truth is None:
        compatible = upper is None or upper > last_observed
    elif lower is None:
        compatible = int(truth[0]) > int(forecast["cycle"].max())
    else:
        compatible = int(truth[1]) >= lower and (upper is None or int(truth[0]) <= upper)
    return {
        "eol_band_lower_cycle": lower,
        "eol_band_upper_cycle": upper,
        "eol_band_truth_compatible": int(bool(compatible)),
    }


def calibrate(
    config: Path,
    dev_config: Path,
    source_archive: Path,
    inventory: Path,
    evaluator_zip: Path,
    evaluator_manifest: Path,
    evaluator_receipt: Path,
    model_root: Path,
    calibration_dir: Path,
    device_name: str,
    project_root: Path,
) -> None:
    protocol, source_receipt, _ = load_source_gate(source_archive, config, inventory)
    if (Path(calibration_dir) / "calibration_audit.json").exists():
        raise RuntimeError(
            "HUST calibration was already completed in this run root; "
            "overwriting it is prohibited"
        )
    validate_evaluator_receipt(
        evaluator_zip,
        evaluator_manifest,
        evaluator_receipt,
        model_root,
        protocol,
        project_root,
        source_receipt,
    )
    curves = load_prepared_role(calibration_dir, protocol, "calibration")
    models, device = _load_models(dev_config, model_root, device_name)
    full_models = models["mstt_full_K5"]
    rows: list[dict[str, object]] = []
    seed_rows: list[dict[str, object]] = []
    trajectories = Path(calibration_dir) / "trajectories"
    trajectories.mkdir(parents=True, exist_ok=True)
    for cell_id, frame in sorted(curves.items()):
        landmark = _landmark(frame, protocol)
        ensemble, current_seed_rows = _seed_ensemble_forecast(
            "mstt_full_K5",
            full_models,
            frame,
            landmark,
            protocol,
            device,
        )
        metrics = _ensemble_metrics(
            "mstt_full_K5",
            frame,
            ensemble,
            landmark,
            protocol,
        )
        rows.append(
            {
                **metrics,
                "nonconformity_max_abs_SOH": metrics["max_abs_capacity_error_SOH"],
            }
        )
        seed_rows.extend(current_seed_rows)
        _trajectory_output(
            "mstt_full_K5",
            frame,
            ensemble,
            landmark,
        ).to_csv(trajectories / f"{cell_id}_mstt_full_K5.csv", index=False)
    if rows:
        scores = pd.DataFrame(rows).sort_values("cell_id").reset_index(drop=True)
    else:
        scores = pd.DataFrame(
            columns=CORE_RECORD_COLUMNS + ["nonconformity_max_abs_SOH"]
        )
    seed_frame = _records_frame(seed_rows, seed_level=True)
    scores_path = Path(calibration_dir) / "calibration_scores.csv"
    seed_path = Path(calibration_dir) / "calibration_seed_records.csv"
    scores.to_csv(scores_path, index=False)
    seed_frame.to_csv(seed_path, index=False)
    minimum = int(protocol["calibration"]["minimum_eligible_calibration_cells_for_UQ"])
    eligible = len(scores)
    quantiles: dict[str, object] = {
        "status": "PASS" if eligible >= minimum else "UQ_NOT_ESTIMABLE",
        "created_utc": utc_now(),
        "eligible_calibration_cells": eligible,
        "minimum_required": minimum,
        "score": protocol["calibration"]["nonconformity_score"],
        "quantile_rule": protocol["calibration"]["quantile_rule"],
        "quantiles": {},
    }
    if eligible >= minimum:
        values = scores["nonconformity_max_abs_SOH"].to_numpy(float)
        for coverage in protocol["calibration"]["nominal_simultaneous_coverages"]:
            quantiles["quantiles"][str(float(coverage))] = conformal_order_statistic(
                values,
                float(coverage),
            )
    quantile_path = Path(calibration_dir) / "calibration_quantiles.json"
    write_json(quantile_path, quantiles)
    audit = {
        "status": quantiles["status"],
        "created_utc": utc_now(),
        "config_sha256": json_sha256(protocol),
        "evaluator_archive_sha256": sha256_file(evaluator_zip),
        "eligible_calibration_cells": eligible,
        "minimum_required_for_UQ": minimum,
        "calibration_scores_sha256": sha256_file(scores_path),
        "calibration_seed_records_sha256": sha256_file(seed_path),
        "calibration_quantiles_sha256": sha256_file(quantile_path),
        "model_selection_or_update_performed": False,
        "landmark_or_exclusion_update_performed": False,
        "confirmation_members_unpickled": False,
    }
    write_json(Path(calibration_dir) / "calibration_audit.json", audit)
    print(
        f"[{audit['status']}] HUST calibration: eligible={eligible} "
        f"minimum_for_UQ={minimum}"
    )


def _verify_local_calibration_files(
    calibration_dir: Path,
    calibration_manifest: Path,
) -> dict[str, Any]:
    manifest = read_json(calibration_manifest)
    expected = {
        str(row["archive_path"]): str(row["sha256"])
        for row in manifest["files"]
    }
    for filename in ("calibration_quantiles.json", "calibration_audit.json"):
        key = f"calibration/{filename}"
        if key not in expected:
            raise RuntimeError(f"Calibration freeze omitted {key}")
        assert_hash(Path(calibration_dir) / filename, expected[key], key)
    return read_json(Path(calibration_dir) / "calibration_quantiles.json")


def _one_shot_begin(
    state_path: Path,
    protocol: Mapping[str, Any],
    source_sha256: str,
    evaluator_sha256: str,
    calibration_sha256: str,
    resume: bool,
) -> dict[str, Any]:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    if state_path.exists():
        state = read_json(state_path)
        if state.get("status") == "PASS":
            raise RuntimeError("HUST confirmation already completed; rerun is prohibited")
        if not resume:
            raise RuntimeError(
                "HUST confirmation was already opened. Use confirm_resume only to continue "
                "the exact frozen run; do not initialize a new analysis."
            )
        locked = {
            "config_sha256": json_sha256(protocol),
            "source_archive_sha256": source_sha256,
            "evaluator_archive_sha256": evaluator_sha256,
            "calibration_archive_sha256": calibration_sha256,
        }
        if any(state.get(key) != value for key, value in locked.items()):
            raise RuntimeError("One-shot resume hashes differ from the opened run")
        state["status"] = "STARTED"
        state["last_resume_utc"] = utc_now()
        state["invocation_count"] = int(state.get("invocation_count", 1)) + 1
        write_json(state_path, state)
        return state
    if resume:
        raise RuntimeError("No opened HUST confirmation exists to resume")
    state = {
        "status": "STARTED",
        "created_utc": utc_now(),
        "run_uuid": str(uuid.uuid4()),
        "invocation_count": 1,
        "config_sha256": json_sha256(protocol),
        "source_archive_sha256": source_sha256,
        "evaluator_archive_sha256": evaluator_sha256,
        "calibration_archive_sha256": calibration_sha256,
        "confirmation_members_considered_open_from_this_receipt": True,
        "new_analysis_after_failure_prohibited": True,
        "exact_code_data_resume_allowed": True,
    }
    # The receipt is written before any confirmation member is read.
    write_json(state_path, state)
    return state


def _confirmation_uq(
    frame: pd.DataFrame,
    forecast: pd.DataFrame,
    protocol: Mapping[str, Any],
    quantiles: Mapping[str, Any],
) -> dict[str, object]:
    output: dict[str, object] = {}
    _, horizon, threshold, _, _, _ = _model_options(protocol)
    truth = eol_interval(frame, threshold)
    observed = frame[frame["measurement_observed"].eq(1)]
    support_end = int(truth[1]) if truth is not None else int(observed["cycle"].max())
    landmark = int(forecast["cycle"].min()) - 1
    measured = observed[
        observed["cycle"].gt(landmark)
        & observed["cycle"].le(min(landmark + horizon, support_end))
    ][["cycle", "raw_capacity"]]
    aligned = measured.merge(forecast, on="cycle", how="inner")
    residual = np.abs(
        aligned["predicted_capacity"].to_numpy(float)
        - aligned["raw_capacity"].to_numpy(float)
    )
    for coverage in protocol["calibration"]["nominal_simultaneous_coverages"]:
        label = str(float(coverage))
        q = quantiles.get("quantiles", {}).get(label)
        prefix = f"coverage_{int(round(float(coverage) * 100))}"
        if q is None:
            output[f"{prefix}_quantile_SOH"] = math.nan
            output[f"{prefix}_simultaneous_trajectory_covered"] = math.nan
            output[f"{prefix}_pointwise_coverage"] = math.nan
            output[f"{prefix}_mean_interval_width_SOH"] = math.nan
            output[f"{prefix}_eol_band_truth_compatible"] = math.nan
            continue
        q = float(q)
        output[f"{prefix}_quantile_SOH"] = q
        output[f"{prefix}_simultaneous_trajectory_covered"] = int(
            len(residual) > 0 and bool(np.all(residual <= q + 1e-12))
        )
        output[f"{prefix}_pointwise_coverage"] = (
            float(np.mean(residual <= q + 1e-12)) if len(residual) else math.nan
        )
        output[f"{prefix}_mean_interval_width_SOH"] = 2.0 * q
        band = _eol_band_compatibility(frame, forecast, threshold, q)
        output[f"{prefix}_eol_band_truth_compatible"] = band[
            "eol_band_truth_compatible"
        ]
        output[f"{prefix}_eol_band_lower_cycle"] = band["eol_band_lower_cycle"]
        output[f"{prefix}_eol_band_upper_cycle"] = band["eol_band_upper_cycle"]
    return output


def confirm_one_shot(
    config: Path,
    dev_config: Path,
    source_archive: Path,
    inventory: Path,
    evaluator_zip: Path,
    evaluator_manifest: Path,
    evaluator_receipt: Path,
    calibration_zip: Path,
    calibration_manifest: Path,
    calibration_receipt: Path,
    calibration_dir: Path,
    model_root: Path,
    output: Path,
    device_name: str,
    project_root: Path,
    *,
    resume: bool,
    trust_official_pickle: bool,
) -> None:
    protocol, source_receipt, _ = load_source_gate(source_archive, config, inventory)
    evaluator = validate_evaluator_receipt(
        evaluator_zip,
        evaluator_manifest,
        evaluator_receipt,
        model_root,
        protocol,
        project_root,
        source_receipt,
    )
    calibration = validate_calibration_receipt(
        calibration_zip,
        calibration_manifest,
        calibration_receipt,
        protocol,
        str(evaluator["archive_sha256"]),
    )
    quantiles = _verify_local_calibration_files(calibration_dir, calibration_manifest)
    output = Path(output)
    state_path = output / "one_shot_state.json"
    state = _one_shot_begin(
        state_path,
        protocol,
        str(source_receipt["archive_sha256"]),
        str(evaluator["archive_sha256"]),
        str(calibration["archive_sha256"]),
        resume,
    )
    try:
        prepared = output / "prepared"
        if resume and (prepared / "confirmation_preflight.json").is_file():
            # Reuse the exact hash-checked prepared confirmation curves after an
            # interruption that occurred later in the same frozen run.
            pass
        else:
            prepare_role(
                source_archive,
                config,
                inventory,
                prepared,
                "confirmation",
                trust_official_pickle=trust_official_pickle,
            )
        preflight_path = prepared / "confirmation_preflight.json"
        preflight = read_json(preflight_path)
        preflight_hash = sha256_file(preflight_path)
        if state.get("confirmation_preflight_sha256") not in {None, preflight_hash}:
            raise RuntimeError("Prepared confirmation receipt changed during exact resume")
        if state.get("confirmation_curves_sha256") not in {
            None,
            preflight.get("curves_sha256"),
        }:
            raise RuntimeError("Prepared confirmation curves changed during exact resume")
        state.update(
            {
                "confirmation_preflight_sha256": preflight_hash,
                "confirmation_curves_sha256": preflight.get("curves_sha256"),
                "prepared_confirmation_bound_utc": utc_now(),
            }
        )
        write_json(state_path, state)
        curves = load_prepared_role(prepared, protocol, "confirmation")
        models, device = _load_models(dev_config, model_root, device_name)
        records: list[dict[str, object]] = []
        seed_records: list[dict[str, object]] = []
        trajectories = output / "trajectories"
        trajectories.mkdir(parents=True, exist_ok=True)
        for cell_id, frame in sorted(curves.items()):
            landmark = _landmark(frame, protocol)
            for model_id in ("mstt_full_K5", "mstt_single_scale_K5"):
                ensemble, current_seed_rows = _seed_ensemble_forecast(
                    model_id,
                    models[model_id],
                    frame,
                    landmark,
                    protocol,
                    device,
                )
                metrics = _ensemble_metrics(
                    model_id,
                    frame,
                    ensemble,
                    landmark,
                    protocol,
                )
                if model_id == "mstt_full_K5":
                    metrics.update(_confirmation_uq(frame, ensemble, protocol, quantiles))
                records.append(metrics)
                seed_records.extend(current_seed_rows)
                _trajectory_output(model_id, frame, ensemble, landmark).to_csv(
                    trajectories / f"{cell_id}_{model_id}.csv",
                    index=False,
                )
            local = _local_forecast(frame, landmark, protocol)
            local_metrics = _ensemble_metrics(
                "local_linear_trend",
                frame,
                local,
                landmark,
                protocol,
            )
            records.append(local_metrics)
            _trajectory_output(
                "local_linear_trend",
                frame,
                local,
                landmark,
            ).to_csv(
                trajectories / f"{cell_id}_local_linear_trend.csv",
                index=False,
            )
        records_frame = _records_frame(records, seed_level=False)
        seed_frame = _records_frame(seed_records, seed_level=True)
        records_path = output / "ensemble_cell_records.csv"
        seed_path = output / "seed_cell_records.csv"
        records_frame.to_csv(records_path, index=False)
        seed_frame.to_csv(seed_path, index=False)
        prepared_gate = read_json(prepared / "confirmation_preflight.json")
        audit = {
            "status": "PASS",
            "created_utc": utc_now(),
            "run_uuid": state["run_uuid"],
            "config_sha256": json_sha256(protocol),
            "source_archive_sha256": source_receipt["archive_sha256"],
            "evaluator_archive_sha256": evaluator["archive_sha256"],
            "calibration_archive_sha256": calibration["archive_sha256"],
            "assigned_confirmation_cells": prepared_gate["assigned_cells"],
            "evaluable_confirmation_cells": prepared_gate["evaluable_cells"],
            "excluded_confirmation_cells": prepared_gate["excluded_cells"],
            "ensemble_cell_records": len(records_frame),
            "seed_cell_records": len(seed_frame),
            "models": sorted(records_frame["model"].unique().tolist()),
            "seed_aggregation": protocol["task"]["seed_aggregation"],
            "post_unblinding_model_or_protocol_change": False,
            "ensemble_records_sha256": sha256_file(records_path),
            "seed_records_sha256": sha256_file(seed_path),
        }
        write_json(output / "evaluation_audit.json", audit)
        state.update(
            {
                "status": "PASS",
                "completed_utc": utc_now(),
                "evaluation_audit_sha256": sha256_file(output / "evaluation_audit.json"),
                "evaluable_confirmation_cells": prepared_gate["evaluable_cells"],
            }
        )
        write_json(state_path, state)
        print(
            "[PASS] HUST one-shot confirmation: "
            f"evaluable_cells={prepared_gate['evaluable_cells']} records={len(records_frame)}"
        )
    except Exception as error:
        state.update(
            {
                "status": "FAILED_EXACT_RESUME_ONLY",
                "failed_utc": utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        write_json(state_path, state)
        raise


def aggregate_confirmation(
    config: Path,
    confirmation_dir: Path,
    output: Path,
) -> None:
    protocol = load_hust_protocol(config)
    if (Path(output) / "aggregate_audit.json").exists():
        raise RuntimeError("HUST confirmation was already aggregated; refusing overwrite")
    state = read_json(Path(confirmation_dir) / "one_shot_state.json")
    audit = read_json(Path(confirmation_dir) / "evaluation_audit.json")
    records_path = Path(confirmation_dir) / "ensemble_cell_records.csv"
    if (
        state.get("status") != "PASS"
        or audit.get("status") != "PASS"
        or audit.get("config_sha256") != json_sha256(protocol)
        or state.get("evaluation_audit_sha256")
        != sha256_file(Path(confirmation_dir) / "evaluation_audit.json")
        or audit.get("ensemble_records_sha256") != sha256_file(records_path)
    ):
        raise RuntimeError("HUST one-shot evaluation gate failed")
    records = pd.read_csv(records_path)
    if records.duplicated(["cell_id", "model"]).any():
        raise RuntimeError("Duplicate physical-cell/model records in confirmation output")
    expected_records = int(audit["evaluable_confirmation_cells"]) * 3
    if len(records) != expected_records:
        raise RuntimeError(
            f"Expected {expected_records} ensemble records, found {len(records)}"
        )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary = model_summary(records)
    summary.to_csv(output / "hust_confirmatory_model_summary.csv", index=False)
    fixed = fixed_sequence_confirmation(records, protocol)
    write_json(output / "hust_fixed_sequence_inference.json", fixed)
    inference_rows: list[dict[str, object]] = []
    for key in ("H1_full_vs_local_linear", "H2_full_vs_single_scale"):
        value = fixed.get(key)
        if value is not None:
            inference_rows.append({"hypothesis": key, **value})
    inference_frame = pd.DataFrame(inference_rows)
    if inference_frame.empty:
        inference_frame = pd.DataFrame(columns=["hypothesis", "status"])
    inference_frame.to_csv(
        output / "hust_fixed_sequence_effects.csv",
        index=False,
    )
    full = records[records["model"].eq("mstt_full_K5")]
    coverage_rows = []
    for coverage in protocol["calibration"]["nominal_simultaneous_coverages"]:
        prefix = f"coverage_{int(round(float(coverage) * 100))}"
        values = pd.to_numeric(
            full[f"{prefix}_simultaneous_trajectory_covered"],
            errors="coerce",
        ).dropna()
        coverage_rows.append(
            {
                "nominal_simultaneous_coverage": float(coverage),
                "eligible_cells": len(values),
                "empirical_simultaneous_coverage": (
                    float(values.mean()) if len(values) else math.nan
                ),
                "mean_pointwise_coverage": float(
                    pd.to_numeric(full[f"{prefix}_pointwise_coverage"], errors="coerce").mean()
                ),
                "mean_interval_width_SOH": float(
                    pd.to_numeric(
                        full[f"{prefix}_mean_interval_width_SOH"], errors="coerce"
                    ).mean()
                ),
                "eol_band_compatibility_rate": float(
                    pd.to_numeric(
                        full[f"{prefix}_eol_band_truth_compatible"], errors="coerce"
                    ).mean()
                ),
            }
        )
    pd.DataFrame(coverage_rows).to_csv(
        output / "hust_calibration_coverage_summary.csv",
        index=False,
    )
    preflight = read_json(
        Path(confirmation_dir) / "prepared" / "confirmation_preflight.json"
    )
    flow = {
        "assigned_confirmation_cells": preflight["assigned_cells"],
        "evaluable_confirmation_cells": preflight["evaluable_cells"],
        "excluded_confirmation_cells": preflight["excluded_cells"],
        "minimum_required_for_confirmation": protocol["statistics"][
            "minimum_eligible_confirmation_cells"
        ],
        "confirmatory_status": fixed["status"],
    }
    write_json(output / "hust_confirmation_flow.json", flow)
    _plots(records, summary, output)
    aggregate_audit = {
        "status": "PASS",
        "created_utc": utc_now(),
        "config_sha256": json_sha256(protocol),
        "run_uuid": state["run_uuid"],
        "confirmatory_status": fixed["status"],
        "physical_cell_unit": True,
        "seed_counted_as_independent_sample": False,
        "post_confirmation_model_or_protocol_change": False,
        "summary_sha256": sha256_file(output / "hust_confirmatory_model_summary.csv"),
        "inference_sha256": sha256_file(output / "hust_fixed_sequence_inference.json"),
    }
    write_json(output / "aggregate_audit.json", aggregate_audit)
    print(
        f"[PASS] HUST aggregation: status={fixed['status']} "
        f"cells={fixed['complete_paired_cells']}"
    )


def _plots(records: pd.DataFrame, summary: pd.DataFrame, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    del summary
    order = ["mstt_full_K5", "local_linear_trend", "mstt_single_scale_K5"]
    colors = ["#2b6cb0", "#dd6b20", "#805ad5"]
    if records.empty:
        figure, axis = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
        axis.axis("off")
        axis.text(
            0.5,
            0.5,
            "No evaluable confirmation cell under the frozen landmark",
            ha="center",
            va="center",
        )
        figure.savefig(output / "hust_confirmatory_rmse_boxplot.png", dpi=300)
        figure.savefig(output / "hust_confirmatory_rmse_boxplot.pdf")
        figure.savefig(output / "hust_paired_rmse_scatter.png", dpi=300)
        figure.savefig(output / "hust_paired_rmse_scatter.pdf")
        plt.close(figure)
        return
    values = [
        records.loc[records["model"].eq(model), "future_capacity_RMSE_SOH"].to_numpy(float)
        for model in order
    ]
    figure, axis = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    box = axis.boxplot(values, labels=order, patch_artist=True, showmeans=True)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    axis.set_ylabel("Cell-level future SOH RMSE")
    axis.set_title("HUST one-shot confirmation")
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(output / "hust_confirmatory_rmse_boxplot.png", dpi=300)
    figure.savefig(output / "hust_confirmatory_rmse_boxplot.pdf")
    plt.close(figure)

    pivot = records.pivot(
        index="cell_id",
        columns="model",
        values="future_capacity_RMSE_SOH",
    ).dropna()
    if pivot.empty:
        figure, axis = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
        axis.axis("off")
        axis.text(
            0.5,
            0.5,
            "No complete three-model confirmation pair",
            ha="center",
            va="center",
        )
        figure.savefig(output / "hust_paired_rmse_scatter.png", dpi=300)
        figure.savefig(output / "hust_paired_rmse_scatter.pdf")
        plt.close(figure)
        return
    figure, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), constrained_layout=True)
    for axis, comparator, title in (
        (axes[0], "local_linear_trend", "Full vs local linear"),
        (axes[1], "mstt_single_scale_K5", "Full vs single-scale"),
    ):
        axis.scatter(
            pivot[comparator],
            pivot["mstt_full_K5"],
            alpha=0.75,
            s=28,
            color="#2b6cb0",
        )
        upper = float(np.nanmax([pivot[comparator].max(), pivot["mstt_full_K5"].max()]))
        axis.plot([0, upper], [0, upper], linestyle="--", color="black", linewidth=1)
        axis.set_xlabel(f"{comparator} RMSE")
        axis.set_ylabel("mstt_full_K5 RMSE")
        axis.set_title(title)
        axis.grid(alpha=0.2)
    figure.savefig(output / "hust_paired_rmse_scatter.png", dpi=300)
    figure.savefig(output / "hust_paired_rmse_scatter.pdf")
    plt.close(figure)


def pack_results(
    config: Path,
    inventory: Path,
    evaluator_manifest: Path,
    evaluator_receipt: Path,
    calibration_manifest: Path,
    calibration_receipt: Path,
    calibration_dir: Path,
    confirmation_dir: Path,
    aggregate_dir: Path,
    project_root: Path,
    output_zip: Path,
) -> None:
    protocol = load_hust_protocol(config)
    for candidate in (Path(output_zip), Path(str(output_zip) + ".sha256")):
        if candidate.exists():
            raise FileExistsError(f"Refusing to overwrite result artifact: {candidate}")
    state = read_json(Path(confirmation_dir) / "one_shot_state.json")
    evaluation = read_json(Path(confirmation_dir) / "evaluation_audit.json")
    aggregate = read_json(Path(aggregate_dir) / "aggregate_audit.json")
    evaluator_manifest_payload = read_json(evaluator_manifest)
    evaluator_receipt_payload = read_json(evaluator_receipt)
    calibration_manifest_payload = read_json(calibration_manifest)
    calibration_receipt_payload = read_json(calibration_receipt)
    if (
        state.get("status") != "PASS"
        or evaluation.get("status") != "PASS"
        or aggregate.get("status") != "PASS"
        or aggregate.get("config_sha256") != json_sha256(protocol)
        or evaluation.get("config_sha256") != json_sha256(protocol)
        or evaluation.get("run_uuid") != state.get("run_uuid")
        or aggregate.get("run_uuid") != state.get("run_uuid")
        or evaluation.get("evaluator_archive_sha256")
        != evaluator_manifest_payload.get("archive_sha256")
        or evaluator_receipt_payload.get("archive_sha256")
        != evaluator_manifest_payload.get("archive_sha256")
        or evaluator_receipt_payload.get("manifest_sha256")
        != sha256_file(evaluator_manifest)
        or evaluation.get("calibration_archive_sha256")
        != calibration_manifest_payload.get("archive_sha256")
        or calibration_receipt_payload.get("archive_sha256")
        != calibration_manifest_payload.get("archive_sha256")
        or calibration_receipt_payload.get("manifest_sha256")
        != sha256_file(calibration_manifest)
        or state.get("evaluation_audit_sha256")
        != sha256_file(Path(confirmation_dir) / "evaluation_audit.json")
        or aggregate.get("summary_sha256")
        != sha256_file(Path(aggregate_dir) / "hust_confirmatory_model_summary.csv")
        or aggregate.get("inference_sha256")
        != sha256_file(Path(aggregate_dir) / "hust_fixed_sequence_inference.json")
    ):
        raise RuntimeError("Results are not ready to package")
    selected: list[tuple[str, Path]] = []
    selected.extend(_walk("source_inventory", Path(inventory)))
    for label, path in (
        ("receipts/evaluator_manifest.json", evaluator_manifest),
        ("receipts/evaluator_receipt.json", evaluator_receipt),
        ("receipts/calibration_manifest.json", calibration_manifest),
        ("receipts/calibration_receipt.json", calibration_receipt),
    ):
        selected.append((label, Path(path)))
    for directory, prefix in (
        (Path(calibration_dir), "calibration"),
        (Path(confirmation_dir), "confirmation"),
        (Path(aggregate_dir), "aggregate"),
    ):
        for archive_path, source_path in _walk(prefix, directory):
            if "/curves/" in archive_path or archive_path.endswith(".pt"):
                continue
            selected.append((archive_path, source_path))
    for path in (
        Path(project_root) / "configs" / "hust_confirmatory_protocol_v0.3.0.json",
        Path(project_root) / "preregistration" / "HUST_CONFIRMATORY_PROTOCOL_v0.3.0.md",
        Path(project_root) / "README_CN.md",
    ):
        selected.append((f"protocol/{path.name}", path))
    rows = _file_rows(selected)
    manifest = {
        "schema_version": "1.0",
        "result_kind": "HUST_v0.3.0_one_shot_confirmation",
        "created_utc": utc_now(),
        "config_sha256": json_sha256(protocol),
        "run_uuid": state["run_uuid"],
        "confirmatory_status": aggregate["confirmatory_status"],
        "source_archive_sha256": evaluation["source_archive_sha256"],
        "evaluator_archive_sha256": evaluation["evaluator_archive_sha256"],
        "calibration_archive_sha256": evaluation["calibration_archive_sha256"],
        "files": [
            {
                "archive_path": row["archive_path"],
                "size_bytes": row["size_bytes"],
                "sha256": row["sha256"],
            }
            for row in rows
        ],
    }
    manifest_bytes = (
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    digest = _write_deterministic_zip(
        output_zip,
        rows,
        {"RESULT_MANIFEST.json": manifest_bytes},
    )
    Path(str(output_zip) + ".sha256").write_text(
        f"{digest}  {Path(output_zip).name}\n",
        encoding="utf-8",
    )
    print(f"[PASS] HUST result package: {output_zip}")
    print(f"SHA256={digest}")
