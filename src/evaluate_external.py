#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from common import (
    FEATURE_NAMES,
    choose_device,
    curves_sha256,
    eol_interval,
    evaluate_forecast,
    json_sha256,
    load_protocol,
    recursive_forecast,
    sha256_file,
)
from model_variants import PaperMSTTVariant, identity_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--external-dir", type=Path, required=True)
    parser.add_argument("--frozen-model-root", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_external_curves(path: Path, config_hash: str) -> dict[str, pd.DataFrame]:
    gate = json.loads((path / "external_preflight.json").read_text(encoding="utf-8"))
    if gate.get("status") != "PASS" or gate.get("config_sha256") != config_hash:
        raise RuntimeError("External preflight gate is not PASS")
    curves = {}
    for csv_path in sorted((path / "curves").glob("B*.csv")):
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
        curves[str(frame["battery_id"].iloc[0])] = frame
    counts = {
        batch: sum(int(frame["batch"].iloc[0]) == batch for frame in curves.values())
        for batch in (4, 5, 6)
    }
    if counts != {4: 8, 5: 8, 6: 8}:
        raise ValueError(f"Expected 8 external cells per batch, got {counts}")
    return curves


def load_frozen_models(
    protocol: dict,
    config_hash: str,
    frozen_root: Path,
    device: torch.device,
) -> dict[str, dict[int, tuple[torch.nn.Module, object]]]:
    manifest = json.loads(
        (frozen_root / "frozen_model_manifest.json").read_text(encoding="utf-8")
    )
    if (
        manifest.get("status") != "PASS"
        or manifest.get("external_data_loaded") is not False
        or manifest.get("config_sha256") != config_hash
    ):
        raise RuntimeError("Frozen-model manifest gate failed")
    seeds = [int(value) for value in protocol["training"]["seeds"]]
    result = {}
    for spec in protocol["models"]:
        model_id = str(spec["id"])
        if not model_id.startswith("mstt_"):
            continue
        result[model_id] = {}
        for seed in seeds:
            job_dir = frozen_root / "models" / model_id / f"seed_{seed}"
            audit = json.loads((job_dir / "job_audit.json").read_text(encoding="utf-8"))
            model_path = job_dir / "model.pt"
            scaler_path = job_dir / "scaler.joblib"
            if (
                audit.get("status") != "PASS"
                or audit.get("external_data_loaded") is not False
                or audit.get("config_sha256") != config_hash
                or audit.get("development_data_sha256")
                != manifest.get("development_data_sha256")
                or sha256_file(model_path) != audit.get("model_sha256")
                or sha256_file(scaler_path) != audit.get("scaler_sha256")
            ):
                raise RuntimeError(f"Frozen checkpoint gate failed: {job_dir}")
            model = PaperMSTTVariant(
                len(FEATURE_NAMES), ablation=str(spec["ablation"])
            ).to(device)
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            if checkpoint["config_sha256"] != config_hash or checkpoint.get(
                "development_data_sha256"
            ) != manifest.get("development_data_sha256"):
                raise RuntimeError(f"Checkpoint config mismatch: {model_path}")
            if identity_record(model) != checkpoint["identity"]:
                raise RuntimeError(f"Checkpoint identity mismatch: {model_path}")
            model.eval()
            result[model_id][seed] = (model, joblib.load(scaler_path))
    return result


def merge_seed_forecasts(seed_forecasts: dict[int, pd.DataFrame]) -> pd.DataFrame:
    merged = None
    for seed, frame in sorted(seed_forecasts.items()):
        current = frame.rename(
            columns={"predicted_capacity": f"predicted_capacity_seed_{seed}"}
        )
        merged = current if merged is None else merged.merge(current, on="cycle")
    seed_columns = [
        column for column in merged if column.startswith("predicted_capacity_seed_")
    ]
    merged["predicted_capacity"] = merged[seed_columns].mean(axis=1)
    return merged


def predicted_eol(forecast: pd.DataFrame, threshold: float) -> int | None:
    hits = forecast[forecast["predicted_capacity"] <= threshold]
    return None if hits.empty else int(hits.iloc[0]["cycle"])


def uncertainty_metrics(
    frame: pd.DataFrame,
    forecast: pd.DataFrame,
    cutoff: int,
    horizon: int,
    threshold: float,
    q: float,
) -> dict[str, float | int]:
    interval = eol_interval(frame, threshold)
    observed_truth = frame[frame["measurement_observed"].eq(1)].dropna(
        subset=["raw_capacity"]
    )
    if observed_truth.empty:
        raise ValueError("No genuine reference-capacity observations")
    right_censor_cycle = int(observed_truth["cycle"].max())
    truth_support_end = int(interval[1]) if interval is not None else right_censor_cycle
    measured = frame[
        frame["measurement_observed"].eq(1)
        & frame["cycle"].gt(cutoff)
        & frame["cycle"].le(min(cutoff + horizon, truth_support_end))
    ][["cycle", "raw_capacity"]]
    aligned = measured.merge(
        forecast[["cycle", "predicted_capacity"]], on="cycle", how="inner"
    )
    max_error = float(
        np.max(
            np.abs(
                aligned["predicted_capacity"].to_numpy(float)
                - aligned["raw_capacity"].to_numpy(float)
            )
        )
    )
    lower_hits = forecast[forecast["predicted_capacity"] - q <= threshold]
    upper_hits = forecast[forecast["predicted_capacity"] + q <= threshold]
    earliest = (
        int(lower_hits.iloc[0]["cycle"])
        if not lower_hits.empty
        else cutoff + horizon + 1
    )
    latest = (
        int(upper_hits.iloc[0]["cycle"])
        if not upper_hits.empty
        else cutoff + horizon + 1
    )
    if interval is not None:
        eol_band_compatible = int(earliest <= interval[1] and latest >= interval[0])
    else:
        eol_band_compatible = int(latest > right_censor_cycle)
    return {
        "trajectory_band_half_width_Ah": q,
        "trajectory_band_covered": int(max_error <= q),
        "trajectory_band_max_abs_error_Ah": max_error,
        "eol_band_earliest_cycle": earliest,
        "eol_band_latest_cycle": latest,
        "eol_band_truth_compatible": eol_band_compatible,
    }


def main() -> None:
    args = parse_args()
    protocol = load_protocol(args.config)
    config_hash = json_sha256(protocol)
    task = protocol["task"]
    threshold = float(task["eol_threshold_Ah"])
    window = int(task["input_window_cycles"])
    horizon = int(task["forecast_horizon_cycles"])
    cutoffs = [int(value) for value in task["cutoffs"]]
    device = choose_device(args.device)
    curves = load_external_curves(args.external_dir, config_hash)
    external_data_hash = curves_sha256(curves)
    models = load_frozen_models(protocol, config_hash, args.frozen_model_root, device)
    calibration = json.loads(
        (args.calibration_dir / "calibration_quantiles.json").read_text(
            encoding="utf-8"
        )
    )
    if (
        calibration.get("status") != "PASS"
        or calibration.get("config_sha256") != config_hash
        or calibration.get("external_data_loaded") is not False
    ):
        raise RuntimeError("Calibration gate failed")
    quantiles = {
        key: float(value) for key, value in calibration["coverage_quantiles_Ah"].items()
    }
    primary_id = str(protocol["uncertainty"]["model"])

    output = args.output_dir.resolve()
    audit_path = output / "evaluation_audit.json"
    if audit_path.is_file() and not args.overwrite:
        old = json.loads(audit_path.read_text(encoding="utf-8"))
        if (
            old.get("status") == "PASS"
            and old.get("config_sha256") == config_hash
            and old.get("external_data_sha256") == external_data_hash
        ):
            print(f"[SKIP] external evaluation already passed: {output}")
            return
    output.mkdir(parents=True, exist_ok=True)
    trajectories = output / "trajectories"
    trajectories.mkdir(exist_ok=True)

    ensemble_records = []
    seed_records = []
    exclusions = []
    for cell_id, frame in sorted(curves.items()):
        batch = int(frame["batch"].iloc[0])
        interval = eol_interval(frame, threshold)
        last_reference_cycle = int(
            frame.loc[
                frame["measurement_observed"].eq(1) & frame["raw_capacity"].notna(),
                "cycle",
            ].max()
        )
        for cutoff in cutoffs:
            if cutoff > int(frame["cycle"].max()):
                exclusions.append(
                    {
                        "cell_id": cell_id,
                        "batch": batch,
                        "cutoff": cutoff,
                        "reason": "cutoff_after_last_recorded_cycle",
                    }
                )
                continue
            if cutoff >= last_reference_cycle:
                exclusions.append(
                    {
                        "cell_id": cell_id,
                        "batch": batch,
                        "cutoff": cutoff,
                        "reason": "cutoff_at_or_after_last_reference_observation",
                    }
                )
                continue
            history = int((frame["cycle"] <= cutoff).sum())
            if history < window:
                exclusions.append(
                    {
                        "cell_id": cell_id,
                        "batch": batch,
                        "cutoff": cutoff,
                        "reason": "insufficient_causal_history",
                    }
                )
                continue
            if interval is not None and cutoff >= interval[1]:
                exclusions.append(
                    {
                        "cell_id": cell_id,
                        "batch": batch,
                        "cutoff": cutoff,
                        "reason": "cutoff_at_or_after_observed_eol_upper_bound",
                    }
                )
                continue

            for model_id, seed_models in models.items():
                seed_forecasts = {}
                for seed, (model, scaler) in seed_models.items():
                    forecast, seed_eol = recursive_forecast(
                        model_id,
                        frame,
                        cutoff,
                        horizon,
                        threshold,
                        window,
                        model=model,
                        scaler=scaler,
                        device=device,
                    )
                    seed_forecasts[seed] = forecast
                    seed_metric = evaluate_forecast(
                        frame,
                        forecast,
                        cutoff,
                        seed_eol,
                        threshold,
                        horizon,
                    )
                    seed_records.append(
                        {
                            "batch": batch,
                            "cell_id": cell_id,
                            "cutoff": cutoff,
                            "model": model_id,
                            "seed": seed,
                            **seed_metric,
                        }
                    )
                ensemble = merge_seed_forecasts(seed_forecasts)
                ensemble_eol = predicted_eol(ensemble, threshold)
                metrics = evaluate_forecast(
                    frame,
                    ensemble,
                    cutoff,
                    ensemble_eol,
                    threshold,
                    horizon,
                )
                record = {
                    "batch": batch,
                    "cell_id": cell_id,
                    "cutoff": cutoff,
                    "model": model_id,
                    "aggregation": "mean_capacity_trajectory_across_seeds",
                    **metrics,
                }
                if model_id == primary_id:
                    for key, q in quantiles.items():
                        suffix = key.replace("coverage_", "cov_")
                        uq = uncertainty_metrics(
                            frame, ensemble, cutoff, horizon, threshold, q
                        )
                        for metric_name, value in uq.items():
                            record[f"{suffix}_{metric_name}"] = value
                        ensemble[f"{suffix}_lower"] = ensemble["predicted_capacity"] - q
                        ensemble[f"{suffix}_upper"] = ensemble["predicted_capacity"] + q
                ensemble_records.append(record)
                ensemble.insert(0, "model", model_id)
                ensemble.insert(0, "cutoff", cutoff)
                ensemble.insert(0, "cell_id", cell_id)
                ensemble.insert(0, "batch", batch)
                ensemble.to_csv(
                    trajectories / f"{cell_id}_c{cutoff}_{model_id}.csv",
                    index=False,
                )

            for baseline in ("local_linear_trend", "persistence"):
                forecast, baseline_eol = recursive_forecast(
                    baseline,
                    frame,
                    cutoff,
                    horizon,
                    threshold,
                    window,
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
                        "batch": batch,
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
                forecast.insert(0, "batch", batch)
                forecast.to_csv(
                    trajectories / f"{cell_id}_c{cutoff}_{baseline}.csv",
                    index=False,
                )

    records = pd.DataFrame(ensemble_records)
    seed_frame = pd.DataFrame(seed_records)
    exclusion_frame = pd.DataFrame(exclusions)
    records.to_csv(output / "ensemble_cell_records.csv", index=False)
    seed_frame.to_csv(output / "seed_level_records.csv", index=False)
    exclusion_frame.to_csv(output / "exclusions.csv", index=False)
    expected_models = {
        "mstt_single_scale_K5",
        "mstt_full_K5",
        "mstt_full_K1",
        "local_linear_trend",
        "persistence",
    }
    if set(records["model"]) != expected_models:
        raise RuntimeError("External evaluation model set is incomplete")
    if records.duplicated(["batch", "cell_id", "cutoff", "model"]).any():
        raise RuntimeError("Duplicate ensemble result keys")
    audit = {
        "status": "PASS",
        "config_sha256": config_hash,
        "external_data_sha256": external_data_hash,
        "external_data_loaded_after_freeze_receipt": True,
        "physical_cells": int(records["cell_id"].nunique()),
        "ensemble_records": len(records),
        "seed_records": len(seed_frame),
        "exclusions": len(exclusion_frame),
        "models": sorted(expected_models),
        "batches": sorted(records["batch"].unique().astype(int).tolist()),
        "interval_observed_records": int(records["truth_event_observed"].eq(1).sum()),
        "right_censored_records": int(records["truth_event_observed"].eq(0).sum()),
        "right_censored_cells_retained": True,
        "test_selection_or_retraining": False,
    }
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[PASS] external evaluation: records={len(records)}, "
        f"seed_records={len(seed_frame)}, exclusions={len(exclusion_frame)}"
    )


if __name__ == "__main__":
    main()
