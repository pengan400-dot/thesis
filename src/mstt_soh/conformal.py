from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
import torch

from .model_variants import PaperMSTTVariant, identity_record
from .pipeline import (
    build_window_sets,
    choose_device,
    curves_sha256,
    eol_interval,
    evaluate_forecast,
    json_sha256,
    load_development_curves,
    load_protocol,
    merge_seed_forecasts,
    recursive_forecast,
    sha256_file,
    train_final,
    train_selection,
    utc_now,
    write_json,
)


def finite_sample_qhat(scores: np.ndarray, coverage: float) -> tuple[float, int]:
    values = np.asarray(scores, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Conformal scores must be a non-empty finite array")
    if not 0.0 < coverage < 1.0:
        raise ValueError("Coverage must lie in (0,1)")
    rank = min(len(values), math.ceil((len(values) + 1) * coverage))
    return float(np.sort(values)[rank - 1]), int(rank)


def stratified_folds(
    curves: dict[str, pd.DataFrame],
    folds: int,
) -> list[dict[str, list[str]]]:
    by_batch = {
        batch: sorted(
            cell_id
            for cell_id, frame in curves.items()
            if int(frame["batch"].iloc[0]) == batch
        )
        for batch in (1, 3)
    }
    if any(len(cells) % folds for cells in by_batch.values()):
        raise ValueError(
            f"Batch counts must be divisible by {folds}: "
            f"{ {key: len(value) for key, value in by_batch.items()} }"
        )
    result: list[dict[str, list[str]]] = []
    chunks = {
        batch: [
            list(chunk)
            for chunk in np.array_split(np.asarray(cells, dtype=object), folds)
        ]
        for batch, cells in by_batch.items()
    }
    for fold in range(folds):
        result.append(
            {
                "B1": [str(value) for value in chunks[1][fold]],
                "B3": [str(value) for value in chunks[3][fold]],
            }
        )
    return result


def model_options(protocol: dict) -> dict[str, object]:
    training = protocol["training"]
    return {
        "attention_heads": training["attention_heads"],
        "dropout": training["dropout"],
        "residual_scale_SOH": training["residual_scale_SOH"],
        "rope_base": training["rope_base"],
        "trend_delta_window": training["trend_delta_window"],
        "trend_delta_clip_SOH": protocol["ah_to_soh_conversion"][
            "trend_delta_clip_SOH"
        ],
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


def load_fold_model(
    model_path: Path,
    scaler_path: Path,
    protocol: dict,
    device: torch.device,
) -> tuple[PaperMSTTVariant, object]:
    options = model_options(protocol)
    model = PaperMSTTVariant(
        7,
        ablation="full",
        n_heads=int(options["attention_heads"]),
        dropout=float(options["dropout"]),
        residual_scale=float(options["residual_scale_SOH"]),
        rope_base=float(options["rope_base"]),
        trend_delta_window=int(options["trend_delta_window"]),
        trend_delta_clip_soh=tuple(
            float(value) for value in options["trend_delta_clip_SOH"]
        ),
    ).to(device)
    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    if identity_record(model) != checkpoint["identity"]:
        raise RuntimeError(f"Cross-fit model identity mismatch: {model_path}")
    model.eval()
    return model, joblib.load(scaler_path)


def run(args: argparse.Namespace) -> None:
    protocol = load_protocol(args.config)
    curves = load_development_curves(args.development, protocol)
    config_hash = json_sha256(protocol)
    development_hash = curves_sha256(curves)
    device = choose_device(args.device)
    training = protocol["training"]
    task = protocol["task"]
    conformal = protocol["conformal"]
    folds = stratified_folds(curves, int(conformal["outer_folds"]))
    seeds = [int(value) for value in training["seeds"]]
    max_epochs = int(training["max_epochs"])
    patience = int(training["patience"])
    forecast_horizon = int(task["forecast_horizon_cycles"])
    if args.quick_test:
        folds = folds[:1]
        seeds = seeds[:1]
        max_epochs = 2
        patience = 1
        forecast_horizon = 30
    window = int(task["input_window_cycles"])
    warmup = int(task["rolling_warmup_cycles"])
    support_horizon = 5
    threshold = float(task["eol_threshold_SOH"])
    options = model_options(protocol)
    all_cells = set(curves)
    forecasts: dict[
        tuple[int, str, int],
        dict[int, pd.DataFrame],
    ] = {}
    training_rows: list[dict[str, object]] = []
    args.output.mkdir(parents=True, exist_ok=True)

    for fold_index, held_out in enumerate(folds, start=1):
        held_out_cells = sorted(held_out["B1"] + held_out["B3"])
        inner_train = sorted(
            cell_id
            for cell_id, frame in curves.items()
            if int(frame["batch"].iloc[0]) == 1
            and cell_id not in held_out_cells
        )
        inner_validation = sorted(
            cell_id
            for cell_id, frame in curves.items()
            if int(frame["batch"].iloc[0]) == 3
            and cell_id not in held_out_cells
        )
        outer_fit = sorted(all_cells - set(held_out_cells))
        if set(held_out_cells) & set(outer_fit):
            raise AssertionError("Outer-held-out cells leaked into cross-fit")
        selection = build_window_sets(
            curves,
            inner_train,
            inner_validation,
            window,
            support_horizon,
            threshold,
            warmup,
        )
        final = build_window_sets(
            curves,
            outer_fit,
            None,
            window,
            support_horizon,
            threshold,
            warmup,
        )
        for seed in seeds:
            job = args.output / "fold_models" / f"fold_{fold_index}" / f"seed_{seed}"
            job.mkdir(parents=True, exist_ok=True)
            model_path = job / "model.pt"
            scaler_path = job / "scaler.joblib"
            audit_path = job / "job_audit.json"
            started = time.time()
            if model_path.is_file() and scaler_path.is_file() and audit_path.is_file():
                audit = json.loads(audit_path.read_text(encoding="utf-8"))
                if (
                    audit.get("status") != "PASS"
                    or audit.get("config_sha256") != config_hash
                    or audit.get("development_data_sha256") != development_hash
                    or audit.get("model_sha256") != sha256_file(model_path)
                    or audit.get("scaler_sha256") != sha256_file(scaler_path)
                ):
                    raise RuntimeError(
                        f"Inconsistent cross-fit resume state: {job}"
                    )
                selected_epoch = int(audit["selected_epoch"])
            else:
                selected_epoch, selection_history = train_selection(
                    selection,
                    "full",
                    5,
                    seed,
                    max_epochs,
                    patience,
                    int(training["batch_size"]),
                    float(training["learning_rate"]),
                    float(training["weight_decay"]),
                    threshold,
                    device,
                    args.num_workers,
                    options,
                )
                model, final_history = train_final(
                    final,
                    "full",
                    5,
                    seed,
                    selected_epoch,
                    int(training["batch_size"]),
                    float(training["learning_rate"]),
                    float(training["weight_decay"]),
                    threshold,
                    device,
                    args.num_workers,
                    options,
                )
                identity = identity_record(model)
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "identity": identity,
                        "fold": fold_index,
                        "seed": seed,
                        "held_out_cells": held_out_cells,
                        "selected_epoch": selected_epoch,
                        "config_sha256": config_hash,
                        "development_data_sha256": development_hash,
                    },
                    model_path,
                )
                joblib.dump(final.scaler, scaler_path)
                selection_history.to_csv(
                    job / "epoch_selection.csv",
                    index=False,
                )
                final_history.to_csv(
                    job / "final_training.csv",
                    index=False,
                )
                audit = {
                    "status": "PASS",
                    "fold": fold_index,
                    "seed": seed,
                    "held_out_cells": held_out_cells,
                    "inner_train_cells": inner_train,
                    "inner_validation_cells": inner_validation,
                    "outer_fit_cells": outer_fit,
                    "selected_epoch": selected_epoch,
                    "identity": identity,
                    "config_sha256": config_hash,
                    "development_data_sha256": development_hash,
                    "model_sha256": sha256_file(model_path),
                    "scaler_sha256": sha256_file(scaler_path),
                    "external_data_loaded": False,
                    "quick_test": bool(args.quick_test),
                    "runtime_seconds": time.time() - started,
                }
                write_json(audit_path, audit)
            training_rows.append(audit)
            model, scaler = load_fold_model(
                model_path,
                scaler_path,
                protocol,
                device,
            )
            for cell_id in held_out_cells:
                frame = curves[cell_id]
                truth = eol_interval(frame, threshold)
                last_observed = int(
                    frame.loc[
                        frame["measurement_observed"].eq(1),
                        "cycle",
                    ].max()
                )
                for cutoff in map(int, task["cutoffs"]):
                    if cutoff >= last_observed:
                        raise RuntimeError(
                            f"Development cell {cell_id} ends by cutoff {cutoff}"
                        )
                    if truth is not None and cutoff >= truth[1]:
                        raise RuntimeError(
                            f"Development cell {cell_id} reaches EOL by "
                            f"cutoff {cutoff}"
                        )
                    forecast, _ = recursive_forecast(
                        "mstt_full_K5",
                        frame,
                        cutoff,
                        forecast_horizon,
                        threshold,
                        window,
                        tuple(map(float, task["prediction_clip_SOH"])),
                        model=model,
                        scaler=scaler,
                        device=device,
                        minimum_history=int(
                            task["minimum_pre_cutoff_physical_cycles"]
                        ),
                    )
                    forecasts.setdefault(
                        (fold_index, cell_id, cutoff),
                        {},
                    )[seed] = forecast
            print(
                f"[CROSSFIT] fold={fold_index} seed={seed} "
                f"epoch={selected_epoch}"
            )

    scores: list[dict[str, object]] = []
    expected_seed_set = set(seeds)
    for (fold_index, cell_id, cutoff), seed_forecasts in sorted(
        forecasts.items()
    ):
        if set(seed_forecasts) != expected_seed_set:
            raise RuntimeError(
                f"Incomplete seed forecasts for {cell_id} cutoff {cutoff}"
            )
        ensemble = merge_seed_forecasts(seed_forecasts)
        metrics = evaluate_forecast(
            curves[cell_id],
            ensemble,
            cutoff,
            None,
            threshold,
            forecast_horizon,
            minimum_future_observations=int(
                task["minimum_future_observations_for_RMSE"]
            ),
        )
        scores.append(
            {
                "fold": fold_index,
                "cell_id": cell_id,
                "batch": int(curves[cell_id]["batch"].iloc[0]),
                "cutoff": cutoff,
                "seed_aggregation": "pointwise_mean_trajectory",
                "future_observed_points": metrics[
                    "future_observed_points"
                ],
                "simultaneous_max_abs_error_SOH": metrics[
                    "max_abs_capacity_error_SOH"
                ],
                "future_RMSE_SOH": metrics[
                    "future_capacity_RMSE_SOH"
                ],
            }
        )
    score_frame = pd.DataFrame(scores).sort_values(
        ["cutoff", "cell_id"]
    )
    score_frame.to_csv(args.output / "calibration_scores.csv", index=False)
    quantile_rows: list[dict[str, object]] = []
    for cutoff, group in score_frame.groupby("cutoff"):
        values = group["simultaneous_max_abs_error_SOH"].to_numpy(float)
        for coverage in map(float, conformal["nominal_coverages"]):
            half_width, rank = finite_sample_qhat(values, coverage)
            quantile_rows.append(
                {
                    "model": conformal["model"],
                    "cutoff": int(cutoff),
                    "nominal_simultaneous_coverage": coverage,
                    "calibration_cells": len(values),
                    "finite_sample_rank": rank,
                    "half_width_SOH": half_width,
                    "bit_updates_quantile": False,
                }
            )
    quantiles = pd.DataFrame(quantile_rows).sort_values(
        ["cutoff", "nominal_simultaneous_coverage"]
    )
    quantiles.to_csv(args.output / "conformal_quantiles.csv", index=False)
    expected_cells = len(curves) if not args.quick_test else 4
    per_cutoff = score_frame.groupby("cutoff")["cell_id"].nunique().to_dict()
    status = (
        "PASS"
        if all(int(value) == expected_cells for value in per_cutoff.values())
        and len(per_cutoff) == len(task["cutoffs"])
        else "FAIL"
    )
    manifest = {
        "status": status,
        "created_utc": utc_now(),
        "method": conformal["method"],
        "config_sha256": config_hash,
        "development_data_sha256": development_hash,
        "physical_calibration_cells": expected_cells,
        "cells_per_cutoff": {
            str(key): int(value) for key, value in per_cutoff.items()
        },
        "folds": folds,
        "seeds": seeds,
        "nonconformity_score": conformal["nonconformity_score"],
        "external_data_loaded": False,
        "hnei_data_loaded": False,
        "bit_data_loaded": False,
        "bit_updates_quantiles": False,
        "quick_test": bool(args.quick_test),
        "training_jobs": training_rows,
        "calibration_scores_sha256": sha256_file(
            args.output / "calibration_scores.csv"
        ),
        "conformal_quantiles_sha256": sha256_file(
            args.output / "conformal_quantiles.csv"
        ),
    }
    write_json(args.output / "calibration_manifest.json", manifest)
    if status != "PASS":
        raise RuntimeError(
            f"Cross-fitted calibration coverage gate failed: {per_cutoff}"
        )
    print(
        f"[PASS] cross-fitted conformal calibration: "
        f"cells={expected_cells}, cutoffs={sorted(per_cutoff)}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="XJTU-only cell-level cross-fitted SOH calibration"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--quick-test", action="store_true")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
