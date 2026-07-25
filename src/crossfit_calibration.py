#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from common import (
    FEATURE_NAMES,
    build_window_sets,
    choose_device,
    conformal_order_statistic,
    curves_sha256,
    eol_interval,
    json_sha256,
    load_prepared_development,
    load_protocol,
    recursive_forecast,
    sha256_file,
)
from model_variants import identity_record
from train_freeze_models import train_final, train_selection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--frozen-model-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--quick-test", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def fold_map(curves: dict[str, pd.DataFrame]) -> dict[str, int]:
    assignment: dict[str, int] = {}
    for batch in (1, 3):
        cells = sorted(
            cell
            for cell, frame in curves.items()
            if int(frame["batch"].iloc[0]) == batch
        )
        for index, cell in enumerate(cells):
            assignment[cell] = index % 4
    return assignment


def validate_frozen_primary(
    frozen_root: Path,
    model_id: str,
    seeds: list[int],
    config_hash: str,
    development_data_hash: str,
) -> None:
    manifest_path = frozen_root / "frozen_model_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "PASS"
        or manifest.get("external_data_loaded") is not False
        or manifest.get("config_sha256") != config_hash
        or manifest.get("development_data_sha256") != development_data_hash
    ):
        raise RuntimeError(f"Frozen-model gate failed: {manifest_path}")
    for seed in seeds:
        path = frozen_root / "models" / model_id / f"seed_{seed}" / "job_audit.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("status") != "PASS"
            or payload.get("config_sha256") != config_hash
            or payload.get("development_data_sha256") != development_data_hash
        ):
            raise RuntimeError(f"Frozen-model audit is not PASS: {path}")


def observed_error_support(
    frame: pd.DataFrame, cutoff: int, horizon: int, threshold: float
) -> pd.DataFrame:
    truth = frame.copy()
    if "measurement_observed" not in truth:
        truth["measurement_observed"] = 1
    interval = eol_interval(truth, threshold)
    end_cycle = int(interval[1]) if interval is not None else int(truth["cycle"].max())
    return truth[
        truth["measurement_observed"].eq(1)
        & truth["cycle"].gt(cutoff)
        & truth["cycle"].le(min(cutoff + horizon, end_cycle))
    ][["cycle", "raw_capacity"]]


def main() -> None:
    args = parse_args()
    protocol = load_protocol(args.config)
    config_hash = json_sha256(protocol)
    uncertainty = protocol["uncertainty"]
    training = protocol["training"]
    task = protocol["task"]
    primary_id = str(uncertainty["model"])
    spec = next(item for item in protocol["models"] if item["id"] == primary_id)
    seeds = [int(value) for value in training["seeds"]]
    coverages = [
        float(uncertainty["primary_coverage"]),
        float(uncertainty["secondary_coverage"]),
    ]
    if args.quick_test:
        seeds = seeds[:1]

    curves = load_prepared_development(args.prepared_dir)
    development_data_hash = curves_sha256(curves)
    assignments = fold_map(curves)
    folds = [0] if args.quick_test else list(range(4))
    validate_frozen_primary(
        args.frozen_model_root,
        primary_id,
        seeds,
        config_hash,
        development_data_hash,
    )
    device = choose_device(args.device)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

    threshold = float(task["eol_threshold_Ah"])
    window = int(task["input_window_cycles"])
    support_horizon = int(spec["common_target_support_horizon"])
    steps = int(spec["loss_rollout_steps"])
    horizon = int(task["forecast_horizon_cycles"])
    cutoffs = [int(value) for value in task["cutoffs"]]
    batch_size = int(training["batch_size"])
    learning_rate = float(training["learning_rate"])
    weight_decay = float(training["weight_decay"])
    max_epochs = int(training["max_epochs"])
    patience = int(training["patience"])
    if args.quick_test:
        max_epochs, patience = 2, 1

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    all_prediction_rows = []
    score_rows = []

    for fold in folds:
        held_out = sorted(cell for cell, value in assignments.items() if value == fold)
        if args.quick_test:
            held_out = held_out[:1]
        fit_cells = sorted(set(curves) - set(held_out))
        selection_train_cells = sorted(
            cell
            for cell in fit_cells
            if int(curves[cell]["batch"].iloc[0])
            == int(training["selection_train_batch"])
        )
        selection_validation_cells = sorted(
            cell
            for cell in fit_cells
            if int(curves[cell]["batch"].iloc[0])
            == int(training["selection_validation_batch"])
        )
        selection_windows = build_window_sets(
            curves,
            selection_train_cells,
            selection_validation_cells,
            window,
            support_horizon,
            threshold,
        )
        final_windows = build_window_sets(
            curves,
            fit_cells,
            None,
            window,
            support_horizon,
            threshold,
        )
        trained: dict[int, tuple[torch.nn.Module, object]] = {}
        for seed in seeds:
            fold_dir = output / "crossfit_models" / f"fold_{fold}" / f"seed_{seed}"
            audit_path = fold_dir / "audit.json"
            model_path = fold_dir / "model.pt"
            scaler_path = fold_dir / "scaler.joblib"
            if (
                audit_path.is_file()
                and model_path.is_file()
                and scaler_path.is_file()
                and not args.overwrite
            ):
                audit = json.loads(audit_path.read_text(encoding="utf-8"))
                if (
                    audit.get("status") == "PASS"
                    and audit.get("config_sha256") == config_hash
                    and audit.get("development_data_sha256") == development_data_hash
                    and audit.get("model_sha256") == sha256_file(model_path)
                    and audit.get("scaler_sha256") == sha256_file(scaler_path)
                ):
                    from model_variants import PaperMSTTVariant

                    model = PaperMSTTVariant(
                        len(FEATURE_NAMES), ablation=str(spec["ablation"])
                    ).to(device)
                    checkpoint = torch.load(
                        model_path, map_location=device, weights_only=False
                    )
                    if (
                        checkpoint.get("config_sha256") != config_hash
                        or checkpoint.get("development_data_sha256")
                        != development_data_hash
                    ):
                        raise RuntimeError(
                            f"Cross-fit checkpoint gate failed: {model_path}"
                        )
                    model.load_state_dict(checkpoint["state_dict"], strict=True)
                    model.eval()
                    trained[seed] = (model, joblib.load(scaler_path))
                    print(f"[SKIP] crossfit fold={fold} seed={seed}")
                    continue
            fold_dir.mkdir(parents=True, exist_ok=True)
            selected_epoch, selection_history = train_selection(
                selection_windows,
                str(spec["ablation"]),
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
            )
            model, history = train_final(
                final_windows,
                str(spec["ablation"]),
                steps,
                seed,
                selected_epoch,
                batch_size,
                learning_rate,
                weight_decay,
                threshold,
                device,
                args.num_workers,
            )
            selection_history.to_csv(fold_dir / "epoch_selection.csv", index=False)
            history.to_csv(fold_dir / "training.csv", index=False)
            joblib.dump(final_windows.scaler, scaler_path)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_id": primary_id,
                    "fold": fold,
                    "seed": seed,
                    "fit_cells": fit_cells,
                    "held_out_cells": held_out,
                    "selection_train_cells": selection_train_cells,
                    "selection_validation_cells": selection_validation_cells,
                    "selected_epoch": selected_epoch,
                    "config_sha256": config_hash,
                    "development_data_sha256": development_data_hash,
                },
                model_path,
            )
            audit = {
                "status": "PASS",
                "model_id": primary_id,
                "fold": fold,
                "seed": seed,
                "fit_cells": fit_cells,
                "held_out_cells": held_out,
                "selection_train_cells": selection_train_cells,
                "selection_validation_cells": selection_validation_cells,
                "selected_epoch": selected_epoch,
                "outer_held_out_cell_used_for_epoch_selection": False,
                "external_data_loaded": False,
                "config_sha256": config_hash,
                "development_data_sha256": development_data_hash,
                "model_sha256": sha256_file(model_path),
                "scaler_sha256": sha256_file(scaler_path),
                "identity": identity_record(model),
            }
            audit_path.write_text(
                json.dumps(audit, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            trained[seed] = (model, final_windows.scaler)
            print(f"[PASS] crossfit fold={fold} seed={seed}")

        for cell in held_out:
            frame = curves[cell].copy()
            frame["measurement_observed"] = 1
            cell_errors = []
            interval = eol_interval(frame, threshold)
            for cutoff in cutoffs:
                if interval is not None and cutoff >= interval[1]:
                    continue
                if int((frame["cycle"] <= cutoff).sum()) < window:
                    continue
                seed_forecasts = []
                for seed, (model, scaler) in trained.items():
                    forecast, _ = recursive_forecast(
                        primary_id,
                        frame,
                        cutoff,
                        horizon,
                        threshold,
                        window,
                        model=model,
                        scaler=scaler,
                        device=device,
                    )
                    forecast = forecast.rename(
                        columns={"predicted_capacity": f"seed_{seed}"}
                    )
                    seed_forecasts.append(forecast)
                ensemble = seed_forecasts[0]
                for other in seed_forecasts[1:]:
                    ensemble = ensemble.merge(other, on="cycle", how="inner")
                seed_columns = [
                    column for column in ensemble if column.startswith("seed_")
                ]
                ensemble["predicted_capacity"] = ensemble[seed_columns].mean(axis=1)
                truth = observed_error_support(frame, cutoff, horizon, threshold)
                aligned = truth.merge(
                    ensemble[["cycle", "predicted_capacity"]],
                    on="cycle",
                    how="inner",
                )
                errors = np.abs(
                    aligned["predicted_capacity"].to_numpy(float)
                    - aligned["raw_capacity"].to_numpy(float)
                )
                if not len(errors):
                    continue
                cell_errors.extend(errors.tolist())
                aligned.insert(0, "cutoff", cutoff)
                aligned.insert(0, "fold", fold)
                aligned.insert(0, "cell_id", cell)
                all_prediction_rows.append(aligned)
            if not cell_errors:
                raise RuntimeError(f"{cell}: no cross-fit calibration support")
            score_rows.append(
                {
                    "cell_id": cell,
                    "fold": fold,
                    "max_abs_recursive_capacity_error_Ah": float(np.max(cell_errors)),
                    "n_scored_points": len(cell_errors),
                }
            )

    scores = pd.DataFrame(score_rows).sort_values("cell_id")
    if not args.quick_test and len(scores) != 16:
        raise RuntimeError(f"Expected 16 cell scores, got {len(scores)}")
    scores.to_csv(output / "calibration_cell_scores.csv", index=False)
    if all_prediction_rows:
        pd.concat(all_prediction_rows, ignore_index=True).to_csv(
            output / "calibration_predictions.csv", index=False
        )
    quantiles = {
        f"coverage_{coverage:.2f}": conformal_order_statistic(
            scores["max_abs_recursive_capacity_error_Ah"], coverage
        )
        for coverage in coverages
    }
    audit = {
        "status": "PASS",
        "model_id": primary_id,
        "fold_assignment": assignments,
        "n_physical_cells": len(scores),
        "score_definition": uncertainty["score"],
        "nested_epoch_selection": True,
        "outer_held_out_cells_used_for_epoch_selection": False,
        "coverage_quantiles_Ah": quantiles,
        "config_sha256": config_hash,
        "development_data_sha256": development_data_hash,
        "external_data_loaded": False,
        "quick_test": bool(args.quick_test),
    }
    (output / "calibration_quantiles.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[PASS] calibration complete: n={len(scores)}")
    print(json.dumps(quantiles, ensure_ascii=False))


if __name__ == "__main__":
    main()
