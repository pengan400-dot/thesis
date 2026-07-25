#!/usr/bin/env python3
"""
Batch-4/5/6 confirmatory calibration:
resume-safe two-worker parallel runner for one GPU.

This changes execution scheduling only. It preserves:
- protocol/config hash
- development-data hash
- four deterministic outer folds
- nested epoch selection
- seeds, batch size, optimizer, patience, epochs
- model architecture and recursive scoring
- held-out-cell exclusion from epoch selection

Recommended on one V100 32GB:
  --gpu-ids 0 --workers-per-gpu 2 --num-workers 0
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch


@dataclass(frozen=True)
class Job:
    fold: int
    seed: int

    @property
    def label(self) -> str:
        return f"fold={self.fold} seed={self.seed}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--prepared-dir", required=True)
    p.add_argument("--frozen-model-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--gpu-ids", default="0")
    p.add_argument("--workers-per-gpu", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--keep-going", action="store_true")
    p.add_argument("--single-job-json", default=None, help=argparse.SUPPRESS)
    p.add_argument("--physical-gpu", default=None, help=argparse.SUPPRESS)
    return p.parse_args()


def install_project(project: Path) -> None:
    src = str((project / "src").resolve())
    if src not in sys.path:
        sys.path.insert(0, src)


def load_context(args: argparse.Namespace):
    project = Path(args.project).resolve()
    install_project(project)

    from common import (
        curves_sha256,
        json_sha256,
        load_prepared_development,
        load_protocol,
    )
    from crossfit_calibration import fold_map, validate_frozen_primary

    config_path = Path(args.config).resolve()
    prepared_dir = Path(args.prepared_dir).resolve()
    frozen_root = Path(args.frozen_model_root).resolve()
    output = Path(args.output_dir).resolve()

    protocol = load_protocol(config_path)
    config_hash = json_sha256(protocol)
    curves = load_prepared_development(prepared_dir)
    development_hash = curves_sha256(curves)
    assignments = fold_map(curves)

    uncertainty = protocol["uncertainty"]
    training = protocol["training"]
    primary_id = str(uncertainty["model"])
    spec = next(
        item for item in protocol["models"]
        if str(item["id"]) == primary_id
    )
    seeds = [int(v) for v in training["seeds"]]

    validate_frozen_primary(
        frozen_root,
        primary_id,
        seeds,
        config_hash,
        development_hash,
    )

    return {
        "project": project,
        "config_path": config_path,
        "prepared_dir": prepared_dir,
        "frozen_root": frozen_root,
        "output": output,
        "protocol": protocol,
        "config_hash": config_hash,
        "curves": curves,
        "development_hash": development_hash,
        "assignments": assignments,
        "primary_id": primary_id,
        "spec": spec,
        "seeds": seeds,
    }


def job_dir(output: Path, job: Job) -> Path:
    return output / "crossfit_models" / f"fold_{job.fold}" / f"seed_{job.seed}"


def valid_job(
    output: Path,
    job: Job,
    config_hash: str,
    development_hash: str,
) -> bool:
    from common import sha256_file

    folder = job_dir(output, job)
    audit_path = folder / "audit.json"
    model_path = folder / "model.pt"
    scaler_path = folder / "scaler.joblib"

    if not (audit_path.is_file() and model_path.is_file() and scaler_path.is_file()):
        return False
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except Exception:
        return False

    return (
        audit.get("status") == "PASS"
        and int(audit.get("fold", -1)) == job.fold
        and int(audit.get("seed", -1)) == job.seed
        and audit.get("config_sha256") == config_hash
        and audit.get("development_data_sha256") == development_hash
        and audit.get("model_sha256") == sha256_file(model_path)
        and audit.get("scaler_sha256") == sha256_file(scaler_path)
        and audit.get("outer_held_out_cell_used_for_epoch_selection") is False
        and audit.get("external_data_loaded") is False
    )


def run_single_job(args: argparse.Namespace) -> int:
    payload = json.loads(args.single_job_json)
    job = Job(fold=int(payload["fold"]), seed=int(payload["seed"]))
    ctx = load_context(args)

    install_project(ctx["project"])
    from common import (
        FEATURE_NAMES,
        build_window_sets,
        choose_device,
        sha256_file,
    )
    from model_variants import identity_record
    from train_freeze_models import train_final, train_selection

    protocol = ctx["protocol"]
    curves = ctx["curves"]
    assignments = ctx["assignments"]
    spec = ctx["spec"]
    output = ctx["output"]
    output.mkdir(parents=True, exist_ok=True)

    if valid_job(
        output,
        job,
        ctx["config_hash"],
        ctx["development_hash"],
    ):
        print(f"[SKIP] {job.label}", flush=True)
        return 0

    training = protocol["training"]
    task = protocol["task"]

    held_out = sorted(
        cell for cell, fold in assignments.items() if int(fold) == job.fold
    )
    fit_cells = sorted(set(curves) - set(held_out))
    selection_train_cells = sorted(
        cell for cell in fit_cells
        if int(curves[cell]["batch"].iloc[0])
        == int(training["selection_train_batch"])
    )
    selection_validation_cells = sorted(
        cell for cell in fit_cells
        if int(curves[cell]["batch"].iloc[0])
        == int(training["selection_validation_batch"])
    )

    window = int(task["input_window_cycles"])
    threshold = float(task["eol_threshold_Ah"])
    support_horizon = int(spec["common_target_support_horizon"])
    steps = int(spec["loss_rollout_steps"])

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

    batch_size = int(training["batch_size"])
    learning_rate = float(training["learning_rate"])
    weight_decay = float(training["weight_decay"])
    max_epochs = int(training["max_epochs"])
    patience = int(training["patience"])

    device = choose_device("cuda")
    torch.set_num_threads(max(1, min(2, os.cpu_count() or 1)))

    folder = job_dir(output, job)
    folder.mkdir(parents=True, exist_ok=True)
    audit_path = folder / "audit.json"
    model_path = folder / "model.pt"
    scaler_path = folder / "scaler.joblib"

    # Audit is written last, so interrupted work is never accepted as complete.
    if audit_path.exists():
        audit_path.unlink()

    print(
        f"[START] {job.label} held_out={held_out} physical_gpu={args.physical_gpu}",
        flush=True,
    )
    started = time.time()

    selected_epoch, selection_history = train_selection(
        selection_windows,
        str(spec["ablation"]),
        steps,
        job.seed,
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
        job.seed,
        selected_epoch,
        batch_size,
        learning_rate,
        weight_decay,
        threshold,
        device,
        args.num_workers,
    )

    selection_history.to_csv(folder / "epoch_selection.csv", index=False)
    history.to_csv(folder / "training.csv", index=False)
    joblib.dump(final_windows.scaler, scaler_path)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_id": ctx["primary_id"],
            "fold": job.fold,
            "seed": job.seed,
            "fit_cells": fit_cells,
            "held_out_cells": held_out,
            "selection_train_cells": selection_train_cells,
            "selection_validation_cells": selection_validation_cells,
            "selected_epoch": selected_epoch,
            "config_sha256": ctx["config_hash"],
            "development_data_sha256": ctx["development_hash"],
        },
        model_path,
    )

    elapsed = time.time() - started
    audit = {
        "status": "PASS",
        "model_id": ctx["primary_id"],
        "fold": job.fold,
        "seed": job.seed,
        "fit_cells": fit_cells,
        "held_out_cells": held_out,
        "selection_train_cells": selection_train_cells,
        "selection_validation_cells": selection_validation_cells,
        "selected_epoch": selected_epoch,
        "runtime_seconds": elapsed,
        "outer_held_out_cell_used_for_epoch_selection": False,
        "external_data_loaded": False,
        "config_sha256": ctx["config_hash"],
        "development_data_sha256": ctx["development_hash"],
        "model_sha256": sha256_file(model_path),
        "scaler_sha256": sha256_file(scaler_path),
        "identity": identity_record(model),
        "execution_mode": "parallel_independent_fold_seed_jobs",
    }
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"[PASS] {job.label} epoch={selected_epoch} runtime={elapsed/60:.1f} min",
        flush=True,
    )
    return 0


def aggregate(ctx: dict, workers: int) -> None:
    install_project(ctx["project"])
    from common import (
        FEATURE_NAMES,
        choose_device,
        conformal_order_statistic,
        eol_interval,
        recursive_forecast,
    )
    from crossfit_calibration import observed_error_support
    from model_variants import PaperMSTTVariant

    protocol = ctx["protocol"]
    curves = ctx["curves"]
    assignments = ctx["assignments"]
    output = ctx["output"]
    spec = ctx["spec"]
    seeds = ctx["seeds"]

    jobs = [Job(fold=f, seed=s) for f in range(4) for s in seeds]
    missing = [
        job.label for job in jobs
        if not valid_job(
            output,
            job,
            ctx["config_hash"],
            ctx["development_hash"],
        )
    ]
    if missing:
        raise RuntimeError(f"Cannot aggregate; incomplete jobs: {missing[:5]}")

    device = choose_device("cuda")
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

    task = protocol["task"]
    uncertainty = protocol["uncertainty"]
    threshold = float(task["eol_threshold_Ah"])
    window = int(task["input_window_cycles"])
    horizon = int(task["forecast_horizon_cycles"])
    cutoffs = [int(v) for v in task["cutoffs"]]
    coverages = [
        float(uncertainty["primary_coverage"]),
        float(uncertainty["secondary_coverage"]),
    ]

    all_prediction_rows = []
    score_rows = []

    for fold in range(4):
        held_out = sorted(
            cell for cell, value in assignments.items() if int(value) == fold
        )
        trained = {}
        for seed in seeds:
            folder = job_dir(output, Job(fold, seed))
            model_path = folder / "model.pt"
            scaler_path = folder / "scaler.joblib"

            model = PaperMSTTVariant(
                len(FEATURE_NAMES),
                ablation=str(spec["ablation"]),
            ).to(device)
            checkpoint = torch.load(
                model_path,
                map_location=device,
                weights_only=False,
            )
            if (
                checkpoint.get("config_sha256") != ctx["config_hash"]
                or checkpoint.get("development_data_sha256")
                != ctx["development_hash"]
            ):
                raise RuntimeError(f"Checkpoint gate failed: {model_path}")
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            model.eval()
            trained[seed] = (model, joblib.load(scaler_path))

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
                        ctx["primary_id"],
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
                    col for col in ensemble.columns if col.startswith("seed_")
                ]
                ensemble["predicted_capacity"] = (
                    ensemble[seed_columns].mean(axis=1)
                )
                truth = observed_error_support(
                    frame,
                    cutoff,
                    horizon,
                    threshold,
                )
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
                    "max_abs_recursive_capacity_error_Ah": float(
                        np.max(cell_errors)
                    ),
                    "n_scored_points": len(cell_errors),
                }
            )

    scores = pd.DataFrame(score_rows).sort_values("cell_id")
    if len(scores) != 16:
        raise RuntimeError(f"Expected 16 cell scores, got {len(scores)}")

    scores.to_csv(output / "calibration_cell_scores.csv", index=False)
    if all_prediction_rows:
        pd.concat(all_prediction_rows, ignore_index=True).to_csv(
            output / "calibration_predictions.csv",
            index=False,
        )

    quantiles = {
        f"coverage_{coverage:.2f}": conformal_order_statistic(
            scores["max_abs_recursive_capacity_error_Ah"],
            coverage,
        )
        for coverage in coverages
    }
    audit = {
        "status": "PASS",
        "model_id": ctx["primary_id"],
        "fold_assignment": assignments,
        "n_physical_cells": len(scores),
        "score_definition": uncertainty["score"],
        "nested_epoch_selection": True,
        "outer_held_out_cells_used_for_epoch_selection": False,
        "coverage_quantiles_Ah": quantiles,
        "config_sha256": ctx["config_hash"],
        "development_data_sha256": ctx["development_hash"],
        "external_data_loaded": False,
        "quick_test": False,
        "execution_mode": "parallel_independent_fold_seed_jobs",
        "parallel_workers": workers,
        "completed_jobs": 12,
        "expected_jobs": 12,
    }
    (output / "calibration_quantiles.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[PASS] calibration complete: n={len(scores)}", flush=True)
    print(json.dumps(quantiles, ensure_ascii=False), flush=True)


def supervisor(args: argparse.Namespace) -> int:
    ctx = load_context(args)
    output = ctx["output"]
    output.mkdir(parents=True, exist_ok=True)

    jobs = [Job(fold=f, seed=s) for f in range(4) for s in ctx["seeds"]]
    complete = [
        job for job in jobs
        if valid_job(
            output,
            job,
            ctx["config_hash"],
            ctx["development_hash"],
        )
    ]
    missing = [job for job in jobs if job not in complete]

    print(
        f"[SCAN] complete={len(complete)}/{len(jobs)} missing={len(missing)}",
        flush=True,
    )

    if not missing:
        aggregate(ctx, args.workers_per_gpu)
        return 0

    gpu_ids = [v.strip() for v in args.gpu_ids.split(",") if v.strip()]
    if not gpu_ids:
        raise ValueError("--gpu-ids cannot be empty")
    if args.workers_per_gpu < 1:
        raise ValueError("--workers-per-gpu must be >= 1")

    slots = [
        gpu
        for gpu in gpu_ids
        for _ in range(args.workers_per_gpu)
    ]
    print(
        f"[PARALLEL] gpu_ids={gpu_ids} "
        f"workers_per_gpu={args.workers_per_gpu} "
        f"total_workers={len(slots)}",
        flush=True,
    )

    script_path = Path(__file__).resolve()

    def launch(pair: tuple[int, Job]) -> tuple[Job, int]:
        index, job = pair
        physical_gpu = slots[index % len(slots)]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = physical_gpu
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("OMP_NUM_THREADS", "2")
        env.setdefault("MKL_NUM_THREADS", "2")
        env.setdefault("OPENBLAS_NUM_THREADS", "2")
        env.setdefault("NUMEXPR_NUM_THREADS", "2")

        payload = json.dumps({"fold": job.fold, "seed": job.seed})
        command = [
            sys.executable,
            str(script_path),
            "--project", str(ctx["project"]),
            "--config", str(ctx["config_path"]),
            "--prepared-dir", str(ctx["prepared_dir"]),
            "--frozen-model-root", str(ctx["frozen_root"]),
            "--output-dir", str(output),
            "--num-workers", str(args.num_workers),
            "--single-job-json", payload,
            "--physical-gpu", physical_gpu,
        ]
        result = subprocess.run(
            command,
            cwd=ctx["project"],
            env=env,
            check=False,
        )
        return job, result.returncode

    failures = []
    with cf.ThreadPoolExecutor(max_workers=len(slots)) as executor:
        futures = {
            executor.submit(launch, pair): pair[1]
            for pair in enumerate(missing)
        }
        for future in cf.as_completed(futures):
            job = futures[future]
            try:
                _, rc = future.result()
            except Exception as exc:
                failures.append(f"{job.label}: {exc!r}")
                if not args.keep_going:
                    continue
            else:
                if rc != 0:
                    failures.append(f"{job.label}: exit={rc}")

    if failures:
        print("[FAILURES]", flush=True)
        for item in failures:
            print(item, flush=True)
        raise RuntimeError(f"{len(failures)} calibration jobs failed")

    aggregate(ctx, args.workers_per_gpu)
    return 0


def main() -> int:
    args = parse_args()
    if args.single_job_json:
        return run_single_job(args)
    return supervisor(args)


if __name__ == "__main__":
    raise SystemExit(main())
