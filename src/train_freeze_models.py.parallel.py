#!/usr/bin/env python3
"""Parallel, resumable runner for the preregistered Batch-4/5/6 freeze training.

This runner preserves the frozen protocol. It does not change batch size, AMP,
model definitions, target support, seeds, optimizer, or stopping rules. It only
runs independent model/seed jobs concurrently and rebuilds the official
frozen_model_manifest.json after every job passes its audit.

Recommended for one V100-32GB:
    --workers 2
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
import torch


@dataclass(frozen=True)
class Job:
    model_id: str
    seed: int

    @property
    def label(self) -> str:
        return f"{self.model_id} seed={self.seed}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--prepared-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--single-job", default=None, help=argparse.SUPPRESS)
    return p.parse_args()


def load_modules(project: Path):
    src = project.resolve() / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    import train_freeze_models as tfm  # type: ignore
    from common import (  # type: ignore
        assert_identical_support,
        build_window_sets,
        choose_device,
        curves_sha256,
        json_sha256,
        load_prepared_development,
        load_protocol,
        sha256_file,
        support_frame,
    )
    from model_variants import identity_record  # type: ignore

    return {
        "tfm": tfm,
        "assert_identical_support": assert_identical_support,
        "build_window_sets": build_window_sets,
        "choose_device": choose_device,
        "curves_sha256": curves_sha256,
        "json_sha256": json_sha256,
        "load_prepared_development": load_prepared_development,
        "load_protocol": load_protocol,
        "sha256_file": sha256_file,
        "support_frame": support_frame,
        "identity_record": identity_record,
    }


def context(args: argparse.Namespace):
    mods = load_modules(args.project)
    protocol = mods["load_protocol"](args.config)
    training = protocol["training"]
    task = protocol["task"]
    curves = mods["load_prepared_development"](args.prepared_dir)

    train_cells = sorted(
        cell
        for cell, frame in curves.items()
        if int(frame["batch"].iloc[0]) == int(training["selection_train_batch"])
    )
    validation_cells = sorted(
        cell
        for cell, frame in curves.items()
        if int(frame["batch"].iloc[0])
        == int(training["selection_validation_batch"])
    )
    development_cells = sorted(curves)
    model_specs = [
        spec for spec in protocol["models"] if str(spec["id"]).startswith("mstt_")
    ]
    seeds = [int(x) for x in training["seeds"]]
    window = int(task["input_window_cycles"])
    support_horizon = max(
        int(spec["common_target_support_horizon"]) for spec in model_specs
    )
    threshold = float(task["eol_threshold_Ah"])

    selection = mods["build_window_sets"](
        curves,
        train_cells,
        validation_cells,
        window,
        support_horizon,
        threshold,
    )
    final = mods["build_window_sets"](
        curves,
        development_cells,
        None,
        window,
        support_horizon,
        threshold,
    )
    supports = {
        str(spec["id"]): mods["support_frame"](selection.train, str(spec["id"]))
        for spec in model_specs
    }
    support_hash = mods["assert_identical_support"](supports)

    return {
        "mods": mods,
        "protocol": protocol,
        "training": training,
        "task": task,
        "curves": curves,
        "train_cells": train_cells,
        "validation_cells": validation_cells,
        "development_cells": development_cells,
        "model_specs": model_specs,
        "seeds": seeds,
        "support_horizon": support_horizon,
        "threshold": threshold,
        "selection": selection,
        "final": final,
        "supports": supports,
        "support_hash": support_hash,
        "config_hash": mods["json_sha256"](protocol),
        "development_data_hash": mods["curves_sha256"](curves),
    }


def job_dir(output: Path, job: Job) -> Path:
    return output / "models" / job.model_id / f"seed_{job.seed}"


def valid_job(output: Path, job: Job, ctx: dict) -> bool:
    directory = job_dir(output, job)
    audit_path = directory / "job_audit.json"
    model_path = directory / "model.pt"
    scaler_path = directory / "scaler.joblib"
    if not (audit_path.is_file() and model_path.is_file() and scaler_path.is_file()):
        return False
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    mods = ctx["mods"]
    return bool(
        audit.get("status") == "PASS"
        and audit.get("model_id") == job.model_id
        and int(audit.get("seed", -1)) == job.seed
        and audit.get("config_sha256") == ctx["config_hash"]
        and audit.get("support_sha256") == ctx["support_hash"]
        and audit.get("development_data_sha256") == ctx["development_data_hash"]
        and audit.get("model_sha256") == mods["sha256_file"](model_path)
        and audit.get("scaler_sha256") == mods["sha256_file"](scaler_path)
    )


def write_support(output: Path, ctx: dict) -> None:
    support_dir = output / "support"
    support_dir.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(ctx["supports"].values(), ignore_index=True)
    combined.to_csv(support_dir / "training_target_support_all_arms.csv", index=False)
    audit = {
        "status": "PASS",
        "common_support_horizon": ctx["support_horizon"],
        "arms": list(ctx["supports"]),
        "rows_per_arm": {
            name: len(frame) for name, frame in ctx["supports"].items()
        },
        "support_sha256": ctx["support_hash"],
    }
    (support_dir / "common_support_gate.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run_one(args: argparse.Namespace, job: Job) -> int:
    ctx = context(args)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    spec = next(
        spec for spec in ctx["model_specs"] if str(spec["id"]) == job.model_id
    )
    training = ctx["training"]
    tfm = ctx["mods"]["tfm"]
    device = ctx["mods"]["choose_device"](args.device)
    torch.set_num_threads(max(1, min(2, os.cpu_count() or 1)))

    directory = job_dir(output, job)
    if valid_job(output, job, ctx):
        print(f"[SKIP] {job.label}", flush=True)
        return 0

    directory.mkdir(parents=True, exist_ok=True)
    # Remove only incomplete products from an interrupted run of this exact job.
    for name in (
        "epoch_selection.csv",
        "final_training.csv",
        "scaler.joblib",
        "model.pt",
        "job_audit.json",
    ):
        (directory / name).unlink(missing_ok=True)

    started = time.time()
    print(f"[START] {job.label}", flush=True)
    ablation = str(spec["ablation"])
    steps = int(spec["loss_rollout_steps"])
    selected_epoch, selection_history = tfm.train_selection(
        ctx["selection"],
        ablation,
        steps,
        job.seed,
        int(training["max_epochs"]),
        int(training["patience"]),
        int(training["batch_size"]),
        float(training["learning_rate"]),
        float(training["weight_decay"]),
        ctx["threshold"],
        device,
        args.num_workers,
    )
    model, final_history = tfm.train_final(
        ctx["final"],
        ablation,
        steps,
        job.seed,
        selected_epoch,
        int(training["batch_size"]),
        float(training["learning_rate"]),
        float(training["weight_decay"]),
        ctx["threshold"],
        device,
        args.num_workers,
    )

    selection_history.to_csv(directory / "epoch_selection.csv", index=False)
    final_history.to_csv(directory / "final_training.csv", index=False)
    scaler_path = directory / "scaler.joblib"
    model_path = directory / "model.pt"
    joblib.dump(ctx["final"].scaler, scaler_path)
    identity = ctx["mods"]["identity_record"](model)
    if ablation == "full" and int(identity["trainable_parameters"]) != 74405:
        raise RuntimeError(
            f"{job.model_id}: expected 74,405 parameters, got "
            f"{identity['trainable_parameters']}"
        )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "identity": identity,
            "model_id": job.model_id,
            "ablation": ablation,
            "loss_rollout_steps": steps,
            "common_target_support_horizon": ctx["support_horizon"],
            "seed": job.seed,
            "selected_epoch": selected_epoch,
            "config_sha256": ctx["config_hash"],
            "support_sha256": ctx["support_hash"],
            "development_data_sha256": ctx["development_data_hash"],
        },
        model_path,
    )
    audit = {
        "status": "PASS",
        "model_id": job.model_id,
        "seed": job.seed,
        "ablation": ablation,
        "loss_rollout_steps": steps,
        "common_target_support_horizon": ctx["support_horizon"],
        "selected_epoch": selected_epoch,
        "train_cells": ctx["train_cells"],
        "validation_cells": ctx["validation_cells"],
        "final_fit_cells": ctx["development_cells"],
        "external_data_loaded": False,
        "batch_size": int(training["batch_size"]),
        "amp": False,
        "device": str(device),
        "parameter_identity": identity,
        "config_sha256": ctx["config_hash"],
        "support_sha256": ctx["support_hash"],
        "development_data_sha256": ctx["development_data_hash"],
        "model_sha256": ctx["mods"]["sha256_file"](model_path),
        "scaler_sha256": ctx["mods"]["sha256_file"](scaler_path),
        "runtime_seconds": time.time() - started,
        "quick_test": False,
        "execution_mode": "parallel_independent_job",
    }
    (directory / "job_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not valid_job(output, job, ctx):
        raise RuntimeError(f"Post-write audit failed: {job.label}")
    print(
        f"[PASS] {job.label} epoch={selected_epoch} "
        f"runtime={audit['runtime_seconds']/60:.1f} min",
        flush=True,
    )
    return 0


def build_manifest(args: argparse.Namespace, ctx: dict, jobs: list[Job]) -> None:
    output = args.output_dir.resolve()
    rows = []
    failures = []
    for job in jobs:
        if not valid_job(output, job, ctx):
            failures.append(job.label)
            continue
        rows.append(
            json.loads(
                (job_dir(output, job) / "job_audit.json").read_text(
                    encoding="utf-8"
                )
            )
        )
    if failures:
        raise RuntimeError(f"Incomplete jobs: {failures}")
    manifest = {
        "status": "PASS",
        "config_sha256": ctx["config_hash"],
        "support_sha256": ctx["support_hash"],
        "development_data_sha256": ctx["development_data_hash"],
        "external_data_loaded": False,
        "quick_test": False,
        "execution_mode": "parallel_independent_jobs",
        "jobs": rows,
    }
    (output / "frozen_model_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[PASS] frozen models written to {output}", flush=True)


def supervisor(args: argparse.Namespace) -> int:
    if args.workers < 1 or args.workers > 3:
        raise ValueError("--workers must be 1, 2, or 3")
    ctx = context(args)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_support(output, ctx)
    jobs = [
        Job(str(spec["id"]), seed)
        for spec in ctx["model_specs"]
        for seed in ctx["seeds"]
    ]
    complete = [job for job in jobs if valid_job(output, job, ctx)]
    missing = [job for job in jobs if job not in complete]
    print(
        f"[SCAN] complete={len(complete)}/{len(jobs)} missing={len(missing)}",
        flush=True,
    )
    if not missing:
        build_manifest(args, ctx, jobs)
        return 0

    script = Path(__file__).resolve()

    def launch(job: Job) -> tuple[Job, int]:
        import subprocess

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"
        env["OMP_NUM_THREADS"] = "2"
        env["MKL_NUM_THREADS"] = "2"
        env["OPENBLAS_NUM_THREADS"] = "2"
        env["PYTHONUNBUFFERED"] = "1"
        command = [
            sys.executable,
            str(script),
            "--project",
            str(args.project),
            "--config",
            str(args.config),
            "--prepared-dir",
            str(args.prepared_dir),
            "--output-dir",
            str(args.output_dir),
            "--device",
            args.device,
            "--workers",
            "1",
            "--num-workers",
            str(args.num_workers),
            "--single-job",
            json.dumps({"model_id": job.model_id, "seed": job.seed}),
        ]
        result = subprocess.run(
            command,
            cwd=args.project,
            env=env,
            check=False,
        )
        return job, int(result.returncode)

    failures = []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_map = {pool.submit(launch, job): job for job in missing}
        for future in cf.as_completed(future_map):
            job = future_map[future]
            try:
                _, code = future.result()
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{job.label}: {exc!r}")
                continue
            if code != 0:
                failures.append(f"{job.label}: exit={code}")
    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}", flush=True)
        raise RuntimeError(f"{len(failures)} jobs failed")

    # Rebuild context once after all workers finish and perform final hash audit.
    final_ctx = context(args)
    write_support(output, final_ctx)
    build_manifest(args, final_ctx, jobs)
    return 0


def main() -> int:
    args = parse_args()
    args.project = args.project.resolve()
    args.config = args.config.resolve()
    args.prepared_dir = args.prepared_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.single_job:
        payload = json.loads(args.single_job)
        return run_one(args, Job(str(payload["model_id"]), int(payload["seed"])))
    return supervisor(args)


if __name__ == "__main__":
    raise SystemExit(main())
