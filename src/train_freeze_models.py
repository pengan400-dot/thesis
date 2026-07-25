#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from common import (
    FEATURE_NAMES,
    RolloutDataset,
    assert_identical_support,
    build_window_sets,
    choose_device,
    curves_sha256,
    json_sha256,
    load_prepared_development,
    load_protocol,
    rollout_loss,
    set_seed,
    sha256_file,
    support_frame,
)
from model_variants import PaperMSTTVariant, identity_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--quick-test", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def make_loader(arrays, batch_size: int, shuffle: bool, workers: int, device):
    return DataLoader(
        RolloutDataset(arrays),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )


def autocast_disabled():
    return nullcontext()


def train_selection(
    windows,
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
) -> tuple[int, pd.DataFrame]:
    if windows.validation is None:
        raise ValueError("Validation windows are required")
    set_seed(seed)
    model = PaperMSTTVariant(len(FEATURE_NAMES), ablation=ablation).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    train_loader = make_loader(windows.train, batch_size, True, workers, device)
    validation_loader = make_loader(
        windows.validation, batch_size, False, workers, device
    )
    mean = torch.tensor(windows.scaler.mean_, dtype=torch.float32, device=device)
    scale = torch.tensor(windows.scaler.scale_, dtype=torch.float32, device=device)
    best_epoch, best_loss, stale = 1, float("inf"), 0
    rows = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_values = []
        for X, caps, cycles, q0, targets in train_loader:
            X, caps, cycles = X.to(device), caps.to(device), cycles.to(device)
            q0, targets = q0.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_disabled():
                loss = rollout_loss(
                    model,
                    X,
                    caps,
                    cycles,
                    q0,
                    targets,
                    steps,
                    mean,
                    scale,
                    threshold,
                    include_penalty=True,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_values.append(float(loss.detach().cpu()))
        model.eval()
        validation_values = []
        with torch.no_grad():
            for X, caps, cycles, q0, targets in validation_loader:
                loss = rollout_loss(
                    model,
                    X.to(device),
                    caps.to(device),
                    cycles.to(device),
                    q0.to(device),
                    targets.to(device),
                    steps,
                    mean,
                    scale,
                    threshold,
                    include_penalty=False,
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
            best_epoch, best_loss, stale = epoch, validation_loss, 0
        else:
            stale += 1
        if stale >= patience:
            break
    return best_epoch, pd.DataFrame(rows)


def train_final(
    windows,
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
) -> tuple[nn.Module, pd.DataFrame]:
    set_seed(seed)
    model = PaperMSTTVariant(len(FEATURE_NAMES), ablation=ablation).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    loader = make_loader(windows.train, batch_size, True, workers, device)
    mean = torch.tensor(windows.scaler.mean_, dtype=torch.float32, device=device)
    scale = torch.tensor(windows.scaler.scale_, dtype=torch.float32, device=device)
    rows = []
    for epoch in range(1, epochs + 1):
        model.train()
        values = []
        for X, caps, cycles, q0, targets in loader:
            X, caps, cycles = X.to(device), caps.to(device), cycles.to(device)
            q0, targets = q0.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = rollout_loss(
                model,
                X,
                caps,
                cycles,
                q0,
                targets,
                steps,
                mean,
                scale,
                threshold,
                include_penalty=True,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            values.append(float(loss.detach().cpu()))
        rows.append({"epoch": epoch, "train_loss": float(np.mean(values))})
    model.eval()
    return model, pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    protocol = load_protocol(args.config)
    config_hash = json_sha256(protocol)
    task = protocol["task"]
    training = protocol["training"]
    curves = load_prepared_development(args.prepared_dir)
    development_data_hash = curves_sha256(curves)
    train_cells = sorted(
        cell
        for cell, frame in curves.items()
        if int(frame["batch"].iloc[0]) == int(training["selection_train_batch"])
    )
    validation_cells = sorted(
        cell
        for cell, frame in curves.items()
        if int(frame["batch"].iloc[0]) == int(training["selection_validation_batch"])
    )
    development_cells = sorted(curves)
    model_specs = [
        spec for spec in protocol["models"] if str(spec["id"]).startswith("mstt_")
    ]
    seeds = [int(value) for value in training["seeds"]]
    max_epochs = int(training["max_epochs"])
    patience = int(training["patience"])
    if args.quick_test:
        train_cells = train_cells[:1]
        validation_cells = validation_cells[:1]
        development_cells = train_cells + validation_cells
        model_specs = model_specs[:1]
        seeds = seeds[:1]
        max_epochs, patience = 2, 1

    window = int(task["input_window_cycles"])
    support_horizon = max(
        int(spec["common_target_support_horizon"]) for spec in model_specs
    )
    threshold = float(task["eol_threshold_Ah"])
    selection = build_window_sets(
        curves,
        train_cells,
        validation_cells,
        window,
        support_horizon,
        threshold,
    )
    final = build_window_sets(
        curves,
        development_cells,
        None,
        window,
        support_horizon,
        threshold,
    )
    supports = {
        str(spec["id"]): support_frame(selection.train, str(spec["id"]))
        for spec in model_specs
    }
    support_hash = assert_identical_support(supports)

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    support_dir = output / "support"
    support_dir.mkdir(exist_ok=True)
    combined_support = pd.concat(supports.values(), ignore_index=True)
    combined_support.to_csv(
        support_dir / "training_target_support_all_arms.csv", index=False
    )
    support_audit = {
        "status": "PASS",
        "common_support_horizon": support_horizon,
        "arms": list(supports),
        "rows_per_arm": {name: len(frame) for name, frame in supports.items()},
        "support_sha256": support_hash,
    }
    (support_dir / "common_support_gate.json").write_text(
        json.dumps(support_audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    device = choose_device(args.device)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    batch_size = int(training["batch_size"])
    learning_rate = float(training["learning_rate"])
    weight_decay = float(training["weight_decay"])
    job_rows = []

    for spec in model_specs:
        model_id = str(spec["id"])
        ablation = str(spec["ablation"])
        steps = int(spec["loss_rollout_steps"])
        for seed in seeds:
            job_dir = output / "models" / model_id / f"seed_{seed}"
            audit_path = job_dir / "job_audit.json"
            scaler_path = job_dir / "scaler.joblib"
            model_path = job_dir / "model.pt"
            if (
                audit_path.is_file()
                and model_path.is_file()
                and scaler_path.is_file()
                and not args.overwrite
            ):
                old = json.loads(audit_path.read_text(encoding="utf-8"))
                if (
                    old.get("status") == "PASS"
                    and old.get("config_sha256") == config_hash
                    and old.get("support_sha256") == support_hash
                    and old.get("development_data_sha256") == development_data_hash
                    and old.get("model_sha256") == sha256_file(model_path)
                    and old.get("scaler_sha256") == sha256_file(scaler_path)
                ):
                    print(f"[SKIP] {model_id} seed={seed}")
                    job_rows.append(old)
                    continue
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
            )
            selection_history.to_csv(job_dir / "epoch_selection.csv", index=False)
            final_history.to_csv(job_dir / "final_training.csv", index=False)
            joblib.dump(final.scaler, scaler_path)
            identity = identity_record(model)
            if ablation == "full" and int(identity["trainable_parameters"]) != 74405:
                raise RuntimeError(
                    f"{model_id}: expected 74,405 parameters, got "
                    f"{identity['trainable_parameters']}"
                )
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "identity": identity,
                    "model_id": model_id,
                    "ablation": ablation,
                    "loss_rollout_steps": steps,
                    "common_target_support_horizon": support_horizon,
                    "seed": seed,
                    "selected_epoch": selected_epoch,
                    "config_sha256": config_hash,
                    "support_sha256": support_hash,
                    "development_data_sha256": development_data_hash,
                },
                model_path,
            )
            audit = {
                "status": "PASS",
                "model_id": model_id,
                "seed": seed,
                "ablation": ablation,
                "loss_rollout_steps": steps,
                "common_target_support_horizon": support_horizon,
                "selected_epoch": selected_epoch,
                "train_cells": train_cells,
                "validation_cells": validation_cells,
                "final_fit_cells": development_cells,
                "external_data_loaded": False,
                "batch_size": batch_size,
                "amp": False,
                "device": str(device),
                "parameter_identity": identity,
                "config_sha256": config_hash,
                "support_sha256": support_hash,
                "development_data_sha256": development_data_hash,
                "model_sha256": sha256_file(model_path),
                "scaler_sha256": sha256_file(scaler_path),
                "runtime_seconds": time.time() - started,
                "quick_test": bool(args.quick_test),
            }
            audit_path.write_text(
                json.dumps(audit, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            job_rows.append(audit)
            print(
                f"[PASS] {model_id} seed={seed} epoch={selected_epoch} "
                f"runtime={audit['runtime_seconds'] / 60:.1f} min"
            )

    manifest = {
        "status": "PASS",
        "config_sha256": config_hash,
        "support_sha256": support_hash,
        "development_data_sha256": development_data_hash,
        "external_data_loaded": False,
        "quick_test": bool(args.quick_test),
        "jobs": job_rows,
    }
    (output / "frozen_model_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[PASS] frozen models written to {output}")


if __name__ == "__main__":
    main()
