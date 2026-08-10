from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd
import yaml

try:
    from ._bootstrap import add_project_root
except ImportError:
    from _bootstrap import add_project_root
add_project_root()

from src.utils.config import load_config
from experiments.exp_36_trail_dgr_prefix_replay_v3 import (
    CANDIDATES,
    checkpoint_for,
    load_domain,
    load_reference_tables,
    recursive_predict_frozen,
    reference_row,
    runtime_from_checkpoint,
)
from experiments.trail_recursive_tools import local_recursive, score_future


def main():
    project = Path(__file__).resolve().parents[1]
    base_nasa = load_config(project / "config_trail.yaml")
    protocol_path = project / "config_trail_crossdomain.yaml"
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    base_cross = load_config(project / protocol.get("base_config", "config_trail.yaml"))

    nasa_ref_path, nasa_ref, cross_ref_path, cross_ref = load_reference_tables(project)

    domain = "NASA"
    seed = 42
    test = "B0005"
    cutoffs = [50, 70, 90]

    cfg_domain, frames, cells_all = load_domain(
        project, domain, base_nasa, base_cross, protocol
    )
    cfg = copy.deepcopy(cfg_domain)
    cfg["training"]["seed"] = seed
    train = [b for b in cells_all if b != test]

    fr = frames[test].sort_values("cycle_index").reset_index(drop=True)
    true = fr["capacity"].to_numpy(float)
    thr = float(cfg["dataset"]["eol_threshold"][test])

    rows = []

    for cutoff in cutoffs:
        p = local_recursive(true[:cutoff], len(true), cfg)
        met = score_future(true, p, cutoff, thr)
        rr = reference_row(
            domain, nasa_ref, cross_ref,
            seed, test, cutoff, "Local_linear_anchor"
        )
        rows.append({
            "model": "Local_linear_anchor",
            "cutoff": cutoff,
            "generated_RMSE": met["future_RMSE"],
            "frozen_RMSE": float(rr["future_RMSE"]),
            "abs_delta_RMSE": abs(met["future_RMSE"] - float(rr["future_RMSE"])),
            "generated_MAE": met["future_MAE"],
            "frozen_MAE": float(rr["future_MAE"]),
            "abs_delta_MAE": abs(met["future_MAE"] - float(rr["future_MAE"])),
            "model_training_flag": False,
        })

    for out_name, backend in CANDIDATES:
        ck = checkpoint_for(project, domain, out_name, test, seed)
        trainer, scaler, nparam = runtime_from_checkpoint(
            cfg, frames, train, test, backend, ck
        )
        print(
            f"[MODE] {out_name}: trainer.model.training={trainer.model.training} "
            f"checkpoint={ck.name}"
        )

        for cutoff in cutoffs:
            p = recursive_predict_frozen(
                trainer, scaler, true[:cutoff], len(true), cfg
            )
            met = score_future(true, p, cutoff, thr)
            rr = reference_row(
                domain, nasa_ref, cross_ref,
                seed, test, cutoff, out_name
            )
            rows.append({
                "model": out_name,
                "cutoff": cutoff,
                "generated_RMSE": met["future_RMSE"],
                "frozen_RMSE": float(rr["future_RMSE"]),
                "abs_delta_RMSE": abs(met["future_RMSE"] - float(rr["future_RMSE"])),
                "generated_MAE": met["future_MAE"],
                "frozen_MAE": float(rr["future_MAE"]),
                "abs_delta_MAE": abs(met["future_MAE"] - float(rr["future_MAE"])),
                "model_training_flag": bool(trainer.model.training),
            })

    out = pd.DataFrame(rows)
    print("\n========== EXP36-v3 EVAL-MODE REPRO PROBE ==========")
    print(out.to_string(index=False))

    tol = 1e-6
    max_delta = max(
        float(out["abs_delta_RMSE"].max()),
        float(out["abs_delta_MAE"].max()),
    )
    print("\nmax_abs_delta_Ah =", max_delta)
    print("tolerance_Ah =", tol)

    if max_delta <= tol:
        print("PROBE_STATUS = PASS_EXACT_REPRO")
    else:
        print("PROBE_STATUS = FAIL_STILL_NOT_REPRODUCED")
        raise SystemExit(3)


if __name__ == "__main__":
    main()
