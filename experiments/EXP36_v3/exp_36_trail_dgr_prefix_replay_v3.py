from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

try:
    from ._bootstrap import add_project_root
except ImportError:
    from _bootstrap import add_project_root
add_project_root()

from src.data.crossdomain_curves import load_prepared_domain
from src.data.feature_engineering import prepare_all_features
from src.data.dataset import (
    make_windows,
    train_val_split,
    fit_transform_windows,
)
from src.models.factory import build_model
from src.models.common import count_parameters
from src.training.trainer import Trainer
from src.experiment_utils import load_prepared_nasa
from src.utils.config import load_config
from src.utils.io import project_root_from_config
from src.utils.seed import set_seed

from experiments.trail_common import CAUSAL_FEATURES
from experiments.trail_recursive_tools import (
    local_recursive,
    recursive_predict,
    score_future,
)

# ---------------------------------------------------------------------
# EXP36-v3: mixed historical-protocol adapter
#
# NASA:
#   - data/config semantics from EXP27 + config_trail.yaml
#   - cutoffs 50/70/90
#   - exact checkpoints:
#       recursive_TRAIL_Lite_<cell>_seed<seed>.pt
#       recursive_TRAIL_RUL_<cell>_seed<seed>.pt
#   - frozen reference:
#       results/trail_lite_recursive/trail_lite_recursive_seed_cell_cutoff.csv
#
# CALCE_CS2 / XJTU_B2:
#   - data/config semantics from EXP28 + config_trail_crossdomain.yaml
#   - exact recursive_cross_* checkpoints
#   - frozen reference:
#       results/trail_lite_crossdomain/crossdomain_recursive_seed_cell_cutoff.csv
#
# IMPORTANT:
#   This is source-only prefix-replay feasibility. It is NOT true LODO.
#   HNEI/SNL/HUST are not read.
# ---------------------------------------------------------------------

DOMAINS = ["NASA", "CALCE_CS2", "XJTU_B2"]
SEEDS_DEFAULT = [42, 2024, 3407]
NASA_CUTOFFS = [50, 70, 90]
REPLAY_FRACTIONS = (0.40, 0.60, 0.80)
REPLAY_EPS = 1e-6

CANDIDATES = [
    ("TRAIL_Lite", "TRAIL_Lite"),
    ("TRAIL_full", "TRAIL_RUL"),
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def qref_first5(cap: np.ndarray) -> float:
    cap = np.asarray(cap, float)
    if len(cap) < 5:
        raise ValueError("fewer than 5 capacity observations")
    q = float(np.median(cap[:5]))
    if not np.isfinite(q) or q <= 0:
        raise ValueError(f"invalid Qref={q}")
    return q


def valid_cutoffs_cross(dc: dict, n: int, window: int) -> list[int]:
    vals = []
    for x in dc.get("cutoff_cycles", []) or []:
        vals.append(int(x))
    for f in dc.get("cutoff_fractions", []) or []:
        vals.append(int(round(float(f) * n)))
    return sorted({x for x in vals if x >= window and x < n - 5})


def valid_cutoffs_nasa(n: int, window: int) -> list[int]:
    return [c for c in NASA_CUTOFFS if c >= window and c < n - 5]


def domain_cfg_cross(base: dict, dc: dict, cells: list[str]) -> dict:
    # Exact EXP28 semantics.
    c = copy.deepcopy(base)
    c["dataset"]["name"] = dc.get("name", "crossdomain")
    c["dataset"]["batteries"] = list(cells)
    c["dataset"]["rated_capacity"] = float(dc["rated_capacity"])
    c["dataset"]["eol_threshold"] = {
        b: float(dc["eol_threshold"]) for b in cells
    }
    c["dataset"]["smooth_method"] = "moving_average"
    c["dataset"]["smooth_window"] = int(dc.get("smooth_window", 7))
    return c


def load_state(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def runtime_from_checkpoint(
    cfg: dict,
    frames: dict[str, pd.DataFrame],
    train_bats: list[str],
    test_bat: str,
    backend_model: str,
    checkpoint_path: Path,
):
    """
    Rebuild the historical source-cell scaler/model shell and load the exact
    historical checkpoint. The held-out cell is not used for scaler fitting.
    """
    set_seed(int(cfg["training"]["seed"]))
    task = cfg["task"]
    thr = cfg["dataset"]["eol_threshold"]
    loss = cfg.get("loss", {})

    x, y, w, _ = make_windows(
        frames,
        train_bats,
        CAUSAL_FEATURES,
        int(task["window_size"]),
        int(task.get("prediction_horizon", 1)),
        task.get("mode", "capacity_prediction"),
        thr,
        late_stage_weight=loss.get("late_stage_weight", 2.0),
        eol_bandwidth=loss.get("eol_bandwidth", 0.12),
    )
    (tx, ty, tw), (vx, vy, vw) = train_val_split(
        x, y, w, cfg["training"].get("val_ratio", 0.2)
    )
    tx, vx, scaler = fit_transform_windows(
        tx, vx,
        scaler_kind=cfg["dataset"].get("scaler", "standard"),
    )

    mc = dict(cfg.get("model", {}))
    mc.pop("name", None)
    model = build_model(
        backend_model,
        input_dim=tx.shape[-1],
        window_size=int(task["window_size"]),
        model_cfg=mc,
    )
    model.load_state_dict(load_state(checkpoint_path), strict=True)

    trainer = Trainer(
        model,
        cfg,
        mode=task.get("mode", "capacity_prediction"),
        threshold=float(thr[test_bat]),
    )
    # Critical EXP36-v3 fix: historical Trainer.fit() ends in eval mode after
    # validation, whereas a newly constructed Trainer is in train mode.
    trainer.model.eval()
    return trainer, scaler, count_parameters(model)


def checkpoint_for(
    project: Path,
    domain: str,
    out_name: str,
    test_bat: str,
    seed: int,
) -> Path:
    ckdir = project / "results/checkpoints"
    if domain == "NASA":
        if out_name == "TRAIL_Lite":
            return ckdir / f"recursive_TRAIL_Lite_{test_bat}_seed{seed}.pt"
        if out_name == "TRAIL_full":
            return ckdir / f"recursive_TRAIL_RUL_{test_bat}_seed{seed}.pt"
        raise KeyError(out_name)

    safe_tag = f"cross_{domain}_{out_name}".replace("/", "_").replace(" ", "_")
    return ckdir / f"recursive_{safe_tag}_{test_bat}_seed{seed}.pt"


def replay_origins(cutoff: int, window: int) -> list[int]:
    vals = []
    for frac in REPLAY_FRACTIONS:
        p = max(window, int(math.floor(frac * cutoff)))
        if p < cutoff:
            vals.append(p)
    return sorted(set(vals))


def rmse(a, b) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a, b) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return float(np.mean(np.abs(a - b)))


def err_growth_slope(a, b) -> float:
    e = np.abs(np.asarray(a, float) - np.asarray(b, float))
    if len(e) < 2:
        return 0.0
    u = np.linspace(0.0, 1.0, len(e))
    return float(np.polyfit(u, e, 1)[0])


def recursive_predict_frozen(
    trainer: Trainer,
    scaler,
    prefix_cap,
    total_len: int,
    cfg: dict,
) -> np.ndarray:
    """
    Historical EXP27/EXP28 train_recursive_model() returns a Trainer whose
    final validation call leaves model.eval() active. A freshly reconstructed
    Trainer defaults to train() mode. Explicitly forcing eval() is therefore
    required for checkpoint replay to reproduce the historical inference state
    (and to disable dropout / training-mode behavior).
    """
    trainer.model.eval()
    if trainer.model.training:
        raise RuntimeError("model remained in training mode after eval()")
    return recursive_predict(trainer, scaler, prefix_cap, total_len, cfg)


def load_reference_tables(project: Path):
    nasa_path = (
        project / "results/trail_lite_recursive/"
        "trail_lite_recursive_seed_cell_cutoff.csv"
    )
    cross_path = (
        project / "results/trail_lite_crossdomain/"
        "crossdomain_recursive_seed_cell_cutoff.csv"
    )
    nasa = pd.read_csv(nasa_path)
    cross = pd.read_csv(cross_path)
    cross = cross.copy()
    return nasa_path, nasa, cross_path, cross


def reference_row(
    domain: str,
    table_nasa: pd.DataFrame,
    table_cross: pd.DataFrame,
    seed: int,
    cell: str,
    cutoff: int,
    model: str,
) -> pd.Series:
    if domain == "NASA":
        d = table_nasa
        mask = (
            (d["seed"].astype(int) == seed)
            & (d["test_battery"] == cell)
            & (d["cutoff"].astype(int) == cutoff)
            & (d["model"] == model)
        )
    else:
        d = table_cross
        mask = (
            (d["domain"] == domain)
            & (d["seed"].astype(int) == seed)
            & (d["test_battery"] == cell)
            & (d["cutoff"].astype(int) == cutoff)
            & (d["model"] == model)
        )
    g = d.loc[mask]
    if len(g) != 1:
        raise RuntimeError(
            f"reference row count={len(g)} for "
            f"{domain}/{cell}/seed{seed}/cutoff{cutoff}/{model}"
        )
    return g.iloc[0]


def load_domain(
    project: Path,
    domain: str,
    base_nasa: dict,
    base_cross: dict,
    protocol_cross: dict,
):
    if domain == "NASA":
        cfg = copy.deepcopy(base_nasa)
        frames = load_prepared_nasa(cfg)
        cells_all = [
            b for b in cfg["dataset"]["batteries"]
            if b in frames
        ]
        if len(cells_all) != 4:
            raise RuntimeError(
                f"NASA expected 4 cells, got {cells_all}"
            )
        return cfg, frames, cells_all

    dc = protocol_cross["domains"][domain]
    prepared_root = project / protocol_cross.get(
        "prepared_root", "data/crossdomain"
    )
    raw = load_prepared_domain(prepared_root, domain)
    expected = list(dc.get("batteries", []))
    if expected:
        missing = sorted(set(expected) - set(raw))
        if missing:
            raise RuntimeError(f"{domain}: missing prepared cells {missing}")
        raw = {b: raw[b] for b in expected}
    cells_all = sorted(raw)
    cfg = domain_cfg_cross(base_cross, dc, cells_all)
    frames = prepare_all_features(raw, cfg, use_denoising=True)
    return cfg, frames, cells_all


def deployment_cutoffs(
    domain: str,
    protocol_cross: dict,
    n: int,
    window: int,
):
    if domain == "NASA":
        return valid_cutoffs_nasa(n, window)
    return valid_cutoffs_cross(
        protocol_cross["domains"][domain], n, window
    )


def preflight(
    project: Path,
    seeds: list[int],
    base_nasa: dict,
    base_cross: dict,
    protocol_cross: dict,
    ref_nasa: pd.DataFrame,
    ref_cross: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for domain in DOMAINS:
        cfg, frames, cells = load_domain(
            project, domain, base_nasa, base_cross, protocol_cross
        )
        ws = int(cfg["task"]["window_size"])
        for cell in cells:
            true = (
                frames[cell].sort_values("cycle_index")["capacity"]
                .to_numpy(float)
            )
            cuts = deployment_cutoffs(
                domain, protocol_cross, len(true), ws
            )
            for seed in seeds:
                for model, _backend in CANDIDATES:
                    ck = checkpoint_for(
                        project, domain, model, cell, seed
                    )
                    ref_rows = 0
                    for cutoff in cuts:
                        try:
                            reference_row(
                                domain, ref_nasa, ref_cross,
                                seed, cell, cutoff, model
                            )
                            ref_rows += 1
                        except Exception:
                            pass
                    rows.append({
                        "domain": domain,
                        "cell": cell,
                        "seed": seed,
                        "model": model,
                        "checkpoint_exists": ck.exists(),
                        "checkpoint": str(ck),
                        "n_valid_cutoffs": len(cuts),
                        "n_reference_rows": ref_rows,
                    })
    return pd.DataFrame(rows)


def stratified_cell_bootstrap(
    cells: pd.DataFrame,
    n_boot: int = 5000,
    seed: int = 3407,
):
    rng = np.random.default_rng(seed)
    domains = sorted(cells["domain"].unique())
    obs = float(
        cells.groupby("domain")["delta_policy_minus_lite"].mean().mean()
    )
    boots = np.empty(n_boot, float)
    for i in range(n_boot):
        dm = []
        for d in domains:
            x = cells.loc[
                cells["domain"] == d,
                "delta_policy_minus_lite"
            ].to_numpy(float)
            dm.append(
                float(rng.choice(x, size=len(x), replace=True).mean())
            )
        boots[i] = float(np.mean(dm))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return obs, float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS_DEFAULT)
    ap.add_argument(
        "--output-dir",
        default="results/trail_dgr_prefix_replay_v3",
    )
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--reproduction-tol-ah", type=float, default=1e-6)
    ap.add_argument("--n-boot", type=int, default=5000)
    args = ap.parse_args()

    project = Path(__file__).resolve().parents[1]
    seeds = list(map(int, args.seeds))

    # Historical configs.
    nasa_cfg_path = project / "config_trail.yaml"
    cross_protocol_path = project / "config_trail_crossdomain.yaml"
    protocol_cross = yaml.safe_load(
        cross_protocol_path.read_text(encoding="utf-8")
    )
    cross_base_path = project / protocol_cross.get(
        "base_config", "config_trail.yaml"
    )

    base_nasa = load_config(nasa_cfg_path)
    base_cross = load_config(cross_base_path)

    if project_root_from_config(base_nasa).resolve() != project.resolve():
        raise RuntimeError("NASA config project root mismatch")
    if project_root_from_config(base_cross).resolve() != project.resolve():
        raise RuntimeError("cross config project root mismatch")

    nasa_ref_path, nasa_ref, cross_ref_path, cross_ref = (
        load_reference_tables(project)
    )

    out = project / args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    pf = preflight(
        project, seeds, base_nasa, base_cross, protocol_cross,
        nasa_ref, cross_ref
    )
    pf.to_csv(out / "EXP36_V3_PREFLIGHT.csv", index=False)

    print("\n========== EXP36-v3 PREFLIGHT ==========")
    sm = (
        pf.groupby(["domain", "model"], as_index=False)
        .agg(
            expected_checkpoints=("checkpoint_exists", "size"),
            existing_checkpoints=("checkpoint_exists", "sum"),
            expected_reference_rows=("n_valid_cutoffs", "sum"),
            found_reference_rows=("n_reference_rows", "sum"),
        )
    )
    print(sm.to_string(index=False))

    n_missing_ck = int((~pf["checkpoint_exists"]).sum())
    n_missing_ref = int(
        (pf["n_reference_rows"] != pf["n_valid_cutoffs"]).sum()
    )
    print("missing checkpoints =", n_missing_ck)
    print("cell-seed-model groups with missing reference rows =", n_missing_ref)

    if args.preflight_only:
        print("[PREFLIGHT ONLY]")
        if n_missing_ck or n_missing_ref:
            raise SystemExit(2)
        return

    if n_missing_ck:
        miss = pf.loc[
            ~pf["checkpoint_exists"],
            ["domain", "cell", "seed", "model", "checkpoint"],
        ]
        print(miss.to_string(index=False))
        raise RuntimeError("exact historical checkpoints missing")
    if n_missing_ref:
        bad = pf.loc[
            pf["n_reference_rows"] != pf["n_valid_cutoffs"]
        ]
        print(bad.to_string(index=False))
        raise RuntimeError("frozen reference rows missing")

    repro_rows = []
    checkpoint_rows = []
    replay_metric_rows = []
    replay_traj_rows = []
    future_traj_rows = []
    decision_rows = []
    exclusions = []

    for domain in DOMAINS:
        cfg_domain, frames, cells_all = load_domain(
            project, domain, base_nasa, base_cross, protocol_cross
        )
        ws = int(cfg_domain["task"]["window_size"])

        for seed in seeds:
            for test in cells_all:
                train = [b for b in cells_all if b != test]
                fr = (
                    frames[test]
                    .sort_values("cycle_index")
                    .reset_index(drop=True)
                )
                true = fr["capacity"].to_numpy(float)
                cyc = fr["cycle_index"].to_numpy()
                qref = qref_first5(true)
                cuts = deployment_cutoffs(
                    domain, protocol_cross, len(true), ws
                )
                if not cuts:
                    exclusions.append({
                        "domain": domain,
                        "seed": seed,
                        "test_battery": test,
                        "reason": "no_valid_historical_cutoff",
                    })
                    continue

                cfg = copy.deepcopy(cfg_domain)
                cfg["training"]["seed"] = seed
                thr = float(cfg["dataset"]["eol_threshold"][test])

                # Local future predictions + reproduction.
                local_future = {}
                local_met = {}
                for cutoff in cuts:
                    p = local_recursive(true[:cutoff], len(true), cfg)
                    m = score_future(true, p, cutoff, thr)
                    local_future[cutoff] = p
                    local_met[cutoff] = m

                    rr = reference_row(
                        domain, nasa_ref, cross_ref,
                        seed, test, cutoff, "Local_linear_anchor"
                    )
                    repro_rows.append({
                        "domain": domain,
                        "seed": seed,
                        "test_battery": test,
                        "cutoff": cutoff,
                        "model": "Local_linear_anchor",
                        "generated_RMSE": float(m["future_RMSE"]),
                        "frozen_RMSE": float(rr["future_RMSE"]),
                        "abs_delta_RMSE": abs(
                            float(m["future_RMSE"]) - float(rr["future_RMSE"])
                        ),
                        "generated_MAE": float(m["future_MAE"]),
                        "frozen_MAE": float(rr["future_MAE"]),
                        "abs_delta_MAE": abs(
                            float(m["future_MAE"]) - float(rr["future_MAE"])
                        ),
                    })

                runtimes = {}
                candidate_future = {c: {} for c in cuts}
                candidate_met = {c: {} for c in cuts}

                for out_name, backend in CANDIDATES:
                    ck = checkpoint_for(
                        project, domain, out_name, test, seed
                    )
                    trainer, scaler, nparam = runtime_from_checkpoint(
                        cfg, frames, train, test, backend, ck
                    )
                    runtimes[out_name] = (trainer, scaler)
                    checkpoint_rows.append({
                        "domain": domain,
                        "seed": seed,
                        "test_battery": test,
                        "model": out_name,
                        "backend_model": backend,
                        "checkpoint": str(ck),
                        "checkpoint_sha256": sha256_file(ck),
                        "num_parameters": int(nparam),
                        "historical_protocol": (
                            "EXP27_NASA"
                            if domain == "NASA"
                            else "EXP28_CROSSDOMAIN"
                        ),
                    })

                    for cutoff in cuts:
                        pred = recursive_predict_frozen(
                            trainer, scaler, true[:cutoff], len(true), cfg
                        )
                        met = score_future(true, pred, cutoff, thr)
                        candidate_future[cutoff][out_name] = pred
                        candidate_met[cutoff][out_name] = met

                        rr = reference_row(
                            domain, nasa_ref, cross_ref,
                            seed, test, cutoff, out_name
                        )
                        repro_rows.append({
                            "domain": domain,
                            "seed": seed,
                            "test_battery": test,
                            "cutoff": cutoff,
                            "model": out_name,
                            "generated_RMSE": float(met["future_RMSE"]),
                            "frozen_RMSE": float(rr["future_RMSE"]),
                            "abs_delta_RMSE": abs(
                                float(met["future_RMSE"])
                                - float(rr["future_RMSE"])
                            ),
                            "generated_MAE": float(met["future_MAE"]),
                            "frozen_MAE": float(rr["future_MAE"]),
                            "abs_delta_MAE": abs(
                                float(met["future_MAE"])
                                - float(rr["future_MAE"])
                            ),
                        })

                # STRICT historical reproduction gate before replay.
                repro_df = pd.DataFrame(repro_rows)
                current_bad = repro_df[
                    (repro_df["domain"] == domain)
                    & (repro_df["seed"] == seed)
                    & (repro_df["test_battery"] == test)
                    & (
                        (repro_df["abs_delta_RMSE"] > args.reproduction_tol_ah)
                        | (repro_df["abs_delta_MAE"] > args.reproduction_tol_ah)
                    )
                ]
                if not current_bad.empty:
                    repro_df.to_csv(
                        out / "EXP36_V3_REPRODUCTION_PARTIAL.csv",
                        index=False,
                    )
                    print("\n========== REPRODUCTION FAILURE ==========")
                    print(current_bad.to_string(index=False))
                    raise RuntimeError(
                        "historical protocol reproduction failed; "
                        "prefix replay not accepted"
                    )

                for cutoff in cuts:
                    origins = replay_origins(cutoff, ws)
                    if len(origins) < 2:
                        exclusions.append({
                            "domain": domain,
                            "seed": seed,
                            "test_battery": test,
                            "cutoff": cutoff,
                            "reason": "fewer_than_two_unique_replay_origins",
                            "origins": json.dumps(origins),
                        })
                        continue

                    local_E = {}
                    for p0 in origins:
                        lp = local_recursive(true[:p0], cutoff, cfg)
                        y = true[p0:cutoff]
                        yp = lp[p0:cutoff]
                        E = rmse(y / qref, yp / qref)
                        local_E[p0] = E

                        replay_metric_rows.append({
                            "domain": domain,
                            "seed": seed,
                            "test_battery": test,
                            "deployment_cutoff": cutoff,
                            "replay_origin": p0,
                            "model": "Local_linear_anchor",
                            "Qref_Ah": qref,
                            "replay_SOH_RMSE": E,
                            "replay_SOH_MAE": mae(y / qref, yp / qref),
                            "relative_gain_vs_local": 0.0,
                            "error_growth_slope_SOH": err_growth_slope(
                                y / qref, yp / qref
                            ),
                            "max_abs_error_SOH": float(
                                np.max(np.abs(yp - y)) / qref
                            ),
                        })

                        for k in range(p0, cutoff):
                            replay_traj_rows.append({
                                "domain": domain,
                                "seed": seed,
                                "test_battery": test,
                                "deployment_cutoff": cutoff,
                                "replay_origin": p0,
                                "model": "Local_linear_anchor",
                                "observation_index": k + 1,
                                "cycle_index": cyc[k],
                                "Qref_Ah": qref,
                                "y_true_Ah": float(true[k]),
                                "y_pred_recursive_Ah": float(lp[k]),
                                "y_true_SOH": float(true[k] / qref),
                                "y_pred_recursive_SOH": float(lp[k] / qref),
                            })

                    R = {}
                    diag = {}

                    for out_name, _backend in CANDIDATES:
                        trainer, scaler = runtimes[out_name]
                        gains, es, slopes, maxerrs = [], [], [], []

                        for p0 in origins:
                            pred = recursive_predict_frozen(
                                trainer, scaler, true[:p0], cutoff, cfg
                            )
                            y = true[p0:cutoff]
                            yp = pred[p0:cutoff]
                            E = rmse(y / qref, yp / qref)
                            gain = (
                                (local_E[p0] - E)
                                / (local_E[p0] + REPLAY_EPS)
                            )
                            gains.append(float(gain))
                            es.append(float(E))
                            slopes.append(
                                err_growth_slope(y / qref, yp / qref)
                            )
                            maxerrs.append(
                                float(np.max(np.abs(yp - y)) / qref)
                            )

                            replay_metric_rows.append({
                                "domain": domain,
                                "seed": seed,
                                "test_battery": test,
                                "deployment_cutoff": cutoff,
                                "replay_origin": p0,
                                "model": out_name,
                                "Qref_Ah": qref,
                                "replay_SOH_RMSE": E,
                                "replay_SOH_MAE": mae(
                                    y / qref, yp / qref
                                ),
                                "relative_gain_vs_local": gain,
                                "error_growth_slope_SOH": slopes[-1],
                                "max_abs_error_SOH": maxerrs[-1],
                            })

                            for k in range(p0, cutoff):
                                replay_traj_rows.append({
                                    "domain": domain,
                                    "seed": seed,
                                    "test_battery": test,
                                    "deployment_cutoff": cutoff,
                                    "replay_origin": p0,
                                    "model": out_name,
                                    "observation_index": k + 1,
                                    "cycle_index": cyc[k],
                                    "Qref_Ah": qref,
                                    "y_true_Ah": float(true[k]),
                                    "y_pred_recursive_Ah": float(pred[k]),
                                    "y_true_SOH": float(true[k] / qref),
                                    "y_pred_recursive_SOH": float(pred[k] / qref),
                                })

                        R[out_name] = float(np.min(gains))
                        diag[out_name] = {
                            "mean_replay_SOH_RMSE": float(np.mean(es)),
                            "worst_replay_SOH_RMSE": float(np.max(es)),
                            "replay_SOH_RMSE_MAD": float(
                                np.median(np.abs(es - np.median(es)))
                            ),
                            "worst_error_growth_slope_SOH": float(
                                np.max(slopes)
                            ),
                            "worst_max_abs_error_SOH": float(np.max(maxerrs)),
                        }

                    # Exact tie -> Lite.
                    chosen = max(
                        ["TRAIL_Lite", "TRAIL_full"],
                        key=lambda m: (
                            R[m],
                            1 if m == "TRAIL_Lite" else 0,
                        ),
                    )
                    rstar = float(R[chosen])
                    alpha = float(np.clip(rstar, 0.0, 1.0))
                    effective = chosen
                    if rstar <= 0:
                        alpha = 0.0
                        effective = "Local_linear_anchor"

                    lp = local_future[cutoff]
                    cp = candidate_future[cutoff][chosen]
                    policy = lp.copy()
                    policy[cutoff:] = (
                        lp[cutoff:]
                        + alpha * (cp[cutoff:] - lp[cutoff:])
                    )
                    pmet = score_future(true, policy, cutoff, thr)

                    lmet = local_met[cutoff]
                    limet = candidate_met[cutoff]["TRAIL_Lite"]
                    fumet = candidate_met[cutoff]["TRAIL_full"]

                    decision_rows.append({
                        "domain": domain,
                        "historical_protocol": (
                            "EXP27_NASA"
                            if domain == "NASA"
                            else "EXP28_CROSSDOMAIN"
                        ),
                        "seed": seed,
                        "test_battery": test,
                        "cutoff": cutoff,
                        "Qref_Ah": qref,
                        "replay_origins": json.dumps(origins),
                        "R_TRAIL_Lite": R["TRAIL_Lite"],
                        "R_TRAIL_full": R["TRAIL_full"],
                        "chosen_candidate": chosen,
                        "effective_model": effective,
                        "alpha": alpha,
                        "Local_SOH_RMSE": float(
                            lmet["future_RMSE"] / qref
                        ),
                        "Lite_SOH_RMSE": float(
                            limet["future_RMSE"] / qref
                        ),
                        "Full_SOH_RMSE": float(
                            fumet["future_RMSE"] / qref
                        ),
                        "Policy_SOH_RMSE": float(
                            pmet["future_RMSE"] / qref
                        ),
                        "Local_RMSE_Ah": float(lmet["future_RMSE"]),
                        "Lite_RMSE_Ah": float(limet["future_RMSE"]),
                        "Full_RMSE_Ah": float(fumet["future_RMSE"]),
                        "Policy_RMSE_Ah": float(pmet["future_RMSE"]),
                        "Policy_MAE_Ah": float(pmet["future_MAE"]),
                        "complexity_adopted": int(alpha > 0),
                        "policy_worse_than_local": int(
                            pmet["future_RMSE"] > lmet["future_RMSE"]
                        ),
                        "chosen_candidate_worse_than_local": int(
                            candidate_met[cutoff][chosen]["future_RMSE"]
                            > lmet["future_RMSE"]
                        ),
                        "policy_error_growth_slope_SOH": err_growth_slope(
                            true[cutoff:] / qref,
                            policy[cutoff:] / qref,
                        ),
                        "policy_max_abs_error_SOH": float(
                            np.max(
                                np.abs(
                                    policy[cutoff:] - true[cutoff:]
                                )
                            ) / qref
                        ),
                        "chosen_mean_replay_SOH_RMSE": (
                            diag[chosen]["mean_replay_SOH_RMSE"]
                        ),
                        "chosen_worst_replay_SOH_RMSE": (
                            diag[chosen]["worst_replay_SOH_RMSE"]
                        ),
                        "chosen_replay_SOH_RMSE_MAD": (
                            diag[chosen]["replay_SOH_RMSE_MAD"]
                        ),
                        "chosen_worst_replay_error_growth_slope_SOH": (
                            diag[chosen]["worst_error_growth_slope_SOH"]
                        ),
                        "chosen_worst_replay_max_abs_error_SOH": (
                            diag[chosen]["worst_max_abs_error_SOH"]
                        ),
                    })

                    for model_name, pred in {
                        "Local_linear_anchor": lp,
                        "TRAIL_Lite": candidate_future[cutoff]["TRAIL_Lite"],
                        "TRAIL_full": candidate_future[cutoff]["TRAIL_full"],
                        "TRAIL_DGR_Replay_v3": policy,
                    }.items():
                        for k in range(cutoff, len(true)):
                            future_traj_rows.append({
                                "domain": domain,
                                "seed": seed,
                                "test_battery": test,
                                "cutoff": cutoff,
                                "model": model_name,
                                "observation_index": k + 1,
                                "cycle_index": cyc[k],
                                "Qref_Ah": qref,
                                "y_true_Ah": float(true[k]),
                                "y_pred_recursive_Ah": float(pred[k]),
                                "y_true_SOH": float(true[k] / qref),
                                "y_pred_recursive_SOH": float(pred[k] / qref),
                            })

    repro = pd.DataFrame(repro_rows)
    cks = pd.DataFrame(checkpoint_rows)
    replay_metrics = pd.DataFrame(replay_metric_rows)
    replay_traj = pd.DataFrame(replay_traj_rows)
    future_traj = pd.DataFrame(future_traj_rows)
    decisions = pd.DataFrame(decision_rows)
    exclusions_df = pd.DataFrame(exclusions)

    max_repro = float(
        max(
            repro["abs_delta_RMSE"].max(),
            repro["abs_delta_MAE"].max(),
        )
    )
    if max_repro > args.reproduction_tol_ah:
        repro.to_csv(
            out / "EXP36_V3_REPRODUCTION_AUDIT.csv",
            index=False,
        )
        raise RuntimeError(
            f"reproduction failed: max delta {max_repro} Ah"
        )

    units = (
        decisions.groupby(
            ["domain", "test_battery", "cutoff"], as_index=False
        )
        .agg(
            Local=("Local_SOH_RMSE", "mean"),
            Always_Lite=("Lite_SOH_RMSE", "mean"),
            Always_Full=("Full_SOH_RMSE", "mean"),
            TRAIL_DGR_Replay_v3=("Policy_SOH_RMSE", "mean"),
            alpha=("alpha", "mean"),
            adoption_fraction=("complexity_adopted", "mean"),
        )
    )
    units["Policy_FCA"] = (
        (units["adoption_fraction"] > 0)
        & (units["TRAIL_DGR_Replay_v3"] > units["Local"])
    ).astype(int)
    units["Lite_FCA"] = (
        units["Always_Lite"] > units["Local"]
    ).astype(int)

    cells = (
        units.groupby(
            ["domain", "test_battery"], as_index=False
        )
        .agg(
            Local=("Local", "mean"),
            Always_Lite=("Always_Lite", "mean"),
            Always_Full=("Always_Full", "mean"),
            TRAIL_DGR_Replay_v3=("TRAIL_DGR_Replay_v3", "mean"),
            mean_alpha=("alpha", "mean"),
            adoption_fraction=("adoption_fraction", "mean"),
            Policy_FCAR=("Policy_FCA", "mean"),
            Lite_FCAR=("Lite_FCA", "mean"),
        )
    )
    cells["delta_policy_minus_lite"] = (
        cells["TRAIL_DGR_Replay_v3"] - cells["Always_Lite"]
    )

    domain_summary = (
        cells.groupby("domain", as_index=False)
        .agg(
            n_cells=("test_battery", "nunique"),
            Local=("Local", "mean"),
            Always_Lite=("Always_Lite", "mean"),
            Always_Full=("Always_Full", "mean"),
            TRAIL_DGR_Replay_v3=("TRAIL_DGR_Replay_v3", "mean"),
            mean_alpha=("mean_alpha", "mean"),
            adoption_rate=("adoption_fraction", "mean"),
            Policy_FCAR=("Policy_FCAR", "mean"),
            Lite_FCAR=("Lite_FCAR", "mean"),
        )
    )
    domain_summary["DGR_minus_Lite"] = (
        domain_summary["TRAIL_DGR_Replay_v3"]
        - domain_summary["Always_Lite"]
    )

    equal = {
        k: float(domain_summary[k].mean())
        for k in [
            "Local",
            "Always_Lite",
            "Always_Full",
            "TRAIL_DGR_Replay_v3",
        ]
    }
    wins = int((domain_summary["DGR_minus_Lite"] < 0).sum())
    worst_policy = float(
        domain_summary["TRAIL_DGR_Replay_v3"].max()
    )
    worst_lite = float(domain_summary["Always_Lite"].max())
    policy_fcar = float(cells["Policy_FCAR"].mean())
    lite_fcar = float(cells["Lite_FCAR"].mean())
    fallback = float(1.0 - cells["adoption_fraction"].mean())

    delta, lo, hi = stratified_cell_bootstrap(
        cells[
            ["domain", "test_battery", "delta_policy_minus_lite"]
        ],
        n_boot=int(args.n_boot),
        seed=3407,
    )

    pass_items = {
        "equal_domain_cell_SOH_RMSE_better_than_Always_Lite":
            bool(equal["TRAIL_DGR_Replay_v3"] < equal["Always_Lite"]),
        "at_least_2_of_3_domains_better_than_Always_Lite":
            bool(wins >= 2),
        "worst_domain_better_than_Always_Lite":
            bool(worst_policy < worst_lite),
        "FCAR_not_worse_than_Always_Lite":
            bool(policy_fcar <= lite_fcar),
        "local_fallback_rate_below_0p90":
            bool(fallback < 0.90),
    }
    status = (
        "PASS_FEASIBILITY"
        if all(pass_items.values())
        else "STOP_BEFORE_TRAIL_DGR_TRAINING"
    )

    summary = {
        "status": status,
        "role": "SOURCE_ONLY_PREFIX_REPLAY_FEASIBILITY",
        "scope": (
            "NASA uses EXP27 historical protocol; CALCE_CS2 and XJTU_B2 "
            "use EXP28 historical protocol. This is not true LODO."
        ),
        "normalization": (
            "SOH = capacity / median(first five observed capacities)"
        ),
        "replay_fractions": list(REPLAY_FRACTIONS),
        "reliability_rule": (
            "R_m=min_j((E_local-E_m)/(E_local+1e-6)); "
            "choose max R; exact tie -> Lite; alpha=clip(R*,0,1); "
            "R*<=0 -> Local."
        ),
        "domains": DOMAINS,
        "seeds": seeds,
        "max_historical_reproduction_delta_Ah": max_repro,
        "equal_domain_cell_mean_SOH_RMSE": equal,
        "domain_wins_vs_Always_Lite": wins,
        "worst_domain": {
            "TRAIL_DGR_Replay_v3": worst_policy,
            "Always_Lite": worst_lite,
        },
        "FCAR": {
            "TRAIL_DGR_Replay_v3": policy_fcar,
            "Always_Lite": lite_fcar,
        },
        "local_fallback_rate": fallback,
        "bootstrap_equal_domain_cell_delta_DGR_minus_Lite": {
            "observed": delta,
            "ci95_low": lo,
            "ci95_high": hi,
            "n_boot": int(args.n_boot),
            "unit": "physical_cell",
            "stratified_by_domain": True,
        },
        "pass_items": pass_items,
        "HNEI_read": False,
        "SNL_read": False,
        "HUST_read": False,
    }

    # Write outputs.
    repro.to_csv(out / "EXP36_V3_REPRODUCTION_AUDIT.csv", index=False)
    cks.to_csv(out / "EXP36_V3_CHECKPOINT_PROVENANCE.csv", index=False)
    replay_metrics.to_csv(out / "EXP36_V3_PREFIX_REPLAY_METRICS.csv", index=False)
    replay_traj.to_csv(out / "EXP36_V3_PREFIX_REPLAY_TRAJECTORIES.csv", index=False)
    future_traj.to_csv(out / "EXP36_V3_FUTURE_TRAJECTORIES.csv", index=False)
    decisions.to_csv(out / "EXP36_V3_SEED_DECISIONS.csv", index=False)
    units.to_csv(out / "EXP36_V3_CELL_CUTOFF_UNITS.csv", index=False)
    cells.to_csv(out / "EXP36_V3_PHYSICAL_CELLS.csv", index=False)
    domain_summary.to_csv(out / "EXP36_V3_DOMAIN_SUMMARY.csv", index=False)
    exclusions_df.to_csv(out / "EXP36_V3_STRUCTURAL_EXCLUSIONS.csv", index=False)

    (out / "EXP36_V3_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    audit = {
        "status": status,
        "script": Path(__file__).name,
        "script_sha256": sha256_file(Path(__file__)),
        "NASA_config": str(nasa_cfg_path.relative_to(project)),
        "NASA_config_sha256": sha256_file(nasa_cfg_path),
        "cross_protocol": str(cross_protocol_path.relative_to(project)),
        "cross_protocol_sha256": sha256_file(cross_protocol_path),
        "NASA_reference": str(nasa_ref_path.relative_to(project)),
        "NASA_reference_sha256": sha256_file(nasa_ref_path),
        "cross_reference": str(cross_ref_path.relative_to(project)),
        "cross_reference_sha256": sha256_file(cross_ref_path),
        "n_seed_decisions": int(len(decisions)),
        "n_cell_cutoff_units": int(len(units)),
        "n_physical_cells": int(len(cells)),
        "physical_inference_unit": "cell",
        "future_truth_used_for_selection": False,
        "observed_prefix_truth_used_for_selection": True,
        "HNEI_read": False,
        "SNL_read": False,
        "HUST_read": False,
    }
    (out / "EXP36_V3_AUDIT.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    checksum = out / "EXP36_V3_SHA256SUMS.txt"
    files = sorted(
        p for p in out.rglob("*")
        if p.is_file() and p != checksum
    )
    with checksum.open("w", encoding="utf-8") as f:
        for p in files:
            f.write(f"{sha256_file(p)}  {p.relative_to(out)}\n")

    print("\n========== EXP36-v3 DOMAIN SUMMARY ==========")
    print(domain_summary.to_string(index=False))

    print("\n========== EQUAL-DOMAIN CELL MEAN ==========")
    for k, v in equal.items():
        print(f"{k}: {v:.10f}")

    print("\n========== DGR - LITE CELL BOOTSTRAP ==========")
    print(f"delta={delta:.10f}  CI95=[{lo:.10f}, {hi:.10f}]")

    print("\n========== PASS ITEMS ==========")
    for k, v in pass_items.items():
        print(k, "=", v)

    print("\nEXP36-v3 STATUS:", status)
    print("[WROTE]", out)


if __name__ == "__main__":
    main()
