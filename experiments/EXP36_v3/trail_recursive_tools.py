from __future__ import annotations

import copy
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from src.data.dataset import BatteryWindowDataset, make_windows, train_val_split, fit_transform_windows
from src.models.factory import build_model
from src.models.common import count_parameters
from src.training.trainer import Trainer
from src.utils.io import project_root_from_config
from src.utils.seed import set_seed
from experiments.trail_common import CAUSAL_FEATURES, local_linear_one_step


def rebuild_causal(cap_history: Iterable[float], rated_capacity: float) -> pd.DataFrame:
    """Rebuild only features that can be recursively reconstructed from past/predicted capacity."""
    cap = np.asarray(list(cap_history), dtype=float)
    cyc = np.arange(1, len(cap) + 1, dtype=float)
    s = pd.Series(cap)
    diff = s.diff().fillna(0.0).to_numpy(float)
    roll_mean = s.rolling(5, min_periods=1).mean().to_numpy(float)
    roll_std = s.rolling(5, min_periods=1).std().fillna(0.0).to_numpy(float)
    degr = (-s.diff().fillna(0.0)).clip(lower=-0.05, upper=0.05).to_numpy(float)
    return pd.DataFrame({
        'cycle_index': cyc,
        'capacity': cap,
        'SOH': cap / max(float(rated_capacity), 1e-9),
        'capacity_diff': diff,
        'capacity_rolling_mean': roll_mean,
        'capacity_rolling_std': roll_std,
        'degradation_rate': degr,
    })


def train_recursive_model(cfg: dict, frames: dict[str, pd.DataFrame], train_bats: list[str], test_bat: str,
                          model_name: str, feature_cols: list[str] | None = None,
                          model_overrides: dict | None = None, checkpoint_tag: str | None = None):
    """Train one LOBO model once; it can then be recursively evaluated at many cutoffs."""
    feature_cols = feature_cols or CAUSAL_FEATURES
    set_seed(int(cfg['training']['seed']))
    task = cfg['task']
    thr = cfg['dataset']['eol_threshold']
    loss = cfg.get('loss', {})
    x, y, w, _ = make_windows(
        frames, train_bats, feature_cols,
        int(task['window_size']), int(task.get('prediction_horizon', 1)),
        task.get('mode', 'capacity_prediction'), thr,
        late_stage_weight=loss.get('late_stage_weight', 2.0),
        eol_bandwidth=loss.get('eol_bandwidth', 0.12),
    )
    (tx, ty, tw), (vx, vy, vw) = train_val_split(x, y, w, cfg['training'].get('val_ratio', 0.2))
    tx, vx, scaler = fit_transform_windows(tx, vx, scaler_kind=cfg['dataset'].get('scaler', 'standard'))
    tr = BatteryWindowDataset(tx, ty, tw)
    va = BatteryWindowDataset(vx, vy, vw)

    mc = dict(cfg.get('model', {}))
    mc.pop('name', None)
    if model_overrides:
        mc.update(model_overrides)
    model = build_model(model_name, input_dim=tx.shape[-1], window_size=int(task['window_size']), model_cfg=mc)
    trainer = Trainer(model, cfg, mode=task.get('mode', 'capacity_prediction'), threshold=float(thr[test_bat]))
    root = project_root_from_config(cfg)
    tag = checkpoint_tag or model_name
    safe_tag = str(tag).replace('/', '_').replace(' ', '_')
    ck = root / f'results/checkpoints/recursive_{safe_tag}_{test_bat}_seed{cfg["training"]["seed"]}.pt'
    hist = trainer.fit(tr, va, checkpoint_path=ck)
    return trainer, scaler, hist, count_parameters(model)


def transform_window(arr: np.ndarray, scaler) -> np.ndarray:
    z = scaler.transform(arr)
    # Keep capacity in physical Ah, matching the original TRAIL/MSTT training pipeline.
    z[:, 1] = arr[:, 1]
    return z


def recursive_predict(trainer: Trainer, scaler, prefix_cap: Iterable[float], total_len: int, cfg: dict) -> np.ndarray:
    hist = list(map(float, prefix_cap))
    ws = int(cfg['task']['window_size'])
    rated = float(cfg['dataset'].get('rated_capacity', 2.0))
    while len(hist) < int(total_len):
        fr = rebuild_causal(hist, rated)
        arr = fr[CAUSAL_FEATURES].to_numpy(float)[-ws:]
        if len(arr) < ws:
            raise ValueError('prefix shorter than window size')
        x = transform_window(arr, scaler)[None, :, :]
        with torch.no_grad():
            p = float(trainer.model(torch.tensor(x, dtype=torch.float32, device=trainer.device)).detach().cpu().reshape(-1)[0])
        # Broad physical guardrail only. No future truth is used.
        q0 = max(abs(hist[0]), 1e-6)
        p = float(np.clip(p, 0.25 * q0, 1.10 * q0))
        hist.append(p)
    return np.asarray(hist, float)


def local_recursive(prefix_cap: Iterable[float], total_len: int, cfg: dict) -> np.ndarray:
    hist = list(map(float, prefix_cap))
    ws = int(cfg['task']['window_size'])
    rated = float(cfg['dataset'].get('rated_capacity', 2.0))
    while len(hist) < int(total_len):
        fr = rebuild_causal(hist, rated)
        arr = fr[CAUSAL_FEATURES].to_numpy(float)[-ws:]
        hist.append(float(local_linear_one_step(
            arr, 1, int(cfg['model'].get('trend_window', 8)), float(cfg['model'].get('trend_decay', 0.22))
        )))
    return np.asarray(hist, float)


def score_future(true_cap: Iterable[float], pred: Iterable[float], cutoff: int, threshold: float) -> dict:
    true_cap = np.asarray(true_cap, float)
    pred = np.asarray(pred, float)
    y = true_cap[cutoff:]
    p = pred[cutoff:]
    rmse = float(np.sqrt(np.mean((p - y) ** 2)))
    mae = float(np.mean(np.abs(p - y)))

    def eol(a):
        idx = np.flatnonzero(np.asarray(a) <= threshold)
        return int(idx[0] + 1) if len(idx) else None

    te, pe = eol(true_cap), eol(pred)
    eol_err = abs(pe - te) if te is not None and pe is not None else np.nan
    return {
        'future_RMSE': rmse,
        'future_MAE': mae,
        'true_EOL': te,
        'pred_EOL': pe,
        'EOL_abs_error': eol_err,
        'threshold_hit': int(pe is not None),
        'n_future': int(len(y)),
    }


def _linear_slope(y: np.ndarray) -> float:
    y = np.asarray(y, float)
    if len(y) < 2:
        return 0.0
    t = np.arange(len(y), dtype=float)
    return float(np.polyfit(t, y, 1)[0])


def prefix_descriptors(prefix_cap: Iterable[float], total_len: int, cutoff: int,
                       rated_capacity: float, threshold: float) -> dict:
    """Outcome-free descriptors computed from the observed prefix only."""
    cap = np.asarray(list(prefix_cap), float)
    if len(cap) < 4:
        raise ValueError('prefix too short for failure descriptors')
    short = cap[-min(8, len(cap)):]
    long = cap[-min(16, len(cap)):]
    slope_short = _linear_slope(short)
    slope_long = _linear_slope(long)
    diff = np.diff(long)
    # Quadratic coefficient is a simple curvature proxy; it is not called a physical knee detector.
    if len(long) >= 5:
        t = np.linspace(-1.0, 1.0, len(long))
        curvature = float(np.polyfit(t, long, 2)[0] / max(abs(long[-1]), 1e-9))
    else:
        curvature = 0.0
    mono_violation = float(np.mean(diff > 0)) if len(diff) else 0.0
    recent_drop = float(cap[-1] - cap[-min(6, len(cap))])
    horizon = int(total_len - cutoff)
    return {
        'prefix_last_capacity': float(cap[-1]),
        'prefix_soh': float(cap[-1] / max(rated_capacity, 1e-9)),
        'distance_to_eol_capacity': float(cap[-1] - threshold),
        'slope_short': slope_short,
        'slope_long': slope_long,
        'slope_instability': float(abs(slope_short - slope_long)),
        'curvature': curvature,
        'curvature_abs': float(abs(curvature)),
        'diff_std': float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0,
        'diff_mad': float(np.median(np.abs(diff - np.median(diff)))) if len(diff) else 0.0,
        'monotonicity_violation': mono_violation,
        'recent_drop': recent_drop,
        'cutoff_fraction': float(cutoff / max(total_len, 1)),
        'horizon_cycles': horizon,
        'horizon_fraction': float(horizon / max(total_len, 1)),
    }


def bootstrap_mean_ci(values: Iterable[float], n_boot: int = 5000, seed: int = 3407) -> tuple[float, float]:
    x = np.asarray(list(values), float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (np.nan, np.nan)
    if len(x) == 1:
        return (float(x[0]), float(x[0]))
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, float)
    for i in range(n_boot):
        means[i] = rng.choice(x, size=len(x), replace=True).mean()
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def load_existing_recursive_reference(cfg: dict) -> pd.DataFrame:
    root = project_root_from_config(cfg)
    p = root / 'results/trail_recursive/trail_recursive_seed_cell_cutoff.csv'
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    needed = {'seed', 'test_battery', 'cutoff', 'model', 'future_RMSE'}
    if not needed.issubset(df.columns):
        return pd.DataFrame()
    return df
