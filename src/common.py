from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import Dataset

FEATURE_NAMES = (
    "cycle_scaled",
    "capacity",
    "soh",
    "capacity_diff",
    "capacity_rolling_mean",
    "capacity_rolling_std",
    "degradation_rate",
)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def curves_sha256(curves: Mapping[str, pd.DataFrame]) -> str:
    """Hash deterministic curve content, independent of source file paths."""
    digest = hashlib.sha256()
    preferred = (
        "cycle",
        "raw_capacity",
        "capacity",
        "measurement_observed",
        "battery_id",
        "batch",
        "initial_capacity_Ah",
    )
    for cell_id in sorted(curves):
        frame = curves[cell_id].sort_values("cycle").reset_index(drop=True)
        columns = [column for column in preferred if column in frame.columns]
        digest.update(f"{cell_id}\n".encode())
        digest.update(
            frame[columns]
            .to_csv(
                index=False,
                na_rep="NA",
                float_format="%.17g",
                lineterminator="\n",
            )
            .encode()
        )
    return digest.hexdigest()


def load_protocol(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "1.0":
        raise ValueError("Unsupported confirmatory protocol schema")
    return payload


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def causal_smooth(values: Sequence[float], window: int) -> np.ndarray:
    return (
        pd.Series(np.asarray(values, dtype=float))
        .rolling(window=max(1, int(window)), min_periods=1)
        .mean()
        .to_numpy(float)
    )


def compute_features(
    cycles: Sequence[int], capacities: Sequence[float], q0: float
) -> np.ndarray:
    cycle = np.asarray(cycles, dtype=float)
    cap = np.asarray(capacities, dtype=float)
    if len(cycle) != len(cap) or not len(cap):
        raise ValueError("Cycle and capacity arrays must have equal non-zero length")
    diff = np.r_[0.0, np.diff(cap)]
    rolling = pd.Series(cap).rolling(7, min_periods=1)
    roll_mean = rolling.mean().to_numpy(float)
    roll_std = rolling.std(ddof=0).fillna(0.0).to_numpy(float)
    degradation = np.zeros_like(cap)
    for i in range(1, len(cap)):
        j = max(0, i - 7)
        delta_cycle = max(1.0, cycle[i] - cycle[j])
        degradation[i] = (cap[i] - cap[j]) / delta_cycle
    if len(cap) > 1:
        degradation[0] = degradation[1]
    return np.column_stack(
        [
            cycle / 1000.0,
            cap,
            cap / float(q0),
            diff,
            roll_mean,
            roll_std,
            degradation,
        ]
    ).astype(np.float32)


def load_prepared_development(prepared_dir: Path) -> dict[str, pd.DataFrame]:
    curves_dir = prepared_dir / "curves"
    if not curves_dir.is_dir():
        raise FileNotFoundError(f"Missing prepared curves directory: {curves_dir}")
    curves: dict[str, pd.DataFrame] = {}
    for path in sorted(curves_dir.glob("*.csv")):
        frame = pd.read_csv(path)
        required = {"cycle", "raw_capacity", "capacity", "battery_id", "batch"}
        if not required.issubset(frame.columns):
            continue
        batch = int(frame["batch"].iloc[0])
        if batch not in (1, 3):
            continue
        frame = frame.sort_values("cycle").reset_index(drop=True)
        cycle = frame["cycle"].to_numpy(int)
        if not np.array_equal(cycle, np.arange(1, len(cycle) + 1)):
            raise ValueError(f"{path}: development cycles must be contiguous from 1")
        curves[str(frame["battery_id"].iloc[0])] = frame
    counts = {
        batch: sum(int(frame["batch"].iloc[0]) == batch for frame in curves.values())
        for batch in (1, 3)
    }
    if counts != {1: 8, 3: 8}:
        raise ValueError(f"Expected B1=8 and B3=8 development cells, got {counts}")
    return curves


def observed_training_end(frame: pd.DataFrame, threshold: float) -> int:
    hits = np.flatnonzero(frame["raw_capacity"].to_numpy(float) <= threshold)
    return int(hits[0]) if len(hits) else len(frame) - 1


@dataclass
class ArraySet:
    X: np.ndarray
    caps: np.ndarray
    cycles: np.ndarray
    q0: np.ndarray
    targets: np.ndarray
    cells: np.ndarray
    target_cycles: np.ndarray


@dataclass
class WindowSets:
    train: ArraySet
    validation: ArraySet | None
    scaler: StandardScaler


def windows_for_cells(
    curves: Mapping[str, pd.DataFrame],
    cell_ids: Sequence[str],
    window: int,
    support_horizon: int,
    threshold: float,
) -> ArraySet:
    parts: dict[str, list[np.ndarray]] = {
        key: []
        for key in ("X", "caps", "cycles", "q0", "targets", "cells", "target_cycles")
    }
    for cell_id in cell_ids:
        frame = curves[cell_id].copy()
        q0 = float(frame.loc[frame["cycle"].eq(1), "raw_capacity"].iloc[0])
        support_end = observed_training_end(frame, threshold)
        use = frame.iloc[: support_end + 1].reset_index(drop=True)
        cycles = use["cycle"].to_numpy(int)
        caps = use["capacity"].to_numpy(float)
        features = compute_features(cycles, caps, q0)
        stop = len(use) - support_horizon + 1
        rows = list(range(window, stop))
        if len(rows) < 10:
            raise ValueError(f"{cell_id}: only {len(rows)} common-support windows")
        parts["X"].append(
            np.asarray([features[i - window : i] for i in rows], np.float32)
        )
        parts["caps"].append(
            np.asarray([caps[i - window : i] for i in rows], np.float32)
        )
        parts["cycles"].append(
            np.asarray([cycles[i - window : i] for i in rows], np.float32)
        )
        parts["q0"].append(np.asarray([q0] * len(rows), np.float32))
        parts["targets"].append(
            np.asarray([caps[i : i + support_horizon] for i in rows], np.float32)
        )
        parts["cells"].append(np.asarray([cell_id] * len(rows), dtype=object))
        parts["target_cycles"].append(np.asarray([cycles[i] for i in rows], np.int32))
    return ArraySet(
        X=np.concatenate(parts["X"]),
        caps=np.concatenate(parts["caps"]),
        cycles=np.concatenate(parts["cycles"]),
        q0=np.concatenate(parts["q0"]),
        targets=np.concatenate(parts["targets"]),
        cells=np.concatenate(parts["cells"]),
        target_cycles=np.concatenate(parts["target_cycles"]),
    )


def transform_array_set(data: ArraySet, scaler: StandardScaler) -> ArraySet:
    shape = data.X.shape
    transformed = (
        scaler.transform(data.X.reshape(-1, shape[-1]))
        .reshape(shape)
        .astype(np.float32)
    )
    transformed[:, :, 1] = data.caps
    return ArraySet(
        X=transformed,
        caps=data.caps,
        cycles=data.cycles,
        q0=data.q0,
        targets=data.targets,
        cells=data.cells,
        target_cycles=data.target_cycles,
    )


def build_window_sets(
    curves: Mapping[str, pd.DataFrame],
    train_cells: Sequence[str],
    validation_cells: Sequence[str] | None,
    window: int,
    support_horizon: int,
    threshold: float,
) -> WindowSets:
    raw_train = windows_for_cells(
        curves, train_cells, window, support_horizon, threshold
    )
    scaler = StandardScaler().fit(raw_train.X.reshape(-1, raw_train.X.shape[-1]))
    validation = None
    if validation_cells:
        validation = transform_array_set(
            windows_for_cells(
                curves, validation_cells, window, support_horizon, threshold
            ),
            scaler,
        )
    return WindowSets(
        train=transform_array_set(raw_train, scaler),
        validation=validation,
        scaler=scaler,
    )


def support_frame(data: ArraySet, arm: str) -> pd.DataFrame:
    return (
        pd.DataFrame(
            {
                "arm": arm,
                "cell_id": data.cells.astype(str),
                "target_cycle": data.target_cycles.astype(int),
            }
        )
        .sort_values(["cell_id", "target_cycle"])
        .reset_index(drop=True)
    )


def assert_identical_support(frames: Mapping[str, pd.DataFrame]) -> str:
    reference_name = next(iter(frames))
    reference = frames[reference_name][["cell_id", "target_cycle"]].reset_index(
        drop=True
    )
    for name, frame in frames.items():
        current = frame[["cell_id", "target_cycle"]].reset_index(drop=True)
        if not current.equals(reference):
            raise RuntimeError(
                f"Common target-support gate failed: {name} differs from {reference_name}"
            )
    return hashlib.sha256(reference.to_csv(index=False).encode("utf-8")).hexdigest()


class RolloutDataset(Dataset):
    def __init__(self, arrays: ArraySet):
        self.X = torch.from_numpy(arrays.X)
        self.caps = torch.from_numpy(arrays.caps)
        self.cycles = torch.from_numpy(arrays.cycles)
        self.q0 = torch.from_numpy(arrays.q0)
        self.targets = torch.from_numpy(arrays.targets)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        return (
            self.X[index],
            self.caps[index],
            self.cycles[index],
            self.q0[index],
            self.targets[index],
        )


def append_predicted_step(
    feature_window: torch.Tensor,
    cap_window: torch.Tensor,
    cycle_window: torch.Tensor,
    q0: torch.Tensor,
    prediction: torch.Tensor,
    scaler_mean: torch.Tensor,
    scaler_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    next_cycle = cycle_window[:, -1] + 1.0
    last_capacity = cap_window[:, -1]
    recent = torch.cat([cap_window[:, -6:], prediction[:, None]], dim=1)
    roll_mean = recent.mean(dim=1)
    roll_std = recent.std(dim=1, unbiased=False)
    lag_index = max(0, cap_window.shape[1] - 7)
    cycle_delta = (next_cycle - cycle_window[:, lag_index]).clamp(min=1.0)
    degradation = (prediction - cap_window[:, lag_index]) / cycle_delta
    raw = torch.stack(
        [
            next_cycle / 1000.0,
            prediction,
            prediction / q0,
            prediction - last_capacity,
            roll_mean,
            roll_std,
            degradation,
        ],
        dim=1,
    )
    scaled = (raw - scaler_mean) / scaler_scale
    scaled[:, 1] = prediction
    return (
        torch.cat([feature_window[:, 1:, :], scaled[:, None, :]], dim=1),
        torch.cat([cap_window[:, 1:], prediction[:, None]], dim=1),
        torch.cat([cycle_window[:, 1:], next_cycle[:, None]], dim=1),
    )


def rollout_loss(
    model: nn.Module,
    X: torch.Tensor,
    caps: torch.Tensor,
    cycles: torch.Tensor,
    q0: torch.Tensor,
    targets: torch.Tensor,
    steps: int,
    scaler_mean: torch.Tensor,
    scaler_scale: torch.Tensor,
    threshold: float,
    include_penalty: bool,
) -> torch.Tensor:
    if steps > targets.shape[1]:
        raise ValueError("Loss horizon exceeds frozen common target support")
    huber = nn.HuberLoss(reduction="none", delta=0.01)
    feature_window, cap_window, cycle_window = X, caps, cycles
    weights = torch.pow(
        torch.tensor(0.8, device=X.device, dtype=X.dtype),
        torch.arange(steps, device=X.device, dtype=X.dtype),
    )
    weights = weights / weights.sum()
    losses = []
    for step in range(steps):
        prediction = model(feature_window, cap_window)
        target = targets[:, step]
        eol_weight = 1.0 + 2.0 * torch.exp(-torch.abs(target - threshold) / 0.03)
        loss = huber(prediction, target) * eol_weight
        if include_penalty:
            loss = (
                loss + 0.1 * torch.relu(prediction - cap_window[:, -1] - 0.005).square()
            )
        losses.append(loss.mean() * weights[step])
        feature_window, cap_window, cycle_window = append_predicted_step(
            feature_window,
            cap_window,
            cycle_window,
            q0,
            prediction,
            scaler_mean,
            scaler_scale,
        )
    return torch.stack(losses).sum()


def local_linear_next(capacities: Sequence[float], lookback: int = 12) -> float:
    values = np.asarray(capacities[-lookback:], dtype=float)
    if len(values) < 2:
        return float(values[-1])
    slope, intercept = np.polyfit(np.arange(len(values), dtype=float), values, 1)
    slope = float(np.clip(slope, -0.04, -1e-6))
    return float(intercept + slope * len(values))


def predict_neural(
    model: nn.Module,
    scaler: StandardScaler,
    cycles: Sequence[int],
    capacities: Sequence[float],
    q0: float,
    window: int,
    device: torch.device,
) -> float:
    features = compute_features(cycles, capacities, q0)[-window:]
    scaled = scaler.transform(features).astype(np.float32)
    scaled[:, 1] = np.asarray(capacities[-window:], dtype=np.float32)
    with torch.no_grad():
        X = torch.from_numpy(scaled[None]).to(device)
        caps = torch.from_numpy(
            np.asarray(capacities[-window:], dtype=np.float32)[None]
        ).to(device)
        return float(model(X, caps).detach().cpu().item())


def recursive_forecast(
    model_name: str,
    frame: pd.DataFrame,
    cutoff: int,
    horizon: int,
    threshold: float,
    window: int,
    model: nn.Module | None = None,
    scaler: StandardScaler | None = None,
    device: torch.device | None = None,
) -> tuple[pd.DataFrame, int | None]:
    observed = frame[frame["cycle"] <= cutoff]
    cycles = observed["cycle"].astype(int).tolist()
    capacities = observed["capacity"].astype(float).tolist()
    if len(cycles) < window:
        raise ValueError(f"cutoff {cutoff} has fewer than {window} input cycles")
    if "initial_capacity_Ah" in frame and frame["initial_capacity_Ah"].notna().any():
        q0 = float(frame["initial_capacity_Ah"].dropna().iloc[0])
    else:
        q0 = float(frame["raw_capacity"].dropna().iloc[0])
    lower, upper = 0.40 * q0, 1.20 * q0
    pred_eol: int | None = None
    rows = []
    for step in range(1, horizon + 1):
        cycle = cutoff + step
        if model_name == "persistence":
            prediction = float(capacities[-1])
        elif model_name == "local_linear_trend":
            prediction = local_linear_next(capacities)
        else:
            if model is None or scaler is None or device is None:
                raise ValueError("Neural forecast requires model, scaler, and device")
            prediction = predict_neural(
                model, scaler, cycles, capacities, q0, window, device
            )
        prediction = float(np.clip(prediction, lower, upper))
        cycles.append(int(cycle))
        capacities.append(prediction)
        if pred_eol is None and prediction <= threshold:
            pred_eol = int(cycle)
        rows.append({"cycle": int(cycle), "predicted_capacity": prediction})
    return pd.DataFrame(rows), pred_eol


def eol_interval(frame: pd.DataFrame, threshold: float) -> tuple[int, int] | None:
    measured = frame[frame["measurement_observed"].eq(1)].copy()
    measured = measured.dropna(subset=["raw_capacity"]).sort_values("cycle")
    hits = measured[measured["raw_capacity"] <= threshold]
    if hits.empty:
        return None
    upper = int(hits.iloc[0]["cycle"])
    earlier = measured[measured["cycle"] < upper]
    lower = 1 if earlier.empty else int(earlier.iloc[-1]["cycle"]) + 1
    return lower, upper


def interval_distance(value: int, interval: tuple[int, int]) -> float:
    lower, upper = interval
    if value < lower:
        return float(lower - value)
    if value > upper:
        return float(value - upper)
    return 0.0


def censor_aware_timing_score(
    pred_eol: int | None,
    cutoff: int,
    horizon: int,
    truth_interval: tuple[int, int] | None,
    right_censor_cycle: int,
) -> float:
    """Distance compatible with interval-observed or right-censored EOL truth.

    If the true threshold crossing is interval observed, an in-horizon crossing
    is scored by its distance to that interval. A forecast with no crossing is
    known only to exceed the forecast horizon, so it is penalized only when the
    complete true interval is already before that lower bound.

    If truth is right censored, only a predicted crossing on or before the last
    genuine reference-capacity observation is contradicted by the data.
    """
    if truth_interval is not None:
        if pred_eol is not None:
            return interval_distance(int(pred_eol), truth_interval)
        prediction_lower_bound = int(cutoff + horizon + 1)
        return float(max(0, prediction_lower_bound - int(truth_interval[1])))
    if pred_eol is None or int(pred_eol) > int(right_censor_cycle):
        return 0.0
    return float(int(right_censor_cycle) + 1 - int(pred_eol))


def evaluate_forecast(
    frame: pd.DataFrame,
    forecast: pd.DataFrame,
    cutoff: int,
    pred_eol: int | None,
    threshold: float,
    horizon: int,
) -> dict[str, object]:
    truth_interval = eol_interval(frame, threshold)
    observed_truth = frame[frame["measurement_observed"].eq(1)].dropna(
        subset=["raw_capacity"]
    )
    if observed_truth.empty:
        raise ValueError("No genuine reference-capacity observations")
    right_censor_cycle = int(observed_truth["cycle"].max())
    truth_support_end = (
        int(truth_interval[1]) if truth_interval is not None else right_censor_cycle
    )
    measured = frame[
        frame["measurement_observed"].eq(1)
        & frame["cycle"].gt(cutoff)
        & frame["cycle"].le(min(cutoff + horizon, truth_support_end))
    ][["cycle", "raw_capacity"]]
    aligned = measured.merge(forecast, on="cycle", how="inner")
    if aligned.empty:
        raise ValueError("No observed future reference-capacity support")
    residual = aligned["predicted_capacity"].to_numpy(float) - aligned[
        "raw_capacity"
    ].to_numpy(float)
    timing_score = censor_aware_timing_score(
        pred_eol,
        cutoff,
        horizon,
        truth_interval,
        right_censor_cycle,
    )
    result: dict[str, object] = {
        "truth_event_type": (
            "interval_observed" if truth_interval is not None else "right_censored"
        ),
        "truth_event_observed": int(truth_interval is not None),
        "right_censor_cycle": (
            np.nan if truth_interval is not None else right_censor_cycle
        ),
        "right_censored_rul_lower_bound_cycles": (
            np.nan
            if truth_interval is not None
            else max(0, right_censor_cycle - cutoff + 1)
        ),
        "pred_eol_cycle": np.nan if pred_eol is None else int(pred_eol),
        "prediction_censor_cycle": (
            int(cutoff + horizon) if pred_eol is None else np.nan
        ),
        "predicted_crossing_within_horizon": int(pred_eol is not None),
        "censor_aware_timing_score_cycles": timing_score,
        "future_capacity_MAE_Ah": float(np.mean(np.abs(residual))),
        "future_capacity_RMSE_Ah": float(np.sqrt(np.mean(residual**2))),
        "future_observed_points": len(aligned),
        "max_abs_capacity_error_Ah": float(np.max(np.abs(residual))),
    }
    if truth_interval is None:
        result.update(
            {
                "true_eol_interval_lower": np.nan,
                "true_eol_interval_upper": np.nan,
                "true_rul_interval_lower": np.nan,
                "true_rul_interval_upper": np.nan,
            }
        )
    else:
        eol_lower, eol_upper = truth_interval
        result.update(
            {
                "true_eol_interval_lower": eol_lower,
                "true_eol_interval_upper": eol_upper,
                "true_rul_interval_lower": max(0, eol_lower - cutoff),
                "true_rul_interval_upper": max(0, eol_upper - cutoff),
            }
        )
    return result


def conformal_order_statistic(scores: Sequence[float], coverage: float) -> float:
    values = np.sort(np.asarray(scores, dtype=float))
    if not len(values):
        raise ValueError("No calibration scores")
    rank = math.ceil((len(values) + 1) * float(coverage))
    rank = min(max(rank, 1), len(values))
    return float(values[rank - 1])


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device
