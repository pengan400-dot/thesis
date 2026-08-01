from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

import numpy as np


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_safe(value: object) -> object:
    """Convert NumPy scalars and non-finite values to strict JSON values."""

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            _json_safe(payload),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def load_hust_protocol(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if payload.get("schema_version") != "3.0":
        raise ValueError("Unsupported HUST protocol schema")
    if payload.get("version_label") != "v0.3.0-HUST":
        raise ValueError("Unexpected HUST protocol version")
    dataset = payload["dataset"]
    if dataset.get("doi") != "10.17632/nsc7hnsg4s.2":
        raise ValueError("The registered HUST dataset DOI changed")
    if int(dataset.get("expected_physical_cells")) != 77:
        raise ValueError("The registered HUST physical-cell count changed")
    if (
        int(dataset.get("version")) != 2
        or str(dataset.get("raw_archive_name")) != "hust_data.zip"
        or str(dataset.get("expected_pickle_member_pattern")) != "our_data/*.pkl"
        or not math.isclose(
            float(dataset.get("nominal_capacity_Ah")),
            1.1,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("The registered HUST source definition changed")
    split = payload["split"]
    if (int(split["calibration_cells"]), int(split["confirmation_cells"])) != (
        20,
        57,
    ):
        raise ValueError("The registered 20/57 HUST split changed")
    if str(split.get("sha256_salt")) != "MSTT_RUL_v0.3.0_HUST_split_20260801":
        raise ValueError("The registered HUST split salt changed")
    if int(payload["task"]["forecast_horizon_cycles"]) != 700:
        raise ValueError("The registered recursive horizon changed")
    if int(payload["task"]["input_window_cycles"]) != 16:
        raise ValueError("The registered input window changed")
    if (
        int(payload["task"]["rolling_warmup_cycles"]) != 4
        or int(payload["task"]["causal_smoothing_window_cycles"]) != 7
        or [float(value) for value in payload["task"]["prediction_clip_SOH"]]
        != [0.2, 1.2]
    ):
        raise ValueError("The registered HUST task preprocessing changed")
    if int(payload["dynamic_landmark"]["minimum_observed_prefix_cycles"]) != 20:
        raise ValueError("The registered landmark history gate changed")
    landmark = payload["dynamic_landmark"]
    if (
        not math.isclose(
            float(landmark["lower_open_raw_SOH"]),
            0.8,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(landmark["upper_closed_raw_SOH"]),
            0.9,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or int(landmark["minimum_future_observations_for_RMSE"]) != 5
    ):
        raise ValueError("The registered dynamic landmark changed")
    if not math.isclose(
        float(payload["soh_definition"]["eol_threshold_raw_SOH"]),
        0.8,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("The registered SOH EOL threshold changed")
    if (
        str(payload["soh_definition"]["formula"])
        != "SOH_i_t = C_i_t / median(C_i_first_5_valid)"
        or int(payload["soh_definition"]["reference_observations"]) != 5
    ):
        raise ValueError("The registered SOH normalization changed")
    if [int(value) for value in payload["model_lineage"]["seeds"]] != [
        42,
        2024,
        3407,
    ]:
        raise ValueError("The registered random seeds changed")
    models = {str(item["id"]): item for item in payload["model_lineage"]["models"]}
    if set(models) != {
        "mstt_full_K5",
        "mstt_single_scale_K5",
        "local_linear_trend",
    }:
        raise ValueError("The registered model set changed")
    if (
        int(models["mstt_full_K5"]["expected_parameters"]) != 74405
        or int(models["mstt_single_scale_K5"]["expected_parameters"]) != 69789
        or int(models["local_linear_trend"]["lookback_physical_cycles"]) != 20
        or [float(value) for value in models["local_linear_trend"]["slope_clip_SOH_per_cycle"]]
        != [-0.02, -1e-6]
    ):
        raise ValueError("The registered model identity or comparator changed")
    calibration = payload["calibration"]
    if (
        [float(value) for value in calibration["nominal_simultaneous_coverages"]]
        != [0.8, 0.9]
        or int(calibration["minimum_eligible_calibration_cells_for_UQ"]) != 15
    ):
        raise ValueError("The registered HUST calibration rule changed")
    statistics = payload["statistics"]
    if (
        int(statistics["minimum_eligible_confirmation_cells"]) != 40
        or not math.isclose(
            float(statistics["alpha"]),
            0.05,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or int(statistics["bootstrap_repetitions"]) != 10000
        or int(statistics["bootstrap_seed"]) != 20260801
        or not math.isclose(
            float(statistics["minimum_practical_mean_RMSE_improvement_SOH"]),
            0.005,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or str(statistics["paired_test"])
        != "two-sided Wilcoxon signed-rank at the physical-cell level"
    ):
        raise ValueError("The registered HUST inferential rule changed")
    return payload


def normalized_zip_name(name: str) -> str:
    normalized = str(name).replace("\\", "/")
    path = PurePosixPath(normalized)
    if not normalized or normalized.startswith("/") or path.is_absolute():
        raise ValueError(f"Unsafe archive member: {name!r}")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"Unsafe archive member: {name!r}")
    return path.as_posix()


def stable_cell_order(cell_ids: Iterable[str], salt: str) -> list[tuple[str, str]]:
    rows = []
    for cell_id in cell_ids:
        digest = hashlib.sha256(
            str(salt).encode("utf-8") + b"\0" + str(cell_id).encode("utf-8")
        ).hexdigest()
        rows.append((str(cell_id), digest))
    return sorted(rows, key=lambda item: (item[1], item[0]))


def files_manifest(
    roots: Mapping[str, Path],
    *,
    include_suffixes: set[str] | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for prefix, root in sorted(roots.items()):
        root = Path(root)
        if root.is_file():
            candidates = [root]
            base = root.parent
        elif root.is_dir():
            candidates = sorted(path for path in root.rglob("*") if path.is_file())
            base = root
        else:
            raise FileNotFoundError(root)
        for path in candidates:
            if include_suffixes is not None and path.suffix not in include_suffixes:
                continue
            relative = path.name if root.is_file() else path.relative_to(base).as_posix()
            archive_path = PurePosixPath(prefix, relative).as_posix()
            rows.append(
                {
                    "archive_path": archive_path,
                    "source_path": str(path.resolve()),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    archive_paths = [str(row["archive_path"]) for row in rows]
    if len(archive_paths) != len(set(archive_paths)):
        raise ValueError("Duplicate archive paths in file manifest")
    return sorted(rows, key=lambda row: str(row["archive_path"]))


def assert_hash(path: Path, expected: str, label: str) -> None:
    observed = sha256_file(Path(path))
    if observed != str(expected):
        raise RuntimeError(
            f"{label} SHA-256 mismatch: expected={expected}, observed={observed}"
        )
