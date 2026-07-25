#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

CYCLE_CANDIDATES = (
    "cycle",
    "cycle_index",
    "cycle_number",
    "Cycle",
    "Cycle_Index",
    "cyc",
)
CAPACITY_CANDIDATES = (
    "raw_capacity",
    "capacity_raw",
    "discharge_capacity",
    "capacity",
    "Capacity",
    "capacity_ah",
    "Capacity(Ah)",
    "Qd",
)
CELL_CANDIDATES = ("cell_id", "cell", "battery", "battery_id")
BATCH_CANDIDATES = ("batch", "batch_id")


def first_present(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    cols = list(columns)
    lower = {str(c).strip().lower(): c for c in cols}
    for name in candidates:
        if name in cols:
            return name
        if name.lower() in lower:
            return str(lower[name.lower()])
    return None


def infer_batch(text: str) -> int | None:
    low = text.lower().replace("\\", "/")
    patterns = [
        r"(?:^|[^a-z0-9])batch[_\- ]?([123])(?:[^0-9]|$)",
        r"(?:^|[^a-z0-9])b([123])[_\-]",
        r"(?:^|[^a-z0-9])batch([123])(?:[^0-9]|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, low)
        if match:
            return int(match.group(1))
    return None


def infer_cell(path: Path, frame: pd.DataFrame, batch: int) -> str:
    col = first_present(frame.columns, CELL_CANDIDATES)
    if col and frame[col].notna().any():
        value = str(frame[col].dropna().iloc[0]).strip()
    else:
        value = path.stem
    if re.match(r"^B[123]_", value, flags=re.IGNORECASE):
        return value
    return f"B{batch}_{value}"


def causal_smooth(values: np.ndarray, window: int) -> np.ndarray:
    return (
        pd.Series(np.asarray(values, dtype=float))
        .rolling(window=window, min_periods=1)
        .mean()
        .to_numpy(float)
    )


def protocol_hash(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: Path) -> list[tuple[Path, int, str]]:
    frame = pd.read_csv(path)
    required = {"path", "batch", "cell_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    return [
        (Path(row.path), int(row.batch), str(row.cell_id))
        for row in frame.itertuples(index=False)
    ]


def scan_files(root: Path) -> list[tuple[Path, int | None, str | None]]:
    return [(path, None, None) for path in sorted(root.rglob("*.csv"))]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert the 31 XJTU per-cell CSVs into the paper v4 prepared format."
    )
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoothing-window", type=int, default=7)
    args = parser.parse_args()

    if bool(args.input_root) == bool(args.manifest):
        parser.error("Provide exactly one of --input-root or --manifest")

    specs = (
        read_manifest(args.manifest) if args.manifest else scan_files(args.input_root)
    )

    accepted: dict[tuple[int, str], dict[str, object]] = {}
    failures: list[str] = []

    for path, explicit_batch, explicit_cell in specs:
        try:
            frame = pd.read_csv(path)
            cycle_col = first_present(frame.columns, CYCLE_CANDIDATES)
            capacity_col = first_present(frame.columns, CAPACITY_CANDIDATES)
            if cycle_col is None or capacity_col is None:
                raise ValueError("missing cycle/capacity columns")

            batch = explicit_batch
            if batch is None:
                batch_col = first_present(frame.columns, BATCH_CANDIDATES)
                if batch_col and frame[batch_col].notna().any():
                    raw = str(frame[batch_col].dropna().iloc[0])
                    match = re.search(r"([123])", raw)
                    batch = int(match.group(1)) if match else None
            if batch is None:
                batch = infer_batch(str(path))
            if batch not in (1, 2, 3):
                raise ValueError("cannot infer batch 1/2/3")

            work = frame[[cycle_col, capacity_col]].copy()
            work.columns = ["cycle", "raw_capacity"]
            work["cycle"] = pd.to_numeric(work["cycle"], errors="coerce")
            work["raw_capacity"] = pd.to_numeric(work["raw_capacity"], errors="coerce")
            work = work.dropna()
            work = work[work["raw_capacity"].between(0.2, 3.0, inclusive="neither")]
            work = (
                work.groupby("cycle", as_index=False)["raw_capacity"]
                .median()
                .sort_values("cycle")
                .reset_index(drop=True)
            )
            work["cycle"] = work["cycle"].round().astype(int)
            if len(work) < 40:
                raise ValueError(f"only {len(work)} usable cycles")
            if int(work.iloc[0]["cycle"]) != 1:
                raise ValueError("first usable cycle is not 1")
            if np.any(np.diff(work["cycle"].to_numpy(int)) <= 0):
                raise ValueError("cycle is not strictly increasing")

            cell = explicit_cell or infer_cell(path, frame, batch)
            key = (batch, cell)
            if key in accepted:
                raise ValueError(f"duplicate cell key {key}")

            q0 = float(work.loc[work["cycle"].eq(1), "raw_capacity"].iloc[0])
            work["capacity"] = causal_smooth(
                work["raw_capacity"].to_numpy(float), args.smoothing_window
            )
            work["soh_raw"] = work["raw_capacity"] / q0
            work["soh_model"] = work["capacity"] / q0
            work["battery_id"] = cell
            work["batch"] = int(batch)

            cycles = work["cycle"].to_numpy(int)
            contiguous = bool(
                np.array_equal(cycles, np.arange(cycles[0], cycles[-1] + 1))
            )
            hits = work.loc[work["raw_capacity"] <= 1.6, "cycle"]
            raw_eol = int(hits.iloc[0]) if len(hits) else None

            accepted[key] = {
                "path": path.resolve(),
                "frame": work,
                "capacity_source_column": capacity_col,
                "raw_eol": raw_eol,
                "contiguous": contiguous,
                "q0": q0,
            }
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{path}: {exc}")

    counts = {
        batch: sum(1 for (value, _) in accepted if value == batch)
        for batch in (1, 2, 3)
    }
    expected = {1: 8, 2: 15, 3: 8}
    if counts != expected:
        detail = "\n".join(failures[:30])
        raise RuntimeError(
            f"Expected batch counts {expected}, got {counts}. First failures:\n{detail}"
        )

    output = args.output_dir.resolve()
    curves_dir = output / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)

    audit_rows: list[dict[str, object]] = []
    for (batch, cell), item in sorted(accepted.items()):
        frame = item["frame"]
        frame.to_csv(curves_dir / f"{cell}.csv", index=False)
        audit_rows.append(
            {
                "battery_id": cell,
                "batch": batch,
                "source_path": str(item["path"]),
                "source_sha256": file_sha256(item["path"]),
                "capacity_source_column": item["capacity_source_column"],
                "n_cycles": len(frame),
                "first_cycle": int(frame["cycle"].iloc[0]),
                "last_cycle": int(frame["cycle"].iloc[-1]),
                "initial_capacity_Ah": item["q0"],
                "raw_observed_eol_cycle": item["raw_eol"],
                "has_raw_direct_crossing": int(item["raw_eol"] is not None),
                "cycles_contiguous": int(item["contiguous"]),
            }
        )

    audit = pd.DataFrame(audit_rows).sort_values(["batch", "battery_id"])
    audit.to_csv(output / "xjtu_prepared_csv_audit.csv", index=False)
    (output / "scan_failures.txt").write_text("\n".join(failures), encoding="utf-8")

    development = audit[audit["batch"].isin([1, 3])]
    test = audit[audit["batch"].eq(2)]
    censored = development.loc[
        development["has_raw_direct_crossing"].eq(0), "battery_id"
    ].tolist()
    observed = development.loc[
        development["has_raw_direct_crossing"].eq(1), "battery_id"
    ].tolist()

    protocol: dict[str, object] = {
        "protocol_name": "XJTU Q2 priority-A paper-faithful supplement",
        "development_batches": [1, 3],
        "epoch_selection_train_batch": 1,
        "epoch_selection_validation_batch": 3,
        "external_test_batch": 2,
        "nominal_capacity_Ah": 2.0,
        "eol_threshold_Ah": 1.6,
        "eol_definition": "first RAW observed capacity <= 1.6 Ah",
        "model_capacity_target": f"causal rolling mean, window={args.smoothing_window}",
        "fixed_absolute_cutoff_cycles": [70, 130, 190],
        "window": 16,
        "multi_step_rollout_steps": 10,
        "max_recursive_horizon": 700,
        "seeds": [11, 42, 73, 2024, 3407],
        "development_observed_eol_cells": sorted(observed),
        "development_right_censored_cells": sorted(censored),
        "capacity_channel_scaling": "physical Ah restored after train-only StandardScaler",
        "test_data_used_for_training_or_scaler": False,
        "source_contract": "31 previously prepared per-cell CSV curves",
    }
    protocol["protocol_sha256"] = protocol_hash(protocol)
    (output / "frozen_protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    gate = {
        "status": "PASS",
        "batch_counts": counts,
        "external_test_cells": len(test),
        "external_test_raw_direct_crossings": int(
            test["has_raw_direct_crossing"].sum()
        ),
        "external_test_contiguous_curves": int(test["cycles_contiguous"].sum()),
        "primary_observed_eol_gate_passed": bool(
            counts == expected
            and test["has_raw_direct_crossing"].eq(1).all()
            and test["cycles_contiguous"].eq(1).all()
        ),
        "protocol_sha256": protocol["protocol_sha256"],
        "note": (
            "The input is the previously audited per-cell CSV export; MATLAB "
            "cycle_life metadata is not re-derived in this conversion."
        ),
    }
    if not gate["primary_observed_eol_gate_passed"]:
        raise RuntimeError(f"Prepared-data gate failed: {gate}")
    (output / "preflight_gate.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("[PASS] paper prepared curves written")
    print(f"output={output}")
    print(f"batch_counts={counts}")
    print(f"B2 direct crossings={gate['external_test_raw_direct_crossings']}/15")
    print(f"protocol_sha256={protocol['protocol_sha256']}")


if __name__ == "__main__":
    main()
