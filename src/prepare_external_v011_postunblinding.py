#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--shared-root", type=Path, required=True)
    parser.add_argument("--archive-inventory", type=Path, required=True)
    parser.add_argument("--original-freeze-receipt", type=Path, required=True)
    parser.add_argument("--amendment-receipt", type=Path, required=True)
    parser.add_argument("--freeze-zip", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project = args.project.resolve()
    sys.path.insert(0, str(project / "src"))

    from common import json_sha256, load_protocol, sha256_file
    from prepare_external import (
        batch_from_filename,
        extract_archives,
        prepare_one,
        validate_receipt,
    )

    protocol = load_protocol(args.config)
    config_hash = json_sha256(protocol)
    threshold = float(protocol["task"]["eol_threshold_Ah"])
    smoothing_window = int(
        protocol["task"]["causal_smoothing_window_cycles"]
    )

    validate_receipt(args.original_freeze_receipt, args.freeze_zip)

    amendment = json.loads(
        args.amendment_receipt.read_text(encoding="utf-8")
    )
    if (
        amendment.get("status") != "PASS"
        or amendment.get("amendment_class")
        != "post_unblinding_pre_model_evaluation"
        or amendment.get("model_performance_evaluated_before_amendment")
        is not False
        or amendment.get("model_weights_changed") is not False
        or amendment.get("cutoffs_changed") is not False
    ):
        raise RuntimeError("Post-unblinding amendment receipt is invalid")
    if sha256_file(args.freeze_zip) != amendment.get(
        "original_freeze_zip_sha256"
    ):
        raise RuntimeError("Freeze ZIP does not match amendment receipt")

    output = args.output_dir.resolve()
    preflight = output / "external_preflight.json"
    if preflight.is_file() and not args.overwrite:
        old = json.loads(preflight.read_text(encoding="utf-8"))
        if old.get("status") == "PASS":
            print(f"[SKIP] amended external preparation already passed: {output}")
            return
        raise RuntimeError(
            "Existing amended preparation is incomplete; use --overwrite"
        )

    if args.overwrite:
        if args.raw_dir.exists():
            shutil.rmtree(args.raw_dir)
        if output.exists():
            shutil.rmtree(output)

    inventory_rows = list(
        csv.DictReader(
            args.archive_inventory.open(encoding="utf-8", newline="")
        )
    )
    extract_archives(
        args.shared_root.resolve(),
        inventory_rows,
        args.raw_dir.resolve(),
    )

    expected_paths = {
        Path(row["inner_path"]).name: {
            "batch": int(row["batch"].split("-")[1]),
            "size": int(row["uncompressed_bytes"]),
        }
        for row in inventory_rows
    }
    extracted: dict[str, Path] = {}
    for path in args.raw_dir.rglob("*.mat"):
        if path.name in expected_paths:
            if path.name in extracted:
                raise RuntimeError(f"Duplicate extracted file: {path.name}")
            extracted[path.name] = path
    if set(extracted) != set(expected_paths):
        raise RuntimeError(
            "Extracted filename mismatch; "
            f"missing={sorted(set(expected_paths) - set(extracted))}, "
            f"extra={sorted(set(extracted) - set(expected_paths))}"
        )
    for name, path in extracted.items():
        if path.stat().st_size != expected_paths[name]["size"]:
            raise RuntimeError(f"{name}: extracted byte size mismatch")

    curves_dir = output / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)
    audit_rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []

    for name in sorted(extracted):
        path = extracted[name]
        batch = batch_from_filename(name)
        try:
            frame, audit = prepare_one(
                path, batch, threshold, smoothing_window
            )
            reference_cycles = frame.loc[
                frame["measurement_observed"].eq(1), "cycle"
            ].to_numpy(dtype=int)
            gaps = np.diff(reference_cycles)
            audit["min_reference_gap_cycles"] = (
                int(gaps.min()) if len(gaps) else 0
            )
            audit["median_reference_gap_cycles"] = (
                float(np.median(gaps)) if len(gaps) else 0.0
            )
            audit["reference_gap_pattern"] = ",".join(
                map(str, sorted(set(gaps.tolist())))
            )
            audit["all_reference_gaps_equal_6"] = bool(
                len(gaps) and np.all(gaps == 6)
            )
            frame.to_csv(
                curves_dir / f"{audit['battery_id']}.csv",
                index=False,
            )
            audit_rows.append(audit)
            print(
                f"[PARSED] {name}: references="
                f"{audit['reference_capacity_tests']} "
                f"gap_pattern={audit['reference_gap_pattern']}"
            )
        except Exception as exc:
            failures.append({"file": name, "error": str(exc)})

    audit = pd.DataFrame(audit_rows)
    if not audit.empty:
        audit = audit.sort_values(["batch", "battery_id"])
        audit.to_csv(output / "external_curve_audit.csv", index=False)

    batch_counts = {
        batch: int((audit["batch"] == batch).sum())
        if not audit.empty
        else 0
        for batch in (4, 5, 6)
    }
    batch4 = audit[audit["batch"].eq(4)] if not audit.empty else audit

    batch4_exact_period6 = bool(
        len(batch4) == 8
        and batch4["first_reference_cycle"].eq(1).all()
        and batch4["last_reference_cycle"].eq(
            batch4["cycle_life_metadata"]
        ).all()
        and batch4["min_reference_gap_cycles"].eq(6).all()
        and batch4["median_reference_gap_cycles"].eq(6).all()
        and batch4["max_reference_gap_cycles"].eq(6).all()
        and batch4["all_reference_gaps_equal_6"].astype(bool).all()
    )

    observed_eol_count = (
        int(audit["has_observed_eol_crossing"].sum())
        if not audit.empty
        else 0
    )
    status = (
        "PASS"
        if (
            not failures
            and batch_counts == {4: 8, 5: 8, 6: 8}
            and batch4_exact_period6
        )
        else "FAIL"
    )

    gate = {
        "status": status,
        "analysis_class": "post_unblinding_amended_external_validation",
        "config_sha256": config_hash,
        "original_freeze_receipt": json.loads(
            args.original_freeze_receipt.read_text(encoding="utf-8")
        ),
        "postunblinding_amendment_receipt": amendment,
        "batch_counts": batch_counts,
        "original_v010_preflight_outcome": "FAIL",
        "original_every_cycle_assumption_rejected": True,
        "batch4_reference_capacity_pattern": (
            "exactly every 6 MATLAB cycle records, beginning at record 1 "
            "and ending at cycle_life_metadata"
        ),
        "batch4_exact_period6_reference_pattern_verified": batch4_exact_period6,
        "observed_fixed_threshold_eol_crossings": observed_eol_count,
        "right_censored_at_last_reference_observation": 24
        - observed_eol_count,
        "cells_without_observed_fixed_threshold_crossing_retained": True,
        "sparse_input_fill": "causal_LOCF",
        "future_interpolation_used": False,
        "model_weights_changed_after_unblinding": False,
        "test_selection_or_retraining": False,
        "failures": failures,
        "claim_limit": (
            "Post-unblinding pre-model-evaluation amendment; not untouched "
            "preregistered confirmation."
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    preflight.write_text(
        json.dumps(gate, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if status != "PASS":
        print(json.dumps(gate, ensure_ascii=False, indent=2), file=sys.stderr)
        raise SystemExit(2)
    print("[PASS] Amended external curves prepared")
    print(f"[OUT] {output}")


if __name__ == "__main__":
    main()
