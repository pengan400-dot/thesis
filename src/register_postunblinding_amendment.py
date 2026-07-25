#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")
DOI_RE = re.compile(r"^10\.\d{4,9}/zenodo\.\d+$", re.IGNORECASE)


def sha256_file(path: Path) -> str:
    digest = __import__("hashlib").sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", required=True)
    parser.add_argument("--doi", required=True)
    parser.add_argument("--original-freeze-receipt", type=Path, required=True)
    parser.add_argument("--original-failed-preflight", type=Path, required=True)
    parser.add_argument("--semantics-summary", type=Path, required=True)
    parser.add_argument("--original-freeze-zip", type=Path, required=True)
    parser.add_argument("--evaluation-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not COMMIT_RE.fullmatch(args.commit):
        raise ValueError("Commit must be a full 40-character hexadecimal hash")

    doi = args.doi.strip().removeprefix("https://doi.org/")
    if not DOI_RE.fullmatch(doi):
        raise ValueError("DOI must look like 10.xxxx/zenodo.xxxxxxxx")

    if args.evaluation_audit.exists():
        raise RuntimeError(
            "Model evaluation already exists. This amendment receipt must be "
            "registered before any external model-performance evaluation."
        )

    freeze_receipt = json.loads(
        args.original_freeze_receipt.read_text(encoding="utf-8")
    )
    if (
        freeze_receipt.get("status") != "PASS"
        or freeze_receipt.get("external_opening_authorized") is not True
    ):
        raise RuntimeError("Original freeze receipt is invalid")
    if sha256_file(args.original_freeze_zip) != freeze_receipt.get(
        "freeze_zip_sha256"
    ):
        raise RuntimeError("Original freeze ZIP no longer matches its receipt")

    failed = json.loads(
        args.original_failed_preflight.read_text(encoding="utf-8")
    )
    if (
        failed.get("status") != "FAIL"
        or failed.get("batch_counts") != {"4": 8, "5": 8, "6": 8}
        or failed.get("batch4_reference_capacity_observed_every_cycle")
        is not False
        or failed.get("failures") != []
    ):
        raise RuntimeError("Original preflight is not the expected Batch-4 gate failure")

    summary = pd.read_csv(args.semantics_summary)
    required = {
        "file",
        "cycle_life_metadata",
        "reference_capacity_tests",
        "first_reference_record",
        "last_reference_record",
        "min_reference_gap_records",
        "median_reference_gap_records",
        "max_reference_gap_records",
        "unique_reference_gaps",
        "all_reference_gaps_equal_6",
    }
    if len(summary) != 8 or not required.issubset(summary.columns):
        raise RuntimeError("Batch-4 semantics summary is incomplete")
    if not (
        summary["first_reference_record"].eq(1).all()
        and summary["last_reference_record"].eq(
            summary["cycle_life_metadata"]
        ).all()
        and summary["min_reference_gap_records"].eq(6).all()
        and summary["median_reference_gap_records"].eq(6).all()
        and summary["max_reference_gap_records"].eq(6).all()
        and summary["all_reference_gaps_equal_6"].astype(bool).all()
    ):
        raise RuntimeError("Batch-4 exact period-6 reference pattern was not verified")

    payload = {
        "status": "PASS",
        "amendment_class": "post_unblinding_pre_model_evaluation",
        "release_tag": "v0.1.1-postunblinding-pre-evaluation-amendment",
        "commit": args.commit.lower(),
        "zenodo_version_doi": doi.lower(),
        "registered_utc": datetime.now(timezone.utc).isoformat(),
        "original_freeze_receipt": freeze_receipt,
        "original_failed_preflight_sha256": sha256_file(
            args.original_failed_preflight
        ),
        "batch4_semantics_summary_sha256": sha256_file(
            args.semantics_summary
        ),
        "original_freeze_zip_sha256": sha256_file(
            args.original_freeze_zip
        ),
        "external_data_structure_parsed_before_amendment": True,
        "external_capacity_threshold_crossing_count_seen_before_amendment": True,
        "model_performance_evaluated_before_amendment": False,
        "model_weights_changed": False,
        "hyperparameters_changed": False,
        "cutoffs_changed": False,
        "statistical_plan_changed": False,
        "changed_item": (
            "Batch-4 reference-capacity observation semantics and the "
            "external-preflight gate only"
        ),
        "batch4_verified_pattern": (
            "One genuine reference-capacity test at MATLAB records "
            "1, 7, 13, ... through the final record; exact gap 6 in all 8 cells"
        ),
        "claim_limit": (
            "Results after this amendment are post-unblinding amended external "
            "validation, not untouched preregistered confirmation."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[PASS] post-unblinding amendment receipt registered: {args.output}")


if __name__ == "__main__":
    main()
