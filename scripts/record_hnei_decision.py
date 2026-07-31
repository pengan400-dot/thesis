#!/usr/bin/env python3
"""Record whether HNEI performance changed the frozen v0.2.0 specification."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--decision",
        choices=("unchanged", "modified"),
        required=True,
    )
    parser.add_argument("--hnei-results", type=Path, required=True)
    parser.add_argument("--aggregate-audit", type=Path, required=True)
    parser.add_argument("--development-freeze", type=Path, required=True)
    parser.add_argument("--rationale", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.rationale.strip():
        raise ValueError("A non-empty rationale is required")
    audit = json.loads(args.aggregate_audit.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS":
        raise RuntimeError("HNEI aggregate audit is not PASS")
    unchanged = args.decision == "unchanged"
    receipt = {
        "status": (
            "PASS_V0_2_0_UNCHANGED"
            if unchanged
            else "BLOCK_BIT_V0_2_0_REQUIRES_V0_2_1"
        ),
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "decision": args.decision,
        "rationale": args.rationale.strip(),
        "hnei_results_file": args.hnei_results.name,
        "hnei_results_sha256": sha256(args.hnei_results),
        "hnei_aggregate_audit_sha256": sha256(args.aggregate_audit),
        "development_freeze_file": args.development_freeze.name,
        "development_freeze_sha256": sha256(args.development_freeze),
        "hnei_used_to_select_or_change_v0_2_0": not unchanged,
        "bit_v0_2_0_structure_preflight_unlocked": unchanged,
        "required_next_version": None if unchanged else "v0.2.1",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[{receipt['status']}] {args.output}")


if __name__ == "__main__":
    main()
