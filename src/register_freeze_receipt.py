#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from common import sha256_file

COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")
DOI_RE = re.compile(r"^10\.\d{4,9}/zenodo\.\d+$", re.IGNORECASE)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", required=True)
    parser.add_argument("--doi", required=True)
    parser.add_argument("--freeze-zip", type=Path, required=True)
    parser.add_argument("--ready-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not COMMIT_RE.fullmatch(args.commit):
        raise ValueError("Commit must be a full 40-character hexadecimal hash")
    doi = args.doi.strip().removeprefix("https://doi.org/")
    if not DOI_RE.fullmatch(doi):
        raise ValueError("DOI must look like 10.xxxx/zenodo.xxxxxxxx")
    ready = json.loads(args.ready_json.read_text(encoding="utf-8"))
    if (
        ready.get("status") != "PASS"
        or ready.get("external_mat_payload_loaded") is not False
    ):
        raise RuntimeError("FREEZE_READY gate is not valid")
    with zipfile.ZipFile(args.freeze_zip) as archive:
        candidates = [
            name
            for name in archive.namelist()
            if name.endswith("/freeze_artifacts/FREEZE_READY.json")
        ]
        if len(candidates) != 1:
            raise RuntimeError("Freeze ZIP must contain exactly one FREEZE_READY.json")
        archived_ready = archive.read(candidates[0])
    if archived_ready != args.ready_json.read_bytes():
        raise RuntimeError(
            "Local FREEZE_READY.json differs from the immutable freeze ZIP"
        )
    payload = {
        "status": "PASS",
        "release_tag": "v0.1.0-freeze",
        "commit": args.commit.lower(),
        "zenodo_version_doi": doi.lower(),
        "freeze_zip": args.freeze_zip.name,
        "freeze_zip_sha256": sha256_file(args.freeze_zip),
        "registered_utc": datetime.now(timezone.utc).isoformat(),
        "external_opening_authorized": True,
        "statement": (
            "The immutable freeze release and version DOI existed before this "
            "workflow parsed Batch-4/5/6 MATLAB payloads."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[PASS] freeze receipt registered: {args.output}")


if __name__ == "__main__":
    main()
