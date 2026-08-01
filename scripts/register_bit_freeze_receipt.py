#!/usr/bin/env python3
"""Record immutable registration for the v0.2.1 BIT freeze."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import zipfile

COMMIT = re.compile(r"^[0-9a-fA-F]{40}$")
ZENODO = re.compile(r"^10\.\d{4,9}/zenodo\.\d+$", re.IGNORECASE)
GITHUB_RELEASE = re.compile(
    r"^https://github\.com/[^/]+/[^/]+/releases/tag/[^/]+$",
    re.IGNORECASE,
)
OSF = re.compile(
    r"^https://(?:www\.)?osf\.io/[a-z0-9]+/?$",
    re.IGNORECASE,
)
EXPECTED_VERSION = "v0.2.1-BIT-structural-feasibility-amendment"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_doi(value: str) -> str:
    return (
        value.strip()
        .removeprefix("https://doi.org/")
        .removeprefix("http://doi.org/")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-zip", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--github-release", required=True)
    parser.add_argument("--osf-registration", required=True)
    parser.add_argument("--zenodo-doi", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path)
    args = parser.parse_args()

    commit = args.commit.strip()
    github_release = args.github_release.strip()
    osf_registration = args.osf_registration.strip()
    zenodo_doi = normalize_doi(args.zenodo_doi)

    if not COMMIT.fullmatch(commit):
        raise ValueError("Commit must be the complete 40-character hash")
    if not GITHUB_RELEASE.fullmatch(github_release):
        raise ValueError("A concrete GitHub release-tag URL is required")
    if not OSF.fullmatch(osf_registration):
        raise ValueError("A concrete immutable OSF registration URL is required")
    if not ZENODO.fullmatch(zenodo_doi):
        raise ValueError("A Zenodo specific-version DOI is required")
    if args.evaluation_dir and args.evaluation_dir.exists() and any(
        args.evaluation_dir.rglob("*")
    ):
        raise RuntimeError(
            "BIT evaluation output already exists; registration is too late"
        )

    manifest = json.loads(
        args.freeze_manifest.read_text(encoding="utf-8")
    )
    if (
        manifest.get("status") != "PASS"
        or manifest.get("version") != EXPECTED_VERSION
        or manifest.get("primary_landmark") != 70
        or sorted(manifest.get("structurally_unavailable_landmarks", []))
        != [130, 190]
        or manifest.get("bit_capacity_opened_before_freeze") is not False
        or manifest.get("bit_model_output_generated_before_freeze") is not False
    ):
        raise RuntimeError("v0.2.1 pre-BIT freeze manifest is invalid")

    with zipfile.ZipFile(args.freeze_zip) as archive:
        archived = json.loads(
            archive.read(
                "PRE_BIT_MODEL_EVALUATION_FREEZE.json"
            ).decode("utf-8")
        )
        bad = archive.testzip()
    if bad is not None or archived != manifest:
        raise RuntimeError("Freeze ZIP/manifest integrity gate failed")

    receipt = {
        "status": "PASS",
        "registered_utc": datetime.now(timezone.utc).isoformat(),
        "version": EXPECTED_VERSION,
        "freeze_zip": args.freeze_zip.name,
        "freeze_zip_sha256": sha256(args.freeze_zip),
        "freeze_manifest_sha256": sha256(args.freeze_manifest),
        "commit": commit.lower(),
        "github_release": github_release,
        "osf_registration": osf_registration,
        "zenodo_specific_version_doi": zenodo_doi.lower(),
        "governance_label": (
            "post-BIT-structure-only, pre-model-evaluation amendment"
        ),
        "wording": (
            "immutably frozen before any BIT model evaluation; not an "
            "untouched confirmation"
        ),
        "primary_landmark": 70,
        "structurally_unavailable_landmarks": [130, 190],
        "bit_capacity_opened_before_receipt": False,
        "bit_model_output_existed_before_receipt": False,
        "bit_evaluation_unlocked": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[PASS] BIT v0.2.1 registration receipt: {args.output}")


if __name__ == "__main__":
    main()
