#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from common import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the frozen Batch-4/5/6 7z archives without extracting or "
            "parsing MATLAB payloads."
        )
    )
    parser.add_argument("--shared-root", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def list_7z(path: Path) -> tuple[list[dict[str, object]], bool]:
    try:
        import py7zr
    except ImportError as exc:
        raise RuntimeError(
            "py7zr is required. Run `bash manager.sh setup` first."
        ) from exc
    with py7zr.SevenZipFile(path, mode="r") as archive:
        infos = archive.list()
    rows = []
    for info in infos:
        name = str(getattr(info, "filename", ""))
        is_directory = bool(getattr(info, "is_directory", False))
        if is_directory or not name:
            continue
        size = getattr(info, "uncompressed", None)
        rows.append(
            {
                "inner_path": name.replace("\\", "/"),
                "uncompressed_bytes": int(size) if size is not None else -1,
            }
        )
    with py7zr.SevenZipFile(path, mode="r") as archive:
        tested = archive.test()
    return rows, tested is not False


def main() -> int:
    args = parse_args()
    expected_rows = list(
        csv.DictReader(args.expected.open(encoding="utf-8", newline=""))
    )
    if not expected_rows:
        raise ValueError("Expected archive inventory is empty")

    expected_by_archive: dict[str, list[dict[str, str]]] = {}
    for row in expected_rows:
        expected_by_archive.setdefault(row["archive_name"], []).append(row)

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    observed_rows: list[dict[str, object]] = []
    failures: list[str] = []

    for archive_name, rows in expected_by_archive.items():
        archive_path = args.shared_root.resolve() / archive_name
        if not archive_path.is_file():
            failures.append(f"missing archive: {archive_path}")
            continue
        observed_hash = sha256_file(archive_path)
        expected_hashes = {row["archive_sha256"].lower() for row in rows}
        if expected_hashes != {observed_hash.lower()}:
            failures.append(
                f"{archive_name}: SHA-256 mismatch; observed {observed_hash}"
            )
            continue
        entries, integrity = list_7z(archive_path)
        if not integrity:
            failures.append(f"{archive_name}: 7z integrity test failed")
        expected_entries = {
            (row["inner_path"], int(row["uncompressed_bytes"])) for row in rows
        }
        observed_entries = {
            (str(row["inner_path"]), int(row["uncompressed_bytes"])) for row in entries
        }
        if observed_entries != expected_entries:
            failures.append(
                f"{archive_name}: entry mismatch; expected {sorted(expected_entries)}, "
                f"observed {sorted(observed_entries)}"
            )
        batch_by_inner = {row["inner_path"]: row["batch"] for row in rows}
        for entry in entries:
            observed_rows.append(
                {
                    "archive_name": archive_name,
                    "archive_path": str(archive_path),
                    "archive_sha256": observed_hash,
                    "integrity_test": "PASS" if integrity else "FAIL",
                    "inner_path": entry["inner_path"],
                    "uncompressed_bytes": entry["uncompressed_bytes"],
                    "batch": batch_by_inner.get(str(entry["inner_path"]), ""),
                    "mat_payload_parsed": "no",
                }
            )

    columns = [
        "archive_name",
        "archive_path",
        "archive_sha256",
        "integrity_test",
        "inner_path",
        "uncompressed_bytes",
        "batch",
        "mat_payload_parsed",
    ]
    with (output / "runtime_archive_inventory.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(observed_rows)

    expected_mat_count = sum(
        1 for row in expected_rows if row["inner_path"].endswith(".mat")
    )
    audit = {
        "status": "PASS" if not failures else "FAIL",
        "archives_expected": len(expected_by_archive),
        "archives_observed": len({str(row["archive_name"]) for row in observed_rows}),
        "mat_entries_expected": expected_mat_count,
        "mat_entries_observed": sum(
            str(row["inner_path"]).endswith(".mat") for row in observed_rows
        ),
        "mat_payload_parsed": False,
        "checks": {
            "outer_sha256": True if not failures else "see_failures",
            "7z_integrity": True if not failures else "see_failures",
            "inner_filename_and_size": True if not failures else "see_failures",
        },
        "failures": failures,
    }
    (output / "archive_gate.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if failures:
        print(json.dumps(audit, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    print(f"[PASS] {len(expected_by_archive)} archives and {expected_mat_count} MATLAB filenames verified")
    print("[PASS] No MATLAB payload was parsed")
    print(f"[OUT] {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
