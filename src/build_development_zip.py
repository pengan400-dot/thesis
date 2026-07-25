#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import pandas as pd

from common import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package the validated 31-cell Batch-1/2/3 CSV set."
    )
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--output-zip", type=Path, required=True)
    args = parser.parse_args()

    curves_dir = args.prepared_dir.resolve() / "curves"
    paths = sorted(curves_dir.glob("*.csv"))
    rows = []
    for path in paths:
        frame = pd.read_csv(path, nrows=1)
        required = {"battery_id", "batch", "cycle", "raw_capacity"}
        if not required.issubset(frame.columns):
            raise ValueError(f"{path}: missing standardized development columns")
        rows.append(
            {
                "path": path,
                "battery_id": str(frame["battery_id"].iloc[0]),
                "batch": int(frame["batch"].iloc[0]),
            }
        )
    counts = {batch: sum(row["batch"] == batch for row in rows) for batch in (1, 2, 3)}
    if len(rows) != 31 or counts != {1: 8, 2: 15, 3: 8}:
        raise RuntimeError(f"Expected 31 cells with 8/15/8 counts, got {counts}")

    output = args.output_zip.resolve()
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing development ZIP: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for row in rows:
            path = row["path"]
            archive.write(path, f"xjtu_q2_prepared_csv/{path.name}")
        audit = {
            "status": "PASS",
            "cell_counts": counts,
            "files": [
                {
                    "battery_id": row["battery_id"],
                    "batch": row["batch"],
                    "filename": row["path"].name,
                    "sha256": sha256_file(row["path"]),
                }
                for row in rows
            ],
        }
        archive.writestr(
            "xjtu_q2_prepared_csv/package_audit.json",
            json.dumps(audit, ensure_ascii=False, indent=2),
        )
    with zipfile.ZipFile(output) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"Development ZIP CRC failure at {bad}")
    digest = sha256_file(output)
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{digest}  {output.name}\n", encoding="utf-8"
    )
    print(f"[PASS] {output}")
    print(f"[SHA256] {digest}")


if __name__ == "__main__":
    main()
