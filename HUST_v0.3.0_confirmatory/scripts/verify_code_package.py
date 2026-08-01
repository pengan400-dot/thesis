#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "PACKAGE_MANIFEST.json"
EXCLUDED_PARTS = {
    "__pycache__",
    ".git",
    ".pytest_cache",
    ".test-mpl-cache",
    ".venv",
    "run_v030",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def current_files() -> dict[str, tuple[int, str]]:
    rows: dict[str, tuple[int, str]] = {}
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path == MANIFEST:
            continue
        relative = path.relative_to(ROOT)
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        if path.suffix in {".pyc", ".pyo"}:
            continue
        rows[relative.as_posix()] = (path.stat().st_size, digest(path))
    return rows


def main() -> None:
    if not MANIFEST.is_file():
        raise SystemExit("[FAIL] PACKAGE_MANIFEST.json is missing")
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    expected = {
        str(row["path"]): (int(row["size_bytes"]), str(row["sha256"]))
        for row in payload["files"]
    }
    observed = current_files()
    if expected != observed:
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        changed = sorted(
            name
            for name in set(expected) & set(observed)
            if expected[name] != observed[name]
        )
        raise SystemExit(
            "[FAIL] package integrity mismatch: "
            f"missing={missing}, extra={extra}, changed={changed}"
        )
    print(f"[PASS] code package integrity: {len(observed)} files")


if __name__ == "__main__":
    main()
