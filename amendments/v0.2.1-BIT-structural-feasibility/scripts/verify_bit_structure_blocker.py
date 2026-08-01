#!/usr/bin/env python3
"""Guard against freezing the unchanged cycle-190 BIT protocol."""

from __future__ import annotations

import json
from pathlib import Path
import sys

def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python verify_bit_structure_blocker.py <bit_structural_preflight.json>")

    path = Path(sys.argv[1])
    data = json.loads(path.read_text(encoding="utf-8"))

    assert data.get("status") == "STRUCTURE_ONLY_PASS", data
    assert data.get("model_output_generated") is False, data
    assert data.get("soh_computed") is False, data
    assert data.get("eol_computed") is False, data
    assert data.get("rmse_computed") is False, data

    print("[PASS] Structure-only preflight identity verified")
    print("[BLOCK] Unchanged physical-cycle-190 BIT freeze is not permitted")
    print("[NEXT] Freeze a separately registered structural-feasibility amendment before any model output")
    raise SystemExit(3)

if __name__ == "__main__":
    main()
