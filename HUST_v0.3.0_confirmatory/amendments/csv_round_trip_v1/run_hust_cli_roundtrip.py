from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd


AMENDMENT_ID = "HUST_CSV_ROUND_TRIP_PARSER_v1"

run_root_raw = os.environ.get("HUST_RUN_ROOT")

if not run_root_raw:
    raise RuntimeError(
        "HUST_RUN_ROOT must be defined before running the amendment wrapper"
    )

RUN_ROOT = Path(run_root_raw).resolve()
_original_read_csv = pd.read_csv
_applied_paths: set[str] = set()


def _is_prepared_hust_curve(value: object) -> bool:
    try:
        path = Path(os.fspath(value)).resolve()
        relative = path.relative_to(RUN_ROOT)
    except (TypeError, ValueError, OSError):
        return False

    parts = relative.parts

    return (
        len(parts) >= 3
        and parts[0] in {
            "04_hust_calibration",
            "confirmation",
        }
        and parts[1] == "curves"
        and path.name.startswith("HUST_")
        and path.suffix.lower() == ".csv"
    )


def _read_csv_round_trip(
    filepath_or_buffer,
    *args,
    **kwargs,
):
    if _is_prepared_hust_curve(filepath_or_buffer):
        supplied = kwargs.get("float_precision")

        if supplied not in (None, "round_trip"):
            raise RuntimeError(
                "Prepared HUST curves may only use "
                "float_precision='round_trip' under "
                f"{AMENDMENT_ID}; received {supplied!r}"
            )

        kwargs["float_precision"] = "round_trip"

        resolved = str(
            Path(os.fspath(filepath_or_buffer)).resolve()
        )

        if resolved not in _applied_paths:
            print(
                f"[AMENDMENT] {AMENDMENT_ID} "
                f"float_precision=round_trip path={resolved}",
                file=sys.stderr,
            )
            _applied_paths.add(resolved)

    return _original_read_csv(
        filepath_or_buffer,
        *args,
        **kwargs,
    )


pd.read_csv = _read_csv_round_trip

os.environ["HUST_IMPLEMENTATION_AMENDMENT"] = AMENDMENT_ID

from hust_v030.cli import main

main()
