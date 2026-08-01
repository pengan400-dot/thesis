from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def bol_reference_from_values(
    values: Iterable[float],
    *,
    nominal_capacity_Ah: float,
    count: int = 5,
    reference_relative_bounds: tuple[float, float] = (0.8, 1.2),
) -> tuple[float, list[float]]:
    """Return the median of the chronologically first valid Ah values."""

    lower = 0.5 * float(nominal_capacity_Ah)
    upper = 1.5 * float(nominal_capacity_Ah)
    valid: list[float] = []
    for value in values:
        numeric = float(value)
        if math.isfinite(numeric) and lower <= numeric <= upper:
            valid.append(numeric)
            if len(valid) == count:
                break
    if len(valid) < count:
        raise ValueError(
            f"BOL quality gate failed: required {count} valid Ah values "
            f"in [{lower:g}, {upper:g}], observed {len(valid)}"
        )
    reference = float(np.median(valid))
    reference_lower = (
        float(reference_relative_bounds[0]) * float(nominal_capacity_Ah)
    )
    reference_upper = (
        float(reference_relative_bounds[1]) * float(nominal_capacity_Ah)
    )
    if not reference_lower <= reference <= reference_upper:
        raise ValueError(
            "BOL quality gate failed: first-five median "
            f"{reference:g} Ah lies outside "
            f"[{reference_lower:g}, {reference_upper:g}] Ah"
        )
    return reference, valid


def capacity_to_soh(
    capacities_Ah: Iterable[float],
    *,
    nominal_capacity_Ah: float,
    reference_count: int = 5,
) -> tuple[np.ndarray, float, list[float]]:
    capacities = np.asarray(list(capacities_Ah), dtype=float)
    reference, values = bol_reference_from_values(
        capacities,
        nominal_capacity_Ah=nominal_capacity_Ah,
        count=reference_count,
    )
    return capacities / reference, reference, values
