from __future__ import annotations

from itertools import product
from typing import Iterable

import numpy as np


def paired_bootstrap_mean_ci(
    differences: Iterable[float],
    *,
    repetitions: int = 10_000,
    seed: int = 20_260_727,
) -> tuple[float, float]:
    values = np.asarray(list(differences), dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Paired differences must be non-empty and finite")
    generator = np.random.default_rng(seed)
    indices = generator.integers(
        0,
        len(values),
        size=(int(repetitions), len(values)),
    )
    estimates = values[indices].mean(axis=1)
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return float(lower), float(upper)


def paired_sign_flip_test(
    differences: Iterable[float],
    *,
    exact_max_nonzero_pairs: int = 20,
    repetitions: int = 100_000,
    seed: int = 20_260_728,
) -> dict[str, float | int | str]:
    values = np.asarray(list(differences), dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Paired differences must be non-empty and finite")
    nonzero = values[values != 0.0]
    observed = abs(float(values.mean()))
    if not len(nonzero):
        return {
            "method": "all_zero",
            "n_pairs": len(values),
            "n_nonzero_pairs": 0,
            "observed_abs_mean": observed,
            "two_sided_p": 1.0,
            "randomizations": 1,
        }
    zero_count = len(values) - len(nonzero)
    denominator = len(values)
    if len(nonzero) <= exact_max_nonzero_pairs:
        extreme = 0
        total = 0
        for signs in product((-1.0, 1.0), repeat=len(nonzero)):
            randomized = float(
                np.dot(nonzero, np.asarray(signs, dtype=float))
                / denominator
            )
            extreme += int(abs(randomized) >= observed - 1e-15)
            total += 1
        p_value = extreme / total
        method = "exact_sign_flip"
    else:
        generator = np.random.default_rng(seed)
        extreme = 0
        total = int(repetitions)
        chunk = 10_000
        remaining = total
        while remaining:
            current = min(chunk, remaining)
            signs = generator.choice(
                np.asarray([-1.0, 1.0]),
                size=(current, len(nonzero)),
            )
            randomized = (signs @ nonzero) / denominator
            extreme += int(
                np.count_nonzero(
                    np.abs(randomized) >= observed - 1e-15
                )
            )
            remaining -= current
        p_value = (extreme + 1) / (total + 1)
        method = "monte_carlo_sign_flip_plus_one"
    return {
        "method": method,
        "n_pairs": len(values),
        "n_nonzero_pairs": len(nonzero),
        "n_zero_pairs": zero_count,
        "observed_abs_mean": observed,
        "two_sided_p": float(p_value),
        "randomizations": int(total),
    }


def fixed_sequence_gatekeeping(
    h1_differences: Iterable[float],
    h2_differences: Iterable[float],
    *,
    alpha: float = 0.05,
    bootstrap_repetitions: int = 10_000,
    bootstrap_seed: int = 20_260_727,
    permutation_repetitions: int = 100_000,
    permutation_seed: int = 20_260_728,
    exact_max_nonzero_pairs: int = 20,
) -> dict[str, object]:
    h1 = np.asarray(list(h1_differences), dtype=float)
    h2 = np.asarray(list(h2_differences), dtype=float)
    if not len(h1) or not len(h2):
        raise ValueError("H1 and H2 each require at least one complete pair")

    def summarize(values: np.ndarray, seed_offset: int) -> dict[str, object]:
        test = paired_sign_flip_test(
            values,
            exact_max_nonzero_pairs=exact_max_nonzero_pairs,
            repetitions=permutation_repetitions,
            seed=permutation_seed + seed_offset,
        )
        lower, upper = paired_bootstrap_mean_ci(
            values,
            repetitions=bootstrap_repetitions,
            seed=bootstrap_seed + seed_offset,
        )
        return {
            "n_cells": len(values),
            "mean_comparator_minus_full_RMSE": float(values.mean()),
            "median_comparator_minus_full_RMSE": float(
                np.median(values)
            ),
            "bootstrap_95ci": [lower, upper],
            **test,
        }

    h1_summary = summarize(h1, 0)
    h1_pass = bool(float(h1_summary["two_sided_p"]) < alpha)
    h2_summary = summarize(h2, 1)
    h2_summary["confirmatory_test_performed"] = h1_pass
    h2_summary["inference"] = (
        "confirmatory_fixed_sequence"
        if h1_pass
        else "descriptive_exploratory_gate_closed"
    )
    h2_pass = bool(
        h1_pass and float(h2_summary["two_sided_p"]) < alpha
    )
    if h1_pass and h2_pass:
        interpretation = (
            "Multiscale contribution confirmed and superiority to the "
            "local-linear deployment baseline confirmed."
        )
    elif h1_pass:
        interpretation = (
            "Multiscale structural contribution confirmed, but superiority "
            "to local linear was not established."
        )
    else:
        interpretation = (
            "BIT did not replicate the late-stage multiscale advantage."
        )
    return {
        "alpha": alpha,
        "difference_direction": (
            "positive comparator-minus-full RMSE favors full K5"
        ),
        "H1_full_vs_single_scale": {
            **h1_summary,
            "confirmatory_test_performed": True,
            "passes_alpha": h1_pass,
        },
        "H2_full_vs_local_linear": {
            **h2_summary,
            "passes_alpha_under_gate": h2_pass,
        },
        "holm_applied_to_H1_H2": False,
        "interpretation": interpretation,
    }
