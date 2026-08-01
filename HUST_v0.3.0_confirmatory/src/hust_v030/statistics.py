from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def conformal_order_statistic(scores: Sequence[float], coverage: float) -> float:
    values = np.sort(np.asarray(scores, dtype=float))
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError("No finite calibration scores")
    rank = math.ceil((len(values) + 1) * float(coverage))
    rank = min(max(rank, 1), len(values))
    return float(values[rank - 1])


def paired_bootstrap_interval(
    differences: Sequence[float],
    repetitions: int,
    seed: int,
) -> tuple[float, float]:
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return math.nan, math.nan
    generator = np.random.default_rng(int(seed))
    indices = generator.integers(
        0,
        len(values),
        size=(int(repetitions), len(values)),
    )
    means = values[indices].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)


def paired_effect(
    records: pd.DataFrame,
    reference: str,
    comparator: str,
    metric: str,
    repetitions: int,
    seed: int,
    *,
    perform_test: bool,
) -> dict[str, Any]:
    pivot = records.pivot(index="cell_id", columns="model", values=metric)
    pair = pivot[[reference, comparator]].dropna()
    differences = (
        pair[comparator].to_numpy(float) - pair[reference].to_numpy(float)
    )
    lower, upper = paired_bootstrap_interval(differences, repetitions, seed)
    statistic = math.nan
    p_value = math.nan
    if perform_test and len(differences):
        if np.allclose(differences, 0.0):
            statistic, p_value = 0.0, 1.0
        else:
            result = wilcoxon(
                differences,
                zero_method="wilcox",
                alternative="two-sided",
                method="auto",
            )
            statistic = float(result.statistic)
            p_value = float(result.pvalue)
    return {
        "reference_model": reference,
        "comparator_model": comparator,
        "metric": metric,
        "paired_physical_cells": len(pair),
        "effect_definition": "comparator minus reference; positive favors reference",
        "mean_paired_effect": (
            float(np.mean(differences)) if len(differences) else math.nan
        ),
        "median_paired_effect": (
            float(np.median(differences)) if len(differences) else math.nan
        ),
        "bootstrap_95ci_lower": lower,
        "bootstrap_95ci_upper": upper,
        "wilcoxon_test_performed": bool(perform_test),
        "wilcoxon_statistic": statistic,
        "two_sided_p": p_value,
    }


def fixed_sequence_confirmation(
    records: pd.DataFrame,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    specification = protocol["statistics"]
    minimum_cells = int(specification["minimum_eligible_confirmation_cells"])
    alpha = float(specification["alpha"])
    practical = float(specification["minimum_practical_mean_RMSE_improvement_SOH"])
    repetitions = int(specification["bootstrap_repetitions"])
    seed = int(specification["bootstrap_seed"])
    metric = "future_capacity_RMSE_SOH"
    required_models = [
        "mstt_full_K5",
        "local_linear_trend",
        "mstt_single_scale_K5",
    ]
    if not {"cell_id", "model", metric}.issubset(records.columns):
        complete = pd.DataFrame(columns=required_models)
    else:
        complete = records.pivot(
            index="cell_id",
            columns="model",
            values=metric,
        ).reindex(columns=required_models).dropna()
    if len(complete) < minimum_cells:
        return {
            "status": "NOT_ESTIMABLE_MINIMUM_CONFIRMATION_COHORT_NOT_MET",
            "minimum_required_cells": minimum_cells,
            "complete_paired_cells": len(complete),
            "H1_full_vs_local_linear": None,
            "H2_full_vs_single_scale": None,
            "claim_limit": "Feasibility and descriptive results only; the landmark and exclusions must not be changed.",
        }
    eligible_ids = set(complete.index.astype(str))
    eligible = records[records["cell_id"].astype(str).isin(eligible_ids)]
    h1 = paired_effect(
        eligible,
        "mstt_full_K5",
        "local_linear_trend",
        metric,
        repetitions,
        seed,
        perform_test=True,
    )
    h1_p_gate = bool(float(h1["two_sided_p"]) < alpha)
    h1_substantive = bool(
        h1_p_gate
        and float(h1["bootstrap_95ci_lower"]) > 0.0
        and float(h1["mean_paired_effect"]) >= practical
    )
    h1.update(
        {
            "alpha": alpha,
            "minimum_practical_mean_improvement_SOH": practical,
            "opens_H2_p_value_gate": h1_p_gate,
            "confirmatory_and_practical_success": h1_substantive,
        }
    )
    h2 = paired_effect(
        eligible,
        "mstt_full_K5",
        "mstt_single_scale_K5",
        metric,
        repetitions,
        seed + 1,
        perform_test=h1_p_gate,
    )
    h2_success = bool(
        h1_p_gate
        and float(h2["two_sided_p"]) < alpha
        and float(h2["bootstrap_95ci_lower"]) > 0.0
    )
    h2.update(
        {
            "gate_open": h1_p_gate,
            "confirmatory_success": h2_success,
            "if_gate_closed": (
                None
                if h1_p_gate
                else "Effect and bootstrap interval are descriptive; no H2 confirmatory p value was computed."
            ),
        }
    )
    return {
        "status": "ESTIMABLE",
        "minimum_required_cells": minimum_cells,
        "complete_paired_cells": len(complete),
        "H1_full_vs_local_linear": h1,
        "H2_full_vs_single_scale": h2,
    }


def model_summary(records: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "future_capacity_RMSE_SOH",
        "future_capacity_MAE_SOH",
        "censor_aware_timing_score_cycles",
        "threshold_hit",
    ]
    rows: list[dict[str, object]] = []
    for model, group in records.groupby("model"):
        row: dict[str, object] = {
            "model": model,
            "physical_cells": int(group["cell_id"].nunique()),
        }
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(float)
            row[f"{metric}_mean"] = float(np.mean(values)) if len(values) else math.nan
            row[f"{metric}_std"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else math.nan
            )
            row[f"{metric}_median"] = (
                float(np.median(values)) if len(values) else math.nan
            )
        rows.append(row)
    if not rows:
        columns = ["model", "physical_cells"]
        for metric in metrics:
            columns.extend(
                [
                    f"{metric}_mean",
                    f"{metric}_std",
                    f"{metric}_median",
                ]
            )
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).sort_values("model").reset_index(drop=True)
