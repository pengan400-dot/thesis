#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from common import load_protocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def bootstrap_mean_ci(
    values: np.ndarray, repetitions: int, seed: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(repetitions, len(values)), replace=True)
    means = draws.mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def paired_test(
    frame: pd.DataFrame,
    reference: str,
    comparator: str,
    metric: str,
    repetitions: int,
    seed: int,
) -> dict[str, float | int | str]:
    left = frame[frame["model"].eq(reference)][["cell_id", metric]].rename(
        columns={metric: "reference"}
    )
    right = frame[frame["model"].eq(comparator)][["cell_id", metric]].rename(
        columns={metric: "comparator"}
    )
    paired = left.merge(right, on="cell_id", how="inner")
    delta = paired["comparator"].to_numpy(float) - paired["reference"].to_numpy(float)
    if not len(delta):
        raise ValueError(f"No paired cells for {reference} vs {comparator}")
    if np.allclose(delta, 0):
        p_value = 1.0
    else:
        p_value = float(
            wilcoxon(delta, alternative="two-sided", zero_method="wilcox").pvalue
        )
    low, high = bootstrap_mean_ci(delta, repetitions, seed)
    return {
        "reference_model": reference,
        "comparator_model": comparator,
        "metric": metric,
        "n_cells": len(delta),
        "reference_mean": float(paired["reference"].mean()),
        "comparator_mean": float(paired["comparator"].mean()),
        "paired_delta_comparator_minus_reference": float(np.mean(delta)),
        "bootstrap_ci95_low": low,
        "bootstrap_ci95_high": high,
        "wilcoxon_p": p_value,
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    count = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(count, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (count - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def summary_table(records: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "truth_event_observed",
        "predicted_crossing_within_horizon",
        "censor_aware_timing_score_cycles",
        "future_capacity_MAE_Ah",
        "future_capacity_RMSE_Ah",
        "future_observed_points",
    ]
    rows = []
    for keys, group in records.groupby(["batch", "cutoff", "model"], sort=True):
        row = {
            "batch": int(keys[0]),
            "cutoff": int(keys[1]),
            "model": keys[2],
            "n_cells": int(group["cell_id"].nunique()),
        }
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def plot_primary(summary: pd.DataFrame, output: Path, primary_cutoff: int) -> None:
    primary = summary[
        (summary["batch"] == 4) & (summary["cutoff"] == int(primary_cutoff))
    ].copy()
    order = [
        "mstt_single_scale_K5",
        "mstt_full_K5",
        "mstt_full_K1",
        "local_linear_trend",
        "persistence",
    ]
    if set(primary["model"].astype(str)) != set(order):
        raise RuntimeError(
            "Primary figure requires all five frozen models at the primary cutoff"
        )
    primary["model"] = pd.Categorical(primary["model"], order, ordered=True)
    primary = primary.sort_values("model")
    labels = [
        "MSTT single-scale K5",
        "MSTT full K5",
        "MSTT full K1",
        "Local linear",
        "Persistence",
    ]
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].bar(
        labels,
        primary["future_capacity_RMSE_Ah_mean"],
        color="#3b6ea8",
    )
    axes[0].set_ylabel("Future capacity RMSE (Ah)")
    axes[0].set_title(f"Batch-4, cutoff {primary_cutoff}")
    axes[1].bar(
        labels,
        primary["censor_aware_timing_score_cycles_mean"],
        color="#c46a35",
    )
    axes[1].set_ylabel("Censor-aware timing score (cycles)")
    axes[1].set_title(f"Batch-4, cutoff {primary_cutoff}")
    for axis in axes:
        axis.tick_params(axis="x", rotation=35)
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "batch4_cutoff190_primary_comparison.png", dpi=300)
    figure.savefig(output / "batch4_cutoff190_primary_comparison.pdf")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    protocol = load_protocol(args.config)
    evaluation_audit = json.loads(
        (args.evaluation_dir / "evaluation_audit.json").read_text(encoding="utf-8")
    )
    if evaluation_audit.get("status") != "PASS":
        raise RuntimeError("Evaluation audit is not PASS")
    records = pd.read_csv(args.evaluation_dir / "ensemble_cell_records.csv")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    summary = summary_table(records)
    summary.to_csv(output / "paper_summary_cell_first.csv", index=False)
    stats = protocol["statistics"]
    primary = stats["primary_family"]
    primary_frame = records[
        records["batch"].eq(4) & records["cutoff"].eq(int(primary["cutoff"]))
    ]
    primary_rows = []
    for comparator in primary["comparators"]:
        for metric in primary["metrics"]:
            primary_rows.append(
                paired_test(
                    primary_frame,
                    str(primary["reference_model"]),
                    str(comparator),
                    str(metric),
                    int(stats["bootstrap_repetitions"]),
                    int(stats["bootstrap_seed"]),
                )
            )
    adjusted = holm_adjust([float(row["wilcoxon_p"]) for row in primary_rows])
    for row, value in zip(primary_rows, adjusted):
        row["holm_p_primary_family"] = value
        row["favourable_direction"] = (
            "negative comparator-minus-reference means comparator is better; "
            "positive means frozen primary is better"
        )
    pd.DataFrame(primary_rows).to_csv(
        output / "primary_paired_effects_holm.csv", index=False
    )

    secondary_rows = []
    reference = str(primary["reference_model"])
    for (batch, cutoff), group in records.groupby(["batch", "cutoff"]):
        for comparator in sorted(set(group["model"]) - {reference}):
            for metric in (
                "future_capacity_RMSE_Ah",
                "censor_aware_timing_score_cycles",
            ):
                row = paired_test(
                    group,
                    reference,
                    comparator,
                    metric,
                    int(stats["bootstrap_repetitions"]),
                    int(stats["bootstrap_seed"]) + int(batch) * 1000 + int(cutoff),
                )
                row.update(
                    {
                        "batch": int(batch),
                        "cutoff": int(cutoff),
                        "analysis_status": (
                            "primary"
                            if batch == 4
                            and cutoff == int(primary["cutoff"])
                            and comparator in primary["comparators"]
                            else "secondary_or_exploratory"
                        ),
                    }
                )
                secondary_rows.append(row)
    pd.DataFrame(secondary_rows).to_csv(output / "all_paired_effects.csv", index=False)

    uq_columns = [
        column
        for column in records
        if column.endswith(
            (
                "_trajectory_band_covered",
                "_eol_band_truth_compatible",
                "_trajectory_band_half_width_Ah",
            )
        )
    ]
    primary_records = records[records["model"].eq(reference)]
    uq_rows = []
    for keys, group in primary_records.groupby(["batch", "cutoff"]):
        row = {
            "batch": int(keys[0]),
            "cutoff": int(keys[1]),
            "n_cells": int(group["cell_id"].nunique()),
        }
        for column in uq_columns:
            row[f"{column}_mean"] = float(group[column].mean())
        uq_rows.append(row)
    pd.DataFrame(uq_rows).to_csv(output / "conformal_coverage_summary.csv", index=False)
    plot_primary(summary, output, int(primary["cutoff"]))

    audit = {
        "status": "PASS",
        "config_sha256": evaluation_audit["config_sha256"],
        "external_data_sha256": evaluation_audit["external_data_sha256"],
        "cell_first_records": len(records),
        "primary_comparisons": len(primary_rows),
        "primary_holm_family_size": 6,
        "bootstrap_repetitions": int(stats["bootstrap_repetitions"]),
        "physical_cell_is_statistical_unit": True,
        "batches_reported_separately": True,
        "right_censored_cells_retained": True,
        "primary_minimum_paired_cells": int(
            min(row["n_cells"] for row in primary_rows)
        ),
        "primary_inference_underpowered": bool(
            min(row["n_cells"] for row in primary_rows) < 5
        ),
    }
    (output / "aggregation_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[PASS] cell-first aggregation and primary Holm family complete")
    print(f"[OUT] {output}")


if __name__ == "__main__":
    main()
