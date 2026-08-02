#!/usr/bin/env python3
"""Independently recompute HUST confirmation summaries from the final result ZIP.

This script does not import the training or evaluator package. It reads the
archived protocol, cell-level records, calibration scores, and published
aggregate outputs, then reproduces the statistical summaries independently.
"""
from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

EXPECTED_RESULT_ZIP_SHA256 = (
    "b694230e3ed124f6a6d9f1aaf9dbf147"
    "15d64605a11bb33fb6de1ecce5cadf9c"
)
EXPECTED_RUN_UUID = "ca19cabf-061c-4e6a-8ac0-8d41072f627c"


def sha256_path(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(archive: zipfile.ZipFile, name: str) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(archive.read(name)), float_precision="round_trip")


def read_json(archive: zipfile.ZipFile, name: str) -> dict:
    return json.loads(archive.read(name).decode("utf-8"))


def json_dump(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


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
            row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else math.nan
            row[f"{metric}_median"] = float(np.median(values)) if len(values) else math.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("model").reset_index(drop=True)


def bootstrap_interval(values: np.ndarray, repetitions: int, seed: int) -> tuple[float, float]:
    generator = np.random.default_rng(int(seed))
    indices = generator.integers(0, len(values), size=(int(repetitions), len(values)))
    means = values[indices].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)


def paired_effect(
    records: pd.DataFrame,
    reference: str,
    comparator: str,
    repetitions: int,
    seed: int,
    perform_test: bool,
) -> dict[str, object]:
    metric = "future_capacity_RMSE_SOH"
    pivot = records.pivot(index="cell_id", columns="model", values=metric)
    pair = pivot[[reference, comparator]].dropna()
    differences = pair[comparator].to_numpy(float) - pair[reference].to_numpy(float)
    lower, upper = bootstrap_interval(differences, repetitions, seed)
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
        "mean_paired_effect": float(np.mean(differences)),
        "median_paired_effect": float(np.median(differences)),
        "bootstrap_95ci_lower": lower,
        "bootstrap_95ci_upper": upper,
        "wilcoxon_test_performed": bool(perform_test),
        "wilcoxon_statistic": statistic,
        "two_sided_p": p_value,
    }


def fixed_sequence(records: pd.DataFrame, protocol: dict) -> dict:
    specification = protocol["statistics"]
    minimum = int(specification["minimum_eligible_confirmation_cells"])
    alpha = float(specification["alpha"])
    practical = float(specification["minimum_practical_mean_RMSE_improvement_SOH"])
    repetitions = int(specification["bootstrap_repetitions"])
    seed = int(specification["bootstrap_seed"])
    required = ["mstt_full_K5", "local_linear_trend", "mstt_single_scale_K5"]
    complete = (
        records.pivot(index="cell_id", columns="model", values="future_capacity_RMSE_SOH")
        .reindex(columns=required)
        .dropna()
    )
    if len(complete) < minimum:
        return {
            "status": "NOT_ESTIMABLE_MINIMUM_CONFIRMATION_COHORT_NOT_MET",
            "minimum_required_cells": minimum,
            "complete_paired_cells": len(complete),
            "H1_full_vs_local_linear": None,
            "H2_full_vs_single_scale": None,
        }

    eligible = records[records["cell_id"].astype(str).isin(set(complete.index.astype(str)))]
    h1 = paired_effect(
        eligible,
        "mstt_full_K5",
        "local_linear_trend",
        repetitions,
        seed,
        True,
    )
    h1_gate = bool(float(h1["two_sided_p"]) < alpha)
    h1_success = bool(
        h1_gate
        and float(h1["bootstrap_95ci_lower"]) > 0.0
        and float(h1["mean_paired_effect"]) >= practical
    )
    h1.update(
        {
            "alpha": alpha,
            "minimum_practical_mean_improvement_SOH": practical,
            "opens_H2_p_value_gate": h1_gate,
            "confirmatory_and_practical_success": h1_success,
        }
    )

    h2 = paired_effect(
        eligible,
        "mstt_full_K5",
        "mstt_single_scale_K5",
        repetitions,
        seed + 1,
        h1_gate,
    )
    h2_success = bool(
        h1_gate
        and float(h2["two_sided_p"]) < alpha
        and float(h2["bootstrap_95ci_lower"]) > 0.0
    )
    h2.update(
        {
            "gate_open": h1_gate,
            "confirmatory_success": h2_success,
            "if_gate_closed": None if h1_gate else (
                "Effect and bootstrap interval are descriptive; "
                "no H2 confirmatory p value was computed."
            ),
        }
    )
    return {
        "status": "ESTIMABLE",
        "minimum_required_cells": minimum,
        "complete_paired_cells": len(complete),
        "H1_full_vs_local_linear": h1,
        "H2_full_vs_single_scale": h2,
    }


def coverage_summary(records: pd.DataFrame, protocol: dict) -> pd.DataFrame:
    full = records[records["model"].eq("mstt_full_K5")]
    rows = []
    for coverage in protocol["calibration"]["nominal_simultaneous_coverages"]:
        prefix = f"coverage_{int(round(float(coverage) * 100))}"
        simultaneous = pd.to_numeric(
            full[f"{prefix}_simultaneous_trajectory_covered"], errors="coerce"
        ).dropna()
        rows.append(
            {
                "nominal_simultaneous_coverage": float(coverage),
                "eligible_cells": len(simultaneous),
                "empirical_simultaneous_coverage": float(simultaneous.mean()),
                "mean_pointwise_coverage": float(
                    pd.to_numeric(full[f"{prefix}_pointwise_coverage"], errors="coerce").mean()
                ),
                "mean_interval_width_SOH": float(
                    pd.to_numeric(full[f"{prefix}_mean_interval_width_SOH"], errors="coerce").mean()
                ),
                "eol_band_compatibility_rate": float(
                    pd.to_numeric(
                        full[f"{prefix}_eol_band_truth_compatible"], errors="coerce"
                    ).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def conformal_quantiles(scores: pd.DataFrame, protocol: dict) -> dict[str, float]:
    values = np.sort(
        pd.to_numeric(scores["nonconformity_max_abs_SOH"], errors="coerce")
        .dropna()
        .to_numpy(float)
    )
    result: dict[str, float] = {}
    for coverage in protocol["calibration"]["nominal_simultaneous_coverages"]:
        rank = math.ceil((len(values) + 1) * float(coverage))
        rank = min(max(rank, 1), len(values))
        result[str(float(coverage))] = float(values[rank - 1])
    return result


def assert_frame_close(name: str, computed: pd.DataFrame, archived: pd.DataFrame) -> None:
    if list(computed.columns) != list(archived.columns):
        raise RuntimeError(f"{name}: columns differ")
    if len(computed) != len(archived):
        raise RuntimeError(f"{name}: row count differs")
    for column in computed.columns:
        left = computed[column]
        right = archived[column]
        if pd.api.types.is_numeric_dtype(left) or pd.api.types.is_numeric_dtype(right):
            a = pd.to_numeric(left, errors="coerce").to_numpy(float)
            b = pd.to_numeric(right, errors="coerce").to_numpy(float)
            if not np.allclose(a, b, rtol=1e-12, atol=1e-12, equal_nan=True):
                raise RuntimeError(f"{name}: numeric mismatch in {column}")
        elif left.astype(str).tolist() != right.astype(str).tolist():
            raise RuntimeError(f"{name}: text mismatch in {column}")


def assert_nested_close(name: str, computed: object, archived: object, path: str = "") -> None:
    label = f"{name}{path}"
    if isinstance(computed, dict) and isinstance(archived, dict):
        if set(computed) != set(archived):
            raise RuntimeError(f"{label}: key set differs")
        for key in computed:
            assert_nested_close(name, computed[key], archived[key], f"{path}.{key}")
        return
    if isinstance(computed, list) and isinstance(archived, list):
        if len(computed) != len(archived):
            raise RuntimeError(f"{label}: list length differs")
        for index, (left, right) in enumerate(zip(computed, archived)):
            assert_nested_close(name, left, right, f"{path}[{index}]")
        return
    if isinstance(computed, (int, float)) and isinstance(archived, (int, float)):
        if math.isnan(float(computed)) and math.isnan(float(archived)):
            return
        if not math.isclose(float(computed), float(archived), rel_tol=1e-12, abs_tol=1e-12):
            raise RuntimeError(f"{label}: {computed!r} != {archived!r}")
        return
    if computed != archived:
        raise RuntimeError(f"{label}: {computed!r} != {archived!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_zip", type=Path)
    parser.add_argument("--output", type=Path, default=Path("recomputed"))
    parser.add_argument("--allow-different-outer-zip-hash", action="store_true")
    args = parser.parse_args()

    result_zip = args.result_zip.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    outer_hash = sha256_path(result_zip)
    if outer_hash != EXPECTED_RESULT_ZIP_SHA256 and not args.allow_different_outer_zip_hash:
        raise SystemExit(f"[STOP] outer ZIP hash differs: {outer_hash}")

    with zipfile.ZipFile(result_zip, "r") as archive:
        protocol = read_json(archive, "protocol/hust_confirmatory_protocol_v0.3.0.json")
        state = read_json(archive, "confirmation/one_shot_state.json")
        records = read_csv(archive, "confirmation/ensemble_cell_records.csv")
        calibration_scores = read_csv(archive, "calibration/calibration_scores.csv")
        archived_summary = read_csv(
            archive, "aggregate/hust_confirmatory_model_summary.csv"
        )
        archived_fixed = read_json(
            archive, "aggregate/hust_fixed_sequence_inference.json"
        )
        archived_coverage = read_csv(
            archive, "aggregate/hust_calibration_coverage_summary.csv"
        )
        archived_quantiles = read_json(
            archive, "calibration/calibration_quantiles.json"
        )

    if state.get("status") != "PASS":
        raise SystemExit("[STOP] archived confirmation state is not PASS")
    if state.get("run_uuid") != EXPECTED_RUN_UUID:
        raise SystemExit("[STOP] archived run UUID differs")
    if records.duplicated(["cell_id", "model"]).any():
        raise SystemExit("[STOP] duplicate cell/model records")
    if len(records) != 171 or records["cell_id"].nunique() != 57:
        raise SystemExit("[STOP] expected 171 records from 57 physical cells")

    computed_summary = model_summary(records)
    computed_fixed = fixed_sequence(records, protocol)
    computed_coverage = coverage_summary(records, protocol)
    computed_quantiles = conformal_quantiles(calibration_scores, protocol)

    assert_frame_close("model_summary", computed_summary, archived_summary)
    assert_nested_close("fixed_sequence", computed_fixed, archived_fixed)
    assert_frame_close("coverage_summary", computed_coverage, archived_coverage)
    assert_nested_close(
        "calibration_quantiles", computed_quantiles, archived_quantiles["quantiles"]
    )

    computed_summary.to_csv(output / "recomputed_model_summary.csv", index=False)
    json_dump(output / "recomputed_fixed_sequence_inference.json", computed_fixed)
    computed_coverage.to_csv(
        output / "recomputed_calibration_coverage_summary.csv", index=False
    )
    json_dump(
        output / "recomputed_calibration_quantiles.json",
        {"quantiles": computed_quantiles},
    )
    report = {
        "status": "PASS",
        "source_result_zip": str(result_zip),
        "source_result_zip_sha256": outer_hash,
        "run_uuid": state["run_uuid"],
        "physical_cells": int(records["cell_id"].nunique()),
        "model_records": int(len(records)),
        "calibration_cells": int(len(calibration_scores)),
        "H1_supported": bool(
            computed_fixed["H1_full_vs_local_linear"][
                "confirmatory_and_practical_success"
            ]
        ),
        "H2_supported": bool(
            computed_fixed["H2_full_vs_single_scale"]["confirmatory_success"]
        ),
        "checks": {
            "model_summary_matches_archive": True,
            "fixed_sequence_matches_archive": True,
            "coverage_summary_matches_archive": True,
            "calibration_quantiles_match_archive": True,
        },
    }
    json_dump(output / "RECOMPUTATION_REPORT.json", report)

    print("[PASS] independent HUST recomputation matches archived outputs")
    print(f"result_zip_sha256={outer_hash}")
    print(f"physical_cells={report['physical_cells']}")
    print(f"model_records={report['model_records']}")
    print(f"calibration_cells={report['calibration_cells']}")
    print(f"output={output}")


if __name__ == "__main__":
    main()
