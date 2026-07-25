from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from common import (
    assert_identical_support,
    build_window_sets,
    censor_aware_timing_score,
    conformal_order_statistic,
    curves_sha256,
    eol_interval,
    evaluate_forecast,
    support_frame,
)


def synthetic_curves() -> dict[str, pd.DataFrame]:
    curves = {}
    for batch in (1, 3):
        for cell_index in range(1, 9):
            cycles = np.arange(1, 121)
            raw = 2.05 - 0.0042 * cycles - 0.00001 * cell_index * cycles
            frame = pd.DataFrame(
                {
                    "cycle": cycles,
                    "raw_capacity": raw,
                    "capacity": pd.Series(raw).rolling(7, min_periods=1).mean(),
                    "battery_id": f"B{batch}_cell_{cell_index}",
                    "batch": batch,
                }
            )
            curves[f"B{batch}_cell_{cell_index}"] = frame
    return curves


class CommonSupportTests(unittest.TestCase):
    def test_k1_k5_support_is_identical(self):
        curves = synthetic_curves()
        train = sorted(cell for cell in curves if cell.startswith("B1_"))
        validation = sorted(cell for cell in curves if cell.startswith("B3_"))
        windows = build_window_sets(
            curves,
            train,
            validation,
            window=16,
            support_horizon=5,
            threshold=1.6,
        )
        frames = {
            "mstt_full_K1": support_frame(windows.train, "mstt_full_K1"),
            "mstt_full_K5": support_frame(windows.train, "mstt_full_K5"),
        }
        digest = assert_identical_support(frames)
        self.assertEqual(len(digest), 64)
        self.assertEqual(len(frames["mstt_full_K1"]), len(frames["mstt_full_K5"]))

    def test_curve_hash_changes_with_capacity_content(self):
        curves = synthetic_curves()
        original = curves_sha256(curves)
        curves["B1_cell_1"].loc[0, "raw_capacity"] += 0.001
        self.assertNotEqual(original, curves_sha256(curves))

    def test_conformal_finite_sample_rule(self):
        scores = np.arange(1, 17, dtype=float)
        self.assertEqual(conformal_order_statistic(scores, 0.8), 14.0)
        self.assertEqual(conformal_order_statistic(scores, 0.9), 16.0)


class IntervalMetricTests(unittest.TestCase):
    def test_sparse_eol_interval_and_distance(self):
        frame = pd.DataFrame(
            {
                "cycle": np.arange(1, 31),
                "raw_capacity": np.nan,
                "measurement_observed": 0,
                "capacity": 2.0,
            }
        )
        for cycle, value in ((1, 2.0), (11, 1.8), (21, 1.55)):
            frame.loc[frame["cycle"].eq(cycle), "raw_capacity"] = value
            frame.loc[frame["cycle"].eq(cycle), "measurement_observed"] = 1
        self.assertEqual(eol_interval(frame, 1.6), (12, 21))
        forecast = pd.DataFrame(
            {
                "cycle": np.arange(6, 26),
                "predicted_capacity": np.linspace(1.9, 1.5, 20),
            }
        )
        metrics = evaluate_forecast(
            frame,
            forecast,
            cutoff=5,
            pred_eol=18,
            threshold=1.6,
            horizon=20,
        )
        self.assertEqual(metrics["censor_aware_timing_score_cycles"], 0.0)
        self.assertEqual(metrics["future_observed_points"], 2)

    def test_right_censoring_is_retained_and_one_sided(self):
        frame = pd.DataFrame(
            {
                "cycle": np.arange(1, 31),
                "raw_capacity": np.nan,
                "measurement_observed": 0,
                "capacity": 2.0,
            }
        )
        for cycle, value in ((1, 2.0), (11, 1.9), (21, 1.8)):
            frame.loc[frame["cycle"].eq(cycle), "raw_capacity"] = value
            frame.loc[frame["cycle"].eq(cycle), "measurement_observed"] = 1
        forecast = pd.DataFrame(
            {
                "cycle": np.arange(6, 26),
                "predicted_capacity": np.linspace(1.9, 1.7, 20),
            }
        )
        metrics = evaluate_forecast(
            frame,
            forecast,
            cutoff=5,
            pred_eol=None,
            threshold=1.6,
            horizon=20,
        )
        self.assertEqual(metrics["truth_event_type"], "right_censored")
        self.assertEqual(metrics["censor_aware_timing_score_cycles"], 0.0)
        self.assertEqual(
            censor_aware_timing_score(
                pred_eol=18,
                cutoff=5,
                horizon=20,
                truth_interval=None,
                right_censor_cycle=21,
            ),
            4.0,
        )


if __name__ == "__main__":
    unittest.main()
