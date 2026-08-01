from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT / "scripts" / "bit_v021_evaluator.py"
SPEC = importlib.util.spec_from_file_location("bit_v021_evaluator", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class BitV021EvaluatorContractTests(unittest.TestCase):
    def test_frozen_identity(self) -> None:
        self.assertEqual(MODULE.PRIMARY_CUTOFF, 70)
        self.assertEqual(MODULE.STRUCTURALLY_UNAVAILABLE_CUTOFFS, [130, 190])
        self.assertEqual(MODULE.EXPECTED_INCLUDED_CELLS, 72)
        self.assertEqual(MODULE.EXCLUDED_CELL_IDS, ["BIT_#2"])
        self.assertEqual(
            MODULE.MODEL_IDS,
            ["mstt_full_K5", "mstt_single_scale_K5", "local_linear_trend"],
        )
        self.assertEqual(MODULE.SEEDS, [42, 2024, 3407])

    def test_prepare_cell_is_causal_and_soh_native(self) -> None:
        cycles = np.arange(1, 81)
        capacities = 2.4 - 0.006 * (cycles - 1)
        raw = pd.DataFrame(
            {
                "cell_id": "BIT_#1",
                "cell_number": 1,
                "cohort": "arbitrary_use",
                "physical_cycle": cycles,
                "raw_discharge_capacity_Ah": capacities,
                "capacity_quality_pass": True,
            }
        )
        frame, audit = MODULE.prepare_cell_frame(raw, smoothing_window=7)
        self.assertEqual(len(frame), 80)
        self.assertAlmostEqual(audit["bol_reference_capacity_Ah"], np.median(capacities[:5]))
        self.assertFalse(audit["backward_fill_used"])
        self.assertFalse(audit["future_interpolation_used"])
        self.assertAlmostEqual(
            frame.loc[0, "raw_capacity"], capacities[0] / np.median(capacities[:5])
        )
        self.assertAlmostEqual(
            frame.loc[6, "capacity"],
            frame.loc[:6, "raw_capacity"].mean(),
        )

    def test_missing_cycle_uses_only_past(self) -> None:
        cycles = np.arange(1, 81)
        capacities = 2.4 - 0.005 * (cycles - 1)
        values = capacities.astype(object)
        values[10] = np.nan
        raw = pd.DataFrame(
            {
                "cell_id": "BIT_#3",
                "cell_number": 3,
                "cohort": "fixed_profile",
                "physical_cycle": cycles,
                "raw_discharge_capacity_Ah": values,
                "capacity_quality_pass": [i != 10 for i in range(80)],
            }
        )
        frame, _ = MODULE.prepare_cell_frame(raw, smoothing_window=1)
        self.assertEqual(frame.loc[10, "measurement_observed"], 0)
        self.assertAlmostEqual(frame.loc[10, "capacity"], frame.loc[9, "raw_capacity"])
        self.assertNotAlmostEqual(frame.loc[10, "capacity"], frame.loc[11, "raw_capacity"])

    def test_metric_keeps_rmse_and_timing_sets_separate(self) -> None:
        cycles = np.arange(1, 81)
        frame = pd.DataFrame(
            {
                "cycle": cycles,
                "raw_capacity": np.linspace(1.0, 0.85, len(cycles)),
                "capacity": np.linspace(1.0, 0.85, len(cycles)),
                "measurement_observed": 1,
            }
        )
        forecast = pd.DataFrame(
            {
                "cycle": np.arange(71, 771),
                "predicted_capacity": np.linspace(0.9, 0.2, 700),
            }
        )
        record = MODULE.metric_record(
            frame,
            forecast,
            cutoff=70,
            predicted_eol=170,
            threshold=0.8,
            horizon=700,
            minimum_future=5,
        )
        self.assertTrue(record["rmse_evaluable"])
        self.assertEqual(record["truth_event_type"], "right_censored")
        self.assertGreaterEqual(record["censor_aware_timing_score_cycles"], 0.0)

    def test_no_network_and_no_pooled_p_value_contract(self) -> None:
        text = MODULE_PATH.read_text(encoding="utf-8")
        self.assertNotIn("requests.", text)
        self.assertNotIn("httpx.", text)
        self.assertIn("ordinary_pooled_p_value_computed", text)
        self.assertIn("fixed_sequence_gatekeeping", text)
        self.assertNotIn("primary_cutoff = 190", text.lower())


if __name__ == "__main__":
    unittest.main()
