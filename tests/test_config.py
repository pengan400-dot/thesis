from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class FrozenConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(
            (
                ROOT
                / "configs"
                / "soh_development_protocol_v0.2.0.json"
            ).read_text(encoding="utf-8")
        )

    def test_ah_scale_conversion(self) -> None:
        conversion = self.config["ah_to_soh_conversion"]
        reference = conversion["xjtu_reference_capacity_Ah"]
        self.assertEqual(
            conversion["legacy_residual_scale_Ah"] / reference,
            conversion["residual_scale_SOH"],
        )
        self.assertEqual(
            conversion["legacy_huber_delta_Ah"] / reference,
            conversion["huber_delta_SOH"],
        )
        self.assertEqual(
            conversion["legacy_threshold_weight_band_Ah"] / reference,
            conversion["threshold_weight_band_SOH"],
        )
        self.assertEqual(
            conversion["legacy_upward_tolerance_Ah"] / reference,
            conversion["upward_tolerance_SOH"],
        )

    def test_bit_primary_design(self) -> None:
        bit = self.config["bit"]
        self.assertEqual(bit["primary_cohort"], "arbitrary_use")
        self.assertEqual(bit["primary_cutoff"], 190)
        self.assertIn("single_scale", bit["H1_difference"])
        self.assertIn("local_linear", bit["H2_difference"])
        self.assertIn("No ordinary pooled", bit["cohort_pooling"])

    def test_unknown_future_inputs_are_forbidden(self) -> None:
        forbidden = self.config["task"]["unknown_future_forbidden_inputs"]
        self.assertIn("post-cutoff actual current schedule", forbidden)
        self.assertIn("true data termination position", forbidden)
        self.assertIn("future missingness pattern", forbidden)


if __name__ == "__main__":
    unittest.main()
