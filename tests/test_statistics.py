from __future__ import annotations

import unittest

from mstt_soh.statistics import (
    fixed_sequence_gatekeeping,
    paired_sign_flip_test,
)


class StatisticsTests(unittest.TestCase):
    def test_exact_sign_flip(self) -> None:
        result = paired_sign_flip_test([1.0, 1.0, 1.0])
        self.assertEqual(result["method"], "exact_sign_flip")
        self.assertAlmostEqual(float(result["two_sided_p"]), 0.25)

    def test_zeros_are_retained_but_do_not_change_sign_count(self) -> None:
        result = paired_sign_flip_test([1.0, 0.0, -1.0, 0.0])
        self.assertEqual(result["n_pairs"], 4)
        self.assertEqual(result["n_nonzero_pairs"], 2)
        self.assertEqual(result["n_zero_pairs"], 2)

    def test_h2_gate_closes_when_h1_fails(self) -> None:
        result = fixed_sequence_gatekeeping(
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            bootstrap_repetitions=100,
        )
        h2 = result["H2_full_vs_local_linear"]
        self.assertFalse(h2["confirmatory_test_performed"])
        self.assertEqual(
            h2["inference"],
            "descriptive_exploratory_gate_closed",
        )
        self.assertFalse(result["holm_applied_to_H1_H2"])

    def test_h1_and_h2_may_have_different_pair_counts(self) -> None:
        result = fixed_sequence_gatekeeping(
            [0.1, 0.2, 0.3],
            [0.05, 0.1],
            bootstrap_repetitions=100,
        )
        self.assertEqual(
            result["H1_full_vs_single_scale"]["n_cells"],
            3,
        )
        self.assertEqual(
            result["H2_full_vs_local_linear"]["n_cells"],
            2,
        )


if __name__ == "__main__":
    unittest.main()
