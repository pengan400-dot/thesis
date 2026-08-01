from __future__ import annotations

import unittest

import numpy as np

from mstt_soh.soh import bol_reference_from_values, capacity_to_soh


class SOHDefinitionTests(unittest.TestCase):
    def test_formula_direction_declines_with_capacity(self) -> None:
        soh, reference, values = capacity_to_soh(
            [2.00, 2.02, 1.98, 2.01, 1.99, 1.80, 1.60],
            nominal_capacity_Ah=2.0,
        )
        self.assertAlmostEqual(reference, 2.0)
        self.assertEqual(len(values), 5)
        self.assertAlmostEqual(soh[0], 1.0)
        self.assertAlmostEqual(soh[-1], 0.8)
        self.assertLess(soh[-1], soh[0])

    def test_later_values_never_replace_valid_first_five(self) -> None:
        values = [2.0, 2.01, 1.7, 2.02, 1.99, 2.0, 2.0, 2.0]
        reference, used = bol_reference_from_values(
            values,
            nominal_capacity_Ah=2.0,
        )
        self.assertEqual(used, values[:5])
        self.assertAlmostEqual(reference, float(np.median(values[:5])))

    def test_physically_invalid_value_is_not_a_reference_observation(self) -> None:
        reference, used = bol_reference_from_values(
            [2000.0, 2.0, 2.01, 1.99, 2.02, 1.98],
            nominal_capacity_Ah=2.0,
        )
        self.assertEqual(len(used), 5)
        self.assertNotIn(2000.0, used)
        self.assertAlmostEqual(reference, 2.0)

    def test_fewer_than_five_valid_values_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "required 5"):
            bol_reference_from_values(
                [2.0, 2.0, float("nan"), 2.0],
                nominal_capacity_Ah=2.0,
            )

    def test_pre_normalized_input_is_rejected_as_not_Ah(self) -> None:
        with self.assertRaisesRegex(ValueError, "first-five median"):
            bol_reference_from_values(
                [1.0, 1.01, 1.0, 1.0, 1.0],
                nominal_capacity_Ah=2.0,
            )


if __name__ == "__main__":
    unittest.main()
