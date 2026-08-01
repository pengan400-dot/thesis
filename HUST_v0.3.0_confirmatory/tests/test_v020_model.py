from __future__ import annotations

import unittest


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is installed by manager.sh setup")
class ModelIdentityTests(unittest.TestCase):
    def test_full_and_single_scale_parameter_counts(self) -> None:
        from mstt_soh.model_variants import (
            PaperMSTTVariant,
            identity_record,
        )

        full = identity_record(PaperMSTTVariant(7, ablation="full"))
        single = identity_record(
            PaperMSTTVariant(7, ablation="single_scale")
        )
        self.assertEqual(full["trainable_parameters"], 74_405)
        self.assertEqual(single["trainable_parameters"], 69_789)
        self.assertEqual(full["residual_scale_soh"], 0.04)

    def test_forward_shape(self) -> None:
        from mstt_soh.model_variants import PaperMSTTVariant

        model = PaperMSTTVariant(7)
        features = torch.zeros(2, 16, 7)
        soh = torch.ones(2, 16)
        prediction = model(features, soh)
        self.assertEqual(tuple(prediction.shape), (2,))


if __name__ == "__main__":
    unittest.main()
