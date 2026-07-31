#!/usr/bin/env python3
"""Verify the frozen protocol and analytical model counts without PyTorch."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def local_parameter_count(kernels: list[int]) -> int:
    branch_count = len(kernels)
    branches = sum(7 * kernel + 57 for kernel in kernels)
    projection = 64 * (5 * branch_count) + 64 + 2 * 64
    return branches + projection


def common_parameter_count() -> int:
    attention = (64 * 192 + 192) + (64 * 64 + 64)
    feed_forward = (64 * 128 + 128) + (128 * 64 + 64)
    encoder_block = 2 * 64 + attention + 2 * 64 + feed_forward
    head = 2 * 64 + (64 * 32 + 32) + (32 * 1 + 1)
    return 2 * encoder_block + head


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    full = local_parameter_count(
        [kernel for kernel in (3, 5, 7, 9) for _ in (1, 2, 4)]
    ) + common_parameter_count()
    single = local_parameter_count([5]) + common_parameter_count()
    expected = config["source_implementation"]
    checks = {
        "formula_direction": config["soh_definition"]["formula"]
        == "SOH_i_t = C_i_t / median(C_i_first_5_valid)",
        "full_parameters": full
        == int(expected["full_model_trainable_parameters"])
        == 74_405,
        "single_parameters": single
        == int(expected["single_scale_trainable_parameters"])
        == 69_789,
        "residual_scale": math.isclose(
            float(config["training"]["residual_scale_SOH"]),
            0.04,
        ),
        "huber_delta": math.isclose(
            float(config["training"]["huber_delta_SOH"]),
            0.01,
        ),
        "eol_weight_band": math.isclose(
            float(config["training"]["threshold_weight_band_SOH"]),
            0.04,
        ),
        "upward_tolerance": math.isclose(
            float(config["training"]["upward_tolerance_SOH"]),
            0.001,
        ),
        "physical_clip": list(
            map(float, config["task"]["prediction_clip_SOH"])
        )
        == [0.2, 1.2],
        "minimum_history": int(
            config["task"]["minimum_pre_cutoff_physical_cycles"]
        )
        == 20,
        "minimum_future_rmse_support": int(
            config["task"]["minimum_future_observations_for_RMSE"]
        )
        == 5,
        "bit_not_evaluated": config["status"]
        == "DRAFT_TO_BE_FROZEN_BEFORE_ANY_BIT_MODEL_OUTPUT",
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Frozen contract checks failed: {failed}")
    print(
        "[PASS] frozen contract: "
        f"full={full}, single_scale={single}, target=SOH"
    )


if __name__ == "__main__":
    main()
