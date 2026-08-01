from __future__ import annotations

import json
import math
import pickle
import os
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


# Source parsing itself does not require torch. Stub only causal_smooth so the
# structure/parser tests can run in a lightweight pre-install environment.
pipeline_stub = types.ModuleType("mstt_soh.pipeline")
pipeline_stub.causal_smooth = lambda values, window: (
    pd.Series(np.asarray(values, dtype=float))
    .rolling(window=max(1, int(window)), min_periods=1)
    .mean()
    .to_numpy(float)
)
for stub_name in (
    "choose_device",
    "eol_interval",
    "evaluate_forecast",
    "first_predicted_eol",
    "load_frozen_models",
    "load_protocol",
    "merge_seed_forecasts",
    "recursive_forecast",
):
    setattr(pipeline_stub, stub_name, lambda *args, **kwargs: None)
sys.modules.setdefault("mstt_soh.pipeline", pipeline_stub)

torch_stub = types.ModuleType("torch")
torch_stub.device = lambda value: value
torch_stub.nn = types.SimpleNamespace(Module=object)
torch_stub.Tensor = type("Tensor", (), {})
sys.modules.setdefault("torch", torch_stub)

from hust_v030.common import (  # noqa: E402
    json_sha256,
    load_hust_protocol,
    read_json,
    sha256_file,
    write_json,
)
from hust_v030.evaluation import _records_frame, aggregate_confirmation  # noqa: E402
from hust_v030.source import (  # noqa: E402
    discharge_capacity_Ah,
    find_dynamic_landmark,
    extract_xjtu_b1_b3,
    inventory_source,
    parse_hust_cell,
)
from hust_v030.statistics import (  # noqa: E402
    conformal_order_statistic,
    fixed_sequence_confirmation,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "hust_confirmatory_protocol_v0.3.0.json"
PROTOCOL = json.loads(CONFIG.read_text(encoding="utf-8"))


def cycle_frame(capacity_Ah: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Time (s)": [0.0, 3600.0],
            "Current (mA)": [0.0, -1000.0 * float(capacity_Ah)],
            "Voltage (V)": [3.3, 3.1],
        }
    )


def synthetic_cell(cell_id: str) -> bytes:
    capacities = {
        cycle: (
            1.10
            if cycle <= 19
            else {
                20: 0.99,
                21: 0.97,
                22: 0.95,
                23: 0.93,
                24: 0.91,
                25: 0.87,
            }.get(cycle, 0.86)
        )
        for cycle in range(1, 31)
    }
    payload = {
        cell_id: {
            "data": {
                cycle: cycle_frame(capacity)
                for cycle, capacity in capacities.items()
            }
        }
    }
    return pickle.dumps(payload, protocol=4)


class StrictJsonTests(unittest.TestCase):
    def test_nonfinite_numpy_values_become_null(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "value.json"
            write_json(path, {"a": np.float64(np.nan), "b": np.int64(7)})
            self.assertEqual(read_json(path), {"a": None, "b": 7})

    def test_critical_protocol_mutation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mutated.json"
            payload = json.loads(CONFIG.read_text(encoding="utf-8"))
            payload["statistics"]["minimum_eligible_confirmation_cells"] = 39
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "inferential rule"):
                load_hust_protocol(path)


class StructureTests(unittest.TestCase):
    def test_bundled_xjtu_extractor_reads_only_b1_b3(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "xjtu"
            extract_xjtu_b1_b3(
                ROOT / "inputs" / "xjtu_q2_prepared_csv.zip",
                output,
            )
            names = sorted(path.name for path in (output / "curves").glob("*.csv"))
            self.assertEqual(len(names), 16)
            self.assertTrue(all(name.startswith(("B1_", "B3_")) for name in names))
            receipt = read_json(output / "xjtu_input_receipt.json")
            self.assertFalse(receipt["external_hust_data_loaded"])

    def test_inventory_is_deterministic_20_57_and_does_not_unpickle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "hust_data.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                for index in range(1, 78):
                    handle.writestr(f"our_data/1-{index}.pkl", b"not-a-pickle")
            first = root / "first"
            second = root / "second"
            inventory_source(archive, CONFIG, first)
            inventory_source(archive, CONFIG, second)
            split_a = pd.read_csv(first / "HUST_split_manifest.csv", dtype=str)
            split_b = pd.read_csv(second / "HUST_split_manifest.csv", dtype=str)
            self.assertEqual(split_a["role"].value_counts().to_dict(), {
                "confirmation": 57,
                "calibration": 20,
            })
            pd.testing.assert_frame_equal(split_a, split_b)
            receipt = read_json(first / "source_receipt.json")
            self.assertFalse(receipt["pickle_unpickled"])
            self.assertFalse(receipt["capacity_value_read"])

    def test_capacity_integration(self) -> None:
        self.assertAlmostEqual(discharge_capacity_Ah(cycle_frame(1.1)), 1.1, places=12)

    def test_dynamic_landmark_and_future_support(self) -> None:
        frame, audit = parse_hust_cell(synthetic_cell("1-1"), "1-1", PROTOCOL)
        self.assertEqual(find_dynamic_landmark(frame, PROTOCOL), 20)
        self.assertEqual(audit["landmark_cycle"], 20)
        self.assertGreaterEqual(audit["future_metric_support_points"], 5)
        self.assertEqual(audit["observed_eol_cycle"], 25)
        self.assertAlmostEqual(audit["bol_reference_capacity_Ah"], 1.1, places=12)

    def test_special_cell_skips_exactly_first_two_sorted_cycles(self) -> None:
        frame, audit = parse_hust_cell(synthetic_cell("7-5"), "7-5", PROTOCOL)
        self.assertEqual(int(frame["cycle"].min()), 3)
        self.assertEqual(audit["raw_cycles_in_pickle"], 30)
        self.assertEqual(audit["cycles_after_problem_cell_rule"], 28)
        self.assertTrue(audit["problem_cell_first_two_cycles_excluded"])


class StatisticsTests(unittest.TestCase):
    def _records(self, cells: int) -> pd.DataFrame:
        rows = []
        for index in range(cells):
            cell = f"HUST_{index:02d}"
            full = 0.020 + index * 0.00001
            for model, rmse in (
                ("mstt_full_K5", full),
                ("local_linear_trend", full + 0.010),
                ("mstt_single_scale_K5", full + 0.006),
            ):
                rows.append(
                    {
                        "cell_id": cell,
                        "model": model,
                        "future_capacity_RMSE_SOH": rmse,
                    }
                )
        return pd.DataFrame(rows)

    def test_conformal_finite_sample_order(self) -> None:
        self.assertEqual(conformal_order_statistic([1, 2, 3, 4], 0.8), 4.0)

    def test_minimum_confirmation_gate(self) -> None:
        result = fixed_sequence_confirmation(self._records(39), PROTOCOL)
        self.assertTrue(result["status"].startswith("NOT_ESTIMABLE"))
        self.assertIsNone(result["H1_full_vs_local_linear"])

    def test_fixed_sequence_success(self) -> None:
        result = fixed_sequence_confirmation(self._records(40), PROTOCOL)
        self.assertEqual(result["status"], "ESTIMABLE")
        h1 = result["H1_full_vs_local_linear"]
        h2 = result["H2_full_vs_single_scale"]
        self.assertTrue(h1["confirmatory_and_practical_success"])
        self.assertTrue(h2["gate_open"])
        self.assertTrue(math.isfinite(h2["two_sided_p"]))

    def test_empty_confirmation_aggregates_as_not_estimable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            confirmation = root / "confirmation"
            prepared = confirmation / "prepared"
            aggregate = root / "aggregate"
            prepared.mkdir(parents=True)
            records_path = confirmation / "ensemble_cell_records.csv"
            _records_frame([], seed_level=False).to_csv(records_path, index=False)
            write_json(
                prepared / "confirmation_preflight.json",
                {
                    "assigned_cells": 57,
                    "evaluable_cells": 0,
                    "excluded_cells": 57,
                },
            )
            audit = {
                "status": "PASS",
                "run_uuid": "synthetic-run",
                "config_sha256": json_sha256(PROTOCOL),
                "evaluable_confirmation_cells": 0,
                "ensemble_records_sha256": sha256_file(records_path),
            }
            audit_path = confirmation / "evaluation_audit.json"
            write_json(audit_path, audit)
            write_json(
                confirmation / "one_shot_state.json",
                {
                    "status": "PASS",
                    "run_uuid": "synthetic-run",
                    "evaluation_audit_sha256": sha256_file(audit_path),
                },
            )
            os.environ["MPLCONFIGDIR"] = str(root / "matplotlib-cache")
            aggregate_confirmation(CONFIG, confirmation, aggregate)
            inference = read_json(aggregate / "hust_fixed_sequence_inference.json")
            self.assertTrue(inference["status"].startswith("NOT_ESTIMABLE"))
            self.assertTrue((aggregate / "hust_confirmatory_rmse_boxplot.png").is_file())
            self.assertTrue((aggregate / "hust_paired_rmse_scatter.pdf").is_file())


if __name__ == "__main__":
    unittest.main()
