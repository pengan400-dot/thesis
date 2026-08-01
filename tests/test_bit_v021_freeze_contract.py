#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import unittest

import yaml

PROJECT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT / "configs/soh_development_protocol_v0.2.1_BIT_amendment.json"
MAPPING = (
    PROJECT
    / "amendments/v0.2.1-BIT-structural-feasibility/configs"
    / "bit_schema_mapping_v0.2.1.yaml"
)
RESOLUTION = (
    PROJECT
    / "amendments/v0.2.1-BIT-structural-feasibility/AUDIT"
    / "CYCLE_MAPPING_PASS_OFFSET20_20260801"
    / "bit_cycle_mapping_resolution_v0.2.1.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_relative(mapping_path: Path, value: object) -> Path:
    candidate = Path(str(value))
    return candidate if candidate.is_absolute() else (mapping_path.parent / candidate).resolve()


class TestBitV021FreezeContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(CONFIG.read_text(encoding="utf-8"))
        cls.mapping = yaml.safe_load(MAPPING.read_text(encoding="utf-8"))
        cls.resolution = json.loads(RESOLUTION.read_text(encoding="utf-8"))

    def test_landmark_contract(self) -> None:
        self.assertEqual(self.config["task"]["primary_cutoff"], 70)
        self.assertEqual(self.config["bit"]["primary_cutoff"], 70)
        self.assertEqual(
            sorted(self.config["bit"]["structurally_unavailable_cutoffs"]),
            [130, 190],
        )
        self.assertEqual(self.mapping["landmark"]["primary_physical_cycle"], 70)
        self.assertEqual(
            sorted(self.mapping["landmark"]["structurally_unavailable_cycles"]),
            [130, 190],
        )

    def test_mapping_contract(self) -> None:
        self.assertEqual(
            self.config["bit"]["physical_cycle_mapping"],
            {
                "first20_local_1_to_20": "physical 1 to 20",
                "first20_local_21": "excluded terminal workbook bucket",
                "later_formula": "physical_cycle = 20 + local_cycle",
            },
        )
        self.assertTrue(
            self.resolution["cycle70_representable_for_all_included_cells"]
        )
        self.assertFalse(
            self.resolution["cycle130_representable_for_any_included_cell"]
        )
        self.assertFalse(
            self.resolution["cycle190_representable_for_any_included_cell"]
        )

    def test_frozen_cell_set(self) -> None:
        manifest = resolve_relative(
            MAPPING,
            self.mapping["resolved_structure"]["evaluation_manifest_csv"],
        )
        self.assertEqual(
            sha256(manifest),
            "34b08a5f401aee9cde564bc49e542768517722b682ee5b927109e8ed5e0fd3b4",
        )
        with manifest.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        included = [
            row for row in rows
            if row["include_in_amended_BIT_evaluation"].lower() == "true"
        ]
        excluded = [
            row for row in rows
            if row["include_in_amended_BIT_evaluation"].lower() != "true"
        ]
        self.assertEqual(len(rows), 73)
        self.assertEqual(len(included), 72)
        self.assertEqual([row["cell_id"] for row in excluded], ["BIT_#2"])
        self.assertEqual(
            sum(row["cohort"] == "arbitrary_use" for row in included),
            55,
        )
        self.assertEqual(
            sum(row["cohort"] == "fixed_profile" for row in included),
            17,
        )

    def test_parser_static_guard(self) -> None:
        parser_path = resolve_relative(
            MAPPING,
            self.mapping["parser"]["source_file"],
        )
        self.assertEqual(
            sha256(parser_path),
            self.mapping["parser"]["source_sha256"],
        )
        text = parser_path.read_text(encoding="utf-8")
        lower = text.lower()
        self.assertNotIn("import torch", lower)
        self.assertNotIn("from torch", lower)
        guard = text.find(
            "validate_registration_receipt(args.registration_receipt)"
        )
        archive_open = text.find("zipfile.ZipFile(args.archive")
        self.assertGreaterEqual(guard, 0)
        self.assertGreater(archive_open, guard)
        self.assertFalse(
            self.mapping["parser"]["executed_on_capacity_values_before_freeze"]
        )

    def test_registration_script_version(self) -> None:
        text = (PROJECT / "scripts/register_bit_freeze_receipt.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            'EXPECTED_VERSION = "v0.2.1-BIT-structural-feasibility-amendment"',
            text,
        )
        self.assertIn('"primary_landmark": 70', text)


if __name__ == "__main__":
    unittest.main()
