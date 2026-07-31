from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile


ROOT = Path(__file__).resolve().parents[1]


class BITStructurePreflightTests(unittest.TestCase):
    def test_preflight_redacts_capacity_range_and_imports_no_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive_path = root / "bit_fixture.zip"
            csv_data = (
                "physical_cycle,discharge_capacity_Ah,cohort\n"
                "1,2.40,arbitrary_use\n"
                "190,2.10,arbitrary_use\n"
                "200,2.00,arbitrary_use\n"
            )
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr(
                    "arbitrary/cell_001.csv",
                    csv_data,
                )
            output = root / "report"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "bit_structure_preflight.py"),
                    "--source",
                    str(archive_path),
                    "--out",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            report = json.loads(
                (output / "bit_structural_preflight.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(report["status"], "STRUCTURE_ONLY_PASS")
            self.assertFalse(report["model_imported"])
            self.assertFalse(report["model_output_generated"])
            self.assertFalse(report["numeric_capacity_values_summarized"])
            with (
                output / "bit_field_inventory.csv"
            ).open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            capacity = next(
                row
                for row in rows
                if row["candidate_role"] == "capacity_candidate"
            )
            cycle = next(
                row
                for row in rows
                if row["candidate_role"] == "physical_cycle_candidate"
            )
            self.assertEqual(capacity["numeric_min"], "")
            self.assertEqual(capacity["numeric_max"], "")
            self.assertEqual(float(cycle["numeric_max"]), 200.0)


if __name__ == "__main__":
    unittest.main()
