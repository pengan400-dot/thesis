#!/usr/bin/env python3
"""Static pre-capacity verification for the BIT v0.2.1 evaluator source."""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


EXPECTED_BRANCH = "implementation/v0.2.1-BIT-evaluator"
REGISTERED_COMMIT = "780a057d26cd4a0fc2dc50a80161d66c936155b5"
REGISTERED_VERSION = "v0.2.1-BIT-structural-feasibility-amendment"
REGISTERED_FREEZE_SHA = (
    "d66a7ff79b76c497f73e842b9accd6c18035f4169720e796366ea6f7a7f48aed"
)
REGISTERED_GITHUB = (
    "https://github.com/pengan400-dot/thesis/releases/tag/"
    "v0.2.1-BIT-structural-feasibility-amendment"
)
REGISTERED_OSF = "https://osf.io/gr6xj/"
REGISTERED_ZENODO = "10.5281/zenodo.21734477"
EXPECTED_EXISTING_HASHES = {
    "configs/soh_development_protocol_v0.2.1_BIT_amendment.json": (
        "c899834230072cb69362b6359c92b8cf0f7f8a34fad6412babf1570070f67f7f"
    ),
    "configs/soh_development_protocol_v0.2.0.json": (
        "41e9ae88fccc46291f08450d335b49799ee185b1cc0e0e395e3b03a9c5c489a5"
    ),
    "scripts/parse_bit_capacity_v0_2_1.py": (
        "51ef6683a69b7594d5da0e37fc7d4ed24e52a6ab31bef890f0a2aa42c043fdb8"
    ),
    "src/mstt_soh/pipeline.py": (
        "9a78d76dffce2ecf08e0301d99013c14068cb7fe5c8e66ea3a841b27e2427d84"
    ),
    "src/mstt_soh/model_variants.py": (
        "25310d52576878df555f23acbff86f5659b445b94cfceddcde7073fa03b31ef2"
    ),
    "src/mstt_soh/soh.py": (
        "bf8bfd359366481f195ab96d5dce295337fda6b37e68c340b17cabf8c259da4a"
    ),
    "src/mstt_soh/statistics.py": (
        "0d5e9f259329de8d828164830a757df15c16012fea0f1ecbb4c8791795e02b43"
    ),
}
EVALUATOR_FILES = (
    "scripts/bit_v021_evaluator.py",
    "scripts/verify_bit_evaluator_source_v0_2_1.py",
    "scripts/freeze_bit_evaluator_source_v0_2_1.py",
    "tests/test_bit_v021_evaluator_contract.py",
    "manager.sh",
    "preregistration/IMPLEMENTATION_NOTE_v0.2.1_BIT_EVALUATOR_PRE_CAPACITY.md",
)
FORBIDDEN_IMPORTS = {"requests", "httpx", "urllib", "socket"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(project: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=project, text=True).strip()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def imports_in(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--registration-receipt", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    project = args.project_root.resolve()
    failures: list[str] = []
    branch = git(project, "branch", "--show-current")
    head = git(project, "rev-parse", "HEAD")
    status = git(project, "status", "--short")
    if branch != EXPECTED_BRANCH:
        failures.append(f"branch={branch!r}, expected={EXPECTED_BRANCH!r}")
    if not status == "":
        failures.append("working tree is not clean")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", REGISTERED_COMMIT, head],
        cwd=project,
        check=False,
    ).returncode == 0
    if not ancestor:
        failures.append("registered protocol commit is not an ancestor")
    try:
        upstream = git(project, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
        upstream_head = git(project, "rev-parse", "@{u}")
    except subprocess.CalledProcessError:
        upstream = None
        upstream_head = None
        failures.append("evaluator branch has no configured upstream")
    if upstream_head is not None and upstream_head != head:
        failures.append("local evaluator commit has not been pushed to its upstream")

    receipt = read_json(args.registration_receipt)
    expected_receipt = {
        "status": "PASS",
        "version": REGISTERED_VERSION,
        "commit": REGISTERED_COMMIT,
        "freeze_zip_sha256": REGISTERED_FREEZE_SHA,
        "github_release": REGISTERED_GITHUB,
        "osf_registration": REGISTERED_OSF,
        "zenodo_specific_version_doi": REGISTERED_ZENODO,
        "bit_capacity_opened_before_receipt": False,
        "bit_model_output_existed_before_receipt": False,
        "bit_evaluation_unlocked": True,
    }
    for key, expected in expected_receipt.items():
        if receipt.get(key) != expected:
            failures.append(f"registration receipt mismatch: {key}")

    freeze = read_json(args.freeze_manifest)
    if freeze.get("status") != "PASS":
        failures.append("freeze manifest status is not PASS")
    if freeze.get("frozen_commit") != REGISTERED_COMMIT:
        failures.append("freeze manifest commit changed")
    if freeze.get("primary_landmark") != 70:
        failures.append("primary landmark is not 70")
    if freeze.get("structurally_unavailable_landmarks") != [130, 190]:
        failures.append("unavailable landmark contract changed")
    if freeze.get("bit_capacity_opened_before_freeze") is not False:
        failures.append("freeze says capacity was opened")
    if freeze.get("bit_model_output_generated_before_freeze") is not False:
        failures.append("freeze says model output existed")

    for relative, expected in EXPECTED_EXISTING_HASHES.items():
        path = project / relative
        if not path.is_file():
            failures.append(f"missing frozen source: {relative}")
        elif sha256(path) != expected:
            failures.append(f"frozen source hash changed: {relative}")

    source_hashes: dict[str, str] = {}
    for relative in EVALUATOR_FILES:
        path = project / relative
        if not path.is_file():
            failures.append(f"missing evaluator source: {relative}")
            continue
        source_hashes[relative] = sha256(path)
        if path.suffix == ".py":
            try:
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except SyntaxError as exc:
                failures.append(f"syntax error {relative}: {exc}")
            bad_imports = imports_in(path) & FORBIDDEN_IMPORTS
            if bad_imports:
                failures.append(f"network imports in {relative}: {sorted(bad_imports)}")

    evaluator_text = (project / "scripts/bit_v021_evaluator.py").read_text(
        encoding="utf-8"
    )
    required_tokens = (
        'PRIMARY_CUTOFF = 70',
        'STRUCTURALLY_UNAVAILABLE_CUTOFFS = [130, 190]',
        'EXPECTED_INCLUDED_CELLS = 72',
        'EXCLUDED_CELL_IDS = ["BIT_#2"]',
        '"mstt_full_K5"',
        '"mstt_single_scale_K5"',
        '"local_linear_trend"',
        'fixed_sequence_gatekeeping',
        'ordinary_pooled_p_value_computed',
        'post_cutoff_actual_current_schedule_loaded',
    )
    for token in required_tokens:
        if token not in evaluator_text:
            failures.append(f"evaluator contract token absent: {token}")
    manager_text = (project / "manager.sh").read_text(encoding="utf-8")
    for command in (
        "bit_eval_verify)",
        "bit_eval_source_freeze)",
        "bit_parse)",
        "bit_prepare)",
        "bit_evaluate)",
        "bit_aggregate)",
        "bit_pack_results)",
    ):
        if command not in manager_text:
            failures.append(f"manager command missing: {command}")

    if args.evaluation_dir.exists() and any(args.evaluation_dir.rglob("*")):
        failures.append("BIT evaluation directory already contains files")

    output = {
        "status": (
            "PASS_BIT_V0_2_1_EVALUATOR_SOURCE_VERIFY"
            if not failures
            else "FAIL_BIT_V0_2_1_EVALUATOR_SOURCE_VERIFY"
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "branch": branch,
        "upstream": upstream,
        "upstream_commit": upstream_head,
        "evaluator_source_commit": head,
        "registered_protocol_commit": REGISTERED_COMMIT,
        "registration_receipt_sha256": sha256(args.registration_receipt),
        "freeze_manifest_sha256": sha256(args.freeze_manifest),
        "source_files": source_hashes,
        "capacity_opened": False,
        "model_weights_loaded": False,
        "model_output_generated": False,
        "evaluation_dir_empty": not (
            args.evaluation_dir.exists() and any(args.evaluation_dir.rglob("*"))
        ),
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(4)
    print("[PASS] BIT v0.2.1 evaluator source verified without capacity access")


if __name__ == "__main__":
    main()
