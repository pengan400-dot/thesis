from __future__ import annotations

import argparse
import json
from pathlib import Path
import zipfile

from .pipeline import (
    json_sha256,
    load_protocol,
    sha256_file,
    utc_now,
    write_json,
)


def files_under(path: Path) -> list[Path]:
    return [item for item in sorted(path.rglob("*")) if item.is_file()]


def run(args: argparse.Namespace) -> None:
    protocol = load_protocol(args.config)
    config_hash = json_sha256(protocol)
    development_gate = json.loads(
        (args.development / "development_preflight.json").read_text(
            encoding="utf-8"
        )
    )
    model_manifest = json.loads(
        (args.models / "frozen_model_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    calibration_manifest = json.loads(
        (args.conformal / "calibration_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    runtime_manifest_path = args.runtime_environment / "runtime_environment.json"
    pip_freeze_path = args.runtime_environment / "pip_freeze.txt"
    runtime_manifest = json.loads(
        runtime_manifest_path.read_text(encoding="utf-8")
    )
    gates = (
        development_gate.get("status") == "PASS",
        development_gate.get("config_sha256") == config_hash,
        development_gate.get("external_data_loaded") is False,
        development_gate.get("hnei_data_loaded") is False,
        development_gate.get("bit_data_loaded") is False,
        model_manifest.get("status") == "PASS",
        model_manifest.get("config_sha256") == config_hash,
        model_manifest.get("external_data_loaded") is False,
        model_manifest.get("quick_test") is False,
        calibration_manifest.get("status") == "PASS",
        calibration_manifest.get("config_sha256") == config_hash,
        calibration_manifest.get("external_data_loaded") is False,
        calibration_manifest.get("hnei_data_loaded") is False,
        calibration_manifest.get("bit_data_loaded") is False,
        calibration_manifest.get("bit_updates_quantiles") is False,
        calibration_manifest.get("quick_test") is False,
        int(calibration_manifest.get("physical_calibration_cells", -1)) == 16,
        runtime_manifest.get("status") == "PASS",
        runtime_manifest.get("pip_freeze_sha256")
        == sha256_file(pip_freeze_path),
    )
    if not all(gates):
        raise RuntimeError(
            "XJTU-only development/model/calibration gate failed"
        )
    expected_jobs = {
        (str(specification["id"]), int(seed))
        for specification in protocol["models"]
        if str(specification["id"]).startswith("mstt_")
        for seed in protocol["training"]["seeds"]
    }
    observed_jobs = {
        (str(job["model_id"]), int(job["seed"]))
        for job in model_manifest["jobs"]
    }
    if observed_jobs != expected_jobs:
        raise RuntimeError(
            f"Expected model jobs {sorted(expected_jobs)}, "
            f"observed {sorted(observed_jobs)}"
        )
    identities = {
        str(job["model_id"]): job["identity"]
        for job in model_manifest["jobs"]
    }
    if any(
        int(job["identity"]["trainable_parameters"])
        != (
            int(protocol["source_implementation"]["full_model_trainable_parameters"])
            if job["ablation"] == "full"
            else int(
                protocol["source_implementation"][
                    "single_scale_trainable_parameters"
                ]
            )
        )
        for job in model_manifest["jobs"]
    ):
        raise RuntimeError("Frozen model parameter identity failed")
    if args.output_zip.exists():
        raise FileExistsError(
            f"Development freeze already exists: {args.output_zip}"
        )
    source_files = [
        path
        for path in files_under(args.project_root)
        if ".venv" not in path.parts
        and "__pycache__" not in path.parts
        and "results" not in path.parts
        and ".git" not in path.parts
    ]
    model_files = files_under(args.models)
    conformal_files = files_under(args.conformal)
    manifest = {
        "status": "PASS",
        "created_utc": utc_now(),
        "version": protocol["version_labels"]["development"],
        "config_sha256": config_hash,
        "target": "SOH_fraction",
        "soh_formula": protocol["soh_definition"]["formula"],
        "model_job_count": len(observed_jobs),
        "model_identities": identities,
        "development_data_sha256": model_manifest[
            "development_data_sha256"
        ],
        "common_training_support_sha256": model_manifest[
            "common_support_sha256"
        ],
        "calibration_scores_sha256": calibration_manifest[
            "calibration_scores_sha256"
        ],
        "conformal_quantiles_sha256": calibration_manifest[
            "conformal_quantiles_sha256"
        ],
        "xjtu_only_development": True,
        "hnei_performance_seen_by_workflow": False,
        "bit_performance_seen_by_workflow": False,
        "hnei_model_predictions_generated_before_freeze": False,
        "bit_model_predictions_generated_before_freeze": False,
        "legacy_absolute_Ah_weights_included": False,
        "runtime_environment_sha256": sha256_file(runtime_manifest_path),
        "pip_freeze_sha256": sha256_file(pip_freeze_path),
        "source_files": {
            str(path.relative_to(args.project_root)): sha256_file(path)
            for path in source_files
        },
        "model_files": {
            str(path.relative_to(args.models)): sha256_file(path)
            for path in model_files
        },
        "conformal_files": {
            str(path.relative_to(args.conformal)): sha256_file(path)
            for path in conformal_files
        },
    }
    manifest_path = args.output_zip.with_suffix(".manifest.json")
    write_json(manifest_path, manifest)
    args.output_zip.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_zip.with_suffix(args.output_zip.suffix + ".partial")
    with zipfile.ZipFile(
        temporary,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        for path in source_files:
            archive.write(
                path,
                Path("code") / path.relative_to(args.project_root),
            )
        for path in model_files:
            archive.write(
                path,
                Path("frozen_models") / path.relative_to(args.models),
            )
        for path in conformal_files:
            archive.write(
                path,
                Path("crossfit_conformal")
                / path.relative_to(args.conformal),
            )
        archive.write(
            args.development / "development_preflight.json",
            "gates/development_preflight.json",
        )
        archive.write(
            args.development / "development_soh_audit.csv",
            "gates/development_soh_audit.csv",
        )
        archive.write(
            runtime_manifest_path,
            "environment/runtime_environment.json",
        )
        archive.write(pip_freeze_path, "environment/pip_freeze.txt")
        archive.write(manifest_path, "SOH_DEVELOPMENT_FREEZE.json")
    temporary.replace(args.output_zip)
    with zipfile.ZipFile(args.output_zip) as archive:
        bad = archive.testzip()
    if bad is not None:
        raise RuntimeError(f"Development freeze ZIP CRC failed: {bad}")
    digest = sha256_file(args.output_zip)
    Path(str(args.output_zip) + ".sha256").write_text(
        f"{digest}  {args.output_zip.name}\n",
        encoding="utf-8",
    )
    print(f"[PASS] {protocol['version_labels']['development']}")
    print(f"ZIP={args.output_zip}")
    print(f"SHA256={digest}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze XJTU-only SOH weights, scalers, and calibration"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--conformal", type=Path, required=True)
    parser.add_argument("--runtime-environment", type=Path, required=True)
    parser.add_argument("--output-zip", type=Path, required=True)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
