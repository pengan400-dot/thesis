from __future__ import annotations

import json
import os
import re
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

import torch

from mstt_soh.pipeline import load_frozen_models, load_protocol as load_dev_protocol

from .common import (
    assert_hash,
    json_sha256,
    load_hust_protocol,
    read_json,
    sha256_file,
    utc_now,
    write_json,
)
from .source import load_source_gate


def _walk(prefix: str, path: Path) -> list[tuple[str, Path]]:
    path = Path(path)
    if path.is_file():
        return [(PurePosixPath(prefix, path.name).as_posix(), path)]
    if not path.is_dir():
        raise FileNotFoundError(path)
    return [
        (PurePosixPath(prefix, item.relative_to(path).as_posix()).as_posix(), item)
        for item in sorted(path.rglob("*"))
        if item.is_file()
        and "__pycache__" not in item.parts
        and ".pytest_cache" not in item.parts
        and item.suffix not in {".pyc", ".pyo"}
    ]


def _file_rows(files: Iterable[tuple[str, Path]]) -> list[dict[str, object]]:
    rows = []
    seen: set[str] = set()
    for archive_path, source_path in sorted(files):
        if archive_path in seen:
            raise ValueError(f"Duplicate archive path: {archive_path}")
        seen.add(archive_path)
        rows.append(
            {
                "archive_path": archive_path,
                "source_path": str(Path(source_path).resolve()),
                "size_bytes": Path(source_path).stat().st_size,
                "sha256": sha256_file(Path(source_path)),
            }
        )
    return rows


def _write_deterministic_zip(
    output_zip: Path,
    rows: list[dict[str, object]],
    extra_bytes: Mapping[str, bytes] | None = None,
) -> str:
    output_zip = Path(output_zip)
    if output_zip.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing frozen archive: {output_zip}"
        )
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_zip.with_suffix(output_zip.suffix + ".tmp")
    with zipfile.ZipFile(
        temporary,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as handle:
        for row in rows:
            info = zipfile.ZipInfo(str(row["archive_path"]), (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            handle.writestr(info, Path(str(row["source_path"])).read_bytes())
        for archive_path, data in sorted((extra_bytes or {}).items()):
            info = zipfile.ZipInfo(str(archive_path), (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            handle.writestr(info, data)
    os.replace(temporary, output_zip)
    return sha256_file(output_zip)


def _model_rows(model_root: Path) -> list[dict[str, object]]:
    rows = _file_rows(_walk("frozen_models", model_root))
    return [
        {
            "relative_path": str(row["archive_path"]).removeprefix("frozen_models/"),
            "size_bytes": row["size_bytes"],
            "sha256": row["sha256"],
        }
        for row in rows
    ]


def _project_files(project_root: Path) -> list[tuple[str, Path]]:
    project_root = Path(project_root).resolve()
    selected: list[tuple[str, Path]] = []
    selected.extend(_walk("project/src", project_root / "src"))
    selected.extend(_walk("project/configs", project_root / "configs"))
    selected.extend(_walk("project/inputs", project_root / "inputs"))
    selected.extend(_walk("project/manifests", project_root / "manifests"))
    selected.extend(_walk("project/preregistration", project_root / "preregistration"))
    selected.extend(_walk("project/scripts", project_root / "scripts"))
    selected.extend(_walk("project/tests", project_root / "tests"))
    for filename in (
        "manager.sh",
        "README_CN.md",
        "requirements.txt",
        "LICENSE",
        "CITATION.cff",
        "THIRD_PARTY_NOTICE.md",
        "VALIDATION_REPORT.md",
        "PACKAGE_MANIFEST.json",
    ):
        path = project_root / filename
        if path.is_file():
            selected.append((f"project/{filename}", path))
    return selected


def _verify_project_files(
    project_root: Path,
    frozen_rows: list[dict[str, object]],
) -> None:
    expected = {
        str(row["archive_path"]): (int(row["size_bytes"]), str(row["sha256"]))
        for row in frozen_rows
        if str(row["archive_path"]).startswith("project/")
    }
    observed_rows = _file_rows(_project_files(project_root))
    observed = {
        str(row["archive_path"]): (int(row["size_bytes"]), str(row["sha256"]))
        for row in observed_rows
    }
    if observed != expected:
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        changed = sorted(
            key
            for key in set(expected) & set(observed)
            if expected[key] != observed[key]
        )
        raise RuntimeError(
            "Local evaluator project differs from the registered freeze: "
            f"missing={missing}, extra={extra}, changed={changed}"
        )


def validate_model_root(dev_config: Path, model_root: Path) -> dict[str, Any]:
    protocol = load_dev_protocol(Path(dev_config))
    model_root = Path(model_root)
    manifest = read_json(model_root / "frozen_model_manifest.json")
    if (
        manifest.get("status") != "PASS"
        or manifest.get("config_sha256") != json_sha256(protocol)
        or manifest.get("external_data_loaded") is not False
        or manifest.get("quick_test") is not False
    ):
        raise RuntimeError("Formal XJTU-only SOH model manifest gate failed")
    loaded = load_frozen_models(protocol, model_root, torch.device("cpu"))
    expected_models = {"mstt_full_K5", "mstt_single_scale_K5"}
    if set(loaded) != expected_models:
        raise RuntimeError(f"Frozen neural model set changed: {sorted(loaded)}")
    for model_id, expected_parameters in (
        ("mstt_full_K5", 74405),
        ("mstt_single_scale_K5", 69789),
    ):
        if set(loaded[model_id]) != {42, 2024, 3407}:
            raise RuntimeError(f"Frozen seed set changed for {model_id}")
        for model, _ in loaded[model_id].values():
            count = sum(parameter.numel() for parameter in model.parameters())
            if count != expected_parameters:
                raise RuntimeError(
                    f"{model_id}: expected {expected_parameters} parameters, got {count}"
                )
    return manifest


def _assert_development_gate(dev_config: Path, development: Path) -> dict[str, Any]:
    protocol = load_dev_protocol(Path(dev_config))
    gate = read_json(Path(development) / "development_preflight.json")
    if (
        gate.get("status") != "PASS"
        or gate.get("config_sha256") != json_sha256(protocol)
        or gate.get("physical_cells") != 16
        or gate.get("external_data_loaded") is not False
        or gate.get("hnei_data_loaded") is not False
        or gate.get("bit_data_loaded") is not False
    ):
        raise RuntimeError("XJTU-only normalized-SOH development gate failed")
    return gate


def freeze_evaluator(
    config: Path,
    dev_config: Path,
    archive: Path,
    inventory: Path,
    development: Path,
    model_root: Path,
    project_root: Path,
    output_zip: Path,
    calibration_preflight: Path,
) -> None:
    for candidate in (
        Path(output_zip),
        Path(str(output_zip) + ".manifest.json"),
        Path(str(output_zip) + ".sha256"),
    ):
        if candidate.exists():
            raise FileExistsError(f"Refusing to overwrite evaluator freeze: {candidate}")
    if Path(calibration_preflight).exists():
        raise RuntimeError(
            "Cannot freeze the evaluator after the HUST calibration arm was opened"
        )
    protocol, source_receipt, _ = load_source_gate(archive, config, inventory)
    dev_protocol = load_dev_protocol(dev_config)
    development_gate = _assert_development_gate(dev_config, development)
    model_manifest = validate_model_root(dev_config, model_root)
    project_root = Path(project_root).resolve()
    selected_files: list[tuple[str, Path]] = _project_files(project_root)
    selected_files.extend(_walk("source_inventory", Path(inventory)))
    for filename in (
        "development_preflight.json",
        "development_soh_audit.csv",
        "xjtu_input_receipt.json",
    ):
        candidates = [Path(development) / filename, Path(development).parent / filename]
        path = next((item for item in candidates if item.is_file()), None)
        if path is not None:
            selected_files.append((f"development_receipts/{filename}", path))
    selected_files.extend(_walk("frozen_models", Path(model_root)))
    rows = _file_rows(selected_files)
    content_manifest = {
        "schema_version": "1.0",
        "freeze_kind": "HUST_evaluator_pre_calibration",
        "created_utc": utc_now(),
        "hust_config_sha256": json_sha256(protocol),
        "development_config_sha256": json_sha256(dev_protocol),
        "source_archive_sha256": source_receipt["archive_sha256"],
        "source_inventory_sha256": source_receipt["inventory_sha256"],
        "split_manifest_sha256": source_receipt["split_manifest_sha256"],
        "development_curves_sha256": development_gate["curves_sha256"],
        "model_manifest_sha256": sha256_file(Path(model_root) / "frozen_model_manifest.json"),
        "model_files": _model_rows(model_root),
        "files": [
            {
                "archive_path": row["archive_path"],
                "size_bytes": row["size_bytes"],
                "sha256": row["sha256"],
            }
            for row in rows
        ],
        "hust_pickle_unpickled": False,
        "hust_capacity_value_read": False,
        "hust_model_output_computed": False,
        "formal_model_jobs": len(model_manifest["jobs"]),
    }
    manifest_bytes = (
        json.dumps(content_manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    digest = _write_deterministic_zip(
        output_zip,
        rows,
        {"freeze_manifest/evaluator_content_manifest.json": manifest_bytes},
    )
    sidecar = {
        **content_manifest,
        "archive_name": Path(output_zip).name,
        "archive_sha256": digest,
        "archive_size_bytes": Path(output_zip).stat().st_size,
    }
    manifest_path = Path(str(output_zip) + ".manifest.json")
    write_json(manifest_path, sidecar)
    Path(str(output_zip) + ".sha256").write_text(
        f"{digest}  {Path(output_zip).name}\n",
        encoding="utf-8",
    )
    print(f"[PASS] evaluator freeze: {output_zip}")
    print(f"SHA256={digest}")


def _verify_model_files(model_root: Path, rows: list[dict[str, object]]) -> None:
    for row in rows:
        path = Path(model_root) / str(row["relative_path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(row["size_bytes"]):
            raise RuntimeError(f"Frozen model file size changed: {path}")
        assert_hash(path, str(row["sha256"]), "Frozen model file")


def register_freeze(
    kind: str,
    archive: Path,
    manifest: Path,
    commit: str,
    locator: str,
    output: Path,
    calibration_preflight: Path,
    confirmation_state: Path,
) -> None:
    if kind not in {"evaluator", "calibration"}:
        raise ValueError(kind)
    if Path(output).exists():
        raise FileExistsError(f"Refusing to overwrite an existing receipt: {output}")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", str(commit)):
        raise ValueError("Commit must be exactly 40 hexadecimal characters")
    commit_value = str(commit).lower()
    if commit_value in {
        "0" * 40,
        "0123456789abcdef0123456789abcdef01234567",
    }:
        raise ValueError("Placeholder Git commit is prohibited")
    locator_value = str(locator).strip()
    if (
        not re.match(r"^(?:https://|doi:)", locator_value, flags=re.IGNORECASE)
        or " " in locator_value
        or "example" in locator_value.lower()
        or "xxxx" in locator_value.lower()
    ):
        raise ValueError("Immutable public release/DOI locator is required")
    if Path(confirmation_state).exists():
        raise RuntimeError("Cannot register a freeze after confirmation was opened")
    if kind == "evaluator" and Path(calibration_preflight).exists():
        raise RuntimeError(
            "Cannot register the evaluator freeze after calibration was opened"
        )
    if kind == "calibration" and not Path(calibration_preflight).is_file():
        raise RuntimeError("Calibration preflight is missing")
    manifest_payload = read_json(manifest)
    expected_kind = (
        "HUST_evaluator_pre_calibration"
        if kind == "evaluator"
        else "HUST_calibration_pre_confirmation"
    )
    if manifest_payload.get("freeze_kind") != expected_kind:
        raise RuntimeError("Freeze manifest kind mismatch")
    assert_hash(archive, str(manifest_payload["archive_sha256"]), f"{kind} freeze")
    receipt = {
        "status": "PASS",
        "created_utc": utc_now(),
        "freeze_kind": expected_kind,
        "archive_name": Path(archive).name,
        "archive_sha256": manifest_payload["archive_sha256"],
        "manifest_name": Path(manifest).name,
        "manifest_sha256": sha256_file(manifest),
        "git_commit": commit_value,
        "immutable_public_locator": locator_value,
        "calibration_data_opened_before_receipt": kind == "calibration",
        "confirmation_data_opened_before_receipt": False,
    }
    write_json(output, receipt)
    print(f"[PASS] {kind} freeze receipt: {output}")


def validate_evaluator_receipt(
    archive: Path,
    manifest: Path,
    receipt: Path,
    model_root: Path,
    protocol: Mapping[str, Any],
    project_root: Path,
    source_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_payload = read_json(manifest)
    receipt_payload = read_json(receipt)
    if (
        manifest_payload.get("freeze_kind") != "HUST_evaluator_pre_calibration"
        or manifest_payload.get("hust_config_sha256") != json_sha256(protocol)
        or manifest_payload.get("source_archive_sha256")
        != source_receipt.get("archive_sha256")
        or manifest_payload.get("source_inventory_sha256")
        != source_receipt.get("inventory_sha256")
        or manifest_payload.get("split_manifest_sha256")
        != source_receipt.get("split_manifest_sha256")
        or receipt_payload.get("status") != "PASS"
        or receipt_payload.get("freeze_kind") != "HUST_evaluator_pre_calibration"
        or receipt_payload.get("archive_sha256") != manifest_payload.get("archive_sha256")
        or receipt_payload.get("manifest_sha256") != sha256_file(manifest)
        or receipt_payload.get("calibration_data_opened_before_receipt") is not False
        or receipt_payload.get("confirmation_data_opened_before_receipt") is not False
    ):
        raise RuntimeError("Evaluator freeze receipt gate failed")
    assert_hash(archive, str(manifest_payload["archive_sha256"]), "Evaluator freeze")
    _verify_project_files(Path(project_root), list(manifest_payload["files"]))
    _verify_model_files(Path(model_root), list(manifest_payload["model_files"]))
    return receipt_payload


def freeze_calibration(
    config: Path,
    archive: Path,
    inventory: Path,
    evaluator_zip: Path,
    evaluator_manifest: Path,
    evaluator_receipt: Path,
    model_root: Path,
    calibration_dir: Path,
    project_root: Path,
    output_zip: Path,
) -> None:
    for candidate in (
        Path(output_zip),
        Path(str(output_zip) + ".manifest.json"),
        Path(str(output_zip) + ".sha256"),
    ):
        if candidate.exists():
            raise FileExistsError(f"Refusing to overwrite calibration freeze: {candidate}")
    protocol, source_receipt, _ = load_source_gate(archive, config, inventory)
    evaluator = validate_evaluator_receipt(
        evaluator_zip,
        evaluator_manifest,
        evaluator_receipt,
        model_root,
        protocol,
        project_root,
        source_receipt,
    )
    calibration_preflight = read_json(Path(calibration_dir) / "calibration_preflight.json")
    calibration_audit = read_json(Path(calibration_dir) / "calibration_audit.json")
    if (
        calibration_preflight.get("status") != "PASS"
        or calibration_preflight.get("role") != "calibration"
        or calibration_preflight.get("confirmation_members_unpickled") is not False
        or calibration_audit.get("status") not in {"PASS", "UQ_NOT_ESTIMABLE"}
        or calibration_audit.get("config_sha256") != json_sha256(protocol)
    ):
        raise RuntimeError("Calibration freeze gate failed")
    confirmation_state = Path(calibration_dir).parent / "confirmation" / "one_shot_state.json"
    if confirmation_state.exists():
        raise RuntimeError("Cannot freeze calibration after confirmation was opened")
    selected_files: list[tuple[str, Path]] = []
    selected_files.extend(_walk("project/src", Path(project_root) / "src"))
    selected_files.extend(_walk("project/configs", Path(project_root) / "configs"))
    selected_files.extend(_walk("source_inventory", Path(inventory)))
    for filename in (
        "calibration_preflight.json",
        "calibration_cell_audit.csv",
        "calibration_exclusions.csv",
        "calibration_scores.csv",
        "calibration_seed_records.csv",
        "calibration_quantiles.json",
        "calibration_audit.json",
    ):
        path = Path(calibration_dir) / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        selected_files.append((f"calibration/{filename}", path))
    for label, path in (
        ("receipts/evaluator_freeze_receipt.json", evaluator_receipt),
        ("receipts/evaluator_freeze_manifest.json", evaluator_manifest),
    ):
        selected_files.append((label, Path(path)))
    rows = _file_rows(selected_files)
    content_manifest = {
        "schema_version": "1.0",
        "freeze_kind": "HUST_calibration_pre_confirmation",
        "created_utc": utc_now(),
        "hust_config_sha256": json_sha256(protocol),
        "source_archive_sha256": source_receipt["archive_sha256"],
        "split_manifest_sha256": source_receipt["split_manifest_sha256"],
        "evaluator_archive_sha256": evaluator["archive_sha256"],
        "calibration_curves_sha256": calibration_preflight["curves_sha256"],
        "calibration_status": calibration_audit["status"],
        "eligible_calibration_cells": calibration_audit["eligible_calibration_cells"],
        "confirmation_members_unpickled": False,
        "files": [
            {
                "archive_path": row["archive_path"],
                "size_bytes": row["size_bytes"],
                "sha256": row["sha256"],
            }
            for row in rows
        ],
    }
    manifest_bytes = (
        json.dumps(content_manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    digest = _write_deterministic_zip(
        output_zip,
        rows,
        {"freeze_manifest/calibration_content_manifest.json": manifest_bytes},
    )
    sidecar = {
        **content_manifest,
        "archive_name": Path(output_zip).name,
        "archive_sha256": digest,
        "archive_size_bytes": Path(output_zip).stat().st_size,
    }
    manifest_path = Path(str(output_zip) + ".manifest.json")
    write_json(manifest_path, sidecar)
    Path(str(output_zip) + ".sha256").write_text(
        f"{digest}  {Path(output_zip).name}\n",
        encoding="utf-8",
    )
    print(f"[PASS] calibration freeze: {output_zip}")
    print(f"SHA256={digest}")


def validate_calibration_receipt(
    archive: Path,
    manifest: Path,
    receipt: Path,
    protocol: Mapping[str, Any],
    evaluator_archive_sha256: str,
) -> dict[str, Any]:
    manifest_payload = read_json(manifest)
    receipt_payload = read_json(receipt)
    if (
        manifest_payload.get("freeze_kind") != "HUST_calibration_pre_confirmation"
        or manifest_payload.get("hust_config_sha256") != json_sha256(protocol)
        or manifest_payload.get("evaluator_archive_sha256") != evaluator_archive_sha256
        or manifest_payload.get("confirmation_members_unpickled") is not False
        or receipt_payload.get("status") != "PASS"
        or receipt_payload.get("freeze_kind") != "HUST_calibration_pre_confirmation"
        or receipt_payload.get("archive_sha256") != manifest_payload.get("archive_sha256")
        or receipt_payload.get("manifest_sha256") != sha256_file(manifest)
        or receipt_payload.get("calibration_data_opened_before_receipt") is not True
        or receipt_payload.get("confirmation_data_opened_before_receipt") is not False
    ):
        raise RuntimeError("Calibration freeze receipt gate failed")
    assert_hash(archive, str(manifest_payload["archive_sha256"]), "Calibration freeze")
    return receipt_payload
