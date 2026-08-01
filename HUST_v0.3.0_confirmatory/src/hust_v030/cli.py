from __future__ import annotations

import argparse
from pathlib import Path

from .evaluation import (
    aggregate_confirmation,
    calibrate,
    confirm_one_shot,
    pack_results,
)
from .freeze import freeze_calibration, freeze_evaluator, register_freeze
from .source import extract_xjtu_b1_b3, inventory_source, prepare_role


def _inventory(args: argparse.Namespace) -> None:
    inventory_source(args.archive, args.config, args.output)


def _extract_xjtu(args: argparse.Namespace) -> None:
    extract_xjtu_b1_b3(args.source_zip, args.output)


def _open_calibration(args: argparse.Namespace) -> None:
    from .common import load_hust_protocol
    from .freeze import validate_evaluator_receipt
    from .source import load_source_gate

    protocol = load_hust_protocol(args.config)
    _, source_receipt, _ = load_source_gate(
        args.archive,
        args.config,
        args.inventory,
    )
    validate_evaluator_receipt(
        args.evaluator_zip,
        args.evaluator_manifest,
        args.evaluator_receipt,
        args.models,
        protocol,
        args.project_root,
        source_receipt,
    )
    prepare_role(
        args.archive,
        args.config,
        args.inventory,
        args.output,
        "calibration",
        trust_official_pickle=args.trust_official_pickle,
    )


def _calibrate(args: argparse.Namespace) -> None:
    calibrate(
        args.config,
        args.dev_config,
        args.archive,
        args.inventory,
        args.evaluator_zip,
        args.evaluator_manifest,
        args.evaluator_receipt,
        args.models,
        args.calibration,
        args.device,
        args.project_root,
    )


def _freeze_evaluator(args: argparse.Namespace) -> None:
    freeze_evaluator(
        args.config,
        args.dev_config,
        args.archive,
        args.inventory,
        args.development,
        args.models,
        args.project_root,
        args.output_zip,
        args.calibration_preflight,
    )


def _register(args: argparse.Namespace) -> None:
    register_freeze(
        args.kind,
        args.archive,
        args.manifest,
        args.commit,
        args.locator,
        args.output,
        args.calibration_preflight,
        args.confirmation_state,
    )


def _freeze_calibration(args: argparse.Namespace) -> None:
    freeze_calibration(
        args.config,
        args.archive,
        args.inventory,
        args.evaluator_zip,
        args.evaluator_manifest,
        args.evaluator_receipt,
        args.models,
        args.calibration,
        args.project_root,
        args.output_zip,
    )


def _confirm(args: argparse.Namespace) -> None:
    confirm_one_shot(
        args.config,
        args.dev_config,
        args.archive,
        args.inventory,
        args.evaluator_zip,
        args.evaluator_manifest,
        args.evaluator_receipt,
        args.calibration_zip,
        args.calibration_manifest,
        args.calibration_receipt,
        args.calibration,
        args.models,
        args.output,
        args.device,
        args.project_root,
        resume=args.resume,
        trust_official_pickle=args.trust_official_pickle,
    )


def _aggregate(args: argparse.Namespace) -> None:
    aggregate_confirmation(args.config, args.confirmation, args.output)


def _pack(args: argparse.Namespace) -> None:
    pack_results(
        args.config,
        args.inventory,
        args.evaluator_manifest,
        args.evaluator_receipt,
        args.calibration_manifest,
        args.calibration_receipt,
        args.calibration,
        args.confirmation,
        args.aggregate,
        args.project_root,
        args.output_zip,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "MSTT_RUL v0.3.0 HUST structure-first calibration and one-shot "
            "external confirmation pipeline"
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)

    inventory = commands.add_parser(
        "inventory", help="Structure-only HUST inventory and frozen 20/57 split"
    )
    inventory.add_argument("--config", type=Path, required=True)
    inventory.add_argument("--archive", type=Path, required=True)
    inventory.add_argument("--output", type=Path, required=True)
    inventory.set_defaults(function=_inventory)

    xjtu = commands.add_parser(
        "extract-xjtu", help="Extract the bundled XJTU B1+B3 development CSVs"
    )
    xjtu.add_argument("--source-zip", type=Path, required=True)
    xjtu.add_argument("--output", type=Path, required=True)
    xjtu.set_defaults(function=_extract_xjtu)

    calibration = commands.add_parser(
        "open-calibration",
        help="Unpickle only the 20 frozen calibration cells",
    )
    calibration.add_argument("--config", type=Path, required=True)
    calibration.add_argument("--archive", type=Path, required=True)
    calibration.add_argument("--inventory", type=Path, required=True)
    calibration.add_argument("--output", type=Path, required=True)
    calibration.add_argument("--evaluator-zip", type=Path, required=True)
    calibration.add_argument("--evaluator-manifest", type=Path, required=True)
    calibration.add_argument("--evaluator-receipt", type=Path, required=True)
    calibration.add_argument("--models", type=Path, required=True)
    calibration.add_argument("--project-root", type=Path, required=True)
    calibration.add_argument(
        "--trust-official-pickle",
        action="store_true",
        help="Acknowledge that this is the official hash-bound HUST archive",
    )
    calibration.set_defaults(function=_open_calibration)

    calibration_run = commands.add_parser(
        "calibrate", help="Run frozen full-K5 calibration on the 20-cell arm"
    )
    calibration_run.add_argument("--config", type=Path, required=True)
    calibration_run.add_argument("--dev-config", type=Path, required=True)
    calibration_run.add_argument("--archive", type=Path, required=True)
    calibration_run.add_argument("--inventory", type=Path, required=True)
    calibration_run.add_argument("--evaluator-zip", type=Path, required=True)
    calibration_run.add_argument("--evaluator-manifest", type=Path, required=True)
    calibration_run.add_argument("--evaluator-receipt", type=Path, required=True)
    calibration_run.add_argument("--models", type=Path, required=True)
    calibration_run.add_argument("--calibration", type=Path, required=True)
    calibration_run.add_argument("--device", default="auto")
    calibration_run.add_argument("--project-root", type=Path, required=True)
    calibration_run.set_defaults(function=_calibrate)

    evaluator = commands.add_parser(
        "freeze-evaluator",
        help="Freeze exact source, protocol, development receipts, and weights",
    )
    evaluator.add_argument("--config", type=Path, required=True)
    evaluator.add_argument("--dev-config", type=Path, required=True)
    evaluator.add_argument("--archive", type=Path, required=True)
    evaluator.add_argument("--inventory", type=Path, required=True)
    evaluator.add_argument("--development", type=Path, required=True)
    evaluator.add_argument("--models", type=Path, required=True)
    evaluator.add_argument("--project-root", type=Path, required=True)
    evaluator.add_argument("--output-zip", type=Path, required=True)
    evaluator.add_argument("--calibration-preflight", type=Path, required=True)
    evaluator.set_defaults(function=_freeze_evaluator)

    registration = commands.add_parser(
        "register-freeze", help="Bind a freeze archive to commit and immutable locator"
    )
    registration.add_argument("--kind", choices=("evaluator", "calibration"), required=True)
    registration.add_argument("--archive", type=Path, required=True)
    registration.add_argument("--manifest", type=Path, required=True)
    registration.add_argument("--commit", required=True)
    registration.add_argument("--locator", required=True)
    registration.add_argument("--output", type=Path, required=True)
    registration.add_argument("--calibration-preflight", type=Path, required=True)
    registration.add_argument("--confirmation-state", type=Path, required=True)
    registration.set_defaults(function=_register)

    calibration_freeze = commands.add_parser(
        "freeze-calibration",
        help="Freeze calibration outputs before any confirmation member is opened",
    )
    calibration_freeze.add_argument("--config", type=Path, required=True)
    calibration_freeze.add_argument("--archive", type=Path, required=True)
    calibration_freeze.add_argument("--inventory", type=Path, required=True)
    calibration_freeze.add_argument("--evaluator-zip", type=Path, required=True)
    calibration_freeze.add_argument("--evaluator-manifest", type=Path, required=True)
    calibration_freeze.add_argument("--evaluator-receipt", type=Path, required=True)
    calibration_freeze.add_argument("--models", type=Path, required=True)
    calibration_freeze.add_argument("--calibration", type=Path, required=True)
    calibration_freeze.add_argument("--project-root", type=Path, required=True)
    calibration_freeze.add_argument("--output-zip", type=Path, required=True)
    calibration_freeze.set_defaults(function=_freeze_calibration)

    confirmation = commands.add_parser(
        "confirm", help="Open and evaluate the 57-cell confirmation arm once"
    )
    confirmation.add_argument("--config", type=Path, required=True)
    confirmation.add_argument("--dev-config", type=Path, required=True)
    confirmation.add_argument("--archive", type=Path, required=True)
    confirmation.add_argument("--inventory", type=Path, required=True)
    confirmation.add_argument("--evaluator-zip", type=Path, required=True)
    confirmation.add_argument("--evaluator-manifest", type=Path, required=True)
    confirmation.add_argument("--evaluator-receipt", type=Path, required=True)
    confirmation.add_argument("--calibration-zip", type=Path, required=True)
    confirmation.add_argument("--calibration-manifest", type=Path, required=True)
    confirmation.add_argument("--calibration-receipt", type=Path, required=True)
    confirmation.add_argument("--calibration", type=Path, required=True)
    confirmation.add_argument("--models", type=Path, required=True)
    confirmation.add_argument("--output", type=Path, required=True)
    confirmation.add_argument("--device", default="auto")
    confirmation.add_argument("--project-root", type=Path, required=True)
    confirmation.add_argument("--resume", action="store_true")
    confirmation.add_argument(
        "--trust-official-pickle",
        action="store_true",
        help="Acknowledge that this is the official hash-bound HUST archive",
    )
    confirmation.set_defaults(function=_confirm)

    aggregate = commands.add_parser(
        "aggregate", help="Cell-first fixed-sequence inference and figures"
    )
    aggregate.add_argument("--config", type=Path, required=True)
    aggregate.add_argument("--confirmation", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    aggregate.set_defaults(function=_aggregate)

    pack = commands.add_parser("pack", help="Create the final audited result archive")
    pack.add_argument("--config", type=Path, required=True)
    pack.add_argument("--inventory", type=Path, required=True)
    pack.add_argument("--evaluator-manifest", type=Path, required=True)
    pack.add_argument("--evaluator-receipt", type=Path, required=True)
    pack.add_argument("--calibration-manifest", type=Path, required=True)
    pack.add_argument("--calibration-receipt", type=Path, required=True)
    pack.add_argument("--calibration", type=Path, required=True)
    pack.add_argument("--confirmation", type=Path, required=True)
    pack.add_argument("--aggregate", type=Path, required=True)
    pack.add_argument("--project-root", type=Path, required=True)
    pack.add_argument("--output-zip", type=Path, required=True)
    pack.set_defaults(function=_pack)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    arguments.function(arguments)


if __name__ == "__main__":
    main()
