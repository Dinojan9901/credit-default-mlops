"""Command-line entry point: ``credit-default <command>``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from credit_default.config import PRODUCTION_ALIAS, REGISTERED_MODEL_NAME, get_settings


def _cmd_download(args: argparse.Namespace) -> int:
    from credit_default.data import download_dataset

    download_dataset(get_settings().csv_path, force=args.force)
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    from credit_default.data import load_dataset
    from credit_default.validation import validate_training_data

    report = validate_training_data(load_dataset(get_settings().csv_path))
    print(json.dumps(report.to_dict(), indent=2))
    return 0 if report.ok else 1


def _cmd_train(args: argparse.Namespace) -> int:
    from credit_default.train import TrainConfig, run_training

    cfg = TrainConfig(
        n_trials=args.n_trials,
        cv_folds=args.cv_folds,
        seed=args.seed,
        search_seed=args.search_seed,
        min_delta=args.min_delta,
        register=not args.no_register,
    )
    summary = run_training(cfg, get_settings())
    if args.summary_out:
        Path(args.summary_out).write_text(json.dumps(summary.to_dict(), indent=2))
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    import mlflow
    from mlflow.tracking import MlflowClient

    from credit_default.registry import export_model

    mlflow.set_tracking_uri(get_settings().tracking_uri)
    export_model(MlflowClient(), args.name, args.alias, Path(args.out))
    return 0


def _cmd_simulate(args: argparse.Namespace) -> int:
    from credit_default.monitoring.simulate import write_batches

    write_batches(get_settings(), n_rows=args.n_rows, seed=args.seed)
    return 0


def _cmd_drift(args: argparse.Namespace) -> int:
    from credit_default.monitoring.drift import run_drift_check

    result = run_drift_check(
        current_path=Path(args.current),
        settings=get_settings(),
        threshold=args.threshold,
        model_uri=args.model_uri,
        model_dir=Path(args.model_dir) if args.model_dir else None,
    )
    print(json.dumps(result.to_dict(), indent=2))
    return 2 if result.alert else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="credit-default", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="download + checksum the UCI dataset")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=_cmd_download)

    p = sub.add_parser("validate", help="run data validation and print the report")
    p.set_defaults(func=_cmd_validate)

    p = sub.add_parser("train", help="tune, train, evaluate, register and gate-promote")
    p.add_argument("--n-trials", type=int, default=40)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument(
        "--seed", type=int, default=get_settings().seed, help="train/val/test split seed"
    )
    p.add_argument(
        "--search-seed",
        type=int,
        default=None,
        help="seed for CV folds, Optuna and the model (defaults to --seed)",
    )
    p.add_argument(
        "--min-delta",
        type=float,
        default=0.001,
        help="ROC-AUC improvement required to replace the production model",
    )
    p.add_argument("--no-register", action="store_true")
    p.add_argument("--summary-out", help="write the training summary JSON here")
    p.set_defaults(func=_cmd_train)

    p = sub.add_parser("export-model", help="download a registered model to a directory")
    p.add_argument("--name", default=REGISTERED_MODEL_NAME)
    p.add_argument("--alias", default=PRODUCTION_ALIAS)
    p.add_argument("--out", default="build/model-export")
    p.set_defaults(func=_cmd_export)

    p = sub.add_parser("simulate-drift", help="write reference / no-drift / drifted batches")
    p.add_argument("--n-rows", type=int, default=2000)
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(func=_cmd_simulate)

    p = sub.add_parser("drift", help="Evidently drift report for a new batch (exit 2 = alert)")
    p.add_argument("--current", required=True, help="CSV with the new batch")
    p.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="alert if the share of drifted columns exceeds this",
    )
    p.add_argument("--model-uri", default=f"models:/{REGISTERED_MODEL_NAME}@{PRODUCTION_ALIAS}")
    p.add_argument("--model-dir", help="use an exported model directory instead of the registry")
    p.set_defaults(func=_cmd_drift)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
