"""Project-wide constants and environment-driven settings."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --- Dataset -----------------------------------------------------------------
# UCI "Default of Credit Card Clients" (Yeh, 2009), CC BY 4.0.
DATASET_NAME = "uci-default-of-credit-card-clients"
DATASET_URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
DATASET_DOI = "10.24432/C55S3H"
DATASET_LICENSE = "CC BY 4.0"
# SHA-256 of the .xls inside the UCI zip; guards against silent upstream changes.
RAW_XLS_SHA256 = "30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933"
RAW_XLS_NAME = "default of credit card clients.xls"
CSV_NAME = "credit_default.csv"

TARGET = "default"
CATEGORICAL_FEATURES: tuple[str, ...] = ("sex", "education", "marriage")
PAY_STATUS = tuple(f"pay_{i}" for i in range(1, 7))
BILL_AMOUNTS = tuple(f"bill_amt{i}" for i in range(1, 7))
PAY_AMOUNTS = tuple(f"pay_amt{i}" for i in range(1, 7))
NUMERIC_FEATURES: tuple[str, ...] = ("limit_bal", "age", *PAY_STATUS, *BILL_AMOUNTS, *PAY_AMOUNTS)
RAW_FEATURES: tuple[str, ...] = (
    "limit_bal",
    "sex",
    "education",
    "marriage",
    "age",
    *PAY_STATUS,
    *BILL_AMOUNTS,
    *PAY_AMOUNTS,
)

# --- MLflow ------------------------------------------------------------------
EXPERIMENT_NAME = "credit-default"
REGISTERED_MODEL_NAME = "credit-default-classifier"
PRODUCTION_ALIAS = "production"
CHALLENGER_ALIAS = "challenger"


def _env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default))


@dataclass(frozen=True)
class Settings:
    """Runtime settings, overridable through environment variables."""

    data_dir: Path = field(default_factory=lambda: _env_path("CD_DATA_DIR", "data"))
    reports_dir: Path = field(default_factory=lambda: _env_path("CD_REPORTS_DIR", "reports"))
    tracking_uri: str = field(
        default_factory=lambda: os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    )
    # Only used when the experiment is first created against a local backend; a remote
    # tracking server decides artifact storage itself (e.g. --serve-artifacts).
    artifact_root: str | None = field(default_factory=lambda: os.environ.get("CD_ARTIFACT_ROOT"))
    seed: int = field(default_factory=lambda: int(os.environ.get("CD_SEED", "42")))

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def csv_path(self) -> Path:
        return self.raw_dir / CSV_NAME


def get_settings() -> Settings:
    return Settings()
