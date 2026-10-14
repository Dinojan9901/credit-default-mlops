"""Dataset download, loading, fingerprinting and splitting."""

from __future__ import annotations

import hashlib
import io
import logging
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from credit_default.config import (
    DATASET_URL,
    RAW_FEATURES,
    RAW_XLS_NAME,
    RAW_XLS_SHA256,
    TARGET,
)

logger = logging.getLogger(__name__)

# The UCI file uses PAY_0 for the most recent month but BILL_AMT1/PAY_AMT1 for the
# same month; we rename to pay_1 so the three month-indexed families line up.
_RENAME = {"PAY_0": "pay_1", "default payment next month": TARGET}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=_RENAME)
    df.columns = [str(c).lower() for c in df.columns]
    return df.drop(columns=["id"]).loc[:, [*RAW_FEATURES, TARGET]]


def download_dataset(csv_path: Path, url: str = DATASET_URL, force: bool = False) -> Path:
    """Fetch the UCI archive, verify its checksum and store it as a tidy CSV."""
    if csv_path.exists() and not force:
        logger.info("Dataset already present at %s", csv_path)
        return csv_path

    logger.info("Downloading %s", url)
    with urllib.request.urlopen(url, timeout=60) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        xls_bytes = zf.read(RAW_XLS_NAME)

    digest = sha256_bytes(xls_bytes)
    if digest != RAW_XLS_SHA256:
        raise RuntimeError(
            f"Checksum mismatch for {RAW_XLS_NAME}: expected {RAW_XLS_SHA256}, got {digest}"
        )

    df = pd.read_excel(io.BytesIO(xls_bytes), header=1, engine="xlrd")
    df = _normalise_columns(df)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, lineterminator="\n")
    logger.info("Wrote %d rows to %s", len(df), csv_path)
    return csv_path


def load_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found - run `credit-default download` first")
    return pd.read_csv(csv_path)


def dataset_fingerprint(df: pd.DataFrame) -> str:
    """Content hash of a DataFrame, independent of file encoding or row index.

    Used as the dataset version in lineage tags: same rows + columns -> same hash.
    """
    row_hashes = pd.util.hash_pandas_object(df, index=False).to_numpy()
    header = ",".join(map(str, df.columns)).encode()
    return sha256_bytes(header + row_hashes.tobytes())


@dataclass(frozen=True)
class DataSplits:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series

    def sizes(self) -> dict[str, int]:
        return {"train": len(self.X_train), "val": len(self.X_val), "test": len(self.X_test)}


def split_dataset(
    df: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> DataSplits:
    """Stratified train / validation / test split (default 70 / 15 / 15).

    - train: hyperparameter search with stratified k-fold cross-validation
    - val:   model selection (baseline vs boosted) and decision-threshold choice
    - test:  touched once, for the final reported metrics and the promotion gate
    """
    X = df.loc[:, list(RAW_FEATURES)]
    y = df[TARGET]
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    rel_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=rel_val, stratify=y_rest, random_state=seed
    )
    return DataSplits(X_train, X_val, X_test, y_train, y_val, y_test)
