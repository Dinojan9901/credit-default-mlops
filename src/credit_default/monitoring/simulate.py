"""Simulated "new data" batches for exercising drift detection.

The UCI dataset is a single static snapshot, so production traffic is simulated:

* reference.csv        rows from the training split (what the model learned from)
* current_nodrift.csv  unseen rows from the test split - same distribution, should NOT alert
* current_drift.csv    unseen test rows with a plausible economic-downturn shift applied:
                       lower credit limits, more late payments, higher bills, lower
                       repayments - should alert
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from credit_default.config import BILL_AMOUNTS, PAY_AMOUNTS, Settings
from credit_default.data import download_dataset, load_dataset, split_dataset

logger = logging.getLogger(__name__)


def apply_downturn_shift(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Shift feature distributions while keeping every row schema-valid."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    out["limit_bal"] = (out["limit_bal"] * 0.6).round(-4).clip(lower=10_000).astype("int64")
    worse = rng.random(len(out)) < 0.4
    for col in ("pay_1", "pay_2", "pay_3"):
        out.loc[worse, col] = (out.loc[worse, col].clip(lower=0) + 2).clip(upper=8)
    for col in BILL_AMOUNTS:
        out[col] = (out[col] * 1.3).round().clip(-10_000_000, 10_000_000).astype("int64")
    for col in PAY_AMOUNTS:
        out[col] = (out[col] * 0.5).round().astype("int64")
    return out


def monitoring_dir(settings: Settings) -> Path:
    return settings.data_dir / "monitoring"


def write_batches(settings: Settings, n_rows: int = 2000, seed: int = 7) -> dict[str, Path]:
    download_dataset(settings.csv_path)
    splits = split_dataset(load_dataset(settings.csv_path), seed=settings.seed)
    train = splits.X_train.assign(default=splits.y_train)
    test = splits.X_test.assign(default=splits.y_test)

    n_ref = min(n_rows * 2, len(train))
    n_cur = min(n_rows, len(test))
    batches = {
        "reference": train.sample(n_ref, random_state=seed),
        "current_nodrift": test.sample(n_cur, random_state=seed),
        "current_drift": apply_downturn_shift(test.sample(n_cur, random_state=seed + 1), seed),
    }
    out_dir = monitoring_dir(settings)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, frame in batches.items():
        paths[name] = out_dir / f"{name}.csv"
        frame.to_csv(paths[name], index=False)
        logger.info("Wrote %s (%d rows)", paths[name], len(frame))
    return paths
