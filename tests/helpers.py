"""Shared test helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from credit_default.config import BILL_AMOUNTS, PAY_AMOUNTS, PAY_STATUS, TARGET


def make_synthetic(n: int = 1200, seed: int = 0) -> pd.DataFrame:
    """Schema-valid fake data with a learnable signal (late payers default more)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "limit_bal": rng.integers(10, 500, n) * 1000,
            "sex": rng.integers(1, 3, n),
            "education": rng.integers(0, 7, n),
            "marriage": rng.integers(0, 4, n),
            "age": rng.integers(21, 75, n),
        }
    )
    for col in PAY_STATUS:
        df[col] = rng.choice([-2, -1, 0, 0, 0, 1, 2, 3], n)
    for col in BILL_AMOUNTS:
        df[col] = rng.integers(-1000, 200_000, n)
    for col in PAY_AMOUNTS:
        df[col] = rng.integers(0, 20_000, n)
    logit = -1.6 + 0.8 * df["pay_1"].clip(lower=0) - 0.000002 * df["limit_bal"]
    df[TARGET] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df
