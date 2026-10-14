"""Lightweight, explicit data validation for the raw training table.

Hard failures (``errors``) stop the pipeline; soft findings (``warnings``) are logged
and stored with the run so they show up in MLflow next to the model they affected.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from credit_default.config import (
    BILL_AMOUNTS,
    PAY_AMOUNTS,
    PAY_STATUS,
    RAW_FEATURES,
    TARGET,
)

# Allowed value domains, taken from the UCI dataset documentation. EDUCATION 0/5/6 and
# MARRIAGE 0 are undocumented but present in the source data, so they are allowed here
# and folded into "other" during feature engineering.
ALLOWED_VALUES: dict[str, set[int]] = {
    "sex": {1, 2},
    "education": {0, 1, 2, 3, 4, 5, 6},
    "marriage": {0, 1, 2, 3},
    **{c: set(range(-2, 9)) for c in PAY_STATUS},
}
RANGES: dict[str, tuple[float, float]] = {
    "limit_bal": (1.0, 10_000_000.0),
    "age": (18.0, 100.0),
    **dict.fromkeys(PAY_AMOUNTS, (0.0, 10000000.0)),
    **dict.fromkeys(BILL_AMOUNTS, (-10000000.0, 10000000.0)),
}
MIN_ROWS = 1_000
TARGET_RATE_BOUNDS = (0.05, 0.50)


class DataValidationError(ValueError):
    """Raised when the dataset fails a hard validation check."""


@dataclass
class ValidationReport:
    n_rows: int
    n_columns: int
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "ok": self.ok}


def validate_features(df: pd.DataFrame, report: ValidationReport) -> None:
    """Checks that apply to any feature frame (training data or a new scoring batch)."""
    missing = [c for c in RAW_FEATURES if c not in df.columns]
    if missing:
        report.errors.append(f"missing columns: {missing}")
        return

    nulls = df[list(RAW_FEATURES)].isna().sum()
    for col, n in nulls[nulls > 0].items():
        report.errors.append(f"{col}: {n} null values")

    for col in RAW_FEATURES:
        if not pd.api.types.is_numeric_dtype(df[col]):
            report.errors.append(f"{col}: expected numeric dtype, got {df[col].dtype}")

    if report.errors:
        return

    for col, allowed in ALLOWED_VALUES.items():
        bad = ~df[col].isin(allowed)
        if bad.any():
            report.errors.append(
                f"{col}: {int(bad.sum())} values outside {sorted(allowed)} "
                f"(e.g. {df.loc[bad, col].iloc[0]})"
            )
    for col, (lo, hi) in RANGES.items():
        bad = ~df[col].between(lo, hi)
        if bad.any():
            report.errors.append(f"{col}: {int(bad.sum())} values outside [{lo}, {hi}]")


def validate_training_data(df: pd.DataFrame) -> ValidationReport:
    report = ValidationReport(n_rows=len(df), n_columns=df.shape[1])

    if len(df) < MIN_ROWS:
        report.errors.append(f"only {len(df)} rows, need at least {MIN_ROWS}")

    validate_features(df, report)

    if TARGET not in df.columns:
        report.errors.append(f"missing target column {TARGET!r}")
    else:
        bad_target = ~df[TARGET].isin({0, 1})
        if bad_target.any():
            report.errors.append(f"{TARGET}: {int(bad_target.sum())} values not in {{0, 1}}")
        else:
            rate = float(df[TARGET].mean())
            report.stats["target_rate"] = round(rate, 4)
            lo, hi = TARGET_RATE_BOUNDS
            if not lo <= rate <= hi:
                report.errors.append(f"target rate {rate:.3f} outside [{lo}, {hi}]")

    n_dupes = int(df.duplicated().sum())
    report.stats["duplicate_rows"] = n_dupes
    if n_dupes:
        report.warnings.append(f"{n_dupes} fully duplicated rows (kept)")

    undocumented = int((~df["education"].isin({1, 2, 3, 4})).sum()) if "education" in df else 0
    if undocumented:
        report.warnings.append(f"education: {undocumented} undocumented codes (0/5/6) -> 'other'")

    return report


def assert_valid(report: ValidationReport) -> None:
    if not report.ok:
        raise DataValidationError("; ".join(report.errors))
