from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_default.config import TARGET
from credit_default.validation import (
    DataValidationError,
    assert_valid,
    validate_training_data,
)


def test_valid_data_passes(synthetic_df: pd.DataFrame) -> None:
    report = validate_training_data(synthetic_df)
    assert report.ok, report.errors
    assert 0 < report.stats["target_rate"] < 1
    assert_valid(report)


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda d: d.drop(columns=["age"]), "missing columns"),
        (lambda d: d.assign(age=d["age"].astype(float).where(d.index != 0, np.nan)), "null"),
        (lambda d: d.assign(sex=d["sex"].where(d.index != 0, 9)), "sex"),
        (lambda d: d.assign(pay_3=d["pay_3"].where(d.index != 0, 12)), "pay_3"),
        (lambda d: d.assign(age=d["age"].where(d.index != 0, 7)), "age"),
        (lambda d: d.assign(limit_bal=d["limit_bal"].where(d.index != 0, -5)), "limit_bal"),
        (lambda d: d.assign(**{TARGET: d[TARGET].where(d.index != 0, 3)}), TARGET),
        (lambda d: d.assign(**{TARGET: 0}), "target rate"),
        (lambda d: d.head(10), "rows"),
        (lambda d: d.assign(age=d["age"].astype(str)), "numeric dtype"),
    ],
)
def test_invalid_data_fails(synthetic_df: pd.DataFrame, mutate, expected: str) -> None:  # type: ignore[no-untyped-def]
    report = validate_training_data(mutate(synthetic_df))
    assert not report.ok
    assert any(expected in e for e in report.errors), report.errors
    with pytest.raises(DataValidationError):
        assert_valid(report)


def test_duplicates_are_a_warning_not_an_error(synthetic_df: pd.DataFrame) -> None:
    df = pd.concat([synthetic_df, synthetic_df.head(3)], ignore_index=True)
    report = validate_training_data(df)
    assert report.ok
    assert report.stats["duplicate_rows"] >= 3
