from __future__ import annotations

import pandas as pd

from credit_default.config import RAW_FEATURES, TARGET
from credit_default.data import _normalise_columns, dataset_fingerprint, split_dataset


def test_split_sizes_and_stratification(synthetic_df: pd.DataFrame) -> None:
    splits = split_dataset(synthetic_df, seed=1)
    sizes = splits.sizes()
    assert sum(sizes.values()) == len(synthetic_df)
    assert abs(sizes["test"] - 0.15 * len(synthetic_df)) <= 1
    assert abs(sizes["val"] - 0.15 * len(synthetic_df)) <= 1
    overall = synthetic_df[TARGET].mean()
    for y in (splits.y_train, splits.y_val, splits.y_test):
        assert abs(y.mean() - overall) < 0.02
    # no row lands in two splits
    idx = [set(s.index) for s in (splits.X_train, splits.X_val, splits.X_test)]
    assert not (idx[0] & idx[1]) and not (idx[0] & idx[2]) and not (idx[1] & idx[2])


def test_split_is_deterministic(synthetic_df: pd.DataFrame) -> None:
    a = split_dataset(synthetic_df, seed=3)
    b = split_dataset(synthetic_df, seed=3)
    assert a.X_test.index.equals(b.X_test.index)


def test_fingerprint_stable_and_content_sensitive(synthetic_df: pd.DataFrame) -> None:
    h1 = dataset_fingerprint(synthetic_df)
    assert h1 == dataset_fingerprint(synthetic_df.copy())
    assert h1 == dataset_fingerprint(synthetic_df.set_axis(range(100, 100 + len(synthetic_df))))
    changed = synthetic_df.copy()
    changed.loc[0, "age"] += 1
    assert dataset_fingerprint(changed) != h1


def test_normalise_columns_maps_uci_names() -> None:
    raw = {"ID": [1], "PAY_0": [0], "default payment next month": [1]}
    for col in RAW_FEATURES:
        if col != "pay_1":
            raw[col.upper()] = [1]
    out = _normalise_columns(pd.DataFrame(raw))
    assert list(out.columns) == [*RAW_FEATURES, TARGET]
