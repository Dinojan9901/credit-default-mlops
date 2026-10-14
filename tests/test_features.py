from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from credit_default.config import RAW_FEATURES, TARGET
from credit_default.evaluation import choose_threshold, classification_metrics
from credit_default.features import ENGINEERED_FEATURES, FeatureEngineer, build_pipeline


def test_feature_engineer_adds_columns_and_folds_codes(synthetic_df: pd.DataFrame) -> None:
    out = FeatureEngineer().fit_transform(synthetic_df[list(RAW_FEATURES)])
    assert set(ENGINEERED_FEATURES) <= set(out.columns)
    assert set(out["education"].unique()) <= {1, 2, 3, 4}
    assert set(out["marriage"].unique()) <= {1, 2, 3}
    assert np.isfinite(out[list(ENGINEERED_FEATURES)].to_numpy()).all()


def test_payment_ratio_defaults_to_one_without_bills(synthetic_df: pd.DataFrame) -> None:
    row = synthetic_df[list(RAW_FEATURES)].head(1).copy()
    for i in range(1, 7):
        row[f"bill_amt{i}"] = 0
        row[f"pay_amt{i}"] = 0
    out = FeatureEngineer().transform(row)
    assert out["payment_ratio_avg"].iloc[0] == 1.0


def test_transform_does_not_mutate_input(synthetic_df: pd.DataFrame) -> None:
    X = synthetic_df[list(RAW_FEATURES)]
    before = X.copy()
    FeatureEngineer().transform(X)
    pd.testing.assert_frame_equal(X, before)


def test_pipeline_fits_and_handles_unseen_category(synthetic_df: pd.DataFrame) -> None:
    X, y = synthetic_df[list(RAW_FEATURES)], synthetic_df[TARGET]
    pipe = build_pipeline(LogisticRegression(max_iter=1000)).fit(X, y)
    proba = pipe.predict_proba(X.head(3).assign(sex=7))[:, 1]  # unseen sex code
    assert proba.shape == (3,)
    assert ((proba >= 0) & (proba <= 1)).all()
    names = pipe[:-1].get_feature_names_out()
    assert "utilization_avg" in names and "sex_1" in names


def test_threshold_and_metrics() -> None:
    y = np.array([0, 0, 1, 1, 0, 1])
    p = np.array([0.1, 0.2, 0.8, 0.7, 0.3, 0.9])
    t = choose_threshold(y, p)
    m = classification_metrics(y, p, t)
    assert m["roc_auc"] == 1.0
    assert m["f1"] == 1.0
    assert 0.3 < t <= 0.7
