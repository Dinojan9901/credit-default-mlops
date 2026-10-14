"""Feature engineering and preprocessing, packaged as scikit-learn transformers.

Everything that turns raw columns into model inputs lives inside the fitted Pipeline,
so training, cross-validation and serving all apply exactly the same transformations
(no train/serve skew, no leakage of validation folds into scaler statistics).
"""

from __future__ import annotations

from typing import Any, Self

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from credit_default.config import (
    BILL_AMOUNTS,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    PAY_AMOUNTS,
    PAY_STATUS,
)

ENGINEERED_FEATURES: tuple[str, ...] = (
    "utilization_last",
    "utilization_avg",
    "payment_ratio_avg",
    "bill_trend",
    "max_delay",
    "n_months_delayed",
)


class FeatureEngineer(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Stateless domain features derived from the raw credit-card columns."""

    def fit(self, X: pd.DataFrame, y: Any = None) -> Self:
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        # Undocumented codes are folded into the documented "other" bucket.
        out["education"] = out["education"].where(out["education"].isin([1, 2, 3, 4]), 4)
        out["marriage"] = out["marriage"].where(out["marriage"].isin([1, 2, 3]), 3)

        limit = out["limit_bal"].astype(float)
        bills = out[list(BILL_AMOUNTS)].astype(float)
        payments = out[list(PAY_AMOUNTS)].astype(float)
        status = out[list(PAY_STATUS)]

        out["utilization_last"] = bills["bill_amt1"] / limit
        out["utilization_avg"] = bills.mean(axis=1) / limit
        # Share of the outstanding bills that was actually paid; no positive bill -> 1.0.
        total_bill = bills.clip(lower=0).sum(axis=1)
        ratio = payments.sum(axis=1) / total_bill.where(total_bill > 0)
        out["payment_ratio_avg"] = ratio.fillna(1.0).clip(0.0, 10.0)
        out["bill_trend"] = (bills["bill_amt1"] - bills["bill_amt6"]) / limit
        out["max_delay"] = status.max(axis=1).clip(lower=0)
        out["n_months_delayed"] = (status > 0).sum(axis=1)
        return out

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        base = list(input_features if input_features is not None else self.feature_names_in_)
        return np.asarray([*base, *ENGINEERED_FEATURES], dtype=object)


def build_preprocessor() -> ColumnTransformer:
    numeric = [*NUMERIC_FEATURES, *ENGINEERED_FEATURES]
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64),
                list(CATEGORICAL_FEATURES),
            ),
            ("num", StandardScaler(), numeric),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(estimator: ClassifierMixin) -> Pipeline:
    pipe = Pipeline(
        steps=[
            ("features", FeatureEngineer()),
            ("preprocess", build_preprocessor()),
            ("model", estimator),
        ]
    )
    # DataFrame outputs keep feature names, which LightGBM and SHAP both use.
    pipe[:-1].set_output(transform="pandas")
    return pipe
