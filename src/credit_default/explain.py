"""SHAP explanations for a fitted pipeline."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: CI runners and containers have no display

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def shap_values(
    pipeline: Pipeline, X: pd.DataFrame, background_size: int = 200
) -> shap.Explanation:
    """SHAP values (log-odds scale) of the final estimator on transformed features."""
    Xt = pipeline[:-1].transform(X)
    model = pipeline[-1]
    if isinstance(model, LGBMClassifier):
        explainer = shap.TreeExplainer(model)
    elif isinstance(model, LogisticRegression):
        background = Xt.sample(min(background_size, len(Xt)), random_state=0)
        explainer = shap.LinearExplainer(model, background)
    else:
        raise TypeError(f"No SHAP explainer configured for {type(model).__name__}")
    explanation = explainer(Xt)
    if explanation.values.ndim == 3:  # some versions return one slice per class
        explanation = explanation[:, :, 1]
    return explanation


def write_shap_artifacts(
    pipeline: Pipeline, X: pd.DataFrame, out_dir: Path, max_display: int = 15
) -> dict[str, Path]:
    """Beeswarm + bar summary plots and a mean |SHAP| table."""
    out_dir.mkdir(parents=True, exist_ok=True)
    explanation = shap_values(pipeline, X)

    paths = {
        "beeswarm": out_dir / "shap_summary_beeswarm.png",
        "bar": out_dir / "shap_summary_bar.png",
        "importance": out_dir / "shap_mean_abs.csv",
    }

    plt.figure()
    shap.plots.beeswarm(explanation, max_display=max_display, show=False)
    plt.title("SHAP summary (impact on log-odds of default)")
    plt.tight_layout()
    plt.savefig(paths["beeswarm"], dpi=120, bbox_inches="tight")
    plt.close("all")

    plt.figure()
    shap.plots.bar(explanation, max_display=max_display, show=False)
    plt.title("Mean |SHAP| per feature")
    plt.tight_layout()
    plt.savefig(paths["bar"], dpi=120, bbox_inches="tight")
    plt.close("all")

    importance = (
        pd.DataFrame(
            {
                "feature": explanation.feature_names,
                "mean_abs_shap": np.abs(explanation.values).mean(axis=0),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    importance.to_csv(paths["importance"], index=False)
    return paths
