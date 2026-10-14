"""Candidate models and their hyperparameter searches.

Both searches score candidates with stratified k-fold cross-validation on the training
split only, so the validation and test splits never influence hyperparameters.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from credit_default.features import build_pipeline

LOGREG_GRID: dict[str, list[Any]] = {"model__C": [0.01, 0.1, 1.0, 10.0]}


@dataclass
class TuningResult:
    name: str
    best_params: dict[str, Any]
    cv_mean: float
    cv_std: float
    pipeline: Pipeline  # unfitted, configured with best_params
    history: pd.DataFrame


def make_cv(folds: int, seed: int) -> StratifiedKFold:
    return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)


def logreg_pipeline(**params: Any) -> Pipeline:
    return build_pipeline(LogisticRegression(max_iter=2000, **params))


def lightgbm_pipeline(seed: int, **params: Any) -> Pipeline:
    return build_pipeline(LGBMClassifier(random_state=seed, n_jobs=-1, verbosity=-1, **params))


def tune_logreg(X: pd.DataFrame, y: pd.Series, folds: int, seed: int) -> TuningResult:
    """Baseline: small grid over the regularisation strength C."""
    search = GridSearchCV(
        logreg_pipeline(),
        LOGREG_GRID,
        scoring="roc_auc",
        cv=make_cv(folds, seed),
        n_jobs=1,
    )
    search.fit(X, y)
    i = search.best_index_
    history = pd.DataFrame(search.cv_results_)[
        ["param_model__C", "mean_test_score", "std_test_score", "rank_test_score"]
    ]
    return TuningResult(
        name="logreg",
        best_params={k.removeprefix("model__"): v for k, v in search.best_params_.items()},
        cv_mean=float(search.cv_results_["mean_test_score"][i]),
        cv_std=float(search.cv_results_["std_test_score"][i]),
        pipeline=clone(search.best_estimator_),
        history=history,
    )


def lightgbm_search_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 8, 128, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 200, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }


TrialCallback = Callable[[optuna.Study, optuna.trial.FrozenTrial], None]


def tune_lightgbm(
    X: pd.DataFrame,
    y: pd.Series,
    folds: int,
    seed: int,
    n_trials: int,
    callbacks: list[TrialCallback] | None = None,
) -> TuningResult:
    """Bayesian search (Optuna TPE) with median pruning on per-fold ROC-AUC."""
    cv = make_cv(folds, seed)
    splits = list(cv.split(X, y))

    def objective(trial: optuna.Trial) -> float:
        params = lightgbm_search_space(trial)
        scores: list[float] = []
        for step, (tr, va) in enumerate(splits):
            model = lightgbm_pipeline(seed, **params)
            model.fit(X.iloc[tr], y.iloc[tr])
            proba = model.predict_proba(X.iloc[va])[:, 1]
            scores.append(float(roc_auc_score(y.iloc[va], proba)))
            trial.report(float(np.mean(scores)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("cv_std", float(np.std(scores)))
        return float(np.mean(scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        study_name="lightgbm",
    )
    study.optimize(objective, n_trials=n_trials, callbacks=callbacks or [])

    best = study.best_trial
    best_params = {**best.params, "subsample_freq": 1}
    return TuningResult(
        name="lightgbm",
        best_params=best_params,
        cv_mean=float(best.value) if best.value is not None else float("nan"),
        cv_std=float(best.user_attrs.get("cv_std", float("nan"))),
        pipeline=lightgbm_pipeline(seed, **best_params),
        history=study.trials_dataframe(attrs=("number", "value", "params", "state")),
    )
