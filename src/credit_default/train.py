"""End-to-end training pipeline: validate -> split -> tune -> select -> test -> register.

MLflow run layout (one experiment, nested runs):

    training-pipeline                 lineage tags, data validation, final decision
    |- logreg-baseline                grid search CV, val metrics, logged model
    |- lightgbm-optuna                Optuna CV, val metrics, logged model
    |  |- trial-000 ... trial-NNN     one run per Optuna trial
    (the winning child also gets test metrics + SHAP artifacts)
"""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mlflow
import optuna
import pandas as pd
from mlflow.data.pandas_dataset import from_pandas
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.pipeline import Pipeline

from credit_default.config import (
    DATASET_NAME,
    DATASET_URL,
    EXPERIMENT_NAME,
    REGISTERED_MODEL_NAME,
    TARGET,
    Settings,
)
from credit_default.data import (
    DataSplits,
    dataset_fingerprint,
    download_dataset,
    load_dataset,
    split_dataset,
)
from credit_default.evaluation import PRIMARY_METRIC, choose_threshold, classification_metrics
from credit_default.explain import write_shap_artifacts
from credit_default.lineage import lineage_tags
from credit_default.models import TuningResult, tune_lightgbm, tune_logreg
from credit_default.registry import PromotionDecision, promote_if_better
from credit_default.validation import assert_valid, validate_training_data

logger = logging.getLogger(__name__)

# MLflow stores sklearn models with skops (no arbitrary code execution on load, unlike
# pickle). skops only loads types it is told to trust; these are the non-sklearn ones.
SKOPS_TRUSTED_TYPES = [
    "collections.OrderedDict",
    "credit_default.features.FeatureEngineer",
    "lightgbm.basic.Booster",
    "lightgbm.sklearn.LGBMClassifier",
]


@dataclass
class TrainConfig:
    n_trials: int = 40
    cv_folds: int = 5
    seed: int = 42  # split seed: fixed, so champion and challenger share one test set
    search_seed: int | None = None  # CV folds / Optuna / model seed; vary it per retrain
    min_delta: float = 0.001
    shap_sample: int = 1000
    register: bool = True

    @property
    def model_seed(self) -> int:
        return self.seed if self.search_seed is None else self.search_seed


@dataclass
class Candidate:
    name: str
    run_id: str
    pipeline: Pipeline
    model_uri: str
    val_metrics: dict[str, float]
    threshold: float


@dataclass
class TrainingSummary:
    parent_run_id: str
    champion: str
    threshold: float
    val_metrics: dict[str, dict[str, float]]
    test_metrics: dict[str, float]
    model_version: str | None
    decision: PromotionDecision | None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["decision"] = self.decision.to_dict() if self.decision else None
        return d


def _prefixed(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{k}": v for k, v in metrics.items()}


def _log_frame(df: pd.DataFrame, artifact_file: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / Path(artifact_file).name
        df.to_csv(path, index=False)
        mlflow.log_artifact(str(path), artifact_path=str(Path(artifact_file).parent))


def _optuna_mlflow_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
    """Record each Optuna trial as a nested MLflow run under the LightGBM run."""
    with mlflow.start_run(run_name=f"trial-{trial.number:03d}", nested=True):
        mlflow.log_params(trial.params)
        mlflow.set_tag("optuna_state", trial.state.name)
        if trial.value is not None:
            mlflow.log_metric("cv_roc_auc", trial.value)
        if "cv_std" in trial.user_attrs:
            mlflow.log_metric("cv_roc_auc_std", trial.user_attrs["cv_std"])


def _fit_and_log_candidate(
    result: TuningResult, splits: DataSplits, run_id: str, tags: dict[str, str]
) -> Candidate:
    pipeline = result.pipeline.fit(splits.X_train, splits.y_train)
    val_proba = pipeline.predict_proba(splits.X_val)[:, 1]
    threshold = choose_threshold(splits.y_val, val_proba)
    val_metrics = classification_metrics(splits.y_val, val_proba, threshold)

    mlflow.set_tags({**tags, "model_type": result.name})
    mlflow.log_params({f"best_{k}": v for k, v in result.best_params.items()})
    mlflow.log_metrics({"cv_roc_auc": result.cv_mean, "cv_roc_auc_std": result.cv_std})
    mlflow.log_metrics({**_prefixed("val", val_metrics), "decision_threshold": threshold})
    _log_frame(result.history, "tuning/search_history.csv")

    example = splits.X_train.head(5)
    signature = infer_signature(example, pipeline.predict_proba(example)[:, 1])
    info = mlflow.sklearn.log_model(
        pipeline,
        name="model",
        signature=signature,
        input_example=example,
        pyfunc_predict_fn="predict_proba",
        skops_trusted_types=SKOPS_TRUSTED_TYPES,
    )
    logger.info("%s: cv_auc=%.4f val_auc=%.4f", result.name, result.cv_mean, val_metrics["roc_auc"])
    return Candidate(result.name, run_id, pipeline, info.model_uri, val_metrics, threshold)


def run_training(cfg: TrainConfig, settings: Settings) -> TrainingSummary:
    download_dataset(settings.csv_path)
    df = load_dataset(settings.csv_path)
    report = validate_training_data(df)
    dataset_hash = dataset_fingerprint(df)
    tags = lineage_tags(dataset_hash)

    mlflow.set_tracking_uri(settings.tracking_uri)
    client = MlflowClient()
    if settings.artifact_root and client.get_experiment_by_name(EXPERIMENT_NAME) is None:
        client.create_experiment(EXPERIMENT_NAME, artifact_location=settings.artifact_root)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="training-pipeline") as parent:
        mlflow.set_tags({**tags, "pipeline_stage": "training"})
        mlflow.log_params({**asdict(cfg), "model_seed": cfg.model_seed, "n_rows": len(df)})
        mlflow.log_dict(report.to_dict(), "data/validation_report.json")
        mlflow.log_input(
            from_pandas(df, source=DATASET_URL, name=DATASET_NAME, targets=TARGET),
            context="training",
        )
        assert_valid(report)  # raising inside the run marks it FAILED in MLflow

        splits = split_dataset(df, seed=cfg.seed)
        mlflow.log_params({f"n_{k}": v for k, v in splits.sizes().items()})

        candidates: list[Candidate] = []
        with mlflow.start_run(run_name="logreg-baseline", nested=True) as run:
            result = tune_logreg(splits.X_train, splits.y_train, cfg.cv_folds, cfg.model_seed)
            candidates.append(_fit_and_log_candidate(result, splits, run.info.run_id, tags))

        with mlflow.start_run(run_name="lightgbm-optuna", nested=True) as run:
            mlflow.log_param("n_trials", cfg.n_trials)
            result = tune_lightgbm(
                splits.X_train,
                splits.y_train,
                cfg.cv_folds,
                cfg.model_seed,
                cfg.n_trials,
                callbacks=[_optuna_mlflow_callback],
            )
            candidates.append(_fit_and_log_candidate(result, splits, run.info.run_id, tags))

        # Model selection on the validation split; the test split is only used below.
        champion = max(candidates, key=lambda c: c.val_metrics[PRIMARY_METRIC])
        test_proba = champion.pipeline.predict_proba(splits.X_test)[:, 1]
        test_metrics = classification_metrics(splits.y_test, test_proba, champion.threshold)

        with mlflow.start_run(run_id=champion.run_id, nested=True):
            mlflow.log_metrics(_prefixed("test", test_metrics))
            mlflow.set_tag("selected", "true")
            sample = splits.X_test.sample(
                min(cfg.shap_sample, len(splits.X_test)), random_state=cfg.seed
            )
            with tempfile.TemporaryDirectory() as tmp:
                write_shap_artifacts(champion.pipeline, sample, Path(tmp))
                mlflow.log_artifacts(tmp, artifact_path="explainability")

        mlflow.set_tag("champion_model_type", champion.name)
        mlflow.log_metrics(_prefixed("test", test_metrics))
        mlflow.log_metric("decision_threshold", champion.threshold)

        version: str | None = None
        decision: PromotionDecision | None = None
        if cfg.register:
            version_tags = {
                **tags,
                "model_type": champion.name,
                "decision_threshold": f"{champion.threshold:.6f}",
                "training_run_id": parent.info.run_id,
                **{k: f"{v:.6f}" for k, v in _prefixed("val", champion.val_metrics).items()},
                **{k: f"{v:.6f}" for k, v in _prefixed("test", test_metrics).items()},
            }
            mv = mlflow.register_model(champion.model_uri, REGISTERED_MODEL_NAME, tags=version_tags)
            version = str(mv.version)
            decision = promote_if_better(
                client,
                REGISTERED_MODEL_NAME,
                version,
                splits.X_test,
                splits.y_test,
                cfg.min_delta,
            )
            mlflow.set_tags(
                {"registered_version": version, "promoted": str(decision.promoted).lower()}
            )
            mlflow.log_dict(decision.to_dict(), "registry/promotion_decision.json")

        summary = TrainingSummary(
            parent_run_id=parent.info.run_id,
            champion=champion.name,
            threshold=champion.threshold,
            val_metrics={c.name: c.val_metrics for c in candidates},
            test_metrics=test_metrics,
            model_version=version,
            decision=decision,
        )
        mlflow.log_dict(summary.to_dict(), "summary.json")

    logger.info("Training summary:\n%s", json.dumps(summary.to_dict(), indent=2))
    return summary
