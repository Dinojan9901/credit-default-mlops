"""MLflow Model Registry helpers: champion/challenger promotion gate and model export."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from sklearn.metrics import roc_auc_score

from credit_default.config import CHALLENGER_ALIAS, PRODUCTION_ALIAS

logger = logging.getLogger(__name__)

METADATA_FILE = "model_metadata.json"


@dataclass
class PromotionDecision:
    promoted: bool
    reason: str
    challenger_version: str
    challenger_roc_auc: float
    champion_version: str | None = None
    champion_roc_auc: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def get_version_by_alias(client: MlflowClient, name: str, alias: str) -> ModelVersion | None:
    try:
        return client.get_model_version_by_alias(name, alias)
    except MlflowException:
        return None


def score_model(model_uri: str, X: pd.DataFrame, y: pd.Series) -> float:
    model = mlflow.sklearn.load_model(model_uri)
    return float(roc_auc_score(y, model.predict_proba(X)[:, 1]))


def promote_if_better(
    client: MlflowClient,
    name: str,
    challenger_version: str,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    min_delta: float = 0.001,
) -> PromotionDecision:
    """Champion/challenger gate on a shared hold-out set.

    The new version gets the ``challenger`` alias. It takes over ``production`` only if
    there is no production model yet, or if it beats the current one by more than
    ``min_delta`` ROC-AUC on the *same* evaluation data (so the comparison is fair even
    if the champion's logged metrics came from an older dataset).
    """
    client.set_registered_model_alias(name, CHALLENGER_ALIAS, challenger_version)
    challenger_score = score_model(f"models:/{name}/{challenger_version}", X_eval, y_eval)
    champion = get_version_by_alias(client, name, PRODUCTION_ALIAS)

    if champion is None:
        decision = PromotionDecision(
            True, "no production model yet", challenger_version, challenger_score
        )
    else:
        champion_score = score_model(f"models:/{name}/{champion.version}", X_eval, y_eval)
        better = challenger_score > champion_score + min_delta
        reason = (
            f"challenger {challenger_score:.4f} vs champion {champion_score:.4f} "
            f"(min_delta {min_delta}) -> {'promote' if better else 'keep champion'}"
        )
        decision = PromotionDecision(
            better,
            reason,
            challenger_version,
            challenger_score,
            str(champion.version),
            champion_score,
        )

    if decision.promoted:
        client.set_registered_model_alias(name, PRODUCTION_ALIAS, challenger_version)
    for key, value in {
        "promotion_decision": "promoted" if decision.promoted else "rejected",
        "promotion_reason": decision.reason,
        "promotion_eval_roc_auc": f"{challenger_score:.6f}",
    }.items():
        client.set_model_version_tag(name, challenger_version, key, value)
    logger.info("Promotion decision for v%s: %s", challenger_version, decision.reason)
    return decision


def version_metadata(name: str, mv: ModelVersion, alias: str | None = None) -> dict[str, Any]:
    return {
        "name": name,
        "version": str(mv.version),
        "alias": alias,
        "run_id": mv.run_id,
        "source": mv.source,
        "created_at_ms": mv.creation_timestamp,
        "tags": dict(mv.tags),
    }


def export_model(client: MlflowClient, name: str, alias: str, out_dir: Path) -> Path:
    """Copy a registered model + its metadata to a plain directory.

    Used to bake an immutable model into the serving image for Kubernetes, so pods
    don't need network access to the tracking server at start-up.
    """
    mv = get_version_by_alias(client, name, alias)
    if mv is None:
        raise LookupError(f"No version of {name!r} has alias {alias!r}")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = out_dir / "model"
    mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/{name}/{mv.version}", dst_path=str(model_dir)
    )
    meta_path = out_dir / METADATA_FILE
    meta_path.write_text(json.dumps(version_metadata(name, mv, alias), indent=2))
    logger.info("Exported %s v%s (@%s) to %s", name, mv.version, alias, out_dir)
    return out_dir
