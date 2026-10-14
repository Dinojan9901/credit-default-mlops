"""Locate and load the model the API serves.

Two modes, chosen by environment variables:

* ``MODEL_DIR`` set   -> load an exported model directory baked into the image
                          (Kubernetes: immutable, no runtime dependency on MLflow)
* otherwise           -> resolve ``MODEL_NAME@MODEL_ALIAS`` in the MLflow registry at
                          ``MLFLOW_TRACKING_URI`` (docker-compose: always the current
                          production version at start-up)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from credit_default.config import PRODUCTION_ALIAS, REGISTERED_MODEL_NAME

METADATA_FILE = "model_metadata.json"  # same name as registry.METADATA_FILE (no mlflow import)


class ProbabilisticModel(Protocol):
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...


@dataclass
class ModelBundle:
    model: ProbabilisticModel
    metadata: dict[str, Any]
    source: str
    loaded_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def tags(self) -> dict[str, str]:
        return dict(self.metadata.get("tags") or {})

    @property
    def version(self) -> str:
        return str(self.metadata.get("version", "unknown"))

    @property
    def threshold(self) -> float:
        return float(self.tags.get("decision_threshold", 0.5))

    @property
    def metrics(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for k, v in self.tags.items():
            if k.startswith(("test_", "val_")):
                try:
                    out[k] = float(v)
                except ValueError:
                    continue
        return out


def load_from_dir(model_dir: Path) -> ModelBundle:
    import mlflow.sklearn

    meta_path = model_dir / METADATA_FILE
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    model = mlflow.sklearn.load_model(str(model_dir / "model"))
    return ModelBundle(model=model, metadata=metadata, source="local")


def load_from_registry(name: str, alias: str, tracking_uri: str | None = None) -> ModelBundle:
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    mv = client.get_model_version_by_alias(name, alias)
    # Load the resolved version (not the alias) so model + metadata can't diverge if the
    # alias moves while we're loading.
    model = mlflow.sklearn.load_model(f"models:/{name}/{mv.version}")
    metadata = {
        "name": name,
        "version": str(mv.version),
        "alias": alias,
        "run_id": mv.run_id,
        "tags": dict(mv.tags),
    }
    return ModelBundle(model=model, metadata=metadata, source="registry")


def load_from_env() -> ModelBundle:
    model_dir = os.environ.get("MODEL_DIR")
    if model_dir:
        return load_from_dir(Path(model_dir))
    return load_from_registry(
        os.environ.get("MODEL_NAME", REGISTERED_MODEL_NAME),
        os.environ.get("MODEL_ALIAS", PRODUCTION_ALIAS),
        os.environ.get("MLFLOW_TRACKING_URI"),
    )
