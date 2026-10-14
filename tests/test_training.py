"""End-to-end training + registry tests against a throw-away local MLflow store."""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from credit_default.config import (
    CHALLENGER_ALIAS,
    PRODUCTION_ALIAS,
    REGISTERED_MODEL_NAME,
    Settings,
)
from credit_default.registry import METADATA_FILE, export_model
from credit_default.train import TrainConfig, run_training

pytestmark = pytest.mark.slow


@pytest.fixture
def settings(tmp_path: Path, synthetic_df: pd.DataFrame) -> Settings:
    s = Settings(
        data_dir=tmp_path / "data",
        reports_dir=tmp_path / "reports",
        tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
        artifact_root=(tmp_path / "artifacts").as_uri(),
        seed=0,
    )
    s.raw_dir.mkdir(parents=True)
    synthetic_df.to_csv(s.csv_path, index=False)  # present -> download is skipped
    return s


def test_train_registers_promotes_and_gates(settings: Settings, tmp_path: Path) -> None:
    cfg = TrainConfig(n_trials=3, cv_folds=2, seed=0, shap_sample=100)
    first = run_training(cfg, settings)

    assert first.model_version == "1"
    assert first.decision is not None and first.decision.promoted
    assert 0.5 < first.test_metrics["roc_auc"] <= 1.0

    client = MlflowClient()
    prod = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, PRODUCTION_ALIAS)
    assert str(prod.version) == "1"
    for tag in ("dataset_sha256", "git_commit", "decision_threshold", "test_roc_auc"):
        assert tag in prod.tags

    runs = mlflow.search_runs(filter_string="tags.selected = 'true'", output_format="list")
    champion_run = runs[0]
    artifacts = {a.path for a in client.list_artifacts(champion_run.info.run_id, "explainability")}
    assert "explainability/shap_summary_beeswarm.png" in artifacts

    # Same data + same seed -> identical challenger, which must NOT replace the champion.
    second = run_training(cfg, settings)
    assert second.model_version == "2"
    assert second.decision is not None and not second.decision.promoted

    def alias_version(alias: str) -> str:
        return str(client.get_model_version_by_alias(REGISTERED_MODEL_NAME, alias).version)

    assert alias_version(PRODUCTION_ALIAS) == "1"
    assert alias_version(CHALLENGER_ALIAS) == "2"

    out = export_model(client, REGISTERED_MODEL_NAME, PRODUCTION_ALIAS, tmp_path / "export")
    meta = json.loads((out / METADATA_FILE).read_text())
    assert meta["version"] == "1" and meta["alias"] == PRODUCTION_ALIAS
    assert (out / "model" / "MLmodel").exists()
