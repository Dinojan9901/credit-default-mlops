from __future__ import annotations

from pathlib import Path

import mlflow.sklearn
import pytest
from sklearn.linear_model import LogisticRegression

from credit_default.config import RAW_FEATURES, TARGET, Settings
from credit_default.features import build_pipeline
from credit_default.monitoring.drift import run_drift_check
from credit_default.monitoring.simulate import apply_downturn_shift
from credit_default.train import SKOPS_TRUSTED_TYPES
from credit_default.validation import DataValidationError, validate_training_data
from helpers import make_synthetic


@pytest.fixture(scope="module")
def setup(tmp_path_factory: pytest.TempPathFactory) -> tuple[Settings, Path, Path]:
    root = tmp_path_factory.mktemp("drift")
    settings = Settings(data_dir=root / "data", reports_dir=root / "reports")
    mon = settings.data_dir / "monitoring"
    mon.mkdir(parents=True)
    make_synthetic(3000, seed=10).to_csv(mon / "reference.csv", index=False)

    train = make_synthetic(1000, seed=11)
    model = build_pipeline(LogisticRegression(max_iter=1000)).fit(
        train[list(RAW_FEATURES)], train[TARGET]
    )
    model_dir = root / "export"
    mlflow.sklearn.save_model(
        model, str(model_dir / "model"), skops_trusted_types=SKOPS_TRUSTED_TYPES
    )
    return settings, model_dir, mon


def test_same_distribution_does_not_alert(setup: tuple[Settings, Path, Path]) -> None:
    settings, model_dir, mon = setup
    path = mon / "nodrift.csv"
    make_synthetic(1500, seed=12).to_csv(path, index=False)
    result = run_drift_check(path, settings, threshold=0.3, model_dir=model_dir)
    assert not result.alert, result.drifted_features
    assert Path(result.report_html).exists()


def test_downturn_shift_alerts(setup: tuple[Settings, Path, Path]) -> None:
    settings, model_dir, mon = setup
    path = mon / "drift.csv"
    apply_downturn_shift(make_synthetic(1500, seed=13)).to_csv(path, index=False)
    result = run_drift_check(path, settings, threshold=0.3, model_dir=model_dir)
    assert result.alert
    assert "limit_bal" in result.drifted_features
    assert result.mean_prediction_current > result.mean_prediction_reference


def test_shifted_batch_is_still_schema_valid() -> None:
    shifted = apply_downturn_shift(make_synthetic(1500, seed=14))
    assert validate_training_data(shifted).ok


def test_malformed_batch_is_rejected(setup: tuple[Settings, Path, Path]) -> None:
    settings, model_dir, mon = setup
    path = mon / "broken.csv"
    make_synthetic(1500, seed=15).drop(columns=["age"]).to_csv(path, index=False)
    with pytest.raises(DataValidationError):
        run_drift_check(path, settings, model_dir=model_dir)
