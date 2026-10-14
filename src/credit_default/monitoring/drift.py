"""Data and prediction drift check with Evidently.

Compares a new batch against the reference sample of the training data:
* per-feature drift (Evidently picks K-S / chi-squared / Wasserstein etc. per column type
  and sample size)
* prediction drift - the model's scores on both datasets, compared the same way

The alert fires when the share of drifted features exceeds ``threshold`` or when the
prediction distribution itself drifted. Labels are deliberately not used: in production
they arrive weeks later (a default is only known after the payment is missed).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from credit_default.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, RAW_FEATURES, Settings
from credit_default.monitoring.simulate import monitoring_dir, write_batches
from credit_default.validation import ValidationReport, assert_valid, validate_features

logger = logging.getLogger(__name__)

PREDICTION = "prediction"


@dataclass
class DriftResult:
    reference_rows: int
    current_rows: int
    n_features: int
    n_drifted_features: int
    share_drifted_features: float
    threshold: float
    prediction_drift: bool
    prediction_drift_score: float
    mean_prediction_reference: float
    mean_prediction_current: float
    alert: bool
    drifted_features: list[str] = field(default_factory=list)
    column_scores: dict[str, dict[str, Any]] = field(default_factory=dict)
    report_html: str = ""
    report_json: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_model(model_uri: str, model_dir: Path | None, tracking_uri: str) -> Any:
    import mlflow.sklearn

    if model_dir is not None:
        return mlflow.sklearn.load_model(str(model_dir / "model"))
    mlflow.set_tracking_uri(tracking_uri)
    return mlflow.sklearn.load_model(model_uri)


def _column_drift(metrics: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per-column drift decisions from the snapshot's ValueDrift metrics."""
    out: dict[str, dict[str, Any]] = {}
    for m in metrics:
        cfg = m.get("config", {})
        if not str(cfg.get("type", "")).endswith("ValueDrift"):
            continue
        method, threshold, score = str(cfg["method"]), float(cfg["threshold"]), float(m["value"])
        # p-value tests: drift when p < threshold; distance metrics: drift when d >= threshold.
        drifted = score < threshold if "p_value" in method else score >= threshold
        out[str(cfg["column"])] = {
            "method": method,
            "score": score,
            "threshold": threshold,
            "drifted": drifted,
        }
    return out


def run_drift_check(
    current_path: Path,
    settings: Settings,
    threshold: float = 0.3,
    model_uri: str = "",
    model_dir: Path | None = None,
    reference_path: Path | None = None,
    out_dir: Path | None = None,
) -> DriftResult:
    reference_path = reference_path or monitoring_dir(settings) / "reference.csv"
    if not reference_path.exists():
        write_batches(settings)
    reference = pd.read_csv(reference_path)
    current = pd.read_csv(current_path)

    report = ValidationReport(n_rows=len(current), n_columns=current.shape[1])
    validate_features(current, report)
    assert_valid(report)  # a malformed batch is a data-quality incident, not drift

    model = _load_model(model_uri, model_dir, settings.tracking_uri)
    features = list(RAW_FEATURES)
    ref = reference[features].assign(**{PREDICTION: model.predict_proba(reference[features])[:, 1]})
    cur = current[features].assign(**{PREDICTION: model.predict_proba(current[features])[:, 1]})

    definition = DataDefinition(
        numerical_columns=[*NUMERIC_FEATURES, PREDICTION],
        categorical_columns=list(CATEGORICAL_FEATURES),
    )
    snapshot = Report([DataDriftPreset(drift_share=threshold)]).run(
        Dataset.from_pandas(cur, data_definition=definition),
        Dataset.from_pandas(ref, data_definition=definition),
    )

    columns = _column_drift(snapshot.dict()["metrics"])
    feature_cols = {k: v for k, v in columns.items() if k != PREDICTION}
    drifted = sorted(k for k, v in feature_cols.items() if v["drifted"])
    share = len(drifted) / max(len(feature_cols), 1)
    pred = columns.get(PREDICTION, {"drifted": False, "score": float("nan")})

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = out_dir or settings.reports_dir / "drift" / f"{stamp}_{current_path.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path, json_path = out_dir / "drift_report.html", out_dir / "drift_summary.json"
    snapshot.save_html(str(html_path))

    result = DriftResult(
        reference_rows=len(ref),
        current_rows=len(cur),
        n_features=len(feature_cols),
        n_drifted_features=len(drifted),
        share_drifted_features=round(share, 4),
        threshold=threshold,
        prediction_drift=bool(pred["drifted"]),
        prediction_drift_score=float(pred["score"]),
        mean_prediction_reference=round(float(ref[PREDICTION].mean()), 4),
        mean_prediction_current=round(float(cur[PREDICTION].mean()), 4),
        alert=share > threshold or bool(pred["drifted"]),
        drifted_features=drifted,
        column_scores=columns,
        report_html=str(html_path),
        report_json=str(json_path),
    )
    json_path.write_text(json.dumps(result.to_dict(), indent=2))
    level = logging.WARNING if result.alert else logging.INFO
    logger.log(
        level,
        "Drift: %d/%d features drifted (share %.2f, threshold %.2f), "
        "prediction drift=%s -> alert=%s",
        result.n_drifted_features,
        result.n_features,
        share,
        threshold,
        result.prediction_drift,
        result.alert,
    )
    return result
