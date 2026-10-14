from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from credit_default.config import RAW_FEATURES, TARGET
from credit_default.features import build_pipeline
from credit_default.serving.app import create_app
from credit_default.serving.model_loader import ModelBundle
from credit_default.serving.schemas import CreditApplication
from credit_default.validation import ALLOWED_VALUES, RANGES
from helpers import make_synthetic

EXAMPLE: dict[str, Any] = CreditApplication.model_config["json_schema_extra"]["example"]  # type: ignore[index]


@pytest.fixture(scope="module")
def bundle() -> ModelBundle:
    df = make_synthetic(800, seed=1)
    model = build_pipeline(LogisticRegression(max_iter=1000)).fit(
        df[list(RAW_FEATURES)], df[TARGET]
    )
    return ModelBundle(
        model=model,
        metadata={
            "name": "credit-default-classifier",
            "version": "7",
            "alias": "production",
            "run_id": "abc123",
            "tags": {
                "decision_threshold": "0.3",
                "model_type": "logreg",
                "dataset_sha256": "deadbeef",
                "git_commit": "cafe",
                "test_roc_auc": "0.77",
            },
        },
        source="local",
    )


@pytest.fixture
def client(bundle: ModelBundle) -> Iterator[TestClient]:
    with TestClient(create_app(loader=lambda: bundle, retry_seconds=0)) as c:
        yield c


def test_health_ready(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True, "model_version": "7", "detail": None}
    assert client.get("/health/live").status_code == 200


def test_health_unavailable_when_model_fails_to_load() -> None:
    def broken() -> ModelBundle:
        raise ConnectionError("registry unreachable")

    with TestClient(create_app(loader=broken, retry_seconds=0)) as c:
        r = c.get("/health")
        assert r.status_code == 503
        assert "registry unreachable" in r.json()["detail"]
        assert c.get("/health/live").status_code == 200
        assert c.post("/predict", json=EXAMPLE).status_code == 503


def test_predict(client: TestClient) -> None:
    r = client.post("/predict", json=EXAMPLE)
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0.0 <= body["default_probability"] <= 1.0
    assert body["threshold"] == 0.3
    assert body["predicted_default"] == (body["default_probability"] >= 0.3)
    assert body["model_version"] == "7"


def test_predict_batch_matches_single(client: TestClient) -> None:
    late = {**EXAMPLE, "pay_1": 3, "pay_2": 2}
    r = client.post("/predict-batch", json={"records": [EXAMPLE, late]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_records"] == 2
    single = client.post("/predict", json=EXAMPLE).json()
    assert body["predictions"][0]["default_probability"] == single["default_probability"]


@pytest.mark.parametrize(
    "patch",
    [
        {"sex": 3},
        {"age": 12},
        {"pay_1": 9},
        {"limit_bal": 0},
        {"pay_amt1": -1},
        {"age": "thirty"},
        {"unexpected": 1},
    ],
)
def test_predict_rejects_invalid_input(client: TestClient, patch: dict[str, Any]) -> None:
    assert client.post("/predict", json={**EXAMPLE, **patch}).status_code == 422


def test_predict_rejects_missing_field(client: TestClient) -> None:
    payload = {k: v for k, v in EXAMPLE.items() if k != "age"}
    assert client.post("/predict", json=payload).status_code == 422


def test_batch_limits(client: TestClient) -> None:
    assert client.post("/predict-batch", json={"records": []}).status_code == 422
    too_many = {"records": [EXAMPLE] * 1001}
    assert client.post("/predict-batch", json=too_many).status_code == 422


def test_model_info(client: TestClient) -> None:
    r = client.get("/model-info")
    assert r.status_code == 200
    info = r.json()
    assert info["version"] == "7"
    assert info["dataset_sha256"] == "deadbeef"
    assert info["metrics"] == {"test_roc_auc": 0.77}


def test_metrics_endpoint_counts_requests_and_predictions(client: TestClient) -> None:
    client.post("/predict", json=EXAMPLE)
    client.post("/predict", json={**EXAMPLE, "sex": 3})
    text = client.get("/metrics").text
    assert 'http_requests_total{method="POST",route="/predict",status="200"} 1.0' in text
    assert 'http_requests_total{method="POST",route="/predict",status="422"} 1.0' in text
    assert "model_prediction_probability_count 1.0" in text
    assert "model_loaded 1.0" in text
    assert 'model_version_info{name="credit-default-classifier",source="local",version="7"}' in text


def test_schema_bounds_match_training_validation() -> None:
    """A request the API accepts must be a row the training validator accepts."""
    props = CreditApplication.model_json_schema()["properties"]
    for col in RAW_FEATURES:
        assert col in props
    for col, allowed in ALLOWED_VALUES.items():
        spec = props[col]
        if "enum" in spec:
            assert set(spec["enum"]) == allowed
        else:
            assert (spec["minimum"], spec["maximum"]) == (min(allowed), max(allowed))
    for col, (lo, hi) in RANGES.items():
        assert (props[col]["minimum"], props[col]["maximum"]) == (lo, hi)
