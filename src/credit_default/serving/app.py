"""FastAPI inference service.

Run locally:  uvicorn credit_default.serving.app:app --port 8000
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from credit_default.config import RAW_FEATURES
from credit_default.serving.metrics import ApiMetrics
from credit_default.serving.model_loader import ModelBundle, load_from_env
from credit_default.serving.schemas import (
    BatchRequest,
    BatchResponse,
    CreditApplication,
    HealthResponse,
    ModelInfo,
    Prediction,
    PredictResponse,
)

logger = logging.getLogger("credit_default.api")

Loader = Callable[[], ModelBundle]


@dataclass
class ServiceState:
    bundle: ModelBundle | None = None
    last_error: str | None = None


def _to_frame(records: list[CreditApplication]) -> pd.DataFrame:
    return pd.DataFrame([r.model_dump() for r in records], columns=list(RAW_FEATURES))


def create_app(loader: Loader = load_from_env, retry_seconds: float | None = None) -> FastAPI:
    """App factory. ``loader`` is injectable so tests can serve a tiny in-memory model."""
    retry = (
        float(os.environ.get("MODEL_LOAD_RETRY_SECONDS", "15"))
        if retry_seconds is None
        else retry_seconds
    )
    state = ServiceState()
    metrics = ApiMetrics()

    def try_load() -> bool:
        try:
            bundle = loader()
        except Exception as exc:
            state.last_error = f"{type(exc).__name__}: {exc}"
            logger.warning("Model load failed: %s", state.last_error)
            return False
        state.bundle, state.last_error = bundle, None
        metrics.model_loaded.set(1)
        metrics.model_version.labels(
            name=str(bundle.metadata.get("name", "unknown")),
            version=bundle.version,
            source=bundle.source,
        ).set(1)
        logger.info("Serving model version %s (%s)", bundle.version, bundle.source)
        return True

    async def retry_loop() -> None:
        while not await asyncio.to_thread(try_load):
            await asyncio.sleep(retry)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        metrics.model_loaded.set(0)
        task = None
        # First attempt is synchronous so a healthy pod is ready as soon as it starts;
        # if the registry isn't reachable yet (compose start-up), keep retrying.
        if not await asyncio.to_thread(try_load) and retry > 0:
            task = asyncio.create_task(retry_loop())
        yield
        if task:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    app = FastAPI(
        title="Credit default risk API",
        version="0.1.0",
        description="Scores the probability that a credit-card client defaults next month.",
        lifespan=lifespan,
    )
    app.state.service = state
    app.state.metrics = metrics

    @app.middleware("http")
    async def prometheus_middleware(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            route = request.scope.get("route")
            # Route *templates* keep label cardinality bounded.
            template = getattr(route, "path", "unmatched")
            if template != "/metrics":
                metrics.requests.labels(request.method, template, str(status)).inc()
                metrics.latency.labels(request.method, template).observe(
                    time.perf_counter() - start
                )

    def require_model() -> ModelBundle:
        if state.bundle is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        return state.bundle

    def score(records: list[CreditApplication]) -> tuple[ModelBundle, list[Prediction]]:
        bundle = require_model()
        proba = bundle.model.predict_proba(_to_frame(records))[:, 1].tolist()
        metrics.observe_predictions(proba, bundle.threshold)
        preds = [
            Prediction(
                default_probability=round(p, 6),
                predicted_default=p >= bundle.threshold,
                threshold=bundle.threshold,
            )
            for p in proba
        ]
        return bundle, preds

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    def health() -> JSONResponse:
        """Readiness: 200 only when a model is loaded and can serve."""
        bundle = state.bundle
        body = HealthResponse(
            status="ok" if bundle else "unavailable",
            model_loaded=bundle is not None,
            model_version=bundle.version if bundle else None,
            detail=None if bundle else state.last_error,
        )
        return JSONResponse(body.model_dump(), status_code=200 if bundle else 503)

    @app.get("/health/live", tags=["ops"])
    def live() -> dict[str, str]:
        """Liveness: the process is up and the event loop responds."""
        return {"status": "alive"}

    @app.get("/model-info", response_model=ModelInfo, tags=["model"])
    def model_info() -> ModelInfo:
        bundle = require_model()
        tags = bundle.tags
        return ModelInfo(
            name=str(bundle.metadata.get("name", "unknown")),
            version=bundle.version,
            alias=bundle.metadata.get("alias"),
            source="local" if bundle.source == "local" else "registry",
            model_type=tags.get("model_type"),
            decision_threshold=bundle.threshold,
            run_id=bundle.metadata.get("run_id"),
            dataset_sha256=tags.get("dataset_sha256"),
            git_commit=tags.get("git_commit"),
            metrics=bundle.metrics,
            loaded_at=bundle.loaded_at,
        )

    @app.post("/predict", response_model=PredictResponse, tags=["model"])
    def predict(application: CreditApplication) -> PredictResponse:
        bundle, preds = score([application])
        return PredictResponse(**preds[0].model_dump(), model_version=bundle.version)

    @app.post("/predict-batch", response_model=BatchResponse, tags=["model"])
    def predict_batch(batch: BatchRequest) -> BatchResponse:
        bundle, preds = score(batch.records)
        return BatchResponse(predictions=preds, model_version=bundle.version, n_records=len(preds))

    @app.get("/metrics", include_in_schema=False)
    def prometheus_metrics() -> Response:
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
