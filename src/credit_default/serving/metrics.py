"""Prometheus metrics exposed on /metrics."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram


class ApiMetrics:
    """Own registry per app instance, so tests can create apps without global clashes."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "http_requests_total",
            "HTTP requests by route template, method and status code",
            ["method", "route", "status"],
            registry=self.registry,
        )
        self.latency = Histogram(
            "http_request_duration_seconds",
            "Request latency by route template",
            ["method", "route"],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
            registry=self.registry,
        )
        self.predictions = Counter(
            "model_predictions_total",
            "Scored records by predicted class",
            ["predicted_class"],
            registry=self.registry,
        )
        self.probability = Histogram(
            "model_prediction_probability",
            "Distribution of predicted default probabilities",
            buckets=tuple(round(0.05 * i, 2) for i in range(1, 21)),
            registry=self.registry,
        )
        self.model_loaded = Gauge(
            "model_loaded", "1 if a model is loaded and serving", registry=self.registry
        )
        self.model_version = Gauge(
            "model_version_info",
            "Constant 1, labelled with the served model version",
            ["name", "version", "source"],
            registry=self.registry,
        )

    def observe_predictions(self, probabilities: list[float], threshold: float) -> None:
        for p in probabilities:
            self.probability.observe(p)
            self.predictions.labels(predicted_class=str(int(p >= threshold))).inc()
