# syntax=docker/dockerfile:1.7
#
# Targets
#   api             inference service; loads the model from the MLflow registry at start-up
#   api-with-model  same, plus an exported model baked in (MODEL_DIR) - used for Kubernetes
#   trainer         training pipeline (Optuna, SHAP, MLflow client), for docker-compose / CT
#
# Build stages install dependencies from uv.lock into /opt/venv; runtime stages copy only
# that venv onto a clean slim image (no compilers, no uv, no build cache, non-root user).

ARG PYTHON_IMAGE=python:3.12-slim-bookworm

FROM ghcr.io/astral-sh/uv:0.13.1 AS uv

# ---------------------------------------------------------------- build stages
FROM ${PYTHON_IMAGE} AS builder-base
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /src
COPY pyproject.toml uv.lock README.md ./

FROM builder-base AS builder-serve
# Dependencies first (cached layer), then the project itself.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra serve --no-install-project
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra serve --no-editable

FROM builder-base AS builder-train
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra train --extra monitor --no-install-project
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra train --extra monitor --no-editable

# ---------------------------------------------------------------- runtime base
FROM ${PYTHON_IMAGE} AS runtime-base
# libgomp1: OpenMP runtime required by LightGBM.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 10001 --create-home --shell /usr/sbin/nologin app
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

# ---------------------------------------------------------------- trainer
FROM runtime-base AS trainer
COPY --from=builder-train /opt/venv /opt/venv
RUN mkdir -p /app/data /app/reports && chown -R app:app /app
USER 10001
ENTRYPOINT ["credit-default"]
CMD ["train"]

# ---------------------------------------------------------------- api
FROM runtime-base AS api
COPY --from=builder-serve /opt/venv /opt/venv
ARG GIT_COMMIT=unknown
LABEL org.opencontainers.image.revision=$GIT_COMMIT \
      org.opencontainers.image.title="credit-default-api"
USER 10001
EXPOSE 8000
# Exec-based check for plain Docker / compose (Kubernetes uses its own HTTP probes).
# Generous timeout: interpreter start-up is slow on resource-constrained hosts.
HEALTHCHECK --interval=30s --timeout=15s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=2)"]
CMD ["uvicorn", "credit_default.serving.app:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]

# ---------------------------------------------------------------- api-with-model
FROM api AS api-with-model
# Produced by: credit-default export-model --alias production --out build/model-export
COPY --chown=10001:10001 build/model-export /opt/model
ENV MODEL_DIR=/opt/model
