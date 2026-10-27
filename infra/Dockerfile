FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS base

WORKDIR /app

# Enable bytecode compilation and optimal settings
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV PYTHONUNBUFFERED=1

# Install curl for container health checks
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Create dedicated non-root user (appuser, UID 10001) for security hardening
RUN groupadd -g 10001 appuser && \
    useradd -u 10001 -g appuser -m -s /bin/bash appuser

# Install project dependencies
COPY pyproject.toml uv.lock* ./
COPY README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev || uv sync --no-install-project --no-dev

# Copy application source code (without unversioned local data)
COPY src/ ./src/
COPY config/ ./config/
COPY datagen/ ./datagen/
COPY evals/ ./evals/
COPY ui/ ./ui/
COPY data/demo/ ./data/demo/
COPY .streamlit/ ./.streamlit/
COPY infra/entrypoint.sh ./infra/entrypoint.sh

# Install the application package
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev || uv sync --no-dev

# Ensure permissions and ownership for appuser
RUN chmod +x ./infra/entrypoint.sh && \
    mkdir -p data/synthetic data/real && \
    chown -R appuser:appuser /app

# Place virtual environment on PATH
ENV PATH="/app/.venv/bin:$PATH"

# Switch to non-root user
USER appuser

EXPOSE 8000 8501

# Healthcheck probing /healthz
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/healthz || exit 1

ENTRYPOINT ["/app/infra/entrypoint.sh"]
CMD ["api"]
