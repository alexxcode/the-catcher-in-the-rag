FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS base

WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV PYTHONUNBUFFERED=1

# Install project dependencies
COPY pyproject.toml uv.lock* ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev || uv sync --no-install-project --no-dev

# Copy application source code
COPY src/ ./src/
COPY config/ ./config/
COPY evals/ ./evals/
COPY ui/ ./ui/
COPY data/ ./data/

# Install the application package
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev || uv sync --no-dev

# Place virtual environment on PATH
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000 8501

CMD ["uvicorn", "catcher_in_the_rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
