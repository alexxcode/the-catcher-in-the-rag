#!/usr/bin/env bash
set -e

# The Catcher in the RAG — Production Container Entrypoint
# Supports running as non-root user (appuser, UID 10001)

# Ensure baseline synthetic packages exist for reproducible execution if directory is missing
if [ ! -d "data/synthetic/PKG-0001" ]; then
    echo "[Entrypoint] Initializing baseline synthetic procurement packages for container..."
    python -m datagen.generate --count 30 --seed 42 || true
fi

MODE="${SERVICE_MODE:-${1:-api}}"

if [ "$MODE" = "ui" ]; then
    PORT="${PORT:-8501}"
    echo "[Entrypoint] Starting Streamlit UI on port ${PORT}..."
    exec streamlit run ui/app.py \
        --server.port "${PORT}" \
        --server.address 0.0.0.0 \
        --server.headless true \
        --browser.gatherUsageStats false
elif [ "$MODE" = "api" ]; then
    PORT="${PORT:-8000}"
    echo "[Entrypoint] Starting FastAPI on port ${PORT}..."
    exec uvicorn catcher_in_the_rag.api.main:app \
        --host 0.0.0.0 \
        --port "${PORT}"
else
    echo "[Entrypoint] Executing custom command: $@"
    exec "$@"
fi
