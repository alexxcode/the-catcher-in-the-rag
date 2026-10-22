from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from catcher_in_the_rag.providers.base import EmbeddingProvider, LLMProvider
from catcher_in_the_rag.providers.stub import StubEmbeddingProvider, StubLLMProvider


def load_model_config() -> dict[str, Any]:
    """Load models configuration from config/models.yaml."""
    config_path = Path("config/models.yaml")
    if not config_path.exists():
        # Fallback to search from repository root
        for parent in Path.cwd().parents:
            candidate = parent / "config" / "models.yaml"
            if candidate.exists():
                config_path = candidate
                break
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def get_llm_provider(provider_type: str | None = None) -> LLMProvider:
    """Get configured LLM provider (defaults to stub for offline execution and tests)."""
    p_type = provider_type or os.getenv("LLM_PROVIDER", "stub")

    if p_type.lower() == "stub":
        return StubLLMProvider()

    if p_type.lower() in ("vertex_ai", "gemini", "gcp", "real"):
        # Real GCP generative AI provider (opt-in for eval/production)
        from catcher_in_the_rag.providers.gemini import GeminiProvider

        cfg = load_model_config()
        return GeminiProvider(config=cfg)

    msg = f"Unknown LLM provider type: {p_type}. Expected 'stub', 'vertex_ai', 'gemini', or 'real'."
    raise ValueError(msg)


def get_embedding_provider(provider_type: str | None = None) -> EmbeddingProvider:
    """Get configured embedding provider (defaults to stub)."""
    p_type = provider_type or os.getenv("EMBEDDING_PROVIDER", "stub")

    if p_type.lower() == "stub":
        return StubEmbeddingProvider()

    if p_type.lower() in ("vertex_ai", "gemini", "gcp", "real"):
        from catcher_in_the_rag.providers.gemini import GeminiEmbeddingProvider

        cfg = load_model_config()
        return GeminiEmbeddingProvider(config=cfg)

    msg = f"Unknown embedding provider type: {p_type}. Expected 'stub', 'vertex_ai', 'gemini', or 'real'."
    raise ValueError(msg)
