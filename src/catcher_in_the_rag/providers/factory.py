from __future__ import annotations

import os
import secrets
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


def get_llm_provider(
    provider_type: str | None = None,
    model_id: str | None = None,
) -> LLMProvider:
    """Get configured LLM provider (defaults to stub for offline execution and tests)."""
    p_type = provider_type or os.getenv("LLM_PROVIDER", "stub")

    if p_type.lower() == "stub":
        return StubLLMProvider()

    if p_type.lower() in ("vertex_ai", "gemini", "gcp", "real"):
        # Real GCP generative AI provider (opt-in for eval/production)
        from catcher_in_the_rag.providers.gemini import GeminiProvider

        cfg = load_model_config()
        return GeminiProvider(config=cfg, model_id=model_id)

    msg = f"Unknown LLM provider type: {p_type}. Expected 'stub', 'vertex_ai', 'gemini', or 'real'."
    raise ValueError(msg)


def resolve_llm_provider(credential: str | None = None) -> LLMProvider:
    """Single provider decision for API, UI and pipeline (spend safeguard).

    - DEMO_ACCESS_PIN unset (local dev): LLM_PROVIDER from env (stub if unset).
    - DEMO_ACCESS_PIN set and ``credential`` matches: LLM_PROVIDER from env.
    - Otherwise: stub if ALLOW_PUBLIC_STUB (default true), else PermissionError.
    """
    configured_pin = os.getenv("DEMO_ACCESS_PIN", "").strip()
    if not configured_pin:
        return get_llm_provider()
    if credential and secrets.compare_digest(credential.strip(), configured_pin):
        return get_llm_provider()
    if os.getenv("ALLOW_PUBLIC_STUB", "true").lower() in ("1", "true", "yes"):
        return get_llm_provider(provider_type="stub")
    raise PermissionError("valid DEMO_ACCESS_PIN required")


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
