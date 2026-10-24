"""Centralized pricing loader and cost calculation for Gemini / Vertex AI models.

Loads rates dynamically from config/pricing.yaml to avoid hardcoded pricing in code.
"""

from pathlib import Path
from typing import Any

import yaml

_PRICING_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "pricing.yaml"

_FALLBACK_RATES: dict[str, dict[str, Any]] = {
    "gemini-3.5-flash": {
        "prompt_cost_per_1m": 0.50,
        "candidate_cost_per_1m": 3.00,
    },
    "gemini-3.8-flash": {
        "prompt_cost_per_1m": 0.50,
        "candidate_cost_per_1m": 3.00,
    },
    "gemini-1.5-flash-legacy": {
        "prompt_cost_per_1m": 0.075,
        "candidate_cost_per_1m": 0.30,
    },
}


def load_pricing_config() -> dict[str, Any]:
    """Load pricing rates from config/pricing.yaml."""
    if _PRICING_CONFIG_PATH.exists():
        try:
            with _PRICING_CONFIG_PATH.open(encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            pass
    return {"models": _FALLBACK_RATES, "last_verified": "2026-10-24", "source": "fallback"}


def get_model_rates(model_id: str) -> tuple[float, float]:
    """Return (prompt_cost_per_1m, candidate_cost_per_1m) for a given model ID.

    If model is unknown, falls back to gemini-3.5-flash standard rates ($0.50 / $3.00).
    """
    config = load_pricing_config()
    models = config.get("models", {})

    # Direct match or substring match (e.g. 'gemini-3.5-flash' in 'vertex_ai/gemini-3.5-flash')
    for key, data in models.items():
        if key == model_id or key in model_id:
            return float(data["prompt_cost_per_1m"]), float(data["candidate_cost_per_1m"])

    # Fallback to standard 3.5 Flash rate
    default_data = _FALLBACK_RATES["gemini-3.5-flash"]
    return float(default_data["prompt_cost_per_1m"]), float(default_data["candidate_cost_per_1m"])


def calculate_cost_usd(
    prompt_tokens: int,
    candidate_tokens: int,
    model_id: str = "gemini-3.5-flash",
) -> float:
    """Calculate total USD cost from token counts and model pricing."""
    p_rate, c_rate = get_model_rates(model_id)
    return (prompt_tokens * p_rate / 1_000_000.0) + (candidate_tokens * c_rate / 1_000_000.0)
