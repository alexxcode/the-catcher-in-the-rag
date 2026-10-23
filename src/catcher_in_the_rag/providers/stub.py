from __future__ import annotations

import hashlib
import math
from typing import Any, TypeVar

from pydantic import BaseModel

from catcher_in_the_rag.providers.base import EmbeddingProvider, LLMProvider

T = TypeVar("T", bound=BaseModel)


class StubLLMProvider(LLMProvider):
    """Deterministic, zero-network LLM provider for tests, CI, and offline execution."""

    def __init__(
        self, registered_responses: dict[type[BaseModel], BaseModel] | None = None
    ) -> None:
        self.registered_responses = registered_responses or {}

    def register_response(self, schema: type[T], response: T) -> None:
        """Register a canned structured response for a specific schema type."""
        self.registered_responses[schema] = response

    def extract_structured(
        self,
        text: str,
        schema: type[T],
        system_prompt: str | None = None,
    ) -> T:
        """Extract structured object.

        First checks registered canned responses, then attempts JSON deserialization,
        or constructs a default instance.
        """
        if schema in self.registered_responses:
            return self.registered_responses[schema]  # type: ignore[return-value]

        # If text is already JSON matching the schema
        try:
            return schema.model_validate_json(text)
        except Exception:
            pass

        # Try parsing JSON from markdown code fence
        if "```json" in text:
            start = text.index("```json") + 7
            end = text.index("```", start)
            snippet = text[start:end].strip()
            try:
                return schema.model_validate_json(snippet)
            except Exception:
                pass

        # Fallback to constructing an empty/mock model if allowed
        try:
            return schema.model_validate({})
        except Exception as err:
            msg = f"StubLLMProvider cannot instantiate {schema.__name__} from provided text: {err}"
            raise ValueError(msg) from err

    def generate_text(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        """Deterministic text generation based on prompt."""
        return f"[STUB_RESPONSE for prompt of length {len(prompt)}]"

    def check_model_health(self) -> dict[str, Any]:
        """Offline deterministic health check."""
        return {
            "status": "healthy",
            "model_id": "stub-offline",
            "provider": "stub",
            "latency_ms": 0.05,
        }


class StubEmbeddingProvider(EmbeddingProvider):
    """Deterministic embedding provider that maps text to a normalized 768-dim float vector."""

    def __init__(self, dimensionality: int = 768) -> None:
        self.dim = dimensionality

    def _hash_to_vector(self, text: str) -> list[float]:
        # Generate deterministic vector from sha256 hash repeated across dimensions
        h = hashlib.sha256(text.encode("utf-8")).digest()
        raw_vals: list[float] = []
        for i in range(self.dim):
            byte_val = h[i % len(h)]
            # Add positional variation
            raw_vals.append(float(byte_val) * (1.0 + (i % 17) * 0.1) - 128.0)

        # L2 normalize
        norm = math.sqrt(sum(v * v for v in raw_vals)) or 1.0
        return [round(v / norm, 6) for v in raw_vals]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_to_vector(t) for t in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._hash_to_vector(query)
