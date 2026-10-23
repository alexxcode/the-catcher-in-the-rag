from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMProvider(ABC):
    """Abstract interface for LLM completions and structured extractions."""

    @abstractmethod
    def extract_structured(
        self,
        text: str,
        schema: type[T],
        system_prompt: str | None = None,
    ) -> T:
        """Extract structured data conforming to a Pydantic schema from input text."""
        ...

    @abstractmethod
    def generate_text(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        """Generate a text completion for a given prompt."""
        ...

    @abstractmethod
    def check_model_health(self) -> dict[str, Any]:
        """Verify model availability and fail fast if model is deprecated or unreachable."""
        ...


class EmbeddingProvider(ABC):
    """Abstract interface for dense text embeddings."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document texts."""
        ...

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Embed a single search query."""
        ...
