"""Provider interfaces and implementations for LLM and embeddings."""

from catcher_in_the_rag.providers.base import EmbeddingProvider, LLMProvider
from catcher_in_the_rag.providers.factory import (
    get_embedding_provider,
    get_llm_provider,
    load_model_config,
)
from catcher_in_the_rag.providers.stub import StubEmbeddingProvider, StubLLMProvider
from catcher_in_the_rag.providers.tracing import (
    BaseTracer,
    LangfuseTracer,
    NoOpTracer,
    get_tracer,
    is_tracing_enabled,
)

__all__ = [
    "BaseTracer",
    "EmbeddingProvider",
    "LLMProvider",
    "LangfuseTracer",
    "NoOpTracer",
    "StubEmbeddingProvider",
    "StubLLMProvider",
    "get_embedding_provider",
    "get_llm_provider",
    "get_tracer",
    "is_tracing_enabled",
    "load_model_config",
]
