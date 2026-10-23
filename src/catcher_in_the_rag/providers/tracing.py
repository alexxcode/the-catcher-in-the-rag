"""Tracing interface adhering to SPEC §11 and §13.

Tracing rules:
- Default is Langfuse Cloud on the free Hobby tier.
- Call it only through this tracing interface in `providers/`, with a no-op implementation
  for tests and offline mode.
- Bulk evaluation runs trace a sample, not every package, to stay under the quota.
- Source of truth for metrics is the versioned JSON report, never Langfuse.
"""

from __future__ import annotations

import contextlib
import os
from abc import ABC, abstractmethod
from typing import Any


class BaseTracer(ABC):
    """Abstract interface for application and evaluation tracing."""

    @abstractmethod
    def trace_execution(
        self,
        name: str,
        input_data: Any = None,
        output_data: Any = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        latency_ms: float | None = None,
    ) -> None:
        """Record an execution trace (e.g. audit graph, extraction, retrieval)."""
        ...

    @abstractmethod
    def trace_generation(
        self,
        name: str,
        model: str,
        prompt: str,
        completion: str,
        latency_ms: float | None = None,
        usage: dict[str, int] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record an LLM call with token usage and latency."""
        ...

    @abstractmethod
    def flush(self) -> None:
        """Flush pending traces to remote sink."""
        ...


class NoOpTracer(BaseTracer):
    """Zero-overhead tracer for offline mode and tests."""

    def trace_execution(
        self,
        name: str,
        input_data: Any = None,
        output_data: Any = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        latency_ms: float | None = None,
    ) -> None:
        pass

    def trace_generation(
        self,
        name: str,
        model: str,
        prompt: str,
        completion: str,
        latency_ms: float | None = None,
        usage: dict[str, int] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        pass

    def flush(self) -> None:
        pass


class LangfuseTracer(BaseTracer):
    """Real Langfuse Cloud tracer for live monitoring and token cost audit."""

    def __init__(
        self,
        public_key: str | None = None,
        secret_key: str | None = None,
        host: str | None = None,
    ) -> None:
        self.public_key = public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
        self.secret_key = secret_key or os.getenv("LANGFUSE_SECRET_KEY")
        self.host = (
            host
            or os.getenv("LANGFUSE_BASE_URL")
            or os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        )
        self._client: Any = None

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from langfuse import Langfuse

            self._client = Langfuse(
                public_key=self.public_key,
                secret_key=self.secret_key,
                host=self.host,
            )
            return self._client
        except Exception:
            return None

    def trace_execution(
        self,
        name: str,
        input_data: Any = None,
        output_data: Any = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        latency_ms: float | None = None,
    ) -> None:
        client = self._ensure_client()
        if not client:
            return
        with contextlib.suppress(Exception):
            meta = dict(metadata or {})
            if latency_ms is not None:
                meta["latency_ms"] = latency_ms
            if tags:
                meta["tags"] = tags

            # Langfuse v5+ OpenTelemetry observation API
            if hasattr(client, "start_observation"):
                obs = client.start_observation(
                    name=name,
                    as_type="span",
                    input=input_data,
                    output=output_data,
                    metadata=meta,
                )
                if obs and hasattr(obs, "end"):
                    obs.end()
            # Langfuse v2/v3 legacy API
            elif hasattr(client, "trace"):
                trace = client.trace(
                    name=name,
                    input=input_data,
                    output=output_data,
                    metadata=meta,
                    tags=tags or ["the-catcher-in-the-rag"],
                )
                if latency_ms is not None and hasattr(trace, "update"):
                    trace.update(metadata={"latency_ms": latency_ms})

    def trace_generation(
        self,
        name: str,
        model: str,
        prompt: str,
        completion: str,
        latency_ms: float | None = None,
        usage: dict[str, int] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        client = self._ensure_client()
        if not client:
            return
        with contextlib.suppress(Exception):
            meta = dict(metadata or {})
            if latency_ms is not None:
                meta["latency_ms"] = latency_ms

            # Langfuse v5+ OpenTelemetry generation API
            if hasattr(client, "start_observation"):
                gen = client.start_observation(
                    name=name,
                    as_type="generation",
                    model=model,
                    input=prompt,
                    output=completion,
                    usage_details=usage,
                    metadata=meta,
                )
                if gen and hasattr(gen, "end"):
                    gen.end()
            # Langfuse v2/v3 legacy API
            elif hasattr(client, "trace"):
                trace = client.trace(name=f"trace-{name}", tags=["llm-generation"])
                trace.generation(
                    name=name,
                    model=model,
                    input=prompt,
                    output=completion,
                    usage=usage,
                    metadata=meta,
                )

    def flush(self) -> None:
        client = self._ensure_client()
        if client and hasattr(client, "flush"):
            with contextlib.suppress(Exception):
                client.flush()


def is_tracing_enabled() -> bool:
    """Return True if Langfuse environment variables are configured."""
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def get_tracer(sample_rate: float = 1.0) -> BaseTracer:
    """Factory to get the configured tracer.

    Defaults to NoOpTracer unless LANGFUSE_PUBLIC_KEY is set and sample check passes.
    """
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")

    if not public_key or not secret_key:
        return NoOpTracer()

    # Sample rate control to conserve Hobby tier units
    if sample_rate < 1.0:
        import random

        if random.random() > sample_rate:
            return NoOpTracer()

    return LangfuseTracer(public_key=public_key, secret_key=secret_key)
