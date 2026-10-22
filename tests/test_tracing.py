from __future__ import annotations

import pytest

from catcher_in_the_rag.providers.tracing import (
    BaseTracer,
    LangfuseTracer,
    NoOpTracer,
    get_tracer,
)


def test_noop_tracer_methods() -> None:
    tracer = NoOpTracer()
    assert isinstance(tracer, BaseTracer)
    # Ensure none of these raise exceptions
    tracer.trace_execution("test_run", input_data={"k": "v"}, latency_ms=10.0)
    tracer.trace_generation("test_gen", "model-x", "prompt", "completion", 5.0)
    tracer.flush()


def test_get_tracer_defaults_to_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    tracer = get_tracer()
    assert isinstance(tracer, NoOpTracer)


def test_langfuse_tracer_graceful_init() -> None:
    tracer = LangfuseTracer(public_key="pk-dummy", secret_key="sk-dummy")
    assert tracer.public_key == "pk-dummy"
    # Even if langfuse network is unreachable, it handles it gracefully
    tracer.trace_execution("test_step", {"a": 1})
    tracer.flush()
