from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from catcher_in_the_rag.providers.tracing import (
    BaseTracer,
    LangfuseTracer,
    NoOpTracer,
    get_tracer,
    is_tracing_enabled,
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
    assert not is_tracing_enabled()
    tracer = get_tracer()
    assert isinstance(tracer, NoOpTracer)


def test_is_tracing_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    assert is_tracing_enabled()
    monkeypatch.delenv("LANGFUSE_ENABLED", raising=False)
    assert isinstance(get_tracer(), NoOpTracer)  # under pytest without LANGFUSE_ENABLED=1
    monkeypatch.setenv("LANGFUSE_ENABLED", "1")
    tracer = get_tracer()
    assert isinstance(tracer, LangfuseTracer)
    assert tracer.public_key == "pk-lf-test"


def test_langfuse_tracer_host_from_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://us.cloud.langfuse.com")
    monkeypatch.delenv("LANGFUSE_HOST", raising=False)
    tracer = LangfuseTracer(public_key="pk", secret_key="sk")
    assert tracer.host == "https://us.cloud.langfuse.com"


def test_langfuse_tracer_v5_observation_api() -> None:
    tracer = LangfuseTracer(public_key="pk-dummy", secret_key="sk-dummy")
    mock_client = MagicMock()
    mock_obs = MagicMock()
    mock_client.start_observation.return_value = mock_obs
    tracer._client = mock_client

    tracer.trace_execution(
        name="audit_run",
        input_data={"pkg": "PKG-01"},
        output_data={"decision": "APPROVE"},
        latency_ms=25.0,
        tags=["audit"],
    )
    mock_client.start_observation.assert_called_once()
    mock_obs.end.assert_called_once()

    mock_client.reset_mock()
    tracer.trace_generation(
        name="llm_call",
        model="gemini-2.5-flash",
        prompt="hello",
        completion="world",
        usage={"input": 10, "output": 5, "total": 15},
        latency_ms=120.0,
    )
    mock_client.start_observation.assert_called_once()
    tracer.flush()
    mock_client.flush.assert_called_once()
