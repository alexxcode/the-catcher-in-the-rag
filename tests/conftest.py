"""
pytest configuration and shared fixtures.

Rules (SPEC §11):
  - Never call a real LLM in tests.
  - All tests run with the stub provider unless explicitly marked @pytest.mark.real_llm.
  - Real-LLM tests live under evals/ and are opt-in via `make eval`.
"""
from __future__ import annotations

import pytest


@pytest.fixture(scope="session", autouse=True)
def force_stub_provider(monkeypatch_session: pytest.MonkeyPatch) -> None:  # type: ignore[name-defined]
    """Ensure the stub provider is always active during the test session."""
    import os
    os.environ.setdefault("LLM_PROVIDER", "stub")
    os.environ.setdefault("EMBEDDING_PROVIDER", "stub")


# ---------------------------------------------------------------------------
# Marker: tests that call real LLMs (excluded from CI, run only in evals/)
# ---------------------------------------------------------------------------
def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "real_llm: marks tests that require real LLM/embedding API calls "
        "(deselected in CI with -m 'not real_llm')",
    )
