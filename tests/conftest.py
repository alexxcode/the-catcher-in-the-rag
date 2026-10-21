"""
pytest configuration and shared fixtures.

Rules (SPEC §11):
  - Never call a real LLM in tests.
  - All tests run with the stub provider unless explicitly marked @pytest.mark.real_llm.
  - Real-LLM tests live under evals/ and are opt-in via `make eval`.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def force_stub_provider() -> None:
    """Ensure the stub provider is always active during the test session.

    Sets environment variables before any test runs. Using os.environ directly
    (not monkeypatch) because monkeypatch is function-scoped in standard pytest.
    """
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
