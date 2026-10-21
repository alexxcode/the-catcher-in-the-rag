"""
Scaffold smoke tests.

These tests verify that the package installs correctly and all modules are
importable. They run with the stub provider (no LLM calls, no network).
"""

from __future__ import annotations

import importlib

PACKAGE_MODULES = [
    "catcher_in_the_rag",
    "catcher_in_the_rag.schemas",
    "catcher_in_the_rag.ingest",
    "catcher_in_the_rag.extract",
    "catcher_in_the_rag.match",
    "catcher_in_the_rag.retrieve",
    "catcher_in_the_rag.bartleby",
    "catcher_in_the_rag.apply_terms",
    "catcher_in_the_rag.validate",
    "catcher_in_the_rag.policy",
    "catcher_in_the_rag.providers",
    "catcher_in_the_rag.api",
]


def test_all_modules_importable() -> None:
    """Every package module must import without error (stub provider active)."""
    for module in PACKAGE_MODULES:
        imported = importlib.import_module(module)
        assert imported is not None, f"Failed to import {module}"


def test_stub_provider_env_set(monkeypatch: object) -> None:
    """conftest.py must have set the stub provider env vars."""
    import os

    assert os.environ.get("LLM_PROVIDER") == "stub"
    assert os.environ.get("EMBEDDING_PROVIDER") == "stub"
