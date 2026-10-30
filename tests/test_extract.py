from __future__ import annotations

import json
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from catcher_in_the_rag.extract.contract_extractor import (
    HybridContractExtractor,
    RuleBasedContractExtractor,
)
from catcher_in_the_rag.providers.gemini import GeminiProvider
from datagen.templates import render_contract_text


def test_rule_based_contract_extractor() -> None:
    rate_items = [
        ("SKU-101", "Enterprise Workstation 32GB RAM", Decimal("1450.00")),
        ("SKU-102", "Ultra-Wide Monitor 34-inch", Decimal("420.00")),
    ]
    raw_text, _ = render_contract_text(
        contract_id="CTR-TEST-001",
        vendor_name="Apex Systems & Hardware LLC",
        vendor_id="VND-001",
        effective_date=date(2025, 2, 1),
        expiration_date=date(2026, 2, 1),
        rate_items=rate_items,
        payment_terms_days=45,
        late_penalty_pct=Decimal("1.5"),
        auto_renew=True,
    )

    extractor = RuleBasedContractExtractor()
    contract = extractor.extract(raw_text)

    assert contract.contract_id == "CTR-TEST-001"
    assert contract.vendor_name == "Apex Systems & Hardware LLC"
    assert contract.vendor_id == "VND-001"
    assert contract.effective_date == date(2025, 2, 1)
    assert contract.expiration_date == date(2026, 2, 1)
    assert contract.auto_renew is True
    assert contract.payment_terms_days == 45
    assert contract.late_delivery_penalty_pct_per_day == Decimal("1.5")
    assert len(contract.rate_sheet) == 2
    assert contract.rate_sheet[0].item_id == "SKU-101"
    assert contract.rate_sheet[0].max_unit_price == Decimal("1450.00")
    assert len(contract.clauses) >= 10


def test_hybrid_contract_extractor_fallback() -> None:
    rate_items = [
        ("SKU-201", "Cloud Storage Block", Decimal("220.00")),
    ]
    raw_text, _ = render_contract_text(
        contract_id="CTR-TEST-002",
        vendor_name="BluePeak Cloud Inc",
        vendor_id="VND-002",
        effective_date=date(2025, 3, 1),
        expiration_date=date(2026, 3, 1),
        rate_items=rate_items,
        payment_terms_days=30,
        late_penalty_pct=Decimal("0.5"),
    )

    extractor = HybridContractExtractor()
    contract = extractor.extract(raw_text)

    assert contract.contract_id == "CTR-TEST-002"
    assert contract.payment_terms_days == 30
    assert contract.late_delivery_penalty_pct_per_day == Decimal("0.5")
    assert len(contract.rate_sheet) == 1


def test_diverse_contract_fails_rule_extractor() -> None:
    """Demonstrate that rigid regex rules fail on independently/diversely phrased contracts.

    Proves that 100% extraction is an artifact of rigid regex on identical templates,
    and justifies why an LLM extractor is required for natural contracts.
    """
    rate_items = [
        ("SKU-101", "Enterprise Workstation 32GB RAM", Decimal("1450.00")),
    ]
    raw_text, _ = render_contract_text(
        contract_id="CTR-DIV-001",
        vendor_name="Apex Systems",
        vendor_id="VND-001",
        effective_date=date(2025, 1, 1),
        expiration_date=date(2026, 1, 1),
        rate_items=rate_items,
        payment_terms_days=45,
        late_penalty_pct=Decimal("1.5"),
        style="diverse",
    )

    rule_extractor = RuleBasedContractExtractor()
    extracted = rule_extractor.extract(raw_text)

    # Rigid regex expectations:
    # 1. Rate sheet fails because diverse format is "Item [SKU] (Desc): Agreed Cap of $..."
    assert len(extracted.rate_sheet) == 0
    # 2. Payment terms fails because text says "payable within 45 calendar days" without "Net"
    assert extracted.payment_terms_days == 30  # falls back to default 30 instead of 45


def test_llm_failure_is_recorded_and_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("RAW_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("RUN_ID", "test_llm_failure")
    raw_text, _ = render_contract_text(
        contract_id="CTR-TEST-ERR",
        vendor_id="VND-TEST-ERR",
        vendor_name="Apex Systems & Hardware LLC",
        effective_date=date(2025, 3, 1),
        expiration_date=date(2026, 3, 1),
        rate_items=[("SKU-101", "Enterprise Workstation 32GB RAM", Decimal("1450.00"))],
        payment_terms_days=30,
        late_penalty_pct=Decimal("0.5"),
    )

    provider = GeminiProvider(config={})  # no network: extract_structured is replaced below

    def _fail(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("404 NOT_FOUND: model unavailable")

    monkeypatch.setattr(provider, "extract_structured", _fail)
    extractor = HybridContractExtractor(provider=provider)
    with caplog.at_level("WARNING"):
        contract = extractor.extract(raw_text)

    assert extractor.last_method_used == "fallback_rule"
    assert extractor.last_error == "RuntimeError: 404 NOT_FOUND: model unavailable"
    assert contract.contract_id == "CTR-TEST-ERR"
    assert "404 NOT_FOUND" in caplog.text
    record = json.loads((tmp_path / "test_llm_failure.jsonl").read_text(encoding="utf-8"))
    assert record["status"] == "error"
    assert record["error_type"] == "RuntimeError"
