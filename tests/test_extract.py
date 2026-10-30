from __future__ import annotations

import json
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from catcher_in_the_rag.apply_terms.evaluator import ContractTermsEvaluator
from catcher_in_the_rag.extract.contract_extractor import (
    ExtractedContractTerms,
    HybridContractExtractor,
    RuleBasedContractExtractor,
)
from catcher_in_the_rag.ingest.cuad_loader import DEMO_CUAD_PATH, CUADLoader
from catcher_in_the_rag.ingest.cuad_simulation import SCENARIO_MATCH, build_simulated_cuad_package
from catcher_in_the_rag.providers.gemini import GeminiEmbeddingProvider, GeminiProvider
from catcher_in_the_rag.schemas import AnomalyType
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


def test_gemini_flash_always_routes_to_global(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_REGION", "us-central1")
    assert GeminiProvider(config={}, model_id="gemini-3.5-flash").region == "global"
    assert GeminiProvider(config={}, model_id="gemini-3.8-flash").region == "global"
    assert GeminiProvider(config={}, model_id="gemini-2.5-flash").region == "us-central1"
    assert GeminiEmbeddingProvider(config={}).region == "us-central1"


def test_hybrid_llm_path_requests_extracted_contract_terms(monkeypatch: pytest.MonkeyPatch) -> None:
    raw_text, _ = render_contract_text(
        contract_id="CTR-TEST-LLM",
        vendor_id="VND-TEST-LLM",
        vendor_name="Apex Systems & Hardware LLC",
        effective_date=date(2025, 3, 1),
        expiration_date=date(2026, 3, 1),
        rate_items=[("SKU-101", "Enterprise Workstation 32GB RAM", Decimal("1450.00"))],
        payment_terms_days=45,
        late_penalty_pct=Decimal("0.5"),
    )
    requested: list[Any] = []

    def _fake(text: str, schema: Any, system_prompt: str | None = None, **kwargs: Any) -> Any:
        requested.append(schema)
        return ExtractedContractTerms(
            contract_id="CTR-FROM-LLM",
            vendor_id="VND-FROM-LLM",
            vendor_name="Vendor From LLM",
            effective_date=date(2025, 3, 1),
            expiration_date=date(2026, 3, 1),
            payment_terms_days=45,
            late_delivery_penalty_pct_per_day=Decimal("0.5"),
        )

    provider = GeminiProvider(config={})  # no network: extract_structured is replaced
    monkeypatch.setattr(provider, "extract_structured", _fake)
    extractor = HybridContractExtractor(provider=provider)
    contract = extractor.extract(raw_text)

    assert requested == [ExtractedContractTerms]
    assert extractor.last_method_used == "llm"
    assert extractor.last_error is None
    assert contract.contract_id == "CTR-FROM-LLM"
    assert contract.payment_terms_days == 45
    assert contract.clauses == RuleBasedContractExtractor().extract(raw_text).clauses
    assert contract.raw_text == raw_text


def test_null_expiration_from_llm_is_not_verifiable_never_expired(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LLM returns expiration_date null: Contract keeps None and A06 is NOT_VERIFIABLE only."""
    loader = CUADLoader(cuad_json_path=DEMO_CUAD_PATH)
    # CUAD-0002's annotations carry a past date (2021-12-31): a guessed date would read as expired.
    cuad = next(c for c in loader.load_contracts() if c.contract_id == "CUAD-0002")
    package = build_simulated_cuad_package(cuad, loader, SCENARIO_MATCH)

    def _fake(text: str, schema: Any, system_prompt: str | None = None, **kwargs: Any) -> Any:
        assert "devuelve null" in (system_prompt or "")
        return schema.model_validate(
            {
                "contract_id": "CUAD-0002",
                "vendor_id": "VND-X",
                "vendor_name": "Vendor X",
                "effective_date": "2019-01-01",
                "expiration_date": None,
                "payment_terms_days": None,
                "late_delivery_penalty_pct_per_day": None,
                "rate_sheet": None,
            }
        )

    provider = GeminiProvider(config={})  # no network: extract_structured is replaced
    monkeypatch.setattr(provider, "extract_structured", _fake)
    extractor = HybridContractExtractor(provider=provider)
    contract = extractor.extract(package.contract.raw_text)

    assert extractor.last_method_used == "llm"
    assert contract.expiration_date is None
    assert contract.rate_sheet == []
    assert contract.late_delivery_penalty_pct_per_day == Decimal("0.0")

    findings, _, _ = ContractTermsEvaluator().evaluate(
        package.model_copy(update={"contract": contract})
    )
    a06 = [f for f in findings if f.anomaly_type == AnomalyType.A06]
    assert a06
    assert all(f.verification_status == "NOT_VERIFIABLE" for f in a06)
    assert not any("subsequent to" in f.description for f in a06)
