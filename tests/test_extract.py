from __future__ import annotations

from datetime import date
from decimal import Decimal

from catcher_in_the_rag.extract.contract_extractor import (
    HybridContractExtractor,
    RuleBasedContractExtractor,
)
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
