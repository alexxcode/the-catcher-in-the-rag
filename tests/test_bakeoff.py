from __future__ import annotations

import tempfile
from pathlib import Path

from catcher_in_the_rag.schemas.contract import Contract, RateItem
from evals.bakeoff import (
    CandidateResult,
    ExtractionBakeOff,
    print_bakeoff_summary,
    save_bakeoff_report,
)


def test_candidate_result_structure() -> None:
    res = CandidateResult(
        candidate_name="Test Model",
        model_id="test-id",
        provider_type="vertex_ai",
        sample_size=10,
        mean_latency_ms=12.5,
        estimated_cost_per_1k_usd=0.50,
        error_count=0,
        error_rate=0.0,
        field_accuracies={"payment_terms_days": 1.0, "rate_sheet": 1.0},
        overall_perfect_accuracy=1.0,
        is_simulated=False,
    )
    assert res.candidate_name == "Test Model"
    assert res.overall_perfect_accuracy == 1.0


def test_extraction_bakeoff_execution() -> None:
    from datetime import date
    from decimal import Decimal

    from datagen.templates import render_contract_text

    rate_items = [
        ("SKU-TEST", "Standard Unit Test Hardware", Decimal("45.00")),
    ]
    raw_text, _ = render_contract_text(
        contract_id="CTR-TEST",
        vendor_name="Test Vendor LLC",
        vendor_id="VEND-TEST",
        effective_date=date(2025, 1, 1),
        expiration_date=date(2026, 12, 31),
        rate_items=rate_items,
        payment_terms_days=30,
        late_penalty_pct=Decimal("0.5"),
        auto_renew=False,
    )
    contract = Contract(
        contract_id="CTR-TEST",
        vendor_name="Test Vendor LLC",
        vendor_id="VEND-TEST",
        effective_date=date(2025, 1, 1),
        expiration_date=date(2026, 12, 31),
        auto_renew=False,
        raw_text=raw_text,
        payment_terms_days=30,
        late_delivery_penalty_pct_per_day=Decimal("0.5"),
        rate_sheet=[
            RateItem(
                item_id="SKU-TEST",
                description="Standard Unit Test Hardware",
                max_unit_price=Decimal("45.00"),
            )
        ],
    )

    bakeoff = ExtractionBakeOff(
        contracts=[(raw_text, contract)],
        sample_size=1,
    )
    results = bakeoff.run_bakeoff()

    assert len(results) >= 2
    rule_res = results[0]
    assert rule_res.field_accuracies["payment_terms_days"] == 1.0
    assert rule_res.field_accuracies["expiration_date"] == 1.0
    assert rule_res.overall_perfect_accuracy == 1.0

    # Test summary printing and save
    print_bakeoff_summary(results)

    with tempfile.TemporaryDirectory() as tmp_dir:
        report_path = save_bakeoff_report(results, output_dir=Path(tmp_dir))
        assert report_path.exists()


def test_generate_diverse_contracts() -> None:
    from datagen.generator import generate_diverse_contracts

    contracts = generate_diverse_contracts(count=3, base_seed=123)
    assert len(contracts) == 3
    for raw_text, contract in contracts:
        assert contract.contract_id.startswith("CTR-")
        assert "Agreed Cap of" in raw_text or "calendar days" in raw_text


def test_model_health_check() -> None:
    from catcher_in_the_rag.providers.factory import get_llm_provider

    provider = get_llm_provider(provider_type="stub")
    health = provider.check_model_health()
    assert health["status"] == "simulated"
    assert health["provider"] == "stub"
    assert "warning" in health
