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
        sample_size=10,
        mean_latency_ms=12.5,
        estimated_cost_per_1k_usd=0.50,
        field_accuracies={"payment_terms_days": 1.0, "rate_sheet": 1.0},
        overall_perfect_accuracy=1.0,
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
