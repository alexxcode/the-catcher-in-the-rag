from __future__ import annotations

from datetime import date
from decimal import Decimal

from catcher_in_the_rag.schemas import (
    AnomalyType,
    Citation,
    Contract,
    Finding,
    RateItem,
)
from catcher_in_the_rag.validate.validator import validate_findings


def test_validator_keeps_verbatim_and_drops_hallucinated() -> None:
    raw_text = "Section 2. Invoices payable Net 30 days from confirmed receipt."
    contract = Contract(
        contract_id="CTR-V1",
        vendor_id="V-1",
        vendor_name="Vendor One",
        effective_date=date(2025, 1, 1),
        expiration_date=date(2026, 1, 1),
        payment_terms_days=30,
        rate_sheet=[RateItem(item_id="S1", description="Item", max_unit_price=Decimal("10.00"))],
        clauses=[],
        raw_text=raw_text,
    )

    span = contract.find_span("payable Net 30 days")
    assert span is not None

    valid_finding = Finding(
        finding_id="F-1",
        anomaly_type=AnomalyType.A04,
        field="payment_terms",
        description="Payment terms issue",
        citation=Citation(
            document_id="CTR-V1",
            start_char=span[0],
            end_char=span[1],
            text="payable Net 30 days",
        ),
    )

    # Hallucinated quote
    invalid_finding = Finding(
        finding_id="F-2",
        anomaly_type=AnomalyType.A04,
        field="payment_terms",
        description="Hallucinated clause",
        citation=Citation(
            document_id="CTR-V1",
            start_char=0,
            end_char=15,
            text="Nonexistent quote not in source",
        ),
    )

    # Non-clause finding without citation (should be preserved)
    non_clause_finding = Finding(
        finding_id="F-3",
        anomaly_type=AnomalyType.A08,
        field="total_amount",
        description="Amount above limit",
        citation=None,
    )

    # Contract clause finding WITHOUT citation (must be dropped per Principle #2)
    contract_finding_no_cit = Finding(
        finding_id="F-4",
        anomaly_type=AnomalyType.A04,
        field="payment_terms",
        description="Contract finding lacking mandatory citation",
        citation=None,
    )

    valid, dropped = validate_findings(
        [valid_finding, invalid_finding, non_clause_finding, contract_finding_no_cit],
        contract,
    )
    assert len(valid) == 2
    assert len(dropped) == 2
    assert {d.finding_id for d in dropped} == {"F-2", "F-4"}
    assert valid_finding in valid
    assert non_clause_finding in valid
