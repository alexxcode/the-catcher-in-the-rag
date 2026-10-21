from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from catcher_in_the_rag.schemas import (
    AnomalyType,
    Citation,
    Contract,
    Decision,
    DecisionType,
    Finding,
    GoodsReceipt,
    Invoice,
    LineItem,
    PackageLabels,
    PurchaseOrder,
    RateItem,
    ReceiptItem,
)


def test_citation_validation() -> None:
    cit = Citation(
        document_id="CTR-001",
        start_char=10,
        end_char=25,
        text="payable Net 30",
    )
    assert cit.start_char == 10
    assert cit.end_char == 25

    with pytest.raises(ValueError, match="end_char .* must be greater than start_char"):
        Citation(document_id="CTR-001", start_char=20, end_char=10, text="invalid")


def test_line_item_math_validation() -> None:
    item = LineItem(
        item_id="SKU-1",
        description="Monitor",
        quantity=Decimal("3"),
        unit_price=Decimal("150.00"),
        total_amount=Decimal("450.00"),
    )
    assert item.total_amount == Decimal("450.00")

    # Mismatched total should raise ValueError
    with pytest.raises(ValueError, match="total_amount .* does not match"):
        LineItem(
            item_id="SKU-1",
            description="Monitor",
            quantity=Decimal("3"),
            unit_price=Decimal("150.00"),
            total_amount=Decimal("999.00"),
        )


def test_contract_span_finder() -> None:
    raw_text = "Section 2. All invoices payable Net 30 days from confirmed receipt."
    contract = Contract(
        contract_id="CTR-TEST",
        vendor_id="VND-01",
        vendor_name="Test Vendor",
        effective_date=date(2025, 1, 1),
        expiration_date=date(2026, 1, 1),
        payment_terms_days=30,
        rate_sheet=[RateItem(item_id="S1", description="Item", max_unit_price=Decimal("10.00"))],
        clauses=[],
        raw_text=raw_text,
    )

    span = contract.find_span("payable Net 30")
    assert span is not None
    start, end = span
    assert raw_text[start:end] == "payable Net 30"
    assert contract.find_span("nonexistent text") is None


def test_procurement_package_models_roundtrip() -> None:
    item = LineItem(
        item_id="SKU-1",
        description="Widget",
        quantity=Decimal("2"),
        unit_price=Decimal("50.00"),
        total_amount=Decimal("100.00"),
    )

    po = PurchaseOrder(
        po_id="PO-001",
        contract_id="CTR-001",
        vendor_id="V-01",
        vendor_name="Acme",
        issue_date=date(2025, 1, 10),
        expected_delivery_date=date(2025, 1, 20),
        line_items=[item],
        total_amount=Decimal("100.00"),
    )

    receipt = GoodsReceipt(
        receipt_id="RCV-001",
        po_id="PO-001",
        delivery_date=date(2025, 1, 18),
        received_by="John Doe",
        items=[ReceiptItem(item_id="SKU-1", quantity_received=Decimal("2"))],
    )

    inv = Invoice(
        invoice_id="INV-001",
        po_id="PO-001",
        vendor_id="V-01",
        vendor_name="Acme",
        invoice_date=date(2025, 1, 19),
        payment_due_date=date(2025, 2, 18),
        line_items=[item],
        total_amount=Decimal("100.00"),
    )

    # JSON serialization roundtrip
    inv_json = inv.model_dump_json()
    reloaded_inv = Invoice.model_validate_json(inv_json)
    assert reloaded_inv.invoice_id == "INV-001"
    assert reloaded_inv.total_amount == Decimal("100.00")
    assert po.po_id == "PO-001"
    assert receipt.receipt_id == "RCV-001"


def test_finding_and_decision_models() -> None:
    finding = Finding(
        finding_id="F-01",
        anomaly_type=AnomalyType.A01,
        field="unit_price",
        description="Unit price exceeds contract ceiling",
        severity="ERROR",
        citation=Citation(
            document_id="CTR-001",
            start_char=5,
            end_char=20,
            text="rate ceiling $50",
        ),
        calculated_difference=Decimal("25.00"),
    )

    decision = Decision(
        package_id="PKG-01",
        decision=DecisionType.REJECT,
        findings=[finding],
        requires_human_review=False,
        justification="Rejected due to rate violation A01",
    )

    assert decision.decision == DecisionType.REJECT
    assert len(decision.findings) == 1
    assert decision.findings[0].citation is not None


def test_package_labels_model() -> None:
    labels = PackageLabels(
        package_id="PKG-01",
        anomalies=[AnomalyType.A05],
        expected_decision=DecisionType.REVIEW,
        metadata={"delay_days": 7},
    )
    assert labels.anomalies == [AnomalyType.A05]
    assert labels.expected_decision == DecisionType.REVIEW
