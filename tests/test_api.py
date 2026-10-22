from __future__ import annotations

import json
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from catcher_in_the_rag.api.main import app
from catcher_in_the_rag.schemas import (
    Contract,
    DecisionType,
    GoodsReceipt,
    Invoice,
    LineItem,
    ProcurementPackage,
    PurchaseOrder,
    RateItem,
    ReceiptItem,
)


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def clean_procurement_package() -> ProcurementPackage:
    invoice = Invoice(
        invoice_id="INV-API-001",
        po_id="PO-API-001",
        vendor_id="VEND-ACME",
        vendor_name="Acme Industrial",
        invoice_date=date(2026, 3, 15),
        payment_due_date=date(2026, 4, 14),
        line_items=[
            LineItem(
                item_id="ITEM-1",
                description="Industrial Widget",
                quantity=Decimal("10"),
                unit_price=Decimal("100.00"),
                total_amount=Decimal("1000.00"),
            )
        ],
        total_amount=Decimal("1000.00"),
    )
    po = PurchaseOrder(
        po_id="PO-API-001",
        contract_id="CTR-API-001",
        vendor_id="VEND-ACME",
        vendor_name="Acme Industrial",
        issue_date=date(2026, 3, 1),
        expected_delivery_date=date(2026, 3, 15),
        line_items=[
            LineItem(
                item_id="ITEM-1",
                description="Industrial Widget",
                quantity=Decimal("10"),
                unit_price=Decimal("100.00"),
                total_amount=Decimal("1000.00"),
            )
        ],
        total_amount=Decimal("1000.00"),
    )
    receipt = GoodsReceipt(
        receipt_id="RC-API-001",
        po_id="PO-API-001",
        delivery_date=date(2026, 3, 10),
        received_by="John Staff",
        items=[
            ReceiptItem(
                item_id="ITEM-1",
                quantity_received=Decimal("10"),
                quantity_rejected=Decimal("0"),
                condition="GOOD",
            )
        ],
    )
    contract = Contract(
        contract_id="CTR-API-001",
        title="Master Services Agreement",
        vendor_name="Acme Industrial",
        effective_date=date(2025, 1, 1),
        expiration_date=date(2027, 1, 1),
        payment_terms="NET30",
        rates=[
            RateItem(
                item_id="ITEM-1",
                description="Industrial Widget",
                unit_rate=Decimal("100.00"),
            )
        ],
        clauses=[],
    )
    return ProcurementPackage(
        package_id="PKG-API-TEST",
        contract=contract,
        po=po,
        receipt=receipt,
        invoice=invoice,
    )


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["app"] == "The Catcher in the RAG"


def test_list_packages(client: TestClient) -> None:
    response = client.get("/api/packages")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_audit_package_payload(
    client: TestClient, clean_procurement_package: ProcurementPackage
) -> None:
    pkg_dict = json.loads(clean_procurement_package.model_dump_json())
    response = client.post("/api/audit", json={"package": pkg_dict})
    assert response.status_code == 200
    data = response.json()
    assert data["package_id"] == "PKG-API-TEST"
    assert data["decision"]["decision"] == DecisionType.APPROVE.value
    assert len(data["validated_findings"]) == 0


def test_audit_missing_payload(client: TestClient) -> None:
    response = client.post("/api/audit", json={})
    assert response.status_code == 400


def test_review_override_flow(client: TestClient) -> None:
    override_payload = {
        "package_id": "PKG-API-TEST",
        "override_decision": "APPROVE",
        "notes": "Reviewed and approved by chief auditor.",
        "auditor_id": "auditor_alex",
    }
    response = client.post("/api/review/override", json=override_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["package_id"] == "PKG-API-TEST"
    assert data["final_decision"] == "APPROVE"
    assert "successfully" in data["status"]


def test_reports_endpoints(client: TestClient, tmp_path: Path) -> None:
    # Test with custom reports dir
    fake_report = tmp_path / "eval_report_test.json"
    fake_report.write_text(
        json.dumps(
            {
                "timestamp": "2026-10-22T00:00:00Z",
                "mode": "baseline",
                "total_packages": 10,
                "decision_accuracy": 1.0,
            }
        ),
        encoding="utf-8",
    )

    response = client.get(f"/api/reports?reports_dir={tmp_path}")
    assert response.status_code == 200
    assert "eval_report_test.json" in response.json()

    response_latest = client.get(f"/api/reports/latest?reports_dir={tmp_path}")
    assert response_latest.status_code == 200
    assert response_latest.json()["decision_accuracy"] == 1.0
