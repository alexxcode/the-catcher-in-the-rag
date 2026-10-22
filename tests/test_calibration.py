from __future__ import annotations

import tempfile
from datetime import date
from decimal import Decimal
from pathlib import Path

import yaml

from catcher_in_the_rag.schemas import (
    AnomalyType,
    DecisionType,
    GoodsReceipt,
    Invoice,
    PackageLabels,
    ProcurementPackage,
    PurchaseOrder,
    ReceiptItem,
)
from catcher_in_the_rag.schemas.common import LineItem
from catcher_in_the_rag.schemas.contract import Contract, RateItem
from datagen.templates import render_contract_text
from evals.calibrate_bartleby import (
    BartlebyCalibrator,
    apply_calibrated_thresholds,
    print_calibration_summary,
    save_calibration_report,
)


def _create_test_package(is_a09: bool) -> tuple[ProcurementPackage, PackageLabels]:
    raw_text, clauses = render_contract_text(
        contract_id="CTR-CALIB-01",
        vendor_name="Test Vendor Corp",
        vendor_id="VND-01",
        effective_date=date(2025, 1, 1),
        expiration_date=date(2026, 1, 1),
        rate_items=[("SKU-1", "Standard Widget", Decimal("100.00"))],
        payment_terms_days=30,
        late_penalty_pct=Decimal("1.0"),
    )
    contract = Contract(
        contract_id="CTR-CALIB-01",
        vendor_name="Test Vendor Corp",
        vendor_id="VND-01",
        effective_date=date(2025, 1, 1),
        expiration_date=date(2026, 1, 1),
        auto_renew=False,
        raw_text=raw_text,
        payment_terms_days=30,
        late_delivery_penalty_pct_per_day=Decimal("1.0"),
        rate_sheet=[
            RateItem(
                item_id="SKU-1", description="Standard Widget", max_unit_price=Decimal("100.00")
            )
        ],
        clauses=clauses,
    )

    po = PurchaseOrder(
        po_id="PO-01",
        contract_id="CTR-CALIB-01",
        vendor_id="VND-01",
        vendor_name="Test Vendor Corp",
        issue_date=date(2025, 5, 1),
        expected_delivery_date=date(2025, 6, 1),
        line_items=[
            LineItem(
                item_id="SKU-1",
                description="Standard Widget",
                quantity=Decimal("1"),
                unit_price=Decimal("100.00"),
                total_amount=Decimal("100.00"),
            )
        ],
        total_amount=Decimal("100.00"),
    )

    receipt = GoodsReceipt(
        receipt_id="REC-01",
        po_id="PO-01",
        delivery_date=date(2025, 6, 1),
        received_by="Receiving Clerk",
        items=[
            ReceiptItem(
                item_id="SKU-1",
                quantity_received=Decimal("1"),
            )
        ],
    )

    if is_a09:
        # Extra uncontracted surcharge
        inv_items = [
            LineItem(
                item_id="EXP-FEE",
                description="Expedited Rush Delivery Surcharge",
                quantity=Decimal("1"),
                unit_price=Decimal("50.00"),
                total_amount=Decimal("50.00"),
            )
        ]
        inv_total = Decimal("50.00")
    else:
        inv_items = [
            LineItem(
                item_id="SKU-1",
                description="Standard Widget",
                quantity=Decimal("1"),
                unit_price=Decimal("100.00"),
                total_amount=Decimal("100.00"),
            )
        ]
        inv_total = Decimal("100.00")

    invoice = Invoice(
        invoice_id="INV-01",
        vendor_id="VND-01",
        vendor_name="Test Vendor Corp",
        po_id="PO-01",
        invoice_date=date(2025, 6, 1),
        payment_due_date=date(2025, 7, 1),
        line_items=inv_items,
        total_amount=inv_total,
    )

    pkg = ProcurementPackage(
        package_id="PKG-TEST",
        invoice=invoice,
        po=po,
        receipt=receipt,
        contract=contract,
    )
    labels = PackageLabels(
        package_id="PKG-TEST",
        anomalies=[AnomalyType.A09] if is_a09 else [AnomalyType.A10],
        expected_decision=DecisionType.REJECT if is_a09 else DecisionType.APPROVE,
    )
    return pkg, labels


def test_bartleby_calibration_sweep() -> None:
    pkg_a09, lbl_a09 = _create_test_package(is_a09=True)
    pkg_a10, lbl_a10 = _create_test_package(is_a09=False)

    calibrator = BartlebyCalibrator(dataset=[(pkg_a09, lbl_a09), (pkg_a10, lbl_a10)])
    report = calibrator.run_sweep(relevance_steps=[0.3, 0.5, 0.7], span_steps=[1])

    assert len(report.candidates) == 3
    assert report.total_packages == 2
    assert report.optimal_threshold > 0.0

    print_calibration_summary(report)

    with tempfile.TemporaryDirectory() as tmp_dir:
        report_file = save_calibration_report(report, output_dir=Path(tmp_dir))
        assert report_file.exists()

        # Test applying thresholds to dummy YAML
        cfg_file = Path(tmp_dir) / "thresholds.yaml"
        cfg_file.write_text(
            yaml.safe_dump({"bartleby": {"min_relevance_score": 0.1}}), encoding="utf-8"
        )
        applied = apply_calibrated_thresholds(report, config_path=cfg_file)
        assert applied is True

        reloaded = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
        assert reloaded["bartleby"]["min_relevance_score"] == report.optimal_threshold
