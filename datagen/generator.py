from __future__ import annotations

import json
import random
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

from catcher_in_the_rag.schemas.common import LineItem
from catcher_in_the_rag.schemas.contract import Clause, Contract, RateItem
from catcher_in_the_rag.schemas.findings import AnomalyType
from catcher_in_the_rag.schemas.labels import PackageLabels
from catcher_in_the_rag.schemas.procurement import (
    GoodsReceipt,
    Invoice,
    ProcurementPackage,
    PurchaseOrder,
    ReceiptItem,
)
from datagen.injector import inject_anomaly
from datagen.templates import VENDORS, render_contract_text


def generate_base_package(seed: int, package_id: str) -> ProcurementPackage:
    """Deterministically generate a clean 3-way matching procurement package with contract."""
    rng = random.Random(seed)

    # 1. Select vendor & items
    vendor_template = rng.choice(VENDORS)
    num_items = rng.randint(1, 3)
    selected_items = rng.sample(vendor_template.items, k=min(num_items, len(vendor_template.items)))

    contract_id = f"CTR-{vendor_template.vendor_id}-{rng.randint(1000, 9999)}"
    po_id = f"PO-{rng.randint(100000, 999999)}"
    receipt_id = f"RCV-{rng.randint(100000, 999999)}"
    invoice_id = f"INV-{rng.randint(100000, 999999)}"

    # 2. Timeline
    base_year = 2025
    effective_date = date(base_year, rng.randint(1, 3), rng.randint(1, 28))
    expiration_date = effective_date + timedelta(days=365)
    po_date = effective_date + timedelta(days=rng.randint(30, 90))
    expected_delivery_date = po_date + timedelta(days=rng.randint(7, 14))
    delivery_date = expected_delivery_date  # Clean on-time delivery
    invoice_date = delivery_date + timedelta(days=rng.randint(1, 3))
    payment_terms_days = 30
    payment_due_date = invoice_date + timedelta(days=payment_terms_days)

    # 3. Rate Sheet & Contract Clauses
    rate_items: list[tuple[str, str, Decimal]] = [
        (it.item_id, it.description, it.base_unit_price) for it in selected_items
    ]
    raw_text, clause_data = render_contract_text(
        contract_id=contract_id,
        vendor_name=vendor_template.name,
        vendor_id=vendor_template.vendor_id,
        effective_date=effective_date,
        expiration_date=expiration_date,
        rate_items=rate_items,
        payment_terms_days=payment_terms_days,
        late_penalty_pct=Decimal("1.0"),
        auto_renew=False,
    )

    contract = Contract(
        contract_id=contract_id,
        vendor_id=vendor_template.vendor_id,
        vendor_name=vendor_template.name,
        effective_date=effective_date,
        expiration_date=expiration_date,
        auto_renew=False,
        payment_terms_days=payment_terms_days,
        late_delivery_penalty_pct_per_day=Decimal("1.0"),
        rate_sheet=[
            RateItem(item_id=sku, description=desc, max_unit_price=price)
            for sku, desc, price in rate_items
        ],
        clauses=[Clause(**cd) for cd in clause_data],  # type: ignore[arg-type]
        raw_text=raw_text,
    )

    # 4. Line items for PO, Receipt, and Invoice
    line_items: list[LineItem] = []
    receipt_items: list[ReceiptItem] = []

    for it in selected_items:
        qty = Decimal(str(rng.randint(2, 10)))
        unit_price = it.base_unit_price
        total_amt = (qty * unit_price).quantize(Decimal("0.01"))

        line_items.append(
            LineItem(
                item_id=it.item_id,
                description=it.description,
                quantity=qty,
                unit_price=unit_price,
                total_amount=total_amt,
            )
        )
        receipt_items.append(
            ReceiptItem(
                item_id=it.item_id,
                quantity_received=qty,
                quantity_rejected=Decimal("0"),
                condition="GOOD",
            )
        )

    total_amount = sum((it.total_amount for it in line_items), Decimal("0"))

    po = PurchaseOrder(
        po_id=po_id,
        contract_id=contract_id,
        vendor_id=vendor_template.vendor_id,
        vendor_name=vendor_template.name,
        issue_date=po_date,
        expected_delivery_date=expected_delivery_date,
        line_items=line_items,
        total_amount=total_amount,
        currency="USD",
    )

    receipt = GoodsReceipt(
        receipt_id=receipt_id,
        po_id=po_id,
        delivery_date=delivery_date,
        received_by="Warehouse Operations Staff",
        items=receipt_items,
    )

    invoice = Invoice(
        invoice_id=invoice_id,
        po_id=po_id,
        vendor_id=vendor_template.vendor_id,
        vendor_name=vendor_template.name,
        invoice_date=invoice_date,
        payment_due_date=payment_due_date,
        line_items=[it.model_copy() for it in line_items],
        total_amount=total_amount,
        currency="USD",
    )

    return ProcurementPackage(
        package_id=package_id,
        contract=contract,
        po=po,
        receipt=receipt,
        invoice=invoice,
        related_invoices=[],
    )


def generate_package(
    seed: int,
    anomaly: AnomalyType,
    package_id: str,
) -> tuple[ProcurementPackage, PackageLabels]:
    """Generate a procurement package with the requested anomaly and its ground truth labels."""
    base_pkg = generate_base_package(seed=seed, package_id=package_id)
    return inject_anomaly(base_pkg=base_pkg, anomaly=anomaly)


def generate_dataset(
    output_dir: Path,
    count_per_anomaly: int = 30,
    base_seed: int = 42,
) -> list[str]:
    """Generate a balanced synthetic dataset of packages (default: 30 x 10 = 300 packages)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    pkg_index = 1

    for anomaly in AnomalyType:
        for i in range(count_per_anomaly):
            seed = base_seed + (pkg_index * 1337)
            pkg_id = f"PKG-{pkg_index:04d}"
            pkg, labels = generate_package(seed=seed, anomaly=anomaly, package_id=pkg_id)

            pkg_dir = output_dir / pkg_id
            pkg_dir.mkdir(parents=True, exist_ok=True)

            (pkg_dir / "package.json").write_text(
                pkg.model_dump_json(indent=2),
                encoding="utf-8",
            )
            (pkg_dir / "labels.json").write_text(
                labels.model_dump_json(indent=2),
                encoding="utf-8",
            )

            manifest.append(
                {
                    "package_id": pkg_id,
                    "anomaly": anomaly.value,
                    "expected_decision": labels.expected_decision.value,
                    "vendor": pkg.contract.vendor_name,
                    "total_amount": float(pkg.invoice.total_amount),
                    "dir": str(pkg_dir.name),
                }
            )
            pkg_index += 1

    (output_dir / "index.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return [m["package_id"] for m in manifest]
