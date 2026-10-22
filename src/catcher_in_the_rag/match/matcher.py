from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

from catcher_in_the_rag.schemas import (
    AnomalyType,
    Citation,
    Finding,
    ProcurementPackage,
)


def load_matching_thresholds() -> dict[str, Any]:
    """Load matching tolerances from config/thresholds.yaml."""
    cfg_file = Path("config/thresholds.yaml")
    if not cfg_file.exists():
        for parent in Path.cwd().parents:
            candidate = parent / "config" / "thresholds.yaml"
            if candidate.exists():
                cfg_file = candidate
                break
    if cfg_file.exists():
        with cfg_file.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            return data.get("matching", {})
    return {}


def run_three_way_match(
    package: ProcurementPackage,
    price_tolerance_pct: Decimal = Decimal("0.01"),
    quantity_tolerance_pct: Decimal = Decimal("0.0"),
) -> list[Finding]:
    """Perform deterministic three-way matching across Invoice, PO, GoodsReceipt, and Contract."""
    findings: list[Finding] = []
    invoice = package.invoice
    po = package.po
    receipt = package.receipt
    contract = package.contract

    # 1. Vendor and PO Linking check (A07)
    if invoice.po_id != po.po_id:
        findings.append(
            Finding(
                finding_id=f"F-A07-{len(findings) + 1}",
                anomaly_type=AnomalyType.A07,
                field="po_id",
                description=(
                    f"Invoice references invalid/unmatched PO number '{invoice.po_id}' "
                    f"(expected '{po.po_id}')"
                ),
                severity="ERROR",
            )
        )

    if invoice.vendor_id != po.vendor_id:
        findings.append(
            Finding(
                finding_id=f"F-A07-{len(findings) + 1}",
                anomaly_type=AnomalyType.A07,
                field="vendor_id",
                description=(
                    f"Invoice vendor ID '{invoice.vendor_id}' does not match "
                    f"authorized PO vendor '{po.vendor_id}'"
                ),
                severity="ERROR",
            )
        )

    # 2. Duplicate Invoice check (A03)
    for prior in package.related_invoices:
        # Check matching ID or same invoice number prefix/duplicate tag
        is_same_id = (
            prior.invoice_id == invoice.invoice_id
            or prior.invoice_id.startswith(invoice.invoice_id)
            or invoice.invoice_id.startswith(prior.invoice_id)
        )
        is_same_po_and_amount = (
            prior.po_id == invoice.po_id
            and prior.total_amount == invoice.total_amount
            and prior.vendor_id == invoice.vendor_id
        )
        if is_same_id or is_same_po_and_amount:
            findings.append(
                Finding(
                    finding_id=f"F-A03-{len(findings) + 1}",
                    anomaly_type=AnomalyType.A03,
                    field="invoice_id",
                    description=(
                        f"Duplicate invoice detected: matches prior invoice '{prior.invoice_id}' "
                        f"for PO '{prior.po_id}' with amount ${prior.total_amount:.2f}"
                    ),
                    severity="ERROR",
                )
            )
            break

    # 3. Quantity matching (A02: Quantity billed > received)
    receipt_qty_map: dict[str, Decimal] = {}
    for item in receipt.items:
        receipt_qty_map[item.item_id] = (
            receipt_qty_map.get(item.item_id, Decimal("0")) + item.quantity_received
        )

    for inv_item in invoice.line_items:
        qty_received = receipt_qty_map.get(inv_item.item_id)
        if qty_received is None:
            # Item was not received at all
            findings.append(
                Finding(
                    finding_id=f"F-A02-{len(findings) + 1}",
                    anomaly_type=AnomalyType.A02,
                    field=f"line_items.{inv_item.item_id}.quantity",
                    description=(
                        f"Item '{inv_item.item_id}' billed quantity {inv_item.quantity} "
                        "has no corresponding delivery on Goods Receipt"
                    ),
                    severity="ERROR",
                    calculated_difference=inv_item.total_amount,
                )
            )
        elif inv_item.quantity > (qty_received * (Decimal("1.0") + quantity_tolerance_pct)):
            delta_qty = inv_item.quantity - qty_received
            diff_amount = (delta_qty * inv_item.unit_price).quantize(Decimal("0.01"))
            findings.append(
                Finding(
                    finding_id=f"F-A02-{len(findings) + 1}",
                    anomaly_type=AnomalyType.A02,
                    field=f"line_items.{inv_item.item_id}.quantity",
                    description=(
                        f"Line item '{inv_item.item_id}' billed quantity ({inv_item.quantity}) "
                        f"exceeds confirmed received quantity ({qty_received})"
                    ),
                    severity="ERROR",
                    calculated_difference=diff_amount,
                )
            )

    # 4. Pricing matching (A01: Unit price above contract rate)
    rate_map: dict[str, Decimal] = {r.item_id: r.max_unit_price for r in contract.rate_sheet}
    po_rate_map: dict[str, Decimal] = {p.item_id: p.unit_price for p in po.line_items}

    for inv_item in invoice.line_items:
        contract_rate = rate_map.get(inv_item.item_id) or po_rate_map.get(inv_item.item_id)
        if contract_rate is not None:
            max_allowed = (contract_rate * (Decimal("1.0") + price_tolerance_pct)).quantize(
                Decimal("0.01")
            )
            if inv_item.unit_price > max_allowed:
                unit_diff = inv_item.unit_price - contract_rate
                total_diff = (unit_diff * inv_item.quantity).quantize(Decimal("0.01"))

                # Try to locate citation in contract
                citation: Citation | None = None
                quote = (
                    f"{inv_item.item_id} - {inv_item.description}: "
                    f"Maximum Unit Rate of ${contract_rate:.2f} USD"
                )
                span = contract.find_span(quote)
                if span:
                    citation = Citation(
                        document_id=contract.contract_id,
                        start_char=span[0],
                        end_char=span[1],
                        text=quote,
                    )

                findings.append(
                    Finding(
                        finding_id=f"F-A01-{len(findings) + 1}",
                        anomaly_type=AnomalyType.A01,
                        field=f"line_items.{inv_item.item_id}.unit_price",
                        description=(
                            f"Unit price ${inv_item.unit_price:.2f} for '{inv_item.item_id}' "
                            f"exceeds contract ceiling ${contract_rate:.2f}"
                        ),
                        severity="ERROR",
                        citation=citation,
                        calculated_difference=total_diff,
                    )
                )

    return findings
