from __future__ import annotations

from copy import deepcopy
from datetime import timedelta
from decimal import Decimal
from typing import Any

from catcher_in_the_rag.schemas.common import Citation, LineItem
from catcher_in_the_rag.schemas.findings import AnomalyType, DecisionType
from catcher_in_the_rag.schemas.labels import PackageLabels
from catcher_in_the_rag.schemas.procurement import (
    ProcurementPackage,
    ReceiptItem,
)


def inject_anomaly(
    base_pkg: ProcurementPackage,
    anomaly: AnomalyType,
) -> tuple[ProcurementPackage, PackageLabels]:
    """Inject a specific anomaly into a clean base procurement package and generate ground truth labels."""
    pkg = deepcopy(base_pkg)
    contract = pkg.contract
    po = pkg.po
    receipt = pkg.receipt
    invoice = pkg.invoice
    expected_citations: list[Citation] = []
    metadata: dict[str, Any] = {"anomaly": anomaly.value}

    match anomaly:
        case AnomalyType.A01:
            # A01: Unit price above contract rate
            target_item = invoice.line_items[0]
            contract_rate = next(
                r.max_unit_price for r in contract.rate_sheet if r.item_id == target_item.item_id
            )
            inflated_price = (contract_rate * Decimal("1.25")).quantize(Decimal("0.01"))
            new_item = LineItem(
                item_id=target_item.item_id,
                description=target_item.description,
                quantity=target_item.quantity,
                unit_price=inflated_price,
                total_amount=(target_item.quantity * inflated_price).quantize(Decimal("0.01")),
            )
            invoice.line_items[0] = new_item
            invoice.total_amount = sum((it.total_amount for it in invoice.line_items), Decimal("0"))

            pricing_clause = next(c for c in contract.clauses if c.clause_type == "pricing")
            quote = f"{target_item.item_id} - {target_item.description}: Maximum Unit Rate of ${contract_rate:.2f} USD"
            span = contract.find_span(quote)
            if span:
                expected_citations.append(
                    Citation(
                        document_id=contract.contract_id,
                        start_char=span[0],
                        end_char=span[1],
                        text=quote,
                    )
                )
            metadata["contract_rate"] = float(contract_rate)
            metadata["invoiced_rate"] = float(inflated_price)
            expected_decision = DecisionType.REJECT

        case AnomalyType.A02:
            # A02: Quantity billed > received
            # Reduce accepted receipt quantity so billed > received
            target_item = invoice.line_items[0]
            received_qty = (target_item.quantity * Decimal("0.5")).quantize(Decimal("1"))
            if received_qty == target_item.quantity:
                received_qty = max(Decimal("0"), target_item.quantity - Decimal("2"))

            receipt.items = [
                ReceiptItem(
                    item_id=target_item.item_id,
                    quantity_received=received_qty,
                    quantity_rejected=target_item.quantity - received_qty,
                    condition="DAMAGED_PARTIAL",
                )
            ]
            metadata["billed_qty"] = float(target_item.quantity)
            metadata["received_qty"] = float(received_qty)
            expected_decision = DecisionType.REJECT

        case AnomalyType.A03:
            # A03: Duplicate invoice (already submitted or currently active)
            prior_invoice = deepcopy(invoice)
            prior_invoice.invoice_id = f"{invoice.invoice_id}-DUP"
            pkg.related_invoices.append(prior_invoice)
            metadata["duplicate_invoice_id"] = prior_invoice.invoice_id
            expected_decision = DecisionType.REJECT

        case AnomalyType.A04:
            # A04: Payment terms differ from contract (e.g. Net 15 instead of Net 30)
            contract_days = contract.payment_terms_days
            shorter_days = max(10, contract_days // 2)
            invoice.payment_due_date = invoice.invoice_date + timedelta(days=shorter_days)

            terms_clause = next(c for c in contract.clauses if c.clause_type == "payment_terms")
            quote = f"payable Net {contract_days} days"
            span = contract.find_span(quote)
            if span:
                expected_citations.append(
                    Citation(
                        document_id=contract.contract_id,
                        start_char=span[0],
                        end_char=span[1],
                        text=quote,
                    )
                )
            metadata["contract_terms_days"] = contract_days
            metadata["invoice_terms_days"] = shorter_days
            expected_decision = DecisionType.REVIEW

        case AnomalyType.A05:
            # A05: Late-delivery penalty applicable
            delay_days = 7
            receipt.delivery_date = po.expected_delivery_date + timedelta(days=delay_days)
            penalty_clause = next(c for c in contract.clauses if c.clause_type == "penalties")
            quote = f"liquidated penalty of {contract.late_delivery_penalty_pct_per_day:.1f}% per calendar day of delay"
            span = contract.find_span(quote)
            if span:
                expected_citations.append(
                    Citation(
                        document_id=contract.contract_id,
                        start_char=span[0],
                        end_char=span[1],
                        text=quote,
                    )
                )
            metadata["delay_days"] = delay_days
            metadata["penalty_rate"] = float(contract.late_delivery_penalty_pct_per_day)
            expected_decision = DecisionType.REVIEW

        case AnomalyType.A06:
            # A06: Contract expired / auto-renewal triggered
            # Shift invoice date to 30 days after expiration date
            invoice.invoice_date = contract.expiration_date + timedelta(days=30)
            invoice.payment_due_date = invoice.invoice_date + timedelta(
                days=contract.payment_terms_days
            )
            term_clause = next(
                c for c in contract.clauses if c.clause_type == "term_and_termination"
            )
            quote = f'until {contract.expiration_date.isoformat()} ("Expiration Date")'
            span = contract.find_span(quote)
            if span:
                expected_citations.append(
                    Citation(
                        document_id=contract.contract_id,
                        start_char=span[0],
                        end_char=span[1],
                        text=quote,
                    )
                )
            metadata["contract_expiration"] = contract.expiration_date.isoformat()
            metadata["invoice_date"] = invoice.invoice_date.isoformat()
            expected_decision = DecisionType.REJECT

        case AnomalyType.A07:
            # A07: Unknown vendor or no PO
            invoice.po_id = "PO-NONEXISTENT-999"
            metadata["mismatched_po_id"] = invoice.po_id
            expected_decision = DecisionType.REJECT

        case AnomalyType.A08:
            # A08: Amount above approval threshold (e.g. $50,000 policy threshold)
            large_qty = Decimal("50")
            item = invoice.line_items[0]
            expensive_item = LineItem(
                item_id=item.item_id,
                description=item.description,
                quantity=large_qty,
                unit_price=item.unit_price,
                total_amount=(large_qty * item.unit_price).quantize(Decimal("0.01")),
            )
            invoice.line_items = [expensive_item]
            invoice.total_amount = expensive_item.total_amount

            # Also update PO and receipt so 3-way match passes, but amount exceeds policy threshold
            po.line_items = [expensive_item]
            po.total_amount = expensive_item.total_amount
            receipt.items = [
                ReceiptItem(
                    item_id=expensive_item.item_id,
                    quantity_received=large_qty,
                    quantity_rejected=Decimal("0"),
                    condition="GOOD",
                )
            ]
            metadata["threshold_exceeded"] = float(invoice.total_amount)
            expected_decision = DecisionType.REVIEW

        case AnomalyType.A09:
            # A09: Relevant clause absent -> must abstain (bartleby negative control)
            # Add an extra surcharge line item claiming special contractual authorization
            surcharge = LineItem(
                item_id="FEE-URGENT",
                description="Expedited Rush Surcharge pursuant to Clause 8.2",
                quantity=Decimal("1"),
                unit_price=Decimal("1500.00"),
                total_amount=Decimal("1500.00"),
            )
            invoice.line_items.append(surcharge)
            invoice.total_amount += surcharge.total_amount
            metadata["unsupported_claim"] = surcharge.description
            expected_decision = DecisionType.ABSTAIN

        case AnomalyType.A10:
            # A10: Clean package (false-positive control)
            # No changes, perfect match
            expected_decision = DecisionType.APPROVE

    labels = PackageLabels(
        package_id=pkg.package_id,
        anomalies=[anomaly],
        expected_decision=expected_decision,
        expected_clause_spans=expected_citations,
        metadata=metadata,
    )

    return pkg, labels
