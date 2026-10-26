from __future__ import annotations

from decimal import Decimal
from typing import Any

from catcher_in_the_rag.bartleby.guard import BartlebyGuard
from catcher_in_the_rag.retrieve.retriever import ContractRetriever
from catcher_in_the_rag.schemas import (
    AnomalyType,
    Citation,
    Finding,
    ProcurementPackage,
)


class ContractTermsEvaluator:
    """Evaluates invoice and delivery against retrieved contract clauses."""

    def __init__(
        self, bartleby_guard: BartlebyGuard | None = None, llm_provider: Any = None
    ) -> None:
        self.bartleby = bartleby_guard or BartlebyGuard()
        self.llm_provider = llm_provider

    @staticmethod
    def _not_verifiable(
        finding_id: str, anomaly: AnomalyType, field: str, description: str, reason: str | None
    ) -> Finding:
        """Finding kept for human review when Bartleby abstains on its clause."""
        return Finding(
            finding_id=finding_id,
            anomaly_type=anomaly,
            field=field,
            description=f"{description} NOT_VERIFIABLE: {reason}",
            severity="WARNING",
            citation=None,
            verification_status="NOT_VERIFIABLE",
        )

    def evaluate(self, package: ProcurementPackage) -> tuple[list[Finding], bool, str | None]:
        """Apply contract clauses to package.

        Returns: (findings, bartleby_abstained, bartleby_reason)
        """
        findings: list[Finding] = []
        contract = package.contract
        po = package.po
        receipt = package.receipt
        invoice = package.invoice

        retriever = ContractRetriever(contract)

        # ── 1. Term and Expiration Check (A06) ──────────────────────────────────
        if contract.expiration_date is None:
            findings.append(
                self._not_verifiable(
                    f"F-A06-{len(findings) + 1}",
                    AnomalyType.A06,
                    "expiration_date",
                    "Contract expiration date not found; term validity cannot be checked.",
                    "expiration date missing from contract text",
                )
            )
        elif invoice.invoice_date > contract.expiration_date and not contract.auto_renew:
            scored = retriever.retrieve("term expiration termination date agreement", top_k=2)
            assessment = self.bartleby.evaluate_retrieval(
                scored, query_topic="term_and_termination", llm_provider=self.llm_provider
            )
            description = (
                f"Invoice date {invoice.invoice_date.isoformat()} is subsequent to "
                f"contract expiration date {contract.expiration_date.isoformat()} "
                "without automatic renewal."
            )
            finding_id = f"F-A06-{len(findings) + 1}"
            if assessment.abstained:
                findings.append(
                    self._not_verifiable(
                        finding_id,
                        AnomalyType.A06,
                        "expiration_date",
                        description,
                        assessment.reason,
                    )
                )
            citation: Citation | None = None
            quote = f'until {contract.expiration_date.isoformat()} ("Expiration Date")'
            span = contract.find_span(quote)
            if span:
                citation = Citation(
                    document_id=contract.contract_id,
                    start_char=span[0],
                    end_char=span[1],
                    text=quote,
                )

            if not assessment.abstained:
                findings.append(
                    Finding(
                        finding_id=finding_id,
                        anomaly_type=AnomalyType.A06,
                        field="expiration_date",
                        description=description,
                        severity="ERROR",
                        citation=citation,
                    )
                )

        # ── 2. Payment Terms Check (A04) ────────────────────────────────────────
        actual_days = (invoice.payment_due_date - invoice.invoice_date).days
        if actual_days < contract.payment_terms_days:
            scored = retriever.retrieve("payment terms net days invoice receipt", top_k=2)
            assessment = self.bartleby.evaluate_retrieval(
                scored, query_topic="payment_terms", llm_provider=self.llm_provider
            )
            description = (
                f"Invoice payment due date demands Net {actual_days} days, "
                f"which violates agreed contract terms of Net {contract.payment_terms_days} days."
            )
            finding_id = f"F-A04-{len(findings) + 1}"
            if assessment.abstained:
                findings.append(
                    self._not_verifiable(
                        finding_id,
                        AnomalyType.A04,
                        "payment_due_date",
                        description,
                        assessment.reason,
                    )
                )
            quote = f"payable Net {contract.payment_terms_days} days"
            span = contract.find_span(quote)
            citation = None
            if span:
                citation = Citation(
                    document_id=contract.contract_id,
                    start_char=span[0],
                    end_char=span[1],
                    text=quote,
                )

            if not assessment.abstained:
                findings.append(
                    Finding(
                        finding_id=finding_id,
                        anomaly_type=AnomalyType.A04,
                        field="payment_due_date",
                        description=description,
                        severity="WARNING",
                        citation=citation,
                    )
                )

        # ── 3. Late Delivery Penalty Check (A05) ────────────────────────────────
        if receipt.delivery_date > po.expected_delivery_date:
            delay_days = (receipt.delivery_date - po.expected_delivery_date).days
            if contract.late_delivery_penalty_pct_per_day > 0:
                scored = retriever.retrieve(
                    "liquidated penalty delay late delivery percentage", top_k=2
                )
                assessment = self.bartleby.evaluate_retrieval(
                    scored, query_topic="penalties", llm_provider=self.llm_provider
                )
                penalty_rate = contract.late_delivery_penalty_pct_per_day
                rate_decimal = penalty_rate / Decimal("100.0")
                penalty_amt = (Decimal(str(delay_days)) * rate_decimal * po.total_amount).quantize(
                    Decimal("0.01")
                )

                description = (
                    f"Delivery delayed by {delay_days} day(s) beyond scheduled date. "
                    f"Applicable contract liquidated penalty is ${penalty_amt:.2f} "
                    f"({penalty_rate:.1f}% per day)."
                )
                finding_id = f"F-A05-{len(findings) + 1}"
                if assessment.abstained:
                    findings.append(
                        self._not_verifiable(
                            finding_id,
                            AnomalyType.A05,
                            "delivery_date",
                            description,
                            assessment.reason,
                        )
                    )
                quote = f"liquidated penalty of {penalty_rate:.1f}% per calendar day of delay"
                span = contract.find_span(quote)
                citation = None
                if span:
                    citation = Citation(
                        document_id=contract.contract_id,
                        start_char=span[0],
                        end_char=span[1],
                        text=quote,
                    )

                if not assessment.abstained:
                    findings.append(
                        Finding(
                            finding_id=finding_id,
                            anomaly_type=AnomalyType.A05,
                            field="delivery_date",
                            description=description,
                            severity="WARNING",
                            citation=citation,
                            calculated_difference=penalty_amt,
                        )
                    )

        # ── 4. Uncontracted Surcharges / Missing Clause Check (A09) ──────────────
        # Inspect if any invoice item claims a special surcharge or clause authorization
        abstain_reasons: list[str] = []
        po_item_ids = {it.item_id for it in po.line_items}
        for inv_item in invoice.line_items:
            if inv_item.item_id not in po_item_ids:
                # Query contract for clause supporting this extra item/fee
                scored = retriever.retrieve(inv_item.description, top_k=3)
                assessment = self.bartleby.evaluate_retrieval(
                    scored, query_topic=inv_item.description, llm_provider=self.llm_provider
                )
                if assessment.abstained:
                    # Bartleby invokes "I would prefer not to": no clause authorizes this fee
                    findings.append(
                        Finding(
                            finding_id=f"F-A09-{len(findings) + 1}",
                            anomaly_type=AnomalyType.A09,
                            field=f"line_items.{inv_item.item_id}",
                            description=(
                                f"Invoice charges '{inv_item.description}' without contractual authorization. "
                                f"{assessment.reason}"
                            ),
                            severity="WARNING",
                            citation=None,
                            calculated_difference=inv_item.total_amount,
                        )
                    )
                    abstain_reasons.append(assessment.reason or "")

        if abstain_reasons:
            return findings, True, " | ".join(abstain_reasons)
        return findings, False, None
