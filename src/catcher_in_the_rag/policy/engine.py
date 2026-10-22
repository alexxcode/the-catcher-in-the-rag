from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

from catcher_in_the_rag.schemas import (
    AnomalyType,
    Decision,
    DecisionType,
    Finding,
)


def load_policy_config() -> dict[str, Any]:
    """Load policy rules from config/policy.yaml."""
    cfg_file = Path("config/policy.yaml")
    if not cfg_file.exists():
        for parent in Path.cwd().parents:
            candidate = parent / "config" / "policy.yaml"
            if candidate.exists():
                cfg_file = candidate
                break
    if cfg_file.exists():
        with cfg_file.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


class PolicyEngine:
    """Deterministic policy engine enforcing approval thresholds and multi-finding resolution."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or load_policy_config()
        approval_cfg = self.config.get("approval", {})
        self.auto_approve_max = Decimal(str(approval_cfg.get("auto_approve_max_usd", 5000.0)))
        self.requires_review_max = Decimal(
            str(approval_cfg.get("requires_review_max_usd", 50000.0))
        )
        self.auto_reject_above = Decimal(str(approval_cfg.get("auto_reject_above_usd", 250000.0)))
        self.anomaly_map = self.config.get(
            "anomaly_decisions",
            {
                "A01": "reject",
                "A02": "reject",
                "A03": "reject",
                "A04": "review",
                "A05": "review",
                "A06": "review",
                "A07": "reject",
                "A08": "review",
                "A09": "review",
                "A10": "approve",
            },
        )

    def evaluate(
        self,
        package_id: str,
        invoice_total: Decimal,
        findings: list[Finding],
        bartleby_abstained: bool = False,
        bartleby_reason: str | None = None,
    ) -> Decision:
        """Evaluate invoice total and findings against policy rules to produce a deterministic decision."""
        all_findings = list(findings)

        # 1. Evaluate hard escalation reject threshold
        if invoice_total > self.auto_reject_above and not any(
            f.anomaly_type == AnomalyType.A08 and f.severity == "ERROR" for f in all_findings
        ):
            all_findings.append(
                Finding(
                    finding_id=f"F-A08-REJECT-{len(all_findings) + 1}",
                    anomaly_type=AnomalyType.A08,
                    field="total_amount",
                    description=(
                        f"Invoice amount ${invoice_total:.2f} exceeds hard policy limit "
                        f"${self.auto_reject_above:.2f}. Auto-reject policy triggered."
                    ),
                    severity="ERROR",
                    calculated_difference=invoice_total - self.auto_reject_above,
                )
            )
        # 2. Evaluate approval review threshold (A08)
        elif invoice_total > self.requires_review_max and not any(
            f.anomaly_type == AnomalyType.A08 for f in all_findings
        ):
            all_findings.append(
                Finding(
                    finding_id=f"F-A08-{len(all_findings) + 1}",
                    anomaly_type=AnomalyType.A08,
                    field="total_amount",
                    description=(
                        f"Invoice amount ${invoice_total:.2f} exceeds "
                        f"standard approval limit ${self.requires_review_max:.2f}. "
                        "Requires executive approval."
                    ),
                    severity="WARNING",
                    calculated_difference=invoice_total - self.requires_review_max,
                )
            )

        # 3. Check for Bartleby abstention (A09)
        has_a09 = any(f.anomaly_type == AnomalyType.A09 for f in all_findings)
        if bartleby_abstained or has_a09:
            reason = bartleby_reason or "Evidence absent in contract terms"
            return Decision(
                package_id=package_id,
                decision=DecisionType.ABSTAIN,
                findings=all_findings,
                requires_human_review=True,
                justification=f"Abstained: {reason}",
                bartleby_abstained=True,
                bartleby_reason=reason,
            )

        # 4. Multi-finding resolution
        # Priority: REJECT > REVIEW > APPROVE
        has_reject = False
        has_review = False

        for f in all_findings:
            rule = self.anomaly_map.get(f.anomaly_type.value, "review").lower()
            if rule == "reject" or f.severity == "ERROR":
                has_reject = True
            elif rule == "review" or f.severity == "WARNING":
                has_review = True

        if has_reject:
            decision_type = DecisionType.REJECT
            justification = (
                f"Rejected: {len(all_findings)} finding(s) detected with hard policy violations "
                f"({', '.join(f.anomaly_type.value for f in all_findings)})."
            )
            requires_human_review = False
        elif has_review:
            decision_type = DecisionType.REVIEW
            justification = (
                f"Requires Human Review: {len(all_findings)} finding(s) require review "
                f"({', '.join(f.anomaly_type.value for f in all_findings)})."
            )
            requires_human_review = True
        else:
            decision_type = DecisionType.APPROVE
            if invoice_total <= self.auto_approve_max:
                justification = (
                    f"Auto-Approved: Package passed all checks and total ${invoice_total:.2f} "
                    f"is within auto-approval threshold (${self.auto_approve_max:.2f})."
                )
            else:
                justification = (
                    f"Approved: Package passed three-way match and contract terms. "
                    f"Total ${invoice_total:.2f} within authorized limit."
                )
            requires_human_review = False

        return Decision(
            package_id=package_id,
            decision=decision_type,
            findings=all_findings,
            requires_human_review=requires_human_review,
            justification=justification,
        )
