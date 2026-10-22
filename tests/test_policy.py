from __future__ import annotations

from decimal import Decimal

from catcher_in_the_rag.policy.engine import PolicyEngine
from catcher_in_the_rag.schemas import (
    AnomalyType,
    DecisionType,
    Finding,
)


def test_policy_auto_approve_clean() -> None:
    engine = PolicyEngine()
    decision = engine.evaluate(
        package_id="PKG-001",
        invoice_total=Decimal("1500.00"),
        findings=[],
    )
    assert decision.decision == DecisionType.APPROVE
    assert not decision.requires_human_review
    assert len(decision.findings) == 0


def test_policy_detects_a08_threshold_exceeded() -> None:
    engine = PolicyEngine()
    decision = engine.evaluate(
        package_id="PKG-002",
        invoice_total=Decimal("75000.00"),
        findings=[],
    )
    assert decision.decision == DecisionType.REVIEW
    assert decision.requires_human_review
    assert any(f.anomaly_type == AnomalyType.A08 for f in decision.findings)


def test_policy_reject_priority_over_review() -> None:
    engine = PolicyEngine()
    # Pass an A08 finding (review) and an A01 finding (reject)
    f_a01 = Finding(
        finding_id="F-1",
        anomaly_type=AnomalyType.A01,
        field="unit_price",
        description="Price ceiling exceeded",
        severity="ERROR",
    )
    decision = engine.evaluate(
        package_id="PKG-003",
        invoice_total=Decimal("80000.00"),  # also triggers A08
        findings=[f_a01],
    )
    # The hardest violation (reject) takes precedence
    assert decision.decision == DecisionType.REJECT
    assert any(f.anomaly_type == AnomalyType.A01 for f in decision.findings)
    assert any(f.anomaly_type == AnomalyType.A08 for f in decision.findings)
