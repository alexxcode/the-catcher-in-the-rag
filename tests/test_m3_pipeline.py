from __future__ import annotations

from catcher_in_the_rag.graph import run_audit
from catcher_in_the_rag.schemas import AnomalyType, DecisionType
from datagen.generator import generate_package


def test_m3_pipeline_detects_a04_payment_terms() -> None:
    pkg, labels = generate_package(seed=401, anomaly=AnomalyType.A04, package_id="PKG-M3-A04")
    decision = run_audit(pkg)
    assert decision.decision == DecisionType.REVIEW
    assert decision.requires_human_review
    a04 = next(f for f in decision.findings if f.anomaly_type == AnomalyType.A04)
    assert a04.citation is not None
    assert a04.citation.text in pkg.contract.raw_text


def test_m3_pipeline_detects_a05_late_delivery_penalty() -> None:
    pkg, labels = generate_package(seed=402, anomaly=AnomalyType.A05, package_id="PKG-M3-A05")
    decision = run_audit(pkg)
    assert decision.decision == DecisionType.REVIEW
    a05 = next(f for f in decision.findings if f.anomaly_type == AnomalyType.A05)
    assert a05.citation is not None
    assert a05.calculated_difference is not None and a05.calculated_difference > 0
    assert a05.citation.text in pkg.contract.raw_text


def test_m3_pipeline_detects_a06_expired_contract() -> None:
    pkg, labels = generate_package(seed=403, anomaly=AnomalyType.A06, package_id="PKG-M3-A06")
    decision = run_audit(pkg)
    assert decision.decision == DecisionType.REJECT
    a06 = next(f for f in decision.findings if f.anomaly_type == AnomalyType.A06)
    assert a06.citation is not None
    assert a06.citation.text in pkg.contract.raw_text


def test_m3_pipeline_abstains_on_a09_unsupported_claim() -> None:
    pkg, labels = generate_package(seed=404, anomaly=AnomalyType.A09, package_id="PKG-M3-A09")
    decision = run_audit(pkg)
    assert decision.decision == DecisionType.ABSTAIN
    assert decision.bartleby_abstained
    assert decision.bartleby_reason is not None
    assert "NO_EVIDENCE" in decision.bartleby_reason
    assert any(f.anomaly_type == AnomalyType.A09 for f in decision.findings)
