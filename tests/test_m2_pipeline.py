from __future__ import annotations

from catcher_in_the_rag.graph import run_audit
from catcher_in_the_rag.schemas import AnomalyType, DecisionType
from datagen.generator import generate_package


def test_m2_pipeline_detects_a01() -> None:
    pkg, labels = generate_package(seed=201, anomaly=AnomalyType.A01, package_id="PKG-M2-A01")
    decision = run_audit(pkg)
    assert decision.decision == DecisionType.REJECT
    assert any(f.anomaly_type == AnomalyType.A01 for f in decision.findings)


def test_m2_pipeline_detects_a02() -> None:
    pkg, labels = generate_package(seed=202, anomaly=AnomalyType.A02, package_id="PKG-M2-A02")
    decision = run_audit(pkg)
    assert decision.decision == DecisionType.REJECT
    assert any(f.anomaly_type == AnomalyType.A02 for f in decision.findings)


def test_m2_pipeline_detects_a03() -> None:
    pkg, labels = generate_package(seed=203, anomaly=AnomalyType.A03, package_id="PKG-M2-A03")
    decision = run_audit(pkg)
    assert decision.decision == DecisionType.REJECT
    assert any(f.anomaly_type == AnomalyType.A03 for f in decision.findings)


def test_m2_pipeline_detects_a07() -> None:
    pkg, labels = generate_package(seed=204, anomaly=AnomalyType.A07, package_id="PKG-M2-A07")
    decision = run_audit(pkg)
    assert decision.decision == DecisionType.REJECT
    assert any(f.anomaly_type == AnomalyType.A07 for f in decision.findings)


def test_m2_pipeline_detects_a08() -> None:
    pkg, labels = generate_package(seed=205, anomaly=AnomalyType.A08, package_id="PKG-M2-A08")
    decision = run_audit(pkg)
    assert decision.decision == DecisionType.REVIEW
    assert decision.requires_human_review
    assert any(f.anomaly_type == AnomalyType.A08 for f in decision.findings)


def test_m2_pipeline_approves_clean_a10() -> None:
    pkg, labels = generate_package(seed=206, anomaly=AnomalyType.A10, package_id="PKG-M2-A10")
    decision = run_audit(pkg)
    assert decision.decision == DecisionType.APPROVE
    assert not decision.requires_human_review
    assert len(decision.findings) == 0
