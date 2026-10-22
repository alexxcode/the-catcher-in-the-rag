from __future__ import annotations

from catcher_in_the_rag.match.matcher import run_three_way_match
from catcher_in_the_rag.schemas import AnomalyType
from datagen.generator import generate_package


def test_match_clean_package() -> None:
    pkg, labels = generate_package(seed=100, anomaly=AnomalyType.A10, package_id="PKG-CLEAN")
    findings = run_three_way_match(pkg)
    assert len(findings) == 0


def test_match_detects_a01_price_violation() -> None:
    pkg, labels = generate_package(seed=101, anomaly=AnomalyType.A01, package_id="PKG-A01")
    findings = run_three_way_match(pkg)
    assert len(findings) >= 1
    a01 = next(f for f in findings if f.anomaly_type == AnomalyType.A01)
    assert a01.calculated_difference is not None and a01.calculated_difference > 0
    assert a01.citation is not None
    assert a01.citation.text in pkg.contract.raw_text


def test_match_detects_a02_quantity_mismatch() -> None:
    pkg, labels = generate_package(seed=102, anomaly=AnomalyType.A02, package_id="PKG-A02")
    findings = run_three_way_match(pkg)
    assert len(findings) >= 1
    a02 = next(f for f in findings if f.anomaly_type == AnomalyType.A02)
    assert a02.calculated_difference is not None and a02.calculated_difference > 0
    assert "billed quantity" in a02.description


def test_match_detects_a03_duplicate_invoice() -> None:
    pkg, labels = generate_package(seed=103, anomaly=AnomalyType.A03, package_id="PKG-A03")
    findings = run_three_way_match(pkg)
    assert len(findings) >= 1
    a03 = next(f for f in findings if f.anomaly_type == AnomalyType.A03)
    assert "Duplicate invoice detected" in a03.description


def test_match_detects_a07_unknown_po() -> None:
    pkg, labels = generate_package(seed=104, anomaly=AnomalyType.A07, package_id="PKG-A07")
    findings = run_three_way_match(pkg)
    assert len(findings) >= 1
    a07 = next(f for f in findings if f.anomaly_type == AnomalyType.A07)
    assert a07.field == "po_id"
