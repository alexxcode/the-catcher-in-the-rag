from __future__ import annotations

import json
from pathlib import Path

from catcher_in_the_rag.schemas import AnomalyType, DecisionType
from datagen.generator import generate_base_package, generate_dataset, generate_package


def test_generate_base_package_is_compliant() -> None:
    pkg = generate_base_package(seed=123, package_id="PKG-TEST")
    assert pkg.package_id == "PKG-TEST"
    assert pkg.contract.vendor_name == pkg.po.vendor_name == pkg.invoice.vendor_name
    assert pkg.po.po_id == pkg.receipt.po_id == pkg.invoice.po_id
    assert pkg.po.total_amount == pkg.invoice.total_amount
    assert len(pkg.po.line_items) == len(pkg.invoice.line_items)


def test_generate_all_anomaly_types() -> None:
    expected_decisions = {
        AnomalyType.A01: DecisionType.REJECT,
        AnomalyType.A02: DecisionType.REJECT,
        AnomalyType.A03: DecisionType.REJECT,
        AnomalyType.A04: DecisionType.REVIEW,
        AnomalyType.A05: DecisionType.REVIEW,
        AnomalyType.A06: DecisionType.REJECT,
        AnomalyType.A07: DecisionType.REJECT,
        AnomalyType.A08: DecisionType.REVIEW,
        AnomalyType.A09: DecisionType.ABSTAIN,
        AnomalyType.A10: DecisionType.APPROVE,
    }

    for anomaly, exp_decision in expected_decisions.items():
        pkg, labels = generate_package(seed=42, anomaly=anomaly, package_id=f"PKG-{anomaly.value}")
        assert labels.anomalies == [anomaly]
        assert labels.expected_decision == exp_decision

        # For anomalies requiring citations, verify citations exist verbatim in contract
        for cit in labels.expected_clause_spans:
            assert cit.text in pkg.contract.raw_text
            extracted = pkg.contract.raw_text[cit.start_char : cit.end_char]
            assert extracted == cit.text, (
                f"Offset mismatch for {anomaly}: '{extracted}' != '{cit.text}'"
            )


def test_generator_determinism() -> None:
    pkg1, labels1 = generate_package(seed=999, anomaly=AnomalyType.A01, package_id="PKG-D1")
    pkg2, labels2 = generate_package(seed=999, anomaly=AnomalyType.A01, package_id="PKG-D1")

    assert pkg1.model_dump_json() == pkg2.model_dump_json()
    assert labels1.model_dump_json() == labels2.model_dump_json()


def test_generate_mini_dataset(tmp_path: Path) -> None:
    pkg_ids = generate_dataset(output_dir=tmp_path, count_per_anomaly=1, base_seed=100)
    assert len(pkg_ids) == 10  # 1 per each of the 10 anomaly types

    index_file = tmp_path / "index.json"
    assert index_file.exists()
    index_data = json.loads(index_file.read_text(encoding="utf-8"))
    assert len(index_data) == 10

    # Verify first package directory
    first_dir = tmp_path / pkg_ids[0]
    assert (first_dir / "package.json").exists()
    assert (first_dir / "labels.json").exists()
