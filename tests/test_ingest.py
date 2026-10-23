"""Unit tests for CUAD and OCDS data ingestion and benchmark runners."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from catcher_in_the_rag.ingest.cuad_loader import CUADLoader
from catcher_in_the_rag.ingest.ocds_loader import OCDSLoader
from evals.cuad_benchmark import CUADBenchmarkRunner

CUAD_PATH = Path("data/real/cuad/CUADv1.json")
OCDS_DIR = Path("data/real/ocds")


def test_cuad_loader_with_fixture(tmp_path: Path) -> None:
    """Test CUADLoader parsing and conversion to Contract schema with a fixture."""
    fixture_data = {
        "version": "test_v1",
        "data": [
            {
                "title": "TEST_DISTRIBUTOR_AGREEMENT",
                "paragraphs": [
                    {
                        "context": "MASTER SERVICES AGREEMENT\nBetween Acme Corp and Beta LLC.\nSection 1. Governing Law\nThis agreement is governed by the laws of Delaware.",
                        "qas": [
                            {
                                "id": "test__Governing Law",
                                "question": "Governing Law",
                                "is_impossible": False,
                                "answers": [{"text": "laws of Delaware", "answer_start": 80}],
                            },
                            {
                                "id": "test__Liquidated Damages",
                                "question": "Liquidated Damages",
                                "is_impossible": True,
                                "answers": [],
                            },
                        ],
                    }
                ],
            }
        ],
    }
    fixture_file = tmp_path / "CUAD_fixture.json"
    fixture_file.write_text(json.dumps(fixture_data), encoding="utf-8")

    loader = CUADLoader(cuad_json_path=fixture_file)
    assert loader.count() == 1
    contracts = loader.load_contracts()
    assert len(contracts) == 1
    c0 = contracts[0]
    assert c0.contract_id == "CUAD-0001"
    assert "Governing Law" in c0.annotations
    assert "Liquidated Damages" in c0.absent_categories

    catcher = loader.to_catcher_contract(c0)
    assert catcher.contract_id == "CUAD-0001"
    assert len(catcher.clauses) > 0


def test_ocds_loader_with_fixture(tmp_path: Path) -> None:
    """Test OCDSLoader parsing releases into OCDSContractRecord with a fixture."""
    ocds_dir = tmp_path / "ocds"
    ocds_dir.mkdir()
    release_data = {
        "releases": [
            {
                "ocid": "ocds-test-001",
                "id": "rel-01",
                "buyer": {"id": "BUYER-1", "name": "City Transport Agency"},
                "awards": [{"suppliers": [{"id": "SUP-1", "name": "Global Rail Corp"}]}],
                "contracts": [
                    {
                        "id": "CTR-001",
                        "title": "Rail Maintenance Contract",
                        "value": {"amount": 500000, "currency": "USD"},
                        "period": {"startDate": "2025-01-01", "endDate": "2026-01-01"},
                        "items": [
                            {
                                "id": "ITEM-1",
                                "description": "Track Maintenance",
                                "quantity": 10,
                                "unit": {"value": {"amount": 50000, "currency": "USD"}},
                            }
                        ],
                    }
                ],
            }
        ]
    }
    (ocds_dir / "release.json").write_text(json.dumps(release_data), encoding="utf-8")
    loader = OCDSLoader(ocds_dir=ocds_dir)
    records = loader.load_contracts()
    assert len(records) == 1
    assert records[0].buyer.name == "City Transport Agency"
    assert records[0].supplier is not None
    assert records[0].supplier.name == "Global Rail Corp"
    assert records[0].total_amount == 500000


@pytest.mark.skipif(
    not CUAD_PATH.exists(),
    reason="CUAD dataset not downloaded. Run `make download-data` to enable.",
)
def test_cuad_loader_live_dataset() -> None:
    """Test loading from real CUAD dataset if available locally."""
    loader = CUADLoader()
    assert loader.count() >= 10
    contracts = loader.load_contracts(limit=2)
    assert len(contracts) == 2


@pytest.mark.skipif(
    not CUAD_PATH.exists(),
    reason="CUAD dataset not downloaded. Run `make download-data` to enable.",
)
def test_cuad_benchmark_live_dataset() -> None:
    """Test running CUAD benchmark on real contracts if downloaded."""
    runner = CUADBenchmarkRunner(sample_size=2)
    report, ret_cases, abs_cases = runner.run_benchmark()
    assert report.total_contracts_evaluated == 2
    assert report.retrieval_cases_count > 0
    assert 0.0 <= report.recall_at_5 <= 1.0
