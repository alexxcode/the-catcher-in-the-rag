"""Unit tests for CUAD and OCDS data ingestion and benchmark runners."""

from __future__ import annotations

from catcher_in_the_rag.ingest.cuad_loader import CUADLoader
from catcher_in_the_rag.ingest.ocds_loader import OCDSLoader
from evals.cuad_benchmark import CUADBenchmarkRunner


def test_cuad_loader_structure() -> None:
    loader = CUADLoader()
    assert loader.count() >= 10

    contracts = loader.load_contracts(limit=3)
    assert len(contracts) == 3

    c0 = contracts[0]
    assert c0.contract_id == "CUAD-0001"
    assert len(c0.raw_text) > 5000
    assert len(c0.annotations) > 0
    assert len(c0.absent_categories) > 0

    # Test conversion to Catcher Contract schema
    catcher_contract = loader.to_catcher_contract(c0)
    assert catcher_contract.contract_id == "CUAD-0001"
    assert len(catcher_contract.clauses) > 0
    assert len(catcher_contract.raw_text) == len(c0.raw_text)


def test_ocds_loader_structure() -> None:
    loader = OCDSLoader()
    records = loader.load_contracts()
    assert len(records) > 0

    rec = records[0]
    assert rec.buyer.name is not None
    assert rec.total_amount >= 0
    assert rec.currency in ("GBP", "USD", "EUR")


def test_cuad_benchmark_execution() -> None:
    runner = CUADBenchmarkRunner(sample_size=2)
    report, ret_cases, abs_cases = runner.run_benchmark()

    assert report.total_contracts_evaluated == 2
    assert report.retrieval_cases_count > 0
    assert 0.0 <= report.recall_at_5 <= 1.0
    assert 0.0 <= report.mean_reciprocal_rank <= 1.0
    assert report.absence_cases_count > 0
    assert report.bartleby_correct_abstention_rate >= 0.80
