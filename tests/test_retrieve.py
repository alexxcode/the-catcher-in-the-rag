from __future__ import annotations

from catcher_in_the_rag.retrieve.retriever import ContractRetriever
from catcher_in_the_rag.schemas import AnomalyType
from datagen.generator import generate_package


def test_contract_retriever_hybrid_search() -> None:
    pkg, labels = generate_package(seed=301, anomaly=AnomalyType.A10, package_id="PKG-RET")
    contract = pkg.contract
    retriever = ContractRetriever(contract)

    # Search for payment terms
    results = retriever.retrieve("payment terms net days invoice receipt", top_k=2)
    assert len(results) > 0
    top = results[0]
    assert top.clause_type == "payment_terms"
    assert top.score > 0.4
    assert top.text in contract.raw_text

    # Search for late delivery penalties
    penalty_results = retriever.retrieve(
        "liquidated penalty delay late delivery percentage", top_k=2
    )
    assert len(penalty_results) > 0
    assert penalty_results[0].clause_type == "penalties"
