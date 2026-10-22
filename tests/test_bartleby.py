from __future__ import annotations

from catcher_in_the_rag.bartleby.guard import BartlebyGuard
from catcher_in_the_rag.retrieve.retriever import ScoredClause
from catcher_in_the_rag.schemas import AnomalyType, Citation, Finding


def test_bartleby_abstains_when_no_evidence() -> None:
    guard = BartlebyGuard(thresholds={"min_relevance_score": 0.5, "min_evidence_spans": 1})

    # Low scoring clause below 0.5
    low_clause = ScoredClause(
        clause_id="C-1",
        clause_type="governing_law",
        title="Governing Law",
        text="State of Delaware",
        start_char=10,
        end_char=27,
        score=0.25,
    )

    assessment = guard.evaluate_retrieval([low_clause], query_topic="unauthorized fee")
    assert assessment.abstained
    assert "NO_EVIDENCE" in assessment.reason
    assert "I would prefer not to" in assessment.reason


def test_bartleby_passes_when_evidence_exists() -> None:
    guard = BartlebyGuard(thresholds={"min_relevance_score": 0.5, "min_evidence_spans": 1})

    good_clause = ScoredClause(
        clause_id="C-2",
        clause_type="payment_terms",
        title="Payment Terms",
        text="payable Net 30 days",
        start_char=10,
        end_char=29,
        score=0.85,
    )

    assessment = guard.evaluate_retrieval([good_clause])
    assert not assessment.abstained
    assert len(assessment.surviving_clauses) == 1


def test_bartleby_evaluates_verbatim_finding() -> None:
    guard = BartlebyGuard()
    raw_text = "Section 2. All invoices payable Net 30 days."

    # Correct verbatim citation
    valid_cit = Citation(
        document_id="CTR-01",
        start_char=24,
        end_char=43,
        text="payable Net 30 days",
    )
    f_valid = Finding(
        finding_id="F-1",
        anomaly_type=AnomalyType.A04,
        field="payment_terms",
        description="terms mismatch",
        citation=valid_cit,
    )
    assert guard.evaluate_finding(f_valid, raw_text)

    # Hallucinated quote
    fake_cit = Citation(
        document_id="CTR-01",
        start_char=24,
        end_char=43,
        text="payable Net 60 days",
    )
    f_fake = Finding(
        finding_id="F-2",
        anomaly_type=AnomalyType.A04,
        field="payment_terms",
        description="terms mismatch",
        citation=fake_cit,
    )
    assert not guard.evaluate_finding(f_fake, raw_text)
