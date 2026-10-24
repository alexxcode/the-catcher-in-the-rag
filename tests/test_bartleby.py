from __future__ import annotations

from typing import Any

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


class MockLLMVerificationProvider:
    def __init__(self, is_relevant: bool, should_fail: bool = False) -> None:
        self.provider_type = "mock_vertex_ai"
        self._is_relevant = is_relevant
        self._should_fail = should_fail
        self.call_count = 0

    def generate_structured(self, prompt: str, response_schema: type) -> Any:
        self.call_count += 1
        if self._should_fail:
            raise RuntimeError("API quota exceeded")
        return response_schema(
            is_relevant=self._is_relevant,
            reason="Mock verified by test suite",
        )


def test_bartleby_llm_verification_confirms_valid_clause() -> None:
    guard = BartlebyGuard(thresholds={"min_relevance_score": 0.5, "min_evidence_spans": 1})
    clause = ScoredClause(
        clause_id="C-10",
        clause_type="audit_rights",
        title="Audit Rights",
        text="Buyer may inspect and audit Supplier books annually.",
        start_char=0,
        end_char=55,
        score=0.75,
    )

    mock_llm = MockLLMVerificationProvider(is_relevant=True)
    is_rel = guard.verify_relevance_with_llm(
        clause, query_topic="audit rights", llm_provider=mock_llm
    )
    assert is_rel is True
    assert mock_llm.call_count == 1

    assessment = guard.evaluate_retrieval(
        [clause], query_topic="audit rights", llm_provider=mock_llm
    )
    assert not assessment.abstained
    assert len(assessment.surviving_clauses) == 1


def test_bartleby_llm_verification_abstains_on_false_positive_clause() -> None:
    guard = BartlebyGuard(thresholds={"min_relevance_score": 0.5, "min_evidence_spans": 1})
    # Clause has high hybrid score due to keyword overlap, but doesn't actually regulate audit rights
    distractor_clause = ScoredClause(
        clause_id="C-11",
        clause_type="general",
        title="Accounting Records",
        text="Supplier will maintain ordinary books in accordance with GAAP.",
        start_char=0,
        end_char=64,
        score=0.72,  # Passes vector threshold!
    )

    mock_llm = MockLLMVerificationProvider(is_relevant=False)
    is_rel = guard.verify_relevance_with_llm(
        distractor_clause, query_topic="audit rights", llm_provider=mock_llm
    )
    assert is_rel is False

    assessment = guard.evaluate_retrieval(
        [distractor_clause], query_topic="audit rights", llm_provider=mock_llm
    )
    assert assessment.abstained
    assert "NO_EVIDENCE" in assessment.reason
    assert "LLM verification" in assessment.reason and "determined" in assessment.reason
    assert len(assessment.surviving_clauses) == 0


def test_bartleby_llm_verification_graceful_fallback_on_error() -> None:
    guard = BartlebyGuard(thresholds={"min_relevance_score": 0.5, "min_evidence_spans": 1})
    clause = ScoredClause(
        clause_id="C-12",
        clause_type="warranties",
        title="Warranties",
        text="Goods will be free from defects.",
        start_char=0,
        end_char=33,
        score=0.65,
    )

    failing_llm = MockLLMVerificationProvider(is_relevant=True, should_fail=True)
    # When LLM raises error, should fallback to passing through rather than hard crash
    is_rel = guard.verify_relevance_with_llm(
        clause, query_topic="warranties", llm_provider=failing_llm
    )
    assert is_rel is True


def test_bartleby_llm_multi_candidate_inspection_finds_second_clause() -> None:
    guard = BartlebyGuard(
        thresholds={"min_relevance_score": 0.5, "min_evidence_spans": 1},
        max_verification_candidates=5,
    )
    distractor = ScoredClause(
        clause_id="C-20",
        clause_type="general",
        title="Preamble",
        text="General definitions and boilerplate terms.",
        start_char=0,
        end_char=43,
        score=0.88,
    )
    good_clause = ScoredClause(
        clause_id="C-21",
        clause_type="audit_rights",
        title="Audit Rights",
        text="Buyer may inspect and audit Supplier books annually.",
        start_char=50,
        end_char=102,
        score=0.82,
    )

    class SelectiveMockLLM:
        def extract_structured(self, text: str, schema: Any, **kwargs: Any) -> Any:
            is_good = "inspect and audit" in text
            return schema(is_relevant=is_good, reason="Test matching")

    assessment = guard.evaluate_retrieval(
        [distractor, good_clause],
        query_topic="audit rights",
        llm_provider=SelectiveMockLLM(),
    )
    assert not assessment.abstained
    assert len(assessment.surviving_clauses) == 1
    assert assessment.surviving_clauses[0].clause_id == "C-21"


def test_bartleby_template_heuristic_toggle() -> None:
    clause = ScoredClause(
        clause_id="C-30",
        clause_type="freight",
        title="Freight",
        text="Buyer shall under no circumstances reimburse expedited courier fees.",
        start_char=0,
        end_char=70,
        score=0.75,
    )

    # With heuristics enabled, Rule 3a forces abstention
    guard_with_heuristics = BartlebyGuard(
        thresholds={"min_relevance_score": 0.5, "min_evidence_spans": 1},
        enable_template_heuristics=True,
    )
    res_heuristic = guard_with_heuristics.evaluate_retrieval([clause], query_topic="expedited courier fee")
    assert res_heuristic.abstained is True
    assert "explicitly denies authorization" in res_heuristic.reason

    # With heuristics disabled, pure retrieval threshold does not force denial abstention
    guard_pure = BartlebyGuard(
        thresholds={"min_relevance_score": 0.5, "min_evidence_spans": 1},
        enable_template_heuristics=False,
    )
    res_pure = guard_pure.evaluate_retrieval([clause], query_topic="expedited courier fee")
    assert res_pure.abstained is False
