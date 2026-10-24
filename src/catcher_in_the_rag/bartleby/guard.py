from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from catcher_in_the_rag.retrieve.retriever import ScoredClause
from catcher_in_the_rag.schemas import Finding


def load_bartleby_thresholds() -> dict[str, Any]:
    """Load bartleby abstention thresholds from config/thresholds.yaml."""
    cfg_file = Path("config/thresholds.yaml")
    if not cfg_file.exists():
        for parent in Path.cwd().parents:
            candidate = parent / "config" / "thresholds.yaml"
            if candidate.exists():
                cfg_file = candidate
                break
    if cfg_file.exists():
        with cfg_file.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            return data.get("bartleby", {})
    return {}


class BartlebyAssessment(BaseModel):
    """Result of Bartleby abstention inspection."""

    abstained: bool = Field(..., description="Whether Bartleby preferred to abstain")
    reason: str | None = Field(default=None, description="Deterministic reason for abstention")
    surviving_clauses: list[ScoredClause] = Field(
        default_factory=list, description="Clauses that passed relevance thresholds"
    )


class BartlebyGuard:
    """Abstention guard ensuring the agent declines to guess when contract evidence is absent."""

    def __init__(self, thresholds: dict[str, Any] | None = None) -> None:
        cfg = thresholds or load_bartleby_thresholds()
        self.min_relevance_score = float(cfg.get("min_relevance_score", 0.5))
        self.min_evidence_spans = int(cfg.get("min_evidence_spans", 1))

    def verify_relevance_with_llm(
        self,
        candidate_clause: ScoredClause,
        query_topic: str,
        llm_provider: Any = None,
    ) -> bool:
        """Condition 3: Use LLM to verify whether candidate clause actually answers or governs the topic."""
        if not llm_provider or getattr(llm_provider, "provider_type", "") == "stub":
            return True

        class RelevanceVerification(BaseModel):
            is_relevant: bool = Field(
                ..., description="Whether the excerpt actually answers or governs the requested topic"
            )
            reason: str = Field(..., description="Brief rationale")

        prompt = (
            f"You are a strict legal contract auditor. Determine whether the provided contract passage "
            f"actually addresses or contains legal terms regarding the requested topic.\n\n"
            f"Requested Topic: {query_topic}\n"
            f"Contract Clause Title: {candidate_clause.title}\n"
            f"Contract Clause Text:\n\"\"\"{candidate_clause.text[:2500]}\"\"\"\n\n"
            f"Return is_relevant=true ONLY if this passage specifically and directly governs or sets obligations/rights "
            f"for '{query_topic}'. Return is_relevant=false if the passage is unrelated, merely mentions a tangential word, "
            f"or does not contain substantive terms on '{query_topic}'."
        )

        try:
            decision = llm_provider.generate_structured(prompt, RelevanceVerification)
            return bool(decision.is_relevant)
        except Exception:
            return True

    def evaluate_retrieval(
        self,
        retrieved_clauses: list[ScoredClause],
        query_topic: str | None = None,
        llm_provider: Any = None,
    ) -> BartlebyAssessment:
        """Evaluate candidate clauses against relevance thresholds and authorization criteria.

        Implements SPEC §5 (Bartleby Abstention Conditions):
        Condition 1: No retrieved span passes the relevance threshold.
        Condition 3: The cited span does not actually grant authorization (e.g. explicitly
                     prohibits the fee or requires a written change order not present),
                     or fails LLM semantic verification of topical relevance.
        """
        passing = [c for c in retrieved_clauses if c.score >= self.min_relevance_score]

        # Condition 1: Low relevance
        if len(passing) < self.min_evidence_spans:
            topic_msg = f" for '{query_topic}'" if query_topic else ""
            return BartlebyAssessment(
                abstained=True,
                reason=(
                    f"NO_EVIDENCE: No contract clause{topic_msg} met relevance threshold "
                    f"({self.min_relevance_score:.2f}). I would prefer not to guess."
                ),
                surviving_clauses=[],
            )

        # Condition 3a: Span does not actually grant authorization for requested surcharge/fee
        if query_topic:
            is_surcharge_query = any(
                term in query_topic.lower()
                for term in (
                    "expedite",
                    "rush",
                    "surcharge",
                    "emergency",
                    "courier",
                    "unauthorized",
                    "fee",
                )
            )
            if is_surcharge_query:
                denial_patterns = (
                    "under no circumstances reimburse",
                    "shall not reimburse",
                    "shall not be reimbursed",
                    "buyer shall under no circumstances pay",
                    "unless explicitly authorized in advance by a written change order",
                )
                has_authorizing_clause = False
                for c in passing:
                    text_lower = c.text.lower()
                    is_denied = any(pat in text_lower for pat in denial_patterns)
                    if not is_denied:
                        has_authorizing_clause = True
                        break

                if not has_authorizing_clause:
                    return BartlebyAssessment(
                        abstained=True,
                        reason=(
                            f"NO_EVIDENCE: Governing contract clause for '{query_topic}' explicitly "
                            "denies authorization or conditions reimbursement on unexecuted change orders. "
                            "I would prefer not to guess."
                        ),
                        surviving_clauses=[],
                    )

        # Condition 3b: LLM Semantic Verification Guard
        if llm_provider and query_topic and passing:
            top_clause = passing[0]
            is_confirmed = self.verify_relevance_with_llm(top_clause, query_topic, llm_provider)
            if not is_confirmed:
                return BartlebyAssessment(
                    abstained=True,
                    reason=(
                        f"NO_EVIDENCE: LLM verification determined candidate clause does not "
                        f"materially govern '{query_topic}'. I would prefer not to guess."
                    ),
                    surviving_clauses=[],
                )

        return BartlebyAssessment(
            abstained=False,
            reason=None,
            surviving_clauses=passing,
        )

    def evaluate_finding(
        self,
        finding: Finding,
        contract_raw_text: str,
    ) -> bool:
        """Verify that a proposed finding carries a valid, verbatim quote from the contract."""
        if not finding.citation:
            # Findings requiring contractual backing must have citations
            return False

        cit = finding.citation
        if cit.text not in contract_raw_text:
            return False

        # Verify character offset alignment
        actual_span = contract_raw_text[cit.start_char : cit.end_char]
        return actual_span == cit.text
