from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from catcher_in_the_rag.retrieve.retriever import ScoredClause
from catcher_in_the_rag.schemas import Finding


def load_bartleby_thresholds(tau: float | None = None) -> dict[str, Any]:
    """Single loader for Bartleby thresholds (tau and evidence minimum).

    Reads the ``bartleby`` block of config/thresholds.yaml. If ``tau`` is given it
    overrides ``min_relevance_score``.
    """
    cfg_file = Path("config/thresholds.yaml")
    if not cfg_file.exists():
        for parent in Path.cwd().parents:
            candidate = parent / "config" / "thresholds.yaml"
            if candidate.exists():
                cfg_file = candidate
                break
    data: dict[str, Any] = {}
    if cfg_file.exists():
        with cfg_file.open("r", encoding="utf-8") as f:
            data = (yaml.safe_load(f) or {}).get("bartleby", {})
    return {
        "min_relevance_score": float(tau if tau is not None else data.get("min_relevance_score", 0.5)),
        "min_evidence_spans": int(data.get("min_evidence_spans", 1)),
    }


class BartlebyAssessment(BaseModel):
    """Result of Bartleby abstention inspection."""

    abstained: bool = Field(..., description="Whether Bartleby preferred to abstain")
    reason: str | None = Field(default=None, description="Deterministic reason for abstention")
    surviving_clauses: list[ScoredClause] = Field(
        default_factory=list, description="Clauses that passed relevance thresholds"
    )


_LEGAL_TOPIC_DESCRIPTIONS: dict[str, str] = {
    "governing law": "governing law, jurisdiction, choice of law, or forum selection",
    "payment terms": "payment terms, invoice payment schedules, late fees, or billing procedures",
    "liquidated damages": "liquidated damages, predetermined damage amounts, or penalty fees for default",
    "cap on liability": "cap or limitation on liability, including aggregate monetary liability limits or exclusion/waiver of consequential, punitive, or indirect damages",
    "audit rights": "audit or inspection rights regarding books, records, facilities, or work performance",
    "termination": "termination rights, termination for convenience, termination for cause, or notice periods",
    "warranties": "warranties, representations, disclaimers of warranties, or covenants",
}


class BartlebyGuard:
    """Abstention guard ensuring the agent declines to guess when contract evidence is absent."""

    def __init__(
        self,
        thresholds: dict[str, Any] | None = None,
        enable_template_heuristics: bool = True,
        max_verification_candidates: int = 5,
        category_descriptions: dict[str, str] | None = None,
    ) -> None:
        cfg = thresholds or load_bartleby_thresholds()
        self.min_relevance_score = float(cfg.get("min_relevance_score", 0.5))
        self.min_evidence_spans = int(cfg.get("min_evidence_spans", 1))
        self.enable_template_heuristics = enable_template_heuristics
        self.max_verification_candidates = max_verification_candidates
        self.category_descriptions = category_descriptions or {}

    def verify_relevance_with_llm(
        self,
        candidate_clause: ScoredClause,
        query_topic: str,
        llm_provider: Any = None,
        query: str | None = None,
        contract_id: str | None = None,
        passage_ids: list[str] | None = None,
    ) -> bool:
        """Condition 3: Use LLM to verify whether candidate clause actually answers or governs the topic."""
        if not llm_provider or getattr(llm_provider, "provider_type", "") == "stub":
            return True

        class RelevanceVerification(BaseModel):
            is_relevant: bool = Field(
                ...,
                description="Whether the excerpt actually answers or governs the requested topic",
            )
            reason: str = Field(..., description="Brief rationale")

        topic_lower = query_topic.strip().lower()
        topic_desc = self.category_descriptions.get(query_topic) or self.category_descriptions.get(
            topic_lower
        )
        if not topic_desc:
            for k, v in self.category_descriptions.items():
                if k.lower() == topic_lower:
                    topic_desc = v
                    break
        if not topic_desc:
            topic_desc = _LEGAL_TOPIC_DESCRIPTIONS.get(topic_lower)

        topic_spec = f"{query_topic} ({topic_desc})" if topic_desc else query_topic

        prompt = (
            f"You are a strict legal contract auditor. Determine whether the provided contract passage "
            f"actually addresses or contains legal terms regarding the requested topic.\n\n"
            f"Requested Topic: {topic_spec}\n"
            f"Contract Clause Title: {candidate_clause.title}\n"
            f'Contract Clause Text:\n"""{candidate_clause.text[:2500]}"""\n\n'
            f"Return is_relevant=true if this passage specifically and directly governs, limits, or sets obligations/rights "
            f"for '{query_topic}'. Return is_relevant=false if the passage is unrelated, merely mentions a tangential word, "
            f"or does not contain substantive terms on this topic."
        )

        extra_kwargs = {
            "query": query,
            "category": query_topic,
            "contract_id": contract_id,
            "passage_ids": passage_ids or [candidate_clause.clause_id],
        }

        try:
            if hasattr(llm_provider, "extract_structured"):
                decision = llm_provider.extract_structured(
                    prompt, RelevanceVerification, **extra_kwargs
                )
            elif hasattr(llm_provider, "generate_structured"):
                decision = llm_provider.generate_structured(
                    prompt, RelevanceVerification, **extra_kwargs
                )
            else:
                return True
            return bool(decision.is_relevant)
        except Exception:
            return True

    def evaluate_retrieval(
        self,
        retrieved_clauses: list[ScoredClause],
        query_topic: str | None = None,
        llm_provider: Any = None,
        query: str | None = None,
        contract_id: str | None = None,
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

        # Condition 3a: Explicit Denial / Restrictive Clause Semantic Filter (Template Heuristic)
        # Note: This regex-based denial heuristic targets closed-domain synthetic templates
        # ("under no circumstances reimburse"). It is labeled as a template-specific heuristic.
        if self.enable_template_heuristics and query_topic:
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

        # Condition 3b: LLM Semantic Verification Guard (Multi-candidate inspection)
        if llm_provider and query_topic and passing:
            candidates_to_inspect = passing[: self.max_verification_candidates]
            all_passage_ids = [c.clause_id for c in retrieved_clauses]
            confirmed_clause = None
            for candidate in candidates_to_inspect:
                if self.verify_relevance_with_llm(
                    candidate,
                    query_topic,
                    llm_provider,
                    query=query,
                    contract_id=contract_id,
                    passage_ids=all_passage_ids,
                ):
                    confirmed_clause = candidate
                    break

            if not confirmed_clause:
                return BartlebyAssessment(
                    abstained=True,
                    reason=(
                        f"NO_EVIDENCE: LLM verification inspected top {len(candidates_to_inspect)} candidate "
                        f"clause(s) and determined none materially govern '{query_topic}'. "
                        "I would prefer not to guess."
                    ),
                    surviving_clauses=[],
                )
            return BartlebyAssessment(
                abstained=False,
                reason=None,
                surviving_clauses=[confirmed_clause],
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
