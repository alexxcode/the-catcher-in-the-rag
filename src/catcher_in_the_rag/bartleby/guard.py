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

    def evaluate_retrieval(
        self,
        retrieved_clauses: list[ScoredClause],
        query_topic: str | None = None,
    ) -> BartlebyAssessment:
        """Evaluate candidate clauses against relevance thresholds."""
        passing = [c for c in retrieved_clauses if c.score >= self.min_relevance_score]

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
