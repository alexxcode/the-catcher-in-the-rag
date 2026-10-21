from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from catcher_in_the_rag.schemas.common import Citation
from catcher_in_the_rag.schemas.findings import AnomalyType, DecisionType


class PackageLabels(BaseModel):
    """Ground truth sidecar metadata stored in labels.json for evaluation."""

    package_id: str = Field(..., description="Target package identifier")
    anomalies: list[AnomalyType] = Field(
        ..., description="List of ground truth anomaly IDs injected (e.g. ['A01'])"
    )
    expected_decision: DecisionType = Field(
        ..., description="Expected deterministic decision (APPROVE, REVIEW, REJECT, ABSTAIN)"
    )
    expected_clause_spans: list[Citation] = Field(
        default_factory=list,
        description="Expected verbatim contract clause citations supporting findings",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Detailed injection parameters (e.g. injected price, discrepancy amount)",
    )
