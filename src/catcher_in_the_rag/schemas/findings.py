from decimal import Decimal
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

from catcher_in_the_rag.schemas.common import Citation


class AnomalyType(StrEnum):
    """Standardized anomaly taxonomy per Section 7 of the specification."""

    A01 = "A01"  # Unit price above contract rate (Match + clause)
    A02 = "A02"  # Quantity billed > received (Match)
    A03 = "A03"  # Duplicate invoice (Match)
    A04 = "A04"  # Payment terms differ from contract (Clause)
    A05 = "A05"  # Late-delivery penalty applicable (Clause)
    A06 = "A06"  # Contract expired / auto-renewal triggered (Clause)
    A07 = "A07"  # Unknown vendor or no PO (Match)
    A08 = "A08"  # Amount above approval threshold (Policy)
    A09 = "A09"  # Relevant clause absent -> must abstain (Clause / negative)
    A10 = "A10"  # Clean package (false-positive control)


FindingSeverity = Literal["INFO", "WARNING", "ERROR"]
VerificationStatus = Literal["VERIFIED", "NOT_VERIFIABLE"]


class Finding(BaseModel):
    """Individual discrepancy or audit finding produced by matching or clause evaluation."""

    finding_id: str = Field(..., description="Unique finding ID")
    anomaly_type: AnomalyType = Field(..., description="Anomaly classification code")
    field: str = Field(..., description="Invoice / PO field or clause topic in question")
    description: str = Field(..., description="Human-readable explanation of discrepancy")
    severity: FindingSeverity = Field(
        default="ERROR", description="Severity impact of the discrepancy"
    )
    citation: Citation | None = Field(
        default=None,
        description="Verbatim contract citation justifying this finding. Required for clause-based findings.",
    )
    calculated_difference: Decimal | None = Field(
        default=None,
        description="Quantitative discrepancy amount (e.g. overbilled amount, penalty amount)",
    )
    verification_status: VerificationStatus = Field(
        default="VERIFIED",
        description="NOT_VERIFIABLE when Bartleby abstained: routed to human review, not an accusation",
    )


class DecisionType(StrEnum):
    """Final decision category for an invoice package."""

    APPROVE = "APPROVE"
    REVIEW = "REVIEW"
    REJECT = "REJECT"
    ABSTAIN = "ABSTAIN"


class Decision(BaseModel):
    """Audit outcome produced by the deterministic policy engine."""

    package_id: str = Field(..., description="ID of audited package")
    decision: DecisionType = Field(..., description="Deterministic decision outcome")
    findings: list[Finding] = Field(default_factory=list, description="All verified findings")
    requires_human_review: bool = Field(
        default=False, description="Whether package is routed to human review workspace"
    )
    justification: str = Field(..., description="Deterministic summary justification for decision")
    bartleby_abstained: bool = Field(
        default=False, description="True if bartleby abstained due to missing evidence"
    )
    bartleby_reason: str | None = Field(
        default=None, description="Reason code or message from bartleby abstention guard"
    )
