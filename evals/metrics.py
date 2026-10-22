from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ClassMetric(BaseModel):
    """Precision, Recall, and F1 for a single anomaly class."""

    anomaly: str
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    support: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0

    def compute(self) -> None:
        if self.true_positives + self.false_positives > 0:
            self.precision = round(
                self.true_positives / (self.true_positives + self.false_positives), 4
            )
        else:
            self.precision = 1.0 if self.support == 0 else 0.0

        if self.true_positives + self.false_negatives > 0:
            self.recall = round(
                self.true_positives / (self.true_positives + self.false_negatives), 4
            )
        else:
            self.recall = 1.0 if self.support == 0 else 0.0

        if self.precision + self.recall > 0:
            self.f1 = round(2 * (self.precision * self.recall) / (self.precision + self.recall), 4)
        else:
            self.f1 = 0.0


class RetrievalMetric(BaseModel):
    """Retrieval Recall@k and Mean Reciprocal Rank against labeled clause spans."""

    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    mrr: float = 0.0


class BartlebyMetric(BaseModel):
    """Abstention performance metrics for Bartleby."""

    total_a09: int = 0
    correct_abstentions_a09: int = 0
    correct_abstention_rate: float = 0.0

    total_non_a09: int = 0
    false_abstentions: int = 0
    over_abstention_rate: float = 0.0


class CitationMetric(BaseModel):
    """Citation validity metrics ensuring quotes exist verbatim in source contracts."""

    total_citations: int = 0
    valid_citations: int = 0
    citation_validity_rate: float = 1.0


class FailureCase(BaseModel):
    """Detailed record of a failed package evaluation for inspection and debugging."""

    package_id: str
    expected_decision: str
    actual_decision: str
    expected_anomalies: list[str]
    detected_anomalies: list[str]
    justification: str


class EvaluationReport(BaseModel):
    """Versioned benchmark evaluation report."""

    version: str = "0.4.0"
    timestamp: str
    environment: str = "offline_stub"
    config_name: str = "baseline"
    total_packages: int
    mean_latency_ms: float
    decision_accuracy: float
    false_positive_rate_a10: float
    per_class: dict[str, ClassMetric]
    retrieval: RetrievalMetric
    bartleby: BartlebyMetric
    citations: CitationMetric
    failure_cases: list[FailureCase] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
