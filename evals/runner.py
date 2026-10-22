from __future__ import annotations

import time
from datetime import UTC, datetime

from catcher_in_the_rag.graph import run_audit
from catcher_in_the_rag.match.matcher import run_three_way_match
from catcher_in_the_rag.policy.engine import PolicyEngine
from catcher_in_the_rag.retrieve.retriever import ContractRetriever
from catcher_in_the_rag.schemas import (
    AnomalyType,
    DecisionType,
    PackageLabels,
    ProcurementPackage,
)
from catcher_in_the_rag.validate.validator import validate_findings
from evals.metrics import (
    BartlebyMetric,
    CitationMetric,
    ClassMetric,
    EvaluationReport,
    FailureCase,
    RetrievalMetric,
)


class EvaluationRunner:
    """Evaluates the audit pipeline against ground-truth labeled packages."""

    def __init__(self, mode: str = "baseline") -> None:
        self.mode = mode  # baseline | no-rag | no-bartleby

    def _evaluate_retrieval(
        self,
        package: ProcurementPackage,
        labels: PackageLabels,
        k_values: list[int],
    ) -> tuple[dict[int, bool], float]:
        """Measure whether expected clause spans appear in top-k retrieval using realistic audit queries.

        Decouples the retrieval query from the ground truth answer text (Phase 2 requirement).
        """
        if not labels.expected_clause_spans:
            return (dict.fromkeys(k_values, True), 1.0)

        retriever = ContractRetriever(package.contract)

        audit_topic_queries = {
            AnomalyType.A01: "negotiated rate sheet maximum unit prices SKU rate ceilings",
            AnomalyType.A04: "payment terms invoice due date net remittance schedule",
            AnomalyType.A05: "liquidated damages delivery delay penalty calendar day deduction",
            AnomalyType.A06: "term and termination agreement expiration date renewal",
            AnomalyType.A09: "freight transportation expedited courier rush delivery fee surcharge authorization",
        }
        topic_queries = [
            audit_topic_queries[a] for a in labels.anomalies if a in audit_topic_queries
        ]
        if not topic_queries:
            topic_queries = ["contractual governing terms and conditions"]

        search_query = " ".join(topic_queries)
        top_k_clauses = retriever.retrieve(search_query, top_k=max(k_values))

        found_at_k: dict[int, bool] = dict.fromkeys(k_values, False)
        reciprocal_rank = 0.0

        for expected_span in labels.expected_clause_spans:
            for rank_idx, candidate in enumerate(top_k_clauses, start=1):
                if expected_span.text in candidate.text or candidate.text in expected_span.text:
                    if reciprocal_rank == 0.0:
                        reciprocal_rank = 1.0 / rank_idx
                    for k in k_values:
                        if rank_idx <= k:
                            found_at_k[k] = True
                    break

        return found_at_k, reciprocal_rank

    def run_eval(
        self,
        packages_with_labels: list[tuple[ProcurementPackage, PackageLabels]],
    ) -> EvaluationReport:
        """Run full evaluation suite over the provided packages."""
        start_eval_time = time.perf_counter()

        per_class: dict[str, ClassMetric] = {
            a.value: ClassMetric(anomaly=a.value) for a in AnomalyType
        }
        for _, labels in packages_with_labels:
            for anomaly in labels.anomalies:
                per_class[anomaly.value].support += 1

        k_values = [1, 3, 5, 10]
        retrieval_hits: dict[int, int] = dict.fromkeys(k_values, 0)
        total_retrieval_queries = 0
        sum_mrr = 0.0

        total_a09 = 0
        correct_abstentions_a09 = 0
        total_non_a09 = 0
        false_abstentions = 0

        total_citations = 0
        valid_citations = 0

        correct_decisions = 0
        clean_a10_false_positives = 0
        total_clean_a10 = 0
        latencies: list[float] = []
        failure_cases: list[FailureCase] = []

        policy_engine = PolicyEngine()

        for pkg, labels in packages_with_labels:
            t0 = time.perf_counter()

            # Retrieval benchmark (only when contract retrieval is enabled)
            if self.mode != "no-rag" and labels.expected_clause_spans:
                hits, rr = self._evaluate_retrieval(pkg, labels, k_values)
                total_retrieval_queries += 1
                sum_mrr += rr
                for k, hit in hits.items():
                    if hit:
                        retrieval_hits[k] += 1

            # Run pipeline based on ablation mode
            if self.mode == "no-rag":
                # Ablation: Only 3-way match, no contract clauses
                match_findings = run_three_way_match(pkg)
                decision = policy_engine.evaluate(
                    package_id=pkg.package_id,
                    invoice_total=pkg.invoice.total_amount,
                    findings=match_findings,
                )
            else:
                decision = run_audit(pkg)

            t_elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(t_elapsed_ms)

            # Decision check
            if decision.decision == labels.expected_decision:
                correct_decisions += 1
            else:
                failure_cases.append(
                    FailureCase(
                        package_id=pkg.package_id,
                        expected_decision=labels.expected_decision.value,
                        actual_decision=decision.decision.value,
                        expected_anomalies=[a.value for a in labels.anomalies],
                        detected_anomalies=[f.anomaly_type.value for f in decision.findings],
                        justification=decision.justification,
                    )
                )

            # Bartleby metrics
            is_a09 = AnomalyType.A09 in labels.anomalies
            if is_a09:
                total_a09 += 1
                if decision.decision == DecisionType.ABSTAIN or decision.bartleby_abstained:
                    correct_abstentions_a09 += 1
            else:
                total_non_a09 += 1
                if decision.bartleby_abstained and decision.decision == DecisionType.ABSTAIN:
                    false_abstentions += 1

            # Clean A10 False Positive Check
            if AnomalyType.A10 in labels.anomalies:
                total_clean_a10 += 1
                if decision.decision != DecisionType.APPROVE or len(decision.findings) > 0:
                    clean_a10_false_positives += 1

            # Citation validity check
            for f in decision.findings:
                if f.citation:
                    total_citations += 1
                    valid_spans, _ = validate_findings([f], pkg.contract)
                    if valid_spans:
                        valid_citations += 1

            # Per-class confusion matrix
            detected_types = {f.anomaly_type for f in decision.findings}
            for anomaly_enum in AnomalyType:
                a_str = anomaly_enum.value
                is_ground_truth = anomaly_enum in labels.anomalies
                is_detected = anomaly_enum in detected_types

                if is_ground_truth and is_detected:
                    per_class[a_str].true_positives += 1
                elif is_ground_truth and not is_detected:
                    per_class[a_str].false_negatives += 1
                elif not is_ground_truth and is_detected:
                    per_class[a_str].false_positives += 1

        # Compute per-class precision / recall / F1
        for metric in per_class.values():
            metric.compute()

        # Aggregate retrieval metric
        retrieval_metric = RetrievalMetric(
            recall_at_1=round(retrieval_hits[1] / max(1, total_retrieval_queries), 4),
            recall_at_3=round(retrieval_hits[3] / max(1, total_retrieval_queries), 4),
            recall_at_5=round(retrieval_hits[5] / max(1, total_retrieval_queries), 4),
            recall_at_10=round(retrieval_hits[10] / max(1, total_retrieval_queries), 4),
            mrr=round(sum_mrr / max(1, total_retrieval_queries), 4),
        )

        # Aggregate bartleby metric
        bartleby_metric = BartlebyMetric(
            total_a09=total_a09,
            correct_abstentions_a09=correct_abstentions_a09,
            correct_abstention_rate=round(correct_abstentions_a09 / max(1, total_a09), 4),
            total_non_a09=total_non_a09,
            false_abstentions=false_abstentions,
            over_abstention_rate=round(false_abstentions / max(1, total_non_a09), 4),
        )

        # Aggregate citations
        citation_metric = CitationMetric(
            total_citations=total_citations,
            valid_citations=valid_citations,
            citation_validity_rate=round(valid_citations / max(1, total_citations), 4)
            if total_citations > 0
            else 1.0,
        )

        total_pkgs = len(packages_with_labels)
        mean_latency = round(sum(latencies) / max(1, total_pkgs), 2)
        accuracy = round(correct_decisions / max(1, total_pkgs), 4)
        fp_rate_a10 = (
            round(clean_a10_false_positives / max(1, total_clean_a10), 4)
            if total_clean_a10 > 0
            else 0.0
        )

        return EvaluationReport(
            timestamp=datetime.now(UTC).isoformat(),
            config_name=self.mode,
            total_packages=total_pkgs,
            mean_latency_ms=mean_latency,
            decision_accuracy=accuracy,
            false_positive_rate_a10=fp_rate_a10,
            per_class=per_class,
            retrieval=retrieval_metric,
            bartleby=bartleby_metric,
            citations=citation_metric,
            failure_cases=failure_cases,
            metadata={
                "duration_seconds": round(time.perf_counter() - start_eval_time, 2),
                "mode": self.mode,
            },
        )
