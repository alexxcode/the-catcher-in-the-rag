"""Empirical calibration harness for Bartleby thresholds (Phase 4).

Sweeps relevance thresholds (min_relevance_score, min_evidence_spans) across
A09 (ground-truth abstention target) and control/other packages (non-abstention target)
to determine the optimal decision boundary that maximizes correct abstention while
strictly avoiding over-abstention.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml

from catcher_in_the_rag.bartleby.guard import BartlebyGuard
from catcher_in_the_rag.retrieve.retriever import ContractRetriever, ScoredClause
from catcher_in_the_rag.schemas import PackageLabels, ProcurementPackage


@dataclass
class SweepCandidate:
    min_relevance_score: float
    min_evidence_spans: int
    a09_total: int
    a09_abstained: int
    a09_abstention_rate: float
    non_a09_total: int
    non_a09_over_abstained: int
    non_a09_over_abstention_rate: float
    f1_score: float
    is_optimal: bool = False


@dataclass
class CalibrationReport:
    timestamp: str
    total_packages: int
    optimal_threshold: float
    optimal_spans: int
    candidates: list[SweepCandidate]


class BartlebyCalibrator:
    """Executes empirical grid search over Bartleby abstention parameters."""

    def __init__(
        self,
        dataset: list[tuple[ProcurementPackage, PackageLabels]],
    ) -> None:
        self.dataset = dataset

    def run_sweep(
        self,
        relevance_steps: list[float] | None = None,
        span_steps: list[int] | None = None,
    ) -> CalibrationReport:
        if relevance_steps is None:
            relevance_steps = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        if span_steps is None:
            span_steps = [1, 2]

        # 1. Pre-compute retrieved clauses for all packages with unlinked/discrepant items
        package_retrievals: list[
            tuple[ProcurementPackage, PackageLabels, list[ScoredClause], str | None]
        ] = []
        for pkg, lbl in self.dataset:
            if not pkg.contract or not pkg.invoice:
                continue

            # Identify if there is an unlinked invoice item requiring contract retrieval
            po_item_ids = {it.item_id for it in pkg.po.line_items} if pkg.po else set()
            unlinked_items = [it for it in pkg.invoice.line_items if it.item_id not in po_item_ids]

            topic = (
                unlinked_items[0].description
                if unlinked_items
                else (pkg.invoice.line_items[0].description if pkg.invoice.line_items else None)
            )

            # Index contract and retrieve
            if pkg.contract.clauses:
                retriever = ContractRetriever(contract=pkg.contract)
                scored = retriever.retrieve(
                    query=topic or "governing contract terms",
                    top_k=5,
                )
            else:
                scored = []

            package_retrievals.append((pkg, lbl, scored, topic))

        # 2. Sweep across threshold candidates
        candidates: list[SweepCandidate] = []
        best_score = -1.0
        best_cand: SweepCandidate | None = None

        for score_th in relevance_steps:
            for spans_th in span_steps:
                guard = BartlebyGuard(
                    thresholds={
                        "min_relevance_score": score_th,
                        "min_evidence_spans": spans_th,
                    }
                )

                a09_total = 0
                a09_abstained = 0
                non_a09_total = 0
                non_a09_over_abstained = 0

                for pkg, lbl, scored_clauses, topic in package_retrievals:
                    is_a09 = any(
                        str(a) == "A09" or getattr(a, "value", None) == "A09" for a in lbl.anomalies
                    )
                    assessment = guard.evaluate_retrieval(scored_clauses, query_topic=topic)

                    if is_a09:
                        a09_total += 1
                        if assessment.abstained:
                            a09_abstained += 1
                    else:
                        # Only count packages where unlinked items triggered contract check
                        po_item_ids = {it.item_id for it in pkg.po.line_items} if pkg.po else set()
                        has_unlinked = any(
                            it.item_id not in po_item_ids for it in pkg.invoice.line_items
                        )
                        if has_unlinked:
                            non_a09_total += 1
                            if assessment.abstained:
                                non_a09_over_abstained += 1

                # Calculate metrics
                abst_rate = (a09_abstained / a09_total) if a09_total > 0 else 0.0
                over_rate = (non_a09_over_abstained / non_a09_total) if non_a09_total > 0 else 0.0

                precision = (
                    a09_abstained / (a09_abstained + non_a09_over_abstained)
                    if (a09_abstained + non_a09_over_abstained) > 0
                    else 0.0
                )
                recall = abst_rate
                f1 = (
                    2 * (precision * recall) / (precision + recall)
                    if (precision + recall) > 0
                    else 0.0
                )

                cand = SweepCandidate(
                    min_relevance_score=round(score_th, 2),
                    min_evidence_spans=spans_th,
                    a09_total=a09_total,
                    a09_abstained=a09_abstained,
                    a09_abstention_rate=round(abst_rate, 4),
                    non_a09_total=non_a09_total,
                    non_a09_over_abstained=non_a09_over_abstained,
                    non_a09_over_abstention_rate=round(over_rate, 4),
                    f1_score=round(f1, 4),
                )
                candidates.append(cand)

                # Prioritize zero over-abstention and maximum F1
                if over_rate == 0.0 and f1 >= best_score:
                    best_score = f1
                    best_cand = cand

        if best_cand:
            best_cand.is_optimal = True

        return CalibrationReport(
            timestamp=datetime.now(UTC).isoformat(),
            total_packages=len(self.dataset),
            optimal_threshold=best_cand.min_relevance_score if best_cand else 0.5,
            optimal_spans=best_cand.min_evidence_spans if best_cand else 1,
            candidates=candidates,
        )


def print_calibration_summary(report: CalibrationReport) -> None:
    print("=" * 86)
    print(" THE CATCHER IN THE RAG - BARTLEBY THRESHOLD CALIBRATION (PHASE 4)")
    print(f" Timestamp: {report.timestamp} | Total Packages: {report.total_packages}")
    print("=" * 86)
    header = f"{'Threshold':<11} | {'Min Spans':<10} | {'A09 Abstention':<16} | {'Over-Abstention':<16} | {'F1-Score':<10} | {'Status'}"
    print(header)
    print("-" * 86)
    for c in report.candidates:
        th = f"{c.min_relevance_score:.2f}"
        spans = str(c.min_evidence_spans)
        a09 = f"{c.a09_abstention_rate * 100:.1f}% ({c.a09_abstained}/{c.a09_total})"
        over = f"{c.non_a09_over_abstention_rate * 100:.1f}% ({c.non_a09_over_abstained}/{c.non_a09_total})"
        f1 = f"{c.f1_score:.4f}"
        tag = "[OPTIMAL]" if c.is_optimal else ""
        print(f"{th:<11} | {spans:<10} | {a09:<16} | {over:<16} | {f1:<10} | {tag}")
    print("=" * 86)
    print(
        f" [RESULT] Recommended Operational Threshold: min_relevance_score = {report.optimal_threshold:.2f}, "
        f"min_evidence_spans = {report.optimal_spans}"
    )
    print("=" * 86)


def save_calibration_report(
    report: CalibrationReport,
    output_dir: str | Path = "evals/reports",
) -> Path:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    report_file = out_path / "calibration_bartleby_report.json"

    data = {
        "timestamp": report.timestamp,
        "total_packages": report.total_packages,
        "optimal_threshold": report.optimal_threshold,
        "optimal_spans": report.optimal_spans,
        "candidates": [asdict(c) for c in report.candidates],
    }

    with report_file.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return report_file


def apply_calibrated_thresholds(
    report: CalibrationReport,
    config_path: str | Path = "config/thresholds.yaml",
) -> bool:
    """Update config/thresholds.yaml with optimal calibrated values."""
    path = Path(config_path)
    if not path.exists():
        return False

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if "bartleby" not in data:
        data["bartleby"] = {}

    data["bartleby"]["min_relevance_score"] = float(report.optimal_threshold)
    data["bartleby"]["min_evidence_spans"] = int(report.optimal_spans)

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    return True
