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
from typing import Any

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
    calibration_packages: int
    validation_packages: int
    optimal_threshold: float
    optimal_spans: int
    auc_roc_train: float
    auc_roc_val: float
    validation_macro_f1: float
    candidates: list[SweepCandidate]
    validation_candidate: SweepCandidate | None = None


class BartlebyCalibrator:
    """Executes empirical grid search over Bartleby abstention parameters.

    Features:
    - Train / held-out validation split
    - Non-zero denominator for over-abstention testing legitimate contract clauses
    - Hard cases / near-misses testing boundary discrimination
    - Center-of-plateau threshold selection (avoiding brittle edge thresholds)
    """

    def __init__(
        self,
        dataset: list[tuple[ProcurementPackage, PackageLabels]],
        train_ratio: float = 0.6,
        seed: int = 42,
    ) -> None:
        self.dataset = dataset
        self.train_ratio = train_ratio
        self.seed = seed

    def _split_dataset(
        self,
    ) -> tuple[
        list[tuple[ProcurementPackage, PackageLabels]],
        list[tuple[ProcurementPackage, PackageLabels]],
    ]:
        if len(self.dataset) <= 2:
            # Minimal dataset (e.g. unit tests)
            return self.dataset, self.dataset

        # Deterministic stratified split by A09 vs non-A09
        a09_pkgs = [
            (pkg_item, lbl_item)
            for pkg_item, lbl_item in self.dataset
            if any(
                str(a) == "A09" or getattr(a, "value", None) == "A09" for a in lbl_item.anomalies
            )
        ]
        non_a09_pkgs = [
            (pkg_item, lbl_item)
            for pkg_item, lbl_item in self.dataset
            if not any(
                str(a) == "A09" or getattr(a, "value", None) == "A09" for a in lbl_item.anomalies
            )
        ]

        n_a09_train = max(1, int(len(a09_pkgs) * self.train_ratio))
        n_non_train = max(1, int(len(non_a09_pkgs) * self.train_ratio))

        train = a09_pkgs[:n_a09_train] + non_a09_pkgs[:n_non_train]
        val = a09_pkgs[n_a09_train:] + non_a09_pkgs[n_non_train:]

        if not val:
            val = train
        return train, val

    def _prepare_retrievals(
        self,
        subset: list[tuple[ProcurementPackage, PackageLabels]],
    ) -> list[tuple[ProcurementPackage, PackageLabels, list[ScoredClause], str | None, bool]]:
        """Pre-compute retrievals for both A09 (uncontracted items) and non-A09 (legitimate/near-miss terms)."""
        retrievals: list[
            tuple[ProcurementPackage, PackageLabels, list[ScoredClause], str | None, bool]
        ] = []

        for pkg, lbl in subset:
            if not pkg.contract or not pkg.invoice:
                continue

            is_a09 = any(
                str(a) == "A09" or getattr(a, "value", None) == "A09" for a in lbl.anomalies
            )

            po_item_ids = {it.item_id for it in pkg.po.line_items} if pkg.po else set()
            unlinked_items = [it for it in pkg.invoice.line_items if it.item_id not in po_item_ids]

            if is_a09:
                topic = (
                    unlinked_items[0].description
                    if unlinked_items
                    else "unauthorized delivery surcharge"
                )
            else:
                # In non-A09, test legitimate contract clause queries that must NOT trigger false abstention
                anomaly_set = {
                    str(a) if isinstance(a, str) else getattr(a, "value", str(a))
                    for a in lbl.anomalies
                }
                if "A04" in anomaly_set:
                    topic = "payment terms net days invoice receipt"
                elif "A05" in anomaly_set:
                    topic = "liquidated penalty delay late delivery percentage"
                elif "A06" in anomaly_set:
                    topic = "term and termination agreement expiration date renewal"
                elif "A01" in anomaly_set:
                    topic = "pricing rate ceilings negotiated rate sheet maximum unit rates"
                else:
                    topic = "standard shipping freight carriage protocol"

            if pkg.contract.clauses:
                retriever = ContractRetriever(contract=pkg.contract)
                scored = retriever.retrieve(
                    query=topic,
                    top_k=5,
                )
            else:
                scored = []

            retrievals.append((pkg, lbl, scored, topic, is_a09))

        return retrievals

    def _evaluate_candidate(
        self,
        guard: BartlebyGuard,
        retrievals: list[
            tuple[ProcurementPackage, PackageLabels, list[ScoredClause], str | None, bool]
        ],
        score_th: float,
        spans_th: int,
    ) -> SweepCandidate:
        a09_total = 0
        a09_abstained = 0
        non_a09_total = 0
        non_a09_over_abstained = 0

        for _pkg, _lbl, scored_clauses, topic, is_a09 in retrievals:
            assessment = guard.evaluate_retrieval(scored_clauses, query_topic=topic)

            if is_a09:
                a09_total += 1
                if assessment.abstained:
                    a09_abstained += 1
            else:
                # For non-A09 packages, the contract contains the required terms
                # If guard abstains, that is an over-abstention!
                non_a09_total += 1
                if assessment.abstained:
                    non_a09_over_abstained += 1

        abst_rate = (a09_abstained / a09_total) if a09_total > 0 else 0.0
        over_rate = (non_a09_over_abstained / non_a09_total) if non_a09_total > 0 else 0.0

        precision = (
            a09_abstained / (a09_abstained + non_a09_over_abstained)
            if (a09_abstained + non_a09_over_abstained) > 0
            else 0.0
        )
        recall = abst_rate
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return SweepCandidate(
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

    def run_sweep(
        self,
        relevance_steps: list[float] | None = None,
        span_steps: list[int] | None = None,
    ) -> CalibrationReport:
        if relevance_steps is None:
            relevance_steps = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        if span_steps is None:
            span_steps = [1, 2]

        train_data, val_data = self._split_dataset()
        train_retrievals = self._prepare_retrievals(train_data)
        val_retrievals = self._prepare_retrievals(val_data)

        # 1. Sweep on Training (Calibration) Split
        candidates: list[SweepCandidate] = []
        for score_th in relevance_steps:
            for spans_th in span_steps:
                guard = BartlebyGuard(
                    thresholds={
                        "min_relevance_score": score_th,
                        "min_evidence_spans": spans_th,
                    }
                )
                cand = self._evaluate_candidate(
                    guard=guard,
                    retrievals=train_retrievals,
                    score_th=score_th,
                    spans_th=spans_th,
                )
                candidates.append(cand)

        # 2. Find optimal plateau and select center of plateau
        # Maximize F1-score first, breaking ties by lowest over-abstention
        best_f1 = max(c.f1_score for c in candidates)
        # Identify optimal candidates within top F1 range
        plateau_candidates = [c for c in candidates if c.f1_score >= max(0.1, best_f1 - 0.02)]
        if not plateau_candidates:
            plateau_candidates = candidates

        min_over = min(c.non_a09_over_abstention_rate for c in plateau_candidates)
        top_candidates = [
            c for c in plateau_candidates if c.non_a09_over_abstention_rate == min_over
        ]

        # Select center/median of the plateau (NOT the boundary edge)
        if top_candidates:
            median_idx = len(top_candidates) // 2
            best_cand = top_candidates[median_idx]
            best_cand.is_optimal = True
        else:
            best_cand = candidates[0]
            best_cand.is_optimal = True

        # 3. Evaluate optimal candidate on held-out Validation Split
        val_guard = BartlebyGuard(
            thresholds={
                "min_relevance_score": best_cand.min_relevance_score,
                "min_evidence_spans": best_cand.min_evidence_spans,
            }
        )
        val_cand = self._evaluate_candidate(
            guard=val_guard,
            retrievals=val_retrievals,
            score_th=best_cand.min_relevance_score,
            spans_th=best_cand.min_evidence_spans,
        )

        def _get_auc(retrieval_set: list[Any]) -> float:
            pres = [r[2][0].score if r[2] else 0.0 for r in retrieval_set if not r[4]]
            abs_ = [r[2][0].score if r[2] else 0.0 for r in retrieval_set if r[4]]
            if not pres or not abs_:
                return 0.5
            wins = sum(1.0 if p > a else (0.5 if p == a else 0.0) for p in pres for a in abs_)
            return round(wins / (len(pres) * len(abs_)), 4)

        auc_train = _get_auc(train_retrievals)
        auc_val = _get_auc(val_retrievals)

        # Macro F1 across Absence and Presence classes on Validation set
        tp_a = val_cand.a09_abstained
        fp_a = val_cand.non_a09_over_abstained
        fn_a = val_cand.a09_total - val_cand.a09_abstained
        p_a = tp_a / (tp_a + fp_a) if (tp_a + fp_a) > 0 else 0.0
        r_a = tp_a / (tp_a + fn_a) if (tp_a + fn_a) > 0 else 0.0
        f1_a = (2 * p_a * r_a) / (p_a + r_a) if (p_a + r_a) > 0 else 0.0

        tp_p = val_cand.non_a09_total - val_cand.non_a09_over_abstained
        fp_p = fn_a
        fn_p = fp_a
        p_p = tp_p / (tp_p + fp_p) if (tp_p + fp_p) > 0 else 0.0
        r_p = tp_p / (tp_p + fn_p) if (tp_p + fn_p) > 0 else 0.0
        f1_p = (2 * p_p * r_p) / (p_p + r_p) if (p_p + r_p) > 0 else 0.0
        val_macro_f1 = round((f1_a + f1_p) / 2.0, 4)

        return CalibrationReport(
            timestamp=datetime.now(UTC).isoformat(),
            total_packages=len(self.dataset),
            calibration_packages=len(train_data),
            validation_packages=len(val_data),
            optimal_threshold=best_cand.min_relevance_score,
            optimal_spans=best_cand.min_evidence_spans,
            auc_roc_train=auc_train,
            auc_roc_val=auc_val,
            validation_macro_f1=val_macro_f1,
            candidates=candidates,
            validation_candidate=val_cand,
        )


def print_calibration_summary(report: CalibrationReport) -> None:
    print("=" * 96)
    print(" THE CATCHER IN THE RAG - BARTLEBY THRESHOLD CALIBRATION (PHASE 4)")
    print(
        f" Timestamp: {report.timestamp} | Total: {report.total_packages} "
        f"(Train: {report.calibration_packages}, Val: {report.validation_packages})"
    )
    print("=" * 96)
    header = f"{'Threshold':<11} | {'Min Spans':<10} | {'A09 Abstention':<18} | {'Over-Abstention':<18} | {'F1-Score':<10} | {'Status'}"
    print(header)
    print("-" * 96)
    for c in report.candidates:
        th = f"{c.min_relevance_score:.2f}"
        spans = str(c.min_evidence_spans)
        a09 = f"{c.a09_abstention_rate * 100:.1f}% ({c.a09_abstained}/{c.a09_total})"
        over = f"{c.non_a09_over_abstention_rate * 100:.1f}% ({c.non_a09_over_abstained}/{c.non_a09_total})"
        f1 = f"{c.f1_score:.4f}"
        tag = "[OPTIMAL-PLATEAU-CENTER]" if c.is_optimal else ""
        print(f"{th:<11} | {spans:<10} | {a09:<18} | {over:<18} | {f1:<10} | {tag}")
    print("=" * 96)
    if report.validation_candidate:
        vc = report.validation_candidate
        print(
            f" [VALIDATION SET EVALUATION (HELD-OUT {report.validation_packages} PKGS)]\n"
            f"   * A09 Abstention (Recall)    : {vc.a09_abstention_rate * 100:.1f}% ({vc.a09_abstained}/{vc.a09_total})\n"
            f"   * Over-Abstention (False Pos): {vc.non_a09_over_abstention_rate * 100:.1f}% ({vc.non_a09_over_abstained}/{vc.non_a09_total})\n"
            f"   * Held-out F1-Score          : {vc.f1_score:.4f}\n"
            f"   * Held-out Macro F1          : {report.validation_macro_f1:.4f}\n"
            f"   * AUC-ROC Separability       : {report.auc_roc_val:.4f}"
        )
    print(
        f" [RESULT] Calibrated Optimal Plateau Center: min_relevance_score = {report.optimal_threshold:.2f}, "
        f"min_evidence_spans = {report.optimal_spans}"
    )
    print("=" * 96)


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
        "calibration_packages": report.calibration_packages,
        "validation_packages": report.validation_packages,
        "optimal_threshold": report.optimal_threshold,
        "optimal_spans": report.optimal_spans,
        "auc_roc_train": report.auc_roc_train,
        "auc_roc_val": report.auc_roc_val,
        "validation_macro_f1": report.validation_macro_f1,
        "candidates": [asdict(c) for c in report.candidates],
        "validation_candidate": asdict(report.validation_candidate)
        if report.validation_candidate
        else None,
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
