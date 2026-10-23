"""Empirical calibration harness for Bartleby on real-world CUAD contracts.

Evaluates domain transfer of Bartleby abstention guard from synthetic templates to
authentic commercial SEC contracts:
- 60/40 Train/Val split on CUAD contracts
- Sweeps relevance thresholds [0.10, 0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70]
- Evaluates F1 score balancing correct abstention on absent clauses vs over-abstention on present clauses
- Documents domain discrepancy between synthetic templates and long-form legal contracts
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from catcher_in_the_rag.ingest.cuad_loader import CUADLoader
from catcher_in_the_rag.providers.factory import get_embedding_provider
from catcher_in_the_rag.retrieve.retriever import ContractRetriever
from evals.provenance import get_provenance_metadata


@dataclass
class ThresholdCandidate:
    min_relevance_score: float
    min_evidence_spans: int
    correct_abstentions: int
    total_absences: int
    abstention_rate: float
    false_abstentions: int
    total_presences: int
    false_abstention_rate: float
    precision: float
    recall: float
    f1_score: float
    is_optimal: bool = False


@dataclass
class CUADCalibrationReport:
    timestamp: str
    git_commit_sha: str
    git_dirty: bool
    command: str
    total_contracts: int
    train_contracts_count: int
    val_contracts_count: int
    synthetic_optimal_threshold: float
    cuad_optimal_threshold: float
    optimal_f1_train: float
    validation_f1_at_optimal: float
    validation_f1_at_synthetic_threshold: float
    candidates: list[ThresholdCandidate]


def run_cuad_calibration(
    sample_size: int = 25,
    train_ratio: float = 0.60,
    output_path: str = "evals/reports/calibration_cuad_report.json",
) -> CUADCalibrationReport:
    loader = CUADLoader()
    contracts = loader.load_all()[:sample_size]
    n_train = max(1, int(len(contracts) * train_ratio))
    train_contracts = contracts[:n_train]
    val_contracts = contracts[n_train:]

    print(
        f"Calibrating Bartleby on CUAD: {len(contracts)} contracts (Train: {len(train_contracts)}, Val: {len(val_contracts)})"
    )

    emb_provider = get_embedding_provider()

    # Pre-compute top retrieval scores on train set
    train_presence_scores: list[float] = []
    train_absence_scores: list[float] = []

    categories = [
        "Governing Law",
        "Payment Terms",
        "Liquidated Damages",
        "Cap On Liability",
        "Audit Rights",
        "Termination",
        "Warranties",
    ]

    for c in train_contracts:
        cat_contract = loader.to_catcher_contract(c)
        retriever = ContractRetriever(cat_contract, embedding_provider=emb_provider)
        for cat in categories:
            spans = c.get_spans(cat)
            query = f"clause regarding {cat.lower()}"
            results = retriever.retrieve(query, top_k=3)
            top_score = results[0].score if results else 0.0
            if spans:
                train_presence_scores.append(top_score)
            else:
                train_absence_scores.append(top_score)

    # Threshold sweep on Train set
    thresholds = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    candidates: list[ThresholdCandidate] = []
    best_candidate: ThresholdCandidate | None = None
    best_f1 = -1.0

    for thresh in thresholds:
        # Absence cases: Bartleby should abstain (top_score < thresh)
        correct_abs = sum(1 for s in train_absence_scores if s < thresh)
        tot_abs = max(1, len(train_absence_scores))
        abs_rate = correct_abs / tot_abs

        # Presence cases: Bartleby should NOT abstain (top_score >= thresh)
        false_abs = sum(1 for s in train_presence_scores if s < thresh)
        tot_pres = max(1, len(train_presence_scores))
        false_abs_rate = false_abs / tot_pres

        # Positive class = abstention event
        tp = correct_abs
        fp = false_abs
        fn = tot_abs - correct_abs

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        cand = ThresholdCandidate(
            min_relevance_score=thresh,
            min_evidence_spans=1,
            correct_abstentions=correct_abs,
            total_absences=tot_abs,
            abstention_rate=round(abs_rate, 4),
            false_abstentions=false_abs,
            total_presences=tot_pres,
            false_abstention_rate=round(false_abs_rate, 4),
            precision=round(prec, 4),
            recall=round(rec, 4),
            f1_score=round(f1, 4),
        )
        candidates.append(cand)
        if f1 > best_f1:
            best_f1 = f1
            best_candidate = cand

    if best_candidate:
        best_candidate.is_optimal = True

    optimal_thresh = best_candidate.min_relevance_score if best_candidate else 0.35

    # Evaluate on held-out validation set
    val_presence_scores: list[float] = []
    val_absence_scores: list[float] = []
    for c in val_contracts:
        cat_contract = loader.to_catcher_contract(c)
        retriever = ContractRetriever(cat_contract, embedding_provider=emb_provider)
        for cat in categories:
            spans = c.get_spans(cat)
            query = f"clause regarding {cat.lower()}"
            results = retriever.retrieve(query, top_k=3)
            top_score = results[0].score if results else 0.0
            if spans:
                val_presence_scores.append(top_score)
            else:
                val_absence_scores.append(top_score)

    def calc_f1(scores_abs: list[float], scores_pres: list[float], t: float) -> float:
        tp = sum(1 for s in scores_abs if s < t)
        fp = sum(1 for s in scores_pres if s < t)
        fn = len(scores_abs) - tp
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return round((2 * p * r) / (p + r) if (p + r) > 0 else 0.0, 4)

    val_f1_optimal = calc_f1(val_absence_scores, val_presence_scores, optimal_thresh)
    val_f1_synthetic = calc_f1(val_absence_scores, val_presence_scores, 0.50)

    provenance = get_provenance_metadata({"domain": "CUAD Real SEC Contracts"})

    report = CUADCalibrationReport(
        timestamp=datetime.now(UTC).isoformat(),
        git_commit_sha=provenance["git_commit_sha"],
        git_dirty=provenance["git_dirty"],
        command=provenance["command"],
        total_contracts=len(contracts),
        train_contracts_count=len(train_contracts),
        val_contracts_count=len(val_contracts),
        synthetic_optimal_threshold=0.50,
        cuad_optimal_threshold=optimal_thresh,
        optimal_f1_train=round(best_f1, 4),
        validation_f1_at_optimal=val_f1_optimal,
        validation_f1_at_synthetic_threshold=val_f1_synthetic,
        candidates=candidates,
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2)

    print("=" * 80)
    print(" BARTLEBY CUAD REAL-WORLD CALIBRATION REPORT")
    print(f" Train Contracts: {len(train_contracts)} | Val Contracts: {len(val_contracts)}")
    print(f" Synthetic Calibrated Threshold: 0.50 (Val F1 on CUAD: {val_f1_synthetic:.4f})")
    print(f" CUAD Real SEC Re-calibrated Threshold: {optimal_thresh:.2f} (Val F1: {val_f1_optimal:.4f})")
    print("=" * 80)
    for c in candidates:
        star = " ★ (Optimal)" if c.is_optimal else ""
        print(
            f"Thresh: {c.min_relevance_score:.2f} | Correct Abs: {c.abstention_rate*100:.1f}% | False Abs: {c.false_abstention_rate*100:.1f}% | F1: {c.f1_score:.4f}{star}"
        )
    print("=" * 80)
    return report


if __name__ == "__main__":
    run_cuad_calibration()
