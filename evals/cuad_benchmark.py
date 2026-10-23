"""CUAD Real-World Contract Evaluation Benchmark.

Evaluates The Catcher in the RAG retrieval and Bartleby abstention guard
on real commercial contracts from The Atticus Project (CC BY 4.0):
- Retrieval Recall@1, Recall@5, and MRR against expert-annotated clause spans
- Bartleby abstention accuracy against genuine clause absences (ground truth negatives)
- Transparent latency, MRR, and failure analysis on 50k-char SEC filings
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from catcher_in_the_rag.bartleby.guard import BartlebyGuard
from catcher_in_the_rag.ingest.cuad_loader import CUADLoader
from catcher_in_the_rag.retrieve.retriever import ContractRetriever


@dataclass
class RetrievalCaseResult:
    contract_id: str
    category: str
    query: str
    num_ground_truth_spans: int
    retrieved_top_5: list[str]
    hit_at_1: bool
    hit_at_3: bool
    hit_at_5: bool
    reciprocal_rank: float
    latency_ms: float


@dataclass
class BartlebyAbsenceCaseResult:
    contract_id: str
    absent_category: str
    query: str
    abstained: bool
    top_score: float
    latency_ms: float


@dataclass
class CUADBenchmarkReport:
    timestamp: str
    total_contracts_evaluated: int
    retrieval_cases_count: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mean_reciprocal_rank: float
    mean_retrieval_latency_ms: float
    absence_cases_count: int
    bartleby_correct_abstention_rate: float
    bartleby_false_assertion_rate: float
    mean_bartleby_latency_ms: float


class CUADBenchmarkRunner:
    """Executes empirical benchmarks on real CUAD contracts."""

    def __init__(
        self,
        loader: CUADLoader | None = None,
        sample_size: int = 25,
    ) -> None:
        self.loader = loader or CUADLoader()
        self.sample_size = sample_size
        self.guard = BartlebyGuard()

    def run_benchmark(
        self,
    ) -> tuple[CUADBenchmarkReport, list[RetrievalCaseResult], list[BartlebyAbsenceCaseResult]]:
        contracts = self.loader.load_contracts(limit=self.sample_size)
        if not contracts:
            raise RuntimeError("No CUAD contracts loaded. Run `make download-data` first.")

        print(
            "\n===================================================================================================="
        )
        print(" THE CATCHER IN THE RAG - REAL-WORLD CUAD BENCHMARK (THE ATTICUS PROJECT)")
        print(f" Sample Size: {len(contracts)} real commercial contracts (~10k-80k chars each)")
        print(
            "===================================================================================================="
        )

        retrieval_results: list[RetrievalCaseResult] = []
        absence_results: list[BartlebyAbsenceCaseResult] = []

        query_templates: dict[str, str] = {
            "Governing Law": "governing law jurisdiction dispute resolution state courts",
            "Audit Rights": "audit rights inspection books records accounts examination",
            "Termination For Convenience": "termination for convenience without cause advance written notice",
            "Expiration Date": "agreement expiration date term duration renewal",
            "Liquidated Damages": "liquidated damages delay penalty compensation per day",
            "Cap On Liability": "limitation of liability aggregate damages maximum liability cap",
            "Warranty Duration": "warranties standard of performance defect remedy guarantee period",
        }

        # 1. Retrieval Benchmark on Real Labeled Spans
        print(
            "\n[Phase 1] Evaluating Retrieval on Human-Annotated Real Clause Spans...", flush=True
        )
        for idx, cuad in enumerate(contracts):
            catcher_contract = self.loader.to_catcher_contract(cuad)
            retriever = ContractRetriever(catcher_contract)

            for cat, query in query_templates.items():
                gt_spans = cuad.get_spans(cat)
                if not gt_spans:
                    continue

                t0 = time.perf_counter()
                results = retriever.retrieve(query, top_k=5)
                dt_ms = (time.perf_counter() - t0) * 1000

                # Check hit: overlap with any ground-truth character span
                hit_1 = False
                hit_3 = False
                hit_5 = False
                rr = 0.0

                top_texts: list[str] = []
                for rank, clause in enumerate(results, start=1):
                    top_texts.append(clause.text[:60] + "...")
                    # Overlap verification: [clause.start_char, clause.end_char] overlaps [gt.start_char, gt.end_char]
                    overlaps = any(
                        max(clause.start_char, gt.start_char) < min(clause.end_char, gt.end_char)
                        for gt in gt_spans
                    )
                    if overlaps:
                        if rank == 1:
                            hit_1 = True
                        if rank <= 3:
                            hit_3 = True
                        if rank <= 5:
                            hit_5 = True
                        if rr == 0.0:
                            rr = 1.0 / rank

                retrieval_results.append(
                    RetrievalCaseResult(
                        contract_id=cuad.contract_id,
                        category=cat,
                        query=query,
                        num_ground_truth_spans=len(gt_spans),
                        retrieved_top_5=top_texts,
                        hit_at_1=hit_1,
                        hit_at_3=hit_3,
                        hit_at_5=hit_5,
                        reciprocal_rank=rr,
                        latency_ms=dt_ms,
                    )
                )

            print(
                f"  [{idx + 1}/{len(contracts)}] Processed {cuad.contract_id} ({cuad.char_length:,} chars) | Total queries: {len(retrieval_results)}",
                flush=True,
            )

        # 2. Bartleby Abstention Benchmark on Real Ground-Truth Absences
        print(
            "\n[Phase 2] Evaluating Bartleby Abstention on Genuine Real-World Clause Absences...",
            flush=True,
        )
        for idx, cuad in enumerate(contracts):
            catcher_contract = self.loader.to_catcher_contract(cuad)
            retriever = ContractRetriever(catcher_contract)

            for absent_cat, query in query_templates.items():
                if absent_cat not in cuad.absent_categories:
                    continue

                t0 = time.perf_counter()
                results = retriever.retrieve(query, top_k=5)
                top_score = results[0].score if results else 0.0

                # Bartleby evaluation
                assessment = self.guard.evaluate_retrieval(results, query_topic=absent_cat)
                dt_ms = (time.perf_counter() - t0) * 1000

                absence_results.append(
                    BartlebyAbsenceCaseResult(
                        contract_id=cuad.contract_id,
                        absent_category=absent_cat,
                        query=query,
                        abstained=assessment.abstained,
                        top_score=round(top_score, 4),
                        latency_ms=dt_ms,
                    )
                )

        # Aggregate Metrics
        n_ret = len(retrieval_results)
        r1 = sum(r.hit_at_1 for r in retrieval_results) / max(1, n_ret)
        r3 = sum(r.hit_at_3 for r in retrieval_results) / max(1, n_ret)
        r5 = sum(r.hit_at_5 for r in retrieval_results) / max(1, n_ret)
        mrr = sum(r.reciprocal_rank for r in retrieval_results) / max(1, n_ret)
        mean_ret_lat = sum(r.latency_ms for r in retrieval_results) / max(1, n_ret)

        n_abs = len(absence_results)
        correct_abs = sum(a.abstained for a in absence_results) / max(1, n_abs)
        false_assertion = 1.0 - correct_abs
        mean_bart_lat = sum(a.latency_ms for a in absence_results) / max(1, n_abs)

        report = CUADBenchmarkReport(
            timestamp=datetime.now(UTC).isoformat(),
            total_contracts_evaluated=len(contracts),
            retrieval_cases_count=n_ret,
            recall_at_1=round(r1, 4),
            recall_at_3=round(r3, 4),
            recall_at_5=round(r5, 4),
            mean_reciprocal_rank=round(mrr, 4),
            mean_retrieval_latency_ms=round(mean_ret_lat, 2),
            absence_cases_count=n_abs,
            bartleby_correct_abstention_rate=round(correct_abs, 4),
            bartleby_false_assertion_rate=round(false_assertion, 4),
            mean_bartleby_latency_ms=round(mean_bart_lat, 2),
        )

        return report, retrieval_results, absence_results


def print_cuad_summary(report: CUADBenchmarkReport) -> None:
    print("\n" + "=" * 90)
    print(" REAL-WORLD CUAD BENCHMARK RESULTS (THE ATTICUS PROJECT)")
    print(f" Evaluated Contracts: {report.total_contracts_evaluated} real commercial agreements")
    print(f" Timestamp: {report.timestamp}")
    print("=" * 90)
    print(" [RETRIEVAL ON REAL HUMAN-ANNOTATED CLAUSES]")
    print(f"   Total Test Queries:        {report.retrieval_cases_count}")
    print(f"   Recall@1:                  {report.recall_at_1 * 100:.1f}%")
    print(f"   Recall@3:                  {report.recall_at_3 * 100:.1f}%")
    print(f"   Recall@5:                  {report.recall_at_5 * 100:.1f}%")
    print(f"   Mean Reciprocal Rank (MRR):{report.mean_reciprocal_rank:.4f}")
    print(f"   Mean Latency per Query:    {report.mean_retrieval_latency_ms:.2f} ms")
    print("-" * 90)
    print(" [BARTLEBY ABSTENTION ON GENUINE REAL-WORLD CLAUSE ABSENCES]")
    print(f"   Total Absence Test Cases:  {report.absence_cases_count}")
    print(
        f"   Correct Abstention Rate:   {report.bartleby_correct_abstention_rate * 100:.1f}% (Ground Truth Negatives)"
    )
    print(
        f"   False Assertion Rate:      {report.bartleby_false_assertion_rate * 100:.1f}% (Spurious Claims)"
    )
    print(f"   Mean Latency per Decision: {report.mean_bartleby_latency_ms:.2f} ms")
    print("=" * 90)


def save_cuad_report(
    report: CUADBenchmarkReport,
    retrieval_cases: list[RetrievalCaseResult],
    absence_cases: list[BartlebyAbsenceCaseResult],
    output_dir: str | Path = "evals/reports",
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    report_file = out / f"cuad_benchmark_{ts}.json"
    latest_file = out / "cuad_benchmark_latest.json"

    data = {
        "report": asdict(report),
        "sample_retrieval_cases": [asdict(c) for c in retrieval_cases[:20]],
        "sample_absence_cases": [asdict(a) for a in absence_cases[:20]],
    }

    with report_file.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    with latest_file.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return report_file
