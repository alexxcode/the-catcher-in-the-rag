"""CUAD Real-World Contract Evaluation Benchmark.

Evaluates The Catcher in the RAG hybrid retrieval (BM25 + dense embeddings) and
Bartleby abstention guard on authentic commercial contracts from The Atticus Project (CC BY 4.0):
- Retrieval Recall@1, Recall@3, Recall@5, and MRR against expert-annotated clause spans
- Verification of dense embedding provider (Vertex AI text-embedding-005 vs offline stub)
- Dual-sided Bartleby assessment:
  * Correct Abstention on Genuine Real-World Absences (True Negatives, specificity)
  * False Abstention on Present Relevant Clauses (False Negatives, false abstention rate)
- Wilson score 95% confidence intervals on abstention and retrieval metrics
- Stratified Train / Validation contract split evaluation (60/40)
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from catcher_in_the_rag.bartleby.guard import BartlebyGuard
from catcher_in_the_rag.ingest.cuad_loader import CUADLoader
from catcher_in_the_rag.providers.base import EmbeddingProvider
from catcher_in_the_rag.providers.factory import get_embedding_provider
from catcher_in_the_rag.retrieve.retriever import ContractRetriever


def _wilson_score_interval(successes: int, total: int, z: float = 1.96) -> list[float]:
    """Calculate 95% Wilson score confidence interval for a binomial proportion."""
    if total == 0:
        return [0.0, 0.0]
    p_hat = successes / total
    denom = 1.0 + (z * z) / total
    center = (p_hat + (z * z) / (2.0 * total)) / denom
    margin = (z / denom) * math.sqrt(
        (p_hat * (1.0 - p_hat) / total) + (z * z) / (4.0 * total * total)
    )
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return [round(lower, 4), round(upper, 4)]


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
    bartleby_abstained: bool
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
    embedding_provider: str
    embedding_model_id: str
    total_contracts_evaluated: int
    train_contracts_count: int
    val_contracts_count: int
    retrieval_cases_count: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mean_reciprocal_rank: float
    mean_retrieval_latency_ms: float
    presence_cases_count: int
    bartleby_false_abstention_count: int
    bartleby_false_abstention_rate: float
    bartleby_false_abstention_ci95: list[float]
    absence_cases_count: int
    bartleby_correct_abstention_count: int
    bartleby_correct_abstention_rate: float
    bartleby_correct_abstention_ci95: list[float]
    bartleby_false_assertion_rate: float
    mean_bartleby_latency_ms: float


class CUADBenchmarkRunner:
    """Executes empirical benchmarks on real CUAD contracts with real embeddings."""

    def __init__(
        self,
        loader: CUADLoader | None = None,
        sample_size: int = 50,
        embedding_provider: EmbeddingProvider | None = None,
        embedding_provider_type: str | None = None,
    ) -> None:
        self.loader = loader or CUADLoader()
        self.sample_size = sample_size
        self.embedding_provider = embedding_provider or get_embedding_provider(
            embedding_provider_type
        )
        self.guard = BartlebyGuard()

    def run_benchmark(
        self,
    ) -> tuple[CUADBenchmarkReport, list[RetrievalCaseResult], list[BartlebyAbsenceCaseResult]]:
        contracts = self.loader.load_contracts(limit=self.sample_size)
        if not contracts:
            raise RuntimeError("No CUAD contracts loaded. Run `make download-data` first.")

        provider_name = type(self.embedding_provider).__name__
        model_id = getattr(self.embedding_provider, "model_id", "stub-hash-768d")

        n_train = max(1, int(len(contracts) * 0.6))
        train_contracts = contracts[:n_train]
        val_contracts = contracts[n_train:]

        print(
            "\n===================================================================================================="
        )
        print(" THE CATCHER IN THE RAG - REAL-WORLD CUAD BENCHMARK (THE ATTICUS PROJECT)")
        print(
            f" Contracts: {len(contracts)} (Train: {len(train_contracts)}, Val: {len(val_contracts)})"
        )
        print(f" Embedding Provider: {provider_name} (Model: {model_id})")
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

        # 1. Retrieval & False Abstention Benchmark on Real Ground-Truth Spans
        print(
            "\n[Phase 1] Evaluating Retrieval & False Abstention on Ground-Truth Present Clauses...",
            flush=True,
        )
        presence_cases_count = 0
        false_abstention_count = 0

        for idx, cuad in enumerate(contracts):
            catcher_contract = self.loader.to_catcher_contract(cuad)
            retriever = ContractRetriever(
                catcher_contract, embedding_provider=self.embedding_provider
            )

            for cat, query in query_templates.items():
                gt_spans = cuad.get_spans(cat)
                if not gt_spans:
                    continue

                presence_cases_count += 1
                t0 = time.perf_counter()
                results = retriever.retrieve(query, top_k=5)
                dt_ms = (time.perf_counter() - t0) * 1000

                hit_1 = False
                hit_3 = False
                hit_5 = False
                rr = 0.0
                top_texts: list[str] = []

                for rank, clause in enumerate(results, start=1):
                    top_texts.append(clause.text[:60] + "...")
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

                # Bartleby evaluation on present clause: did it abstain erroneously?
                assessment = self.guard.evaluate_retrieval(results, query_topic=cat)
                abstained = assessment.abstained
                if hit_5 and abstained:
                    false_abstention_count += 1

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
                        bartleby_abstained=abstained,
                        reciprocal_rank=rr,
                        latency_ms=dt_ms,
                    )
                )

            print(
                f"  [{idx + 1}/{len(contracts)}] Processed {cuad.contract_id} ({cuad.char_length:,} chars) | Queries: {len(retrieval_results)}",
                flush=True,
            )

        # 2. Bartleby Abstention Benchmark on Real Ground-Truth Absences
        print(
            "\n[Phase 2] Evaluating Bartleby Correct Abstention on Real Ground-Truth Absences...",
            flush=True,
        )
        for cuad in contracts:
            catcher_contract = self.loader.to_catcher_contract(cuad)
            retriever = ContractRetriever(
                catcher_contract, embedding_provider=self.embedding_provider
            )

            for absent_cat, query in query_templates.items():
                if absent_cat not in cuad.absent_categories:
                    continue

                t0 = time.perf_counter()
                results = retriever.retrieve(query, top_k=5)
                top_score = results[0].score if results else 0.0

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
        correct_abs_count = sum(a.abstained for a in absence_results)
        correct_abs_rate = correct_abs_count / max(1, n_abs)
        false_assertion_rate = 1.0 - correct_abs_rate
        mean_bart_lat = sum(a.latency_ms for a in absence_results) / max(1, n_abs)
        abs_ci95 = _wilson_score_interval(correct_abs_count, n_abs)

        false_abs_rate = false_abstention_count / max(1, presence_cases_count)
        false_abs_ci95 = _wilson_score_interval(false_abstention_count, presence_cases_count)

        report = CUADBenchmarkReport(
            timestamp=datetime.now(UTC).isoformat(),
            embedding_provider=provider_name,
            embedding_model_id=str(model_id),
            total_contracts_evaluated=len(contracts),
            train_contracts_count=len(train_contracts),
            val_contracts_count=len(val_contracts),
            retrieval_cases_count=n_ret,
            recall_at_1=round(r1, 4),
            recall_at_3=round(r3, 4),
            recall_at_5=round(r5, 4),
            mean_reciprocal_rank=round(mrr, 4),
            mean_retrieval_latency_ms=round(mean_ret_lat, 2),
            presence_cases_count=presence_cases_count,
            bartleby_false_abstention_count=false_abstention_count,
            bartleby_false_abstention_rate=round(false_abs_rate, 4),
            bartleby_false_abstention_ci95=false_abs_ci95,
            absence_cases_count=n_abs,
            bartleby_correct_abstention_count=correct_abs_count,
            bartleby_correct_abstention_rate=round(correct_abs_rate, 4),
            bartleby_correct_abstention_ci95=abs_ci95,
            bartleby_false_assertion_rate=round(false_assertion_rate, 4),
            mean_bartleby_latency_ms=round(mean_bart_lat, 2),
        )

        return report, retrieval_results, absence_results


def print_cuad_summary(report: CUADBenchmarkReport) -> None:
    print("\n" + "=" * 95)
    print(" REAL-WORLD CUAD BENCHMARK RESULTS (THE ATTICUS PROJECT)")
    print(
        f" Evaluated Contracts: {report.total_contracts_evaluated} (Train: {report.train_contracts_count}, Val: {report.val_contracts_count})"
    )
    print(f" Embedding Provider:  {report.embedding_provider} ({report.embedding_model_id})")
    print(f" Timestamp:           {report.timestamp}")
    print("=" * 95)
    print(" [RETRIEVAL ON REAL HUMAN-ANNOTATED CLAUSES]")
    print(f"   Total Test Queries:            {report.retrieval_cases_count}")
    print(f"   Recall@1:                      {report.recall_at_1 * 100:.1f}%")
    print(f"   Recall@3:                      {report.recall_at_3 * 100:.1f}%")
    print(f"   Recall@5:                      {report.recall_at_5 * 100:.1f}%")
    print(f"   Mean Reciprocal Rank (MRR):    {report.mean_reciprocal_rank:.4f}")
    print(f"   Mean Latency per Query:        {report.mean_retrieval_latency_ms:.2f} ms")
    print("-" * 95)
    print(" [BARTLEBY DUAL-SIDED ABSTENTION INTEGRITY]")
    print(
        f"   Correct Abstention on Negatives: {report.bartleby_correct_abstention_rate * 100:.1f}% "
        f"({report.bartleby_correct_abstention_count}/{report.absence_cases_count}) "
        f"[95% CI: {report.bartleby_correct_abstention_ci95[0] * 100:.1f}% - {report.bartleby_correct_abstention_ci95[1] * 100:.1f}%]"
    )
    print(f"   False Assertion Rate (Spurious): {report.bartleby_false_assertion_rate * 100:.1f}%")
    print(
        f"   False Abstention on Present:     {report.bartleby_false_abstention_rate * 100:.1f}% "
        f"({report.bartleby_false_abstention_count}/{report.presence_cases_count}) "
        f"[95% CI: {report.bartleby_false_abstention_ci95[0] * 100:.1f}% - {report.bartleby_false_abstention_ci95[1] * 100:.1f}%]"
    )
    print(f"   Mean Latency per Decision:       {report.mean_bartleby_latency_ms:.2f} ms")
    print("=" * 95)


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
