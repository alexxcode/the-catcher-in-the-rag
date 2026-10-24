"""Empirical evaluation harness for Bartleby Condition 3 LLM Verification Guard.

Measures the impact of LLM semantic verification on real CUAD SEC contracts:
1. Pure Vector Hybrid Retrieval (tau=0.50 without LLM)
2. Pure Vector Hybrid Retrieval (tau=0.70 without LLM)
3. Vector Hybrid + Bartleby LLM Verification (Condition 3 with Gemini)

Metrics measured:
- Correct Abstention Rate on true absences
- False Abstention Rate on present valid clauses
- Macro F1 across both classes
- Query latency (ms) and measured cost per 1k queries ($) based on actual tokens
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path.cwd()))

from catcher_in_the_rag.bartleby.guard import BartlebyGuard
from catcher_in_the_rag.ingest.cuad_loader import CUADLoader
from catcher_in_the_rag.providers.factory import get_embedding_provider, get_llm_provider
from catcher_in_the_rag.retrieve.retriever import ContractRetriever
from evals.provenance import get_provenance_metadata


@dataclass
class GuardEvaluationModeResult:
    mode_name: str
    description: str
    relevance_threshold: float
    use_llm_verification: bool
    total_queries: int
    total_absences: int
    correct_abstentions: int
    correct_abstention_rate: float
    total_presences: int
    false_abstentions: int
    false_abstention_rate: float
    precision_abstain: float
    recall_abstain: float
    f1_abstain: float
    f1_presence: float
    macro_f1: float
    mean_latency_ms: float
    estimated_cost_usd_per_1k: float
    prompt_tokens: int = 0
    candidate_tokens: int = 0


@dataclass
class BartlebyLLMVerificationReport:
    timestamp: str
    git_commit_sha: str
    git_dirty: bool
    command: str
    total_contracts_evaluated: int
    llm_provider_used: str
    modes: list[GuardEvaluationModeResult]
    findings_summary: str


def run_bartleby_llm_eval(
    sample_size: int = 15,
    output_path: str = "evals/reports/bartleby_llm_verification_report.json",
    llm_provider_type: str = "vertex_ai",
    allow_stub: bool = False,
) -> BartlebyLLMVerificationReport:
    # Fail fast if attempting to measure LLM verification with stub without explicit override
    if llm_provider_type == "stub" and not allow_stub:
        raise ValueError(
            "Cannot evaluate Bartleby Condition 3 LLM verification with stub provider because stub always returns True "
            "and does not perform semantic discrimination. Pass --llm-provider vertex_ai (or set --allow-stub for offline testing)."
        )

    loader = CUADLoader()
    contracts = loader.load_contracts(limit=sample_size)
    if not contracts:
        raise RuntimeError("No CUAD contracts loaded. Run `make download-data` first.")

    categories = [
        "Governing Law",
        "Payment Terms",
        "Liquidated Damages",
        "Cap On Liability",
        "Audit Rights",
        "Termination",
        "Warranties",
    ]

    emb_provider = get_embedding_provider()
    llm_provider = get_llm_provider(provider_type=llm_provider_type)

    if llm_provider_type != "stub":
        health = llm_provider.check_model_health()
        print(f"Connected to LLM provider '{llm_provider_type}': {health}")

    # Build query test cases: (contract, category, is_present, query)
    test_cases: list[dict[str, Any]] = []
    retrievers: dict[str, ContractRetriever] = {}

    for c in contracts:
        cat_contract = loader.to_catcher_contract(c)
        retriever = ContractRetriever(cat_contract, embedding_provider=emb_provider)
        retrievers[c.contract_id] = retriever
        for cat in categories:
            has_clause = c.has_clause(cat)
            test_cases.append(
                {
                    "contract_id": c.contract_id,
                    "category": cat,
                    "query": f"clause regarding {cat.lower()}",
                    "is_present": has_clause,
                }
            )

    print(
        f"Evaluating Bartleby Verification on {len(contracts)} CUAD contracts "
        f"({len(test_cases)} query evaluations, LLM: {llm_provider_type})"
    )

    modes_to_test = [
        {
            "name": "Pure Vector Hybrid (tau=0.50)",
            "description": "Baseline transfer: hybrid retrieval thresholded at synthetic tau=0.50 without LLM",
            "threshold": 0.50,
            "use_llm": False,
        },
        {
            "name": "Pure Vector Hybrid (tau=0.70)",
            "description": "Recalibrated vector: hybrid retrieval thresholded at tau=0.70 without LLM",
            "threshold": 0.70,
            "use_llm": False,
        },
        {
            "name": "Bartleby Condition 3 (tau=0.50 + LLM Verification)",
            "description": "Hybrid retrieval with tau=0.50 passed to Condition 3 LLM Semantic Verification",
            "threshold": 0.50,
            "use_llm": True,
        },
    ]

    mode_results: list[GuardEvaluationModeResult] = []

    for mode in modes_to_test:
        guard = BartlebyGuard(
            thresholds={"min_relevance_score": mode["threshold"], "min_evidence_spans": 1}
        )
        t_start = time.perf_counter()

        # Track tokens consumed by LLM calls in this mode
        prompt_tokens_before = getattr(llm_provider, "total_prompt_tokens", 0)
        candidate_tokens_before = getattr(llm_provider, "total_candidate_tokens", 0)

        correct_abs = 0
        tot_abs = 0
        false_abs = 0
        tot_pres = 0

        for tc in test_cases:
            retriever = retrievers[tc["contract_id"]]
            scored_clauses = retriever.retrieve(tc["query"], top_k=3)

            llm_to_pass = llm_provider if mode["use_llm"] else None
            assessment = guard.evaluate_retrieval(
                scored_clauses,
                query_topic=tc["category"],
                llm_provider=llm_to_pass,
            )

            if tc["is_present"]:
                tot_pres += 1
                if assessment.abstained:
                    false_abs += 1
            else:
                tot_abs += 1
                if assessment.abstained:
                    correct_abs += 1

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        mean_lat = round(elapsed_ms / max(1, len(test_cases)), 2)

        p_tokens = getattr(llm_provider, "total_prompt_tokens", 0) - prompt_tokens_before
        c_tokens = getattr(llm_provider, "total_candidate_tokens", 0) - candidate_tokens_before

        if mode["use_llm"] and (p_tokens > 0 or c_tokens > 0):
            # Gemini 3.5 Flash: $0.075 / 1M prompt, $0.30 / 1M completion
            measured_cost = (p_tokens * 0.075 / 1_000_000) + (c_tokens * 0.30 / 1_000_000)
            cost_per_1k = round((measured_cost / max(1, len(test_cases))) * 1000, 4)
        else:
            cost_per_1k = 0.00

        # Class 1: Absence (Abstain)
        tp_a = correct_abs
        fp_a = false_abs
        fn_a = tot_abs - correct_abs
        p_a = tp_a / (tp_a + fp_a) if (tp_a + fp_a) > 0 else 0.0
        r_a = tp_a / (tp_a + fn_a) if (tp_a + fn_a) > 0 else 0.0
        f1_a = (2 * p_a * r_a) / (p_a + r_a) if (p_a + r_a) > 0 else 0.0

        # Class 0: Presence (Accept)
        tp_p = tot_pres - false_abs
        fp_p = fn_a
        fn_p = fp_a
        p_p = tp_p / (tp_p + fp_p) if (tp_p + fp_p) > 0 else 0.0
        r_p = tp_p / (tp_p + fn_p) if (tp_p + fn_p) > 0 else 0.0
        f1_p = (2 * p_p * r_p) / (p_p + r_p) if (p_p + r_p) > 0 else 0.0

        macro_f1 = round((f1_a + f1_p) / 2.0, 4)
        c_rate = round(correct_abs / max(1, tot_abs), 4)
        f_rate = round(false_abs / max(1, tot_pres), 4)

        mode_results.append(
            GuardEvaluationModeResult(
                mode_name=mode["name"],
                description=mode["description"],
                relevance_threshold=mode["threshold"],
                use_llm_verification=mode["use_llm"],
                total_queries=len(test_cases),
                total_absences=tot_abs,
                correct_abstentions=correct_abs,
                correct_abstention_rate=c_rate,
                total_presences=tot_pres,
                false_abstentions=false_abs,
                false_abstention_rate=f_rate,
                precision_abstain=round(p_a, 4),
                recall_abstain=round(r_a, 4),
                f1_abstain=round(f1_a, 4),
                f1_presence=round(f1_p, 4),
                macro_f1=macro_f1,
                mean_latency_ms=mean_lat,
                estimated_cost_usd_per_1k=cost_per_1k,
                prompt_tokens=p_tokens,
                candidate_tokens=c_tokens,
            )
        )

    provenance = get_provenance_metadata({"eval": "Bartleby LLM Verification"})

    if llm_provider_type == "stub":
        summary = (
            "[HYPOTHESIS ONLY - SIMULATED STUB] Mode 3 ran with stub provider, which returns True unconditionally. "
            "Mode 3 results are identical to Mode 1 (tau=0.50). Real semantic discrimination must be evaluated "
            "against live Vertex AI (`--llm-provider vertex_ai`)."
        )
    else:
        summary = (
            f"Evaluated 3 guard architectures on {len(contracts)} authentic CUAD contracts ({len(test_cases)} cases) "
            f"using live Vertex AI ({llm_provider_type}, model={getattr(llm_provider, 'model_id', 'unknown')}). "
            f"Pure vector retrieval at tau=0.50 achieves {mode_results[0].correct_abstention_rate * 100:.1f}% correct abstention "
            f"(false abs: {mode_results[0].false_abstention_rate * 100:.1f}%, Macro F1: {mode_results[0].macro_f1:.4f}). "
            f"Raising threshold to tau=0.70 yields {mode_results[1].correct_abstention_rate * 100:.1f}% correct abstention "
            f"at cost of {mode_results[1].false_abstention_rate * 100:.1f}% false abstentions (Macro F1: {mode_results[1].macro_f1:.4f}). "
            f"Condition 3 LLM verification achieves {mode_results[2].correct_abstention_rate * 100:.1f}% correct abstention "
            f"with {mode_results[2].false_abstention_rate * 100:.1f}% false abstentions (Macro F1: {mode_results[2].macro_f1:.4f}), "
            f"with measured latency {mode_results[2].mean_latency_ms:.1f}ms and measured cost ${mode_results[2].estimated_cost_usd_per_1k:.2f}/1k queries."
        )

    report = BartlebyLLMVerificationReport(
        timestamp=datetime.now(UTC).isoformat(),
        git_commit_sha=provenance["git_commit_sha"],
        git_dirty=provenance["git_dirty"],
        command=provenance["command"],
        total_contracts_evaluated=len(contracts),
        llm_provider_used=llm_provider_type,
        modes=mode_results,
        findings_summary=summary,
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2)

    print("=" * 80)
    print(" BARTLEBY LLM VERIFICATION EVALUATION REPORT")
    print(f" LLM Provider: {llm_provider_type} | Contracts: {len(contracts)}")
    print("=" * 80)
    for m in mode_results:
        tokens_info = (
            f" | Tokens: {m.prompt_tokens}in/{m.candidate_tokens}out"
            if m.use_llm_verification
            else ""
        )
        print(
            f"Mode: {m.mode_name} | Correct Abs: {m.correct_abstention_rate * 100:.1f}% | "
            f"False Abs: {m.false_abstention_rate * 100:.1f}% | Macro F1: {m.macro_f1:.4f} | "
            f"Latency: {m.mean_latency_ms}ms | Cost/1k: ${m.estimated_cost_usd_per_1k:.4f}{tokens_info}"
        )
    print("=" * 80)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Bartleby LLM Verification")
    parser.add_argument("--sample-size", type=int, default=15, help="Number of contracts")
    parser.add_argument(
        "--llm-provider",
        type=str,
        default="vertex_ai",
        help="LLM provider (default: vertex_ai; must pass --allow-stub for stub)",
    )
    parser.add_argument(
        "--allow-stub",
        action="store_true",
        help="Allow running with stub provider (offline simulation only; results are hypothetical)",
    )
    parser.add_argument(
        "--output", type=str, default="evals/reports/bartleby_llm_verification_report.json"
    )
    args = parser.parse_args()
    run_bartleby_llm_eval(
        sample_size=args.sample_size,
        output_path=args.output,
        llm_provider_type=args.llm_provider,
        allow_stub=args.allow_stub,
    )
