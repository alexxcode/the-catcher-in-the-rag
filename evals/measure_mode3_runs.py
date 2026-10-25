from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from catcher_in_the_rag.bartleby.guard import BartlebyGuard
from catcher_in_the_rag.ingest.cuad_loader import CUADLoader
from catcher_in_the_rag.providers.factory import get_embedding_provider, get_llm_provider
from catcher_in_the_rag.retrieve.retriever import ContractRetriever


def run_mode3_repetition(num_runs: int = 3, sample_size: int = 15) -> None:
    loader = CUADLoader()
    contracts = loader.load_contracts(limit=sample_size)
    categories = [
        "Governing Law",
        "Payment Terms",
        "Liquidated Damages",
        "Cap On Liability",
        "Audit Rights",
        "Termination",
        "Warranties",
    ]

    emb_provider = get_embedding_provider("stub")
    llm_provider = get_llm_provider("vertex_ai")

    # Pre-build test cases and retrievers
    test_cases = []
    retrievers = {}
    for c in contracts:
        cat_contract = loader.to_catcher_contract(c)
        retriever = ContractRetriever(cat_contract, embedding_provider=emb_provider)
        retrievers[c.contract_id] = retriever
        for cat in categories:
            has_clause = c.has_clause(cat)
            gt_spans = c.get_spans(cat) if has_clause else []
            test_cases.append({
                "contract_id": c.contract_id,
                "category": cat,
                "query": f"clause regarding {cat.lower()}",
                "is_present": has_clause,
                "gt_spans": gt_spans,
            })

    total_pres = sum(1 for tc in test_cases if tc["is_present"])
    total_abs = sum(1 for tc in test_cases if not tc["is_present"])
    print(f"Loaded {len(contracts)} contracts, {len(test_cases)} cases ({total_pres} presences, {total_abs} absences).", flush=True)

    runs_data = []

    for run_idx in range(1, num_runs + 1):
        print(f"\n--- Starting Run {run_idx}/{num_runs} ---", flush=True)
        guard = BartlebyGuard(
            thresholds={"min_relevance_score": 0.50, "min_evidence_spans": 1},
            max_verification_candidates=5,
        )

        tokens_prompt_start = getattr(llm_provider, "total_prompt_tokens", 0)
        tokens_cand_start = getattr(llm_provider, "total_candidate_tokens", 0)

        t_start = time.perf_counter()
        correct_abs = 0
        false_abs = 0
        llm_accepted_on_hit = 0
        llm_rejected_on_miss = 0

        for i, tc in enumerate(test_cases):
            retriever = retrievers[tc["contract_id"]]
            scored_clauses = retriever.retrieve(tc["query"], top_k=5)
            assessment = guard.evaluate_retrieval(
                scored_clauses,
                query_topic=tc["category"],
                llm_provider=llm_provider,
            )

            if tc["is_present"]:
                gt_list = tc["gt_spans"]
                has_hit_in_top5 = any(
                    any(sc.start_char <= sp.end_char and sc.end_char >= sp.start_char for sp in gt_list)
                    for sc in scored_clauses
                )
                if assessment.abstained:
                    false_abs += 1
                else:
                    if has_hit_in_top5:
                        llm_accepted_on_hit += 1
                if not has_hit_in_top5 and assessment.abstained:
                    llm_rejected_on_miss += 1
            else:
                if assessment.abstained:
                    correct_abs += 1

            if (i + 1) % 25 == 0 or (i + 1) == len(test_cases):
                print(f"  [Run {run_idx}] Completed {i+1}/{len(test_cases)} queries...", flush=True)

        elapsed = time.perf_counter() - t_start
        mean_lat_ms = (elapsed / len(test_cases)) * 1000

        correct_abs_rate = correct_abs / total_abs
        false_abs_rate = false_abs / total_pres

        # Macro F1
        tp = total_pres - false_abs
        fn = false_abs
        fp = total_abs - correct_abs
        tn = correct_abs

        p_pres = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r_pres = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_pres = 2 * p_pres * r_pres / (p_pres + r_pres) if (p_pres + r_pres) > 0 else 0.0

        p_abs = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        r_abs = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1_abs = 2 * p_abs * r_abs / (p_abs + r_abs) if (p_abs + r_abs) > 0 else 0.0

        macro_f1 = (f1_pres + f1_abs) / 2.0

        tokens_p = getattr(llm_provider, "total_prompt_tokens", 0) - tokens_prompt_start
        tokens_c = getattr(llm_provider, "total_candidate_tokens", 0) - tokens_cand_start

        run_res = {
            "run_index": run_idx,
            "correct_abs_count": correct_abs,
            "correct_abs_rate": round(correct_abs_rate, 4),
            "false_abs_count": false_abs,
            "false_abs_rate": round(false_abs_rate, 4),
            "macro_f1": round(macro_f1, 4),
            "llm_accepted_on_hit": llm_accepted_on_hit,
            "llm_rejected_on_miss": llm_rejected_on_miss,
            "mean_latency_ms": round(mean_lat_ms, 1),
            "prompt_tokens": tokens_p,
            "candidate_tokens": tokens_c,
        }
        runs_data.append(run_res)
        print(f"Run {run_idx} finished: Correct Abs: {correct_abs}/{total_abs} ({correct_abs_rate*100:.1f}%), "
              f"False Abs: {false_abs}/{total_pres} ({false_abs_rate*100:.1f}%), "
              f"Macro F1: {macro_f1:.4f}, Hit Accept: {llm_accepted_on_hit}/16, Miss Reject: {llm_rejected_on_miss}/10, "
              f"Latency: {mean_lat_ms:.1f} ms", flush=True)

    # Compute variance / std dev across runs
    f1s = [r["macro_f1"] for r in runs_data]
    c_rates = [r["correct_abs_rate"] for r in runs_data]
    f_rates = [r["false_abs_rate"] for r in runs_data]
    lats = [r["mean_latency_ms"] for r in runs_data]

    summary = {
        "runs": runs_data,
        "metrics_summary": {
            "macro_f1": {
                "mean": round(statistics.mean(f1s), 4),
                "stdev": round(statistics.stdev(f1s) if len(f1s) > 1 else 0.0, 4),
                "values": f1s,
            },
            "correct_abstention_rate": {
                "mean": round(statistics.mean(c_rates), 4),
                "stdev": round(statistics.stdev(c_rates) if len(c_rates) > 1 else 0.0, 4),
                "values": c_rates,
            },
            "false_abstention_rate": {
                "mean": round(statistics.mean(f_rates), 4),
                "stdev": round(statistics.stdev(f_rates) if len(f_rates) > 1 else 0.0, 4),
                "values": f_rates,
            },
            "mean_latency_ms": {
                "mean": round(statistics.mean(lats), 1),
                "stdev": round(statistics.stdev(lats) if len(lats) > 1 else 0.0, 1),
                "values": lats,
            },
        }
    }

    out_path = Path("evals/reports/mode3_variance_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nVariance report saved to {out_path}", flush=True)

if __name__ == "__main__":
    run_mode3_repetition(num_runs=3, sample_size=15)
