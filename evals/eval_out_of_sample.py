from __future__ import annotations

import json
import math
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from catcher_in_the_rag.bartleby.guard import BartlebyGuard, load_bartleby_thresholds  # noqa: E402
from catcher_in_the_rag.ingest.cuad_loader import CUADLoader  # noqa: E402
from catcher_in_the_rag.providers.factory import get_embedding_provider, get_llm_provider  # noqa: E402
from catcher_in_the_rag.retrieve.retriever import ContractRetriever  # noqa: E402
from evals.provenance import get_provenance_metadata  # noqa: E402


def wilson_interval(successes: int, total: int, z: float = 1.95996) -> tuple[float, float]:
    """95% two-sided Wilson score interval."""
    if total <= 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1.0 + (z**2) / total
    center = (p + (z**2) / (2 * total)) / denom
    margin = (z / denom) * math.sqrt((p * (1.0 - p) / total) + ((z**2) / (4 * (total**2))))
    return (max(0.0, center - margin), min(1.0, center + margin))


OOS_CATEGORIES = (
    "Governing Law",
    "Audit Rights",
    "Termination For Convenience",
    "Cap On Liability",
    "Liquidated Damages",
    "Warranty Duration",
    "Price Restrictions",
    "Non-Compete",
    "Exclusivity",
    "Anti-Assignment",
    "Insurance",
    "Uncapped Liability",
)


def run_retrieval_ablation(
    start_contract: int = 15,
    end_contract: int = 50,
    embedding_provider_name: str = "vertex_ai",
    output_path: str = "evals/reports/retrieval_ablation_oos_report.json",
) -> dict[str, Any]:
    """bm25 / dense / hybrid Hit@1 and Hit@5 on out-of-sample presence cases. No LLM calls.

    One retrieve() per case returns bm25_score, dense_score and the hybrid score for every
    clause; each mode is that list re-ranked by its own score only. Ties are broken by
    order of appearance in the document (start_char), never by another mode's score.
    """
    provenance = get_provenance_metadata(
        {"evaluation_type": "retrieval_ablation_oos", "embedding_provider": embedding_provider_name}
    )
    loader = CUADLoader()
    contracts = loader.load_contracts(limit=end_contract)[start_contract:end_contract]
    if not contracts:
        raise RuntimeError("No out-of-sample contracts loaded.")

    emb_provider = get_embedding_provider(embedding_provider_name)
    modes = {"bm25": "bm25_score", "dense": "dense_score", "hybrid": "score"}
    hits = {m: {"hit_at_1": 0, "hit_at_5": 0} for m in modes}
    total = 0

    for c in contracts:
        retriever = ContractRetriever(
            loader.to_catcher_contract(c), embedding_provider=emb_provider
        )
        for cat in OOS_CATEGORIES:
            if not c.has_clause(cat):
                continue
            total += 1
            gt = c.get_spans(cat)
            scored = retriever.retrieve(
                f"clause regarding {cat.lower()}", top_k=len(retriever.clauses)
            )
            for mode, attr in modes.items():
                ranked = sorted(scored, key=lambda sc, a=attr: (-getattr(sc, a), sc.start_char))
                is_gold = [
                    any(sc.start_char <= sp.end_char and sc.end_char >= sp.start_char for sp in gt)
                    for sc in ranked[:5]
                ]
                hits[mode]["hit_at_1"] += int(bool(is_gold[:1] and is_gold[0]))
                hits[mode]["hit_at_5"] += int(any(is_gold))

    metrics: dict[str, Any] = {}
    for mode, h in hits.items():
        metrics[mode] = {
            k: {
                "count": v,
                "total": total,
                "rate": round(v / total, 4) if total else 0.0,
                "wilson_ci95": [round(x, 4) for x in wilson_interval(v, total)],
            }
            for k, v in h.items()
        }

    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "git_commit_sha": provenance["git_commit_sha"],
        "git_dirty": provenance["git_dirty"],
        "command": provenance["command"],
        "evaluation_type": "Retrieval ablation (bm25 / dense / hybrid) on out-of-sample presence cases",
        "contract_slice": f"CUAD-{start_contract + 1:04d} to CUAD-{end_contract:04d} ({len(contracts)} contracts)",
        "embedding_provider": embedding_provider_name,
        "embedding_model_id": getattr(emb_provider, "model_id", "stub"),
        "presence_cases": total,
        "metrics": metrics,
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    for mode, m in metrics.items():
        print(
            f" {mode:<7} Hit@1 {m['hit_at_1']['count']}/{total}  Hit@5 {m['hit_at_5']['count']}/{total}",
            flush=True,
        )
    return report


def run_out_of_sample_evaluation(
    start_contract: int = 15,
    end_contract: int = 50,
    categories: list[str] | None = None,
    output_path: str = "evals/reports/out_of_sample_cuad_report.json",
    embedding_provider_name: str = "stub",
    llm_provider_name: str = "vertex_ai",
    tau: float | None = None,
) -> dict[str, Any]:
    initial_provenance = get_provenance_metadata(
        {
            "dataset": "CUAD_v1",
            "sample": f"contracts_{start_contract}_to_{end_contract}",
            "categories_count": len(categories) if categories else 12,
            "evaluation_type": "out_of_sample_held_out",
        }
    )
    loader = CUADLoader()
    all_contracts = loader.load_contracts(limit=end_contract)
    out_of_sample_contracts = all_contracts[start_contract:end_contract]

    if not out_of_sample_contracts:
        raise RuntimeError("No out-of-sample contracts loaded.")

    official_descriptions = loader.get_category_descriptions()

    if categories is None:
        categories = list(OOS_CATEGORIES)

    emb_provider = get_embedding_provider(embedding_provider_name)
    llm_provider = get_llm_provider(llm_provider_name)

    test_cases = []
    retrievers = {}

    for c in out_of_sample_contracts:
        cat_contract = loader.to_catcher_contract(c)
        retriever = ContractRetriever(cat_contract, embedding_provider=emb_provider)
        retrievers[c.contract_id] = retriever
        for cat in categories:
            has_clause = c.has_clause(cat)
            gt_spans = c.get_spans(cat) if has_clause else []
            test_cases.append(
                {
                    "contract_id": c.contract_id,
                    "category": cat,
                    "query": f"clause regarding {cat.lower()}",
                    "is_present": has_clause,
                    "gt_spans": gt_spans,
                }
            )

    total_pres = sum(1 for tc in test_cases if tc["is_present"])
    total_abs = sum(1 for tc in test_cases if not tc["is_present"])

    print("\n" + "=" * 70, flush=True)
    print(" OUT-OF-SAMPLE CUAD EVALUATION (HELD-OUT CONTRACTS)", flush=True)
    print(
        f" Contracts Evaluated: {len(out_of_sample_contracts)} (CUAD-{start_contract + 1:04d} to CUAD-{end_contract:04d})",
        flush=True,
    )
    print(f" Categories Evaluated: {len(categories)} categories", flush=True)
    print(
        f" Total Queries: {len(test_cases)} ({total_pres} Present, {total_abs} Absent)", flush=True
    )
    print(" Prompt Grounding: 100% Official Atticus Dataset Definitions", flush=True)
    print("=" * 70 + "\n", flush=True)

    guard = BartlebyGuard(
        thresholds=load_bartleby_thresholds(tau=tau),
        max_verification_candidates=5,
        category_descriptions=official_descriptions,
    )

    t_start = time.perf_counter()
    tokens_p_start = getattr(llm_provider, "total_prompt_tokens", 0)
    tokens_c_start = getattr(llm_provider, "total_candidate_tokens", 0)

    correct_abs = 0
    false_abs = 0
    llm_accepted_on_hit = 0
    llm_rejected_on_miss = 0
    gold_in_top5_count = 0
    miss_top5_count = 0

    for i, tc in enumerate(test_cases):
        retriever = retrievers[tc["contract_id"]]
        scored_clauses = retriever.retrieve(tc["query"], top_k=5)
        assessment = guard.evaluate_retrieval(
            scored_clauses,
            query_topic=tc["category"],
            llm_provider=llm_provider,
            query=tc["query"],
            contract_id=tc["contract_id"],
        )

        if tc["is_present"]:
            gt_list = tc["gt_spans"]
            has_hit_in_top5 = any(
                any(sc.start_char <= sp.end_char and sc.end_char >= sp.start_char for sp in gt_list)
                for sc in scored_clauses
            )
            if has_hit_in_top5:
                gold_in_top5_count += 1
            else:
                miss_top5_count += 1

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
            print(f"  Processed {i + 1}/{len(test_cases)} out-of-sample queries...", flush=True)

    elapsed = time.perf_counter() - t_start
    mean_lat_ms = (elapsed / len(test_cases)) * 1000

    correct_abs_rate = correct_abs / total_abs if total_abs > 0 else 0.0
    false_abs_rate = false_abs / total_pres if total_pres > 0 else 0.0

    wilson_correct_abs = wilson_interval(correct_abs, total_abs)
    wilson_false_abs = wilson_interval(false_abs, total_pres)

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

    tokens_p = getattr(llm_provider, "total_prompt_tokens", 0) - tokens_p_start
    tokens_c = getattr(llm_provider, "total_candidate_tokens", 0) - tokens_c_start

    provenance = initial_provenance

    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "git_commit_sha": provenance["git_commit_sha"],
        "git_dirty": provenance["git_dirty"],
        "command": provenance["command"],
        "tau_override": tau,
        "evaluation_type": "Out-of-sample held-out evaluation with frozen prompt and official Atticus category definitions",
        "contract_slice": f"CUAD-{start_contract + 1:04d} to CUAD-{end_contract:04d} ({len(out_of_sample_contracts)} contracts)",
        "num_contracts": len(out_of_sample_contracts),
        "num_categories": len(categories),
        "total_queries_evaluated": len(test_cases),
        "presence_cases": total_pres,
        "absence_cases": total_abs,
        "metrics": {
            "correct_abstention_on_negatives": {
                "count": correct_abs,
                "total": total_abs,
                "rate": round(correct_abs_rate, 4),
                "wilson_ci95": [round(wilson_correct_abs[0], 4), round(wilson_correct_abs[1], 4)],
            },
            "false_abstention_on_positives": {
                "count": false_abs,
                "total": total_pres,
                "rate": round(false_abs_rate, 4),
                "wilson_ci95": [round(wilson_false_abs[0], 4), round(wilson_false_abs[1], 4)],
            },
            "macro_f1": round(macro_f1, 4),
            "retrieval_conditioned_verifier": {
                "gold_in_top5_count": gold_in_top5_count,
                "llm_accepted_on_hit": llm_accepted_on_hit,
                "hit_acceptance_rate": round(llm_accepted_on_hit / gold_in_top5_count, 4)
                if gold_in_top5_count > 0
                else 0.0,
                "miss_top5_count": miss_top5_count,
                "llm_rejected_on_miss": llm_rejected_on_miss,
                "miss_rejection_rate": round(llm_rejected_on_miss / miss_top5_count, 4)
                if miss_top5_count > 0
                else 0.0,
            },
            "performance": {
                "mean_latency_ms": round(mean_lat_ms, 1),
                "prompt_tokens": tokens_p,
                "candidate_tokens": tokens_c,
            },
        },
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70, flush=True)
    print(" OUT-OF-SAMPLE EVALUATION RESULTS", flush=True)
    print("=" * 70, flush=True)
    print(
        f" Correct Abstention (Absence):  {correct_abs_rate * 100:.1f}% ({correct_abs}/{total_abs}) "
        f"[95% Wilson CI: {wilson_correct_abs[0] * 100:.1f}% - {wilson_correct_abs[1] * 100:.1f}%]",
        flush=True,
    )
    print(
        f" False Abstention (Presence):   {false_abs_rate * 100:.1f}% ({false_abs}/{total_pres}) "
        f"[95% Wilson CI: {wilson_false_abs[0] * 100:.1f}% - {wilson_false_abs[1] * 100:.1f}%]",
        flush=True,
    )
    print(f" Macro F1 Score:                {macro_f1:.4f}", flush=True)
    print(
        f" Verifier on Hit:               {llm_accepted_on_hit}/{gold_in_top5_count} ({llm_accepted_on_hit / max(1, gold_in_top5_count) * 100:.1f}%)",
        flush=True,
    )
    print(
        f" Verifier on Miss:              {llm_rejected_on_miss}/{miss_top5_count} ({llm_rejected_on_miss / max(1, miss_top5_count) * 100:.1f}%)",
        flush=True,
    )
    print(f" Mean Latency:                  {mean_lat_ms:.1f} ms", flush=True)
    print("=" * 70 + "\n", flush=True)
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=15)
    parser.add_argument("--end", type=int, default=50)
    parser.add_argument("--embedding-provider", type=str, default="stub")
    parser.add_argument("--llm-provider", type=str, default="vertex_ai")
    parser.add_argument(
        "--output", type=str, default="evals/reports/out_of_sample_cuad_report.json"
    )
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument(
        "--retrieval-ablation",
        action="store_true",
        help="bm25/dense/hybrid Hit@k on presence cases only (no LLM calls)",
    )
    args = parser.parse_args()
    if args.retrieval_ablation:
        out = args.output
        if out == "evals/reports/out_of_sample_cuad_report.json":
            out = "evals/reports/retrieval_ablation_oos_report.json"
        run_retrieval_ablation(args.start, args.end, args.embedding_provider, out)
        sys.exit(0)
    run_out_of_sample_evaluation(
        start_contract=args.start,
        end_contract=args.end,
        embedding_provider_name=args.embedding_provider,
        llm_provider_name=args.llm_provider,
        output_path=args.output,
        tau=args.tau,
    )
