"""Audit script to inspect all 26 present cases in CUAD benchmark.

Logs per-case:
- query topic & contract_id
- ground truth text snippet and char range
- top-5 retrieved clauses, whether each overlaps with ground truth
- LLM verification decision (is_relevant) and exact rationale
- analysis of why any gold-containing candidates were rejected
"""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from catcher_in_the_rag.ingest.cuad_loader import CUADLoader
from catcher_in_the_rag.providers.factory import get_embedding_provider, get_llm_provider
from catcher_in_the_rag.retrieve.retriever import ContractRetriever


class RelevanceVerification(BaseModel):
    is_relevant: bool = Field(..., description="Whether the excerpt addresses the topic")
    reason: str = Field(..., description="Brief rationale")


def audit_present_cases(sample_size: int = 15, max_candidates: int = 5) -> dict[str, Any]:
    loader = CUADLoader()
    contracts = loader.load_contracts(limit=sample_size)
    if not contracts:
        raise RuntimeError("No CUAD contracts loaded.")

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
    llm_provider = get_llm_provider(provider_type="vertex_ai")

    cases_log: list[dict[str, Any]] = []
    gold_in_topk_count = 0
    gold_in_topk_accepted = 0
    gold_in_topk_rejected = 0

    category_descriptions = {
        "Governing Law": "Governing law, jurisdiction, choice of law, or forum selection",
        "Payment Terms": "Payment terms, invoice payment schedules, late fees, or billing procedures",
        "Liquidated Damages": "Liquidated damages, predetermined damage amounts, or penalty fees for default",
        "Cap On Liability": "Cap or limitation on liability, including aggregate monetary liability limits or exclusion/waiver of consequential, punitive, or indirect damages",
        "Audit Rights": "Audit or inspection rights regarding books, records, facilities, or work performance",
        "Termination": "Termination rights, termination for convenience, termination for cause, or notice periods",
        "Warranties": "Warranties, representations, disclaimers of warranties, or covenants",
    }

    for c in contracts:
        cat_contract = loader.to_catcher_contract(c)
        retriever = ContractRetriever(cat_contract, embedding_provider=emb_provider)

        for cat in categories:
            if not c.has_clause(cat):
                continue

            gt_spans = c.get_spans(cat)
            query = f"clause regarding {cat.lower()}"
            scored_clauses = retriever.retrieve(query, top_k=max_candidates)

            # Check overlap for each candidate
            candidate_details: list[dict[str, Any]] = []
            has_gold_in_topk = False
            topk_accepted = False

            topic_desc = category_descriptions.get(cat, cat)

            for rank, cand in enumerate(scored_clauses, start=1):
                is_gold_candidate = any(
                    max(cand.start_char, gt.start_char) < min(cand.end_char, gt.end_char)
                    for gt in gt_spans
                )
                if is_gold_candidate:
                    has_gold_in_topk = True

                prompt = (
                    f"You are a strict legal contract auditor. Determine whether the provided contract passage "
                    f"actually addresses or contains legal terms regarding the requested topic.\n\n"
                    f"Requested Topic: {cat} ({topic_desc})\n"
                    f"Contract Clause Title: {cand.title}\n"
                    f'Contract Clause Text:\n"""{cand.text[:2500]}"""\n\n'
                    f"Return is_relevant=true if this passage specifically and directly governs, limits, or sets obligations/rights "
                    f"regarding '{topic_desc}'. Return is_relevant=false if the passage is unrelated, merely mentions a tangential word, "
                    f"or does not contain substantive terms on this topic."
                )

                decision = llm_provider.extract_structured(prompt, RelevanceVerification)
                is_rel = bool(decision.is_relevant)
                if is_rel:
                    topk_accepted = True

                candidate_details.append({
                    "rank": rank,
                    "clause_id": cand.clause_id,
                    "title": cand.title,
                    "score": round(cand.score, 4),
                    "is_gold": is_gold_candidate,
                    "llm_is_relevant": is_rel,
                    "llm_reason": decision.reason,
                    "text_snippet": cand.text[:300].replace("\n", " "),
                })

            if has_gold_in_topk:
                gold_in_topk_count += 1
                if topk_accepted:
                    gold_in_topk_accepted += 1
                else:
                    gold_in_topk_rejected += 1

            cases_log.append({
                "contract_id": c.contract_id,
                "category": cat,
                "has_gold_in_topk": has_gold_in_topk,
                "topk_accepted": topk_accepted,
                "gt_snippet": gt_spans[0].text[:300].replace("\n", " ") if gt_spans else "",
                "candidates": candidate_details,
            })

    summary = {
        "total_present_cases": len(cases_log),
        "gold_in_topk_count": gold_in_topk_count,
        "gold_in_topk_accepted": gold_in_topk_accepted,
        "gold_in_topk_rejected": gold_in_topk_rejected,
        "cases": cases_log,
    }

    out_file = Path("evals/reports/audit_rejected_cases_report.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAudit complete. Processed {len(cases_log)} present cases.")
    print(f"Gold in top-k: {gold_in_topk_count} (Accepted: {gold_in_topk_accepted}, Rejected: {gold_in_topk_rejected})")
    return summary


if __name__ == "__main__":
    audit_present_cases()
