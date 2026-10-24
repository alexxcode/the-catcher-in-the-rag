import json
from pathlib import Path

with Path("evals/reports/audit_rejected_cases_report.json").open(encoding="utf-8") as f:
    data = json.load(f)

print(f"Total present cases: {data['total_present_cases']}")
print(f"Gold in top-k: {data['gold_in_topk_count']}")
print(f"  - Accepted: {data['gold_in_topk_accepted']}/{data['gold_in_topk_count']} ({data['gold_in_topk_accepted']/data['gold_in_topk_count']*100:.1f}%)")
print(f"  - Rejected: {data['gold_in_topk_rejected']}/{data['gold_in_topk_count']}\n")

miss_cases = [c for c in data["cases"] if not c["has_gold_in_topk"]]
miss_rej = [c for c in miss_cases if not c["topk_accepted"]]
print(f"Gold NOT in top-k: {len(miss_cases)}")
print(f"  - Correctly Rejected (Spurious Distractor Filtered): {len(miss_rej)}/{len(miss_cases)} ({len(miss_rej)/len(miss_cases)*100:.1f}%)")
print(f"  - Falsely Accepted Distractor: {len(miss_cases) - len(miss_rej)}/{len(miss_cases)}\n")

for i, c in enumerate(data["cases"], 1):
    if c["has_gold_in_topk"] and not c["topk_accepted"]:
        print("=" * 80)
        print(f"CASE {i}: Contract: {c['contract_id']} | Category: {c['category']}")
        print(f"Ground Truth Span (CUAD):\n  {c['gt_snippet']}\n")
        print("Candidates inspected:")
        for cand in c["candidates"]:
            print(f"  [Rank {cand['rank']}] score={cand['score']} is_gold={cand['is_gold']} LLM_rel={cand['llm_is_relevant']}")
            print(f"    Title: {cand['title']}")
            print(f"    LLM Reason: {cand['llm_reason']}")
            print(f"    Text: {cand['text_snippet'][:180]}...")
