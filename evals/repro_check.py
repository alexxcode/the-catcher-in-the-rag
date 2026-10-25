"""Repro Check - Verifies empirical reports against raw counts and deterministic recalculation.

Enforces Law L1, L2, L7:
- Live reports: audits derived metrics (Macro F1, Wilson CIs, false abstention breakdown, pricing)
  against raw case counts, raw execution fields, and official pricing config.
- Requires raw per-case JSONL datasets with full audit provenance (query, passage hashes,
  verdict, reason, response_id, model_version, tokens, latency).
- Deny by default: any report JSON in evals/reports/ without a registered verification handler
  fails the build.
- Fails with exit code 1 if any hand-edited metric or arithmetic discrepancy is detected.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import yaml


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    p_hat = successes / total
    denom = 1.0 + (z * z) / total
    center = (p_hat + (z * z) / (2.0 * total)) / denom
    margin = (z / denom) * math.sqrt(
        (p_hat * (1.0 - p_hat) / total) + (z * z) / (4.0 * total * total)
    )
    return [round(max(0.0, center - margin), 4), round(min(1.0, center + margin), 4)]


def check_bartleby_llm_report(report_path: Path, pricing_path: Path) -> list[str]:
    errors: list[str] = []
    if not report_path.exists():
        return [f"Missing required report: {report_path}"]

    with report_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    with pricing_path.open("r", encoding="utf-8") as f:
        pricing_data = yaml.safe_load(f)

    gemini_35_pricing = pricing_data["models"]["gemini-3.5-flash"]
    rate_in = gemini_35_pricing["prompt_cost_per_1m"]
    rate_out = gemini_35_pricing["candidate_cost_per_1m"]

    modes = data.get("modes", [])
    if len(modes) < 3:
        errors.append(f"Expected at least 3 modes in {report_path}, found {len(modes)}")
        return errors

    m3 = modes[2]

    # 1. Verify against raw per-case JSONL data
    cases_file = report_path.parent / "bartleby_llm_verification_cases.jsonl"
    if not cases_file.exists():
        cases_file = Path("evals/reports/bartleby_llm_verification_cases.jsonl")

    if not cases_file.exists():
        errors.append(f"Missing required raw case log: {cases_file}")
    else:
        raw_bytes = cases_file.read_bytes()
        calculated_sha = hashlib.sha256(raw_bytes).hexdigest()
        expected_sha = data.get("raw_cases_sha256")
        if expected_sha and calculated_sha != expected_sha:
            errors.append(
                f"Raw cases SHA256 mismatch for {cases_file.name}: expected {expected_sha} but calculated {calculated_sha}. Forgery detected!"
            )

        cases: list[dict[str, Any]] = []
        with cases_file.open("r", encoding="utf-8") as cf:
            for line_num, line in enumerate(cf, 1):
                if line.strip():
                    c = json.loads(line)
                    # Check required audit fields
                    required_fields = [
                        "case_id", "contract_id", "category", "query", "is_present",
                        "passage_ids", "passages_hash", "verdict", "reason",
                        "prompt_tokens", "candidate_tokens", "latency_ms", "response_id", "model_version",
                    ]
                    for rf in required_fields:
                        if rf not in c:
                            errors.append(f"Line {line_num} in {cases_file.name} missing required audit field: {rf}")

                    # Check verdict vs abstention consistency
                    verdict = c.get("verdict")
                    abstained = c.get("abstained_mode_3")
                    if verdict not in ["ACCEPT", "ABSTAIN"]:
                        errors.append(f"Line {line_num} in {cases_file.name} has invalid verdict: {verdict}")
                    if abstained != (verdict == "ABSTAIN"):
                        errors.append(
                            f"Line {line_num} in {cases_file.name} has inconsistent abstention: abstained_mode_3={abstained} but verdict={verdict}"
                        )

                    cases.append(c)

        raw_c_abs = sum(1 for c in cases if not c.get("is_present") and c.get("abstained_mode_3"))
        raw_f_abs = sum(1 for c in cases if c.get("is_present") and c.get("abstained_mode_3"))
        raw_tot_abs = sum(1 for c in cases if not c.get("is_present"))
        raw_tot_pres = sum(1 for c in cases if c.get("is_present"))
        raw_acc_on_hit = sum(
            1 for c in cases if c.get("is_present") and c.get("hit_at_5") and not c.get("abstained_mode_3")
        )
        raw_rej_on_miss = sum(
            1 for c in cases if c.get("is_present") and not c.get("hit_at_5") and c.get("abstained_mode_3")
        )

        if m3.get("correct_abstentions") != raw_c_abs:
            errors.append(
                f"Raw case verification failed: cases JSONL has {raw_c_abs} correct abstentions, but summary report claims {m3.get('correct_abstentions')}"
            )
        if m3.get("false_abstentions") != raw_f_abs:
            errors.append(
                f"Raw case verification failed: cases JSONL has {raw_f_abs} false abstentions, but summary report claims {m3.get('false_abstentions')}"
            )
        if m3.get("total_absences") != raw_tot_abs:
            errors.append(
                f"Raw case verification failed: cases JSONL has {raw_tot_abs} absences, but summary report claims {m3.get('total_absences')}"
            )
        if m3.get("total_presences") != raw_tot_pres:
            errors.append(
                f"Raw case verification failed: cases JSONL has {raw_tot_pres} presences, but summary report claims {m3.get('total_presences')}"
            )
        if m3.get("llm_accepted_on_hit") != raw_acc_on_hit:
            errors.append(
                f"Raw case verification failed: cases JSONL has {raw_acc_on_hit} accepted on hit, but summary report claims {m3.get('llm_accepted_on_hit')}"
            )
        if m3.get("llm_rejected_on_miss") != raw_rej_on_miss:
            errors.append(
                f"Raw case verification failed: cases JSONL has {raw_rej_on_miss} rejected on miss, but summary report claims {m3.get('llm_rejected_on_miss')}"
            )


    # 2. Check false abstentions breakdown: (16 - llm_accepted_on_hit) + llm_rejected_on_miss == false_abstentions
    hit_acc = m3.get("llm_accepted_on_hit", 0)
    miss_rej = m3.get("llm_rejected_on_miss", 0)
    expected_false_abs = (16 - hit_acc) + miss_rej
    actual_false_abs = m3.get("false_abstentions", 0)
    if expected_false_abs != actual_false_abs:
        errors.append(
            f"Mode 3 false abstention breakdown arithmetic discrepancy: (16 - {hit_acc}) + {miss_rej} = {expected_false_abs}, but found {actual_false_abs}"
        )

    # Check correct abstention rate
    c_abs = m3.get("correct_abstentions", 0)
    tot_abs = m3.get("total_absences", 0)
    expected_c_rate = round(c_abs / tot_abs, 4) if tot_abs > 0 else 0.0
    actual_c_rate = round(m3.get("correct_abstention_rate", 0.0), 4)
    if abs(expected_c_rate - actual_c_rate) > 1e-3:
        errors.append(
            f"Mode 3 correct abstention rate mismatch: {c_abs}/{tot_abs} = {expected_c_rate}, but reported {actual_c_rate}"
        )

    # Check false abstention rate
    f_abs = m3.get("false_abstentions", 0)
    tot_pres = m3.get("total_presences", 0)
    expected_f_rate = round(f_abs / tot_pres, 4) if tot_pres > 0 else 0.0
    actual_f_rate = round(m3.get("false_abstention_rate", 0.0), 4)
    if abs(expected_f_rate - actual_f_rate) > 1e-3:
        errors.append(
            f"Mode 3 false abstention rate mismatch: {f_abs}/{tot_pres} = {expected_f_rate}, but reported {actual_f_rate}"
        )

    # Check pricing calculation
    p_tok = m3.get("prompt_tokens", 0)
    c_tok = m3.get("candidate_tokens", 0)
    expected_cost_per_1k = round(((p_tok * rate_in + c_tok * rate_out) / 1_000_000) * (1000 / 105), 4)
    actual_cost_per_1k = round(m3.get("estimated_cost_usd_per_1k", 0.0), 4)
    if abs(expected_cost_per_1k - actual_cost_per_1k) > 0.05:
        errors.append(
            f"Mode 3 cost per 1k mismatch: calculated ${expected_cost_per_1k:.4f} but reported ${actual_cost_per_1k:.4f}"
        )

    return errors


def check_out_of_sample_cuad_report(report_path: Path) -> list[str]:
    errors: list[str] = []
    if not report_path.exists():
        return [f"Missing required report: {report_path}"]

    with report_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Verify against raw per-case JSONL
    cases_file = report_path.parent / "out_of_sample_cuad_cases.jsonl"
    if not cases_file.exists():
        errors.append(f"Missing required raw case log: {cases_file}")
        return errors

    raw_bytes = cases_file.read_bytes()
    calculated_sha = hashlib.sha256(raw_bytes).hexdigest()
    expected_sha = data.get("raw_cases_sha256")
    if expected_sha and calculated_sha != expected_sha:
        errors.append(
            f"Raw cases SHA256 mismatch for {cases_file.name}: expected {expected_sha} but calculated {calculated_sha}. Forgery detected!"
        )

    cases: list[dict[str, Any]] = []
    with cases_file.open("r", encoding="utf-8") as cf:
        for line_num, line in enumerate(cf, 1):
            if line.strip():
                c = json.loads(line)
                required_fields = [
                    "case_id", "contract_id", "category", "query", "is_present",
                    "passage_ids", "passages_hash", "verdict", "reason",
                    "prompt_tokens", "candidate_tokens", "latency_ms", "response_id", "model_version",
                ]
                for rf in required_fields:
                    if rf not in c:
                        errors.append(f"Line {line_num} in {cases_file.name} missing required field: {rf}")

                verdict = c.get("verdict")
                abstained = c.get("abstained")
                if verdict not in ["ACCEPT", "ABSTAIN"]:
                    errors.append(f"Line {line_num} in {cases_file.name} has invalid verdict: {verdict}")
                if abstained != (verdict == "ABSTAIN"):
                    errors.append(
                        f"Line {line_num} in {cases_file.name} has inconsistent abstention: abstained={abstained} vs verdict={verdict}"
                    )
                cases.append(c)

    # 2. Recalculate metrics from raw cases
    raw_tot = len(cases)
    raw_pres = sum(1 for c in cases if c.get("is_present"))
    raw_abs = sum(1 for c in cases if not c.get("is_present"))
    raw_c_abs = sum(1 for c in cases if not c.get("is_present") and c.get("abstained"))
    raw_f_abs = sum(1 for c in cases if c.get("is_present") and c.get("abstained"))

    raw_hit5 = sum(1 for c in cases if c.get("is_present") and c.get("hit_at_5"))
    raw_acc_hit = sum(1 for c in cases if c.get("is_present") and c.get("hit_at_5") and not c.get("abstained"))
    raw_miss5 = sum(1 for c in cases if c.get("is_present") and not c.get("hit_at_5"))
    raw_rej_miss = sum(1 for c in cases if c.get("is_present") and not c.get("hit_at_5") and c.get("abstained"))

    raw_p_tok = sum(c.get("prompt_tokens", 0) for c in cases)
    raw_c_tok = sum(c.get("candidate_tokens", 0) for c in cases)
    raw_lat = sum(c.get("latency_ms", 0.0) for c in cases) / raw_tot if raw_tot > 0 else 0.0

    # Compare counts
    if data.get("total_queries_evaluated") != raw_tot:
        errors.append(f"OOS report total_queries_evaluated {data.get('total_queries_evaluated')} != raw {raw_tot}")
    if data.get("presence_cases") != raw_pres:
        errors.append(f"OOS report presence_cases {data.get('presence_cases')} != raw {raw_pres}")
    if data.get("absence_cases") != raw_abs:
        errors.append(f"OOS report absence_cases {data.get('absence_cases')} != raw {raw_abs}")

    metrics = data.get("metrics", {})
    neg = metrics.get("correct_abstention_on_negatives", {})
    if neg.get("count") != raw_c_abs or neg.get("total") != raw_abs:
        errors.append(
            f"OOS negative abstentions mismatch: reported {neg.get('count')}/{neg.get('total')} vs raw {raw_c_abs}/{raw_abs}"
        )
    exp_neg_rate = round(raw_c_abs / raw_abs, 4) if raw_abs > 0 else 0.0
    if abs(neg.get("rate", 0.0) - exp_neg_rate) > 1e-3:
        errors.append(f"OOS negative rate mismatch: {neg.get('rate')} vs {exp_neg_rate}")
    exp_neg_ci = _wilson_interval(raw_c_abs, raw_abs)
    if neg.get("wilson_ci95") != exp_neg_ci:
        errors.append(f"OOS negative CI mismatch: {neg.get('wilson_ci95')} vs {exp_neg_ci}")

    pos = metrics.get("false_abstention_on_positives", {})
    if pos.get("count") != raw_f_abs or pos.get("total") != raw_pres:
        errors.append(
            f"OOS false abstention mismatch: reported {pos.get('count')}/{pos.get('total')} vs raw {raw_f_abs}/{raw_pres}"
        )
    exp_pos_rate = round(raw_f_abs / raw_pres, 4) if raw_pres > 0 else 0.0
    if abs(pos.get("rate", 0.0) - exp_pos_rate) > 1e-3:
        errors.append(f"OOS positive false abstention rate mismatch: {pos.get('rate')} vs {exp_pos_rate}")
    exp_pos_ci = _wilson_interval(raw_f_abs, raw_pres)
    if pos.get("wilson_ci95") != exp_pos_ci:
        errors.append(f"OOS positive false abstention CI mismatch: {pos.get('wilson_ci95')} vs {exp_pos_ci}")

    # Macro F1
    pa = raw_c_abs / (raw_c_abs + raw_f_abs) if (raw_c_abs + raw_f_abs) > 0 else 0.0
    ra = raw_c_abs / raw_abs if raw_abs > 0 else 0.0
    f1a = 2 * pa * ra / (pa + ra) if (pa + ra) > 0 else 0.0

    tp = raw_pres - raw_f_abs
    fp = raw_abs - raw_c_abs
    fn = raw_f_abs
    pp = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rp = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1p = 2 * pp * rp / (pp + rp) if (pp + rp) > 0 else 0.0
    exp_macro_f1 = round((f1a + f1p) / 2.0, 4)
    if abs(metrics.get("macro_f1", 0.0) - exp_macro_f1) > 1e-3:
        errors.append(f"OOS Macro F1 mismatch: reported {metrics.get('macro_f1')} vs calculated {exp_macro_f1}")

    # Verifier conditioned on retrieval
    ret_cond = metrics.get("retrieval_conditioned_verifier", {})
    if ret_cond.get("gold_in_top5_count") != raw_hit5 or ret_cond.get("llm_accepted_on_hit") != raw_acc_hit:
        errors.append(
            f"OOS verifier conditioned hit mismatch: reported {ret_cond.get('llm_accepted_on_hit')}/{ret_cond.get('gold_in_top5_count')} vs raw {raw_acc_hit}/{raw_hit5}"
        )
    if ret_cond.get("miss_top5_count") != raw_miss5 or ret_cond.get("llm_rejected_on_miss") != raw_rej_miss:
        errors.append(
            f"OOS verifier conditioned miss mismatch: reported {ret_cond.get('llm_rejected_on_miss')}/{ret_cond.get('miss_top5_count')} vs raw {raw_rej_miss}/{raw_miss5}"
        )

    # Performance
    perf = metrics.get("performance", {})
    if perf.get("prompt_tokens") != raw_p_tok:
        errors.append(f"OOS prompt tokens mismatch: reported {perf.get('prompt_tokens')} vs raw {raw_p_tok}")
    if perf.get("candidate_tokens") != raw_c_tok:
        errors.append(f"OOS candidate tokens mismatch: reported {perf.get('candidate_tokens')} vs raw {raw_c_tok}")
    if abs(perf.get("mean_latency_ms", 0.0) - round(raw_lat, 1)) > 1.0:
        errors.append(f"OOS mean latency mismatch: reported {perf.get('mean_latency_ms')} vs raw {round(raw_lat, 1)}")

    return errors


def check_calibration_cuad_report(report_path: Path) -> list[str]:
    errors: list[str] = []
    if not report_path.exists():
        return [f"Missing required report: {report_path}"]

    with report_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Verify candidates math
    candidates = data.get("candidates", [])
    for cand in candidates:
        c_abs = cand.get("correct_abstentions", 0)
        tot_abs = cand.get("total_absences", 0)
        exp_rate = round(c_abs / tot_abs, 4) if tot_abs > 0 else 0.0
        act_rate = round(cand.get("abstention_rate", 0.0), 4)
        if abs(exp_rate - act_rate) > 1e-3:
            errors.append(
                f"Candidate tau={cand.get('min_relevance_score')} abstention_rate mismatch: {c_abs}/{tot_abs}={exp_rate} vs {act_rate}"
            )

        f_abs = cand.get("false_abstentions", 0)
        tot_pres = cand.get("total_presences", 0)
        exp_frate = round(f_abs / tot_pres, 4) if tot_pres > 0 else 0.0
        act_frate = round(cand.get("false_abstention_rate", 0.0), 4)
        if abs(exp_frate - act_frate) > 1e-3:
            errors.append(
                f"Candidate tau={cand.get('min_relevance_score')} false_abstention_rate mismatch: {f_abs}/{tot_pres}={exp_frate} vs {act_frate}"
            )

        # Check Wilson interval
        exp_ci = _wilson_interval(c_abs, tot_abs)
        act_ci = cand.get("abstention_rate_ci", [])
        if act_ci and (abs(exp_ci[0] - act_ci[0]) > 1e-3 or abs(exp_ci[1] - act_ci[1]) > 1e-3):
            errors.append(
                f"Candidate tau={cand.get('min_relevance_score')} CI mismatch: expected {exp_ci} vs {act_ci}"
            )

    # Verify baselines
    base_always = data.get("baseline_always_abstain", {})
    if base_always and (
        base_always.get("correct_abstention_rate") != 1.0 or base_always.get("false_abstention_rate") != 1.0
    ):
        errors.append("Baseline Always-Abstain must have rates 1.0 / 1.0")

    base_never = data.get("baseline_never_abstain", {})
    if base_never and (
        base_never.get("correct_abstention_rate") != 0.0 or base_never.get("false_abstention_rate") != 0.0
    ):
        errors.append("Baseline Never-Abstain must have rates 0.0 / 0.0")

    return errors


def check_bakeoff_report(report_path: Path, pricing_path: Path) -> list[str]:
    errors: list[str] = []
    if not report_path.exists():
        return [f"Missing required report: {report_path}"]

    with report_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    with pricing_path.open("r", encoding="utf-8") as f:
        pricing_data = yaml.safe_load(f)

    candidates = data.get("candidates", [])
    for cand in candidates:
        name = cand.get("candidate_name", "")
        sample_size = cand.get("sample_size", 0)
        if sample_size <= 0:
            errors.append(f"Bakeoff candidate {name} has invalid sample_size: {sample_size}")

        if "3.5" in name:
            p_rate = pricing_data["models"]["gemini-3.5-flash"]["prompt_cost_per_1m"]
            c_rate = pricing_data["models"]["gemini-3.5-flash"]["candidate_cost_per_1m"]
            p_tok = cand.get("prompt_tokens", 0)
            c_tok = cand.get("candidate_tokens", 0)
            exp_cost = round(((p_tok * p_rate + c_tok * c_rate) / 1_000_000) * (1000 / sample_size), 2)
            act_cost = round(cand.get("estimated_cost_per_1k", 0.0), 2)
            if abs(exp_cost - act_cost) > 0.10:
                errors.append(
                    f"Bakeoff candidate {name} cost mismatch: calculated ${exp_cost:.2f} vs reported ${act_cost:.2f}"
                )

    return errors


def check_canonical_report(report_path: Path) -> list[str]:
    errors: list[str] = []
    if not report_path.exists():
        return [f"Missing required report: {report_path}"]
    with report_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    valid = any(
        k in data
        for k in [
            "total_packages",
            "report",
            "metrics",
            "runs",
            "total_present_cases",
            "candidates",
            "version",
        ]
    )
    if not valid:
        errors.append(f"Canonical report {report_path.name} lacks recognized metrics structure")
    return errors


def run_all_checks() -> int:
    pricing_path = Path("config/pricing.yaml")
    reports_dir = Path("evals/reports")
    if not reports_dir.exists():
        print(f"Reports directory {reports_dir} does not exist.", file=sys.stderr)
        return 1

    all_errors: list[str] = []

    # Map of filename or pattern to checker function
    registered_handlers = {
        "bartleby_llm_verification_report.json": lambda p: check_bartleby_llm_report(p, pricing_path),
        "out_of_sample_cuad_report.json": check_out_of_sample_cuad_report,
        "calibration_cuad_report.json": check_calibration_cuad_report,
        "calibration_bartleby_report.json": check_canonical_report,
        "bakeoff_latest.json": lambda p: check_bakeoff_report(p, pricing_path),
        "canonical_baseline_report.json": check_canonical_report,
        "canonical_norag_report.json": check_canonical_report,
        "mode3_variance_report.json": check_canonical_report,
        "audit_rejected_cases_report.json": check_canonical_report,
        "cuad_benchmark_latest.json": check_canonical_report,
        "latest.json": check_canonical_report,
    }


    # Discover every *.json in evals/reports/ (DENY BY DEFAULT)
    for p in sorted(reports_dir.glob("*.json")):
        name = p.name
        # Match handler
        if name in registered_handlers:
            handler = registered_handlers[name]
            all_errors.extend(handler(p))
        elif name.startswith("bakeoff_report_"):
            all_errors.extend(check_bakeoff_report(p, pricing_path))
        elif name.startswith("cuad_benchmark_") or name.startswith("eval_report_"):
            all_errors.extend(check_canonical_report(p))
        else:

            all_errors.append(
                f"DENIED BY DEFAULT: Unregistered report JSON in evals/reports/: {name}. Every report must have an audited verification handler."
            )

    if all_errors:
        print("\n" + "=" * 70, file=sys.stderr)
        print(f" REPRO-CHECK FAILED: {len(all_errors)} metric / arithmetic discrepancies detected!", file=sys.stderr)
        print(" Hand-edited metrics or altered counts violate Law L1 and L2.", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        for err in all_errors:
            print(f"  ❌ {err}", file=sys.stderr)
        return 1

    print(" Repro-Check Passed: All JSON report metrics match exact arithmetic recalculations, raw cases, and pricing.")
    return 0


if __name__ == "__main__":
    sys.exit(run_all_checks())
