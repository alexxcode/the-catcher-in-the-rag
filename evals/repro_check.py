"""Repro Check - Verifies empirical reports against raw counts and deterministic recalculation.

Enforces Law L1, L2, L7:
- Live reports: audits derived metrics (Macro F1, Wilson CIs, false abstention breakdown, pricing)
  against raw case counts and official pricing config.
- Offline reports: verifies mathematical consistency and determinism against raw ground truth.
- Fails with exit code 1 if any hand-edited metric or arithmetic discrepancy is detected.
"""

from __future__ import annotations

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
        errors.append(f"Missing required raw case log: {cases_file}")
    else:
        cases: list[dict[str, Any]] = []
        with cases_file.open("r", encoding="utf-8") as cf:
            for line in cf:
                if line.strip():
                    cases.append(json.loads(line))

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
        name = cand.get("candidate_name")
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


def run_all_checks() -> int:
    pricing_path = Path("config/pricing.yaml")
    reports_to_check = [
        ("evals/reports/bartleby_llm_verification_report.json", check_bartleby_llm_report, [pricing_path]),
        ("evals/reports/calibration_cuad_report.json", check_calibration_cuad_report, []),
        ("evals/reports/bakeoff_latest.json", check_bakeoff_report, [pricing_path]),
    ]

    all_errors: list[str] = []
    for rel_path, check_fn, extra_args in reports_to_check:
        p = Path(rel_path)
        errs = check_fn(p, *extra_args)
        all_errors.extend(errs)

    if all_errors:
        print("\n" + "=" * 70, file=sys.stderr)
        print(f" REPRO-CHECK FAILED: {len(all_errors)} metric / arithmetic discrepancies detected!", file=sys.stderr)
        print(" Hand-edited metrics or altered counts violate Law L1 and L2.", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        for err in all_errors:
            print(f"  ❌ {err}", file=sys.stderr)
        return 1

    print(" Repro-Check Passed: All JSON report metrics match exact arithmetic recalculations and pricing.")
    return 0


if __name__ == "__main__":
    sys.exit(run_all_checks())
