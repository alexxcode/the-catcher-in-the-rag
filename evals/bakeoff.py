"""Extraction Bake-Off Runner: Flash vs Pro vs Rule-Based Baseline (Phase 3).

Evaluates candidate extraction strategies on unstructured contracts:
- Field-by-field accuracy: payment_terms_days, expiration_date, penalty %, rate sheet
- Latency per contract (ms)
- Estimated cost per 1,000 contracts
- Tracing integration with Langfuse via providers.get_tracer()
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from catcher_in_the_rag.extract.contract_extractor import (
    RuleBasedContractExtractor,
)
from catcher_in_the_rag.providers.factory import get_llm_provider
from catcher_in_the_rag.providers.tracing import get_tracer
from catcher_in_the_rag.schemas.contract import Contract
from evals.extraction_eval import evaluate_contract_extraction
from evals.provenance import get_provenance_metadata


@dataclass
class CandidateResult:
    candidate_name: str
    model_id: str
    provider_type: str
    sample_size: int
    mean_latency_ms: float
    estimated_cost_per_1k_usd: float
    error_count: int
    error_rate: float
    field_accuracies: dict[str, float]
    overall_perfect_accuracy: float
    is_simulated: bool = False


class ExtractionBakeOff:
    """Executes comparative evaluation across extraction candidates."""

    def __init__(
        self,
        contracts: list[tuple[str, Contract]],
        sample_size: int = 50,
        allow_simulation_fallback: bool = True,
    ) -> None:
        self.contracts = contracts[:sample_size]
        self.sample_size = len(self.contracts)
        self.allow_simulation_fallback = allow_simulation_fallback
        self.tracer = get_tracer(sample_rate=0.2)

    def evaluate_rule_based(self) -> CandidateResult:
        extractor = RuleBasedContractExtractor()
        print(
            f"\nEvaluating: Rule-Based Baseline (Regex/Parsing) | Sample size: {self.sample_size}",
            flush=True,
        )
        start = time.perf_counter()
        field_matches: dict[str, int] = {
            "payment_terms_days": 0,
            "expiration_date": 0,
            "late_delivery_penalty_pct_per_day": 0,
            "auto_renew": 0,
            "rate_sheet": 0,
        }
        perfect_count = 0

        for raw_text, gt_contract in self.contracts:
            t0 = time.perf_counter()
            extracted = extractor.extract(raw_text)
            dt_ms = (time.perf_counter() - t0) * 1000
            self.tracer.trace_execution(
                name="extract_contract_rule_based",
                input_data={"length": len(raw_text)},
                output_data={"extracted": extracted.model_dump(mode="json")},
                latency_ms=dt_ms,
                tags=["bake-off", "rule-based"],
            )
            matches = evaluate_contract_extraction(extracted, gt_contract)
            all_ok = True
            for k, ok in matches.items():
                if ok:
                    field_matches[k] += 1
                else:
                    all_ok = False
            if all_ok:
                perfect_count += 1

        total_time_ms = (time.perf_counter() - start) * 1000
        mean_lat = total_time_ms / max(1, self.sample_size)

        return CandidateResult(
            candidate_name="Rule-Based Baseline (Regex/Parsing)",
            model_id="deterministic-rules",
            provider_type="rule_based",
            sample_size=self.sample_size,
            mean_latency_ms=round(mean_lat, 2),
            estimated_cost_per_1k_usd=0.00,
            error_count=0,
            error_rate=0.0,
            field_accuracies={k: round(v / self.sample_size, 4) for k, v in field_matches.items()},
            overall_perfect_accuracy=round(perfect_count / self.sample_size, 4),
            is_simulated=False,
        )

    def evaluate_llm_candidate(
        self,
        candidate_name: str,
        provider_type: str,
        model_id: str,
        prompt_cost_per_1m: float = 0.075,
        completion_cost_per_1m: float = 0.30,
    ) -> CandidateResult:
        from catcher_in_the_rag.extract.contract_extractor import LLMContractExtractor

        is_simulated = False
        actual_provider_type = provider_type
        display_name = candidate_name

        try:
            llm = get_llm_provider(provider_type=provider_type, model_id=model_id)
            health = llm.check_model_health()
            print(
                f"  [Health Check] {model_id}: status={health.get('status')} latency={health.get('latency_ms')}ms",
                flush=True,
            )
        except Exception as e:
            if not self.allow_simulation_fallback:
                raise RuntimeError(
                    f"Failed to connect to LLM provider {provider_type} for model {model_id}: {e}"
                ) from e
            # Transparently label as offline simulation stub
            llm = get_llm_provider(provider_type="stub")
            actual_provider_type = "stub"
            is_simulated = True
            display_name = f"{candidate_name} [Offline Stub Simulation]"

        # Strict LLM extractor: does NOT silently fallback to regex
        extractor = LLMContractExtractor(provider=llm)

        print(
            f"\nEvaluating: {display_name} (provider: {actual_provider_type}) | Sample size: {self.sample_size}",
            flush=True,
        )
        start = time.perf_counter()
        field_matches: dict[str, int] = {
            "payment_terms_days": 0,
            "expiration_date": 0,
            "late_delivery_penalty_pct_per_day": 0,
            "auto_renew": 0,
            "rate_sheet": 0,
        }
        perfect_count = 0
        error_count = 0
        total_prompt_tokens = 0
        total_completion_tokens = 0

        for idx, (raw_text, gt_contract) in enumerate(self.contracts):
            t0 = time.perf_counter()
            # Estimate token lengths: ~4 chars per token for prompt, ~500 tokens structured response
            est_prompt_tokens = max(1, len(raw_text) // 4)
            est_completion_tokens = 250
            total_prompt_tokens += est_prompt_tokens
            total_completion_tokens += est_completion_tokens

            all_ok = False
            try:
                extracted = extractor.extract(raw_text)
                matches = evaluate_contract_extraction(extracted, gt_contract)
                all_ok = True
                for k, ok in matches.items():
                    if ok:
                        field_matches[k] += 1
                    else:
                        all_ok = False
                if all_ok:
                    perfect_count += 1
                out_data = extracted.model_dump(mode="json")
            except Exception as err:
                error_count += 1
                out_data = {"error": str(err)}

            dt_ms = (time.perf_counter() - t0) * 1000
            print(
                f"  [{idx + 1}/{self.sample_size}] {gt_contract.contract_id}: "
                f"{dt_ms:.0f}ms | perfect={all_ok} | err_count={error_count}",
                flush=True,
            )
            self.tracer.trace_execution(
                name=f"extract_contract_{model_id}",
                input_data={"length": len(raw_text)},
                output_data=out_data,
                latency_ms=dt_ms,
                tags=["bake-off", model_id, actual_provider_type],
            )

        total_time_ms = (time.perf_counter() - start) * 1000
        mean_lat = total_time_ms / max(1, self.sample_size)
        err_rate = error_count / max(1, self.sample_size)

        if is_simulated:
            cost_per_1k = 0.00
        else:
            cost_per_contract = (total_prompt_tokens / max(1, self.sample_size)) * (
                prompt_cost_per_1m / 1_000_000
            ) + (total_completion_tokens / max(1, self.sample_size)) * (
                completion_cost_per_1m / 1_000_000
            )
            cost_per_1k = round(cost_per_contract * 1000, 2)

        return CandidateResult(
            candidate_name=display_name,
            model_id=model_id,
            provider_type=actual_provider_type,
            sample_size=self.sample_size,
            mean_latency_ms=round(mean_lat, 2),
            estimated_cost_per_1k_usd=cost_per_1k,
            error_count=error_count,
            error_rate=round(err_rate, 4),
            field_accuracies={k: round(v / self.sample_size, 4) for k, v in field_matches.items()},
            overall_perfect_accuracy=round(perfect_count / self.sample_size, 4),
            is_simulated=is_simulated,
        )

    def run_bakeoff(self) -> list[CandidateResult]:
        from catcher_in_the_rag.pricing import get_model_rates

        results: list[CandidateResult] = []

        # 1. Baseline Rule-Based
        results.append(self.evaluate_rule_based())

        # 2. Gemini 3.5 Flash (GA Active Model on Vertex AI)
        p35, c35 = get_model_rates("gemini-3.5-flash")
        results.append(
            self.evaluate_llm_candidate(
                candidate_name="Gemini 3.5 Flash (GA Active)",
                provider_type="vertex_ai",
                model_id="gemini-3.5-flash",
                prompt_cost_per_1m=p35,
                completion_cost_per_1m=c35,
            )
        )

        # 3. Gemini 3.8 Flash (GA Active Model on Vertex AI)
        p38, c38 = get_model_rates("gemini-3.8-flash")
        results.append(
            self.evaluate_llm_candidate(
                candidate_name="Gemini 3.8 Flash (GA Active)",
                provider_type="vertex_ai",
                model_id="gemini-3.8-flash",
                prompt_cost_per_1m=p38,
                completion_cost_per_1m=c38,
            )
        )

        self.tracer.flush()
        return results


def print_bakeoff_summary(results: list[CandidateResult]) -> None:
    print("=" * 100)
    print(" THE CATCHER IN THE RAG - EXTRACTION BAKE-OFF COMPARISON (PHASE 3)")
    print(f" Timestamp: {datetime.now(UTC).isoformat()}")
    print("=" * 100)
    header = f"{'Candidate':<38} | {'Provider':<10} | {'Latency':<9} | {'Cost/1k':<8} | {'Err%':<6} | {'Pmt Days':<8} | {'Rates':<8} | {'Overall':<8}"
    print(header)
    print("-" * 100)
    for r in results:
        pmt = f"{r.field_accuracies.get('payment_terms_days', 0.0) * 100:.1f}%"
        rate = f"{r.field_accuracies.get('rate_sheet', 0.0) * 100:.1f}%"
        overall = f"{r.overall_perfect_accuracy * 100:.1f}%"
        lat = f"{r.mean_latency_ms:.1f} ms"
        cost = f"${r.estimated_cost_per_1k_usd:.2f}" if not r.is_simulated else "N/A (Stub)"
        err = f"{r.error_rate * 100:.1f}%"
        prov = "stub" if r.is_simulated else r.provider_type
        print(
            f"{r.candidate_name:<38} | {prov:<10} | {lat:<9} | {cost:<8} | {err:<6} | {pmt:<8} | {rate:<8} | {overall:<8}"
        )
    print("=" * 100)


def save_bakeoff_report(
    results: list[CandidateResult],
    output_dir: str | Path = "evals/reports",
) -> Path:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    report_file = out_path / f"bakeoff_report_{ts}.json"
    latest_file = out_path / "bakeoff_latest.json"

    provenance = get_provenance_metadata(
        {
            "models_evaluated": [r.model_id for r in results],
        }
    )

    data = {
        "timestamp": datetime.now(UTC).isoformat(),
        "git_commit_sha": provenance["git_commit_sha"],
        "git_dirty": provenance["git_dirty"],
        "command": provenance["command"],
        "provenance": provenance,
        "sample_size": results[0].sample_size if results else 0,
        "candidates": [asdict(r) for r in results],
    }

    with report_file.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    with latest_file.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return report_file
