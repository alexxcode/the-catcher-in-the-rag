from __future__ import annotations

from pydantic import BaseModel

from catcher_in_the_rag.extract.contract_extractor import (
    HybridContractExtractor,
    RuleBasedContractExtractor,
)
from catcher_in_the_rag.schemas import PackageLabels, ProcurementPackage


class ContractFieldAccuracy(BaseModel):
    """Evaluation result for a single contract's extracted fields."""

    package_id: str
    payment_terms_match: bool
    expiration_date_match: bool
    penalty_rate_match: bool
    auto_renew_match: bool
    rate_sheet_match: bool
    perfect_match: bool


class ExtractionReport(BaseModel):
    """Aggregate benchmark report measuring contract extraction field accuracy."""

    total_evaluated: int
    payment_terms_accuracy: float
    expiration_date_accuracy: float
    penalty_rate_accuracy: float
    rate_sheet_accuracy: float
    overall_perfect_match_accuracy: float
    details: list[ContractFieldAccuracy] = []


class ExtractionBenchmarkRunner:
    """Benchmark harness evaluating unstructured contract extraction (Phase 2)."""

    def __init__(self, extractor_type: str = "rule") -> None:
        if extractor_type == "hybrid":
            self.extractor = HybridContractExtractor()
        else:
            self.extractor = RuleBasedContractExtractor()

    def evaluate_dataset(
        self,
        dataset: list[tuple[ProcurementPackage, PackageLabels]],
    ) -> ExtractionReport:
        """Run field-by-field extraction evaluation across all packages in the dataset."""
        total = len(dataset)
        if total == 0:
            return ExtractionReport(
                total_evaluated=0,
                payment_terms_accuracy=0.0,
                expiration_date_accuracy=0.0,
                penalty_rate_accuracy=0.0,
                rate_sheet_accuracy=0.0,
                overall_perfect_match_accuracy=0.0,
            )

        match_terms = 0
        match_exp = 0
        match_penalty = 0
        match_rates = 0
        match_perfect = 0
        details: list[ContractFieldAccuracy] = []

        for pkg, _ in dataset:
            raw_text = pkg.contract.raw_text
            extracted = self.extractor.extract(raw_text)

            terms_ok = extracted.payment_terms_days == pkg.contract.payment_terms_days
            exp_ok = extracted.expiration_date == pkg.contract.expiration_date
            pen_ok = (
                extracted.late_delivery_penalty_pct_per_day
                == pkg.contract.late_delivery_penalty_pct_per_day
            )
            renew_ok = extracted.auto_renew == pkg.contract.auto_renew

            # Check rate sheet
            gt_rates = {r.item_id: r.max_unit_price for r in pkg.contract.rate_sheet}
            ext_rates = {r.item_id: r.max_unit_price for r in extracted.rate_sheet}
            rates_ok = gt_rates == ext_rates

            perfect = terms_ok and exp_ok and pen_ok and renew_ok and rates_ok

            if terms_ok:
                match_terms += 1
            if exp_ok:
                match_exp += 1
            if pen_ok:
                match_penalty += 1
            if rates_ok:
                match_rates += 1
            if perfect:
                match_perfect += 1

            details.append(
                ContractFieldAccuracy(
                    package_id=pkg.package_id,
                    payment_terms_match=terms_ok,
                    expiration_date_match=exp_ok,
                    penalty_rate_match=pen_ok,
                    auto_renew_match=renew_ok,
                    rate_sheet_match=rates_ok,
                    perfect_match=perfect,
                )
            )

        return ExtractionReport(
            total_evaluated=total,
            payment_terms_accuracy=match_terms / total,
            expiration_date_accuracy=match_exp / total,
            penalty_rate_accuracy=match_penalty / total,
            rate_sheet_accuracy=match_rates / total,
            overall_perfect_match_accuracy=match_perfect / total,
            details=details,
        )


def print_extraction_summary(report: ExtractionReport) -> None:
    """Print an executive table of field extraction accuracy."""
    sep = "=" * 78
    print(sep)
    print(" THE CATCHER IN THE RAG - FIELD-BY-FIELD EXTRACTION ACCURACY (PHASE 2)")
    print(f" Total Contracts Evaluated from Raw Text: {report.total_evaluated}")
    print(sep)
    print(f" {'Field Name':<35} | {'Exact Match Accuracy':<25}")
    print("-" * 78)
    print(f" {'payment_terms_days':<35} | {report.payment_terms_accuracy * 100:>20.1f}%")
    print(f" {'expiration_date':<35} | {report.expiration_date_accuracy * 100:>20.1f}%")
    print(
        f" {'late_delivery_penalty_pct_per_day':<35} | {report.penalty_rate_accuracy * 100:>20.1f}%"
    )
    print(f" {'rate_sheet (SKUs & Price Limits)':<35} | {report.rate_sheet_accuracy * 100:>20.1f}%")
    print("-" * 78)
    print(
        f" {'OVERALL PERFECT EXTRACTION':<35} | {report.overall_perfect_match_accuracy * 100:>20.1f}%"
    )
    print(sep)
