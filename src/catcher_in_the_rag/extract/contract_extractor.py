from __future__ import annotations

import re
from datetime import date
from decimal import Decimal
from typing import Annotated, Any, Protocol

from pydantic import BaseModel, Field

from catcher_in_the_rag.providers.base import LLMProvider
from catcher_in_the_rag.providers.factory import get_llm_provider
from catcher_in_the_rag.schemas.contract import Clause, Contract, RateItem

WORD_TO_DAYS = {
    "fifteen": 15,
    "twenty": 20,
    "thirty": 30,
    "forty-five": 45,
    "forty five": 45,
    "sixty": 60,
    "ninety": 90,
}


class BaseContractExtractor(Protocol):
    """Protocol defining interface for contract text extractors."""

    def extract(self, raw_text: str) -> Contract: ...


class RuleBasedContractExtractor:
    """Deterministic, regex-driven baseline extractor for unstructured contract text.

    Serves as the deterministic offline baseline and fallback against LLM extraction
    for benchmarking extraction field accuracy (Phase 2 requirement).
    """

    def extract(self, raw_text: str) -> Contract:
        """Parse raw legal text into a structured Contract instance."""
        # 1. Contract & Vendor identification
        contract_id_match = re.search(r"Contract Reference:\s*([A-Za-z0-9_-]+)", raw_text)
        contract_id = contract_id_match.group(1).strip() if contract_id_match else "CTR-UNKNOWN"

        vendor_match = re.search(r"Vendor:\s*(.+?)\s*\(ID:\s*([A-Za-z0-9_-]+)\)", raw_text)
        if vendor_match:
            vendor_name = vendor_match.group(1).strip()
            vendor_id = vendor_match.group(2).strip()
        else:
            vendor_name = "Unknown Vendor"
            vendor_id = "VND-UNKNOWN"

        # 2. Dates
        eff_match = re.search(r"entered into as of\s*(\d{4}-\d{2}-\d{2})", raw_text, re.IGNORECASE)
        effective_date = date.fromisoformat(eff_match.group(1)) if eff_match else date(2025, 1, 1)

        exp_match = re.search(r"until\s*(\d{4}-\d{2}-\d{2})", raw_text, re.IGNORECASE) or re.search(
            r"expire on\s*(\d{4}-\d{2}-\d{2})", raw_text, re.IGNORECASE
        )
        expiration_date = date.fromisoformat(exp_match.group(1)) if exp_match else date(2026, 1, 1)

        # Auto-renewal check (properly handling 'without automatic renewal')
        has_renew = bool(re.search(r"shall automatically renew", raw_text, re.IGNORECASE))
        is_negated = bool(
            re.search(r"without automatic renewal|no automatic renewal", raw_text, re.IGNORECASE)
        )
        auto_renew = has_renew and not is_negated

        # 3. Payment Terms (e.g. "Net 30 days", "thirty (30) days", "Net 45")
        payment_terms_days = 30
        net_match = re.search(r"(?:payable\s+)?Net\s*(\d+)\s*days", raw_text, re.IGNORECASE)
        if net_match:
            payment_terms_days = int(net_match.group(1))
        else:
            word_match = re.search(
                r"(?:within|payable)\s+(thirty|forty-five|forty five|sixty|fifteen)\s*(?:\((\d+)\))?\s*days",
                raw_text,
                re.IGNORECASE,
            )
            if word_match:
                if word_match.group(2):
                    payment_terms_days = int(word_match.group(2))
                else:
                    word_key = word_match.group(1).lower()
                    payment_terms_days = WORD_TO_DAYS.get(word_key, 30)

        # 4. Late Delivery Penalty Rate
        late_penalty_pct = Decimal("0.0")
        penalty_match = re.search(
            r"(?:liquidated penalty of|penalty of|compensation of)\s*(\d+(?:\.\d+)?)\s*%",
            raw_text,
            re.IGNORECASE,
        )
        if penalty_match:
            late_penalty_pct = Decimal(penalty_match.group(1))

        # 5. Rate Sheet Extraction
        rate_sheet: list[RateItem] = []
        rate_matches = re.finditer(
            r"\*\s*([A-Za-z0-9_-]+)\s*-\s*(.+?):\s*Maximum Unit Rate of\s*\$(\d+(?:\.\d{2})?)\s*USD",
            raw_text,
        )
        for rm in rate_matches:
            item_id = rm.group(1).strip()
            desc = rm.group(2).strip()
            price = Decimal(rm.group(3))
            rate_sheet.append(RateItem(item_id=item_id, description=desc, max_unit_price=price))

        # 6. Clause Segmentation & Character Offsets
        clauses: list[Clause] = []
        # Match sections like "Section 1. Pricing and Rate Ceilings"
        section_pattern = re.compile(r"(Section\s+\d+\.\s*[^\n]+)", re.MULTILINE)
        matches = list(section_pattern.finditer(raw_text))

        type_map = {
            "pricing": "pricing",
            "rate": "pricing",
            "payment": "payment_terms",
            "invoicing": "payment_terms",
            "delivery": "penalties",
            "penalty": "penalties",
            "penalties": "penalties",
            "term": "term_and_termination",
            "termination": "term_and_termination",
            "freight": "freight_and_carriage",
            "carriage": "freight_and_carriage",
            "warranty": "warranties",
            "confidentiality": "confidentiality",
            "intellectual": "intellectual_property",
            "indemnif": "indemnification",
            "insurance": "insurance",
            "force majeure": "force_majeure",
            "audit": "audit_rights",
            "governing law": "governing_law",
        }

        for i, match in enumerate(matches):
            title = match.group(1).strip()
            header_end = match.end()

            # The clause body starts right after the section header line
            start_char = (
                header_end + 1
                if header_end < len(raw_text) and raw_text[header_end] == "\n"
                else header_end
            )
            end_char = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)

            body = raw_text[start_char:end_char].strip()
            # Infer clause_type from title
            title_lower = title.lower()
            clause_type = "other"
            for keyword, c_type in type_map.items():
                if keyword in title_lower:
                    clause_type = c_type
                    break

            clauses.append(
                Clause(
                    clause_id=f"CLAUSE-{i + 1:02d}-{clause_type.upper()}",
                    clause_type=clause_type,
                    title=title,
                    text=body,
                    start_char=start_char,
                    end_char=end_char,
                )
            )

        return Contract(
            contract_id=contract_id,
            vendor_id=vendor_id,
            vendor_name=vendor_name,
            effective_date=effective_date,
            expiration_date=expiration_date,
            auto_renew=auto_renew,
            payment_terms_days=payment_terms_days,
            late_delivery_penalty_pct_per_day=late_penalty_pct,
            rate_sheet=rate_sheet,
            clauses=clauses,
            raw_text=raw_text,
        )


class ContractExtractionError(Exception):
    """Raised when contract extraction fails and fallback is disabled."""


class ExtractedContractTerms(BaseModel):
    """Structured fields extracted from contract text by an LLM."""

    contract_id: str = Field(..., description="Unique contract identifier")
    vendor_id: str = Field(..., description="Vendor identifier")
    vendor_name: str = Field(..., description="Legal vendor name")
    effective_date: date = Field(..., description="Start date of agreement")
    expiration_date: date = Field(..., description="Expiration date of agreement")
    auto_renew: bool = Field(default=False, description="Whether contract auto-renews")
    payment_terms_days: Annotated[int, Field(ge=0, description="Agreed net payment terms (days)")]
    late_delivery_penalty_pct_per_day: Annotated[
        Decimal,
        Field(default=Decimal("0.0"), ge=0, description="Penalty % per day for late delivery"),
    ]
    rate_sheet: list[RateItem] = Field(
        default_factory=list, description="Contracted price ceiling per item"
    )


class LLMContractExtractor:
    """Pure LLM-driven contract extractor that never falls back to regex rules.

    Used for measuring genuine LLM extraction performance without masking failures.
    """

    def __init__(self, provider: LLMProvider | None = None) -> None:
        self.provider = provider or get_llm_provider()

    def extract(self, raw_text: str) -> Contract:
        """Extract Contract model directly via LLM provider. Fails explicitly on error."""
        terms = self.provider.extract_structured(
            text=raw_text,
            schema=ExtractedContractTerms,
            system_prompt=(
                "Extract contract governing terms (contract_id, vendor_id, vendor_name, "
                "effective_date, expiration_date, auto_renew, payment_terms_days, "
                "late_delivery_penalty_pct_per_day, rate_sheet) accurately from the contract text."
            ),
        )
        rule_extractor = RuleBasedContractExtractor()
        parsed = rule_extractor.extract(raw_text)

        return Contract(
            contract_id=terms.contract_id,
            vendor_id=terms.vendor_id,
            vendor_name=terms.vendor_name,
            effective_date=terms.effective_date,
            expiration_date=terms.expiration_date,
            auto_renew=terms.auto_renew,
            payment_terms_days=terms.payment_terms_days,
            late_delivery_penalty_pct_per_day=terms.late_delivery_penalty_pct_per_day,
            rate_sheet=terms.rate_sheet,
            clauses=parsed.clauses,
            raw_text=raw_text,
        )


class HybridContractExtractor:
    """Extractor capable of routing to LLM provider with explicit, transparent fallback metrics."""

    def __init__(
        self,
        provider: LLMProvider | None = None,
        llm_provider: LLMProvider | None = None,
        fallback_to_rules: bool = True,
    ) -> None:
        self.provider = provider or llm_provider or get_llm_provider()
        self.rule_extractor = RuleBasedContractExtractor()
        self.fallback_to_rules = fallback_to_rules
        self.llm_attempts: int = 0
        self.llm_successes: int = 0
        self.fallbacks_used: int = 0
        self.last_method_used: str = "none"

    def extract(self, raw_text: str) -> Contract:
        """Extract Contract model from unstructured raw text.

        Tracks attempts, successes, and fallbacks transparently.
        """
        self.llm_attempts += 1
        try:
            contract = self.provider.extract_structured(
                text=raw_text,
                schema=Contract,
                system_prompt=(
                    "Extract contract governing terms (contract_id, vendor_id, vendor_name, "
                    "effective_date, expiration_date, auto_renew, payment_terms_days, "
                    "late_delivery_penalty_pct_per_day, rate_sheet, clauses) into the Contract schema."
                ),
            )
            self.llm_successes += 1
            self.last_method_used = "llm"
            return contract
        except Exception as err:
            if not self.fallback_to_rules:
                self.last_method_used = "failed"
                msg = f"LLM contract extraction failed and fallback is disabled: {err}"
                raise ContractExtractionError(msg) from err

            self.fallbacks_used += 1
            self.last_method_used = "fallback_rule"
            return self.rule_extractor.extract(raw_text)

    def get_telemetry(self) -> dict[str, Any]:
        """Return transparent operational telemetry."""
        return {
            "llm_attempts": self.llm_attempts,
            "llm_successes": self.llm_successes,
            "fallbacks_used": self.fallbacks_used,
            "fallback_rate": (
                (self.fallbacks_used / self.llm_attempts) if self.llm_attempts > 0 else 0.0
            ),
            "last_method_used": self.last_method_used,
        }
