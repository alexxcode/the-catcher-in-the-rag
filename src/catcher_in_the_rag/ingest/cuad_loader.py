"""CUAD (Contract Understanding Atticus Dataset) Ingestion Loader.

Parses real-world commercial contracts from The Atticus Project (CC BY 4.0):
- 510 commercial contracts filed with the SEC
- 41 clause annotation categories with verbatim character offsets (start_char, text)
- Labeled absent clauses (answers: []) serving as ground truth negatives for Bartleby
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

from catcher_in_the_rag.schemas.contract import Clause, Contract, RateItem

# Primary CUAD categories mapped to procurement audit topics
CATEGORY_TOPIC_MAP: dict[str, str] = {
    "Document Name": "contract_title",
    "Parties": "vendor_buyer",
    "Agreement Date": "effective_date",
    "Effective Date": "effective_date",
    "Expiration Date": "expiration_date",
    "Renewal Term": "auto_renew",
    "Notice Period To Terminate Renewal": "auto_renew",
    "Governing Law": "governing_law",
    "Audit Rights": "audit_rights",
    "Liquidated Damages": "penalties",
    "Termination For Convenience": "term_and_termination",
    "Cap On Liability": "liability_cap",
    "Warranty Duration": "warranties",
    "Price Restrictions": "pricing",
    "Anti-Assignment": "subcontracting",
}


@dataclass
class CUADAnnotation:
    """A human-annotated clause span from CUAD."""

    category: str
    text: str
    start_char: int
    end_char: int


@dataclass
class CUADContract:
    """A parsed real-world contract from CUAD with all 41 category annotations."""

    contract_id: str
    title: str
    raw_text: str
    annotations: dict[str, list[CUADAnnotation]] = field(default_factory=dict)
    absent_categories: list[str] = field(default_factory=list)

    @property
    def char_length(self) -> int:
        return len(self.raw_text)

    def get_spans(self, category: str) -> list[CUADAnnotation]:
        return self.annotations.get(category, [])

    def has_clause(self, category: str) -> bool:
        return bool(self.annotations.get(category))


class CUADLoader:
    """Loads and converts CUAD v1 dataset contracts into The Catcher in the RAG schemas."""

    def __init__(self, cuad_json_path: str | Path = "data/real/cuad/CUADv1.json") -> None:
        self.path = Path(cuad_json_path)
        self._data: list[dict[str, Any]] | None = None

    def _ensure_loaded(self) -> None:
        if self._data is not None:
            return
        if not self.path.exists():
            raise FileNotFoundError(
                f"CUAD dataset not found at {self.path}. Run `make download-data` to fetch it."
            )
        with self.path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        self._data = payload.get("data", [])

    def count(self) -> int:
        self._ensure_loaded()
        return len(self._data or [])

    def load_contracts(self, limit: int | None = None) -> list[CUADContract]:
        """Parse CUAD contracts and their 41 annotated clause categories."""
        self._ensure_loaded()
        raw_items = self._data or []
        if limit:
            raw_items = raw_items[:limit]

        contracts: list[CUADContract] = []
        for idx, item in enumerate(raw_items):
            title = item.get("title", f"CUAD-CONTRACT-{idx + 1:04d}")
            contract_id = f"CUAD-{idx + 1:04d}"
            paragraphs = item.get("paragraphs", [])
            if not paragraphs:
                continue

            p0 = paragraphs[0]
            context = p0.get("context", "")
            annotations: dict[str, list[CUADAnnotation]] = {}
            absent_categories: list[str] = []

            for qa in p0.get("qas", []):
                q_id = qa.get("id", "")
                cat_name = q_id.split("__")[-1] if "__" in q_id else qa.get("question", "Unknown")
                answers = qa.get("answers", [])

                if not answers or qa.get("is_impossible", False):
                    absent_categories.append(cat_name)
                    continue

                spans: list[CUADAnnotation] = []
                for ans in answers:
                    txt = ans.get("text", "").strip()
                    start = ans.get("answer_start", -1)
                    if start >= 0 and txt:
                        spans.append(
                            CUADAnnotation(
                                category=cat_name,
                                text=txt,
                                start_char=start,
                                end_char=start + len(txt),
                            )
                        )
                if spans:
                    annotations[cat_name] = spans
                else:
                    absent_categories.append(cat_name)

            contracts.append(
                CUADContract(
                    contract_id=contract_id,
                    title=title,
                    raw_text=context,
                    annotations=annotations,
                    absent_categories=absent_categories,
                )
            )

        return contracts

    def to_catcher_contract(self, cuad: CUADContract) -> Contract:
        """Convert a real CUAD contract into a Catcher Contract schema instance."""
        # 1. Extract vendor / buyer hints from annotations or fallback
        parties = cuad.get_spans("Parties")
        vendor_name = parties[0].text if parties else "Commercial Vendor"
        if len(parties) > 1:
            buyer_name = parties[1].text
        else:
            buyer_name = "Enterprise Buyer"

        # 2. Extract dates if annotated
        eff_spans = cuad.get_spans("Effective Date") or cuad.get_spans("Agreement Date")
        effective_date = date(2025, 1, 1)
        if eff_spans:
            # Try parsing a year
            match = re.search(r"\b(19\d\d|20\d\d)\b", eff_spans[0].text)
            if match:
                try:
                    effective_date = date(int(match.group(1)), 1, 1)
                except Exception:
                    pass

        exp_spans = cuad.get_spans("Expiration Date")
        expiration_date = date(2026, 12, 31)
        if exp_spans:
            match = re.search(r"\b(19\d\d|20\d\d)\b", exp_spans[0].text)
            if match:
                try:
                    expiration_date = date(int(match.group(1)), 12, 31)
                except Exception:
                    pass

        auto_renew = bool(cuad.get_spans("Renewal Term"))

        # 3. Build Clause segmentation from paragraph breaks or sections
        clauses: list[Clause] = []
        paragraphs = re.split(r"\n\s*\n", cuad.raw_text)
        current_offset = 0

        for i, para in enumerate(paragraphs):
            para_clean = para.strip()
            if not para_clean:
                current_offset += len(para) + 2
                continue

            start_char = cuad.raw_text.find(para_clean, current_offset)
            if start_char == -1:
                start_char = current_offset
            end_char = start_char + len(para_clean)
            current_offset = end_char

            # Check if paragraph contains any labeled category
            clause_type = "general"
            for cat, spans in cuad.annotations.items():
                for sp in spans:
                    if sp.start_char >= start_char and sp.end_char <= end_char:
                        clause_type = CATEGORY_TOPIC_MAP.get(cat, "contract_clause")
                        break
                if clause_type != "general":
                    break

            first_line = para_clean.split("\n")[0][:80]
            clauses.append(
                Clause(
                    clause_id=f"{cuad.contract_id}-SEC-{i + 1:03d}",
                    clause_type=clause_type,
                    title=first_line if len(first_line) > 5 else f"Section {i + 1}",
                    text=para_clean,
                    start_char=start_char,
                    end_char=end_char,
                )
            )

        return Contract(
            contract_id=cuad.contract_id,
            vendor_id=f"VND-{cuad.contract_id}",
            vendor_name=vendor_name[:100],
            effective_date=effective_date,
            expiration_date=expiration_date,
            auto_renew=auto_renew,
            payment_terms_days=30,
            late_delivery_penalty_pct_per_day=Decimal("0.0"),
            rate_sheet=[
                RateItem(
                    item_id="SKU-REAL",
                    description="Procured Services",
                    max_unit_price=Decimal("1000.00"),
                )
            ],
            clauses=clauses,
            raw_text=cuad.raw_text,
        )
