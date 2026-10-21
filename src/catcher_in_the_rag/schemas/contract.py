from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Annotated

from pydantic import BaseModel, Field

from catcher_in_the_rag.schemas.common import Citation


class Clause(BaseModel):
    """Specific clause extracted or indexed from a contract."""

    clause_id: str = Field(..., description="Unique clause identifier")
    clause_type: str = Field(
        ...,
        description="Type of clause (e.g. pricing, payment_terms, penalties, term_and_termination)",
    )
    title: str = Field(..., description="Heading or title of the clause section")
    text: str = Field(..., description="Full text of the clause")
    start_char: Annotated[
        int, Field(ge=0, description="Start character offset in contract raw text")
    ]
    end_char: Annotated[int, Field(ge=0, description="End character offset in contract raw text")]

    def to_citation(self, contract_id: str, quote: str | None = None) -> Citation:
        """Convert this clause or quote to a Citation."""
        quote_text = quote or self.text
        start = self.start_char
        end = self.end_char
        if quote and quote in self.text:
            offset = self.text.index(quote)
            start += offset
            end = start + len(quote)
        return Citation(
            document_id=contract_id,
            start_char=start,
            end_char=end,
            text=quote_text,
        )


class RateItem(BaseModel):
    """Agreed contracted rate for a good or service."""

    item_id: str = Field(..., description="SKU or service code")
    description: str = Field(..., description="Item or service description")
    max_unit_price: Annotated[Decimal, Field(ge=0, description="Agreed maximum unit price")]


class Contract(BaseModel):
    """Governing contract between the buyer and a vendor."""

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
    clauses: list[Clause] = Field(
        default_factory=list, description="Sectional clauses with character offsets"
    )
    raw_text: str = Field(..., description="Full text of the contract document")

    def find_span(self, quote: str) -> tuple[int, int] | None:
        """Find the start and end character offsets of a verbatim quote."""
        idx = self.raw_text.find(quote)
        if idx == -1:
            return None
        return idx, idx + len(quote)
