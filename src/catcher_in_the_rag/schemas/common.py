from __future__ import annotations

from decimal import Decimal
from typing import Annotated

from pydantic import BaseModel, Field, model_validator


class Citation(BaseModel):
    """Verbatim citation pointing to a specific clause or segment in a document."""

    document_id: str = Field(..., description="Unique ID of the document (e.g. contract ID)")
    start_char: Annotated[int, Field(ge=0, description="0-indexed start character offset")]
    end_char: Annotated[int, Field(ge=0, description="0-indexed end character offset")]
    text: Annotated[str, Field(min_length=1, description="Verbatim quote from the source document")]

    @model_validator(mode="after")
    def validate_offsets(self) -> Citation:
        if self.end_char <= self.start_char:
            msg = f"end_char ({self.end_char}) must be greater than start_char ({self.start_char})"
            raise ValueError(msg)
        return self


class LineItem(BaseModel):
    """Line item in a Purchase Order or Invoice."""

    item_id: str = Field(..., description="Unique line item code / SKU")
    description: str = Field(..., description="Description of goods or services")
    quantity: Annotated[Decimal, Field(gt=0, description="Quantity billed or ordered")]
    unit_price: Annotated[Decimal, Field(ge=0, description="Unit price")]
    total_amount: Annotated[Decimal, Field(ge=0, description="Total amount (qty * unit_price)")]

    @model_validator(mode="after")
    def compute_or_validate_total(self) -> LineItem:
        expected = (self.quantity * self.unit_price).quantize(Decimal("0.01"))
        if self.total_amount != expected and abs(self.total_amount - expected) > Decimal("0.02"):
            msg = f"total_amount {self.total_amount} does not match qty * unit_price {expected}"
            raise ValueError(msg)
        return self
