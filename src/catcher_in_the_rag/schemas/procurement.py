from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Annotated

from pydantic import BaseModel, Field, model_validator

from catcher_in_the_rag.schemas.common import LineItem
from catcher_in_the_rag.schemas.contract import Contract


class ReceiptItem(BaseModel):
    """Line item received at delivery."""

    item_id: str = Field(..., description="SKU or item identifier matching PO")
    quantity_received: Annotated[Decimal, Field(ge=0, description="Quantity accepted")]
    quantity_rejected: Annotated[
        Decimal, Field(default=Decimal("0"), ge=0, description="Quantity rejected / damaged")
    ]
    condition: str = Field(default="GOOD", description="Condition of items received")


class GoodsReceipt(BaseModel):
    """Proof of delivery document (Goods Receipt Note)."""

    receipt_id: str = Field(..., description="Unique receipt identifier")
    po_id: str = Field(..., description="Referenced Purchase Order ID")
    delivery_date: date = Field(..., description="Date goods or services were delivered")
    received_by: str = Field(..., description="Staff member who accepted delivery")
    items: list[ReceiptItem] = Field(..., min_length=1, description="List of items delivered")


class PurchaseOrder(BaseModel):
    """Authorized Purchase Order (PO)."""

    po_id: str = Field(..., description="Unique PO number")
    contract_id: str = Field(..., description="Governing contract ID")
    vendor_id: str = Field(..., description="Vendor identifier")
    vendor_name: str = Field(..., description="Vendor legal name")
    issue_date: date = Field(..., description="Date PO was issued")
    expected_delivery_date: date = Field(..., description="Promised/scheduled delivery date")
    line_items: list[LineItem] = Field(..., min_length=1, description="Ordered items")
    total_amount: Annotated[Decimal, Field(ge=0, description="Total authorized amount")]
    currency: str = Field(default="USD", description="Currency code (e.g. USD)")

    @model_validator(mode="after")
    def validate_total(self) -> PurchaseOrder:
        sum_items = sum((item.total_amount for item in self.line_items), Decimal("0"))
        if abs(self.total_amount - sum_items) > Decimal("0.05"):
            msg = f"PO total {self.total_amount} does not match sum of items {sum_items}"
            raise ValueError(msg)
        return self


class Invoice(BaseModel):
    """Vendor invoice submitted for payment."""

    invoice_id: str = Field(..., description="Vendor invoice number")
    po_id: str = Field(..., description="Referenced PO number")
    vendor_id: str = Field(..., description="Vendor identifier")
    vendor_name: str = Field(..., description="Vendor name")
    invoice_date: date = Field(..., description="Date invoice was issued")
    payment_due_date: date = Field(..., description="Payment due date")
    line_items: list[LineItem] = Field(..., min_length=1, description="Invoiced items")
    total_amount: Annotated[Decimal, Field(ge=0, description="Total amount claimed")]
    currency: str = Field(default="USD", description="Currency code")

    @model_validator(mode="after")
    def validate_total(self) -> Invoice:
        sum_items = sum((item.total_amount for item in self.line_items), Decimal("0"))
        if abs(self.total_amount - sum_items) > Decimal("0.05"):
            msg = f"Invoice total {self.total_amount} does not match sum of items {sum_items}"
            raise ValueError(msg)
        return self


class ProcurementPackage(BaseModel):
    """Complete document package for 3-way matching and contract compliance."""

    package_id: str = Field(..., description="Package identifier")
    contract: Contract = Field(..., description="Governing contract")
    po: PurchaseOrder = Field(..., description="Purchase order")
    receipt: GoodsReceipt = Field(..., description="Goods receipt note")
    invoice: Invoice = Field(..., description="Invoice under review")
    related_invoices: list[Invoice] = Field(
        default_factory=list,
        description="Historical or concurrent invoices (used to test duplicate detection A03)",
    )
