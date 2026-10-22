from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from catcher_in_the_rag.providers.base import LLMProvider
from catcher_in_the_rag.providers.factory import get_llm_provider
from catcher_in_the_rag.schemas import (
    Contract,
    GoodsReceipt,
    Invoice,
    ProcurementPackage,
    PurchaseOrder,
)


class DocumentExtractor:
    """Extracts structured Pydantic procurement models from JSON, dict, or text using an LLM provider."""

    def __init__(self, provider: LLMProvider | None = None) -> None:
        self.provider = provider or get_llm_provider()

    def extract_package(
        self, source: str | dict[str, Any] | Path | ProcurementPackage
    ) -> ProcurementPackage:
        """Extract or parse a full procurement package from a source file, dict, or raw text."""
        if isinstance(source, ProcurementPackage):
            return source

        if isinstance(source, Path):
            with source.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return ProcurementPackage.model_validate(data)

        if isinstance(source, dict):
            return ProcurementPackage.model_validate(source)

        if isinstance(source, str):
            # Check if JSON
            try:
                data = json.loads(source)
                if isinstance(data, dict):
                    return ProcurementPackage.model_validate(data)
            except Exception:
                pass

            # Otherwise use LLM provider structured extraction
            return self.provider.extract_structured(
                text=source,
                schema=ProcurementPackage,
                system_prompt=(
                    "Extract the ProcurementPackage structure (Contract, PurchaseOrder, "
                    "GoodsReceipt, Invoice) exactly matching the schema."
                ),
            )

        msg = f"Unsupported source type for extract_package: {type(source)}"
        raise TypeError(msg)

    def extract_invoice(self, text: str) -> Invoice:
        """Extract a structured Invoice from raw text."""
        return self.provider.extract_structured(
            text=text,
            schema=Invoice,
            system_prompt="Extract invoice fields strictly into the Invoice schema.",
        )

    def extract_po(self, text: str) -> PurchaseOrder:
        """Extract a structured PurchaseOrder from raw text."""
        return self.provider.extract_structured(
            text=text,
            schema=PurchaseOrder,
            system_prompt="Extract purchase order fields strictly into the PurchaseOrder schema.",
        )

    def extract_receipt(self, text: str) -> GoodsReceipt:
        """Extract a structured GoodsReceipt from raw text."""
        return self.provider.extract_structured(
            text=text,
            schema=GoodsReceipt,
            system_prompt="Extract goods receipt note fields strictly into the GoodsReceipt schema.",
        )

    def extract_contract(self, text: str) -> Contract:
        """Extract a structured Contract from raw text."""
        return self.provider.extract_structured(
            text=text,
            schema=Contract,
            system_prompt="Extract contract clauses and rate sheet into the Contract schema.",
        )
