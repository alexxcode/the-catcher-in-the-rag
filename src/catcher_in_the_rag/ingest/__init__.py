"""Data ingestion module for external real-world procurement and legal contract corpora."""

from __future__ import annotations

from catcher_in_the_rag.ingest.cuad_loader import (
    CATEGORY_TOPIC_MAP,
    CUADAnnotation,
    CUADContract,
    CUADLoader,
)
from catcher_in_the_rag.ingest.ocds_loader import (
    OCDSContractRecord,
    OCDSEntity,
    OCDSLineItem,
    OCDSLoader,
)

__all__ = [
    "CATEGORY_TOPIC_MAP",
    "CUADAnnotation",
    "CUADContract",
    "CUADLoader",
    "OCDSContractRecord",
    "OCDSEntity",
    "OCDSLineItem",
    "OCDSLoader",
]
