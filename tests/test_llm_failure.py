"""Tests simulating LLM failures (429 Rate Limit / Quota Exhaustion and 500 Model Errors).

Verifies that:
1. Transient 429 errors execute bounded retries.
2. Quota exhaustion raises explicit, structured errors without silent degradation.
3. Errors never cause silent fallbacks into erroneous 'clean' package approvals.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from catcher_in_the_rag.graph import audit_pipeline
from catcher_in_the_rag.providers.gemini import GeminiProvider


class MockFailingClient:
    """Mock simulating quota exhaustion (429) and internal errors."""

    def __init__(self, fail_count: int = 1, error_message: str = "429 Resource Exhausted") -> None:
        self.attempts = 0
        self.fail_count = fail_count
        self.error_message = error_message

    def generate_content(self, *args: Any, **kwargs: Any) -> Any:
        self.attempts += 1
        if self.attempts <= self.fail_count:
            raise RuntimeError(self.error_message)
        mock_response = MagicMock()
        mock_response.text = '{"contract_id": "CTR-OK", "vendor_id": "V-1", "vendor_name": "V", "effective_date": "2025-01-01", "expiration_date": "2026-01-01", "payment_terms_days": 30, "auto_renew": false, "late_delivery_penalty_pct_per_day": 0.0, "rate_sheet": [], "clauses": []}'
        return mock_response


def test_gemini_retry_recovers_after_transient_429() -> None:
    """Verify that a transient 429 recovers on subsequent retry."""
    provider = GeminiProvider(model_id="gemini-3.5-flash")
    mock_client = MockFailingClient(
        fail_count=2, error_message="429 ResourceExhausted: Quota exceeded"
    )
    provider._client = mock_client
    provider._initialized = True
    provider._use_genai = False

    # Should succeed on 3rd attempt
    result = provider.generate_text("test prompt")
    assert mock_client.attempts == 3
    assert result != ""


def test_gemini_retry_exhausted_raises_explicit_runtime_error() -> None:
    """Verify that persistent 429 fails fast with clear error message, never silent degradation."""
    provider = GeminiProvider(model_id="gemini-3.5-flash")
    mock_client = MockFailingClient(
        fail_count=10, error_message="429 ResourceExhausted: Quota exceeded"
    )
    provider._client = mock_client
    provider._initialized = True
    provider._use_genai = False

    with pytest.raises(RuntimeError) as exc_info:
        provider.generate_text("test prompt")

    assert "429" in str(exc_info.value) or "Quota exceeded" in str(exc_info.value)
    assert mock_client.attempts == 3  # Max bounded retries


def test_llm_fatal_error_does_not_silently_approve() -> None:
    """Verify that an unrecoverable model 500 error halts pipeline or flags human review, never approving blindly."""
    failing_provider = MagicMock()
    failing_provider.extract_structured.side_effect = RuntimeError(
        "500 Internal Server Error: Model overloaded"
    )
    failing_provider.generate_text.side_effect = RuntimeError("500 Internal Server Error")

    fake_package_dict = {
        "package_id": "PKG-FAIL-TEST",
        "invoice": {
            "invoice_id": "INV-001",
            "po_id": "PO-001",
            "vendor_id": "V-1",
            "vendor_name": "Acme",
            "invoice_date": "2026-01-01",
            "payment_due_date": "2026-02-01",
            "total_amount": 1000.0,
            "line_items": [
                {
                    "item_id": "I1",
                    "description": "Item",
                    "quantity": 1,
                    "unit_price": 1000.0,
                    "total_amount": 1000.0,
                }
            ],
        },
        "po": {
            "po_id": "PO-001",
            "contract_id": "CTR-001",
            "vendor_id": "V-1",
            "vendor_name": "Acme",
            "issue_date": "2026-01-01",
            "expected_delivery_date": "2026-01-05",
            "total_amount": 1000.0,
            "line_items": [
                {
                    "item_id": "I1",
                    "description": "Item",
                    "quantity": 1,
                    "unit_price": 1000.0,
                    "total_amount": 1000.0,
                }
            ],
        },
        "receipt": {
            "receipt_id": "GR-001",
            "po_id": "PO-001",
            "delivery_date": "2026-01-04",
            "received_by": "Auditor",
            "items": [
                {
                    "item_id": "I1",
                    "quantity_received": 1,
                    "quantity_rejected": 0,
                    "condition": "GOOD",
                }
            ],
        },
        "contract": {
            "contract_id": "CTR-001",
            "vendor_id": "V-1",
            "vendor_name": "Acme",
            "effective_date": "2025-01-01",
            "expiration_date": "2027-01-01",
            "payment_terms_days": 30,
            "auto_renew": False,
            "late_delivery_penalty_pct_per_day": 0.0,
            "rate_sheet": [{"item_id": "I1", "description": "Item", "max_unit_price": 1000.0}],
            "raw_text": "Non-standard unstructured legal text that requires LLM parsing.",
        },
    }

    # If the LLM extractor raises RuntimeError, it should either propagate or mark requires_human_review,
    # NEVER return DecisionType.APPROVE with zero findings!
    try:
        final_state = audit_pipeline(fake_package_dict, llm_provider=failing_provider)
        # If it reached state, it MUST require human review
        assert (
            final_state.get("requires_human_review") is True
            or final_state["decision"].decision.value != "APPROVE"
        )
    except RuntimeError as err:
        assert "500" in str(err)
