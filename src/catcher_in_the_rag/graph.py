from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypedDict

try:
    from langgraph.graph import END, StateGraph
except ImportError:
    StateGraph = None
    END = "__end__"

from catcher_in_the_rag.apply_terms.evaluator import ContractTermsEvaluator
from catcher_in_the_rag.extract.extractor import DocumentExtractor
from catcher_in_the_rag.match.matcher import run_three_way_match
from catcher_in_the_rag.policy.engine import PolicyEngine
from catcher_in_the_rag.providers.base import LLMProvider
from catcher_in_the_rag.providers.factory import get_llm_provider
from catcher_in_the_rag.providers.tracing import get_tracer
from catcher_in_the_rag.schemas import (
    Decision,
    Finding,
    ProcurementPackage,
)
from catcher_in_the_rag.validate.validator import validate_findings


class AuditState(TypedDict, total=False):
    """Typed state for The Catcher in the RAG LangGraph state machine."""

    package_id: str
    raw_input: Any
    package: ProcurementPackage
    match_findings: list[Finding]
    clause_findings: list[Finding]
    validated_findings: list[Finding]
    bartleby_abstained: bool
    bartleby_reason: str | None
    decision: Decision
    requires_human_review: bool
    llm_extraction_failed: bool


NodeFn = Callable[[AuditState], dict[str, Any]]


def create_extract_node(
    extractor: DocumentExtractor,
    extract_contract_from_text: bool = True,
) -> NodeFn:
    """Factory for the extraction graph node.

    When extract_contract_from_text is True (default), extracts governing contract
    terms directly from raw_text via DocumentExtractor rather than relying on
    pre-structured input fields.
    """

    def extract_node(state: AuditState) -> dict[str, Any]:
        raw = state.get("raw_input")
        if state.get("package") is not None:
            pkg = state["package"]
        else:
            pkg = extractor.extract_package(raw)

        llm_extraction_failed = False
        if extract_contract_from_text and pkg.contract and pkg.contract.raw_text:
            extracted_contract = extractor.extract_contract(pkg.contract.raw_text)
            if (
                hasattr(extractor, "contract_extractor")
                and getattr(extractor.contract_extractor, "last_method_used", None)
                == "fallback_rule"
            ):
                llm_extraction_failed = True

            updated_contract = pkg.contract.model_copy(
                update={
                    "payment_terms_days": extracted_contract.payment_terms_days,
                    "expiration_date": extracted_contract.expiration_date,
                    "late_delivery_penalty_pct_per_day": extracted_contract.late_delivery_penalty_pct_per_day,
                    "auto_renew": extracted_contract.auto_renew,
                    "rate_sheet": extracted_contract.rate_sheet,
                }
            )
            pkg = pkg.model_copy(update={"contract": updated_contract})

        return {
            "package_id": pkg.package_id,
            "package": pkg,
            "llm_extraction_failed": llm_extraction_failed,
            "requires_human_review": llm_extraction_failed,
        }

    return extract_node


def match_node(state: AuditState) -> dict[str, Any]:
    """Perform deterministic three-way matching."""
    pkg = state["package"]
    findings = run_three_way_match(pkg)
    return {"match_findings": findings}


def create_apply_terms_node(evaluator: ContractTermsEvaluator) -> NodeFn:
    """Factory for applying contract clauses via RAG and Bartleby gate."""

    def apply_terms_node(state: AuditState) -> dict[str, Any]:
        pkg = state["package"]
        findings, abstained, reason = evaluator.evaluate(pkg)
        return {
            "clause_findings": findings,
            "bartleby_abstained": abstained,
            "bartleby_reason": reason,
        }

    return apply_terms_node


def validate_node(state: AuditState) -> dict[str, Any]:
    """Validate that all citations exist verbatim in the contract source."""
    pkg = state["package"]
    raw_findings = list(state.get("match_findings", [])) + list(state.get("clause_findings", []))
    valid, _dropped = validate_findings(raw_findings, pkg.contract)
    return {"validated_findings": valid}


def create_policy_node(policy_engine: PolicyEngine) -> NodeFn:
    """Factory for the policy decision graph node."""

    def policy_node(state: AuditState) -> dict[str, Any]:
        pkg = state["package"]
        findings = state.get("validated_findings", [])
        abstained = state.get("bartleby_abstained", False)
        reason = state.get("bartleby_reason")

        decision = policy_engine.evaluate(
            package_id=pkg.package_id,
            invoice_total=pkg.invoice.total_amount,
            findings=findings,
            bartleby_abstained=abstained,
            bartleby_reason=reason,
        )
        # Extraction failure is decided once, in the extract node; carry it forward.
        return {
            "decision": decision,
            "requires_human_review": state.get("requires_human_review", False)
            or decision.requires_human_review,
        }

    return policy_node


class SimplePipeline:
    """Lightweight fallback pipeline when langgraph is not installed."""

    def __init__(
        self,
        extract_fn: NodeFn,
        match_fn: NodeFn,
        apply_terms_fn: NodeFn,
        validate_fn: NodeFn,
        policy_fn: NodeFn,
    ) -> None:
        self.extract_fn = extract_fn
        self.match_fn = match_fn
        self.apply_terms_fn = apply_terms_fn
        self.validate_fn = validate_fn
        self.policy_fn = policy_fn

    def invoke(self, state: AuditState) -> AuditState:
        s = dict(state)
        s.update(self.extract_fn(s))
        s.update(self.match_fn(s))
        s.update(self.apply_terms_fn(s))
        s.update(self.validate_fn(s))
        s.update(self.policy_fn(s))
        return s  # type: ignore[return-value]

    def compile(self) -> SimplePipeline:
        return self


def build_audit_graph(
    llm_provider: LLMProvider | None = None,
    policy_engine: PolicyEngine | None = None,
    extract_contract_from_text: bool = True,
) -> StateGraph | SimplePipeline:
    """Build and wire the LangGraph state machine for document auditing."""
    provider = llm_provider or get_llm_provider()
    extractor = DocumentExtractor(provider=provider)
    policy = policy_engine or PolicyEngine()
    terms_evaluator = ContractTermsEvaluator(llm_provider=provider)

    extract_fn = create_extract_node(
        extractor, extract_contract_from_text=extract_contract_from_text
    )
    apply_terms_fn = create_apply_terms_node(terms_evaluator)
    policy_fn = create_policy_node(policy)

    if StateGraph is None:
        return SimplePipeline(
            extract_fn=extract_fn,
            match_fn=match_node,
            apply_terms_fn=apply_terms_fn,
            validate_fn=validate_node,
            policy_fn=policy_fn,
        )

    graph = StateGraph(AuditState)

    graph.add_node("extract", extract_fn)
    graph.add_node("match", match_node)
    graph.add_node("apply_terms", apply_terms_fn)
    graph.add_node("validate", validate_node)
    graph.add_node("decide", policy_fn)

    graph.set_entry_point("extract")
    graph.add_edge("extract", "match")
    graph.add_edge("match", "apply_terms")
    graph.add_edge("apply_terms", "validate")
    graph.add_edge("validate", "decide")
    graph.add_edge("decide", END)

    return graph


def audit_pipeline(
    source: str | dict[str, Any] | Path | ProcurementPackage,
    llm_provider: LLMProvider | None = None,
    extract_contract_from_text: bool = True,
) -> AuditState:
    """Run full audit pipeline on a package and return the full final AuditState."""
    compiled = build_audit_graph(
        llm_provider=llm_provider,
        extract_contract_from_text=extract_contract_from_text,
    ).compile()
    init_state: AuditState = {}
    if isinstance(source, ProcurementPackage):
        init_state["package"] = source
        init_state["package_id"] = source.package_id
    else:
        init_state["raw_input"] = source

    t0 = time.perf_counter()
    final_state: AuditState = compiled.invoke(init_state)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    # Observability: Trace execution via Langfuse (or NoOp if offline)
    pkg = final_state.get("package")
    decision = final_state.get("decision")
    findings = final_state.get("validated_findings", [])
    tracer = get_tracer()

    tracer.trace_execution(
        name="audit_pipeline",
        input_data={
            "package_id": final_state.get("package_id"),
            "vendor_name": pkg.invoice.vendor_name if pkg else None,
            "invoice_total": float(pkg.invoice.total_amount) if pkg else None,
        },
        output_data={
            "decision": decision.decision.value if decision else "UNKNOWN",
            "justification": decision.justification if decision else None,
            "findings_count": len(findings),
        },
        metadata={
            "bartleby_abstained": final_state.get("bartleby_abstained", False),
            "bartleby_reason": final_state.get("bartleby_reason"),
            "requires_human_review": final_state.get("requires_human_review", False),
        },
        tags=[
            "audit",
            decision.decision.value.lower() if decision else "unknown",
        ],
        latency_ms=elapsed_ms,
    )
    tracer.flush()

    return final_state


def run_audit(
    source: str | dict[str, Any] | Path | ProcurementPackage,
    llm_provider: LLMProvider | None = None,
    extract_contract_from_text: bool = True,
) -> Decision:
    """Convenience helper to run full audit pipeline on a package and return the final Decision."""
    final_state = audit_pipeline(
        source=source,
        llm_provider=llm_provider,
        extract_contract_from_text=extract_contract_from_text,
    )
    return final_state["decision"]
