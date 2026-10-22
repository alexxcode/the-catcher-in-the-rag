from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, TypedDict

try:
    from langgraph.graph import END, StateGraph
except ImportError:
    StateGraph = None
    END = "__end__"

from catcher_in_the_rag.extract.extractor import DocumentExtractor
from catcher_in_the_rag.match.matcher import run_three_way_match
from catcher_in_the_rag.policy.engine import PolicyEngine
from catcher_in_the_rag.providers.base import LLMProvider
from catcher_in_the_rag.providers.factory import get_llm_provider
from catcher_in_the_rag.schemas import (
    Decision,
    Finding,
    ProcurementPackage,
)


class AuditState(TypedDict, total=False):
    """Typed state for The Catcher in the RAG LangGraph state machine."""

    package_id: str
    raw_input: Any
    package: ProcurementPackage
    match_findings: list[Finding]
    clause_findings: list[Finding]
    decision: Decision
    requires_human_review: bool


NodeFn = Callable[[AuditState], dict[str, Any]]


def create_extract_node(extractor: DocumentExtractor) -> NodeFn:
    """Factory for the extraction graph node."""

    def extract_node(state: AuditState) -> dict[str, Any]:
        raw = state.get("raw_input")
        if state.get("package") is not None:
            pkg = state["package"]
        else:
            pkg = extractor.extract_package(raw)
        return {
            "package_id": pkg.package_id,
            "package": pkg,
        }

    return extract_node


def match_node(state: AuditState) -> dict[str, Any]:
    """Perform deterministic three-way matching."""
    pkg = state["package"]
    findings = run_three_way_match(pkg)
    return {"match_findings": findings}


def create_policy_node(policy_engine: PolicyEngine) -> NodeFn:
    """Factory for the policy decision graph node."""

    def policy_node(state: AuditState) -> dict[str, Any]:
        pkg = state["package"]
        all_findings = list(state.get("match_findings", [])) + list(
            state.get("clause_findings", [])
        )
        decision = policy_engine.evaluate(
            package_id=pkg.package_id,
            invoice_total=pkg.invoice.total_amount,
            findings=all_findings,
        )
        return {
            "decision": decision,
            "requires_human_review": decision.requires_human_review,
        }

    return policy_node


class SimplePipeline:
    """Lightweight fallback pipeline when langgraph is not installed."""

    def __init__(self, extract_fn: NodeFn, match_fn: NodeFn, policy_fn: NodeFn) -> None:
        self.extract_fn = extract_fn
        self.match_fn = match_fn
        self.policy_fn = policy_fn

    def invoke(self, state: AuditState) -> AuditState:
        s = dict(state)
        s.update(self.extract_fn(s))
        s.update(self.match_fn(s))
        s.update(self.policy_fn(s))
        return s  # type: ignore[return-value]

    def compile(self) -> SimplePipeline:
        return self


def build_audit_graph(
    llm_provider: LLMProvider | None = None,
    policy_engine: PolicyEngine | None = None,
) -> StateGraph | SimplePipeline:
    """Build and wire the LangGraph state machine for document auditing."""
    provider = llm_provider or get_llm_provider()
    extractor = DocumentExtractor(provider=provider)
    policy = policy_engine or PolicyEngine()

    extract_fn = create_extract_node(extractor)
    policy_fn = create_policy_node(policy)

    if StateGraph is None:
        return SimplePipeline(extract_fn, match_node, policy_fn)

    graph = StateGraph(AuditState)

    graph.add_node("extract", extract_fn)
    graph.add_node("match", match_node)
    graph.add_node("decide", policy_fn)

    graph.set_entry_point("extract")
    graph.add_edge("extract", "match")
    graph.add_edge("match", "decide")
    graph.add_edge("decide", END)

    return graph


def run_audit(
    source: str | dict[str, Any] | Path | ProcurementPackage,
    llm_provider: LLMProvider | None = None,
) -> Decision:
    """Convenience helper to run full audit pipeline on a package and return the final Decision."""
    compiled = build_audit_graph(llm_provider=llm_provider).compile()
    init_state: AuditState = {}
    if isinstance(source, ProcurementPackage):
        init_state["package"] = source
        init_state["package_id"] = source.package_id
    else:
        init_state["raw_input"] = source

    final_state = compiled.invoke(init_state)
    return final_state["decision"]
