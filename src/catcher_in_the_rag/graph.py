"""
LangGraph wiring for The Catcher in the RAG agent.

Architecture (SPEC §5):
  ingest → classify → extract → link → match → retrieve →
  bartleby (abstention gate) → apply_terms → validate → decide

Rules (SPEC §11):
  - Implement schemas/ first; change a schema only with a note in the PR.
  - Every node must be a pure function over typed state (unit-testable without LLM).
  - Every node emits a trace span.
  - Never call a real LLM in tests; all nodes must work with the stub provider.
"""

from __future__ import annotations

# TODO (M2): Wire LangGraph state machine.
# Suggested structure:
#
#   from langgraph.graph import StateGraph, END
#   from catcher_in_the_rag.schemas import AgentState
#   from catcher_in_the_rag import (
#       ingest, classify, extract, match, retrieve, bartleby,
#       apply_terms, validate, policy,
#   )
#
#   def build_graph(provider) -> StateGraph:
#       graph = StateGraph(AgentState)
#       graph.add_node("ingest", ingest.run)
#       graph.add_node("classify", classify.run)
#       graph.add_node("extract", extract.run)
#       graph.add_node("link", match.link)
#       graph.add_node("match", match.run)
#       graph.add_node("retrieve", retrieve.run)
#       graph.add_node("bartleby", bartleby.run)
#       graph.add_node("apply_terms", apply_terms.run)
#       graph.add_node("validate", validate.run)
#       graph.add_node("decide", policy.run)
#       # Edges ...
#       # Human-in-the-loop interrupt at "decide" for review queue
#       return graph.compile(checkpointer=...)
