from __future__ import annotations

import math
import re

from pydantic import BaseModel, Field

try:
    from rank_bm25 import BM25Okapi
except ImportError:

    class BM25Okapi:  # type: ignore[no-redef]
        def __init__(self, corpus: list[list[str]]) -> None:
            self.corpus = corpus

        def get_scores(self, query: list[str]) -> list[float]:
            scores: list[float] = []
            q_set = set(query)
            for doc in self.corpus:
                common = q_set.intersection(doc)
                scores.append(float(len(common)))
            return scores


from catcher_in_the_rag.providers.base import EmbeddingProvider
from catcher_in_the_rag.providers.factory import get_embedding_provider
from catcher_in_the_rag.schemas import Clause, Contract


class ScoredClause(BaseModel):
    """A contract clause scored by hybrid retrieval."""

    clause_id: str
    clause_type: str
    title: str
    text: str
    start_char: int
    end_char: int
    bm25_score: float = 0.0
    dense_score: float = 0.0
    score: float = Field(..., description="Combined normalized hybrid score [0.0 - 1.0]")


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two float vectors."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b, strict=False))
    norm_a = math.sqrt(sum(a * a for a in vec_a)) or 1.0
    norm_b = math.sqrt(sum(b * b for b in vec_b)) or 1.0
    return max(0.0, min(1.0, dot / (norm_a * norm_b)))


STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
    "or",
    "pursuant",
    "clause",
    "section",
}


def _tokenize(text: str) -> list[str]:
    """Simple alphanumeric tokenization with stop-word filtering for BM25."""
    tokens = re.findall(r"\b[a-zA-Z0-9]{3,}\b", text.lower())
    return [t for t in tokens if t not in STOP_WORDS]


class ContractRetriever:
    """Hybrid BM25 + dense retrieval engine scoped to a single contract."""

    def __init__(
        self,
        contract: Contract,
        embedding_provider: EmbeddingProvider | None = None,
        use_reranker: bool = False,
    ) -> None:
        self.contract = contract
        self.clauses: list[Clause] = contract.clauses
        self.embedding_provider = embedding_provider or get_embedding_provider()
        self.use_reranker = use_reranker
        self._ranker = None
        if self.use_reranker:
            try:
                from flashrank import Ranker

                self._ranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2")
            except Exception:
                self._ranker = None

        # BM25 corpus preparation
        corpus_tokens = [_tokenize(f"{c.title} {c.text}") for c in self.clauses]
        self.bm25: BM25Okapi | None = (
            BM25Okapi(corpus_tokens) if corpus_tokens and any(corpus_tokens) else None
        )

        # Dense corpus embeddings
        clause_texts = [f"{c.title}\n{c.text}" for c in self.clauses]
        self.dense_embeddings: list[list[float]] = (
            self.embedding_provider.embed_texts(clause_texts) if clause_texts else []
        )

    def retrieve(self, query: str, top_k: int = 5) -> list[ScoredClause]:
        """Search contract clauses for relevant provisions using hybrid scoring."""
        if not self.clauses:
            return []

        query_tokens = _tokenize(query)
        bm25_scores = [0.0] * len(self.clauses)
        if self.bm25 and query_tokens:
            raw_bm25 = self.bm25.get_scores(query_tokens)
            # Normalize by query token count to reflect query keyword coverage
            bm25_scores = [min(1.0, float(s) / float(len(query_tokens))) for s in raw_bm25]

        # Dense similarity
        query_vec = self.embedding_provider.embed_query(query)
        dense_scores = [_cosine_similarity(query_vec, doc_vec) for doc_vec in self.dense_embeddings]

        scored: list[ScoredClause] = []
        for i, clause in enumerate(self.clauses):
            b_score = bm25_scores[i]
            d_score = dense_scores[i] if i < len(dense_scores) else 0.0
            # Hybrid combination (50% BM25, 50% dense)
            hybrid_score = round(0.5 * b_score + 0.5 * d_score, 4)

            scored.append(
                ScoredClause(
                    clause_id=clause.clause_id,
                    clause_type=clause.clause_type,
                    title=clause.title,
                    text=clause.text,
                    start_char=clause.start_char,
                    end_char=clause.end_char,
                    bm25_score=round(b_score, 4),
                    dense_score=round(d_score, 4),
                    score=hybrid_score,
                )
            )

        scored.sort(key=lambda x: x.score, reverse=True)

        if self.use_reranker and self._ranker and scored:
            try:
                from flashrank import RerankRequest

                candidates = scored[: max(15, top_k)]
                passages = [
                    {"id": i, "text": f"{c.title}\n{c.text[:1000]}", "clause": c}
                    for i, c in enumerate(candidates)
                ]
                req = RerankRequest(query=query, passages=passages)
                reranked = self._ranker.rerank(req)
                return [p["clause"] for p in reranked[:top_k]]
            except Exception:
                pass

        return scored[:top_k]
