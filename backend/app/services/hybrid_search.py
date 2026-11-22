"""Hybrid retrieval: semantic (ChromaDB) + keyword (BM25) -> RRF fusion -> rerank.

Step by step for one question:
1. Semantic search: top HYBRID_CANDIDATES chunks by embedding similarity.
2. BM25 search:     top HYBRID_CANDIDATES chunks by keyword score.
3. Reciprocal Rank Fusion: each chunk gets sum(1 / (RRF_K + rank)) over the lists it
   appears in. Using ranks (not raw scores) lets us merge two scales that are not
   comparable. Chunks found by BOTH retrievers rise to the top.
4. Cross-encoder rerank of the best RERANK_CANDIDATES fused chunks.
"""
import time
from dataclasses import dataclass, field

from app.core.config import Settings
from app.services.embeddings import Embedder
from app.services.keyword_search import KeywordSearcher
from app.services.reranker import Reranker
from app.services.vector_store import SearchHit, VectorStore


@dataclass
class HybridHit:
    hit: SearchHit
    semantic_rank: int | None = None  # 1-based rank in semantic results (None = not found)
    semantic_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    rrf_score: float = 0.0
    rerank_score: float | None = None

    @property
    def retrieved_by(self) -> list[str]:
        sources = []
        if self.semantic_rank is not None:
            sources.append("semantic")
        if self.bm25_rank is not None:
            sources.append("bm25")
        return sources


@dataclass
class HybridResult:
    hits: list[HybridHit]
    reranker_model: str
    reranked: bool
    timings_ms: dict[str, float] = field(default_factory=dict)


def reciprocal_rank_fusion(
    semantic: list[SearchHit], keyword: list[SearchHit], k: int = 60
) -> list[HybridHit]:
    merged: dict[str, HybridHit] = {}
    for rank, hit in enumerate(semantic, start=1):
        entry = merged.setdefault(hit.chunk_id, HybridHit(hit=hit))
        entry.semantic_rank, entry.semantic_score = rank, hit.score
        entry.rrf_score += 1.0 / (k + rank)
    for rank, hit in enumerate(keyword, start=1):
        entry = merged.setdefault(hit.chunk_id, HybridHit(hit=hit))
        entry.bm25_rank, entry.bm25_score = rank, hit.score
        entry.rrf_score += 1.0 / (k + rank)
    fused = sorted(merged.values(), key=lambda h: h.rrf_score, reverse=True)
    for h in fused:
        h.rrf_score = round(h.rrf_score, 6)
    return fused


def hybrid_search(
    query: str,
    *,
    embedder: Embedder,
    store: VectorStore,
    keyword: KeywordSearcher,
    reranker: Reranker,
    settings: Settings,
    top_k: int,
    document_ids: list[int] | None = None,
    rerank: bool = True,
) -> HybridResult:
    timings: dict[str, float] = {}

    t = time.perf_counter()
    query_vector = embedder.embed([query])[0]
    semantic = store.query(query_vector, settings.HYBRID_CANDIDATES, document_ids)
    timings["semantic"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    bm25 = keyword.search(query, settings.HYBRID_CANDIDATES, document_ids)
    timings["bm25"] = (time.perf_counter() - t) * 1000

    fused = reciprocal_rank_fusion(semantic, bm25, k=settings.RRF_K)

    reranked = False
    if rerank and fused:
        t = time.perf_counter()
        candidates = fused[: settings.RERANK_CANDIDATES]
        scores = reranker.score(query, [h.hit.text for h in candidates])
        for h, s in zip(candidates, scores, strict=True):
            h.rerank_score = round(s, 4)
        fused = sorted(candidates, key=lambda h: h.rerank_score, reverse=True)
        reranked = True
        timings["rerank"] = (time.perf_counter() - t) * 1000

    return HybridResult(
        hits=fused[:top_k],
        reranker_model=reranker.model_name if reranked else "none",
        reranked=reranked,
        timings_ms={k: round(v, 1) for k, v in timings.items()},
    )
