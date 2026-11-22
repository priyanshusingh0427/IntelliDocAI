"""Search endpoints.

/search         Phase 3 semantic search (embeddings only).
/search/hybrid  Phase 4 hybrid search: semantic + BM25, fused with RRF, then reranked.
"""
from fastapi import APIRouter, Depends
from fastapi.concurrency import run_in_threadpool

from sqlalchemy.orm import Session

from app.api.deps import get_current_user, owned_document_ids
from app.core.config import Settings, get_settings
from app.db.session import get_db
from app.models.user import User
from app.schemas.rag import HybridSearchRequest, HybridSearchResponse, HybridSearchResult
from app.schemas.search import SearchRequest, SearchResponse, SearchResult
from app.services.embeddings import Embedder, get_embedder
from app.services.hybrid_search import hybrid_search
from app.services.keyword_search import get_keyword_searcher
from app.services.reranker import Reranker, get_reranker
from app.services.vector_store import VectorStore, get_vector_store

router = APIRouter(prefix="/search")


@router.post(
    "/hybrid",
    response_model=HybridSearchResponse,
    summary="Hybrid search: semantic + BM25 keyword, fused (RRF) and reranked (cross-encoder)",
)
async def hybrid(
    request: HybridSearchRequest,
    settings: Settings = Depends(get_settings),
    embedder: Embedder = Depends(get_embedder),
    store: VectorStore = Depends(get_vector_store),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    reranker: Reranker = Depends(get_reranker),
) -> HybridSearchResponse:
    top_k = min(request.top_k or settings.SEARCH_DEFAULT_TOP_K, settings.SEARCH_MAX_TOP_K)
    scope = owned_document_ids(db, user, request.document_ids)
    if not scope:  # the user has no (matching) documents: nothing to search
        return HybridSearchResponse(
            query=request.query, reranked=False, reranker_model=reranker.model_name,
            timings_ms={}, results=[],
        )
    result = await run_in_threadpool(
        lambda: hybrid_search(
            request.query.strip(),
            embedder=embedder,
            store=store,
            keyword=get_keyword_searcher(store),
            reranker=reranker,
            settings=settings,
            top_k=top_k,
            document_ids=scope,
            rerank=request.rerank,
        )
    )
    return HybridSearchResponse(
        query=request.query,
        reranked=result.reranked,
        reranker_model=result.reranker_model,
        timings_ms=result.timings_ms,
        results=[
            HybridSearchResult(
                document_id=h.hit.document_id,
                filename=h.hit.filename,
                page_number=h.hit.page_number,
                chunk_index=h.hit.chunk_index,
                chunk_id=h.hit.chunk_id,
                text=h.hit.text,
                retrieved_by=h.retrieved_by,
                semantic_rank=h.semantic_rank,
                semantic_score=h.semantic_score,
                bm25_rank=h.bm25_rank,
                bm25_score=h.bm25_score,
                rrf_score=h.rrf_score,
                rerank_score=h.rerank_score,
            )
            for h in result.hits
        ],
    )


@router.post("", response_model=SearchResponse, summary="Semantic search across indexed documents")
async def semantic_search(
    request: SearchRequest,
    settings: Settings = Depends(get_settings),
    embedder: Embedder = Depends(get_embedder),
    store: VectorStore = Depends(get_vector_store),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> SearchResponse:
    top_k = min(request.top_k or settings.SEARCH_DEFAULT_TOP_K, settings.SEARCH_MAX_TOP_K)
    scope = owned_document_ids(db, user, request.document_ids)
    if not scope:
        return SearchResponse(query=request.query, embedding_model=embedder.model_name, results=[])
    query_vector = (await run_in_threadpool(embedder.embed, [request.query.strip()]))[0]
    hits = await run_in_threadpool(store.query, query_vector, top_k, scope)
    return SearchResponse(
        query=request.query,
        embedding_model=embedder.model_name,
        results=[
            SearchResult(
                score=h.score,
                document_id=h.document_id,
                filename=h.filename,
                page_number=h.page_number,
                chunk_index=h.chunk_index,
                chunk_id=h.chunk_id,
                text=h.text,
            )
            for h in hits
        ],
    )
