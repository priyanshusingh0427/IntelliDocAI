"""BM25 keyword search over the stored chunks.

Semantic search (embeddings) is good at meaning but can miss exact terms: product
codes, names, numbers, rare jargon. BM25 is the classic search-engine ranking: it
scores chunks by how often the query's words appear in them, giving rare words more
weight and not over-rewarding long chunks. Phase 4 combines both (hybrid search).

The BM25 index is built in memory from the chunks stored in ChromaDB and rebuilt
automatically whenever documents are added or removed (VectorStore.version changes).
"""
import re
import threading
from dataclasses import dataclass, replace

from rank_bm25 import BM25Plus

from app.services.vector_store import SearchHit, VectorStore

_TOKEN = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")

# Very common English words carry no meaning for keyword matching.
STOPWORDS = frozenset(
    """a an and are as at be but by can did do does for from had has have how i if in into is
    it its me my no not of on or our so than that the their them then there these they this
    to was we were what when where which who why will with would you your""".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, keeping codes like "zx-4471" or "v2.1" whole."""
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


@dataclass
class _Index:
    version: int
    hits: list[SearchHit]
    token_sets: list[frozenset[str]]
    # BM25Plus rather than classic BM25Okapi: Okapi gives NEGATIVE weights to words that
    # appear in more than half the chunks, which breaks ranking on small collections.
    bm25: BM25Plus | None


class KeywordSearcher:
    def __init__(self, store: VectorStore) -> None:
        self._store = store
        self._index: _Index | None = None
        self._lock = threading.Lock()

    def _get_index(self) -> _Index:
        with self._lock:
            if self._index is None or self._index.version != self._store.version:
                hits = self._store.get_all_chunks()
                corpus = [tokenize(h.text) for h in hits]
                # rank_bm25 fails on an empty corpus; also skip if every chunk is empty.
                bm25 = BM25Plus(corpus) if any(corpus) else None
                self._index = _Index(
                    version=self._store.version,
                    hits=hits,
                    token_sets=[frozenset(tokens) for tokens in corpus],
                    bm25=bm25,
                )
            return self._index

    def search(
        self, query: str, top_k: int, document_ids: list[int] | None = None
    ) -> list[SearchHit]:
        index = self._get_index()
        terms = tokenize(query)
        if index.bm25 is None or not terms:
            return []
        scores = index.bm25.get_scores(terms)
        allowed = set(document_ids) if document_ids else None
        query_terms = set(terms)
        # A chunk is a keyword match only if it contains at least one query word
        # (BM25Plus gives every chunk a small baseline score, so the score alone can't tell).
        candidates = [
            i
            for i, tokens in enumerate(index.token_sets)
            if tokens & query_terms
            and (allowed is None or index.hits[i].document_id in allowed)
        ]
        ranked = sorted(candidates, key=lambda i: scores[i], reverse=True)
        results: list[SearchHit] = []
        for i in ranked:
            hit = index.hits[i]
            results.append(replace(hit, score=round(float(scores[i]), 4)))
            if len(results) >= top_k:
                break
        return results


_searchers: dict[int, KeywordSearcher] = {}
_searchers_lock = threading.Lock()


def get_keyword_searcher(store: VectorStore) -> KeywordSearcher:
    """One cached searcher per vector store (so the BM25 index is reused)."""
    with _searchers_lock:
        searcher = _searchers.get(id(store))
        if searcher is None or searcher._store is not store:
            searcher = KeywordSearcher(store)
            _searchers[id(store)] = searcher
        return searcher
