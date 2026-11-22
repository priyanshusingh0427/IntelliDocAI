"""Rerank candidate chunks with a cross-encoder.

Embedding search compares the question and each chunk *separately* (fast, but rough).
A cross-encoder reads the question and one chunk *together* and outputs a relevance
score, which is much more accurate but too slow to run on every chunk. So we use it
only on the ~12 best candidates from hybrid search, and reorder them.

Providers:
- "cross-encoder": cross-encoder/ms-marco-MiniLM-L-6-v2 (local, ~90 MB). Scores are
  logits: above 0 = relevant, strongly negative = unrelated.
- "none": keep the hybrid order (score = fused score). For tests/offline use.
"""
import logging
import threading
from functools import lru_cache
from typing import Protocol

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)


class Reranker(Protocol):
    model_name: str
    produces_relevance_scores: bool  # False -> scores are not comparable to RERANK_MIN_SCORE

    def score(self, query: str, passages: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    produces_relevance_scores = True

    def __init__(self, model_name: str, cache_dir: str) -> None:
        self.model_name = model_name
        self._cache_dir = cache_dir
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import CrossEncoder

                    logger.info("Loading reranker model %s ...", self.model_name)
                    self._model = CrossEncoder(
                        self.model_name, device="cpu", cache_dir=self._cache_dir
                    )
                    logger.info("Reranker model loaded")
        return self._model

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        scores = self._load().predict(
            [(query, p) for p in passages], show_progress_bar=False, convert_to_numpy=True
        )
        return [float(s) for s in scores]


class NoopReranker:
    """Keeps the incoming order: scores descend from 0 so sorting is stable."""

    model_name = "none"
    produces_relevance_scores = False

    def score(self, query: str, passages: list[str]) -> list[float]:
        return [-float(i) for i in range(len(passages))]


def build_reranker(settings: Settings) -> Reranker:
    provider = settings.RERANKER_PROVIDER.lower()
    if provider == "cross-encoder":
        return CrossEncoderReranker(settings.RERANKER_MODEL, str(settings.embedding_cache_path))
    if provider == "none":
        return NoopReranker()
    raise ValueError(f"Unknown RERANKER_PROVIDER: {settings.RERANKER_PROVIDER}")


@lru_cache
def get_reranker() -> Reranker:
    """FastAPI dependency: one shared reranker (model loaded once)."""
    return build_reranker(get_settings())
