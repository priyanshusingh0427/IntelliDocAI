"""Turn text into vectors (embeddings).

An embedding is a list of numbers that captures the *meaning* of a text, so texts
about similar things end up close together even when they use different words.

Providers:
- "sentence-transformers": real local model (default all-MiniLM-L6-v2, 384 numbers
  per text). Downloaded once into EMBEDDING_CACHE_DIR, then runs offline on the CPU.
- "hash": tiny deterministic bag-of-words embedder with no downloads. Used by the
  automated tests and as an offline fallback; it only matches shared words.
"""
import hashlib
import logging
import re
import threading
from functools import lru_cache
from typing import Protocol

import numpy as np

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    model_name: str
    dimension: int

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dimension) float32 array of L2-normalised vectors."""
        ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, cache_dir: str, batch_size: int = 32) -> None:
        self.model_name = model_name
        self._cache_dir = cache_dir
        self._batch_size = batch_size
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        # Loading takes a few seconds (and downloads the model the first time),
        # so it happens lazily on first use, once, even with concurrent requests.
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    logger.info("Loading embedding model %s ...", self.model_name)
                    self._model = SentenceTransformer(
                        self.model_name, cache_folder=self._cache_dir, device="cpu"
                    )
                    logger.info("Embedding model loaded")
        return self._model

    @property
    def dimension(self) -> int:
        return int(self._load().get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vectors = self._load().encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.astype(np.float32)


_TOKEN = re.compile(r"[a-z0-9]+")


class HashEmbedder:
    """Deterministic bag-of-words vectors via feature hashing (no model download)."""

    def __init__(self, dimension: int = 256) -> None:
        self.dimension = dimension
        self.model_name = f"hash-bow-{dimension}"

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dimension, dtype=np.float32)
        for token in _TOKEN.findall(text.lower()):
            digest = hashlib.md5(token.encode(), usedforsecurity=False).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimension
            vec[index] += 1.0 if digest[4] % 2 == 0 else -1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return np.stack([self._vector(t) for t in texts])


def build_embedder(settings: Settings) -> Embedder:
    provider = settings.EMBEDDING_PROVIDER.lower()
    if provider == "sentence-transformers":
        return SentenceTransformerEmbedder(
            settings.EMBEDDING_MODEL,
            cache_dir=str(settings.embedding_cache_path),
            batch_size=settings.EMBEDDING_BATCH_SIZE,
        )
    if provider == "hash":
        return HashEmbedder()
    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {settings.EMBEDDING_PROVIDER}")


@lru_cache
def get_embedder() -> Embedder:
    """FastAPI dependency: one shared embedder (the model is loaded once)."""
    return build_embedder(get_settings())
