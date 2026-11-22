"""ChromaDB wrapper: stores chunk vectors + text + source metadata, and searches them.

ChromaDB is a vector database. We give it each chunk's embedding together with the
chunk text and where it came from (document id, file name, page number...). At search
time we give it the query's embedding and it returns the closest chunks.

We compute embeddings ourselves (embeddings.py) and pass them in, so Chroma never
downloads its own model, and the same embedder is used for documents and queries.
"""
import logging
import threading
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable

import numpy as np

from app.core.config import get_settings
from app.services.chunking import Chunk

logger = logging.getLogger(__name__)

# NOTE: chromadb is imported lazily (only when the vector store is first used), so the
# API - health, docs, upload, listing - still starts if ChromaDB is missing or broken.


class VectorStoreUnavailable(RuntimeError):
    """ChromaDB is not installed or could not be opened. Mapped to HTTP 503."""


@dataclass
class SearchHit:
    chunk_id: str
    document_id: int
    filename: str
    page_number: int
    chunk_index: int
    char_start: int
    char_end: int
    text: str
    score: float  # cosine similarity: 1.0 = identical meaning, ~0 = unrelated


def chunk_id(document_id: int, chunk_index: int) -> str:
    return f"doc{document_id}-chunk{chunk_index}"


class VectorStore:
    """Pass either a ready Chroma `client` (tests) or a `client_factory` called on first use."""

    def __init__(
        self,
        client: Any = None,
        collection_name: str = "intellidoc_chunks",
        client_factory: Callable[[], Any] | None = None,
    ) -> None:
        if client is None and client_factory is None:
            raise ValueError("Provide a Chroma client or a client_factory")
        self._client = client
        self._client_factory = client_factory
        self._collection_name = collection_name
        self._collection_obj = None
        # First use can come from several request threads at once (e.g. an upload and the
        # readiness check). Creating two Chroma clients on a fresh folder races its schema
        # setup ("no such table: tenants") and leaves a broken client cached for the folder.
        self._init_lock = threading.Lock()
        # Bumped on every add/delete so caches built from the chunks (BM25) know to rebuild.
        self.version = 0

    @property
    def _collection(self):
        if self._collection_obj is not None:
            return self._collection_obj
        with self._init_lock:
            if self._collection_obj is not None:  # another thread finished while we waited
                return self._collection_obj
            try:
                if self._client is None:
                    self._client = self._client_factory()
                self._collection_obj = self._client.get_or_create_collection(
                    name=self._collection_name,
                    embedding_function=None,  # we always supply embeddings ourselves
                    metadata={"hnsw:space": "cosine"},
                )
            except ImportError as exc:
                raise VectorStoreUnavailable(
                    "ChromaDB is not installed. Run: pip install -r requirements.txt"
                ) from exc
            except Exception as exc:  # noqa: BLE001
                raise VectorStoreUnavailable(f"ChromaDB could not be opened: {exc}") from exc
        return self._collection_obj

    @property
    def collection(self):
        return self._collection

    def count(self) -> int:
        return self._collection.count()

    def add_chunks(
        self,
        document_id: int,
        filename: str,
        chunks: list[Chunk],
        embeddings: np.ndarray,
        embedding_model: str,
        batch_size: int = 500,
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        for i in range(0, len(chunks), batch_size):
            part = chunks[i : i + batch_size]
            self._collection.upsert(
                ids=[chunk_id(document_id, c.chunk_index) for c in part],
                embeddings=embeddings[i : i + batch_size].tolist(),
                documents=[c.text for c in part],
                metadatas=[
                    {
                        "document_id": document_id,
                        "filename": filename,
                        "page_number": c.page_number,
                        "chunk_index": c.chunk_index,
                        "char_start": c.char_start,
                        "char_end": c.char_end,
                        "embedding_model": embedding_model,
                    }
                    for c in part
                ],
            )
        self.version += 1

    def delete_document(self, document_id: int) -> None:
        self._collection.delete(where={"document_id": document_id})
        self.version += 1

    def get_all_chunks(self) -> list[SearchHit]:
        """Every stored chunk (text + metadata, no vectors). Used to build the BM25 index."""
        result = self._collection.get(include=["documents", "metadatas"])
        return [
            self._hit(cid, text, meta, score=0.0)
            for cid, text, meta in zip(result["ids"], result["documents"], result["metadatas"], strict=True)
        ]

    def get_document_chunks(self, document_id: int) -> list[SearchHit]:
        result = self._collection.get(
            where={"document_id": document_id}, include=["documents", "metadatas"]
        )
        hits = [
            self._hit(cid, text, meta, score=1.0)
            for cid, text, meta in zip(result["ids"], result["documents"], result["metadatas"], strict=True)
        ]
        return sorted(hits, key=lambda h: h.chunk_index)

    def query(
        self,
        embedding: np.ndarray,
        top_k: int,
        document_ids: list[int] | None = None,
    ) -> list[SearchHit]:
        total = self.count()
        if total == 0:
            return []
        top_k = min(top_k, total)  # asking Chroma for more than it holds only logs a warning
        where = None
        if document_ids:
            where = (
                {"document_id": document_ids[0]}
                if len(document_ids) == 1
                else {"document_id": {"$in": document_ids}}
            )
        result = self._collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        return [
            # Chroma's cosine "distance" is 1 - cosine similarity.
            self._hit(cid, text, meta, score=round(1.0 - float(dist), 4))
            for cid, text, meta, dist in zip(
                result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0], strict=True
            )
        ]

    @staticmethod
    def _hit(cid: str, text: str, meta: dict, score: float) -> SearchHit:
        return SearchHit(
            chunk_id=cid,
            document_id=int(meta["document_id"]),
            filename=str(meta.get("filename", "")),
            page_number=int(meta["page_number"]),
            chunk_index=int(meta["chunk_index"]),
            char_start=int(meta.get("char_start", 0)),
            char_end=int(meta.get("char_end", 0)),
            text=text,
            score=score,
        )


def make_persistent_store(path: str, collection_name: str) -> VectorStore:
    def factory():
        import chromadb  # lazy: see note at top of module
        from chromadb.config import Settings as ChromaSettings

        return chromadb.PersistentClient(
            path=path, settings=ChromaSettings(anonymized_telemetry=False)
        )

    return VectorStore(collection_name=collection_name, client_factory=factory)


@lru_cache
def get_vector_store() -> VectorStore:
    """FastAPI dependency: one shared ChromaDB collection stored in CHROMA_DIR."""
    settings = get_settings()
    settings.chroma_path.mkdir(parents=True, exist_ok=True)
    return make_persistent_store(str(settings.chroma_path), settings.CHROMA_COLLECTION)
