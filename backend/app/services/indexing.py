"""Index a document: chunk its pages -> embed the chunks -> store them in ChromaDB."""
import logging
from datetime import datetime, timezone

from app.core.config import Settings
from app.models.document import Document, IndexStatus
from app.services.chunking import chunk_pages
from app.services.embeddings import Embedder
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)


class IndexingError(Exception):
    pass


def index_document(
    document: Document, embedder: Embedder, store: VectorStore, settings: Settings
) -> int:
    """(Re)build the vector index for one document and record the result on it.

    Safe to call repeatedly: old vectors for the document are removed first.
    Returns the number of chunks stored. On failure the document is marked
    index_status="failed" with the error, and IndexingError is raised.
    The caller commits the database session.
    """
    try:
        chunks = chunk_pages(document.pages, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
        store.delete_document(document.id)
        if chunks:
            vectors = embedder.embed([c.text for c in chunks])
            store.add_chunks(
                document.id, document.original_filename, chunks, vectors, embedder.model_name
            )
    except Exception as exc:  # noqa: BLE001 - model, Chroma or chunking failure
        logger.exception("Indexing failed for document %s", document.id)
        document.index_status = IndexStatus.FAILED
        document.index_error = f"{type(exc).__name__}: {exc}"
        document.chunk_count = 0
        raise IndexingError(document.index_error) from exc

    document.index_status = IndexStatus.INDEXED
    document.index_error = None
    document.chunk_count = len(chunks)
    document.embedding_model = embedder.model_name
    document.indexed_at = datetime.now(timezone.utc)
    logger.info("Indexed document %s: %d chunks", document.id, len(chunks))
    return len(chunks)
