import logging

from fastapi import APIRouter, Depends, File, HTTPException, Path, Query, Response, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.api.deps import get_current_user
from app.core.config import Settings, get_settings
from app.db.session import get_db
from app.models.document import Document, DocumentPage, DocumentStatus, IndexStatus
from app.models.user import User
from app.schemas.document import (
    DocumentDetail,
    DocumentRead,
    DocumentUploadResponse,
    ErrorResponse,
    PageRead,
)
from app.schemas.search import ChunkRead, IndexResponse
from app.services import storage
from app.services.embeddings import Embedder, get_embedder
from app.services.extraction import ExtractionError, extract
from app.services.indexing import IndexingError, index_document
from app.services.vector_store import VectorStore, VectorStoreUnavailable, get_vector_store

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents")

_ERRORS = {
    400: {"model": ErrorResponse, "description": "Bad request (wrong extension, empty file)"},
    404: {"model": ErrorResponse, "description": "Document not found"},
    413: {"model": ErrorResponse, "description": "File too large or too many pages"},
    415: {"model": ErrorResponse, "description": "Not a real file of the declared type"},
    422: {"model": ErrorResponse, "description": "Corrupted or password-protected file"},
    500: {"model": ErrorResponse, "description": "Indexing (embedding / ChromaDB) failed"},
}


def _get_or_404(db: Session, document_id: int, user: User, with_pages: bool = False) -> Document:
    # Another user's document is reported as "not found", never as "forbidden".
    stmt = select(Document).where(Document.id == document_id, Document.owner_id == user.id)
    if with_pages:
        stmt = stmt.options(selectinload(Document.pages))
    document = db.scalar(stmt)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    return document


@router.post(
    "",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    responses={k: _ERRORS[k] for k in (400, 413, 415, 422)},
    summary="Upload a PDF (or DOCX/TXT), extract its text page by page and index it for search",
)
async def upload_document(
    file: UploadFile = File(..., description="PDF, DOCX or TXT file"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
    embedder: Embedder = Depends(get_embedder),
    store: VectorStore = Depends(get_vector_store),
) -> Document:
    stored = await storage.save_upload(file, settings)

    try:
        # PyMuPDF is CPU-bound; keep the event loop free.
        result = await run_in_threadpool(extract, stored.path, stored.extension, settings.MAX_PDF_PAGES)
    except ExtractionError as exc:
        storage.delete_file(stored.stored_filename, settings)
        logger.info("Rejected upload %r: %s", file.filename, exc)
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    except Exception:
        storage.delete_file(stored.stored_filename, settings)
        raise

    full_text = result.full_text
    document = Document(
        original_filename=storage.sanitize_filename(file.filename or stored.stored_filename),
        stored_filename=stored.stored_filename,
        content_type=file.content_type,
        extension=stored.extension,
        size_bytes=stored.size_bytes,
        sha256=stored.sha256,
        owner_id=user.id,
        status=DocumentStatus.PROCESSED,
        extracted_text=full_text,
        char_count=len(full_text),
        page_count=result.page_count,
        doc_metadata=result.metadata,
        pages=[
            DocumentPage(
                page_number=p.page_number,
                text=p.text,
                char_count=len(p.text),
                width=p.width,
                height=p.height,
            )
            for p in result.pages
        ],
    )

    try:
        db.add(document)
        db.commit()
    except Exception:
        db.rollback()
        storage.delete_file(stored.stored_filename, settings)
        raise
    logger.info(
        "Stored document %s (%s, %d pages, %d chars)",
        document.id, document.original_filename, result.page_count, len(full_text),
    )

    # Phase 3: chunk + embed + store in ChromaDB. An indexing failure does not undo the
    # upload: the document is kept with index_status="failed" and can be re-indexed later.
    if settings.AUTO_INDEX_ON_UPLOAD:
        try:
            await run_in_threadpool(index_document, document, embedder, store, settings)
        except IndexingError:
            pass  # status + error already recorded on the document
        db.commit()
    else:
        document.index_status = IndexStatus.NOT_INDEXED
        db.commit()
    return document


@router.get("", response_model=list[DocumentRead], summary="List uploaded documents (newest first)")
def list_documents(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> list[Document]:
    stmt = select(Document).where(Document.owner_id == user.id).order_by(Document.id.desc()).offset(skip).limit(limit)
    return list(db.scalars(stmt))


@router.get(
    "/{document_id}",
    response_model=DocumentDetail,
    responses={404: _ERRORS[404]},
    summary="Get a document with metadata, page summary and full text",
)
def get_document(
    document_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)
) -> Document:
    return _get_or_404(db, document_id, user, with_pages=True)


@router.get(
    "/{document_id}/pages",
    response_model=list[PageRead],
    responses={404: _ERRORS[404]},
    summary="Get the extracted text of every page, with page numbers",
)
def list_pages(
    document_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)
) -> list[DocumentPage]:
    return _get_or_404(db, document_id, user, with_pages=True).pages


@router.get(
    "/{document_id}/pages/{page_number}",
    response_model=PageRead,
    responses={404: _ERRORS[404]},
    summary="Get the extracted text of one page (1-based page number)",
)
def get_page(
    document_id: int,
    page_number: int = Path(..., ge=1),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> DocumentPage:
    _get_or_404(db, document_id, user)
    page = db.scalar(
        select(DocumentPage).where(
            DocumentPage.document_id == document_id, DocumentPage.page_number == page_number
        )
    )
    if page is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Page not found")
    return page


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={404: _ERRORS[404]},
    summary="Delete a document, its pages, its search vectors and the stored file",
)
def delete_document(
    document_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
    store: VectorStore = Depends(get_vector_store),
) -> Response:
    document = _get_or_404(db, document_id, user)
    stored_filename = document.stored_filename
    # Remove vectors first: if ChromaDB errors, the database row is kept (no orphans).
    # If ChromaDB is not available at all, still allow deleting the document itself.
    try:
        store.delete_document(document_id)
    except VectorStoreUnavailable as exc:
        logger.warning("Deleting document %s without removing vectors: %s", document_id, exc)
    db.delete(document)
    db.commit()
    storage.delete_file(stored_filename, settings)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ---------------------------------------------------------------- Phase 3


@router.post(
    "/{document_id}/index",
    response_model=IndexResponse,
    responses={404: _ERRORS[404], 500: _ERRORS[500]},
    summary="(Re)build the chunks and embeddings for a document",
)
async def reindex_document(
    document_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
    embedder: Embedder = Depends(get_embedder),
    store: VectorStore = Depends(get_vector_store),
) -> IndexResponse:
    document = _get_or_404(db, document_id, user, with_pages=True)
    try:
        count = await run_in_threadpool(index_document, document, embedder, store, settings)
    except IndexingError as exc:
        db.commit()  # persist the "failed" status
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Indexing failed: {exc}"
        ) from exc
    db.commit()
    return IndexResponse(
        document_id=document.id,
        index_status=document.index_status,
        chunk_count=count,
        embedding_model=document.embedding_model or embedder.model_name,
    )


@router.get(
    "/{document_id}/chunks",
    response_model=list[ChunkRead],
    responses={404: _ERRORS[404]},
    summary="List the stored chunks of a document with page numbers and offsets",
)
def list_chunks(
    document_id: int,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    store: VectorStore = Depends(get_vector_store),
) -> list[ChunkRead]:
    _get_or_404(db, document_id, user)
    hits = store.get_document_chunks(document_id)[skip : skip + limit]
    return [
        ChunkRead(
            chunk_id=h.chunk_id,
            document_id=h.document_id,
            chunk_index=h.chunk_index,
            page_number=h.page_number,
            char_start=h.char_start,
            char_end=h.char_end,
            char_count=len(h.text),
            text=h.text,
        )
        for h in hits
    ]
