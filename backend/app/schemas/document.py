from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class DocumentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    original_filename: str
    content_type: str | None
    extension: str
    size_bytes: int
    status: str
    error_message: str | None
    char_count: int
    page_count: int | None = None
    sha256: str | None = None
    created_at: datetime
    # Phase 3: embedding index
    index_status: str | None = None  # indexed / failed / not_indexed (null = uploaded before Phase 3)
    index_error: str | None = None
    chunk_count: int | None = None
    embedding_model: str | None = None
    indexed_at: datetime | None = None


class PageSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    page_number: int
    char_count: int
    width: float | None = None
    height: float | None = None


class PageRead(PageSummary):
    text: str


class DocumentUploadResponse(DocumentRead):
    """Returned by POST: metadata and per-page stats, without the (possibly huge) full text."""

    # The ORM attribute is doc_metadata (SQLAlchemy reserves .metadata); the API calls it "metadata".
    metadata: dict | None = Field(default=None, validation_alias="doc_metadata")
    pages: list[PageSummary] = []


class DocumentDetail(DocumentUploadResponse):
    extracted_text: str | None


class ErrorResponse(BaseModel):
    detail: str
