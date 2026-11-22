from pydantic import BaseModel, Field


class ChunkRead(BaseModel):
    chunk_id: str
    document_id: int
    chunk_index: int
    page_number: int
    char_start: int
    char_end: int
    char_count: int
    text: str


class IndexResponse(BaseModel):
    document_id: int
    index_status: str
    chunk_count: int
    embedding_model: str


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, examples=["What is the methodology?"])
    top_k: int | None = Field(default=None, ge=1, description="Number of results (default 5)")
    document_ids: list[int] | None = Field(
        default=None, description="Only search these documents (default: all)"
    )


class SearchResult(BaseModel):
    score: float = Field(..., description="Cosine similarity, higher is more relevant")
    document_id: int
    filename: str
    page_number: int
    chunk_index: int
    chunk_id: str
    text: str


class SearchResponse(BaseModel):
    query: str
    embedding_model: str
    results: list[SearchResult]
