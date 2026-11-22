from pydantic import BaseModel, Field


class HybridSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1)
    document_ids: list[int] | None = None
    rerank: bool = Field(default=True, description="Reorder results with the cross-encoder")


class HybridSearchResult(BaseModel):
    document_id: int
    filename: str
    page_number: int
    chunk_index: int
    chunk_id: str
    text: str
    retrieved_by: list[str] = Field(..., description='"semantic", "bm25" or both')
    semantic_rank: int | None
    semantic_score: float | None
    bm25_rank: int | None
    bm25_score: float | None
    rrf_score: float
    rerank_score: float | None


class HybridSearchResponse(BaseModel):
    query: str
    reranked: bool
    reranker_model: str
    timings_ms: dict[str, float]
    results: list[HybridSearchResult]


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, examples=["What is the methodology?"])
    document_ids: list[int] | None = Field(default=None, description="Limit to these documents")
    chat_id: int | None = Field(
        default=None, description="Add to this chat (default: start a new chat titled by the question)"
    )


class Citation(BaseModel):
    source: int = Field(..., description="The [n] number used in the answer")
    document_id: int
    filename: str
    page_number: int
    chunk_id: str
    rerank_score: float | None
    text: str = Field(..., description="The source passage given to the LLM")


class AskResponse(BaseModel):
    question: str
    answer: str
    found: bool = Field(..., description="False if the documents do not contain the answer")
    grounded: bool = Field(..., description="True if the answer cites at least one source")
    citations: list[Citation] = Field(..., description="Sources the answer actually cites")
    sources: list[Citation] = Field(..., description="All sources given to the LLM")
    llm_model: str
    reranker_model: str
    warnings: list[str]
    timings_ms: dict[str, float]
    chat_id: int | None = Field(default=None, description="The chat this question was saved to")
    message_id: int | None = Field(default=None, description="The saved answer message")
