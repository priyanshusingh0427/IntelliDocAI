from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ChatSessionCreate(BaseModel):
    title: str | None = Field(default=None, max_length=200)


class ChatSessionUpdate(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)


class ChatSessionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int = 0


class ChatMessageRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    role: str
    content: str
    created_at: datetime
    document_ids: list[int] | None = None
    found: bool | None = None
    grounded: bool | None = None
    citations: list[dict] | None = None
    warnings: list[str] | None = None
    llm_model: str | None = None
    timings_ms: dict[str, float] | None = None


class ChatSessionDetail(ChatSessionRead):
    messages: list[ChatMessageRead]
