"""Phase 4: ask a question, get a grounded answer with page citations (RAG).
Phase 6: only the user's own documents are searched, and every answer is saved to chat history."""
from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, owned_document_ids
from app.api.v1.endpoints.chats import get_owned_chat
from app.core.config import Settings, get_settings
from app.db.session import get_db
from app.models.chat import ChatMessage, ChatRole, ChatSession
from app.models.user import User
from app.schemas.document import ErrorResponse
from app.schemas.rag import AskRequest, AskResponse, Citation
from app.services.embeddings import Embedder, get_embedder
from app.services.keyword_search import get_keyword_searcher
from app.services.llm import LLM, get_llm
from app.services.rag import NOT_FOUND_ANSWER, RagAnswer, Source, answer_question
from app.services.reranker import Reranker, get_reranker
from app.services.vector_store import VectorStore, get_vector_store

router = APIRouter(prefix="/ask")

TITLE_LENGTH = 60


def _citation(source: Source) -> Citation:
    hit = source.hit.hit
    return Citation(
        source=source.number,
        document_id=hit.document_id,
        filename=hit.filename,
        page_number=hit.page_number,
        chunk_id=hit.chunk_id,
        rerank_score=source.hit.rerank_score,
        text=source.text,
    )


def _chat_title(question: str) -> str:
    title = " ".join(question.split())
    return title if len(title) <= TITLE_LENGTH else title[: TITLE_LENGTH - 1].rstrip() + "…"


@router.post(
    "",
    response_model=AskResponse,
    responses={
        404: {"model": ErrorResponse, "description": "chat_id not found"},
        503: {"model": ErrorResponse, "description": "LLM or search index unavailable"},
    },
    summary="Ask a question about your documents (hybrid retrieval + rerank + local LLM)",
)
async def ask(
    request: AskRequest,
    settings: Settings = Depends(get_settings),
    embedder: Embedder = Depends(get_embedder),
    store: VectorStore = Depends(get_vector_store),
    reranker: Reranker = Depends(get_reranker),
    llm: LLM = Depends(get_llm),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> AskResponse:
    # Check the chat first, so a bad chat_id fails before the (slow) LLM runs.
    chat = get_owned_chat(db, user, request.chat_id) if request.chat_id is not None else None
    scope = owned_document_ids(db, user, request.document_ids)

    if scope:
        result = await run_in_threadpool(
            lambda: answer_question(
                request.question,
                embedder=embedder,
                store=store,
                keyword=get_keyword_searcher(store),
                reranker=reranker,
                llm=llm,
                settings=settings,
                document_ids=scope,
            )
        )
    else:  # no documents of this user to search
        result = RagAnswer(
            answer=NOT_FOUND_ANSWER, grounded=False, found=False, citations=[], sources=[],
            llm_model=llm.model_name, reranker_model=reranker.model_name, timings_ms={},
        )

    citations = [_citation(s) for s in result.citations]

    # Save the exchange only after a successful answer (errors leave history unchanged).
    now = datetime.now(timezone.utc)
    if chat is None:
        chat = ChatSession(user_id=user.id, title=_chat_title(request.question), created_at=now)
        db.add(chat)
    chat.updated_at = now
    db.add(
        ChatMessage(
            session=chat, role=ChatRole.USER, content=request.question.strip(),
            document_ids=request.document_ids or None, created_at=now,
        )
    )
    answer_message = ChatMessage(
        session=chat,
        role=ChatRole.ASSISTANT,
        content=result.answer,
        found=result.found,
        grounded=result.grounded,
        citations=[c.model_dump() for c in citations],
        warnings=result.warnings,
        llm_model=result.llm_model,
        timings_ms=result.timings_ms,
        created_at=now,
    )
    db.add(answer_message)
    db.commit()

    return AskResponse(
        question=request.question,
        answer=result.answer,
        found=result.found,
        grounded=result.grounded,
        citations=citations,
        sources=[_citation(s) for s in result.sources],
        llm_model=result.llm_model,
        reranker_model=result.reranker_model,
        warnings=result.warnings,
        timings_ms=result.timings_ms,
        chat_id=chat.id,
        message_id=answer_message.id,
    )
