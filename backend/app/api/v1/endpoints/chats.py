"""Phase 6: chat history. Each user sees and manages only their own conversations."""
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app.api.deps import get_current_user
from app.db.session import get_db
from app.models.chat import ChatMessage, ChatSession
from app.models.user import User
from app.schemas.chat import ChatSessionCreate, ChatSessionDetail, ChatSessionRead, ChatSessionUpdate
from app.schemas.document import ErrorResponse

router = APIRouter(prefix="/chats")
_404 = {404: {"model": ErrorResponse, "description": "Chat not found"}}


def get_owned_chat(db: Session, user: User, chat_id: int, with_messages: bool = False) -> ChatSession:
    stmt = select(ChatSession).where(ChatSession.id == chat_id, ChatSession.user_id == user.id)
    if with_messages:
        stmt = stmt.options(selectinload(ChatSession.messages))
    chat = db.scalar(stmt)
    if chat is None:  # someone else's chat looks exactly like a missing one
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Chat not found")
    return chat


def _read(chat: ChatSession, count: int) -> ChatSessionRead:
    return ChatSessionRead(
        id=chat.id, title=chat.title, created_at=chat.created_at,
        updated_at=chat.updated_at, message_count=count,
    )


@router.get("", response_model=list[ChatSessionRead], summary="My chats, most recently active first")
def list_chats(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> list[ChatSessionRead]:
    counts = (
        select(ChatMessage.session_id, func.count(ChatMessage.id).label("n"))
        .group_by(ChatMessage.session_id)
        .subquery()
    )
    rows = db.execute(
        select(ChatSession, func.coalesce(counts.c.n, 0))
        .outerjoin(counts, counts.c.session_id == ChatSession.id)
        .where(ChatSession.user_id == user.id)
        .order_by(ChatSession.updated_at.desc(), ChatSession.id.desc())
        .offset(skip)
        .limit(limit)
    ).all()
    return [_read(chat, n) for chat, n in rows]


@router.post("", response_model=ChatSessionRead, status_code=status.HTTP_201_CREATED, summary="Start an empty chat")
def create_chat(
    data: ChatSessionCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> ChatSessionRead:
    chat = ChatSession(user_id=user.id, title=(data.title or "").strip() or "New chat")
    db.add(chat)
    db.commit()
    return _read(chat, 0)


@router.get("/{chat_id}", response_model=ChatSessionDetail, responses=_404, summary="A chat with all its messages")
def get_chat(chat_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> ChatSessionDetail:
    chat = get_owned_chat(db, user, chat_id, with_messages=True)
    return ChatSessionDetail(**_read(chat, len(chat.messages)).model_dump(), messages=chat.messages)


@router.patch("/{chat_id}", response_model=ChatSessionRead, responses=_404, summary="Rename a chat")
def rename_chat(
    chat_id: int,
    data: ChatSessionUpdate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> ChatSessionRead:
    chat = get_owned_chat(db, user, chat_id, with_messages=True)
    chat.title = data.title.strip()
    chat.updated_at = datetime.now(timezone.utc)
    db.commit()
    return _read(chat, len(chat.messages))


@router.delete("/{chat_id}", status_code=status.HTTP_204_NO_CONTENT, responses=_404, summary="Delete a chat and its messages")
def delete_chat(chat_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> Response:
    chat = get_owned_chat(db, user, chat_id)
    db.delete(chat)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
