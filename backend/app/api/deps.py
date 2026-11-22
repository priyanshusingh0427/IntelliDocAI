"""Shared FastAPI dependencies for login and per-user data access."""
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.security import InvalidToken, decode_access_token
from app.db.session import get_db
from app.models.document import Document
from app.models.user import User

# tokenUrl powers the "Authorize" button in /docs (username field = email).
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login", auto_error=False)

_UNAUTHORIZED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Not logged in or the session expired. Please log in again.",
    headers={"WWW-Authenticate": "Bearer"},
)


def get_current_user(
    token: str | None = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> User:
    if not token:
        raise _UNAUTHORIZED
    try:
        user_id = decode_access_token(token, settings)
    except InvalidToken:
        raise _UNAUTHORIZED from None
    user = db.get(User, user_id)
    if user is None or not user.is_active:
        raise _UNAUTHORIZED
    return user


def owned_document_ids(db: Session, user: User, requested: list[int] | None) -> list[int]:
    """The documents a search/question may use: the user's own, optionally narrowed to
    `requested`. Ids the user does not own are silently ignored (never searched).
    An empty result means "nothing to search", so callers must not pass it on as None/all."""
    owned = set(db.scalars(select(Document.id).where(Document.owner_id == user.id)))
    if requested:
        return [i for i in dict.fromkeys(requested) if i in owned]
    return sorted(owned)
