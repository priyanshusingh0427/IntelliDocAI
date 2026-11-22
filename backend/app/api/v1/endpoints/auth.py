"""Phase 6: create an account, log in (JWT bearer token), and read the current user."""
import logging

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.config import Settings, get_settings
from app.core.security import create_access_token, hash_password, verify_dummy, verify_password
from app.db.session import get_db
from app.models.document import Document
from app.models.user import User
from app.schemas.auth import RegisterResponse, Token, UserCreate, UserRead
from app.schemas.document import ErrorResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth")


def _token_response(user: User, settings: Settings) -> dict:
    token, expires_in = create_access_token(user.id, settings)
    return {"access_token": token, "expires_in": expires_in, "user": user}


@router.post(
    "/register",
    response_model=RegisterResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        403: {"model": ErrorResponse, "description": "Registration is disabled"},
        409: {"model": ErrorResponse, "description": "Email already registered"},
    },
    summary="Create an account and get a login token",
)
def register(
    data: UserCreate,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> dict:
    if not settings.ALLOW_REGISTRATION:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Registration is disabled on this server")
    if len(data.password) < settings.PASSWORD_MIN_LENGTH:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"Password must be at least {settings.PASSWORD_MIN_LENGTH} characters",
        )
    if db.scalar(select(User.id).where(User.email == data.email)) is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, "An account with this email already exists")

    is_first_user = db.scalar(select(func.count(User.id))) == 0
    user = User(email=data.email, hashed_password=hash_password(data.password), full_name=data.full_name)
    db.add(user)
    try:
        db.flush()
    except IntegrityError:  # two registrations with the same email at the same time
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, "An account with this email already exists") from None

    adopted = 0
    if is_first_user and settings.ADOPT_UNOWNED_DOCUMENTS:
        result = db.execute(
            update(Document).where(Document.owner_id.is_(None)).values(owner_id=user.id)
        )
        adopted = result.rowcount or 0
    db.commit()
    logger.info("Registered user %s (adopted %d older documents)", user.id, adopted)
    return {**_token_response(user, settings), "adopted_documents": adopted}


@router.post(
    "/login",
    response_model=Token,
    responses={401: {"model": ErrorResponse, "description": "Wrong email or password"}},
    summary="Log in with email + password (form fields: username, password)",
)
def login(
    form: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> dict:
    # OAuth2 form field is called "username"; IntelliDoc uses the email address.
    user = db.scalar(select(User).where(User.email == form.username.strip().lower()))
    if user is None:
        verify_dummy(form.password)
        ok = False
    else:
        ok = verify_password(form.password, user.hashed_password) and user.is_active
    if not ok:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return _token_response(user, settings)


@router.get("/me", response_model=UserRead, summary="The logged-in user")
def me(user: User = Depends(get_current_user)) -> User:
    return user
