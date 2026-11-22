"""Password hashing (Argon2) and login tokens (JWT)."""
from datetime import datetime, timedelta, timezone

import jwt
from pwdlib import PasswordHash

from app.core.config import Settings

# Argon2id with pwdlib's recommended parameters. Passwords are never stored, only the hash.
_hasher = PasswordHash.recommended()


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password: str, hashed: str) -> bool:
    try:
        return _hasher.verify(password, hashed)
    except Exception:  # noqa: BLE001 - malformed hash = no match
        return False


# Verifying against a real hash when the email is unknown keeps response times similar,
# so attackers cannot tell which emails have accounts by timing the login request.
_DUMMY_HASH = _hasher.hash("timing-equalizer-password")


def verify_dummy(password: str) -> None:
    verify_password(password, _DUMMY_HASH)


class InvalidToken(Exception):
    pass


def create_access_token(user_id: int, settings: Settings) -> tuple[str, int]:
    """Return (token, lifetime in seconds)."""
    lifetime = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    now = datetime.now(timezone.utc)
    payload = {"sub": str(user_id), "iat": now, "exp": now + lifetime, "type": "access"}
    token = jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    return token, int(lifetime.total_seconds())


def decode_access_token(token: str, settings: Settings) -> int:
    """Return the user id in a valid, unexpired token, else raise InvalidToken."""
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
            options={"require": ["sub", "exp"]},
        )
        if payload.get("type") != "access":
            raise InvalidToken("wrong token type")
        return int(payload["sub"])
    except (jwt.PyJWTError, ValueError, TypeError) as exc:
        raise InvalidToken(str(exc)) from exc
