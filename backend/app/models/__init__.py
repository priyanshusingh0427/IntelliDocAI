"""Importing this package registers every table on Base.metadata."""
from app.models.chat import ChatMessage, ChatRole, ChatSession  # noqa: F401
from app.models.document import Document, DocumentPage  # noqa: F401
from app.models.user import User  # noqa: F401
