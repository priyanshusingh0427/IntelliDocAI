"""IntelliDoc AI - backwards-compatible entry point.

The application now lives in app/main.py, so both of these work from backend/:
    uvicorn app.main:app --reload
    uvicorn main:app --reload
"""
from app.main import app, create_app, lifespan  # noqa: F401

__all__ = ["app", "create_app", "lifespan"]
