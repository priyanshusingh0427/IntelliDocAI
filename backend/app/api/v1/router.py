from fastapi import APIRouter

from app.api.v1.endpoints import ask, auth, chats, documents, health, search

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(auth.router, tags=["auth"])
api_router.include_router(documents.router, tags=["documents"])
api_router.include_router(search.router, tags=["search"])
api_router.include_router(ask.router, tags=["ask (RAG)"])
api_router.include_router(chats.router, tags=["chat history"])
