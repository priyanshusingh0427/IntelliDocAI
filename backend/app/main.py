"""IntelliDoc AI - FastAPI application.

Start the server from the backend/ folder with either command:
    uvicorn app.main:app --reload
    uvicorn main:app --reload        (backend/main.py re-exports this app)

Or from the project root (C:\\Users\\priya\\IntelliDocAI):
    uvicorn app.main:app --reload --app-dir backend
"""
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.router import api_router
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.db.session import init_db
from app.services.llm import LLMUnavailable
from app.services.vector_store import VectorStoreUnavailable

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    init_db()  # create / upgrade database tables on startup
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    setup_logging(settings.LOG_LEVEL)

    application = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        # Never FastAPI/Starlette debug mode: it bypasses the error handler below and sends
        # full Python tracebacks (file paths, code) to the browser. Tracebacks are logged
        # server-side instead (use LOG_LEVEL=DEBUG for more detail). The DEBUG setting is kept
        # so existing .env files stay valid, but it no longer switches this on.
        debug=False,
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.include_router(api_router, prefix=settings.API_V1_PREFIX)

    @application.exception_handler(VectorStoreUnavailable)
    async def vector_store_unavailable_handler(_: Request, exc: VectorStoreUnavailable) -> JSONResponse:
        logger.error("Vector store unavailable: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": f"Search index unavailable: {exc}"},
        )

    @application.exception_handler(LLMUnavailable)
    async def llm_unavailable_handler(_: Request, exc: LLMUnavailable) -> JSONResponse:
        logger.error("LLM unavailable: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": f"Answer generator unavailable: {exc}"},
        )

    @application.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        # Log the full traceback server-side; never leak internals to the client.
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"},
        )

    @application.get("/", tags=["root"])
    def root() -> dict[str, str]:
        return {"message": f"Welcome to {settings.APP_NAME}", "docs": "/docs"}

    logger.info("%s v%s started (%s)", settings.APP_NAME, settings.APP_VERSION, settings.ENVIRONMENT)
    return application


app = create_app()
