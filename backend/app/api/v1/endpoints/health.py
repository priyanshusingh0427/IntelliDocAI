from fastapi import APIRouter, Depends, Response, status
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.db.session import get_db
from app.schemas.health import ComponentStatus, HealthResponse, ReadinessResponse
from app.services.llm import LLM, get_llm
from app.services.vector_store import VectorStore, get_vector_store

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check(settings: Settings = Depends(get_settings)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT,
    )


def _check_database(db: Session) -> ComponentStatus:
    try:
        db.execute(text("SELECT 1"))
        return ComponentStatus(ok=True, detail=f"{db.get_bind().dialect.name} reachable")
    except Exception as exc:  # noqa: BLE001
        return ComponentStatus(ok=False, detail=f"Database error: {exc.__class__.__name__}: {str(exc)[:200]}")


def _check_vector_store(store: VectorStore) -> ComponentStatus:
    try:
        return ComponentStatus(ok=True, detail=f"ChromaDB reachable, {store.count()} chunks indexed")
    except Exception as exc:  # noqa: BLE001
        return ComponentStatus(ok=False, detail=f"ChromaDB error: {exc.__class__.__name__}: {str(exc)[:200]}")


def _check_llm(llm: LLM) -> ComponentStatus:
    try:
        ok, detail = llm.check()
        return ComponentStatus(ok=ok, detail=detail)
    except Exception as exc:  # noqa: BLE001
        return ComponentStatus(ok=False, detail=f"LLM check failed: {exc.__class__.__name__}: {str(exc)[:200]}")


@router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    responses={503: {"model": ReadinessResponse, "description": "A component is not ready"}},
    summary="Readiness: checks the database, the vector store and the LLM (no login needed)",
)
async def readiness(
    response: Response,
    db: Session = Depends(get_db),
    store: VectorStore = Depends(get_vector_store),
    llm: LLM = Depends(get_llm),
) -> ReadinessResponse:
    components = {
        "database": await run_in_threadpool(_check_database, db),
        "vector_store": await run_in_threadpool(_check_vector_store, store),
        "llm": await run_in_threadpool(_check_llm, llm),
    }
    ready = all(c.ok for c in components.values())
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadinessResponse(status="ready" if ready else "degraded", components=components)
