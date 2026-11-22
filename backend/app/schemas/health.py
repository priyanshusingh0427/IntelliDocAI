from pydantic import BaseModel


class ComponentStatus(BaseModel):
    ok: bool
    detail: str


class ReadinessResponse(BaseModel):
    status: str  # "ready" or "degraded"
    components: dict[str, ComponentStatus]


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    environment: str
