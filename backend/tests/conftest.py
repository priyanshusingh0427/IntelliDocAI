import os
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import Settings, get_settings
from app.db.session import Base, get_db, make_engine
import app.models  # noqa: F401  (register every model on Base)
from app.services.embeddings import Embedder, HashEmbedder, get_embedder
from app.services.llm import ExtractiveLLM, get_llm
from app.services.reranker import NoopReranker, get_reranker
from app.services.vector_store import VectorStore, get_vector_store
from main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        DATABASE_URL="sqlite://",
        UPLOAD_DIR=str(tmp_path / "uploads"),
        MAX_UPLOAD_SIZE_MB=1,
        EMBEDDING_PROVIDER="hash",
        CHROMA_DIR=str(tmp_path / "chroma"),
        CHUNK_SIZE=300,
        CHUNK_OVERLAP=60,
        RERANKER_PROVIDER="none",
        LLM_PROVIDER="extractive",
    )


@pytest.fixture
def embedder() -> Embedder:
    """Offline deterministic embedder (no model download) for fast tests."""
    return HashEmbedder()


@pytest.fixture
def vector_store() -> Iterator[VectorStore]:
    """In-memory ChromaDB collection, unique per test."""
    chromadb = pytest.importorskip("chromadb")
    from chromadb.config import Settings as ChromaSettings

    client = chromadb.EphemeralClient(settings=ChromaSettings(anonymized_telemetry=False))
    name = f"test-{uuid.uuid4().hex}"
    store = VectorStore(client=client, collection_name=name)
    try:
        yield store
    finally:
        # The store creates its collection lazily, so tests that never touch it have none.
        try:
            client.delete_collection(name)
        except ValueError:
            pass


# Tests use in-memory SQLite by default. To run the same suite against PostgreSQL:
#   $env:TEST_DATABASE_URL="postgresql+psycopg://intellidoc:PASSWORD@127.0.0.1:5432/intellidoc_test"
# (all tables in that database are dropped and recreated for every test).
TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL", "sqlite://")
TEST_PASSWORD = "correct-horse-battery"


@pytest.fixture
def db_engine() -> Iterator[Engine]:
    if TEST_DATABASE_URL.startswith("sqlite"):
        engine = make_engine("sqlite://", poolclass=StaticPool)  # in-memory, FKs enabled
    else:
        engine = make_engine(TEST_DATABASE_URL)
        Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    try:
        yield engine
    finally:
        if not TEST_DATABASE_URL.startswith("sqlite"):
            Base.metadata.drop_all(bind=engine)
        engine.dispose()


@pytest.fixture
def db_session_factory(db_engine: Engine) -> sessionmaker:
    return sessionmaker(bind=db_engine, autoflush=False, expire_on_commit=False)


@pytest.fixture
def api_client(
    test_settings: Settings,
    embedder: Embedder,
    vector_store: VectorStore,
    db_session_factory: sessionmaker,
) -> Iterator[TestClient]:
    """Logged-OUT client wired to the test database, a temp upload folder,
    the offline embedder and an in-memory ChromaDB collection."""

    def override_get_db() -> Iterator[Session]:
        db = db_session_factory()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_settings] = lambda: test_settings
    app.dependency_overrides[get_embedder] = lambda: embedder
    app.dependency_overrides[get_vector_store] = lambda: vector_store
    app.dependency_overrides[get_reranker] = lambda: NoopReranker()
    app.dependency_overrides[get_llm] = lambda: ExtractiveLLM()
    try:
        # No `with` block: skip the lifespan so the real database is never touched.
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


def register(client: TestClient, email: str, password: str = TEST_PASSWORD, **extra) -> dict:
    """Create an account; returns the JSON (token, user, adopted_documents)."""
    response = client.post(
        "/api/v1/auth/register", json={"email": email, "password": password, **extra}
    )
    assert response.status_code == 201, response.text
    return response.json()


def auth_headers(token_json: dict) -> dict:
    return {"Authorization": f"Bearer {token_json['access_token']}"}


@pytest.fixture
def doc_client(api_client: TestClient) -> TestClient:
    """Client logged in as a test user (every Phase 2-5 test runs through this)."""
    api_client.headers.update(auth_headers(register(api_client, "tester@example.com")))
    return api_client
