"""Phase 7: readiness checks and cross-phase integration (upload -> index -> ask -> history)."""
import httpx
import pymupdf
import pytest
from sqlalchemy.orm import sessionmaker

from app.db.session import get_db, make_engine
from app.services.llm import ExtractiveLLM, OllamaLLM, TransformersLLM, get_llm
from main import app
from tests.conftest import auth_headers, register

API = "/api/v1"
READY = f"{API}/health/ready"


def build_pdf(pages: list[str]) -> bytes:
    doc = pymupdf.open()
    for text in pages:
        doc.new_page().insert_textbox(pymupdf.Rect(50, 50, 550, 800), text, fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


# ------------------------------------------------------------------ readiness


def test_ready_when_everything_works(api_client):
    response = api_client.get(READY)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert set(body["components"]) == {"database", "vector_store", "llm"}
    assert all(c["ok"] for c in body["components"].values())
    assert "chunks indexed" in body["components"]["vector_store"]["detail"]


def test_ready_needs_no_login(api_client):
    assert "authorization" not in {k.lower() for k in api_client.headers}
    assert api_client.get(READY).status_code == 200


def test_degraded_when_llm_is_down(api_client):
    class DownLLM(ExtractiveLLM):
        def check(self):
            return False, "Ollama is not reachable at http://127.0.0.1:11434"

    app.dependency_overrides[get_llm] = lambda: DownLLM()
    response = api_client.get(READY)
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["components"]["llm"] == {"ok": False, "detail": "Ollama is not reachable at http://127.0.0.1:11434"}
    assert body["components"]["database"]["ok"] is True


def test_degraded_when_database_is_down(api_client, tmp_path):
    # A SQLite path inside a missing folder cannot be opened -> connection error.
    broken = make_engine(f"sqlite:///{(tmp_path / 'missing' / 'db.sqlite').as_posix()}")
    Broken = sessionmaker(bind=broken)

    def broken_db():
        db = Broken()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = broken_db
    response = api_client.get(READY)
    assert response.status_code == 503
    db_status = response.json()["components"]["database"]
    assert db_status["ok"] is False and "OperationalError" in db_status["detail"]


def test_degraded_when_vector_store_is_down(api_client):
    from app.services.vector_store import VectorStoreUnavailable, get_vector_store

    class DownStore:
        def count(self):
            raise VectorStoreUnavailable("chromadb is not installed")

    app.dependency_overrides[get_vector_store] = lambda: DownStore()
    response = api_client.get(READY)
    assert response.status_code == 503
    assert "chromadb is not installed" in response.json()["components"]["vector_store"]["detail"]


class _FakeResponse:
    def __init__(self, payload, status=200):
        self._payload, self.status_code = payload, status

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("bad", request=None, response=None)


@pytest.mark.parametrize(
    "fake,ok,needle",
    [
        (httpx.ConnectError("refused"), False, "not reachable"),
        (_FakeResponse({"models": [{"name": "qwen2:1.5b"}]}), False, "ollama pull llama3.2:3b"),
        (_FakeResponse({"models": [{"name": "llama3.2:3b"}]}), True, "ready with llama3.2:3b"),
    ],
)
def test_ollama_check(monkeypatch, fake, ok, needle):
    def fake_get(url, timeout):
        assert url == "http://127.0.0.1:11434/api/tags" and timeout <= 5
        if isinstance(fake, Exception):
            raise fake
        return fake

    monkeypatch.setattr(httpx, "get", fake_get)
    llm = OllamaLLM("http://127.0.0.1:11434/", "llama3.2:3b", max_new_tokens=10, temperature=0, timeout=5)
    result_ok, detail = llm.check()
    assert result_ok is ok and needle in detail


def test_ollama_check_accepts_implicit_latest_tag(monkeypatch):
    monkeypatch.setattr(httpx, "get", lambda url, timeout: _FakeResponse({"models": [{"name": "mistral:latest"}]}))
    assert OllamaLLM("http://x", "mistral", 10, 0, 5).check()[0] is True


def test_transformers_check_does_not_load_the_model():
    llm = TransformersLLM("Qwen/Qwen2.5-0.5B-Instruct", cache_dir="unused", max_new_tokens=10, temperature=0)
    ok, detail = llm.check()
    assert ok is True and "not loaded yet" in detail
    assert llm._model is None  # the probe must stay cheap


# ------------------------------------------------------------------ full journey


def test_full_user_journey(api_client):
    """Everything a user does, in order, through the public API only."""
    headers = auth_headers(register(api_client, "journey@example.com", full_name="Journey"))
    pdf = build_pdf([
        "Garden Pump GP-9 manual. Introduction to the pump.",
        "Cleaning. Rinse the intake sieve every month. Spare sieve part number SV-118.",
        "Safety. Never run the pump dry for more than one minute.",
    ])

    # upload -> extracted page by page -> indexed
    doc = api_client.post(
        f"{API}/documents", files={"file": ("GP9.pdf", pdf, "application/pdf")}, headers=headers
    ).json()
    assert doc["page_count"] == 3 and doc["index_status"] == "indexed" and doc["chunk_count"] >= 3
    page2 = api_client.get(f"{API}/documents/{doc['id']}/pages/2", headers=headers).json()
    assert "SV-118" in page2["text"]

    # search finds the right page with both retrievers
    hybrid = api_client.post(f"{API}/search/hybrid", json={"query": "spare sieve part number"}, headers=headers).json()
    assert hybrid["results"][0]["page_number"] == 2
    assert set(hybrid["results"][0]["retrieved_by"]) == {"semantic", "bm25"}

    # ask -> grounded answer with a page citation, saved to a new chat
    first = api_client.post(f"{API}/ask", json={"question": "What is the spare sieve part number?"}, headers=headers).json()
    assert first["found"] and first["grounded"]
    assert first["citations"][0]["page_number"] == 2 and "SV-118" in first["citations"][0]["text"]

    # follow-up in the same chat, limited to this document
    second = api_client.post(
        f"{API}/ask",
        json={"question": "How long may the pump run dry?", "chat_id": first["chat_id"], "document_ids": [doc["id"]]},
        headers=headers,
    ).json()
    assert second["chat_id"] == first["chat_id"]
    assert second["citations"][0]["page_number"] == 3

    # history shows both exchanges with their citations
    chat = api_client.get(f"{API}/chats/{first['chat_id']}", headers=headers).json()
    assert [m["role"] for m in chat["messages"]] == ["user", "assistant"] * 2
    assert chat["messages"][3]["citations"][0]["page_number"] == 3
    assert chat["messages"][2]["document_ids"] == [doc["id"]]

    # deleting the document removes it from search; the chat history stays
    assert api_client.delete(f"{API}/documents/{doc['id']}", headers=headers).status_code == 204
    gone = api_client.post(f"{API}/search/hybrid", json={"query": "spare sieve part number"}, headers=headers).json()
    assert gone["results"] == []
    assert api_client.get(f"{API}/chats/{first['chat_id']}", headers=headers).json()["message_count"] == 4

    # log in again later: same data, same history
    login = api_client.post(f"{API}/auth/login", data={"username": "journey@example.com", "password": "correct-horse-battery"})
    chats = api_client.get(f"{API}/chats", headers=auth_headers(login.json())).json()
    assert [c["id"] for c in chats] == [first["chat_id"]]


def test_reupload_after_delete_reindexes_cleanly(api_client, vector_store):
    headers = auth_headers(register(api_client, "again@example.com"))
    pdf = build_pdf(["Alpha page about meadows.", "Beta page about rivers."])
    first = api_client.post(f"{API}/documents", files={"file": ("a.pdf", pdf, "application/pdf")}, headers=headers).json()
    api_client.delete(f"{API}/documents/{first['id']}", headers=headers)
    second = api_client.post(f"{API}/documents", files={"file": ("a.pdf", pdf, "application/pdf")}, headers=headers).json()
    chunks = vector_store.get_all_chunks()
    assert {c.document_id for c in chunks} == {second["id"]}  # no leftovers from the deleted copy
    assert len(chunks) == second["chunk_count"]


# ------------------------------------------------------------------ error paths across layers


def test_unexpected_extraction_crash_returns_clean_500_and_leaves_nothing(api_client, test_settings, monkeypatch):
    from fastapi.testclient import TestClient

    import app.api.v1.endpoints.documents as documents_endpoint

    def explode(*args, **kwargs):
        raise RuntimeError("secret internal path C:/x/y")

    monkeypatch.setattr(documents_endpoint, "extract", explode)
    headers = auth_headers(register(api_client, "crash@example.com"))
    client = TestClient(app, raise_server_exceptions=False)  # see the response the browser gets
    pdf = build_pdf(["Hello"])
    response = client.post(f"{API}/documents", files={"file": ("a.pdf", pdf, "application/pdf")}, headers=headers)
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}  # no internals leaked
    assert list(test_settings.upload_path.glob("*")) == []  # stored file was cleaned up
    assert client.get(f"{API}/documents", headers=headers).json() == []


def test_search_index_unavailable_gives_503_with_reason(api_client):
    from app.services.vector_store import VectorStoreUnavailable, get_vector_store

    class DownStore:
        def __getattr__(self, name):
            def fail(*args, **kwargs):
                raise VectorStoreUnavailable("chromadb is not installed")

            return fail

    headers = auth_headers(register(api_client, "nostore@example.com"))
    pdf = build_pdf(["Rivers and meadows."])
    doc = api_client.post(f"{API}/documents", files={"file": ("r.pdf", pdf, "application/pdf")}, headers=headers).json()
    app.dependency_overrides[get_vector_store] = lambda: DownStore()
    response = api_client.post(f"{API}/search", json={"query": "rivers"}, headers=headers)
    assert response.status_code == 503
    assert response.json()["detail"] == "Search index unavailable: chromadb is not installed"
    # deleting still works without the index (the database row and file are removed)
    assert api_client.delete(f"{API}/documents/{doc['id']}", headers=headers).status_code == 204


def test_non_english_filename_is_kept_readable(api_client):
    headers = auth_headers(register(api_client, "unicode@example.com"))
    pdf = build_pdf(["Résumé page with café notes."])
    doc = api_client.post(
        f"{API}/documents", files={"file": ("Résumé café 2026.pdf", pdf, "application/pdf")}, headers=headers
    ).json()
    assert doc["original_filename"].endswith(".pdf")
    assert "R" in doc["original_filename"] and "/" not in doc["original_filename"]
    page = api_client.get(f"{API}/documents/{doc['id']}/pages/1", headers=headers).json()
    assert "café" in page["text"]


# ------------------------------------------------------------------ concurrency (found by the screenshot run)


def _run_together(n: int, fn) -> list:
    """Start n threads that call fn at the same instant; return their exceptions."""
    import threading

    barrier = threading.Barrier(n)
    errors: list = []

    def worker():
        barrier.wait()
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    return errors


def test_vector_store_is_created_once_under_concurrent_first_use():
    import time

    from app.services.vector_store import VectorStore

    calls = []

    class SlowClient:
        def get_or_create_collection(self, **kwargs):
            time.sleep(0.05)  # widen the race window
            return type("Collection", (), {"count": lambda self: 0})()

    def factory():
        calls.append(1)
        time.sleep(0.05)
        return SlowClient()

    store = VectorStore(client_factory=factory, collection_name="race")
    assert _run_together(10, store.count) == []
    assert len(calls) == 1


def test_real_chromadb_fresh_folder_survives_concurrent_first_use(tmp_path):
    pytest.importorskip("chromadb")
    from app.services.vector_store import make_persistent_store

    store = make_persistent_store(str(tmp_path / "chroma"), "race")
    assert _run_together(8, store.count) == []
    assert store.count() == 0
