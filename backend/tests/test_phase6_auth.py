"""Phase 6: accounts, login tokens, per-user data isolation and chat history."""
import pymupdf
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.core.config import DEV_SECRET_KEY, Settings, get_settings
from app.core.security import create_access_token
from app.models.document import Document
from app.models.user import User
from app.services.rag import NOT_FOUND_ANSWER
from main import app
from tests.conftest import TEST_PASSWORD, auth_headers, register

API = "/api/v1"


def build_pdf(pages: list[str]) -> bytes:
    doc = pymupdf.open()
    for text in pages:
        doc.new_page().insert_textbox(pymupdf.Rect(50, 50, 550, 800), text, fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


MANUAL = [
    "Installation. Mount the X200 water purifier under the sink.",
    "Reset. To reset the X200, hold the red RESET button for ten seconds. Replace the filter "
    "cartridge ZX-4471 every six months.",
]


def upload(client: TestClient, headers: dict, name: str = "X200_Manual.pdf", pages=MANUAL) -> dict:
    response = client.post(
        f"{API}/documents", files={"file": (name, build_pdf(pages), "application/pdf")}, headers=headers
    )
    assert response.status_code == 201, response.text
    return response.json()


def login(client: TestClient, email: str, password: str = TEST_PASSWORD):
    return client.post(f"{API}/auth/login", data={"username": email, "password": password})


# ------------------------------------------------------------------ accounts and login


def test_register_returns_token_and_user_without_password(api_client, db_session_factory):
    body = register(api_client, "Alice@Example.com", full_name="Alice")
    assert body["token_type"] == "bearer"
    assert body["expires_in"] == 24 * 60 * 60
    assert body["user"]["email"] == "alice@example.com"  # stored lower-case
    assert body["user"]["full_name"] == "Alice"
    assert "hashed_password" not in body["user"]
    assert TEST_PASSWORD not in str(body)
    with db_session_factory() as db:
        user = db.query(User).one()
        assert user.hashed_password.startswith("$argon2id$")
        assert TEST_PASSWORD not in user.hashed_password


def test_duplicate_email_is_rejected_case_insensitively(api_client):
    register(api_client, "bob@example.com")
    response = api_client.post(f"{API}/auth/register", json={"email": "BOB@example.com", "password": TEST_PASSWORD})
    assert response.status_code == 409
    assert "already exists" in response.json()["detail"]


@pytest.mark.parametrize(
    "payload",
    [{"email": "not-an-email", "password": TEST_PASSWORD}, {"email": "c@example.com", "password": "short"}],
)
def test_register_validation(api_client, payload):
    assert api_client.post(f"{API}/auth/register", json=payload).status_code == 422


def test_registration_can_be_disabled(api_client, test_settings):
    app.dependency_overrides[get_settings] = lambda: test_settings.model_copy(update={"ALLOW_REGISTRATION": False})
    response = api_client.post(f"{API}/auth/register", json={"email": "d@example.com", "password": TEST_PASSWORD})
    assert response.status_code == 403


def test_login_and_me(api_client):
    register(api_client, "erin@example.com", full_name="Erin")
    response = login(api_client, " ERIN@example.com ")
    assert response.status_code == 200
    me = api_client.get(f"{API}/auth/me", headers=auth_headers(response.json()))
    assert me.status_code == 200
    assert me.json()["email"] == "erin@example.com"


@pytest.mark.parametrize("email,password", [("frank@example.com", "wrong-password"), ("nobody@example.com", TEST_PASSWORD)])
def test_login_failures_give_the_same_401(api_client, email, password):
    register(api_client, "frank@example.com")
    response = login(api_client, email, password)
    assert response.status_code == 401
    assert response.json()["detail"] == "Incorrect email or password"


def test_bad_and_expired_tokens_are_rejected(api_client, test_settings, db_session_factory):
    register(api_client, "gina@example.com")
    with db_session_factory() as db:
        user_id = db.query(User).one().id
    expired, _ = create_access_token(user_id, test_settings.model_copy(update={"ACCESS_TOKEN_EXPIRE_MINUTES": -1}))
    other_key, _ = create_access_token(user_id, test_settings.model_copy(update={"SECRET_KEY": "another-secret-key-x" * 2}))
    for token in ["garbage", expired, other_key]:
        response = api_client.get(f"{API}/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 401, token
        assert response.headers["www-authenticate"] == "Bearer"


@pytest.mark.parametrize(
    "method,path,body",
    [
        ("get", "/documents", None),
        ("get", "/documents/1", None),
        ("delete", "/documents/1", None),
        ("post", "/search", {"query": "x"}),
        ("post", "/search/hybrid", {"query": "x"}),
        ("post", "/ask", {"question": "x"}),
        ("get", "/chats", None),
        ("get", "/auth/me", None),
    ],
)
def test_endpoints_require_login(api_client, method, path, body):
    kwargs = {"json": body} if body else {}
    assert getattr(api_client, method)(f"{API}{path}", **kwargs).status_code == 401


def test_health_stays_public(api_client):
    assert api_client.get(f"{API}/health").status_code == 200


def test_production_requires_a_real_secret_key():
    with pytest.raises(ValidationError, match="SECRET_KEY"):
        Settings(_env_file=None, ENVIRONMENT="production", SECRET_KEY=DEV_SECRET_KEY)
    Settings(_env_file=None, ENVIRONMENT="production", SECRET_KEY="a-long-random-production-secret")


# ------------------------------------------------------------------ per-user isolation


@pytest.fixture
def two_users(api_client):
    alice = auth_headers(register(api_client, "alice@example.com"))
    bob = auth_headers(register(api_client, "bob@example.com"))
    return alice, bob


def test_users_only_see_their_own_documents(api_client, two_users):
    alice, bob = two_users
    doc = upload(api_client, alice)
    doc_id = doc["id"]
    assert [d["id"] for d in api_client.get(f"{API}/documents", headers=alice).json()] == [doc_id]
    assert api_client.get(f"{API}/documents", headers=bob).json() == []
    for path in [f"/documents/{doc_id}", f"/documents/{doc_id}/pages", f"/documents/{doc_id}/pages/1", f"/documents/{doc_id}/chunks"]:
        assert api_client.get(f"{API}{path}", headers=bob).status_code == 404, path
    assert api_client.post(f"{API}/documents/{doc_id}/index", headers=bob).status_code == 404
    assert api_client.delete(f"{API}/documents/{doc_id}", headers=bob).status_code == 404
    # still there for Alice
    assert api_client.get(f"{API}/documents/{doc_id}", headers=alice).status_code == 200


def test_search_and_ask_never_use_other_users_documents(api_client, two_users):
    alice, bob = two_users
    doc_id = upload(api_client, alice)["id"]
    q = "How do I reset the X200?"

    assert api_client.post(f"{API}/search", json={"query": q}, headers=alice).json()["results"]
    assert api_client.post(f"{API}/search", json={"query": q}, headers=bob).json()["results"] == []
    assert api_client.post(f"{API}/search/hybrid", json={"query": q}, headers=bob).json()["results"] == []
    # asking for Alice's document id explicitly does not help Bob
    hybrid = api_client.post(f"{API}/search/hybrid", json={"query": q, "document_ids": [doc_id]}, headers=bob)
    assert hybrid.json()["results"] == []

    alice_answer = api_client.post(f"{API}/ask", json={"question": q}, headers=alice).json()
    assert alice_answer["found"] is True
    assert alice_answer["citations"][0]["document_id"] == doc_id
    for body in [{"question": q}, {"question": q, "document_ids": [doc_id]}]:
        bob_answer = api_client.post(f"{API}/ask", json=body, headers=bob).json()
        assert bob_answer["found"] is False
        assert bob_answer["answer"] == NOT_FOUND_ANSWER
        assert bob_answer["citations"] == []


def test_two_users_searching_with_both_documents_indexed(api_client, two_users):
    alice, bob = two_users
    upload(api_client, alice, "alice.pdf")
    bob_doc = upload(api_client, bob, "bob.pdf", ["Bob's notes. The red RESET button resets the Y300 router."])
    results = api_client.post(f"{API}/search/hybrid", json={"query": "red RESET button"}, headers=bob).json()["results"]
    assert results and {r["document_id"] for r in results} == {bob_doc["id"]}


def test_first_account_adopts_documents_uploaded_before_accounts(api_client, db_session_factory):
    with db_session_factory() as db:
        db.add(Document(original_filename="old.pdf", stored_filename="old-uuid.pdf", extension=".pdf", size_bytes=10))
        db.commit()
    first = register(api_client, "owner@example.com")
    assert first["adopted_documents"] == 1
    second = register(api_client, "later@example.com")
    assert second["adopted_documents"] == 0
    assert [d["original_filename"] for d in api_client.get(f"{API}/documents", headers=auth_headers(first)).json()] == ["old.pdf"]
    assert api_client.get(f"{API}/documents", headers=auth_headers(second)).json() == []


# ------------------------------------------------------------------ chat history


def test_ask_saves_history_and_continues_a_chat(doc_client):
    upload(doc_client, {})
    first = doc_client.post(f"{API}/ask", json={"question": "  How do I reset the X200?  "}).json()
    chat_id = first["chat_id"]
    assert chat_id and first["message_id"]
    second = doc_client.post(f"{API}/ask", json={"question": "Which filter cartridge?", "chat_id": chat_id}).json()
    assert second["chat_id"] == chat_id

    chats = doc_client.get(f"{API}/chats").json()
    assert len(chats) == 1
    assert chats[0]["title"] == "How do I reset the X200?"
    assert chats[0]["message_count"] == 4

    detail = doc_client.get(f"{API}/chats/{chat_id}").json()
    roles = [m["role"] for m in detail["messages"]]
    assert roles == ["user", "assistant", "user", "assistant"]
    assert detail["messages"][0]["content"] == "How do I reset the X200?"
    answer = detail["messages"][1]
    assert answer["content"] == first["answer"]
    assert answer["found"] is True
    assert answer["citations"][0]["page_number"] == first["citations"][0]["page_number"]
    assert answer["llm_model"] == "extractive"


def test_new_question_without_chat_id_starts_a_new_chat_and_long_titles_are_cut(doc_client):
    long_q = "Please explain " + "very " * 30 + "carefully"
    doc_client.post(f"{API}/ask", json={"question": "first"})
    doc_client.post(f"{API}/ask", json={"question": long_q})
    chats = doc_client.get(f"{API}/chats").json()
    assert len(chats) == 2
    assert chats[0]["title"].endswith("…") and len(chats[0]["title"]) <= 60  # newest first


def test_create_rename_delete_chat(doc_client):
    chat = doc_client.post(f"{API}/chats", json={}).json()
    assert chat["title"] == "New chat" and chat["message_count"] == 0
    renamed = doc_client.patch(f"{API}/chats/{chat['id']}", json={"title": "  Warranty questions "}).json()
    assert renamed["title"] == "Warranty questions"
    assert doc_client.patch(f"{API}/chats/{chat['id']}", json={"title": ""}).status_code == 422
    doc_client.post(f"{API}/ask", json={"question": "anything", "chat_id": chat["id"]})
    assert doc_client.delete(f"{API}/chats/{chat['id']}").status_code == 204
    assert doc_client.get(f"{API}/chats/{chat['id']}").status_code == 404
    assert doc_client.get(f"{API}/chats").json() == []


def test_chats_are_private(api_client, two_users):
    alice, bob = two_users
    chat_id = api_client.post(f"{API}/ask", json={"question": "hello"}, headers=alice).json()["chat_id"]
    assert api_client.get(f"{API}/chats", headers=bob).json() == []
    assert api_client.get(f"{API}/chats/{chat_id}", headers=bob).status_code == 404
    assert api_client.patch(f"{API}/chats/{chat_id}", json={"title": "x"}, headers=bob).status_code == 404
    assert api_client.delete(f"{API}/chats/{chat_id}", headers=bob).status_code == 404
    # Bob cannot add messages to Alice's chat either, and nothing is saved
    assert api_client.post(f"{API}/ask", json={"question": "sneaky", "chat_id": chat_id}, headers=bob).status_code == 404
    assert api_client.get(f"{API}/chats/{chat_id}", headers=alice).json()["message_count"] == 2


def test_failed_answer_is_not_saved(doc_client):
    from app.services.llm import LLMUnavailable, get_llm

    class BrokenLLM:
        model_name = "broken"
        is_generative = True

        def generate(self, system, user):
            raise LLMUnavailable("Ollama is not reachable")

    upload(doc_client, {})
    app.dependency_overrides[get_llm] = lambda: BrokenLLM()
    response = doc_client.post(f"{API}/ask", json={"question": "How do I reset the X200?"})
    assert response.status_code == 503
    assert doc_client.get(f"{API}/chats").json() == []
