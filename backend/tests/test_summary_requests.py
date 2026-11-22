"""Whole-document requests ("Summarize this document") used to answer "not found": similarity
search finds nothing for a question without topic words. They now get evenly spread excerpts."""
import pymupdf
import pytest

from app.services.llm import get_llm
from app.services.rag import _spread, build_summary_prompt, is_summary_request, summary_sources
from main import app

URL = "/api/v1/documents"
ASK = "/api/v1/ask"


def build_pdf(pages: list[str]) -> bytes:
    doc = pymupdf.open()
    for text in pages:
        doc.new_page().insert_textbox(pymupdf.Rect(50, 50, 550, 800), text, fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


NOTES = [
    f"Chapter {i}. {topic} is explained here with definitions, properties and an example." * 3
    for i, topic in enumerate(
        ["AES encryption", "Diffie-Hellman key exchange", "Hash functions", "Digital signatures",
         "Kerberos authentication", "Firewalls", "K-anonymity", "L-diversity", "Differential privacy",
         "Active attacks", "Passive attacks", "Stream ciphers"],
        start=1,
    )
]


def upload(client, pages, name):
    res = client.post(URL, files={"file": (name, build_pdf(pages), "application/pdf")})
    assert res.status_code == 201, res.text
    return res.json()


class RecordingLLM:
    is_generative = True
    model_name = "recording-llm"

    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def generate(self, system, user):
        self.calls.append((system, user))
        return self.reply

    def check(self):
        return True, "ok"


@pytest.mark.parametrize(
    "question",
    ["Summarize this document", "Summarize", "summarise the PDF briefly", "What is this document about?",
     "What's this about?", "Give me the main topics covered", "Key points of the notes", "tl;dr",
     "Give an overview of the whole file"],
)
def test_detects_whole_document_requests(question):
    assert is_summary_request(question)


@pytest.mark.parametrize(
    "question",
    ["Summarize the AES section", "Give me the key points about Diffie-Hellman", "Overview of k-anonymity",
     "What is AES?", "What is the warranty about?", "What is the capital of France?", "How do I reset the X200?"],
)
def test_topic_questions_use_normal_retrieval(question):
    assert not is_summary_request(question)


def test_spread_takes_start_middle_and_end():
    assert _spread(list(range(65)), 8) == [0, 9, 18, 27, 37, 46, 55, 64]
    assert _spread([1, 2], 8) == [1, 2]
    assert _spread([1, 2, 3], 1) == [1]


def test_summary_of_a_long_document_uses_pages_from_start_to_end(doc_client, vector_store, test_settings):
    doc = upload(doc_client, NOTES, "CNS.pdf")
    llm = RecordingLLM("The notes cover AES [1], hashing [3] and differential privacy [7].")
    app.dependency_overrides[get_llm] = lambda: llm

    body = doc_client.post(ASK, json={"question": "Summarize this document", "document_ids": [doc["id"]]}).json()
    assert body["found"] is True and body["grounded"] is True
    assert body["answer"].startswith("The notes cover AES [1]")
    pages = [s["page_number"] for s in body["sources"]]
    assert pages[0] == 1 and pages[-1] == 12  # first and last page are included
    assert pages == sorted(pages) and len(pages) == 8
    assert {c["source"] for c in body["citations"]} == {1, 3, 7}
    assert "summary_sampling" in body["timings_ms"]
    _, prompt = llm.calls[0]
    assert "Excerpts taken evenly from the start, middle and end of CNS.pdf" in prompt
    assert "Request: Summarize this document" in prompt
    assert len(prompt) < test_settings.RAG_SUMMARY_MAX_CONTEXT_CHARS + 2000  # fits the context window


def test_summary_with_several_documents_covers_each_and_says_so(doc_client):
    upload(doc_client, NOTES[:3], "notes.pdf")
    upload(doc_client, ["Controller X200 manual. Reset with the red button. Filter ZX-4471."], "X200.pdf")
    llm = RecordingLLM("notes.pdf is about cryptography [1]. X200.pdf is a controller manual [8].")
    app.dependency_overrides[get_llm] = lambda: llm

    body = doc_client.post(ASK, json={"question": "Summarize"}).json()
    assert body["found"] is True
    assert {s["filename"] for s in body["sources"]} == {"notes.pdf", "X200.pdf"}
    assert any("covers 2 documents" in w and "Tick one document" in w for w in body["warnings"])
    assert "Summarize each file separately" in llm.calls[0][1]


def test_summary_without_documents_is_not_found(doc_client):
    body = doc_client.post(ASK, json={"question": "Summarize this document"}).json()
    assert body["found"] is False and body["sources"] == []


def test_summary_with_extractive_provider_returns_spread_passages(doc_client):
    upload(doc_client, NOTES, "CNS.pdf")
    body = doc_client.post(ASK, json={"question": "Give me the main topics covered"}).json()
    assert body["found"] is True
    assert body["llm_model"] == "extractive"
    assert len(body["citations"]) == 8


def test_summary_is_saved_to_chat_history(doc_client):
    upload(doc_client, NOTES, "CNS.pdf")
    app.dependency_overrides[get_llm] = lambda: RecordingLLM("It covers AES [1].")
    first = doc_client.post(ASK, json={"question": "Summarize this document"}).json()
    chat = doc_client.get(f"/api/v1/chats/{first['chat_id']}").json()
    assert chat["messages"][1]["found"] is True and chat["messages"][1]["citations"][0]["page_number"] == 1


def test_summary_sources_respect_the_character_budget(vector_store, test_settings, embedder):
    from app.services.chunking import Chunk

    chunks = [Chunk(i, i + 1, 0, 2000, "x" * 2000) for i in range(20)]
    vector_store.add_chunks(5, "big.pdf", chunks, embedder.embed([c.text for c in chunks]), embedder.model_name)
    sources = summary_sources(vector_store, test_settings, [5])
    assert len(sources) == test_settings.RAG_SUMMARY_CHUNKS
    assert sum(len(s.text) for s in sources) <= test_settings.RAG_SUMMARY_MAX_CONTEXT_CHARS
    assert "[1] (file: big.pdf, page 1)" in build_summary_prompt("Summarize", sources)


def test_citations_after_the_full_stop_are_moved_before_it():
    from app.services.rag import place_citations_before_punctuation as fix

    assert fix("Used in VPNs. [1][2]\n2. **SHA-512**: hashes. [2]") == "Used in VPNs [1][2].\n2. **SHA-512**: hashes [2]."
    assert fix("It lasts three years [1].") == "It lasts three years [1]."  # already correct
    assert fix("Two sources. [1, 3] Next sentence.") == "Two sources [1, 3]. Next sentence."
    assert fix("No citations here.") == "No citations here."


def test_no_duplicate_citations_when_model_cites_after_the_period(doc_client):
    upload(doc_client, NOTES, "CNS.pdf")
    app.dependency_overrides[get_llm] = lambda: RecordingLLM(
        "The main topics are:\n\n1. **AES** is explained here with definitions. [1]\n2. **Hash functions** are explained. [3]"
    )
    body = doc_client.post(ASK, json={"question": "Summarize this document"}).json()
    assert "[1]. [1]" not in body["answer"] and "] [" not in body["answer"]
    assert "definitions [1]." in body["answer"]
    assert not any("added automatically" in w for w in body["warnings"])
