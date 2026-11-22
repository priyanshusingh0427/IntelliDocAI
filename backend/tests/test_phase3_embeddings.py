"""Phase 3: chunking, embeddings, ChromaDB storage and semantic search."""
import os
from dataclasses import dataclass

import numpy as np
import pymupdf
import pytest

from app.core.config import Settings, get_settings
from app.services.chunking import chunk_pages, split_text
from app.services.embeddings import HashEmbedder, get_embedder
from main import app

URL = "/api/v1/documents"
SEARCH = "/api/v1/search"

SOLAR = (
    "Solar panels convert sunlight into electricity using photovoltaic cells. "
    "Panel efficiency depends on sunlight intensity, temperature and the angle of the panels. "
    "Most residential solar panels produce electricity for twenty five years."
)
BAKING = (
    "Sourdough bread needs flour, water, salt and an active starter. "
    "The dough rises slowly overnight and is baked in a very hot oven. "
    "A good loaf of bread has a crisp crust and an open crumb."
)


@dataclass
class P:
    page_number: int
    text: str


def build_pdf(pages: list[str]) -> bytes:
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(50, 50, 545, 790), text, fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


def upload_pdf(client, pages: list[str], name: str = "doc.pdf"):
    return client.post(URL, files={"file": (name, build_pdf(pages), "application/pdf")})


# ----------------------------------------------------------------- chunking


def test_short_page_is_one_chunk():
    chunks = chunk_pages([P(1, "Hello world.")], chunk_size=100, overlap=20)
    assert len(chunks) == 1
    assert chunks[0].text == "Hello world." and chunks[0].page_number == 1


def test_long_text_chunks_respect_size_overlap_and_offsets():
    text = " ".join(f"Sentence number {i} talks about topic {i % 7}." for i in range(120))
    spans = split_text(text, chunk_size=200, overlap=50)
    assert len(spans) > 5
    for start, end in spans:
        assert 0 < end - start <= 200
        assert not text[start].isspace() and not text[end - 1].isspace()
    for (s1, e1), (s2, _) in zip(spans, spans[1:], strict=False):
        assert s2 < e1, "consecutive chunks must overlap"
        assert s2 > s1, "chunks must move forward"
    assert spans[0][0] == 0 and spans[-1][1] == len(text)  # whole text covered


def test_chunks_prefer_sentence_boundaries():
    text = "First sentence is here. " * 30
    for start, end in split_text(text, chunk_size=120, overlap=0):
        assert text[start:end].endswith(".")


def test_chunks_never_cross_pages_and_keep_page_numbers():
    pages = [P(1, "alpha " * 100), P(2, ""), P(3, "gamma " * 100)]
    chunks = chunk_pages(pages, chunk_size=150, overlap=30)
    assert {c.page_number for c in chunks} == {1, 3}  # blank page 2 yields nothing
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    for c in chunks:
        page_text = pages[c.page_number - 1].text
        assert page_text[c.char_start : c.char_end] == c.text
        assert ("alpha" in c.text) != ("gamma" in c.text)


def test_unbroken_text_is_hard_cut():
    spans = split_text("x" * 1000, chunk_size=300, overlap=50)
    assert all(e - s <= 300 for s, e in spans)
    assert spans[-1][1] == 1000


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (100, 100), (100, -1)])
def test_invalid_chunk_parameters(size, overlap):
    with pytest.raises(ValueError):
        split_text("abc", size, overlap)


def test_settings_reject_overlap_not_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        Settings(_env_file=None, CHUNK_SIZE=100, CHUNK_OVERLAP=100)


# --------------------------------------------------------------- embeddings


def test_hash_embedder_is_normalised_and_deterministic():
    emb = HashEmbedder()
    a, b = emb.embed(["solar panel power", "solar panel power"])
    assert a.shape == (emb.dimension,)
    assert np.isclose(np.linalg.norm(a), 1.0)
    assert np.allclose(a, b)
    assert emb.embed([]).shape == (0, emb.dimension)


# ------------------------------------------------- upload -> index -> ChromaDB


def test_upload_auto_indexes_into_chromadb(doc_client, vector_store):
    body = upload_pdf(doc_client, [SOLAR, BAKING], "mixed.pdf").json()
    assert body["index_status"] == "indexed"
    assert body["chunk_count"] >= 2
    assert body["embedding_model"] == "hash-bow-256"
    assert body["indexed_at"]
    assert vector_store.count() == body["chunk_count"]


def test_chunks_endpoint_keeps_source_metadata(doc_client):
    doc_id = upload_pdf(doc_client, [SOLAR, BAKING]).json()["id"]
    chunks = doc_client.get(f"{URL}/{doc_id}/chunks").json()
    pages = {p["page_number"]: p["text"] for p in doc_client.get(f"{URL}/{doc_id}/pages").json()}
    assert [c["chunk_index"] for c in chunks] == list(range(len(chunks)))
    assert {c["page_number"] for c in chunks} == {1, 2}
    for c in chunks:
        assert c["document_id"] == doc_id
        assert pages[c["page_number"]][c["char_start"] : c["char_end"]] == c["text"]
    assert doc_client.get(f"{URL}/999/chunks").status_code == 404


def test_semantic_search_returns_relevant_page(doc_client):
    upload_pdf(doc_client, [SOLAR], "solar.pdf")
    upload_pdf(doc_client, ["Intro page", BAKING], "bread.pdf")

    res = doc_client.post(SEARCH, json={"query": "how is sourdough bread baked in the oven"})
    assert res.status_code == 200
    top = res.json()["results"][0]
    assert top["filename"] == "bread.pdf"
    assert top["page_number"] == 2
    assert "bread" in top["text"].lower()

    res = doc_client.post(SEARCH, json={"query": "solar panels electricity", "top_k": 1})
    results = res.json()["results"]
    assert len(results) == 1 and results[0]["filename"] == "solar.pdf"
    scores = [r["score"] for r in doc_client.post(SEARCH, json={"query": "solar"}).json()["results"]]
    assert scores == sorted(scores, reverse=True)


def test_search_filter_by_document(doc_client):
    solar_id = upload_pdf(doc_client, [SOLAR], "solar.pdf").json()["id"]
    upload_pdf(doc_client, [BAKING], "bread.pdf")
    res = doc_client.post(SEARCH, json={"query": "bread", "document_ids": [solar_id]}).json()
    assert res["results"] and all(r["document_id"] == solar_id for r in res["results"])


def test_search_validation_and_empty_index(doc_client):
    assert doc_client.post(SEARCH, json={"query": ""}).status_code == 422
    assert doc_client.post(SEARCH, json={"query": "x", "top_k": 0}).status_code == 422
    assert doc_client.post(SEARCH, json={"query": "anything"}).json()["results"] == []


def test_reindex_replaces_old_chunks(doc_client, vector_store, test_settings):
    body = upload_pdf(doc_client, [SOLAR * 4]).json()
    doc_id, before = body["id"], body["chunk_count"]

    bigger = test_settings.model_copy(update={"CHUNK_SIZE": 2000, "CHUNK_OVERLAP": 0})
    app.dependency_overrides[get_settings] = lambda: bigger
    res = doc_client.post(f"{URL}/{doc_id}/index")
    assert res.status_code == 200
    after = res.json()["chunk_count"]
    assert after < before
    assert vector_store.count() == after  # no leftover vectors from the old index
    assert doc_client.post(f"{URL}/999/index").status_code == 404


def test_delete_removes_vectors(doc_client, vector_store):
    keep = upload_pdf(doc_client, [BAKING], "keep.pdf").json()
    gone = upload_pdf(doc_client, [SOLAR], "gone.pdf").json()
    assert doc_client.delete(f"{URL}/{gone['id']}").status_code == 204
    assert vector_store.count() == keep["chunk_count"]
    results = doc_client.post(SEARCH, json={"query": "solar"}).json()["results"]
    assert all(r["document_id"] == keep["id"] for r in results)


class BrokenEmbedder(HashEmbedder):
    def embed(self, texts):
        raise RuntimeError("model crashed")


def test_indexing_failure_keeps_upload_and_reports_error(doc_client):
    app.dependency_overrides[get_embedder] = lambda: BrokenEmbedder()
    res = upload_pdf(doc_client, [SOLAR])
    assert res.status_code == 201  # extraction succeeded; upload is kept
    body = res.json()
    assert body["index_status"] == "failed"
    assert "model crashed" in body["index_error"]

    retry = doc_client.post(f"{URL}/{body['id']}/index")
    assert retry.status_code == 500
    assert "model crashed" in retry.json()["detail"]

    app.dependency_overrides[get_embedder] = lambda: HashEmbedder()
    fixed = doc_client.post(f"{URL}/{body['id']}/index").json()
    assert fixed["index_status"] == "indexed" and fixed["chunk_count"] >= 1
    assert doc_client.get(f"{URL}/{body['id']}").json()["index_error"] is None


def test_auto_index_can_be_disabled(doc_client, vector_store, test_settings):
    off = test_settings.model_copy(update={"AUTO_INDEX_ON_UPLOAD": False})
    app.dependency_overrides[get_settings] = lambda: off
    body = upload_pdf(doc_client, [SOLAR]).json()
    assert body["index_status"] == "not_indexed"
    assert vector_store.count() == 0


# ---------------------------------------- real Sentence Transformers model


@pytest.mark.skipif(
    os.getenv("RUN_MODEL_TESTS") != "1",
    reason="downloads/loads the real model; run with RUN_MODEL_TESTS=1",
)
def test_real_sentence_transformer_understands_meaning(doc_client, test_settings):
    from app.services.embeddings import build_embedder

    real = build_embedder(
        Settings(_env_file=None, EMBEDDING_PROVIDER="sentence-transformers")
    )
    vectors = real.embed(["The car would not start this morning.",
                          "My automobile's engine failed to turn on.",
                          "I baked a chocolate cake for the party."])
    assert vectors.shape == (3, 384)
    sim_related = float(vectors[0] @ vectors[1])
    sim_unrelated = float(vectors[0] @ vectors[2])
    assert sim_related > sim_unrelated + 0.2  # synonyms match without shared words

    app.dependency_overrides[get_embedder] = lambda: real
    upload_pdf(doc_client, [SOLAR], "solar.pdf")
    upload_pdf(doc_client, [BAKING], "bread.pdf")
    # No word overlap with the bread text: only real semantic understanding finds it.
    top = doc_client.post(SEARCH, json={"query": "a recipe for cooking food"}).json()["results"][0]
    assert top["filename"] == "bread.pdf"
