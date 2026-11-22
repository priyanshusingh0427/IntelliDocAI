"""Phase 4: BM25, hybrid fusion, reranking, RAG answers with citations."""
import os

import httpx
import pymupdf
import pytest

from app.core.config import Settings, get_settings
from app.services.hybrid_search import reciprocal_rank_fusion
from app.services.keyword_search import KeywordSearcher, tokenize
from app.services.llm import LLMUnavailable, OllamaLLM, get_llm
from app.services.rag import NOT_FOUND_ANSWER, parse_citations, strip_invalid_citations
from app.services.reranker import get_reranker
from app.services.vector_store import SearchHit
from main import app

URL = "/api/v1/documents"
HYBRID = "/api/v1/search/hybrid"
ASK = "/api/v1/ask"

SOLAR = (
    "Solar panels convert sunlight into electricity using photovoltaic cells. "
    "Most residential solar panels keep producing electricity for twenty five years."
)
BAKING = (
    "Sourdough bread needs flour, water, salt and an active starter. "
    "The dough rises overnight and is baked in a very hot oven."
)
MANUAL = (
    "Maintenance manual. To reset the controller, hold the red button for ten seconds. "
    "Replacement part number ZX-4471 fits all models made after 2019."
)


def build_pdf(pages: list[str]) -> bytes:
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(50, 50, 545, 790), text, fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


def upload(client, pages: list[str], name: str) -> dict:
    res = client.post(URL, files={"file": (name, build_pdf(pages), "application/pdf")})
    assert res.status_code == 201, res.text
    return res.json()


def hit(cid: str, doc: int = 1, page: int = 1, text: str = "t") -> SearchHit:
    return SearchHit(cid, doc, "f.pdf", page, 0, 0, len(text), text, 0.0)


# --------------------------------------------------------------- test doubles


class OverlapReranker:
    """Deterministic stand-in for the cross-encoder: score = shared word count."""

    model_name = "overlap-test"
    produces_relevance_scores = True

    def score(self, query, passages):
        q = set(tokenize(query))
        return [float(len(q & set(tokenize(p)))) for p in passages]


class FakeLLM:
    is_generative = True
    model_name = "fake-llm"

    def __init__(self, reply: str):
        self.reply = reply
        self.calls: list[tuple[str, str]] = []

    def generate(self, system, user):
        self.calls.append((system, user))
        return self.reply


class BrokenLLM(FakeLLM):
    def generate(self, system, user):
        raise LLMUnavailable("Ollama is not reachable")


def use(dependency, value):
    app.dependency_overrides[dependency] = lambda: value


# ------------------------------------------------------------------- BM25


def test_tokenize_keeps_codes_and_drops_stopwords():
    assert tokenize("What is part ZX-4471 for the v2.1 model?") == ["part", "zx-4471", "v2.1", "model"]


def test_bm25_finds_exact_code(doc_client, vector_store):
    upload(doc_client, [SOLAR], "solar.pdf")
    manual = upload(doc_client, [BAKING, MANUAL], "manual.pdf")
    results = KeywordSearcher(vector_store).search("ZX-4471", top_k=5)
    assert results[0].document_id == manual["id"] and results[0].page_number == 2
    assert all("zx-4471" in r.text.lower() for r in results)  # only real matches returned


def test_bm25_filter_and_empty_query(doc_client, vector_store):
    solar = upload(doc_client, [SOLAR], "solar.pdf")
    upload(doc_client, [MANUAL], "manual.pdf")
    searcher = KeywordSearcher(vector_store)
    assert searcher.search("ZX-4471", 5, document_ids=[solar["id"]]) == []
    assert searcher.search("the and of", 5) == []  # only stopwords


def test_bm25_index_rebuilds_after_upload_and_delete(doc_client, vector_store):
    searcher = KeywordSearcher(vector_store)
    assert searcher.search("sourdough", 5) == []
    bread = upload(doc_client, [BAKING], "bread.pdf")
    assert searcher.search("sourdough", 5)[0].document_id == bread["id"]
    doc_client.delete(f"{URL}/{bread['id']}")
    assert searcher.search("sourdough", 5) == []


# -------------------------------------------------------------------- RRF


def test_rrf_rewards_chunks_found_by_both_retrievers():
    semantic = [hit("a"), hit("b"), hit("c")]
    keyword = [hit("c"), hit("d")]
    fused = reciprocal_rank_fusion(semantic, keyword, k=60)
    assert fused[0].hit.chunk_id == "c"  # rank 3 + rank 1 beats rank 1 alone
    assert fused[0].retrieved_by == ["semantic", "bm25"]
    assert fused[0].rrf_score == pytest.approx(1 / 63 + 1 / 61, abs=1e-6)
    assert {h.hit.chunk_id for h in fused} == {"a", "b", "c", "d"}


# ---------------------------------------------------------- hybrid endpoint


def test_hybrid_search_endpoint(doc_client):
    upload(doc_client, [SOLAR], "solar.pdf")
    manual = upload(doc_client, [BAKING, MANUAL], "manual.pdf")
    use(get_reranker, OverlapReranker())

    res = doc_client.post(HYBRID, json={"query": "replacement part ZX-4471", "top_k": 3})
    assert res.status_code == 200
    body = res.json()
    assert body["reranked"] is True and body["reranker_model"] == "overlap-test"
    top = body["results"][0]
    assert top["document_id"] == manual["id"] and top["page_number"] == 2
    assert "bm25" in top["retrieved_by"]
    assert top["rerank_score"] is not None
    assert {"semantic", "bm25", "rerank"} <= set(body["timings_ms"])
    assert len(body["results"]) <= 3


def test_hybrid_search_without_rerank(doc_client):
    upload(doc_client, [SOLAR], "solar.pdf")
    body = doc_client.post(HYBRID, json={"query": "solar panels", "rerank": False}).json()
    assert body["reranked"] is False
    assert all(r["rerank_score"] is None for r in body["results"])
    assert doc_client.post(HYBRID, json={"query": ""}).status_code == 422


# ------------------------------------------------------------- RAG / ask


def test_ask_extractive_returns_cited_passages(doc_client):
    upload(doc_client, [SOLAR], "solar.pdf")
    manual = upload(doc_client, [BAKING, MANUAL], "manual.pdf")
    body = doc_client.post(ASK, json={"question": "How do I reset the controller?"}).json()
    assert body["found"] and body["grounded"]
    assert body["llm_model"] == "extractive"
    top = body["citations"][0]
    assert top["source"] == 1 and "[1]" in body["answer"]
    assert any(c["document_id"] == manual["id"] and c["page_number"] == 2 for c in body["citations"])


def test_ask_with_llm_maps_citations_to_pages(doc_client):
    upload(doc_client, [SOLAR], "solar.pdf")
    manual = upload(doc_client, [BAKING, MANUAL], "manual.pdf")
    use(get_reranker, OverlapReranker())
    llm = FakeLLM("Hold the red button for ten seconds [1]. Part ZX-4471 fits newer models [1][9].")
    use(get_llm, llm)

    body = doc_client.post(ASK, json={"question": "How do I reset the controller with the red button?"}).json()
    assert body["found"] and body["grounded"]
    assert [c["source"] for c in body["citations"]] == [1]
    assert body["citations"][0]["document_id"] == manual["id"]
    assert body["citations"][0]["page_number"] == 2
    assert "[9]" not in body["answer"] and "[1]" in body["answer"]
    assert any("non-existent" in w for w in body["warnings"])
    assert "llm" in body["timings_ms"]

    system, user = llm.calls[0]
    assert "ONLY the numbered sources" in system
    assert "[1] (file: manual.pdf, page 2)" in user
    assert "Question: How do I reset the controller" in user


def test_ask_not_found_skips_llm_when_nothing_relevant(doc_client, test_settings):
    upload(doc_client, [SOLAR], "solar.pdf")
    strict = test_settings.model_copy(update={"RERANK_MIN_SCORE": 1.0})
    app.dependency_overrides[get_settings] = lambda: strict
    use(get_reranker, OverlapReranker())
    llm = FakeLLM("should never be called")
    use(get_llm, llm)

    body = doc_client.post(ASK, json={"question": "Who won the football world cup?"}).json()
    assert body["found"] is False and body["grounded"] is False
    assert body["answer"] == NOT_FOUND_ANSWER
    assert body["citations"] == [] and body["sources"] == []
    assert llm.calls == []  # no LLM call -> no chance to hallucinate


def test_ask_llm_says_not_found(doc_client):
    upload(doc_client, [SOLAR], "solar.pdf")
    use(get_llm, FakeLLM("I could not find the answer in the uploaded documents."))
    body = doc_client.post(ASK, json={"question": "What is the battery capacity?"}).json()
    assert body["found"] is False and body["citations"] == []
    assert body["sources"]  # sources are still shown for transparency


def test_ask_uncited_answer_gets_auto_attributed(doc_client):
    solar = upload(doc_client, [SOLAR], "solar.pdf")
    use(get_llm, FakeLLM("Solar panels keep producing electricity for twenty five years."))
    body = doc_client.post(ASK, json={"question": "How long do solar panels last?"}).json()
    assert body["found"] is True and body["grounded"] is True
    assert body["answer"].endswith("years [1].")
    assert body["citations"][0]["document_id"] == solar["id"]
    assert any("added automatically" in w for w in body["warnings"])


def test_ask_uncited_unsupported_answer_is_flagged(doc_client):
    upload(doc_client, [SOLAR], "solar.pdf")
    use(get_llm, FakeLLM("The moon is mostly made of green cheese."))
    body = doc_client.post(ASK, json={"question": "How long do solar panels last?"}).json()
    assert body["found"] is True and body["grounded"] is False
    assert "[" not in body["answer"]
    assert any("no citations" in w for w in body["warnings"])


def test_ask_no_documents_and_validation(doc_client):
    body = doc_client.post(ASK, json={"question": "anything?"}).json()
    assert body["found"] is False and body["answer"] == NOT_FOUND_ANSWER
    assert doc_client.post(ASK, json={"question": ""}).status_code == 422


def test_ask_document_filter(doc_client):
    solar = upload(doc_client, [SOLAR], "solar.pdf")
    upload(doc_client, [MANUAL], "manual.pdf")
    body = doc_client.post(ASK, json={"question": "part number", "document_ids": [solar["id"]]}).json()
    assert all(s["document_id"] == solar["id"] for s in body["sources"])


def test_ask_llm_unavailable_returns_503(doc_client):
    upload(doc_client, [SOLAR], "solar.pdf")
    use(get_llm, BrokenLLM(""))
    res = doc_client.post(ASK, json={"question": "solar panels?"})
    assert res.status_code == 503
    assert "Ollama is not reachable" in res.json()["detail"]


# ------------------------------------------------------- citation helpers


def test_parse_citations():
    valid, invalid = parse_citations("A [1][2, 3]. B [2]. C [9]. D [0]", source_count=3)
    assert valid == [1, 2, 3] and invalid == [9, 0]


def test_strip_invalid_citations():
    assert strip_invalid_citations("Fact [1, 9]. Other [7].", 3) == "Fact [1]. Other."


# ----------------------------------------------------------- Ollama client


def test_ollama_request_and_errors(monkeypatch):
    sent = {}

    def fake_post(url, json, timeout):
        sent.update(url=url, json=json)
        return httpx.Response(200, json={"message": {"content": "Answer [1]"}})

    monkeypatch.setattr(httpx, "post", fake_post)
    llm = OllamaLLM("http://127.0.0.1:11434/", "llama3.2:3b", 100, 0.1, 30)
    assert llm.generate("sys", "user") == "Answer [1]"
    assert sent["url"] == "http://127.0.0.1:11434/api/chat"
    assert sent["json"]["model"] == "llama3.2:3b" and sent["json"]["stream"] is False
    assert sent["json"]["messages"][0] == {"role": "system", "content": "sys"}
    assert sent["json"]["options"]["num_ctx"] == 4096

    monkeypatch.setattr(httpx, "post", lambda *a, **k: httpx.Response(404, text="not found"))
    with pytest.raises(LLMUnavailable, match="ollama pull"):
        llm.generate("s", "u")

    def refuse(*a, **k):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "post", refuse)
    with pytest.raises(LLMUnavailable, match="not reachable"):
        llm.generate("s", "u")


# ------------------------------------------------- real models (opt-in)

real_models = pytest.mark.skipif(
    os.getenv("RUN_MODEL_TESTS") != "1", reason="loads real models; run with RUN_MODEL_TESTS=1"
)


@real_models
def test_real_cross_encoder_ranks_relevant_passage_first():
    from app.services.reranker import build_reranker

    reranker = build_reranker(Settings(_env_file=None, RERANKER_PROVIDER="cross-encoder"))
    scores = reranker.score("How long do solar panels last?", [BAKING, SOLAR, MANUAL])
    assert scores[1] == max(scores)
    assert scores[1] > 0 > scores[0]  # relevant positive, unrelated negative


@real_models
def test_real_rag_pipeline_end_to_end(doc_client):
    from app.services.embeddings import build_embedder
    from app.services.llm import build_llm
    from app.services.reranker import build_reranker
    from app.services.embeddings import get_embedder

    real = Settings(_env_file=None)
    use(get_embedder, build_embedder(real))
    use(get_reranker, build_reranker(real))
    use(get_llm, build_llm(real))
    upload(doc_client, [SOLAR], "solar.pdf")
    manual = upload(doc_client, [BAKING, MANUAL], "manual.pdf")

    body = doc_client.post(ASK, json={"question": "How do I reset the controller?"}).json()
    print("\nREAL ANSWER:", body["answer"], "\nWARNINGS:", body["warnings"])
    assert body["found"] is True
    assert body["answer"].strip()
    assert body["sources"][0]["document_id"] == manual["id"]
    assert body["sources"][0]["page_number"] == 2
    assert body["grounded"] is True  # model citation or automatic attribution
    assert any(c["page_number"] == 2 for c in body["citations"])

    off_topic = doc_client.post(ASK, json={"question": "Who won the 2018 football world cup?"}).json()
    print("OFF-TOPIC:", off_topic["answer"])
    assert off_topic["found"] is False


# ------------------------------------------- low-confidence fallback (rerank threshold tuning)

from app.services.hybrid_search import HybridHit  # noqa: E402
from app.services.rag import LOW_CONFIDENCE_WARNING, keyword_fallback_hit, select_sources  # noqa: E402


def _hybrid(page: int, rerank: float, semantic: float | None, bm25_rank: int | None) -> HybridHit:
    return HybridHit(
        hit=hit(f"doc1-chunk{page}", 1, page, f"text of page {page}"),
        semantic_rank=1 if semantic is not None else None,
        semantic_score=semantic,
        bm25_rank=bm25_rank,
        rerank_score=rerank,
    )


def test_fallback_needs_keyword_match_and_semantic_similarity(test_settings):
    s = test_settings.model_copy(update={"RERANK_MIN_SCORE": -6.0, "RERANK_FALLBACK_MIN_SEMANTIC": 0.25})
    # Real scores measured for "Which filter cartridge should I buy?" vs off-topic questions
    rescued = [_hybrid(2, -9.58, 0.306, 1), _hybrid(3, -11.26, 0.2, None)]
    off_topic = [_hybrid(2, -11.13, 0.119, None), _hybrid(3, -11.2, 0.05, None)]
    keyword_only = [_hybrid(2, -10.14, 0.171, 2)]  # BM25 hit but weak meaning match
    assert keyword_fallback_hit(rescued, s) is rescued[0]
    assert keyword_fallback_hit(off_topic, s) is None
    assert keyword_fallback_hit(keyword_only, s) is None

    sources = select_sources(rescued, s, use_threshold=True)
    assert [src.hit for src in sources] == [rescued[0]] and sources[0].number == 1
    assert select_sources(off_topic, s, use_threshold=True) == []


def test_fallback_only_when_nothing_passes_threshold(test_settings):
    s = test_settings.model_copy(update={"RERANK_MIN_SCORE": -6.0, "RERANK_FALLBACK_MIN_SEMANTIC": 0.25})
    hits = [_hybrid(1, 2.0, 0.5, 1), _hybrid(2, -9.0, 0.4, 1)]
    assert [src.hit for src in select_sources(hits, s, use_threshold=True)] == [hits[0]]


def test_fallback_can_be_disabled(test_settings):
    s = test_settings.model_copy(update={"RERANK_FALLBACK_MIN_SEMANTIC": 1.5})
    assert select_sources([_hybrid(2, -9.58, 0.9, 1)], s, use_threshold=True) == []


def test_ask_uses_fallback_with_low_confidence_warning(doc_client, test_settings):
    upload(doc_client, [MANUAL], "manual.pdf")
    # Nothing can pass the threshold; any semantic similarity is enough for the fallback,
    # so the result depends only on the BM25 keyword match.
    s = test_settings.model_copy(update={"RERANK_MIN_SCORE": 100.0, "RERANK_FALLBACK_MIN_SEMANTIC": -1.0})
    app.dependency_overrides[get_settings] = lambda: s
    use(get_reranker, OverlapReranker())
    llm = FakeLLM("Order part number ZX-4471 [1].")
    use(get_llm, llm)

    body = doc_client.post(ASK, json={"question": "Which replacement part should I order?"}).json()
    assert body["found"] is True and body["grounded"] is True
    assert body["citations"][0]["page_number"] == 1
    assert body["warnings"][0] == LOW_CONFIDENCE_WARNING
    assert len(llm.calls) == 1

    # Off-topic: no keyword overlap -> still "not found" and the LLM is not called
    body = doc_client.post(ASK, json={"question": "Who won the football world cup?"}).json()
    assert body["found"] is False and body["answer"] == NOT_FOUND_ANSWER
    assert len(llm.calls) == 1


def test_confident_answers_have_no_low_confidence_warning(doc_client):
    upload(doc_client, [MANUAL], "manual.pdf")
    use(get_llm, FakeLLM("Hold the red button for ten seconds [1]."))
    body = doc_client.post(ASK, json={"question": "How do I reset the controller?"}).json()
    assert body["found"] is True
    assert LOW_CONFIDENCE_WARNING not in body["warnings"]


def test_best_sentence_support():
    from types import SimpleNamespace

    from app.services.rag import best_sentence_support

    src = [SimpleNamespace(text=MANUAL)]
    assert best_sentence_support("Replacement part number ZX-4471 fits models after 2019 [1].", src) >= 0.9
    # Any one well-supported sentence is enough
    assert best_sentence_support("Check the manual. Replacement part number ZX-4471 fits all models [1].", src) >= 0.8
    # Invented answer (the real WiFi case: only the product name overlaps)
    assert best_sentence_support("To connect it to WiFi, refer to page 5 of the guide [1].", src) < 0.3


def test_ask_fallback_rejects_answer_not_supported_by_source(doc_client, test_settings):
    from app.services.rag import UNSUPPORTED_LOW_CONFIDENCE_WARNING

    upload(doc_client, [MANUAL], "manual.pdf")
    s = test_settings.model_copy(update={"RERANK_MIN_SCORE": 100.0, "RERANK_FALLBACK_MIN_SEMANTIC": -1.0})
    app.dependency_overrides[get_settings] = lambda: s
    use(get_reranker, OverlapReranker())
    use(get_llm, FakeLLM("To connect the controller to WiFi, open the settings app on page 5 [1]."))
    body = doc_client.post(ASK, json={"question": "How do I connect the controller to WiFi?"}).json()
    assert body["found"] is False and body["answer"] == NOT_FOUND_ANSWER
    assert body["citations"] == []
    assert UNSUPPORTED_LOW_CONFIDENCE_WARNING in body["warnings"]


def test_support_check_does_not_apply_to_confident_answers(doc_client):
    # Normal (above-threshold) answers keep the Phase 4 behaviour: a paraphrase is fine.
    upload(doc_client, [MANUAL], "manual.pdf")
    use(get_llm, FakeLLM("Press and keep holding it for a while [1]."))
    body = doc_client.post(ASK, json={"question": "How do I reset the controller?"}).json()
    assert body["found"] is True
