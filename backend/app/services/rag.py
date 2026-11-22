"""RAG (Retrieval-Augmented Generation): answer a question from the user's documents.

1. Retrieve: hybrid search + rerank finds the most relevant chunks.
2. Filter:   drop chunks the reranker judges unrelated (RERANK_MIN_SCORE). If nothing
             is left, answer "not found" WITHOUT calling the LLM - this is the main
             guard against made-up answers.
3. Augment:  put the chunks in the prompt as numbered sources [1], [2], ... each
             labelled with its file name and page number.
4. Generate: the LLM answers using only those sources and cites them like [1].
5. Verify:   citations are parsed and mapped back to file + page. Citation numbers
             that do not match a source are dropped.
"""
import re
import time
from dataclasses import dataclass, field

from app.core.config import Settings
from app.services.hybrid_search import HybridHit, hybrid_search
from app.services.embeddings import Embedder
from app.services.keyword_search import KeywordSearcher
from app.services.llm import LLM
from app.services.reranker import Reranker
from app.services.vector_store import VectorStore

NOT_FOUND_ANSWER = "I could not find the answer in the uploaded documents."

SYSTEM_PROMPT = f"""You are IntelliDoc AI, an assistant that answers questions about the user's documents.
Rules:
1. Use ONLY the numbered sources provided. Do not use outside knowledge.
2. After every sentence that uses a source, cite it with its number in square brackets, like [1] or [2][3].
3. If the sources do not contain the answer, reply exactly: "{NOT_FOUND_ANSWER}"
4. Be concise and factual."""

_CITATION = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")


@dataclass
class Source:
    number: int  # the [n] used in the prompt / answer
    hit: HybridHit
    text: str  # possibly truncated to fit RAG_MAX_CONTEXT_CHARS


@dataclass
class RagAnswer:
    answer: str
    grounded: bool  # True when the answer cites at least one valid source
    found: bool  # False when no relevant source existed (LLM not called) or LLM said so
    citations: list[Source]
    sources: list[Source]
    llm_model: str
    reranker_model: str
    warnings: list[str] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)


def select_sources(hits: list[HybridHit], settings: Settings, use_threshold: bool) -> list[Source]:
    sources: list[Source] = []
    budget = settings.RAG_MAX_CONTEXT_CHARS
    for h in hits:
        if len(sources) >= settings.RAG_CONTEXT_CHUNKS or budget <= 0:
            break
        if use_threshold and h.rerank_score is not None and h.rerank_score < settings.RERANK_MIN_SCORE:
            continue
        text = h.hit.text[:budget]
        budget -= len(text)
        sources.append(Source(number=len(sources) + 1, hit=h, text=text))
    if use_threshold and not sources:
        fallback = keyword_fallback_hit(hits, settings)
        if fallback is not None:
            sources.append(Source(number=1, hit=fallback, text=fallback.hit.text[: settings.RAG_MAX_CONTEXT_CHARS]))
    return sources


def keyword_fallback_hit(hits: list[HybridHit], settings: Settings) -> HybridHit | None:
    """Best-reranked chunk that both retrievers agree on (BM25 keyword match AND semantic
    similarity >= RERANK_FALLBACK_MIN_SEMANTIC). Used only when nothing passed the reranker
    threshold; off-topic questions fail one or both checks, so they still get "not found"."""
    for h in hits:  # already in rerank order
        if (
            h.bm25_rank is not None
            and h.semantic_score is not None
            and h.semantic_score >= settings.RERANK_FALLBACK_MIN_SEMANTIC
        ):
            return h
    return None


LOW_CONFIDENCE_WARNING = (
    "Low-confidence match: no passage scored as clearly relevant, so the answer uses the closest "
    "keyword + meaning match. Check it against the cited page."
)


def is_low_confidence(sources: list[Source], settings: Settings, use_threshold: bool) -> bool:
    return use_threshold and bool(sources) and all(
        s.hit.rerank_score is not None and s.hit.rerank_score < settings.RERANK_MIN_SCORE
        for s in sources
    )


def build_user_prompt(question: str, sources: list[Source]) -> str:
    blocks = [
        f"[{s.number}] (file: {s.hit.hit.filename}, page {s.hit.hit.page_number})\n{s.text}"
        for s in sources
    ]
    return (
        "Sources:\n\n"
        + "\n\n".join(blocks)
        + "\n\nExample of the required format: The warranty lasts two years [2]. "
        "Repairs are free during that time [1][2].\n\n"
        + f"Question: {question}\nAnswer with citations:"
    )


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def best_sentence_support(answer: str, sources: list[Source]) -> float:
    """Largest share of one answer sentence's (non-stopword) words found in a single source.

    1.0 = some sentence is fully made of source words; ~0 = nothing in the answer comes from
    the sources (likely invented). Sentences with fewer than 3 content words are ignored."""
    from app.services.keyword_search import tokenize

    source_tokens = [set(tokenize(s.text)) for s in sources]
    best = 0.0
    for sentence in _SENTENCE_SPLIT.split(_CITATION.sub(" ", answer).strip()):
        tokens = set(tokenize(sentence))
        if len(tokens) < 3:
            continue
        best = max(best, max(len(tokens & st) / len(tokens) for st in source_tokens))
    return best


UNSUPPORTED_LOW_CONFIDENCE_WARNING = (
    "The closest passage was only a weak match and the generated answer was not supported by "
    "its text, so no answer is given."
)


def attribute_citations(answer: str, sources: list[Source], min_overlap: float = 0.6) -> tuple[str, int]:
    """Add [n] to uncited sentences whose words mostly come from one source.

    Small local models often answer correctly but forget the [n] markers. A sentence is
    attributed to the source containing the largest share of its (non-stopword) words,
    only if that share is >= min_overlap. Returns (new answer, citations added).
    """
    from app.services.keyword_search import tokenize

    source_tokens = [(s.number, set(tokenize(s.text))) for s in sources]
    added = 0
    out: list[str] = []
    for sentence in _SENTENCE_SPLIT.split(answer.strip()):
        tokens = set(tokenize(sentence))
        if _CITATION.search(sentence) or len(tokens) < 3:
            out.append(sentence)
            continue
        number, best = max(
            ((n, len(tokens & st) / len(tokens)) for n, st in source_tokens), key=lambda x: x[1]
        )
        if best >= min_overlap:
            match = re.search(r"[.!?]+$", sentence)
            if match:
                sentence = f"{sentence[: match.start()]} [{number}]{match.group()}"
            else:
                sentence = f"{sentence} [{number}]"
            added += 1
        out.append(sentence)
    return " ".join(out), added


def parse_citations(answer: str, source_count: int) -> tuple[list[int], list[int]]:
    """Return (valid source numbers in order of first use, invalid numbers)."""
    valid: list[int] = []
    invalid: list[int] = []
    for group in _CITATION.findall(answer):
        for num in re.split(r"\s*[,;]\s*", group):
            n = int(num)
            target = valid if 1 <= n <= source_count else invalid
            if n not in target:
                target.append(n)
    return valid, invalid


def strip_invalid_citations(answer: str, source_count: int) -> str:
    """Rewrite each [..] group keeping only numbers that match a real source."""

    def fix(match: re.Match) -> str:
        nums = [n for n in re.split(r"\s*[,;]\s*", match.group(1)) if 1 <= int(n) <= source_count]
        return "".join(f"[{n}]" for n in dict.fromkeys(nums))

    cleaned = _CITATION.sub(fix, answer)
    return re.sub(r"[ \t]{2,}", " ", cleaned).replace(" .", ".")


def _is_not_found(answer: str) -> bool:
    normalized = answer.lower()
    return "could not find the answer" in normalized or "couldn't find the answer" in normalized


def _extractive_answer(sources: list[Source]) -> str:
    lines = ["Most relevant passages from your documents:"]
    for s in sources:
        snippet = " ".join(s.text.split())
        if len(snippet) > 300:
            snippet = snippet[:300].rsplit(" ", 1)[0] + " ..."
        lines.append(f"[{s.number}] {snippet}")
    return "\n".join(lines)


def answer_question(
    question: str,
    *,
    embedder: Embedder,
    store: VectorStore,
    keyword: KeywordSearcher,
    reranker: Reranker,
    llm: LLM,
    settings: Settings,
    document_ids: list[int] | None = None,
) -> RagAnswer:
    question = question.strip()
    retrieval = hybrid_search(
        question,
        embedder=embedder,
        store=store,
        keyword=keyword,
        reranker=reranker,
        settings=settings,
        top_k=max(settings.RAG_CONTEXT_CHUNKS, settings.RERANK_CANDIDATES),
        document_ids=document_ids,
        rerank=True,
    )
    timings = dict(retrieval.timings_ms)
    use_threshold = reranker.produces_relevance_scores
    sources = select_sources(retrieval.hits, settings, use_threshold=use_threshold)
    low_confidence = is_low_confidence(sources, settings, use_threshold)

    common = dict(llm_model=llm.model_name, reranker_model=retrieval.reranker_model)
    if not sources:
        return RagAnswer(
            answer=NOT_FOUND_ANSWER, grounded=False, found=False, citations=[], sources=[],
            timings_ms=timings, **common,
        )

    if not llm.is_generative:
        return RagAnswer(
            answer=_extractive_answer(sources), grounded=True, found=True,
            citations=sources, sources=sources, timings_ms=timings,
            warnings=[LOW_CONFIDENCE_WARNING] if low_confidence else [], **common,
        )

    t = time.perf_counter()
    raw = llm.generate(SYSTEM_PROMPT, build_user_prompt(question, sources))
    timings["llm"] = round((time.perf_counter() - t) * 1000, 1)

    warnings: list[str] = []
    if not raw:
        raw = NOT_FOUND_ANSWER
        warnings.append("The LLM returned an empty answer.")
    if _is_not_found(raw):
        return RagAnswer(
            answer=NOT_FOUND_ANSWER, grounded=False, found=False, citations=[],
            sources=sources, warnings=warnings, timings_ms=timings, **common,
        )
    # A low-confidence (fallback) source must visibly back the answer: at least one sentence
    # mostly made of the source's own words. Otherwise the model probably invented it.
    if low_confidence and best_sentence_support(raw, sources) < settings.LOW_CONFIDENCE_MIN_SUPPORT:
        return RagAnswer(
            answer=NOT_FOUND_ANSWER, grounded=False, found=False, citations=[], sources=sources,
            warnings=[*warnings, UNSUPPORTED_LOW_CONFIDENCE_WARNING], timings_ms=timings, **common,
        )

    valid, invalid = parse_citations(raw, len(sources))
    if invalid:
        warnings.append(f"Removed citations to non-existent sources: {invalid}")
        raw = strip_invalid_citations(raw, len(sources))
    raw, auto_added = attribute_citations(raw, sources)
    if auto_added:
        warnings.append(
            f"{auto_added} citation(s) were added automatically by matching answer sentences "
            "to the sources (the model did not cite them itself)."
        )
        valid, _ = parse_citations(raw, len(sources))
    if not valid:
        warnings.append("The answer has no citations; check it against the sources.")
    if low_confidence:
        warnings.insert(0, LOW_CONFIDENCE_WARNING)
    by_number = {s.number: s for s in sources}
    return RagAnswer(
        answer=raw.strip(),
        grounded=bool(valid),
        found=True,
        citations=[by_number[n] for n in valid],
        sources=sources,
        warnings=warnings,
        timings_ms=timings,
        **common,
    )
