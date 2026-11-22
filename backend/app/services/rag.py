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


_CITATIONS_AFTER_PUNCT = re.compile(r"([.!?])([ \t]*)((?:\[\d+(?:\s*[,;]\s*\d+)*\][ \t]*)+)")


def place_citations_before_punctuation(answer: str) -> str:
    """'...communications. [1][2]' -> '...communications [1][2].'

    Models often put citations after the full stop. The sentence splitter then sees the
    sentence as uncited, auto-attribution adds a duplicate ('[1]. [1][2]') and a misleading
    "the model did not cite" warning."""
    def move(m: re.Match) -> str:
        cites = m.group(3)
        trailing = cites[len(cites.rstrip()):]  # keep the space before the next sentence
        return f" {cites.strip()}{m.group(1)}{trailing}"

    return _CITATIONS_AFTER_PUNCT.sub(move, answer)


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


# ------------------------------------------------------------ whole-document (summary) requests

# Words that only express "give me an overview"; anything else in the question is a topic.
_SUMMARY_TRIGGERS = re.compile(
    r"\b(summari[sz]e|summari[sz]ing|summary|summaries|overview|tl;?dr|gist|outline|recap|"
    r"main (?:points?|topics?|ideas?|themes?)|key (?:points?|topics?|ideas?|takeaways?|concepts?)|"
    r"topics? covered|what(?:'s| is) (?:this|the|it) (?:document |pdf |file |text |notes? )?about)\b",
    re.IGNORECASE,
)
_SUMMARY_FILLER = frozenset(
    "summarize summarise summarizing summarising summary summaries overview tldr tl dr gist outline "
    "recap main key points point topics topic ideas idea themes theme takeaways takeaway concepts "
    "concept covered cover covers document documents doc docs pdf pdfs file files text notes note "
    "whole entire give tell list please brief briefly short quick simple simply words explain "
    "describe provide write make about contents content".split()
)


def is_summary_request(question: str) -> bool:
    """True for whole-document requests ("Summarize this document", "What is this PDF about?",
    "main topics"). A request that also names a topic ("Summarize the AES section") is NOT
    one: normal retrieval finds that topic and the LLM summarizes it."""
    if not _SUMMARY_TRIGGERS.search(question):
        return False
    from app.services.keyword_search import tokenize

    topic_words = [t for t in tokenize(question) if t not in _SUMMARY_FILLER and t != "s"]
    return len(topic_words) == 0


def _spread(items: list, count: int) -> list:
    """`count` items evenly spaced from first to last (all of them if there are fewer)."""
    if count >= len(items):
        return list(items)
    if count == 1:
        return [items[0]]
    step = (len(items) - 1) / (count - 1)
    return [items[round(i * step)] for i in range(count)]


def summary_sources(store: VectorStore, settings: Settings, document_ids: list[int] | None) -> list[Source]:
    """Excerpts spread evenly across each document (start, middle, end), in reading order.
    With several documents the slots are shared so each one is represented."""
    if document_ids:
        per_doc = {d: store.get_document_chunks(d) for d in document_ids}
    else:
        per_doc = {}
        for h in sorted(store.get_all_chunks(), key=lambda h: (h.document_id, h.chunk_index)):
            per_doc.setdefault(h.document_id, []).append(h)
    per_doc = {d: chunks for d, chunks in per_doc.items() if chunks}
    if not per_doc:
        return []

    slots = max(settings.RAG_SUMMARY_CHUNKS, len(per_doc))
    base, extra = divmod(slots, len(per_doc))
    picked: list = []
    for i, (_, chunks) in enumerate(sorted(per_doc.items())):
        picked.extend(_spread(chunks, base + (1 if i < extra else 0)))

    per_source = max(200, settings.RAG_SUMMARY_MAX_CONTEXT_CHARS // len(picked))
    return [
        Source(number=n, hit=HybridHit(hit=h), text=h.text[:per_source])
        for n, h in enumerate(picked, start=1)
    ]


def build_summary_prompt(question: str, sources: list[Source]) -> str:
    files = sorted({s.hit.hit.filename for s in sources})
    blocks = [
        f"[{s.number}] (file: {s.hit.hit.filename}, page {s.hit.hit.page_number})\n{s.text}"
        for s in sources
    ]
    return (
        f"Excerpts taken evenly from the start, middle and end of {', '.join(files)}:\n\n"
        + "\n\n".join(blocks)
        + "\n\nWrite a short summary of the main topics covered, using only these excerpts. "
        "Cite the excerpt numbers like [1][4]."
        + (" Summarize each file separately, naming it." if len(files) > 1 else "")
        + f"\n\nRequest: {question}\nSummary with citations:"
    )


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
    if is_summary_request(question):
        return _answer_summary(question, store=store, llm=llm, settings=settings,
                               document_ids=document_ids, reranker_model=reranker.model_name)
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
    raw = place_citations_before_punctuation(raw)
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


def _answer_summary(
    question: str,
    *,
    store: VectorStore,
    llm: LLM,
    settings: Settings,
    document_ids: list[int] | None,
    reranker_model: str,
) -> RagAnswer:
    t = time.perf_counter()
    sources = summary_sources(store, settings, document_ids)
    timings = {"summary_sampling": round((time.perf_counter() - t) * 1000, 1)}
    common = dict(llm_model=llm.model_name, reranker_model=reranker_model)
    if not sources:
        return RagAnswer(answer=NOT_FOUND_ANSWER, grounded=False, found=False, citations=[],
                         sources=[], timings_ms=timings, **common)

    warnings: list[str] = []
    files = sorted({s.hit.hit.filename for s in sources})
    if len(files) > 1:
        warnings.append(
            f"This summary covers {len(files)} documents ({', '.join(files)}). "
            "Tick one document in the list to summarize only that one."
        )
    if not llm.is_generative:
        return RagAnswer(answer=_extractive_answer(sources), grounded=True, found=True, citations=sources,
                         sources=sources, warnings=warnings, timings_ms=timings, **common)

    t = time.perf_counter()
    raw = place_citations_before_punctuation(
        llm.generate(SYSTEM_PROMPT, build_summary_prompt(question, sources)).strip()
    )
    timings["llm"] = round((time.perf_counter() - t) * 1000, 1)
    if not raw or _is_not_found(raw):
        warnings.append("The model could not summarize these excerpts.")
        return RagAnswer(answer=NOT_FOUND_ANSWER, grounded=False, found=False, citations=[],
                         sources=sources, warnings=warnings, timings_ms=timings, **common)

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
    by_number = {s.number: s for s in sources}
    return RagAnswer(answer=raw, grounded=bool(valid), found=True, citations=[by_number[n] for n in valid],
                     sources=sources, warnings=warnings, timings_ms=timings, **common)
