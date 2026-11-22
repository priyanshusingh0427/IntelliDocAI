"""Split extracted page text into overlapping chunks.

Why chunk at all? Embedding models only "see" a few hundred words at a time, and
search results are more precise when each vector represents one focused passage.

Design choices:
- Chunks never cross a page boundary, so every chunk has exactly one page number
  (this is what lets Phase 4 cite "page 3" in answers).
- We cut at the most natural boundary available inside the size limit:
  paragraph break > line break > sentence end > space > hard cut.
- Consecutive chunks on a page overlap by about CHUNK_OVERLAP characters so a
  sentence split across two chunks is still fully present in one of them.
- char_start / char_end are offsets into the page text: page_text[start:end] == chunk text.
"""
from dataclasses import dataclass
from typing import Iterable, Protocol

# Preferred split points, best first.
_SEPARATORS = ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ")


class PageLike(Protocol):
    page_number: int
    text: str


@dataclass(frozen=True)
class Chunk:
    chunk_index: int  # 0-based position within the whole document
    page_number: int  # 1-based
    char_start: int
    char_end: int
    text: str

    @property
    def char_count(self) -> int:
        return len(self.text)


def _find_break(text: str, start: int, hard_end: int, min_end: int) -> int:
    """Best cut position in text[min_end:hard_end], or hard_end if none found."""
    for sep in _SEPARATORS:
        pos = text.rfind(sep, min_end, hard_end)
        if pos != -1:
            return pos + len(sep)  # keep the separator with the left chunk
    return hard_end


def _trimmed_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def split_text(text: str, chunk_size: int, overlap: int) -> list[tuple[int, int]]:
    """Return (start, end) spans covering `text`, each at most chunk_size long."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    spans: list[tuple[int, int]] = []
    length = len(text)
    start = 0
    while start < length:
        hard_end = min(start + chunk_size, length)
        if hard_end < length:
            # Don't accept a break that would make the chunk tiny (< half the size).
            end = _find_break(text, start, hard_end, start + chunk_size // 2)
        else:
            end = hard_end

        s, e = _trimmed_span(text, start, end)
        if e > s:
            spans.append((s, e))
        if end >= length:
            break

        # Step back by `overlap`, then forward to the next word start so chunks
        # don't begin mid-word. Always make progress.
        next_start = max(end - overlap, start + 1)
        if overlap:
            space = text.find(" ", next_start, end)
            if space != -1:
                next_start = space + 1
        start = next_start
    return spans


def chunk_pages(pages: Iterable[PageLike], chunk_size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in pages:
        text = page.text or ""
        for start, end in split_text(text, chunk_size, overlap):
            chunks.append(
                Chunk(
                    chunk_index=len(chunks),
                    page_number=page.page_number,
                    char_start=start,
                    char_end=end,
                    text=text[start:end],
                )
            )
    return chunks
