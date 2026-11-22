"""Text extraction.

PDF  -> PyMuPDF, page by page (1-based page numbers) plus document metadata.
DOCX -> python-docx (no fixed pages, stored as a single page 1).
TXT  -> decoded text (stored as a single page 1).
"""
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pymupdf
from docx import Document as DocxDocument

# MuPDF otherwise prints repair warnings for damaged files straight to stderr.
pymupdf.TOOLS.mupdf_display_errors(False)

PDF_SIGNATURE = b"%PDF-"


class ExtractionError(Exception):
    """Base class: the file could not be turned into text. Carries an HTTP status."""

    status_code = 422


class InvalidFileTypeError(ExtractionError):
    status_code = 415


class CorruptedFileError(ExtractionError):
    status_code = 422


class EncryptedFileError(ExtractionError):
    status_code = 422


class TooManyPagesError(ExtractionError):
    status_code = 413


@dataclass
class PageText:
    page_number: int  # 1-based
    text: str
    width: float | None = None
    height: float | None = None


@dataclass
class ExtractionResult:
    pages: list[PageText]
    metadata: dict = field(default_factory=dict)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text).strip()


# ---------------------------------------------------------------- PDF helpers

_PDF_DATE = re.compile(
    r"^D:(\d{4})(\d{2})?(\d{2})?(\d{2})?(\d{2})?(\d{2})?(?:([Zz+\-])(\d{2})?'?(\d{2})?'?)?"
)


def parse_pdf_date(value: str | None) -> str | None:
    """Convert a PDF date like "D:20240115103000+05'30'" to ISO 8601 (raw value if unparsable)."""
    if not value:
        return None
    match = _PDF_DATE.match(value.strip())
    if not match:
        return value
    year, month, day, hour, minute, second, sign, tz_h, tz_m = match.groups()
    try:
        tz = timezone.utc
        if sign in ("+", "-"):
            offset = timedelta(hours=int(tz_h or 0), minutes=int(tz_m or 0))
            tz = timezone(offset if sign == "+" else -offset)
        parsed = datetime(
            int(year), int(month or 1), int(day or 1),
            int(hour or 0), int(minute or 0), int(second or 0), tzinfo=tz,
        )
        return parsed.isoformat()
    except ValueError:
        return value


def has_pdf_signature(path: Path) -> bool:
    """True if a %PDF- header appears in the first 1 KB (the spec allows leading bytes)."""
    with path.open("rb") as fh:
        return PDF_SIGNATURE in fh.read(1024)


def _extract_pdf(path: Path, max_pages: int) -> ExtractionResult:
    if not has_pdf_signature(path):
        raise InvalidFileTypeError("File has a .pdf extension but is not a PDF (missing %PDF header).")

    try:
        # Open from memory, not by path: on Windows a failed open can leave the file
        # handle locked, which would stop us deleting the rejected upload.
        doc = pymupdf.open(stream=path.read_bytes(), filetype="pdf")
    except Exception as exc:  # noqa: BLE001 - PyMuPDF raises several types for bad files
        raise CorruptedFileError("The PDF is corrupted or unreadable.") from exc

    with doc:
        if doc.needs_pass or doc.is_encrypted:
            raise EncryptedFileError("The PDF is password-protected; upload an unencrypted copy.")
        if doc.page_count == 0:
            raise CorruptedFileError("The PDF has no pages.")
        if doc.page_count > max_pages:
            raise TooManyPagesError(
                f"The PDF has {doc.page_count} pages; the limit is {max_pages}."
            )

        pages: list[PageText] = []
        try:
            for page in doc:
                pages.append(
                    PageText(
                        page_number=page.number + 1,
                        text=page.get_text("text").strip(),
                        width=round(page.rect.width, 2),
                        height=round(page.rect.height, 2),
                    )
                )
        except Exception as exc:  # noqa: BLE001
            raise CorruptedFileError(
                f"The PDF is corrupted: page {len(pages) + 1} could not be read."
            ) from exc

        raw = doc.metadata or {}
        metadata = {
            "title": raw.get("title") or None,
            "author": raw.get("author") or None,
            "subject": raw.get("subject") or None,
            "keywords": raw.get("keywords") or None,
            "creator": raw.get("creator") or None,
            "producer": raw.get("producer") or None,
            "creation_date": parse_pdf_date(raw.get("creationDate")),
            "modification_date": parse_pdf_date(raw.get("modDate")),
            "pdf_version": raw.get("format") or None,
            "page_count": doc.page_count,
            "is_repaired": bool(getattr(doc, "is_repaired", False)),
        }

    return ExtractionResult(pages=pages, metadata=metadata)


# --------------------------------------------------------- DOCX / TXT helpers

def _extract_docx(path: Path) -> ExtractionResult:
    try:
        doc = DocxDocument(str(path))
    except Exception as exc:  # noqa: BLE001 - bad zip, missing parts, etc.
        raise CorruptedFileError("The DOCX file is corrupted or unreadable.") from exc
    text = "\n".join(p.text for p in doc.paragraphs).strip()
    props = doc.core_properties
    metadata = {
        "title": props.title or None,
        "author": props.author or None,
        "subject": props.subject or None,
        "keywords": props.keywords or None,
        "creation_date": props.created.isoformat() if props.created else None,
        "modification_date": props.modified.isoformat() if props.modified else None,
    }
    return ExtractionResult(pages=[PageText(page_number=1, text=text)], metadata=metadata)


def _extract_txt(path: Path) -> ExtractionResult:
    raw = path.read_bytes()
    text = None
    used = "latin-1"
    for encoding in ("utf-8-sig", "utf-16"):
        try:
            text = raw.decode(encoding)
            used = encoding
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        text = raw.decode("latin-1")  # never fails; last resort
    return ExtractionResult(
        pages=[PageText(page_number=1, text=text.strip())], metadata={"encoding": used}
    )


def extract(path: Path, extension: str, max_pages: int = 2000) -> ExtractionResult:
    """Extract text, page structure and metadata. Raises ExtractionError subclasses."""
    if extension == ".pdf":
        return _extract_pdf(path, max_pages)
    if extension == ".docx":
        return _extract_docx(path)
    if extension == ".txt":
        return _extract_txt(path)
    raise InvalidFileTypeError(f"No extractor for {extension}")


def extract_text(path: Path, extension: str) -> str:
    """Phase 2 v1 helper kept for compatibility: plain text only."""
    return extract(path, extension).full_text
