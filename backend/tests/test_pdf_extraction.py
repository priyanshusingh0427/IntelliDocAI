"""Phase 2: PyMuPDF extraction, page numbers, metadata, validation and secure storage."""
from pathlib import Path

import pymupdf
import pytest
from sqlalchemy import inspect, text

from app.core.config import Settings, get_settings
from app.db.session import init_db, make_engine
from app.services.extraction import parse_pdf_date
from main import app

URL = "/api/v1/documents"


def build_pdf(pages: list[str], metadata: dict | None = None, password: str | None = None) -> bytes:
    doc = pymupdf.open()
    for page_text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), page_text, fontsize=12)
    if metadata:
        doc.set_metadata(metadata)
    kwargs = {}
    if password:
        kwargs = {
            "encryption": pymupdf.PDF_ENCRYPT_AES_256,
            "owner_pw": password,
            "user_pw": password,
        }
    data = doc.tobytes(**kwargs)
    doc.close()
    return data


def upload_pdf(client, data: bytes, name: str = "doc.pdf", content_type: str = "application/pdf"):
    return client.post(URL, files={"file": (name, data, content_type)})


def stored_files(settings: Settings) -> list[Path]:
    folder = Path(settings.UPLOAD_DIR)
    return list(folder.iterdir()) if folder.exists() else []


# ------------------------------------------------------------ happy path


def test_multipage_pdf_preserves_page_numbers_and_metadata(doc_client, test_settings):
    data = build_pdf(
        ["Introduction to IntelliDoc", "Second page body", "Final conclusions"],
        metadata={"title": "Test Report", "author": "QA Bot", "subject": "Testing",
                  "creationDate": "D:20240115103000+05'30'"},
    )
    response = upload_pdf(doc_client, data, "report.pdf")
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["page_count"] == 3
    assert [p["page_number"] for p in body["pages"]] == [1, 2, 3]
    assert body["metadata"]["title"] == "Test Report"
    assert body["metadata"]["author"] == "QA Bot"
    assert body["metadata"]["creation_date"] == "2024-01-15T10:30:00+05:30"
    assert body["metadata"]["page_count"] == 3
    assert len(body["sha256"]) == 64
    assert "extracted_text" not in body

    doc_id = body["id"]
    pages = doc_client.get(f"{URL}/{doc_id}/pages").json()
    assert [p["page_number"] for p in pages] == [1, 2, 3]
    assert "Second page body" in pages[1]["text"]
    assert pages[0]["width"] == pytest.approx(595, abs=1)  # A4 default

    page3 = doc_client.get(f"{URL}/{doc_id}/pages/3").json()
    assert page3["page_number"] == 3 and "Final conclusions" in page3["text"]

    detail = doc_client.get(f"{URL}/{doc_id}").json()
    assert "Introduction to IntelliDoc" in detail["extracted_text"]
    assert "Final conclusions" in detail["extracted_text"]
    assert len(stored_files(test_settings)) == 1


def test_pdf_with_blank_page_keeps_numbering(doc_client):
    body = upload_pdf(doc_client, build_pdf(["first", "", "third"])).json()
    assert [(p["page_number"], p["char_count"] > 0) for p in body["pages"]] == [
        (1, True), (2, False), (3, True)
    ]


def test_missing_page_returns_404(doc_client):
    doc_id = upload_pdf(doc_client, build_pdf(["only page"])).json()["id"]
    assert doc_client.get(f"{URL}/{doc_id}/pages/2").status_code == 404
    assert doc_client.get(f"{URL}/{doc_id}/pages/0").status_code == 422  # must be >= 1
    assert doc_client.get(f"{URL}/999/pages").status_code == 404


# ------------------------------------------------------------ validation


def test_corrupted_pdf_rejected(doc_client, test_settings):
    data = b"%PDF-1.7\n" + b"\x00\xffgarbage" * 200
    response = upload_pdf(doc_client, data, "corrupt.pdf")
    assert response.status_code == 422
    assert "corrupt" in response.json()["detail"].lower() or "no pages" in response.json()["detail"]
    assert stored_files(test_settings) == []


def test_truncated_pdf_is_rejected_or_repaired(doc_client, test_settings):
    good = build_pdf(["page one text", "page two text"])
    response = upload_pdf(doc_client, good[: len(good) // 3], "truncated.pdf")
    # MuPDF can sometimes repair a truncated file; either way the API must not 500.
    assert response.status_code in (201, 422)
    if response.status_code == 422:
        assert stored_files(test_settings) == []


def test_encrypted_pdf_rejected(doc_client, test_settings):
    response = upload_pdf(doc_client, build_pdf(["secret"], password="pw123"), "locked.pdf")
    assert response.status_code == 422
    assert "password" in response.json()["detail"].lower()
    assert stored_files(test_settings) == []


def test_wrong_content_type_rejected(doc_client):
    response = upload_pdf(doc_client, build_pdf(["x"]), "image.pdf", "image/png")
    assert response.status_code == 415


def test_png_renamed_to_pdf_rejected(doc_client, test_settings):
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
    response = upload_pdf(doc_client, png, "photo.pdf")
    assert response.status_code == 415
    assert stored_files(test_settings) == []


def test_too_many_pages_rejected(doc_client, test_settings):
    limited = test_settings.model_copy(update={"MAX_PDF_PAGES": 2})
    app.dependency_overrides[get_settings] = lambda: limited
    response = upload_pdf(doc_client, build_pdf(["1", "2", "3"]))
    assert response.status_code == 413
    assert stored_files(test_settings) == []


def test_oversized_pdf_rejected(doc_client, test_settings):
    big = b"%PDF-1.4\n" + b"0" * (test_settings.max_upload_bytes + 10)
    assert upload_pdf(doc_client, big).status_code == 413
    assert stored_files(test_settings) == []


# ------------------------------------------------------------ secure storage


def test_path_traversal_filename_is_neutralised(doc_client, test_settings):
    response = upload_pdf(doc_client, build_pdf(["safe"]), "..\\..\\evil<>.pdf")
    assert response.status_code == 201
    assert response.json()["original_filename"] == "evil__.pdf"
    files = stored_files(test_settings)
    assert len(files) == 1
    assert files[0].parent == Path(test_settings.UPLOAD_DIR).resolve()
    assert "evil" not in files[0].name  # stored under a random UUID


def test_delete_cascades_pages(doc_client):
    doc_id = upload_pdf(doc_client, build_pdf(["a", "b"])).json()["id"]
    assert doc_client.delete(f"{URL}/{doc_id}").status_code == 204
    assert doc_client.get(f"{URL}/{doc_id}/pages").status_code == 404


# ------------------------------------------------------------ helpers & DB upgrade


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("D:20240115103000+05'30'", "2024-01-15T10:30:00+05:30"),
        ("D:20231231235959Z", "2023-12-31T23:59:59+00:00"),
        ("D:2022", "2022-01-01T00:00:00+00:00"),
        ("", None),
        ("not a date", "not a date"),
    ],
)
def test_parse_pdf_date(raw, expected):
    assert parse_pdf_date(raw) == expected


def test_init_db_upgrades_old_schema_without_data_loss(tmp_path):
    engine = make_engine(f"sqlite:///{(tmp_path / 'old.db').as_posix()}")
    with engine.begin() as conn:  # table as created by Phase 2 v1
        conn.execute(text(
            "CREATE TABLE documents (id INTEGER PRIMARY KEY, original_filename VARCHAR(255), "
            "stored_filename VARCHAR(255) UNIQUE, content_type VARCHAR(127), extension VARCHAR(16), "
            "size_bytes INTEGER, status VARCHAR(16), error_message TEXT, extracted_text TEXT, "
            "char_count INTEGER, created_at DATETIME)"
        ))
        conn.execute(text(
            "INSERT INTO documents (original_filename, stored_filename, extension, size_bytes, "
            "status, char_count) VALUES ('old.txt', 'abc.txt', '.txt', 3, 'processed', 3)"
        ))

    init_db(engine)

    inspector = inspect(engine)
    columns = {c["name"] for c in inspector.get_columns("documents")}
    assert {"page_count", "sha256", "doc_metadata"} <= columns
    assert "document_pages" in inspector.get_table_names()
    with engine.connect() as conn:
        assert conn.execute(text("SELECT original_filename FROM documents")).scalar() == "old.txt"
    engine.dispose()


def test_settings_paths_are_anchored_to_backend_folder():
    settings = Settings(_env_file=None, UPLOAD_DIR="uploads", DATABASE_URL="sqlite:///./intellidoc.db")
    backend_dir = Path(__file__).resolve().parents[1]
    assert settings.upload_path == (backend_dir / "uploads").resolve()
    assert settings.database_url.endswith((backend_dir / "intellidoc.db").resolve().as_posix())
    assert Settings(_env_file=None, DATABASE_URL="sqlite://").database_url == "sqlite://"
