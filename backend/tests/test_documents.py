import io
from pathlib import Path

from docx import Document as DocxDocument

URL = "/api/v1/documents"


def make_pdf(text: str) -> bytes:
    """Build a minimal one-page PDF with a correct xref table."""
    stream = f"BT /F1 24 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref_pos = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode()
    return bytes(out)


def make_docx(text: str) -> bytes:
    buf = io.BytesIO()
    doc = DocxDocument()
    doc.add_paragraph(text)
    doc.save(buf)
    return buf.getvalue()


def upload(client, name: str, data: bytes, content_type: str = "application/octet-stream"):
    return client.post(URL, files={"file": (name, data, content_type)})


def test_upload_txt(doc_client, test_settings):
    response = upload(doc_client, "notes.txt", b"Hello IntelliDoc", "text/plain")
    assert response.status_code == 201
    body = response.json()
    assert body["original_filename"] == "notes.txt"
    assert body["extension"] == ".txt"
    assert body["status"] == "processed"
    assert body["char_count"] == len("Hello IntelliDoc")
    assert "extracted_text" not in body  # list/summary schema omits text
    assert len(list(Path(test_settings.UPLOAD_DIR).iterdir())) == 1


def test_upload_docx(doc_client):
    response = upload(doc_client, "report.docx", make_docx("Quarterly report text"))
    assert response.status_code == 201
    doc_id = response.json()["id"]
    detail = doc_client.get(f"{URL}/{doc_id}").json()
    assert detail["status"] == "processed"
    assert "Quarterly report text" in detail["extracted_text"]


def test_upload_pdf(doc_client):
    response = upload(doc_client, "paper.pdf", make_pdf("Hello PDF World"), "application/pdf")
    assert response.status_code == 201
    detail = doc_client.get(f"{URL}/{response.json()['id']}").json()
    assert detail["status"] == "processed"
    assert "Hello PDF World" in detail["extracted_text"]


def test_corrupt_pdf_is_stored_as_failed(doc_client):
    # Phase 2 v2: a ".pdf" that is not really a PDF is now rejected (was stored as "failed").
    response = upload(doc_client, "broken.pdf", b"this is not a pdf", "application/pdf")
    assert response.status_code == 415
    assert "not a PDF" in response.json()["detail"]
    assert doc_client.get("/api/v1/documents").json() == []


def test_rejects_unsupported_extension(doc_client, test_settings):
    response = upload(doc_client, "virus.exe", b"MZ...")
    assert response.status_code == 400
    assert not Path(test_settings.UPLOAD_DIR).exists() or not any(
        Path(test_settings.UPLOAD_DIR).iterdir()
    )


def test_rejects_empty_file(doc_client):
    assert upload(doc_client, "empty.txt", b"").status_code == 400


def test_rejects_too_large_file(doc_client, test_settings):
    too_big = b"a" * (test_settings.max_upload_bytes + 1)
    response = upload(doc_client, "big.txt", too_big, "text/plain")
    assert response.status_code == 413
    assert not any(Path(test_settings.UPLOAD_DIR).iterdir())  # partial file cleaned up


def test_list_documents(doc_client):
    upload(doc_client, "a.txt", b"first")
    upload(doc_client, "b.txt", b"second")
    response = doc_client.get(URL)
    assert response.status_code == 200
    names = [d["original_filename"] for d in response.json()]
    assert names == ["b.txt", "a.txt"]  # newest first
    assert len(doc_client.get(URL, params={"limit": 1}).json()) == 1


def test_delete_document_removes_file(doc_client, test_settings):
    doc_id = upload(doc_client, "gone.txt", b"bye").json()["id"]
    assert doc_client.delete(f"{URL}/{doc_id}").status_code == 204
    assert doc_client.get(f"{URL}/{doc_id}").status_code == 404
    assert not any(Path(test_settings.UPLOAD_DIR).iterdir())


def test_missing_document_returns_404(doc_client):
    assert doc_client.get(f"{URL}/999").status_code == 404
    assert doc_client.delete(f"{URL}/999").status_code == 404
