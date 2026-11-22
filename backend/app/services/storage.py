"""Saving and deleting uploaded files on local disk.

Security measures:
- Files are stored under a random UUID name, never the client-supplied name,
  so path traversal ("../../x.pdf") and name collisions are impossible.
- The resolved path is checked to stay inside the upload folder.
- Uploads are streamed in chunks and aborted as soon as the size limit is hit.
- Partially written or rejected files are always removed.
- The upload folder is not served over HTTP.
"""
import hashlib
import re
import unicodedata
import uuid
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

from fastapi import HTTPException, UploadFile, status

from app.core.config import Settings

CHUNK_SIZE = 1024 * 1024  # 1 MB
_UNSAFE_CHARS = re.compile(r"[\x00-\x1f\x7f<>:\"/\\|?*]")


@dataclass
class StoredFile:
    path: Path
    stored_filename: str
    extension: str
    size_bytes: int
    sha256: str = ""


def sanitize_filename(filename: str) -> str:
    """Display-safe original name: drop any directory parts and control characters."""
    name = PureWindowsPath(filename).name  # handles both "/" and "\" separators
    name = unicodedata.normalize("NFC", name)
    name = _UNSAFE_CHARS.sub("_", name).strip(" .")
    return name[:255] or "unnamed"


def _safe_path(settings: Settings, stored_filename: str) -> Path:
    base = settings.upload_path
    path = (base / stored_filename).resolve()
    if base not in path.parents:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid file path.")
    return path


# Declared MIME types accepted per extension. Generic binary types are allowed because
# some clients (curl, scripts) send them; the real check is the file signature + parser.
_ALLOWED_CONTENT_TYPES = {
    ".pdf": {"application/pdf", "application/x-pdf", "application/octet-stream", "binary/octet-stream"},
    ".docx": {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/octet-stream",
        "application/zip",
    },
    ".txt": {"text/plain", "application/octet-stream"},
}


def validate_content_type(content_type: str | None, extension: str) -> None:
    if not content_type:
        return
    base_type = content_type.split(";")[0].strip().lower()
    allowed = _ALLOWED_CONTENT_TYPES.get(extension)
    if allowed is not None and base_type not in allowed:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Content type {base_type} does not match a {extension} file.",
        )


def get_extension(filename: str) -> str:
    return Path(filename).suffix.lower()


def validate_extension(filename: str, settings: Settings) -> str:
    extension = get_extension(filename)
    if extension not in settings.ALLOWED_EXTENSIONS:
        shown = extension or "(none)"
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type {shown}. Allowed: {', '.join(settings.ALLOWED_EXTENSIONS)}",
        )
    return extension


async def save_upload(upload: UploadFile, settings: Settings) -> StoredFile:
    """Validate the upload and stream it to disk, enforcing the size limit."""
    extension = validate_extension(upload.filename or "", settings)
    validate_content_type(upload.content_type, extension)

    upload_dir = settings.upload_path
    upload_dir.mkdir(parents=True, exist_ok=True)
    stored_filename = f"{uuid.uuid4().hex}{extension}"
    path = _safe_path(settings, stored_filename)

    size = 0
    digest = hashlib.sha256()
    try:
        with path.open("xb") as out:  # "x": never overwrite an existing file
            while chunk := await upload.read(CHUNK_SIZE):
                size += len(chunk)
                digest.update(chunk)
                if size > settings.max_upload_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File exceeds the {settings.MAX_UPLOAD_SIZE_MB} MB limit.",
                    )
                out.write(chunk)
    except BaseException:
        path.unlink(missing_ok=True)
        raise

    if size == 0:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File is empty.")

    return StoredFile(
        path=path,
        stored_filename=stored_filename,
        extension=extension,
        size_bytes=size,
        sha256=digest.hexdigest(),
    )


def delete_file(stored_filename: str, settings: Settings) -> None:
    _safe_path(settings, stored_filename).unlink(missing_ok=True)
