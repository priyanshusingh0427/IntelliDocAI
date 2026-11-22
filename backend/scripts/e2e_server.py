"""Start a throwaway IntelliDoc backend for the browser end-to-end tests.

Playwright (frontend/playwright.config.js) runs this automatically. Every run starts from
nothing: a fresh SQLite database, upload folder and ChromaDB index inside backend/.e2e-run/.
Your real data (PostgreSQL, uploads/, chroma_db/) is never used, because every storage
setting is overridden here and environment variables take priority over backend/.env.

The real embedding model and cross-encoder are used (from the local models/ cache).
Answers use the "extractive" provider by default (fast, deterministic); set
E2E_LLM_PROVIDER=ollama to run the same tests against the local Ollama model.

    ..\\.venv\\Scripts\\python.exe scripts\\e2e_server.py --port 8011
"""
import argparse
import os
import secrets
import shutil
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
RUN_DIR = BACKEND / ".e2e-run"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8011)
    args = parser.parse_args()

    # Only ever deletes its own dedicated folder.
    if RUN_DIR.name != ".e2e-run" or RUN_DIR.parent != BACKEND:
        raise SystemExit(f"Refusing to delete unexpected folder: {RUN_DIR}")
    if RUN_DIR.exists():
        shutil.rmtree(RUN_DIR)
    RUN_DIR.mkdir()

    os.environ.update(
        {
            "ENVIRONMENT": "e2e",
            "DATABASE_URL": os.environ.get("E2E_DATABASE_URL", f"sqlite:///{(RUN_DIR / 'e2e.db').as_posix()}"),
            "UPLOAD_DIR": str(RUN_DIR / "uploads"),
            "CHROMA_DIR": str(RUN_DIR / "chroma"),
            "CHROMA_COLLECTION": "e2e",
            "SECRET_KEY": secrets.token_urlsafe(32),
            "LLM_PROVIDER": os.environ.get("E2E_LLM_PROVIDER", "extractive"),
            "ALLOW_REGISTRATION": "true",
            "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
        }
    )
    os.chdir(BACKEND)
    sys.path.insert(0, str(BACKEND))

    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
