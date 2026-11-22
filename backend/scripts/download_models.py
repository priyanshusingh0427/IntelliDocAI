r"""Download the embedding model and cross-encoder into EMBEDDING_CACHE_DIR and test them once.

Used by the Docker build (so containers start without internet access), and handy on a new PC:
    ..\.venv\Scripts\python.exe scripts\download_models.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.services.embeddings import build_embedder  # noqa: E402
from app.services.reranker import build_reranker  # noqa: E402


def main() -> None:
    settings = get_settings()
    embedder = build_embedder(settings)
    vector = embedder.embed(["warm-up sentence"])[0]
    print(f"embedding model ready: {embedder.model_name} ({len(vector)} dimensions)")
    reranker = build_reranker(settings)
    score = reranker.score("how do I reset it?", ["Hold the reset button for ten seconds."])[0]
    print(f"reranker ready: {reranker.model_name} (test score {score:.2f})")
    print(f"models stored in {settings.embedding_cache_path}")


if __name__ == "__main__":
    main()
