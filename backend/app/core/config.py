"""Application settings loaded from environment variables / .env file."""
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# backend/ folder. Relative paths (.env, uploads, sqlite file) resolve against it,
# so the server behaves the same no matter which folder uvicorn is started from.
BASE_DIR = Path(__file__).resolve().parents[2]

DEV_SECRET_KEY = "dev-only-insecure-secret-key-change-me"  # noqa: S105 - placeholder, refused in production


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    APP_NAME: str = "IntelliDoc AI"
    APP_VERSION: str = "0.1.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True  # kept for compatibility; never enables tracebacks in responses (see app/main.py)
    API_V1_PREFIX: str = "/api/v1"
    CORS_ORIGINS: Annotated[list[str], NoDecode] = [
        "http://localhost:5173",
        "http://localhost:3000",
    ]
    LOG_LEVEL: str = "INFO"

    # ---- Phase 2: documents ----
    DATABASE_URL: str = "sqlite:///./intellidoc.db"
    UPLOAD_DIR: str = "uploads"
    MAX_UPLOAD_SIZE_MB: int = 10
    ALLOWED_EXTENSIONS: Annotated[list[str], NoDecode] = [".pdf", ".docx", ".txt"]
    MAX_PDF_PAGES: int = 2000

    # ---- Phase 3: chunking, embeddings, semantic search ----
    CHUNK_SIZE: int = 1000  # target characters per chunk
    CHUNK_OVERLAP: int = 200  # characters shared between consecutive chunks on a page
    # "sentence-transformers" (real local model) or "hash" (tiny offline embedder for tests)
    EMBEDDING_PROVIDER: str = "sentence-transformers"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim, ~90 MB
    EMBEDDING_BATCH_SIZE: int = 32
    EMBEDDING_CACHE_DIR: str = "models"  # where downloaded model files are kept
    CHROMA_DIR: str = "chroma_db"  # ChromaDB persistent storage folder
    CHROMA_COLLECTION: str = "intellidoc_chunks"
    AUTO_INDEX_ON_UPLOAD: bool = True
    SEARCH_DEFAULT_TOP_K: int = 5
    SEARCH_MAX_TOP_K: int = 50

    # ---- Phase 4: hybrid retrieval, reranking, RAG answers ----
    HYBRID_CANDIDATES: int = 20  # results taken from EACH retriever (semantic, BM25) before fusion
    RRF_K: int = 60  # Reciprocal Rank Fusion constant (standard value from the RRF paper)
    RERANK_CANDIDATES: int = 12  # fused results passed to the cross-encoder
    # "cross-encoder" (real model) or "none" (keep fused order; tests/offline)
    RERANKER_PROVIDER: str = "cross-encoder"
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # ~90 MB
    # Cross-encoder scores are logits: > 0 relevant, very negative = unrelated. Chunks below
    # this are not given to the LLM; if none pass, /ask answers "not found" without the LLM.
    RERANK_MIN_SCORE: float = -6.0
    # Fallback when NO chunk passes RERANK_MIN_SCORE: the cross-encoder under-scores some
    # indirect wording ("Which filter should I buy?" -9.6 vs part-number text). If the best chunk
    # was found by BOTH keyword (BM25) and meaning search, with a semantic similarity of at least
    # this value, it is still given to the LLM, and the answer carries a low-confidence warning.
    # Measured on the sample PDFs: off-topic questions never had a keyword match and stayed
    # <= 0.12 similarity; the rescued questions were 0.31-0.40. Set above 1 to disable.
    RERANK_FALLBACK_MIN_SEMANTIC: float = 0.25
    # A fallback answer is kept only if one of its sentences has at least this share of its words
    # in the source passage (stops the LLM inventing an answer from a weakly related page).
    LOW_CONFIDENCE_MIN_SUPPORT: float = 0.6
    RAG_CONTEXT_CHUNKS: int = 5  # chunks given to the LLM as numbered sources
    RAG_MAX_CONTEXT_CHARS: int = 6000  # cap on total source text in the prompt
    # Whole-document requests ("Summarize this document", "main topics") cannot be answered by
    # similarity search (the question has no topic words), so they get excerpts spread evenly
    # across the document(s) instead. ~8000 chars fits comfortably in OLLAMA_NUM_CTX=4096.
    RAG_SUMMARY_CHUNKS: int = 8
    RAG_SUMMARY_MAX_CONTEXT_CHARS: int = 8000
    # LLM: "transformers" (in-process HF model), "ollama" (local Ollama server),
    # or "extractive" (no LLM: returns the best passages with citations)
    LLM_PROVIDER: str = "transformers"
    LLM_MODEL: str = "Qwen/Qwen2.5-0.5B-Instruct"  # ~1 GB; used by "transformers"
    OLLAMA_URL: str = "http://127.0.0.1:11434"
    OLLAMA_MODEL: str = "llama3.2:3b"
    OLLAMA_NUM_CTX: int = 4096  # context window (tokens); must fit system prompt + sources
    LLM_MAX_NEW_TOKENS: int = 300
    LLM_TEMPERATURE: float = 0.1  # low = factual, sticks to the sources
    LLM_TIMEOUT_SECONDS: int = 180

    # ---- Phase 6: users, login (JWT) and chat history ----
    # PostgreSQL example: DATABASE_URL="postgresql+psycopg://intellidoc:PASSWORD@127.0.0.1:5432/intellidoc"
    # SECRET_KEY signs login tokens. The default is for local development only: set your own
    # long random value in .env (ENVIRONMENT=production refuses to start with the default).
    SECRET_KEY: str = DEV_SECRET_KEY
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 1 day
    ALLOW_REGISTRATION: bool = True
    PASSWORD_MIN_LENGTH: int = 8
    # The first account created adopts documents uploaded before accounts existed (Phase 2-5),
    # so nothing already uploaded becomes invisible.
    ADOPT_UNOWNED_DOCUMENTS: bool = True

    @model_validator(mode="after")
    def secret_key_set_in_production(self) -> "Settings":
        if self.ENVIRONMENT.lower() == "production" and self.SECRET_KEY == DEV_SECRET_KEY:
            raise ValueError("Set SECRET_KEY to a long random value when ENVIRONMENT=production")
        return self

    @field_validator("CORS_ORIGINS", "ALLOWED_EXTENSIONS", mode="before")
    @classmethod
    def split_comma_separated(cls, value: object) -> object:
        """Allow list settings as comma-separated strings in .env."""
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @field_validator("ALLOWED_EXTENSIONS")
    @classmethod
    def normalize_extensions(cls, value: list[str]) -> list[str]:
        return [ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in value]

    @property
    def max_upload_bytes(self) -> int:
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    @property
    def upload_path(self) -> Path:
        path = Path(self.UPLOAD_DIR)
        return (path if path.is_absolute() else BASE_DIR / path).resolve()

    @field_validator("CHUNK_OVERLAP")
    @classmethod
    def overlap_smaller_than_chunk(cls, value: int, info) -> int:
        chunk_size = info.data.get("CHUNK_SIZE", 1000)
        if value < 0 or value >= chunk_size:
            raise ValueError("CHUNK_OVERLAP must be >= 0 and smaller than CHUNK_SIZE")
        return value

    @staticmethod
    def _anchored(value: str) -> Path:
        path = Path(value)
        return (path if path.is_absolute() else BASE_DIR / path).resolve()

    @property
    def embedding_cache_path(self) -> Path:
        return self._anchored(self.EMBEDDING_CACHE_DIR)

    @property
    def chroma_path(self) -> Path:
        return self._anchored(self.CHROMA_DIR)

    @property
    def database_url(self) -> str:
        """DATABASE_URL with a relative SQLite file path anchored to backend/."""
        prefix = "sqlite:///"
        if self.DATABASE_URL.startswith(prefix) and self.DATABASE_URL != "sqlite://":
            file_part = self.DATABASE_URL[len(prefix):]
            if file_part and file_part != ":memory:" and not Path(file_part).is_absolute():
                return f"{prefix}{(BASE_DIR / file_part).resolve().as_posix()}"
        return self.DATABASE_URL


@lru_cache
def get_settings() -> Settings:
    return Settings()
