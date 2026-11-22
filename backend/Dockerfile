# IntelliDoc AI backend (FastAPI + sentence-transformers + ChromaDB).
# Built by docker-compose.yml from the project root. CPU-only PyTorch keeps the image ~2 GB smaller.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    ANONYMIZED_TELEMETRY=False

WORKDIR /app

# Same versions as the development venv; torch from the CPU-only index.
RUN pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt transformers==4.46.3 tokenizers==0.20.3

# Run as an unprivileged user; /data holds uploads + the ChromaDB index (Docker volumes).
RUN useradd --create-home --uid 10001 app \
    && mkdir -p /data/uploads /data/chroma /app/models \
    && chown -R app:app /data /app
USER app

ENV EMBEDDING_CACHE_DIR=/app/models \
    UPLOAD_DIR=/data/uploads \
    CHROMA_DIR=/data/chroma \
    ENVIRONMENT=production

# Bake the embedding model + cross-encoder into the image (~180 MB): no download at runtime.
COPY --chown=app:app app ./app
COPY --chown=app:app scripts/download_models.py ./scripts/download_models.py
RUN ENVIRONMENT=build python scripts/download_models.py
COPY --chown=app:app main.py ./

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=4)"

# One worker on purpose: the models are loaded once per process, and ChromaDB's local
# storage must only be written by one process.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", \
     "--proxy-headers", "--forwarded-allow-ips", "*"]
