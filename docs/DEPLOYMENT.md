# Deployment

Two ways to run IntelliDoc AI: **Docker Compose** (any Linux server, macOS, or Windows with Docker Desktop) or **directly on Windows** (how it is developed).

## Docker Compose

### Requirements

| | Minimum | Recommended |
|---|---|---|
| RAM | 3 GB with `LLM_PROVIDER=extractive` | 8 GB for `llama3.2:3b` |
| Disk | 6 GB (images + model) | 10 GB+ for documents |
| CPU | 2 cores | 4+ cores, or an NVIDIA GPU for Ollama |

### Start

```bash
cp .env.docker.example .env
python -c "import secrets; print(secrets.token_hex(16))"   # paste as POSTGRES_PASSWORD
python -c "import secrets; print(secrets.token_hex(32))"   # paste as SECRET_KEY
docker compose up -d --build
docker compose ps                      # all services "healthy"; ollama-pull "exited (0)"
python backend/scripts/smoke_test.py http://localhost:8080
```

| Service | Image | Role | Published |
|---|---|---|---|
| `frontend` | built from `frontend/Dockerfile` (nginx 1.30) | serves the React build; proxies `/api` and `/docs` | `PUBLIC_PORT` (8080) |
| `backend` | built from `backend/Dockerfile` (Python 3.11, CPU PyTorch, models baked in) | FastAPI, one worker | no |
| `postgres` | `postgres:17-alpine` | users, documents, pages, chats | no |
| `ollama` | `ollama/ollama:0.40.16` | local LLM | no |
| `ollama-pull` | same | downloads `OLLAMA_MODEL` once, then exits | no |

Data lives in named volumes: `pgdata`, `uploads`, `chroma`, `ollama`. `docker compose down` keeps them; `docker compose down -v` **deletes everything**.

### Settings (`.env` next to `docker-compose.yml`)

| Variable | Default | Meaning |
|---|---|---|
| `POSTGRES_PASSWORD` | required | database password (letters and digits) |
| `SECRET_KEY` | required | signs login tokens; changing it logs everyone out |
| `LLM_PROVIDER` | `ollama` | `ollama` or `extractive` (no LLM) |
| `OLLAMA_MODEL` | `llama3.2:3b` | any Ollama model, e.g. `qwen2.5:7b` on bigger machines |
| `PUBLIC_PORT` / `PUBLIC_URL` | `8080` / `http://localhost:8080` | where people open the app |
| `ALLOW_REGISTRATION` | `true` | set `false` after creating your accounts |
| `LOG_LEVEL` | `INFO` | `DEBUG` for more detail in `docker compose logs backend` |

Every backend setting from [backend/.env.example](../backend/.env.example) can also be added under `backend.environment` in `docker-compose.yml`, for example `RERANK_MIN_SCORE` or `CHUNK_SIZE`.

### GPU

With the NVIDIA Container Toolkit installed:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

### HTTPS on a public server

Put a TLS-terminating reverse proxy in front of port 8080, for example Caddy:

```
docs.example.com {
    reverse_proxy localhost:8080
}
```

Then set `PUBLIC_URL=https://docs.example.com`, and `ALLOW_REGISTRATION=false` once your accounts exist.

### Backups

```bash
docker compose exec postgres pg_dump -U intellidoc intellidoc > intellidoc-$(date +%F).sql
docker run --rm -v intellidoc_uploads:/data -v "$PWD":/backup alpine tar czf /backup/uploads.tgz -C /data .
docker run --rm -v intellidoc_chroma:/data -v "$PWD":/backup alpine tar czf /backup/chroma.tgz -C /data .
```

The ChromaDB index can always be rebuilt from the stored documents (`POST /api/v1/documents/{id}/index`), so the database and uploads are the essential backups.

### Update

```bash
git pull
docker compose up -d --build
```

Database tables and columns are upgraded automatically on startup, and nothing is dropped.

### Troubleshooting

| Symptom | Check |
|---|---|
| Header shows "AI model offline" | `docker compose logs ollama-pull` (model still downloading?) and `docker compose logs ollama` |
| First answer takes a minute | normal: Ollama loads the model into memory; later answers take seconds |
| `502 Bad Gateway` | backend still starting (`docker compose ps`), or `docker compose logs backend` |
| Uploads fail with 413 | file over 10 MB (`MAX_UPLOAD_SIZE_MB`; also raise `client_max_body_size` in `frontend/nginx/default.conf`) |
| Readiness at a glance | `curl http://localhost:8080/api/v1/health/ready` |

## Directly on Windows (development setup)

| Piece | How |
|---|---|
| Python | `.venv` with `backend/requirements-dev.txt` |
| PostgreSQL | portable PostgreSQL 17 in `%LOCALAPPDATA%\Programs\PostgreSQL17`, data in `%LOCALAPPDATA%\IntelliDocAI\pgdata`; start with `scripts\start-postgres.ps1`, stop with `scripts\stop-postgres.ps1` |
| Ollama | Windows app (`winget install Ollama.Ollama`), then `ollama pull llama3.2:3b` |
| Settings | `backend/.env` (`DATABASE_URL`, `SECRET_KEY`, `LLM_PROVIDER`; see `.env.example`) |
| Backend | `cd backend; ..\.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000` |
| Frontend | `cd frontend; npm run dev` (Vite proxies `/api` to port 8000) |

Moving existing SQLite data into PostgreSQL:

```bash
cd backend
..\.venv\Scripts\python.exe scripts\migrate_sqlite_to_postgres.py --target "postgresql+psycopg://intellidoc:PASSWORD@127.0.0.1:5432/intellidoc"
```

## Continuous integration

`.github/workflows/ci.yml` runs on every push and pull request:

1. **backend**: ruff, then pytest on SQLite (with coverage) and on a PostgreSQL 17 service
2. **frontend**: Vitest with coverage, then the production build
3. **e2e**: Playwright in Chromium against the real backend and search models (cached)
4. **docker**: `docker compose build`, starts PostgreSQL + backend + frontend with `LLM_PROVIDER=extractive`, and runs `smoke_test.py` through nginx. Container logs are printed on failure.
