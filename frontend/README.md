# IntelliDoc AI: frontend

React 19 + Vite + Tailwind CSS 4 single-page app for IntelliDoc AI.

## Features
- Log in / create an account (Phase 6). Each user sees only their own documents and chats; the login token is kept in the browser, so a refresh stays logged in, and an expired session returns you to the login screen
- Chat history: the **Chats** tab lists saved conversations; open one to see its answers and citations, continue it, rename or delete it, or start a **New chat**
- Upload PDF / DOCX / TXT by drag-and-drop or the file picker (checked for type and the 10 MB limit before upload)
- Document list showing page count, size, index status, delete and re-index buttons
- Tick documents to limit questions to them
- AI chat: answers with clickable `[n]` citations, each showing the file, page number and exact passage
- Loading indicators (upload processing, a "thinking" timer with a cold-start hint), clear error messages with Retry, and a backend health dot

## Run (development)
The FastAPI backend must be running on port 8000 first:

```bash
cd backend
..\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

Then, in a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. Vite forwards every `/api/...` request to the backend, so no CORS setup is needed.
To use a backend somewhere else, set `VITE_BACKEND_URL`, e.g. `VITE_BACKEND_URL=http://192.168.1.10:8000`.

## Test and build
```bash
npm test         # Vitest + Testing Library (backend calls are mocked)
npm run e2e      # Playwright: real browser (Edge) + real backend + real search models, throwaway data
npm run build    # production files in dist/
```

Everything at once (backend lint + tests on SQLite and PostgreSQL, frontend tests, build, browser tests),
from the project root:

```bash
powershell -ExecutionPolicy Bypass -File scripts\run-all-tests.ps1
```

Add `-WithModels` for the real-model backend tests, `-WithOllama` to run the browser tests with Ollama answers,
or `-SkipE2E` for a quick run. The browser tests start their own backend (port 8011) and frontend (port 5181)
with a fresh database and search index in `backend/.e2e-run/`, so your real documents are never touched.
Fixture PDFs are in `e2e/fixtures/` (regenerate with `backend/scripts/make_e2e_fixtures.py`).

## Layout
```
src/
  App.jsx                  page layout, document state, notifications
  lib/api.js               fetch wrapper for /api/v1 with friendly error messages
  lib/citations.js         splits "[1]" / "[2, 3]" markers out of answers
  components/UploadZone.jsx
  components/DocumentList.jsx
  components/ChatPanel.jsx      chat, input, loading indicator
  components/AnswerMessage.jsx  answer, citation buttons, source passages
  test/                    tests
```
