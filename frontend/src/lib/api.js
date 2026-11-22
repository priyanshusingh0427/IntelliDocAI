// Small client for the IntelliDoc FastAPI backend (all routes live under /api/v1).
// In development Vite proxies /api to http://127.0.0.1:8000 (see vite.config.js).
export const API_BASE = import.meta.env.VITE_API_BASE || '/api/v1'

/** An error the UI can show directly: a readable message plus the HTTP status (0 = no response). */
export class ApiError extends Error {
  constructor(message, status = 0, body = null) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.body = body
  }
}

/** Turn FastAPI error bodies ({detail: "..."} or a list of validation errors) into one sentence. */
export function errorMessage(body, status) {
  const detail = body?.detail
  if (typeof detail === 'string' && detail) return detail
  if (Array.isArray(detail) && detail.length) {
    return detail
      .map((d) => (d.loc ? `${d.loc.filter((p) => p !== 'body').join('.')}: ${d.msg}` : d.msg))
      .join('; ')
  }
  if (status === 503) return 'A required service (AI model or vector database) is unavailable.'
  if (status >= 500) return `Server error (${status}). Check the backend log.`
  return `Request failed (${status}).`
}

// ---- Login token (Phase 6). Kept in localStorage so a page refresh stays logged in.
const TOKEN_KEY = 'intellidoc.token'
let token = null
try {
  token = localStorage.getItem(TOKEN_KEY)
} catch {
  /* storage blocked: stay logged out */
}
let onUnauthorized = () => {}

export function getToken() {
  return token
}

export function setToken(value) {
  token = value || null
  try {
    if (token) localStorage.setItem(TOKEN_KEY, token)
    else localStorage.removeItem(TOKEN_KEY)
  } catch {
    /* ignore */
  }
}

/** Called when the backend says the token is missing/expired (HTTP 401 on a logged-in call). */
export function setUnauthorizedHandler(handler) {
  onUnauthorized = handler || (() => {})
}

async function request(path, options = {}) {
  const headers = { ...(options.headers || {}) }
  const sentToken = token
  if (sentToken) headers.Authorization = `Bearer ${sentToken}`
  let response
  try {
    response = await fetch(`${API_BASE}${path}`, { ...options, headers })
  } catch {
    throw new ApiError(
      'Cannot reach the backend. Is the FastAPI server running on port 8000?',
      0,
    )
  }
  if (response.status === 204) return null
  const text = await response.text()
  let body = null
  if (text) {
    try {
      body = JSON.parse(text)
    } catch {
      body = { detail: text.slice(0, 300) }
    }
  }
  if (response.status === 401 && sentToken) onUnauthorized()
  if (!response.ok) throw new ApiError(errorMessage(body, response.status), response.status, body)
  return body
}

const json = (method, data) => ({
  method,
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(data),
})

export const api = {
  health: () => request('/health'),
  /** GET /health/ready: returns the component report for both 200 (ready) and 503 (degraded). */
  ready: async () => {
    try {
      return await request('/health/ready')
    } catch (err) {
      if (err.status === 503 && err.body?.components) return err.body
      throw err
    }
  },
  register: (email, password, fullName) =>
    request('/auth/register', json('POST', { email, password, full_name: fullName || null })),
  login: (email, password) =>
    request('/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: new URLSearchParams({ username: email, password }).toString(),
    }),
  me: () => request('/auth/me'),
  listChats: () => request('/chats'),
  getChat: (id) => request(`/chats/${id}`),
  renameChat: (id, title) => request(`/chats/${id}`, json('PATCH', { title })),
  deleteChat: (id) => request(`/chats/${id}`, { method: 'DELETE' }),
  listDocuments: () => request('/documents'),
  uploadDocument: (file) => {
    const form = new FormData()
    form.append('file', file)
    return request('/documents', { method: 'POST', body: form })
  },
  deleteDocument: (id) => request(`/documents/${id}`, { method: 'DELETE' }),
  reindexDocument: (id) => request(`/documents/${id}/index`, { method: 'POST' }),
  ask: (question, documentIds, chatId = null) =>
    request(
      '/ask',
      json('POST', {
        question,
        document_ids: documentIds?.length ? documentIds : null,
        chat_id: chatId,
      }),
    ),
}

export function formatBytes(bytes) {
  if (bytes == null) return ''
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}
