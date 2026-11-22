import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import App from '../App'
import { getToken, setToken } from '../lib/api'

const DOC = {
  id: 2,
  original_filename: 'X200_Manual.pdf',
  content_type: 'application/pdf',
  extension: '.pdf',
  size_bytes: 4096,
  status: 'processed',
  error_message: null,
  char_count: 1200,
  page_count: 3,
  created_at: '2026-11-22T10:00:00',
  index_status: 'indexed',
  index_error: null,
  chunk_count: 6,
}

const CITATION = {
  source: 1,
  document_id: 2,
  filename: 'X200_Manual.pdf',
  page_number: 2,
  chunk_id: 'doc2-chunk3',
  rerank_score: 5.1,
  text: 'To reset the X200, hold the red RESET button for ten seconds.',
}

const ANSWER = {
  question: 'How do I reset it?',
  answer: 'Hold the red RESET button for ten seconds [1].',
  found: true,
  grounded: true,
  citations: [CITATION],
  sources: [CITATION],
  llm_model: 'ollama/llama3.2:3b',
  reranker_model: 'cross-encoder',
  warnings: [],
  timings_ms: { semantic: 20, bm25: 5, rerank: 75, llm: 900 },
  chat_id: 10,
  message_id: 2,
}

const json = (body, status = 200) => new Response(JSON.stringify(body), { status })

const USER = { id: 1, email: 'priya@example.com', full_name: 'Priya', created_at: '2026-11-22T10:00:00' }

/** Route fetch calls by method + path; each handler returns a Response (or throws).
 *  Logged-in basics (/auth/me, /chats, /health/ready) have defaults that `routes` can override. */
function mockBackend(routes) {
  const all = {
    'GET /health/ready': () => json({ status: 'ready', components: { database: { ok: true, detail: 'ok' }, vector_store: { ok: true, detail: 'ok' }, llm: { ok: true, detail: 'ok' } } }),
    'GET /auth/me': () => json(USER),
    'GET /chats': () => json([]),
    ...routes,
  }
  const fetchMock = vi.fn(async (url, options = {}) => {
    const key = `${options.method || 'GET'} ${url.replace('/api/v1', '')}`
    const handler = all[key]
    if (!handler) throw new Error(`Unexpected request: ${key}`)
    return handler(options)
  })
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}


beforeEach(() => {
  Element.prototype.scrollIntoView = vi.fn()
  setToken('test-token') // most tests start logged in
})
afterEach(() => {
  vi.unstubAllGlobals()
  setToken(null)
})

describe('App', () => {
  it('lists documents and shows backend status', async () => {
    mockBackend({ 'GET /documents': () => json([DOC]) })
    render(<App />)
    expect(await screen.findByText('X200_Manual.pdf')).toBeInTheDocument()
    expect(screen.getByText('Ready')).toBeInTheDocument()
    expect(screen.getByText('3 pages')).toBeInTheDocument()
    expect(await screen.findByText('All systems ready')).toBeInTheDocument()
  })

  it('shows an error with retry when the backend is down', async () => {
    mockBackend({
      'GET /health/ready': () => Promise.reject(new TypeError('Failed to fetch')),
      'GET /documents': () => Promise.reject(new TypeError('Failed to fetch')),
    })
    render(<App />)
    expect(await screen.findByText(/Cannot reach the backend/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Try again' })).toBeInTheDocument()
    expect(await screen.findByText('Backend offline')).toBeInTheDocument()
  })

  it('asks a question and renders the answer with a clickable citation', async () => {
    let resolveAsk
    const fetchMock = mockBackend({
      'GET /documents': () => json([DOC]),
      'POST /ask': () => new Promise((r) => (resolveAsk = () => r(json(ANSWER)))),
    })
    const user = userEvent.setup()
    render(<App />)
    await screen.findByText('X200_Manual.pdf')

    await user.type(screen.getByLabelText('Question'), 'How do I reset it?{Enter}')
    // loading indicator while the backend works
    expect(await screen.findByText(/Searching your documents/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Thinking…' })).toBeDisabled()
    resolveAsk()

    expect(await screen.findByText(/Hold the red RESET button/)).toBeInTheDocument()
    expect(screen.queryByText(/Searching your documents/)).not.toBeInTheDocument()
    expect(screen.getByText('Page 2')).toBeInTheDocument()
    expect(screen.getByText(/ollama\/llama3.2:3b · 1.0s/)).toBeInTheDocument()

    // clicking the [1] marker reveals the cited passage
    expect(screen.queryByText(/hold the red RESET button for ten seconds\.$/)).not.toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Source 1: X200_Manual.pdf, page 2' }))
    expect(screen.getByText(CITATION.text)).toBeInTheDocument()

    const askCall = fetchMock.mock.calls.find(([, o]) => o?.method === 'POST')
    expect(JSON.parse(askCall[1].body)).toEqual({ question: 'How do I reset it?', document_ids: null, chat_id: null })
  })

  it('limits the question to ticked documents', async () => {
    const fetchMock = mockBackend({
      'GET /documents': () => json([DOC]),
      'POST /ask': () => json(ANSWER),
    })
    const user = userEvent.setup()
    render(<App />)
    await user.click(await screen.findByLabelText('Ask about X200_Manual.pdf only'))
    expect(screen.getByText('Asking about: X200_Manual.pdf')).toBeInTheDocument()
    await user.type(screen.getByLabelText('Question'), 'Reset?{Enter}')
    await screen.findByText(/Hold the red RESET button/)
    const askCall = fetchMock.mock.calls.find(([, o]) => o?.method === 'POST')
    expect(JSON.parse(askCall[1].body).document_ids).toEqual([2])
  })

  it('shows "not found" answers distinctly', async () => {
    mockBackend({
      'GET /documents': () => json([DOC]),
      'POST /ask': () =>
        json({ ...ANSWER, found: false, grounded: false, citations: [], answer: 'I could not find this in the documents.' }),
    })
    const user = userEvent.setup()
    render(<App />)
    await screen.findByText('X200_Manual.pdf')
    await user.type(screen.getByLabelText('Question'), 'Capital of France?{Enter}')
    expect(await screen.findByText('Not found in your documents')).toBeInTheDocument()
  })

  it('shows an error message with retry when /ask fails', async () => {
    let calls = 0
    mockBackend({
      'GET /documents': () => json([DOC]),
      'POST /ask': () => (++calls === 1 ? json({ detail: 'Ollama is not reachable' }, 503) : json(ANSWER)),
    })
    const user = userEvent.setup()
    render(<App />)
    await screen.findByText('X200_Manual.pdf')
    await user.type(screen.getByLabelText('Question'), 'Reset?{Enter}')
    const alert = await screen.findByRole('alert')
    expect(within(alert).getByText('Ollama is not reachable')).toBeInTheDocument()
    await user.click(within(alert).getByRole('button', { name: 'Retry' }))
    expect(await screen.findByText(/Hold the red RESET button/)).toBeInTheDocument()
  })

  it('uploads a PDF and adds it to the list', async () => {
    const newDoc = { ...DOC, id: 3, original_filename: 'Report.pdf', page_count: 5 }
    const fetchMock = mockBackend({
      'GET /documents': () => json([]),
      'POST /documents': () => json(newDoc, 201),
    })
    const user = userEvent.setup()
    render(<App />)
    expect(await screen.findByText(/No documents yet/)).toBeInTheDocument()
    expect(screen.getByLabelText('Question')).toBeDisabled()

    await user.upload(screen.getByTestId('file-input'), new File(['%PDF-1.7'], 'Report.pdf', { type: 'application/pdf' }))
    expect(await screen.findByText('Report.pdf')).toBeInTheDocument()
    expect(screen.getByText(/Report.pdf is ready \(5 pages\)/)).toBeInTheDocument()
    expect(screen.getByLabelText('Question')).toBeEnabled()
    const uploadCall = fetchMock.mock.calls.find(([, o]) => o?.method === 'POST')
    expect(uploadCall[1].body).toBeInstanceOf(FormData)
    expect(uploadCall[1].body.get('file').name).toBe('Report.pdf')
  })

  it('shows the backend error when an upload is rejected', async () => {
    mockBackend({
      'GET /documents': () => json([]),
      'POST /documents': () => json({ detail: 'The PDF is password-protected (encrypted).' }, 422),
    })
    const user = userEvent.setup()
    render(<App />)
    await screen.findByText(/No documents yet/)
    await user.upload(screen.getByTestId('file-input'), new File(['%PDF'], 'Secret.pdf', { type: 'application/pdf' }))
    expect(await screen.findByText('Secret.pdf: The PDF is password-protected (encrypted).')).toBeInTheDocument()
  })

  it('rejects unsupported files before uploading', async () => {
    const fetchMock = mockBackend({ 'GET /documents': () => json([]) })
    const user = userEvent.setup({ applyAccept: false })
    render(<App />)
    await screen.findByText(/No documents yet/)
    await user.upload(screen.getByTestId('file-input'), new File(['MZ'], 'virus.exe'))
    expect(await screen.findByText(/only PDF, DOCX and TXT/)).toBeInTheDocument()
    expect(fetchMock.mock.calls.some(([, o]) => o?.method === 'POST')).toBe(false)
  })

  it('deletes a document after confirmation', async () => {
    mockBackend({
      'GET /documents': () => json([DOC]),
      'DELETE /documents/2': () => new Response(null, { status: 204 }),
    })
    vi.stubGlobal('confirm', vi.fn(() => true))
    const user = userEvent.setup()
    render(<App />)
    await user.click(await screen.findByRole('button', { name: 'Delete X200_Manual.pdf' }))
    await waitFor(() => expect(screen.queryByText('X200_Manual.pdf')).not.toBeInTheDocument())
    expect(screen.getByText(/No documents yet/)).toBeInTheDocument()
  })
})
