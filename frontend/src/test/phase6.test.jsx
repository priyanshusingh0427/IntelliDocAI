import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import App from '../App'
import { getToken, setToken } from '../lib/api'

const USER = { id: 1, email: 'priya@example.com', full_name: 'Priya', created_at: '2026-11-22T10:00:00' }
const DOC = {
  id: 2, original_filename: 'X200_Manual.pdf', extension: '.pdf', size_bytes: 4096, status: 'processed',
  char_count: 1200, page_count: 3, created_at: '2026-11-22T10:00:00', index_status: 'indexed', chunk_count: 6,
}
const CITATION = {
  source: 1, document_id: 2, filename: 'X200_Manual.pdf', page_number: 3, chunk_id: 'doc2-chunk5',
  rerank_score: 4.2, text: 'Water damage caused by incorrect installation is not covered.',
}
const CHATS = [
  { id: 7, title: 'Warranty questions', created_at: '2026-11-22T10:00:00Z', updated_at: new Date().toISOString(), message_count: 2 },
  { id: 6, title: 'Older chat', created_at: '2026-11-20T10:00:00Z', updated_at: '2026-11-20T10:00:00Z', message_count: 4 },
]
const CHAT_7 = {
  ...CHATS[0],
  messages: [
    { id: 1, role: 'user', content: 'What does the warranty not cover?', created_at: '2026-11-22T10:00:00Z' },
    {
      id: 2, role: 'assistant', content: 'Water damage caused by incorrect installation is not covered [1].',
      created_at: '2026-11-22T10:00:01Z', found: true, grounded: true, citations: [CITATION], warnings: [],
      llm_model: 'ollama/llama3.2:3b', timings_ms: { llm: 1200 },
    },
  ],
}
const answer = (text, chatId) => ({
  question: 'q', answer: text, found: true, grounded: true, citations: [CITATION], sources: [CITATION],
  llm_model: 'ollama/llama3.2:3b', reranker_model: 'x', warnings: [], timings_ms: { llm: 800 },
  chat_id: chatId, message_id: 99,
})

const json = (body, status = 200) => new Response(JSON.stringify(body), { status })

function mockBackend(routes) {
  const all = {
    'GET /health/ready': () => json({ status: 'ready', components: { database: { ok: true, detail: 'ok' }, vector_store: { ok: true, detail: 'ok' }, llm: { ok: true, detail: 'ok' } } }),
    'GET /auth/me': () => json(USER),
    'GET /documents': () => json([DOC]),
    'GET /chats': () => json([]),
    ...routes,
  }
  const fetchMock = vi.fn(async (url, options = {}) => {
    const key = `${options.method || 'GET'} ${url.replace('/api/v1', '')}`
    if (!all[key]) throw new Error(`Unexpected request: ${key}`)
    return all[key](options)
  })
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

const callsTo = (fetchMock, method, path) =>
  fetchMock.mock.calls.filter(([url, o]) => (o?.method || 'GET') === method && url.endsWith(path))

beforeEach(() => {
  Element.prototype.scrollIntoView = vi.fn()
})
afterEach(() => {
  vi.unstubAllGlobals()
  setToken(null)
})

describe('login', () => {
  it('shows the login screen when not logged in', async () => {
    const fetchMock = mockBackend({})
    render(<App />)
    expect(screen.getByRole('button', { name: 'Log in' })).toBeInTheDocument()
    expect(screen.queryByText('X200_Manual.pdf')).not.toBeInTheDocument()
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('logs in with email + password and opens the workspace', async () => {
    const fetchMock = mockBackend({
      'POST /auth/login': () => json({ access_token: 'tok-123', token_type: 'bearer', expires_in: 86400, user: USER }),
    })
    const user = userEvent.setup()
    render(<App />)
    await user.type(screen.getByLabelText('Email'), 'priya@example.com')
    await user.type(screen.getByLabelText('Password'), 'secret-pass-1')
    await user.click(screen.getByRole('button', { name: 'Log in' }))

    expect(await screen.findByText('X200_Manual.pdf')).toBeInTheDocument()
    expect(screen.getByText('Priya')).toBeInTheDocument()
    expect(getToken()).toBe('tok-123')
    const [, loginOptions] = callsTo(fetchMock, 'POST', '/auth/login')[0]
    expect(new URLSearchParams(loginOptions.body).get('username')).toBe('priya@example.com')
    // later calls carry the token
    const [, docsOptions] = callsTo(fetchMock, 'GET', '/documents')[0]
    expect(docsOptions.headers.Authorization).toBe('Bearer tok-123')
  })

  it('shows the backend error for a wrong password', async () => {
    mockBackend({ 'POST /auth/login': () => json({ detail: 'Incorrect email or password' }, 401) })
    const user = userEvent.setup()
    render(<App />)
    await user.type(screen.getByLabelText('Email'), 'priya@example.com')
    await user.type(screen.getByLabelText('Password'), 'nope-nope')
    await user.click(screen.getByRole('button', { name: 'Log in' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('Incorrect email or password')
    expect(getToken()).toBeNull()
  })

  it('creates an account, checking the password length first', async () => {
    const fetchMock = mockBackend({
      'POST /auth/register': () =>
        json({ access_token: 'tok-new', token_type: 'bearer', expires_in: 86400, user: USER, adopted_documents: 2 }, 201),
    })
    const user = userEvent.setup()
    render(<App />)
    await user.click(screen.getByRole('tab', { name: 'Create account' }))
    await user.type(screen.getByLabelText(/Name/), 'Priya')
    await user.type(screen.getByLabelText('Email'), 'priya@example.com')
    await user.type(screen.getByLabelText(/^Password/), 'short')
    await user.click(screen.getByRole('button', { name: 'Create account' }))
    expect(screen.getByRole('alert')).toHaveTextContent('at least 8 characters')
    expect(callsTo(fetchMock, 'POST', '/auth/register')).toHaveLength(0)

    await user.type(screen.getByLabelText(/^Password/), '-and-longer')
    await user.click(screen.getByRole('button', { name: 'Create account' }))
    expect(await screen.findByText(/2 document\(s\) uploaded earlier now belong to your account/)).toBeInTheDocument()
    const body = JSON.parse(callsTo(fetchMock, 'POST', '/auth/register')[0][1].body)
    expect(body).toEqual({ email: 'priya@example.com', password: 'short-and-longer', full_name: 'Priya' })
  })

  it('keeps you logged in after a refresh while the token is valid', async () => {
    setToken('saved-token')
    mockBackend({})
    render(<App />)
    expect(await screen.findByText('X200_Manual.pdf')).toBeInTheDocument()
  })

  it('drops an invalid saved token and shows the login screen', async () => {
    setToken('stale-token')
    mockBackend({ 'GET /auth/me': () => json({ detail: 'Not logged in' }, 401) })
    render(<App />)
    expect(await screen.findByRole('button', { name: 'Log in' })).toBeInTheDocument()
    expect(getToken()).toBeNull()
  })

  it('logs out when a request says the session expired', async () => {
    setToken('tok')
    mockBackend({ 'POST /ask': () => json({ detail: 'Not logged in or the session expired.' }, 401) })
    const user = userEvent.setup()
    render(<App />)
    await screen.findByText('X200_Manual.pdf')
    await user.type(screen.getByLabelText('Question'), 'Reset?{Enter}')
    expect(await screen.findByText('Your session expired. Please log in again.')).toBeInTheDocument()
    expect(getToken()).toBeNull()
  })

  it('log out button returns to the login screen', async () => {
    setToken('tok')
    mockBackend({})
    const user = userEvent.setup()
    render(<App />)
    await user.click(await screen.findByRole('button', { name: 'Log out' }))
    expect(screen.getByRole('button', { name: 'Log in' })).toBeInTheDocument()
    expect(getToken()).toBeNull()
  })
})

describe('chat history', () => {
  beforeEach(() => setToken('tok'))

  it('lists chats and opens a saved conversation with its citations', async () => {
    mockBackend({ 'GET /chats': () => json(CHATS), 'GET /chats/7': () => json(CHAT_7) })
    const user = userEvent.setup()
    render(<App />)
    await user.click(await screen.findByRole('tab', { name: 'Chats (2)' }))
    const list = screen.getByRole('list', { name: 'Chat history' })
    expect(within(list).getByText('Warranty questions')).toBeInTheDocument()
    expect(within(list).getByText(/just now · 1 question$/)).toBeInTheDocument()

    await user.click(within(list).getByText('Warranty questions'))
    expect(await screen.findByText('What does the warranty not cover?')).toBeInTheDocument()
    expect(screen.getByText(/Water damage caused by incorrect installation is not covered/)).toBeInTheDocument()
    expect(screen.getByText('Page 3')).toBeInTheDocument()
    expect(screen.getByText(/ollama\/llama3.2:3b · 1.2s/)).toBeInTheDocument()
  })

  it('continues an opened chat by sending its chat_id', async () => {
    const fetchMock = mockBackend({
      'GET /chats': () => json(CHATS),
      'GET /chats/7': () => json(CHAT_7),
      'POST /ask': () => json(answer('It is three years [1].', 7)),
    })
    const user = userEvent.setup()
    render(<App />)
    await user.click(await screen.findByRole('tab', { name: 'Chats (2)' }))
    await user.click(screen.getByText('Warranty questions'))
    await screen.findByText('What does the warranty not cover?')
    await user.type(screen.getByLabelText('Question'), 'How long is it?{Enter}')
    expect(await screen.findByText(/It is three years/)).toBeInTheDocument()
    // the earlier messages stay on screen
    expect(screen.getByText('What does the warranty not cover?')).toBeInTheDocument()
    expect(JSON.parse(callsTo(fetchMock, 'POST', '/ask')[0][1].body).chat_id).toBe(7)
    expect(callsTo(fetchMock, 'GET', '/chats/7')).toHaveLength(1) // not reloaded
  })

  it('a first question starts a new chat that appears in the list', async () => {
    let chats = []
    const fetchMock = mockBackend({
      'GET /chats': () => json(chats),
      'POST /ask': () => {
        chats = [{ id: 11, title: 'How do I reset it?', created_at: '2026-11-22T10:00:00Z', updated_at: new Date().toISOString(), message_count: 2 }]
        return json(answer('Hold RESET for ten seconds [1].', 11))
      },
    })
    const user = userEvent.setup()
    render(<App />)
    await screen.findByText('X200_Manual.pdf')
    await user.type(screen.getByLabelText('Question'), 'How do I reset it?{Enter}')
    expect(await screen.findByText(/Hold RESET for ten seconds/)).toBeInTheDocument()
    expect(JSON.parse(callsTo(fetchMock, 'POST', '/ask')[0][1].body).chat_id).toBeNull()
    expect(await screen.findByRole('tab', { name: 'Chats (1)' })).toBeInTheDocument()
    // answer is still visible (the new chat is not re-fetched)
    expect(screen.getByText(/Hold RESET for ten seconds/)).toBeInTheDocument()
    expect(callsTo(fetchMock, 'GET', '/chats/11')).toHaveLength(0)

    // "New chat" clears the conversation
    await user.click(screen.getByRole('tab', { name: 'Chats (1)' }))
    await user.click(screen.getByRole('button', { name: /New chat/ }))
    expect(screen.queryByText(/Hold RESET for ten seconds/)).not.toBeInTheDocument()
    expect(screen.getByText('Ask a question about your documents')).toBeInTheDocument()
  })

  it('deletes the open chat after confirmation', async () => {
    mockBackend({
      'GET /chats': () => json(CHATS),
      'GET /chats/7': () => json(CHAT_7),
      'DELETE /chats/7': () => new Response(null, { status: 204 }),
    })
    vi.stubGlobal('confirm', vi.fn(() => true))
    const user = userEvent.setup()
    render(<App />)
    await user.click(await screen.findByRole('tab', { name: 'Chats (2)' }))
    await user.click(screen.getByText('Warranty questions'))
    await screen.findByText('What does the warranty not cover?')
    await user.click(screen.getByRole('button', { name: 'Delete chat Warranty questions' }))
    await waitFor(() => expect(screen.queryByText('Warranty questions')).not.toBeInTheDocument())
    // the conversation is gone; the empty-chat screen (with its example buttons) is back
    expect(screen.queryByText(/Water damage caused by incorrect installation/)).not.toBeInTheDocument()
    expect(screen.getByText('Ask a question about your documents')).toBeInTheDocument()
    expect(screen.getByRole('tab', { name: 'Chats (1)' })).toBeInTheDocument()
  })

  it('renames a chat', async () => {
    const fetchMock = mockBackend({
      'GET /chats': () => json(CHATS),
      'PATCH /chats/6': () => json({ ...CHATS[1], title: 'Filter notes' }),
    })
    vi.stubGlobal('prompt', vi.fn(() => 'Filter notes'))
    const user = userEvent.setup()
    render(<App />)
    await user.click(await screen.findByRole('tab', { name: 'Chats (2)' }))
    await user.click(screen.getByRole('button', { name: 'Rename chat Older chat' }))
    expect(await screen.findByText('Filter notes')).toBeInTheDocument()
    expect(JSON.parse(callsTo(fetchMock, 'PATCH', '/chats/6')[0][1].body)).toEqual({ title: 'Filter notes' })
  })

  it('shows an error with retry when a chat fails to load', async () => {
    let fail = true
    mockBackend({
      'GET /chats': () => json(CHATS),
      'GET /chats/7': () => (fail ? json({ detail: 'Server error' }, 500) : json(CHAT_7)),
    })
    const user = userEvent.setup()
    render(<App />)
    await user.click(await screen.findByRole('tab', { name: 'Chats (2)' }))
    await user.click(screen.getByText('Warranty questions'))
    expect(await screen.findByText(/Could not load this chat/)).toBeInTheDocument()
    fail = false
    await user.click(screen.getByRole('button', { name: 'Try again' }))
    expect(await screen.findByText('What does the warranty not cover?')).toBeInTheDocument()
  })
})

describe('readiness indicator (Phase 7)', () => {
  beforeEach(() => setToken('tok'))

  it('names the component that is down and shows its detail on hover', async () => {
    mockBackend({
      'GET /health/ready': () =>
        json(
          {
            status: 'degraded',
            components: {
              database: { ok: true, detail: 'postgresql reachable' },
              vector_store: { ok: true, detail: 'ChromaDB reachable, 6 chunks indexed' },
              llm: { ok: false, detail: 'Ollama is not reachable at http://127.0.0.1:11434' },
            },
          },
          503,
        ),
    })
    render(<App />)
    const status = await screen.findByText('AI model offline')
    expect(status.closest('[title]')).toHaveAttribute('title', 'Ollama is not reachable at http://127.0.0.1:11434')
  })

  it('describeHealth covers every state', async () => {
    const { describeHealth } = await import('../App')
    const ok = { ok: true, detail: 'fine' }
    expect(describeHealth({ state: 'checking' })[1]).toBe('Checking backend…')
    expect(describeHealth({ state: 'down', detail: 'Cannot reach' })).toEqual(['bg-red-500', 'Backend offline', 'Cannot reach'])
    expect(describeHealth({ state: 'ready', components: { database: ok, llm: ok } })[1]).toBe('All systems ready')
    expect(
      describeHealth({ state: 'ready', components: { database: { ok: false, detail: 'a' }, llm: { ok: false, detail: 'b' } } }),
    ).toEqual(['bg-amber-500', 'Services degraded', 'a\nb'])
    expect(describeHealth({ state: 'ready', components: { vector_store: { ok: false, detail: 'x' } } })[1]).toBe(
      'Search index offline',
    )
  })
})
