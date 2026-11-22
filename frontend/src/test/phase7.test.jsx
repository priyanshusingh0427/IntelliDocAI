// Phase 7: UI paths the earlier tests did not reach (coverage gaps).
import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest'
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import App from '../App'
import { setToken } from '../lib/api'

const USER = { id: 1, email: 'priya@example.com', full_name: null, created_at: '2026-11-22T10:00:00' }
const READY = { status: 'ready', components: { database: { ok: true, detail: 'ok' } } }
const FAILED_DOC = {
  id: 4, original_filename: 'Scan.pdf', extension: '.pdf', size_bytes: 2048, status: 'processed', char_count: 10,
  page_count: 1, created_at: '2026-11-22T10:00:00', index_status: 'failed', index_error: 'model crashed', chunk_count: null,
}
const json = (body, status = 200) => new Response(JSON.stringify(body), { status })

function mockBackend(routes) {
  const all = {
    'GET /health/ready': () => json(READY),
    'GET /auth/me': () => json(USER),
    'GET /chats': () => json([]),
    'GET /documents': () => json([]),
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

beforeEach(() => {
  Element.prototype.scrollIntoView = vi.fn()
  setToken('tok')
})
afterEach(() => {
  vi.unstubAllGlobals()
  setToken(null)
})

describe('indexing problems', () => {
  it('shows a failed index with its reason, and the Index button repairs it', async () => {
    let docs = [FAILED_DOC]
    const fetchMock = mockBackend({
      'GET /documents': () => json(docs),
      'POST /documents/4/index': () => {
        docs = [{ ...FAILED_DOC, index_status: 'indexed', index_error: null, chunk_count: 2 }]
        return json({ document_id: 4, index_status: 'indexed', chunk_count: 2, embedding_model: 'm' })
      },
    })
    const user = userEvent.setup()
    render(<App />)
    const row = (await screen.findByText('Scan.pdf')).closest('li')
    expect(within(row).getByText('Index failed')).toBeInTheDocument()
    expect(within(row).getByText('model crashed')).toBeInTheDocument()
    // a failed document cannot be ticked for questions, and nothing is "ready" to ask about
    expect(within(row).getByRole('checkbox')).toBeDisabled()
    expect(screen.getByLabelText('Question')).toBeDisabled()

    await user.click(within(row).getByRole('button', { name: 'Index' }))
    expect(await screen.findByText('Scan.pdf indexed.')).toBeInTheDocument()
    expect(within(screen.getByText('Scan.pdf').closest('li')).getByText('Ready')).toBeInTheDocument()
    expect(screen.getByLabelText('Question')).toBeEnabled()
    expect(fetchMock.mock.calls.some(([url, o]) => o?.method === 'POST' && url.endsWith('/documents/4/index'))).toBe(true)
  })

  it('reports a re-index failure', async () => {
    mockBackend({
      'GET /documents': () => json([FAILED_DOC]),
      'POST /documents/4/index': () => json({ detail: 'Indexing failed: model crashed' }, 500),
    })
    const user = userEvent.setup()
    render(<App />)
    await user.click(await screen.findByRole('button', { name: 'Index' }))
    expect(await screen.findByText('Indexing Scan.pdf failed: Indexing failed: model crashed')).toBeInTheDocument()
  })

  it('warns when an upload was stored but could not be indexed', async () => {
    mockBackend({ 'POST /documents': () => json(FAILED_DOC, 201) })
    const user = userEvent.setup()
    render(<App />)
    await screen.findByText(/No documents yet/)
    await user.upload(screen.getByTestId('file-input'), new File(['%PDF'], 'Scan.pdf', { type: 'application/pdf' }))
    expect(await screen.findByText('Scan.pdf was uploaded but indexing failed: model crashed')).toBeInTheDocument()
  })
})

describe('upload interactions', () => {
  it('accepts a file dropped on the upload area and shows progress while processing', async () => {
    let finish
    mockBackend({
      'POST /documents': () =>
        new Promise((resolve) => {
          finish = () => resolve(json({ ...FAILED_DOC, id: 9, original_filename: 'Dropped.pdf', index_status: 'indexed', index_error: null }, 201))
        }),
    })
    render(<App />)
    await screen.findByText(/No documents yet/)
    const zone = screen.getByText(/Drag a PDF here/).closest('div')
    const file = new File(['%PDF'], 'Dropped.pdf', { type: 'application/pdf' })
    fireEvent.dragOver(zone, { dataTransfer: { files: [file] } })
    fireEvent.drop(zone, { dataTransfer: { files: [file] } })
    expect(await screen.findByText('Processing Dropped.pdf…')).toBeInTheDocument()
    finish()
    expect(await screen.findByText('Dropped.pdf is ready (1 page).')).toBeInTheDocument()
    await waitFor(() => expect(screen.queryByText('Processing Dropped.pdf…')).not.toBeInTheDocument())
  })
})

describe('answer details', () => {
  it('shows answer warnings (e.g. low confidence) under the sources', async () => {
    const citation = {
      source: 1, document_id: 2, filename: 'X200_Manual.pdf', page_number: 2, chunk_id: 'c', rerank_score: -9.6, text: 'Part ZX-4471.',
    }
    mockBackend({
      'GET /documents': () => json([{ ...FAILED_DOC, id: 2, original_filename: 'X200_Manual.pdf', index_status: 'indexed', index_error: null }]),
      'POST /ask': () =>
        json({
          question: 'q', answer: 'The filter is ZX-4471 [1].', found: true, grounded: true, citations: [citation],
          sources: [citation], llm_model: 'ollama/llama3.2:3b', reranker_model: 'x', timings_ms: {},
          warnings: ['Low-confidence match: no passage scored as clearly relevant.'], chat_id: 1, message_id: 2,
        }),
    })
    const user = userEvent.setup()
    render(<App />)
    await screen.findByText('X200_Manual.pdf')
    await user.type(screen.getByLabelText('Question'), 'Which filter should I buy?{Enter}')
    expect(await screen.findByText(/Low-confidence match/)).toBeInTheDocument()
    // no timing info -> the model name is shown without a duration
    expect(screen.getByText('ollama/llama3.2:3b')).toBeInTheDocument()
  })
})
