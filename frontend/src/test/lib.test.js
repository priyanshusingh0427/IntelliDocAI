import { describe, expect, it, vi, afterEach } from 'vitest'
import { api, ApiError, errorMessage, formatBytes } from '../lib/api'
import { splitCitations } from '../lib/citations'
import { validateFile } from '../components/UploadZone'

afterEach(() => vi.unstubAllGlobals())

describe('splitCitations', () => {
  it('splits single, grouped and adjacent markers', () => {
    expect(splitCitations('Hold RESET [1]. Warranty is 3 years [2, 3][4].')).toEqual([
      { type: 'text', value: 'Hold RESET ' },
      { type: 'cite', numbers: [1] },
      { type: 'text', value: '. Warranty is 3 years ' },
      { type: 'cite', numbers: [2, 3] },
      { type: 'cite', numbers: [4] },
      { type: 'text', value: '.' },
    ])
  })
  it('returns plain text when there are no markers', () => {
    expect(splitCitations('No citations here')).toEqual([{ type: 'text', value: 'No citations here' }])
  })
})

describe('errorMessage', () => {
  it('uses a string detail', () => {
    expect(errorMessage({ detail: 'File is password-protected' }, 422)).toBe('File is password-protected')
  })
  it('joins FastAPI validation errors', () => {
    const body = { detail: [{ loc: ['body', 'question'], msg: 'String should have at least 1 character' }] }
    expect(errorMessage(body, 422)).toBe('question: String should have at least 1 character')
  })
  it('falls back by status', () => {
    expect(errorMessage(null, 503)).toMatch(/unavailable/)
    expect(errorMessage(null, 500)).toMatch(/Server error \(500\)/)
  })
})

describe('api client', () => {
  it('turns a network failure into a friendly ApiError', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')))
    await expect(api.listDocuments()).rejects.toMatchObject({ name: 'ApiError', status: 0 })
    await expect(api.listDocuments()).rejects.toThrow(/Cannot reach the backend/)
  })
  it('surfaces the backend detail on HTTP errors', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: 'Document not found' }), { status: 404 })),
    )
    const err = await api.deleteDocument(99).catch((e) => e)
    expect(err).toBeInstanceOf(ApiError)
    expect(err.status).toBe(404)
    expect(err.message).toBe('Document not found')
  })
  it('sends document_ids only when some are selected', async () => {
    const fetchMock = vi.fn(async () => new Response('{}', { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)
    await api.ask('q', [])
    await api.ask('q', [2])
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({ question: 'q', document_ids: null, chat_id: null })
    expect(JSON.parse(fetchMock.mock.calls[1][1].body)).toEqual({ question: 'q', document_ids: [2], chat_id: null })
  })
})

describe('validateFile', () => {
  it('accepts a PDF under the limit', () => {
    expect(validateFile(new File(['%PDF'], 'a.PDF'))).toBeNull()
  })
  it('rejects other types, empty files and oversized files', () => {
    expect(validateFile(new File(['x'], 'a.exe'))).toMatch(/only PDF/)
    expect(validateFile(new File([], 'a.pdf'))).toMatch(/empty/)
    const big = new File(['x'], 'big.pdf')
    Object.defineProperty(big, 'size', { value: 11 * 1024 * 1024 })
    expect(validateFile(big)).toMatch(/over the 10 MB limit/)
  })
})

describe('formatBytes', () => {
  it('formats sizes', () => {
    expect(formatBytes(500)).toBe('500 B')
    expect(formatBytes(2048)).toBe('2.0 KB')
    expect(formatBytes(3 * 1024 * 1024)).toBe('3.0 MB')
  })
})
