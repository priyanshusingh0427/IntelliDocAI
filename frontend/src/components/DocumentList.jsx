import { useState } from 'react'
import { formatBytes } from '../lib/api'
import { Spinner } from './UploadZone'

function IndexBadge({ doc }) {
  if (doc.index_status === 'indexed') {
    return <span className="rounded-full bg-emerald-50 px-2 py-0.5 text-xs font-medium text-emerald-700">Ready</span>
  }
  if (doc.index_status === 'failed') {
    return (
      <span className="rounded-full bg-red-50 px-2 py-0.5 text-xs font-medium text-red-700" title={doc.index_error || ''}>
        Index failed
      </span>
    )
  }
  return <span className="rounded-full bg-slate-100 px-2 py-0.5 text-xs font-medium text-slate-600">Not indexed</span>
}

export default function DocumentList({
  documents,
  loading,
  error,
  onRetry,
  selectedIds,
  onToggle,
  onDelete,
  onReindex,
}) {
  const [busyId, setBusyId] = useState(null)

  async function run(id, action) {
    setBusyId(id)
    try {
      await action()
    } finally {
      setBusyId(null)
    }
  }

  if (loading) {
    return (
      <div className="flex items-center gap-2 px-1 py-4 text-sm text-slate-500" role="status">
        <Spinner className="h-4 w-4" /> Loading documents…
      </div>
    )
  }
  if (error) {
    return (
      <div className="rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700">
        <p>{error}</p>
        <button onClick={onRetry} className="mt-2 font-semibold underline hover:no-underline">
          Try again
        </button>
      </div>
    )
  }
  if (!documents.length) {
    return <p className="px-1 py-4 text-sm text-slate-500">No documents yet. Upload a PDF to get started.</p>
  }

  return (
    <ul className="space-y-2" aria-label="Documents">
      {documents.map((doc) => {
        const indexed = doc.index_status === 'indexed'
        const busy = busyId === doc.id
        return (
          <li key={doc.id} className="rounded-lg border border-slate-200 bg-white p-3 shadow-sm">
            <div className="flex items-start gap-3">
              <input
                type="checkbox"
                className="mt-1 h-4 w-4 accent-indigo-600 disabled:opacity-40"
                checked={selectedIds.includes(doc.id)}
                disabled={!indexed}
                onChange={() => onToggle(doc.id)}
                aria-label={`Ask about ${doc.original_filename} only`}
              />
              <div className="min-w-0 flex-1">
                <p className="truncate text-sm font-medium text-slate-800" title={doc.original_filename}>
                  {doc.original_filename}
                </p>
                <div className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-slate-500">
                  <IndexBadge doc={doc} />
                  {doc.page_count != null && <span>{doc.page_count} {doc.page_count === 1 ? 'page' : 'pages'}</span>}
                  <span>{formatBytes(doc.size_bytes)}</span>
                  {doc.chunk_count != null && indexed && <span>{doc.chunk_count} {doc.chunk_count === 1 ? 'chunk' : 'chunks'}</span>}
                </div>
                {doc.index_status === 'failed' && doc.index_error && (
                  <p className="mt-1 text-xs text-red-600">{doc.index_error}</p>
                )}
              </div>
              <div className="flex shrink-0 items-center gap-1">
                {busy ? (
                  <Spinner className="h-4 w-4" />
                ) : (
                  <>
                    {!indexed && (
                      <button
                        onClick={() => run(doc.id, () => onReindex(doc))}
                        className="rounded p-1 text-xs font-medium text-indigo-600 hover:bg-indigo-50"
                        title="Build the search index for this document"
                      >
                        Index
                      </button>
                    )}
                    <button
                      onClick={() => run(doc.id, () => onDelete(doc))}
                      className="rounded p-1 text-slate-400 hover:bg-red-50 hover:text-red-600"
                      aria-label={`Delete ${doc.original_filename}`}
                      title="Delete"
                    >
                      <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
                        <path strokeLinecap="round" strokeLinejoin="round" d="M6 7h12M9 7V5a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2m2 0-.8 12a2 2 0 0 1-2 1.9H8.8a2 2 0 0 1-2-1.9L6 7" />
                      </svg>
                    </button>
                  </>
                )}
              </div>
            </div>
          </li>
        )
      })}
    </ul>
  )
}
