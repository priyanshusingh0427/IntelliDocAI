import { useState } from 'react'
import { splitCitations } from '../lib/citations'

function CitationCard({ citation, active, onToggle }) {
  return (
    <li
      id={`cite-${citation.chunk_id}`}
      className={`rounded-lg border text-sm transition-colors ${
        active ? 'border-indigo-400 bg-indigo-50/60' : 'border-slate-200 bg-white'
      }`}
    >
      <button
        onClick={onToggle}
        className="flex w-full items-center gap-2 px-3 py-2 text-left"
        aria-expanded={active}
      >
        <span className="flex h-5 min-w-5 items-center justify-center rounded bg-indigo-600 px-1 text-xs font-semibold text-white">
          {citation.source}
        </span>
        <span className="min-w-0 flex-1 truncate font-medium text-slate-700">{citation.filename}</span>
        <span className="shrink-0 text-xs text-slate-500">Page {citation.page_number}</span>
        <svg className={`h-4 w-4 shrink-0 text-slate-400 transition-transform ${active ? 'rotate-180' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2" aria-hidden="true">
          <path strokeLinecap="round" strokeLinejoin="round" d="m6 9 6 6 6-6" />
        </svg>
      </button>
      {active && (
        <blockquote className="whitespace-pre-line border-t border-indigo-100 px-3 py-2 text-xs leading-relaxed text-slate-600">
          {citation.text}
        </blockquote>
      )}
    </li>
  )
}

export default function AnswerMessage({ response }) {
  const [open, setOpen] = useState(null) // source number whose passage is expanded
  const bySource = Object.fromEntries(response.citations.map((c) => [c.source, c]))
  // timings_ms has one entry per pipeline step (semantic, bm25, rerank, llm); show their sum.
  const totalMs = Object.values(response.timings_ms || {}).reduce((a, b) => a + b, 0)
  const seconds = totalMs > 0 ? (totalMs / 1000).toFixed(1) : null

  if (!response.found) {
    return (
      <div className="rounded-2xl rounded-tl-sm border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900">
        <p className="font-medium">Not found in your documents</p>
        <p className="mt-1 text-amber-800">{response.answer}</p>
      </div>
    )
  }

  return (
    <div className="space-y-3">
      <div className="rounded-2xl rounded-tl-sm border border-slate-200 bg-white px-4 py-3 text-sm leading-relaxed text-slate-800 shadow-sm">
        <p className="whitespace-pre-line">
          {splitCitations(response.answer).map((part, i) =>
            part.type === 'text' ? (
              <span key={i}>{part.value}</span>
            ) : (
              <span key={i} className="whitespace-nowrap">
                {part.numbers.map((n) => {
                  const c = bySource[n]
                  return (
                    <button
                      key={n}
                      onClick={() => setOpen(open === n ? null : n)}
                      disabled={!c}
                      title={c ? `${c.filename}, page ${c.page_number}` : undefined}
                      className="mx-0.5 inline-flex h-5 min-w-5 items-center justify-center rounded bg-indigo-100 px-1 align-text-top text-xs font-semibold text-indigo-700 hover:bg-indigo-200 disabled:opacity-50"
                      aria-label={c ? `Source ${n}: ${c.filename}, page ${c.page_number}` : `Source ${n}`}
                    >
                      {n}
                    </button>
                  )
                })}
              </span>
            ),
          )}
        </p>
      </div>

      {response.citations.length > 0 && (
        <div>
          <p className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-slate-500">Sources</p>
          <ul className="space-y-1.5">
            {response.citations.map((c) => (
              <CitationCard
                key={c.source}
                citation={c}
                active={open === c.source}
                onToggle={() => setOpen(open === c.source ? null : c.source)}
              />
            ))}
          </ul>
        </div>
      )}

      {response.warnings?.length > 0 && (
        <ul className="space-y-1 text-xs text-amber-700">
          {response.warnings.map((w, i) => (
            <li key={i}>⚠ {w}</li>
          ))}
        </ul>
      )}

      <p className="text-xs text-slate-400">
        {response.llm_model}
        {seconds && ` · ${seconds}s`}
      </p>
    </div>
  )
}
