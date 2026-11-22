import { useEffect, useRef, useState } from 'react'
import { api } from '../lib/api'
import AnswerMessage from './AnswerMessage'
import { Spinner } from './UploadZone'

/** Saved chat messages (GET /chats/{id}) -> the shapes this panel renders. */
export function toDisplayMessages(saved) {
  return saved.map((m) =>
    m.role === 'user'
      ? { role: 'user', text: m.content }
      : {
          role: 'assistant',
          response: {
            answer: m.content,
            found: m.found ?? true,
            grounded: m.grounded ?? false,
            citations: m.citations || [],
            warnings: m.warnings || [],
            llm_model: m.llm_model || '',
            timings_ms: m.timings_ms || {},
          },
        },
  )
}

function ThinkingIndicator() {
  const [elapsed, setElapsed] = useState(0)
  useEffect(() => {
    const start = Date.now()
    const timer = setInterval(() => setElapsed(Math.floor((Date.now() - start) / 1000)), 1000)
    return () => clearInterval(timer)
  }, [])
  return (
    <div className="flex items-start gap-3" role="status" aria-live="polite">
      <Avatar />
      <div className="rounded-2xl rounded-tl-sm border border-slate-200 bg-white px-4 py-3 text-sm shadow-sm">
        <div className="flex items-center gap-2 text-slate-600">
          <span className="flex gap-1" aria-hidden="true">
            <span className="h-2 w-2 animate-bounce rounded-full bg-indigo-500 [animation-delay:-0.3s]" />
            <span className="h-2 w-2 animate-bounce rounded-full bg-indigo-500 [animation-delay:-0.15s]" />
            <span className="h-2 w-2 animate-bounce rounded-full bg-indigo-500" />
          </span>
          Searching your documents and writing an answer… {elapsed > 0 && <span className="text-slate-400">{elapsed}s</span>}
        </div>
        {elapsed >= 10 && (
          <p className="mt-1 text-xs text-slate-500">
            The first question after starting the server loads the AI models and can take about a minute.
          </p>
        )}
      </div>
    </div>
  )
}

function Avatar() {
  return (
    <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-indigo-600 text-xs font-bold text-white">
      AI
    </div>
  )
}

const EXAMPLES = ['Summarize this document', 'What are the key requirements?', 'What does the warranty not cover?']

export default function ChatPanel({ chatId, onChatSaved, selectedIds, readyCount, documentNames }) {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [pending, setPending] = useState(false)
  const [historyLoading, setHistoryLoading] = useState(false)
  const [historyError, setHistoryError] = useState(null)
  const [reloadKey, setReloadKey] = useState(0)
  const bottomRef = useRef(null)
  // The chat this panel itself just created: its messages are already on screen, so
  // selecting it (which App does right after) must not reload it.
  const ownChatRef = useRef(null)
  // Changes whenever the visible chat changes; a late answer for another chat is not shown here.
  const viewRef = useRef(0)

  useEffect(() => {
    viewRef.current += 1
    setHistoryError(null)
    if (chatId != null && chatId === ownChatRef.current) return
    ownChatRef.current = null
    if (chatId == null) {
      setMessages([])
      setHistoryLoading(false)
      return
    }
    let cancelled = false
    setMessages([])
    setHistoryLoading(true)
    api
      .getChat(chatId)
      .then((chat) => !cancelled && setMessages(toDisplayMessages(chat.messages)))
      .catch((err) => !cancelled && setHistoryError(err.message))
      .finally(() => !cancelled && setHistoryLoading(false))
    return () => {
      cancelled = true
    }
  }, [chatId, reloadKey])

  useEffect(() => {
    bottomRef.current?.scrollIntoView?.({ behavior: 'smooth', block: 'end' })
  }, [messages, pending])

  async function send(question) {
    const q = question.trim()
    if (!q || pending) return
    setInput('')
    setMessages((m) => [...m, { role: 'user', text: q, scope: selectedIds }])
    setPending(true)
    const view = viewRef.current
    try {
      const response = await api.ask(q, selectedIds, chatId)
      if (view !== viewRef.current) return // user switched chats meanwhile (answer is saved)
      setMessages((m) => [...m, { role: 'assistant', response }])
      if (response.chat_id != null) {
        if (chatId == null) ownChatRef.current = response.chat_id
        onChatSaved?.(response.chat_id)
      }
    } catch (err) {
      if (view !== viewRef.current) return
      setMessages((m) => [...m, { role: 'error', text: err.message, question: q }])
    } finally {
      setPending(false)
    }
  }

  const noDocs = readyCount === 0
  const scopeLabel = selectedIds.length
    ? `Asking about: ${selectedIds.map((id) => documentNames[id]).filter(Boolean).join(', ')}`
    : `Asking about all ${readyCount} ready ${readyCount === 1 ? 'document' : 'documents'}`

  return (
    <section className="flex h-full min-h-0 flex-col" aria-label="Chat">
      <div className="min-h-0 flex-1 space-y-5 overflow-y-auto px-4 py-6 sm:px-8">
        {historyLoading && (
          <div className="flex items-center justify-center gap-2 py-10 text-sm text-slate-500" role="status">
            <Spinner className="h-4 w-4" /> Loading chat…
          </div>
        )}
        {historyError && (
          <div className="mx-auto max-w-md rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700" role="alert">
            <p>Could not load this chat: {historyError}</p>
            <button onClick={() => setReloadKey((k) => k + 1)} className="mt-2 font-semibold underline hover:no-underline">
              Try again
            </button>
          </div>
        )}
        {messages.length === 0 && !pending && !historyLoading && !historyError && (
          <div className="mx-auto mt-10 max-w-md text-center">
            <h2 className="text-lg font-semibold text-slate-800">Ask a question about your documents</h2>
            <p className="mt-2 text-sm text-slate-500">
              Answers come only from your uploaded files, with numbered citations that show the exact page.
            </p>
            {!noDocs && (
              <div className="mt-5 flex flex-wrap justify-center gap-2">
                {EXAMPLES.map((ex) => (
                  <button
                    key={ex}
                    onClick={() => send(ex)}
                    className="rounded-full border border-slate-200 bg-white px-3 py-1.5 text-xs text-slate-600 hover:border-indigo-300 hover:text-indigo-700"
                  >
                    {ex}
                  </button>
                ))}
              </div>
            )}
          </div>
        )}

        {messages.map((msg, i) => {
          if (msg.role === 'user') {
            return (
              <div key={i} className="flex justify-end">
                <div className="max-w-[85%] rounded-2xl rounded-tr-sm bg-indigo-600 px-4 py-2.5 text-sm whitespace-pre-line text-white">
                  {msg.text}
                </div>
              </div>
            )
          }
          if (msg.role === 'error') {
            return (
              <div key={i} className="flex items-start gap-3">
                <Avatar />
                <div className="rounded-2xl rounded-tl-sm border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700" role="alert">
                  <p className="font-medium">Could not get an answer</p>
                  <p className="mt-1">{msg.text}</p>
                  <button
                    onClick={() => send(msg.question)}
                    disabled={pending}
                    className="mt-2 text-xs font-semibold underline hover:no-underline disabled:opacity-50"
                  >
                    Retry
                  </button>
                </div>
              </div>
            )
          }
          return (
            <div key={i} className="flex items-start gap-3">
              <Avatar />
              <div className="min-w-0 max-w-[85%] flex-1">
                <AnswerMessage response={msg.response} />
              </div>
            </div>
          )
        })}
        {pending && <ThinkingIndicator />}
        <div ref={bottomRef} />
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault()
          send(input)
        }}
        className="border-t border-slate-200 bg-white px-4 py-3 sm:px-8"
      >
        <p className="mb-2 truncate text-xs text-slate-500">
          {noDocs ? 'Upload and index a document to start asking questions.' : scopeLabel}
        </p>
        <div className="flex items-end gap-2">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                send(input)
              }
            }}
            rows={1}
            maxLength={2000}
            disabled={noDocs}
            placeholder={noDocs ? 'No documents ready yet' : 'Ask a question…'}
            title="Enter to send, Shift+Enter for a new line"
            aria-label="Question"
            className="max-h-40 min-h-[42px] flex-1 resize-none rounded-xl border border-slate-300 px-3 py-2.5 text-sm focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200 focus:outline-none disabled:bg-slate-100"
          />
          <button
            type="submit"
            disabled={pending || noDocs || !input.trim()}
            className="h-[42px] rounded-xl bg-indigo-600 px-4 text-sm font-semibold text-white hover:bg-indigo-500 disabled:cursor-not-allowed disabled:bg-slate-300"
          >
            {pending ? 'Thinking…' : 'Ask'}
          </button>
        </div>
      </form>
    </section>
  )
}
