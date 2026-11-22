import { useCallback, useEffect, useState } from 'react'
import { api, getToken, setToken, setUnauthorizedHandler } from './lib/api'
import UploadZone, { Spinner } from './components/UploadZone'
import DocumentList from './components/DocumentList'
import ChatPanel from './components/ChatPanel'
import ChatList from './components/ChatList'
import LoginScreen from './components/LoginScreen'

const COMPONENT_LABELS = {
  llm: 'AI model offline',
  database: 'Database offline',
  vector_store: 'Search index offline',
}

/** Turns GET /health/ready into a dot colour, a short label and a tooltip with details. */
export function describeHealth(health) {
  if (health.state === 'checking') return ['bg-slate-300', 'Checking backend…', '']
  if (health.state === 'down') return ['bg-red-500', 'Backend offline', health.detail || '']
  const failing = Object.entries(health.components || {}).filter(([, c]) => !c.ok)
  if (!failing.length) return ['bg-emerald-500', 'All systems ready', '']
  const label = failing.length === 1 ? COMPONENT_LABELS[failing[0][0]] || 'Service degraded' : 'Services degraded'
  return ['bg-amber-500', label, failing.map(([, c]) => c.detail).join('\n')]
}

function HealthDot({ health }) {
  const [dot, label, detail] = describeHealth(health)
  return (
    <span className="flex items-center gap-1.5 text-xs text-slate-500" title={detail} role="status">
      <span className={`h-2 w-2 rounded-full ${dot}`} aria-hidden="true" />
      <span className="hidden sm:inline">{label}</span>
    </span>
  )
}

function Toasts({ items, onDismiss }) {
  return (
    <div className="pointer-events-none fixed right-4 bottom-4 z-50 flex w-[min(24rem,calc(100vw-2rem))] flex-col gap-2">
      {items.map((t) => (
        <div
          key={t.id}
          role={t.kind === 'error' ? 'alert' : 'status'}
          className={`pointer-events-auto flex items-start gap-2 rounded-lg border px-3 py-2.5 text-sm shadow-lg ${
            t.kind === 'error' ? 'border-red-200 bg-red-50 text-red-800' : 'border-emerald-200 bg-emerald-50 text-emerald-800'
          }`}
        >
          <p className="flex-1">{t.text}</p>
          <button onClick={() => onDismiss(t.id)} className="text-current opacity-60 hover:opacity-100" aria-label="Dismiss">
            ×
          </button>
        </div>
      ))}
    </div>
  )
}

let toastId = 0

function useToasts() {
  const [toasts, setToasts] = useState([])
  const notify = useCallback((text, kind = 'error') => {
    const id = ++toastId
    setToasts((t) => [...t, { id, text, kind }])
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), kind === 'error' ? 8000 : 4000)
  }, [])
  const dismiss = useCallback((id) => setToasts((t) => t.filter((x) => x.id !== id)), [])
  return { toasts, notify, dismiss }
}

/** Decides between the login screen and the workspace (Phase 6). */
export default function App() {
  const [user, setUser] = useState(null)
  const [checking, setChecking] = useState(() => Boolean(getToken()))
  const [notice, setNotice] = useState(null)
  const [welcome, setWelcome] = useState(null)

  const logout = useCallback((message = null) => {
    setToken(null)
    setUser(null)
    setNotice(message)
  }, [])

  useEffect(() => {
    setUnauthorizedHandler(() => logout('Your session expired. Please log in again.'))
    return () => setUnauthorizedHandler(null)
  }, [logout])

  // A saved token from an earlier visit: check it is still valid.
  useEffect(() => {
    if (!getToken()) return
    api
      .me()
      .then(setUser)
      .catch(() => setToken(null))
      .finally(() => setChecking(false))
  }, [])

  function handleLoggedIn(result) {
    setToken(result.access_token)
    setNotice(null)
    setWelcome(
      result.adopted_documents
        ? `Welcome! ${result.adopted_documents} document(s) uploaded earlier now belong to your account.`
        : null,
    )
    setUser(result.user)
  }

  if (checking) {
    return (
      <div className="flex h-full items-center justify-center gap-2 text-sm text-slate-500" role="status">
        <Spinner className="h-5 w-5" /> Loading…
      </div>
    )
  }
  if (!user) return <LoginScreen onLoggedIn={handleLoggedIn} notice={notice} />
  // key: a different user gets a completely fresh workspace (no leftover state).
  return <Workspace key={user.id} user={user} welcome={welcome} onLogout={() => logout()} />
}

function Workspace({ user, welcome, onLogout }) {
  const [documents, setDocuments] = useState([])
  const [docsLoading, setDocsLoading] = useState(true)
  const [docsError, setDocsError] = useState(null)
  const [selectedIds, setSelectedIds] = useState([])
  const [chats, setChats] = useState([])
  const [chatsLoading, setChatsLoading] = useState(true)
  const [chatsError, setChatsError] = useState(null)
  const [activeChatId, setActiveChatId] = useState(null)
  const [tab, setTab] = useState('documents') // sidebar: 'documents' | 'chats'
  const [health, setHealth] = useState({ state: 'checking' })
  const { toasts, notify, dismiss } = useToasts()

  const loadDocuments = useCallback(async () => {
    setDocsLoading(true)
    setDocsError(null)
    try {
      setDocuments(await api.listDocuments())
    } catch (err) {
      setDocsError(err.message)
    } finally {
      setDocsLoading(false)
    }
  }, [])

  const loadChats = useCallback(async ({ quiet = false } = {}) => {
    if (!quiet) setChatsLoading(true)
    setChatsError(null)
    try {
      setChats(await api.listChats())
    } catch (err) {
      setChatsError(err.message)
    } finally {
      setChatsLoading(false)
    }
  }, [])

  useEffect(() => {
    loadDocuments()
    loadChats()
  }, [loadDocuments, loadChats])

  // Readiness of the database, search index and AI model; re-checked every minute so the
  // header recovers by itself when e.g. Ollama is started.
  useEffect(() => {
    let cancelled = false
    const check = () =>
      api
        .ready()
        .then((r) => !cancelled && setHealth({ state: 'ready', components: r.components }))
        .catch((err) => !cancelled && setHealth({ state: 'down', detail: err.message }))
    check()
    const timer = setInterval(check, 60000)
    return () => {
      cancelled = true
      clearInterval(timer)
    }
  }, [])

  useEffect(() => {
    if (welcome) notify(welcome, 'success')
  }, [welcome, notify])

  function handleUploaded(doc) {
    setDocuments((docs) => [doc, ...docs.filter((d) => d.id !== doc.id)])
    if (doc.index_status === 'failed') {
      notify(`${doc.original_filename} was uploaded but indexing failed: ${doc.index_error}`)
    } else {
      const pages = doc.page_count == null ? '' : ` (${doc.page_count} ${doc.page_count === 1 ? 'page' : 'pages'})`
      notify(`${doc.original_filename} is ready${pages}.`, 'success')
    }
  }

  async function handleDelete(doc) {
    if (!window.confirm(`Delete "${doc.original_filename}"? This removes the file and its search index.`)) return
    try {
      await api.deleteDocument(doc.id)
      setDocuments((docs) => docs.filter((d) => d.id !== doc.id))
      setSelectedIds((ids) => ids.filter((id) => id !== doc.id))
    } catch (err) {
      notify(`Could not delete ${doc.original_filename}: ${err.message}`)
    }
  }

  async function handleReindex(doc) {
    try {
      await api.reindexDocument(doc.id)
      await loadDocuments()
      notify(`${doc.original_filename} indexed.`, 'success')
    } catch (err) {
      notify(`Indexing ${doc.original_filename} failed: ${err.message}`)
    }
  }

  function handleChatSaved(chatId) {
    setActiveChatId(chatId)
    loadChats({ quiet: true })
  }

  async function handleRenameChat(chat) {
    const title = window.prompt('Rename chat', chat.title)?.trim()
    if (!title || title === chat.title) return
    try {
      const updated = await api.renameChat(chat.id, title)
      setChats((cs) => cs.map((c) => (c.id === chat.id ? updated : c)))
    } catch (err) {
      notify(`Could not rename the chat: ${err.message}`)
    }
  }

  async function handleDeleteChat(chat) {
    if (!window.confirm(`Delete the chat "${chat.title}"? Its questions and answers cannot be recovered.`)) return
    try {
      await api.deleteChat(chat.id)
      setChats((cs) => cs.filter((c) => c.id !== chat.id))
      if (activeChatId === chat.id) setActiveChatId(null)
    } catch (err) {
      notify(`Could not delete the chat: ${err.message}`)
    }
  }

  const ready = documents.filter((d) => d.index_status === 'indexed')
  const documentNames = Object.fromEntries(documents.map((d) => [d.id, d.original_filename]))
  const displayName = user.full_name || user.email

  return (
    <div className="flex h-full flex-col">
      <header className="flex items-center justify-between gap-3 border-b border-slate-200 bg-white px-4 py-3 sm:px-6">
        <div className="flex min-w-0 items-center gap-2.5">
          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-indigo-600 text-white" aria-hidden="true">
            <svg className="h-5 w-5" viewBox="0 0 24 24" fill="currentColor">
              <path d="M7 3h7l5 5v11a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2zm6 1.5V9h4.5L13 4.5z" />
            </svg>
          </div>
          <div className="min-w-0">
            <h1 className="text-base leading-tight font-semibold text-slate-900">IntelliDoc AI</h1>
            <p className="hidden text-xs text-slate-500 sm:block">Ask questions, get answers with page citations</p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <HealthDot health={health} />
          <span className="hidden max-w-[12rem] truncate text-sm text-slate-600 md:inline" title={user.email}>
            {displayName}
          </span>
          <button
            onClick={onLogout}
            className="rounded-lg border border-slate-200 px-2.5 py-1.5 text-xs font-medium text-slate-600 hover:bg-slate-50"
          >
            Log out
          </button>
        </div>
      </header>

      <div className="flex min-h-0 flex-1 flex-col md:flex-row">
        <aside className="flex max-h-[45vh] shrink-0 flex-col border-b border-slate-200 bg-slate-50 md:max-h-none md:w-[340px] md:border-r md:border-b-0">
          <div className="grid grid-cols-2 gap-1 border-b border-slate-200 p-2 text-sm font-medium" role="tablist">
            {[
              ['documents', `Documents${documents.length ? ` (${documents.length})` : ''}`],
              ['chats', `Chats${chats.length ? ` (${chats.length})` : ''}`],
            ].map(([key, label]) => (
              <button
                key={key}
                role="tab"
                aria-selected={tab === key}
                onClick={() => setTab(key)}
                className={`rounded-md py-1.5 ${tab === key ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-500 hover:text-slate-700'}`}
              >
                {label}
              </button>
            ))}
          </div>

          <div className="min-h-0 flex-1 overflow-y-auto p-4">
            {tab === 'documents' ? (
              <div className="flex flex-col gap-4">
                <UploadZone onUploaded={handleUploaded} onError={notify} />
                <div>
                  <div className="mb-2 flex items-center justify-between px-1">
                    <h2 className="text-xs font-semibold tracking-wide text-slate-500 uppercase">Your documents</h2>
                    {selectedIds.length > 0 && (
                      <button onClick={() => setSelectedIds([])} className="text-xs text-indigo-600 hover:underline">
                        Clear selection
                      </button>
                    )}
                  </div>
                  <DocumentList
                    documents={documents}
                    loading={docsLoading}
                    error={docsError}
                    onRetry={loadDocuments}
                    selectedIds={selectedIds}
                    onToggle={(id) =>
                      setSelectedIds((ids) => (ids.includes(id) ? ids.filter((x) => x !== id) : [...ids, id]))
                    }
                    onDelete={handleDelete}
                    onReindex={handleReindex}
                  />
                  {ready.length > 0 && (
                    <p className="mt-3 px-1 text-xs text-slate-400">Tick documents to limit questions to them.</p>
                  )}
                </div>
              </div>
            ) : (
              <ChatList
                chats={chats}
                loading={chatsLoading}
                error={chatsError}
                activeId={activeChatId}
                onSelect={setActiveChatId}
                onNew={() => setActiveChatId(null)}
                onRename={handleRenameChat}
                onDelete={handleDeleteChat}
                onRetry={() => loadChats()}
              />
            )}
          </div>
        </aside>

        <main className="min-h-0 flex-1">
          <ChatPanel
            chatId={activeChatId}
            onChatSaved={handleChatSaved}
            selectedIds={selectedIds}
            readyCount={ready.length}
            documentNames={documentNames}
          />
        </main>
      </div>

      <Toasts items={toasts} onDismiss={dismiss} />
    </div>
  )
}
