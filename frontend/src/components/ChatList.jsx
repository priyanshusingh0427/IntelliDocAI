import { Spinner } from './UploadZone'

function timeAgo(iso) {
  const then = new Date(iso.endsWith('Z') || iso.includes('+') ? iso : `${iso}Z`).getTime()
  const minutes = Math.round((Date.now() - then) / 60000)
  if (minutes < 1) return 'just now'
  if (minutes < 60) return `${minutes} min ago`
  const hours = Math.round(minutes / 60)
  if (hours < 24) return `${hours} h ago`
  const days = Math.round(hours / 24)
  return days === 1 ? 'yesterday' : `${days} days ago`
}

export default function ChatList({ chats, loading, error, activeId, onSelect, onNew, onRename, onDelete, onRetry }) {
  return (
    <div>
      <button
        onClick={onNew}
        className="mb-3 flex w-full items-center justify-center gap-1.5 rounded-lg border border-indigo-200 bg-white py-2 text-sm font-medium text-indigo-700 hover:bg-indigo-50"
      >
        <span aria-hidden="true">+</span> New chat
      </button>

      {loading && (
        <div className="flex items-center gap-2 px-1 py-3 text-sm text-slate-500" role="status">
          <Spinner className="h-4 w-4" /> Loading chats…
        </div>
      )}
      {!loading && error && (
        <div className="rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700">
          <p>{error}</p>
          <button onClick={onRetry} className="mt-2 font-semibold underline hover:no-underline">
            Try again
          </button>
        </div>
      )}
      {!loading && !error && chats.length === 0 && (
        <p className="px-1 py-3 text-sm text-slate-500">No chats yet. Your questions and answers are saved here.</p>
      )}

      <ul className="space-y-1" aria-label="Chat history">
        {chats.map((chat) => {
          const active = chat.id === activeId
          return (
            <li key={chat.id} className="group">
              <div
                className={`flex items-center gap-1 rounded-lg pr-1 ${active ? 'bg-indigo-100' : 'hover:bg-slate-100'}`}
              >
                <button
                  onClick={() => onSelect(chat.id)}
                  aria-current={active ? 'true' : undefined}
                  className="min-w-0 flex-1 px-3 py-2 text-left"
                >
                  <p className={`truncate text-sm ${active ? 'font-medium text-indigo-900' : 'text-slate-700'}`}>{chat.title}</p>
                  <p className="text-xs text-slate-400">
                    {timeAgo(chat.updated_at)} · {Math.ceil(chat.message_count / 2)} {chat.message_count === 2 ? 'question' : 'questions'}
                  </p>
                </button>
                <button
                  onClick={() => onRename(chat)}
                  className="rounded p-1 text-slate-400 opacity-100 hover:text-indigo-600 sm:opacity-0 sm:group-hover:opacity-100 focus:opacity-100"
                  aria-label={`Rename chat ${chat.title}`}
                  title="Rename"
                >
                  <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
                    <path strokeLinecap="round" strokeLinejoin="round" d="m16.9 4.6 2.5 2.5M4 20l4.2-1 10.4-10.4a1.8 1.8 0 0 0-2.5-2.5L5.7 16.5 4 20z" />
                  </svg>
                </button>
                <button
                  onClick={() => onDelete(chat)}
                  className="rounded p-1 text-slate-400 opacity-100 hover:text-red-600 sm:opacity-0 sm:group-hover:opacity-100 focus:opacity-100"
                  aria-label={`Delete chat ${chat.title}`}
                  title="Delete"
                >
                  <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M6 7h12M9 7V5a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2m2 0-.8 12a2 2 0 0 1-2 1.9H8.8a2 2 0 0 1-2-1.9L6 7" />
                  </svg>
                </button>
              </div>
            </li>
          )
        })}
      </ul>
    </div>
  )
}
