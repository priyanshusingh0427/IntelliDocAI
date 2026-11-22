import { useState } from 'react'
import { api } from '../lib/api'
import { Spinner } from './UploadZone'

const MIN_PASSWORD = 8 // matches the backend's PASSWORD_MIN_LENGTH default

export default function LoginScreen({ onLoggedIn, notice }) {
  const [mode, setMode] = useState('login') // 'login' | 'register'
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [fullName, setFullName] = useState('')
  const [error, setError] = useState(null)
  const [busy, setBusy] = useState(false)

  const registering = mode === 'register'

  async function submit(e) {
    e.preventDefault()
    setError(null)
    if (registering && password.length < MIN_PASSWORD) {
      setError(`Password must be at least ${MIN_PASSWORD} characters.`)
      return
    }
    setBusy(true)
    try {
      const result = registering
        ? await api.register(email.trim(), password, fullName.trim())
        : await api.login(email.trim(), password)
      onLoggedIn(result)
    } catch (err) {
      setError(err.message)
      setBusy(false)
    }
  }

  function switchMode(next) {
    setMode(next)
    setError(null)
  }

  const field =
    'mt-1 block w-full rounded-lg border border-slate-300 px-3 py-2 text-sm focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200 focus:outline-none'

  return (
    <div className="flex min-h-full items-center justify-center bg-slate-50 px-4 py-10">
      <div className="w-full max-w-sm">
        <div className="mb-6 flex flex-col items-center text-center">
          <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-indigo-600 text-white" aria-hidden="true">
            <svg className="h-6 w-6" viewBox="0 0 24 24" fill="currentColor">
              <path d="M7 3h7l5 5v11a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2zm6 1.5V9h4.5L13 4.5z" />
            </svg>
          </div>
          <h1 className="mt-3 text-xl font-semibold text-slate-900">IntelliDoc AI</h1>
          <p className="mt-1 text-sm text-slate-500">Ask questions about your PDFs, with page citations</p>
        </div>

        <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
          <div className="mb-5 grid grid-cols-2 rounded-lg bg-slate-100 p-1 text-sm font-medium" role="tablist">
            {[
              ['login', 'Log in'],
              ['register', 'Create account'],
            ].map(([key, label]) => (
              <button
                key={key}
                role="tab"
                aria-selected={mode === key}
                onClick={() => switchMode(key)}
                className={`rounded-md py-1.5 ${mode === key ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-500 hover:text-slate-700'}`}
              >
                {label}
              </button>
            ))}
          </div>

          {notice && !error && (
            <p className="mb-4 rounded-lg bg-amber-50 px-3 py-2 text-sm text-amber-800" role="status">
              {notice}
            </p>
          )}

          <form onSubmit={submit} className="space-y-4">
            {registering && (
              <label className="block text-sm font-medium text-slate-700">
                Name <span className="font-normal text-slate-400">(optional)</span>
                <input value={fullName} onChange={(e) => setFullName(e.target.value)} autoComplete="name" maxLength={120} className={field} />
              </label>
            )}
            <label className="block text-sm font-medium text-slate-700">
              Email
              <input type="email" required value={email} onChange={(e) => setEmail(e.target.value)} autoComplete="email" className={field} />
            </label>
            <label className="block text-sm font-medium text-slate-700">
              Password
              <input
                type="password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                autoComplete={registering ? 'new-password' : 'current-password'}
                maxLength={128}
                className={field}
              />
              {registering && <span className="mt-1 block text-xs font-normal text-slate-500">At least {MIN_PASSWORD} characters</span>}
            </label>

            {error && (
              <p className="rounded-lg bg-red-50 px-3 py-2 text-sm text-red-700" role="alert">
                {error}
              </p>
            )}

            <button
              type="submit"
              disabled={busy}
              className="flex w-full items-center justify-center gap-2 rounded-lg bg-indigo-600 py-2.5 text-sm font-semibold text-white hover:bg-indigo-500 disabled:opacity-60"
            >
              {busy && <Spinner className="h-4 w-4 text-white" />}
              {registering ? 'Create account' : 'Log in'}
            </button>
          </form>
        </div>
      </div>
    </div>
  )
}
