import { useRef, useState } from 'react'
import { api, formatBytes } from '../lib/api'

// Mirrors the backend defaults (MAX_UPLOAD_SIZE_MB, ALLOWED_EXTENSIONS). The backend still
// checks everything; this just gives instant feedback before uploading.
const ALLOWED = ['.pdf', '.docx', '.txt']
const MAX_MB = 10

export function validateFile(file) {
  const name = file.name.toLowerCase()
  if (!ALLOWED.some((ext) => name.endsWith(ext))) {
    return `${file.name}: only PDF, DOCX and TXT files are supported.`
  }
  if (file.size === 0) return `${file.name}: the file is empty.`
  if (file.size > MAX_MB * 1024 * 1024) {
    return `${file.name}: ${formatBytes(file.size)} is over the ${MAX_MB} MB limit.`
  }
  return null
}

export default function UploadZone({ onUploaded, onError }) {
  const inputRef = useRef(null)
  const [dragging, setDragging] = useState(false)
  const [uploading, setUploading] = useState(null) // file name being processed

  async function handleFiles(fileList) {
    const files = Array.from(fileList || [])
    for (const file of files) {
      const problem = validateFile(file)
      if (problem) {
        onError(problem)
        continue
      }
      setUploading(file.name)
      try {
        const doc = await api.uploadDocument(file)
        onUploaded(doc)
      } catch (err) {
        onError(`${file.name}: ${err.message}`)
      } finally {
        setUploading(null)
      }
    }
    if (inputRef.current) inputRef.current.value = ''
  }

  const busy = uploading !== null

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault()
        if (!busy) setDragging(true)
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault()
        setDragging(false)
        if (!busy) handleFiles(e.dataTransfer.files)
      }}
      className={`rounded-xl border-2 border-dashed p-5 text-center transition-colors ${
        dragging ? 'border-indigo-500 bg-indigo-50' : 'border-slate-300 bg-white'
      }`}
    >
      <input
        ref={inputRef}
        type="file"
        accept={ALLOWED.join(',')}
        multiple
        className="hidden"
        data-testid="file-input"
        onChange={(e) => handleFiles(e.target.files)}
      />
      {busy ? (
        <div className="flex flex-col items-center gap-2 py-1" role="status">
          <Spinner />
          <p className="text-sm font-medium text-slate-700">Processing {uploading}…</p>
          <p className="text-xs text-slate-500">Extracting text and building the search index</p>
        </div>
      ) : (
        <>
          <svg className="mx-auto h-8 w-8 text-indigo-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 16.5V3.75m0 0L7.5 8.25M12 3.75l4.5 4.5M3.75 15v3.75A1.5 1.5 0 0 0 5.25 20.25h13.5a1.5 1.5 0 0 0 1.5-1.5V15" />
          </svg>
          <p className="mt-2 text-sm text-slate-700">
            Drag a PDF here, or{' '}
            <button
              type="button"
              onClick={() => inputRef.current?.click()}
              className="font-semibold text-indigo-600 hover:text-indigo-500 focus:outline-none focus-visible:underline"
            >
              browse
            </button>
          </p>
          <p className="mt-1 text-xs text-slate-500">PDF, DOCX or TXT · up to {MAX_MB} MB</p>
        </>
      )}
    </div>
  )
}

export function Spinner({ className = 'h-5 w-5' }) {
  return (
    <svg className={`animate-spin text-indigo-600 ${className}`} viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 0 1 8-8v4a4 4 0 0 0-4 4H4z" />
    </svg>
  )
}
