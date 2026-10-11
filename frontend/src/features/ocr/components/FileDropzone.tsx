import { useRef, useState } from 'react'
import { FileText, UploadCloud, X } from 'lucide-react'
import { formatBytes } from '../utils'

interface FileDropzoneProps {
  accept: string[]          // extensions including the dot, e.g. ['.pdf', '.png']
  file: File | null
  onFile: (file: File | null) => void
  maxMb: number
  disabled?: boolean
  hint: string
}

export function FileDropzone({ accept, file, onFile, maxMb, disabled, hint }: FileDropzoneProps) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const pick = (candidate: File | undefined) => {
    if (!candidate) return
    const ext = candidate.name.slice(candidate.name.lastIndexOf('.')).toLowerCase()
    if (!accept.includes(ext)) {
      setError(`Unsupported file type. Use ${accept.join(', ')}`)
      return
    }
    if (candidate.size > maxMb * 1024 * 1024) {
      setError(`File is larger than ${maxMb} MB`)
      return
    }
    setError(null)
    onFile(candidate)
  }

  if (file) {
    return (
      <div style={{
        display: 'flex', alignItems: 'center', gap: 10, padding: '.7rem .85rem', borderRadius: 10,
        background: 'hsl(var(--paper) / .5)', border: '1px solid hsl(var(--border) / .6)',
        borderLeft: '3px solid hsl(var(--accent))',
      }}>
        <FileText size={18} style={{ color: 'hsl(var(--accent))', flexShrink: 0 }} />
        <div style={{ minWidth: 0, flex: 1 }}>
          <div style={{ fontSize: '.8rem', fontWeight: 700, color: 'hsl(var(--ink))', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            {file.name}
          </div>
          <div style={{ fontSize: '.68rem', color: 'hsl(var(--pencil))' }}>{formatBytes(file.size)}</div>
        </div>
        <button
          type="button"
          className="btn btn-ghost"
          onClick={() => onFile(null)}
          disabled={disabled}
          aria-label="Remove file"
          style={{ padding: '.3rem', borderRadius: 6 }}
        >
          <X size={14} />
        </button>
      </div>
    )
  }

  return (
    <div>
      <div
        role="button"
        tabIndex={0}
        aria-disabled={disabled}
        onClick={() => !disabled && inputRef.current?.click()}
        onKeyDown={e => { if (!disabled && (e.key === 'Enter' || e.key === ' ')) inputRef.current?.click() }}
        onDragOver={e => { e.preventDefault(); if (!disabled) setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={e => {
          e.preventDefault()
          setDragging(false)
          if (!disabled) pick(e.dataTransfer.files?.[0])
        }}
        style={{
          display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 6,
          padding: '1.4rem 1rem', borderRadius: 12, cursor: disabled ? 'not-allowed' : 'pointer', textAlign: 'center',
          border: `2px dashed ${dragging ? 'hsl(var(--accent))' : 'hsl(var(--border))'}`,
          background: dragging ? 'hsl(var(--accent) / .06)' : 'hsl(var(--paper) / .35)',
          transition: 'all .15s ease', opacity: disabled ? 0.6 : 1,
        }}
      >
        <div style={{
          width: 40, height: 40, borderRadius: 10, display: 'flex', alignItems: 'center', justifyContent: 'center',
          background: 'hsl(var(--accent) / .12)', color: 'hsl(var(--accent))',
        }}>
          <UploadCloud size={20} />
        </div>
        <div style={{ fontSize: '.82rem', fontWeight: 700, color: 'hsl(var(--ink))' }}>
          Drop a file here or click to browse
        </div>
        <div style={{ fontSize: '.7rem', color: 'hsl(var(--pencil))' }}>{hint}</div>
        <input
          ref={inputRef}
          type="file"
          accept={accept.join(',')}
          style={{ display: 'none' }}
          onChange={e => { pick(e.target.files?.[0]); e.target.value = '' }}
        />
      </div>
      {error && <div role="alert" style={{ marginTop: 6, fontSize: '.72rem', color: 'hsl(var(--destructive))', fontWeight: 600 }}>{error}</div>}
    </div>
  )
}
