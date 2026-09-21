/**
 * InlineEdit.tsx
 * Click-to-edit text field. Shows styled text; on click becomes an input.
 * Saves on Enter or blur. Reverts on Escape. Matches VoiceSum design system.
 */
import { useState, useRef, useEffect } from 'react'
import { Pencil, Check, X, Loader } from 'lucide-react'
import { getApiErrorDetail } from '../lib/errors'

interface Props {
  value: string
  onSave: (newValue: string) => Promise<void>
  /** Optional override for the text display style */
  textStyle?: React.CSSProperties
  placeholder?: string
  maxLength?: number
}

export default function InlineEdit({
  value,
  onSave,
  textStyle,
  placeholder = 'Untitled',
  maxLength = 200,
}: Props) {
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(value)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  // Keep draft in sync if parent value changes from outside
  useEffect(() => {
    if (!editing) setDraft(value)
  }, [value, editing])

  // Focus + select-all when entering edit mode
  useEffect(() => {
    if (editing) {
      inputRef.current?.focus()
      inputRef.current?.select()
    }
  }, [editing])

  const startEdit = () => {
    setDraft(value)
    setError(null)
    setEditing(true)
  }

  const cancel = () => {
    setDraft(value)
    setError(null)
    setEditing(false)
  }

  const commit = async () => {
    const trimmed = draft.trim()
    if (!trimmed) {
      setError('Name cannot be empty')
      return
    }
    if (trimmed === value) {
      setEditing(false)
      return
    }
    setSaving(true)
    setError(null)
    try {
      await onSave(trimmed)
      setEditing(false)
    } catch (e: unknown) {
      setError(getApiErrorDetail(e, 'Failed to save'))
    } finally {
      setSaving(false)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') { e.preventDefault(); commit() }
    if (e.key === 'Escape') cancel()
  }

  if (editing) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', minWidth: 0 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <input
            ref={inputRef}
            className="input"
            value={draft}
            onChange={e => setDraft(e.target.value)}
            onKeyDown={handleKeyDown}
            onBlur={commit}
            maxLength={maxLength}
            style={{
              flex: 1,
              padding: '0.3rem 0.6rem',
              fontSize: 'inherit',
              fontWeight: 'inherit',
              fontFamily: 'inherit',
              height: '32px',
              minWidth: 0,
              ...(error ? { borderColor: 'hsl(var(--destructive))' } : {}),
            }}
          />
          {saving
            ? <Loader size={14} className="spin" style={{ color: 'hsl(var(--pencil))', flexShrink: 0 }} />
            : (
              <>
                <button
                  onMouseDown={e => { e.preventDefault(); commit() }}
                  className="icon-btn"
                  style={{ color: 'hsl(var(--success))', width: '28px', height: '28px', flexShrink: 0 }}
                  title="Save (Enter)"
                >
                  <Check size={14} />
                </button>
                <button
                  onMouseDown={e => { e.preventDefault(); cancel() }}
                  className="icon-btn"
                  style={{ color: 'hsl(var(--pencil))', width: '28px', height: '28px', flexShrink: 0 }}
                  title="Cancel (Escape)"
                >
                  <X size={14} />
                </button>
              </>
            )}
        </div>
        {error && (
          <span style={{ fontSize: '0.72rem', color: 'hsl(var(--destructive))', fontFamily: 'Inter, sans-serif' }}>
            {error}
          </span>
        )}
      </div>
    )
  }

  return (
    <div
      onClick={startEdit}
      title="Click to rename"
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: '6px',
        cursor: 'pointer',
        minWidth: 0,
        maxWidth: '100%',
        borderRadius: '6px',
        padding: '2px 4px',
        margin: '-2px -4px',
        transition: 'background .15s',
      }}
      onMouseEnter={e => (e.currentTarget.style.background = 'hsl(var(--accent) / .08)')}
      onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}
    >
      <span style={{
        overflow: 'hidden',
        textOverflow: 'ellipsis',
        whiteSpace: 'nowrap',
        ...textStyle,
      }}>
        {value || placeholder}
      </span>
      <Pencil size={12} style={{ color: 'hsl(var(--pencil))', opacity: 0.6, flexShrink: 0 }} />
    </div>
  )
}
