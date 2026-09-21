import { useState } from 'react'
import { Edit2, Check, X } from 'lucide-react'

interface SpeakerLabelProps {
  speaker: string
  color: string
  onRename?: (oldName: string, newName: string) => void
  editable?: boolean
}

export default function SpeakerLabel({ speaker, color, onRename, editable = false }: SpeakerLabelProps) {
  const [isEditing, setIsEditing] = useState(false)
  const [newName, setNewName] = useState(speaker)

  const handleSave = () => {
    if (newName.trim() && newName !== speaker && onRename) {
      onRename(speaker, newName.trim())
    }
    setIsEditing(false)
  }

  const handleCancel = () => {
    setNewName(speaker)
    setIsEditing(false)
  }

  if (!editable) {
    return (
      <span className={`badge badge-${color}`} style={{ fontSize: '0.75rem', fontWeight: 700 }}>
        {speaker}
      </span>
    )
  }

  if (isEditing) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
        <input
          type="text"
          value={newName}
          onChange={(e) => setNewName(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') handleSave()
            if (e.key === 'Escape') handleCancel()
          }}
          autoFocus
          style={{
            padding: '4px 8px',
            fontSize: '0.75rem',
            fontWeight: 700,
            background: 'hsl(var(--card))',
            border: '1.5px solid hsl(var(--accent))',
            borderRadius: '6px',
            color: 'hsl(var(--ink))',
            outline: 'none',
            width: '120px',
            fontFamily: 'Inter, sans-serif'
          }}
        />
        <button onClick={handleSave} className="icon-btn" style={{ width: '24px', height: '24px' }}>
          <Check size={14} style={{ color: 'hsl(130, 60%, 45%)' }} />
        </button>
        <button onClick={handleCancel} className="icon-btn" style={{ width: '24px', height: '24px' }}>
          <X size={14} style={{ color: 'hsl(var(--destructive))' }} />
        </button>
      </div>
    )
  }

  return (
    <button
      onClick={() => setIsEditing(true)}
      className={`badge badge-${color}`}
      style={{
        fontSize: '0.75rem',
        fontWeight: 700,
        cursor: 'pointer',
        display: 'inline-flex',
        alignItems: 'center',
        gap: '6px',
        border: 'none',
      }}
    >
      {speaker}
      <Edit2 size={11} style={{ opacity: 0.6 }} />
    </button>
  )
}
