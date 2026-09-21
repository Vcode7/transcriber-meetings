import React, { useState } from 'react'
import { X } from 'lucide-react'

interface Props {
  tags: string[]
  onChange: (tags: string[]) => void
  placeholder?: string
}

export default function TagInput({ tags, onChange, placeholder = "Add tag..." }: Props) {
  const [input, setInput] = useState('')

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && input.trim()) {
      e.preventDefault()
      if (!tags.includes(input.trim())) {
        onChange([...tags, input.trim()])
      }
      setInput('')
    } else if (e.key === 'Backspace' && !input && tags.length > 0) {
      onChange(tags.slice(0, -1))
    }
  }

  const removeTag = (indexToRemove: number) => {
    onChange(tags.filter((_, index) => index !== indexToRemove))
  }

  return (
    <div style={{
      display: 'flex',
      flexWrap: 'wrap',
      gap: '8px',
      padding: '0.4rem 0.6rem',
      background: 'var(--input-bg, hsl(var(--paper) / .5))',
      border: '1.5px solid var(--input-border, hsl(var(--border)))',
      borderRadius: '8px',
      alignItems: 'center',
      minHeight: '40px'
    }}>
      {tags.map((tag, index) => (
        <div key={index} style={{
          display: 'flex',
          alignItems: 'center',
          gap: '4px',
          background: 'hsl(var(--accent) / .15)',
          color: 'hsl(var(--accent))',
          padding: '2px 8px',
          borderRadius: '16px',
          fontSize: '0.82rem',
          fontWeight: 500,
          fontFamily: 'Inter, sans-serif'
        }}>
          {tag}
          <button
            onClick={() => removeTag(index)}
            style={{
              background: 'none', border: 'none', padding: 0, cursor: 'pointer',
              color: 'hsl(var(--accent))', opacity: 0.7,
              display: 'flex', alignItems: 'center'
            }}
          >
            <X size={12} />
          </button>
        </div>
      ))}
      <input
        value={input}
        onChange={e => setInput(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder={tags.length === 0 ? placeholder : ''}
        style={{
          flex: 1,
          minWidth: '120px',
          border: 'none',
          background: 'transparent',
          outline: 'none',
          fontSize: '0.9rem',
          fontFamily: 'Inter, sans-serif',
          color: 'hsl(var(--ink))'
        }}
      />
    </div>
  )
}
