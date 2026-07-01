import React, { useState } from 'react'
import { ChevronDown, ChevronUp } from 'lucide-react'

interface MomSectionProps {
  title: string
  children: React.ReactNode
  defaultOpen?: boolean
  headerRight?: React.ReactNode
  className?: string
}

export default function MomSection({ title, children, defaultOpen = true, headerRight, className }: MomSectionProps) {
  const [open, setOpen] = useState(defaultOpen)

  return (
    <div className={className} style={{
      background: 'hsl(var(--card))',
      borderRadius: '12px',
      border: '1px solid hsl(var(--border) / .5)',
      overflow: 'hidden',
      marginBottom: '1rem',
      boxShadow: '0 2px 8px hsl(var(--border) / .1)'
    }}>
      <div 
        onClick={() => setOpen(!open)}
        style={{
          padding: '1rem 1.25rem',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          cursor: 'pointer',
          borderBottom: open ? '1px solid hsl(var(--border) / .3)' : 'none',
          position: 'relative'
        }}
      >
        {/* Accent left border */}
        <div style={{
          position: 'absolute',
          left: 0, top: '15%', bottom: '15%', width: '4px',
          background: 'hsl(var(--accent))',
          borderRadius: '0 4px 4px 0',
          opacity: 0.8
        }} />
        
        <h3 style={{ 
          fontSize: '1rem', 
          fontWeight: 600, 
          color: 'hsl(var(--ink))',
          margin: 0,
          fontFamily: 'Inter, sans-serif'
        }}>
          {title}
        </h3>
        
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          {headerRight && <div onClick={e => e.stopPropagation()}>{headerRight}</div>}
          <div style={{ color: 'hsl(var(--pencil))' }}>
            {open ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
          </div>
        </div>
      </div>
      
      {open && (
        <div style={{ padding: '1.25rem' }}>
          {children}
        </div>
      )}
    </div>
  )
}
