import React, { useState } from 'react'
import { Plus, X } from 'lucide-react'

import { ImportanceBadge } from './AgendaActionPoints'
import { hiddenByImportanceText, passesImportance } from '../lib/actionPoints'
import type { ActionImportance } from '../lib/actionPoints'

export interface ActionItem extends ActionImportance {
  task: string
  owner: string
  deadline: string
}

interface Props {
  items: ActionItem[]
  onChange: (items: ActionItem[]) => void
  /** Items scored below this importance are hidden until "Show all" (0 = show all). */
  minImportance?: number
}

export default function ActionItemsTable({ items, onChange, minImportance = 0 }: Props) {
  const [showAll, setShowAll] = useState(false)
  const hasImportance = items.some(item => item.importance !== undefined && item.importance !== null)
  // Rows keep their index in `items`, so edits and removals of a filtered list hit the right item.
  const rows = items.map((item, idx) => ({ item, idx })).filter(({ item }) => showAll || passesImportance(item, minImportance))
  const hidden = items.length - items.filter(item => passesImportance(item, minImportance)).length

  const handleUpdate = (index: number, field: 'task' | 'owner' | 'deadline', value: string) => {
    const newItems = [...items]
    newItems[index] = { ...newItems[index], [field]: value }
    onChange(newItems)
  }

  const handleAdd = () => {
    onChange([...items, { task: '', owner: '', deadline: '' }])
  }

  const handleRemove = (index: number) => {
    onChange(items.filter((_, i) => i !== index))
  }

  return (
    <div>
      {hidden > 0 && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: '0.5rem', fontSize: '0.8rem', color: 'hsl(var(--pencil))', fontFamily: 'Inter, sans-serif' }}>
          <span style={{ fontStyle: 'italic' }}>
            {showAll ? `Showing all, including ${hidden} below ${minImportance}% importance` : hiddenByImportanceText(hidden, minImportance)}
          </span>
          <button className="btn btn-ghost" style={{ fontSize: '0.78rem', padding: '0.15rem 0.5rem' }} onClick={() => setShowAll(v => !v)}>
            {showAll ? 'Hide them' : 'Show all'}
          </button>
        </div>
      )}
      <table style={{ width: '100%', borderCollapse: 'collapse', marginBottom: '1rem', fontFamily: 'Inter, sans-serif' }}>
        <thead>
          <tr style={{ borderBottom: '1px solid hsl(var(--border) / .5)' }}>
            <th style={{ textAlign: 'left', padding: '0.5rem', color: 'hsl(var(--pencil))', fontSize: '0.82rem', fontWeight: 600 }}>Task</th>
            <th style={{ textAlign: 'left', padding: '0.5rem', color: 'hsl(var(--pencil))', fontSize: '0.82rem', fontWeight: 600, width: '20%' }}>Owner</th>
            <th style={{ textAlign: 'left', padding: '0.5rem', color: 'hsl(var(--pencil))', fontSize: '0.82rem', fontWeight: 600, width: '20%' }}>Deadline</th>
            {hasImportance && <th style={{ textAlign: 'left', padding: '0.5rem', color: 'hsl(var(--pencil))', fontSize: '0.82rem', fontWeight: 600, width: '90px' }}>Importance</th>}
            <th style={{ width: '40px' }}></th>
          </tr>
        </thead>
        <tbody>
          {rows.map(({ item, idx }) => (
            <tr key={idx} style={{ borderBottom: '1px solid hsl(var(--border) / .3)' }}>
              <td style={{ padding: '0.5rem' }}>
                <input 
                  className="input" 
                  value={item.task} 
                  onChange={e => handleUpdate(idx, 'task', e.target.value)} 
                  placeholder="Task description"
                  style={{ width: '100%', padding: '0.4rem 0.6rem', fontSize: '0.9rem' }}
                />
              </td>
              <td style={{ padding: '0.5rem' }}>
                <input 
                  className="input" 
                  value={item.owner} 
                  onChange={e => handleUpdate(idx, 'owner', e.target.value)} 
                  placeholder="Owner"
                  style={{ width: '100%', padding: '0.4rem 0.6rem', fontSize: '0.9rem' }}
                />
              </td>
              <td style={{ padding: '0.5rem' }}>
                <input 
                  className="input" 
                  value={item.deadline} 
                  onChange={e => handleUpdate(idx, 'deadline', e.target.value)} 
                  placeholder="Date"
                  style={{ width: '100%', padding: '0.4rem 0.6rem', fontSize: '0.9rem' }}
                />
              </td>
              {hasImportance && (
                <td style={{ padding: '0.5rem' }}>
                  <ImportanceBadge item={item} />
                </td>
              )}
              <td style={{ padding: '0.5rem', textAlign: 'center' }}>
                <button 
                  onClick={() => handleRemove(idx)}
                  className="icon-btn"
                  style={{ color: 'hsl(var(--destructive))', width: '28px', height: '28px' }}
                  title="Remove item"
                >
                  <X size={14} />
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      
      <button 
        onClick={handleAdd}
        className="btn btn-ghost"
        style={{ fontSize: '0.82rem', padding: '0.4rem 0.8rem', gap: '6px' }}
      >
        <Plus size={14} /> Add Action Item
      </button>
    </div>
  )
}
