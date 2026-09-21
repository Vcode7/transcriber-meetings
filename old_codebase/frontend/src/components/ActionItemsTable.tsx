import React from 'react'
import { Plus, X } from 'lucide-react'

export interface ActionItem {
  task: string
  owner: string
  deadline: string
}

interface Props {
  items: ActionItem[]
  onChange: (items: ActionItem[]) => void
}

export default function ActionItemsTable({ items, onChange }: Props) {
  
  const handleUpdate = (index: number, field: keyof ActionItem, value: string) => {
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
      <table style={{ width: '100%', borderCollapse: 'collapse', marginBottom: '1rem', fontFamily: 'Inter, sans-serif' }}>
        <thead>
          <tr style={{ borderBottom: '1px solid hsl(var(--border) / .5)' }}>
            <th style={{ textAlign: 'left', padding: '0.5rem', color: 'hsl(var(--pencil))', fontSize: '0.82rem', fontWeight: 600 }}>Task</th>
            <th style={{ textAlign: 'left', padding: '0.5rem', color: 'hsl(var(--pencil))', fontSize: '0.82rem', fontWeight: 600, width: '20%' }}>Owner</th>
            <th style={{ textAlign: 'left', padding: '0.5rem', color: 'hsl(var(--pencil))', fontSize: '0.82rem', fontWeight: 600, width: '20%' }}>Deadline</th>
            <th style={{ width: '40px' }}></th>
          </tr>
        </thead>
        <tbody>
          {items.map((item, idx) => (
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
