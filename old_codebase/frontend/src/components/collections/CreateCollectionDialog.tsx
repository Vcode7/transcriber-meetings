import React, { useState, useEffect } from 'react'
import { X, Loader } from 'lucide-react'
import type { Collection } from '../../types/recording'
import { createCollection, updateCollection } from '../../api/collections'

interface CreateCollectionDialogProps {
  open: boolean
  onClose: () => void
  onCreated: (c: Collection) => void
  editCollection?: Collection | null
}

export default function CreateCollectionDialog({ open, onClose, onCreated, editCollection }: CreateCollectionDialogProps) {
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (open) {
      if (editCollection) {
        setName(editCollection.name)
        setDescription(editCollection.description)
      } else {
        setName('')
        setDescription('')
      }
      setError(null)
    }
  }, [open, editCollection])

  if (!open) return null

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!name.trim()) return

    setLoading(true)
    setError(null)

    try {
      if (editCollection) {
        const updated = await updateCollection(editCollection.id, {
          name: name.trim(),
          description: description.trim()
        })
        onCreated(updated)
      } else {
        const created = await createCollection(name.trim(), description.trim())
        onCreated(created)
      }
    } catch (err: any) {
      if (err.response?.status === 409) {
        setError('A collection with this name already exists.')
      } else {
        setError('Failed to save collection. Please try again.')
      }
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{
      position: 'fixed', inset: 0, zIndex: 1000,
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      padding: '1rem'
    }}>
      <div 
        style={{ position: 'absolute', inset: 0, background: 'hsl(var(--ink) / .3)', backdropFilter: 'blur(4px)' }}
        onClick={onClose}
      />
      <div 
        className="animate-scale-in"
        style={{
          position: 'relative', width: '100%', maxWidth: '480px',
          background: 'hsl(var(--card))', borderRadius: '16px',
          boxShadow: '0 24px 48px hsl(var(--ink) / .15)',
          border: '1px solid hsl(var(--border) / .1)',
          overflow: 'hidden'
        }}
      >
        <div style={{ 
          padding: '1.5rem', display: 'flex', alignItems: 'center', 
          justifyContent: 'space-between', borderBottom: '1px solid hsl(var(--border) / .1)' 
        }}>
          <h2 style={{ fontSize: '1.25rem', fontWeight: 600, color: 'hsl(var(--ink))', margin: 0 }}>
            {editCollection ? 'Edit Collection' : 'Create Collection'}
          </h2>
          <button onClick={onClose} className="icon-btn" style={{ width: '32px', height: '32px' }}>
            <X size={18} />
          </button>
        </div>

        <form onSubmit={handleSubmit} style={{ padding: '1.5rem' }}>
          {error && (
            <div style={{ 
              marginBottom: '1rem', padding: '.75rem 1rem', 
              background: 'hsl(var(--destructive) / .1)', border: '1px solid hsl(var(--destructive) / .2)',
              borderRadius: '8px', color: 'hsl(var(--destructive))', fontSize: '.9rem', fontWeight: 500
            }}>
              {error}
            </div>
          )}

          <div style={{ marginBottom: '1.25rem' }}>
            <label style={{ display: 'block', marginBottom: '.5rem', fontSize: '.9rem', fontWeight: 500, color: 'hsl(var(--ink))' }}>
              Name <span style={{ color: 'hsl(var(--destructive))' }}>*</span>
            </label>
            <input
              className="input"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g. Q3 Planning, Weekly Standups"
              required
              autoFocus
              style={{ width: '100%' }}
            />
          </div>

          <div style={{ marginBottom: '2rem' }}>
            <label style={{ display: 'block', marginBottom: '.5rem', fontSize: '.9rem', fontWeight: 500, color: 'hsl(var(--ink))' }}>
              Description <span style={{ color: 'hsl(var(--pencil))', fontWeight: 400 }}>(Optional)</span>
            </label>
            <textarea
              className="input"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="What is this collection about?"
              rows={3}
              style={{ width: '100%', resize: 'none' }}
            />
          </div>

          <div style={{ display: 'flex', gap: '.75rem', justifyContent: 'flex-end' }}>
            <button type="button" className="btn btn-ghost" onClick={onClose} disabled={loading}>
              Cancel
            </button>
            <button type="submit" className="btn btn-primary" disabled={loading || !name.trim()}>
              {loading && <Loader size={16} className="spin" />}
              {editCollection ? 'Save Changes' : 'Create Collection'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}
