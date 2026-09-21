import React, { useState, useEffect } from 'react'
import { X, Search, Loader, FileAudio, Calendar, Mic } from 'lucide-react'
import api from '../../api/client'
import { addMeetingsToCollection } from '../../api/collections'

interface AddMeetingsDialogProps {
  open: boolean
  collectionId: string
  existingMeetingIds: string[]
  onClose: () => void
  onAdded: () => void
}

interface HistoryItem {
  id: string
  filename: string
  duration: number
  status: string
  speakers_detected: string[]
  has_summary: boolean
  created_at: string
}

function fmtDuration(s: number) {
  const hours = Math.floor(s / 3600)
  const mins = Math.floor((s % 3600) / 60)
  const secs = Math.floor(s % 60)
  if (hours > 0) return `${hours}h ${mins}m`
  return `${mins}m ${secs}s`
}

function fmtDate(iso: string) {
  const date = new Date(iso)
  const now = new Date()
  const todayStart = new Date(now.getFullYear(), now.getMonth(), now.getDate())
  const dateStart = new Date(date.getFullYear(), date.getMonth(), date.getDate())
  const diffDays = Math.floor((todayStart.getTime() - dateStart.getTime()) / (1000 * 60 * 60 * 24))
  if (diffDays === 0) return 'Today'
  if (diffDays === 1) return 'Yesterday'
  if (diffDays > 0 && diffDays < 7) return `${diffDays} days ago`
  return date.toLocaleDateString(undefined, {
    month: 'short', day: 'numeric',
    year: date.getFullYear() !== now.getFullYear() ? 'numeric' : undefined
  })
}

export default function AddMeetingsDialog({ open, collectionId, existingMeetingIds, onClose, onAdded }: AddMeetingsDialogProps) {
  const [meetings, setMeetings] = useState<HistoryItem[]>([])
  const [loading, setLoading] = useState(true)
  const [submitting, setSubmitting] = useState(false)
  const [query, setQuery] = useState('')
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set())

  useEffect(() => {
    if (open) {
      setQuery('')
      setSelectedIds(new Set())
      fetchMeetings()
    }
  }, [open])

  const fetchMeetings = async () => {
    setLoading(true)
    try {
      const res = await api.get('/history')
      const allMeetings = res.data as HistoryItem[]
      setMeetings(allMeetings.filter(m => !existingMeetingIds.includes(m.id)))
    } catch (err) {
      console.error('Failed to fetch meetings', err)
    } finally {
      setLoading(false)
    }
  }

  if (!open) return null

  const filtered = query.trim()
    ? meetings.filter(m => m.filename.toLowerCase().includes(query.toLowerCase()))
    : meetings

  const toggleSelect = (id: string) => {
    const newSet = new Set(selectedIds)
    if (newSet.has(id)) newSet.delete(id)
    else newSet.add(id)
    setSelectedIds(newSet)
  }

  const toggleAll = () => {
    if (selectedIds.size === filtered.length) {
      setSelectedIds(new Set())
    } else {
      setSelectedIds(new Set(filtered.map(m => m.id)))
    }
  }

  const handleSubmit = async () => {
    if (selectedIds.size === 0) return
    setSubmitting(true)
    try {
      await addMeetingsToCollection(collectionId, Array.from(selectedIds))
      onAdded()
      onClose()
    } catch (err) {
      console.error('Failed to add meetings', err)
    } finally {
      setSubmitting(false)
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
          position: 'relative', width: '100%', maxWidth: '640px', maxHeight: '90vh',
          background: 'hsl(var(--card))', borderRadius: '16px',
          boxShadow: '0 24px 48px hsl(var(--ink) / .15)',
          border: '1px solid hsl(var(--border) / .1)',
          display: 'flex', flexDirection: 'column'
        }}
      >
        <div style={{ 
          padding: '1.5rem', display: 'flex', alignItems: 'center', 
          justifyContent: 'space-between', borderBottom: '1px solid hsl(var(--border) / .1)' 
        }}>
          <h2 style={{ fontSize: '1.25rem', fontWeight: 600, color: 'hsl(var(--ink))', margin: 0 }}>
            Add Meetings
          </h2>
          <button onClick={onClose} className="icon-btn" style={{ width: '32px', height: '32px' }}>
            <X size={18} />
          </button>
        </div>

        <div style={{ padding: '1rem 1.5rem', borderBottom: '1px solid hsl(var(--border) / .1)' }}>
          <div style={{ position: 'relative' }}>
            <Search size={16} style={{ position: 'absolute', left: '.85rem', top: '50%', transform: 'translateY(-50%)', color: 'hsl(var(--pencil))' }} />
            <input
              className="input"
              value={query}
              onChange={e => setQuery(e.target.value)}
              placeholder="Search meetings..."
              style={{ width: '100%', paddingLeft: '2.5rem' }}
            />
          </div>
        </div>

        <div style={{ flex: 1, overflowY: 'auto', padding: '1rem', minHeight: '300px' }}>
          {loading ? (
            <div style={{ display: 'flex', justifyContent: 'center', padding: '3rem' }}>
              <Loader size={32} className="spin" style={{ color: 'hsl(var(--accent))' }} />
            </div>
          ) : filtered.length === 0 ? (
            <div style={{ textAlign: 'center', padding: '3rem', color: 'hsl(var(--pencil))' }}>
              No meetings found.
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '.5rem' }}>
              <div style={{ padding: '0 .5rem', marginBottom: '.5rem' }}>
                <label style={{ display: 'flex', alignItems: 'center', gap: '.5rem', cursor: 'pointer', fontSize: '.9rem', fontWeight: 500 }}>
                  <input 
                    type="checkbox" 
                    checked={selectedIds.size === filtered.length && filtered.length > 0}
                    onChange={toggleAll}
                  />
                  Select All
                </label>
              </div>
              {filtered.map(m => (
                <div 
                  key={m.id}
                  onClick={() => toggleSelect(m.id)}
                  style={{
                    display: 'flex', alignItems: 'center', gap: '1rem', padding: '.75rem 1rem',
                    background: selectedIds.has(m.id) ? 'hsl(var(--accent) / .05)' : 'hsl(var(--paper))',
                    border: `1px solid ${selectedIds.has(m.id) ? 'hsl(var(--accent) / .3)' : 'hsl(var(--border) / .1)'}`,
                    borderRadius: '8px', cursor: 'pointer', transition: 'all .2s'
                  }}
                >
                  <input type="checkbox" checked={selectedIds.has(m.id)} onChange={() => {}} style={{ pointerEvents: 'none' }} />
                  <div style={{ width: '32px', height: '32px', borderRadius: '8px', background: 'hsl(var(--accent) / .1)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <Mic size={14} style={{ color: 'hsl(var(--accent))' }} />
                  </div>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ fontWeight: 600, fontSize: '.95rem', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                      {m.filename}
                    </div>
                    <div style={{ display: 'flex', gap: '1rem', fontSize: '.8rem', color: 'hsl(var(--pencil))' }}>
                      <span style={{ display: 'flex', alignItems: 'center', gap: '.25rem' }}><Calendar size={12}/> {fmtDate(m.created_at)}</span>
                      {m.duration > 0 && <span style={{ display: 'flex', alignItems: 'center', gap: '.25rem' }}><FileAudio size={12}/> {fmtDuration(m.duration)}</span>}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        <div style={{ 
          padding: '1.25rem 1.5rem', borderTop: '1px solid hsl(var(--border) / .1)',
          display: 'flex', justifyContent: 'space-between', alignItems: 'center'
        }}>
          <span style={{ fontSize: '.9rem', fontWeight: 500, color: 'hsl(var(--pencil))' }}>
            {selectedIds.size} selected
          </span>
          <div style={{ display: 'flex', gap: '.75rem' }}>
            <button className="btn btn-ghost" onClick={onClose} disabled={submitting}>Cancel</button>
            <button className="btn btn-primary" onClick={handleSubmit} disabled={submitting || selectedIds.size === 0}>
              {submitting && <Loader size={16} className="spin" />}
              Add Selected
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
