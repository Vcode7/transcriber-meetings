import React, { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { ArrowLeft, Plus, Trash2, Mic, Clock, FileAudio, Users, Calendar, Sparkles, Loader, ChevronRight, GripVertical, X, Bot } from 'lucide-react'
import type { CollectionDetail, CollectionMeeting } from '../../types/recording'
import { deleteCollection, removeMeetingsFromCollection, reorderMeetings } from '../../api/collections'
import InlineEdit from '../InlineEdit'
import AddMeetingsDialog from './AddMeetingsDialog'
import CollectionAIChat from './CollectionAIChat'

interface CollectionDetailViewProps {
  collection: CollectionDetail
  onBack: () => void
  onDeleted: () => void
  onUpdated: () => void
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

function fmtTime(iso: string) {
  const date = new Date(iso)
  return date.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit', hour12: true })
}

const SPEAKER_PALETTE = ['#f4623a', '#3b9ede', '#34a853', '#9c59d1', '#f5a623', '#e91e8c']
function getSpeakerColor(name: string) {
  let hash = 0
  for (const c of name) hash = (hash * 31 + c.charCodeAt(0)) & 0xffff
  return SPEAKER_PALETTE[hash % SPEAKER_PALETTE.length]
}

export default function CollectionDetailView({ collection, onBack, onDeleted, onUpdated }: CollectionDetailViewProps) {
  const navigate = useNavigate()
  const [addDialogOpen, setAddDialogOpen] = useState(false)
  const [deleting, setDeleting] = useState(false)
  const [confirmDelete, setConfirmDelete] = useState(false)
  const [sortMode, setSortMode] = useState<'manual' | 'newest' | 'oldest'>('manual')
  const [showAIChat, setShowAIChat] = useState(false)
  
  const [meetings, setMeetings] = useState<CollectionMeeting[]>(collection.meetings)
  const [draggedIdx, setDraggedIdx] = useState<number | null>(null)

  useEffect(() => {
    setMeetings(collection.meetings)
  }, [collection.meetings])

  const handleDeleteCollection = async () => {
    setDeleting(true)
    try {
      await deleteCollection(collection.id)
      onDeleted()
    } catch (err) {
      console.error(err)
      setDeleting(false)
    }
  }

  const handleRemoveMeeting = async (e: React.MouseEvent, meetingId: string) => {
    e.stopPropagation()
    try {
      await removeMeetingsFromCollection(collection.id, [meetingId])
      setMeetings(prev => prev.filter(m => m.id !== meetingId))
      onUpdated()
    } catch (err) {
      console.error(err)
    }
  }

  const handleDragStart = (e: React.DragEvent, index: number) => {
    setDraggedIdx(index)
    e.dataTransfer.effectAllowed = 'move'
  }

  const handleDragOver = (e: React.DragEvent, index: number) => {
    e.preventDefault()
    e.dataTransfer.dropEffect = 'move'
    if (draggedIdx === null || draggedIdx === index) return

    const newMeetings = [...meetings]
    const draggedItem = newMeetings[draggedIdx]
    newMeetings.splice(draggedIdx, 1)
    newMeetings.splice(index, 0, draggedItem)
    setMeetings(newMeetings)
    setDraggedIdx(index)
  }

  const handleDrop = async () => {
    setDraggedIdx(null)
    const ids = meetings.map(m => m.id)
    try {
      await reorderMeetings(collection.id, ids)
    } catch (err) {
      console.error(err)
      // On error could restore previous state
    }
  }

  let displayMeetings = [...meetings]
  if (sortMode === 'newest') {
    displayMeetings.sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime())
  } else if (sortMode === 'oldest') {
    displayMeetings.sort((a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime())
  }

  return (
    <div style={{ display: 'flex', height: '100%', width: '100%', overflow: 'hidden' }}>
      {/* Left Pane - Collection Meetings */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', height: '100%', padding: '1.5rem 2rem 2rem', overflowY: 'auto', minWidth: 0 }}>
        <div style={{ display: 'flex', alignItems: 'flex-start', gap: '1rem', marginBottom: '2rem' }}>
          <button onClick={onBack} className="icon-btn" style={{ width: '40px', height: '40px', background: 'hsl(var(--card))', border: '1px solid hsl(var(--border) / .2)' }}>
            <ArrowLeft size={18} />
          </button>
          <div style={{ flex: 1 }}>
            <h2 style={{ fontSize: '1.5rem', fontWeight: 700, margin: '0 0 .25rem 0', color: 'hsl(var(--ink))' }}>
              {collection.name}
            </h2>
            {collection.description && (
              <p style={{ color: 'hsl(var(--pencil))', margin: '0 0 .5rem 0', fontSize: '.95rem' }}>
                {collection.description}
              </p>
            )}
            <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
              <span style={{ fontSize: '.85rem', color: 'hsl(var(--accent))', fontWeight: 600, background: 'hsl(var(--accent) / .1)', padding: '.25rem .75rem', borderRadius: '8px' }}>
                {meetings.length} Meetings
              </span>
            </div>
          </div>
          <div style={{ display: 'flex', gap: '.5rem', alignItems: 'center' }}>
            <button 
              className="btn btn-ghost" 
              onClick={() => setShowAIChat(!showAIChat)} 
              style={{ 
                color: showAIChat ? 'hsl(var(--accent))' : 'hsl(var(--pencil))',
                background: showAIChat ? 'hsl(var(--accent) / .1)' : 'transparent',
                borderColor: showAIChat ? 'hsl(var(--accent) / .25)' : 'transparent',
                borderWidth: '1.5px',
                borderStyle: 'solid',
                padding: '.5rem 1rem',
              }}
            >
              <Bot size={16} style={{ marginRight: '.5rem' }} /> Collection AI
            </button>

            {confirmDelete ? (
              <div style={{ display: 'flex', alignItems: 'center', gap: '.5rem', background: 'hsl(var(--card))', padding: '.25rem', borderRadius: '8px', border: '1px solid hsl(var(--destructive) / .3)' }}>
                <span style={{ fontSize: '.85rem', fontWeight: 600, paddingLeft: '.5rem' }}>Delete?</span>
                <button className="btn" onClick={handleDeleteCollection} disabled={deleting} style={{ background: 'hsl(var(--destructive))', color: 'white', padding: '.4rem .8rem', minHeight: '0' }}>
                  {deleting ? <Loader size={14} className="spin" /> : 'Yes'}
                </button>
                <button className="btn btn-ghost" onClick={() => setConfirmDelete(false)} style={{ padding: '.4rem .8rem', minHeight: '0' }}>No</button>
              </div>
            ) : (
              <button className="btn btn-ghost" onClick={() => setConfirmDelete(true)} style={{ color: 'hsl(var(--destructive))' }}>
                <Trash2 size={16} style={{ marginRight: '.5rem' }} /> Delete Collection
              </button>
            )}
          </div>
        </div>

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '.5rem' }}>
          <span style={{ fontSize: '.9rem', fontWeight: 500, color: 'hsl(var(--pencil))' }}>Sort:</span>
          <select 
            className="input" 
            value={sortMode} 
            onChange={e => setSortMode(e.target.value as any)}
            style={{ height: '32px', padding: '0 2rem 0 .75rem', fontSize: '.85rem' }}
          >
            <option value="manual">Manual Order</option>
            <option value="newest">Newest First</option>
            <option value="oldest">Oldest First</option>
          </select>
        </div>
        <button className="btn btn-primary" onClick={() => setAddDialogOpen(true)} style={{ padding: '.5rem 1rem' }}>
          <Plus size={16} style={{ marginRight: '.5rem' }} /> Add Meetings
        </button>
      </div>

      {meetings.length === 0 ? (
        <div style={{ textAlign: 'center', padding: '4rem 2rem', background: 'hsl(var(--card))', borderRadius: '16px', border: '1px dashed hsl(var(--border))' }}>
          <div style={{ width: '64px', height: '64px', background: 'hsl(var(--accent) / .1)', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 1.5rem' }}>
            <Plus size={24} style={{ color: 'hsl(var(--accent))' }} />
          </div>
          <h3 style={{ fontSize: '1.25rem', fontWeight: 600, marginBottom: '.5rem' }}>Empty Collection</h3>
          <p style={{ color: 'hsl(var(--pencil))', marginBottom: '1.5rem' }}>Add some meetings to this collection to organize them.</p>
          <button className="btn btn-primary" onClick={() => setAddDialogOpen(true)}>
            Add Meetings
          </button>
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          {displayMeetings.map((item, index) => (
            <div
              key={item.id}
              draggable={sortMode === 'manual'}
              onDragStart={e => handleDragStart(e, index)}
              onDragOver={e => handleDragOver(e, index)}
              onDrop={handleDrop}
              onDragEnd={() => setDraggedIdx(null)}
              onClick={() => navigate(`/dashboard/history/${item.id}`)}
              style={{
                background: 'hsl(var(--card))', border: '2px solid hsl(var(--border) / .15)',
                borderRadius: '14px', padding: '1.25rem 1.5rem', cursor: 'pointer',
                transition: 'all .2s cubic-bezier(0.4, 0, 0.2, 1)',
                position: 'relative', overflow: 'hidden', boxShadow: '0 1px 3px hsl(var(--ink) / .04)',
                opacity: draggedIdx === index ? 0.5 : 1,
                display: 'flex', alignItems: 'flex-start', gap: '1rem'
              }}
              onMouseEnter={e => {
                const el = e.currentTarget as HTMLDivElement
                el.style.borderColor = 'hsl(var(--accent) / .35)'
                el.style.transform = 'translateY(-2px)'
              }}
              onMouseLeave={e => {
                const el = e.currentTarget as HTMLDivElement
                el.style.borderColor = 'hsl(var(--border) / .15)'
                el.style.transform = 'translateY(0)'
              }}
            >
              {sortMode === 'manual' && (
                <div style={{ padding: '.5rem 0', color: 'hsl(var(--pencil))', cursor: 'grab' }}>
                  <GripVertical size={20} />
                </div>
              )}
              
              <div style={{ width: '42px', height: '42px', borderRadius: '10px', flexShrink: 0, background: 'linear-gradient(135deg, hsl(var(--accent) / .15), hsl(var(--accent) / .08))', display: 'flex', alignItems: 'center', justifyContent: 'center', border: '2px solid hsl(var(--accent) / .25)' }}>
                <Mic size={18} style={{ color: 'hsl(var(--accent))' }} />
              </div>

              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ fontWeight: 700, fontSize: '1.05rem', marginBottom: '.5rem', color: 'hsl(var(--ink))' }}>
                  {item.filename}
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '.75rem', flexWrap: 'wrap', marginBottom: '.75rem' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '.4rem', fontSize: '.8rem', color: 'hsl(var(--pencil))', fontWeight: 500 }}>
                    <Calendar size={13} /> <span>{fmtDate(item.created_at)}</span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '.4rem', fontSize: '.8rem', color: 'hsl(var(--pencil))', fontWeight: 500 }}>
                    <Clock size={13} /> <span>{fmtTime(item.created_at)}</span>
                  </div>
                </div>

                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '.6rem', alignItems: 'center' }}>
                  <span className={`status-badge ${item.status === 'done' ? 'done' : item.status === 'error' ? 'error' : 'processing'}`} style={{ fontSize: '.75rem', padding: '.3rem .7rem', fontWeight: 600 }}>
                    {item.status === 'done' ? '✓ Complete' : item.status === 'error' ? '✕ Error' : '⟳ Processing'}
                  </span>

                  {item.has_summary && (
                    <span className="status-badge" style={{ fontSize: '.75rem', padding: '.3rem .7rem', background: 'linear-gradient(135deg, hsl(235, 75%, 65% / .15), hsl(235, 75%, 65% / .08))', borderColor: 'hsl(235, 75%, 65% / .3)', color: 'hsl(235, 75%, 55%)', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '.35rem' }}>
                      <Sparkles size={11} /> AI Summary
                    </span>
                  )}

                  {item.duration > 0 && (
                    <span style={{ display: 'inline-flex', alignItems: 'center', gap: '.4rem', fontSize: '.78rem', fontWeight: 600, background: 'hsl(var(--muted))', color: 'hsl(var(--pencil))', border: '1.5px solid hsl(var(--border) / .2)', padding: '.3rem .7rem', borderRadius: '8px' }}>
                      <FileAudio size={12} /> {fmtDuration(item.duration)}
                    </span>
                  )}

                  {item.speakers_detected.length > 0 && (
                    <div style={{ display: 'flex', alignItems: 'center', gap: '.4rem' }}>
                      {item.speakers_detected.slice(0, 3).map((sp, si) => {
                        const col = getSpeakerColor(sp)
                        return (
                          <span key={si} style={{ display: 'inline-flex', alignItems: 'center', gap: '.4rem', padding: '.25rem .65rem .25rem .35rem', borderRadius: '999px', fontSize: '.76rem', fontWeight: 600, background: `${col}15`, border: `1.5px solid ${col}40`, color: col }}>
                            <span style={{ width: '18px', height: '18px', borderRadius: '50%', background: col, color: '#fff', fontSize: '.65rem', fontWeight: 800, display: 'inline-flex', alignItems: 'center', justifyContent: 'center' }}>
                              {sp.charAt(0).toUpperCase()}
                            </span>
                            {sp}
                          </span>
                        )
                      })}
                    </div>
                  )}
                </div>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: '.5rem' }}>
                <button 
                  onClick={(e) => handleRemoveMeeting(e, item.id)}
                  className="icon-btn" 
                  style={{ width: '32px', height: '32px', color: 'hsl(var(--pencil))' }}
                  title="Remove from collection"
                >
                  <X size={16} />
                </button>
                <ChevronRight size={18} style={{ color: 'hsl(var(--accent))' }} />
              </div>
            </div>
          ))}
        </div>
      )}

      {addDialogOpen && (
        <AddMeetingsDialog 
          open={addDialogOpen}
          collectionId={collection.id}
          existingMeetingIds={meetings.map(m => m.id)}
          onClose={() => setAddDialogOpen(false)}
          onAdded={() => {
            onUpdated()
            // reload collection detail? We rely on parent to pass updated data, but since we manage meetings locally for drag and drop:
            // The simplest approach is to trigger onUpdated which re-fetches in parent.
          }}
        />
      )}
      </div>

      {/* Right Pane - Collection AI Chat */}
      {showAIChat && (
        <div style={{ width: '420px', height: '100%', flexShrink: 0 }}>
          <CollectionAIChat
            collectionId={collection.id}
            meetings={meetings}
            onClose={() => setShowAIChat(false)}
          />
        </div>
      )}
    </div>
  )
}
