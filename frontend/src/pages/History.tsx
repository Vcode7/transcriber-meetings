import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  History as HistoryIcon, Clock, FileAudio, Mic,
  Loader, Trash2, ChevronRight, Search, X, Users, Calendar, Sparkles
} from 'lucide-react'
import api from '../api/client'
import InlineEdit from '../components/InlineEdit'

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
  
  if (hours > 0) {
    return `${hours}h ${mins}m`
  }
  return `${mins}m ${secs}s`
}

function fmtDate(iso: string) {
  const date = new Date(iso)
  const now = new Date()
  
  // Compare dates at midnight to properly detect "today" vs "yesterday"
  const todayStart = new Date(now.getFullYear(), now.getMonth(), now.getDate())
  const dateStart = new Date(date.getFullYear(), date.getMonth(), date.getDate())
  const diffDays = Math.floor((todayStart.getTime() - dateStart.getTime()) / (1000 * 60 * 60 * 24))
  
  if (diffDays === 0) return 'Today'
  if (diffDays === 1) return 'Yesterday'
  if (diffDays > 0 && diffDays < 7) return `${diffDays} days ago`
  
  // For older dates, show formatted date
  return date.toLocaleDateString(undefined, { 
    month: 'short', 
    day: 'numeric', 
    year: date.getFullYear() !== now.getFullYear() ? 'numeric' : undefined
  })
}

function fmtTime(iso: string) {
  const date = new Date(iso)
  return date.toLocaleTimeString(undefined, { 
    hour: '2-digit', 
    minute: '2-digit',
    hour12: true // or false for 24-hour format
  })
}

const SPEAKER_PALETTE = [
  '#f4623a', '#3b9ede', '#34a853', '#9c59d1', '#f5a623', '#e91e8c',
]

function getSpeakerColor(name: string) {
  let hash = 0
  for (const c of name) hash = (hash * 31 + c.charCodeAt(0)) & 0xffff
  return SPEAKER_PALETTE[hash % SPEAKER_PALETTE.length]
}

export default function HistoryPage() {
  const [items, setItems] = useState<HistoryItem[]>([])
  const [loading, setLoading] = useState(true)
  const [deletingId, setDeletingId] = useState<string | null>(null)
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null)
  const [query, setQuery] = useState('')
  const navigate = useNavigate()

  const load = async () => {
    setLoading(true)
    try {
      const res = await api.get('/history')
      setItems(res.data)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [])

  const handleDeleteClick = (id: string, e: React.MouseEvent) => {
    e.stopPropagation()
    setConfirmDeleteId(id)
  }

  const handleDeleteConfirm = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation()
    setConfirmDeleteId(null)
    setDeletingId(id)
    try {
      await api.delete(`/history/${id}`)
      setItems((prev) => prev.filter((i) => i.id !== id))
    } finally {
      setDeletingId(null)
    }
  }

  const handleDeleteCancel = (e: React.MouseEvent) => {
    e.stopPropagation()
    setConfirmDeleteId(null)
  }

  const handleRename = async (id: string, newName: string) => {
    await api.patch(`/history/${id}/rename`, { filename: newName })
    setItems((prev) => prev.map((i) => i.id === id ? { ...i, filename: newName } : i))
  }

  const filtered = query.trim()
    ? items.filter((i) =>
        i.filename.toLowerCase().includes(query.toLowerCase()) ||
        i.speakers_detected.some((s) => s.toLowerCase().includes(query.toLowerCase()))
      )
    : items

  return (
    <div className="page-scroll-root" style={{ 
      display: 'flex', 
      flexDirection: 'column', 
      background: 'linear-gradient(to bottom, hsl(var(--paper)), hsl(var(--paper-deep)))'
    }}>

      {/* ═══════════════════════════════════════════════
          HEADER SECTION
      ═══════════════════════════════════════════════ */}
      <div className="panel-header" style={{ flexDirection: 'column', alignItems: 'stretch' }}>
        {/* Title Row */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: items.length > 0 ? '1rem' : 0 }}>
          {/* Icon */}
          <div style={{
            width: '44px', 
            height: '44px', 
            borderRadius: '12px', 
            flexShrink: 0,
            background: 'linear-gradient(135deg, hsl(var(--accent) / .15), hsl(var(--accent) / .08))',
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'center',
            border: '2px solid hsl(var(--accent) / .35)',
          }}>
            <HistoryIcon size={20} style={{ color: 'hsl(var(--accent))' }} />
          </div>

          {/* Title & Description */}
          <div style={{ flex: 1 }}>
            <h1>Recording History</h1>
            <p style={{ fontSize: '.82rem', color: 'hsl(var(--pencil))', fontFamily: 'Inter, sans-serif', fontWeight: 400, marginTop: '1px' }}>
              View, manage, and revisit all your meeting recordings
            </p>
          </div>

          {/* Stats Badge */}
          {items.length > 0 && (
            <div className="animate-scale-in" style={{
              fontSize: '.88rem', 
              fontWeight: 700,
              color: 'hsl(var(--accent))',
              background: 'linear-gradient(135deg, hsl(var(--accent) / .15), hsl(var(--accent) / .08))',
              padding: '.5rem 1rem',
              borderRadius: '10px',
              border: '2px solid hsl(var(--accent) / .3)',
              fontFamily: 'JetBrains Mono, monospace',
            }}>
              {items.length} {items.length === 1 ? 'Recording' : 'Recordings'}
            </div>
          )}
        </div>

        {/* Search Bar */}
        {items.length > 0 && (
          <div style={{ maxWidth: '500px' }}>
            <div style={{ position: 'relative' }}>
              <Search size={16} style={{
                position: 'absolute', 
                left: '.85rem', 
                top: '50%', 
                transform: 'translateY(-50%)',
                color: 'hsl(var(--pencil))', 
                pointerEvents: 'none',
                zIndex: 1
              }} />
              <input
                className="input"
                placeholder="Search by recording name or speaker..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                style={{ 
                  paddingLeft: '2.5rem', 
                  paddingRight: query ? '3rem' : '1rem', 
                  fontSize: '.88rem', 
                  height: '42px',
                  background: 'hsl(var(--card))',
                  border: '2px solid hsl(var(--border) / .2)',
                  fontFamily: 'Inter, sans-serif'
                }}
              />
              {query && (
                <button
                  onClick={() => setQuery('')}
                  className="icon-btn"
                  style={{
                    position: 'absolute', 
                    right: '.4rem', 
                    top: '50%', 
                    transform: 'translateY(-50%)',
                    width: '32px',
                    height: '32px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center'
                  }}
                  title="Clear search"
                >
                  <X size={14} />
                </button>
              )}
            </div>
          </div>
        )}
      </div>

      {/* ═══════════════════════════════════════════════
          RECORDINGS LIST (SCROLLABLE)
      ═══════════════════════════════════════════════ */}
      <div style={{
        flex: 1, 
        overflowY: 'auto',
        padding: '1.5rem 2rem 2rem',
        minHeight: 0
      }}>

        {/* Loading State */}
        {loading && (
          <div style={{ 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'center', 
            padding: '6rem 2rem',
            flexDirection: 'column', 
            gap: '1.5rem' 
          }}>
            <Loader size={36} className="spin" style={{ color: 'hsl(var(--accent))' }} />
            <p style={{ 
              fontFamily: 'Inter, sans-serif', 
              fontSize: '1rem',
              color: 'hsl(var(--pencil))',
              fontWeight: 500
            }}>
              Loading your recordings...
            </p>
          </div>
        )}

        {/* Empty State */}
        {!loading && items.length === 0 && (
          <div style={{ 
            textAlign: 'center', 
            padding: '6rem 2rem',
            maxWidth: '500px',
            margin: '0 auto'
          }}>
            <div style={{
              width: '120px', 
              height: '120px', 
              borderRadius: '50%',
              background: 'linear-gradient(135deg, hsl(var(--accent) / .1), hsl(var(--accent) / .05))',
              border: '3px dashed hsl(var(--accent) / .3)',
              display: 'flex', 
              alignItems: 'center', 
              justifyContent: 'center',
              margin: '0 auto 2rem',
              boxShadow: '0 0 0 16px hsl(var(--accent) / .04)',
            }}>
              <Mic size={48} style={{ color: 'hsl(var(--accent))', opacity: 0.5 }} className="animate-float" />
            </div>
            <h2 style={{ 
              fontSize: '1.75rem', 
              fontWeight: 700,
              marginBottom: '1rem', 
              fontFamily: 'Inter, sans-serif', 
              color: 'hsl(var(--ink))',
              letterSpacing: '-.02em'
            }}>
              No Recordings Yet
            </h2>
            <p style={{ 
              fontSize: '1rem',
              color: 'hsl(var(--pencil))',
              fontFamily: 'Inter, sans-serif', 
              lineHeight: 1.7,
              marginBottom: '2rem'
            }}>
              Start capturing your meetings and conversations. Your recordings will appear here for easy access and management.
            </p>
            <button 
              className="btn btn-primary" 
              onClick={() => navigate('/dashboard/record')} 
              style={{ 
                padding: '.85rem 2rem', 
                fontSize: '1rem',
                gap: '.75rem',
                boxShadow: '0 4px 12px hsl(var(--accent) / .3)'
              }}
            >
              <Mic size={18} /> Start Your First Recording
            </button>
          </div>
        )}

        {/* No Search Results */}
        {!loading && items.length > 0 && filtered.length === 0 && (
          <div style={{ 
            textAlign: 'center', 
            padding: '4rem 2rem',
            maxWidth: '450px',
            margin: '0 auto'
          }}>
            <div style={{
              fontSize: '3rem',
              marginBottom: '1rem',
              opacity: 0.3
            }}>
              🔍
            </div>
            <h3 style={{
              fontSize: '1.3rem',
              fontWeight: 700,
              marginBottom: '.75rem',
              fontFamily: 'Inter, sans-serif',
              color: 'hsl(var(--ink))'
            }}>
              No matches found
            </h3>
            <p style={{ 
              fontFamily: 'Inter, sans-serif', 
              fontSize: '.95rem',
              color: 'hsl(var(--pencil))',
              marginBottom: '1.5rem',
              lineHeight: 1.6
            }}>
              No recordings match "<strong>{query}</strong>". Try a different search term.
            </p>
            <button 
              onClick={() => setQuery('')} 
              className="btn btn-ghost" 
              style={{ 
                fontSize: '.9rem',
                gap: '.5rem',
                padding: '.6rem 1.2rem'
              }}
            >
              <X size={16} /> Clear Search
            </button>
          </div>
        )}

        {/* ═══════════════════════════════════════════════
            RECORDING CARDS GRID
        ═══════════════════════════════════════════════ */}
        <div style={{
          display: 'grid',
          gap: '1rem',
          gridTemplateColumns: 'repeat(auto-fill, minmax(min(100%, 480px), 1fr))'
        }}>
          {filtered.map((item, idx) => (
            <div
              key={item.id}
              className="animate-slide-up"
              onClick={() => navigate(`/dashboard/history/${item.id}`)}
              style={{
                background: 'hsl(var(--card))',
                border: '2px solid hsl(var(--border) / .15)',
                borderRadius: '14px',
                padding: '1.25rem 1.5rem',
                cursor: 'pointer',
                animationDelay: `${idx * 0.04}s`,
                animationFillMode: 'both',
                transition: 'all .2s cubic-bezier(0.4, 0, 0.2, 1)',
                position: 'relative',
                overflow: 'hidden',
                boxShadow: '0 1px 3px hsl(var(--ink) / .04)'
              }}
              onMouseEnter={e => {
                const el = e.currentTarget as HTMLDivElement
                el.style.borderColor = 'hsl(var(--accent) / .35)'
                el.style.boxShadow = '0 4px 16px hsl(var(--ink) / .08), 0 0 0 1px hsl(var(--accent) / .15)'
                el.style.transform = 'translateY(-2px)'
              }}
              onMouseLeave={e => {
                const el = e.currentTarget as HTMLDivElement
                el.style.borderColor = 'hsl(var(--border) / .15)'
                el.style.boxShadow = '0 1px 3px hsl(var(--ink) / .04)'
                el.style.transform = 'translateY(0)'
              }}
            >
              {/* Status Indicator Stripe */}
              <div style={{
                position: 'absolute',
                left: 0,
                top: 0,
                bottom: 0,
                width: '4px',
                background: item.status === 'done'
                  ? 'linear-gradient(180deg, hsl(var(--accent)), hsl(14,95%,65%))'
                  : item.status === 'error'
                  ? 'hsl(var(--destructive))'
                  : 'linear-gradient(180deg, hsl(var(--accent) / .5), hsl(var(--accent) / .3))',
                borderRadius: '14px 0 0 14px',
              }} />

              {/* Card Content */}
              <div style={{ position: 'relative', zIndex: 1 }}>
                
                {/* Header Row */}
                <div style={{ 
                  display: 'flex', 
                  alignItems: 'flex-start', 
                  gap: '1rem',
                  marginBottom: '1rem'
                }}>
                  
                  {/* Mic Icon */}
                  <div style={{
                    width: '42px',
                    height: '42px',
                    borderRadius: '10px',
                    flexShrink: 0,
                    background: 'linear-gradient(135deg, hsl(var(--accent) / .15), hsl(var(--accent) / .08))',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    border: '2px solid hsl(var(--accent) / .25)',
                  }}>
                    <Mic size={18} style={{ color: 'hsl(var(--accent))' }} />
                  </div>

                  {/* Title & Metadata */}
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ marginBottom: '.5rem' }} onClick={e => e.stopPropagation()}>
                      <InlineEdit
                        value={item.filename}
                        onSave={(name) => handleRename(item.id, name)}
                        textStyle={{
                          fontWeight: 700,
                          fontSize: '1rem',
                          fontFamily: 'Inter, sans-serif',
                          color: 'hsl(var(--ink))',
                          letterSpacing: '-.01em',
                          lineHeight: '1.3',
                        }}
                      />
                    </div>
                    
                    {/* Date & Time */}
                    <div style={{ 
                      display: 'flex', 
                      alignItems: 'center', 
                      gap: '.75rem',
                      flexWrap: 'wrap'
                    }}>
                      <div 
                        style={{ 
                          display: 'flex', 
                          alignItems: 'center', 
                          gap: '.4rem',
                          fontSize: '.8rem',
                          color: 'hsl(var(--pencil))',
                          fontFamily: 'Inter, sans-serif',
                          fontWeight: 500
                        }}
                        title={new Date(item.created_at).toLocaleString(undefined, { 
                          dateStyle: 'full', 
                          timeStyle: 'short' 
                        })}
                      >
                        <Calendar size={13} />
                        <span>{fmtDate(item.created_at)}</span>
                      </div>
                      <div 
                        style={{ 
                          display: 'flex', 
                          alignItems: 'center', 
                          gap: '.4rem',
                          fontSize: '.8rem',
                          color: 'hsl(var(--pencil))',
                          fontFamily: 'Inter, sans-serif',
                          fontWeight: 500
                        }}
                        title={`Recorded at ${fmtTime(item.created_at)}`}
                      >
                        <Clock size={13} />
                        <span>{fmtTime(item.created_at)}</span>
                      </div>
                    </div>
                  </div>

                  {/* Action Buttons */}
                  <div
                    style={{ 
                      display: 'flex', 
                      alignItems: 'center', 
                      gap: '.5rem',
                      flexShrink: 0
                    }}
                    onClick={e => e.stopPropagation()}
                  >
                    {/* Delete Confirmation */}
                    {confirmDeleteId === item.id ? (
                      <div style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '.4rem',
                        padding: '.35rem .6rem',
                        background: 'hsl(var(--card))',
                        border: '2px solid hsl(var(--destructive) / .3)',
                        borderRadius: '8px',
                        fontSize: '.8rem',
                        fontFamily: 'Inter, sans-serif',
                        fontWeight: 600
                      }}>
                        <span style={{ color: 'hsl(var(--ink))' }}>Delete?</span>
                        <button
                          className="btn"
                          onClick={(e) => handleDeleteConfirm(item.id, e)}
                          disabled={deletingId === item.id}
                          style={{
                            padding: '.25rem .6rem',
                            fontSize: '.78rem',
                            height: 'auto',
                            background: 'hsl(var(--destructive))',
                            color: 'white',
                            border: 'none',
                            minHeight: 'unset'
                          }}
                        >
                          {deletingId === item.id ? '...' : 'Yes'}
                        </button>
                        <button
                          className="btn btn-ghost"
                          onClick={handleDeleteCancel}
                          style={{
                            padding: '.25rem .6rem',
                            fontSize: '.78rem',
                            height: 'auto',
                            minHeight: 'unset'
                          }}
                        >
                          No
                        </button>
                      </div>
                    ) : (
                      <button
                        onClick={(e) => handleDeleteClick(item.id, e)}
                        disabled={deletingId === item.id}
                        className="icon-btn"
                        style={{
                          width: '34px',
                          height: '34px',
                          color: 'hsl(var(--destructive) / .7)',
                        }}
                        title="Delete recording"
                      >
                        {deletingId === item.id ? (
                          <Loader size={14} className="spin" />
                        ) : (
                          <Trash2 size={14} />
                        )}
                      </button>
                    )}
                    
                    <ChevronRight 
                      size={18} 
                      style={{ 
                        color: 'hsl(var(--accent))',
                        transition: 'transform .2s'
                      }} 
                    />
                  </div>
                </div>

                {/* Divider */}
                <div style={{
                  height: '1px',
                  background: 'linear-gradient(90deg, transparent, hsl(var(--border) / .2), transparent)',
                  marginBottom: '1rem'
                }} />

                {/* Metadata Row */}
                <div style={{
                  display: 'flex',
                  flexWrap: 'wrap',
                  gap: '.6rem',
                  alignItems: 'center'
                }}>
                  
                  {/* Status Badge */}
                  <span 
                    className={`status-badge ${item.status === 'done' ? 'done' : item.status === 'error' ? 'error' : 'processing'}`}
                    style={{
                      fontSize: '.75rem',
                      padding: '.3rem .7rem',
                      fontWeight: 600
                    }}
                  >
                    {item.status === 'done' ? '✓ Complete' : item.status === 'error' ? '✕ Error' : '⟳ Processing'}
                  </span>

                  {/* AI Summary Badge */}
                  {item.has_summary && (
                    <span
                      className="status-badge"
                      style={{
                        fontSize: '.75rem',
                        padding: '.3rem .7rem',
                        background: 'linear-gradient(135deg, hsl(235, 75%, 65% / .15), hsl(235, 75%, 65% / .08))',
                        borderColor: 'hsl(235, 75%, 65% / .3)',
                        color: 'hsl(235, 75%, 55%)',
                        fontWeight: 600,
                        display: 'flex',
                        alignItems: 'center',
                        gap: '.35rem'
                      }}
                    >
                      <Sparkles size={11} /> AI Summary
                    </span>
                  )}

                  {/* Duration */}
                  {item.duration > 0 && (
                    <span style={{
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: '.4rem',
                      fontSize: '.78rem',
                      fontWeight: 600,
                      background: 'hsl(var(--muted))',
                      color: 'hsl(var(--pencil))',
                      border: '1.5px solid hsl(var(--border) / .2)',
                      padding: '.3rem .7rem',
                      borderRadius: '8px',
                      fontFamily: 'JetBrains Mono, monospace',
                    }}>
                      <FileAudio size={12} />
                      {fmtDuration(item.duration)}
                    </span>
                  )}

                  {/* Speakers */}
                  {item.speakers_detected.length > 0 && (
                    <div style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '.4rem',
                      flexWrap: 'wrap'
                    }}>
                      <div style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '.35rem',
                        fontSize: '.78rem',
                        color: 'hsl(var(--pencil))',
                        fontFamily: 'Inter, sans-serif',
                        fontWeight: 600,
                        paddingRight: '.15rem'
                      }}>
                        <Users size={12} />
                        <span>{item.speakers_detected.length} {item.speakers_detected.length === 1 ? 'Speaker' : 'Speakers'}:</span>
                      </div>
                      
                      {item.speakers_detected.slice(0, 3).map((sp, si) => {
                        const col = getSpeakerColor(sp)
                        return (
                          <span
                            key={si}
                            style={{
                              display: 'inline-flex',
                              alignItems: 'center',
                              gap: '.4rem',
                              padding: '.25rem .65rem .25rem .35rem',
                              borderRadius: '999px',
                              fontFamily: 'Inter, sans-serif',
                              fontSize: '.76rem',
                              fontWeight: 600,
                              background: `${col}15`,
                              border: `1.5px solid ${col}40`,
                              color: col,
                              whiteSpace: 'nowrap',
                            }}
                          >
                            <span style={{
                              width: '18px',
                              height: '18px',
                              borderRadius: '50%',
                              background: col,
                              color: '#fff',
                              fontSize: '.65rem',
                              fontWeight: 800,
                              display: 'inline-flex',
                              alignItems: 'center',
                              justifyContent: 'center',
                            }}>
                              {sp.charAt(0).toUpperCase()}
                            </span>
                            {sp}
                          </span>
                        )
                      })}

                      {item.speakers_detected.length > 3 && (
                        <span style={{
                          fontSize: '.76rem',
                          color: 'hsl(var(--pencil))',
                          fontFamily: 'Inter, sans-serif',
                          fontWeight: 600,
                          padding: '.25rem .6rem',
                          background: 'hsl(var(--muted))',
                          borderRadius: '999px',
                          border: '1.5px solid hsl(var(--border) / .2)'
                        }}>
                          +{item.speakers_detected.length - 3} more
                        </span>
                      )}
                    </div>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
