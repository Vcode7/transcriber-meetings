import React, { useState, useEffect, useMemo, useCallback } from 'react'
import {
  RotateCcw, RefreshCw, CheckCircle2, AlertCircle, Clock,
  Search, X, Check, Database, FileText,
  ShieldAlert, Sparkles, Loader2,
} from 'lucide-react'
import type { SyncableMeeting, ResyncResult } from '../types/recording'
import { getSyncableMeetings, resyncMeeting } from '../api/aiChat'

interface MeetingResyncModalProps {
  isOpen: boolean
  onClose: () => void
  onSyncCompleted?: () => void
}

export default function MeetingResyncModal({
  isOpen,
  onClose,
  onSyncCompleted,
}: MeetingResyncModalProps) {
  const [meetings, setMeetings] = useState<SyncableMeeting[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [syncingId, setSyncingId] = useState<string | null>(null)
  const [syncAllRunning, setSyncAllRunning] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [filterMode, setFilterMode] = useState<'all' | 'needs_sync' | 'missing_final_rom' | 'synced'>('all')
  const [lastResults, setLastResults] = useState<Record<string, ResyncResult>>({})

  // Load syncable meetings from backend
  const fetchMeetings = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await getSyncableMeetings()
      setMeetings(data)
    } catch (err: any) {
      console.error('Failed to load syncable meetings:', err)
      setError(err?.message || 'Failed to load syncable meetings')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    if (isOpen) {
      fetchMeetings()
    }
  }, [isOpen, fetchMeetings])

  // Handle single meeting resync
  const handleSyncMeeting = async (mid: string, force = false) => {
    if (syncingId || syncAllRunning) return
    setSyncingId(mid)
    setError(null)
    try {
      const res = await resyncMeeting(mid, force)
      setLastResults(prev => ({ ...prev, [mid]: res }))

      // Update meeting item locally
      setMeetings(prev =>
        prev.map(m => {
          if (m.id === mid) {
            const finalNewly = res.final_rom_result?.newly_indexed ?? 0
            const s2Newly = res.stage2_result?.newly_indexed ?? 0
            const newFinalIndexed = Math.max((m.indexed_final_rom_count ?? 0) + finalNewly, m.final_rom_points_count)
            const newS2Indexed = Math.max((m.indexed_stage2_count ?? 0) + s2Newly, m.stage2_points_count)
            return {
              ...m,
              indexed_final_rom_count: newFinalIndexed,
              indexed_stage2_count: newS2Indexed,
              indexed_points_count: newS2Indexed,
              final_rom_sync_status: m.has_final_rom ? 'synced' : 'not_generated',
              stage2_sync_status: m.has_stage2 ? 'synced' : 'not_generated',
              needs_sync: false,
              sync_status: 'synced',
            }
          }
          return m
        })
      )

      if (onSyncCompleted) {
        onSyncCompleted()
      }
    } catch (err: any) {
      console.error(`Failed to resync meeting ${mid}:`, err)
      setError(err?.message || `Failed to sync meeting: ${mid}`)
    } finally {
      setSyncingId(null)
    }
  }

  // Handle sync all missing meetings sequentially
  const handleSyncAll = async () => {
    const needingSync = meetings.filter(m => m.needs_sync)
    if (needingSync.length === 0 || syncAllRunning || syncingId) return

    setSyncAllRunning(true)
    setError(null)

    for (const m of needingSync) {
      setSyncingId(m.id)
      try {
        const res = await resyncMeeting(m.id, false)
        setLastResults(prev => ({ ...prev, [m.id]: res }))
        setMeetings(prev =>
          prev.map(item => {
            if (item.id === m.id) {
              const finalNewly = res.final_rom_result?.newly_indexed ?? 0
              const s2Newly = res.stage2_result?.newly_indexed ?? 0
              const newFinalIndexed = Math.max((item.indexed_final_rom_count ?? 0) + finalNewly, item.final_rom_points_count)
              const newS2Indexed = Math.max((item.indexed_stage2_count ?? 0) + s2Newly, item.stage2_points_count)
              return {
                ...item,
                indexed_final_rom_count: newFinalIndexed,
                indexed_stage2_count: newS2Indexed,
                indexed_points_count: newS2Indexed,
                final_rom_sync_status: item.has_final_rom ? 'synced' : 'not_generated',
                stage2_sync_status: item.has_stage2 ? 'synced' : 'not_generated',
                needs_sync: false,
                sync_status: 'synced',
              }
            }
            return item
          })
        )
      } catch (err: any) {
        console.error(`Failed to sync meeting ${m.id}:`, err)
      }
    }

    setSyncingId(null)
    setSyncAllRunning(false)
    if (onSyncCompleted) {
      onSyncCompleted()
    }
  }

  // Filtered list
  const filteredMeetings = useMemo(() => {
    return meetings.filter(m => {
      // Filter tab
      if (filterMode === 'needs_sync' && !m.needs_sync) return false
      if (filterMode === 'missing_final_rom' && m.final_rom_sync_status !== 'not_generated') return false
      if (filterMode === 'synced' && m.needs_sync) return false

      // Search query
      if (searchQuery) {
        const q = searchQuery.toLowerCase()
        const nameMatch = (m.name || '').toLowerCase().includes(q)
        const dateMatch = (m.date || '').toLowerCase().includes(q)
        return nameMatch || dateMatch
      }
      return true
    })
  }, [meetings, filterMode, searchQuery])

  // Aggregate stats
  const totalCount = meetings.length
  const needsSyncCount = useMemo(() => meetings.filter(m => m.needs_sync).length, [meetings])
  const missingFinalRomCount = useMemo(() => meetings.filter(m => m.final_rom_sync_status === 'not_generated').length, [meetings])
  const syncedCount = totalCount - needsSyncCount

  const fmtDate = (d: string) => {
    if (!d) return ''
    try {
      const dt = new Date(d)
      return dt.toLocaleDateString(undefined, {
        month: 'short',
        day: 'numeric',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      })
    } catch {
      return d
    }
  }

  if (!isOpen) return null

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 9999,
        background: 'hsl(var(--ink) / .5)',
        backdropFilter: 'blur(4px)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '1rem',
      }}
      onClick={(e) => {
        if (e.target === e.currentTarget && !syncingId && !syncAllRunning) {
          onClose()
        }
      }}
    >
      <div
        style={{
          width: '100%',
          maxWidth: '1000px',
          maxHeight: '90vh',
          background: 'hsl(var(--card))',
          borderRadius: '16px',
          border: '1.5px solid hsl(var(--border) / .3)',
          boxShadow: '0 20px 60px hsl(var(--ink) / .35)',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
          animation: 'fadeInScale 0.2s ease-out',
        }}
      >
        {/* Header */}
        <div
          style={{
            padding: '1.25rem 1.5rem',
            borderBottom: '1px solid hsl(var(--border) / .2)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            background: 'hsl(var(--paper))',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '.85rem' }}>
            <div
              style={{
                width: '42px',
                height: '42px',
                borderRadius: '12px',
                background: 'linear-gradient(135deg, hsl(270 70% 60% / .2), hsl(var(--accent) / .2))',
                border: '1px solid hsl(var(--accent) / .3)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: 'hsl(var(--accent))',
              }}
            >
              <RotateCcw size={22} />
            </div>
            <div>
              <h2 style={{ fontSize: '1.18rem', fontWeight: 700, margin: 0, color: 'hsl(var(--ink))' }}>
                Resync Meeting Context
              </h2>
              <p style={{ fontSize: '.8rem', color: 'hsl(var(--pencil))', margin: '2px 0 0' }}>
                Re-index Final ROM and Stage 2 discussion points into ChromaDB vector stores for AI Chat retrieval
              </p>
            </div>
          </div>

          <button
            className="btn btn-ghost"
            onClick={onClose}
            disabled={syncAllRunning}
            style={{
              padding: '.4rem',
              borderRadius: '8px',
              color: 'hsl(var(--pencil))',
            }}
          >
            <X size={18} />
          </button>
        </div>

        {/* Stats & Actions Bar */}
        <div
          style={{
            padding: '1rem 1.5rem',
            background: 'hsl(var(--paper-deep) / .4)',
            borderBottom: '1px solid hsl(var(--border) / .15)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            flexWrap: 'wrap',
            gap: '.75rem',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '.6rem', flexWrap: 'wrap' }}>
            <div
              style={{
                padding: '.35rem .75rem',
                borderRadius: '8px',
                background: 'hsl(var(--card))',
                border: '1px solid hsl(var(--border) / .2)',
                fontSize: '.78rem',
                display: 'flex',
                alignItems: 'center',
                gap: '.4rem',
              }}
            >
              <Database size={13} style={{ color: 'hsl(var(--pencil))' }} />
              <span style={{ color: 'hsl(var(--pencil))' }}>Total Meetings:</span>
              <strong style={{ color: 'hsl(var(--ink))' }}>{totalCount}</strong>
            </div>

            <div
              style={{
                padding: '.35rem .75rem',
                borderRadius: '8px',
                background: needsSyncCount > 0 ? 'hsl(38 92% 50% / .12)' : 'hsl(var(--card))',
                border: needsSyncCount > 0 ? '1px solid hsl(38 92% 50% / .3)' : '1px solid hsl(var(--border) / .2)',
                fontSize: '.78rem',
                display: 'flex',
                alignItems: 'center',
                gap: '.4rem',
                color: needsSyncCount > 0 ? 'hsl(38 92% 40%)' : 'hsl(var(--pencil))',
              }}
            >
              <AlertCircle size={13} />
              <span>Needs Sync:</span>
              <strong>{needsSyncCount}</strong>
            </div>

            <div
              style={{
                padding: '.35rem .75rem',
                borderRadius: '8px',
                background: missingFinalRomCount > 0 ? 'hsl(0 84% 60% / .1)' : 'hsl(var(--card))',
                border: missingFinalRomCount > 0 ? '1px solid hsl(0 84% 60% / .25)' : '1px solid hsl(var(--border) / .2)',
                fontSize: '.78rem',
                display: 'flex',
                alignItems: 'center',
                gap: '.4rem',
                color: missingFinalRomCount > 0 ? 'hsl(0 84% 45%)' : 'hsl(var(--pencil))',
              }}
            >
              <AlertCircle size={13} />
              <span>Final ROM Missing:</span>
              <strong>{missingFinalRomCount}</strong>
            </div>

            <div
              style={{
                padding: '.35rem .75rem',
                borderRadius: '8px',
                background: syncedCount > 0 ? 'hsl(142 70% 45% / .1)' : 'hsl(var(--card))',
                border: syncedCount > 0 ? '1px solid hsl(142 70% 45% / .25)' : '1px solid hsl(var(--border) / .2)',
                fontSize: '.78rem',
                display: 'flex',
                alignItems: 'center',
                gap: '.4rem',
                color: syncedCount > 0 ? 'hsl(142 70% 35%)' : 'hsl(var(--pencil))',
              }}
            >
              <CheckCircle2 size={13} />
              <span>Synced:</span>
              <strong>{syncedCount}</strong>
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '.5rem' }}>
            <button
              onClick={fetchMeetings}
              disabled={loading || syncAllRunning}
              title="Refresh meeting statuses"
              style={{
                background: 'hsl(var(--card))',
                border: '1px solid hsl(var(--border) / .25)',
                padding: '.4rem .7rem',
                borderRadius: '8px',
                fontSize: '.78rem',
                color: 'hsl(var(--ink))',
                display: 'flex',
                alignItems: 'center',
                gap: '.35rem',
                cursor: 'pointer',
              }}
            >
              <RefreshCw size={12} className={loading ? 'animate-spin' : ''} />
              Refresh
            </button>

            {needsSyncCount > 0 && (
              <button
                onClick={handleSyncAll}
                disabled={syncAllRunning || !!syncingId}
                style={{
                  background: 'hsl(var(--accent))',
                  border: 'none',
                  color: 'white',
                  padding: '.4rem .9rem',
                  borderRadius: '8px',
                  fontSize: '.78rem',
                  fontWeight: 600,
                  display: 'flex',
                  alignItems: 'center',
                  gap: '.4rem',
                  cursor: syncAllRunning ? 'not-allowed' : 'pointer',
                  boxShadow: '0 2px 8px hsl(var(--accent) / .3)',
                }}
              >
                {syncAllRunning ? (
                  <>
                    <Loader2 size={13} className="animate-spin" />
                    Syncing All ({needsSyncCount})...
                  </>
                ) : (
                  <>
                    <Sparkles size={13} />
                    Sync All Missing ({needsSyncCount})
                  </>
                )}
              </button>
            )}
          </div>
        </div>

        {/* Search and Filters */}
        <div
          style={{
            padding: '.75rem 1.5rem',
            borderBottom: '1px solid hsl(var(--border) / .15)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: '1rem',
          }}
        >
          {/* Search box */}
          <div
            style={{
              flex: 1,
              position: 'relative',
              display: 'flex',
              alignItems: 'center',
            }}
          >
            <Search size={14} style={{ position: 'absolute', left: '.75rem', color: 'hsl(var(--pencil))' }} />
            <input
              type="text"
              placeholder="Search meetings by name or date..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              style={{
                width: '100%',
                padding: '.45rem .75rem .45rem 2.2rem',
                borderRadius: '8px',
                border: '1px solid hsl(var(--border) / .25)',
                background: 'hsl(var(--paper))',
                fontSize: '.82rem',
                color: 'hsl(var(--ink))',
                outline: 'none',
              }}
            />
            {searchQuery && (
              <button
                onClick={() => setSearchQuery('')}
                style={{
                  position: 'absolute',
                  right: '.6rem',
                  background: 'none',
                  border: 'none',
                  color: 'hsl(var(--pencil))',
                  cursor: 'pointer',
                  padding: '2px',
                  display: 'flex',
                }}
              >
                <X size={13} />
              </button>
            )}
          </div>

          {/* Filter tabs */}
          <div
            style={{
              display: 'flex',
              background: 'hsl(var(--paper))',
              padding: '3px',
              borderRadius: '8px',
              border: '1px solid hsl(var(--border) / .2)',
              gap: '2px',
            }}
          >
            <button
              onClick={() => setFilterMode('all')}
              style={{
                padding: '.3rem .65rem',
                borderRadius: '6px',
                fontSize: '.74rem',
                fontWeight: filterMode === 'all' ? 600 : 500,
                border: 'none',
                background: filterMode === 'all' ? 'hsl(var(--card))' : 'transparent',
                color: filterMode === 'all' ? 'hsl(var(--accent))' : 'hsl(var(--pencil))',
                cursor: 'pointer',
                boxShadow: filterMode === 'all' ? '0 1px 4px hsl(var(--ink) / .08)' : 'none',
              }}
            >
              All ({totalCount})
            </button>
            <button
              onClick={() => setFilterMode('needs_sync')}
              style={{
                padding: '.3rem .65rem',
                borderRadius: '6px',
                fontSize: '.74rem',
                fontWeight: filterMode === 'needs_sync' ? 600 : 500,
                border: 'none',
                background: filterMode === 'needs_sync' ? 'hsl(var(--card))' : 'transparent',
                color: filterMode === 'needs_sync' ? 'hsl(38 92% 40%)' : 'hsl(var(--pencil))',
                cursor: 'pointer',
                boxShadow: filterMode === 'needs_sync' ? '0 1px 4px hsl(var(--ink) / .08)' : 'none',
              }}
            >
              Needs Sync ({needsSyncCount})
            </button>
            <button
              onClick={() => setFilterMode('missing_final_rom')}
              style={{
                padding: '.3rem .65rem',
                borderRadius: '6px',
                fontSize: '.74rem',
                fontWeight: filterMode === 'missing_final_rom' ? 600 : 500,
                border: 'none',
                background: filterMode === 'missing_final_rom' ? 'hsl(var(--card))' : 'transparent',
                color: filterMode === 'missing_final_rom' ? 'hsl(0 84% 45%)' : 'hsl(var(--pencil))',
                cursor: 'pointer',
                boxShadow: filterMode === 'missing_final_rom' ? '0 1px 4px hsl(var(--ink) / .08)' : 'none',
              }}
            >
              Final ROM Missing ({missingFinalRomCount})
            </button>
            <button
              onClick={() => setFilterMode('synced')}
              style={{
                padding: '.3rem .65rem',
                borderRadius: '6px',
                fontSize: '.74rem',
                fontWeight: filterMode === 'synced' ? 600 : 500,
                border: 'none',
                background: filterMode === 'synced' ? 'hsl(var(--card))' : 'transparent',
                color: filterMode === 'synced' ? 'hsl(142 70% 35%)' : 'hsl(var(--pencil))',
                cursor: 'pointer',
                boxShadow: filterMode === 'synced' ? '0 1px 4px hsl(var(--ink) / .08)' : 'none',
              }}
            >
              Synced ({syncedCount})
            </button>
          </div>
        </div>

        {/* Error alert */}
        {error && (
          <div
            style={{
              padding: '.65rem 1.5rem',
              background: 'hsl(0 84% 60% / .12)',
              borderBottom: '1px solid hsl(0 84% 60% / .25)',
              color: 'hsl(0 84% 45%)',
              fontSize: '.78rem',
              display: 'flex',
              alignItems: 'center',
              gap: '.5rem',
            }}
          >
            <ShieldAlert size={14} />
            <span>{error}</span>
          </div>
        )}

        {/* Meeting List */}
        <div
          style={{
            flex: 1,
            overflowY: 'auto',
            padding: '1rem 1.5rem',
            display: 'flex',
            flexDirection: 'column',
            gap: '.65rem',
          }}
        >
          {loading ? (
            <div
              style={{
                padding: '3rem',
                textAlign: 'center',
                color: 'hsl(var(--pencil))',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                gap: '.75rem',
              }}
            >
              <Loader2 size={24} className="animate-spin" style={{ color: 'hsl(var(--accent))' }} />
              <span style={{ fontSize: '.84rem' }}>Scanning meetings for Final ROM & Stage 2 points...</span>
            </div>
          ) : filteredMeetings.length === 0 ? (
            <div
              style={{
                padding: '3rem 1.5rem',
                textAlign: 'center',
                color: 'hsl(var(--pencil))',
              }}
            >
              <Database size={36} style={{ opacity: 0.35, margin: '0 auto .75rem' }} />
              <p style={{ fontWeight: 600, fontSize: '.92rem', margin: '0 0 .25rem', color: 'hsl(var(--ink))' }}>
                {totalCount === 0
                  ? 'No meetings found'
                  : 'No matching meetings found'}
              </p>
              <p style={{ fontSize: '.78rem', margin: 0, opacity: 0.8 }}>
                {totalCount === 0
                  ? 'Record or process meetings in History first.'
                  : 'Try clearing your search or switching filters.'}
              </p>
            </div>
          ) : (
            filteredMeetings.map((m) => {
              const isSyncing = syncingId === m.id
              const lastResult = lastResults[m.id]
              const hasAnyPoints = m.has_final_rom || m.has_stage2

              return (
                <div
                  key={m.id}
                  style={{
                    background: 'hsl(var(--card))',
                    borderRadius: '12px',
                    border: m.needs_sync
                      ? '1.5px solid hsl(38 92% 50% / .3)'
                      : '1px solid hsl(var(--border) / .25)',
                    padding: '.9rem 1.1rem',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '.6rem',
                    transition: 'all .15s ease',
                  }}
                >
                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      gap: '1rem',
                    }}
                  >
                    {/* Meeting Info */}
                    <div style={{ display: 'flex', alignItems: 'flex-start', gap: '.75rem', minWidth: 0, flex: 1 }}>
                      <div
                        style={{
                          width: '34px',
                          height: '34px',
                          borderRadius: '8px',
                          background: m.needs_sync
                            ? 'hsl(38 92% 50% / .12)'
                            : 'hsl(142 70% 45% / .12)',
                          color: m.needs_sync
                            ? 'hsl(38 92% 40%)'
                            : 'hsl(142 70% 35%)',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          flexShrink: 0,
                          marginTop: '2px',
                        }}
                      >
                        <FileText size={17} />
                      </div>

                      <div style={{ minWidth: 0, flex: 1 }}>
                        <div
                          style={{
                            fontSize: '.88rem',
                            fontWeight: 600,
                            color: 'hsl(var(--ink))',
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                            whiteSpace: 'nowrap',
                          }}
                          title={m.name}
                        >
                          {m.name}
                        </div>

                        <div
                          style={{
                            display: 'flex',
                            alignItems: 'center',
                            gap: '.5rem',
                            marginTop: '3px',
                            fontSize: '.74rem',
                            color: 'hsl(var(--pencil))',
                            flexWrap: 'wrap',
                          }}
                        >
                          {m.date && <span>{fmtDate(m.date)}</span>}
                          <span>•</span>
                          <span style={{ fontWeight: 500 }}>
                            {m.has_final_rom ? `${m.final_rom_points_count} Final ROM pts` : 'No Final ROM'}
                          </span>
                          <span>•</span>
                          <span style={{ fontWeight: 500 }}>
                            {m.has_stage2 ? `${m.stage2_points_count} Stage 2 pts` : 'No Stage 2'}
                          </span>
                        </div>
                      </div>
                    </div>

                    {/* Status Badges & Action Button */}
                    <div style={{ display: 'flex', alignItems: 'center', gap: '.6rem', flexShrink: 0, flexWrap: 'wrap', justifyContent: 'flex-end' }}>
                      {/* Final ROM status badge */}
                      {m.final_rom_sync_status === 'not_generated' ? (
                        <span
                          title="Final ROM discussion points have not been generated yet for this meeting"
                          style={{
                            fontSize: '.72rem',
                            fontWeight: 600,
                            padding: '3px 8px',
                            borderRadius: '6px',
                            background: 'hsl(0 84% 60% / .12)',
                            color: 'hsl(0 84% 45%)',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '.3rem',
                          }}
                        >
                          <AlertCircle size={12} />
                          Final ROM Point Missing
                        </span>
                      ) : m.final_rom_sync_status === 'not_indexed' ? (
                        <span
                          title="Final ROM points exist but ChromaDB embeddings are missing"
                          style={{
                            fontSize: '.72rem',
                            fontWeight: 600,
                            padding: '3px 8px',
                            borderRadius: '6px',
                            background: 'hsl(38 92% 50% / .14)',
                            color: 'hsl(38 92% 35%)',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '.3rem',
                          }}
                        >
                          <AlertCircle size={12} />
                          Final ROM Embedding Missing (0/{m.final_rom_points_count})
                        </span>
                      ) : m.final_rom_sync_status === 'partial' ? (
                        <span
                          style={{
                            fontSize: '.72rem',
                            fontWeight: 600,
                            padding: '3px 8px',
                            borderRadius: '6px',
                            background: 'hsl(38 92% 50% / .14)',
                            color: 'hsl(38 92% 35%)',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '.3rem',
                          }}
                        >
                          <AlertCircle size={12} />
                          Final ROM {m.indexed_final_rom_count || 0}/{m.final_rom_points_count}
                        </span>
                      ) : (
                        <span
                          style={{
                            fontSize: '.72rem',
                            fontWeight: 600,
                            padding: '3px 8px',
                            borderRadius: '6px',
                            background: 'hsl(142 70% 45% / .14)',
                            color: 'hsl(142 70% 32%)',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '.3rem',
                          }}
                        >
                          <CheckCircle2 size={12} />
                          Final ROM Indexed ({m.final_rom_points_count})
                        </span>
                      )}

                      {/* Stage 2 status badge */}
                      {m.stage2_sync_status === 'not_generated' ? (
                        <span
                          title="Stage 2 points have not been generated yet for this meeting"
                          style={{
                            fontSize: '.72rem',
                            fontWeight: 600,
                            padding: '3px 8px',
                            borderRadius: '6px',
                            background: 'hsl(var(--paper-deep) / .8)',
                            color: 'hsl(var(--pencil))',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '.3rem',
                          }}
                        >
                          <AlertCircle size={12} />
                          Stage 2 Point Missing
                        </span>
                      ) : m.stage2_sync_status === 'not_indexed' ? (
                        <span
                          title="Stage 2 points exist but ChromaDB embeddings are missing"
                          style={{
                            fontSize: '.72rem',
                            fontWeight: 600,
                            padding: '3px 8px',
                            borderRadius: '6px',
                            background: 'hsl(38 92% 50% / .14)',
                            color: 'hsl(38 92% 35%)',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '.3rem',
                          }}
                        >
                          <AlertCircle size={12} />
                          Stage 2 Embedding Missing (0/{m.stage2_points_count})
                        </span>
                      ) : m.stage2_sync_status === 'partial' ? (
                        <span
                          style={{
                            fontSize: '.72rem',
                            fontWeight: 600,
                            padding: '3px 8px',
                            borderRadius: '6px',
                            background: 'hsl(38 92% 50% / .14)',
                            color: 'hsl(38 92% 35%)',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '.3rem',
                          }}
                        >
                          <AlertCircle size={12} />
                          Stage 2 {m.indexed_stage2_count || 0}/{m.stage2_points_count}
                        </span>
                      ) : (
                        <span
                          style={{
                            fontSize: '.72rem',
                            fontWeight: 600,
                            padding: '3px 8px',
                            borderRadius: '6px',
                            background: 'hsl(142 70% 45% / .14)',
                            color: 'hsl(142 70% 32%)',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '.3rem',
                          }}
                        >
                          <CheckCircle2 size={12} />
                          Stage 2 Indexed ({m.stage2_points_count})
                        </span>
                      )}

                      {/* Action Button */}
                      <button
                        onClick={() => handleSyncMeeting(m.id, false)}
                        disabled={isSyncing || syncAllRunning || !hasAnyPoints}
                        title={
                          !hasAnyPoints
                            ? 'Neither Final ROM nor Stage 2 points generated yet'
                            : m.needs_sync
                            ? 'Embed missing discussion points into ChromaDB'
                            : 'Re-sync all discussion points'
                        }
                        style={{
                          padding: '.4rem .85rem',
                          borderRadius: '8px',
                          fontSize: '.76rem',
                          fontWeight: 600,
                          cursor: isSyncing || syncAllRunning || !hasAnyPoints ? 'not-allowed' : 'pointer',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '.35rem',
                          border: m.needs_sync
                            ? 'none'
                            : '1px solid hsl(var(--border) / .3)',
                          background: m.needs_sync
                            ? 'hsl(var(--accent))'
                            : 'hsl(var(--paper))',
                          color: m.needs_sync
                            ? 'white'
                            : hasAnyPoints
                            ? 'hsl(var(--ink))'
                            : 'hsl(var(--pencil))',
                          boxShadow: m.needs_sync
                            ? '0 2px 6px hsl(var(--accent) / .25)'
                            : 'none',
                          opacity: !hasAnyPoints ? 0.6 : 1,
                        }}
                      >
                        {isSyncing ? (
                          <>
                            <Loader2 size={12} className="animate-spin" />
                            Indexing...
                          </>
                        ) : !hasAnyPoints ? (
                          <>
                            <AlertCircle size={12} />
                            No Points
                          </>
                        ) : m.needs_sync ? (
                          <>
                            <Sparkles size={12} />
                            Sync to ChromaDB
                          </>
                        ) : (
                          <>
                            <RotateCcw size={12} />
                            Resync
                          </>
                        )}
                      </button>
                    </div>
                  </div>

                  {/* Last sync result report */}
                  {lastResult && (
                    <div
                      style={{
                        marginTop: '.2rem',
                        padding: '.5rem .75rem',
                        borderRadius: '8px',
                        background: 'hsl(142 70% 45% / .08)',
                        border: '1px solid hsl(142 70% 45% / .2)',
                        fontSize: '.74rem',
                        color: 'hsl(var(--ink))',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        flexWrap: 'wrap',
                        gap: '.4rem',
                      }}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', gap: '.4rem', flexWrap: 'wrap' }}>
                        <Check size={13} style={{ color: 'hsl(142 70% 35%)' }} />
                        <span>
                          <strong>Sync report:</strong> {lastResult.total_found} points processed •{' '}
                          <span style={{ color: 'hsl(142 70% 35%)', fontWeight: 600 }}>
                            +{lastResult.newly_indexed} newly indexed
                          </span>{' '}
                          • {lastResult.already_indexed} verified
                          {lastResult.skipped_failed > 0 && (
                            <span style={{ color: 'hsl(0 84% 45%)' }}>
                              {' '}• {lastResult.skipped_failed} skipped
                            </span>
                          )}
                        </span>
                      </div>
                      <span style={{ fontSize: '.7rem', color: 'hsl(142 70% 35%)', fontWeight: 600 }}>
                        Ready for AI Chat (Final ROM Priority)
                      </span>
                    </div>
                  )}
                </div>
              )
            })
          )}
        </div>

        {/* Footer */}
        <div
          style={{
            padding: '.9rem 1.5rem',
            borderTop: '1px solid hsl(var(--border) / .2)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            background: 'hsl(var(--paper))',
          }}
        >
          <p style={{ fontSize: '.76rem', color: 'hsl(var(--pencil))', margin: 0 }}>
            Syncing embeds discussion points into isolated collections (Final ROM and Stage 2) using text embeddings without modifying points.
          </p>
          <button
            onClick={onClose}
            disabled={syncAllRunning}
            style={{
              padding: '.45rem 1.25rem',
              borderRadius: '8px',
              fontSize: '.82rem',
              fontWeight: 600,
              background: 'hsl(var(--ink))',
              color: 'hsl(var(--paper))',
              border: 'none',
              cursor: 'pointer',
            }}
          >
            Done
          </button>
        </div>
      </div>
    </div>
  )
}
