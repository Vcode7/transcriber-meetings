/**
 * TranscriptReviewPanel — Missing Transcription Recovery review interface.
 *
 * Appears when pipeline is paused in 'pending_transcript_review' status.
 * Shows raw aligned transcription segments, automatically flags potential silence/speech gaps,
 * and allows users to manually insert missed speech text before speaker diarization & insights run.
 */

import React, { useState, useEffect, useMemo } from 'react'
import { Check, Plus, Trash2, ArrowRight, SkipForward, AlertCircle, Clock, FileText, Sparkles, Loader2 } from 'lucide-react'
import { toast } from 'sonner'
import api from '../api/client'
import { getApiErrorDetail } from '../lib/errors'

interface RawSegment {
  start: number
  end: number
  text: string
  words?: Array<{ word: string; start: number; end: number; score?: number; probability?: number }>
  manually_added?: boolean
}

interface NewGapCorrection {
  id: string
  start: number
  end: number
  text: string
}

interface TranscriptReviewPanelProps {
  recordingId: string
  onResumed: () => void
}

function formatTime(sec: number): string {
  if (!isFinite(sec) || sec < 0) return '0:00'
  const m = Math.floor(sec / 60)
  const s = Math.floor(sec % 60)
  const ms = Math.floor((sec % 1) * 10)
  return `${m}:${String(s).padStart(2, '0')}.${ms}`
}

export default function TranscriptReviewPanel({ recordingId, onResumed }: TranscriptReviewPanelProps) {
  const [segments, setSegments] = useState<RawSegment[]>([])
  const [loading, setLoading] = useState(true)
  const [submitting, setSubmitting] = useState(false)
  const [corrections, setCorrections] = useState<NewGapCorrection[]>([])
  const [error, setError] = useState<string | null>(null)

  // Fetch raw transcript
  useEffect(() => {
    let active = true
    setLoading(true)
    setError(null)

    api.get(`/audio/${recordingId}/transcript/raw`)
      .then((res) => {
        if (!active) return
        setSegments(res.data.segments || [])
      })
      .catch((err) => {
        if (!active) return
        setError(getApiErrorDetail(err, 'Failed to fetch raw transcript for review.'))
      })
      .finally(() => {
        if (active) setLoading(false)
      })

    return () => {
      active = false
    }
  }, [recordingId])

  // Detect gaps >= 2.5 seconds between segments
  const detectedGaps = useMemo(() => {
    const gaps: Array<{ start: number; end: number; indexBefore: number }> = []
    if (segments.length === 0) return gaps

    for (let i = 0; i < segments.length - 1; i++) {
      const currentEnd = segments[i].end
      const nextStart = segments[i + 1].start
      const gapDuration = nextStart - currentEnd
      if (gapDuration >= 2.0) {
        gaps.push({
          start: currentEnd,
          end: nextStart,
          indexBefore: i,
        })
      }
    }
    return gaps
  }, [segments])

  const handleAddGapCorrection = (start: number, end: number) => {
    const newCorr: NewGapCorrection = {
      id: Math.random().toString(36).substring(2, 9),
      start: Math.round(start * 100) / 100,
      end: Math.round(end * 100) / 100,
      text: '',
    }
    setCorrections((prev) => [...prev, newCorr])
  }

  const handleUpdateCorrection = (id: string, text: string) => {
    setCorrections((prev) =>
      prev.map((c) => (c.id === id ? { ...c, text } : c))
    )
  }

  const handleRemoveCorrection = (id: string) => {
    setCorrections((prev) => prev.filter((c) => c.id !== id))
  }

  const handleSubmit = async (skip: boolean = false) => {
    setSubmitting(true)
    try {
      const validCorrections = skip
        ? []
        : corrections
            .filter((c) => c.text.trim().length > 0)
            .map((c) => ({
              start: c.start,
              end: c.end,
              text: c.text.trim(),
            }))

      await api.post(`/audio/${recordingId}/transcript/corrections`, {
        corrections: validCorrections,
      })

      if (skip || validCorrections.length === 0) {
        toast.info('Pipeline resumed with original transcript')
      } else {
        toast.success(`Pipeline resumed with ${validCorrections.length} recovered segment(s)`)
      }
      onResumed()
    } catch (err: unknown) {
      toast.error(getApiErrorDetail(err, 'Failed to resume pipeline.'))
      setSubmitting(false)
    }
  }

  if (loading) {
    return (
      <div style={{
        padding: '3rem',
        textAlign: 'center',
        background: 'hsl(var(--card))',
        borderRadius: '16px',
        border: '1.5px solid hsl(var(--accent) / .2)',
        margin: '1.5rem',
      }}>
        <Loader2 className="spin" size={32} style={{ color: 'hsl(var(--accent))', margin: '0 auto 1rem' }} />
        <div style={{ fontWeight: 600, color: 'hsl(var(--ink))' }}>Loading Transcription for Review...</div>
        <div style={{ fontSize: '.8rem', color: 'hsl(var(--pencil))', marginTop: '4px' }}>
          Preparing word-aligned segments to check for missing speech
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div style={{
        padding: '2rem',
        background: 'hsl(var(--destructive) / .08)',
        borderRadius: '14px',
        border: '1.5px solid hsl(var(--destructive) / .3)',
        margin: '1.5rem',
        color: 'hsl(var(--destructive))',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontWeight: 700 }}>
          <AlertCircle size={18} />
          Error Loading Review
        </div>
        <div style={{ fontSize: '.85rem', marginTop: '6px' }}>{error}</div>
        <button
          onClick={() => handleSubmit(true)}
          style={{
            marginTop: '1rem',
            padding: '6px 14px',
            borderRadius: '8px',
            background: 'hsl(var(--destructive))',
            color: '#fff',
            border: 'none',
            fontWeight: 600,
            cursor: 'pointer',
          }}
        >
          Skip & Resume Pipeline
        </button>
      </div>
    )
  }

  return (
    <div style={{
      margin: '1.5rem',
      borderRadius: '16px',
      border: '1.5px solid hsl(var(--accent) / .25)',
      background: 'hsl(var(--card))',
      boxShadow: '0 8px 30px hsl(var(--ink) / .06)',
      overflow: 'hidden',
      fontFamily: 'Inter, sans-serif',
    }}>
      {/* Header */}
      <div style={{
        padding: '1.25rem 1.5rem',
        background: 'hsl(var(--accent) / .08)',
        borderBottom: '1.5px solid hsl(var(--accent) / .15)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: '12px',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{
            width: '38px', height: '38px', borderRadius: '10px',
            background: 'hsl(var(--accent) / .15)',
            border: '1.5px solid hsl(var(--accent) / .3)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <Sparkles size={18} style={{ color: 'hsl(var(--accent))' }} />
          </div>
          <div>
            <div style={{ fontSize: '1rem', fontWeight: 800, color: 'hsl(var(--ink))', display: 'flex', alignItems: 'center', gap: '8px' }}>
              Missing Transcription Recovery
              <span style={{
                fontSize: '.68rem', fontWeight: 700, padding: '2px 8px',
                borderRadius: '999px', background: 'hsl(var(--accent))', color: '#fff',
              }}>
                Checkpoint
              </span>
            </div>
            <div style={{ fontSize: '.78rem', color: 'hsl(var(--pencil))', marginTop: '2px' }}>
              Review raw speech alignment. Insert any silent gaps or missed phrases before Speaker Diarization runs.
            </div>
          </div>
        </div>

        {/* Stats */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div style={{
            padding: '4px 10px', borderRadius: '8px',
            background: 'hsl(var(--muted))',
            fontSize: '.75rem', fontWeight: 600, color: 'hsl(var(--ink))',
          }}>
            {segments.length} transcribed segments
          </div>
          {detectedGaps.length > 0 && (
            <div style={{
              padding: '4px 10px', borderRadius: '8px',
              background: 'hsl(45 90% 50% / .15)',
              border: '1px solid hsl(45 90% 50% / .3)',
              fontSize: '.75rem', fontWeight: 700, color: 'hsl(45 90% 40%)',
            }}>
              {detectedGaps.length} speech gap{detectedGaps.length > 1 ? 's' : ''} detected
            </div>
          )}
        </div>
      </div>

      {/* Main Content List */}
      <div style={{
        padding: '1.25rem 1.5rem',
        maxHeight: '480px',
        overflowY: 'auto',
        display: 'flex',
        flexDirection: 'column',
        gap: '10px',
      }}>
        {segments.map((seg, idx) => {
          // Check if there is a gap after this segment
          const nextSeg = segments[idx + 1]
          const hasGapAfter = nextSeg && (nextSeg.start - seg.end) >= 2.5
          const gapDuration = nextSeg ? nextSeg.start - seg.end : 0

          // Existing corrections inside this gap
          const gapCorrections = corrections.filter(
            (c) => c.start >= seg.end - 0.1 && c.end <= (nextSeg?.start ?? 99999) + 0.1
          )

          return (
            <React.Fragment key={idx}>
              {/* Transcript Segment Card */}
              <div style={{
                padding: '0.85rem 1rem',
                borderRadius: '10px',
                background: 'hsl(var(--muted) / .3)',
                border: '1px solid hsl(var(--ink) / .08)',
                display: 'flex',
                gap: '12px',
                alignItems: 'flex-start',
              }}>
                <div style={{
                  fontSize: '.72rem',
                  fontFamily: 'JetBrains Mono, monospace',
                  fontWeight: 700,
                  color: 'hsl(var(--accent))',
                  background: 'hsl(var(--accent) / .08)',
                  padding: '3px 8px',
                  borderRadius: '6px',
                  whiteSpace: 'nowrap',
                  marginTop: '1px',
                }}>
                  {formatTime(seg.start)} → {formatTime(seg.end)}
                </div>
                <div style={{
                  flex: 1,
                  fontSize: '.85rem',
                  color: 'hsl(var(--ink))',
                  lineHeight: 1.5,
                }}>
                  {seg.text}
                </div>
              </div>

              {/* Gap Alert / Recovery Insertion */}
              {hasGapAfter && (
                <div style={{
                  margin: '4px 0 4px 1.5rem',
                  padding: '0.75rem 1rem',
                  borderRadius: '10px',
                  background: 'hsl(45 90% 50% / .06)',
                  border: '1.5px dashed hsl(45 90% 50% / .3)',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '8px',
                }}>
                  <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                  }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '.76rem', color: 'hsl(45 90% 35%)', fontWeight: 700 }}>
                      <Clock size={12} />
                      Silence Gap: {gapDuration.toFixed(1)}s ({formatTime(seg.end)} → {formatTime(nextSeg.start)})
                    </div>

                    <button
                      onClick={() => handleAddGapCorrection(seg.end, nextSeg.start)}
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '4px',
                        padding: '3px 9px',
                        borderRadius: '6px',
                        background: 'hsl(45 90% 50% / .15)',
                        border: '1px solid hsl(45 90% 50% / .4)',
                        color: 'hsl(45 90% 30%)',
                        fontSize: '.72rem',
                        fontWeight: 700,
                        cursor: 'pointer',
                      }}
                    >
                      <Plus size={12} />
                      Fill Missing Speech
                    </button>
                  </div>

                  {/* Any corrections added in this gap */}
                  {gapCorrections.map((corr) => (
                    <div
                      key={corr.id}
                      style={{
                        display: 'flex',
                        gap: '8px',
                        alignItems: 'center',
                        background: 'hsl(var(--card))',
                        padding: '6px 10px',
                        borderRadius: '8px',
                        border: '1px solid hsl(var(--accent) / .3)',
                      }}
                    >
                      <span style={{ fontSize: '.72rem', fontWeight: 700, color: 'hsl(var(--accent))', fontFamily: 'JetBrains Mono, monospace' }}>
                        Recovery:
                      </span>
                      <input
                        type="text"
                        placeholder="Type missed speech here..."
                        value={corr.text}
                        onChange={(e) => handleUpdateCorrection(corr.id, e.target.value)}
                        autoFocus
                        style={{
                          flex: 1,
                          background: 'transparent',
                          border: 'none',
                          outline: 'none',
                          fontSize: '.82rem',
                          color: 'hsl(var(--ink))',
                        }}
                      />
                      <button
                        onClick={() => handleRemoveCorrection(corr.id)}
                        style={{
                          background: 'transparent',
                          border: 'none',
                          color: 'hsl(var(--destructive))',
                          cursor: 'pointer',
                          padding: '2px',
                        }}
                        title="Remove correction"
                      >
                        <Trash2 size={13} />
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </React.Fragment>
          )
        })}
      </div>

      {/* Footer Action Controls */}
      <div style={{
        padding: '1.25rem 1.5rem',
        background: 'hsl(var(--muted) / .2)',
        borderTop: '1.5px solid hsl(var(--ink) / .08)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: '10px',
      }}>
        <button
          onClick={() => handleSubmit(true)}
          disabled={submitting}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            padding: '0.6rem 1.1rem',
            borderRadius: '10px',
            background: 'transparent',
            border: '1.5px solid hsl(var(--ink) / .15)',
            color: 'hsl(var(--pencil))',
            fontSize: '.82rem',
            fontWeight: 600,
            cursor: submitting ? 'not-allowed' : 'pointer',
          }}
        >
          <SkipForward size={14} />
          Skip & Keep Original
        </button>

        <button
          onClick={() => handleSubmit(false)}
          disabled={submitting}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            padding: '0.65rem 1.35rem',
            borderRadius: '10px',
            background: 'hsl(var(--accent))',
            color: '#fff',
            border: 'none',
            fontSize: '.85rem',
            fontWeight: 700,
            cursor: submitting ? 'not-allowed' : 'pointer',
            boxShadow: '0 4px 14px hsl(var(--accent) / .35)',
            opacity: submitting ? 0.7 : 1,
          }}
        >
          {submitting ? (
            <>
              <Loader2 className="spin" size={14} />
              Resuming Pipeline...
            </>
          ) : (
            <>
              <Check size={15} />
              Save Recovered Speech & Continue
              <ArrowRight size={14} />
            </>
          )}
        </button>
      </div>
    </div>
  )
}
