import { useState, useEffect, useCallback } from 'react'
import {
  Users, ChevronDown, ChevronUp, Trash2, ArrowRight,
  Play, Loader, AlertTriangle, CheckCircle, X, UserX,
  MoreVertical, Shuffle, CornerDownRight, Plus, UserCheck
} from 'lucide-react'
import api from '../api/client'

// ── Types ─────────────────────────────────────────────────────────────────────

interface SegmentRow {
  seg_id: string
  text: string
  start: number
  end: number
  speaker_label: string
  similarity?: number | null
  scoring_method?: string | null
  is_overlap: boolean
}

interface SpeakerGroup {
  speaker_label: string
  segments: SegmentRow[]
}

interface VoiceProfile {
  id: string
  label: string
  is_self?: boolean
  sample_count?: number
}

interface DissolveProposal {
  seg_id: string
  text: string
  start: number
  end: number
  proposed_speaker: string
  similarity: number | null
  method: string
}

interface DissolvePreviewData {
  dissolving_speaker: string
  remaining_speakers: string[]
  audio_available: boolean
  proposals: DissolveProposal[]
}

interface Props {
  recordingId: string
  onScrollToSegment: (segId: string, startTime: number) => void
  onTranscriptChanged: () => void
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function formatTime(s: number) {
  const m = Math.floor(s / 60)
  const sec = Math.floor(s % 60).toString().padStart(2, '0')
  return `${m}:${sec}`
}

const SPEAKER_PALETTE = [
  { accent: 'hsl(14,90%,56%)', bg: 'hsl(14,90%,56% / .08)', border: 'hsl(14,90%,56% / .25)' },
  { accent: 'hsl(205,85%,55%)', bg: 'hsl(205,85%,55% / .08)', border: 'hsl(205,85%,55% / .25)' },
  { accent: 'hsl(130,60%,45%)', bg: 'hsl(130,60%,45% / .08)', border: 'hsl(130,60%,45% / .25)' },
  { accent: 'hsl(280,65%,58%)', bg: 'hsl(280,65%,58% / .08)', border: 'hsl(280,65%,58% / .25)' },
  { accent: 'hsl(45,90%,50%)', bg: 'hsl(45,90%,50% / .08)', border: 'hsl(45,90%,50% / .25)' },
  { accent: 'hsl(340,75%,58%)', bg: 'hsl(340,75%,58% / .08)', border: 'hsl(340,75%,58% / .25)' },
]

function getSpeakerColor(label: string, allLabels: string[]) {
  const idx = allLabels.indexOf(label)
  return SPEAKER_PALETTE[Math.max(0, idx) % SPEAKER_PALETTE.length]
}

function SimilarityBadge({ score, method }: { score?: number | null; method?: string | null }) {
  if (score == null || score === 0) return null
  const pct = Math.round(score * 100)
  const color = pct >= 80 ? 'hsl(130,60%,38%)' : pct >= 60 ? 'hsl(45,90%,40%)' : 'hsl(var(--pencil))'
  return (
    <span style={{
      fontSize: '.6rem', fontWeight: 700, fontFamily: 'Inter, sans-serif',
      color, background: `${color}18`, border: `1px solid ${color}40`,
      borderRadius: '999px', padding: '.06rem .42rem',
      letterSpacing: '.03em',
    }}>
      {pct}% {method === 'embedding' ? '🎙' : ''}
    </span>
  )
}

// ── Dissolve Preview Modal ────────────────────────────────────────────────────

function DissolveModal({
  data,
  allLabels,
  onConfirm,
  onCancel,
  confirming,
}: {
  data: DissolvePreviewData
  allLabels: string[]
  onConfirm: (proposals: DissolveProposal[]) => void
  onCancel: () => void
  confirming: boolean
}) {
  const [edited, setEdited] = useState<DissolveProposal[]>(data.proposals)

  const updateProposal = (segId: string, newSpeaker: string) => {
    setEdited(prev => prev.map(p => p.seg_id === segId ? { ...p, proposed_speaker: newSpeaker } : p))
  }

  return (
    <div style={{
      position: 'fixed', inset: 0, zIndex: 9999,
      background: 'rgba(0,0,0,.55)', backdropFilter: 'blur(4px)',
      display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '1rem',
    }}>
      <div style={{
        background: 'hsl(var(--card))',
        border: '1.5px solid hsl(var(--border) / .35)',
        borderRadius: '16px',
        width: '100%', maxWidth: '540px',
        maxHeight: '80vh',
        display: 'flex', flexDirection: 'column',
        boxShadow: '0 24px 48px rgba(0,0,0,.25)',
        overflow: 'hidden',
      }}>
        {/* Header */}
        <div style={{
          padding: '1.2rem 1.4rem',
          borderBottom: '1px solid hsl(var(--border) / .2)',
          display: 'flex', alignItems: 'flex-start', gap: '12px', flexShrink: 0,
        }}>
          <div style={{
            width: 38, height: 38, borderRadius: '10px', flexShrink: 0,
            background: 'hsl(280,65%,58% / .12)', border: '1.5px solid hsl(280,65%,58% / .3)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <Shuffle size={18} style={{ color: 'hsl(280,65%,58%)' }} />
          </div>
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: '.9rem', fontWeight: 700, fontFamily: 'Inter, sans-serif', color: 'hsl(var(--ink))' }}>
              Dissolve <span style={{ color: 'hsl(280,65%,58%)' }}>{data.dissolving_speaker}</span>
            </div>
            <div style={{ fontSize: '.7rem', fontFamily: 'Inter, sans-serif', color: 'hsl(var(--pencil))', marginTop: '3px' }}>
              {data.audio_available
                ? `Reassigning ${edited.length} segment${edited.length !== 1 ? 's' : ''} using voice embeddings`
                : `Reassigning ${edited.length} segment${edited.length !== 1 ? 's' : ''} (audio unavailable — using stored similarity)`}
            </div>
            {!data.audio_available && (
              <div style={{
                marginTop: '6px', fontSize: '.65rem', fontFamily: 'Inter, sans-serif',
                color: 'hsl(45,90%,40%)', background: 'hsl(45,90%,50% / .1)',
                border: '1px solid hsl(45,90%,50% / .3)', borderRadius: '6px', padding: '.3rem .6rem',
                display: 'flex', alignItems: 'center', gap: '5px',
              }}>
                <AlertTriangle size={10} /> Audio file not found — similarity estimates may be less accurate
              </div>
            )}
          </div>
          <button onClick={onCancel} style={{ background: 'none', border: 'none', cursor: 'pointer', padding: '4px', color: 'hsl(var(--pencil))' }}>
            <X size={16} />
          </button>
        </div>

        {/* Proposals list */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '1rem 1.2rem', display: 'flex', flexDirection: 'column', gap: '8px' }}>
          {edited.map((p) => {
            const color = getSpeakerColor(p.proposed_speaker, data.remaining_speakers)
            return (
              <div key={p.seg_id} style={{
                background: 'hsl(var(--paper) / .5)',
                border: '1px solid hsl(var(--border) / .18)',
                borderRadius: '10px', padding: '.7rem .9rem',
                display: 'flex', flexDirection: 'column', gap: '6px',
              }}>
                <div style={{ fontSize: '.77rem', fontFamily: 'Inter, sans-serif', color: 'hsl(var(--ink-soft))', lineHeight: 1.45 }}>
                  "{p.text.length > 120 ? p.text.slice(0, 120) + '…' : p.text}"
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
                  <span style={{ fontSize: '.62rem', fontFamily: 'Inter, sans-serif', color: 'hsl(var(--pencil))' }}>
                    {formatTime(p.start)}
                  </span>
                  <CornerDownRight size={10} style={{ color: 'hsl(var(--pencil))' }} />
                  <select
                    value={p.proposed_speaker}
                    onChange={e => updateProposal(p.seg_id, e.target.value)}
                    style={{
                      fontSize: '.72rem', fontFamily: 'Inter, sans-serif',
                      fontWeight: 700, color: color.accent,
                      background: color.bg, border: `1px solid ${color.border}`,
                      borderRadius: '6px', padding: '.2rem .5rem', cursor: 'pointer',
                      outline: 'none',
                    }}
                  >
                    {data.remaining_speakers.map(s => (
                      <option key={s} value={s}>{s}</option>
                    ))}
                  </select>
                  {p.similarity != null && (
                    <SimilarityBadge score={p.similarity} method={p.method} />
                  )}
                </div>
              </div>
            )
          })}
        </div>

        {/* Footer */}
        <div style={{
          padding: '.9rem 1.2rem',
          borderTop: '1px solid hsl(var(--border) / .2)',
          display: 'flex', gap: '10px', justifyContent: 'flex-end', flexShrink: 0,
        }}>
          <button
            onClick={onCancel}
            disabled={confirming}
            style={{
              padding: '.5rem 1rem', borderRadius: '8px', fontFamily: 'Inter, sans-serif',
              fontSize: '.78rem', fontWeight: 600, cursor: 'pointer',
              background: 'hsl(var(--muted))', border: '1px solid hsl(var(--border) / .3)',
              color: 'hsl(var(--ink))', opacity: confirming ? 0.5 : 1,
            }}
          >
            Cancel
          </button>
          <button
            onClick={() => onConfirm(edited)}
            disabled={confirming}
            style={{
              padding: '.5rem 1.2rem', borderRadius: '8px', fontFamily: 'Inter, sans-serif',
              fontSize: '.78rem', fontWeight: 700, cursor: 'pointer',
              background: 'hsl(280,65%,58%)', border: 'none',
              color: '#fff', display: 'flex', alignItems: 'center', gap: '6px',
              opacity: confirming ? 0.7 : 1,
            }}
          >
            {confirming ? <Loader size={13} className="spin" /> : <CheckCircle size={13} />}
            Confirm Dissolve
          </button>
        </div>
      </div>
    </div>
  )
}

// ── Speaker Card ──────────────────────────────────────────────────────────────

function SpeakerCard({
  group,
  allLabels,
  voiceProfiles,
  onJump,
  onMoveSegment,
  onDeleteSegment,
  onDeleteSpeaker,
  onDissolve,
}: {
  group: SpeakerGroup
  allLabels: string[]
  voiceProfiles: VoiceProfile[]
  onJump: (seg: SegmentRow) => void
  onMoveSegment: (segId: string, newLabel: string) => void
  onDeleteSegment: (segId: string) => void
  onDeleteSpeaker: (label: string) => void
  onDissolve: (label: string) => void
}) {
  const [collapsed, setCollapsed] = useState(false)
  const [speakerMenuOpen, setSpeakerMenuOpen] = useState(false)
  const [movingSegId, setMovingSegId] = useState<string | null>(null)
  const [deletingSegId, setDeletingSegId] = useState<string | null>(null)
  const [customName, setCustomName] = useState('')

  const color = getSpeakerColor(group.speaker_label, allLabels)
  const otherSpeakers = allLabels.filter(l => l !== group.speaker_label)
  const initials = group.speaker_label.split(' ').map(w => w[0]).join('').toUpperCase().slice(0, 2)

  return (
    <div style={{
      border: `1.5px solid ${color.border}`,
      borderRadius: '12px',
      background: color.bg,
      overflow: 'visible',
      position: 'relative',
      transition: 'box-shadow .2s',
    }}>
      {/* Speaker header */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: '10px',
        padding: '.7rem .9rem',
        borderBottom: collapsed ? 'none' : `1px solid ${color.border}`,
        borderTopLeftRadius: '11px',
        borderTopRightRadius: '11px',
      }}>
        {/* Avatar */}
        <div style={{
          width: 32, height: 32, borderRadius: '50%', flexShrink: 0,
          background: `${color.accent}20`, border: `2px solid ${color.accent}50`,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: '.72rem', fontWeight: 800, fontFamily: 'Inter, sans-serif',
          color: color.accent,
        }}>
          {initials}
        </div>

        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ fontSize: '.82rem', fontWeight: 700, fontFamily: 'Inter, sans-serif', color: color.accent }}>
            {group.speaker_label}
          </div>
          <div style={{ fontSize: '.62rem', fontFamily: 'Inter, sans-serif', color: 'hsl(var(--pencil))' }}>
            {group.segments.length} segment{group.segments.length !== 1 ? 's' : ''}
          </div>
        </div>

        {/* Speaker actions menu */}
        <div style={{ position: 'relative' }}>
          <button
            id={`speaker-menu-btn-${group.speaker_label.replace(/\s+/g, '-')}`}
            onClick={() => setSpeakerMenuOpen(o => !o)}
            style={{
              background: 'none', border: 'none', cursor: 'pointer',
              padding: '4px', borderRadius: '6px', color: 'hsl(var(--pencil))',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
            }}
            title="Speaker options"
          >
            <MoreVertical size={15} />
          </button>
          {speakerMenuOpen && (
            <>
              <div
                style={{ position: 'fixed', inset: 0, zIndex: 50 }}
                onClick={() => setSpeakerMenuOpen(false)}
              />
              <div style={{
                position: 'absolute', right: 0, top: '100%', marginTop: '4px',
                background: 'hsl(var(--card))',
                border: '1px solid hsl(var(--border) / .3)',
                borderRadius: '10px', padding: '.4rem',
                boxShadow: '0 8px 20px rgba(0,0,0,.15)',
                zIndex: 100, minWidth: '180px',
              }}>
                <button
                  id={`dissolve-speaker-${group.speaker_label.replace(/\s+/g, '-')}`}
                  onClick={() => { setSpeakerMenuOpen(false); onDissolve(group.speaker_label) }}
                  style={{
                    display: 'flex', alignItems: 'center', gap: '8px',
                    width: '100%', background: 'none', border: 'none', cursor: 'pointer',
                    padding: '.45rem .7rem', borderRadius: '7px', textAlign: 'left',
                    fontSize: '.77rem', fontFamily: 'Inter, sans-serif', fontWeight: 600,
                    color: 'hsl(280,65%,50%)',
                  }}
                  onMouseEnter={e => (e.currentTarget.style.background = 'hsl(280,65%,50% / .1)')}
                  onMouseLeave={e => (e.currentTarget.style.background = 'none')}
                >
                  <Shuffle size={13} /> Dissolve Speaker
                </button>
                <div style={{ height: '1px', background: 'hsl(var(--border) / .2)', margin: '.3rem 0' }} />
                <button
                  id={`delete-speaker-${group.speaker_label.replace(/\s+/g, '-')}`}
                  onClick={() => { setSpeakerMenuOpen(false); onDeleteSpeaker(group.speaker_label) }}
                  style={{
                    display: 'flex', alignItems: 'center', gap: '8px',
                    width: '100%', background: 'none', border: 'none', cursor: 'pointer',
                    padding: '.45rem .7rem', borderRadius: '7px', textAlign: 'left',
                    fontSize: '.77rem', fontFamily: 'Inter, sans-serif', fontWeight: 600,
                    color: 'hsl(var(--destructive))',
                  }}
                  onMouseEnter={e => (e.currentTarget.style.background = 'hsl(var(--destructive) / .08)')}
                  onMouseLeave={e => (e.currentTarget.style.background = 'none')}
                >
                  <UserX size={13} /> Delete Speaker &amp; Segments
                </button>
              </div>
            </>
          )}
        </div>

        {/* Collapse toggle */}
        <button
          onClick={() => setCollapsed(o => !o)}
          style={{
            background: 'none', border: 'none', cursor: 'pointer',
            padding: '4px', borderRadius: '6px', color: 'hsl(var(--pencil))',
            display: 'flex', alignItems: 'center',
          }}
        >
          {collapsed ? <ChevronDown size={15} /> : <ChevronUp size={15} />}
        </button>
      </div>

      {/* Segment list */}
      {!collapsed && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 0 }}>
          {group.segments.map((seg, idx) => (
            <div
              key={seg.seg_id}
              style={{
                padding: '.6rem .9rem',
                borderTop: idx > 0 ? `1px solid ${color.border}` : 'none',
                display: 'flex', flexDirection: 'column', gap: '5px',
                transition: 'background .15s',
              }}
              onMouseEnter={e => (e.currentTarget.style.background = `${color.accent}06`)}
              onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}
            >
              {/* Segment text */}
              <div style={{
                fontSize: '.78rem', fontFamily: 'Inter, sans-serif',
                color: 'hsl(var(--ink-soft))', lineHeight: 1.5,
              }}>
                {seg.text}
              </div>

              {/* Meta row */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
                <span style={{ fontSize: '.6rem', fontFamily: 'monospace', color: 'hsl(var(--pencil))' }}>
                  {formatTime(seg.start)} – {formatTime(seg.end)}
                </span>
                <SimilarityBadge score={seg.similarity} method={seg.scoring_method} />
                {seg.is_overlap && (
                  <span style={{
                    fontSize: '.58rem', fontWeight: 700, fontFamily: 'Inter, sans-serif',
                    color: 'hsl(45,90%,40%)', background: 'hsl(45,90%,50% / .12)',
                    border: '1px solid hsl(45,90%,50% / .3)', borderRadius: '4px', padding: '.04rem .36rem',
                  }}>overlap</span>
                )}

                {/* Spacer */}
                <div style={{ flex: 1 }} />

                {/* Jump button */}
                <button
                  id={`jump-seg-${seg.seg_id}`}
                  onClick={() => onJump(seg)}
                  title="Jump to this segment in transcript"
                  style={{
                    display: 'flex', alignItems: 'center', gap: '3px',
                    background: 'none', border: `1px solid ${color.border}`,
                    borderRadius: '5px', padding: '.15rem .45rem', cursor: 'pointer',
                    fontSize: '.6rem', fontFamily: 'Inter, sans-serif', fontWeight: 700,
                    color: color.accent, transition: 'all .15s',
                  }}
                  onMouseEnter={e => { e.currentTarget.style.background = color.bg }}
                  onMouseLeave={e => { e.currentTarget.style.background = 'none' }}
                >
                  <Play size={8} /> Jump
                </button>

                {/* Move to dropdown */}
                <div style={{ position: 'relative' }}>
                  <button
                    id={`move-seg-${seg.seg_id}`}
                    onClick={() => {
                      setMovingSegId(m => m === seg.seg_id ? null : seg.seg_id)
                      setCustomName('')
                    }}
                    title="Move segment to another speaker or create new speaker"
                    style={{
                      display: 'flex', alignItems: 'center', gap: '3px',
                      background: 'none', border: '1px solid hsl(var(--border) / .3)',
                      borderRadius: '5px', padding: '.15rem .45rem', cursor: 'pointer',
                      fontSize: '.6rem', fontFamily: 'Inter, sans-serif', fontWeight: 700,
                      color: 'hsl(var(--pencil))', transition: 'all .15s',
                    }}
                    onMouseEnter={e => { e.currentTarget.style.background = 'hsl(var(--muted))' }}
                    onMouseLeave={e => { e.currentTarget.style.background = 'none' }}
                  >
                    <ArrowRight size={8} /> Move
                  </button>
                  {movingSegId === seg.seg_id && (
                    <>
                      <div style={{ position: 'fixed', inset: 0, zIndex: 50 }} onClick={() => setMovingSegId(null)} />
                      <div style={{
                        position: 'absolute', right: 0, bottom: '100%', marginBottom: '6px',
                        background: 'hsl(var(--card))',
                        border: '1.5px solid hsl(var(--border) / .35)',
                        borderRadius: '10px', padding: '.45rem',
                        boxShadow: '0 12px 28px rgba(0,0,0,.25)',
                        zIndex: 100, minWidth: '200px', maxWidth: '250px',
                        display: 'flex', flexDirection: 'column', gap: '6px',
                      }}>
                        <div style={{
                          fontSize: '.62rem', fontWeight: 700, fontFamily: 'Inter, sans-serif',
                          color: 'hsl(var(--pencil))', textTransform: 'uppercase', letterSpacing: '.05em',
                          borderBottom: '1px solid hsl(var(--border) / .15)', paddingBottom: '.25rem',
                        }}>
                          Move to Speaker
                        </div>

                        <div style={{
                          maxHeight: '180px', overflowY: 'auto',
                          display: 'flex', flexDirection: 'column', gap: '6px',
                        }}>
                          {/* Section 1: Meeting Speakers */}
                          {otherSpeakers.length > 0 && (
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
                              <span style={{ fontSize: '.58rem', fontWeight: 700, fontFamily: 'Inter, sans-serif', color: 'hsl(var(--pencil))', letterSpacing: '.03em' }}>
                                MEETING SPEAKERS
                              </span>
                              {otherSpeakers.map(sp => {
                                const spColor = getSpeakerColor(sp, allLabels)
                                return (
                                  <button
                                    key={sp}
                                    onClick={() => { setMovingSegId(null); onMoveSegment(seg.seg_id, sp) }}
                                    style={{
                                      display: 'flex', alignItems: 'center', gap: '6px',
                                      width: '100%', background: 'none', border: 'none',
                                      cursor: 'pointer', padding: '.3rem .5rem', borderRadius: '6px',
                                      textAlign: 'left', fontSize: '.73rem', fontFamily: 'Inter, sans-serif',
                                      fontWeight: 700, color: spColor.accent, whiteSpace: 'nowrap',
                                    }}
                                    onMouseEnter={e => (e.currentTarget.style.background = spColor.bg)}
                                    onMouseLeave={e => (e.currentTarget.style.background = 'none')}
                                  >
                                    <ArrowRight size={10} style={{ flexShrink: 0 }} />
                                    <span style={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>{sp}</span>
                                  </button>
                                )
                              })}
                            </div>
                          )}

                          {/* Section 2: Saved Voice Profiles */}
                          {voiceProfiles.filter(p => p.label !== group.speaker_label && !otherSpeakers.includes(p.label)).length > 0 && (
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
                              <span style={{ fontSize: '.58rem', fontWeight: 700, fontFamily: 'Inter, sans-serif', color: 'hsl(var(--pencil))', letterSpacing: '.03em' }}>
                                SAVED VOICE PROFILES
                              </span>
                              {voiceProfiles.filter(p => p.label !== group.speaker_label && !otherSpeakers.includes(p.label)).map(p => (
                                <button
                                  key={p.id}
                                  onClick={() => { setMovingSegId(null); onMoveSegment(seg.seg_id, p.label) }}
                                  style={{
                                    display: 'flex', alignItems: 'center', gap: '6px',
                                    width: '100%', background: 'none', border: 'none',
                                    cursor: 'pointer', padding: '.3rem .5rem', borderRadius: '6px',
                                    textAlign: 'left', fontSize: '.73rem', fontFamily: 'Inter, sans-serif',
                                    fontWeight: 700, color: 'hsl(var(--accent))', whiteSpace: 'nowrap',
                                  }}
                                  onMouseEnter={e => (e.currentTarget.style.background = 'hsl(var(--accent) / .1)')}
                                  onMouseLeave={e => (e.currentTarget.style.background = 'none')}
                                >
                                  <UserCheck size={11} style={{ flexShrink: 0, color: 'hsl(var(--accent))' }} />
                                  <span style={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>
                                    {p.label} {p.is_self ? '(Self)' : ''}
                                  </span>
                                </button>
                              ))}
                            </div>
                          )}

                          {/* Section 3: Create New Speaker */}
                          <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', paddingTop: '2px', borderTop: '1px dashed hsl(var(--border) / .2)' }}>
                            <span style={{ fontSize: '.58rem', fontWeight: 700, fontFamily: 'Inter, sans-serif', color: 'hsl(var(--pencil))', letterSpacing: '.03em' }}>
                              NEW SPEAKER
                            </span>
                            <form
                              onSubmit={(e) => {
                                e.preventDefault()
                                const trimmed = customName.trim()
                                if (trimmed) {
                                  onMoveSegment(seg.seg_id, trimmed)
                                  setMovingSegId(null)
                                  setCustomName('')
                                }
                              }}
                              style={{ display: 'flex', gap: '4px' }}
                            >
                              <input
                                type="text"
                                placeholder="Speaker name..."
                                value={customName}
                                onChange={e => setCustomName(e.target.value)}
                                style={{
                                  flex: 1, minWidth: 0, fontSize: '.72rem', fontFamily: 'Inter, sans-serif',
                                  padding: '.25rem .45rem', borderRadius: '5px',
                                  border: '1px solid hsl(var(--border) / .35)',
                                  background: 'hsl(var(--card))', color: 'hsl(var(--ink))',
                                  outline: 'none',
                                }}
                              />
                              <button
                                type="submit"
                                disabled={!customName.trim()}
                                style={{
                                  fontSize: '.68rem', fontFamily: 'Inter, sans-serif', fontWeight: 700,
                                  padding: '.25rem .5rem', borderRadius: '5px',
                                  background: customName.trim() ? 'hsl(var(--accent))' : 'hsl(var(--muted))',
                                  color: customName.trim() ? '#fff' : 'hsl(var(--pencil))',
                                  border: 'none', cursor: customName.trim() ? 'pointer' : 'default',
                                  flexShrink: 0, display: 'flex', alignItems: 'center', gap: '2px',
                                }}
                              >
                                <Plus size={10} /> Add
                              </button>
                            </form>
                          </div>
                        </div>
                      </div>
                    </>
                  )}
                </div>

                {/* Delete segment */}
                <button
                  id={`delete-seg-${seg.seg_id}`}
                  onClick={() => { setDeletingSegId(seg.seg_id); onDeleteSegment(seg.seg_id) }}
                  title="Delete this segment"
                  disabled={deletingSegId === seg.seg_id}
                  style={{
                    display: 'flex', alignItems: 'center',
                    background: 'none', border: 'none', cursor: 'pointer',
                    padding: '.15rem .3rem', borderRadius: '5px',
                    color: 'hsl(var(--destructive) / .6)', transition: 'all .15s',
                    opacity: deletingSegId === seg.seg_id ? 0.5 : 1,
                  }}
                  onMouseEnter={e => { e.currentTarget.style.color = 'hsl(var(--destructive))'; e.currentTarget.style.background = 'hsl(var(--destructive) / .08)' }}
                  onMouseLeave={e => { e.currentTarget.style.color = 'hsl(var(--destructive) / .6)'; e.currentTarget.style.background = 'none' }}
                >
                  <Trash2 size={11} />
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ── Main SpeakerTab ───────────────────────────────────────────────────────────

export default function SpeakerTab({ recordingId, onScrollToSegment, onTranscriptChanged }: Props) {
  const [speakers, setSpeakers] = useState<SpeakerGroup[]>([])
  const [voiceProfiles, setVoiceProfiles] = useState<VoiceProfile[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Dissolve state
  const [dissolvePreviewData, setDissolvePreviewData] = useState<DissolvePreviewData | null>(null)
  const [dissolvePreviewing, setDissolvePreviewing] = useState(false)
  const [dissolveConfirming, setDissolveConfirming] = useState(false)

  const allLabels = speakers.map(s => s.speaker_label)

  const loadSpeakers = useCallback(async () => {
    if (!recordingId) return
    setLoading(true)
    setError(null)
    try {
      const [spkRes, vpRes] = await Promise.allSettled([
        api.get(`/speakers/${recordingId}`),
        api.get('/voice/profiles'),
      ])
      if (spkRes.status === 'fulfilled') {
        setSpeakers(spkRes.value.data.speakers || [])
      } else {
        setError(spkRes.reason?.response?.data?.detail || 'Failed to load speakers')
      }
      if (vpRes.status === 'fulfilled') {
        setVoiceProfiles(vpRes.value.data || [])
      }
    } finally {
      setLoading(false)
    }
  }, [recordingId])

  useEffect(() => {
    loadSpeakers()
  }, [loadSpeakers])

  const handleJump = (seg: SegmentRow) => {
    onScrollToSegment(seg.seg_id, seg.start)
  }

  const handleMoveSegment = async (segId: string, newLabel: string) => {
    try {
      await api.post(`/speakers/${recordingId}/move-segment`, {
        seg_id: segId,
        new_speaker_label: newLabel,
      })
      await loadSpeakers()
      onTranscriptChanged()
    } catch (err: any) {
      setError(err?.response?.data?.detail || 'Failed to move segment')
    }
  }

  const handleDeleteSegment = async (segId: string) => {
    try {
      await api.delete(`/speakers/${recordingId}/segment/${segId}`)
      await loadSpeakers()
      onTranscriptChanged()
    } catch (err: any) {
      setError(err?.response?.data?.detail || 'Failed to delete segment')
    }
  }

  const handleDeleteSpeaker = async (label: string) => {
    if (!window.confirm(`Delete "${label}" and all their segments? This cannot be undone.`)) return
    try {
      await api.delete(`/speakers/${recordingId}/speaker`, {
        data: { speaker_label: label, action: 'delete_segments' },
      })
      await loadSpeakers()
      onTranscriptChanged()
    } catch (err: any) {
      setError(err?.response?.data?.detail || 'Failed to delete speaker')
    }
  }

  const handleDissolveStart = async (label: string) => {
    setDissolvePreviewing(true)
    setError(null)
    try {
      const res = await api.post(`/speakers/${recordingId}/dissolve-preview`, {
        speaker_label: label,
      })
      setDissolvePreviewData(res.data)
    } catch (err: any) {
      setError(err?.response?.data?.detail || 'Failed to calculate dissolve preview')
    } finally {
      setDissolvePreviewing(false)
    }
  }

  const handleDissolveConfirm = async (proposals: DissolveProposal[]) => {
    if (!dissolvePreviewData) return
    setDissolveConfirming(true)
    try {
      await api.post(`/speakers/${recordingId}/dissolve-confirm`, {
        speaker_label: dissolvePreviewData.dissolving_speaker,
        assignments: proposals.map(p => ({
          seg_id: p.seg_id,
          new_speaker_label: p.proposed_speaker,
        })),
      })
      setDissolvePreviewData(null)
      await loadSpeakers()
      onTranscriptChanged()
    } catch (err: any) {
      setError(err?.response?.data?.detail || 'Failed to confirm dissolve')
    } finally {
      setDissolveConfirming(false)
    }
  }

  // ── Render ────────────────────────────────────────────────────────────────

  if (loading) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '3rem 1rem', gap: '12px' }}>
        <Loader size={20} className="spin" style={{ color: 'hsl(var(--accent))' }} />
        <span style={{ fontSize: '.78rem', fontFamily: 'Inter, sans-serif', color: 'hsl(var(--pencil))' }}>
          Loading speakers…
        </span>
      </div>
    )
  }

  if (dissolvePreviewing) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '3rem 1rem', gap: '12px' }}>
        <Loader size={20} className="spin" style={{ color: 'hsl(280,65%,58%)' }} />
        <span style={{ fontSize: '.78rem', fontFamily: 'Inter, sans-serif', color: 'hsl(var(--pencil))' }}>
          Calculating reassignments using voice embeddings…
        </span>
      </div>
    )
  }

  return (
    <>
      {/* Dissolve modal */}
      {dissolvePreviewData && (
        <DissolveModal
          data={dissolvePreviewData}
          allLabels={allLabels}
          onConfirm={handleDissolveConfirm}
          onCancel={() => setDissolvePreviewData(null)}
          confirming={dissolveConfirming}
        />
      )}

      <div style={{ padding: '.85rem .9rem', display: 'flex', flexDirection: 'column', gap: '10px' }}>
        {/* Error banner */}
        {error && (
          <div style={{
            display: 'flex', alignItems: 'center', gap: '8px',
            padding: '.5rem .75rem',
            background: 'hsl(var(--destructive) / .08)',
            border: '1px solid hsl(var(--destructive) / .3)',
            borderRadius: '8px',
            fontSize: '.73rem', fontFamily: 'Inter, sans-serif',
            color: 'hsl(var(--destructive))',
          }}>
            <AlertTriangle size={13} style={{ flexShrink: 0 }} />
            {error}
            <button onClick={() => setError(null)} style={{ marginLeft: 'auto', background: 'none', border: 'none', cursor: 'pointer', color: 'hsl(var(--destructive))', padding: 0 }}>
              <X size={13} />
            </button>
          </div>
        )}

        {/* Empty state */}
        {speakers.length === 0 && !loading && (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '2.5rem 1rem', gap: '12px', textAlign: 'center' }}>
            <div style={{
              width: 52, height: 52, borderRadius: '14px',
              background: 'hsl(var(--accent) / .1)', border: '1.5px solid hsl(var(--accent) / .2)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
            }}>
              <Users size={22} style={{ color: 'hsl(var(--accent))' }} />
            </div>
            <div>
              <p style={{ fontFamily: 'Inter, sans-serif', fontSize: '.85rem', color: 'hsl(var(--ink))', fontWeight: 600, marginBottom: '.25rem' }}>
                No speakers found
              </p>
              <p style={{ fontFamily: 'Inter, sans-serif', fontSize: '.75rem', color: 'hsl(var(--pencil))', lineHeight: 1.5 }}>
                Process a recording with speaker identification enabled to see speakers here.
              </p>
            </div>
          </div>
        )}

        {/* Speaker cards */}
        {speakers.map(group => (
          <SpeakerCard
            key={group.speaker_label}
            group={group}
            allLabels={allLabels}
            voiceProfiles={voiceProfiles}
            onJump={handleJump}
            onMoveSegment={handleMoveSegment}
            onDeleteSegment={handleDeleteSegment}
            onDeleteSpeaker={handleDeleteSpeaker}
            onDissolve={handleDissolveStart}
          />
        ))}
      </div>
    </>
  )
}
