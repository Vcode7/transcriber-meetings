import React from 'react'
import {
  X, BookOpen, FileText, Layers, Clock, User, Sparkles,
  Info
} from 'lucide-react'

export interface ContextGroupPoint {
  id: string
  discussion_point: string
  timeline_start: number
  timeline_end: number
  speakers?: string[]
  action_owner?: string | null
  technical_terms?: string[]
}

export interface ContextChunk {
  filename: string
  text: string
  score: number
  section?: string | null
}

export interface PreviousMeetingChunk {
  meeting_name: string
  date?: string
  text: string
  score: number
  speakers?: string
  action_owner?: string
}

export interface Stage2ContextGroup {
  group_index: number
  points_count: number
  timeline_start: number
  timeline_end: number
  points: ContextGroupPoint[]
  context: {
    meeting_context: ContextChunk[]
    global_context: ContextChunk[]
    previous_meeting_context: PreviousMeetingChunk[]
  }
}

const getStage1OwnerText = (p: any): string | null => {
  if (!p) return null
  let owner = p.action_owner || p.action_owners || p.owner || p.assignee
  if (!owner && Array.isArray(p.action_items) && p.action_items.length > 0) {
    const list: string[] = []
    for (const a of p.action_items) {
      if (typeof a === 'object' && a !== null) {
        const o = a.assignee || a.owner || a.action_owner
        if (o && String(o).toLowerCase() !== 'null' && String(o).toLowerCase() !== 'none') list.push(String(o))
      }
    }
    if (list.length > 0) owner = list.join(', ')
  }
  if (!owner && p.discussion_point && typeof p.discussion_point === 'string') {
    const m = p.discussion_point.match(/\((?:Owner|Assignee|Action Owner):\s*([^)]+)\)/i) ||
              p.discussion_point.match(/\[(?:Owner|Assignee|Action Owner):\s*([^\]]+)\]/i)
    if (m && m[1]) owner = m[1].trim()
  }
  if (!owner) return null
  const str = String(owner).trim()
  if (['null', 'none', 'n/a', 'undefined', 'unassigned', ''].includes(str.toLowerCase())) return null
  return str
}

interface Stage2ContextPreviewModalProps {
  isOpen: boolean
  onClose: () => void
  groups: Stage2ContextGroup[]
  totalGroups?: number
  totalPoints: number
  onEnhance: () => void
  isEnhancing?: boolean
}

function fmtTime(seconds: number): string {
  if (!seconds || isNaN(seconds) || seconds < 0) return '00:00'
  const total = Math.round(seconds)
  const h = Math.floor(total / 3600)
  const m = Math.floor((total % 3600) / 60)
  const s = total % 60
  if (h > 0) {
    return `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
  }
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
}

export const Stage2ContextPreviewModal: React.FC<Stage2ContextPreviewModalProps> = ({
  isOpen,
  onClose,
  groups,
  totalGroups,
  totalPoints,
  onEnhance,
  isEnhancing = false,
}) => {
  if (!isOpen) return null

  const effectiveTotalGroups = totalGroups && totalGroups > groups.length ? totalGroups : groups.length

  const scrollToGroup = (groupIndex: number) => {
    const el = document.getElementById(`preview-group-card-${groupIndex}`)
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
  }

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        backgroundColor: 'rgba(0, 0, 0, 0.7)',
        backdropFilter: 'blur(4px)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 9999,
        padding: '1.25rem',
      }}
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose()
      }}
    >
      <div
        style={{
          background: 'hsl(var(--card))',
          border: '1.5px solid hsl(var(--border)/.8)',
          borderRadius: 14,
          width: '100%',
          maxWidth: 980,
          height: '92vh',
          maxHeight: '92vh',
          display: 'flex',
          flexDirection: 'column',
          boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.35)',
          overflow: 'hidden',
        }}
      >
        {/* ── Modal Header ── */}
        <div
          style={{
            flexShrink: 0,
            padding: '1rem 1.4rem',
            borderBottom: '1px solid hsl(var(--border)/.4)',
            background: 'hsl(205, 90%, 55%/.06)',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            gap: '1rem',
          }}
        >
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
              <span style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '1.05rem', fontWeight: 700, color: 'hsl(var(--ink))' }}>
                <BookOpen size={19} style={{ color: 'hsl(205,90%,55%)' }} />
                Stage 2 Context Retrieval Preview
              </span>
              <span
                style={{
                  fontSize: '.72rem',
                  fontWeight: 700,
                  background: 'hsl(205,90%,55%/.15)',
                  color: 'hsl(205,90%,50%)',
                  padding: '2px 8px',
                  borderRadius: 12,
                }}
              >
                {groups.length < effectiveTotalGroups
                  ? `Previewing Initial ${groups.length} of ${effectiveTotalGroups} Groups`
                  : `${groups.length} Groups`} • {totalPoints} Total Points
              </span>
            </div>

            <div style={{ fontSize: '.76rem', color: 'hsl(var(--pencil))', marginTop: 3 }}>
              {groups.length < effectiveTotalGroups
                ? `Previewing context for the initial ${groups.length} groups. When you click Enhance Points, all ${effectiveTotalGroups} groups will be processed.`
                : 'Preview of retrieved context per point group. Spoken transcript evidence remains primary truth.'}
            </div>

            {/* Quick Navigation Jump Bar */}
            {groups.length > 1 && (
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 8, flexWrap: 'wrap' }}>
                <span style={{ fontSize: '.68rem', fontWeight: 600, color: 'hsl(var(--pencil))', textTransform: 'uppercase', letterSpacing: '.04em' }}>
                  Jump to:
                </span>
                {groups.map((g) => (
                  <button
                    key={g.group_index}
                    type="button"
                    onClick={() => scrollToGroup(g.group_index)}
                    style={{
                      padding: '2px 8px',
                      borderRadius: 6,
                      border: '1px solid hsl(205,90%,55%/.3)',
                      background: 'hsl(var(--card))',
                      color: 'hsl(205,90%,45%)',
                      fontSize: '.68rem',
                      fontWeight: 600,
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      gap: 4,
                    }}
                  >
                    Group {g.group_index} ({g.points_count} pts)
                  </button>
                ))}
              </div>
            )}
          </div>

          <button
            type="button"
            onClick={onClose}
            style={{
              background: 'transparent',
              border: 'none',
              cursor: 'pointer',
              color: 'hsl(var(--pencil))',
              padding: 6,
              borderRadius: 6,
              display: 'flex',
              alignItems: 'center',
              flexShrink: 0,
            }}
            title="Close Preview"
          >
            <X size={20} />
          </button>
        </div>

        {/* ── Single Main Scrollable Area for Entire Modal ── */}
        <div
          style={{
            flex: '1 1 0%',
            minHeight: 0,
            overflowY: 'auto',
            overflowX: 'hidden',
            padding: '1.25rem 1.5rem',
            display: 'flex',
            flexDirection: 'column',
            gap: '1.5rem',
          }}
        >
          {groups.length === 0 ? (
            <div style={{ textAlign: 'center', padding: '3.5rem 1rem', color: 'hsl(var(--pencil))' }}>
              <Info size={36} style={{ margin: '0 auto .75rem', opacity: 0.5 }} />
              <div style={{ fontSize: '.95rem', fontWeight: 600 }}>No discussion point groups available to preview.</div>
            </div>
          ) : (
            groups.map((group) => {
              const totalContextCount =
                group.meeting_context.length +
                group.global_context.length +
                group.previous_meeting_context.length

              return (
                <div
                  key={group.group_index}
                  id={`preview-group-card-${group.group_index}`}
                  style={{
                    borderRadius: 12,
                    border: '1.5px solid hsl(var(--border)/.6)',
                    background: 'hsl(var(--card))',
                    display: 'flex',
                    flexDirection: 'column',
                    overflow: 'visible',
                    boxShadow: '0 2px 8px rgba(0,0,0,0.04)',
                  }}
                >
                  {/* Group Header Banner */}
                  <div
                    style={{
                      padding: '.85rem 1.15rem',
                      background: 'hsl(var(--muted)/.2)',
                      borderBottom: '1px solid hsl(var(--border)/.4)',
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      borderTopLeftRadius: 11,
                      borderTopRightRadius: 11,
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                      <span
                        style={{
                          background: 'hsl(205,90%,55%)',
                          color: 'white',
                          fontWeight: 700,
                          fontSize: '.75rem',
                          padding: '3px 9px',
                          borderRadius: 6,
                          letterSpacing: '.02em',
                        }}
                      >
                        Group {group.group_index}
                      </span>
                      <span style={{ fontSize: '.88rem', fontWeight: 700, color: 'hsl(var(--ink))' }}>
                        {group.points_count} Discussion Point{group.points_count !== 1 ? 's' : ''}
                      </span>
                      <span style={{ fontSize: '.74rem', color: 'hsl(var(--pencil))' }}>
                        • {totalContextCount} context chunk{totalContextCount !== 1 ? 's' : ''} matched
                      </span>
                    </div>

                    {group.query_snippet && (
                      <div
                        style={{
                          fontSize: '.68rem',
                          color: 'hsl(var(--pencil))',
                          maxWidth: 320,
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap',
                          fontStyle: 'italic',
                        }}
                        title={group.query_snippet}
                      >
                        Query: &quot;{group.query_snippet}&quot;
                      </div>
                    )}
                  </div>

                  {/* Group Body: All content flows naturally without fixed heights */}
                  <div style={{ padding: '1.15rem', display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
                    
                    {/* Section 1: Points in this Group */}
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '.45rem' }}>
                      <div style={{ fontSize: '.72rem', fontWeight: 700, color: 'hsl(var(--pencil))', textTransform: 'uppercase', letterSpacing: '.05em' }}>
                        Points in Group {group.group_index} ({group.points.length})
                      </div>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '.5rem' }}>
                        {group.points.map((p, pIdx) => (
                          <div
                            key={p.id || pIdx}
                            style={{
                              padding: '.65rem .85rem',
                              borderRadius: 8,
                              background: 'hsl(var(--paper)/.5)',
                              border: '1px solid hsl(var(--border)/.45)',
                              display: 'flex',
                              flexDirection: 'column',
                              gap: 5,
                            }}
                          >
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 8 }}>
                              <div style={{ fontSize: '.82rem', color: 'hsl(var(--ink))', lineHeight: 1.45, flex: 1, wordBreak: 'break-word' }}>
                                <span style={{ fontWeight: 600, color: 'hsl(var(--pencil))', marginRight: 6 }}>
                                  #{pIdx + 1}.
                                </span>
                                {p.discussion_point}
                              </div>
                              <span
                                style={{
                                  fontSize: '.66rem',
                                  fontFamily: 'JetBrains Mono, monospace',
                                  color: 'hsl(var(--pencil))',
                                  background: 'hsl(var(--muted)/.5)',
                                  padding: '2px 7px',
                                  borderRadius: 4,
                                  flexShrink: 0,
                                }}
                              >
                                {fmtTime(p.timeline_start)} – {fmtTime(p.timeline_end)}
                              </span>
                            </div>

                            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, alignItems: 'center' }}>
                              {p.speakers && p.speakers.length > 0 && (
                                <span style={{ fontSize: '.67rem', color: 'hsl(var(--pencil))', display: 'inline-flex', alignItems: 'center', gap: 3 }}>
                                  <User size={11} /> {p.speakers.join(', ')}
                                </span>
                              )}
                              {(() => {
                                const owner = getStage1OwnerText(p)
                                return owner ? (
                                  <span style={{ fontSize: '.67rem', color: 'hsl(280,75%,60%)', background: 'hsl(280,75%,60%/.1)', padding: '1px 6px', borderRadius: 4, fontWeight: 600 }}>
                                    Action: {owner}
                                  </span>
                                ) : (
                                  <span style={{ fontSize: '.67rem', color: 'hsl(var(--pencil))', background: 'hsl(var(--muted)/.5)', padding: '1px 6px', borderRadius: 4, fontWeight: 600 }}>
                                    No action owner
                                  </span>
                                )
                              })()}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Section 2: Context retrieved from Current Meeting */}
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '.45rem' }}>
                      <div style={{ fontSize: '.72rem', fontWeight: 700, color: 'hsl(205,90%,50%)', textTransform: 'uppercase', letterSpacing: '.05em', display: 'flex', alignItems: 'center', gap: 5 }}>
                        <FileText size={13} /> Context Retrieved from Current Meeting ({group.meeting_context.length})
                      </div>
                      {group.meeting_context.length === 0 ? (
                        <div style={{ fontSize: '.76rem', color: 'hsl(var(--pencil))', fontStyle: 'italic', padding: '.5rem .75rem', background: 'hsl(var(--paper)/.3)', borderRadius: 7, border: '1px dashed hsl(var(--border)/.4)' }}>
                          No matching current meeting context chunks found above similarity threshold.
                        </div>
                      ) : (
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '.5rem' }}>
                          {group.meeting_context.map((chunk, cIdx) => (
                            <div
                              key={cIdx}
                              style={{
                                padding: '.65rem .85rem',
                                borderRadius: 8,
                                background: 'hsl(205,90%,55%/.04)',
                                border: '1px solid hsl(205,90%,55%/.25)',
                                display: 'flex',
                                flexDirection: 'column',
                                gap: 5,
                              }}
                            >
                              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 6 }}>
                                <span style={{ fontSize: '.76rem', fontWeight: 700, color: 'hsl(205,90%,45%)' }}>
                                  {chunk.filename} {chunk.section ? `• ${chunk.section}` : ''}
                                </span>
                                <span style={{ fontSize: '.66rem', fontWeight: 600, color: 'hsl(205,90%,45%)', background: 'hsl(205,90%,55%/.12)', padding: '1px 6px', borderRadius: 8, fontFamily: 'JetBrains Mono' }}>
                                  Score: {chunk.score}
                                </span>
                              </div>
                              <div style={{ fontSize: '.76rem', color: 'hsl(var(--ink))', lineHeight: 1.5, whiteSpace: 'pre-wrap', wordBreak: 'break-word', overflowWrap: 'anywhere' }}>
                                {chunk.text}
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>

                    {/* Section 3: Context retrieved from Global Meeting Knowledge */}
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '.45rem' }}>
                      <div style={{ fontSize: '.72rem', fontWeight: 700, color: 'hsl(30,90%,50%)', textTransform: 'uppercase', letterSpacing: '.05em', display: 'flex', alignItems: 'center', gap: 5 }}>
                        <Layers size={13} /> Context Retrieved from Global Knowledge ({group.global_context.length})
                      </div>
                      {group.global_context.length === 0 ? (
                        <div style={{ fontSize: '.76rem', color: 'hsl(var(--pencil))', fontStyle: 'italic', padding: '.5rem .75rem', background: 'hsl(var(--paper)/.3)', borderRadius: 7, border: '1px dashed hsl(var(--border)/.4)' }}>
                          No matching global knowledge chunks found above similarity threshold.
                        </div>
                      ) : (
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '.5rem' }}>
                          {group.global_context.map((chunk, gIdx) => (
                            <div
                              key={gIdx}
                              style={{
                                padding: '.65rem .85rem',
                                borderRadius: 8,
                                background: 'hsl(30,90%,50%/.04)',
                                border: '1px solid hsl(30,90%,50%/.25)',
                                display: 'flex',
                                flexDirection: 'column',
                                gap: 5,
                              }}
                            >
                              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 6 }}>
                                <span style={{ fontSize: '.76rem', fontWeight: 700, color: 'hsl(30,90%,45%)' }}>
                                  {chunk.filename} {chunk.section ? `• ${chunk.section}` : ''}
                                </span>
                                <span style={{ fontSize: '.66rem', fontWeight: 600, color: 'hsl(30,90%,45%)', background: 'hsl(30,90%,50%/.12)', padding: '1px 6px', borderRadius: 8, fontFamily: 'JetBrains Mono' }}>
                                  Score: {chunk.score}
                                </span>
                              </div>
                              <div style={{ fontSize: '.76rem', color: 'hsl(var(--ink))', lineHeight: 1.5, whiteSpace: 'pre-wrap', wordBreak: 'break-word', overflowWrap: 'anywhere' }}>
                                {chunk.text}
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>

                    {/* Section 4: Context retrieved from Previous Meetings */}
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '.45rem' }}>
                      <div style={{ fontSize: '.72rem', fontWeight: 700, color: 'hsl(280,75%,55%)', textTransform: 'uppercase', letterSpacing: '.05em', display: 'flex', alignItems: 'center', gap: 5 }}>
                        <Clock size={13} /> Context Retrieved from Relevant Previous Meetings ({group.previous_meeting_context.length})
                      </div>
                      {group.previous_meeting_context.length === 0 ? (
                        <div style={{ fontSize: '.76rem', color: 'hsl(var(--pencil))', fontStyle: 'italic', padding: '.5rem .75rem', background: 'hsl(var(--paper)/.3)', borderRadius: 7, border: '1px dashed hsl(var(--border)/.4)' }}>
                          No previous meeting Stage 2 points matched (or previous meeting mode is disabled).
                        </div>
                      ) : (
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '.5rem' }}>
                          {group.previous_meeting_context.map((chunk, pmIdx) => (
                            <div
                              key={pmIdx}
                              style={{
                                padding: '.65rem .85rem',
                                borderRadius: 8,
                                background: 'hsl(280,75%,60%/.04)',
                                border: '1px solid hsl(280,75%,60%/.25)',
                                display: 'flex',
                                flexDirection: 'column',
                                gap: 5,
                              }}
                            >
                              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 6 }}>
                                <span style={{ fontSize: '.76rem', fontWeight: 700, color: 'hsl(280,75%,55%)' }}>
                                  {chunk.meeting_name} {chunk.date ? `(${chunk.date})` : ''}
                                </span>
                                <span style={{ fontSize: '.66rem', fontWeight: 600, color: 'hsl(280,75%,55%)', background: 'hsl(280,75%,60%/.12)', padding: '1px 6px', borderRadius: 8, fontFamily: 'JetBrains Mono' }}>
                                  Score: {chunk.score}
                                </span>
                              </div>
                              <div style={{ fontSize: '.76rem', color: 'hsl(var(--ink))', lineHeight: 1.5, whiteSpace: 'pre-wrap', wordBreak: 'break-word', overflowWrap: 'anywhere' }}>
                                {chunk.text}
                              </div>
                                <div style={{ display: 'flex', gap: 8, fontSize: '.67rem', color: 'hsl(var(--pencil))', marginTop: 2 }}>
                                  {chunk.speakers && <span>Speakers: {chunk.speakers}</span>}
                                  {chunk.action_owner ? (
                                    <span>Action: {chunk.action_owner}</span>
                                  ) : (
                                    <span>No action owner</span>
                                  )}
                                </div>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )
            })
          )}
        </div>

        {/* ── Modal Footer ── */}
        <div
          style={{
            flexShrink: 0,
            padding: '.9rem 1.4rem',
            borderTop: '1px solid hsl(var(--border)/.4)',
            background: 'hsl(var(--card))',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            gap: '1rem',
          }}
        >
          <div style={{ fontSize: '.74rem', color: 'hsl(var(--pencil))' }}>
            Clicking <strong>Enhance Points</strong> will perform fresh retrieval &amp; execute full enhancement across all groups.
          </div>

          <div style={{ display: 'flex', gap: 8, flexShrink: 0 }}>
            <button
              type="button"
              onClick={onClose}
              style={{
                padding: '.45rem .95rem',
                borderRadius: 7,
                border: '1px solid hsl(var(--border)/.5)',
                background: 'hsl(var(--muted)/.3)',
                color: 'hsl(var(--ink))',
                fontSize: '.76rem',
                fontWeight: 600,
                cursor: 'pointer',
              }}
            >
              Close Preview
            </button>
            <button
              type="button"
              onClick={() => {
                onClose()
                onEnhance()
              }}
              disabled={isEnhancing}
              style={{
                padding: '.45rem 1.1rem',
                borderRadius: 7,
                border: 'none',
                background: 'hsl(205, 90%, 55%)',
                color: 'white',
                fontSize: '.76rem',
                fontWeight: 700,
                cursor: isEnhancing ? 'not-allowed' : 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: 6,
              }}
            >
              <Sparkles size={13} />
              {isEnhancing ? 'Enhancing...' : 'Enhance Points'}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
