import React, { useState, useRef, useCallback } from 'react'
import {
  Clock, ArrowRightLeft, RefreshCw, ChevronLeft, ChevronRight,
  Sliders, Info, MoveHorizontal
} from 'lucide-react'

export interface AgendaItemSummary {
  agenda_id: string
  title: string
  description?: string
  speaker?: string | null
  presenter?: string | null
}

export interface AgendaTimelineRange {
  start_sec: number
  end_sec: number
}

interface AgendaTimelineControlsProps {
  agendas: AgendaItemSummary[]
  totalDuration: number // total meeting duration in seconds
  discussionOrder: string[]
  agendaTimeline: Record<string, AgendaTimelineRange>
  onChangeOrder: (newOrder: string[]) => void
  onChangeTimeline: (newTimeline: Record<string, AgendaTimelineRange>) => void
  onReset?: () => void
  enableOrder?: boolean
  onChangeEnableOrder?: (enabled: boolean) => void
  enableTimeline?: boolean
  onChangeEnableTimeline?: (enabled: boolean) => void
}

const AGENDA_COLORS = [
  'hsl(140, 70%, 45%)',
  'hsl(205, 90%, 52%)',
  'hsl(280, 75%, 60%)',
  'hsl(35, 90%, 50%)',
  'hsl(330, 75%, 55%)',
  'hsl(175, 80%, 40%)',
  'hsl(25, 95%, 55%)',
  'hsl(250, 70%, 65%)',
]

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

function parseTimeToSeconds(timeStr: string, fallback: number): number {
  if (!timeStr) return fallback
  const parts = timeStr.trim().split(':').map(Number)
  if (parts.some(isNaN)) return fallback
  if (parts.length === 3) {
    return parts[0] * 3600 + parts[1] * 60 + parts[2]
  }
  if (parts.length === 2) {
    return parts[0] * 60 + parts[1]
  }
  return fallback
}

export const AgendaTimelineControls: React.FC<AgendaTimelineControlsProps> = ({
  agendas,
  totalDuration,
  discussionOrder,
  agendaTimeline,
  onChangeOrder,
  onChangeTimeline,
  onReset,
  enableOrder = true,
  onChangeEnableOrder,
  enableTimeline = true,
  onChangeEnableTimeline,
}) => {
  const timelineBarRef = useRef<HTMLDivElement>(null)
  const [draggingBoundaryIdx, setDraggingBoundaryIdx] = useState<number | null>(null)
  const [dragTooltip, setDragTooltip] = useState<{ x: number; timeStr: string } | null>(null)

  const effectiveDuration = Math.max(totalDuration > 0 ? totalDuration : 1800, 60)

  // Map agenda_id to details
  const agendaMap = React.useMemo(() => {
    const map = new Map<string, AgendaItemSummary>()
    agendas.forEach(a => map.set(a.agenda_id, a))
    return map
  }, [agendas])

  // Color lookup by original agenda index
  const getColorForAgenda = useCallback((agendaId: string) => {
    const originalIdx = agendas.findIndex(a => a.agenda_id === agendaId)
    const idx = originalIdx >= 0 ? originalIdx : 0
    return AGENDA_COLORS[idx % AGENDA_COLORS.length]
  }, [agendas])

  // Reorder handler
  const moveOrder = (fromIdx: number, toIdx: number) => {
    if (toIdx < 0 || toIdx >= discussionOrder.length) return
    const updated = [...discussionOrder]
    const [moved] = updated.splice(fromIdx, 1)
    updated.splice(toIdx, 0, moved)
    onChangeOrder(updated)

    // Re-distribute timeline following the new order
    const count = updated.length
    if (count === 0) return
    const slice = effectiveDuration / count
    const newTimeline: Record<string, AgendaTimelineRange> = {}
    updated.forEach((id, idx) => {
      newTimeline[id] = {
        start_sec: Math.round(idx * slice * 10) / 10,
        end_sec: Math.round((idx + 1) * slice * 10) / 10,
      }
    })
    onChangeTimeline(newTimeline)
  }

  // Auto distribute timeline equally across discussion order
  const handleAutoDistribute = () => {
    const count = discussionOrder.length
    if (count === 0) return
    const slice = effectiveDuration / count
    const newTimeline: Record<string, AgendaTimelineRange> = {}
    discussionOrder.forEach((id, idx) => {
      newTimeline[id] = {
        start_sec: Math.round(idx * slice * 10) / 10,
        end_sec: Math.round((idx + 1) * slice * 10) / 10,
      }
    })
    onChangeTimeline(newTimeline)
  }

  // Handle Dragging Timeline Boundary between agenda i and i+1
  const startDragBoundary = (boundaryIdx: number, e: React.MouseEvent | React.TouchEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDraggingBoundaryIdx(boundaryIdx)

    const barEl = timelineBarRef.current
    if (!barEl) return

    const rect = barEl.getBoundingClientRect()

    const onMove = (moveEvt: MouseEvent | TouchEvent) => {
      const clientX = 'touches' in moveEvt ? moveEvt.touches[0].clientX : moveEvt.clientX
      const offsetX = Math.max(0, Math.min(rect.width, clientX - rect.left))
      const ratio = offsetX / rect.width
      const rawTargetSec = ratio * effectiveDuration

      const leftAgendaId = discussionOrder[boundaryIdx]
      const rightAgendaId = discussionOrder[boundaryIdx + 1]
      if (!leftAgendaId || !rightAgendaId) return

      const leftStart = agendaTimeline[leftAgendaId]?.start_sec ?? 0
      const rightEnd = agendaTimeline[rightAgendaId]?.end_sec ?? effectiveDuration

      // Minimum 5 seconds slice for each agenda
      const minSec = leftStart + 5
      const maxSec = rightEnd - 5

      const clampedSec = Math.max(minSec, Math.min(maxSec, rawTargetSec))

      const updated = { ...agendaTimeline }
      updated[leftAgendaId] = {
        ...(updated[leftAgendaId] || { start_sec: leftStart }),
        end_sec: Math.round(clampedSec * 10) / 10,
      }
      updated[rightAgendaId] = {
        ...(updated[rightAgendaId] || { end_sec: rightEnd }),
        start_sec: Math.round(clampedSec * 10) / 10,
      }

      onChangeTimeline(updated)
      setDragTooltip({
        x: offsetX,
        timeStr: fmtTime(clampedSec),
      })
    }

    const onUp = () => {
      setDraggingBoundaryIdx(null)
      setDragTooltip(null)
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
      window.removeEventListener('touchmove', onMove)
      window.removeEventListener('touchend', onUp)
    }

    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
    window.addEventListener('touchmove', onMove)
    window.addEventListener('touchend', onUp)
  }

  // Handle manual start/end time adjustments
  const handleManualTimeChange = (agendaId: string, type: 'start' | 'end', valStr: string) => {
    const orderIdx = discussionOrder.indexOf(agendaId)
    if (orderIdx === -1) return

    const currentRange = agendaTimeline[agendaId] || { start_sec: 0, end_sec: effectiveDuration }
    const currentVal = type === 'start' ? currentRange.start_sec : currentRange.end_sec
    const parsedSec = parseTimeToSeconds(valStr, currentVal)

    const updated = { ...agendaTimeline }

    if (type === 'start') {
      const prevAgendaId = orderIdx > 0 ? discussionOrder[orderIdx - 1] : null
      const minAllowed = prevAgendaId && updated[prevAgendaId] ? updated[prevAgendaId].start_sec + 5 : 0
      const maxAllowed = currentRange.end_sec - 5
      const clamped = Math.max(minAllowed, Math.min(maxAllowed, parsedSec))

      updated[agendaId] = { ...currentRange, start_sec: clamped }
      if (prevAgendaId && updated[prevAgendaId]) {
        updated[prevAgendaId] = { ...updated[prevAgendaId], end_sec: clamped }
      }
    } else {
      const nextAgendaId = orderIdx < discussionOrder.length - 1 ? discussionOrder[orderIdx + 1] : null
      const minAllowed = currentRange.start_sec + 5
      const maxAllowed = nextAgendaId && updated[nextAgendaId] ? updated[nextAgendaId].end_sec - 5 : effectiveDuration
      const clamped = Math.max(minAllowed, Math.min(maxAllowed, parsedSec))

      updated[agendaId] = { ...currentRange, end_sec: clamped }
      if (nextAgendaId && updated[nextAgendaId]) {
        updated[nextAgendaId] = { ...updated[nextAgendaId], start_sec: clamped }
      }
    }

    onChangeTimeline(updated)
  }

  if (agendas.length === 0) return null

  return (
    <div style={{
      borderRadius: 12,
      border: '1.5px solid hsl(var(--border)/.6)',
      background: 'hsl(var(--card))',
      padding: '1.15rem 1.25rem',
      display: 'flex',
      flexDirection: 'column',
      gap: '1rem',
      boxShadow: '0 2px 10px rgba(0,0,0,0.03)'
    }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 8 }}>
        <div>
          <div style={{ fontSize: '.92rem', fontWeight: 700, color: 'hsl(var(--ink))', display: 'flex', alignItems: 'center', gap: 7 }}>
            <Sliders size={16} style={{ color: 'hsl(140,70%,45%)' }} />
            Agenda Discussion Sequence &amp; Approximate Timeline
          </div>
          <div style={{ fontSize: '.74rem', color: 'hsl(var(--pencil))', marginTop: 3, display: 'flex', alignItems: 'center', gap: 5 }}>
            <Info size={12} style={{ color: 'hsl(205,90%,55%)', flexShrink: 0 }} />
            <span>
              Total Meeting Duration: <strong style={{ color: 'hsl(var(--ink))', fontFamily: 'JetBrains Mono, monospace' }}>{fmtTime(effectiveDuration)}</strong>.
              Sequence and timelines provide soft contextual guidance for LLM mapping while spoken evidence remains the ground truth.
            </span>
          </div>
        </div>

        {/* Quick Actions */}
        <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
          <button
            type="button"
            onClick={handleAutoDistribute}
            title="Evenly distribute timeline windows across agendas"
            style={{
              padding: '.32rem .65rem',
              borderRadius: 6,
              border: '1px solid hsl(140,70%,45%/.35)',
              background: 'hsl(140,70%,45%/.08)',
              color: 'hsl(140,70%,40%)',
              fontSize: '.72rem',
              fontWeight: 600,
              display: 'flex',
              alignItems: 'center',
              gap: 5,
              cursor: 'pointer'
            }}
          >
            <MoveHorizontal size={12} /> Auto-Distribute Timeline
          </button>
          {onReset && (
            <button
              type="button"
              onClick={onReset}
              title="Reset order and timeline to natural sequence"
              style={{
                padding: '.32rem .65rem',
                borderRadius: 6,
                border: '1px solid hsl(var(--border)/.5)',
                background: 'hsl(var(--muted)/.3)',
                color: 'hsl(var(--pencil))',
                fontSize: '.72rem',
                fontWeight: 600,
                display: 'flex',
                alignItems: 'center',
                gap: 5,
                cursor: 'pointer'
              }}
            >
              <RefreshCw size={11} /> Reset Order
            </button>
          )}
        </div>
      </div>

      {/* ── 1. DISCUSSION SEQUENCE CONTROLS ── */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '.4rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <label style={{ display: 'inline-flex', alignItems: 'center', gap: 7, cursor: 'pointer', userSelect: 'none' }}>
            <input
              type="checkbox"
              checked={enableOrder}
              onChange={(e) => onChangeEnableOrder?.(e.target.checked)}
              style={{ cursor: 'pointer', accentColor: 'hsl(140,70%,45%)' }}
            />
            <span style={{ fontSize: '.72rem', fontWeight: 700, color: enableOrder ? 'hsl(var(--ink))' : 'hsl(var(--pencil))', textTransform: 'uppercase', letterSpacing: '.04em' }}>
              1. Discussion Order Sequence
            </span>
          </label>
          {!enableOrder && (
            <span style={{ fontSize: '.68rem', color: 'hsl(38,90%,40%)', background: 'hsl(38,90%,50%/.12)', padding: '2px 7px', borderRadius: 4, fontWeight: 600 }}>
              Disabled — Omitted from LLM input
            </span>
          )}
        </div>

        <div style={{
          display: 'flex', flexWrap: 'wrap', gap: 6, alignItems: 'center',
          opacity: enableOrder ? 1 : 0.45,
          transition: 'opacity 0.2s ease',
          pointerEvents: enableOrder ? 'auto' : 'none'
        }}>
          {discussionOrder.map((aid, idx) => {
            const agenda = agendaMap.get(aid)
            const color = getColorForAgenda(aid)
            const isFirst = idx === 0
            const isLast = idx === discussionOrder.length - 1

            return (
              <React.Fragment key={aid}>
                <div style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: 6,
                  padding: '4px 8px',
                  borderRadius: 8,
                  border: `1.5px solid ${color}40`,
                  background: `${color}12`,
                  fontSize: '.76rem',
                  fontWeight: 600,
                  color: 'hsl(var(--ink))',
                  userSelect: 'none'
                }}>
                  <span style={{
                    background: color,
                    color: 'white',
                    padding: '1px 6px',
                    borderRadius: 4,
                    fontSize: '.68rem',
                    fontWeight: 700,
                    fontFamily: 'JetBrains Mono, monospace'
                  }} title={agenda?.title || aid}>
                    {idx + 1}. {aid}
                  </span>
                  

                  {/* Reorder Arrows */}
                  <div style={{ display: 'inline-flex', gap: 2, marginLeft: 2 }}>
                    <button
                      type="button"
                      onClick={() => moveOrder(idx, idx - 1)}
                      disabled={isFirst}
                      title="Move Left / Earlier in discussion"
                      style={{
                        background: 'transparent',
                        border: 'none',
                        padding: '1px 2px',
                        cursor: isFirst ? 'not-allowed' : 'pointer',
                        color: isFirst ? 'hsl(var(--border))' : 'hsl(var(--pencil))',
                        display: 'flex',
                        alignItems: 'center'
                      }}
                    >
                      <ChevronLeft size={13} />
                    </button>
                    <button
                      type="button"
                      onClick={() => moveOrder(idx, idx + 1)}
                      disabled={isLast}
                      title="Move Right / Later in discussion"
                      style={{
                        background: 'transparent',
                        border: 'none',
                        padding: '1px 2px',
                        cursor: isLast ? 'not-allowed' : 'pointer',
                        color: isLast ? 'hsl(var(--border))' : 'hsl(var(--pencil))',
                        display: 'flex',
                        alignItems: 'center'
                      }}
                    >
                      <ChevronRight size={13} />
                    </button>
                  </div>
                </div>

                {!isLast && (
                  <span style={{ color: 'hsl(var(--pencil)/.6)', fontWeight: 700, fontSize: '.8rem' }}>→</span>
                )}
              </React.Fragment>
            )
          })}
        </div>
      </div>

      {/* ── 2. VISUAL MULTI-SEGMENT TIMELINE BAR WITH DRAGGABLE PARTITIONS ── */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '.4rem', marginTop: '.2rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <label style={{ display: 'inline-flex', alignItems: 'center', gap: 7, cursor: 'pointer', userSelect: 'none' }}>
            <input
              type="checkbox"
              checked={enableTimeline}
              onChange={(e) => onChangeEnableTimeline?.(e.target.checked)}
              style={{ cursor: 'pointer', accentColor: 'hsl(140,70%,45%)' }}
            />
            <span style={{ fontSize: '.72rem', fontWeight: 700, color: enableTimeline ? 'hsl(var(--ink))' : 'hsl(var(--pencil))', textTransform: 'uppercase', letterSpacing: '.04em' }}>
              2. Approximate Timeline Windows (Drag Partitions to Adjust)
            </span>
          </label>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            {!enableTimeline && (
              <span style={{ fontSize: '.68rem', color: 'hsl(38,90%,40%)', background: 'hsl(38,90%,50%/.12)', padding: '2px 7px', borderRadius: 4, fontWeight: 600 }}>
                Disabled — Omitted from LLM input
              </span>
            )}
            <div style={{ fontSize: '.68rem', color: 'hsl(var(--pencil))', fontFamily: 'JetBrains Mono, monospace' }}>
              00:00 ── {fmtTime(effectiveDuration)}
            </div>
          </div>
        </div>

        {/* Continuous Bar Container */}
        <div
          ref={timelineBarRef}
          style={{
            position: 'relative',
            height: 42,
            borderRadius: 8,
            overflow: 'hidden',
            display: 'flex',
            background: 'hsl(var(--muted)/.3)',
            border: '1px solid hsl(var(--border)/.8)',
            userSelect: 'none',
            opacity: enableTimeline ? 1 : 0.45,
            transition: 'opacity 0.2s ease',
            pointerEvents: enableTimeline ? 'auto' : 'none',
          }}
        >
          {discussionOrder.map((aid, idx) => {
            const range = agendaTimeline[aid] || { start_sec: 0, end_sec: effectiveDuration }
            const startSec = Math.max(0, range.start_sec)
            const endSec = Math.min(effectiveDuration, range.end_sec)
            const durationSec = Math.max(0, endSec - startSec)
            const widthPct = (durationSec / effectiveDuration) * 100
            const color = getColorForAgenda(aid)
            const agenda = agendaMap.get(aid)

            return (
              <div
                key={aid}
                style={{
                  width: `${widthPct}%`,
                  height: '100%',
                  background: `${color}25`,
                  borderRight: idx < discussionOrder.length - 1 ? 'none' : 'none',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  padding: '0 6px',
                  position: 'relative',
                  overflow: 'hidden',
                  transition: draggingBoundaryIdx !== null ? 'none' : 'width 0.15s ease',
                  borderTop: `3px solid ${color}`,
                }}
                title={`${aid}: ${agenda?.title || ''} (${fmtTime(startSec)} – ${fmtTime(endSec)})`}
              >
                <div style={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  gap: 1,
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                  pointerEvents: 'none'
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                    <span style={{ fontSize: '.68rem', fontWeight: 800, color }}>{aid}</span>
                    {widthPct >= 18 && (
                      <span style={{ fontSize: '.64rem', color: 'hsl(var(--ink))', fontWeight: 600, overflow: 'hidden', textOverflow: 'ellipsis', maxWidth: 90 }}>
                        {agenda?.title}
                      </span>
                    )}
                  </div>
                  <span style={{ fontSize: '.62rem', fontFamily: 'JetBrains Mono, monospace', color: 'hsl(var(--pencil))' }}>
                    {fmtTime(startSec)} – {fmtTime(endSec)}
                  </span>
                </div>
              </div>
            )
          })}

          {/* Draggable Partition Boundary Handles */}
          {discussionOrder.slice(0, -1).map((aid, idx) => {
            const range = agendaTimeline[aid] || { start_sec: 0, end_sec: 0 }
            const boundarySec = range.end_sec
            const leftPct = (boundarySec / effectiveDuration) * 100
            const isDraggingThis = draggingBoundaryIdx === idx

            return (
              <div
                key={`handle-${idx}`}
                onMouseDown={(e) => startDragBoundary(idx, e)}
                onTouchStart={(e) => startDragBoundary(idx, e)}
                style={{
                  position: 'absolute',
                  left: `${leftPct}%`,
                  top: 0,
                  bottom: 0,
                  width: 14,
                  transform: 'translateX(-50%)',
                  cursor: 'ew-resize',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  zIndex: isDraggingThis ? 30 : 10,
                  touchAction: 'none',
                }}
                title="Drag to adjust boundary between agendas"
              >
                <div style={{
                  width: isDraggingThis ? 4 : 2.5,
                  height: '100%',
                  background: isDraggingThis ? 'hsl(45,95%,50%)' : 'hsl(var(--ink)/.4)',
                  borderRadius: 2,
                  boxShadow: isDraggingThis ? '0 0 8px hsl(45,95%,50%)' : '0 0 3px rgba(0,0,0,0.3)',
                  transition: 'all .1s',
                }} />
              </div>
            )
          })}

          {/* Active Dragging Tooltip */}
          {dragTooltip && (
            <div style={{
              position: 'absolute',
              left: dragTooltip.x,
              bottom: 2,
              transform: 'translateX(-50%)',
              background: 'hsl(var(--ink))',
              color: 'hsl(var(--paper))',
              fontSize: '.65rem',
              fontWeight: 700,
              padding: '1px 6px',
              borderRadius: 4,
              pointerEvents: 'none',
              zIndex: 40,
              fontFamily: 'JetBrains Mono, monospace',
              boxShadow: '0 2px 6px rgba(0,0,0,0.2)'
            }}>
              {dragTooltip.timeStr}
            </div>
          )}
        </div>
      </div>

      {/* ── 3. DETAILED PER-AGENDA TIME WINDOW EDITORS ── */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(210px, 1fr))',
        gap: '.6rem',
        marginTop: '.2rem',
        opacity: enableTimeline ? 1 : 0.45,
        transition: 'opacity 0.2s ease',
        pointerEvents: enableTimeline ? 'auto' : 'none'
      }}>
        {discussionOrder.map((aid, idx) => {
          const range = agendaTimeline[aid] || { start_sec: 0, end_sec: effectiveDuration }
          const color = getColorForAgenda(aid)
          const agenda = agendaMap.get(aid)
          const durSec = Math.max(0, range.end_sec - range.start_sec)
          const pct = Math.round((durSec / effectiveDuration) * 100)

          return (
            <div
              key={aid}
              style={{
                borderRadius: 8,
                border: `1.2px solid ${color}35`,
                background: 'hsl(var(--card))',
                padding: '.55rem .75rem',
                display: 'flex',
                flexDirection: 'column',
                gap: 5
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 5, overflow: 'hidden' }}>
                  <span style={{ background: color, color: 'white', fontSize: '.65rem', fontWeight: 700, padding: '1px 5px', borderRadius: 4, flexShrink: 0 }}>
                    {aid}
                  </span>
                  <span style={{ fontSize: '.72rem', fontWeight: 700, color: 'hsl(var(--ink))', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={agenda?.title || aid}>
                    {agenda?.title || aid}
                  </span>
                </div>
                <span style={{ fontSize: '.63rem', fontWeight: 600, color: 'hsl(var(--pencil))', background: 'hsl(var(--muted)/.4)', padding: '1px 5px', borderRadius: 4, fontFamily: 'JetBrains Mono' }}>
                  {pct}% (~{Math.round(durSec / 60)}m)
                </span>
              </div>

              {/* Time inputs */}
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 2 }}>
                <div style={{ flex: 1, display: 'flex', alignItems: 'center', gap: 3 }}>
                  <span style={{ fontSize: '.62rem', color: 'hsl(var(--pencil))' }}>Start</span>
                  <input
                    type="text"
                    defaultValue={fmtTime(range.start_sec)}
                    key={`start-${range.start_sec}`}
                    onBlur={(e) => handleManualTimeChange(aid, 'start', e.target.value)}
                    style={{
                      width: '100%',
                      padding: '2px 4px',
                      fontSize: '.68rem',
                      fontFamily: 'JetBrains Mono, monospace',
                      borderRadius: 4,
                      border: '1px solid hsl(var(--border)/.6)',
                      background: 'hsl(var(--paper))',
                      color: 'hsl(var(--ink))',
                      textAlign: 'center'
                    }}
                  />
                </div>
                <span style={{ fontSize: '.68rem', color: 'hsl(var(--pencil)/.6)' }}>–</span>
                <div style={{ flex: 1, display: 'flex', alignItems: 'center', gap: 3 }}>
                  <span style={{ fontSize: '.62rem', color: 'hsl(var(--pencil))' }}>End</span>
                  <input
                    type="text"
                    defaultValue={fmtTime(range.end_sec)}
                    key={`end-${range.end_sec}`}
                    onBlur={(e) => handleManualTimeChange(aid, 'end', e.target.value)}
                    style={{
                      width: '100%',
                      padding: '2px 4px',
                      fontSize: '.68rem',
                      fontFamily: 'JetBrains Mono, monospace',
                      borderRadius: 4,
                      border: '1px solid hsl(var(--border)/.6)',
                      background: 'hsl(var(--paper))',
                      color: 'hsl(var(--ink))',
                      textAlign: 'center'
                    }}
                  />
                </div>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
