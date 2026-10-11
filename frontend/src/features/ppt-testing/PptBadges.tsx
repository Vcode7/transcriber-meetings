import type { CSSProperties } from 'react'
import { STATUS_STYLE } from './pptStyles'

const TYPE_LABEL: Record<string, string> = {
  delay: 'Delay', risk: 'Risk', decision: 'Decision', dependency: 'Dependency',
  action: 'Action', milestone: 'Milestone', progress: 'Progress', status: 'Status',
}

const TEMPORAL_LABEL: Record<string, string> = {
  current: 'Active now', upcoming_soon: 'Due ≤ 30 days', upcoming: 'Upcoming', recent: 'Recent', undated: 'Undated',
}

const pill = (color: string): CSSProperties => ({
  fontSize: '.64rem', fontWeight: 700, padding: '1px 7px', borderRadius: 999, whiteSpace: 'nowrap',
  color, background: color.replace(')', ' / .12)'), border: `1px solid ${color.replace(')', ' / .3)')}`,
})

export function StatusBadge({ status }: { status: string }) {
  const s = STATUS_STYLE[status] ?? STATUS_STYLE.unknown
  return <span style={pill(s.color)}>{s.label}</span>
}

export function TypeBadge({ type }: { type: string }) {
  return (
    <span style={{ fontSize: '.64rem', fontWeight: 600, color: 'hsl(var(--pencil))', textTransform: 'uppercase', letterSpacing: '.04em' }}>
      {TYPE_LABEL[type] ?? type}
    </span>
  )
}

export function TemporalBadge({ temporal }: { temporal: string }) {
  const color = temporal === 'current' || temporal === 'upcoming_soon' ? 'hsl(14 85% 50%)' : 'hsl(220 10% 46%)'
  return <span style={pill(color)}>{TEMPORAL_LABEL[temporal] ?? temporal}</span>
}

export function SlideChips({ slides, kind = 'Slide' }: { slides: number[]; kind?: string }) {
  if (!slides?.length) return null
  return (
    <span
      title={`Source: ${kind.toLowerCase()} ${slides.join(', ')}`}
      style={{
        fontSize: '.64rem', fontWeight: 700, padding: '1px 7px', borderRadius: 5, whiteSpace: 'nowrap',
        fontFamily: 'JetBrains Mono, monospace', color: 'hsl(260 60% 52%)',
        background: 'hsl(260 60% 55% / .1)', border: '1px solid hsl(260 60% 55% / .3)',
      }}
    >
      {kind} {slides.join(', ')}
    </span>
  )
}
