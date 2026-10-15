import { ArrowRight, BarChart3 } from 'lucide-react'

import { formatReduction, type Reduction } from '../../lib/romVersions'

const HUE = '200,90%,45%'

function reductionColor(pct: number | null): string {
  if (pct === null) return 'hsl(var(--pencil))'
  return pct >= 0 ? 'hsl(142,70%,32%)' : 'hsl(30,90%,40%)'
}

function Stat({ label, words, points }: { label: string; words: number; points: number }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
      <span style={{ fontSize: '.64rem', fontWeight: 700, color: 'hsl(var(--pencil))', textTransform: 'uppercase', letterSpacing: '.04em' }}>
        {label}
      </span>
      <span style={{ fontSize: '.86rem', fontWeight: 800, color: 'hsl(var(--ink))', fontVariantNumeric: 'tabular-nums' }}>
        {words.toLocaleString()} words
        <span style={{ fontSize: '.7rem', fontWeight: 600, color: 'hsl(var(--pencil))', marginLeft: 5 }}>
          · {points} {points === 1 ? 'pt' : 'pts'}
        </span>
      </span>
    </div>
  )
}

/** Word counts of the long points against the Short/Medium points, and the reduction. */
export function VersionReductionSummary({ version, reduction }: { version: 'short' | 'medium'; reduction: Reduction }) {
  const label = version === 'short' ? 'Short' : 'Medium'
  const color = reductionColor(reduction.reductionPct)
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 14, flexWrap: 'wrap',
      background: 'hsl(var(--card))', padding: '.6rem 1rem', borderRadius: 10,
      border: `1.5px solid hsl(${HUE}/.25)`, boxShadow: '0 1px 4px rgba(0,0,0,0.04)',
    }}>
      <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6, fontSize: '.75rem', fontWeight: 800, color: `hsl(${HUE})`, textTransform: 'uppercase', letterSpacing: '.05em' }}>
        <BarChart3 size={14} /> Reduction Analysis
      </span>
      <Stat label="Original (Long)" words={reduction.originalWords} points={reduction.originalPoints} />
      <ArrowRight size={14} style={{ color: 'hsl(var(--pencil))' }} />
      <Stat label={`Generated (${label})`} words={reduction.generatedWords} points={reduction.generatedPoints} />
      <span style={{
        fontSize: '.8rem', fontWeight: 800, color, padding: '3px 10px', borderRadius: 6,
        background: reduction.reductionPct !== null && reduction.reductionPct < 0 ? 'hsl(30,90%,55%/.12)' : 'hsl(142,70%,45%/.14)',
        fontVariantNumeric: 'tabular-nums',
      }}>
        {formatReduction(reduction.reductionPct)}
      </span>
      <span style={{ fontSize: '.7rem', color: 'hsl(var(--pencil))', flex: '1 1 220px' }}>
        Original words come from the long points each agenda was condensed from; edits and expansions update the generated count.
      </span>
    </div>
  )
}

/** Compact per-agenda chip: "412 → 160 words · −61%". */
export function AgendaReductionChip({ reduction }: { reduction: Reduction }) {
  if (!reduction.originalWords) return null
  return (
    <span
      title={`Long: ${reduction.originalWords} words in ${reduction.originalPoints} points → ${reduction.generatedWords} words in ${reduction.generatedPoints} points (${formatReduction(reduction.reductionPct)})`}
      style={{
        fontSize: '.68rem', fontWeight: 700, padding: '2px 7px', borderRadius: 8, flexShrink: 0,
        color: 'hsl(var(--pencil))', background: 'hsl(var(--muted)/.45)', fontVariantNumeric: 'tabular-nums',
      }}
    >
      {reduction.originalWords} → {reduction.generatedWords} words ·{' '}
      <span style={{ color: reductionColor(reduction.reductionPct) }}>{formatReduction(reduction.reductionPct, true)}</span>
    </span>
  )
}
