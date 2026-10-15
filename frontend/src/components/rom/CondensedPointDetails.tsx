import { useState, type CSSProperties } from 'react'
import { ChevronDown, ChevronUp, Info, Loader, Maximize2, RefreshCw, Undo2, User } from 'lucide-react'

import { DiscussionTypeBadge } from '../AgendaActionPoints'
import type { CondenseExplanation, CondenseSource } from '../../lib/romVersions'

const HUE = '200,90%,45%'

export interface CondensedPoint {
  id: string
  text?: string
  polished_text?: string
  source_point_ids?: string[]
  condense_explanation?: CondenseExplanation
  condense_notes?: string[]
  expanded_sentences?: string[]
  pre_expand_text?: string
}

interface CondensedPointDetailsProps {
  point: CondensedPoint
  sources: Array<CondenseSource & { missing?: boolean }>
  isVeryImportant: boolean
  explaining: boolean
  expanding: boolean
  reverting: boolean
  onExplain: (regenerate: boolean) => void
  onExpand: () => void
  onRevert: () => void
}

const EXPLANATION_ROWS: Array<[keyof CondenseExplanation, string]> = [
  ['retained', 'Retained'],
  ['removed', 'Removed'],
  ['merged', 'Merged'],
  ['why', 'Why'],
]

const norm = (text?: string) => (text || '').replace(/\s+/g, ' ').trim().toLowerCase()

function buttonStyle(active: boolean, disabled = false): CSSProperties {
  return {
    display: 'inline-flex', alignItems: 'center', gap: 4, fontSize: '.68rem', fontWeight: 700, fontFamily: 'Inter',
    padding: '2px 8px', borderRadius: 6, cursor: disabled ? 'not-allowed' : 'pointer', opacity: disabled ? 0.55 : 1,
    border: `1px solid ${active ? `hsl(${HUE}/.45)` : 'hsl(var(--border)/.6)'}`,
    background: active ? `hsl(${HUE}/.1)` : 'transparent',
    color: active ? `hsl(${HUE})` : 'hsl(var(--pencil))',
  }
}

const sectionLabel: CSSProperties = {
  fontSize: '.64rem', fontWeight: 800, color: 'hsl(var(--pencil))', textTransform: 'uppercase', letterSpacing: '.05em',
}

/** Explanation dropdown (what the point kept, removed and merged from its original long points, and
 *  the originals themselves by point ID), Expand Point and Revert for one Short/Medium point. */
export function CondensedPointDetails({
  point, sources, isVeryImportant, explaining, expanding, reverting, onExplain, onExpand, onRevert,
}: CondensedPointDetailsProps) {
  const [open, setOpen] = useState(false)
  const explanation = point.condense_explanation
  const notes = point.condense_notes || []
  const expanded = point.expanded_sentences || []
  const text = point.text || point.polished_text || ''
  const isVerbatim = sources.length === 1 && !sources[0].missing && norm(sources[0].text) === norm(text)
  const expandDisabledReason = isVeryImportant
    ? 'Very Important points are kept verbatim'
    : isVerbatim ? 'This point is already the full original text' : ''
  const busy = expanding || reverting

  const toggle = () => {
    const next = !open
    setOpen(next)
    if (next && !explanation && !explaining) onExplain(false)
  }

  return (
    <div style={{ marginTop: 6 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
        <button type="button" onClick={toggle} aria-expanded={open} style={buttonStyle(open)}
          title="What this point retained, removed and merged from its original long points, and why">
          <Info size={11} /> Explanation {open ? <ChevronUp size={11} /> : <ChevronDown size={11} />}
        </button>
        <button
          type="button"
          onClick={onExpand}
          disabled={busy || !!expandDisabledReason}
          style={buttonStyle(false, busy || !!expandDisabledReason)}
          title={expandDisabledReason || 'Add 2–3 sentences of detail from this point’s original long points only'}
        >
          {expanding ? <Loader size={11} className="spin" /> : <Maximize2 size={11} />} Expand Point
        </button>
        {point.pre_expand_text && (
          <>
            <span style={{ fontSize: '.64rem', fontWeight: 800, padding: '1px 6px', borderRadius: 4, background: `hsl(${HUE}/.12)`, color: `hsl(${HUE})`, border: `1px solid hsl(${HUE}/.3)` }}>
              Expanded (+{expanded.length})
            </span>
            <button type="button" onClick={onRevert} disabled={busy} style={buttonStyle(false, busy)} title="Restore the text from before the expansion">
              {reverting ? <Loader size={11} className="spin" /> : <Undo2 size={11} />} Revert
            </button>
          </>
        )}
        <span style={{ fontSize: '.66rem', color: 'hsl(var(--pencil))' }}>
          From {sources.map(s => s.label || s.id.slice(0, 8)).join(', ')}
        </span>
      </div>

      {open && (
        <div style={{
          marginTop: 6, padding: '.6rem .75rem', borderRadius: 8, display: 'flex', flexDirection: 'column', gap: 10,
          border: `1px solid hsl(${HUE}/.25)`, background: `hsl(${HUE}/.04)`,
        }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 4 }}>
              <span style={sectionLabel}>Explanation</span>
              {explanation && (
                <button type="button" onClick={() => onExplain(true)} disabled={explaining}
                  style={{ ...buttonStyle(false, explaining), border: 'none', padding: '1px 4px' }}
                  title="Generate the explanation again (e.g. after editing the point)">
                  <RefreshCw size={10} className={explaining ? 'spin' : ''} /> Regenerate
                </button>
              )}
            </div>
            {explaining && !explanation ? (
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '.74rem', color: 'hsl(var(--pencil))' }}>
                <Loader size={12} className="spin" /> Generating explanation…
              </div>
            ) : explanation ? (
              <div style={{ display: 'grid', gridTemplateColumns: 'max-content 1fr', columnGap: 10, rowGap: 4 }}>
                {EXPLANATION_ROWS.filter(([key]) => explanation[key]).map(([key, label]) => (
                  <div key={key} style={{ display: 'contents' }}>
                    <span style={{ fontSize: '.7rem', fontWeight: 700, color: `hsl(${HUE})` }}>{label}</span>
                    <span style={{ fontSize: '.76rem', color: 'hsl(var(--ink))', lineHeight: 1.45 }}>{explanation[key]}</span>
                  </div>
                ))}
              </div>
            ) : (
              <div style={{ fontSize: '.74rem', color: 'hsl(var(--pencil))' }}>
                No explanation yet.{' '}
                <button type="button" onClick={() => onExplain(false)} style={{ ...buttonStyle(true), display: 'inline-flex' }}>Generate</button>
              </div>
            )}
          </div>

          {notes.length > 0 && (
            <div>
              <div style={{ ...sectionLabel, marginBottom: 3 }}>Verification notes</div>
              <ul style={{ margin: 0, paddingLeft: 16, fontSize: '.72rem', color: 'hsl(var(--ink))', lineHeight: 1.45 }}>
                {notes.map(note => <li key={note}>{note}</li>)}
              </ul>
            </div>
          )}

          {expanded.length > 0 && (
            <div>
              <div style={{ ...sectionLabel, marginBottom: 3 }}>Added by Expand Point</div>
              <ul style={{ margin: 0, paddingLeft: 16, fontSize: '.72rem', color: 'hsl(var(--ink))', lineHeight: 1.45 }}>
                {expanded.map((sentence, i) => <li key={`${i}-${sentence}`}>{sentence}</li>)}
              </ul>
            </div>
          )}

          <div>
            <div style={{ ...sectionLabel, marginBottom: 4 }}>
              Original long points used ({sources.length}){sources.length > 1 ? ' — merged into this point' : ''}
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
              {sources.map(src => (
                <div key={src.id} style={{ padding: '.45rem .6rem', borderRadius: 6, border: '1px solid hsl(var(--border)/.4)', background: 'hsl(var(--card))' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap', marginBottom: src.missing ? 0 : 3 }}>
                    {src.label && (
                      <span style={{ fontFamily: 'JetBrains Mono', fontSize: '.66rem', fontWeight: 800, padding: '1px 6px', borderRadius: 4, background: `hsl(${HUE})`, color: 'white' }}>
                        {src.label}
                      </span>
                    )}
                    <span title={src.id} style={{ fontFamily: 'JetBrains Mono', fontSize: '.62rem', color: 'hsl(var(--pencil))' }}>
                      ID {src.id.length > 12 ? `${src.id.slice(0, 8)}…` : src.id}
                    </span>
                    {src.type && <DiscussionTypeBadge type={src.type} includeActionPoint={src.include_action_point} />}
                    {!!src.speakers?.length && (
                      <span style={{ display: 'inline-flex', alignItems: 'center', gap: 3, fontSize: '.66rem', color: 'hsl(var(--pencil))' }}>
                        <User size={9} /> {src.speakers.join(', ')}
                      </span>
                    )}
                  </div>
                  <div style={{ fontSize: '.76rem', lineHeight: 1.45, color: src.missing ? 'hsl(var(--pencil))' : 'hsl(var(--ink))', fontStyle: src.missing ? 'italic' : 'normal' }}>
                    {src.missing ? 'This original point is no longer in the long version.' : src.text}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
