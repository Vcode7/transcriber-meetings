import { Clock, Info, UserCheck } from 'lucide-react'

import { NOT_SPECIFIED } from '../lib/actionPoints'
import type { ActionPointAudit, AgendaActionPoint } from '../lib/actionPoints'

type PointType = 'decision' | 'action_item' | 'question' | 'risk' | 'discussion'

const POINT_TYPE_STYLE: Record<PointType, { label: string; hue: string }> = {
  decision: { label: 'Decision', hue: '140,65%,38%' },
  action_item: { label: 'Action', hue: '30,90%,45%' },
  question: { label: 'Question', hue: '205,85%,45%' },
  risk: { label: 'Risk', hue: '0,75%,50%' },
  discussion: { label: 'Discussion', hue: '220,10%,50%' },
}

export function PointTypeBadge({ type, types }: { type?: string; types?: string[] }) {
  const style = type ? POINT_TYPE_STYLE[type as PointType] : undefined
  if (!style || type === 'discussion') return null
  const others = (types || []).filter(t => t !== type).map(t => POINT_TYPE_STYLE[t as PointType]?.label).filter(Boolean)
  return (
    <span
      title={others.length ? `Also: ${others.join(', ')}` : style.label}
      style={{
        fontSize: '.64rem', padding: '1px 6px', borderRadius: 4, fontWeight: 700,
        background: `hsl(${style.hue}/.1)`, color: `hsl(${style.hue})`, border: `1px solid hsl(${style.hue}/.3)`,
      }}
    >
      {style.label}
    </span>
  )
}

const PRIORITY_HUE: Record<string, string> = { high: '0,75%,50%', medium: '30,90%,45%', low: '220,10%,50%' }

function isUnspecified(value?: string | null) {
  if (!value) return true
  const v = String(value).trim().toLowerCase()
  return v === '' || v === 'none' || v === 'null' || v === 'n/a' || v === NOT_SPECIFIED.toLowerCase()
}

function auditSummary(audit: ActionPointAudit): string {
  const lines = [`${audit.generated} proposed, ${audit.kept} kept`]
  if (audit.source === 'fallback_stage_items') lines.push('AI generation failed: built from the extracted action items')
  for (const r of audit.rejected) lines.push(`Removed: "${r.task}" (${r.reason})`)
  for (const a of audit.adjusted) lines.push(`${a.field === 'owner' ? 'Owner' : 'Deadline'} "${a.value}" set to Not specified for "${a.task}" (${a.reason})`)
  return lines.join('\n')
}

interface SectionProps {
  agendaId: string
  agendaMode: boolean
  actionPoints: AgendaActionPoint[]
  audit?: ActionPointAudit
  pointNumbers: Record<string, number>
}

export function AgendaActionPointsSection({ agendaId, agendaMode, actionPoints, audit, pointNumbers }: SectionProps) {
  if (!agendaMode && actionPoints.length === 0) return null
  const removed = audit ? audit.rejected.length : 0
  return (
    <div style={{ marginTop: '.85rem', display: 'flex', flexDirection: 'column', gap: 6 }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8, paddingTop: '.6rem', borderTop: '1px solid hsl(var(--border)/.4)', marginBottom: 2 }}>
        <div style={{ fontSize: '.84rem', fontWeight: 700, color: 'hsl(var(--ink))' }}>Action Points</div>
        {audit && (removed > 0 || audit.adjusted.length > 0 || audit.source === 'fallback_stage_items') && (
          <span title={auditSummary(audit)} style={{ display: 'inline-flex', alignItems: 'center', gap: 4, fontSize: '.67rem', color: 'hsl(var(--pencil))', cursor: 'help' }}>
            <Info size={11} />
            {removed > 0 ? `${removed} vague, unsupported or duplicate item${removed === 1 ? '' : 's'} removed` : 'Validation notes'}
          </span>
        )}
      </div>
      {actionPoints.length === 0 && (
        <div style={{ color: 'hsl(var(--pencil))', fontStyle: 'italic', fontSize: '.76rem' }}>
          No specific, supported action points were found in this agenda's discussion.
        </div>
      )}
      {actionPoints.map((ap, apIdx) => {
        const task = ap.task || ap.item || ap.description || ''
        const owner = ap.assignee || ap.owner || ''
        const deadline = ap.deadline || ''
        const ownerMissing = isUnspecified(owner)
        const deadlineMissing = isUnspecified(deadline) || (!agendaMode && String(deadline).toLowerCase() === 'asap')
        const sources = (ap.source_point_ids || []).map(pid => pointNumbers[pid]).filter((n): n is number => typeof n === 'number')
        const hue = ap.priority ? PRIORITY_HUE[ap.priority] : undefined
        return (
          <div
            key={`agenda-ap-${agendaId}-${apIdx}`}
            title={ap.evidence_quote ? `Evidence: ${ap.evidence_quote}` : undefined}
            style={{ borderRadius: 8, border: '1px solid hsl(var(--border)/.3)', background: 'hsl(var(--paper)/.4)', padding: '.6rem .8rem', display: 'flex', gap: 8, alignItems: 'flex-start' }}
          >
            <span style={{ color: 'hsl(var(--pencil))', fontSize: '1rem', lineHeight: '1.2rem', flexShrink: 0, marginTop: 1, userSelect: 'none' }}>•</span>
            <div style={{ flex: 1 }}>
              <div style={{ color: 'hsl(var(--ink))', lineHeight: 1.45, fontSize: '.83rem' }}>{task}</div>
              {(agendaMode || !ownerMissing || !deadlineMissing) && (
                <div style={{ marginTop: 4, display: 'flex', gap: 10, fontSize: '.7rem', color: 'hsl(var(--pencil))', flexWrap: 'wrap', alignItems: 'center' }}>
                  {(agendaMode || !ownerMissing) && (
                    <span style={{ display: 'flex', alignItems: 'center', gap: 4, fontStyle: ownerMissing ? 'italic' : 'normal' }}>
                      <UserCheck size={10} /> {ownerMissing ? NOT_SPECIFIED : owner}
                      {!ownerMissing && ap.owner_confidence === 'low' && (
                        <span
                          title={`Speaker attribution uncertain — verify the owner.${ap.evidence?.quote ? `\nTranscript: ${ap.evidence.quote}` : ''}`}
                          style={{ color: 'hsl(38,90%,45%)', fontWeight: 700, cursor: 'help' }}
                        >
                          (?)
                        </span>
                      )}
                    </span>
                  )}
                  {(agendaMode || !deadlineMissing) && (
                    <span style={{ display: 'flex', alignItems: 'center', gap: 4, fontStyle: deadlineMissing ? 'italic' : 'normal' }}>
                      <Clock size={10} /> {deadlineMissing ? NOT_SPECIFIED : deadline}
                    </span>
                  )}
                  {hue && agendaMode && (
                    <span style={{ fontSize: '.63rem', fontWeight: 700, padding: '0 5px', borderRadius: 4, textTransform: 'capitalize', color: `hsl(${hue})`, background: `hsl(${hue}/.1)`, border: `1px solid hsl(${hue}/.3)` }}>
                      {ap.priority}
                    </span>
                  )}
                  {sources.length > 0 && (
                    <span style={{ fontSize: '.66rem' }}>from {sources.map(n => `#${n}`).join(', ')}</span>
                  )}
                </div>
              )}
            </div>
          </div>
        )
      })}
    </div>
  )
}
