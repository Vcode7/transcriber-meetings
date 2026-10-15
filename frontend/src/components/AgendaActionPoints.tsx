import type { CSSProperties } from 'react'
import { Clock, Info, UserCheck } from 'lucide-react'

import { IMPORTANCE_THRESHOLDS, NOT_SPECIFIED, filterByImportance, hiddenByImportanceText, importanceBand } from '../lib/actionPoints'
import type { ActionImportance, ActionPointAudit, AgendaActionPoint, ImportanceBand } from '../lib/actionPoints'

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

// Stage 2 discussion types (backend services/point_classification.py).
const DISCUSSION_TYPES = ['decision', 'debate', 'question_answer', 'issue', 'update', 'discussion'] as const
export type DiscussionType = typeof DISCUSSION_TYPES[number]

const DISCUSSION_TYPE_STYLE: Record<DiscussionType, { label: string; hue: string }> = {
  decision: { label: 'Decision', hue: '140,65%,38%' },
  debate: { label: 'Debate', hue: '280,60%,52%' },
  question_answer: { label: 'Q&A', hue: '205,85%,45%' },
  issue: { label: 'Issue', hue: '0,75%,50%' },
  update: { label: 'Update', hue: '180,60%,36%' },
  discussion: { label: 'Discussion', hue: '220,10%,50%' },
}
const ACTION_HUE = '30,90%,45%'

function chipStyle(hue: string, active = true): CSSProperties {
  return {
    fontSize: '.64rem', padding: '1px 6px', borderRadius: 4, fontWeight: 700, fontFamily: 'inherit',
    background: active ? `hsl(${hue}/.1)` : 'transparent',
    color: active ? `hsl(${hue})` : 'hsl(var(--pencil))',
    border: `1px solid ${active ? `hsl(${hue}/.3)` : 'hsl(var(--border)/.5)'}`,
  }
}

interface DiscussionTypeBadgeProps {
  type?: string
  includeActionPoint?: boolean
  /** When given, the type and the action flag can be changed (Stage 2). */
  onChange?: (change: { type?: DiscussionType; include_action_point?: boolean }) => void
  disabled?: boolean
}

export function DiscussionTypeBadge({ type, includeActionPoint, onChange, disabled }: DiscussionTypeBadgeProps) {
  const known = type && type in DISCUSSION_TYPE_STYLE ? (type as DiscussionType) : undefined
  if (!onChange) {
    const style = known && known !== 'discussion' ? DISCUSSION_TYPE_STYLE[known] : undefined
    if (!style && !includeActionPoint) return null
    return (
      <span style={{ display: 'inline-flex', gap: 4 }}>
        {style && <span style={chipStyle(style.hue)}>{style.label}</span>}
        {includeActionPoint && (
          <span title="Contains an action: action points are generated from this point" style={chipStyle(ACTION_HUE)}>Action</span>
        )}
      </span>
    )
  }
  const current = known || 'discussion'
  return (
    <span style={{ display: 'inline-flex', gap: 4, alignItems: 'center' }}>
      <select
        value={current}
        disabled={disabled}
        title="Point type: decides how the point is shortened in the Short ROM"
        onChange={e => onChange({ type: e.target.value as DiscussionType })}
        style={{ ...chipStyle(DISCUSSION_TYPE_STYLE[current].hue), cursor: disabled ? 'default' : 'pointer' }}
      >
        {DISCUSSION_TYPES.map(t => <option key={t} value={t}>{DISCUSSION_TYPE_STYLE[t].label}</option>)}
      </select>
      <button
        type="button"
        disabled={disabled}
        onClick={() => onChange({ include_action_point: !includeActionPoint })}
        title={includeActionPoint
          ? 'Contains an action: Final ROM action points are generated from this point. Click to unflag.'
          : 'No action flagged: action points are not generated from this point. Click to flag it.'}
        style={{ ...chipStyle(ACTION_HUE, !!includeActionPoint), cursor: disabled ? 'default' : 'pointer' }}
      >
        {includeActionPoint ? 'Action' : 'No action'}
      </button>
    </span>
  )
}

const PRIORITY_HUE: Record<string, string> = { high: '0,75%,50%', medium: '30,90%,45%', low: '220,10%,50%' }

const IMPORTANCE_BAND_STYLE: Record<ImportanceBand, { label: string; hue: string }> = {
  critical: { label: 'Critical', hue: PRIORITY_HUE.high },
  moderate: { label: 'Moderate', hue: PRIORITY_HUE.medium },
  general: { label: 'General / low priority', hue: PRIORITY_HUE.low },
}

/** "85%" chip coloured by importance band; the tooltip gives the reason and how the score was set. */
export function ImportanceBadge({ item }: { item: ActionImportance }) {
  const score = Number(item.importance)
  if (item.importance === undefined || item.importance === null || !Number.isFinite(score)) return null
  const band = IMPORTANCE_BAND_STYLE[importanceBand(score)]
  const estimated = item.importance_source === 'heuristic'
  const lines = [`Importance ${score}% (${band.label})`]
  if (item.importance_reason) lines.push(item.importance_reason)
  if (estimated) lines.push('Estimated: AI scoring was unavailable')
  if (item.importance_adjusted) lines.push(`Adjusted: ${item.importance_adjusted}`)
  return (
    <span
      title={lines.join('\n')}
      style={{
        fontSize: '.63rem', fontWeight: 700, padding: '0 5px', borderRadius: 4, cursor: 'help', whiteSpace: 'nowrap',
        fontStyle: estimated ? 'italic' : 'normal',
        color: `hsl(${band.hue})`, background: `hsl(${band.hue}/.1)`, border: `1px solid hsl(${band.hue}/.3)`,
      }}
    >
      {score}%
    </span>
  )
}

interface ImportanceThresholdSelectProps {
  value: number
  onChange: (value: number) => void
  disabled?: boolean
}

/** Minimum importance shown (and exported); saved as a user setting. */
export function ImportanceThresholdSelect({ value, onChange, disabled }: ImportanceThresholdSelectProps) {
  const choices: readonly number[] = IMPORTANCE_THRESHOLDS
  const options = choices.includes(value) ? choices : [...choices, value].sort((a, b) => a - b)
  return (
    <label
      title="Hide action points scored below this importance. DOCX exports leave them out too. Action points without a score are always shown."
      style={{ display: 'inline-flex', alignItems: 'center', gap: 5, fontSize: '.72rem', fontWeight: 600, color: 'hsl(var(--pencil))', whiteSpace: 'nowrap' }}
    >
      Min importance
      <select
        value={value}
        disabled={disabled}
        onChange={e => onChange(Number(e.target.value))}
        style={{
          fontSize: '.72rem', fontFamily: 'inherit', padding: '1px 4px', borderRadius: 6, cursor: disabled ? 'default' : 'pointer',
          border: '1px solid hsl(var(--border)/.6)', background: 'hsl(var(--card))', color: 'hsl(var(--ink))',
        }}
      >
        {options.map(t => <option key={t} value={t}>{t === 0 ? 'Off' : `${t}%`}</option>)}
      </select>
    </label>
  )
}

function isUnspecified(value?: string | null) {
  if (!value) return true
  const v = String(value).trim().toLowerCase()
  return v === '' || v === 'none' || v === 'null' || v === 'n/a' || v === NOT_SPECIFIED.toLowerCase()
}

function auditSummary(audit: ActionPointAudit): string {
  const lines = [`${audit.generated} proposed, ${audit.kept} kept`]
  if (audit.mode === 'flagged_groups') {
    lines.push(`${audit.flagged_points ?? 0} point(s) flagged as containing an action; ${audit.groups_called ?? 0} of ${audit.groups ?? 0} group(s) analysed`)
  }
  if (audit.source === 'fallback_stage_items') lines.push('AI generation failed: built from the extracted action items')
  if (audit.source === 'llm_with_fallback') lines.push('AI generation failed for some groups: their extracted action items were used')
  for (const r of audit.rejected) lines.push(`Removed: "${r.task}" (${r.reason})`)
  for (const a of audit.adjusted) lines.push(`${a.field === 'owner' ? 'Owner' : 'Deadline'} "${a.value}" set to Not specified for "${a.task}" (${a.reason})`)
  const imp = audit.importance
  if (imp && imp.scored > 0) {
    if (imp.source === 'heuristic') lines.push('Importance estimated: AI scoring was unavailable')
    else if (imp.source === 'llm_with_fallback') lines.push(`Importance scored by AI for ${imp.llm_scored} of ${imp.scored}; the rest estimated`)
    if (imp.capped > 0) lines.push(`${imp.capped} importance score${imp.capped === 1 ? '' : 's'} capped at 75%: no deadline, date, decision or stated urgency`)
  }
  return lines.join('\n')
}

interface SectionProps {
  agendaId: string
  agendaMode: boolean
  actionPoints: AgendaActionPoint[]
  audit?: ActionPointAudit
  pointNumbers: Record<string, number>
  /** Action points scored below this importance are hidden (0 = show all). */
  minImportance?: number
}

export function AgendaActionPointsSection({ agendaId, agendaMode, actionPoints, audit, pointNumbers, minImportance = 0 }: SectionProps) {
  if (!agendaMode && actionPoints.length === 0) return null
  const removed = audit ? audit.rejected.length : 0
  const importanceNotes = !!audit?.importance && (audit.importance.source === 'heuristic' || audit.importance.source === 'llm_with_fallback' || audit.importance.capped > 0)
  const { visible, hidden } = filterByImportance(actionPoints, minImportance)
  return (
    <div style={{ marginTop: '.85rem', display: 'flex', flexDirection: 'column', gap: 6 }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8, paddingTop: '.6rem', borderTop: '1px solid hsl(var(--border)/.4)', marginBottom: 2 }}>
        <div style={{ fontSize: '.84rem', fontWeight: 700, color: 'hsl(var(--ink))' }}>Action Points</div>
        {audit && (removed > 0 || audit.adjusted.length > 0 || audit.source === 'fallback_stage_items' || audit.source === 'llm_with_fallback' || importanceNotes) && (
          <span title={auditSummary(audit)} style={{ display: 'inline-flex', alignItems: 'center', gap: 4, fontSize: '.67rem', color: 'hsl(var(--pencil))', cursor: 'help' }}>
            <Info size={11} />
            {removed > 0 ? `${removed} vague, unsupported or duplicate item${removed === 1 ? '' : 's'} removed` : 'Validation notes'}
          </span>
        )}
      </div>
      {actionPoints.length === 0 && (
        <div style={{ color: 'hsl(var(--pencil))', fontStyle: 'italic', fontSize: '.76rem' }}>
          {audit?.source === 'no_flagged_points'
            ? "No discussion point of this agenda was flagged as containing an action in Stage 2."
            : "No specific, supported action points were found in this agenda's discussion."}
        </div>
      )}
      {hidden > 0 && (
        <div style={{ color: 'hsl(var(--pencil))', fontStyle: 'italic', fontSize: '.72rem' }}>
          {hiddenByImportanceText(hidden, minImportance)}
        </div>
      )}
      {visible.map((ap, apIdx) => {
        const task = ap.task || ap.item || ap.description || ''
        const owner = ap.assignee || ap.owner || ''
        const deadline = ap.deadline || ''
        const ownerMissing = isUnspecified(owner)
        const deadlineMissing = isUnspecified(deadline) || (!agendaMode && String(deadline).toLowerCase() === 'asap')
        const sources = (ap.source_point_ids || []).map(pid => pointNumbers[pid]).filter((n): n is number => typeof n === 'number')
        // A scored item shows its importance; older unscored items keep the priority chip.
        const scored = ap.importance !== undefined && ap.importance !== null
        const hue = ap.priority && !scored ? PRIORITY_HUE[ap.priority] : undefined
        return (
          <div
            key={`agenda-ap-${agendaId}-${apIdx}`}
            title={ap.evidence_quote ? `Evidence: ${ap.evidence_quote}` : undefined}
            style={{ borderRadius: 8, border: '1px solid hsl(var(--border)/.3)', background: 'hsl(var(--paper)/.4)', padding: '.6rem .8rem', display: 'flex', gap: 8, alignItems: 'flex-start' }}
          >
            <span style={{ color: 'hsl(var(--pencil))', fontSize: '1rem', lineHeight: '1.2rem', flexShrink: 0, marginTop: 1, userSelect: 'none' }}>•</span>
            <div style={{ flex: 1 }}>
              <div style={{ color: 'hsl(var(--ink))', lineHeight: 1.45, fontSize: '.83rem' }}>{task}</div>
              {(agendaMode || !ownerMissing || !deadlineMissing || scored) && (
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
                  {scored && <ImportanceBadge item={ap} />}
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
