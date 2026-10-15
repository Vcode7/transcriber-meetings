export const NOT_SPECIFIED = 'Not specified'

/** Importance score (40-100%) of an action point, set by backend services/action_importance.py. */
export interface ActionImportance {
  importance?: number
  importance_source?: 'llm' | 'heuristic'
  importance_reason?: string
  importance_adjusted?: string
}

export interface AgendaActionPoint extends ActionImportance {
  task: string
  assignee?: string | null
  owner?: string | null
  deadline?: string | null
  source_point_ids?: string[]
  priority?: 'high' | 'medium' | 'low'
  basis?: string
  evidence_quote?: string
  owner_confidence?: 'high' | 'medium' | 'low'
  evidence?: { quote?: string }
  item?: string
  description?: string
}

export interface ImportanceSummary {
  source: 'llm' | 'heuristic' | 'llm_with_fallback' | 'none'
  scored: number
  llm_scored: number
  capped: number
}

/** Minimum-importance choices; 0 shows every action point. */
export const IMPORTANCE_THRESHOLDS = [0, 50, 60, 70, 80, 90] as const

export type ImportanceBand = 'critical' | 'moderate' | 'general'

/** Same bands as the scoring prompt: 80-100 critical, 60-79 moderate, below 60 general. */
export function importanceBand(score: number): ImportanceBand {
  if (score >= 80) return 'critical'
  if (score >= 60) return 'moderate'
  return 'general'
}

/** Same rule as the backend: unscored items (built before scoring existed) and a threshold of 0 always pass. */
export function passesImportance(ap: ActionImportance | null | undefined, minImportance: number): boolean {
  if (!ap || !(minImportance > 0) || ap.importance === undefined || ap.importance === null) return true
  const score = Number(ap.importance)
  return !Number.isFinite(score) || score >= minImportance
}

/** The action points shown under the threshold, and how many it hides. */
export function filterByImportance<T extends ActionImportance>(items: T[], minImportance: number): { visible: T[]; hidden: number } {
  const visible = items.filter(item => passesImportance(item, minImportance))
  return { visible, hidden: items.length - visible.length }
}

/** "3 action points below 70% importance hidden" (empty when nothing is hidden). */
export function hiddenByImportanceText(hidden: number, minImportance: number): string {
  if (hidden <= 0) return ''
  return `${hidden} action point${hidden === 1 ? '' : 's'} below ${minImportance}% importance hidden`
}

export interface ActionPointAudit {
  source: string
  // flagged_groups: only points Stage 2 flagged, in groups of 5; all_points: older Final ROMs without flags
  mode?: 'flagged_groups' | 'all_points' | 'none'
  groups?: number
  groups_called?: number
  flagged_points?: number | null
  generated: number
  kept: number
  rejected: Array<{ task: string; reason: string }>
  adjusted: Array<{ task: string; field: string; value: string; reason: string }>
  generated_at?: string
  importance?: ImportanceSummary
}

interface AgendaLike {
  skipped?: boolean
  include_action_points?: boolean
  action_points_mode?: string
  action_points?: AgendaActionPoint[]
  discussion_points?: Array<{ id?: string; action_points?: AgendaActionPoint[]; action_items?: Array<AgendaActionPoint | string> }>
}

/** Same rule as the backend: per-agenda option, falling back to the old global flag. */
export function agendaShowsActionPoints(agenda: AgendaLike, globalFallback: boolean): boolean {
  if (agenda.skipped) return false
  if (typeof agenda.include_action_points === 'boolean') return agenda.include_action_points
  return globalFallback
}

/** Agenda-level action points; Final ROMs saved before per-agenda generation keep point-level items. */
export function collectAgendaActionPoints(agenda: AgendaLike): AgendaActionPoint[] {
  if (agenda.action_points_mode === 'agenda') return agenda.action_points || []
  const collected: AgendaActionPoint[] = []
  const seen = new Set<string>()
  const add = (ap: AgendaActionPoint | string) => {
    if (typeof ap !== 'object' || ap === null) return
    const key = (ap.task || ap.item || ap.description || '').trim().toLowerCase()
    if (key && !seen.has(key)) {
      seen.add(key)
      collected.push(ap)
    }
  }
  ;(agenda.action_points || []).forEach(add)
  for (const pt of agenda.discussion_points || []) {
    const list = pt.action_points && pt.action_points.length > 0 ? pt.action_points : (pt.action_items || [])
    list.forEach(add)
  }
  return collected
}
