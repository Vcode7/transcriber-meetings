export const NOT_SPECIFIED = 'Not specified'

export interface AgendaActionPoint {
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

export interface ActionPointAudit {
  source: string
  generated: number
  kept: number
  rejected: Array<{ task: string; reason: string }>
  adjusted: Array<{ task: string; field: string; value: string; reason: string }>
  generated_at?: string
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
