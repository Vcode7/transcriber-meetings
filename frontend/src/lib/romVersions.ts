// Short/Medium Final ROM versions: word reduction against the long points, and each condensed
// point's original (long) points. Mirrors backend services/rom_condense.py and the
// _condense_source_index helper in routers/rom_router.py.

/** One original (long) point as the condense call saw it (agenda.condense_sources). */
export interface CondenseSource {
  label?: string
  id: string
  text: string
  speakers?: string[]
  action_owner?: string | null
  type?: string
  include_action_point?: boolean
  is_very_important?: boolean
  is_doc_point?: boolean
}

/** On-demand explanation of how a condensed point was written from its original points. */
export interface CondenseExplanation {
  retained?: string
  removed?: string
  merged?: string
  why?: string
  generated_at?: string
}

interface PointLike {
  id?: string
  text?: string
  polished_text?: string
  source_point_ids?: string[]
  speakers?: string[] | string
  speaker?: string
  type?: string
  include_action_point?: boolean
}

interface AgendaLike {
  agenda_id?: string
  skipped?: boolean
  discussion_points?: PointLike[]
  condense_sources?: CondenseSource[]
}

interface RomLike {
  agendas?: AgendaLike[]
}

export interface Reduction {
  originalWords: number
  originalPoints: number
  generatedWords: number
  generatedPoints: number
  /** Share of the original words removed (negative when the version is longer); null without originals. */
  reductionPct: number | null
}

export function countWords(text?: string | null): number {
  const trimmed = (text || '').trim()
  return trimmed ? trimmed.split(/\s+/).length : 0
}

const pointText = (point: PointLike) => point.text || point.polished_text || ''

const sumWords = (texts: string[]) => texts.reduce((total, text) => total + countWords(text), 0)

function reduction(originalWords: number, originalPoints: number, generatedWords: number, generatedPoints: number): Reduction {
  return {
    originalWords,
    originalPoints,
    generatedWords,
    generatedPoints,
    reductionPct: originalWords ? ((originalWords - generatedWords) / originalWords) * 100 : null,
  }
}

/** Original words: the points the agenda was condensed from (its snapshot, else the long agenda);
 *  generated words: the agenda's current points, so edits and expansions count. */
export function agendaReduction(agenda: AgendaLike, longAgenda?: AgendaLike): Reduction {
  const originals = agenda.condense_sources?.length
    ? agenda.condense_sources.map(s => s.text || '')
    : (longAgenda?.discussion_points || []).map(pointText)
  const generated = (agenda.discussion_points || []).map(pointText)
  return reduction(sumWords(originals), originals.length, sumWords(generated), generated.length)
}

/** Totals over the version's non-skipped agendas, plus each agenda's reduction by agenda_id. */
export function versionReduction(versionRom?: RomLike | null, longRom?: RomLike | null): Reduction & { agendas: Record<string, Reduction> } {
  const longById = new Map((longRom?.agendas || []).map(a => [a.agenda_id, a]))
  const agendas: Record<string, Reduction> = {}
  let originalWords = 0, originalPoints = 0, generatedWords = 0, generatedPoints = 0
  for (const agenda of versionRom?.agendas || []) {
    if (agenda.skipped) continue
    const r = agendaReduction(agenda, longById.get(agenda.agenda_id))
    if (agenda.agenda_id) agendas[agenda.agenda_id] = r
    originalWords += r.originalWords
    originalPoints += r.originalPoints
    generatedWords += r.generatedWords
    generatedPoints += r.generatedPoints
  }
  return { ...reduction(originalWords, originalPoints, generatedWords, generatedPoints), agendas }
}

export function formatReduction(pct: number | null, short = false): string {
  if (pct === null) return '—'
  const value = Math.abs(pct).toFixed(short ? 0 : 1)
  if (pct >= 0) return short ? `−${value}%` : `${value}% fewer words`
  return short ? `+${value}%` : `${value}% more words`
}

/** Original points by ID across the whole version (points can move between agendas): the
 *  condense snapshots first, then the long version's points labelled by agenda position. */
export function condenseSourceIndex(versionRom?: RomLike | null, longRom?: RomLike | null): Map<string, CondenseSource> {
  const index = new Map<string, CondenseSource>()
  for (const agenda of versionRom?.agendas || []) {
    for (const src of agenda.condense_sources || []) {
      if (src?.id && !index.has(src.id)) index.set(src.id, src)
    }
  }
  for (const agenda of longRom?.agendas || []) {
    (agenda.discussion_points || []).forEach((point, i) => {
      if (!point.id || index.has(point.id)) return
      const speakers = Array.isArray(point.speakers) ? point.speakers : point.speaker ? [point.speaker] : []
      index.set(point.id, {
        id: point.id,
        label: `P${i + 1}`,
        text: pointText(point),
        speakers,
        type: point.type,
        include_action_point: point.include_action_point,
      })
    })
  }
  return index
}

/** A condensed point's original points, in cited order; IDs that no longer resolve are kept with empty text. */
export function pointSources(point: PointLike, index: Map<string, CondenseSource>): Array<CondenseSource & { missing?: boolean }> {
  return Array.from(new Set(point.source_point_ids || [])).map(id => index.get(id) || { id, text: '', missing: true })
}
