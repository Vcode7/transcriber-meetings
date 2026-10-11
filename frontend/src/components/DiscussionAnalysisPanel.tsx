import { useState } from 'react'
import { ChevronDown, ChevronUp, Layers, Loader, Plus } from 'lucide-react'

interface TopicEntry {
  text: string
  point_ids: string[]
}

export interface DiscussionTopic {
  topic_id: string
  title: string
  description: string
  supporting_context?: string
  decisions?: TopicEntry[]
  action_items?: TopicEntry[]
  open_questions?: TopicEntry[]
  risks?: TopicEntry[]
  point_ids: string[]
  added_by_validation?: string[]
  time_start?: number
  time_end?: number
  type_counts?: Record<string, number>
  generated_by?: string
}

export interface TopicMatch {
  topic_id: string
  topic_title?: string
  status: 'matched' | 'unmatched' | 'derived' | 'added'
  matched_agenda_id?: string | null
  matched_agenda_title?: string
  best_agenda_id?: string | null
  best_agenda_title?: string
  score?: number
  margin?: number
  vote_agenda_id?: string | null
  vote_share?: number
  evidence?: Array<{ point_id: string; text: string; score: number }>
}

export interface DiscussionAnalysis {
  mode: 'derived_agendas' | 'matched_existing' | 'default_agenda' | null
  type_counts?: Record<string, number>
  near_duplicates?: Record<string, string>
  topics: DiscussionTopic[]
  topic_matches: TopicMatch[]
  validation?: {
    topics_in: number
    topics_out: number
    merged: Array<{ kept: string; merged: string; reason: string }>
    dropped: Array<{ title: string; reason: string }>
    uncovered_point_ids: string[]
    unsupported_removed: Array<{ topic: string; field: string; text: string; unsupported: string[] }>
    coverage_ratio: number
  }
}

export interface SuggestedAgenda {
  suggestion_id: string
  topic_id: string
  title: string
  description: string
  point_ids: string[]
  best_agenda_id?: string | null
  best_agenda_title?: string
  best_score?: number
  evidence?: Array<{ point_id: string; text: string; score: number }>
}

const TYPE_LABELS: Array<[string, string, string]> = [
  ['decision', 'Decisions', '140,65%,38%'],
  ['action_item', 'Action items', '30,90%,45%'],
  ['question', 'Open questions', '205,85%,45%'],
  ['risk', 'Risks', '0,75%,50%'],
  ['discussion', 'Discussion', '220,10%,50%'],
]

function fmtTime(sec?: number) {
  if (!sec) return ''
  const m = Math.floor(sec / 60)
  const s = Math.floor(sec % 60)
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
}

function Chip({ label, hue }: { label: string; hue: string }) {
  return (
    <span style={{ fontSize: '.68rem', fontWeight: 700, padding: '2px 7px', borderRadius: 6, color: `hsl(${hue})`, background: `hsl(${hue}/.1)`, border: `1px solid hsl(${hue}/.3)` }}>
      {label}
    </span>
  )
}

function Evidence({ items }: { items?: Array<{ point_id: string; text: string; score: number }> }) {
  if (!items || items.length === 0) return null
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 3, marginTop: 4 }}>
      {items.map(ev => (
        <div key={ev.point_id} style={{ fontSize: '.7rem', color: 'hsl(var(--pencil))', borderLeft: '2px solid hsl(var(--border))', paddingLeft: 6 }}>
          “{ev.text}” <span style={{ fontFamily: 'JetBrains Mono', fontSize: '.64rem' }}>({ev.score.toFixed(2)})</span>
        </div>
      ))}
    </div>
  )
}

interface Props {
  analysis: DiscussionAnalysis
  suggestions: SuggestedAgenda[]
  addingSuggestionId: string | null
  onAddSuggestion: (suggestion: SuggestedAgenda) => void
}

export function DiscussionAnalysisPanel({ analysis, suggestions, addingSuggestionId, onAddSuggestion }: Props) {
  const [showTopics, setShowTopics] = useState(false)
  const [openTopic, setOpenTopic] = useState<string | null>(null)
  const matchByTopic = new Map(analysis.topic_matches.map(m => [m.topic_id, m]))
  const validation = analysis.validation
  const totalPoints = Object.values(analysis.type_counts || {}).reduce((a, b) => a + b, 0)

  return (
    <div style={{ borderRadius: 10, border: '1.5px solid hsl(205,85%,50%/.35)', background: 'hsl(205,85%,50%/.04)', padding: '.8rem 1rem', display: 'flex', flexDirection: 'column', gap: 10 }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8, flexWrap: 'wrap' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '.84rem', fontWeight: 800, color: 'hsl(205,85%,38%)' }}>
          <Layers size={14} /> Discussion Analysis
        </div>
        <div style={{ fontSize: '.72rem', color: 'hsl(var(--pencil))' }}>
          {analysis.mode === 'derived_agendas' && 'No agenda document — agendas were built from the discussion.'}
          {analysis.mode === 'matched_existing' && 'Discussion topics matched against your agendas.'}
          {analysis.mode === 'default_agenda' && 'No distinct topics found; using General Discussion.'}
        </div>
      </div>

      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', alignItems: 'center' }}>
        <span style={{ fontSize: '.72rem', color: 'hsl(var(--ink))', fontWeight: 600 }}>{totalPoints} points · {analysis.topics.length} topics</span>
        {TYPE_LABELS.filter(([key]) => (analysis.type_counts || {})[key]).map(([key, label, hue]) => (
          <Chip key={key} label={`${label}: ${analysis.type_counts?.[key]}`} hue={hue} />
        ))}
      </div>

      {validation && (
        <div style={{ fontSize: '.71rem', color: 'hsl(var(--pencil))', display: 'flex', gap: 12, flexWrap: 'wrap' }}>
          <span title="Share of discussion points covered by a topic before validation filled the gaps">
            Coverage: <strong style={{ color: 'hsl(var(--ink))' }}>{Math.round(validation.coverage_ratio * 100)}%</strong>
            {validation.uncovered_point_ids.length > 0 && ` (${validation.uncovered_point_ids.length} point(s) attached during validation)`}
          </span>
          {Object.keys(analysis.near_duplicates || {}).length > 0 && <span>Near-duplicate points: {Object.keys(analysis.near_duplicates || {}).length}</span>}
          {validation.merged.length > 0 && (
            <span title={validation.merged.map(m => `"${m.merged}" merged into "${m.kept}" (${m.reason})`).join('\n')} style={{ cursor: 'help' }}>
              Duplicate topics merged: {validation.merged.length}
            </span>
          )}
          {validation.unsupported_removed.length > 0 && (
            <span title={validation.unsupported_removed.map(r => `${r.topic} · ${r.field}: ${r.text} [${r.unsupported.join(', ')}]`).join('\n')} style={{ cursor: 'help' }}>
              Unsupported details removed: {validation.unsupported_removed.length}
            </span>
          )}
        </div>
      )}

      {suggestions.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          <div style={{ fontSize: '.74rem', fontWeight: 700, color: 'hsl(var(--ink))' }}>
            Suggested agendas <span style={{ fontWeight: 400, color: 'hsl(var(--pencil))' }}>— discussed topics that match none of your agendas</span>
          </div>
          {suggestions.map(s => (
            <div key={s.suggestion_id} style={{ borderRadius: 8, border: '1px dashed hsl(205,85%,50%/.45)', background: 'hsl(var(--card))', padding: '.55rem .75rem', display: 'flex', gap: 10, alignItems: 'flex-start' }}>
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: '.82rem', fontWeight: 700, color: 'hsl(var(--ink))' }}>{s.title}</div>
                <div style={{ fontSize: '.73rem', color: 'hsl(var(--pencil))', marginTop: 2, lineHeight: 1.4 }}>{s.description}</div>
                <div style={{ fontSize: '.68rem', color: 'hsl(var(--pencil))', marginTop: 4 }}>
                  {s.point_ids.length} point(s)
                  {s.best_agenda_id && typeof s.best_score === 'number' && ` · closest existing agenda: ${s.best_agenda_id} ${s.best_agenda_title || ''} (${s.best_score.toFixed(2)})`}
                </div>
                <Evidence items={s.evidence} />
              </div>
              <button
                onClick={() => onAddSuggestion(s)}
                disabled={addingSuggestionId !== null}
                style={{ padding: '.3rem .7rem', borderRadius: 7, border: 'none', background: 'hsl(205,85%,45%)', color: 'white', fontSize: '.72rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: 4, cursor: addingSuggestionId !== null ? 'not-allowed' : 'pointer', flexShrink: 0, fontFamily: 'Inter' }}
              >
                {addingSuggestionId === s.suggestion_id ? <Loader size={11} className="spin" /> : <Plus size={11} />}
                Add as agenda
              </button>
            </div>
          ))}
        </div>
      )}

      {analysis.topics.length > 0 && (
        <div>
          <button
            type="button"
            onClick={() => setShowTopics(v => !v)}
            style={{ background: 'none', border: 'none', padding: 0, cursor: 'pointer', fontSize: '.73rem', fontWeight: 700, color: 'hsl(205,85%,40%)', display: 'flex', alignItems: 'center', gap: 4 }}
          >
            {showTopics ? <ChevronUp size={12} /> : <ChevronDown size={12} />} {showTopics ? 'Hide' : 'Show'} topics and matches
          </button>
          {showTopics && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 5, marginTop: 6 }}>
              {analysis.topics.map(topic => {
                const match = matchByTopic.get(topic.topic_id)
                const isOpen = openTopic === topic.topic_id
                const counts = [
                  ['decisions', topic.decisions?.length || 0],
                  ['open questions', topic.open_questions?.length || 0],
                  ['risks', topic.risks?.length || 0],
                ].filter(([, n]) => n) as Array<[string, number]>
                return (
                  <div key={topic.topic_id} style={{ borderRadius: 7, border: '1px solid hsl(var(--border)/.4)', background: 'hsl(var(--card))', padding: '.45rem .65rem' }}>
                    <div onClick={() => setOpenTopic(isOpen ? null : topic.topic_id)} style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
                      <span style={{ fontFamily: 'JetBrains Mono', fontSize: '.66rem', color: 'hsl(var(--pencil))' }}>{topic.topic_id}</span>
                      <span style={{ flex: 1, fontSize: '.78rem', fontWeight: 600, color: 'hsl(var(--ink))' }}>{topic.title}</span>
                      <span style={{ fontSize: '.66rem', color: 'hsl(var(--pencil))' }}>
                        {fmtTime(topic.time_start)}{topic.time_end ? `–${fmtTime(topic.time_end)}` : ''} · {topic.point_ids.length} pts
                      </span>
                      {match?.status === 'matched' && <Chip label={`→ ${match.matched_agenda_id} (${(match.score ?? 0).toFixed(2)})`} hue="140,65%,38%" />}
                      {match?.status === 'derived' && <Chip label={`Agenda ${match.matched_agenda_id}`} hue="140,65%,38%" />}
                      {match?.status === 'added' && <Chip label={`Added as ${match.matched_agenda_id}`} hue="140,65%,38%" />}
                      {match?.status === 'unmatched' && <Chip label="No matching agenda" hue="38,90%,42%" />}
                    </div>
                    {isOpen && (
                      <div style={{ marginTop: 6, fontSize: '.73rem', color: 'hsl(var(--ink))', lineHeight: 1.45 }}>
                        <div>{topic.description}</div>
                        {topic.supporting_context && <div style={{ color: 'hsl(var(--pencil))', marginTop: 3 }}>{topic.supporting_context}</div>}
                        {counts.length > 0 && (
                          <div style={{ color: 'hsl(var(--pencil))', marginTop: 3 }}>{counts.map(([label, n]) => `${n} ${label}`).join(' · ')}</div>
                        )}
                        {match && (match.status === 'matched' || match.status === 'unmatched') && (
                          <div style={{ marginTop: 4, color: 'hsl(var(--pencil))' }}>
                            Best agenda: {match.best_agenda_id} {match.best_agenda_title} · score {(match.score ?? 0).toFixed(2)}
                            {typeof match.vote_share === 'number' && match.vote_agenda_id && ` · ${Math.round(match.vote_share * 100)}% of its points prefer ${match.vote_agenda_id}`}
                          </div>
                        )}
                        <Evidence items={match?.evidence} />
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
