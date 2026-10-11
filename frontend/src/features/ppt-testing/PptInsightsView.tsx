import { useMemo, useState, type ReactNode } from 'react'
import {
  AlertTriangle, CalendarClock, CalendarDays, CheckCircle2, ChevronDown, ChevronRight, Download, Flag,
  Layers, ListChecks, Loader, Scissors,
} from 'lucide-react'
import type { PptAnalysis, SlideInsight, TimelineEntry } from './types'
import { SlideChips, StatusBadge, TemporalBadge, TypeBadge } from './PptBadges'
import { statusColor } from './pptStyles'

const STATUS_FILTERS = ['all', 'delayed', 'at_risk', 'ongoing', 'planned', 'completed'] as const
type StatusFilter = typeof STATUS_FILTERS[number]

const FILTER_LABEL: Record<StatusFilter, string> = {
  all: 'All', delayed: 'Delayed', at_risk: 'At risk', ongoing: 'Ongoing', planned: 'Planned', completed: 'Completed',
}

function fmtDate(iso: string): string {
  const d = new Date(`${iso}T00:00:00`)
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' })
}

interface Props {
  analysis: PptAnalysis
  onExport?: () => void
  exporting?: boolean
}

export default function PptInsightsView({ analysis, onExport, exporting }: Props) {
  const [filter, setFilter] = useState<StatusFilter>('all')
  const [view, setView] = useState<'project' | 'slide'>('project')
  const [showCoverage, setShowCoverage] = useState(false)
  const unit = analysis.slides[0]?.kind === 'page' ? 'Page' : analysis.slides[0]?.kind === 'section' ? 'Section' : 'Slide'

  const byId = useMemo(() => Object.fromEntries(analysis.insights.map(i => [i.id, i])), [analysis.insights])
  const matches = (i: SlideInsight) => filter === 'all' || i.status === filter
  const counts = useMemo(() => {
    const c: Record<string, number> = { all: analysis.insights.length }
    for (const i of analysis.insights) c[i.status] = (c[i.status] ?? 0) + 1
    return c
  }, [analysis.insights])

  const tl = analysis.timeline
  const skipped = analysis.slides.filter(s => s.status === 'skipped_out_of_window')

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
      {/* Briefing header */}
      <div style={card('hsl(260 60% 55%)')}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap', marginBottom: '.5rem' }}>
          <CalendarDays size={15} style={{ color: 'hsl(260 60% 55%)' }} />
          <span style={{ fontWeight: 700, fontSize: '.86rem', color: 'hsl(var(--ink))' }}>
            Meeting brief · {fmtDate(analysis.meeting_date)}
          </span>
          <span style={{ fontSize: '.7rem', color: 'hsl(var(--pencil))' }}>
            Focus window {fmtDate(analysis.window.from)} – {fmtDate(analysis.window.to)}
          </span>
          {onExport && (
            <button onClick={onExport} disabled={exporting} style={{ ...smallBtn, marginLeft: 'auto' }}>
              {exporting ? <Loader size={12} className="spin" /> : <Download size={12} />} Export brief
            </button>
          )}
        </div>
        {analysis.overview && (
          <p style={{ margin: '0 0 .65rem', fontSize: '.86rem', lineHeight: 1.55, color: 'hsl(var(--ink))' }}>{analysis.overview}</p>
        )}
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '.4rem 1rem', fontSize: '.72rem', color: 'hsl(var(--pencil))' }}>
          <span><strong style={{ color: 'hsl(var(--ink))' }}>{analysis.slides_analyzed}</strong> of {analysis.slides_total} {unit.toLowerCase()}s analysed</span>
          <span><strong style={{ color: 'hsl(var(--ink))' }}>{analysis.insights.length}</strong> insights in {analysis.projects.length} project{analysis.projects.length === 1 ? '' : 's'}</span>
          {skipped.length > 0 && <span>{skipped.length} {unit.toLowerCase()}(s) outside the focus window skipped</span>}
          {analysis.dropped_out_of_window > 0 && <span>{analysis.dropped_out_of_window} far-dated point(s) dropped</span>}
        </div>
        {analysis.cutoff && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: '.55rem', fontSize: '.72rem', color: 'hsl(var(--pencil))' }}>
            <Scissors size={12} /> {analysis.cutoff.message}
          </div>
        )}
      </div>

      {/* Timeline around the meeting date */}
      {(tl.current_phase.length > 0 || tl.upcoming.length > 0 || tl.recent.length > 0) && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(230px, 1fr))', gap: '.6rem' }}>
          <TimelineColumn title="Current phase" icon={<Flag size={13} />} color="hsl(14 85% 50%)" entries={tl.current_phase} unit={unit} range />
          <TimelineColumn title="Upcoming" icon={<CalendarClock size={13} />} color="hsl(265 60% 55%)" entries={tl.upcoming} unit={unit} />
          <TimelineColumn
            title={tl.possibly_overdue.length ? 'Past due (not shown done)' : 'Recently finished'}
            icon={tl.possibly_overdue.length ? <AlertTriangle size={13} /> : <CheckCircle2 size={13} />}
            color={tl.possibly_overdue.length ? 'hsl(0 72% 50%)' : 'hsl(150 60% 36%)'}
            entries={tl.possibly_overdue.length ? tl.possibly_overdue : tl.recent}
            unit={unit}
          />
        </div>
      )}

      {/* Filters */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
        {STATUS_FILTERS.filter(f => f === 'all' || counts[f]).map(f => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            aria-pressed={filter === f}
            style={{
              ...smallBtn,
              color: filter === f ? (f === 'all' ? 'hsl(var(--ink))' : statusColor(f)) : 'hsl(var(--pencil))',
              border: filter === f ? '1.5px solid hsl(var(--ink) / .35)' : '1px solid hsl(var(--border) / .6)',
              fontWeight: filter === f ? 700 : 500,
            }}
          >
            {FILTER_LABEL[f]} <span style={{ opacity: 0.7 }}>{counts[f] ?? 0}</span>
          </button>
        ))}
        <div style={{ marginLeft: 'auto', display: 'flex', gap: 4 }}>
          {(['project', 'slide'] as const).map(v => (
            <button key={v} onClick={() => setView(v)} aria-pressed={view === v}
              style={{ ...smallBtn, fontWeight: view === v ? 700 : 500, background: view === v ? 'hsl(var(--muted))' : 'transparent' }}>
              {v === 'project' ? <Layers size={12} /> : <ListChecks size={12} />} By {v === 'project' ? 'project' : unit.toLowerCase()}
            </button>
          ))}
        </div>
      </div>

      {/* Insights */}
      {view === 'project' ? (
        analysis.projects.map(project => {
          const items = project.insight_ids.map(id => byId[id]).filter(Boolean).filter(matches)
          if (!items.length) return null
          return (
            <div key={project.name} style={card(statusColor(project.overall_status))}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap', marginBottom: '.4rem' }}>
                <span style={{ fontWeight: 700, fontSize: '.92rem', color: 'hsl(var(--ink))' }}>{project.name}</span>
                <StatusBadge status={project.overall_status} />
                <SlideChips slides={project.slides} kind={unit} />
              </div>
              <p style={{ margin: '0 0 .5rem', fontSize: '.82rem', lineHeight: 1.55, color: 'hsl(var(--ink))' }}>{project.summary}</p>
              {project.needs_attention.length > 0 && (
                <div style={{ display: 'flex', gap: 6, alignItems: 'flex-start', marginBottom: '.6rem', fontSize: '.76rem', color: 'hsl(25 85% 40%)' }}>
                  <AlertTriangle size={13} style={{ flexShrink: 0, marginTop: 2 }} />
                  <span><strong>Needs attention:</strong> {project.needs_attention.join(' · ')}</span>
                </div>
              )}
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                {items.map(i => <InsightRow key={i.id} insight={i} unit={unit} />)}
              </div>
            </div>
          )
        })
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          {[...analysis.insights].filter(matches).sort((a, b) => a.slides[0] - b.slides[0]).map(i => (
            <InsightRow key={i.id} insight={i} unit={unit} showProject />
          ))}
        </div>
      )}
      {analysis.insights.filter(matches).length === 0 && (
        <div style={{ textAlign: 'center', padding: '1.5rem', fontSize: '.8rem', color: 'hsl(var(--pencil))' }}>
          No insights with this status.
        </div>
      )}

      {/* Slide coverage */}
      <div style={{ border: '1px solid hsl(var(--border) / .5)', borderRadius: 9 }}>
        <button onClick={() => setShowCoverage(s => !s)} style={{ ...smallBtn, border: 'none', width: '100%', padding: '.55rem .8rem' }}>
          {showCoverage ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
          {unit} coverage — which {unit.toLowerCase()}s were analysed, skipped or excluded
        </button>
        {showCoverage && (
          <div style={{ padding: '0 .8rem .7rem', display: 'flex', flexDirection: 'column', gap: 3 }}>
            {[...analysis.slides, ...analysis.excluded_slides].sort((a, b) => a.number - b.number).map(s => (
              <div key={s.number} style={{ display: 'flex', gap: 8, fontSize: '.74rem', alignItems: 'baseline' }}>
                <span style={{ fontFamily: 'JetBrains Mono, monospace', minWidth: 28, color: 'hsl(var(--pencil))' }}>{s.number}</span>
                <span style={{ flex: 1, color: 'hsl(var(--ink))', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{s.title || '(untitled)'}</span>
                <span style={{ color: coverageColor(s.status ?? s.reason ?? ''), fontWeight: 600, whiteSpace: 'nowrap' }}>
                  {coverageLabel(s.status ?? s.reason ?? '', s.insight_count)}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

function InsightRow({ insight, unit, showProject }: { insight: SlideInsight; unit: string; showProject?: boolean }) {
  return (
    <div style={{
      padding: '.6rem .75rem', borderRadius: 8, background: 'hsl(var(--paper) / .45)',
      borderLeft: `3px solid ${statusColor(insight.status)}`,
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap', marginBottom: 3 }}>
        <span style={{ fontSize: '.62rem', fontWeight: 700, color: 'hsl(var(--pencil))', fontFamily: 'JetBrains Mono, monospace' }}>{insight.ref}</span>
        <SlideChips slides={insight.slides} kind={unit} />
        {showProject && insight.project && <span style={{ fontSize: '.7rem', fontWeight: 700, color: 'hsl(var(--ink))' }}>{insight.project}</span>}
        <StatusBadge status={insight.status} />
        <TypeBadge type={insight.type} />
        <span style={{ marginLeft: 'auto' }}><TemporalBadge temporal={insight.temporal} /></span>
      </div>
      <div style={{ fontSize: '.84rem', fontWeight: 700, color: 'hsl(var(--ink))', lineHeight: 1.4 }}>{insight.headline}</div>
      <div style={{ fontSize: '.8rem', color: 'hsl(var(--ink))', lineHeight: 1.5, marginTop: 2 }}>{insight.summary}</div>
      {insight.attention && (
        <div style={{ fontSize: '.76rem', color: 'hsl(25 85% 40%)', marginTop: 4 }}>
          <strong>Attention:</strong> {insight.attention}
        </div>
      )}
      {(insight.dates.length > 0 || insight.owner) && (
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginTop: 5 }}>
          {insight.dates.map(d => (
            <span key={d} style={{ fontSize: '.66rem', padding: '1px 6px', borderRadius: 4, background: 'hsl(var(--muted))', color: 'hsl(var(--ink))' }}>{d}</span>
          ))}
          {insight.owner && <span style={{ fontSize: '.66rem', color: 'hsl(var(--pencil))' }}>Owner: {insight.owner}</span>}
        </div>
      )}
    </div>
  )
}

function TimelineColumn({ title, icon, color, entries, unit, range }: {
  title: string; icon: ReactNode; color: string; entries: TimelineEntry[]; unit: string; range?: boolean
}) {
  return (
    <div style={{ ...card(color), padding: '.7rem .8rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '.74rem', fontWeight: 700, color, marginBottom: 6 }}>
        {icon}{title}
      </div>
      {entries.length === 0 ? (
        <div style={{ fontSize: '.74rem', color: 'hsl(var(--pencil))' }}>Not stated in the deck</div>
      ) : entries.slice(0, 4).map((e, idx) => (
        <div key={idx} style={{ fontSize: '.76rem', lineHeight: 1.4, marginBottom: 5, color: 'hsl(var(--ink))' }}>
          <div style={{ fontWeight: 600 }}>{e.label}</div>
          <div style={{ fontSize: '.68rem', color: 'hsl(var(--pencil))' }}>
            {range || e.start !== e.end ? `${fmtDate(e.start)} – ${fmtDate(e.end)}` : fmtDate(e.start)} · {unit} {e.slide}
          </div>
        </div>
      ))}
    </div>
  )
}

function coverageLabel(status: string, count?: number): string {
  switch (status) {
    case 'analyzed': return count ? `${count} insight${count === 1 ? '' : 's'}` : 'analysed — nothing notable'
    case 'skipped_out_of_window': return 'skipped — dates far from meeting'
    case 'skipped_empty': return 'skipped — no content'
    case 'skipped_structural': return 'skipped — title / agenda / divider'
    case 'error': return 'analysis failed'
    case 'closing_slide': return 'closing slide — stopped here'
    case 'after_conclusion': return 'excluded — after conclusion'
    case 'back_matter_after_conclusion': return 'excluded — reference / appendix'
    default: return status
  }
}

function coverageColor(status: string): string {
  if (status === 'analyzed') return 'hsl(150 60% 36%)'
  if (status === 'error') return 'hsl(0 72% 50%)'
  return 'hsl(var(--pencil))'
}

function card(color: string) {
  return {
    padding: '.85rem 1rem', borderRadius: 10, background: 'hsl(var(--card))',
    border: `1.5px solid ${color.startsWith('hsl(var') ? 'hsl(var(--border))' : color.replace(')', ' / .28)')}`,
  } as const
}

const smallBtn = {
  display: 'flex', alignItems: 'center', gap: 5, padding: '.3rem .65rem', borderRadius: 7, cursor: 'pointer',
  fontSize: '.72rem', fontFamily: 'Inter, sans-serif', background: 'transparent', color: 'hsl(var(--ink))',
  border: '1px solid hsl(var(--border) / .6)',
} as const
