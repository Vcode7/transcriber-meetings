import { useEffect, useState, useCallback, useRef } from 'react'
import { isAxiosError } from 'axios'
import { useParams, useNavigate } from 'react-router-dom'
import {
  ArrowLeft, Sparkles, Loader, FileDown, Copy, CheckCircle,
  AlertTriangle, Clock, RotateCcw, Plus, X, History
} from 'lucide-react'
import api from '../api/client'
import MomSection from '../components/MomSection'
import ActionItemsTable, { ActionItem } from '../components/ActionItemsTable'
import TagInput from '../components/TagInput'
import { getApiErrorDetail } from '../lib/errors'

// ── Types ─────────────────────────────────────────────────────
interface MomData {
  title: string
  date: string
  duration: number
  participants: string[]
  agenda_items: string[]
  discussion_summary: string
  decisions: string[]
  action_items: ActionItem[]
  risks_concerns: string[]
  next_steps: string[]
  next_meeting_date: string | null
}

interface VersionEntry {
  version: number
  saved_at: string
  data: MomData
}

interface RecordingSummary {
  filename: string
  duration: number
}

type PageState = 'loading' | 'idle' | 'generating' | 'editing'
type SaveState = 'saved' | 'saving' | 'unsaved'

// ── Normalize API response to safe MomData ─────────────────────
function normalizeMom(raw: Partial<MomData> | null | undefined): MomData | null {
  if (!raw) return null
  return {
    title: typeof raw.title === 'string' ? raw.title : '',
    date: typeof raw.date === 'string' ? raw.date : '',
    duration: typeof raw.duration === 'number' ? raw.duration : 0,
    participants: Array.isArray(raw.participants) ? raw.participants.map(String) : [],
    agenda_items: Array.isArray(raw.agenda_items) ? raw.agenda_items.map(String) : [],
    discussion_summary: typeof raw.discussion_summary === 'string' ? raw.discussion_summary : '',
    decisions: Array.isArray(raw.decisions) ? raw.decisions.map(String) : [],
    action_items: Array.isArray(raw.action_items)
      ? raw.action_items.map((a: unknown) => {
          if (a && typeof a === 'object' && !Array.isArray(a)) {
            const obj = a as Record<string, unknown>
            return {
              task: String(obj.task ?? ''),
              owner: String(obj.owner ?? 'Unassigned'),
              deadline: String(obj.deadline ?? 'ASAP'),
            }
          }
          return { task: String(a ?? ''), owner: 'Unassigned', deadline: 'ASAP' }
        })
      : [],
    risks_concerns: Array.isArray(raw.risks_concerns) ? raw.risks_concerns.map(String) : [],
    next_steps: Array.isArray(raw.next_steps) ? raw.next_steps.map(String) : [],
    next_meeting_date: raw.next_meeting_date ?? null,
  }
}

function fmtDuration(s: number) {
  if (!s) return 'N/A'
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const sec = Math.floor(s % 60).toString().padStart(2, '0')
  if (h > 0) return `${h}h ${m}m ${sec}s`
  return `${m}m ${sec}s`
}

// ── Editable List Component ────────────────────────────────────
function EditableList({
  items,
  onChange,
  placeholder = 'Add item...',
  ordered = false,
}: {
  items: string[]
  onChange: (items: string[]) => void
  placeholder?: string
  ordered?: boolean
}) {
  const handleChange = (index: number, value: string) => {
    const newItems = [...items]
    newItems[index] = value
    onChange(newItems)
  }
  const handleAdd = () => onChange([...items, ''])
  const handleRemove = (index: number) => onChange(items.filter((_, i) => i !== index))
  const handleKeyDown = (e: React.KeyboardEvent, index: number) => {
    if (e.key === 'Enter') {
      e.preventDefault()
      onChange([...items.slice(0, index + 1), '', ...items.slice(index + 1)])
    } else if (e.key === 'Backspace' && !items[index] && items.length > 1) {
      e.preventDefault()
      handleRemove(index)
    }
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
      {items.map((item, idx) => (
        <div key={idx} style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{
            fontSize: '0.82rem', color: 'hsl(var(--pencil))', width: '22px',
            textAlign: 'right', flexShrink: 0, fontFamily: 'Inter, sans-serif'
          }}>
            {ordered ? `${idx + 1}.` : '\u2022'}
          </span>
          <input
            className="input"
            value={item}
            onChange={e => handleChange(idx, e.target.value)}
            onKeyDown={e => handleKeyDown(e, idx)}
            placeholder={placeholder}
            style={{ flex: 1, padding: '0.45rem 0.75rem', fontSize: '0.9rem' }}
          />
          <button onClick={() => handleRemove(idx)} className="icon-btn"
            style={{ color: 'hsl(var(--pencil))', width: '28px', height: '28px' }}
            title="Remove">
            <X size={13} />
          </button>
        </div>
      ))}
      <button onClick={handleAdd} className="btn btn-ghost"
        style={{ fontSize: '0.8rem', padding: '0.3rem 0.7rem', gap: '5px', alignSelf: 'flex-start', marginTop: '4px' }}>
        <Plus size={13} /> Add
      </button>
    </div>
  )
}

// ── Main Page ─────────────────────────────────────────────────
export default function MomPage() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()

  const [pageState, setPageState] = useState<PageState>('loading')
  const [saveState, setSaveState] = useState<SaveState>('saved')
  const [mom, setMom] = useState<MomData | null>(null)
  const [recording, setRecording] = useState<RecordingSummary | null>(null)
  const [versions, setVersions] = useState<VersionEntry[]>([])
  const [historyOpen, setHistoryOpen] = useState(false)
  const [copyStatus, setCopyStatus] = useState<'idle' | 'copied'>('idle')
  const [pdfStatus, setPdfStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle')
  const [error, setError] = useState<string | null>(null)
  const [genStep, setGenStep] = useState(0)

  const saveTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const momRef = useRef<MomData | null>(null)
  momRef.current = mom

  const GENERATING_STEPS = [
    'Reading transcript...',
    'Extracting agenda items...',
    'Identifying decisions...',
    'Building action items...',
    'Generating summary...',
    'Finalizing MoM...',
  ]

  // ── Generating step animation ──
  useEffect(() => {
    if (pageState !== 'generating') return
    const t = setInterval(() => setGenStep(s => (s + 1) % GENERATING_STEPS.length), 1600)
    return () => clearInterval(t)
  }, [pageState, GENERATING_STEPS.length])

  // ── Load recording info + check for existing MoM ──
  useEffect(() => {
    if (!id) return
    const init = async () => {
      try {
        const recRes = await api.get(`/history/${id}`)
        setRecording(recRes.data)
        try {
          const momRes = await api.get(`/mom/${id}`)
          setMom(normalizeMom(momRes.data))
          setPageState('editing')
        } catch (e: unknown) {
          if (isAxiosError(e) && e.response?.status === 404) {
            setPageState('idle')
          } else {
            throw e
          }
        }
      } catch {
        setError('Failed to load recording.')
        setPageState('idle')
      }
    }
    init()
  }, [id])

  // ── Auto-save with 3s debounce ──
  const scheduleSave = useCallback((data: MomData) => {
    setSaveState('unsaved')
    if (saveTimer.current) clearTimeout(saveTimer.current)
    saveTimer.current = setTimeout(async () => {
      setSaveState('saving')
      try {
        await api.patch(`/mom/${id}`, data)
        setSaveState('saved')
      } catch {
        setSaveState('unsaved')
      }
    }, 3000)
  }, [id])

  const update = useCallback(<K extends keyof MomData>(field: K, value: MomData[K]) => {
    setMom(prev => {
      if (!prev) return prev
      const next = { ...prev, [field]: value }
      scheduleSave(next)
      return next
    })
  }, [scheduleSave])

  // ── Generate MoM ──
  const handleGenerate = async () => {
    setPageState('generating')
    setError(null)
    try {
      const res = await api.post(`/mom/${id}/generate`)
      setMom(normalizeMom(res.data))
      setPageState('editing')
    } catch (e: unknown) {
      setError(getApiErrorDetail(e, 'Generation failed. Please try again.'))
      setPageState('idle')
    }
  }

  // ── Version history ──
  const loadVersions = async () => {
    try {
      const res = await api.get(`/mom/${id}/versions`)
      setVersions(res.data.versions || [])
    } catch { /* ignore */ }
  }

  const toggleHistory = () => {
    if (!historyOpen) loadVersions()
    setHistoryOpen(v => !v)
  }

  const restoreVersion = async (v: VersionEntry) => {
    if (!confirm(`Restore version ${v.version} from ${new Date(v.saved_at).toLocaleString()}?`)) return
    const normalized = normalizeMom(v.data)
    if (normalized) {
      setMom(normalized)
      setHistoryOpen(false)
      scheduleSave(normalized)
    }
  }

  // ── Export PDF ──
  const handlePdf = async () => {
    setPdfStatus('loading')
    try {
      const res = await api.post(`/mom/${id}/pdf`, {}, { responseType: 'blob' })
      const blob = new Blob([res.data], { type: 'application/pdf' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `MoM_${mom?.title?.replace(/\s+/g, '_') || 'Meeting'}.pdf`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
      setPdfStatus('success')
      setTimeout(() => setPdfStatus('idle'), 2500)
    } catch {
      setPdfStatus('error')
      setTimeout(() => setPdfStatus('idle'), 3000)
    }
  }

  // ── Copy to Clipboard ──
  const handleCopy = async () => {
    if (!mom) return
    const text = [
      'MINUTES OF MEETING',
      '==================',
      `Title: ${mom.title}`,
      `Date: ${mom.date}`,
      `Duration: ${fmtDuration(mom.duration)}`,
      `Participants: ${mom.participants.join(', ')}`,
      '',
      'AGENDA',
      '------',
      ...mom.agenda_items.map((a, i) => `${i + 1}. ${a}`),
      '',
      'DISCUSSION SUMMARY',
      '------------------',
      mom.discussion_summary,
      '',
      'DECISIONS TAKEN',
      '---------------',
      ...mom.decisions.map(d => `\u2022 ${d}`),
      '',
      'ACTION ITEMS',
      '------------',
      ...mom.action_items.map(a => `\u2022 [${a.owner}] ${a.task} \u2014 Due: ${a.deadline}`),
      '',
      'RISKS / CONCERNS',
      '----------------',
      ...mom.risks_concerns.map(r => `\u2022 ${r}`),
      '',
      'NEXT STEPS',
      '----------',
      ...mom.next_steps.map((s, i) => `${i + 1}. ${s}`),
      ...(mom.next_meeting_date ? ['', 'NEXT MEETING', '------------', mom.next_meeting_date] : []),
    ].join('\n')

    await navigator.clipboard.writeText(text)
    setCopyStatus('copied')
    setTimeout(() => setCopyStatus('idle'), 2000)
  }

  // ── Render ────────────────────────────────────────────────────

  if (pageState === 'loading') {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', flexDirection: 'column', gap: '1rem' }}>
        <Loader size={24} className="spin" style={{ color: 'hsl(var(--accent))' }} />
        <p style={{ fontSize: '.9rem', color: 'hsl(var(--pencil))', fontFamily: 'Inter, sans-serif' }}>Loading...</p>
      </div>
    )
  }

  return (
    <div className="page-scroll-root mom-page" style={{ display: 'flex', flexDirection: 'column', background: 'hsl(var(--paper) / .4)' }}>

      {/* Header */}
      <div className="panel-header" style={{ flexShrink: 0, flexWrap: 'wrap', gap: '8px' }}>
        <button className="icon-btn" onClick={() => navigate(`/dashboard/history/${id}`)} title="Back to transcript">
          <ArrowLeft size={15} />
        </button>

        <div style={{ flex: 1, minWidth: 0 }}>
          <h1 style={{ fontSize: '1rem', fontWeight: 700, color: 'hsl(var(--ink))', fontFamily: 'Inter, sans-serif', marginBottom: '2px' }}>
            Minutes of Meeting
          </h1>
          {recording && (
            <p style={{ fontSize: '0.78rem', color: 'hsl(var(--pencil))', fontFamily: 'Inter, sans-serif' }}>
              {recording.filename} \u00b7 {fmtDuration(recording.duration)}
            </p>
          )}
        </div>

        {pageState === 'editing' && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.78rem', color: 'hsl(var(--pencil))', fontFamily: 'Inter, sans-serif', flexShrink: 0 }}>
            {saveState === 'saving' && <><Loader size={11} className="spin" /> Saving...</>}
            {saveState === 'saved' && <><CheckCircle size={11} style={{ color: 'hsl(var(--success))' }} /><span style={{ color: 'hsl(var(--success))' }}>Saved</span></>}
            {saveState === 'unsaved' && <><Clock size={11} /> Unsaved</>}
          </div>
        )}

        {pageState === 'editing' && (
          <div style={{ display: 'flex', gap: '8px', flexShrink: 0 }}>
            <button className="btn btn-ghost" onClick={toggleHistory} style={{ fontSize: '0.82rem', padding: '0.4rem 0.85rem', height: '36px', gap: '6px' }}>
              <History size={14} />
              {historyOpen ? 'Hide History' : 'History'}
            </button>
            <button className="btn btn-ghost" onClick={handleCopy} style={{ fontSize: '0.82rem', padding: '0.4rem 0.85rem', height: '36px', gap: '6px' }}>
              {copyStatus === 'copied'
                ? <><CheckCircle size={14} style={{ color: 'hsl(var(--success))' }} /><span style={{ color: 'hsl(var(--success))' }}>Copied!</span></>
                : <><Copy size={14} />Copy</>}
            </button>
            <button className="btn btn-ghost" onClick={handleGenerate} style={{ fontSize: '0.82rem', padding: '0.4rem 0.85rem', height: '36px', gap: '6px' }}>
              <RotateCcw size={14} /> Regenerate
            </button>
            <button className="btn btn-primary" onClick={handlePdf} disabled={pdfStatus === 'loading'} style={{ fontSize: '0.82rem', padding: '0.4rem 0.9rem', height: '36px', gap: '6px' }}>
              {pdfStatus === 'loading' && <><Loader size={13} className="spin" />Generating...</>}
              {pdfStatus === 'success' && <><CheckCircle size={13} />Downloaded!</>}
              {pdfStatus === 'error' && <><AlertTriangle size={13} />Failed</>}
              {pdfStatus === 'idle' && <><FileDown size={13} />Export PDF</>}
            </button>
          </div>
        )}
      </div>

      {/* Body */}
      <div style={{ flex: 1, overflowY: 'auto', minHeight: 0, display: 'flex' }}>

        {/* Idle - Generate CTA */}
        {pageState === 'idle' && (
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '3rem 2rem', gap: '1.5rem', textAlign: 'center' }}>
            <div style={{ width: '100px', height: '100px', borderRadius: '50%', background: 'hsl(var(--accent) / .08)', border: '2.5px dashed hsl(var(--accent) / .3)', display: 'flex', alignItems: 'center', justifyContent: 'center', boxShadow: '0 0 0 12px hsl(var(--accent) / .04)' }}>
              <Sparkles size={42} style={{ color: 'hsl(var(--accent))', opacity: 0.75 }} className="animate-float" />
            </div>
            <div>
              <h2 style={{ fontSize: '1.3rem', fontWeight: 700, color: 'hsl(var(--ink))', fontFamily: 'Inter, sans-serif', marginBottom: '.5rem' }}>Generate Minutes of Meeting</h2>
              <p style={{ fontSize: '0.9rem', color: 'hsl(var(--pencil))', maxWidth: '400px', lineHeight: 1.6, fontFamily: 'Inter, sans-serif' }}>
                Let AI extract the full structured MoM from your transcript - agenda, decisions, action items, owners, risks and next steps.
              </p>
            </div>
            {error && (
              <div style={{ padding: '0.75rem 1.25rem', borderRadius: '10px', background: 'hsl(var(--destructive) / .1)', border: '1px solid hsl(var(--destructive) / .3)', color: 'hsl(var(--destructive))', fontSize: '0.88rem', fontFamily: 'Inter, sans-serif' }}>
                <AlertTriangle size={14} style={{ display: 'inline', marginRight: '6px' }} />
                {error}
              </div>
            )}
            <button className="btn btn-primary" onClick={handleGenerate} style={{ fontSize: '0.95rem', padding: '0.75rem 2.25rem', gap: '8px', borderRadius: '12px' }}>
              <Sparkles size={16} /> Generate Minutes of Meeting
            </button>
          </div>
        )}

        {/* Generating skeleton */}
        {pageState === 'generating' && (
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '1.5rem', padding: '3rem' }}>
            <div style={{ width: '80px', height: '80px', borderRadius: '50%', background: 'hsl(var(--accent) / .08)', border: '3px solid hsl(var(--accent) / .2)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <Loader size={32} className="spin" style={{ color: 'hsl(var(--accent))' }} />
            </div>
            <div style={{ textAlign: 'center' }}>
              <p style={{ fontSize: '1.05rem', fontWeight: 600, color: 'hsl(var(--ink))', fontFamily: 'Inter, sans-serif', marginBottom: '8px' }}>Analyzing transcript...</p>
              <p className="animate-slide-up" key={genStep} style={{ fontSize: '0.9rem', color: 'hsl(var(--accent))', fontFamily: 'Inter, sans-serif' }}>
                {GENERATING_STEPS[genStep]}
              </p>
            </div>
            <div style={{ width: '100%', maxWidth: '680px', display: 'flex', flexDirection: 'column', gap: '12px', marginTop: '1rem' }}>
              {[120, 80, 200, 100, 150, 80].map((h, i) => (
                <div key={i} style={{ height: h, borderRadius: '12px', background: 'linear-gradient(90deg, hsl(var(--muted)) 0%, hsl(var(--card)) 50%, hsl(var(--muted)) 100%)', backgroundSize: '200% 100%', animation: 'shimmer 1.5s ease-in-out infinite', animationDelay: `${i * 0.15}s`, border: '1px solid hsl(var(--border) / .3)' }} />
              ))}
            </div>
          </div>
        )}

        {/* Editing view */}
        {pageState === 'editing' && mom && (
          <div className="mom-editor-layout" style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>

            <div className="mom-editor-main" style={{ flex: 1, overflowY: 'auto', padding: '1.5rem', minWidth: 0 }}>
              <MomSection title="Meeting Information" className="mom-section">
                <div className="mom-info-grid" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                  <div>
                    <label style={{ fontSize: '0.78rem', fontWeight: 600, color: 'hsl(var(--pencil))', fontFamily: 'Inter, sans-serif', display: 'block', marginBottom: '6px', textTransform: 'uppercase', letterSpacing: '0.04em' }}>Meeting Title</label>
                    <input className="input" value={mom.title} onChange={e => update('title', e.target.value)} placeholder="Meeting title" style={{ width: '100%', padding: '0.5rem 0.75rem', fontSize: '0.95rem', fontWeight: 600 }} />
                  </div>
                  <div>
                    <label style={{ fontSize: '0.78rem', fontWeight: 600, color: 'hsl(var(--pencil))', fontFamily: 'Inter, sans-serif', display: 'block', marginBottom: '6px', textTransform: 'uppercase', letterSpacing: '0.04em' }}>Date</label>
                    <input className="input" value={mom.date} onChange={e => update('date', e.target.value)} placeholder="Meeting date" style={{ width: '100%', padding: '0.5rem 0.75rem', fontSize: '0.9rem' }} />
                  </div>
                  <div>
                    <label style={{ fontSize: '0.78rem', fontWeight: 600, color: 'hsl(var(--pencil))', fontFamily: 'Inter, sans-serif', display: 'block', marginBottom: '6px', textTransform: 'uppercase', letterSpacing: '0.04em' }}>Duration</label>
                    <p style={{ fontSize: '0.9rem', color: 'hsl(var(--ink))', fontFamily: 'Inter, sans-serif', padding: '0.5rem 0' }}>{fmtDuration(mom.duration)}</p>
                  </div>
                  <div>
                    <label style={{ fontSize: '0.78rem', fontWeight: 600, color: 'hsl(var(--pencil))', fontFamily: 'Inter, sans-serif', display: 'block', marginBottom: '6px', textTransform: 'uppercase', letterSpacing: '0.04em' }}>Next Meeting</label>
                    <input className="input" value={mom.next_meeting_date || ''} onChange={e => update('next_meeting_date', e.target.value)} placeholder="Next meeting date (if known)" style={{ width: '100%', padding: '0.5rem 0.75rem', fontSize: '0.9rem' }} />
                  </div>
                </div>
                <div style={{ marginTop: '1rem' }}>
                  <label style={{ fontSize: '0.78rem', fontWeight: 600, color: 'hsl(var(--pencil))', fontFamily: 'Inter, sans-serif', display: 'block', marginBottom: '6px', textTransform: 'uppercase', letterSpacing: '0.04em' }}>Participants</label>
                  <TagInput tags={mom.participants} onChange={tags => update('participants', tags)} placeholder="Type name and press Enter..." />
                </div>
              </MomSection>

              <MomSection title="Agenda Items">
                <EditableList items={mom.agenda_items} onChange={items => update('agenda_items', items)} placeholder="Agenda item..." ordered />
              </MomSection>

              <MomSection title="Discussion Summary">
                <textarea className="input" value={mom.discussion_summary} onChange={e => update('discussion_summary', e.target.value)} placeholder="Write a summary of the discussion..." rows={7} style={{ width: '100%', resize: 'vertical', padding: '0.75rem', fontSize: '0.9rem', lineHeight: 1.7, fontFamily: 'Inter, sans-serif' }} />
              </MomSection>

              <MomSection title="Decisions Taken">
                <EditableList items={mom.decisions} onChange={items => update('decisions', items)} placeholder="Decision made..." />
              </MomSection>

              <MomSection title="Action Items">
                <ActionItemsTable items={mom.action_items} onChange={items => update('action_items', items)} />
              </MomSection>

              <MomSection title="Risks & Concerns">
                <EditableList items={mom.risks_concerns} onChange={items => update('risks_concerns', items)} placeholder="Risk or concern..." />
              </MomSection>

              <MomSection title="Next Steps">
                <EditableList items={mom.next_steps} onChange={items => update('next_steps', items)} placeholder="Next step..." ordered />
              </MomSection>
            </div>

            {/* Version History sidebar */}
            {historyOpen && (
              <div className="mom-version-panel" style={{ width: '280px', flexShrink: 0, overflowY: 'auto', borderLeft: '1px solid hsl(var(--border) / .25)', background: 'hsl(var(--card))', padding: '1.25rem 1rem' }}>
                <h3 style={{ fontSize: '0.9rem', fontWeight: 700, color: 'hsl(var(--ink))', fontFamily: 'Inter, sans-serif', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <History size={14} style={{ color: 'hsl(var(--accent))' }} /> Version History
                </h3>
                {versions.length === 0 && (
                  <p style={{ fontSize: '0.82rem', color: 'hsl(var(--pencil))', fontFamily: 'Inter, sans-serif', lineHeight: 1.6 }}>No saved versions yet. Versions are saved automatically.</p>
                )}
                {versions.map((v, idx) => (
                  <div key={idx} style={{ borderRadius: '10px', padding: '0.75rem 1rem', border: idx === 0 ? '1.5px solid hsl(var(--sticky-yellow) / .5)' : '1px solid hsl(var(--border) / .3)', background: idx === 0 ? 'hsl(var(--sticky-yellow) / .08)' : 'hsl(var(--paper) / .5)', marginBottom: '8px' }}>
                    <div style={{ fontWeight: 700, fontSize: '0.85rem', color: 'hsl(var(--ink))', fontFamily: 'Inter, sans-serif', marginBottom: '4px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                      {idx === 0 && <span style={{ fontSize: '.72rem', background: 'hsl(var(--sticky-yellow))', color: 'hsl(var(--ink))', padding: '.08rem .4rem', borderRadius: '4px', fontWeight: 700 }}>* Original</span>}
                      {idx === 0 ? 'AI Generated' : `Version ${v.version}`}
                    </div>
                    <div style={{ fontSize: '0.74rem', color: 'hsl(var(--pencil))', fontFamily: 'JetBrains Mono, monospace', marginBottom: '10px' }}>
                      {new Date(v.saved_at).toLocaleString()}
                    </div>
                    <button onClick={() => restoreVersion(v)} className="btn btn-ghost" style={{ fontSize: '0.78rem', padding: '0.3rem 0.7rem', gap: '5px', width: '100%' }}>
                      <RotateCcw size={12} /> Restore this version
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

      </div>
    </div>
  )
}
