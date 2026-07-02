import { useEffect, useState, useCallback, useRef } from 'react'
import { isAxiosError } from 'axios'
import { useParams, useNavigate } from 'react-router-dom'
import {
  ArrowLeft, Sparkles, Loader, FileDown, Copy, CheckCircle,
  AlertTriangle, Clock, RotateCcw, Plus, X, History, User, Users
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
  planned_start_time: string
  actual_start_time: string
  participants: string[]
  introduction: string
  points_discussed: string[]
  action_items: ActionItem[]
  conclusion: string
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
    planned_start_time: typeof raw.planned_start_time === 'string' ? raw.planned_start_time : '',
    actual_start_time: typeof raw.actual_start_time === 'string' ? raw.actual_start_time : '',
    participants: Array.isArray(raw.participants) ? raw.participants.map(String) : [],
    introduction: typeof raw.introduction === 'string' ? raw.introduction : '',
    points_discussed: Array.isArray(raw.points_discussed) ? raw.points_discussed.map(String) : [],
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
    conclusion: typeof raw.conclusion === 'string' ? raw.conclusion : '',
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
  minItems = 0,
  maxItems,
}: {
  items: string[]
  onChange: (items: string[]) => void
  placeholder?: string
  ordered?: boolean
  minItems?: number
  maxItems?: number
}) {
  const handleChange = (index: number, value: string) => {
    const newItems = [...items]
    newItems[index] = value
    onChange(newItems)
  }
  const handleAdd = () => {
    if (maxItems && items.length >= maxItems) return
    onChange([...items, ''])
  }
  const handleRemove = (index: number) => {
    if (minItems && items.length <= minItems) return
    onChange(items.filter((_, i) => i !== index))
  }
  const handleKeyDown = (e: React.KeyboardEvent, index: number) => {
    if (e.key === 'Enter') {
      e.preventDefault()
      if (!maxItems || items.length < maxItems)
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
          <button
            onClick={() => handleRemove(idx)}
            className="icon-btn"
            disabled={!!(minItems && items.length <= minItems)}
            style={{ color: 'hsl(var(--pencil))', width: '28px', height: '28px', opacity: (minItems && items.length <= minItems) ? 0.35 : 1 }}
            title="Remove"
          >
            <X size={13} />
          </button>
        </div>
      ))}
      {(!maxItems || items.length < maxItems) && (
        <button onClick={handleAdd} className="btn btn-ghost"
          style={{ fontSize: '0.8rem', padding: '0.3rem 0.7rem', gap: '5px', alignSelf: 'flex-start', marginTop: '4px' }}>
          <Plus size={13} /> Add
        </button>
      )}
      {maxItems && (
        <p style={{ fontSize: '0.74rem', color: 'hsl(var(--pencil))', fontFamily: 'Inter, sans-serif', marginTop: '2px' }}>
          {items.length}/{maxItems} points
        </p>
      )}
    </div>
  )
}

// ── Action Items grouped by speaker ───────────────────────────
function ActionPointsSection({
  items,
  onChange,
}: {
  items: ActionItem[]
  onChange: (items: ActionItem[]) => void
}) {
  // Group
  const speakerItems = items.filter(a => a.owner && a.owner !== 'Unassigned')
  const generalItems = items.filter(a => !a.owner || a.owner === 'Unassigned')

  const speakers = [...new Set(speakerItems.map(a => a.owner))].sort()

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
      {/* Speaker-based */}
      {speakers.length > 0 && (
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '8px' }}>
            <User size={13} style={{ color: 'hsl(var(--accent))' }} />
            <span style={{ fontSize: '0.78rem', fontWeight: 700, color: 'hsl(var(--pencil))', textTransform: 'uppercase', letterSpacing: '0.04em', fontFamily: 'Inter, sans-serif' }}>
              By Speaker
            </span>
          </div>
          <ActionItemsTable items={items} onChange={onChange} />
        </div>
      )}

      {/* If no speakers yet, just show the table */}
      {speakers.length === 0 && (
        <ActionItemsTable items={items} onChange={onChange} />
      )}

      {/* General hint */}
      {generalItems.length > 0 && (
        <div style={{ padding: '0.6rem 0.9rem', borderRadius: '8px', background: 'hsl(var(--muted) / .4)', border: '1px dashed hsl(var(--border) / .4)', display: 'flex', alignItems: 'center', gap: '6px' }}>
          <Users size={13} style={{ color: 'hsl(var(--pencil))' }} />
          <span style={{ fontSize: '0.8rem', color: 'hsl(var(--pencil))', fontFamily: 'Inter, sans-serif' }}>
            {generalItems.length} general action item{generalItems.length !== 1 ? 's' : ''} (no specific owner)
          </span>
        </div>
      )}
    </div>
  )
}

// ── Label helper ──────────────────────────────────────────────
function FieldLabel({ children }: { children: React.ReactNode }) {
  return (
    <label style={{
      fontSize: '0.78rem', fontWeight: 600, color: 'hsl(var(--pencil))',
      fontFamily: 'Inter, sans-serif', display: 'block', marginBottom: '6px',
      textTransform: 'uppercase', letterSpacing: '0.04em'
    }}>{children}</label>
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
    'Extracting meeting topics...',
    'Identifying action points...',
    'Drafting introduction...',
    'Writing conclusion...',
    'Finalizing MoM...',
  ]

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

    // Action items grouped
    const speakerAI = mom.action_items.filter(a => a.owner && a.owner !== 'Unassigned')
    const generalAI = mom.action_items.filter(a => !a.owner || a.owner === 'Unassigned')
    const speakers = [...new Set(speakerAI.map(a => a.owner))].sort()

    const aiLines: string[] = []
    for (const sp of speakers) {
      aiLines.push(`  ${sp}:`)
      for (const a of speakerAI.filter(x => x.owner === sp)) {
        aiLines.push(`    \u2022 ${a.task} \u2014 Due: ${a.deadline}`)
      }
    }
    if (generalAI.length > 0) {
      aiLines.push('  General:')
      for (const a of generalAI) {
        aiLines.push(`    \u2022 ${a.task} \u2014 Due: ${a.deadline}`)
      }
    }

    const text = [
      'MINUTES OF MEETING',
      '==================',
      `Meeting Title    : ${mom.title}`,
      `Date             : ${mom.date}`,
      `Members          : ${mom.participants.join(', ')}`,
      ...(mom.planned_start_time ? [`Planned Start    : ${mom.planned_start_time}`] : []),
      ...(mom.actual_start_time  ? [`Actual Start     : ${mom.actual_start_time}`]  : []),
      '',
      'INTRODUCTION',
      '------------',
      mom.introduction,
      '',
      'POINTS DISCUSSED',
      '----------------',
      ...mom.points_discussed.map((p, i) => `${i + 1}. ${p}`),
      '',
      'ACTION POINTS',
      '-------------',
      ...aiLines,
      '',
      'CONCLUSION',
      '----------',
      mom.conclusion,
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
              {recording.filename} · {fmtDuration(recording.duration)}
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
              <p style={{ fontSize: '0.9rem', color: 'hsl(var(--pencil))', maxWidth: '420px', lineHeight: 1.6, fontFamily: 'Inter, sans-serif' }}>
                AI will extract a structured MoM — introduction, discussion points, action items by speaker, and conclusion.
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
              {[80, 200, 120, 160, 100, 140].map((h, i) => (
                <div key={i} style={{ height: h, borderRadius: '12px', background: 'linear-gradient(90deg, hsl(var(--muted)) 0%, hsl(var(--card)) 50%, hsl(var(--muted)) 100%)', backgroundSize: '200% 100%', animation: 'shimmer 1.5s ease-in-out infinite', animationDelay: `${i * 0.15}s`, border: '1px solid hsl(var(--border) / .3)' }} />
              ))}
            </div>
          </div>
        )}

        {/* Editing view */}
        {pageState === 'editing' && mom && (
          <div className="mom-editor-layout" style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>

            <div className="mom-editor-main" style={{ flex: 1, overflowY: 'auto', padding: '1.5rem', minWidth: 0 }}>

              {/* ── 1. Meeting Title ─────────────────────────────── */}
              <MomSection title="1. Meeting Title" className="mom-section">
                <input
                  className="input"
                  value={mom.title}
                  onChange={e => update('title', e.target.value)}
                  placeholder="Enter meeting title..."
                  style={{ width: '100%', padding: '0.55rem 0.85rem', fontSize: '1rem', fontWeight: 600 }}
                />
              </MomSection>

              {/* ── 2. Date ──────────────────────────────────────── */}
              <MomSection title="2. Date">
                <input
                  className="input"
                  value={mom.date}
                  onChange={e => update('date', e.target.value)}
                  placeholder="e.g. 02 July 2026"
                  style={{ width: '100%', maxWidth: '340px', padding: '0.5rem 0.75rem', fontSize: '0.9rem' }}
                />
              </MomSection>

              {/* ── 3. Members ───────────────────────────────────── */}
              <MomSection title="3. Members">
                <TagInput
                  tags={mom.participants}
                  onChange={tags => update('participants', tags)}
                  placeholder="Type name and press Enter..."
                />
              </MomSection>

              {/* ── 4 & 5. Start Times ───────────────────────────── */}
              <MomSection title="4 & 5. Meeting Times">
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.25rem' }}>
                  <div>
                    <FieldLabel>4. Planned Starting Time <span style={{ fontWeight: 400, color: 'hsl(var(--accent))', fontSize: '0.7rem', marginLeft: '4px' }}>(manual entry)</span></FieldLabel>
                    <input
                      className="input"
                      value={mom.planned_start_time}
                      onChange={e => update('planned_start_time', e.target.value)}
                      placeholder="e.g. 10:00 AM"
                      style={{ width: '100%', padding: '0.5rem 0.75rem', fontSize: '0.9rem' }}
                    />
                  </div>
                  <div>
                    <FieldLabel>5. Actual Starting Time</FieldLabel>
                    <input
                      className="input"
                      value={mom.actual_start_time}
                      onChange={e => update('actual_start_time', e.target.value)}
                      placeholder="e.g. 10:12 AM"
                      style={{ width: '100%', padding: '0.5rem 0.75rem', fontSize: '0.9rem' }}
                    />
                  </div>
                </div>
              </MomSection>

              {/* ── 6. Introduction ──────────────────────────────── */}
              <MomSection title="6. Introduction">
                <p style={{ fontSize: '0.78rem', color: 'hsl(var(--pencil))', fontFamily: 'Inter, sans-serif', marginBottom: '8px', lineHeight: 1.5 }}>
                  A paragraph that clearly defines the meeting agenda and topics discussed.
                </p>
                <textarea
                  className="input"
                  value={mom.introduction}
                  onChange={e => update('introduction', e.target.value)}
                  placeholder="Write the meeting introduction — agenda, purpose, and topics discussed..."
                  rows={5}
                  style={{ width: '100%', resize: 'vertical', padding: '0.75rem', fontSize: '0.9rem', lineHeight: 1.7, fontFamily: 'Inter, sans-serif' }}
                />
              </MomSection>

              {/* ── 7. Points Discussed ──────────────────────────── */}
              <MomSection title="7. Points Discussed">
                <p style={{ fontSize: '0.78rem', color: 'hsl(var(--pencil))', fontFamily: 'Inter, sans-serif', marginBottom: '8px', lineHeight: 1.5 }}>
                  Minimum 3 points, maximum 10 points. Each point should be a complete sentence.
                </p>
                <EditableList
                  items={mom.points_discussed.length > 0 ? mom.points_discussed : ['', '', '']}
                  onChange={items => update('points_discussed', items)}
                  placeholder="Describe a discussion point..."
                  ordered
                  minItems={3}
                  maxItems={10}
                />
              </MomSection>

              {/* ── 8. Action Points ─────────────────────────────── */}
              <MomSection title="8. Action Points">
                <p style={{ fontSize: '0.78rem', color: 'hsl(var(--pencil))', fontFamily: 'Inter, sans-serif', marginBottom: '10px', lineHeight: 1.5 }}>
                  General and speaker-based action items. Set <strong>Owner</strong> to a speaker name for speaker-based items.
                </p>
                <ActionPointsSection items={mom.action_items} onChange={items => update('action_items', items)} />
              </MomSection>

              {/* ── 9. Conclusion ────────────────────────────────── */}
              <MomSection title="9. Conclusion">
                <textarea
                  className="input"
                  value={mom.conclusion}
                  onChange={e => update('conclusion', e.target.value)}
                  placeholder="Summarize the outcomes, agreements reached, and overall conclusion of the meeting..."
                  rows={5}
                  style={{ width: '100%', resize: 'vertical', padding: '0.75rem', fontSize: '0.9rem', lineHeight: 1.7, fontFamily: 'Inter, sans-serif' }}
                />
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
