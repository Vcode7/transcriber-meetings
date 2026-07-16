import { useEffect, useState, useCallback, useRef } from 'react'
import { isAxiosError } from 'axios'
import { useParams, useNavigate } from 'react-router-dom'
import {
  ArrowLeft, Sparkles, Loader, FileDown, Copy, CheckCircle,
  AlertTriangle, Clock, RotateCcw, Plus, X, History, User, Users,
  FileText, Upload, ChevronDown, ChevronUp, Trash2, Brain, Zap, ChevronRight,
  Database
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

interface AttachmentFile {
  id: string
  filename: string
  type: 'agenda' | 'context'
}

type PageState = 'loading' | 'idle' | 'generating' | 'editing'
type SaveState = 'saved' | 'saving' | 'unsaved'
type ProcessState = 'idle' | 'processing' | 'done' | 'error'

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

// ── RAW MOM SECTION (completely independent) ══════════════════════
function RawMomSection({
  id,
  pageState,
}: {
  id: string
  pageState: PageState
}) {
  const [rawMom, setRawMom] = useState<any>(null)
  const [rawMomLoading, setRawMomLoading] = useState(false)
  const [rawMomError, setRawMomError] = useState<string | null>(null)
  const [rawMomLoaded, setRawMomLoaded] = useState(false)
  const [rawMomExpanded, setRawMomExpanded] = useState(true)
  const [openAgendas, setOpenAgendas] = useState<Record<number, boolean>>({})
  const [forceReembed, setForceReembed] = useState(false)

  const handleGenerateRawMom = async () => {
    setRawMomLoading(true)
    setRawMomError(null)
    try {
      const res = await api.post(`/raw-mom/${id}/generate${forceReembed ? '?force_reembed=true' : ''}`)
      setRawMom(res.data)
      setRawMomLoaded(true)
      setRawMomExpanded(true)
      if (res.data?.meeting?.agendas?.length > 0) {
        setOpenAgendas({ 0: true })
      }
    } catch (e: any) {
      setRawMomError(e.response?.data?.detail || 'Failed to generate Raw MoM')
    } finally {
      setRawMomLoading(false)
    }
  }

  // Load existing raw_mom on mount
  useEffect(() => {
    api.get(`/raw-mom/${id}`)
      .then(res => {
        setRawMom(res.data)
        setRawMomLoaded(true)
        if (res.data?.meeting?.agendas?.length > 0) {
          setOpenAgendas({ 0: true })
        }
      })
      .catch(() => { /* 404 = not generated yet */ })
  }, [id])

  const handleExportRawMomJson = () => {
    if (!rawMom) return
    const blob = new Blob([JSON.stringify(rawMom, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `raw-mom-${id}.json`
    a.click()
  }

  return (
    <div className='page-scroll-root' style={{
      borderTop: '2px solid hsl(35,90%,55% / .2)',
      background: 'linear-gradient(180deg, hsl(35,90%,55% / .04) 0%, transparent 100%)',
      padding: '1.25rem 1.5rem',
      flexShrink: 0,
    }}>
      {/* Header Row */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: rawMomExpanded ? '1rem' : 0 }}>
        <div style={{
          width: 30, height: 30, borderRadius: 8, flexShrink: 0,
          background: 'hsl(35,90%,55% / .14)',
          border: '1.5px solid hsl(35,90%,55% / .25)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}>
          <Database size={15} style={{ color: 'hsl(35,90%,55%)' }} />
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ fontSize: '.85rem', fontWeight: 700, color: 'hsl(var(--ink))', fontFamily: 'Inter, sans-serif' }}>
            Raw MoM
            <span style={{
              marginLeft: 8, fontSize: '.68rem', fontWeight: 600,
              color: 'hsl(35,90%,55%)',
              background: 'hsl(35,90%,55% / .12)',
              border: '1px solid hsl(35,90%,55% / .25)',
              padding: '2px 7px', borderRadius: 999,
            }}>RAG Pipeline</span>
          </div>
          <div style={{ fontSize: '.72rem', color: 'hsl(var(--pencil))', fontFamily: 'Inter, sans-serif' }}>
            Per-agenda fact extraction via FAISS retrieval · independent of Generate MoM
          </div>
        </div>
        {/* Controls */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flexShrink: 0 }}>
          {rawMomLoaded && (
            <button
              className="btn btn-ghost"
              onClick={handleExportRawMomJson}
              style={{ fontSize: '.75rem', padding: '.3rem .65rem', gap: '5px', height: 32 }}
            >
              <FileDown size={13} /> JSON
            </button>
          )}
          <label style={{
            display: 'flex', alignItems: 'center', gap: '5px',
            fontSize: '.72rem', color: 'hsl(var(--pencil))', cursor: 'pointer',
            userSelect: 'none', fontFamily: 'Inter, sans-serif',
          }}>
            <input
              type="checkbox"
              checked={forceReembed}
              onChange={e => setForceReembed(e.target.checked)}
              style={{ width: 13, height: 13, accentColor: 'hsl(35,90%,55%)' }}
            />
            Re-embed
          </label>
          <button
            onClick={handleGenerateRawMom}
            disabled={rawMomLoading || pageState === 'generating'}
            className="btn"
            style={{
              fontSize: '.8rem', padding: '.38rem .85rem', gap: '6px', height: 32,
              background: 'linear-gradient(135deg, hsl(35,90%,48%), hsl(28,90%,50%))',
              color: '#fff', border: 'none',
              opacity: (rawMomLoading || pageState === 'generating') ? 0.65 : 1,
            }}
          >
            {rawMomLoading
              ? <><Loader size={13} className="spin" /> Generating...</>
              : <><Zap size={13} /> {rawMomLoaded ? 'Regenerate Raw MoM' : 'Generate Raw MoM'}</>
            }
          </button>
          <button
            onClick={() => setRawMomExpanded(v => !v)}
            className="icon-btn"
            style={{ color: 'hsl(var(--pencil))', width: 30, height: 30 }}
          >
            {rawMomExpanded ? <ChevronUp size={15} /> : <ChevronDown size={15} />}
          </button>
        </div>
      </div>

      {/* Expanded Content */}
      {rawMomExpanded && (
        <div>
          {/* Error */}
          {rawMomError && (
            <div style={{
              padding: '.65rem 1rem', borderRadius: 8, marginBottom: '.75rem',
              background: 'hsl(var(--destructive) / .1)',
              border: '1px solid hsl(var(--destructive) / .25)',
              color: 'hsl(var(--destructive))', fontSize: '.82rem',
              display: 'flex', alignItems: 'center', gap: '7px',
              fontFamily: 'Inter, sans-serif',
            }}>
              <AlertTriangle size={13} style={{ flexShrink: 0 }} />
              {rawMomError}
            </div>
          )}

          {/* Loading Skeleton */}
          {rawMomLoading && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              {[120, 80, 140].map((h, i) => (
                <div key={i} style={{
                  height: h, borderRadius: 10,
                  background: 'linear-gradient(90deg, hsl(35,90%,55% / .08) 0%, hsl(35,90%,55% / .03) 50%, hsl(35,90%,55% / .08) 100%)',
                  backgroundSize: '200% 100%',
                  animation: 'shimmer 1.6s ease-in-out infinite',
                  animationDelay: `${i * 0.2}s`,
                  border: '1px solid hsl(35,90%,55% / .12)',
                }} />
              ))}
              <p style={{ fontSize: '.78rem', color: 'hsl(35,90%,55%)', fontFamily: 'Inter, sans-serif', textAlign: 'center', marginTop: 4 }}>
                Retrieving evidence and extracting facts per agenda item…
              </p>
            </div>
          )}

          {/* No raw mom yet */}
          {!rawMomLoading && !rawMomLoaded && !rawMomError && (
            <div style={{
              textAlign: 'center', padding: '1.5rem',
              borderRadius: 10, border: '1.5px dashed hsl(35,90%,55% / .2)',
              color: 'hsl(var(--pencil))', fontFamily: 'Inter, sans-serif',
            }}>
              <Database size={24} style={{ margin: '0 auto 8px', color: 'hsl(35,90%,55%)', opacity: 0.5 }} />
              <p style={{ margin: 0, fontSize: '.85rem' }}>Click <strong>Generate Raw MoM</strong> to extract structured facts using the RAG pipeline.</p>
              <p style={{ margin: '4px 0 0', fontSize: '.75rem', opacity: .7 }}>
                Retrieves from transcript, meeting context files, and your Global Knowledge Base.
              </p>
            </div>
          )}

          {/* Agenda Panels */}
          {!rawMomLoading && rawMom?.meeting?.agendas && rawMom.meeting.agendas.length > 0 && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              {rawMom.meeting.agendas.map((agenda: any, idx: number) => (
                <div key={idx} style={{
                  borderRadius: 10,
                  border: '1.5px solid hsl(35,90%,55% / .2)',
                  background: 'hsl(35,90%,55% / .03)',
                  overflow: 'hidden',
                }}>
                  {/* Agenda header */}
                  <button
                    onClick={() => setOpenAgendas(prev => ({ ...prev, [idx]: !prev[idx] }))}
                    style={{
                      width: '100%', display: 'flex', alignItems: 'center', gap: '10px',
                      padding: '.7rem .9rem',
                      background: openAgendas[idx] ? 'hsl(35,90%,55% / .07)' : 'transparent',
                      border: 'none', cursor: 'pointer', textAlign: 'left',
                      borderBottom: openAgendas[idx] ? '1px solid hsl(35,90%,55% / .15)' : 'none',
                      transition: 'background 0.15s',
                    }}
                  />
                    <span style={{
                      fontSize: '.68rem', fontWeight: 700, color: 'hsl(35,90%,50%)',
                      background: 'hsl(35,90%,55% / .12)',
                      border: '1px solid hsl(35,90%,55% / .2)',
                      padding: '2px 6px', borderRadius: 6, flexShrink: 0,
                    }}>A{idx + 1}</span>
                    <span style={{ flex: 1, fontSize: '.82rem', fontWeight: 600, color: 'hsl(var(--ink))', fontFamily: 'Inter, sans-serif' }}>
                      {agenda.agenda_topic}
                    </span>
                    {agenda.agenda_speaker && (
                      <span style={{
                        fontSize: '.7rem', color: 'hsl(var(--pencil))',
                        background: 'hsl(var(--muted) / .6)', padding: '2px 7px',
                        borderRadius: 999, flexShrink: 0, fontFamily: 'Inter, sans-serif',
                      }}>{agenda.agenda_speaker}</span>
                    )}
                  </div>
                ))}
              </div>
            )}
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

  // ── Attachment state ──────────────────────────────────────────
  const [contextPanelOpen, setContextPanelOpen] = useState(false)
  const [agendaFiles, setAgendaFiles] = useState<AttachmentFile[]>([])
  const [contextFiles, setContextFiles] = useState<AttachmentFile[]>([])
  const [agendaSummary, setAgendaSummary] = useState<string | null>(null)
  const [referenceSummary, setReferenceSummary] = useState<string | null>(null)
  const [agendaProcessState, setAgendaProcessState] = useState<ProcessState>('idle')
  const [contextProcessState, setContextProcessState] = useState<ProcessState>('idle')
  const agendaInputRef = useRef<HTMLInputElement>(null)
  const contextInputRef = useRef<HTMLInputElement>(null)

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

  // Fetch MoM and recording info
  const fetchData = useCallback(async () => {
    if (!id) return
    try {
      const recRes = await api.get(`/history/${id}`)
      setRecording(recRes.data)
      // Load existing MoM if available
      try {
        const momRes = await api.get(`/mom/${id}`)
        setMom(normalizeMom(momRes.data))
        setPageState('editing')
        const vRes = await api.get(`/mom/${id}/versions`)
        setVersions((vRes.data.versions || []).slice(0, 10))
      } catch (e: unknown) {
        if (isAxiosError(e) && e.response?.status === 404) {
          setPageState('idle')
        } else {
          throw e
        }
      }
      // Load attachments (non-critical)
      try {
        const [attRes, sumRes] = await Promise.all([
          api.get(`/attachments/${id}`),
          api.get(`/attachments/${id}/summaries`),
        ])
        const allFiles: AttachmentFile[] = attRes.data.files || []
        setAgendaFiles(allFiles.filter((f: AttachmentFile) => f.type === 'agenda'))
        setContextFiles(allFiles.filter((f: AttachmentFile) => f.type === 'context'))
        setAgendaSummary(sumRes.data.agenda_summary || null)
        setReferenceSummary(sumRes.data.reference_summary || null)
      } catch { /* attachments optional */ }
    } catch {
      setError('Failed to load recording.')
      setPageState('idle')
    }
  }, [id])

  useEffect(() => { fetchData() }, [fetchData])


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

  // ── Generate MoM ── (EXISTING — untouched)
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

  // ── Raw MoM state (completely independent of existing MoM) ───────────────
  type RawAgendaDiscussion = {
    type: string
    speaker: string | null
    point: string
    dates: { value: string; purpose: string }[]
    action: { owner: string | null; description: string | null; deadline: string | null; status: string | null }
  }
  type RawAgenda = {
    agenda_topic: string
    agenda_speaker: string | null
    discussion: RawAgendaDiscussion[]
  }
  type RawMomData = { meeting: { agendas: RawAgenda[] } }

  const [rawMom, setRawMom] = useState<RawMomData | null>(null)
  const [rawMomLoading, setRawMomLoading] = useState(false)
  const [rawMomError, setRawMomError] = useState<string | null>(null)
  const [rawMomLoaded, setRawMomLoaded] = useState(false)
  const [rawMomExpanded, setRawMomExpanded] = useState(true)
  const [openAgendas, setOpenAgendas] = useState<Record<number, boolean>>({})
  const [forceReembed, setForceReembed] = useState(false)

  // Load existing raw_mom on mount (non-blocking)
  useEffect(() => {
    if (!id) return
    api.get(`/raw-mom/${id}`)
      .then(res => {
        setRawMom(res.data)
        setRawMomLoaded(true)
        // Expand first agenda by default
        if (res.data?.meeting?.agendas?.length > 0) {
          setOpenAgendas({ 0: true })
        }
      })
      .catch(() => { /* 404 means not generated yet */ })
  }, [id])

  const handleGenerateRawMom = async () => {
    if (!id) return
    setRawMomLoading(true)
    setRawMomError(null)
    try {
      const res = await api.post(`/raw-mom/${id}/generate${forceReembed ? '?force_reembed=true' : ''}`)
      setRawMom(res.data)
      setRawMomLoaded(true)
      setRawMomExpanded(true)
      if (res.data?.meeting?.agendas?.length > 0) {
        setOpenAgendas({ 0: true })
      }
    } catch (e: unknown) {
      setRawMomError(getApiErrorDetail(e, 'Raw MoM generation failed. Please try again.'))
    } finally {
      setRawMomLoading(false)
    }
  }

  const handleExportRawMomJson = () => {
    if (!rawMom) return
    const json = JSON.stringify(rawMom, null, 2)
    const blob = new Blob([json], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `RawMoM_${id}.json`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  // ── Attachment handlers ────────────────────────────────────────
  const handleUploadFiles = async (files: FileList | null, type: 'agenda' | 'context') => {
    if (!files || files.length === 0 || !id) return
    const formData = new FormData()
    formData.append('type', type)
    Array.from(files).forEach(f => formData.append('files', f))
    try {
      await api.post(`/attachments/${id}/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      // Refresh file list
      const res = await api.get(`/attachments/${id}`)
      const all: AttachmentFile[] = res.data.files || []
      const filtered = all.filter(f => f.type === type)
      if (type === 'agenda') setAgendaFiles(filtered)
      else setContextFiles(filtered)

      // Automatically generate/update summary in background
      if (filtered.length > 0) {
        await handleProcessFiles(type)
      }
    } catch (e) {
      alert(`Upload failed: ${getApiErrorDetail(e)}`)
    }
  }

  const handleDeleteFile = async (fileId: string, type: 'agenda' | 'context') => {
    if (!id) return
    try {
      await api.delete(`/attachments/${id}/${fileId}`)
      let remainingCount = 0
      if (type === 'agenda') {
        const remaining = agendaFiles.filter(f => f.id !== fileId)
        setAgendaFiles(remaining)
        remainingCount = remaining.length
        if (remainingCount === 0) setAgendaSummary(null)
      } else {
        const remaining = contextFiles.filter(f => f.id !== fileId)
        setContextFiles(remaining)
        remainingCount = remaining.length
        if (remainingCount === 0) setReferenceSummary(null)
      }

      // Automatically regenerate/update summary in background if files still exist
      if (remainingCount > 0) {
        await handleProcessFiles(type)
      }
    } catch (e) {
      alert(`Delete failed: ${getApiErrorDetail(e)}`)
    }
  }

  const handleProcessFiles = async (type: 'agenda' | 'context') => {
    if (!id) return
    const setState = type === 'agenda' ? setAgendaProcessState : setContextProcessState
    setState('processing')
    try {
      const formData = new FormData()
      formData.append('type', type)
      const res = await api.post(`/attachments/${id}/process`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      if (type === 'agenda') setAgendaSummary(res.data.summary)
      else setReferenceSummary(res.data.summary)
      setState('done')
    } catch (e) {
      alert(`Processing failed: ${getApiErrorDetail(e)}`)
      setState('error')
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
    if (!mom) return
    setPdfStatus('loading')
    try {
      // Send current editor state directly — bypasses the 3s auto-save debounce
      // so the exported PDF always reflects the latest unsaved edits.
      const res = await api.post(`/mom/${id}/pdf`, mom, { responseType: 'blob' })
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
    
      {(pageState === 'editing' || pageState === 'idle') && (
        <div style={{
          flexShrink: 0, borderTop: '1px solid hsl(var(--border) / .3)',
          background: 'hsl(var(--card))',
        }}>
          {/* Collapse toggle */}
          <button
            onClick={() => setContextPanelOpen(v => !v)}
            style={{
              width: '100%', display: 'flex', alignItems: 'center', gap: '8px',
              padding: '0.65rem 1.25rem', background: 'transparent', border: 'none',
              cursor: 'pointer', color: 'hsl(var(--ink))', fontFamily: 'Inter, sans-serif',
              fontSize: '0.82rem', fontWeight: 600,
            }}
          >
            <Brain size={14} style={{ color: 'hsl(var(--accent))' }} />
            AI Context Files
            <span style={{ fontSize: '0.72rem', color: 'hsl(var(--pencil))', fontWeight: 400, marginLeft: 2 }}>
              — Upload agenda &amp; reference docs to improve MoM quality
            </span>
            {(agendaSummary || referenceSummary) && (
              <span style={{
                fontSize: '0.68rem', padding: '0.1rem 0.5rem', borderRadius: '99px',
                background: 'hsl(var(--accent) / .12)', color: 'hsl(var(--accent))',
                fontWeight: 700, marginLeft: 4,
              }}>Active</span>
            )}
            <span style={{ marginLeft: 'auto', color: 'hsl(var(--pencil))' }}>
              {contextPanelOpen ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
            </span>
          </button>

          {contextPanelOpen && (
            <div style={{ padding: '0 1.25rem 1.25rem', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>

              {/* Hidden file inputs */}
              <input ref={agendaInputRef} type="file" multiple accept=".pdf,.docx,.pptx,.txt,.md,.png,.jpg,.jpeg,.webp,.xlsx,.xls,.csv" style={{ display: 'none' }}
                onChange={e => handleUploadFiles(e.target.files, 'agenda')} />
              <input ref={contextInputRef} type="file" multiple accept=".pdf,.docx,.pptx,.txt,.md,.png,.jpg,.jpeg,.webp,.xlsx,.xls,.csv" style={{ display: 'none' }}
                onChange={e => handleUploadFiles(e.target.files, 'context')} />

              {/* ── Agenda Files ── */}
              <div style={{ border: '1px solid hsl(var(--border) / .4)', borderRadius: '12px', padding: '1rem', background: 'hsl(var(--paper) / .5)' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '0.75rem' }}>
                  <FileText size={13} style={{ color: 'hsl(var(--accent))' }} />
                  <span style={{ fontSize: '0.8rem', fontWeight: 700, color: 'hsl(var(--ink))', fontFamily: 'Inter, sans-serif' }}>Agenda Files</span>
                  <span style={{ fontSize: '0.72rem', color: 'hsl(var(--pencil))', marginLeft: 'auto' }}>PDF, DOCX, PPTX, TXT, MD, Images</span>
                </div>

                {/* Uploaded files list */}
                {agendaFiles.length > 0 && (
                  <div style={{ marginBottom: '0.6rem', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                    {agendaFiles.map(f => (
                      <div key={f.id} style={{ display: 'flex', alignItems: 'center', gap: '6px', padding: '4px 8px', borderRadius: '6px', background: 'hsl(var(--muted) / .4)', fontSize: '0.78rem', fontFamily: 'Inter, sans-serif' }}>
                        <FileText size={11} style={{ flexShrink: 0, color: 'hsl(var(--pencil))' }} />
                        <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', color: 'hsl(var(--ink))' }}>{f.filename}</span>
                        <button onClick={() => handleDeleteFile(f.id, 'agenda')} style={{ background: 'none', border: 'none', cursor: 'pointer', padding: '2px', color: 'hsl(var(--destructive))', flexShrink: 0 }}>
                          <Trash2 size={11} />
                        </button>
                      </div>
                    ))}
                  </div>
                )}

                {/* Drop zone */}
                <div
                  onClick={() => agendaInputRef.current?.click()}
                  style={{
                    border: '1.5px dashed hsl(var(--border) / .6)', borderRadius: '8px',
                    padding: '0.6rem', textAlign: 'center', cursor: 'pointer',
                    fontSize: '0.78rem', color: 'hsl(var(--pencil))', fontFamily: 'Inter, sans-serif',
                    display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px',
                    transition: 'border-color 0.2s',
                    marginBottom: '0.6rem',
                  }}
                  onMouseOver={e => (e.currentTarget.style.borderColor = 'hsl(var(--accent))')}
                  onMouseOut={e => (e.currentTarget.style.borderColor = 'hsl(var(--border) / .6)')}
                >
                  <Upload size={12} /> Click to upload agenda files
                </div>

                {/* Process button */}
                <button
                  className="btn btn-ghost"
                  disabled={agendaFiles.length === 0 || agendaProcessState === 'processing'}
                  onClick={() => handleProcessFiles('agenda')}
                  style={{ width: '100%', fontSize: '0.8rem', padding: '0.4rem 0.7rem', gap: '6px' }}
                >
                  {agendaProcessState === 'processing'
                    ? <><Loader size={12} className="spin" /> Processing...</>
                    : agendaProcessState === 'done'
                    ? <><CheckCircle size={12} style={{ color: 'hsl(var(--success))' }} /> Re-process</>
                    : <><Brain size={12} /> Extract &amp; Summarize</>}
                </button>

                {/* Summary preview */}
                {agendaSummary && (
                  <div style={{ marginTop: '0.6rem', padding: '0.6rem 0.75rem', borderRadius: '8px', background: 'hsl(var(--accent) / .07)', border: '1px solid hsl(var(--accent) / .2)' }}>
                    <p style={{ fontSize: '0.72rem', fontWeight: 700, color: 'hsl(var(--accent))', fontFamily: 'Inter, sans-serif', marginBottom: '4px' }}>Agenda Summary ✓</p>
                    <p style={{ fontSize: '0.74rem', color: 'hsl(var(--ink))', fontFamily: 'Inter, sans-serif', lineHeight: 1.6, whiteSpace: 'pre-wrap', maxHeight: '80px', overflow: 'scroll' }}>{agendaSummary}</p>
                  </div>
                )}
              </div>

              {/* ── Context Files ── */}
              <div style={{ border: '1px solid hsl(var(--border) / .4)', borderRadius: '12px', padding: '1rem', background: 'hsl(var(--paper) / .5)' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '0.75rem' }}>
                  <FileText size={13} style={{ color: '#8b5cf6' }} />
                  <span style={{ fontSize: '0.8rem', fontWeight: 700, color: 'hsl(var(--ink))', fontFamily: 'Inter, sans-serif' }}>Context Files</span>
                  <span style={{ fontSize: '0.72rem', color: 'hsl(var(--pencil))', marginLeft: 'auto' }}>Background knowledge</span>
                </div>

                {contextFiles.length > 0 && (
                  <div style={{ marginBottom: '0.6rem', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                    {contextFiles.map(f => (
                      <div key={f.id} style={{ display: 'flex', alignItems: 'center', gap: '6px', padding: '4px 8px', borderRadius: '6px', background: 'hsl(var(--muted) / .4)', fontSize: '0.78rem', fontFamily: 'Inter, sans-serif' }}>
                        <FileText size={11} style={{ flexShrink: 0, color: 'hsl(var(--pencil))' }} />
                        <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', color: 'hsl(var(--ink))' }}>{f.filename}</span>
                        <button onClick={() => handleDeleteFile(f.id, 'context')} style={{ background: 'none', border: 'none', cursor: 'pointer', padding: '2px', color: 'hsl(var(--destructive))', flexShrink: 0 }}>
                          <Trash2 size={11} />
                        </button>
                      </div>
                    ))}
                  </div>
                )}

                <div
                  onClick={() => contextInputRef.current?.click()}
                  style={{
                    border: '1.5px dashed hsl(var(--border) / .6)', borderRadius: '8px',
                    padding: '0.6rem', textAlign: 'center', cursor: 'pointer',
                    fontSize: '0.78rem', color: 'hsl(var(--pencil))', fontFamily: 'Inter, sans-serif',
                    display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px',
                    transition: 'border-color 0.2s',
                    marginBottom: '0.6rem',
                  }}
                  onMouseOver={e => (e.currentTarget.style.borderColor = '#8b5cf6')}
                  onMouseOut={e => (e.currentTarget.style.borderColor = 'hsl(var(--border) / .6)')}
                >
                  <Upload size={12} /> Click to upload context files
                </div>

                <button
                  className="btn btn-ghost"
                  disabled={contextFiles.length === 0 || contextProcessState === 'processing'}
                  onClick={() => handleProcessFiles('context')}
                  style={{ width: '100%', fontSize: '0.8rem', padding: '0.4rem 0.7rem', gap: '6px' }}
                >
                  {contextProcessState === 'processing'
                    ? <><Loader size={12} className="spin" /> Processing...</>
                    : contextProcessState === 'done'
                    ? <><CheckCircle size={12} style={{ color: 'hsl(var(--success))' }} /> Re-process</>
                    : <><Brain size={12} /> Extract &amp; Summarize</>}
                </button>

                {referenceSummary && (
                  <div style={{ marginTop: '0.6rem', padding: '0.6rem 0.75rem', borderRadius: '8px', background: 'hsl(#8b5cf6 / .07)', border: '1px solid hsl(#8b5cf6 / .2)', borderColor: '#8b5cf620' }}>
                    <p style={{ fontSize: '0.72rem', fontWeight: 700, color: '#8b5cf6', fontFamily: 'Inter, sans-serif', marginBottom: '4px' }}>Context Summary ✓</p>
                    <p style={{ fontSize: '0.74rem', color: 'hsl(var(--ink))', fontFamily: 'Inter, sans-serif', lineHeight: 1.6, whiteSpace: 'pre-wrap', maxHeight: '80px', overflow: 'scroll' }}>{referenceSummary}</p>
                  </div>
                )}
              </div>

            </div>
          )}
        </div>
      )}

      {/* ══ Raw MoM Section (completely independent of MoM pipeline) ══ */}
      {/* {id && <RawMomSection id={id} pageState={pageState} />} */}

    </div>
  )
}
