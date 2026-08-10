import { useEffect, useState, useCallback, useRef } from 'react'
import { isAxiosError } from 'axios'
import { useParams, useNavigate } from 'react-router-dom'
import {
  ArrowLeft, Sparkles, Loader, FileDown, Copy, CheckCircle,
  AlertTriangle, Clock, RotateCcw, Plus, X, History, User, Users,
  FileText, Upload, ChevronDown, ChevronUp, Trash2, Brain, Zap, ChevronRight,
  Wand2, Settings2, SlidersHorizontal, Save, RefreshCw, AlignLeft
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
  planned_end_time: string
  actual_end_time: string
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
type RewriteTarget = 'discussion_points' | 'action_items' | 'introduction' | 'conclusion' | null
type RewritePhase = 'upload' | 'analyzing' | 'rules' | 'rewriting' | 'done'

// ── Normalize API response to safe MomData ─────────────────────
// Convert a raw points_discussed entry (string or object) to a clean display string
function normalizePoint(pt: unknown): string {
  if (typeof pt === 'string') return pt
  if (pt && typeof pt === 'object' && !Array.isArray(pt)) {
    const obj = pt as Record<string, unknown>
    const topic = typeof obj.topic === 'string' ? obj.topic.trim() : ''
    const summary = typeof obj.summary === 'string' ? obj.summary.trim()
      : typeof obj.discussion_point === 'string' ? obj.discussion_point.trim()
        : typeof obj.text === 'string' ? obj.text.trim() : ''
    if (topic && summary && !summary.toLowerCase().startsWith(topic.toLowerCase())) {
      return `${topic}: ${summary}`
    }
    return summary || topic || JSON.stringify(pt)
  }
  return String(pt ?? '')
}

function normalizeMom(raw: Partial<MomData> | null | undefined): MomData | null {
  if (!raw) return null
  return {
    title: typeof raw.title === 'string' ? raw.title : '',
    date: typeof raw.date === 'string' ? raw.date : '',
    duration: typeof raw.duration === 'number' ? raw.duration : 0,
    planned_start_time: typeof raw.planned_start_time === 'string' ? raw.planned_start_time : '',
    actual_start_time: typeof raw.actual_start_time === 'string' ? raw.actual_start_time : '',
    planned_end_time: typeof raw.planned_end_time === 'string' ? raw.planned_end_time : '',
    actual_end_time: typeof raw.actual_end_time === 'string' ? raw.actual_end_time : '',
    participants: Array.isArray(raw.participants) ? raw.participants.map(String) : [],
    introduction: typeof raw.introduction === 'string' ? raw.introduction : '',
    points_discussed: Array.isArray(raw.points_discussed)
      ? raw.points_discussed.map(normalizePoint).filter(Boolean)
      : typeof raw.points_discussed === 'string' && (raw.points_discussed as string).trim()
        ? (raw.points_discussed as string).split('\n').map(s => s.replace(/^[\s\u2022\-*]+/, '').trim()).filter(Boolean)
        : [],
    action_items: Array.isArray(raw.action_items)
      ? raw.action_items.map((a: unknown) => {
        if (a && typeof a === 'object' && !Array.isArray(a)) {
          const obj = a as Record<string, unknown>
          return {
            task: String(obj.task ?? obj.item ?? obj.description ?? ''),
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
}: {
  items: string[]
  onChange: (items: string[]) => void
  placeholder?: string
  ordered?: boolean
  minItems?: number
}) {
  const handleChange = (index: number, value: string) => {
    const newItems = [...items]
    newItems[index] = value
    onChange(newItems)
  }
  const handleAdd = () => {
    onChange([...items, ''])
  }
  const handleRemove = (index: number) => {
    if (minItems && items.length <= minItems) return
    onChange(items.filter((_, i) => i !== index))
  }
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
      <button onClick={handleAdd} className="btn btn-ghost"
        style={{ fontSize: '0.8rem', padding: '0.3rem 0.7rem', gap: '5px', alignSelf: 'flex-start', marginTop: '4px' }}>
        <Plus size={13} /> Add
      </button>
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
  const [docxStatus, setDocxStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle')
  const [versionSaveStatus, setVersionSaveStatus] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle')
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

  // ── Rewrite state ─────────────────────────────────────────────
  const [rewriteTarget, setRewriteTarget] = useState<RewriteTarget>(null)
  const [rewritePhase, setRewritePhase] = useState<RewritePhase>('upload')
  const [rewriteRules, setRewriteRules] = useState('')
  const [rewriteCustomPrompt, setRewriteCustomPrompt] = useState('')
  const [rewriteWindowSize, setRewriteWindowSize] = useState(20)
  const [rewriteFiles, setRewriteFiles] = useState<File[]>([])
  const [rewriteProgress, setRewriteProgress] = useState({ current: 0, total: 0 })
  const [rewriteError, setRewriteError] = useState<string | null>(null)
  const rewriteFileInputRef = useRef<HTMLInputElement>(null)

  // ── Regenerate Action Points state ────────────────────────────
  const [regenActionStatus, setRegenActionStatus] = useState<'idle' | 'processing' | 'done' | 'error'>('idle')
  const [regenWindowMinutes, setRegenWindowMinutes] = useState(10)
  const [regenActionOpen, setRegenActionOpen] = useState(false)
  const [regenActionError, setRegenActionError] = useState<string | null>(null)

  // ── Summarize Long Points state ────────────────────────────
  const [summarizeLongOpen, setSummarizeLongOpen] = useState(false)
  const [summarizeThreshold, setSummarizeThreshold] = useState(70)
  const [summarizeStatus, setSummarizeStatus] = useState<'idle' | 'processing' | 'done' | 'error'>('idle')
  const [summarizeError, setSummarizeError] = useState<string | null>(null)
  const [summarizeInfo, setSummarizeInfo] = useState<string | null>(null)

  const GENERATING_STEPS = [
    'Reading transcript...',
    'Extracting meeting topics...',
    'Identifying action points...',
    'Drafting introduction...',
    'Writing conclusion...',
    'Finalizing MoM...',
  ]

  // ── Regenerate Action Points handler ──────────────────────────
  const handleRegenerateActionPoints = useCallback(async () => {
    if (!id || regenActionStatus === 'processing') return
    setRegenActionStatus('processing')
    setRegenActionError(null)
    try {
      const res = await api.post(`/mom/${id}/regenerate-action-points`, {
        window_minutes: regenWindowMinutes,
      })
      const newItems: ActionItem[] = (res.data.action_items || []).map((a: Record<string, string>) => ({
        task: String(a.task ?? ''),
        owner: String(a.owner ?? 'Unassigned'),
        deadline: String(a.deadline ?? 'ASAP'),
      }))
      setMom(prev => prev ? { ...prev, action_items: newItems } : prev)
      setSaveState('unsaved')
      setRegenActionStatus('done')
      setRegenActionOpen(false)
    } catch (err) {
      const detail = isAxiosError(err) ? (err.response?.data?.detail ?? err.message) : String(err)
      setRegenActionError(detail)
      setRegenActionStatus('error')
    }
  }, [id, regenWindowMinutes, regenActionStatus])

  // ── Summarize Long Points handler ───────────────────────────
  const handleSummarizeLongPoints = useCallback(async () => {
    if (!id || summarizeStatus === 'processing') return
    setSummarizeStatus('processing')
    setSummarizeError(null)
    setSummarizeInfo(null)

    if (saveTimer.current) {
      clearTimeout(saveTimer.current)
      saveTimer.current = null
    }

    try {
      const res = await api.post(`/mom/${id}/summarize-long-points`, {
        threshold: summarizeThreshold,
      })
      const updated = normalizeMom(res.data.mom)
      if (updated) {
        setMom(updated)
        setSaveState('saved')
      }
      try {
        const vRes = await api.get(`/mom/${id}/versions`)
        setVersions(vRes.data.versions || [])
      } catch { /* ignore */ }
      setSummarizeStatus('done')
      const condensedCount = res.data.condensed_count || 0
      if (condensedCount > 0) {
        setSummarizeInfo(`Condensed ${condensedCount} long point${condensedCount !== 1 ? 's' : ''} (> ${summarizeThreshold} words) as Version #${res.data.version_number}. Revert anytime from History.`)
      } else {
        setSummarizeInfo(`No points exceeded the ${summarizeThreshold}-word threshold. All points remain unchanged.`)
      }
    } catch (err) {
      const detail = isAxiosError(err) ? (err.response?.data?.detail ?? err.message) : String(err)
      setSummarizeError(detail)
      setSummarizeStatus('error')
    }
  }, [id, summarizeThreshold, summarizeStatus])

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

  // ── Rewrite handlers ──────────────────────────────────────────
  const openRewrite = (target: NonNullable<RewriteTarget>) => {
    setRewriteTarget(target)
    setRewritePhase(target === 'introduction' || target === 'conclusion' ? 'rules' : 'upload')
    setRewriteRules(target === 'discussion_points'
      ? '1. Merge discussion points covering the same or highly similar topics into a single coherent point without losing any factual information, decisions, questions, or outcomes.'
      : ''
    )
    setRewriteCustomPrompt('')
    setRewriteWindowSize(20)
    setRewriteFiles([])
    setRewriteError(null)
    setRewriteProgress({ current: 0, total: 0 })
  }

  const closeRewrite = () => {
    setRewriteTarget(null)
    setRewriteFiles([])
    setRewriteError(null)
  }

  const handleAnalyzeStyle = async () => {
    if (!id || rewriteFiles.length === 0 || !rewriteTarget) return
    setRewritePhase('analyzing')
    setRewriteError(null)
    try {
      const formData = new FormData()
      formData.append('section', rewriteTarget)
      rewriteFiles.forEach(f => formData.append('files', f))
      const res = await api.post(`/mom/${id}/rewrite/analyze-style`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      const generated = res.data.rules || ''
      // Prepend the default merge rule for discussion points
      if (rewriteTarget === 'discussion_points') {
        setRewriteRules(
          '1. Merge discussion points covering the same or highly similar topics into a single coherent point without losing any factual information, decisions, questions, or outcomes.\n\n' +
          generated
        )
      } else {
        setRewriteRules(generated)
      }
      setRewritePhase('rules')
    } catch (e) {
      setRewriteError(getApiErrorDetail(e, 'Style analysis failed. Please try again.'))
      setRewritePhase('upload')
    }
  }

  const handleRewrite = async () => {
    if (!id || !mom || !rewriteTarget) return
    setRewritePhase('rewriting')
    setRewriteError(null)

    // Cancel any pending debounced auto-save timer to prevent it from overwriting the rewritten DB record!
    if (saveTimer.current) {
      clearTimeout(saveTimer.current)
      saveTimer.current = null
      console.log('[MoM Rewrite FE Step 1/6] Cancelled pending auto-save timer prior to rewrite.')
    }

    let content: string[] | string
    if (rewriteTarget === 'discussion_points') {
      content = mom.points_discussed.map(pt => normalizePoint(pt))
    } else if (rewriteTarget === 'action_items') {
      content = mom.action_items as any
    } else if (rewriteTarget === 'introduction') {
      content = mom.introduction
    } else {
      content = mom.conclusion
    }

    const isList = rewriteTarget === 'discussion_points' || rewriteTarget === 'action_items'
    const itemCount = Array.isArray(content) ? content.length : 1
    const totalBatches = isList ? Math.ceil(itemCount / rewriteWindowSize) : 1
    setRewriteProgress({ current: 0, total: totalBatches })

    console.log('[MoM Rewrite FE Step 2/6] Sending rewrite request to backend:', {
      recording_id: id,
      section: rewriteTarget,
      window_size: rewriteWindowSize,
      item_count: itemCount,
      rules_preview: rewriteRules.slice(0, 100),
    })

    try {
      const progressInterval = setInterval(() => {
        setRewriteProgress(prev => ({
          ...prev,
          current: Math.min(prev.current + 1, Math.max(1, totalBatches - 1)),
        }))
      }, 2000)

      const res = await api.post(`/mom/${id}/rewrite/section`, {
        section: rewriteTarget,
        content,
        rules: rewriteRules,
        custom_prompt: rewriteCustomPrompt,
        window_size: rewriteWindowSize,
      })

      clearInterval(progressInterval)
      setRewriteProgress({ current: totalBatches, total: totalBatches })

      console.log('[MoM Rewrite FE Step 3/6] API Response received:', res.data)

      const updated = normalizeMom(res.data.mom)
      if (updated) {
        // Cancel any auto-save timer again
        if (saveTimer.current) {
          clearTimeout(saveTimer.current)
          saveTimer.current = null
        }
        momRef.current = updated
        setMom(updated)
        setSaveState('saved')

        console.log('[MoM Rewrite FE Step 4/6] React state updated via setMom(). New points_discussed count:', updated.points_discussed.length)
        console.log('[MoM Rewrite FE Step 4/6] New points_discussed content:', updated.points_discussed)
      } else {
        console.error('[MoM Rewrite FE Step 4/6 ERROR] Failed to normalize MoM from server response:', res.data)
      }

      // Refresh version history so the new rewrite version appears
      loadVersions()
      setRewritePhase('done')
    } catch (e) {
      console.error('[MoM Rewrite FE Step 4/6 ERROR] Rewrite request failed:', e)
      setRewriteError(getApiErrorDetail(e, 'Rewrite failed. Please try again.'))
      setRewritePhase('rules')
    }
  }

  // ── Save manual version entry ──
  const handleSaveVersion = async () => {
    if (!mom || !id) return
    setVersionSaveStatus('saving')
    if (saveTimer.current) {
      clearTimeout(saveTimer.current)
      saveTimer.current = null
    }
    try {
      const res = await api.post(`/mom/${id}/version`, mom)
      setSaveState('saved')
      setVersionSaveStatus('saved')
      if (res.data.versions) {
        setVersions(res.data.versions.slice(0, 10))
      } else {
        loadVersions()
      }
      setTimeout(() => setVersionSaveStatus('idle'), 2500)
    } catch {
      setVersionSaveStatus('error')
      setTimeout(() => setVersionSaveStatus('idle'), 3000)
    }
  }

  // ── Export DOCX ──
  const handleDocx = async () => {
    if (!mom) return
    setDocxStatus('loading')
    try {
      // Send current editor state directly — bypasses the 3s auto-save debounce
      // so the exported DOCX always reflects the latest unsaved edits.
      const res = await api.post(`/mom/${id}/docx`, mom, { responseType: 'blob' })
      const blob = new Blob([res.data], { type: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `MoM_${mom?.title?.replace(/\s+/g, '_') || 'Meeting'}.docx`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
      setDocxStatus('success')
      setTimeout(() => setDocxStatus('idle'), 2500)
    } catch {
      setDocxStatus('error')
      setTimeout(() => setDocxStatus('idle'), 3000)
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
      `Meeting Title      : ${mom.title}`,
      `Date               : ${mom.date}`,
      `Members            : ${mom.participants.join(', ')}`,
      ...(mom.planned_start_time ? [`Planned Start Time : ${mom.planned_start_time}`] : []),
      ...(mom.actual_start_time ? [`Actual Start Time  : ${mom.actual_start_time}`] : []),
      ...(mom.planned_end_time ? [`Planned End Time   : ${mom.planned_end_time}`] : []),
      ...(mom.actual_end_time ? [`Actual End Time    : ${mom.actual_end_time}`] : []),
      '',
      'INTRODUCTION',
      '------------',
      mom.introduction,
      '',
      'POINTS DISCUSSED',
      '----------------',
      ...mom.points_discussed.map((p, i) => `${i + 1}. ${normalizePoint(p)}`),
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
            <button className="btn btn-ghost" onClick={handleSaveVersion} disabled={versionSaveStatus === 'saving'} style={{ fontSize: '0.82rem', padding: '0.4rem 0.85rem', height: '36px', gap: '6px' }} title="Save current MoM as a new version entry in History">
              {versionSaveStatus === 'saving' && <><Loader size={13} className="spin" />Saving...</>}
              {versionSaveStatus === 'saved' && <><CheckCircle size={13} style={{ color: 'hsl(var(--success))' }} /><span style={{ color: 'hsl(var(--success))' }}>Saved!</span></>}
              {versionSaveStatus === 'error' && <><AlertTriangle size={13} />Failed</>}
              {versionSaveStatus === 'idle' && <><Save size={14} style={{ color: 'hsl(var(--accent))' }} />Save Version</>}
            </button>
            <button className="btn btn-ghost" onClick={handleCopy} style={{ fontSize: '0.82rem', padding: '0.4rem 0.85rem', height: '36px', gap: '6px' }}>
              {copyStatus === 'copied'
                ? <><CheckCircle size={14} style={{ color: 'hsl(var(--success))' }} /><span style={{ color: 'hsl(var(--success))' }}>Copied!</span></>
                : <><Copy size={14} />Copy</>}
            </button>
            <button className="btn btn-ghost" onClick={handleGenerate} style={{ fontSize: '0.82rem', padding: '0.4rem 0.85rem', height: '36px', gap: '6px' }}>
              <RotateCcw size={14} /> Regenerate
            </button>
            <button className="btn btn-primary" onClick={handleDocx} disabled={docxStatus === 'loading'} style={{ fontSize: '0.82rem', padding: '0.4rem 0.9rem', height: '36px', gap: '6px' }}>
              {docxStatus === 'loading' && <><Loader size={13} className="spin" />Generating...</>}
              {docxStatus === 'success' && <><CheckCircle size={13} />Downloaded!</>}
              {docxStatus === 'error' && <><AlertTriangle size={13} />Failed</>}
              {docxStatus === 'idle' && <><FileDown size={13} />Export DOCX</>}
            </button>
          </div>
        )}
      </div>

      {/* Body */}
      <div style={{ flex: 1, overflowY: 'auto', minHeight: 0, display: 'flex' }}>

        {/* Idle - Actions CTA */}
        {pageState === 'idle' && (
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '3rem 2rem', gap: '1.5rem', textAlign: 'center' }}>
            <div style={{ width: '100px', height: '100px', borderRadius: '50%', background: 'hsl(var(--accent) / .08)', border: '2.5px dashed hsl(var(--accent) / .3)', display: 'flex', alignItems: 'center', justifyContent: 'center', boxShadow: '0 0 0 12px hsl(var(--accent) / .04)' }}>
              <Sparkles size={42} style={{ color: 'hsl(var(--accent))', opacity: 0.75 }} className="animate-float" />
            </div>
            <div>
              <h2 style={{ fontSize: '1.3rem', fontWeight: 700, color: 'hsl(var(--ink))', fontFamily: 'Inter, sans-serif', marginBottom: '.5rem' }}>No Minutes of Meeting Generated</h2>
              <p style={{ fontSize: '0.9rem', color: 'hsl(var(--pencil))', maxWidth: '420px', lineHeight: 1.6, fontFamily: 'Inter, sans-serif' }}>
                Select an action below to generate a structured Minutes of Meeting.
              </p>
            </div>
            {error && (
              <div style={{ padding: '0.75rem 1.25rem', borderRadius: '10px', background: 'hsl(var(--destructive) / .1)', border: '1px solid hsl(var(--destructive) / .3)', color: 'hsl(var(--destructive))', fontSize: '0.88rem', fontFamily: 'Inter, sans-serif' }}>
                <AlertTriangle size={14} style={{ display: 'inline', marginRight: '6px' }} />
                {error}
              </div>
            )}
            <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap', justifyContent: 'center' }}>
              <button id="btn-generate-mom" className="btn btn-primary" onClick={handleGenerate} style={{ fontSize: '0.95rem', padding: '0.75rem 1.75rem', gap: '8px', borderRadius: '12px' }}>
                <Sparkles size={16} /> Generate MoM
              </button>
            </div>
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

              {/* ── 4 & 5. Meeting Times ──────────────────────────── */}
              <MomSection title="4 & 5. Meeting Times">
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.25rem' }}>
                  <div>
                    <FieldLabel>4. Planned Starting Time <span style={{ fontWeight: 400, color: 'hsl(var(--accent))', fontSize: '0.7rem', marginLeft: '4px' }}>(manual)</span></FieldLabel>
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
                  <div>
                    <FieldLabel>Planned End Time</FieldLabel>
                    <input
                      className="input"
                      value={mom.planned_end_time}
                      onChange={e => update('planned_end_time', e.target.value)}
                      placeholder="e.g. 11:00 AM"
                      style={{ width: '100%', padding: '0.5rem 0.75rem', fontSize: '0.9rem' }}
                    />
                  </div>
                  <div>
                    <FieldLabel>Actual End Time</FieldLabel>
                    <input
                      className="input"
                      value={mom.actual_end_time}
                      onChange={e => update('actual_end_time', e.target.value)}
                      placeholder="e.g. 11:18 AM"
                      style={{ width: '100%', padding: '0.5rem 0.75rem', fontSize: '0.9rem' }}
                    />
                  </div>
                </div>
              </MomSection>

              {/* ── 6. Introduction ──────────────────────────────── */}
              <MomSection title="6. Introduction">
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
                  <p style={{ fontSize: '0.78rem', color: 'hsl(var(--pencil))', fontFamily: 'Inter, sans-serif', lineHeight: 1.5 }}>
                    A paragraph that clearly defines the meeting agenda and topics discussed.
                  </p>
                  <button
                    className="btn btn-ghost"
                    onClick={() => openRewrite('introduction')}
                    style={{ fontSize: '0.78rem', padding: '0.3rem 0.7rem', gap: '5px', flexShrink: 0, marginLeft: '12px' }}
                    title="Rewrite Introduction with AI"
                  >
                    <Wand2 size={13} style={{ color: 'hsl(var(--accent))' }} /> Rewrite
                  </button>
                </div>
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
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: summarizeLongOpen ? '6px' : '8px' }}>
                  <p style={{ fontSize: '0.78rem', color: 'hsl(var(--pencil))', fontFamily: 'Inter, sans-serif', lineHeight: 1.5 }}>
                    Each point is displayed as <strong>Topic: Summary</strong>. Minimum 3 points.
                  </p>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flexShrink: 0, marginLeft: '12px' }}>
                    <button
                      id="btn-summarize-long-points"
                      className="btn btn-ghost"
                      onClick={() => {
                        setSummarizeLongOpen(v => !v)
                        setSummarizeError(null)
                        setSummarizeInfo(null)
                        if (summarizeStatus === 'done' || summarizeStatus === 'error') setSummarizeStatus('idle')
                      }}
                      style={{ fontSize: '0.78rem', padding: '0.3rem 0.7rem', gap: '5px' }}
                      title="Summarize long discussion points exceeding word threshold"
                    >
                      <AlignLeft size={13} style={{ color: 'hsl(var(--accent))' }} /> Summarize Long Points
                    </button>
                    <button
                      className="btn btn-ghost"
                      onClick={() => openRewrite('discussion_points')}
                      style={{ fontSize: '0.78rem', padding: '0.3rem 0.7rem', gap: '5px' }}
                      title="Rewrite Discussion Points with AI"
                    >
                      <Wand2 size={13} style={{ color: 'hsl(var(--accent))' }} /> Rewrite
                    </button>
                  </div>
                </div>

                {/* ── Summarize Long Points panel ─────────────────────────── */}
                {summarizeLongOpen && (
                  <div style={{
                    marginBottom: '12px',
                    padding: '0.9rem 1rem',
                    borderRadius: '10px',
                    border: '1.5px solid hsl(var(--accent) / .3)',
                    background: 'hsl(var(--accent) / .05)',
                  }}>
                    <div style={{ fontSize: '0.8rem', fontWeight: 700, color: 'hsl(var(--ink))', marginBottom: '8px', fontFamily: 'Inter, sans-serif' }}>
                      Summarize Long Discussion Points
                    </div>
                    <p style={{ fontSize: '0.75rem', color: 'hsl(var(--pencil))', lineHeight: 1.5, marginBottom: '12px', fontFamily: 'Inter, sans-serif' }}>
                      Sends discussion points exceeding the word threshold to the LLM to be condensed while preserving key factual information. Points below the threshold remain unchanged. Creates a new version in History.
                    </p>

                    {/* Threshold control */}
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '12px' }}>
                      <label style={{ fontSize: '0.75rem', fontWeight: 600, color: 'hsl(var(--pencil))', fontFamily: 'Inter, sans-serif', whiteSpace: 'nowrap' }}>
                        Word threshold
                      </label>
                      <input
                        type="range" min={20} max={300} step={5}
                        value={summarizeThreshold}
                        onChange={e => setSummarizeThreshold(Number(e.target.value))}
                        style={{ flex: 1, accentColor: 'hsl(var(--accent))' }}
                      />
                      <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '0.88rem', fontWeight: 700, minWidth: 44, color: 'hsl(var(--accent))' }}>
                        {summarizeThreshold} words
                      </span>
                      <input
                        type="number" min={10} max={1000}
                        value={summarizeThreshold}
                        onChange={e => setSummarizeThreshold(Math.max(10, Math.min(1000, Number(e.target.value))))}
                        style={{ width: 64, padding: '0.3rem 0.5rem', fontSize: '0.82rem', borderRadius: 6, border: '1.5px solid hsl(var(--border)/.6)', background: 'hsl(var(--surface))', color: 'hsl(var(--ink))', fontFamily: 'JetBrains Mono, monospace', textAlign: 'center' }}
                      />
                    </div>

                    {/* Live count preview */}
                    {mom && (
                      <div style={{ fontSize: '0.73rem', color: 'hsl(var(--pencil))', marginBottom: '10px', fontFamily: 'Inter, sans-serif' }}>
                        {(() => {
                          const longCount = mom.points_discussed.filter(p => normalizePoint(p).split(/\s+/).filter(Boolean).length > summarizeThreshold).length
                          const totalCount = mom.points_discussed.length
                          return longCount > 0
                            ? <><span style={{ fontWeight: 600, color: 'hsl(var(--accent))' }}>{longCount}</span> of {totalCount} point{totalCount !== 1 ? 's' : ''} exceed {summarizeThreshold} words and will be condensed.</>
                            : <>0 of {totalCount} point{totalCount !== 1 ? 's' : ''} exceed {summarizeThreshold} words (all points are below threshold).</>
                        })()}
                      </div>
                    )}

                    {summarizeError && (
                      <div style={{ fontSize: '0.78rem', color: 'hsl(0,75%,55%)', background: 'hsl(0,75%,55%/.08)', border: '1px solid hsl(0,75%,55%/.2)', borderRadius: 6, padding: '0.45rem 0.7rem', marginBottom: '10px', fontFamily: 'Inter, sans-serif' }}>
                        <AlertTriangle size={12} style={{ display: 'inline', marginRight: '5px' }} />{summarizeError}
                      </div>
                    )}

                    {summarizeInfo && (
                      <div style={{ fontSize: '0.78rem', color: 'hsl(var(--success))', background: 'hsl(var(--success) / .08)', border: '1px solid hsl(var(--success) / .25)', borderRadius: 6, padding: '0.45rem 0.7rem', marginBottom: '10px', fontFamily: 'Inter, sans-serif' }}>
                        <CheckCircle size={12} style={{ display: 'inline', marginRight: '5px' }} />{summarizeInfo}
                      </div>
                    )}

                    <div style={{ display: 'flex', gap: '8px' }}>
                      <button
                        className="btn btn-ghost"
                        onClick={() => { setSummarizeLongOpen(false); setSummarizeError(null); setSummarizeInfo(null); setSummarizeStatus('idle') }}
                        style={{ fontSize: '0.82rem', padding: '0.35rem 0.8rem' }}
                        disabled={summarizeStatus === 'processing'}
                      >
                        Close
                      </button>
                      <button
                        id="mom-summarize-long-points-btn"
                        className="btn btn-primary"
                        onClick={handleSummarizeLongPoints}
                        disabled={summarizeStatus === 'processing'}
                        style={{ fontSize: '0.82rem', padding: '0.35rem 1rem', gap: '6px' }}
                      >
                        {summarizeStatus === 'processing'
                          ? <><Loader size={13} className="spin" /> Condensing long points...</>
                          : <><AlignLeft size={13} /> Summarize Long Points</>}
                      </button>
                    </div>
                  </div>
                )}
                <EditableList
                  items={mom.points_discussed.length > 0 ? mom.points_discussed.map(pt => normalizePoint(pt)) : ['', '', '']}
                  onChange={items => update('points_discussed', items)}
                  placeholder="e.g. Budget Review: The committee reviewed the Q3 budget allocations and approved..."
                  ordered
                  minItems={3}
                />
              </MomSection>

              {/* ── 8. Action Points ─────────────────────────────── */}
              <MomSection title="8. Action Points">
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: regenActionOpen ? '6px' : '10px' }}>
                  <p style={{ fontSize: '0.78rem', color: 'hsl(var(--pencil))', fontFamily: 'Inter, sans-serif', lineHeight: 1.5 }}>
                    General and speaker-based action items. Set <strong>Owner</strong> to a speaker name for speaker-based items.
                  </p>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flexShrink: 0, marginLeft: '12px' }}>
                    <button
                      className="btn btn-ghost"
                      onClick={() => {
                        setRegenActionOpen(v => !v)
                        setRegenActionError(null)
                        if (regenActionStatus === 'done' || regenActionStatus === 'error') setRegenActionStatus('idle')
                      }}
                      style={{ fontSize: '0.78rem', padding: '0.3rem 0.7rem', gap: '5px' }}
                      title="Regenerate Action Points from transcript"
                    >
                      <RefreshCw size={13} style={{ color: 'hsl(160,70%,42%)' }} /> Regenerate
                    </button>
                    <button
                      className="btn btn-ghost"
                      onClick={() => openRewrite('action_items')}
                      style={{ fontSize: '0.78rem', padding: '0.3rem 0.7rem', gap: '5px' }}
                      title="Rewrite Action Points with AI"
                    >
                      <Wand2 size={13} style={{ color: 'hsl(var(--accent))' }} /> Rewrite
                    </button>
                  </div>
                </div>

                {/* ── Regenerate panel ─────────────────────────── */}
                {regenActionOpen && (
                  <div style={{
                    marginBottom: '12px',
                    padding: '0.9rem 1rem',
                    borderRadius: '10px',
                    border: '1.5px solid hsl(160,70%,42%/.3)',
                    background: 'hsl(160,70%,42%/.06)',
                  }}>
                    <div style={{ fontSize: '0.8rem', fontWeight: 700, color: 'hsl(160,60%,35%)', marginBottom: '8px', fontFamily: 'Inter, sans-serif' }}>
                      Regenerate Action Points from Transcript
                    </div>
                    <p style={{ fontSize: '0.75rem', color: 'hsl(var(--pencil))', lineHeight: 1.5, marginBottom: '12px', fontFamily: 'Inter, sans-serif' }}>
                      Extracts action items directly from the original meeting transcript using a dedicated LLM call.
                      Existing action points will be replaced. All other MoM sections remain unchanged.
                    </p>

                    {/* Window size control */}
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '12px' }}>
                      <label style={{ fontSize: '0.75rem', fontWeight: 600, color: 'hsl(var(--pencil))', fontFamily: 'Inter, sans-serif', whiteSpace: 'nowrap' }}>
                        Window size
                      </label>
                      <input
                        type="range" min={1} max={60} step={1}
                        value={regenWindowMinutes}
                        onChange={e => setRegenWindowMinutes(Number(e.target.value))}
                        style={{ flex: 1, accentColor: 'hsl(160,70%,42%)' }}
                      />
                      <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '0.88rem', fontWeight: 700, minWidth: 44, color: 'hsl(160,60%,35%)' }}>
                        {regenWindowMinutes} min
                      </span>
                      <input
                        type="number" min={1} max={60}
                        value={regenWindowMinutes}
                        onChange={e => setRegenWindowMinutes(Math.max(1, Math.min(60, Number(e.target.value))))}
                        style={{ width: 56, padding: '0.3rem 0.5rem', fontSize: '0.82rem', borderRadius: 6, border: '1.5px solid hsl(var(--border)/.6)', background: 'hsl(var(--surface))', color: 'hsl(var(--ink))', fontFamily: 'JetBrains Mono, monospace', textAlign: 'center' }}
                      />
                    </div>

                    {regenActionError && (
                      <div style={{ fontSize: '0.78rem', color: 'hsl(0,75%,55%)', background: 'hsl(0,75%,55%/.08)', border: '1px solid hsl(0,75%,55%/.2)', borderRadius: 6, padding: '0.45rem 0.7rem', marginBottom: '10px', fontFamily: 'Inter, sans-serif' }}>
                        <AlertTriangle size={12} style={{ display: 'inline', marginRight: '5px' }} />{regenActionError}
                      </div>
                    )}

                    <div style={{ display: 'flex', gap: '8px' }}>
                      <button
                        className="btn btn-ghost"
                        onClick={() => { setRegenActionOpen(false); setRegenActionError(null); setRegenActionStatus('idle') }}
                        style={{ fontSize: '0.82rem', padding: '0.35rem 0.8rem' }}
                        disabled={regenActionStatus === 'processing'}
                      >
                        Cancel
                      </button>
                      <button
                        id="mom-regen-action-points-btn"
                        className="btn btn-primary"
                        onClick={handleRegenerateActionPoints}
                        disabled={regenActionStatus === 'processing'}
                        style={{ fontSize: '0.82rem', padding: '0.35rem 1rem', gap: '6px', background: 'hsl(160,70%,38%)', borderColor: 'hsl(160,70%,38%)' }}
                      >
                        {regenActionStatus === 'processing'
                          ? <><Loader size={13} className="spin" /> Extracting from transcript...</>
                          : <><RefreshCw size={13} /> Regenerate from Transcript</>}
                      </button>
                      {regenActionStatus === 'processing' && (
                        <span style={{ fontSize: '0.73rem', color: 'hsl(var(--pencil))', alignSelf: 'center', fontFamily: 'Inter, sans-serif' }}>
                          ~{regenWindowMinutes} min windows · this may take a moment
                        </span>
                      )}
                    </div>
                  </div>
                )}

                <ActionPointsSection items={mom.action_items} onChange={items => update('action_items', items)} />
              </MomSection>

              {/* ── 9. Conclusion ────────────────────────────────── */}
              <MomSection title="9. Conclusion">
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
                  <p style={{ fontSize: '0.78rem', color: 'hsl(var(--pencil))', fontFamily: 'Inter, sans-serif', lineHeight: 1.5 }}>
                    Meeting outcomes and conclusions.
                  </p>
                  <button
                    className="btn btn-ghost"
                    onClick={() => openRewrite('conclusion')}
                    style={{ fontSize: '0.78rem', padding: '0.3rem 0.7rem', gap: '5px', flexShrink: 0, marginLeft: '12px' }}
                    title="Rewrite Conclusion with AI"
                  >
                    <Wand2 size={13} style={{ color: 'hsl(var(--accent))' }} /> Rewrite
                  </button>
                </div>
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
                {versions.map((v, idx) => {
                  const vAny = v as any
                  const label: string | undefined = vAny.label
                  const isRewrite = label?.startsWith('Rewrite:')
                  const isManual = label === 'Manual Save' || label === 'User Saved'
                  return (
                    <div key={idx} style={{ borderRadius: '10px', padding: '0.75rem 1rem', border: idx === 0 ? '1.5px solid hsl(var(--sticky-yellow) / .5)' : isRewrite ? '1.5px solid hsl(var(--accent) / .35)' : isManual ? '1.5px solid hsl(var(--success) / .35)' : '1px solid hsl(var(--border) / .3)', background: idx === 0 ? 'hsl(var(--sticky-yellow) / .08)' : isRewrite ? 'hsl(var(--accent) / .05)' : isManual ? 'hsl(var(--success) / .05)' : 'hsl(var(--paper) / .5)', marginBottom: '8px' }}>
                      <div style={{ fontWeight: 700, fontSize: '0.85rem', color: 'hsl(var(--ink))', fontFamily: 'Inter, sans-serif', marginBottom: '4px', display: 'flex', alignItems: 'center', gap: '6px', flexWrap: 'wrap' }}>
                        {idx === 0 && <span style={{ fontSize: '.72rem', background: 'hsl(var(--sticky-yellow))', color: 'hsl(var(--ink))', padding: '.08rem .4rem', borderRadius: '4px', fontWeight: 700 }}>* Original</span>}
                        {isRewrite && <span style={{ fontSize: '.72rem', background: 'hsl(var(--accent) / .15)', color: 'hsl(var(--accent))', padding: '.08rem .4rem', borderRadius: '4px', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '3px' }}><Wand2 size={9} />{label}</span>}
                        {isManual && <span style={{ fontSize: '.72rem', background: 'hsl(var(--success) / .15)', color: 'hsl(var(--success))', padding: '.08rem .4rem', borderRadius: '4px', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '3px' }}><Save size={9} />{label}</span>}
                        {idx === 0 ? 'AI Generated' : (isRewrite || isManual) ? '' : `Version ${v.version}`}
                      </div>
                      <div style={{ fontSize: '0.74rem', color: 'hsl(var(--pencil))', fontFamily: 'JetBrains Mono, monospace', marginBottom: '10px' }}>
                        {new Date(v.saved_at).toLocaleString()}
                      </div>
                      <button onClick={() => restoreVersion(v)} className="btn btn-ghost" style={{ fontSize: '0.78rem', padding: '0.3rem 0.7rem', gap: '5px', width: '100%' }}>
                        <RotateCcw size={12} /> Restore this version
                      </button>
                    </div>
                  )
                })}
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

      {/* ── Rewrite Modal ──────────────────────────────────────────── */}
      {rewriteTarget && (
        <div style={{
          position: 'fixed', inset: 0, zIndex: 9999,
          background: 'rgba(0,0,0,0.55)', backdropFilter: 'blur(4px)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          padding: '1rem',
        }}>
          <div style={{
            width: '100%', maxWidth: '620px', background: 'hsl(var(--card))',
            borderRadius: '18px', border: '1px solid hsl(var(--border) / .3)',
            boxShadow: '0 24px 64px rgba(0,0,0,0.4)',
            overflow: 'hidden', display: 'flex', flexDirection: 'column',
            maxHeight: '90vh',
          }}>
            {/* Modal header */}
            <div style={{
              padding: '1.25rem 1.5rem', borderBottom: '1px solid hsl(var(--border) / .2)',
              display: 'flex', alignItems: 'center', gap: '10px', background: 'hsl(var(--paper) / .5)',
            }}>
              <div style={{ width: '34px', height: '34px', borderRadius: '10px', background: 'hsl(var(--accent) / .12)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <Wand2 size={17} style={{ color: 'hsl(var(--accent))' }} />
              </div>
              <div style={{ flex: 1 }}>
                <h2 style={{ fontSize: '0.95rem', fontWeight: 700, color: 'hsl(var(--ink))', fontFamily: 'Inter, sans-serif', margin: 0 }}>
                  Rewrite {
                    rewriteTarget === 'discussion_points' ? 'Discussion Points'
                      : rewriteTarget === 'action_items' ? 'Action Points'
                        : rewriteTarget === 'introduction' ? 'Introduction'
                          : 'Conclusion'
                  }
                </h2>
                <p style={{ fontSize: '0.75rem', color: 'hsl(var(--pencil))', fontFamily: 'Inter, sans-serif', margin: 0, marginTop: '2px' }}>
                  {rewritePhase === 'upload' && 'Upload reference documents to extract writing style'}
                  {rewritePhase === 'analyzing' && 'Analyzing writing style from documents...'}
                  {rewritePhase === 'rules' && 'Review & edit style rules, then rewrite'}
                  {rewritePhase === 'rewriting' && 'Rewriting content with AI...'}
                  {rewritePhase === 'done' && 'Rewrite complete! Content updated.'}
                </p>
              </div>
              {rewritePhase !== 'analyzing' && rewritePhase !== 'rewriting' && (
                <button className="icon-btn" onClick={closeRewrite} style={{ color: 'hsl(var(--pencil))' }}><X size={16} /></button>
              )}
            </div>

            {/* Modal body */}
            <div style={{ flex: 1, overflowY: 'auto', padding: '1.5rem' }}>

              {/* Phase: Upload (for discussion_points / action_items) */}
              {rewritePhase === 'upload' && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                  <p style={{ fontSize: '0.85rem', color: 'hsl(var(--ink))', fontFamily: 'Inter, sans-serif', lineHeight: 1.6 }}>
                    Upload one or more reference MoM documents. The AI will analyze how {
                      rewriteTarget === 'discussion_points' ? 'discussion points are written' : 'action items are structured'
                    } in those documents and generate style rules you can review before rewriting.
                  </p>

                  {/* Hidden file input */}
                  <input
                    ref={rewriteFileInputRef}
                    type="file"
                    multiple
                    accept=".pdf,.docx,.pptx,.txt,.md,.png,.jpg,.jpeg,.webp,.xlsx,.xls,.csv"
                    style={{ display: 'none' }}
                    onChange={e => {
                      const files = Array.from(e.target.files || [])
                      setRewriteFiles(prev => [...prev, ...files])
                      e.target.value = ''
                    }}
                  />

                  {/* Drop zone */}
                  <div
                    onClick={() => rewriteFileInputRef.current?.click()}
                    style={{
                      border: '2px dashed hsl(var(--accent) / .35)', borderRadius: '12px',
                      padding: '2rem', textAlign: 'center', cursor: 'pointer',
                      background: 'hsl(var(--accent) / .03)', transition: 'all 0.2s',
                    }}
                    onMouseOver={e => { e.currentTarget.style.borderColor = 'hsl(var(--accent))'; e.currentTarget.style.background = 'hsl(var(--accent) / .06)' }}
                    onMouseOut={e => { e.currentTarget.style.borderColor = 'hsl(var(--accent) / .35)'; e.currentTarget.style.background = 'hsl(var(--accent) / .03)' }}
                  >
                    <Upload size={28} style={{ color: 'hsl(var(--accent))', marginBottom: '8px' }} />
                    <p style={{ fontSize: '0.88rem', fontWeight: 600, color: 'hsl(var(--ink))', fontFamily: 'Inter, sans-serif', marginBottom: '4px' }}>Click to select reference documents</p>
                    <p style={{ fontSize: '0.75rem', color: 'hsl(var(--pencil))', fontFamily: 'Inter, sans-serif' }}>PDF, DOCX, PPTX, TXT, MD, Images (max 30 MB each)</p>
                  </div>

                  {/* Selected files */}
                  {rewriteFiles.length > 0 && (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                      {rewriteFiles.map((f, i) => (
                        <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '6px 10px', borderRadius: '8px', background: 'hsl(var(--muted) / .4)', fontSize: '0.82rem', fontFamily: 'Inter, sans-serif' }}>
                          <FileText size={13} style={{ color: 'hsl(var(--accent))' }} />
                          <span style={{ flex: 1, color: 'hsl(var(--ink))', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{f.name}</span>
                          <span style={{ color: 'hsl(var(--pencil))', fontSize: '0.72rem' }}>{(f.size / 1024 / 1024).toFixed(2)} MB</span>
                          <button
                            onClick={() => setRewriteFiles(prev => prev.filter((_, j) => j !== i))}
                            style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'hsl(var(--destructive))', padding: '2px' }}
                          >
                            <X size={13} />
                          </button>
                        </div>
                      ))}
                    </div>
                  )}

                  {rewriteError && (
                    <div style={{ padding: '0.65rem 1rem', borderRadius: '8px', background: 'hsl(var(--destructive) / .1)', border: '1px solid hsl(var(--destructive) / .3)', color: 'hsl(var(--destructive))', fontSize: '0.82rem', fontFamily: 'Inter, sans-serif' }}>
                      <AlertTriangle size={13} style={{ display: 'inline', marginRight: '6px' }} />{rewriteError}
                    </div>
                  )}

                  <div style={{ display: 'flex', gap: '10px', justifyContent: 'flex-end', marginTop: '0.5rem' }}>
                    <button className="btn btn-ghost" onClick={closeRewrite} style={{ fontSize: '0.85rem' }}>Cancel</button>
                    <button
                      className="btn btn-ghost"
                      onClick={() => { setRewriteFiles([]); setRewriteRules(rewriteTarget === 'discussion_points' ? '1. Merge discussion points covering the same or highly similar topics into a single coherent point without losing any factual information, decisions, questions, or outcomes.' : ''); setRewritePhase('rules') }}
                      style={{ fontSize: '0.85rem' }}
                    >
                      Skip — Enter Rules Manually
                    </button>
                    <button
                      className="btn btn-primary"
                      disabled={rewriteFiles.length === 0}
                      onClick={handleAnalyzeStyle}
                      style={{ fontSize: '0.85rem', gap: '7px' }}
                    >
                      <Sparkles size={14} /> Analyze Style
                    </button>
                  </div>
                </div>
              )}

              {/* Phase: Analyzing */}
              {rewritePhase === 'analyzing' && (
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1.5rem', padding: '2rem 0' }}>
                  <div style={{ width: '72px', height: '72px', borderRadius: '50%', background: 'hsl(var(--accent) / .08)', border: '3px solid hsl(var(--accent) / .2)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <Loader size={30} className="spin" style={{ color: 'hsl(var(--accent))' }} />
                  </div>
                  <div style={{ textAlign: 'center' }}>
                    <p style={{ fontSize: '1rem', fontWeight: 600, color: 'hsl(var(--ink))', fontFamily: 'Inter, sans-serif', marginBottom: '6px' }}>Analyzing Writing Style</p>
                    <p style={{ fontSize: '0.85rem', color: 'hsl(var(--pencil))', fontFamily: 'Inter, sans-serif' }}>Extracting formatting and style rules from your reference documents...</p>
                  </div>
                </div>
              )}

              {/* Phase: Rules */}
              {rewritePhase === 'rules' && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>

                  {/* Style Rules */}
                  <div>
                    <label style={{ fontSize: '0.78rem', fontWeight: 700, color: 'hsl(var(--ink))', fontFamily: 'Inter, sans-serif', display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                      <Settings2 size={13} style={{ color: 'hsl(var(--accent))' }} />
                      Style Rules
                      <span style={{ fontSize: '0.68rem', fontWeight: 400, color: 'hsl(var(--pencil))', textTransform: 'none', letterSpacing: 0 }}>(editable — the AI will follow these when rewriting)</span>
                    </label>
                    <textarea
                      className="input"
                      value={rewriteRules}
                      onChange={e => setRewriteRules(e.target.value)}
                      placeholder="Enter style rules (one per line or numbered list)..."
                      rows={8}
                      style={{ width: '100%', resize: 'vertical', padding: '0.75rem', fontSize: '0.85rem', lineHeight: 1.7, fontFamily: 'Inter, sans-serif' }}
                    />
                  </div>

                  {/* Custom prompt */}
                  <div>
                    <label style={{ fontSize: '0.78rem', fontWeight: 700, color: 'hsl(var(--ink))', fontFamily: 'Inter, sans-serif', display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                      <Sparkles size={13} style={{ color: 'hsl(var(--accent))' }} />
                      Additional Instructions
                      <span style={{ fontSize: '0.68rem', fontWeight: 400, color: 'hsl(var(--pencil))', textTransform: 'none', letterSpacing: 0 }}>(optional)</span>
                    </label>
                    <textarea
                      className="input"
                      value={rewriteCustomPrompt}
                      onChange={e => setRewriteCustomPrompt(e.target.value)}
                      placeholder="e.g. Keep it concise. Use formal language. Start each point with an action verb..."
                      rows={3}
                      style={{ width: '100%', resize: 'vertical', padding: '0.75rem', fontSize: '0.85rem', lineHeight: 1.7, fontFamily: 'Inter, sans-serif' }}
                    />
                  </div>

                  {/* Window size (list types only) */}
                  {(rewriteTarget === 'discussion_points' || rewriteTarget === 'action_items') && (
                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                      <label style={{ fontSize: '0.78rem', fontWeight: 700, color: 'hsl(var(--ink))', fontFamily: 'Inter, sans-serif', display: 'flex', alignItems: 'center', gap: '6px', textTransform: 'uppercase', letterSpacing: '0.04em', flexShrink: 0 }}>
                        <SlidersHorizontal size={13} style={{ color: 'hsl(var(--pencil))' }} />
                        Batch Size
                      </label>
                      <input
                        type="number"
                        className="input"
                        value={rewriteWindowSize}
                        onChange={e => setRewriteWindowSize(Math.max(1, Math.min(200, parseInt(e.target.value) || 20)))}
                        min={1} max={200}
                        style={{ width: '80px', padding: '0.4rem 0.6rem', fontSize: '0.85rem' }}
                      />
                      <span style={{ fontSize: '0.75rem', color: 'hsl(var(--pencil))', fontFamily: 'Inter, sans-serif' }}>
                        items per AI request (default 20)
                      </span>
                    </div>
                  )}

                  {rewriteError && (
                    <div style={{ padding: '0.65rem 1rem', borderRadius: '8px', background: 'hsl(var(--destructive) / .1)', border: '1px solid hsl(var(--destructive) / .3)', color: 'hsl(var(--destructive))', fontSize: '0.82rem', fontFamily: 'Inter, sans-serif' }}>
                      <AlertTriangle size={13} style={{ display: 'inline', marginRight: '6px' }} />{rewriteError}
                    </div>
                  )}

                  <div style={{ display: 'flex', gap: '10px', justifyContent: 'flex-end' }}>
                    {(rewriteTarget === 'discussion_points' || rewriteTarget === 'action_items') && (
                      <button className="btn btn-ghost" onClick={() => setRewritePhase('upload')} style={{ fontSize: '0.85rem' }}>← Back</button>
                    )}
                    <button className="btn btn-ghost" onClick={closeRewrite} style={{ fontSize: '0.85rem' }}>Cancel</button>
                    <button
                      className="btn btn-primary"
                      onClick={handleRewrite}
                      style={{ fontSize: '0.85rem', gap: '7px' }}
                    >
                      <Wand2 size={14} /> Rewrite Now
                    </button>
                  </div>
                </div>
              )}

              {/* Phase: Rewriting */}
              {rewritePhase === 'rewriting' && (
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1.75rem', padding: '2rem 0' }}>
                  <div style={{ width: '72px', height: '72px', borderRadius: '50%', background: 'hsl(var(--accent) / .08)', border: '3px solid hsl(var(--accent) / .2)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <Wand2 size={30} style={{ color: 'hsl(var(--accent))' }} className="spin" />
                  </div>
                  <div style={{ textAlign: 'center', width: '100%' }}>
                    <p style={{ fontSize: '1rem', fontWeight: 600, color: 'hsl(var(--ink))', fontFamily: 'Inter, sans-serif', marginBottom: '6px' }}>Rewriting with AI...</p>
                    <p style={{ fontSize: '0.85rem', color: 'hsl(var(--pencil))', fontFamily: 'Inter, sans-serif', marginBottom: '1rem' }}>
                      {rewriteProgress.total > 1
                        ? `Processing batch ${rewriteProgress.current} of ${rewriteProgress.total}...`
                        : 'Processing...'}
                    </p>
                    {rewriteProgress.total > 0 && (
                      <div style={{ width: '100%', background: 'hsl(var(--muted))', borderRadius: '99px', height: '6px', overflow: 'hidden' }}>
                        <div style={{
                          height: '100%',
                          width: `${Math.round((rewriteProgress.current / Math.max(rewriteProgress.total, 1)) * 100)}%`,
                          background: 'hsl(var(--accent))',
                          borderRadius: '99px',
                          transition: 'width 0.4s ease',
                        }} />
                      </div>
                    )}
                  </div>
                  <p style={{ fontSize: '0.75rem', color: 'hsl(var(--pencil))', fontFamily: 'Inter, sans-serif' }}>Please wait — do not close this window</p>
                </div>
              )}

              {/* Phase: Done */}
              {rewritePhase === 'done' && (
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1.25rem', padding: '2rem 0' }}>
                  <div style={{ width: '72px', height: '72px', borderRadius: '50%', background: 'hsl(var(--success) / .12)', border: '3px solid hsl(var(--success) / .3)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <CheckCircle size={30} style={{ color: 'hsl(var(--success))' }} />
                  </div>
                  <div style={{ textAlign: 'center' }}>
                    <p style={{ fontSize: '1rem', fontWeight: 700, color: 'hsl(var(--ink))', fontFamily: 'Inter, sans-serif', marginBottom: '6px' }}>Rewrite Complete!</p>
                    <p style={{ fontSize: '0.85rem', color: 'hsl(var(--pencil))', fontFamily: 'Inter, sans-serif', lineHeight: 1.6 }}>
                      The content has been updated. A new version has been saved to History — use the History panel to revert if needed.
                    </p>
                  </div>
                  <button className="btn btn-primary" onClick={closeRewrite} style={{ fontSize: '0.85rem', gap: '7px' }}>
                    <CheckCircle size={14} /> Done
                  </button>
                </div>
              )}

            </div>
          </div>
        </div>
      )}

    </div>
  )
}
