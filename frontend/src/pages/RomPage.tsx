import { useState, useCallback, useEffect, useRef } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import {
  ArrowLeft, Sparkles, Download, FileText, Clock, User,
  Loader, Pencil, Upload, Save, X, Play, Plus, Brain,
  Target, List, RefreshCw, FileDown, Layers, Video,
  ChevronUp, ChevronDown, ArrowRightLeft, Tag, Trash2, Sliders,
  RotateCcw, WandSparkles, FileUp
} from 'lucide-react'
import api from '../api/client'
import { toast } from 'sonner'
import { getApiErrorDetail } from '../lib/errors'

// ── Types ─────────────────────────────────────────────────────────────────────

interface TranscriptSegment {
  speaker_label: string
  start: number
  end: number
  text: string
}

interface RecordingDetail {
  id: string
  filename: string
  duration: number
  status: string
  speakers_detected: string[]
  created_at: string
  transcript?: TranscriptSegment[]
}

interface DiscussionPoint {
  id: string
  discussion_point: string
  timeline_start: number
  timeline_end: number
  speakers: string[]
  action_owner: string | null
  decisions: string[]
  questions: string[]
  technical_terms: string[]
  dates: string[]
  numbers: string[]
  project_names: string[]
  action_items: string[]
  references: string[]
  required_information: string[]
  window_index: number
  raw_transcript_text?: string
  video_transcript_context?: string
}

const formatItemText = (item: any): string => {
  if (item === null || item === undefined) return ''
  if (typeof item === 'string') return item.trim()
  if (typeof item === 'number' || typeof item === 'boolean') return String(item)
  if (typeof item === 'object') {
    const task = item.task || item.description || item.item || item.text || item.action || item.decision || item.point
    const owner = item.assignee || item.owner || item.assigner
    const deadline = item.deadline || item.due || item.date
    const cond = item.conditions || item.condition

    const parts: string[] = []
    if (task) parts.push(String(task).trim())
    if (owner) parts.push(`(Owner: ${String(owner).trim()})`)
    if (deadline) parts.push(`[Due: ${String(deadline).trim()}]`)
    if (cond) parts.push(`(If: ${String(cond).trim()})`)

    if (parts.length > 0) return parts.join(' ')
    try {
      return JSON.stringify(item)
    } catch {
      return String(item)
    }
  }
  return String(item)
}

const cleanCalendarDates = (datesList: any): string[] => {
  if (!datesList) return []
  const list = Array.isArray(datesList) ? datesList : [datesList]
  const clean: string[] = []

  for (const d of list) {
    const val = typeof d === 'object' && d !== null ? String(d.value || d.date || '').trim() : String(d || '').trim()
    if (!val) continue

    if (/^\d+(\.\d+)?\s*[\-–—]\s*\d+(\.\d+)?$/.test(val)) continue
    if (/^\d{1,2}:\d{2}(:\d{2})?\s*[\-–—]\s*\d{1,2}:\d{2}(:\d{2})?$/.test(val)) continue
    if (/^\d{4,}\.\d+$/.test(val) || /^\d+\.\d{2,}$/.test(val)) continue
    if (/\b(timeline|window|timestamp|seconds?|offset)\b/i.test(val)) continue

    const hasMonth = /\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|june|july|august|september|october|november|december)\b/i.test(val)
    const hasYear = /\b(19|20)\d{2}\b/.test(val)
    const hasDateFmt = /\b\d{1,4}[/\-.]\d{1,2}[/\-.]\d{1,4}\b/.test(val)
    const hasDaySpec = /\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|today|tomorrow|yesterday|next week|end of month|q[1-4])\b/i.test(val)

    if (hasMonth || hasYear || hasDateFmt || hasDaySpec || val.length >= 4) {
      if (!/^[\d\s.:\-–—]+$/.test(val) || hasDateFmt || hasYear) {
        clean.push(val)
      }
    }
  }

  return clean
}

const formatPrecisePointText = (pt: any): string => {
  const rawDecisions = pt.decisions || []
  const decList = (Array.isArray(rawDecisions) ? rawDecisions : [rawDecisions])
    .map((d: any) => formatItemText(d))
    .filter((d: string) => d && !['none', 'n/a', 'null'].includes(d.toLowerCase()))

  const rawActions = pt.action_items || []
  const actList = (Array.isArray(rawActions) ? rawActions : [rawActions])
    .map((a: any) => formatItemText(a))
    .filter((a: string) => a && !['none', 'n/a', 'null'].includes(a.toLowerCase()))

  const validDates = cleanCalendarDates(pt.dates)

  const parts: string[] = []
  if (decList.length) parts.push(...decList)
  if (actList.length) parts.push(...actList)

  if (!parts.length) {
    const ptText = formatItemText(pt.text || pt.polished_text || pt.discussion_point || '')
    if (ptText) parts.push(ptText)
  }

  if (!parts.length) return 'Discussion noted.'

  const mainText = parts.map(p => p.replace(/\.$/, '')).join('. ') + '.'

  if (validDates.length) {
    return `${mainText} (${validDates.join(', ')})`
  }
  return mainText
}

interface PolishedPoint {
  id: string
  original_point_id: string
  polished_text: string
  timeline_start: number
  timeline_end: number
  speakers: string[]
  action_owner: string | null
  decisions: string[]
  technical_terms: string[]
  dates: string[]
  numbers: string[]
  references: string[]
  action_items: string[]
  retrieved_context: {
    meeting_chunks: Array<{ text: string; score: number; filename: string }>
    global_chunks: Array<{ text: string; score: number; filename: string }>
  }
}

interface AgendaItem {
  id: string
  agenda_id: string
  title: string
  description: string
  keywords: string[]
  related_concepts: string[]
  alternative_terminology: string[]
  expected_themes: string[]
}

interface FinalRomAgenda {
  agenda_id: string
  title: string
  discussion_points: Array<{
    id: string
    text: string
    speaker: string
    action_owner: string | null
    timeline_start: number
    timeline_end: number
    references: string[]
    action_items: string[]
  }>
}

interface AgendaDocEntry {
  doc_name: string
  points: string[]
}

interface SpeakerMapping {
  speaker_id: string
  real_name: string
}

interface RomData {
  stage1: {
    status: string
    transcript_window_minutes: number
    discussion_points: DiscussionPoint[]
    windows_processed: number
    completed_at: string | null
    video_ocr_blocks_used?: number
    source_type?: string
  } | null
  stage2: {
    status: string
    polished_points: PolishedPoint[]
    completed_at: string | null
  } | null
  stage3: {
    status: string
    agendas: AgendaItem[]
    expanded_agendas: any[]
    similarity_matrix: number[][]
    point_mappings: Record<string, string>
    batch_assignments: any[]
    agenda_groups: Record<string, string[]>
    agenda_doc_points: Record<string, AgendaDocEntry>
    completed_at: string | null
  } | null
  final_rom: {
    agendas: FinalRomAgenda[]
    speaker_mappings: Record<string, string>
    last_edited_at: string | null
    include_agenda_doc_points: boolean
  } | null
}

type ProcessState = 'idle' | 'processing' | 'done' | 'error'

// ── Helpers ───────────────────────────────────────────────────────────────────

function fmtTime(secs: number): string {
  if (!secs || isNaN(secs)) return '0:00'
  const h = Math.floor(secs / 3600)
  const m = Math.floor((secs % 3600) / 60)
  const s = Math.floor(secs % 60)
  if (h > 0) return `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
}

function SectionHeader({ icon, label, count, color }: { icon: React.ReactNode; label: string; count?: number; color: string }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '.35rem .6rem', borderRadius: 7, background: `${color}12`, border: `1px solid ${color}28`, marginBottom: '.35rem' }}>
      <span style={{ color, display: 'flex' }}>{icon}</span>
      <span style={{ fontSize: '.71rem', fontWeight: 700, color, textTransform: 'uppercase', letterSpacing: '.04em' }}>{label}</span>
      {count !== undefined && <span style={{ fontSize: '.65rem', fontWeight: 600, padding: '1px 6px', borderRadius: 999, background: `${color}20`, border: `1px solid ${color}40`, color, marginLeft: 'auto' }}>{count}</span>}
    </div>
  )
}

function StatusBadge({ state }: { state: ProcessState }) {
  const configs = {
    idle: { color: 'hsl(var(--pencil))', bg: 'hsl(var(--pencil) / 0.12)', border: 'hsl(var(--pencil) / 0.25)', text: 'Ready' },
    processing: { color: 'hsl(280, 75%, 65%)', bg: 'hsl(280, 75%, 60% / 0.12)', border: 'hsl(280, 75%, 60% / 0.3)', text: 'Running' },
    done: { color: 'hsl(140, 70%, 45%)', bg: 'hsl(140, 70%, 45% / 0.12)', border: 'hsl(140, 70%, 45% / 0.3)', text: 'Done' },
    error: { color: 'hsl(0, 70%, 50%)', bg: 'hsl(0, 70%, 50% / 0.12)', border: 'hsl(0, 70%, 50% / 0.3)', text: 'Error' },
  }
  const c = configs[state] || configs.idle
  return (
    <div style={{
      display: 'inline-flex', alignItems: 'center', gap: 4,
      padding: '2px 7px', borderRadius: 10,
      background: c.bg, border: `1px solid ${c.border}`, color: c.color,
      fontSize: '0.68rem', fontWeight: 700, fontFamily: 'Inter, sans-serif'
    }}>
      {state === 'processing' ? <Loader size={10} className="spin" /> : <div style={{ width: 5, height: 5, borderRadius: '50%', background: c.color }} />}
      {c.text}
    </div>
  )
}

const BUILTIN_PROMPT_TEMPLATES = [
  {
    id: 'executive_polish',
    name: 'Executive Summary Polish',
    description: 'Polishes discussion points for C-suite executive reporting with high-level outcome bullets and strategic alignment.',
    prompt: `Enhance all agenda discussion points to be crisp, concise, and structured for C-suite executive review.\n- Focus on strategic decisions, financial/business impacts, and key milestones.\n- Use strong, active business verbs and clear bullet structures.\n- Keep technical jargon minimal unless essential to the decision context.\n- Ensure all action items clearly state the task, owner, and deadline.`
  },
  {
    id: 'technical_spec',
    name: 'Technical & Architectural Focus',
    description: 'Emphasizes technical specifications, architecture decisions, system impacts, and engineering tasks.',
    prompt: `Enhance discussion points with technical precision for engineering and product teams.\n- Highlight specific architectural choices, system components, API contracts, and technology stacks mentioned.\n- Explicitly document technical dependencies, performance/scaling metrics, and security/compliance considerations.\n- Ensure action items clearly define technical deliverables, owner, and implementation target.`
  },
  {
    id: 'action_oriented',
    name: 'Action-Oriented & Operational',
    description: 'Prioritizes operational deliverables, task owners, deadlines, and project execution milestones.',
    prompt: `Format all discussion points into clear, action-oriented operational meeting notes.\n- Highlight operational deliverables, task ownership, and target execution timelines.\n- Clearly separate key discussion takeaways from explicit follow-up commitments.\n- Ensure all action items are listed with complete descriptions, owners, and explicit target dates.`
  },
  {
    id: 'formal_governance',
    name: 'Formal Governance & Minutes',
    description: 'Structures notes into formal corporate governance meeting minutes with clear motions and approvals.',
    prompt: `Format the meeting discussion into formal corporate governance minutes style.\n- Maintain formal, objective, third-person corporate tone throughout.\n- Clearly record formal proposals, consensus reached, and official decisions made.\n- Ensure all task assignments and responsibilities are formally documented with assigned owners.`
  }
]

// ── Component ─────────────────────────────────────────────────────────────────

export default function RomPage() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()

  // Recording data
  const [recording, setRecording] = useState<RecordingDetail | null>(null)
  const [loadingRec, setLoadingRec] = useState(true)

  // ROM data from backend
  const [romData, setRomData] = useState<RomData | null>(null)

  // Active tab
  const [activeTab, setActiveTab] = useState<'stage1' | 'stage2' | 'stage3' | 'final' | 'advanced'>('stage1')

  // Advanced MoM controls
  const [advancedCustomPrompt, setAdvancedCustomPrompt] = useState('')
  const [advancedRegenTitle, setAdvancedRegenTitle] = useState(false)
  const [advancedRegenIntro, setAdvancedRegenIntro] = useState(false)
  const [advancedRegenConclusion, setAdvancedRegenConclusion] = useState(false)
  const [advancedStatus, setAdvancedStatus] = useState<ProcessState>('idle')
  const [selectedTemplateId, setSelectedTemplateId] = useState<string | null>(null)

  // Stage 1 controls
  const [transcriptWindow, setTranscriptWindow] = useState(2)
  const [stage1Status, setStage1Status] = useState<ProcessState>('idle')

  // Stage 2 controls
  const [meetingTopK, setMeetingTopK] = useState(5)
  const [globalTopK, setGlobalTopK] = useState(3)
  const [discussionWindowSize, setDiscussionWindowSize] = useState(5)
  const [stage2Status, setStage2Status] = useState<ProcessState>('idle')
  // Stage 2 meeting documents (uploaded inline, text extracted)
  const [stage2MeetingDocs, setStage2MeetingDocs] = useState<{ id: string; name: string }[]>([])
  const [uploadingStage2Doc, setUploadingStage2Doc] = useState(false)
  const stage2DocInputRef = useRef<HTMLInputElement>(null)

  // Stage 3 controls
  const [agendaText, setAgendaText] = useState('')
  const [stage3MeetingTopK, setStage3MeetingTopK] = useState(5)
  const [stage3GlobalTopK, setStage3GlobalTopK] = useState(3)
  const [stage3Status, setStage3Status] = useState<ProcessState>('idle')
  // Stage 3 – Previous MoM documents
  const [previousMomDocs, setPreviousMomDocs] = useState<{ name: string; text: string }[]>([])
  const [uploadingPrevMom, setUploadingPrevMom] = useState(false)
  const previousMomInputRef = useRef<HTMLInputElement>(null)
  // Stage 3 – Batch size for LLM assignment
  const [stage3BatchSize, setStage3BatchSize] = useState(20)
  // Previous MoM character limit option
  const [prevMomCharLimit, setPrevMomCharLimit] = useState(20000)

  // Stage 3 – split step statuses
  const [stage3AgendaStatus, setStage3AgendaStatus] = useState<ProcessState>('idle')
  const [stage3FinalStatus, setStage3FinalStatus] = useState<ProcessState>('idle')
  const [enhancedMomStatus, setEnhancedMomStatus] = useState<ProcessState>('idle')
  const [enhancedMomData, setEnhancedMomData] = useState<any>(null)

  // Stage 3 – include agenda doc points toggle
  const [includeAgendaDocPoints, setIncludeAgendaDocPoints] = useState(false)

  // Stage 3 – per-agenda supporting document upload tracking
  const [agendaDocUploading, setAgendaDocUploading] = useState<Record<string, boolean>>({})
  const agendaDocInputRefs = useRef<Record<string, HTMLInputElement | null>>({})

  // Speaker name mappings for Final ROM
  const [speakerMappings, setSpeakerMappings] = useState<SpeakerMapping[]>([])
  const [speakerMappingDraft, setSpeakerMappingDraft] = useState<Record<string, string>>({})
  const [showSpeakerMapping, setShowSpeakerMapping] = useState(false)

  // Agenda extraction state (using /raw-mom/{id}/agenda endpoint)
  const [extractedAgendaItems, setExtractedAgendaItems] = useState<{ topic: string; speaker: string | null; details?: string | null }[]>([])
  const [extractingAgenda, setExtractingAgenda] = useState(false)
  const [agendaExtractSource, setAgendaExtractSource] = useState<string>('')

  // Agenda file upload (for text extraction to populate agendaText)
  const [agendaUploadedFiles, setAgendaUploadedFiles] = useState<{ name: string; text: string }[]>([])
  const [uploadingAgendaFile, setUploadingAgendaFile] = useState(false)
  const agendaFileInputRef = useRef<HTMLInputElement>(null)

  const forceReparseRef = useRef(false)

  // Editing state for Stage 3 Agendas
  const [editingAgendaId, setEditingAgendaId] = useState<string | null>(null)
  const [editAgendaTitle, setEditAgendaTitle] = useState('')
  const [editAgendaDescription, setEditAgendaDescription] = useState('')
  const [savingAgendas, setSavingAgendas] = useState(false)

  // Editing state for final ROM
  const [editingPointId, setEditingPointId] = useState<string | null>(null)
  const [editDraft, setEditDraft] = useState('')
  const [savingRom, setSavingRom] = useState(false)
  const [romViewMode, setRomViewMode] = useState<'standard' | 'precise'>('standard')

  // Rewrite ROM state
  const DEFAULT_REWRITE_INSTRUCTION = 'Rewrite the ROM in a formal, professional writing style. Improve grammar, sentence structure, readability, and formatting only. Do not change, add, remove, or reinterpret any facts, discussion points, decisions, action items, speakers, or context. Preserve the exact meaning and structure while presenting it in polished formal language.'
  const [originalFinalRom, setOriginalFinalRom] = useState<any>(null)
  const [rewriteInstruction, setRewriteInstruction] = useState(DEFAULT_REWRITE_INSTRUCTION)
  const [rewriteStatus, setRewriteStatus] = useState<'idle' | 'processing' | 'done' | 'error'>('idle')
  const [isRewritten, setIsRewritten] = useState(false)
  const [showRewritePanel, setShowRewritePanel] = useState(false)
  // Rewrite mode controls
  const [rewriteMode, setRewriteMode] = useState<'window' | 'complete' | 'reference'>('window')
  const [rewriteWindowSize, setRewriteWindowSize] = useState(3)
  const [writingRules, setWritingRules] = useState('')
  const [extractingRules, setExtractingRules] = useState(false)
  const referenceFileInputRef = useRef<HTMLInputElement>(null)

  // Save edited Stage 3 Agendas to backend
  const saveStage3Agendas = async (updatedAgendas: any[]) => {
    if (!id) return
    setSavingAgendas(true)
    try {
      await api.put(`/rom/${id}/stage3/agendas`, { agendas: updatedAgendas })
      setRomData(prev => {
        if (!prev?.stage3) return prev
        return {
          ...prev,
          stage3: { ...prev.stage3, agendas: updatedAgendas }
        }
      })
      toast.success('Agenda updated')
    } catch (e) {
      toast.error(getApiErrorDetail(e) || 'Failed to save agenda')
    } finally {
      setSavingAgendas(false)
    }
  }

  // Complete ROM Orchestration state
  const [isGeneratingAll, setIsGeneratingAll] = useState(false)
  const [generateAllStep, setGenerateAllStep] = useState<string | null>(null)

  // File text extraction helper (shared)
  const extractFileText = async (file: File): Promise<string> => {
    const formData = new FormData()
    formData.append('file', file)
    const res = await api.post(`/raw-mom/${id}/extract-file-text`, formData)
    return res.data.text
  }

  // Handle uploading Previous MoM files for Stage 3 Phase 1
  const handlePreviousMomUpload = async (files: FileList | null) => {
    if (!files?.length || !id) return

    const maxFiles = 5
    if (previousMomDocs.length + files.length > maxFiles) {
      toast.error(`Maximum of ${maxFiles} previous MoM files can be uploaded.`)
      if (previousMomInputRef.current) previousMomInputRef.current.value = ''
      return
    }

    const maxSizeBytes = 10 * 1024 * 1024 // 10MB
    setUploadingPrevMom(true)
    for (const file of Array.from(files)) {
      if (file.size > maxSizeBytes) {
        toast.error(`File ${file.name} exceeds the 10MB size limit.`)
        continue
      }

      try {
        const formData = new FormData()
        formData.append('file', file)
        const res = await api.post(`/rom/${id}/stage3/upload-previous-mom`, formData)
        const momText = res.data.mom_text || ''
        if (momText) {
          setPreviousMomDocs(prev => [...prev, { name: file.name, text: momText }])
          toast.success(`Added previous MoM: ${file.name}`)
        }
      } catch (err) {
        toast.error(getApiErrorDetail(err) || `Failed to upload ${file.name}`)
      }
    }
    setUploadingPrevMom(false)
    if (previousMomInputRef.current) previousMomInputRef.current.value = ''
  }

  // Handle uploading a file for the agenda section
  const handleAgendaFileUpload = async (files: FileList | null) => {
    if (!files?.length || !id) return
    setUploadingAgendaFile(true)
    const results: { name: string; text: string }[] = []
    for (const file of Array.from(files)) {
      try {
        const text = await extractFileText(file)
        results.push({ name: file.name, text })
        toast.success(`Extracted text from ${file.name}`)
      } catch (err) {
        toast.error(getApiErrorDetail(err) || `Failed to extract ${file.name}`)
      }
    }
    if (results.length) {
      const combined = results.map(r => r.text).join('\n\n')
      setAgendaText(prev => prev ? prev + '\n\n' + combined : combined)
      setAgendaUploadedFiles(prev => [...prev, ...results])
    }
    setUploadingAgendaFile(false)
  }

  // Fetch attachments list from backend
  const fetchStage2Attachments = useCallback(async () => {
    if (!id) return
    try {
      const attRes = await api.get(`/attachments/${id}`)
      const contextFiles = (attRes.data.files || []).filter((f: any) => f.type === 'context')
      setStage2MeetingDocs(contextFiles.map((f: any) => ({ id: f.id, name: f.filename })))
    } catch (e) {
      console.error('Failed to fetch meeting context attachments', e)
    }
  }, [id])

  // Handle uploading meeting docs for Stage 2
  const handleStage2DocUpload = async (files: FileList | null) => {
    if (!files?.length || !id) return
    setUploadingStage2Doc(true)
    try {
      const formData = new FormData()
      formData.append('type', 'context')
      for (const file of Array.from(files)) {
        formData.append('files', file)
      }
      // 1. Upload files
      await api.post(`/attachments/${id}/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })
      toast.success('Uploaded files. Indexing into vector store...')

      // 2. Process/index files
      const processData = new FormData()
      processData.append('type', 'context')
      await api.post(`/attachments/${id}/process`, processData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })
      toast.success('Successfully indexed meeting context')

      // 3. Refresh file list
      await fetchStage2Attachments()
    } catch (err) {
      toast.error(getApiErrorDetail(err) || 'Failed to upload/index context documents')
    } finally {
      setUploadingStage2Doc(false)
      if (stage2DocInputRef.current) stage2DocInputRef.current.value = ''
    }
  }

  const handleStage2DocDelete = async (attachmentId: string) => {
    if (!id) return
    try {
      await api.delete(`/attachments/${id}/${attachmentId}`)
      toast.success('Removed context document')
      // If there are other documents left, re-process/re-embed them.
      // If no files remain, the backend delete endpoint clears the vector store automatically.
      const remaining = stage2MeetingDocs.filter(d => d.id !== attachmentId)
      if (remaining.length > 0) {
        const formData = new FormData()
        formData.append('type', 'context')
        await api.post(`/attachments/${id}/process`, formData, {
          headers: { 'Content-Type': 'multipart/form-data' }
        })
      }
      await fetchStage2Attachments()
    } catch (err) {
      toast.error(getApiErrorDetail(err) || 'Failed to delete context document')
    }
  }

  // Extract agenda items using the /raw-mom/{id}/agenda endpoint
  const handleExtractAgendaItems = async () => {
    if (!id) return
    setExtractingAgenda(true)
    try {
      const res = await api.post(`/raw-mom/${id}/agenda`, { force_reparse: forceReparseRef.current })
      const items: { topic: string; speaker: string | null }[] = res.data.agendas || []
      setExtractedAgendaItems(items)
      setAgendaExtractSource(res.data.source || '')
      const formatted = items.map((a, i) => `${i + 1}. ${a.topic}${a.speaker ? ` (Speaker: ${a.speaker})` : ''}`).join('\n')
      setAgendaText(formatted)
      toast.success(`${items.length} agenda item(s) extracted (${res.data.source})`)
    } catch (err: any) {
      toast.error(err?.response?.data?.detail ?? 'Agenda extraction failed')
    } finally {
      setExtractingAgenda(false)
    }
  }

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>, setter: (t: string) => void) => {
    const file = e.target.files?.[0]
    if (!file) return
    try {
      const text = await extractFileText(file)
      setter(text)
      toast.success(`Extracted text from ${file.name}`)
    } catch (err) {
      toast.error(getApiErrorDetail(err) || 'Failed to extract text')
    } finally {
      e.target.value = ''
    }
  }

  // Create Agenda (Step 1)
  const runCreateAgenda = async (forceReextract: boolean = false): Promise<boolean> => {
    setStage3AgendaStatus('processing')
    try {
      const res = await api.post(`/rom/${id}/stage3/create-agenda`, {
        agenda_text: agendaText || undefined,
        meeting_context_top_k: stage3MeetingTopK,
        global_context_top_k: stage3GlobalTopK,
        force_reextract: forceReextract,
        previous_mom_texts: previousMomDocs.length > 0 ? previousMomDocs.map(d => d.text.slice(0, prevMomCharLimit)) : null,
        previous_mom_char_limit: prevMomCharLimit,
      })
      const fullData = res.data.rom_data || { ...(romData || {} as RomData), stage3: res.data.stage3 }
      setRomData(fullData)
      setStage3AgendaStatus('done')
      setStage3FinalStatus('idle') // reset final since agendas changed
      setActiveTab('stage3')
      const agendaCount = fullData.stage3?.agendas?.length ?? 0
      toast.success(`Agendas created: ${agendaCount} agenda item(s)`)
      return true
    } catch (e) {
      setStage3AgendaStatus('error')
      toast.error(getApiErrorDetail(e) || 'Create Agenda failed')
      return false
    }
  }

  // Upload supporting document for a specific agenda
  const handleAgendaDocUpload = async (agendaId: string, file: File) => {
    if (!id || !file) return
    setAgendaDocUploading(prev => ({ ...prev, [agendaId]: true }))
    try {
      const formData = new FormData()
      formData.append('file', file)
      const res = await api.post(`/rom/${id}/stage3/agenda/${agendaId}/upload-document`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })
      // Update romData with the new doc points
      setRomData(prev => {
        if (!prev?.stage3) return prev
        const updatedDocPoints = { ...(prev.stage3.agenda_doc_points || {}) }
        updatedDocPoints[agendaId] = { doc_name: res.data.doc_name, points: res.data.points || [] }
        return {
          ...prev,
          stage3: { ...prev.stage3, agenda_doc_points: updatedDocPoints }
        }
      })
      toast.success(`Extracted ${res.data.points_count || 0} point(s) from ${res.data.doc_name}`)
    } catch (err) {
      toast.error(getApiErrorDetail(err) || 'Failed to upload supporting document')
    } finally {
      setAgendaDocUploading(prev => ({ ...prev, [agendaId]: false }))
    }
  }

  // Generate Final ROM (Step 2 – map points)
  const runGenerateFinalRom = async (): Promise<boolean> => {
    setStage3FinalStatus('processing')
    try {
      const res = await api.post(`/rom/${id}/stage3/generate-final-rom`, {
        meeting_context_top_k: stage3MeetingTopK,
        global_context_top_k: stage3GlobalTopK,
        batch_size: stage3BatchSize,
        include_agenda_doc_points: includeAgendaDocPoints,
        agendas: romData?.stage3?.agendas || undefined,
      })
      const fullData = res.data.rom_data || {
        ...(romData || {} as RomData),
        stage3: res.data.stage3,
        final_rom: res.data.final_rom
      }
      setRomData(fullData)
      // Snapshot original on fresh generation (resets any prior rewrite)
      if (fullData.final_rom) {
        setOriginalFinalRom(JSON.parse(JSON.stringify(fullData.final_rom)))
        setIsRewritten(false)
        setRewriteStatus('idle')
      }
      setStage3FinalStatus('done')
      setStage3Status('done')
      setActiveTab('final')
      toast.success('Final ROM generated successfully')
      return true
    } catch (e) {
      setStage3FinalStatus('error')
      toast.error(getApiErrorDetail(e) || 'Generate Final ROM failed')
      return false
    }
  }


  // Delete an agenda item (points move to General Discussion)
  const handleDeleteAgenda = async (agendaId: string, agendaTitle: string) => {
    if (!id) return
    if (!window.confirm(`Are you sure you want to delete agenda "${agendaTitle}"? All mapped discussion points will be moved to General Discussion.`)) {
      return
    }
    try {
      const res = await api.delete(`/rom/${id}/stage3/agenda/${agendaId}`)
      const fullData = res.data.rom_data || res.data
      setRomData(fullData)
      toast.success('Agenda deleted. Points remapped to General Discussion.')
    } catch (e) {
      toast.error(getApiErrorDetail(e) || 'Failed to delete agenda')
    }
  }

  // Delete an individual enhanced discussion point
  const handleDeletePoint = async (pointId: string) => {
    if (!id) return
    if (!window.confirm('Are you sure you want to delete this discussion point?')) {
      return
    }
    try {
      const res = await api.delete(`/rom/${id}/stage3/point/${pointId}`)
      const fullData = res.data.rom_data || res.data
      setRomData(fullData)
      toast.success('Discussion point deleted')
    } catch (e) {
      toast.error(getApiErrorDetail(e) || 'Failed to delete discussion point')
    }
  }

  // Generate MOM from Enhanced ROM
  const runGenerateMomFromRom = async () => {
    if (!id) return
    setEnhancedMomStatus('processing')
    try {
      const res = await api.post(`/rom/${id}/stage3/generate-mom-from-rom`)
      setEnhancedMomData(res.data.mom || res.data.enhanced_mom)
      if (res.data.rom_data) setRomData(res.data.rom_data)
      setEnhancedMomStatus('done')
      toast.success('Main MOM updated with Stage 2 Enhanced Points!')
    } catch (e) {
      setEnhancedMomStatus('error')
      toast.error(getApiErrorDetail(e) || 'Failed to generate MOM')
    }
  }

  // Generate Advanced MoM using custom prompt and selective regeneration flags
  const runGenerateAdvancedMom = async (): Promise<boolean> => {
    if (!id || !advancedCustomPrompt.trim()) return false
    setAdvancedStatus('processing')
    try {
      const res = await api.post(`/rom/${id}/stage3/generate-advanced-mom`, {
        custom_prompt: advancedCustomPrompt,
        regenerate_title: advancedRegenTitle,
        regenerate_intro: advancedRegenIntro,
        regenerate_conclusion: advancedRegenConclusion,
      })
      const fullData = res.data.rom_data || {
        ...(romData || {} as RomData),
        stage3: res.data.stage3,
        final_rom: res.data.final_rom
      }
      setRomData(fullData)
      setAdvancedStatus('done')
      toast.success('Advanced MoM generated successfully!')
      return true
    } catch (e) {
      setAdvancedStatus('error')
      toast.error(getApiErrorDetail(e) || 'Advanced MoM generation failed')
      return false
    }
  }

  // Download Enhanced MOM DOCX
  const downloadEnhancedMomDocx = async () => {
    if (!id) return
    try {
      const response = await api.get(`/rom/${id}/stage3/enhanced-mom/download/docx`, { responseType: 'blob' })
      const url = window.URL.createObjectURL(new Blob([response.data]))
      const link = document.createElement('a')
      link.href = url
      link.setAttribute('download', `mom_enhanced_${id}.docx`)
      document.body.appendChild(link)
      link.click()
      link.remove()
    } catch (e) {
      toast.error(getApiErrorDetail(e) || 'Failed to download MOM docx')
    }
  }

  // Move a discussion point to a different agenda in the final ROM
  const movePointToAgenda = (fromAgendaIdx: number, pointIdx: number, toAgendaId: string) => {
    if (!romData?.final_rom) return
    const newFinal = JSON.parse(JSON.stringify(romData.final_rom))
    const point = newFinal.agendas[fromAgendaIdx].discussion_points.splice(pointIdx, 1)[0]
    const toAgenda = newFinal.agendas.find((a: any) => a.agenda_id === toAgendaId)
    if (toAgenda) toAgenda.discussion_points.push(point)
    setRomData(prev => ({ ...prev as RomData, final_rom: newFinal }))
    toast.success('Point moved')
  }

  // Reorder a discussion point within an agenda
  const reorderPoint = (agendaIdx: number, pointIdx: number, direction: 'up' | 'down') => {
    if (!romData?.final_rom) return
    const newFinal = JSON.parse(JSON.stringify(romData.final_rom))
    const pts = newFinal.agendas[agendaIdx].discussion_points
    if (direction === 'up' && pointIdx === 0) return
    if (direction === 'down' && pointIdx === pts.length - 1) return
    const swapIdx = direction === 'up' ? pointIdx - 1 : pointIdx + 1
      ;[pts[pointIdx], pts[swapIdx]] = [pts[swapIdx], pts[pointIdx]]
    setRomData(prev => ({ ...prev as RomData, final_rom: newFinal }))
  }

  // Apply speaker name mappings throughout the final ROM
  const applySpeakerMappings = async () => {
    if (!romData?.final_rom) return
    const mappingObj: Record<string, string> = {}
    speakerMappings.forEach(m => {
      if (m.speaker_id.trim() && m.real_name.trim()) {
        mappingObj[m.speaker_id.trim()] = m.real_name.trim()
      }
    })
    const keys = Object.keys(mappingObj)
    keys.sort((a, b) => b.length - a.length)

    const replaceText = (text: any): any => {
      if (typeof text !== 'string' || !text) return text
      let res = text
      for (const k of keys) {
        const val = mappingObj[k]
        const escapedKey = k.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
        const regex = new RegExp(`\\b${escapedKey}\\b`, 'g')
        res = res.replace(regex, val)
      }
      return res
    }

    const replaceDeep = (obj: any): any => {
      if (typeof obj === 'string') {
        return replaceText(obj)
      } else if (Array.isArray(obj)) {
        return obj.map(replaceDeep)
      } else if (obj !== null && typeof obj === 'object') {
        const newObj: any = {}
        for (const [k, v] of Object.entries(obj)) {
          newObj[k] = replaceDeep(v)
        }
        return newObj
      }
      return obj
    }

    const newFinal = JSON.parse(JSON.stringify(romData.final_rom))
    newFinal.speaker_mappings = mappingObj
    if (newFinal.agendas) {
      newFinal.agendas = replaceDeep(newFinal.agendas)
    }

    setRomData(prev => ({ ...prev as RomData, final_rom: newFinal }))
    await saveFinalRom(newFinal)
    toast.success('Speaker names applied throughout Final ROM')
  }

  const loadData = useCallback(async () => {
    try {
      const recRes = await api.get(`/history/${id}`)
      setRecording(recRes.data)
    } catch (e) {
      toast.error(getApiErrorDetail(e) || 'Failed to load recording')
    } finally {
      setLoadingRec(false)
    }

    try {
      const romRes = await api.get(`/rom/${id}`)
      const rawData = romRes.data
      const data = (rawData?.rom_data || rawData) as RomData
      if (data && (data.stage1 || data.stage2 || data.stage3 || data.final_rom)) {
        setRomData(data)
        if (data.stage1?.status === 'done') setStage1Status('done')
        if (data.stage2?.status === 'done') setStage2Status('done')
        if (data.stage3?.agendas?.length) {
          setExtractedAgendaItems(data.stage3.agendas.map(a => ({
            topic: a.title,
            speaker: (a as any).speaker || null,
            details: a.description
          })))
          if (data.stage3.status === 'agendas_ready' || data.stage3.status === 'done') {
            setStage3AgendaStatus('done')
          }
          if (data.stage3.status === 'done' && data.final_rom) {
            setStage3FinalStatus('done')
          }
        }
        if (data.final_rom?.speaker_mappings) {
          const mappings = Object.entries(data.final_rom.speaker_mappings).map(([k, v]) => ({ speaker_id: k, real_name: v as string }))
          setSpeakerMappings(mappings)
        }

        if (data.final_rom) {
          setActiveTab('final')
          // Snapshot the original final_rom so Revert always works
          setOriginalFinalRom(prev => prev ?? JSON.parse(JSON.stringify(data.final_rom)))
        }
        else if (data.stage3) setActiveTab('stage3')
        else if (data.stage2) setActiveTab('stage2')
        else if (data.stage1) setActiveTab('stage1')
      }
    } catch (e) {
      // 404 is fine, means no ROM exists yet
    }
    await fetchStage2Attachments()
  }, [id, fetchStage2Attachments])

  useEffect(() => {
    if (id) loadData()
  }, [id, loadData])

  // ── Actions ─────────────────────────────────────────────────────────────────

  const runStage1 = async (): Promise<boolean> => {
    setStage1Status('processing')
    try {
      const res = await api.post(`/rom/${id}/stage1/generate`, {
        transcript_window_minutes: transcriptWindow
      })
      const fullData = res.data.rom_data || {
        ...(romData || {} as RomData),
        stage1: res.data.stage1 || res.data
      }
      setRomData(fullData)
      setStage1Status('done')
      setActiveTab('stage1')
      const pointCount = fullData.stage1?.discussion_points?.length ?? 0
      toast.success(`Stage 1 complete: ${pointCount} points extracted`)
      return true
    } catch (e) {
      setStage1Status('error')
      toast.error(getApiErrorDetail(e) || 'Stage 1 failed')
      return false
    }
  }

  const runStage2 = async (): Promise<boolean> => {
    setStage2Status('processing')
    try {
      const res = await api.post(`/rom/${id}/stage2/generate`, {
        meeting_context_top_k: Math.max(0, Number(meetingTopK) || 0),
        global_context_top_k: Math.max(0, Number(globalTopK) || 0),
        discussion_window_size: Math.max(3, Number(discussionWindowSize) || 5),
      })
      const fullData = res.data.rom_data || {
        ...(romData || {} as RomData),
        stage2: res.data.stage2 || res.data
      }
      setRomData(fullData)
      setStage2Status('done')
      setActiveTab('stage2')
      toast.success('Stage 2 complete')
      return true
    } catch (e) {
      setStage2Status('error')
      toast.error(getApiErrorDetail(e) || 'Stage 2 failed')
      return false
    }
  }

  const runStage3 = async (forceReextract: boolean = false): Promise<boolean> => {
    setStage3Status('processing')
    try {
      const res = await api.post(`/rom/${id}/stage3/generate`, {
        agenda_text: agendaText,
        meeting_context_top_k: stage3MeetingTopK,
        global_context_top_k: stage3GlobalTopK,
        force_reextract: forceReextract,
        previous_mom_texts: previousMomDocs.length > 0 ? previousMomDocs.map(d => d.text.slice(0, prevMomCharLimit)) : null,
        previous_mom_char_limit: prevMomCharLimit,
        batch_size: stage3BatchSize,
      })
      const fullData = res.data.rom_data || {
        ...(romData || {} as RomData),
        stage3: res.data.stage3 || res.data,
        final_rom: res.data.final_rom
      }
      setRomData(fullData)
      setStage3Status('done')
      if (fullData.stage3?.agendas?.length) {
        setExtractedAgendaItems(fullData.stage3.agendas.map((a: any) => ({
          topic: a.title,
          speaker: a.speaker || null,
          details: a.description
        })))
      }
      setActiveTab('final')
      toast.success('Stage 3 & Final ROM complete')
      return true
    } catch (e) {
      setStage3Status('error')
      toast.error(getApiErrorDetail(e) || 'Stage 3 failed')
      return false
    }
  }

  const handleGeneratePreciseRom = async () => {
    if (stage3Status !== 'done') {
      const ok = await runStage3(false)
      if (!ok) return
    }
    setRomViewMode('precise')
    setActiveTab('final')
    toast.success('Precise ROM generated')
  }

  const runCompleteRom = async () => {
    setIsGeneratingAll(true)
    try {
      setGenerateAllStep('Stage 1 of 3: Window Extraction...')
      const s1Ok = await runStage1()
      if (!s1Ok) {
        toast.error('Sequence stopped: Stage 1 failed.')
        return
      }

      setGenerateAllStep('Stage 2 of 3: RAG Enhancement...')
      const s2Ok = await runStage2()
      if (!s2Ok) {
        toast.error('Sequence stopped: Stage 2 failed.')
        return
      }

      setGenerateAllStep('Step 1 of Stage 3: Creating Agenda...')
      const s3aOk = await runCreateAgenda()
      if (!s3aOk) {
        toast.error('Sequence stopped: Create Agenda failed.')
        return
      }

      setGenerateAllStep('Step 2 of Stage 3: Mapping to Agendas...')
      const s3bOk = await runGenerateFinalRom()
      if (!s3bOk) {
        toast.error('Sequence stopped: Final ROM generation failed.')
        return
      }

      toast.success('Complete ROM pipeline generated successfully!')
    } catch (err) {
      toast.error('An unexpected error occurred during Complete ROM generation')
    } finally {
      setIsGeneratingAll(false)
      setGenerateAllStep(null)
    }
  }

  const downloadDocx = async (stage: string) => {
    try {
      const res = await api.get(`/rom/${id}/${stage}/download/docx`, { responseType: 'blob' })
      const url = URL.createObjectURL(new Blob([res.data]))
      const a = document.createElement('a')
      a.href = url
      a.download = `rom_${stage}_${id}.docx`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
      toast.success(`ROM ${stage.toUpperCase()} DOCX downloaded`)
    } catch {
      toast.error('Download failed')
    }
  }

  const saveFinalRom = async (updatedFinalRom: any) => {
    setSavingRom(true)
    try {
      await api.put(`/rom/${id}/final`, { final_rom: updatedFinalRom })
      setRomData(prev => ({
        ...(prev as RomData),
        final_rom: updatedFinalRom
      }))
      toast.success('Final ROM saved to database')
    } catch (e) {
      toast.error(getApiErrorDetail(e) || 'Save failed')
    } finally {
      setSavingRom(false)
    }
  }

  // Rewrite ROM using LLM (style-only, no content changes)
  const runRewriteRom = async () => {
    if (!id || !romData?.final_rom) return
    setRewriteStatus('processing')
    // Snapshot the original before rewriting (only once)
    if (!originalFinalRom) {
      setOriginalFinalRom(JSON.parse(JSON.stringify(romData.final_rom)))
    }
    try {
      const res = await api.post(`/rom/${id}/final/rewrite`, {
        rewrite_instruction: rewriteInstruction,
        mode: rewriteMode,
        window_size: rewriteWindowSize,
        writing_rules: rewriteMode === 'reference' ? writingRules : '',
      })
      const rewrittenFinalRom = res.data.rewritten_final_rom
      if (rewrittenFinalRom) {
        setRomData(prev => ({ ...(prev as RomData), final_rom: rewrittenFinalRom }))
        setIsRewritten(true)
        setRewriteStatus('done')
        toast.success('ROM rewritten successfully. Review the changes and Save if satisfied.')
      } else {
        throw new Error('No rewritten ROM returned from server')
      }
    } catch (e) {
      setRewriteStatus('error')
      toast.error(getApiErrorDetail(e) || 'Rewrite ROM failed')
    }
  }

  // Upload reference doc and extract writing style rules
  const handleReferenceDocUpload = async (files: FileList | null) => {
    if (!files?.length || !id) return
    const file = files[0]
    setExtractingRules(true)
    try {
      // Step 1: Extract text from uploaded file
      const formData = new FormData()
      formData.append('file', file)
      const extractRes = await api.post(`/raw-mom/${id}/extract-file-text`, formData)
      const refText = extractRes.data.text || ''
      if (!refText.trim()) {
        toast.error('Could not extract text from the uploaded file')
        return
      }
      // Step 2: Analyze writing style and get rules
      const rulesRes = await api.post(`/rom/${id}/final/extract-writing-rules`, {
        reference_text: refText,
      })
      const rules = rulesRes.data.writing_rules || ''
      setWritingRules(rules)
      toast.success(`Writing rules extracted from "${file.name}". Review and adjust before rewriting.`)
    } catch (e) {
      toast.error(getApiErrorDetail(e) || 'Failed to extract writing rules from reference document')
    } finally {
      setExtractingRules(false)
      if (referenceFileInputRef.current) referenceFileInputRef.current.value = ''
    }
  }

  // Revert to the original (pre-rewrite) final ROM
  const revertToOriginal = () => {
    if (!originalFinalRom) return
    setRomData(prev => ({ ...(prev as RomData), final_rom: JSON.parse(JSON.stringify(originalFinalRom)) }))
    setIsRewritten(false)
    setRewriteStatus('idle')
    toast.success('Reverted to original ROM')
  }

  if (loadingRec) {
    return (
      <div style={{ display: 'flex', height: '100%', alignItems: 'center', justifyContent: 'center', flexDirection: 'column', gap: '1rem' }}>
        <Loader size={24} className="spin" style={{ color: 'hsl(var(--accent))' }} />
        <p style={{ fontSize: '.88rem', color: 'hsl(var(--pencil))', fontFamily: 'Inter, sans-serif' }}>Loading ROM workspace...</p>
      </div>
    )
  }

  const stage1Count = romData?.stage1?.discussion_points?.length || 0
  const stage2Count = romData?.stage2?.polished_points?.length || 0
  const stage3Count = romData?.stage3?.agendas?.length || 0
  const finalCount = romData?.final_rom?.agendas?.length || 0

  return (
    <div className="page-scroll-root" style={{ display: 'flex', flexDirection: 'column', height: '100%', minHeight: 0, overflow: 'hidden' }}>

      {/* ── Top Panel Header Bar ── */}
      <div className="panel-header" style={{ flexShrink: 0, gap: '12px' }}>
        <button className="icon-btn" onClick={() => navigate(-1)} title="Go Back"><ArrowLeft size={16} /></button>
        <div style={{ width: 32, height: 32, borderRadius: '8px', flexShrink: 0, background: 'hsl(280,75%,60%/.15)', border: '1.5px solid hsl(280,75%,60%/.3)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <Brain size={16} style={{ color: 'hsl(280,75%,65%)' }} />
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <h1 style={{ fontSize: '1.05rem', fontWeight: 700, margin: 0 }}>Record of Meeting (ROM)</h1>
          <p style={{ fontSize: '.78rem', color: 'hsl(var(--pencil))', fontFamily: 'Inter', fontWeight: 400, marginTop: '1px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
            Multi-stage extraction & agenda RAG pipeline for "{recording?.filename}"
          </p>
        </div>

        {/* Global Download Actions Header Bar */}
        <div style={{ display: 'flex', gap: 6 }}>
          {stage1Count > 0 && (
            <button onClick={() => downloadDocx('stage1')} title="Download Stage 1 DOCX" style={{ display: 'flex', alignItems: 'center', gap: 5, padding: '.36rem .65rem', borderRadius: 8, border: '1.5px solid hsl(var(--border)/.6)', background: 'hsl(var(--card))', color: 'hsl(var(--ink))', fontSize: '.74rem', fontWeight: 600, cursor: 'pointer', fontFamily: 'Inter' }}>
              <Download size={12} /> Stage 1
            </button>
          )}
          {stage2Count > 0 && (
            <button onClick={() => downloadDocx('stage2')} title="Download Stage 2 DOCX" style={{ display: 'flex', alignItems: 'center', gap: 5, padding: '.36rem .65rem', borderRadius: 8, border: '1.5px solid hsl(var(--border)/.6)', background: 'hsl(var(--card))', color: 'hsl(var(--ink))', fontSize: '.74rem', fontWeight: 600, cursor: 'pointer', fontFamily: 'Inter' }}>
              <Download size={12} /> Stage 2
            </button>
          )}
          {finalCount > 0 && (
            <button onClick={() => downloadDocx('final')} title="Download Final ROM DOCX" style={{ display: 'flex', alignItems: 'center', gap: 5, padding: '.36rem .75rem', borderRadius: 8, border: '1.5px solid hsl(205,90%,55%/.4)', background: 'hsl(205,90%,55%/.08)', color: 'hsl(205,90%,60%)', fontSize: '.76rem', fontWeight: 700, cursor: 'pointer', fontFamily: 'Inter' }}>
              <FileDown size={13} /> Final DOCX
            </button>
          )}
        </div>
      </div>

      {/* ── Main Split View ── */}
      <div style={{ flex: 1, display: 'flex', gap: '1.25rem', padding: '1.15rem 1.5rem 1.25rem', minHeight: 0, height: 0, overflow: 'hidden' }}>

        {/* ── LEFT CONTROL PANEL ── */}
        <div style={{ width: 280, flexShrink: 0, minHeight: 0, height: '100%', display: 'flex', flexDirection: 'column', gap: '.6rem', overflowY: 'auto', paddingRight: 4 }}>

          {/* Recording Compact Info */}
          <div style={{ borderRadius: 10, border: '1.5px solid hsl(var(--border)/.4)', background: 'hsl(var(--card))', padding: '.55rem .8rem' }}>
            <div style={{ fontSize: '.68rem', fontWeight: 700, color: 'hsl(var(--pencil))', textTransform: 'uppercase', letterSpacing: '.05em', marginBottom: '.3rem' }}>Recording</div>
            <div style={{ fontSize: '.75rem', color: 'hsl(var(--ink))', fontWeight: 600, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{recording?.filename}</div>
            <div style={{ display: 'flex', gap: 10, marginTop: 4, fontSize: '.71rem', color: 'hsl(var(--pencil))' }}>
              <span><Clock size={10} style={{ display: 'inline', marginRight: 3 }} />{recording ? fmtTime(recording.duration) : '-'}</span>
              <span><User size={10} style={{ display: 'inline', marginRight: 3 }} />{recording?.speakers_detected?.length ?? 0} spk</span>
            </div>
          </div>

          {/* Complete ROM Automated Execution Card */}
          <div style={{
            borderRadius: 10, border: '1.5px solid hsl(280,75%,60%/.35)', background: 'hsl(280,75%,60%/.06)',
            padding: '.65rem .8rem', display: 'flex', flexDirection: 'column', gap: '.45rem'
          }}>
            <div style={{ fontSize: '.71rem', fontWeight: 700, color: 'hsl(280,75%,65%)', textTransform: 'uppercase', letterSpacing: '.04em', display: 'flex', alignItems: 'center', gap: 5 }}>
              <Play size={12} style={{ color: 'hsl(280,75%,65%)' }} /> Full Pipeline
            </div>
            <div style={{ fontSize: '.68rem', color: 'hsl(var(--pencil))', lineHeight: 1.35 }}>
              Sequentially executes Stage 1 → Stage 2 → Stage 3 with zero manual intervention.
            </div>

            {isGeneratingAll && generateAllStep && (
              <div style={{
                fontSize: '.69rem', fontWeight: 600, color: 'hsl(280,75%,65%)',
                display: 'flex', alignItems: 'center', gap: 6, padding: '4px 8px',
                background: 'hsl(280,75%,60%/.12)', borderRadius: 6, border: '1px solid hsl(280,75%,60%/.25)'
              }}>
                <Loader size={11} className="spin" />
                {generateAllStep}
              </div>
            )}

            <button
              onClick={runCompleteRom}
              disabled={isGeneratingAll || stage1Status === 'processing' || stage2Status === 'processing' || stage3Status === 'processing'}
              style={{
                width: '100%', padding: '.45rem .75rem', borderRadius: 8,
                background: isGeneratingAll ? 'hsl(280,75%,60%/.5)' : 'linear-gradient(135deg, hsl(280,75%,60%), hsl(205,90%,55%))',
                color: 'white', fontWeight: 700, fontSize: '.76rem', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6,
                cursor: (isGeneratingAll || stage1Status === 'processing' || stage2Status === 'processing' || stage3Status === 'processing') ? 'not-allowed' : 'pointer',
                border: 'none', boxShadow: '0 2px 6px rgba(0,0,0,0.12)', fontFamily: 'Inter'
              }}
            >
              {isGeneratingAll ? <Loader size={13} className="spin" /> : <Sparkles size={13} />}
              {isGeneratingAll ? 'Generating Full ROM...' : 'Generate Complete ROM'}
            </button>
          </div>

          {/* ── STAGE 1 CONTROLS ── */}
          <div style={{ borderRadius: 10, border: '1.5px solid hsl(var(--border)/.4)', background: 'hsl(var(--card))', padding: '.55rem .8rem', display: 'flex', flexDirection: 'column', gap: '.45rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div style={{ fontSize: '.7rem', fontWeight: 700, color: 'hsl(var(--pencil))', textTransform: 'uppercase', letterSpacing: '.05em', display: 'flex', alignItems: 'center', gap: 4 }}>
                <Brain size={12} style={{ color: 'hsl(280,75%,65%)' }} /> Stage 1
              </div>
              <StatusBadge state={stage1Status} />
            </div>

            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
                <label style={{ fontSize: '.71rem', color: 'hsl(var(--ink))', fontFamily: 'Inter' }}>Window (mins)</label>
                <span style={{ fontSize: '.71rem', fontWeight: 700, color: 'hsl(280,75%,65%)', fontFamily: 'JetBrains Mono' }}>{transcriptWindow}m</span>
              </div>
              <input
                type="range" min={0.5} max={15} step={0.5}
                value={transcriptWindow} onChange={e => setTranscriptWindow(Number(e.target.value))}
                style={{ width: '100%', accentColor: 'hsl(280,75%,60%)' }}
              />
            </div>

            <button
              onClick={runStage1}
              disabled={stage1Status === 'processing'}
              style={{
                width: '100%', padding: '4px 8px', borderRadius: 7, fontSize: '.73rem', fontWeight: 600,
                background: stage1Status === 'processing' ? 'hsl(var(--muted))' : 'hsl(var(--accent))',
                color: 'white', border: 'none', cursor: stage1Status === 'processing' ? 'not-allowed' : 'pointer',
                display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 5
              }}
            >
              {stage1Status === 'processing' ? <Loader size={11} className="spin" /> : <Sparkles size={11} />}
              {stage1Status === 'processing' ? 'Extracting...' : 'Generate Points'}
            </button>
          </div>

          {/* ── STAGE 2 CONTROLS ── */}
          <div style={{ borderRadius: 10, border: '1.5px solid hsl(var(--border)/.4)', background: 'hsl(var(--card))', padding: '.55rem .8rem', display: 'flex', flexDirection: 'column', gap: '.45rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div style={{ fontSize: '.7rem', fontWeight: 700, color: 'hsl(var(--pencil))', textTransform: 'uppercase', letterSpacing: '.05em', display: 'flex', alignItems: 'center', gap: 4 }}>
                <Target size={12} style={{ color: 'hsl(205,90%,55%)' }} /> Stage 2
              </div>
              <StatusBadge state={stage2Status} />
            </div>

            <div style={{ display: 'flex', gap: 8 }}>
              <div style={{ flex: 1 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 1 }}>
                  <label style={{ fontSize: '.68rem', color: 'hsl(var(--pencil))' }}>Meeting K</label>
                  <span style={{ fontSize: '.68rem', fontWeight: 700, color: 'hsl(140,70%,50%)', fontFamily: 'JetBrains Mono' }}>{meetingTopK}</span>
                </div>
                <input type="range" min={0} max={20} value={meetingTopK} onChange={e => setMeetingTopK(Number(e.target.value))} style={{ width: '100%', accentColor: 'hsl(140,70%,50%)' }} />
              </div>
              <div style={{ flex: 1 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 1 }}>
                  <label style={{ fontSize: '.68rem', color: 'hsl(var(--pencil))' }}>Global K</label>
                  <span style={{ fontSize: '.68rem', fontWeight: 700, color: 'hsl(30,90%,55%)', fontFamily: 'JetBrains Mono' }}>{globalTopK}</span>
                </div>
                <input type="range" min={0} max={20} value={globalTopK} onChange={e => setGlobalTopK(Number(e.target.value))} style={{ width: '100%', accentColor: 'hsl(30,90%,55%)' }} />
              </div>
            </div>

            {/* Discussion Window Size slider */}
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 1 }}>
                <label style={{ fontSize: '.68rem', color: 'hsl(var(--pencil))' }}>Discussion Window</label>
                <span style={{ fontSize: '.68rem', fontWeight: 700, color: 'hsl(270,80%,65%)', fontFamily: 'JetBrains Mono' }}>{discussionWindowSize} pts</span>
              </div>
              <input
                type="range" min={3} max={20} step={1}
                value={discussionWindowSize}
                onChange={e => setDiscussionWindowSize(Number(e.target.value))}
                style={{ width: '100%', accentColor: 'hsl(270,80%,65%)' }}
              />
            </div>

            {/* Meeting Docs upload */}
            <div>
              <div style={{ fontSize: '.66rem', fontWeight: 700, color: 'hsl(var(--pencil))', textTransform: 'uppercase', letterSpacing: '.04em', marginBottom: 4 }}>Meeting Context Docs</div>
              <input
                ref={stage2DocInputRef}
                type="file" multiple
                accept=".pdf,.docx,.pptx,.txt,.md,.png,.jpg,.jpeg,.webp,.xlsx,.xls,.csv"
                style={{ display: 'none' }}
                onChange={e => handleStage2DocUpload(e.target.files)}
              />
              {stage2MeetingDocs.length > 0 ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                  {stage2MeetingDocs.map((doc, i) => (
                    <div key={doc.id || i} style={{ display: 'flex', alignItems: 'center', gap: 4, padding: '2px 5px', borderRadius: 5, background: 'hsl(var(--muted)/.5)', fontSize: '.68rem' }}>
                      <FileText size={8} style={{ flexShrink: 0, color: 'hsl(205,90%,55%)' }} />
                      <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{doc.name}</span>
                      <button onClick={() => handleStage2DocDelete(doc.id)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'hsl(var(--destructive))' }}><X size={8} /></button>
                    </div>
                  ))}
                  <button
                    onClick={() => stage2DocInputRef.current?.click()}
                    disabled={uploadingStage2Doc}
                    style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 3, padding: '2px 5px', borderRadius: 5, border: '1px dashed hsl(205,90%,55%/.4)', background: 'hsl(205,90%,55%/.05)', color: 'hsl(205,90%,55%)', fontSize: '.66rem', cursor: 'pointer' }}
                  >
                    <Plus size={8} /> Add More
                  </button>
                </div>
              ) : (
                <div
                  onClick={() => stage2DocInputRef.current?.click()}
                  style={{ border: '1.5px dashed hsl(var(--border)/.5)', borderRadius: 6, padding: '.3rem', textAlign: 'center', cursor: 'pointer', fontSize: '.68rem', color: 'hsl(var(--pencil))', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 3 }}
                >
                  {uploadingStage2Doc ? <Loader size={8} className="spin" /> : <Upload size={8} />}
                  {uploadingStage2Doc ? 'Extracting...' : 'Upload Docs'}
                </div>
              )}
            </div>

            <button
              onClick={runStage2}
              disabled={stage2Status === 'processing' || stage1Status !== 'done'}
              style={{
                width: '100%', padding: '4px 8px', borderRadius: 7, fontSize: '.73rem', fontWeight: 600,
                background: (stage2Status === 'processing' || stage1Status !== 'done') ? 'hsl(var(--muted))' : 'hsl(205,90%,55%)',
                color: 'white', border: 'none', cursor: (stage2Status === 'processing' || stage1Status !== 'done') ? 'not-allowed' : 'pointer',
                display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 5
              }}
            >
              {stage2Status === 'processing' ? <Loader size={11} className="spin" /> : <Sparkles size={11} />}
              {stage2Status === 'processing' ? 'Enhancing...' : 'Enhance Points'}
            </button>
          </div>

          {/* ── STAGE 3 STEP 1 – CREATE AGENDA ── */}
          <div style={{ borderRadius: 10, border: '1.5px solid hsl(var(--border)/.4)', background: 'hsl(var(--card))', padding: '.55rem .8rem', display: 'flex', flexDirection: 'column', gap: '.45rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div style={{ fontSize: '.7rem', fontWeight: 700, color: 'hsl(140,70%,45%)', textTransform: 'uppercase', letterSpacing: '.05em', display: 'flex', alignItems: 'center', gap: 4 }}>
                <List size={12} /> Step 1 — Create Agenda
              </div>
              <StatusBadge state={stage3AgendaStatus} />
            </div>

            {/* Upload Agenda File */}
            <div>
              <div style={{ fontSize: '.66rem', fontWeight: 700, color: 'hsl(var(--pencil))', textTransform: 'uppercase', letterSpacing: '.04em', marginBottom: 4 }}>Upload Agenda File</div>
              <input
                ref={agendaFileInputRef}
                type="file"
                accept=".pdf,.docx,.pptx,.txt,.md,.png,.jpg,.jpeg,.webp,.xlsx,.xls,.csv"
                style={{ display: 'none' }}
                onChange={e => handleAgendaFileUpload(e.target.files)}
              />
              {agendaUploadedFiles.length > 0 ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 2, marginBottom: 4 }}>
                  {agendaUploadedFiles.map((f, i) => (
                    <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 4, padding: '3px 6px', borderRadius: 5, background: 'hsl(140,70%,45%/.1)', border: '1px solid hsl(140,70%,45%/.3)', fontSize: '.68rem' }}>
                      <FileText size={9} style={{ flexShrink: 0, color: 'hsl(140,70%,45%)' }} />
                      <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', color: 'hsl(var(--ink))' }}>{f.name}</span>
                      <button onClick={() => { setAgendaUploadedFiles([]); setAgendaText('') }} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'hsl(var(--destructive))' }}><X size={8} /></button>
                    </div>
                  ))}
                </div>
              ) : (
                <div
                  onClick={() => agendaFileInputRef.current?.click()}
                  style={{ border: '1.5px dashed hsl(140,70%,45%/.4)', borderRadius: 6, padding: '.35rem', textAlign: 'center', cursor: 'pointer', fontSize: '.68rem', color: 'hsl(140,70%,40%)', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 4, background: 'hsl(140,70%,45%/.04)' }}
                >
                  {uploadingAgendaFile ? <Loader size={9} className="spin" /> : <Upload size={9} />}
                  {uploadingAgendaFile ? 'Extracting...' : 'Upload Agenda File'}
                </div>
              )}
            </div>

            {/* Previous Meeting MoMs */}
            <div>
              <div style={{ fontSize: '.66rem', fontWeight: 700, color: 'hsl(var(--pencil))', textTransform: 'uppercase', letterSpacing: '.04em', marginBottom: 2 }}>Previous MoMs <span style={{ fontWeight: 400, textTransform: 'none', color: 'hsl(var(--pencil)/.6)', fontSize: '.63rem' }}>(optional)</span></div>
              <input
                ref={previousMomInputRef}
                type="file" multiple
                accept=".pdf,.docx,.pptx,.txt,.md,.png,.jpg,.jpeg,.webp,.xlsx,.xls,.csv"
                style={{ display: 'none' }}
                onChange={e => handlePreviousMomUpload(e.target.files)}
              />
              {previousMomDocs.length > 0 ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                  {previousMomDocs.map((doc, i) => (
                    <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 4, padding: '3px 6px', borderRadius: 5, background: 'hsl(200,80%,50%/.1)', border: '1px solid hsl(200,80%,50%/.3)', fontSize: '.68rem' }}>
                      <FileText size={9} style={{ flexShrink: 0, color: 'hsl(200,80%,55%)' }} />
                      <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', color: 'hsl(var(--ink))' }}>{doc.name}</span>
                      <button onClick={() => setPreviousMomDocs(prev => prev.filter((_, idx) => idx !== i))} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'hsl(var(--destructive))' }}><X size={8} /></button>
                    </div>
                  ))}
                  <button onClick={() => previousMomInputRef.current?.click()} disabled={uploadingPrevMom} style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 3, padding: '2px 5px', borderRadius: 5, border: '1px dashed hsl(200,80%,50%/.4)', background: 'hsl(200,80%,50%/.05)', color: 'hsl(200,80%,55%)', fontSize: '.66rem', cursor: 'pointer' }}>
                    <Plus size={8} /> Add More
                  </button>
                </div>
              ) : (
                <div onClick={() => previousMomInputRef.current?.click()} style={{ border: '1.5px dashed hsl(200,80%,50%/.35)', borderRadius: 6, padding: '.32rem', textAlign: 'center', cursor: 'pointer', fontSize: '.68rem', color: 'hsl(200,80%,45%)', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 4, background: 'hsl(200,80%,50%/.04)' }}>
                  {uploadingPrevMom ? <Loader size={9} className="spin" /> : <Upload size={9} />}
                  {uploadingPrevMom ? 'Uploading...' : 'Upload Previous MoM'}
                </div>
              )}
            </div>

            {/* Top-K sliders */}
            <div style={{ display: 'flex', gap: 8 }}>
              <div style={{ flex: 1 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 1 }}>
                  <label style={{ fontSize: '.68rem', color: 'hsl(var(--pencil))' }}>Meeting K</label>
                  <span style={{ fontSize: '.68rem', fontWeight: 700, color: 'hsl(140,70%,50%)', fontFamily: 'JetBrains Mono' }}>{stage3MeetingTopK}</span>
                </div>
                <input type="range" min={0} max={20} value={stage3MeetingTopK} onChange={e => setStage3MeetingTopK(Number(e.target.value))} style={{ width: '100%', accentColor: 'hsl(140,70%,50%)' }} />
              </div>
              <div style={{ flex: 1 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 1 }}>
                  <label style={{ fontSize: '.68rem', color: 'hsl(var(--pencil))' }}>Global K</label>
                  <span style={{ fontSize: '.68rem', fontWeight: 700, color: 'hsl(30,90%,55%)', fontFamily: 'JetBrains Mono' }}>{stage3GlobalTopK}</span>
                </div>
                <input type="range" min={0} max={20} value={stage3GlobalTopK} onChange={e => setStage3GlobalTopK(Number(e.target.value))} style={{ width: '100%', accentColor: 'hsl(30,90%,55%)' }} />
              </div>
            </div>

            {/* Prev MoM Char Limit slider */}
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 1 }}>
                <label style={{ fontSize: '.68rem', color: 'hsl(var(--pencil))' }}>Prev MoM Char Limit</label>
                <span style={{ fontSize: '.68rem', fontWeight: 700, color: 'hsl(200,80%,55%)', fontFamily: 'JetBrains Mono' }}>{prevMomCharLimit.toLocaleString()}</span>
              </div>
              <input
                type="range"
                min={1000}
                max={100000}
                step={1000}
                value={prevMomCharLimit}
                onChange={e => setPrevMomCharLimit(Number(e.target.value))}
                style={{ width: '100%', accentColor: 'hsl(200,80%,55%)' }}
              />
            </div>

            <button
              onClick={() => runCreateAgenda(false)}
              disabled={stage3AgendaStatus === 'processing' || stage2Status !== 'done'}
              style={{
                width: '100%', padding: '5px 8px', borderRadius: 7, fontSize: '.74rem', fontWeight: 700,
                background: (stage3AgendaStatus === 'processing' || stage2Status !== 'done') ? 'hsl(var(--muted))' : 'hsl(140,70%,45%)',
                color: 'white', border: 'none', cursor: (stage3AgendaStatus === 'processing' || stage2Status !== 'done') ? 'not-allowed' : 'pointer',
                display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 5, fontFamily: 'Inter'
              }}
            >
              {stage3AgendaStatus === 'processing' ? <Loader size={11} className="spin" /> : <List size={11} />}
              {stage3AgendaStatus === 'processing' ? 'Creating Agendas...' : 'Create Agenda'}
            </button>
          </div>

          {/* ── STAGE 3 STEP 2 – GENERATE FINAL ROM ── */}
          <div style={{ borderRadius: 10, border: '1.5px solid hsl(var(--border)/.4)', background: 'hsl(var(--card))', padding: '.55rem .8rem', display: 'flex', flexDirection: 'column', gap: '.45rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div style={{ fontSize: '.7rem', fontWeight: 700, color: 'hsl(280,75%,60%)', textTransform: 'uppercase', letterSpacing: '.05em', display: 'flex', alignItems: 'center', gap: 4 }}>
                <Sparkles size={12} /> Step 2 — Generate Final ROM
              </div>
              <StatusBadge state={stage3FinalStatus} />
            </div>

            {stage3AgendaStatus !== 'done' && (
              <div style={{ fontSize: '.67rem', color: 'hsl(var(--pencil))', background: 'hsl(var(--muted)/.4)', borderRadius: 6, padding: '.3rem .5rem', lineHeight: 1.4 }}>
                Complete Step 1 (Create Agenda) first
              </div>
            )}

            {/* Batch size slider */}
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 1 }}>
                <label style={{ fontSize: '.68rem', color: 'hsl(var(--pencil))' }}>Assign Batch Size</label>
                <span style={{ fontSize: '.68rem', fontWeight: 700, color: 'hsl(45,90%,55%)', fontFamily: 'JetBrains Mono' }}>{stage3BatchSize} pts</span>
              </div>
              <input type="range" min={5} max={50} step={5} value={stage3BatchSize} onChange={e => setStage3BatchSize(Number(e.target.value))} style={{ width: '100%', accentColor: 'hsl(45,90%,55%)' }} />
            </div>

            {/* Include Agenda Document Points checkbox */}
            <label style={{ display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer', fontSize: '.72rem', color: 'hsl(var(--ink))', padding: '.3rem .4rem', borderRadius: 6, background: includeAgendaDocPoints ? 'hsl(280,75%,60%/.08)' : 'transparent', border: `1px solid ${includeAgendaDocPoints ? 'hsl(280,75%,60%/.3)' : 'hsl(var(--border)/.3)'}`, transition: 'all .15s' }}>
              <input
                type="checkbox"
                checked={includeAgendaDocPoints}
                onChange={e => setIncludeAgendaDocPoints(e.target.checked)}
                style={{ accentColor: 'hsl(280,75%,60%)', width: 12, height: 12 }}
              />
              <span style={{ fontWeight: 600 }}>Include Agenda Document Points</span>
            </label>

            <button
              onClick={runGenerateFinalRom}
              disabled={stage3FinalStatus === 'processing' || stage3AgendaStatus !== 'done'}
              style={{
                width: '100%', padding: '5px 8px', borderRadius: 7, fontSize: '.74rem', fontWeight: 700,
                background: (stage3FinalStatus === 'processing' || stage3AgendaStatus !== 'done') ? 'hsl(var(--muted))' : 'hsl(280,75%,60%)',
                color: 'white', border: 'none', cursor: (stage3FinalStatus === 'processing' || stage3AgendaStatus !== 'done') ? 'not-allowed' : 'pointer',
                display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 5, fontFamily: 'Inter'
              }}
            >
              {stage3FinalStatus === 'processing' ? <Loader size={11} className="spin" /> : <Sparkles size={11} />}
              {stage3FinalStatus === 'processing' ? 'Generating ROM...' : 'Generate Final ROM'}
            </button>
          </div>

          {/* Step 3 — Generate MOM (from Enhanced ROM) */}
          <div style={{ borderRadius: 10, border: '1.5px solid hsl(var(--border)/.4)', background: 'hsl(var(--card))', padding: '.55rem .8rem', display: 'flex', flexDirection: 'column', gap: '.45rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div style={{ fontSize: '.7rem', fontWeight: 700, color: 'hsl(140,70%,45%)', textTransform: 'uppercase', letterSpacing: '.05em', display: 'flex', alignItems: 'center', gap: 4 }}>
                <FileText size={12} /> Step 3 — Generate MOM
              </div>
              <StatusBadge state={enhancedMomStatus || (romData?.stage3?.enhanced_mom ? 'done' : 'idle')} />
            </div>

            <div style={{ fontSize: '.67rem', color: 'hsl(var(--pencil))', lineHeight: 1.35 }}>
              Generate Minutes of Meeting directly from Stage 2 enhanced discussion points.
            </div>

            <button
              onClick={runGenerateMomFromRom}
              disabled={enhancedMomStatus === 'processing' || stage2Count === 0}
              style={{
                width: '100%', padding: '5px 8px', borderRadius: 7, fontSize: '.74rem', fontWeight: 700,
                background: (enhancedMomStatus === 'processing' || stage2Count === 0) ? 'hsl(var(--muted))' : 'hsl(140,70%,45%)',
                color: 'white', border: 'none', cursor: (enhancedMomStatus === 'processing' || stage2Count === 0) ? 'not-allowed' : 'pointer',
                display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 5, fontFamily: 'Inter'
              }}
            >
              {enhancedMomStatus === 'processing' ? <Loader size={11} className="spin" /> : <FileText size={11} />}
              {enhancedMomStatus === 'processing' ? 'Generating MOM...' : 'Generate MOM (from Enhanced ROM)'}
            </button>
            {(enhancedMomData || romData?.stage3?.enhanced_mom) && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
                <button
                  onClick={() => navigate(`/dashboard/history/${id}/mom`)}
                  style={{
                    width: '100%', padding: '5px 8px', borderRadius: 6, fontSize: '.73rem', fontWeight: 700,
                    background: 'hsl(var(--accent))', color: 'hsl(var(--accent-foreground))', border: 'none',
                    cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 4, fontFamily: 'Inter'
                  }}
                >
                  <FileText size={11} /> View Main MOM Page
                </button>
                <button
                  onClick={downloadEnhancedMomDocx}
                  style={{
                    width: '100%', padding: '4px 8px', borderRadius: 6, fontSize: '.71rem', fontWeight: 600,
                    background: 'hsl(140,70%,45%/.1)', color: 'hsl(140,70%,40%)', border: '1px solid hsl(140,70%,45%/.3)',
                    cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 4, fontFamily: 'Inter'
                  }}
                >
                  <Download size={10} /> Download MOM (.docx)
                </button>
              </div>
            )}
          </div>

        </div>


        {/* ── RIGHT CONTENT PANEL ── */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0, height: '100%', overflow: 'hidden', background: 'hsl(var(--card))', borderRadius: 12, border: '1.5px solid hsl(var(--border)/.4)' }}>

          {/* ── Tab Header Bar ── */}
          <div style={{ display: 'flex', borderBottom: '1px solid hsl(var(--border)/.4)', background: 'hsl(var(--muted)/.2)', padding: '0 1.25rem', flexShrink: 0 }}>
            {[
              { id: 'stage1', label: 'Stage 1 (Raw)', count: stage1Count, color: 'hsl(280,75%,65%)' },
              { id: 'stage2', label: 'Stage 2 (Enhanced)', count: stage2Count, color: 'hsl(205,90%,55%)' },
              { id: 'stage3', label: 'Stage 3 (Mapped)', count: stage3Count, color: 'hsl(140,70%,50%)' },
              { id: 'final', label: 'Final ROM', count: finalCount, color: 'hsl(30,90%,55%)' },
              { id: 'advanced', label: 'Advanced MoM', count: 0, color: 'hsl(330,85%,60%)' },
            ].map(tab => {
              const isActive = activeTab === tab.id
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id as any)}
                  style={{
                    padding: '.75rem 1.1rem', fontSize: '.82rem', fontWeight: isActive ? 700 : 500,
                    color: isActive ? tab.color : 'hsl(var(--pencil))',
                    borderBottom: `2px solid ${isActive ? tab.color : 'transparent'}`,
                    background: 'transparent', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 6,
                    transition: 'all .15s', fontFamily: 'Inter'
                  }}
                >
                  {tab.label}
                  {tab.count > 0 && (
                    <span style={{
                      fontSize: '.66rem', fontWeight: 700, padding: '1px 6px', borderRadius: 999,
                      background: isActive ? `${tab.color}22` : 'hsl(var(--muted))',
                      color: isActive ? tab.color : 'hsl(var(--pencil))',
                      border: `1px solid ${isActive ? `${tab.color}44` : 'hsl(var(--border)/.5)'}`
                    }}>
                      {tab.count}
                    </span>
                  )}
                </button>
              )
            })}
          </div>

          {/* ── Tab Content Container ── */}
          <div style={{ flex: 1, overflowY: 'auto', padding: '1.25rem 1.5rem', minHeight: 0 }}>

            {/* ── STAGE 1 TAB ── */}
            {activeTab === 'stage1' && (
              !romData?.stage1?.discussion_points || romData.stage1.discussion_points.length === 0 ? (
                <div style={{ textAlign: 'center', padding: '4rem 1.5rem', color: 'hsl(var(--pencil))' }}>
                  <Brain size={42} style={{ margin: '0 auto 1rem', opacity: 0.4, color: 'hsl(280,75%,65%)' }} />
                  <div style={{ fontSize: '1.05rem', fontWeight: 700, color: 'hsl(var(--ink))', marginBottom: '.4rem' }}>No Stage 1 Points Extracted Yet</div>
                  <div style={{ fontSize: '.84rem', maxWidth: 420, margin: '0 auto 1.25rem', lineHeight: 1.45 }}>
                    Click <strong>Generate Points</strong> on the left panel to execute sliding window transcript extraction.
                  </div>
                  <button onClick={runStage1} disabled={stage1Status === 'processing'} className="btn btn-primary" style={{ fontSize: '.8rem', padding: '.45rem 1rem' }}>
                    <Sparkles size={14} /> Run Stage 1 Extraction
                  </button>
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 10 }}>
                    <SectionHeader icon={<Brain size={14} />} label="Stage 1: Raw Discussion Points" count={stage1Count} color="hsl(280,75%,65%)" />
                    {(Boolean(romData?.stage1?.video_ocr_blocks_used) || romData?.stage1?.discussion_points?.some(p => Boolean(p.video_transcript_context))) && (
                      <div style={{
                        display: 'inline-flex', alignItems: 'center', gap: 5,
                        padding: '.25rem .65rem', borderRadius: 8,
                        background: 'hsl(210,80%,55%/.12)', border: '1px solid hsl(210,80%,55%/.3)',
                        color: 'hsl(210,85%,60%)', fontSize: '.74rem', fontWeight: 600, fontFamily: 'Inter'
                      }}>
                        <Video size={12} /> Video Transcription Used {romData?.stage1?.video_ocr_blocks_used ? `(${romData.stage1.video_ocr_blocks_used} OCR blocks)` : ''}
                      </div>
                    )}
                  </div>

                  <div style={{ overflowX: 'auto', borderRadius: 10, border: '1.5px solid hsl(var(--border)/.4)', background: 'hsl(var(--card))' }}>
                    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '.82rem', fontFamily: 'Inter' }}>
                      <thead>
                        <tr style={{ borderBottom: '1.5px solid hsl(var(--border)/.6)', background: 'hsl(var(--muted)/.4)' }}>
                          <th style={{ padding: '.75rem .85rem', textAlign: 'left', fontWeight: 700, width: 140 }}>Window</th>
                          <th style={{ padding: '.75rem .85rem', textAlign: 'left', fontWeight: 700 }}>Generated Discussion Points</th>
                        </tr>
                      </thead>
                      <tbody>
                        {(() => {
                          const groups: Record<number, DiscussionPoint[]> = {}
                          romData.stage1.discussion_points.forEach((pt, idx) => {
                            const winIdx = pt.window_index ?? 1
                            if (!groups[winIdx]) {
                              groups[winIdx] = []
                            }
                            groups[winIdx].push(pt)
                          })

                          return Object.entries(groups)
                            .map(([winIdxStr, pts]) => {
                              const winIdx = Number(winIdxStr)
                              const timelineStart = Math.min(...pts.map(p => p.timeline_start ?? 0))
                              const timelineEnd = Math.max(...pts.map(p => p.timeline_end ?? 0))
                              return {
                                windowIndex: winIdx,
                                timelineStart,
                                timelineEnd,
                                points: pts
                              }
                            })
                            .sort((a, b) => a.windowIndex - b.windowIndex)
                            .map((group, gIdx) => (
                              <tr key={group.windowIndex} style={{ borderBottom: '1.5px solid hsl(var(--border)/.4)', background: gIdx % 2 === 0 ? 'transparent' : 'hsl(var(--muted)/.1)' }}>
                                <td style={{ padding: '.9rem .85rem', verticalAlign: 'top', borderRight: '1px solid hsl(var(--border)/.3)' }}>
                                  <div style={{ fontWeight: 700, color: 'hsl(280,75%,60%)', fontSize: '.84rem', marginBottom: '.3rem' }}>
                                    Window {group.windowIndex}
                                  </div>
                                  <span style={{
                                    background: 'hsl(280,75%,60%/.08)', color: 'hsl(280,75%,65%)',
                                    border: '1px solid hsl(280,75%,60%/.2)',
                                    padding: '2px 6px', borderRadius: 8, fontSize: '.68rem', fontWeight: 700,
                                    display: 'inline-flex', alignItems: 'center', gap: 3, fontFamily: 'JetBrains Mono',
                                    whiteSpace: 'nowrap'
                                  }}>
                                    <Clock size={10} /> {fmtTime(group.timelineStart)} – {fmtTime(group.timelineEnd)}
                                  </span>
                                </td>
                                <td style={{ padding: '.9rem .85rem', verticalAlign: 'top' }}>
                                  <div style={{ display: 'flex', flexDirection: 'column', gap: '.9rem' }}>
                                    {group.points.map((pt, pIdx) => (
                                      <div key={pt.id || pIdx} style={{
                                        padding: '.65rem .85rem',
                                        background: 'hsl(var(--paper)/.4)',
                                        border: '1px solid hsl(var(--border)/.3)',
                                        borderRadius: 8,
                                        display: 'flex',
                                        flexDirection: 'column',
                                        gap: '.45rem'
                                      }}>
                                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                          <span style={{ fontSize: '.74rem', fontWeight: 700, color: 'hsl(280,75%,65%)', background: 'hsl(280,75%,60%/.08)', padding: '1px 6px', borderRadius: 4 }}>
                                            Point {pIdx + 1}
                                          </span>
                                          {pt.speakers && pt.speakers.length > 0 && (
                                            <div style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
                                              {pt.speakers.map((sp, idx) => (
                                                <span key={idx} style={{
                                                  background: 'hsl(var(--muted)/.6)', color: 'hsl(var(--ink))',
                                                  padding: '1px 7px', borderRadius: 10, fontSize: '.68rem', fontWeight: 600,
                                                  display: 'inline-flex', alignItems: 'center', gap: 3
                                                }}>
                                                  <User size={8} /> {sp}
                                                </span>
                                              ))}
                                            </div>
                                          )}
                                        </div>
                                        <p style={{ fontSize: '.88rem', fontWeight: 500, color: 'hsl(var(--ink))', lineHeight: 1.5, margin: 0 }}>
                                          {formatItemText(pt.discussion_point)}
                                        </p>
                                        {((pt.decisions?.length || 0) > 0 || (pt.action_items?.length || 0) > 0 || (pt.technical_terms?.length || 0) > 0) && (
                                          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5 }}>
                                            {pt.decisions?.map((d, idx) => (
                                              <span key={`dec-${idx}`} style={{ background: 'hsl(140,70%,45%/.12)', color: 'hsl(140,70%,45%)', border: '1px solid hsl(140,70%,45%/.3)', padding: '1px 6px', borderRadius: 8, fontSize: '.66rem', fontWeight: 600 }}>Decision: {formatItemText(d)}</span>
                                            ))}
                                            {pt.action_items?.map((a, idx) => (
                                              <span key={`act-${idx}`} style={{ background: 'hsl(35,90%,50%/.12)', color: 'hsl(35,90%,45%)', border: '1px solid hsl(35,90%,50%/.3)', padding: '1px 6px', borderRadius: 8, fontSize: '.66rem', fontWeight: 600 }}>Action: {formatItemText(a)}</span>
                                            ))}
                                            {pt.technical_terms?.map((t, idx) => (
                                              <span key={`tech-${idx}`} style={{ background: 'hsl(280,70%,60%/.12)', color: 'hsl(280,70%,65%)', border: '1px solid hsl(280,70%,60%/.3)', padding: '1px 6px', borderRadius: 8, fontSize: '.66rem', fontWeight: 600 }}>{formatItemText(t)}</span>
                                            ))}
                                          </div>
                                        )}
                                        {(pt.raw_transcript_text || pt.video_transcript_context) && (
                                          <details style={{ background: 'transparent', marginTop: '.3rem' }}>
                                            <summary style={{ cursor: 'pointer', fontSize: '.72rem', fontWeight: 600, color: 'hsl(var(--pencil))', outline: 'none', display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                                              <span>View Source Transcript & Context</span>
                                              {pt.video_transcript_context && (
                                                <span style={{ background: 'hsl(210,80%,55%/.15)', color: 'hsl(210,85%,60%)', border: '1px solid hsl(210,80%,55%/.3)', padding: '1px 6px', borderRadius: 6, fontSize: '.64rem', fontWeight: 700, display: 'inline-flex', alignItems: 'center', gap: 3 }}>
                                                  <Video size={9} /> Video OCR Included
                                                </span>
                                              )}
                                            </summary>
                                            <div style={{ marginTop: '.4rem', display: 'flex', flexDirection: 'column', gap: '.45rem' }}>
                                              {pt.raw_transcript_text && (
                                                <div>
                                                  <div style={{ fontSize: '.68rem', fontWeight: 700, color: 'hsl(var(--pencil))', marginBottom: 3 }}>Audio Transcript:</div>
                                                  <pre style={{
                                                    padding: '.5rem .65rem', borderRadius: 6, background: 'hsl(var(--muted)/.3)',
                                                    border: '1px solid hsl(var(--border)/.2)', fontSize: '.72rem', whiteSpace: 'pre-wrap',
                                                    fontFamily: 'Inter, sans-serif', color: 'hsl(var(--ink))', lineHeight: 1.4, margin: 0
                                                  }}>
                                                    {pt.raw_transcript_text}
                                                  </pre>
                                                </div>
                                              )}
                                              {pt.video_transcript_context && (
                                                <div>
                                                  <div style={{ fontSize: '.68rem', fontWeight: 700, color: 'hsl(210,85%,60%)', marginBottom: 3, display: 'flex', alignItems: 'center', gap: 4 }}>
                                                    <Video size={10} /> Video Frame Transcription / OCR Context:
                                                  </div>
                                                  <pre style={{
                                                    padding: '.5rem .65rem', borderRadius: 6, background: 'hsl(210,80%,55%/.06)',
                                                    border: '1px solid hsl(210,80%,55%/.2)', fontSize: '.72rem', whiteSpace: 'pre-wrap',
                                                    fontFamily: 'Inter, sans-serif', color: 'hsl(var(--ink))', lineHeight: 1.4, margin: 0
                                                  }}>
                                                    {pt.video_transcript_context}
                                                  </pre>
                                                </div>
                                              )}
                                            </div>
                                          </details>
                                        )}
                                      </div>
                                    ))}
                                  </div>
                                </td>
                              </tr>
                            ))
                        })()}
                      </tbody>
                    </table>
                  </div>
                </div>
              )
            )}

            {/* ── STAGE 2 TAB ── */}
            {activeTab === 'stage2' && (
              !romData?.stage2?.polished_points || romData.stage2.polished_points.length === 0 ? (
                <div style={{ textAlign: 'center', padding: '4rem 1.5rem', color: 'hsl(var(--pencil))' }}>
                  <Target size={42} style={{ margin: '0 auto 1rem', opacity: 0.4, color: 'hsl(205,90%,55%)' }} />
                  <div style={{ fontSize: '1.05rem', fontWeight: 700, color: 'hsl(var(--ink))', marginBottom: '.4rem' }}>No Stage 2 Enhanced Points Yet</div>
                  <div style={{ fontSize: '.84rem', maxWidth: 420, margin: '0 auto 1.25rem', lineHeight: 1.45 }}>
                    Complete Stage 1 first, then click <strong>Enhance Points</strong> to run RAG context enrichment and batch point merging.
                  </div>
                  <button onClick={runStage2} disabled={stage2Status === 'processing' || stage1Count === 0} className="btn btn-primary" style={{ fontSize: '.8rem', padding: '.45rem 1rem' }}>
                    <Sparkles size={14} /> Run Stage 2 Enhancement
                  </button>
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                  <SectionHeader icon={<Target size={14} />} label="Stage 2: RAG Enhanced & Merged Points" count={stage2Count} color="hsl(205,90%,55%)" />
                  {romData?.stage2?.polished_points?.map((pt, i) => (
                    <div key={pt.id || i} style={{
                      borderRadius: 10, border: '1.5px solid hsl(var(--border)/.4)',
                      background: 'hsl(var(--card))', padding: '1rem 1.15rem'
                    }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '.6rem' }}>
                        <span style={{
                          background: 'hsl(205,90%,55%/.12)', color: 'hsl(205,90%,60%)',
                          border: '1px solid hsl(205,90%,55%/.3)',
                          padding: '2px 8px', borderRadius: 12, fontSize: '.72rem', fontWeight: 700,
                          display: 'flex', alignItems: 'center', gap: 4, fontFamily: 'JetBrains Mono'
                        }}>
                          <Clock size={11} /> {fmtTime(pt.timeline_start || 0)} – {fmtTime(pt.timeline_end || 0)}
                        </span>
                        <span style={{ fontSize: '.68rem', fontWeight: 700, color: 'hsl(205,90%,55%)', background: 'hsl(205,90%,55%/.08)', padding: '1px 6px', borderRadius: 6 }}>
                          Point P{i + 1}
                        </span>
                      </div>

                      <div style={{ padding: '.75rem .9rem', background: 'hsl(var(--muted)/.3)', borderRadius: 8, border: '1px solid hsl(var(--border)/.3)', marginBottom: '.75rem' }}>
                        <div style={{ fontSize: '.67rem', fontWeight: 700, color: 'hsl(var(--pencil))', marginBottom: 4, textTransform: 'uppercase', letterSpacing: '.04em' }}>Enhanced Point</div>
                        <p style={{ fontSize: '.92rem', color: 'hsl(var(--ink))', lineHeight: 1.55, margin: 0 }}>
                          {formatItemText(pt.polished_text)}
                        </p>
                      </div>

                      {((pt.retrieved_context?.meeting_chunks?.length || 0) > 0 || (pt.retrieved_context?.global_chunks?.length || 0) > 0) && (
                        <details style={{ background: 'transparent' }}>
                          <summary style={{ cursor: 'pointer', fontSize: '.76rem', fontWeight: 600, color: 'hsl(205,90%,55%)', outline: 'none' }}>
                            View Context Preview ({((pt.retrieved_context?.meeting_chunks?.length || 0) + (pt.retrieved_context?.global_chunks?.length || 0))})
                          </summary>
                          <div style={{ marginTop: '.5rem', display: 'flex', flexDirection: 'column', gap: 6 }}>
                            {pt.retrieved_context?.meeting_chunks?.map((c, idx) => {
                              const txt = (c.text || (c as any).content || (c as any).chunk || '').trim()
                              return txt ? (
                                <div key={`m-${idx}`} style={{ padding: '.5rem .65rem', border: '1px solid hsl(var(--border)/.3)', borderRadius: 7, fontSize: '.76rem', background: 'hsl(var(--paper)/.4)', color: 'hsl(var(--ink))', lineHeight: 1.45 }}>
                                  <span style={{ fontSize: '.62rem', fontWeight: 700, color: 'hsl(205,90%,55%)', textTransform: 'uppercase', letterSpacing: '.04em', display: 'block', marginBottom: 2 }}>📄 Meeting · {c.filename || 'Document'}</span>
                                  {txt}
                                </div>
                              ) : null
                            })}
                            {pt.retrieved_context?.global_chunks?.map((c, idx) => {
                              const txt = (c.text || (c as any).content || (c as any).chunk || '').trim()
                              return txt ? (
                                <div key={`g-${idx}`} style={{ padding: '.5rem .65rem', border: '1px solid hsl(var(--border)/.3)', borderRadius: 7, fontSize: '.76rem', background: 'hsl(var(--paper)/.4)', color: 'hsl(var(--ink))', lineHeight: 1.45 }}>
                                  <span style={{ fontSize: '.62rem', fontWeight: 700, color: 'hsl(30,90%,55%)', textTransform: 'uppercase', letterSpacing: '.04em', display: 'block', marginBottom: 2 }}>🌐 Global · {c.filename || 'Document'}</span>
                                  {txt}
                                </div>
                              ) : null
                            })}
                          </div>
                        </details>
                      )}

                      {/* Context Usage Report */}
                      {(pt.retrieved_context as any)?.context_usage_report && (() => {
                        const rep = (pt.retrieved_context as any).context_usage_report
                        const badges: { label: string; color: string }[] = []
                        if (rep.verified) badges.push({ label: '✅ Verified', color: 'hsl(140,65%,45%)' })
                        if (rep.technical_details_added) badges.push({ label: '🔬 Details added', color: 'hsl(205,90%,50%)' })
                        if (rep.abbreviations_expanded) badges.push({ label: '🔤 Abbrevs expanded', color: 'hsl(270,75%,60%)' })
                        if (rep.references_added) badges.push({ label: '📎 Refs added', color: 'hsl(30,90%,50%)' })
                        if (rep.terminology_clarified) badges.push({ label: '📘 Terms clarified', color: 'hsl(190,80%,45%)' })
                        if (rep.no_useful_context) badges.push({ label: '⚠️ No useful context', color: 'hsl(45,90%,50%)' })
                        const meetingDocs: string[] = rep.meeting_context_docs || []
                        const globalDocs: string[] = rep.global_context_docs || []
                        if (badges.length === 0 && meetingDocs.length === 0 && globalDocs.length === 0) return null
                        return (
                          <details style={{ marginTop: '.5rem' }}>
                            <summary style={{ cursor: 'pointer', fontSize: '.73rem', fontWeight: 600, color: 'hsl(270,75%,60%)', outline: 'none' }}>
                              Context Usage Report
                            </summary>
                            <div style={{ marginTop: '.45rem', padding: '.55rem .7rem', borderRadius: 8, border: '1px solid hsl(270,75%,55%/.25)', background: 'hsl(270,75%,55%/.05)' }}>
                              {badges.length > 0 && (
                                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginBottom: (meetingDocs.length + globalDocs.length > 0) ? '.5rem' : 0 }}>
                                  {badges.map((b, bi) => (
                                    <span key={bi} style={{
                                      fontSize: '.66rem', fontWeight: 700, padding: '1px 7px', borderRadius: 10,
                                      background: b.color + '18', color: b.color, border: `1px solid ${b.color}33`
                                    }}>{b.label}</span>
                                  ))}
                                </div>
                              )}
                              {(meetingDocs.length > 0 || globalDocs.length > 0) && (
                                <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                                  {meetingDocs.map((d, di) => (
                                    <span key={`md-${di}`} style={{ fontSize: '.65rem', color: 'hsl(205,90%,55%)', display: 'flex', alignItems: 'center', gap: 3 }}>📄 {d}</span>
                                  ))}
                                  {globalDocs.map((d, di) => (
                                    <span key={`gd-${di}`} style={{ fontSize: '.65rem', color: 'hsl(30,90%,55%)', display: 'flex', alignItems: 'center', gap: 3 }}>🌐 {d}</span>
                                  ))}
                                </div>
                              )}
                            </div>
                          </details>
                        )
                      })()}
                    </div>
                  ))}
                </div>
              )
            )}

            {/* ── STAGE 3 TAB (Agenda Cards with Doc Upload) ── */}
            {activeTab === 'stage3' && (
              !romData?.stage3?.agendas || romData.stage3.agendas.length === 0 ? (
                <div style={{ textAlign: 'center', padding: '4rem 1.5rem', color: 'hsl(var(--pencil))' }}>
                  <List size={42} style={{ margin: '0 auto 1rem', opacity: 0.4, color: 'hsl(140,70%,50%)' }} />
                  <div style={{ fontSize: '1.05rem', fontWeight: 700, color: 'hsl(var(--ink))', marginBottom: '.4rem' }}>No Agendas Created Yet</div>
                  <div style={{ fontSize: '.84rem', maxWidth: 440, margin: '0 auto 1.25rem', lineHeight: 1.5 }}>
                    Complete Stage 2, then click <strong>Create Agenda</strong> in the left panel.
                  </div>
                  <button onClick={() => runCreateAgenda(false)} disabled={stage3AgendaStatus === 'processing' || stage2Status !== 'done'} className="btn btn-primary" style={{ fontSize: '.8rem', padding: '.45rem 1rem' }}>
                    <List size={14} /> Create Agenda
                  </button>
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <SectionHeader icon={<List size={14} />} label="Agenda Items" count={romData.stage3.agendas.length} color="hsl(140,70%,50%)" />
                    <div style={{ display: 'flex', gap: 6 }}>
                      <button onClick={() => runCreateAgenda(true)} disabled={stage3AgendaStatus === 'processing'} style={{ padding: '.3rem .65rem', borderRadius: 7, border: '1px solid hsl(140,70%,45%/.4)', background: 'hsl(140,70%,45%/.08)', color: 'hsl(140,70%,45%)', fontSize: '.72rem', fontWeight: 600, display: 'flex', alignItems: 'center', gap: 4, cursor: 'pointer', fontFamily: 'Inter' }}>
                        {stage3AgendaStatus === 'processing' ? <Loader size={11} className="spin" /> : <RefreshCw size={11} />}
                        Regenerate
                      </button>
                      <button onClick={() => downloadDocx('stage3')} style={{ padding: '.3rem .65rem', borderRadius: 7, border: '1px solid hsl(var(--border)/.4)', background: 'hsl(var(--muted)/.3)', color: 'hsl(var(--pencil))', fontWeight: 600, fontSize: '.72rem', display: 'flex', alignItems: 'center', gap: 4, cursor: 'pointer', fontFamily: 'Inter' }}>
                        <Download size={11} /> Export
                      </button>
                    </div>
                  </div>

                  <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                    {romData.stage3.agendas.map((agenda: any, aIdx: number) => {
                      const agendaId = agenda.agenda_id || `A${aIdx + 1}`
                      const docEntry = romData.stage3?.agenda_doc_points?.[agendaId]
                      const isUploading = agendaDocUploading[agendaId] || false
                      const expanded = romData.stage3?.expanded_agendas?.find((ea: any) => ea.agenda_id === agendaId)
                      const isEditingThisAgenda = editingAgendaId === agendaId
                      const presenterName = agenda.presenter || agenda.speaker || docEntry?.presenter

                      return (
                        <div key={agendaId} style={{ borderRadius: 10, border: '1.5px solid hsl(var(--border)/.4)', background: 'hsl(var(--card))', overflow: 'hidden' }}>
                          {/* Agenda Header */}
                          <div style={{ background: 'hsl(140,70%,50%/.08)', padding: '.75rem 1rem', borderBottom: '1px solid hsl(var(--border)/.3)', display: 'flex', alignItems: 'flex-start', gap: 8 }}>
                            <span style={{ background: 'hsl(140,70%,45%)', color: 'white', fontWeight: 700, fontSize: '.72rem', padding: '2px 7px', borderRadius: 5, flexShrink: 0, marginTop: 2 }}>{agendaId}</span>
                            <div style={{ flex: 1 }}>
                              {isEditingThisAgenda ? (
                                <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                                  <input
                                    value={editAgendaTitle}
                                    onChange={e => setEditAgendaTitle(e.target.value)}
                                    placeholder="Agenda Title"
                                    style={{ width: '100%', padding: '4px 8px', borderRadius: 6, border: '1.5px solid hsl(140,70%,45%)', fontSize: '.88rem', fontWeight: 700, fontFamily: 'Inter', background: 'hsl(var(--paper))', color: 'hsl(var(--ink))' }}
                                  />
                                  <textarea
                                    value={editAgendaDescription}
                                    onChange={e => setEditAgendaDescription(e.target.value)}
                                    placeholder="Agenda Description / Details"
                                    style={{ width: '100%', minHeight: 50, padding: '4px 8px', borderRadius: 6, border: '1.5px solid hsl(140,70%,45%)', fontSize: '.78rem', fontFamily: 'Inter', background: 'hsl(var(--paper))', color: 'hsl(var(--ink))' }}
                                  />
                                  <div style={{ display: 'flex', gap: 6, justifyContent: 'flex-end', marginTop: 2 }}>
                                    <button onClick={() => setEditingAgendaId(null)} style={{ padding: '2px 8px', fontSize: '.72rem', borderRadius: 4, border: '1px solid hsl(var(--border))', background: 'transparent', cursor: 'pointer' }}>Cancel</button>
                                    <button
                                      disabled={savingAgendas}
                                      onClick={async () => {
                                        const updated = JSON.parse(JSON.stringify(romData!.stage3!.agendas))
                                        updated[aIdx].title = editAgendaTitle
                                        updated[aIdx].description = editAgendaDescription
                                        await saveStage3Agendas(updated)
                                        setEditingAgendaId(null)
                                      }}
                                      style={{ padding: '2px 10px', fontSize: '.72rem', borderRadius: 4, background: 'hsl(140,70%,45%)', color: 'white', border: 'none', fontWeight: 700, cursor: 'pointer' }}
                                    >
                                      {savingAgendas ? 'Saving...' : 'Save Agenda'}
                                    </button>
                                  </div>
                                </div>
                              ) : (
                                <div>
                                  <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                                    <div style={{ fontSize: '.92rem', fontWeight: 700, color: 'hsl(var(--ink))' }}>{agenda.title}</div>
                                    <button
                                      onClick={() => {
                                        setEditingAgendaId(agendaId)
                                        setEditAgendaTitle(agenda.title || '')
                                        setEditAgendaDescription(agenda.description || '')
                                      }}
                                      title="Edit Agenda Title & Description"
                                      style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'hsl(var(--pencil))', padding: '2px', display: 'flex', alignItems: 'center' }}
                                    >
                                      <Pencil size={12} />
                                    </button>
                                  </div>
                                  {agenda.description && <div style={{ fontSize: '.73rem', color: 'hsl(var(--pencil))', marginTop: 2, lineHeight: 1.35 }}>{agenda.description}</div>}
                                  {presenterName && (
                                    <div style={{ marginTop: 4, display: 'inline-flex', alignItems: 'center', gap: 4, background: 'hsl(280,75%,60%/.1)', color: 'hsl(280,75%,60%)', border: '1px solid hsl(280,75%,60%/.3)', padding: '1px 7px', borderRadius: 6, fontSize: '.68rem', fontWeight: 600 }}>
                                      <User size={10} /> Presenter: {presenterName}
                                    </div>
                                  )}
                                </div>
                              )}
                            </div>
                          </div>

                          <div style={{ padding: '.85rem 1rem', display: 'flex', flexDirection: 'column', gap: '.75rem' }}>

                            {/* Context Chips from RAG */}
                            {expanded?.meeting_context_snippet && (
                              <details style={{ background: 'transparent' }}>
                                <summary style={{ cursor: 'pointer', fontSize: '.72rem', fontWeight: 600, color: 'hsl(205,90%,55%)', outline: 'none', display: 'flex', alignItems: 'center', gap: 5 }}>
                                  <FileText size={11} /> View Retrieved Context
                                </summary>
                                <div style={{ marginTop: '.4rem', padding: '.5rem .65rem', borderRadius: 7, background: 'hsl(205,90%,55%/.06)', border: '1px solid hsl(205,90%,55%/.2)', fontSize: '.73rem', color: 'hsl(var(--ink))', lineHeight: 1.45 }}>
                                  {expanded.meeting_context_snippet.slice(0, 500)}{expanded.meeting_context_snippet.length > 500 ? '…' : ''}
                                </div>
                              </details>
                            )}

                            {/* Keywords */}
                            {agenda.keywords?.length > 0 && (
                              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 3 }}>
                                {agenda.keywords.slice(0, 8).map((k: string, ki: number) => (
                                  <span key={ki} style={{ fontSize: '.65rem', padding: '1px 5px', borderRadius: 4, background: 'hsl(140,70%,45%/.1)', color: 'hsl(140,70%,40%)', border: '1px solid hsl(140,70%,45%/.25)' }}>{k}</span>
                                ))}
                              </div>
                            )}

                            {/* Supporting Document Upload */}
                            <div style={{ borderTop: '1px solid hsl(var(--border)/.3)', paddingTop: '.65rem' }}>
                              <div style={{ fontSize: '.68rem', fontWeight: 700, color: 'hsl(var(--pencil))', textTransform: 'uppercase', letterSpacing: '.04em', marginBottom: 6 }}>Supporting Document</div>

                              {docEntry ? (
                                <div style={{ display: 'flex', flexDirection: 'column', gap: '.45rem' }}>
                                  {/* Doc info header */}
                                  <div style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '.35rem .55rem', borderRadius: 7, background: 'hsl(280,75%,60%/.08)', border: '1px solid hsl(280,75%,60%/.25)' }}>
                                    <FileText size={11} style={{ color: 'hsl(280,75%,60%)', flexShrink: 0 }} />
                                    <span style={{ fontSize: '.72rem', fontWeight: 600, color: 'hsl(var(--ink))', flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{docEntry.doc_name}</span>
                                    <span style={{ fontSize: '.65rem', fontWeight: 700, color: 'hsl(280,75%,60%)', background: 'hsl(280,75%,60%/.12)', padding: '1px 6px', borderRadius: 8 }}>{docEntry.points.length} pts</span>
                                    <button
                                      onClick={() => agendaDocInputRefs.current[agendaId]?.click()}
                                      disabled={isUploading}
                                      title="Replace document"
                                      style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'hsl(var(--pencil))', padding: '0 2px' }}
                                    >
                                      <RefreshCw size={10} />
                                    </button>
                                  </div>
                                  {/* Extracted Points */}
                                  {docEntry.points.length > 0 && (
                                    <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                                      {docEntry.points.map((point: string, pIdx: number) => (
                                        <div key={pIdx} style={{ display: 'flex', gap: 6, padding: '.4rem .55rem', borderRadius: 6, background: 'hsl(var(--paper)/.5)', border: '1px solid hsl(var(--border)/.25)' }}>
                                          <span style={{ fontSize: '.66rem', fontWeight: 700, color: 'hsl(280,75%,65%)', flexShrink: 0, marginTop: 1, fontFamily: 'JetBrains Mono' }}>D{pIdx + 1}</span>
                                          <span style={{ fontSize: '.78rem', color: 'hsl(var(--ink))', lineHeight: 1.4 }}>{point}</span>
                                        </div>
                                      ))}
                                    </div>
                                  )}
                                </div>
                              ) : (
                                <div
                                  onClick={() => !isUploading && agendaDocInputRefs.current[agendaId]?.click()}
                                  style={{ border: '1.5px dashed hsl(280,75%,60%/.35)', borderRadius: 7, padding: '.45rem', textAlign: 'center', cursor: isUploading ? 'wait' : 'pointer', fontSize: '.72rem', color: 'hsl(280,75%,55%)', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 5, background: 'hsl(280,75%,60%/.04)' }}
                                >
                                  {isUploading ? <Loader size={10} className="spin" /> : <Upload size={10} />}
                                  {isUploading ? 'Extracting points...' : 'Upload Supporting Document'}
                                </div>
                              )}

                              <input
                                ref={el => { agendaDocInputRefs.current[agendaId] = el }}
                                type="file"
                                accept=".pdf,.docx,.pptx,.txt,.md,.xlsx,.xls,.csv"
                                style={{ display: 'none' }}
                                onChange={e => {
                                  const file = e.target.files?.[0]
                                  if (file) handleAgendaDocUpload(agendaId, file)
                                  if (e.target) e.target.value = ''
                                }}
                              />
                            </div>
                          </div>
                        </div>
                      )
                    })}
                  </div>

                  {/* Generated MOM Section */}
                  {(enhancedMomData || romData?.stage3?.enhanced_mom) && (() => {
                    const mom = enhancedMomData || romData.stage3.enhanced_mom
                    return (
                      <div style={{ borderRadius: 10, border: '1.5px solid hsl(140,70%,45%/.4)', background: 'hsl(var(--card))', padding: '1rem 1.15rem', display: 'flex', flexDirection: 'column', gap: '1rem', marginTop: '1rem' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <SectionHeader icon={<FileText size={14} />} label="Generated Minutes of Meeting (MOM)" color="hsl(140,70%,45%)" />
                          <button onClick={downloadEnhancedMomDocx} style={{ padding: '.35rem .75rem', borderRadius: 7, border: '1px solid hsl(140,70%,45%/.4)', background: 'hsl(140,70%,45%/.1)', color: 'hsl(140,70%,45%)', fontSize: '.74rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: 5, cursor: 'pointer', fontFamily: 'Inter' }}>
                            <Download size={12} /> Download MOM (.docx)
                          </button>
                        </div>
                        {mom.introduction && (
                          <div style={{ padding: '.75rem', borderRadius: 8, background: 'hsl(var(--muted)/.2)', fontSize: '.84rem', lineHeight: 1.5, color: 'hsl(var(--ink))' }}>
                            <strong>Introduction:</strong> {mom.introduction}
                          </div>
                        )}
                        {mom.points_discussed?.length > 0 && (
                          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                            <div style={{ fontSize: '.78rem', fontWeight: 700, color: 'hsl(var(--ink))' }}>Key Discussion Points:</div>
                            {mom.points_discussed.map((pt: any, idx: number) => (
                              <div key={idx} style={{ padding: '.6rem .75rem', borderRadius: 7, border: '1px solid hsl(var(--border)/.3)', background: 'hsl(var(--paper)/.4)', fontSize: '.82rem' }}>
                                <div style={{ fontWeight: 700, color: 'hsl(140,70%,40%)', marginBottom: 2 }}>{pt.topic || `Topic ${idx+1}`}</div>
                                <div>{pt.summary}</div>
                              </div>
                            ))}
                          </div>
                        )}
                        {mom.action_items?.length > 0 && (
                          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                            <div style={{ fontSize: '.78rem', fontWeight: 700, color: 'hsl(var(--ink))' }}>Action Items:</div>
                            <div style={{ overflowX: 'auto', borderRadius: 8, border: '1px solid hsl(var(--border)/.4)' }}>
                              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '.78rem' }}>
                                <thead>
                                  <tr style={{ background: 'hsl(var(--muted)/.4)', borderBottom: '1px solid hsl(var(--border)/.4)' }}>
                                    <th style={{ padding: '6px 8px', textAlign: 'left' }}>Item</th>
                                    <th style={{ padding: '6px 8px', textAlign: 'left', width: 120 }}>Owner</th>
                                    <th style={{ padding: '6px 8px', textAlign: 'left', width: 90 }}>Deadline</th>
                                  </tr>
                                </thead>
                                <tbody>
                                  {mom.action_items.map((act: any, idx: number) => (
                                    <tr key={idx} style={{ borderBottom: '1px solid hsl(var(--border)/.2)' }}>
                                      <td style={{ padding: '6px 8px' }}>{act.item || act.description || '-'}</td>
                                      <td style={{ padding: '6px 8px', fontWeight: 600 }}>{act.owner || 'Unassigned'}</td>
                                      <td style={{ padding: '6px 8px', color: 'hsl(var(--pencil))' }}>{act.deadline || 'ASAP'}</td>
                                    </tr>
                                  ))}
                                </tbody>
                              </table>
                            </div>
                          </div>
                        )}
                        {mom.conclusion && (
                          <div style={{ padding: '.6rem .75rem', borderRadius: 8, background: 'hsl(var(--muted)/.2)', fontSize: '.82rem', color: 'hsl(var(--ink))' }}>
                            <strong>Conclusion:</strong> {mom.conclusion}
                          </div>
                        )}
                      </div>
                    )
                  })()}
                </div>
              )
            )}



          {/* ── FINAL ROM TAB ── */}
          {activeTab === 'final' && (
            !romData?.final_rom?.agendas || romData.final_rom.agendas.length === 0 ? (
              <div style={{ textAlign: 'center', padding: '4rem 1.5rem', color: 'hsl(var(--pencil))' }}>
                <Sparkles size={42} style={{ margin: '0 auto 1rem', opacity: 0.4, color: 'hsl(30,90%,55%)' }} />
                <div style={{ fontSize: '1.05rem', fontWeight: 700, color: 'hsl(var(--ink))', marginBottom: '.4rem' }}>Final Record of Meeting Not Generated Yet</div>
                <div style={{ fontSize: '.84rem', maxWidth: 420, margin: '0 auto 1.25rem', lineHeight: 1.45 }}>
                  Run Stage 3 on the left panel to map discussion points to agendas and generate the Final ROM.
                </div>
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
                {/* Sticky Header Bar for Final ROM */}
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: 'hsl(var(--muted)/.3)', padding: '.65rem 1rem', borderRadius: 9, border: '1px solid hsl(var(--border)/.4)' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                    <SectionHeader icon={<Sparkles size={14} />} label={romViewMode === 'precise' ? 'Precise Record of Meeting' : 'Final Record of Meeting'} count={finalCount} color={romViewMode === 'precise' ? 'hsl(280,75%,60%)' : 'hsl(30,90%,55%)'} />
                    <div style={{ display: 'flex', borderRadius: 6, border: '1px solid hsl(var(--border)/.6)', background: 'hsl(var(--muted)/.4)', padding: 2, gap: 2 }}>
                      <button
                        onClick={() => setRomViewMode('standard')}
                        style={{
                          padding: '3px 8px', border: 'none', borderRadius: 4, cursor: 'pointer',
                          fontSize: '.72rem', fontWeight: romViewMode === 'standard' ? 700 : 500,
                          background: romViewMode === 'standard' ? 'hsl(30,90%,55%)' : 'transparent',
                          color: romViewMode === 'standard' ? 'white' : 'hsl(var(--pencil))',
                          transition: 'all .15s', fontFamily: 'Inter'
                        }}
                      >
                        Standard ROM
                      </button>
                      <button
                        onClick={() => setRomViewMode('precise')}
                        style={{
                          padding: '3px 8px', border: 'none', borderRadius: 4, cursor: 'pointer',
                          fontSize: '.72rem', fontWeight: romViewMode === 'precise' ? 700 : 500,
                          background: romViewMode === 'precise' ? 'hsl(280,75%,60%)' : 'transparent',
                          color: romViewMode === 'precise' ? 'white' : 'hsl(var(--pencil))',
                          transition: 'all .15s', fontFamily: 'Inter'
                        }}
                      >
                        Precise ROM
                      </button>
                    </div>
                  </div>
                  <div style={{ display: 'flex', gap: 8 }}>
                    <button
                      onClick={() => saveFinalRom(romData.final_rom)}
                      disabled={savingRom}
                      style={{
                        padding: '.35rem .85rem', borderRadius: 8,
                        background: 'hsl(280,75%,60%)', color: 'white', fontWeight: 700,
                        fontSize: '.76rem', display: 'flex', alignItems: 'center', gap: 6, cursor: savingRom ? 'not-allowed' : 'pointer',
                        border: 'none', fontFamily: 'Inter'
                      }}
                    >
                      {savingRom ? <Loader size={12} className="spin" /> : <Save size={12} />}
                      {savingRom ? 'Saving...' : 'Save Changes'}
                    </button>
                    <button
                      onClick={() => downloadDocx(romViewMode === 'precise' ? 'precise' : 'final')}
                      style={{
                        padding: '.35rem .85rem', borderRadius: 8,
                        background: 'hsl(205,90%,55%/.1)', color: 'hsl(205,90%,60%)', border: '1px solid hsl(205,90%,55%/.3)',
                        fontWeight: 700, fontSize: '.76rem', display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer', fontFamily: 'Inter'
                      }}
                    >
                      <FileDown size={12} /> Download {romViewMode === 'precise' ? 'Precise' : 'Final'} DOCX
                    </button>
                    <button
                      onClick={() => downloadDocx('agenda-transcript')}
                      style={{
                        padding: '.35rem .85rem', borderRadius: 8,
                        background: 'hsl(150,75%,40%/.1)', color: 'hsl(150,75%,40%)', border: '1px solid hsl(150,75%,40%/.3)',
                        fontWeight: 700, fontSize: '.76rem', display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer', fontFamily: 'Inter'
                      }}
                    >
                      <FileDown size={12} /> Download Agenda Transcript (.docx)
                    </button>

                  </div>
                </div>

                {/* ── Rewrite ROM Panel ── */}
                <div style={{ borderRadius: 9, border: `1.5px solid ${isRewritten ? 'hsl(38,90%,52%/.6)' : 'hsl(var(--border)/.4)'}`, background: isRewritten ? 'hsl(38,90%,52%/.04)' : 'hsl(var(--card))', overflow: 'hidden', transition: 'border-color .2s, background .2s' }}>

                  {/* Panel toggle header */}
                  <button
                    onClick={() => setShowRewritePanel(prev => !prev)}
                    style={{ width: '100%', background: isRewritten ? 'hsl(38,90%,52%/.1)' : 'hsl(var(--muted)/.3)', border: 'none', cursor: 'pointer', padding: '.6rem 1rem', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '.78rem', fontWeight: 700, color: 'hsl(var(--ink))' }}>
                      <WandSparkles size={13} style={{ color: 'hsl(38,90%,52%)' }} />
                      Rewrite ROM
                      {isRewritten && (
                        <span style={{ fontSize: '.65rem', padding: '1px 7px', borderRadius: 8, background: 'hsl(38,90%,52%/.18)', color: 'hsl(38,90%,40%)', border: '1px solid hsl(38,90%,52%/.4)', fontWeight: 700 }}>
                          ✦ Rewritten
                        </span>
                      )}
                      <span style={{ fontSize: '.63rem', padding: '1px 6px', borderRadius: 5, background: 'hsl(var(--muted))', color: 'hsl(var(--pencil))', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '.04em' }}>
                        {rewriteMode === 'window' ? `Window ×${rewriteWindowSize}` : rewriteMode === 'complete' ? 'Complete' : 'Reference'}
                      </span>
                    </div>
                    <span style={{ fontSize: '.68rem', color: 'hsl(var(--pencil))' }}>{showRewritePanel ? '▲' : '▼'}</span>
                  </button>

                  {showRewritePanel && (
                    <div style={{ padding: '.9rem 1rem', display: 'flex', flexDirection: 'column', gap: '.9rem' }}>

                      {/* Style-only info banner */}
                      <div style={{ fontSize: '.73rem', color: 'hsl(var(--pencil))', lineHeight: 1.5, padding: '.55rem .85rem', borderRadius: 7, background: 'hsl(205,90%,55%/.07)', border: '1px solid hsl(205,90%,55%/.2)' }}>
                        <strong style={{ color: 'hsl(var(--ink))' }}>Style-only rewrite</strong> — improves grammar, phrasing, and formatting.
                        Facts, speakers, decisions, dates, and action items are <strong style={{ color: 'hsl(var(--ink))' }}>never changed</strong>.
                        The original ROM is preserved and can be instantly restored.
                      </div>

                      {/* Rewritten notice bar */}
                      {isRewritten && (
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '.5rem .85rem', borderRadius: 7, background: 'hsl(38,90%,52%/.1)', border: '1px solid hsl(38,90%,52%/.35)' }}>
                          <WandSparkles size={12} style={{ color: 'hsl(38,90%,42%)', flexShrink: 0 }} />
                          <span style={{ fontSize: '.74rem', color: 'hsl(38,90%,35%)', fontWeight: 600, flex: 1 }}>
                            Showing rewritten version. Save to persist, or revert.
                          </span>
                          <button
                            onClick={revertToOriginal}
                            style={{ padding: '.28rem .7rem', borderRadius: 6, background: 'hsl(var(--destructive)/.08)', color: 'hsl(var(--destructive))', border: '1px solid hsl(var(--destructive)/.3)', fontSize: '.71rem', fontWeight: 700, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 4, flexShrink: 0, fontFamily: 'Inter' }}
                          >
                            <RotateCcw size={10} /> Revert
                          </button>
                        </div>
                      )}

                      {/* ── Mode Selector ── */}
                      <div>
                        <div style={{ fontSize: '.72rem', fontWeight: 700, color: 'hsl(var(--ink))', marginBottom: '.45rem', textTransform: 'uppercase', letterSpacing: '.05em' }}>Rewrite Mode</div>
                        <div style={{ display: 'flex', gap: 6 }}>
                          {([
                            { id: 'window', label: 'Window Rewrite', desc: 'Batch points into windows' },
                            { id: 'complete', label: 'Complete Rewrite', desc: 'Single call, best consistency' },
                            { id: 'reference', label: 'Reference-Based', desc: 'Learn style from a sample doc' },
                          ] as const).map(m => (
                            <button
                              key={m.id}
                              onClick={() => setRewriteMode(m.id)}
                              style={{
                                flex: 1, padding: '.45rem .6rem', borderRadius: 7, cursor: 'pointer',
                                border: rewriteMode === m.id ? '2px solid hsl(38,90%,52%)' : '1.5px solid hsl(var(--border)/.5)',
                                background: rewriteMode === m.id ? 'hsl(38,90%,52%/.1)' : 'hsl(var(--muted)/.3)',
                                transition: 'all .15s', textAlign: 'left' as const,
                              }}
                            >
                              <div style={{ fontSize: '.73rem', fontWeight: 700, color: rewriteMode === m.id ? 'hsl(38,90%,42%)' : 'hsl(var(--ink))' }}>{m.label}</div>
                              <div style={{ fontSize: '.63rem', color: 'hsl(var(--pencil))', marginTop: 1 }}>{m.desc}</div>
                            </button>
                          ))}
                        </div>
                      </div>

                      {/* Window Size (window / reference modes) */}
                      {(rewriteMode === 'window' || rewriteMode === 'reference') && (
                        <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '.55rem .85rem', borderRadius: 7, background: 'hsl(var(--muted)/.2)', border: '1px solid hsl(var(--border)/.4)' }}>
                          <div style={{ flex: 1 }}>
                            <div style={{ fontSize: '.73rem', fontWeight: 700, color: 'hsl(var(--ink))' }}>Window Size</div>
                            <div style={{ fontSize: '.65rem', color: 'hsl(var(--pencil))', marginTop: 1 }}>Discussion points processed per LLM call (1–10)</div>
                          </div>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                            <button
                              onClick={() => setRewriteWindowSize(v => Math.max(1, v - 1))}
                              style={{ width: 26, height: 26, borderRadius: 6, border: '1px solid hsl(var(--border))', background: 'hsl(var(--muted)/.4)', cursor: 'pointer', fontSize: '.9rem', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'hsl(var(--ink))' }}
                            >−</button>
                            <span style={{ fontFamily: 'JetBrains Mono', fontSize: '.9rem', fontWeight: 700, minWidth: 22, textAlign: 'center', color: 'hsl(38,90%,45%)' }}>{rewriteWindowSize}</span>
                            <button
                              onClick={() => setRewriteWindowSize(v => Math.min(10, v + 1))}
                              style={{ width: 26, height: 26, borderRadius: 6, border: '1px solid hsl(var(--border))', background: 'hsl(var(--muted)/.4)', cursor: 'pointer', fontSize: '.9rem', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'hsl(var(--ink))' }}
                            >+</button>
                          </div>
                        </div>
                      )}

                      {/* Reference Document Upload (reference mode only) */}
                      {rewriteMode === 'reference' && (
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '.65rem', padding: '.7rem .85rem', borderRadius: 8, border: '1.5px dashed hsl(280,75%,60%/.4)', background: 'hsl(280,75%,60%/.04)' }}>
                          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                            <div>
                              <div style={{ fontSize: '.75rem', fontWeight: 700, color: 'hsl(var(--ink))', display: 'flex', alignItems: 'center', gap: 5 }}>
                                <FileUp size={13} style={{ color: 'hsl(280,75%,60%)' }} /> Upload Reference Document
                              </div>
                              <div style={{ fontSize: '.65rem', color: 'hsl(var(--pencil))', marginTop: 2 }}>
                                Upload a sample MoM/ROM (.pdf, .docx, .txt) to learn its writing style
                              </div>
                            </div>
                            <button
                              onClick={() => referenceFileInputRef.current?.click()}
                              disabled={extractingRules}
                              style={{ padding: '.32rem .8rem', borderRadius: 6, background: extractingRules ? 'hsl(var(--muted))' : 'hsl(280,75%,60%)', color: 'white', border: 'none', fontSize: '.72rem', fontWeight: 700, cursor: extractingRules ? 'not-allowed' : 'pointer', display: 'flex', alignItems: 'center', gap: 5, flexShrink: 0, fontFamily: 'Inter' }}
                            >
                              {extractingRules ? <Loader size={11} className="spin" /> : <FileUp size={11} />}
                              {extractingRules ? 'Analyzing...' : 'Upload & Analyze'}
                            </button>
                            <input
                              ref={referenceFileInputRef}
                              type="file"
                              accept=".pdf,.docx,.txt,.doc"
                              style={{ display: 'none' }}
                              onChange={e => handleReferenceDocUpload(e.target.files)}
                            />
                          </div>

                          {/* Writing Rules textarea */}
                          <div>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '.3rem' }}>
                              <div style={{ fontSize: '.72rem', fontWeight: 700, color: 'hsl(var(--ink))' }}>
                                Writing Rules {writingRules ? <span style={{ color: 'hsl(280,75%,60%)', fontWeight: 600 }}>— Editable</span> : <span style={{ color: 'hsl(var(--pencil))', fontWeight: 500 }}>— Upload a document to generate</span>}
                              </div>
                              {writingRules && (
                                <button onClick={() => setWritingRules('')} style={{ background: 'none', border: 'none', color: 'hsl(var(--pencil))', fontSize: '.68rem', cursor: 'pointer', textDecoration: 'underline' }}>Clear</button>
                              )}
                            </div>
                            <textarea
                              value={writingRules}
                              onChange={e => setWritingRules(e.target.value)}
                              rows={8}
                              placeholder="Writing rules will appear here after uploading a reference document. You can also type rules manually."
                              style={{ width: '100%', padding: '.6rem .75rem', borderRadius: 7, border: `1px solid ${writingRules ? 'hsl(280,75%,60%/.5)' : 'hsl(var(--border))'}`, background: 'hsl(var(--muted)/.25)', fontSize: '.76rem', fontFamily: 'Inter', color: 'hsl(var(--ink))', resize: 'vertical', lineHeight: 1.55, boxSizing: 'border-box', transition: 'border-color .2s' }}
                            />
                            {writingRules && (
                              <div style={{ fontSize: '.65rem', color: 'hsl(280,75%,55%)', marginTop: 3 }}>
                                ✓ {writingRules.split('\n').filter(l => l.trim()).length} rule{writingRules.split('\n').filter(l => l.trim()).length !== 1 ? 's' : ''} — will be injected into every rewrite call
                              </div>
                            )}
                          </div>
                        </div>
                      )}

                      {/* Instruction textarea */}
                      <div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '.3rem' }}>
                          <div style={{ fontSize: '.73rem', fontWeight: 700, color: 'hsl(var(--ink))' }}>Rewrite Instruction</div>
                          {rewriteInstruction !== DEFAULT_REWRITE_INSTRUCTION && (
                            <button onClick={() => setRewriteInstruction(DEFAULT_REWRITE_INSTRUCTION)} style={{ background: 'none', border: 'none', color: 'hsl(var(--pencil))', fontSize: '.68rem', cursor: 'pointer', textDecoration: 'underline' }}>Reset to default</button>
                          )}
                        </div>
                        <textarea
                          value={rewriteInstruction}
                          onChange={e => setRewriteInstruction(e.target.value)}
                          rows={4}
                          style={{ width: '100%', padding: '.6rem .75rem', borderRadius: 7, border: '1px solid hsl(var(--border))', background: 'hsl(var(--muted)/.3)', fontSize: '.78rem', fontFamily: 'Inter', color: 'hsl(var(--ink))', resize: 'vertical', lineHeight: 1.5, boxSizing: 'border-box' }}
                        />
                      </div>

                      {/* Validation warning for reference mode */}
                      {rewriteMode === 'reference' && !writingRules.trim() && (
                        <div style={{ fontSize: '.72rem', color: 'hsl(38,90%,42%)', padding: '.4rem .75rem', borderRadius: 6, background: 'hsl(38,90%,52%/.08)', border: '1px solid hsl(38,90%,52%/.3)' }}>
                          ⚠ Upload a reference document or enter writing rules manually before running reference-based rewrite.
                        </div>
                      )}

                      {/* Action buttons */}
                      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
                        <button
                          onClick={runRewriteRom}
                          disabled={
                            rewriteStatus === 'processing' ||
                            !rewriteInstruction.trim() ||
                            (rewriteMode === 'reference' && !writingRules.trim())
                          }
                          style={{
                            padding: '.42rem 1.1rem', borderRadius: 7,
                            background: (
                              rewriteStatus === 'processing' ||
                              !rewriteInstruction.trim() ||
                              (rewriteMode === 'reference' && !writingRules.trim())
                            ) ? 'hsl(var(--muted))' : 'linear-gradient(135deg, hsl(38,90%,52%), hsl(25,90%,55%))',
                            color: 'white', fontWeight: 700, fontSize: '.78rem',
                            border: 'none',
                            cursor: (
                              rewriteStatus === 'processing' ||
                              !rewriteInstruction.trim() ||
                              (rewriteMode === 'reference' && !writingRules.trim())
                            ) ? 'not-allowed' : 'pointer',
                            display: 'flex', alignItems: 'center', gap: 6, fontFamily: 'Inter',
                            boxShadow: '0 2px 8px rgba(0,0,0,0.12)', transition: 'all .15s'
                          }}
                        >
                          {rewriteStatus === 'processing' ? <Loader size={12} className="spin" /> : <WandSparkles size={12} />}
                          {rewriteStatus === 'processing' ? 'Rewriting ROM...' : (isRewritten ? 'Re-Rewrite ROM' : 'Rewrite ROM')}
                        </button>

                        {isRewritten && (
                          <button
                            onClick={revertToOriginal}
                            style={{ padding: '.42rem .95rem', borderRadius: 7, background: 'transparent', color: 'hsl(var(--destructive))', border: '1.5px solid hsl(var(--destructive)/.4)', fontSize: '.78rem', fontWeight: 700, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 5, fontFamily: 'Inter' }}
                          >
                            <RotateCcw size={12} /> Revert to Original
                          </button>
                        )}

                        <div style={{ marginLeft: 'auto', fontSize: '.68rem', color: 'hsl(var(--pencil))', lineHeight: 1.3, textAlign: 'right' as const }}>
                          {rewriteMode === 'window' && <span>~{Math.ceil((romData?.final_rom?.agendas?.reduce((s: number, a: any) => s + (a.discussion_points?.length || 0), 0) || 0) / rewriteWindowSize)} LLM calls</span>}
                          {rewriteMode === 'complete' && <span>1 LLM call</span>}
                          {rewriteMode === 'reference' && <span>~{Math.ceil((romData?.final_rom?.agendas?.reduce((s: number, a: any) => s + (a.discussion_points?.length || 0), 0) || 0) / rewriteWindowSize)} LLM calls + rules</span>}
                        </div>
                      </div>

                    </div>
                  )}
                </div>


                {/* Speaker Name Mapping Panel */}

                <div style={{ borderRadius: 9, border: '1.5px solid hsl(var(--border)/.4)', background: 'hsl(var(--card))', overflow: 'hidden' }}>
                  <button
                    onClick={() => setShowSpeakerMapping(prev => !prev)}
                    style={{ width: '100%', background: 'hsl(var(--muted)/.3)', border: 'none', cursor: 'pointer', padding: '.6rem 1rem', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '.78rem', fontWeight: 700, color: 'hsl(var(--ink))' }}>
                      <Tag size={13} style={{ color: 'hsl(280,75%,60%)' }} /> Speaker Name Mapping
                      {speakerMappings.length > 0 && <span style={{ fontSize: '.65rem', padding: '1px 6px', borderRadius: 8, background: 'hsl(280,75%,60%/.12)', color: 'hsl(280,75%,60%)', border: '1px solid hsl(280,75%,60%/.3)' }}>{speakerMappings.length} mapped</span>}
                    </div>
                    <span style={{ fontSize: '.68rem', color: 'hsl(var(--pencil))' }}>{showSpeakerMapping ? '▲' : '▼'}</span>
                  </button>
                  {showSpeakerMapping && (
                    <div style={{ padding: '.85rem 1rem', display: 'flex', flexDirection: 'column', gap: '.65rem' }}>
                      <div style={{ fontSize: '.72rem', color: 'hsl(var(--pencil))', lineHeight: 1.35 }}>Map Speaker IDs (e.g. Speaker_1) to real names. Click Apply to update throughout the ROM.</div>
                      {speakerMappings.map((m, idx) => (
                        <div key={idx} style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                          <input
                            value={m.speaker_id}
                            onChange={e => setSpeakerMappings(prev => prev.map((x, i) => i === idx ? { ...x, speaker_id: e.target.value } : x))}
                            placeholder="Speaker_1"
                            style={{ flex: 1, padding: '4px 8px', borderRadius: 6, border: '1px solid hsl(var(--border))', background: 'hsl(var(--muted)/.4)', fontSize: '.75rem', fontFamily: 'JetBrains Mono', color: 'hsl(var(--ink))' }}
                          />
                          <span style={{ fontSize: '.72rem', color: 'hsl(var(--pencil))' }}>→</span>
                          <input
                            value={m.real_name}
                            onChange={e => setSpeakerMappings(prev => prev.map((x, i) => i === idx ? { ...x, real_name: e.target.value } : x))}
                            placeholder="Real Name"
                            style={{ flex: 1, padding: '4px 8px', borderRadius: 6, border: '1px solid hsl(var(--border))', background: 'hsl(var(--muted)/.4)', fontSize: '.75rem', fontFamily: 'Inter', color: 'hsl(var(--ink))' }}
                          />
                          <button onClick={() => setSpeakerMappings(prev => prev.filter((_, i) => i !== idx))} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'hsl(var(--destructive))', padding: '0 3px' }}><X size={12} /></button>
                        </div>
                      ))}
                      <div style={{ display: 'flex', gap: 6 }}>
                        <button onClick={() => setSpeakerMappings(prev => [...prev, { speaker_id: '', real_name: '' }])} style={{ padding: '.3rem .65rem', borderRadius: 6, border: '1px dashed hsl(var(--border))', background: 'transparent', fontSize: '.72rem', color: 'hsl(var(--pencil))', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 4 }}>
                          <Plus size={10} /> Add Mapping
                        </button>
                        <button onClick={applySpeakerMappings} style={{ padding: '.3rem .85rem', borderRadius: 6, background: 'hsl(280,75%,60%)', color: 'white', border: 'none', fontSize: '.72rem', fontWeight: 700, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 4 }}>
                          <Tag size={10} /> Apply Names
                        </button>
                      </div>
                    </div>
                  )}
                </div>

                {/* Final ROM agendas */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                  {romData.final_rom.agendas.map((agenda: any, aIdx: number) => {
                    const pts = agenda.discussion_points || []
                    return (
                      <div key={agenda.agenda_id || aIdx} style={{ borderRadius: 10, border: '1.5px solid hsl(var(--border)/.4)', background: 'hsl(var(--card))', overflow: 'hidden' }}>
                        {/* Agenda header */}
                        <div style={{ background: 'hsl(30,90%,55%/.08)', padding: '.65rem 1rem', borderBottom: '1px solid hsl(var(--border)/.3)', display: 'flex', alignItems: 'center', gap: 8 }}>
                          <span style={{ background: 'hsl(30,90%,55%)', color: 'white', fontWeight: 700, fontSize: '.7rem', padding: '2px 7px', borderRadius: 5, flexShrink: 0, fontFamily: 'JetBrains Mono' }}>{agenda.agenda_id || `A${aIdx + 1}`}</span>
                          <div style={{ flex: 1 }}>
                            <div style={{ fontSize: '.92rem', fontWeight: 700, color: 'hsl(var(--ink))' }}>{agenda.title}</div>
                            {agenda.description && <div style={{ fontSize: '.72rem', color: 'hsl(var(--pencil))', marginTop: 1 }}>{agenda.description}</div>}
                            {(agenda.presenter || agenda.speaker) && (
                              <div style={{ marginTop: 3, display: 'inline-flex', alignItems: 'center', gap: 4, background: 'hsl(280,75%,60%/.1)', color: 'hsl(280,75%,60%)', border: '1px solid hsl(280,75%,60%/.3)', padding: '1px 6px', borderRadius: 6, fontSize: '.67rem', fontWeight: 600 }}>
                                <User size={10} /> Presenter: {agenda.presenter || agenda.speaker}
                              </div>
                            )}
                          </div>
                          <span style={{ fontSize: '.7rem', fontWeight: 700, color: 'hsl(30,90%,55%)', background: 'hsl(30,90%,55%/.1)', padding: '2px 7px', borderRadius: 8 }}>{pts.length} pts</span>
                        </div>

                        {/* Points */}
                        <div style={{ padding: '.75rem 1rem', display: 'flex', flexDirection: 'column', gap: 6 }}>
                          {pts.length === 0 ? (
                            <div style={{ color: 'hsl(var(--pencil))', fontStyle: 'italic', fontSize: '.78rem' }}>No discussion points mapped to this agenda.</div>
                          ) : (
                            pts.map((pt: any, pIdx: number) => {
                              const isEditing = editingPointId === pt.id
                              const ptText = pt.text || pt.polished_text || ''
                              const spk = pt.speaker || (pt.speakers?.length ? pt.speakers.join(', ') : '—')
                              const isDocPoint = pt.is_doc_point

                              return (
                                <div key={pt.id || pIdx} style={{ borderRadius: 8, border: `1px solid ${isDocPoint ? 'hsl(280,75%,60%/.3)' : 'hsl(var(--border)/.3)'}`, background: isDocPoint ? 'hsl(280,75%,60%/.05)' : 'hsl(var(--paper)/.4)', padding: '.6rem .8rem' }}>
                                  <div style={{ display: 'flex', gap: 8, alignItems: 'flex-start' }}>
                                    {/* Point label */}
                                    <span style={{ fontWeight: 700, color: isDocPoint ? 'hsl(280,75%,65%)' : 'hsl(280,75%,65%)', fontFamily: 'JetBrains Mono', fontSize: '.7rem', flexShrink: 0, marginTop: 2 }}>
                                      {isDocPoint ? 'D' : `P${pIdx + 1}`}
                                    </span>

                                    {/* Point content */}
                                    <div style={{ flex: 1 }}>
                                      {isEditing ? (
                                        <div>
                                          <textarea
                                            value={editDraft}
                                            onChange={e => setEditDraft(e.target.value)}
                                            style={{ width: '100%', minHeight: 55, padding: '4px 8px', borderRadius: 6, border: '1.5px solid hsl(280,75%,60%)', fontSize: '.8rem', fontFamily: 'Inter', outline: 'none', background: 'hsl(var(--paper))' }}
                                          />
                                          <div style={{ display: 'flex', gap: 4, marginTop: 4, justifyContent: 'flex-end' }}>
                                            <button onClick={() => setEditingPointId(null)} style={{ padding: '2px 8px', fontSize: '.7rem', borderRadius: 4, border: '1px solid hsl(var(--border))', background: 'transparent', cursor: 'pointer' }}>Cancel</button>
                                            <button
                                              onClick={() => {
                                                const newFinal = JSON.parse(JSON.stringify(romData.final_rom!))
                                                newFinal.agendas[aIdx].discussion_points[pIdx].text = editDraft
                                                setRomData(prev => ({ ...prev as RomData, final_rom: newFinal }))
                                                setEditingPointId(null)
                                                toast.success('Point updated')
                                              }}
                                              style={{ padding: '2px 8px', fontSize: '.7rem', borderRadius: 4, background: 'hsl(280,75%,60%)', color: 'white', border: 'none', fontWeight: 700, cursor: 'pointer' }}
                                            >
                                              Save
                                            </button>
                                          </div>
                                        </div>
                                      ) : romViewMode === 'precise' ? (
                                        <div style={{ color: 'hsl(var(--ink))', lineHeight: 1.5, fontSize: '.83rem', fontWeight: 500 }}>{formatPrecisePointText(pt)}</div>
                                      ) : (
                                        <div style={{ color: 'hsl(var(--ink))', lineHeight: 1.45, fontSize: '.83rem' }}>{ptText}</div>
                                      )}

                                      {/* Speaker */}
                                      <div style={{ marginTop: 4, fontSize: '.7rem', color: 'hsl(var(--pencil))', display: 'flex', alignItems: 'center', gap: 4 }}>
                                        <User size={10} />{spk}
                                        {isDocPoint && <span style={{ fontSize: '.64rem', padding: '1px 5px', borderRadius: 4, background: 'hsl(280,75%,60%/.1)', color: 'hsl(280,75%,60%)', border: '1px solid hsl(280,75%,60%/.25)' }}>From Document</span>}
                                      </div>
                                    </div>

                                    {/* Actions column */}
                                    {!isEditing && (
                                      <div style={{ display: 'flex', flexDirection: 'column', gap: 3, flexShrink: 0 }}>
                                        {/* Edit */}
                                        {romViewMode === 'standard' && (
                                          <button onClick={() => { setEditingPointId(pt.id); setEditDraft(ptText) }} title="Edit" style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'hsl(var(--pencil))', padding: '2px' }}><Pencil size={11} /></button>
                                        )}
                                        {/* Move Up */}
                                        <button onClick={() => reorderPoint(aIdx, pIdx, 'up')} disabled={pIdx === 0} title="Move up" style={{ background: 'none', border: 'none', cursor: pIdx === 0 ? 'default' : 'pointer', color: pIdx === 0 ? 'hsl(var(--border))' : 'hsl(var(--pencil))', padding: '2px' }}><ChevronUp size={11} /></button>
                                        {/* Move Down */}
                                        <button onClick={() => reorderPoint(aIdx, pIdx, 'down')} disabled={pIdx === pts.length - 1} title="Move down" style={{ background: 'none', border: 'none', cursor: pIdx === pts.length - 1 ? 'default' : 'pointer', color: pIdx === pts.length - 1 ? 'hsl(var(--border))' : 'hsl(var(--pencil))', padding: '2px' }}><ChevronDown size={11} /></button>
                                        {/* Delete Point */}
                                        <button onClick={() => handleDeletePoint(pt.id)} title="Delete Discussion Point" style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'hsl(var(--destructive))', padding: '2px' }}><Trash2 size={11} /></button>
                                        {/* Move to agenda */}
                                        <select
                                          title="Move to agenda"
                                          value=""
                                          onChange={e => {
                                            if (e.target.value) movePointToAgenda(aIdx, pIdx, e.target.value)
                                            e.target.value = ''
                                          }}
                                          style={{ background: 'none', border: '1px solid hsl(var(--border)/.5)', borderRadius: 4, cursor: 'pointer', color: 'hsl(var(--pencil))', fontSize: '.63rem', padding: '1px', maxWidth: 20, appearance: 'none', textAlign: 'center' }}
                                        >
                                          <option value="">↔</option>
                                          {romData.final_rom!.agendas.filter((a: any) => a.agenda_id !== agenda.agenda_id).map((a: any) => (
                                            <option key={a.agenda_id} value={a.agenda_id}>{a.agenda_id}: {a.title?.slice(0, 20)}</option>
                                          ))}
                                        </select>
                                      </div>
                                    )}
                                  </div>
                                </div>
                              )
                            })
                          )}
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>
            )
          )}

            {/* ── ADVANCED MOM TAB ── */}
            {activeTab === 'advanced' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem', maxWidth: 960, margin: '0 auto' }}>

                {/* Header banner */}
                <div style={{ padding: '1rem 1.25rem', borderRadius: 10, background: 'linear-gradient(135deg, hsl(330,85%,60%/.1), hsl(280,75%,60%/.1))', border: '1px solid hsl(330,85%,60%/.25)', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <div>
                    <div style={{ fontSize: '.95rem', fontWeight: 800, color: 'hsl(var(--ink))', display: 'flex', alignItems: 'center', gap: 6 }}>
                      <Sparkles size={16} style={{ color: 'hsl(330,85%,60%)' }} /> Advanced MoM Generation & Custom Refinement
                    </div>
                    <div style={{ fontSize: '.78rem', color: 'hsl(var(--pencil))', marginTop: 2 }}>
                      Provide custom instructions or pick a template to enhance Final MoM agenda points while preserving core facts, speakers, and action owners.
                    </div>
                  </div>
                  <button
                    onClick={() => navigate(`/dashboard/history/${id}/mom`)}
                    style={{ padding: '.4rem .85rem', borderRadius: 7, background: 'hsl(var(--accent))', color: 'white', border: 'none', fontSize: '.75rem', fontWeight: 700, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 5, flexShrink: 0 }}
                  >
                    <FileText size={12} /> View Main MoM Page
                  </button>
                </div>

                {/* Prompt Template Preset Selector */}
                <div>
                  <div style={{ fontSize: '.78rem', fontWeight: 700, color: 'hsl(var(--ink))', marginBottom: '.4rem', display: 'flex', alignItems: 'center', gap: 5 }}>
                    <Sliders size={13} style={{ color: 'hsl(330,85%,60%)' }} /> Built-in Prompt Templates
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(210px, 1fr))', gap: '.6rem' }}>
                    {BUILTIN_PROMPT_TEMPLATES.map(t => {
                      const isSelected = selectedTemplateId === t.id
                      return (
                        <div
                          key={t.id}
                          onClick={() => {
                            setSelectedTemplateId(t.id)
                            setAdvancedCustomPrompt(t.prompt)
                          }}
                          style={{
                            padding: '.65rem .85rem', borderRadius: 8,
                            border: isSelected ? '1.5px solid hsl(330,85%,60%)' : '1px solid hsl(var(--border)/.5)',
                            background: isSelected ? 'hsl(330,85%,60%/.08)' : 'hsl(var(--muted)/.2)',
                            cursor: 'pointer', transition: 'all .15s'
                          }}
                        >
                          <div style={{ fontSize: '.78rem', fontWeight: 700, color: isSelected ? 'hsl(330,85%,55%)' : 'hsl(var(--ink))' }}>
                            {t.name}
                          </div>
                          <div style={{ fontSize: '.68rem', color: 'hsl(var(--pencil))', marginTop: 2, lineHeight: 1.3 }}>
                            {t.description}
                          </div>
                        </div>
                      )
                    })}
                  </div>
                </div>

                {/* Custom Prompt Textarea */}
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '.4rem' }}>
                    <label style={{ fontSize: '.78rem', fontWeight: 700, color: 'hsl(var(--ink))' }}>
                      Custom LLM Instructions (No Character Limit)
                    </label>
                    {advancedCustomPrompt && (
                      <button
                        onClick={() => { setAdvancedCustomPrompt(''); setSelectedTemplateId(null) }}
                        style={{ background: 'none', border: 'none', color: 'hsl(var(--pencil))', fontSize: '.7rem', cursor: 'pointer', textDecoration: 'underline' }}
                      >
                        Clear prompt
                      </button>
                    )}
                  </div>
                  <textarea
                    value={advancedCustomPrompt}
                    onChange={e => {
                      setAdvancedCustomPrompt(e.target.value)
                      setSelectedTemplateId(null)
                    }}
                    placeholder="Enter custom instructions to refine agenda points (e.g. Focus on financial impact, format key takeaways as bullet points, use active executive tone...)"
                    rows={6}
                    style={{
                      width: '100%', padding: '.75rem', borderRadius: 8,
                      border: '1px solid hsl(var(--border))', background: 'hsl(var(--muted)/.3)',
                      fontSize: '.8rem', fontFamily: 'Inter', color: 'hsl(var(--ink))',
                      resize: 'vertical', lineHeight: 1.45
                    }}
                  />
                </div>

                {/* Selective Regeneration Checkboxes */}
                <div style={{ padding: '.85rem 1rem', borderRadius: 8, border: '1px solid hsl(var(--border)/.5)', background: 'hsl(var(--muted)/.15)', display: 'flex', flexWrap: 'wrap', gap: '1.25rem', alignItems: 'center' }}>
                  <div style={{ fontSize: '.75rem', fontWeight: 700, color: 'hsl(var(--pencil))', textTransform: 'uppercase', letterSpacing: '.04em' }}>
                    Selective Regeneration Options:
                  </div>
                  <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '.78rem', fontWeight: 600, color: 'hsl(var(--ink))', cursor: 'pointer' }}>
                    <input
                      type="checkbox"
                      checked={advancedRegenTitle}
                      onChange={e => setAdvancedRegenTitle(e.target.checked)}
                      style={{ accentColor: 'hsl(330,85%,60%)' }}
                    />
                    Regenerate Title
                  </label>
                  <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '.78rem', fontWeight: 600, color: 'hsl(var(--ink))', cursor: 'pointer' }}>
                    <input
                      type="checkbox"
                      checked={advancedRegenIntro}
                      onChange={e => setAdvancedRegenIntro(e.target.checked)}
                      style={{ accentColor: 'hsl(330,85%,60%)' }}
                    />
                    Regenerate Introduction
                  </label>
                  <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '.78rem', fontWeight: 600, color: 'hsl(var(--ink))', cursor: 'pointer' }}>
                    <input
                      type="checkbox"
                      checked={advancedRegenConclusion}
                      onChange={e => setAdvancedRegenConclusion(e.target.checked)}
                      style={{ accentColor: 'hsl(330,85%,60%)' }}
                    />
                    Regenerate Conclusion
                  </label>
                </div>

                {/* Generate Action Button */}
                <div>
                  <button
                    onClick={runGenerateAdvancedMom}
                    disabled={advancedStatus === 'processing' || !advancedCustomPrompt.trim()}
                    style={{
                      width: '100%', padding: '.65rem 1.25rem', borderRadius: 8,
                      background: (advancedStatus === 'processing' || !advancedCustomPrompt.trim())
                        ? 'hsl(var(--muted))'
                        : 'linear-gradient(135deg, hsl(330,85%,60%), hsl(280,75%,60%))',
                      color: 'white', fontWeight: 700, fontSize: '.84rem',
                      border: 'none', cursor: (advancedStatus === 'processing' || !advancedCustomPrompt.trim()) ? 'not-allowed' : 'pointer',
                      display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6,
                      boxShadow: '0 2px 8px rgba(0,0,0,0.12)'
                    }}
                  >
                    {advancedStatus === 'processing' ? <Loader size={14} className="spin" /> : <Sparkles size={14} />}
                    {advancedStatus === 'processing' ? 'Generating Advanced MoM...' : 'Generate Advanced MoM'}
                  </button>
                </div>

                {/* Display Enhanced Result Preview */}
                {romData?.final_rom?.agendas && romData.final_rom.agendas.length > 0 && (
                  <div style={{ marginTop: '.5rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                    <div style={{ fontSize: '.84rem', fontWeight: 700, color: 'hsl(var(--ink))', display: 'flex', alignItems: 'center', gap: 6 }}>
                      <FileText size={14} style={{ color: 'hsl(330,85%,60%)' }} /> Current Enhanced Agendas & Action Items
                    </div>
                    {romData.final_rom.agendas.map((ag: any, idx: number) => (
                      <div key={idx} style={{ borderRadius: 8, border: '1px solid hsl(var(--border)/.4)', background: 'hsl(var(--card))', padding: '1rem' }}>
                        <div style={{ fontSize: '.84rem', fontWeight: 700, color: 'hsl(var(--ink))', marginBottom: '.5rem' }}>
                          {ag.title || `Agenda ${idx + 1}`}
                        </div>
                        {(ag.discussion_points || []).map((dp: any, dpIdx: number) => (
                          <div key={dpIdx} style={{ fontSize: '.78rem', color: 'hsl(var(--pencil))', lineHeight: 1.45, marginBottom: '.4rem' }}>
                            • {dp.polished_text || dp.text || ''}
                          </div>
                        ))}
                      </div>
                    ))}
                  </div>
                )}

              </div>
            )}

        </div>
      </div>
    </div>
  </div>
)
}


