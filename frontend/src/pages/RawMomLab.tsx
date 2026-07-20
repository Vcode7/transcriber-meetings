import { useState, useCallback, useEffect, useRef } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import {
  FlaskConical, ChevronDown, ChevronRight, Search, Loader,
  CheckCircle, X, Download, FileText, Clock, User,
  Sparkles, FileDown, ArrowLeft, Trash2, Upload, Users, Calendar, Brain,
  ExternalLink, ListChecks, ListOrdered, RefreshCw, Plus, Zap,
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

interface AttachmentFile {
  id: string
  filename: string
  type: 'agenda' | 'context'
}

interface TranscriptChunk {
  chunk_id: string
  source: 'transcript'
  score: number
  text: string
  speakers: string[]
  start: number
  end: number
  chunk_index: number
  word_count: number
  char_count: number
  from_timeline?: boolean
  from_semantic?: boolean
}

interface MeetingChunk {
  chunk_id: string
  source: 'meeting'
  score: number
  text: string
  filename: string
  page: number | null
  chunk_index: number
  char_count: number
}

interface GlobalChunk {
  chunk_id: string
  source: 'global'
  score: number
  text: string
  filename: string
  chunk_index: number
  char_count: number
}

type AnyChunk = TranscriptChunk | MeetingChunk | GlobalChunk

interface AgendaItem {
  topic: string
  speaker: string | null
}

interface AgendaResponse {
  agendas: AgendaItem[]
  source: 'cached' | 'parsed' | 'generated'
}

interface AgendaContextFile {
  name: string
  text: string
  charCount: number
}

interface AgendaResult {
  topic: string
  speaker: string | null
  is_procedural: boolean
  transcript_chunks: TranscriptChunk[]
  meeting_chunks: MeetingChunk[]
  global_chunks: GlobalChunk[]
}

interface DiscussionEntry {
  type: string
  speaker: string | null
  point: string
  dates: Array<{ value: string; purpose: string }>
  action: {
    owner: string | null
    description: string | null
    deadline: string | null
    status: string | null
  }
}

interface AgendaMom {
  agenda_topic: string
  agenda_speaker: string | null
  discussion: DiscussionEntry[]
}

interface RawMomResult {
  meeting: { agendas: AgendaMom[] }
}

interface FinalActionItem {
  task: string
  background?: string | null
  owner: string
  deadline: string
  status?: string | null
}

interface FinalMomData {
  title: string
  date: string
  duration: number
  participants: string[]
  introduction: string
  points_discussed: string[]
  action_items: FinalActionItem[]
  conclusion: string
}

type ProcessState = 'idle' | 'processing' | 'done' | 'error'

// ── Helpers ───────────────────────────────────────────────────────────────────

function fmtTime(secs: number): string {
  const h = Math.floor(secs / 3600)
  const m = Math.floor((secs % 3600) / 60)
  const s = Math.floor(secs % 60)
  if (h > 0) return `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
}

interface CompactBlock {
  speaker: string
  text: string
}

function mergeConsecutiveSegments(segments: TranscriptSegment[]): CompactBlock[] {
  if (!segments.length) return []
  const blocks: CompactBlock[] = []
  let currentBlock: CompactBlock = {
    speaker: segments[0].speaker_label || 'Unknown',
    text: (segments[0].text || '').trim()
  }

  for (let i = 1; i < segments.length; i++) {
    const seg = segments[i]
    const speaker = seg.speaker_label || 'Unknown'
    if (speaker === currentBlock.speaker) {
      const textToAdd = (seg.text || '').trim()
      if (textToAdd) {
        currentBlock.text += (currentBlock.text ? ' ' : '') + textToAdd
      }
    } else {
      blocks.push(currentBlock)
      currentBlock = {
        speaker,
        text: (seg.text || '').trim()
      }
    }
  }
  blocks.push(currentBlock)
  return blocks.filter(b => b.text.length > 0)
}

function compactTranscriptText(text: string): string {
  if (!text) return ''
  const lines = text.split('\n')
  const blocks: CompactBlock[] = []
  let currentBlock: CompactBlock | null = null

  for (const line of lines) {
    const trimmed = line.trim()
    if (!trimmed) continue

    // Regex to match "Speaker Name: text"
    const match = trimmed.match(/^([^:]+):\s*(.*)$/)
    if (match) {
      const speaker = match[1].trim()
      const content = match[2].trim()

      if (currentBlock && currentBlock.speaker === speaker) {
        if (content) {
          currentBlock.text += (currentBlock.text ? ' ' : '') + content
        }
      } else {
        if (currentBlock) {
          blocks.push(currentBlock)
        }
        currentBlock = {
          speaker,
          text: content
        }
      }
    } else {
      // Continuation line without speaker label prefix
      if (currentBlock) {
        currentBlock.text += (currentBlock.text ? ' ' : '') + trimmed
      } else {
        currentBlock = {
          speaker: 'Unknown',
          text: trimmed
        }
      }
    }
  }

  if (currentBlock) {
    blocks.push(currentBlock)
  }

  // Format as:
  // Speaker Name:
  // Text content
  return blocks
    .filter(b => b.text.length > 0)
    .map(b => `${b.speaker}:\n${b.text}`)
    .join('\n\n')
}

function scoreColor(score: number): string {
  if (score >= 0.85) return 'hsl(140,70%,45%)'
  if (score >= 0.70) return 'hsl(45,90%,50%)'
  return 'hsl(0,80%,55%)'
}

function getTimelineSegments(
  segments: TranscriptSegment[],
  agendaIndex: number,
  totalAgendas: number,
  duration: number,
  stride: number,
): TranscriptSegment[] {
  if (!segments.length || !duration || !totalAgendas) return []
  const slotSize = duration / totalAgendas
  const winStart = Math.max(0, agendaIndex * slotSize - stride)
  const winEnd = Math.min(duration, (agendaIndex + 1) * slotSize + stride)
  return segments.filter(seg => seg.start <= winEnd && seg.end >= winStart)
}

function assembleEvidenceForAgenda(
  agendaIndex: number,
  _agendaItem: AgendaItem,
  timelineSegs: TranscriptSegment[],
  contextFiles: AgendaContextFile[],
  retrievedResult: AgendaResult | null,
  disabledChunks: Set<string>,
  charLimit: number,
): string {
  const sections: string[] = []
  if (timelineSegs.length) {
    const formattedRawLines = timelineSegs.map(s => `${s.speaker_label}: ${s.text}`).join('\n')
    const compactTimeline = compactTranscriptText(formattedRawLines)
    sections.push('=== TIMELINE TRANSCRIPT ===\n' + compactTimeline)
  }
  if (contextFiles.length) {
    sections.push('=== AGENDA CONTEXT (Uploaded) ===\n' + contextFiles.map(f => `[${f.name}]\n${f.text}`).join('\n\n'))
  }
  if (retrievedResult) {
    const extraT = retrievedResult.transcript_chunks.filter(c => !disabledChunks.has(`${agendaIndex}_${c.chunk_id}`))
    if (extraT.length) {
      const compactedChunks = extraT.map(c => compactTranscriptText(c.text)).join('\n\n')
      sections.push('=== ADDITIONAL TRANSCRIPT EVIDENCE ===\n' + compactedChunks)
    }
    const meet = retrievedResult.meeting_chunks.filter(c => !disabledChunks.has(`${agendaIndex}_${c.chunk_id}`))
    if (meet.length) sections.push('=== MEETING CONTEXT ===\n' + meet.map(c => c.filename ? `[${c.filename}] ${c.text}` : c.text).join('\n'))
    const glob = retrievedResult.global_chunks.filter(c => !disabledChunks.has(`${agendaIndex}_${c.chunk_id}`))
    if (glob.length) sections.push('=== GLOBAL CONTEXT ===\n' + glob.map(c => c.filename ? `[${c.filename}] ${c.text}` : c.text).join('\n'))
  }
  const assembled = sections.join('\n\n')
  return assembled.slice(0, charLimit)
}

// ── Sub-components ────────────────────────────────────────────────────────────

function ScorePill({ score }: { score: number }) {
  const col = scoreColor(score)
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center',
      fontSize: '.72rem', fontWeight: 700, color: col,
      background: `${col}1a`, border: `1px solid ${col}44`,
      padding: '2px 7px', borderRadius: 999, fontFamily: 'JetBrains Mono, monospace',
    }}>{score.toFixed(3)}</span>
  )
}

function ChunkRow({ chunk, agendaIndex, disabled, onToggle }: {
  chunk: AnyChunk; agendaIndex: number; disabled: boolean; onToggle: (ai: number, id: string) => void
}) {
  const [expanded, setExpanded] = useState(false)
  const tChunk = chunk.source === 'transcript' ? chunk as TranscriptChunk : null
  const mChunk = chunk.source === 'meeting' ? chunk as MeetingChunk : null
  const gChunk = chunk.source === 'global' ? chunk as GlobalChunk : null
  return (
    <div style={{ borderRadius: 10, border: '1px solid hsl(var(--border)/.4)', background: disabled ? 'hsl(var(--muted)/.3)' : 'hsl(var(--card))', overflow: 'hidden', opacity: disabled ? 0.5 : 1 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '.5rem .85rem', cursor: 'pointer' }} onClick={() => setExpanded(e => !e)}>
        {expanded ? <ChevronDown size={13} style={{ flexShrink: 0, color: 'hsl(var(--pencil))' }} /> : <ChevronRight size={13} style={{ flexShrink: 0, color: 'hsl(var(--pencil))' }} />}
        <span style={{ fontSize: '.78rem', fontWeight: 600, color: 'hsl(var(--ink))', minWidth: 56 }}>Chunk {chunk.chunk_index}</span>
        <ScorePill score={chunk.score} />
        {tChunk && tChunk.speakers.length > 0 && <span style={{ fontSize: '.73rem', color: 'hsl(var(--pencil))' }}><User size={10} style={{ display: 'inline', marginRight: 2 }} />{tChunk.speakers.join(', ')}</span>}
        {tChunk && <span style={{ fontSize: '.73rem', color: 'hsl(var(--pencil))' }}><Clock size={10} style={{ display: 'inline', marginRight: 2 }} />{fmtTime(tChunk.start)} - {fmtTime(tChunk.end)}</span>}
        {tChunk && (tChunk.from_timeline || tChunk.from_semantic) && (
          <span style={{ display: 'flex', gap: 3 }}>
            {tChunk.from_timeline && <span style={{ fontSize: '.62rem', fontWeight: 700, padding: '1px 5px', borderRadius: 4, background: 'hsl(205,90%,55%/.12)', color: 'hsl(205,90%,55%)', border: '1px solid hsl(205,90%,55%/.3)' }}>TL</span>}
            {tChunk.from_semantic && <span style={{ fontSize: '.62rem', fontWeight: 700, padding: '1px 5px', borderRadius: 4, background: 'hsl(280,75%,60%/.12)', color: 'hsl(280,75%,65%)', border: '1px solid hsl(280,75%,60%/.3)' }}>SEM</span>}
          </span>
        )}
        {(mChunk || gChunk) && <span style={{ fontSize: '.73rem', color: 'hsl(var(--pencil))', flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}><FileText size={10} style={{ display: 'inline', marginRight: 2 }} />{mChunk?.filename || gChunk?.filename || ''}{mChunk?.page != null ? ` p.${mChunk.page}` : ''}</span>}
        <span style={{ marginLeft: 'auto', fontSize: '.7rem', color: 'hsl(var(--pencil))', whiteSpace: 'nowrap' }}>{chunk.char_count.toLocaleString()} ch</span>
        <button onClick={e => { e.stopPropagation(); onToggle(agendaIndex, chunk.chunk_id) }} style={{ flexShrink: 0, display: 'flex', alignItems: 'center', gap: 4, fontSize: '.7rem', fontWeight: 600, padding: '3px 7px', borderRadius: 7, border: `1px solid ${disabled ? 'hsl(var(--destructive)/.3)' : 'hsl(140,70%,45%/.3)'}`, background: disabled ? 'hsl(var(--destructive)/.08)' : 'hsl(140,70%,45%/.08)', color: disabled ? 'hsl(var(--destructive))' : 'hsl(140,70%,45%)', cursor: 'pointer' }}>
          {disabled ? <><X size={10} /> Out</> : <><CheckCircle size={10} /> In</>}
        </button>
      </div>
      {expanded && (
        <div style={{ borderTop: '1px solid hsl(var(--border)/.25)', padding: '.65rem .9rem' }}>
          <pre style={{ margin: 0, fontSize: '.76rem', color: 'hsl(var(--ink))', fontFamily: 'Inter', lineHeight: 1.55, whiteSpace: 'pre-wrap', wordBreak: 'break-word', background: 'hsl(var(--paper)/.5)', padding: '.65rem', borderRadius: 8, border: '1px solid hsl(var(--border)/.3)', maxHeight: 280, overflowY: 'auto' }}>
            {chunk.source === 'transcript' ? compactTranscriptText(chunk.text) : chunk.text}
          </pre>
        </div>
      )}
    </div>
  )
}

function ChunkGroup({ label, color, chunks, agendaIndex, disabled, onToggle, defaultOpen = false }: {
  label: string; color: string; chunks: AnyChunk[]; agendaIndex: number
  disabled: Set<string>; onToggle: (ai: number, id: string) => void; defaultOpen?: boolean
}) {
  const [open, setOpen] = useState(defaultOpen)
  const active = chunks.filter(c => !disabled.has(`${agendaIndex}_${c.chunk_id}`)).length
  return (
    <div style={{ marginBottom: '.35rem' }}>
      <button onClick={() => setOpen(o => !o)} style={{ display: 'flex', alignItems: 'center', gap: 6, width: '100%', textAlign: 'left', background: 'none', border: 'none', cursor: 'pointer', padding: '.3rem .45rem', borderRadius: 7 }}>
        {open ? <ChevronDown size={12} style={{ color }} /> : <ChevronRight size={12} style={{ color }} />}
        <span style={{ fontSize: '.73rem', fontWeight: 700, color, textTransform: 'uppercase', letterSpacing: '.04em' }}>{label}</span>
        <span style={{ fontSize: '.67rem', fontWeight: 600, color: active > 0 ? color : 'hsl(var(--pencil))', background: `${color}1a`, border: `1px solid ${color}33`, padding: '1px 6px', borderRadius: 999 }}>{active}/{chunks.length}</span>
      </button>
      {open && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4, paddingLeft: '.65rem', marginTop: 3 }}>
          {chunks.length === 0 ? <p style={{ fontSize: '.73rem', color: 'hsl(var(--pencil))', margin: '2px 0' }}>No chunks</p>
            : chunks.map(c => <ChunkRow key={c.chunk_id} chunk={c} agendaIndex={agendaIndex} disabled={disabled.has(`${agendaIndex}_${c.chunk_id}`)} onToggle={onToggle} />)}
        </div>
      )}
    </div>
  )
}

function SectionHeader({ icon, label, count, color }: { icon: React.ReactNode; label: string; count?: number; color: string }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 7, padding: '.4rem .7rem', borderRadius: 8, background: `${color}12`, border: `1px solid ${color}28`, marginBottom: '.45rem' }}>
      <span style={{ color, display: 'flex' }}>{icon}</span>
      <span style={{ fontSize: '.73rem', fontWeight: 700, color, textTransform: 'uppercase', letterSpacing: '.04em' }}>{label}</span>
      {count !== undefined && <span style={{ fontSize: '.67rem', fontWeight: 600, padding: '1px 7px', borderRadius: 999, background: `${color}20`, border: `1px solid ${color}40`, color, marginLeft: 'auto' }}>{count}</span>}
    </div>
  )
}

// ── Main Page ─────────────────────────────────────────────────────────────────

export default function RawMomLab() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()

  const [recording, setRecording] = useState<RecordingDetail | null>(null)
  const [loadingRec, setLoadingRec] = useState(true)

  const [agendaFiles, setAgendaFiles] = useState<AttachmentFile[]>([])
  const [contextFiles, setContextFiles] = useState<AttachmentFile[]>([])
  const [agendaProcessState, setAgendaProcessState] = useState<ProcessState>('idle')
  const [contextProcessState, setContextProcessState] = useState<ProcessState>('idle')
  const agendaInputRef = useRef<HTMLInputElement>(null)
  const contextInputRef = useRef<HTMLInputElement>(null)

  const [kTranscript, setKTranscript] = useState(8)
  const [kMeeting, setKMeeting] = useState(4)
  const [kGlobal, setKGlobal] = useState(2)
  const [cutoff, setCutoff] = useState(0.01)
  const [charLimit, setCharLimit] = useState(15000)
  const [forceReembed, setForceReembed] = useState(false)
  const [timelineStride, setTimelineStride] = useState(60)
  const [highConfThreshold, setHighConfThreshold] = useState(0.70)
  const [forceReparse, setForceReparse] = useState(false)
  const [useMaxTokensForFinal, setUseMaxTokensForFinal] = useState(false)

  const [editedAgendaItems, setEditedAgendaItems] = useState<AgendaItem[]>([])
  const [creatingAgenda, setCreatingAgenda] = useState(false)
  const [agendaCreated, setAgendaCreated] = useState(false)
  const [agendaSource, setAgendaSource] = useState<string>('')
  const [selectedAgendaIndex, setSelectedAgendaIndex] = useState<number | null>(null)

  const [agendaContextFiles, setAgendaContextFiles] = useState<Record<number, AgendaContextFile[]>>({})
  const [agendaEvidenceResult, setAgendaEvidenceResult] = useState<Record<number, AgendaResult>>({})
  const [agendaContextUploading, setAgendaContextUploading] = useState<Record<number, boolean>>({})
  const [showEvidencePreview, setShowEvidencePreview] = useState<Set<number>>(new Set())

  const [disabledChunks, setDisabledChunks] = useState<Set<string>>(new Set())
  const [retrieving, setRetrieving] = useState(false)
  const [generating, setGenerating] = useState(false)
  const [momResult, setMomResult] = useState<RawMomResult | null>(null)
  const [generatingFinalMom, setGeneratingFinalMom] = useState(false)
  const [finalMomResult, setFinalMomResult] = useState<FinalMomData | null>(null)

  const agendaContextInputRefs = useRef<Record<number, HTMLInputElement | null>>({})

  const loadInitialData = useCallback(async () => {
    if (!id) return
    setLoadingRec(true)
    try {
      const recRes = await api.get(`/history/${id}`)
      setRecording(recRes.data)
      try {
        const attRes = await api.get(`/attachments/${id}`)
        const allFiles: AttachmentFile[] = attRes.data.files || []
        setAgendaFiles(allFiles.filter((f: AttachmentFile) => f.type === 'agenda'))
        setContextFiles(allFiles.filter((f: AttachmentFile) => f.type === 'context'))
      } catch { /* optional */ }
      try { const momRes = await api.get(`/raw-mom/${id}`); setMomResult(momRes.data) } catch { /* 404 ok */ }
    } catch { toast.error('Failed to load recording details') }
    finally { setLoadingRec(false) }
  }, [id])

  useEffect(() => { loadInitialData() }, [loadInitialData])

  const handleUploadFiles = async (files: FileList | null, type: 'agenda' | 'context') => {
    if (!files || !files.length || !id) return
    const fd = new FormData()
    fd.append('type', type)
    Array.from(files).forEach(f => fd.append('files', f))
    try {
      await api.post(`/attachments/${id}/upload`, fd, { headers: { 'Content-Type': 'multipart/form-data' } })
      const res = await api.get(`/attachments/${id}`)
      const all: AttachmentFile[] = res.data.files || []
      if (type === 'agenda') setAgendaFiles(all.filter(f => f.type === 'agenda'))
      else setContextFiles(all.filter(f => f.type === 'context'))
    } catch (e) { toast.error(`Upload failed: ${getApiErrorDetail(e)}`) }
  }

  const handleDeleteFile = async (fileId: string, type: 'agenda' | 'context') => {
    if (!id) return
    try {
      await api.delete(`/attachments/${id}/${fileId}`)
      if (type === 'agenda') setAgendaFiles(prev => prev.filter(f => f.id !== fileId))
      else setContextFiles(prev => prev.filter(f => f.id !== fileId))
    } catch (e) { toast.error(`Delete failed: ${getApiErrorDetail(e)}`) }
  }

  const handleProcessFiles = async (type: 'agenda' | 'context') => {
    if (!id) return
    const setState = type === 'agenda' ? setAgendaProcessState : setContextProcessState
    setState('processing')
    try {
      const fd = new FormData(); fd.append('type', type)
      await api.post(`/attachments/${id}/process`, fd, { headers: { 'Content-Type': 'multipart/form-data' } })
      setState('done')
    } catch (e) { toast.error(`Processing failed: ${getApiErrorDetail(e)}`); setState('error') }
  }

  const handleCreateAgenda = async () => {
    if (!id) return
    setCreatingAgenda(true)
    try {
      const res = await api.post(`/raw-mom/${id}/agenda`, { force_reparse: forceReparse })
      const data: AgendaResponse = res.data
      setEditedAgendaItems(data.agendas.map(a => ({ ...a })))
      setAgendaContextFiles({})
      setAgendaEvidenceResult({})
      setDisabledChunks(new Set())
      setAgendaCreated(true)
      setAgendaSource(data.source)
      if (data.agendas.length > 0) setSelectedAgendaIndex(0)
      toast.success(`${data.agendas.length} agenda items (${data.source})`)
    } catch (e: any) { toast.error(e?.response?.data?.detail ?? 'Agenda creation failed') }
    finally { setCreatingAgenda(false) }
  }

  const updateAgendaItem = (index: number, field: 'topic' | 'speaker', value: string) => {
    setEditedAgendaItems(prev => prev.map((item, i) => i === index ? { ...item, [field]: value || null } : item))
  }

  const handleUploadAgendaContext = async (agendaIndex: number, files: FileList | null) => {
    if (!files || !files.length || !id) return
    setAgendaContextUploading(prev => ({ ...prev, [agendaIndex]: true }))
    const results: AgendaContextFile[] = []
    for (const file of Array.from(files)) {
      try {
        const fd = new FormData(); fd.append('file', file)
        const res = await api.post(`/raw-mom/${id}/extract-file-text`, fd, { headers: { 'Content-Type': 'multipart/form-data' } })
        results.push({ name: res.data.filename, text: res.data.text, charCount: res.data.char_count })
      } catch (e) { toast.error(`Failed to extract ${file.name}: ${getApiErrorDetail(e)}`) }
    }
    if (results.length) {
      setAgendaContextFiles(prev => ({ ...prev, [agendaIndex]: [...(prev[agendaIndex] || []), ...results] }))
      toast.success(`${results.length} file(s) added for agenda ${agendaIndex + 1}`)
    }
    setAgendaContextUploading(prev => ({ ...prev, [agendaIndex]: false }))
  }

  const removeAgendaContextFile = (agendaIndex: number, fi: number) => {
    setAgendaContextFiles(prev => ({ ...prev, [agendaIndex]: (prev[agendaIndex] || []).filter((_, i) => i !== fi) }))
  }

  const handleRetrieve = async () => {
    if (!id || !editedAgendaItems.length) return
    setRetrieving(true)
    setMomResult(null)
    setDisabledChunks(new Set())
    try {
      const res = await api.post(`/raw-mom/${id}/retrieve`, {
        k_transcript: kTranscript,
        k_meeting: kMeeting,
        k_global: kGlobal,
        relative_similarity_cutoff: cutoff,
        char_limit: charLimit,
        force_reembed: forceReembed,
        agenda_items: editedAgendaItems,
        timeline_stride_seconds: timelineStride,
        high_confidence_threshold: highConfThreshold,
        recording_duration: recording?.duration ?? 0,
      })
      const resultsMap: Record<number, AgendaResult> = {}
      res.data.agendas.forEach((agendaRes: AgendaResult, idx: number) => {
        resultsMap[idx] = agendaRes
      })
      setAgendaEvidenceResult(resultsMap)
      toast.success(`Evidence retrieved for all ${res.data.agendas.length} agenda items!`)
    } catch (e: any) {
      toast.error(e?.response?.data?.detail ?? 'Retrieval failed')
    } finally {
      setRetrieving(false)
    }
  }

  const toggleChunk = useCallback((agendaIndex: number, chunkId: string) => {
    setDisabledChunks(prev => {
      const next = new Set(prev)
      const key = `${agendaIndex}_${chunkId}`
      if (next.has(key)) next.delete(key); else next.add(key)
      return next
    })
  }, [])

  const handleGenerate = async () => {
    if (!id || !editedAgendaItems.length) { toast.error('Create agenda first'); return }
    setGenerating(true)
    try {
      const agendas = editedAgendaItems.map((item, idx) => {
        const tl = getTimelineSegments(recording?.transcript || [], idx, editedAgendaItems.length, recording?.duration ?? 0, timelineStride)
        const evidence = assembleEvidenceForAgenda(idx, item, tl, agendaContextFiles[idx] || [], agendaEvidenceResult[idx] || null, disabledChunks, charLimit)
        return { topic: item.topic, speaker: item.speaker, evidence }
      })
      const res = await api.post(`/raw-mom/${id}/generate`, { agendas })
      setMomResult(res.data)
      toast.success('Raw MoM generated')
    } catch (e: any) { toast.error(e?.response?.data?.detail ?? 'Generation failed') }
    finally { setGenerating(false) }
  }

  const handleGenerateFinalMom = async () => {
    if (!id || !momResult) { toast.error('Generate Raw MoM first'); return }
    setGeneratingFinalMom(true)
    setFinalMomResult(null)
    try {
      const res = await api.post(`/raw-mom/${id}/generate-final-mom`, {
        char_limit: charLimit,
        use_max_tokens: useMaxTokensForFinal,
      })
      setFinalMomResult(res.data)
      toast.success('Final MoM generated')
    } catch (e: any) { toast.error(e?.response?.data?.detail ?? 'Final MoM generation failed') }
    finally { setGeneratingFinalMom(false) }
  }

  const downloadJson = () => {
    if (!momResult) return
    const blob = new Blob([JSON.stringify(momResult, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a'); a.href = url; a.download = `raw_mom_${id}.json`; a.click()
    URL.revokeObjectURL(url)
  }

  const downloadDocx = async () => {
    if (!id) return
    try {
      const res = await api.get(`/raw-mom/${id}/download/docx`, { responseType: 'blob' })
      const blob = new Blob([res.data], { type: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a'); a.href = url; a.download = `raw_mom_${id}.docx`
      document.body.appendChild(a); a.click(); document.body.removeChild(a)
      URL.revokeObjectURL(url)
      toast.success('DOCX downloaded')
    } catch { toast.error('Failed to download DOCX') }
  }

  if (loadingRec) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', flexDirection: 'column', gap: '1rem' }}>
        <Loader size={24} className="spin" style={{ color: 'hsl(var(--accent))' }} />
        <p style={{ fontSize: '.9rem', color: 'hsl(var(--pencil))', fontFamily: 'Inter, sans-serif' }}>Loading recording details...</p>
      </div>
    )
  }

  const selIdx = selectedAgendaIndex
  const selItem = selIdx !== null ? editedAgendaItems[selIdx] : null
  const selTimeline = selIdx !== null ? getTimelineSegments(recording?.transcript || [], selIdx, editedAgendaItems.length, recording?.duration ?? 0, timelineStride) : []
  const selCtxFiles = selIdx !== null ? (agendaContextFiles[selIdx] || []) : []
  const selEvidence = selIdx !== null ? (agendaEvidenceResult[selIdx] || null) : null

  return (
    <div className="page-scroll-root" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      {/* Header */}
      <div className="panel-header" style={{ flexShrink: 0, gap: '12px' }}>
        <button className="icon-btn" onClick={() => navigate(-1)} title="Go Back"><ArrowLeft size={16} /></button>
        <div style={{ width: 32, height: 32, borderRadius: '8px', flexShrink: 0, background: 'hsl(280,75%,60%/.15)', border: '1.5px solid hsl(280,75%,60%/.3)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <FlaskConical size={15} style={{ color: 'hsl(280,75%,65%)' }} />
        </div>
        <div style={{ flex: 1 }}>
          <h1 style={{ fontSize: '1.05rem', fontWeight: 700 }}>Raw MoM Lab</h1>
          <p style={{ fontSize: '.78rem', color: 'hsl(var(--pencil))', fontFamily: 'Inter', fontWeight: 400, marginTop: '1px' }}>
            Agenda-first RAG evidence pipeline for "{recording?.filename}"
          </p>
        </div>
        {momResult && (
          <div style={{ display: 'flex', gap: 6 }}>
            <button onClick={downloadJson} style={{ display: 'flex', alignItems: 'center', gap: 5, padding: '.38rem .75rem', borderRadius: 8, border: '1.5px solid hsl(var(--border)/.6)', background: 'hsl(var(--card))', color: 'hsl(var(--ink))', fontSize: '.76rem', fontWeight: 600, cursor: 'pointer', fontFamily: 'Inter' }}><Download size={12} /> JSON</button>
            <button onClick={downloadDocx} style={{ display: 'flex', alignItems: 'center', gap: 5, padding: '.38rem .75rem', borderRadius: 8, border: '1.5px solid hsl(205,90%,55%/.4)', background: 'hsl(205,90%,55%/.08)', color: 'hsl(205,90%,60%)', fontSize: '.76rem', fontWeight: 600, cursor: 'pointer', fontFamily: 'Inter' }}><FileDown size={12} /> DOCX</button>
          </div>
        )}
      </div>

      {/* Main split */}
      <div style={{ flex: 1, display: 'flex', gap: '1.25rem', padding: '1.25rem 1.75rem 3rem', minHeight: 0, overflow: 'hidden' }}>

        {/* ── LEFT PANEL ── */}
        <div style={{ width: 268, flexShrink: 0, minHeight: 0, maxHeight: '100%', display: 'flex', flexDirection: 'column', gap: '.6rem', overflowY: 'auto', paddingRight: 4 }}>

          {/* Recording compact */}
          <div style={{ borderRadius: 10, border: '1.5px solid hsl(var(--border)/.4)', background: 'hsl(var(--card))', padding: '.55rem .8rem' }}>
            <div style={{ fontSize: '.69rem', fontWeight: 700, color: 'hsl(var(--pencil))', textTransform: 'uppercase', letterSpacing: '.05em', marginBottom: '.3rem' }}>Recording</div>
            <div style={{ fontSize: '.74rem', color: 'hsl(var(--ink))', fontWeight: 600, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{recording?.filename}</div>
            <div style={{ display: 'flex', gap: 10, marginTop: 3, fontSize: '.71rem', color: 'hsl(var(--pencil))' }}>
              <span><Clock size={9} style={{ display: 'inline', marginRight: 2 }} />{recording ? fmtTime(recording.duration) : '-'}</span>
              <span><User size={9} style={{ display: 'inline', marginRight: 2 }} />{recording?.speakers_detected?.length ?? 0} spk</span>
            </div>
          </div>

          {/* Hidden inputs */}
          <input ref={agendaInputRef} type="file" multiple accept=".pdf,.docx,.pptx,.txt,.md,.png,.jpg,.jpeg,.webp,.xlsx,.xls,.csv" style={{ display: 'none' }} onChange={e => handleUploadFiles(e.target.files, 'agenda')} />
          <input ref={contextInputRef} type="file" multiple accept=".pdf,.docx,.pptx,.txt,.md,.png,.jpg,.jpeg,.webp,.xlsx,.xls,.csv" style={{ display: 'none' }} onChange={e => handleUploadFiles(e.target.files, 'context')} />

          {/* Agenda file */}
          <div style={{ borderRadius: 10, border: '1.5px solid hsl(var(--border)/.4)', background: 'hsl(var(--card))', padding: '.55rem .8rem', display: 'flex', flexDirection: 'column', gap: '.4rem' }}>
            <div style={{ fontSize: '.69rem', fontWeight: 700, color: 'hsl(var(--pencil))', textTransform: 'uppercase', letterSpacing: '.05em' }}>Agenda File</div>
            {agendaFiles.length > 0 ? agendaFiles.map(f => (
              <div key={f.id} style={{ display: 'flex', alignItems: 'center', gap: 5, padding: '3px 6px', borderRadius: 5, background: 'hsl(var(--muted)/.5)', fontSize: '.72rem' }}>
                <FileText size={9} style={{ flexShrink: 0 }} />
                <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{f.filename}</span>
                <button onClick={() => handleDeleteFile(f.id, 'agenda')} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'hsl(var(--destructive))' }}><Trash2 size={9} /></button>
              </div>
            )) : (
              <div onClick={() => agendaInputRef.current?.click()} style={{ border: '1.5px dashed hsl(var(--border)/.5)', borderRadius: 7, padding: '.3rem', textAlign: 'center', cursor: 'pointer', fontSize: '.72rem', color: 'hsl(var(--pencil))', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 3 }}><Upload size={9} /> Upload Agenda</div>
            )}
            {agendaFiles.length > 0 && <button className="btn btn-ghost" disabled={agendaProcessState === 'processing'} onClick={() => handleProcessFiles('agenda')} style={{ width: '100%', fontSize: '.68rem', padding: '2px 4px', height: 20 }}>{agendaProcessState === 'processing' ? 'Processing...' : 'Process'}</button>}
          </div>

          {/* Context files */}
          <div style={{ borderRadius: 10, border: '1.5px solid hsl(var(--border)/.4)', background: 'hsl(var(--card))', padding: '.55rem .8rem', display: 'flex', flexDirection: 'column', gap: '.4rem' }}>
            <div style={{ fontSize: '.69rem', fontWeight: 700, color: 'hsl(var(--pencil))', textTransform: 'uppercase', letterSpacing: '.05em' }}>Context Files</div>
            {contextFiles.length > 0 ? contextFiles.map(f => (
              <div key={f.id} style={{ display: 'flex', alignItems: 'center', gap: 5, padding: '3px 6px', borderRadius: 5, background: 'hsl(var(--muted)/.5)', fontSize: '.72rem' }}>
                <FileText size={9} style={{ flexShrink: 0 }} />
                <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{f.filename}</span>
                <button onClick={() => handleDeleteFile(f.id, 'context')} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'hsl(var(--destructive))' }}><Trash2 size={9} /></button>
              </div>
            )) : (
              <div onClick={() => contextInputRef.current?.click()} style={{ border: '1.5px dashed hsl(var(--border)/.5)', borderRadius: 7, padding: '.3rem', textAlign: 'center', cursor: 'pointer', fontSize: '.72rem', color: 'hsl(var(--pencil))', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 3 }}><Upload size={9} /> Upload Context</div>
            )}
            {contextFiles.length > 0 && <button className="btn btn-ghost" disabled={contextProcessState === 'processing'} onClick={() => handleProcessFiles('context')} style={{ width: '100%', fontSize: '.68rem', padding: '2px 4px', height: 20 }}>{contextProcessState === 'processing' ? 'Processing...' : 'Process'}</button>}
          </div>

          {/* Settings */}
          <div style={{ borderRadius: 10, border: '1.5px solid hsl(var(--border)/.4)', background: 'hsl(var(--card))', padding: '.55rem .8rem', display: 'flex', flexDirection: 'column', gap: '.45rem' }}>
            <div style={{ fontSize: '.69rem', fontWeight: 700, color: 'hsl(var(--pencil))', textTransform: 'uppercase', letterSpacing: '.05em' }}>Retrieval</div>
            {([
              { label: 'Transcript K', value: kTranscript, setter: setKTranscript, color: 'hsl(205,90%,55%)' },
              { label: 'Meeting K', value: kMeeting, setter: setKMeeting, color: 'hsl(140,70%,50%)' },
              { label: 'Global K', value: kGlobal, setter: setKGlobal, color: 'hsl(30,90%,55%)' },
            ] as { label: string; value: number; setter: (v: number) => void; color: string }[]).map(({ label, value, setter, color }) => (
              <div key={label}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 1 }}>
                  <label style={{ fontSize: '.71rem', color: 'hsl(var(--ink))', fontFamily: 'Inter' }}>{label}</label>
                  <span style={{ fontSize: '.71rem', fontWeight: 700, color, fontFamily: 'JetBrains Mono' }}>{value}</span>
                </div>
                <input type="range" min={0} max={20} value={value} onChange={e => setter(Number(e.target.value))} style={{ width: '100%', accentColor: color }} />
              </div>
            ))}
            <div style={{ height: '1px', background: 'hsl(var(--border)/.2)' }} />
            <div style={{ fontSize: '.66rem', fontWeight: 700, color: 'hsl(280,75%,65%)', textTransform: 'uppercase', letterSpacing: '.04em' }}>Timeline</div>
            {[
              { label: 'Stride (sec)', value: timelineStride, setter: setTimelineStride, step: 15, min: 0, max: 300 },
              { label: 'High-Conf', value: highConfThreshold, setter: setHighConfThreshold, step: 0.05, min: 0, max: 1 },
              { label: 'Sim Cutoff', value: cutoff, setter: setCutoff, step: 0.01, min: 0, max: 1 },
              { label: 'Max Tokens', value: charLimit, setter: setCharLimit, step: 1000, min: 1000, max: 50000 },
            ].map(({ label, value, setter, step, min, max }) => (
              <div key={label}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 1 }}>
                  <label style={{ fontSize: '.71rem', color: 'hsl(var(--ink))', fontFamily: 'Inter' }}>{label}</label>
                  <span style={{ fontSize: '.67rem', fontWeight: 700, color: 'hsl(var(--pencil))', fontFamily: 'JetBrains Mono' }}>{value}</span>
                </div>
                <input type="range" min={min} max={max} step={step} value={value} onChange={e => setter(Number(e.target.value))} style={{ width: '100%', accentColor: 'hsl(280,75%,60%)' }} />
              </div>
            ))}
            <div style={{ height: '1px', background: 'hsl(var(--border)/.2)' }} />
            <label style={{ display: 'flex', alignItems: 'center', gap: 5, cursor: 'pointer', fontSize: '.71rem', color: 'hsl(var(--ink))', fontFamily: 'Inter' }}>
              <input type="checkbox" checked={forceReembed} onChange={e => setForceReembed(e.target.checked)} />Force re-embed
            </label>
            <label style={{ display: 'flex', alignItems: 'center', gap: 5, cursor: 'pointer', fontSize: '.71rem', color: 'hsl(var(--ink))', fontFamily: 'Inter' }}>
              <input type="checkbox" checked={forceReparse} onChange={e => setForceReparse(e.target.checked)} />Force re-parse agenda
            </label>
            <label style={{ display: 'flex', alignItems: 'center', gap: 5, cursor: 'pointer', fontSize: '.71rem', color: 'hsl(var(--ink))', fontFamily: 'Inter' }}>
              <input type="checkbox" checked={useMaxTokensForFinal} onChange={e => setUseMaxTokensForFinal(e.target.checked)} />Use Max Tokens for Final MoM
            </label>
          </div>

          {/* Create Agenda */}
          <button onClick={handleCreateAgenda} disabled={creatingAgenda} style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 7, padding: '.6rem', borderRadius: 10, border: `1.5px solid ${agendaCreated ? 'hsl(280,75%,60%/.5)' : 'hsl(280,75%,60%)'}`, background: agendaCreated ? 'hsl(280,75%,60%/.1)' : 'hsl(280,75%,60%)', color: agendaCreated ? 'hsl(280,75%,65%)' : '#fff', fontSize: '.8rem', fontWeight: 700, cursor: creatingAgenda ? 'not-allowed' : 'pointer', opacity: creatingAgenda ? 0.6 : 1, fontFamily: 'Inter', transition: 'all .2s' }}>
            {creatingAgenda ? <Loader size={13} className="spin" /> : agendaCreated ? <RefreshCw size={13} /> : <ListOrdered size={13} />}
            {creatingAgenda ? 'Creating Agenda...' : agendaCreated ? 'Re-create Agenda' : 'Create Agenda'}
          </button>

          {/* Retrieve Evidence */}
          {agendaCreated && (
            <button onClick={handleRetrieve} disabled={retrieving} style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 7, padding: '.6rem', borderRadius: 10, border: 'none', background: 'hsl(280,75%,60%)', color: '#fff', fontSize: '.8rem', fontWeight: 700, cursor: retrieving ? 'not-allowed' : 'pointer', opacity: retrieving ? 0.6 : 1, fontFamily: 'Inter' }}>
              {retrieving ? <Loader size={13} className="spin" /> : <Search size={13} />}
              {retrieving ? 'Retrieving Chunks...' : 'Retrieve Evidence'}
            </button>
          )}

          {/* Agenda list */}
          {agendaCreated && editedAgendaItems.length > 0 && (
            <div style={{ borderRadius: 10, border: '1.5px solid hsl(280,75%,60%/.25)', background: 'hsl(var(--card))', overflow: 'hidden' }}>
              <div style={{ padding: '.4rem .7rem', background: 'linear-gradient(135deg,hsl(280,75%,60%/.1),hsl(260,70%,60%/.06))', borderBottom: '1px solid hsl(280,75%,60%/.2)', display: 'flex', alignItems: 'center', gap: 5 }}>
                <ListOrdered size={11} style={{ color: 'hsl(280,75%,65%)' }} />
                <span style={{ fontSize: '.71rem', fontWeight: 700, color: 'hsl(var(--ink))' }}>Agenda ({editedAgendaItems.length})</span>
                <span style={{ fontSize: '.62rem', padding: '1px 6px', borderRadius: 999, marginLeft: 'auto', background: 'hsl(280,75%,60%/.15)', border: '1px solid hsl(280,75%,60%/.3)', color: 'hsl(280,75%,65%)', fontWeight: 600 }}>
                  {agendaSource === 'cached' ? 'Cached' : agendaSource === 'parsed' ? 'From File' : 'Generated'}
                </span>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column' }}>
                {editedAgendaItems.map((item, idx) => {
                  const isSel = selIdx === idx
                  const hasEv = !!agendaEvidenceResult[idx]
                  return (
                    <button key={idx} onClick={() => setSelectedAgendaIndex(idx)} style={{ display: 'flex', alignItems: 'center', gap: 7, padding: '.45rem .7rem', textAlign: 'left', border: 'none', borderBottom: idx < editedAgendaItems.length - 1 ? '1px solid hsl(var(--border)/.2)' : 'none', background: isSel ? 'hsl(280,75%,60%/.12)' : 'transparent', cursor: 'pointer', transition: 'background .1s' }}>
                      <span style={{ flexShrink: 0, width: 18, height: 18, borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '.62rem', fontWeight: 700, background: isSel ? 'hsl(280,75%,60%)' : 'hsl(280,75%,60%/.15)', color: isSel ? '#fff' : 'hsl(280,75%,65%)', border: `1px solid ${isSel ? 'hsl(280,75%,60%)' : 'hsl(280,75%,60%/.3)'}`, transition: 'all .1s' }}>{idx + 1}</span>
                      <span style={{ flex: 1, fontSize: '.73rem', fontWeight: isSel ? 600 : 400, color: isSel ? 'hsl(var(--ink))' : 'hsl(var(--pencil))', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{item.topic}</span>
                      {retrieving && <Loader size={9} className="spin" style={{ color: 'hsl(280,75%,65%)', flexShrink: 0 }} />}
                      {!retrieving && hasEv && <span style={{ width: 6, height: 6, borderRadius: '50%', background: 'hsl(140,70%,45%)', flexShrink: 0 }} title="Evidence retrieved" />}
                    </button>
                  )
                })}
              </div>
            </div>
          )}

          {/* Generate Raw MoM */}
          {agendaCreated && (
            <button onClick={handleGenerate} disabled={generating} style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 7, padding: '.6rem', borderRadius: 10, border: 'none', background: generating ? 'hsl(140,70%,45%/.5)' : 'hsl(140,70%,45%)', color: '#fff', fontSize: '.8rem', fontWeight: 700, cursor: generating ? 'not-allowed' : 'pointer', fontFamily: 'Inter' }}>
              {generating ? <Loader size={13} className="spin" /> : <Sparkles size={13} />}
              {generating ? 'Generating...' : 'Generate Raw MoM'}
            </button>
          )}

          {/* Generate Final MoM */}
          {momResult && !generating && (
            <button onClick={handleGenerateFinalMom} disabled={generatingFinalMom} style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 7, padding: '.6rem', borderRadius: 10, border: 'none', background: generatingFinalMom ? 'hsl(220,80%,60%/.5)' : 'hsl(220,80%,60%)', color: '#fff', fontSize: '.8rem', fontWeight: 700, cursor: generatingFinalMom ? 'not-allowed' : 'pointer', fontFamily: 'Inter' }}>
              {generatingFinalMom ? <Loader size={13} className="spin" /> : <Zap size={13} />}
              {generatingFinalMom ? 'Generating...' : 'Generate Final MoM'}
            </button>
          )}
        </div>

        {/* ── RIGHT PANEL ── */}
        <div style={{ flex: 1, minWidth: 0, minHeight: 0, maxHeight: '100%', display: 'flex', flexDirection: 'column', gap: '1rem', overflow: 'scroll' }}>

          {/* Empty state */}
          {!agendaCreated && (
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '6rem 2rem', gap: '1rem', borderRadius: 16, border: '2px dashed hsl(var(--border)/.4)', color: 'hsl(var(--pencil))' }}>
              <FlaskConical size={36} style={{ opacity: 0.2 }} />
              <div style={{ textAlign: 'center' }}>
                <p style={{ margin: 0, fontWeight: 600, fontSize: '.95rem', color: 'hsl(var(--ink))' }}>Click "Create Agenda" to begin</p>
                <p style={{ margin: '4px 0 0', fontSize: '.82rem' }}>The agenda will be parsed or generated. Then select an agenda item to view its timeline transcript and retrieve targeted evidence.</p>
              </div>
            </div>
          )}

          {/* Select prompt */}
          {agendaCreated && selIdx === null && (
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '5rem 2rem', gap: '1rem', borderRadius: 16, border: '1.5px dashed hsl(var(--border)/.4)', color: 'hsl(var(--pencil))' }}>
              <ListOrdered size={30} style={{ opacity: 0.2 }} />
              <p style={{ margin: 0, fontSize: '.88rem' }}>Select an agenda item from the left panel</p>
            </div>
          )}

          {/* Agenda Detail */}
          {agendaCreated && selIdx !== null && selItem && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              {/* Header card */}
              <div style={{ borderRadius: 14, border: '1.5px solid hsl(280,75%,60%/.3)', background: 'hsl(var(--card))', overflow: 'hidden' }}>
                {/* Title bar */}
                <div style={{ background: 'linear-gradient(135deg,hsl(280,75%,60%/.1),hsl(260,70%,60%/.06))', borderBottom: '1px solid hsl(280,75%,60%/.2)', padding: '.7rem 1rem', display: 'flex', alignItems: 'center', gap: 10 }}>
                  <span style={{ flexShrink: 0, width: 26, height: 26, borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '.74rem', fontWeight: 700, background: 'hsl(280,75%,60%)', color: '#fff' }}>{selIdx + 1}</span>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <input type="text" value={selItem.topic} onChange={e => updateAgendaItem(selIdx, 'topic', e.target.value)} style={{ width: '100%', background: 'transparent', border: 'none', outline: 'none', fontSize: '.92rem', fontWeight: 700, color: 'hsl(var(--ink))', fontFamily: 'Inter' }} />
                    <input type="text" value={selItem.speaker || ''} onChange={e => updateAgendaItem(selIdx, 'speaker', e.target.value)} placeholder="Speaker (optional)" style={{ width: '100%', background: 'transparent', border: 'none', outline: 'none', fontSize: '.73rem', color: 'hsl(var(--pencil))', fontFamily: 'Inter' }} />
                  </div>
                </div>
                {/* Prev/Next */}
                <div style={{ display: 'flex', padding: '.35rem .85rem', gap: 6, borderBottom: '1px solid hsl(var(--border)/.15)' }}>
                  <button disabled={selIdx === 0} onClick={() => setSelectedAgendaIndex(i => Math.max(0, (i ?? 1) - 1))} style={{ fontSize: '.7rem', color: 'hsl(var(--pencil))', background: 'none', border: 'none', cursor: selIdx > 0 ? 'pointer' : 'default', opacity: selIdx > 0 ? 1 : 0.3 }}>&#8592; Prev</button>
                  <span style={{ flex: 1, textAlign: 'center', fontSize: '.7rem', color: 'hsl(var(--pencil))' }}>{selIdx + 1} / {editedAgendaItems.length}</span>
                  <button disabled={selIdx === editedAgendaItems.length - 1} onClick={() => setSelectedAgendaIndex(i => Math.min(editedAgendaItems.length - 1, (i ?? -1) + 1))} style={{ fontSize: '.7rem', color: 'hsl(var(--pencil))', background: 'none', border: 'none', cursor: selIdx < editedAgendaItems.length - 1 ? 'pointer' : 'default', opacity: selIdx < editedAgendaItems.length - 1 ? 1 : 0.3 }}>Next &#8594;</button>
                </div>
                <div style={{ padding: '.75rem 1rem', display: 'flex', flexDirection: 'column', gap: '.9rem' }}>

                
                  {/* Section 2: Agenda Context Upload */}
                  <div>
                    <SectionHeader icon={<FileText size={13} />} label="Agenda Context" count={selCtxFiles.length} color="hsl(45,90%,50%)" />
                    <input ref={el => { agendaContextInputRefs.current[selIdx] = el }} type="file" multiple accept=".pdf,.docx,.pptx,.txt,.md,.png,.jpg,.jpeg,.webp,.xlsx,.xls,.csv" style={{ display: 'none' }} onChange={e => handleUploadAgendaContext(selIdx, e.target.files)} />
                    <button onClick={() => agendaContextInputRefs.current[selIdx]?.click()} disabled={agendaContextUploading[selIdx]} style={{ display: 'flex', alignItems: 'center', gap: 5, padding: '.35rem .7rem', borderRadius: 8, cursor: 'pointer', fontFamily: 'Inter', fontSize: '.73rem', fontWeight: 600, border: '1.5px dashed hsl(45,90%,50%/.5)', background: 'hsl(45,90%,50%/.06)', color: 'hsl(45,90%,55%)', marginBottom: '.4rem' }}>
                      {agendaContextUploading[selIdx] ? <><Loader size={10} className="spin" /> Extracting...</> : <><Plus size={10} /> Upload Agenda Context File</>}
                    </button>
                    {selCtxFiles.length > 0 && (
                      <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                        {selCtxFiles.map((f, fi) => (
                          <div key={fi} style={{ borderRadius: 8, border: '1px solid hsl(45,90%,50%/.25)', background: 'hsl(45,90%,50%/.04)', padding: '.45rem .7rem' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 3 }}>
                              <FileText size={10} style={{ color: 'hsl(45,90%,55%)', flexShrink: 0 }} />
                              <span style={{ flex: 1, fontSize: '.72rem', fontWeight: 600, color: 'hsl(var(--ink))' }}>{f.name}</span>
                              <span style={{ fontSize: '.67rem', color: 'hsl(var(--pencil))' }}>{f.charCount.toLocaleString()} ch</span>
                              <button onClick={() => removeAgendaContextFile(selIdx, fi)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'hsl(var(--destructive))' }}><X size={11} /></button>
                            </div>
                            <pre style={{ margin: 0, fontSize: '.7rem', color: 'hsl(var(--ink))', fontFamily: 'Inter', lineHeight: 1.45, whiteSpace: 'pre-wrap', wordBreak: 'break-word', maxHeight: 100, overflowY: 'auto' }}>{f.text.slice(0, 500)}{f.text.length > 500 ? '...' : ''}</pre>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
  {/* Section 1: Timeline Transcript */}
                  <div>
                    <SectionHeader icon={<Clock size={13} />} label="Timeline Transcript" count={selTimeline.length} color="hsl(205,90%,55%)" />
                    {selTimeline.length === 0 ? (
                      <p style={{ margin: '4px 0', fontSize: '.75rem', color: 'hsl(var(--pencil))', fontStyle: 'italic' }}>
                        No transcript segments in time window ({!recording?.transcript?.length ? 'no transcript loaded' : `stride ${timelineStride}s around slot ${selIdx + 1}/${editedAgendaItems.length}`})
                      </p>
                    ) : (
                      <div style={{ display: 'flex', flexDirection: 'column', gap: 4, maxHeight: "200px", overflowY: "scroll", backgroundColor: "hsl(var(--muted)/.5)" }}>
                        {mergeConsecutiveSegments(selTimeline).map((block, bi) => (
                          <div key={bi} style={{ borderRadius: 8, border: '1px solid hsl(205,90%,55%/.25)', background: 'hsl(205,90%,55%/.04)', padding: '.45rem .7rem' }}>
                            <div style={{ display: 'flex', gap: 8, marginBottom: 2 }}>
                              <span style={{ fontSize: '.71rem', fontWeight: 700, color: 'hsl(205,90%,60%)' }}><User size={10} style={{ display: 'inline', marginRight: 2 }} />{block.speaker}</span>
                            </div>
                            <p style={{ margin: 0, fontSize: '.77rem', color: 'hsl(var(--ink))', fontFamily: 'Inter', lineHeight: 1.55 }}>{block.text}</p>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>

                  {/* Retrieval loading */}
                  {retrieving && (
                    <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '1.5rem', borderRadius: 10, background: 'hsl(var(--muted)/.3)', border: '1px dashed hsl(var(--border)/.4)' }}>
                      <Loader size={18} className="spin" style={{ color: 'hsl(280,75%,65%)' }} />
                      <span style={{ fontSize: '.8rem', color: 'hsl(var(--pencil))', fontFamily: 'Inter' }}>Embedding transcript and running FAISS retrieval...</span>
                    </div>
                  )}

                  {/* Placeholder when no evidence is retrieved yet */}
                  {!selEvidence && !retrieving && (
                    <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '1.25rem', borderRadius: 10, background: 'hsl(var(--muted)/.2)', border: '1px dashed hsl(var(--border)/.3)' }}>
                      <Search size={16} style={{ color: 'hsl(var(--pencil))', opacity: 0.5 }} />
                      <span style={{ fontSize: '.76rem', color: 'hsl(var(--pencil))', fontFamily: 'Inter' }}>
                        Click "Retrieve Evidence" in the left panel to populate additional transcript, meeting, and global context chunks.
                      </span>
                    </div>
                  )}

                  {/* Retrieved Evidence */}
                  {selEvidence && !retrieving && (
                    <>
                      {/* Additional Transcript */}
                      <div>
                        <SectionHeader icon={<Zap size={13} />} label="Additional Transcript Evidence" count={selEvidence.transcript_chunks.length} color="hsl(280,75%,60%)" />
                        <ChunkGroup label={`Transcript Chunks (${selEvidence.transcript_chunks.length})`} color="hsl(280,75%,60%)" chunks={selEvidence.transcript_chunks} agendaIndex={selIdx} disabled={disabledChunks} onToggle={toggleChunk} defaultOpen={true} />
                      </div>

                      {selEvidence.meeting_chunks.length > 0 && (
                        <div>
                          <SectionHeader icon={<FileText size={13} />} label="Meeting Context" count={selEvidence.meeting_chunks.length} color="hsl(140,70%,45%)" />
                          <ChunkGroup label={`Meeting Chunks (${selEvidence.meeting_chunks.length})`} color="hsl(140,70%,45%)" chunks={selEvidence.meeting_chunks} agendaIndex={selIdx} disabled={disabledChunks} onToggle={toggleChunk} defaultOpen={true} />
                        </div>
                      )}

                      {selEvidence.global_chunks.length > 0 && (
                        <div>
                          <SectionHeader icon={<Brain size={13} />} label="Global Context" count={selEvidence.global_chunks.length} color="hsl(30,90%,55%)" />
                          <ChunkGroup label={`Global Chunks (${selEvidence.global_chunks.length})`} color="hsl(30,90%,55%)" chunks={selEvidence.global_chunks} agendaIndex={selIdx} disabled={disabledChunks} onToggle={toggleChunk} defaultOpen={true} />
                        </div>
                      )}

                      {/* Final Evidence Preview */}
                      <div>
                        <button onClick={() => setShowEvidencePreview(prev => { const next = new Set(prev); if (next.has(selIdx)) next.delete(selIdx); else next.add(selIdx); return next })} style={{ display: 'flex', alignItems: 'center', gap: 6, width: '100%', textAlign: 'left', background: 'hsl(var(--muted)/.3)', border: '1px solid hsl(var(--border)/.4)', borderRadius: 8, padding: '.4rem .75rem', cursor: 'pointer' }}>
                          {showEvidencePreview.has(selIdx) ? <ChevronDown size={12} style={{ color: 'hsl(var(--pencil))' }} /> : <ChevronRight size={12} style={{ color: 'hsl(var(--pencil))' }} />}
                          <span style={{ fontSize: '.73rem', fontWeight: 700, color: 'hsl(var(--pencil))', textTransform: 'uppercase', letterSpacing: '.04em' }}>Final Evidence Preview</span>
                        </button>
                        {showEvidencePreview.has(selIdx) && (
                          <pre style={{ marginTop: 6, fontSize: '.7rem', color: 'hsl(var(--ink))', fontFamily: 'Inter', lineHeight: 1.5, whiteSpace: 'pre-wrap', wordBreak: 'break-word', background: 'hsl(var(--paper)/.5)', padding: '.6rem', borderRadius: 9, border: '1px solid hsl(var(--border)/.3)', maxHeight: 240, overflowY: 'auto' }}>
                            {assembleEvidenceForAgenda(selIdx, selItem, selTimeline, selCtxFiles, selEvidence, disabledChunks, charLimit) || <span style={{ color: 'hsl(var(--pencil))', fontStyle: 'italic' }}>No evidence assembled</span>}
                          </pre>
                        )}
                      </div>
                    </>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Raw MoM Result */}
          {momResult && (
            <div style={{ borderRadius: 14, border: '1.5px solid hsl(140,70%,45%/.3)', background: 'hsl(var(--card))', overflow: 'scroll' }}>
              <div style={{ background: 'linear-gradient(135deg,hsl(140,70%,45%/.1),hsl(160,70%,45%/.06))', borderBottom: '1px solid hsl(140,70%,45%/.2)', padding: '.7rem 1rem', display: 'flex', alignItems: 'center', gap: 8 }}>
                <CheckCircle size={14} style={{ color: 'hsl(140,70%,45%)' }} />
                <span style={{ fontSize: '.78rem', fontWeight: 700, color: 'hsl(var(--ink))' }}>Raw MoM Generated</span>
                <span style={{ fontSize: '.69rem', color: 'hsl(var(--pencil))', marginLeft: 'auto' }}>
                  {(momResult.meeting?.agendas || momResult.agendas || []).length} agendas
                </span>
              </div>
              <div style={{ padding: '.7rem 1rem', display: 'flex', flexDirection: 'column', gap: '.55rem' }}>
                {(momResult.meeting?.agendas || momResult.agendas || []).map((ag, ai) => (
                  <div key={ai} style={{ borderRadius: 9, border: '1px solid hsl(var(--border)/.4)', overflow: 'scroll' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '.5rem .85rem', background: 'hsl(var(--muted)/.3)', borderBottom: '1px solid hsl(var(--border)/.25)' }}>
                      <span style={{ fontSize: '.71rem', fontWeight: 700, color: 'hsl(var(--pencil))' }}>Agenda {ai + 1}</span>
                      <span style={{ flex: 1, fontSize: '.81rem', fontWeight: 600, color: 'hsl(var(--ink))' }}>{ag.agenda_topic}</span>
                      {ag.agenda_speaker && <span style={{ fontSize: '.7rem', color: 'hsl(var(--pencil))' }}><User size={9} style={{ display: 'inline', marginRight: 2 }} />{ag.agenda_speaker}</span>}
                      <span style={{ fontSize: '.69rem', color: 'hsl(140,70%,45%)', background: 'hsl(140,70%,45%/.1)', border: '1px solid hsl(140,70%,45%/.3)', padding: '2px 7px', borderRadius: 999 }}>
                        {Array.isArray(ag.discussion) ? ag.discussion.length : 1} pts
                      </span>
                    </div>
                    {Array.isArray(ag.discussion) ? (
                      ag.discussion.map((disc, di) => (
                        <div key={di} style={{ padding: '.5rem .85rem', borderBottom: di < ag.discussion.length - 1 ? '1px solid hsl(var(--border)/.15)' : 'none' }}>
                          {disc.speaker && <div style={{ fontSize: '.72rem', color: 'hsl(var(--pencil))', fontWeight: 600, marginBottom: 2 }}>{disc.speaker}</div>}
                          <p style={{ margin: 0, fontSize: '.79rem', color: 'hsl(var(--ink))', fontFamily: 'Inter', lineHeight: 1.55 }}>{disc.point}</p>
                          {disc.action?.description && <p style={{ margin: '3px 0 0', fontSize: '.72rem', color: 'hsl(30,90%,55%)', fontFamily: 'Inter' }}>&#8594; {disc.action.description}{disc.action.owner ? ` (${disc.action.owner})` : ''}{disc.action.deadline ? ` | ${disc.action.deadline}` : ''}</p>}
                        </div>
                      ))
                    ) : (
                      <div style={{ padding: '.65rem .85rem', background: 'hsl(var(--paper)/.5)' }}>
                        <pre style={{ margin: 0, fontSize: '.76rem', color: 'hsl(var(--ink))', fontFamily: 'Inter', lineHeight: 1.5, whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>{String(ag.discussion)}</pre>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Final MoM */}
          {finalMomResult && (
            <div style={{ borderRadius: 14, border: '1.5px solid hsl(220,80%,60%/.3)', background: 'hsl(var(--card))', overflow: 'hidden' }}>
              <div style={{ background: 'linear-gradient(135deg,hsl(220,80%,55%/.12),hsl(260,70%,60%/.08))', borderBottom: '1px solid hsl(220,80%,60%/.2)', padding: '.9rem 1.25rem', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <div>
                  <h2 style={{ margin: 0, fontSize: '1rem', fontWeight: 700, color: 'hsl(var(--ink))', fontFamily: 'Inter' }}>{finalMomResult.title || 'Minutes of Meeting'}</h2>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px 16px', marginTop: '.35rem', fontSize: '.72rem', color: 'hsl(var(--pencil))' }}>
                    {finalMomResult.date && <span><Calendar size={10} style={{ display: 'inline', marginRight: 2 }} />{new Date(finalMomResult.date).toLocaleDateString()}</span>}
                    {finalMomResult.participants?.length > 0 && <span><Users size={10} style={{ display: 'inline', marginRight: 2 }} />{finalMomResult.participants.join(', ')}</span>}
                  </div>
                </div>
                <button onClick={() => navigate(`/mom/${id}`)} style={{ display: 'flex', alignItems: 'center', gap: 5, padding: '.38rem .75rem', borderRadius: 8, border: '1.5px solid hsl(220,80%,60%/.4)', background: 'hsl(220,80%,60%/.08)', color: 'hsl(220,80%,65%)', fontSize: '.76rem', fontWeight: 600, cursor: 'pointer', fontFamily: 'Inter' }}>
                  <ExternalLink size={12} /> View in MoM Editor
                </button>
              </div>
              <div style={{ padding: '.9rem 1.25rem', display: 'flex', flexDirection: 'column', gap: '.9rem' }}>
                {finalMomResult.introduction && <p style={{ margin: 0, fontSize: '.8rem', color: 'hsl(var(--ink))', fontFamily: 'Inter', lineHeight: 1.65 }}>{finalMomResult.introduction}</p>}
                {finalMomResult.action_items?.length > 0 && (
                  <div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: '.5rem' }}>
                      <ListChecks size={13} style={{ color: 'hsl(30,90%,55%)' }} />
                      <span style={{ fontSize: '.72rem', fontWeight: 700, color: 'hsl(var(--pencil))', textTransform: 'uppercase', letterSpacing: '.05em' }}>Action Items</span>
                    </div>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                      {finalMomResult.action_items.map((ai, i) => (
                        <div key={i} style={{ borderRadius: 8, border: '1px solid hsl(var(--border)/.4)', padding: '.65rem .85rem', background: 'hsl(var(--card))' }}>
                          <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'space-between', alignItems: 'center', gap: '6px', marginBottom: '4px' }}>
                            <span style={{ fontSize: '.79rem', fontWeight: 700, color: 'hsl(var(--ink))' }}>{ai.task}</span>
                            <div style={{ display: 'flex', gap: '6px', fontSize: '.71rem', fontFamily: 'Inter', fontWeight: 600 }}>
                              <span style={{ padding: '2px 7px', borderRadius: 4, background: 'hsl(var(--muted))', color: 'hsl(var(--pencil))' }}>Owner: {ai.owner || 'Unassigned'}</span>
                              <span style={{ padding: '2px 7px', borderRadius: 4, background: 'hsl(var(--muted))', color: 'hsl(var(--pencil))' }}>Deadline: {ai.deadline || 'ASAP'}</span>
                            </div>
                          </div>
                          {ai.background && (
                            <p style={{ margin: '4px 0 0', fontSize: '.73rem', color: 'hsl(var(--pencil))', lineHeight: 1.45 }}>
                              <strong>Context:</strong> {ai.background}
                            </p>
                          )}
                          {ai.status && (
                            <p style={{ margin: '4px 0 0', fontSize: '.73rem', color: 'hsl(30,90%,55%)', lineHeight: 1.45 }}>
                              <strong>Status/Outcome:</strong> {ai.status}
                            </p>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
