import { useState, useCallback, useEffect, useRef } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import {
  FlaskConical, ChevronDown, ChevronRight, Search, Loader,
  CheckCircle, X, Download, FileText, Clock, User,
  Sparkles, FileDown, ArrowLeft, Trash2, Upload, Users, Calendar, Brain, AlertTriangle
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

interface AgendaResult {
  topic: string
  speaker: string | null
  is_procedural: boolean
  transcript_chunks: TranscriptChunk[]
  meeting_chunks: MeetingChunk[]
  global_chunks: GlobalChunk[]
}

interface RetrieveResponse {
  agendas: AgendaResult[]
  retrieval_params: {
    k_transcript: number
    k_meeting: number
    k_global: number
    relative_similarity_cutoff: number
    char_limit: number
  }
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
  meeting: {
    agendas: AgendaMom[]
  }
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

function scoreColor(score: number): string {
  if (score >= 0.85) return 'hsl(140,70%,45%)'
  if (score >= 0.70) return 'hsl(45,90%,50%)'
  return 'hsl(0,80%,55%)'
}

function assembleEvidence(agenda: AgendaResult, agendaIndex: number, disabled: Set<string>): string {
  const sections: string[] = []
  const tLines = agenda.transcript_chunks
    .filter(c => !disabled.has(`${agendaIndex}_${c.chunk_id}`)).map(c => c.text)
  if (tLines.length) sections.push('=== TRANSCRIPT (Actual Discussion) ===\n' + tLines.join('\n'))
  const mLines = agenda.meeting_chunks
    .filter(c => !disabled.has(`${agendaIndex}_${c.chunk_id}`)).map(c => c.filename ? `[${c.filename}] ${c.text}` : c.text)
  if (mLines.length) sections.push('=== MEETING CONTEXT (Slides / Documents) ===\n' + mLines.join('\n'))
  const gLines = agenda.global_chunks
    .filter(c => !disabled.has(`${agendaIndex}_${c.chunk_id}`)).map(c => c.filename ? `[${c.filename}] ${c.text}` : c.text)
  if (gLines.length) sections.push('=== GLOBAL CONTEXT (Organizational Knowledge) ===\n' + gLines.join('\n'))
  return sections.join('\n\n')
}

// ── Sub-components ────────────────────────────────────────────────────────────

function ScorePill({ score }: { score: number }) {
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center',
      fontSize: '.72rem', fontWeight: 700, color: scoreColor(score),
      background: `${scoreColor(score)}1a`, border: `1px solid ${scoreColor(score)}44`,
      padding: '2px 7px', borderRadius: 999, fontFamily: 'JetBrains Mono, monospace',
    }}>
      {score.toFixed(3)}
    </span>
  )
}

function ChunkRow({ chunk, agendaIndex, disabled, onToggle }: {
  chunk: AnyChunk; agendaIndex: number; disabled: boolean; onToggle: (agendaIndex: number, id: string) => void
}) {
  const [expanded, setExpanded] = useState(false)
  const tChunk = chunk.source === 'transcript' ? chunk as TranscriptChunk : null
  const mChunk = chunk.source === 'meeting' ? chunk as MeetingChunk : null
  const gChunk = chunk.source === 'global' ? chunk as GlobalChunk : null

  return (
    <div style={{
      borderRadius: 10, border: '1px solid hsl(var(--border) / .4)',
      background: disabled ? 'hsl(var(--muted) / .3)' : 'hsl(var(--card))',
      overflow: 'hidden', opacity: disabled ? 0.5 : 1, transition: 'opacity .15s',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '.5rem .85rem', cursor: 'pointer' }}
        onClick={() => setExpanded(e => !e)}>
        {expanded ? <ChevronDown size={13} style={{ flexShrink: 0, color: 'hsl(var(--pencil))' }} />
          : <ChevronRight size={13} style={{ flexShrink: 0, color: 'hsl(var(--pencil))' }} />}
        <span style={{ fontSize: '.78rem', fontWeight: 600, color: 'hsl(var(--ink))', minWidth: 56 }}>
          Chunk {chunk.chunk_index}
        </span>
        <ScorePill score={chunk.score} />
        {tChunk && tChunk.speakers.length > 0 && (
          <span style={{ fontSize: '.73rem', color: 'hsl(var(--pencil))' }}>
            <User size={10} style={{ display: 'inline', marginRight: 2 }} />{tChunk.speakers.join(', ')}
          </span>
        )}
        {tChunk && (
          <span style={{ fontSize: '.73rem', color: 'hsl(var(--pencil))' }}>
            <Clock size={10} style={{ display: 'inline', marginRight: 2 }} />
            {fmtTime(tChunk.start)} – {fmtTime(tChunk.end)}
          </span>
        )}
        {(mChunk || gChunk) && (
          <span style={{ fontSize: '.73rem', color: 'hsl(var(--pencil))', flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            <FileText size={10} style={{ display: 'inline', marginRight: 2 }} />
            {mChunk?.filename || gChunk?.filename || ''}
            {mChunk?.page != null ? ` · p.${mChunk.page}` : ''}
          </span>
        )}
        <span style={{ marginLeft: 'auto', fontSize: '.7rem', color: 'hsl(var(--pencil))', whiteSpace: 'nowrap' }}>
          {chunk.char_count.toLocaleString()} chars
        </span>
        <button
          onClick={e => { e.stopPropagation(); onToggle(agendaIndex, chunk.chunk_id) }}
          style={{
            flexShrink: 0, display: 'flex', alignItems: 'center', gap: 4,
            fontSize: '.7rem', fontWeight: 600, padding: '3px 7px', borderRadius: 7,
            border: `1px solid ${disabled ? 'hsl(var(--destructive) / .3)' : 'hsl(140,70%,45% / .3)'}`,
            background: disabled ? 'hsl(var(--destructive) / .08)' : 'hsl(140,70%,45% / .08)',
            color: disabled ? 'hsl(var(--destructive))' : 'hsl(140,70%,45%)', cursor: 'pointer',
          }}
        >
          {disabled ? <><X size={10} /> Excluded</> : <><CheckCircle size={10} /> Include</>}
        </button>
      </div>
      {expanded && (
        <div style={{ borderTop: '1px solid hsl(var(--border) / .25)', padding: '.65rem .9rem' }}>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px 14px', marginBottom: '.6rem' }}>
            <span style={{ fontSize: '.71rem', color: 'hsl(var(--pencil))', fontFamily: 'Inter' }}>
              <strong>Similarity:</strong> {chunk.score.toFixed(4)}
            </span>
            <span style={{ fontSize: '.71rem', color: 'hsl(var(--pencil))', fontFamily: 'Inter' }}>
              <strong>Chunk #:</strong> {chunk.chunk_index}
            </span>
            {tChunk && <>
              <span style={{ fontSize: '.71rem', color: 'hsl(var(--pencil))', fontFamily: 'Inter' }}>
                <strong>Speakers:</strong> {tChunk.speakers.join(', ') || '—'}
              </span>
              <span style={{ fontSize: '.71rem', color: 'hsl(var(--pencil))', fontFamily: 'Inter' }}>
                <strong>Words:</strong> {tChunk.word_count}
              </span>
            </>}
            {(mChunk || gChunk) && (
              <span style={{ fontSize: '.71rem', color: 'hsl(var(--pencil))', fontFamily: 'Inter' }}>
                <strong>File:</strong> {mChunk?.filename || gChunk?.filename || '—'}
                {mChunk?.page != null ? ` · page ${mChunk.page}` : ''}
              </span>
            )}
          </div>
          <pre style={{
            margin: 0, fontSize: '.76rem', color: 'hsl(var(--ink))', fontFamily: 'Inter',
            lineHeight: 1.55, whiteSpace: 'pre-wrap', wordBreak: 'break-word',
            background: 'hsl(var(--paper) / .5)', padding: '.65rem', borderRadius: 8,
            border: '1px solid hsl(var(--border) / .3)', maxHeight: 280, overflowY: 'auto',
          }}>
            {chunk.text}
          </pre>
        </div>
      )}
    </div>
  )
}

function ChunkGroup({ label, color, chunks, agendaIndex, disabled, onToggle }: {
  label: string; color: string; chunks: AnyChunk[]; agendaIndex: number; disabled: Set<string>; onToggle: (agendaIndex: number, id: string) => void
}) {
  const [open, setOpen] = useState(false) // chunks collapsed by default
  const activeCount = chunks.filter(c => !disabled.has(`${agendaIndex}_${c.chunk_id}`)).length
  return (
    <div style={{ marginBottom: '.4rem' }}>
      <button onClick={() => setOpen(o => !o)} style={{
        display: 'flex', alignItems: 'center', gap: 6, width: '100%', textAlign: 'left',
        background: 'none', border: 'none', cursor: 'pointer', padding: '.35rem .45rem', borderRadius: 7,
      }}>
        {open ? <ChevronDown size={12} style={{ color }} /> : <ChevronRight size={12} style={{ color }} />}
        <span style={{ fontSize: '.74rem', fontWeight: 700, color, textTransform: 'uppercase', letterSpacing: '.04em' }}>
          {label}
        </span>
        <span style={{
          fontSize: '.68rem', fontWeight: 600, color: activeCount > 0 ? color : 'hsl(var(--pencil))',
          background: `${color}1a`, border: `1px solid ${color}33`, padding: '1px 6px', borderRadius: 999,
        }}>
          {activeCount}/{chunks.length}
        </span>
      </button>
      {open && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4, paddingLeft: '.65rem', marginTop: 3 }}>
          {chunks.length === 0
            ? <p style={{ fontSize: '.73rem', color: 'hsl(var(--pencil))', margin: '2px 0' }}>No chunks retrieved</p>
            : chunks.map(c => (
              <ChunkRow key={c.chunk_id} chunk={c} agendaIndex={agendaIndex} disabled={disabled.has(`${agendaIndex}_${c.chunk_id}`)} onToggle={onToggle} />
            ))}
        </div>
      )}
    </div>
  )
}

// ── Main Page ─────────────────────────────────────────────────────────────────

export default function RawMomLab() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()

  // Recording Info
  const [recording, setRecording] = useState<RecordingDetail | null>(null)
  const [loadingRec, setLoadingRec] = useState(true)

  // Attachments Info
  const [agendaFiles, setAgendaFiles] = useState<AttachmentFile[]>([])
  const [contextFiles, setContextFiles] = useState<AttachmentFile[]>([])
  const [agendaProcessState, setAgendaProcessState] = useState<ProcessState>('idle')
  const [contextProcessState, setContextProcessState] = useState<ProcessState>('idle')
  const [agendaSummary, setAgendaSummary] = useState<string | null>(null)
  const [referenceSummary, setReferenceSummary] = useState<string | null>(null)
  const agendaInputRef = useRef<HTMLInputElement>(null)
  const contextInputRef = useRef<HTMLInputElement>(null)

  // Retrieval params
  const [kTranscript, setKTranscript] = useState(8)
  const [kMeeting, setKMeeting] = useState(4)
  const [kGlobal, setKGlobal] = useState(2)
  const [cutoff, setCutoff] = useState(0.01)
  const [charLimit, setCharLimit] = useState(15000)
  const [forceReembed, setForceReembed] = useState(false)

  // Retrieval & Generation States
  const [retrieving, setRetrieving] = useState(false)
  const [generating, setGenerating] = useState(false)
  const [retrieveResult, setRetrieveResult] = useState<RetrieveResponse | null>(null)
  const [momResult, setMomResult] = useState<RawMomResult | null>(null)

  // Interaction States
  const [disabledChunks, setDisabledChunks] = useState<Set<string>>(new Set())
  const [expandedAgendas, setExpandedAgendas] = useState<Set<number>>(new Set())
  const [expandedMomRows, setExpandedMomRows] = useState<Set<number>>(new Set())

  // ── Fetch Recording & Attachments on Mount ─────────────────────────────
  const loadInitialData = useCallback(async () => {
    if (!id) return
    setLoadingRec(true)
    try {
      const recRes = await api.get(`/history/${id}`)
      setRecording(recRes.data)

      // Fetch context / agenda files exactly like MomPage.tsx
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

      // Try load existing Raw MoM if already generated
      try {
        const momRes = await api.get(`/raw-mom/${id}`)
        setMomResult(momRes.data)
      } catch { /* 404 is normal for not generated */ }

    } catch {
      toast.error('Failed to load recording details')
    } finally {
      setLoadingRec(false)
    }
  }, [id])

  useEffect(() => {
    loadInitialData()
  }, [loadInitialData])

  // ── Attachment handlers (exactly like MomPage.tsx) ────────────────────
  const handleUploadFiles = async (files: FileList | null, type: 'agenda' | 'context') => {
    if (!files || files.length === 0 || !id) return
    const formData = new FormData()
    formData.append('type', type)
    Array.from(files).forEach(f => formData.append('files', f))
    try {
      await api.post(`/attachments/${id}/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      const res = await api.get(`/attachments/${id}`)
      const all: AttachmentFile[] = res.data.files || []
      const filtered = all.filter(f => f.type === type)
      if (type === 'agenda') setAgendaFiles(filtered)
      else setContextFiles(filtered)

      if (filtered.length > 0) {
        await handleProcessFiles(type)
      }
    } catch (e) {
      toast.error(`Upload failed: ${getApiErrorDetail(e)}`)
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

      if (remainingCount > 0) {
        await handleProcessFiles(type)
      }
    } catch (e) {
      toast.error(`Delete failed: ${getApiErrorDetail(e)}`)
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
      toast.error(`Processing failed: ${getApiErrorDetail(e)}`)
      setState('error')
    }
  }

  // ── Retrieve Chunks ───────────────────────────────────────────────────────
  const handleRetrieve = async () => {
    if (!id) return
    setRetrieving(true)
    setRetrieveResult(null)
    setMomResult(null)
    setDisabledChunks(new Set())
    try {
      const res = await api.post(`/raw-mom/${id}/retrieve`, {
        k_transcript: kTranscript, k_meeting: kMeeting, k_global: kGlobal,
        relative_similarity_cutoff: cutoff, char_limit: charLimit, force_reembed: forceReembed,
      })
      setRetrieveResult(res.data)
      if (res.data.agendas?.length > 0) setExpandedAgendas(new Set([0]))
      toast.success(`Retrieved ${res.data.agendas?.length} agenda items`)
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

  // ── Generate Raw MoM ──────────────────────────────────────────────────────
  const handleGenerate = async () => {
    if (!id || !retrieveResult) { toast.error('Retrieve evidence first'); return }
    setGenerating(true)
    try {
      const agendas = retrieveResult.agendas.map((a, idx) => ({
        topic: a.topic, speaker: a.speaker, evidence: assembleEvidence(a, idx, disabledChunks)
      }))
      const res = await api.post(`/raw-mom/${id}/generate`, { agendas })
      setMomResult(res.data)
      toast.success('Raw MoM generated')
    } catch (e: any) {
      toast.error(e?.response?.data?.detail ?? 'Generation failed')
    } finally {
      setGenerating(false)
    }
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
      const a = document.createElement('a')
      a.href = url
      a.download = `raw_mom_${id}.docx`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
      toast.success('DOCX downloaded successfully')
    } catch {
      toast.error('Failed to download DOCX')
    }
  }

  const segmentCount = recording?.transcript ? recording.transcript.length : 0
  const speakerCount = recording?.speakers_detected ? recording.speakers_detected.length : 0
  const transcriptPreview = recording?.transcript
    ? recording.transcript.slice(0, 3).map(s => `${s.speaker_label}: ${s.text}`).join('\n')
    : 'No transcript segments available.'

  const totalChunks = retrieveResult?.agendas.reduce(
    (acc, a) => acc + a.transcript_chunks.length + a.meeting_chunks.length + a.global_chunks.length, 0
  ) ?? 0

  if (loadingRec) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', flexDirection: 'column', gap: '1rem' }}>
        <Loader size={24} className="spin" style={{ color: 'hsl(var(--accent))' }} />
        <p style={{ fontSize: '.9rem', color: 'hsl(var(--pencil))', fontFamily: 'Inter, sans-serif' }}>Loading recording details...</p>
      </div>
    )
  }

  return (
    <div className="page-scroll-root" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>

      {/* Header with back button */}
      <div className="panel-header" style={{ flexShrink: 0, gap: '12px' }}>
        <button className="icon-btn" onClick={() => navigate(-1)} title="Go Back">
          <ArrowLeft size={16} />
        </button>
        <div style={{
          width: 32, height: 32, borderRadius: '8px', flexShrink: 0,
          background: 'hsl(280,75%,60% / .15)', border: '1.5px solid hsl(280,75%,60% / .3)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}>
          <FlaskConical size={15} style={{ color: 'hsl(280,75%,65%)' }} />
        </div>
        <div style={{ flex: 1 }}>
          <h1 style={{ fontSize: '1.05rem', fontWeight: 700 }}>Raw MoM Lab (Advanced Debugger)</h1>
          <p style={{ fontSize: '.78rem', color: 'hsl(var(--pencil))', fontFamily: 'Inter', fontWeight: 400, marginTop: '1px' }}>
            Inspect RAG retrieval, toggle evidence chunks, and test facts generation for "{recording?.filename}"
          </p>
        </div>
      </div>

      {/* Main split layout */}
      <div style={{ flex: 1, display: 'flex', gap: '1.25rem', padding: '1.25rem 1.75rem 3rem', minHeight: 0, overflow: 'hidden' }}>

        {/* ── LEFT PANEL: Recording metadata + Uploaders + Settings ── */}
        <div style={{
          width: 330, flexShrink: 0, display: 'flex', flexDirection: 'column', gap: '.85rem',
          overflowY: 'auto', paddingRight: 6
        }}>

          {/* Recording Info Card */}
          <div style={{ borderRadius: 12, border: '1.5px solid hsl(var(--border) / .5)', background: 'hsl(var(--card))', padding: '.9rem' }}>
            <div style={{ fontSize: '.72rem', fontWeight: 700, color: 'hsl(var(--pencil))', textTransform: 'uppercase', letterSpacing: '.05em', marginBottom: '.6rem' }}>Recording Details</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '5px', fontSize: '.76rem', color: 'hsl(var(--ink))', fontFamily: 'Inter' }}>
              <div><strong>Name:</strong> {recording?.filename}</div>
              <div><strong>Duration:</strong> {recording ? fmtTime(recording.duration) : '—'}</div>
              <div><strong>Date:</strong> {recording ? new Date(recording.created_at).toLocaleString() : '—'}</div>
            </div>

            <div style={{ height: '1px', background: 'hsl(var(--border) / .3)', margin: '.7rem 0' }} />

            {/* Transcript Preview */}
            <div style={{ fontSize: '.72rem', fontWeight: 700, color: 'hsl(var(--pencil))', textTransform: 'uppercase', letterSpacing: '.05em', marginBottom: '.4rem' }}>Transcript Info</div>
            <div style={{ display: 'flex', gap: '12px', fontSize: '.76rem', marginBottom: '.5rem', color: 'hsl(var(--pencil))' }}>
              <span>Segments: <strong style={{ color: 'hsl(var(--ink))' }}>{segmentCount}</strong></span>
              <span>Speakers: <strong style={{ color: 'hsl(var(--ink))' }}>{speakerCount}</strong></span>
            </div>
            <pre style={{
              margin: 0, fontSize: '.7rem', color: 'hsl(var(--pencil))', fontFamily: 'Inter',
              lineHeight: 1.4, whiteSpace: 'pre-wrap', background: 'hsl(var(--paper) / .4)',
              padding: '.45rem', borderRadius: 6, border: '1px solid hsl(var(--border) / .3)',
              maxHeight: 70, overflowY: 'auto', fontStyle: 'italic'
            }}>
              {transcriptPreview}
            </pre>
          </div>

          {/* Agenda & Context Uploaders (identical to MomPage.tsx) */}
          <div style={{ borderRadius: 12, border: '1.5px solid hsl(var(--border) / .5)', background: 'hsl(var(--card))', padding: '.9rem', display: 'flex', flexDirection: 'column', gap: '.75rem' }}>
            
            <input ref={agendaInputRef} type="file" multiple accept=".pdf,.docx,.pptx,.txt,.md,.png,.jpg,.jpeg,.webp,.xlsx,.xls,.csv" style={{ display: 'none' }}
              onChange={e => handleUploadFiles(e.target.files, 'agenda')} />
            <input ref={contextInputRef} type="file" multiple accept=".pdf,.docx,.pptx,.txt,.md,.png,.jpg,.jpeg,.webp,.xlsx,.xls,.csv" style={{ display: 'none' }}
              onChange={e => handleUploadFiles(e.target.files, 'context')} />

            {/* Agenda File Selector */}
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '5px', marginBottom: '0.45rem' }}>
                <FileText size={12} style={{ color: 'hsl(var(--accent))' }} />
                <span style={{ fontSize: '.74rem', fontWeight: 700, color: 'hsl(var(--ink))', fontFamily: 'Inter' }}>Agenda Uploader</span>
              </div>
              {agendaFiles.length > 0 ? (
                <div style={{ marginBottom: '.4rem', display: 'flex', flexDirection: 'column', gap: 3 }}>
                  {agendaFiles.map(f => (
                    <div key={f.id} style={{ display: 'flex', alignItems: 'center', gap: 5, padding: '3px 6px', borderRadius: 5, background: 'hsl(var(--muted) / .5)', fontSize: '.74rem' }}>
                      <FileText size={10} style={{ flexShrink: 0 }} />
                      <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{f.filename}</span>
                      <button onClick={() => handleDeleteFile(f.id, 'agenda')} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'hsl(var(--destructive))', flexShrink: 0 }}>
                        <Trash2 size={10} />
                      </button>
                    </div>
                  ))}
                </div>
              ) : (
                <div onClick={() => agendaInputRef.current?.click()} style={{ border: '1.5px dashed hsl(var(--border) / .6)', borderRadius: 7, padding: '.45rem', textAlign: 'center', cursor: 'pointer', fontSize: '.74rem', color: 'hsl(var(--pencil))', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 4, marginBottom: '.4rem' }}>
                  <Upload size={11} /> Upload Agenda File
                </div>
              )}
              {agendaFiles.length > 0 && (
                <button className="btn btn-ghost" disabled={agendaProcessState === 'processing'} onClick={() => handleProcessFiles('agenda')} style={{ width: '100%', fontSize: '.7rem', padding: '3px 6px', height: 24 }}>
                  {agendaProcessState === 'processing' ? 'Processing...' : 'Process / Re-parse Agenda'}
                </button>
              )}
            </div>

            <div style={{ height: '1px', background: 'hsl(var(--border) / .2)' }} />

            {/* Context Files Selector */}
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '5px', marginBottom: '0.45rem' }}>
                <FileText size={12} style={{ color: '#8b5cf6' }} />
                <span style={{ fontSize: '.74rem', fontWeight: 700, color: 'hsl(var(--ink))', fontFamily: 'Inter' }}>Context Uploader</span>
              </div>
              {contextFiles.length > 0 && (
                <div style={{ marginBottom: '.4rem', display: 'flex', flexDirection: 'column', gap: 3 }}>
                  {contextFiles.map(f => (
                    <div key={f.id} style={{ display: 'flex', alignItems: 'center', gap: 5, padding: '3px 6px', borderRadius: 5, background: 'hsl(var(--muted) / .5)', fontSize: '.74rem' }}>
                      <FileText size={10} style={{ flexShrink: 0 }} />
                      <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{f.filename}</span>
                      <button onClick={() => handleDeleteFile(f.id, 'context')} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'hsl(var(--destructive))', flexShrink: 0 }}>
                        <Trash2 size={10} />
                      </button>
                    </div>
                  ))}
                </div>
              )}
              <div onClick={() => contextInputRef.current?.click()} style={{ border: '1.5px dashed hsl(var(--border) / .6)', borderRadius: 7, padding: '.45rem', textAlign: 'center', cursor: 'pointer', fontSize: '.74rem', color: 'hsl(var(--pencil))', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 4, marginBottom: '.4rem' }}>
                <Upload size={11} /> Upload Context File
              </div>
              {contextFiles.length > 0 && (
                <button className="btn btn-ghost" disabled={contextProcessState === 'processing'} onClick={() => handleProcessFiles('context')} style={{ width: '100%', fontSize: '.7rem', padding: '3px 6px', height: 24 }}>
                  {contextProcessState === 'processing' ? 'Processing...' : 'Process / Re-parse Context'}
                </button>
              )}
            </div>

          </div>

          {/* Retrieval settings panel */}
          <div style={{ borderRadius: 12, border: '1.5px solid hsl(var(--border) / .5)', background: 'hsl(var(--card))', padding: '.9rem', display: 'flex', flexDirection: 'column', gap: '.65rem' }}>
            <div style={{ fontSize: '.72rem', fontWeight: 700, color: 'hsl(var(--pencil))', textTransform: 'uppercase', letterSpacing: '.05em' }}>Retrieval Overrides</div>

            {([
              { label: 'Transcript Top-K', value: kTranscript, setter: setKTranscript, color: 'hsl(205,90%,55%)' },
              { label: 'Meeting Context K', value: kMeeting, setter: setKMeeting, color: 'hsl(140,70%,50%)' },
              { label: 'Global Context K', value: kGlobal, setter: setKGlobal, color: 'hsl(30,90%,55%)' },
            ] as { label: string; value: number; setter: (v: number) => void; color: string }[]).map(({ label, value, setter, color }) => (
              <div key={label}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3 }}>
                  <label style={{ fontSize: '.74rem', color: 'hsl(var(--ink))', fontFamily: 'Inter' }}>{label}</label>
                  <span style={{ fontSize: '.74rem', fontWeight: 700, color, fontFamily: 'JetBrains Mono' }}>{value}</span>
                </div>
                <input type="range" min={0} max={20} value={value} onChange={e => setter(Number(e.target.value))}
                  style={{ width: '100%', accentColor: color }} />
              </div>
            ))}

            {[
              { label: 'Similarity Cutoff', value: cutoff, setter: setCutoff, step: 0.01, min: 0, max: 1 },
              { label: 'Max Characters', value: charLimit, setter: setCharLimit, step: 1000, min: 1000, max: 100000 },
            ].map(({ label, value, setter, step, min, max }) => (
              <div key={label}>
                <label style={{ fontSize: '.74rem', color: 'hsl(var(--ink))', fontFamily: 'Inter', display: 'block', marginBottom: 3 }}>{label}</label>
                <input type="number" min={min} max={max} step={step} value={value}
                  onChange={e => setter(Number(e.target.value))}
                  style={{
                    width: '100%', padding: '.32rem .55rem', borderRadius: 7,
                    border: '1.5px solid hsl(var(--border) / .5)',
                    background: 'hsl(var(--muted) / .5)', color: 'hsl(var(--ink))',
                    fontSize: '.78rem', fontFamily: 'JetBrains Mono', outline: 'none', boxSizing: 'border-box',
                  }} />
              </div>
            ))}

            <label style={{ display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer', fontSize: '.74rem', color: 'hsl(var(--ink))', fontFamily: 'Inter' }}>
              <input type="checkbox" checked={forceReembed} onChange={e => setForceReembed(e.target.checked)} />
              Force re-embed
            </label>
          </div>

          {/* Retrieve button */}
          <button onClick={handleRetrieve} disabled={retrieving}
            style={{
              display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 7,
              padding: '.7rem', borderRadius: 10, border: 'none',
              background: 'hsl(280,75%,60%)', color: '#fff',
              fontSize: '.84rem', fontWeight: 700, cursor: retrieving ? 'not-allowed' : 'pointer',
              opacity: retrieving ? 0.6 : 1, fontFamily: 'Inter',
            }}>
            {retrieving ? <Loader size={14} className="spin" /> : <Search size={14} />}
            {retrieving ? 'Retrieving Chunks…' : 'Retrieve Evidence'}
          </button>

          {retrieveResult && (
            <button onClick={handleGenerate} disabled={generating}
              style={{
                display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 7,
                padding: '.7rem', borderRadius: 10, border: 'none',
                background: generating ? 'hsl(140,70%,45% / .5)' : 'hsl(140,70%,45%)', color: '#fff',
                fontSize: '.84rem', fontWeight: 700, cursor: generating ? 'not-allowed' : 'pointer', fontFamily: 'Inter',
              }}>
              {generating ? <Loader size={14} className="spin" /> : <Sparkles size={14} />}
              {generating ? 'Extracting Raw MoM…' : 'Generate Raw MoM'}
            </button>
          )}

          {momResult && (
            <div style={{ display: 'flex', gap: 6 }}>
              <button onClick={downloadJson} style={{
                flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 5,
                padding: '.5rem', borderRadius: 8, border: '1.5px solid hsl(var(--border) / .6)',
                background: 'hsl(var(--card))', color: 'hsl(var(--ink))',
                fontSize: '.76rem', fontWeight: 600, cursor: 'pointer', fontFamily: 'Inter',
              }}>
                <Download size={12} /> JSON
              </button>
              <button onClick={downloadDocx} style={{
                flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 5,
                padding: '.5rem', borderRadius: 8, border: '1.5px solid hsl(205,90%,55% / .4)',
                background: 'hsl(205,90%,55% / .08)', color: 'hsl(205,90%,60%)',
                fontSize: '.76rem', fontWeight: 600, cursor: 'pointer', fontFamily: 'Inter',
              }}>
                <FileDown size={12} /> DOCX
              </button>
            </div>
          )}

          {retrieveResult && (
            <div style={{
              borderRadius: 10, padding: '.7rem', background: 'hsl(var(--card))',
              border: '1.5px solid hsl(var(--border) / .4)',
              fontSize: '.73rem', color: 'hsl(var(--pencil))', fontFamily: 'Inter',
            }}>
              <div style={{ fontWeight: 700, color: 'hsl(var(--ink))', marginBottom: 5, fontSize: '.76rem' }}>Summary</div>
              <div>Agendas: <strong style={{ color: 'hsl(var(--ink))' }}>{retrieveResult.agendas.length}</strong></div>
              <div>Total chunks: <strong style={{ color: 'hsl(var(--ink))' }}>{totalChunks}</strong></div>
              <div>Excluded: <strong style={{ color: 'hsl(var(--destructive))' }}>{disabledChunks.size}</strong></div>
            </div>
          )}
        </div>

        {/* ── RIGHT PANEL: Chunk list, filters, live preview & generated table ── */}
        <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', gap: '1.25rem', overflowY: 'auto' }}>

          {/* Empty initial state */}
          {!retrieveResult && !retrieving && (
            <div style={{
              display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
              padding: '6rem 2rem', gap: '1rem', borderRadius: 16,
              border: '2px dashed hsl(var(--border) / .4)', color: 'hsl(var(--pencil))',
            }}>
              <FlaskConical size={36} style={{ opacity: 0.2 }} />
              <div style={{ textAlign: 'center' }}>
                <p style={{ margin: 0, fontWeight: 600, fontSize: '.95rem', color: 'hsl(var(--ink))' }}>
                  Click "Retrieve Evidence" to run RAG search
                </p>
                <p style={{ margin: '4px 0 0', fontSize: '.82rem' }}>
                  The backend will parse the agenda and fetch matching evidence from the transcript, meeting docs, and global knowledge base.
                </p>
              </div>
            </div>
          )}

          {retrieving && (
            <div style={{
              display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
              padding: '5rem', gap: '1rem', borderRadius: 16,
              border: '1.5px solid hsl(var(--border) / .4)', background: 'hsl(var(--card))',
            }}>
              <Loader size={30} className="spin" style={{ color: 'hsl(280,75%,65%)' }} />
              <p style={{ margin: 0, fontSize: '.88rem', color: 'hsl(var(--pencil))' }}>
                Performing FAISS similarity search and relative similarity cap filtering...
              </p>
            </div>
          )}

          {/* Agenda & Retrieved Evidence list */}
          {retrieveResult && retrieveResult.agendas.length > 0 && (
            <div>
              <div style={{ fontSize: '.7rem', fontWeight: 700, color: 'hsl(var(--pencil))', textTransform: 'uppercase', letterSpacing: '.05em', marginBottom: '.65rem' }}>
                Evidence Chunks per Agenda Item ({retrieveResult.agendas.length})
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '.65rem' }}>
                {retrieveResult.agendas.map((agenda, idx) => {
                  const isOpen = expandedAgendas.has(idx)
                  const allChunks: AnyChunk[] = [...agenda.transcript_chunks, ...agenda.meeting_chunks, ...agenda.global_chunks]
                  const activeChunks = allChunks.filter(c => !disabledChunks.has(`${idx}_${c.chunk_id}`)).length

                  return (
                    <div key={idx} style={{
                      borderRadius: 12, border: '1.5px solid hsl(var(--border) / .5)',
                      background: 'hsl(var(--card))', overflow: 'hidden',
                    }}>
                      <div
                        style={{
                          display: 'flex', alignItems: 'center', gap: 9,
                          padding: '.75rem 1rem', cursor: 'pointer',
                          borderBottom: isOpen ? '1px solid hsl(var(--border) / .3)' : 'none',
                        }}
                        onClick={() => setExpandedAgendas(prev => {
                          const next = new Set(prev)
                          if (next.has(idx)) next.delete(idx); else next.add(idx)
                          return next
                        })}
                      >
                        {isOpen ? <ChevronDown size={14} style={{ flexShrink: 0, color: 'hsl(var(--pencil))' }} />
                          : <ChevronRight size={14} style={{ flexShrink: 0, color: 'hsl(var(--pencil))' }} />}
                        <span style={{ fontSize: '.7rem', fontWeight: 700, color: 'hsl(var(--pencil))', minWidth: 55 }}>
                          Agenda {idx + 1}
                        </span>
                        <span style={{ flex: 1, fontSize: '.87rem', fontWeight: 600, color: 'hsl(var(--ink))', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                          {agenda.topic}
                        </span>
                        {agenda.speaker && (
                          <span style={{ fontSize: '.73rem', color: 'hsl(var(--pencil))', whiteSpace: 'nowrap' }}>
                            <User size={10} style={{ display: 'inline', marginRight: 2 }} />{agenda.speaker}
                          </span>
                        )}
                        {agenda.is_procedural && (
                          <span style={{
                            fontSize: '.68rem', fontWeight: 600, color: 'hsl(45,90%,50%)',
                            background: 'hsl(45,90%,50% / .1)', border: '1px solid hsl(45,90%,50% / .3)',
                            padding: '2px 7px', borderRadius: 999, whiteSpace: 'nowrap',
                          }}>Procedural</span>
                        )}
                        <span style={{
                          fontSize: '.7rem', fontWeight: 600, whiteSpace: 'nowrap',
                          color: activeChunks > 0 ? 'hsl(140,70%,45%)' : 'hsl(var(--pencil))',
                          background: activeChunks > 0 ? 'hsl(140,70%,45% / .1)' : 'hsl(var(--muted) / .5)',
                          border: `1px solid ${activeChunks > 0 ? 'hsl(140,70%,45% / .3)' : 'hsl(var(--border) / .4)'}`,
                          padding: '2px 7px', borderRadius: 999,
                        }}>
                          {activeChunks}/{allChunks.length}
                        </span>
                      </div>

                      {isOpen && (
                        <div style={{ padding: '.65rem .9rem .9rem' }}>
                          <ChunkGroup label={`Transcript (${agenda.transcript_chunks.length})`} color="hsl(205,90%,55%)"
                            chunks={agenda.transcript_chunks} agendaIndex={idx} disabled={disabledChunks} onToggle={toggleChunk} />
                          <ChunkGroup label={`Meeting Context (${agenda.meeting_chunks.length})`} color="hsl(140,70%,50%)"
                            chunks={agenda.meeting_chunks} agendaIndex={idx} disabled={disabledChunks} onToggle={toggleChunk} />
                          <ChunkGroup label={`Global Context (${agenda.global_chunks.length})`} color="hsl(30,90%,55%)"
                            chunks={agenda.global_chunks} agendaIndex={idx} disabled={disabledChunks} onToggle={toggleChunk} />

                          {/* Evidence preview */}
                          <div style={{ marginTop: '.65rem' }}>
                            <div style={{ fontSize: '.68rem', fontWeight: 700, color: 'hsl(var(--pencil))', textTransform: 'uppercase', letterSpacing: '.05em', marginBottom: 5 }}>
                              Final Evidence Preview (Sent to LLM)
                            </div>
                            <pre style={{
                              margin: 0, fontSize: '.71rem', color: 'hsl(var(--ink))', fontFamily: 'Inter',
                              lineHeight: 1.5, whiteSpace: 'pre-wrap', wordBreak: 'break-word',
                              background: 'hsl(var(--paper) / .5)', padding: '.6rem', borderRadius: 9,
                              border: '1px solid hsl(var(--border) / .3)', maxHeight: 180, overflowY: 'auto',
                            }}>
                              {assembleEvidence(agenda, idx, disabledChunks) ||
                                <span style={{ color: 'hsl(var(--pencil))', fontStyle: 'italic' }}>
                                  All chunks excluded — no evidence will be sent
                                </span>}
                            </pre>
                          </div>
                        </div>
                      )}
                    </div>
                  )
                })}
              </div>
            </div>
          )}

          {/* Generated Raw MoM results table */}
          {momResult && (
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: '.65rem' }}>
                <div style={{ fontSize: '.7rem', fontWeight: 700, color: 'hsl(var(--pencil))', textTransform: 'uppercase', letterSpacing: '.05em' }}>Generated Raw Minutes of Meeting</div>
                <span style={{
                  fontSize: '.7rem', fontWeight: 600, color: 'hsl(140,70%,45%)',
                  background: 'hsl(140,70%,45% / .1)', border: '1px solid hsl(140,70%,45% / .3)',
                  padding: '2px 8px', borderRadius: 999,
                }}>
                  <CheckCircle size={10} style={{ display: 'inline', marginRight: 3 }} />Generated
                </span>
              </div>

              <div style={{ borderRadius: 12, border: '1.5px solid hsl(var(--border) / .5)', background: 'hsl(var(--card))', overflow: 'hidden' }}>
                {/* Table header */}
                <div style={{
                  display: 'grid', gridTemplateColumns: '40px 1fr 2fr 180px',
                  background: 'hsl(var(--muted) / .5)', borderBottom: '1px solid hsl(var(--border) / .3)',
                  padding: '.55rem .9rem', gap: '.9rem',
                }}>
                  {['#', 'Agenda', 'Discussion Point', 'Details'].map(h => (
                    <span key={h} style={{ fontSize: '.68rem', fontWeight: 700, color: 'hsl(var(--pencil))', textTransform: 'uppercase', letterSpacing: '.04em' }}>{h}</span>
                  ))}
                </div>

                {momResult.meeting.agendas.flatMap((agenda, ai) =>
                  agenda.discussion.length === 0
                    ? [(
                      <div key={`${ai}-empty`} style={{
                        display: 'grid', gridTemplateColumns: '40px 1fr 2fr 180px',
                        padding: '.6rem .9rem', gap: '.9rem', borderBottom: '1px solid hsl(var(--border) / .2)',
                      }}>
                        <span style={{ fontSize: '.76rem', color: 'hsl(var(--pencil))', fontFamily: 'JetBrains Mono' }}>{ai + 1}</span>
                        <span style={{ fontSize: '.76rem', color: 'hsl(var(--ink))', fontWeight: 600 }}>{agenda.agenda_topic}</span>
                        <span style={{ fontSize: '.74rem', color: 'hsl(var(--pencil))', fontStyle: 'italic' }}>No discussion extracted</span>
                        <span />
                      </div>
                    )]
                    : agenda.discussion.map((entry, ei) => {
                      const rowKey = ai * 1000 + ei
                      const isExpanded = expandedMomRows.has(rowKey)
                      const action = entry.action ?? {}
                      return (
                        <div key={rowKey} style={{ borderBottom: '1px solid hsl(var(--border) / .2)', cursor: 'pointer' }}
                          onClick={() => setExpandedMomRows(prev => {
                            const n = new Set(prev)
                            if (n.has(rowKey)) n.delete(rowKey); else n.add(rowKey)
                            return n
                          })}>
                          <div style={{ display: 'grid', gridTemplateColumns: '40px 1fr 2fr 180px', padding: '.6rem .9rem', gap: '.9rem' }}>
                            <span style={{ fontSize: '.76rem', color: 'hsl(var(--pencil))', fontFamily: 'JetBrains Mono' }}>
                              {ei === 0 ? ai + 1 : ''}
                            </span>
                            <span style={{ fontSize: '.76rem', color: 'hsl(var(--ink))', fontWeight: 600, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                              {ei === 0 ? agenda.agenda_topic : ''}
                            </span>
                            <span style={{ fontSize: '.76rem', color: 'hsl(var(--ink))', lineHeight: 1.5 }}>{entry.point}</span>
                            <div style={{ fontSize: '.7rem', color: 'hsl(var(--pencil))' }}>
                              {entry.speaker && <div><strong>Speaker:</strong> {entry.speaker}</div>}
                              {action.owner && <div><strong>Owner:</strong> {action.owner}</div>}
                              {action.deadline && <div><strong>Due:</strong> {action.deadline}</div>}
                            </div>
                          </div>
                          {isExpanded && (
                            <div style={{
                              padding: '.4rem .9rem .6rem 3rem',
                              background: 'hsl(var(--muted) / .3)',
                              borderTop: '1px solid hsl(var(--border) / .2)',
                              fontSize: '.72rem', color: 'hsl(var(--ink))',
                              display: 'flex', flexWrap: 'wrap', gap: '6px 20px',
                            }}>
                              {action.description && <span><strong>Action:</strong> {action.description}</span>}
                              {action.status && <span><strong>Status:</strong> {action.status}</span>}
                              {(entry.dates || []).map((d, di) => (
                                <span key={di}><strong>{d.purpose}:</strong> {d.value}</span>
                              ))}
                            </div>
                          )}
                        </div>
                      )
                    })
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
