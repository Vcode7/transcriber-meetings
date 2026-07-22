import { useState, useCallback, useEffect, useRef } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import {
  FlaskConical, ChevronDown, ChevronRight, Search, Loader,
  CheckCircle, X, Download, FileText, Clock, User,
  Sparkles, FileDown, ArrowLeft, Trash2, Upload, Users, Calendar, Brain,
  ExternalLink, ListChecks, ListOrdered, RefreshCw, Plus, Zap, Play, Pause,
  Pencil, MessageSquare, ArrowRightLeft, BarChart2, CheckSquare, Square, ChevronUp, Save,
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
  /** Source type: which evidence category this point came from */
  source_type?: 'Transcript' | 'Meeting Context' | 'Agenda Context' | 'Global Context'
  /** Original audio timestamps in seconds (only for Transcript entries) */
  timeline?: { start: number; end: number } | null
  /** Human-readable source reference: timestamp range or filename */
  source_reference?: string | null
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

// ── Review & Correction Types ─────────────────────────────────────────────────

interface SimilarityRow {
  agenda_idx: number
  disc_idx: number
  point: string
  similarities: number[]
}

interface EditingPoint {
  agendaIdx: number
  discIdx: number
}

interface MovingPoint {
  agendaIdx: number
  discIdx: number
}

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

  return blocks
    .filter(b => b.text.length > 0)
    .map(b => `${b.speaker}:\n${b.text}`)
    .join('\n\n')
}

function tryParseRawDiscussionJson(rawText: string): { success: true; discussion: DiscussionEntry[] } | { success: false; text: string } {
  let cleaned = rawText.trim()
  if (cleaned.startsWith('```')) {
    cleaned = cleaned.replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/, '').trim()
  }
  try {
    const parsed = JSON.parse(cleaned)
    let discussionArray: any[] | null = null

    if (Array.isArray(parsed)) {
      discussionArray = parsed
    } else if (parsed && typeof parsed === 'object') {
      if (Array.isArray(parsed.discussion)) {
        discussionArray = parsed.discussion
      } else if (parsed.point) {
        discussionArray = [parsed]
      }
    }

    if (discussionArray && Array.isArray(discussionArray) && discussionArray.length > 0) {
      const normalized: DiscussionEntry[] = discussionArray.map(item => {
        if (typeof item === 'string') {
          return {
            type: 'discussion',
            speaker: null,
            point: item,
            dates: [],
            action: { owner: null, description: null, deadline: null, status: null }
          }
        }
        return {
          type: item.type || 'discussion',
          speaker: item.speaker || null,
          point: item.point || item.statement || String(item),
          source_type: item.source_type || 'Transcript',
          timeline: item.timeline || null,
          source_reference: item.source_reference || null,
          dates: Array.isArray(item.dates) ? item.dates : [],
          action: item.action || { owner: null, description: null, deadline: null, status: null }
        }
      })
      return { success: true, discussion: normalized }
    }
  } catch {
    // Invalid JSON
  }
  return { success: false, text: rawText.trim() }
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
      // Use the raw chunk text directly (now includes [HH:MM:SS - HH:MM:SS] Speaker headers)
      sections.push('=== ADDITIONAL TRANSCRIPT EVIDENCE ===\n' + extraT.map(c => c.text).join('\n\n'))
    }
    const meet = retrievedResult.meeting_chunks.filter(c => !disabledChunks.has(`${agendaIndex}_${c.chunk_id}`))
    if (meet.length) sections.push('=== MEETING CONTEXT ===\n' + meet.map(c => c.filename ? `[${c.filename}]\n${c.text}` : c.text).join('\n\n'))
    const glob = retrievedResult.global_chunks.filter(c => !disabledChunks.has(`${agendaIndex}_${c.chunk_id}`))
    if (glob.length) sections.push('=== GLOBAL CONTEXT ===\n' + glob.map(c => c.filename ? `[${c.filename}]\n${c.text}` : c.text).join('\n\n'))
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
  const [retrieveByTimeline, setRetrieveByTimeline] = useState(false)
  const [retrievalMode, setRetrievalMode] = useState<'agenda_wise' | 'chunk_wise'>('agenda_wise')
  const [maxOverlapChunks, setMaxOverlapChunks] = useState(2)

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
  const [activeRightTab, setActiveRightTab] = useState<'agenda' | 'raw_mom' | 'final_mom'>('agenda')

  // ── Review & Correction state ───────────────────────────────────────────────
  const [editingPoint, setEditingPoint] = useState<EditingPoint | null>(null)
  const [editDraft, setEditDraft] = useState('')
  const [movingPoint, setMovingPoint] = useState<MovingPoint | null>(null)
  const [moveTarget, setMoveTarget] = useState<number | null>(null)
  // Per-point transcript toggle (key = "ai_di")
  const [shownTranscripts, setShownTranscripts] = useState<Set<string>>(new Set())
  // Recheck / similarity matrix
  const [recheckOpen, setRecheckOpen] = useState(false)
  const [recheckLoading, setRecheckLoading] = useState(false)
  const [recheckMatrix, setRecheckMatrix] = useState<SimilarityRow[]>([])
  const [agendaMoveSimilarityDiff, setAgendaMoveSimilarityDiff] = useState(0.35)
  const [dismissedSuggestions, setDismissedSuggestions] = useState<Set<string>>(new Set())
  const [applyingRecheckMoves, setApplyingRecheckMoves] = useState(false)
  // Non-JSON raw agenda edit state
  const [editingRawAgenda, setEditingRawAgenda] = useState<number | null>(null)
  const [rawAgendaDraft, setRawAgendaDraft] = useState('')
  const [savingRawAgenda, setSavingRawAgenda] = useState(false)

  // ── Audio player state ──────────────────────────────────────────────────────
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const [audioUrl, setAudioUrl] = useState<string | null>(null)
  const [audioPlaying, setAudioPlaying] = useState(false)
  const [audioCurrentTime, setAudioCurrentTime] = useState(0)

  /** Seek audio to startSecs and play */
  const seekAudio = useCallback((startSecs: number) => {
    if (!audioRef.current || !audioUrl) return
    audioRef.current.currentTime = startSecs
    audioRef.current.play().catch(() => { })
    setAudioPlaying(true)
  }, [audioUrl])

  const agendaContextInputRefs = useRef<Record<number, HTMLInputElement | null>>({})

  const loadInitialData = useCallback(async () => {
    if (!id) return
    setLoadingRec(true)
    try {
      const recRes = await api.get(`/history/${id}`)
      setRecording(recRes.data)
      // Fetch audio blob for playback with Auth headers
      try {
        const audioRes = await api.get(`/history/${id}/audio`, { responseType: 'blob' })
        setAudioUrl(URL.createObjectURL(audioRes.data))
      } catch { /* audio file missing or not ready */ }
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
        retrieve_by_timeline: retrieveByTimeline,
        retrieval_mode: retrievalMode,
        max_overlap_chunks: maxOverlapChunks,
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
        const tl = retrieveByTimeline ? getTimelineSegments(recording?.transcript || [], idx, editedAgendaItems.length, recording?.duration ?? 0, timelineStride) : []
        const evidence = assembleEvidenceForAgenda(idx, item, tl, agendaContextFiles[idx] || [], agendaEvidenceResult[idx] || null, disabledChunks, charLimit)
        return { topic: item.topic, speaker: item.speaker, evidence }
      })
      const res = await api.post(`/raw-mom/${id}/generate`, { agendas })
      setMomResult(res.data)
      setActiveRightTab('raw_mom')
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
      setActiveRightTab('final_mom')
      toast.success('Final MoM generated')
    } catch (e: any) { toast.error(e?.response?.data?.detail ?? 'Final MoM generation failed') }
    finally { setGeneratingFinalMom(false) }
  }

  // Save updated Raw MoM to database
  const saveMomResultToDb = useCallback(async (updatedMom: RawMomResult) => {
    if (!id) return
    try {
      await api.put(`/raw-mom/${id}`, { raw_mom: updatedMom })
    } catch (err) {
      toast.error(`Failed to sync changes to database: ${getApiErrorDetail(err)}`)
    }
  }, [id])

  // ── Feature 2: Edit discussion point ────────────────────────────────────────
  const updateDiscussionPoint = useCallback((agendaIdx: number, discIdx: number, newText: string) => {
    setMomResult(prev => {
      if (!prev) return prev
      const agendas = (prev.meeting?.agendas || []).map((ag, ai) => {
        if (ai !== agendaIdx) return ag
        return {
          ...ag,
          discussion: (ag.discussion as DiscussionEntry[]).map((d, di) =>
            di !== discIdx ? d : { ...d, point: newText }
          )
        }
      })
      const nextState: RawMomResult = { ...prev, meeting: { ...prev.meeting, agendas } }
      saveMomResultToDb(nextState)
      return nextState
    })
    setEditingPoint(null)
    setEditDraft('')
  }, [saveMomResultToDb])

  // ── Feature 3: Move point between agendas ───────────────────────────────────
  const moveDiscussionPoint = useCallback((fromAi: number, fromDi: number, toAi: number) => {
    setMomResult(prev => {
      if (!prev) return prev
      const agendas = (prev.meeting?.agendas || []).map((ag, ai) => ({
        ...ag,
        discussion: Array.isArray(ag.discussion) ? [...ag.discussion] : ag.discussion
      }))
      if (Array.isArray(agendas[fromAi].discussion) && Array.isArray(agendas[toAi].discussion)) {
        const [entry] = (agendas[fromAi].discussion as DiscussionEntry[]).splice(fromDi, 1)
        ;(agendas[toAi].discussion as DiscussionEntry[]).push(entry)
      }
      const nextState: RawMomResult = { ...prev, meeting: { ...prev.meeting, agendas } }
      saveMomResultToDb(nextState)
      return nextState
    })
    setMovingPoint(null)
    setMoveTarget(null)
  }, [saveMomResultToDb])

  // ── Edit & Fix Raw non-JSON Agenda Discussion ────────────────────────────────
  const handleSaveRawAgendaDiscussion = async (agendaIdx: number, draftText: string) => {
    if (!id || !momResult) return
    setSavingRawAgenda(true)
    try {
      const parseRes = tryParseRawDiscussionJson(draftText)
      let newDiscussionValue: DiscussionEntry[] | string

      if (parseRes.success) {
        newDiscussionValue = parseRes.discussion
        toast.success(`Valid JSON parsed with ${parseRes.discussion.length} points!`)
      } else {
        newDiscussionValue = parseRes.text
        toast.info('Saved as raw text (invalid JSON format)')
      }

      const updatedAgendas = (momResult.meeting?.agendas || []).map((ag, ai) => {
        if (ai !== agendaIdx) return ag
        return { ...ag, discussion: newDiscussionValue }
      })
      const updatedMomResult: RawMomResult = { ...momResult, meeting: { agendas: updatedAgendas } }

      setMomResult(updatedMomResult)
      await api.put(`/raw-mom/${id}`, { raw_mom: updatedMomResult })
      setEditingRawAgenda(null)
      setRawAgendaDraft('')
      toast.success('Raw MoM saved to database')
    } catch (e: any) {
      toast.error(e?.response?.data?.detail ?? 'Failed to save Raw MoM')
    } finally {
      setSavingRawAgenda(false)
    }
  }

  // ── Feature 1: Toggle transcript snippet ────────────────────────────────────
  const toggleTranscriptSnippet = useCallback((agendaIdx: number, discIdx: number) => {
    const key = `${agendaIdx}_${discIdx}`
    setShownTranscripts(prev => {
      const next = new Set(prev)
      if (next.has(key)) next.delete(key); else next.add(key)
      return next
    })
  }, [])

  // ── Feature 4: Recheck analysis ─────────────────────────────────────────────
  const runRecheckAnalysis = async () => {
    if (!id || !momResult) return
    const agendas = momResult.meeting?.agendas || []
    const agenda_topics = agendas.map(ag => ag.agenda_topic)
    const points: Array<{ agenda_idx: number; disc_idx: number; point: string }> = []
    agendas.forEach((ag, ai) => {
      if (Array.isArray(ag.discussion)) {
        ag.discussion.forEach((d, di) => {
          points.push({ agenda_idx: ai, disc_idx: di, point: d.point })
        })
      }
    })
    if (!points.length) { toast.error('No discussion points to analyze'); return }
    setRecheckLoading(true)
    setRecheckMatrix([])
    setDismissedSuggestions(new Set())
    try {
      const res = await api.post(`/raw-mom/${id}/validate-agenda-similarity`, {
        agenda_topics,
        points,
      })
      setRecheckMatrix(res.data.matrix || [])
      toast.success(`Similarity matrix computed for ${res.data.embeddings_computed} items`)
    } catch (e: any) {
      toast.error(e?.response?.data?.detail ?? 'Similarity analysis failed')
    } finally {
      setRecheckLoading(false)
    }
  }

  const applyRecheckMoves = () => {
    if (!momResult) return
    const agendas = momResult.meeting?.agendas || []
    const suggestions = recheckMatrix.filter(row => {
      const key = `${row.agenda_idx}_${row.disc_idx}`
      if (dismissedSuggestions.has(key)) return false
      const current = row.similarities[row.agenda_idx] ?? 0
      const best = Math.max(...row.similarities)
      const bestIdx = row.similarities.indexOf(best)
      return bestIdx !== row.agenda_idx && (best - current) >= agendaMoveSimilarityDiff
    })
    if (!suggestions.length) { toast.info('No suggestions to apply'); return }
    setApplyingRecheckMoves(true)

    // Apply in reverse disc_idx order to avoid index shifting within same agenda
    const sorted = [...suggestions].sort((a, b) =>
      a.agenda_idx !== b.agenda_idx ? 0 : b.disc_idx - a.disc_idx
    )
    let nextMom: RawMomResult | null = null
    setMomResult(prev => {
      if (!prev) return prev
      const ags = (prev.meeting?.agendas || []).map((ag, ai) => ({
        ...ag,
        discussion: Array.isArray(ag.discussion) ? [...ag.discussion] : ag.discussion
      }))
      sorted.forEach(row => {
        const toAi = row.similarities.indexOf(Math.max(...row.similarities))
        if (toAi === row.agenda_idx) return
        if (!ags[row.agenda_idx] || !ags[toAi]) return
        if (!Array.isArray(ags[row.agenda_idx].discussion) || !Array.isArray(ags[toAi].discussion)) return
        const srcDisc = ags[row.agenda_idx].discussion as DiscussionEntry[]
        const dstDisc = ags[toAi].discussion as DiscussionEntry[]
        const idx = srcDisc.findIndex((d, di) => di === row.disc_idx)
        if (idx === -1) return
        const [entry] = srcDisc.splice(idx, 1)
        dstDisc.push(entry)
      })
      nextMom = { ...prev, meeting: { agendas: ags } }
      return nextMom
    })
    if (nextMom) saveMomResultToDb(nextMom)
    setRecheckMatrix([])
    setDismissedSuggestions(new Set())
    setApplyingRecheckMoves(false)
    toast.success(`Moved ${suggestions.length} discussion point(s)`)
  }

  // Helper to get transcript segments matching a point's timeline
  const getPointTranscriptSegments = useCallback((start: number, end: number): TranscriptSegment[] => {
    if (!recording?.transcript) return []
    return recording.transcript.filter(seg => seg.start <= end && seg.end >= start)
  }, [recording])

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
  const selTimeline = selIdx !== null ? (retrieveByTimeline ? getTimelineSegments(recording?.transcript || [], selIdx, editedAgendaItems.length, recording?.duration ?? 0, timelineStride) : []) : []
  const selCtxFiles = selIdx !== null ? (agendaContextFiles[selIdx] || []) : []
  const selEvidence = selIdx !== null ? (agendaEvidenceResult[selIdx] || null) : null

  return (
    <div className="page-scroll-root" style={{ display: 'flex', flexDirection: 'column', height: '100%', minHeight: 0, overflow: 'hidden' }}>
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
      <div style={{ flex: 1, display: 'flex', gap: '1.25rem', padding: '1.25rem 1.75rem 1.5rem', minHeight: 0, height: 0, overflow: 'hidden' }}>

        {/* ── LEFT PANEL ── */}
        <div style={{ width: 268, flexShrink: 0, minHeight: 0, height: '100%', display: 'flex', flexDirection: 'column', gap: '.6rem', overflowY: 'auto', paddingRight: 4 }}>

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

            {/* Retrieval Mode Toggle */}
            <div>
              <div style={{ fontSize: '.66rem', fontWeight: 700, color: 'hsl(280,75%,65%)', textTransform: 'uppercase', letterSpacing: '.04em', marginBottom: '.3rem' }}>Retrieval Mode</div>
              <div style={{ display: 'flex', borderRadius: 8, overflow: 'hidden', border: '1.5px solid hsl(280,75%,60%/.35)', background: 'hsl(var(--muted)/.3)' }}>
                {(['agenda_wise', 'chunk_wise'] as const).map(mode => {
                  const isActive = retrievalMode === mode
                  const label = mode === 'agenda_wise' ? 'Agenda-wise' : 'Chunk-wise'
                  return (
                    <button
                      key={mode}
                      onClick={() => setRetrievalMode(mode)}
                      title={mode === 'agenda_wise'
                        ? 'Each agenda retrieves its top-K transcript chunks by semantic similarity'
                        : 'Each transcript chunk is assigned to agendas within relative threshold of its best match'}
                      style={{
                        flex: 1, padding: '.3rem .2rem', border: 'none', cursor: 'pointer',
                        fontSize: '.68rem', fontWeight: isActive ? 700 : 500,
                        background: isActive ? 'hsl(280,75%,60%)' : 'transparent',
                        color: isActive ? '#fff' : 'hsl(var(--pencil))',
                        transition: 'all .15s', fontFamily: 'Inter',
                      }}
                    >{label}</button>
                  )
                })}
              </div>
              <div style={{ fontSize: '.63rem', color: 'hsl(var(--pencil))', marginTop: '.25rem', fontStyle: 'italic', lineHeight: 1.35 }}>
                {retrievalMode === 'agenda_wise' ? 'Agendas search for chunks.' : 'Chunks are assigned to agendas.'}
              </div>
              {retrievalMode === 'chunk_wise' && (
                <div style={{ marginTop: '.4rem', paddingTop: '.35rem', borderTop: '1px dashed hsl(280,75%,60%/.3)' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 1 }}>
                    <label style={{ fontSize: '.71rem', color: 'hsl(var(--ink))', fontFamily: 'Inter' }}>Max Chunk Overlap</label>
                    <span style={{ fontSize: '.67rem', fontWeight: 700, color: 'hsl(280,75%,65%)', fontFamily: 'JetBrains Mono' }}>{maxOverlapChunks}</span>
                  </div>
                  <input
                    type="range"
                    min={1}
                    max={10}
                    value={maxOverlapChunks}
                    onChange={e => setMaxOverlapChunks(Number(e.target.value))}
                    style={{ width: '100%', accentColor: 'hsl(280,75%,60%)' }}
                  />
                  <div style={{ fontSize: '.61rem', color: 'hsl(var(--pencil))', fontStyle: 'italic', marginTop: 1 }}>
                    Limit chunk to at most {maxOverlapChunks} agenda{maxOverlapChunks > 1 ? 's' : ''}
                  </div>
                </div>
              )}
            </div>

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
            <label style={{ display: 'flex', alignItems: 'center', gap: 5, cursor: 'pointer', fontSize: '.71rem', color: 'hsl(var(--ink))', fontFamily: 'Inter' }}>
              <input type="checkbox" checked={retrieveByTimeline} onChange={e => setRetrieveByTimeline(e.target.checked)} />Retrieve Transcript by Timeline
            </label>
            <div style={{ height: '1px', background: 'hsl(var(--border)/.2)' }} />
            <div style={{ fontSize: '.66rem', fontWeight: 700, color: 'hsl(280,75%,65%)', textTransform: 'uppercase', letterSpacing: '.04em' }}>Agenda Validation</div>
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 1 }}>
                <label style={{ fontSize: '.71rem', color: 'hsl(var(--ink))', fontFamily: 'Inter' }}>Move Similarity Threshold</label>
                <span style={{ fontSize: '.67rem', fontWeight: 700, color: 'hsl(280,75%,65%)', fontFamily: 'JetBrains Mono' }}>{agendaMoveSimilarityDiff.toFixed(2)}</span>
              </div>
              <input type="range" min={0.1} max={0.9} step={0.05} value={agendaMoveSimilarityDiff} onChange={e => setAgendaMoveSimilarityDiff(Number(e.target.value))} style={{ width: '100%', accentColor: 'hsl(280,75%,60%)' }} />
              <div style={{ fontSize: '.62rem', color: 'hsl(var(--pencil))', fontStyle: 'italic' }}>Min difference to suggest a move</div>
            </div>
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
        <div style={{ flex: 1, minWidth: 0, minHeight: 0, height: '100%', display: 'flex', flexDirection: 'column', gap: '.75rem', overflow: 'hidden' }}>

          {/* ── TAB NAVIGATION HEADER ── */}
          <div style={{ display: 'flex', gap: '6px', padding: '4px', borderRadius: '12px', background: 'hsl(var(--muted)/.4)', border: '1.5px solid hsl(var(--border)/.4)', flexShrink: 0 }}>
            {/* Agenda Tab */}
            <button
              onClick={() => setActiveRightTab('agenda')}
              style={{
                flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px',
                padding: '.45rem .75rem', borderRadius: '8px', border: 'none', cursor: 'pointer',
                fontSize: '.78rem', fontWeight: activeRightTab === 'agenda' ? 700 : 500,
                background: activeRightTab === 'agenda' ? 'hsl(var(--card))' : 'transparent',
                color: activeRightTab === 'agenda' ? 'hsl(280,75%,65%)' : 'hsl(var(--pencil))',
                boxShadow: activeRightTab === 'agenda' ? '0 2px 6px rgba(0,0,0,0.06)' : 'none',
                transition: 'all .15s ease', fontFamily: 'Inter',
              }}
            >
              <ListOrdered size={13} />
              <span>Agenda</span>
              {editedAgendaItems.length > 0 && (
                <span style={{ fontSize: '.64rem', padding: '1px 6px', borderRadius: 999, background: activeRightTab === 'agenda' ? 'hsl(280,75%,60%/.15)' : 'hsl(var(--border)/.4)', color: activeRightTab === 'agenda' ? 'hsl(280,75%,65%)' : 'hsl(var(--pencil))', fontWeight: 700 }}>
                  {editedAgendaItems.length}
                </span>
              )}
            </button>

            {/* Raw MoM Tab */}
            <button
              disabled={!momResult}
              onClick={() => momResult && setActiveRightTab('raw_mom')}
              title={momResult ? 'View Raw MoM output' : 'Generate Raw MoM first to view'}
              style={{
                flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px',
                padding: '.45rem .75rem', borderRadius: '8px', border: 'none',
                cursor: momResult ? 'pointer' : 'not-allowed',
                opacity: momResult ? 1 : 0.45,
                fontSize: '.78rem', fontWeight: activeRightTab === 'raw_mom' ? 700 : 500,
                background: activeRightTab === 'raw_mom' ? 'hsl(var(--card))' : 'transparent',
                color: activeRightTab === 'raw_mom' ? 'hsl(140,70%,45%)' : 'hsl(var(--pencil))',
                boxShadow: activeRightTab === 'raw_mom' ? '0 2px 6px rgba(0,0,0,0.06)' : 'none',
                transition: 'all .15s ease', fontFamily: 'Inter',
              }}
            >
              <Sparkles size={13} />
              <span>Raw MoM</span>
              {momResult && (
                <span style={{ fontSize: '.64rem', padding: '1px 6px', borderRadius: 999, background: 'hsl(140,70%,45%/.15)', color: 'hsl(140,70%,45%)', fontWeight: 700 }}>
                  Ready
                </span>
              )}
            </button>

            {/* Final MoM Tab */}
            <button
              disabled={!finalMomResult}
              onClick={() => finalMomResult && setActiveRightTab('final_mom')}
              title={finalMomResult ? 'View Final MoM output' : 'Generate Final MoM first to view'}
              style={{
                flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px',
                padding: '.45rem .75rem', borderRadius: '8px', border: 'none',
                cursor: finalMomResult ? 'pointer' : 'not-allowed',
                opacity: finalMomResult ? 1 : 0.45,
                fontSize: '.78rem', fontWeight: activeRightTab === 'final_mom' ? 700 : 500,
                background: activeRightTab === 'final_mom' ? 'hsl(var(--card))' : 'transparent',
                color: activeRightTab === 'final_mom' ? 'hsl(220,80%,60%)' : 'hsl(var(--pencil))',
                boxShadow: activeRightTab === 'final_mom' ? '0 2px 6px rgba(0,0,0,0.06)' : 'none',
                transition: 'all .15s ease', fontFamily: 'Inter',
              }}
            >
              <Zap size={13} />
              <span>Final MoM</span>
              {finalMomResult && (
                <span style={{ fontSize: '.64rem', padding: '1px 6px', borderRadius: 999, background: 'hsl(220,80%,60%/.15)', color: 'hsl(220,80%,60%)', fontWeight: 700 }}>
                  Ready
                </span>
              )}
            </button>
          </div>

          {/* ── SCROLLABLE TAB CONTENT AREA ── */}
          <div style={{ flex: 1, minHeight: 0, overflowY: 'auto', paddingRight: 6, display: 'flex', flexDirection: 'column', gap: '1rem' }}>

            {/* ── TAB 1: AGENDA ── */}
            <div style={{ display: activeRightTab === 'agenda' ? 'flex' : 'none', flexDirection: 'column', gap: '1rem', minHeight: 0, flex: 1 }}>
              {/* Empty state */}
              {!agendaCreated && (
                <div style={{
                  display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '6rem 2rem', gap: '1rem', minHeight: 0,
                  flex: 1, borderRadius: 16, border: '2px dashed hsl(var(--border)/.4)', color: 'hsl(var(--pencil))'
                }}>
                  <FlaskConical size={36} style={{ opacity: 0.2 }} />
                  <div style={{ textAlign: 'center' }}>
                    <p style={{ margin: 0, fontWeight: 600, fontSize: '.95rem', color: 'hsl(var(--ink))' }}>Click "Create Agenda" to begin</p>
                    <p style={{ margin: '4px 0 0', fontSize: '.82rem' }}>The agenda will be parsed or generated. Then select an agenda item to view its timeline transcript and retrieve targeted evidence.</p>
                  </div>
                </div>
              )}

              {/* Select prompt */}
              {agendaCreated && selIdx === null && (
                <div style={{
                  display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '5rem 2rem', gap: '1rem', minHeight: 0,
                  flex: 1, borderRadius: 16, border: '1.5px dashed hsl(var(--border)/.4)', color: 'hsl(var(--pencil))'
                }}>
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
                        <SectionHeader icon={<FileText size={13} />} label="Agenda Context" count={selCtxFiles.length} color="hsla(45, 100%, 20%, 1.00)"  />
                        <input ref={el => { agendaContextInputRefs.current[selIdx] = el }} type="file" multiple accept=".pdf,.docx,.pptx,.txt,.md,.png,.jpg,.jpeg,.webp,.xlsx,.xls,.csv" style={{ display: 'none' }} onChange={e => handleUploadAgendaContext(selIdx, e.target.files)} />
                        <button onClick={() => agendaContextInputRefs.current[selIdx]?.click()} disabled={agendaContextUploading[selIdx]} style={{ display: 'flex', alignItems: 'center', gap: 5, padding: '.35rem .7rem', borderRadius: 8, cursor: 'pointer', fontFamily: 'Inter', fontSize: '.73rem', fontWeight: 600, border: '1.5px dashed hsl(45,90%,50%)', background: 'hsl(45,90%,50%,.04)', color: 'hsla(45, 100%, 20%, 1.00)', marginBottom: '.4rem' }}>
                          {agendaContextUploading[selIdx] ? <><Loader size={10} className="spin" /> Extracting...</> : <><Plus size={10} /> Upload Agenda Context File</>}
                        </button>
                        {selCtxFiles.length > 0 && (
                          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                            {selCtxFiles.map((f, fi) => (
                              <div key={fi} style={{ borderRadius: 8, border: '1px solid hsl(45,90%,50%)', background: 'hsl(45,90%,50%,.04)', padding: '.45rem .7rem' }}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 3 }}>
                                  <FileText size={10} style={{ color: 'hsla(46, 100%, 12%, 1.00)', flexShrink: 0 }} />
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
                        <SectionHeader
                          icon={<Clock size={13} />}
                          label={retrieveByTimeline ? "Timeline Transcript" : "Timeline Transcript (Disabled)"}
                          count={retrieveByTimeline ? selTimeline.length : undefined}
                          color={retrieveByTimeline ? "hsl(205,90%,55%)" : "hsl(var(--pencil))"}
                        />
                        {!retrieveByTimeline ? (
                          <p style={{ margin: '4px 0', fontSize: '.75rem', color: 'hsl(var(--pencil))', fontStyle: 'italic' }}>
                            Timeline-based transcript retrieval is Disabled. Use the toggle in the settings panel to enable.
                          </p>
                        ) : selTimeline.length === 0 ? (
                          <p style={{ margin: '4px 0', fontSize: '.75rem', color: 'hsl(var(--pencil))', fontStyle: 'italic' }}>
                            No transcript segments in time window ({!recording?.transcript?.length ? 'no transcript loaded' : `stride ${timelineStride}s around slot ${selIdx + 1}/${editedAgendaItems.length}`})
                          </p>
                        ) : (
                          <div style={{ display: 'flex', flexDirection: 'column', gap: 4, maxHeight: "200px", overflowY: "auto", backgroundColor: "hsl(var(--muted)/.5)" }}>
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
            </div>

            {/* ── TAB 2: RAW MOM ── */}
            <div style={{ display: activeRightTab === 'raw_mom' ? 'flex' : 'none', flexDirection: 'column', gap: '1rem', flex: 1 }}>
              {momResult && (
                <div style={{ borderRadius: 14, border: '1.5px solid hsl(140,70%,45%/.3)', background: 'hsl(var(--card))', overflow: 'hidden' }}>
                  {/* Hidden audio element */}
                  {audioUrl && (
                    <audio
                      ref={audioRef}
                      src={audioUrl}
                      onPlay={() => setAudioPlaying(true)}
                      onPause={() => setAudioPlaying(false)}
                      onEnded={() => setAudioPlaying(false)}
                      onTimeUpdate={() => setAudioCurrentTime(audioRef.current?.currentTime ?? 0)}
                      style={{ display: 'none' }}
                    />
                  )}

                  <div style={{ background: 'linear-gradient(135deg,hsl(140,70%,45%/.1),hsl(160,70%,45%/.06))', borderBottom: '1px solid hsl(140,70%,45%/.2)', padding: '.7rem 1rem', display: 'flex', alignItems: 'center', gap: 8 }}>
                    <CheckCircle size={14} style={{ color: 'hsl(140,70%,45%)' }} />
                    <span style={{ fontSize: '.78rem', fontWeight: 700, color: 'hsl(var(--ink))' }}>Raw MoM Generated</span>
                    <span style={{ fontSize: '.69rem', color: 'hsl(var(--pencil))', marginLeft: 'auto' }}>
                      {(momResult.meeting?.agendas || momResult.agendas || []).length} agendas
                    </span>
                  </div>

                  {/* Mini audio player bar */}
                  {audioUrl && (
                    <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '.45rem 1rem', background: 'hsl(205,90%,55%/.06)', borderBottom: '1px solid hsl(205,90%,55%/.15)' }}>
                      <button
                        onClick={() => {
                          if (!audioRef.current) return
                          if (audioPlaying) { audioRef.current.pause(); setAudioPlaying(false) }
                          else { audioRef.current.play().catch(() => { }); setAudioPlaying(true) }
                        }}
                        style={{ display: 'flex', alignItems: 'center', gap: 4, padding: '3px 9px', borderRadius: 7, border: '1px solid hsl(205,90%,55%/.3)', background: 'hsl(205,90%,55%/.1)', color: 'hsl(205,90%,60%)', cursor: 'pointer', fontSize: '.72rem', fontWeight: 600 }}
                      >
                        {audioPlaying ? <Pause size={11} /> : <Play size={11} />}
                        {audioPlaying ? 'Pause' : 'Play'}
                      </button>
                      <span style={{ fontSize: '.7rem', fontFamily: 'JetBrains Mono, monospace', color: 'hsl(var(--pencil))' }}>
                        {fmtTime(audioCurrentTime)}
                      </span>
                      <span style={{ fontSize: '.67rem', color: 'hsl(var(--pencil))', opacity: 0.6 }}>Click ▶ on any transcript point to jump to that moment</span>
                    </div>
                  )}

                  <div style={{ padding: '.7rem 1rem', display: 'flex', flexDirection: 'column', gap: '.55rem' }}>
                    {(momResult.meeting?.agendas || momResult.agendas || []).map((ag, ai) => (
                      <div key={ai} style={{ borderRadius: 9, border: '1px solid hsl(var(--border)/.4)', overflow: 'hidden' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '.5rem .85rem', background: 'hsl(var(--muted)/.3)', borderBottom: '1px solid hsl(var(--border)/.25)' }}>
                          <span style={{ fontSize: '.71rem', fontWeight: 700, color: 'hsl(var(--pencil))' }}>Agenda {ai + 1}</span>
                          <span style={{ flex: 1, fontSize: '.81rem', fontWeight: 600, color: 'hsl(var(--ink))' }}>{ag.agenda_topic}</span>
                          {ag.agenda_speaker && <span style={{ fontSize: '.7rem', color: 'hsl(var(--pencil))' }}><User size={9} style={{ display: 'inline', marginRight: 2 }} />{ag.agenda_speaker}</span>}
                          <span style={{ fontSize: '.69rem', color: 'hsl(140,70%,45%)', background: 'hsl(140,70%,45%/.1)', border: '1px solid hsl(140,70%,45%/.3)', padding: '2px 7px', borderRadius: 999 }}>
                            {Array.isArray(ag.discussion) ? ag.discussion.length : 1} pts
                          </span>
                        </div>
                        {Array.isArray(ag.discussion) ? (
                          ag.discussion.map((disc, di) => {
                            const srcType = disc.source_type || 'Transcript'
                            const isTranscript = srcType === 'Transcript'
                            const isReference = disc.type === 'reference' || srcType === 'Meeting Context' || srcType === 'Agenda Context' || srcType === 'Global Context'
                            const hasTimeline = isTranscript && disc.timeline?.start != null
                            const pointKey = `${ai}_${di}`
                            const isEditing = editingPoint?.agendaIdx === ai && editingPoint?.discIdx === di
                            const isMoving = movingPoint?.agendaIdx === ai && movingPoint?.discIdx === di
                            const transcriptVisible = shownTranscripts.has(pointKey)
                            const tSegs = hasTimeline && disc.timeline ? getPointTranscriptSegments(disc.timeline.start, disc.timeline.end) : []

                            // Source badge styling
                            const srcBadgeStyle: Record<string, { bg: string; border: string; color: string }> = {
                              'Transcript': { bg: 'hsl(205,90%,55%/.12)', border: 'hsl(205,90%,55%/.35)', color: 'hsl(205,90%,60%)' },
                              'Meeting Context': { bg: 'hsl(140,70%,45%/.1)', border: 'hsl(140,70%,45%/.3)', color: 'hsl(140,70%,50%)' },
                              'Agenda Context': { bg: 'hsl(45,90%,50%/.1)', border: 'hsl(45,90%,50%/.3)', color: 'hsl(45,90%,45%)' },
                              'Global Context': { bg: 'hsl(30,90%,55%/.1)', border: 'hsl(30,90%,55%/.3)', color: 'hsl(30,90%,55%)' },
                            }
                            const badge = srcBadgeStyle[srcType] || srcBadgeStyle['Transcript']

                            return (
                              <div key={di} style={{ padding: '.55rem .85rem', borderBottom: di < ag.discussion.length - 1 ? '1px solid hsl(var(--border)/.15)' : 'none' }}>
                                {/* Top row: speaker + source badge + timeline + play + transcript + edit + move */}
                                <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 5, marginBottom: 4 }}>
                                  {disc.speaker && <span style={{ fontSize: '.72rem', color: 'hsl(var(--pencil))', fontWeight: 600 }}>{disc.speaker}</span>}

                                  {/* Source type badge */}
                                  <span style={{
                                    fontSize: '.63rem', fontWeight: 700, padding: '1px 6px', borderRadius: 999,
                                    background: badge.bg, border: `1px solid ${badge.border}`, color: badge.color,
                                  }}>
                                    {isReference && srcType !== 'Transcript' ? 'For Info' : srcType}
                                  </span>

                                  {/* Timeline badge + play button */}
                                  {hasTimeline && disc.timeline && (
                                    <>
                                      <span style={{ fontSize: '.63rem', color: 'hsl(205,90%,60%)', fontFamily: 'JetBrains Mono, monospace', display: 'flex', alignItems: 'center', gap: 3 }}>
                                        <Clock size={9} />
                                        {fmtTime(disc.timeline.start)} – {fmtTime(disc.timeline.end)}
                                      </span>
                                      {audioUrl && (
                                        <button
                                          onClick={() => seekAudio(disc.timeline!.start)}
                                          title={`Play from ${fmtTime(disc.timeline.start)}`}
                                          style={{
                                            display: 'inline-flex', alignItems: 'center', gap: 3,
                                            padding: '1px 6px', borderRadius: 999, cursor: 'pointer',
                                            background: 'hsl(205,90%,55%/.15)', border: '1px solid hsl(205,90%,55%/.4)',
                                            color: 'hsl(205,90%,60%)', fontSize: '.63rem', fontWeight: 700,
                                          }}
                                        >
                                          <Play size={8} /> Play
                                        </button>
                                      )}
                                      {/* Feature 1: Show Transcript button */}
                                      <button
                                        onClick={() => toggleTranscriptSnippet(ai, di)}
                                        title={transcriptVisible ? 'Hide transcript' : 'Show transcript'}
                                        style={{
                                          display: 'inline-flex', alignItems: 'center', gap: 3,
                                          padding: '1px 6px', borderRadius: 999, cursor: 'pointer',
                                          background: transcriptVisible ? 'hsl(280,75%,60%/.15)' : 'transparent',
                                          border: `1px solid ${transcriptVisible ? 'hsl(280,75%,60%/.5)' : 'hsl(var(--border)/.4)'}`,
                                          color: transcriptVisible ? 'hsl(280,75%,65%)' : 'hsl(var(--pencil))',
                                          fontSize: '.63rem', fontWeight: 600,
                                        }}
                                      >
                                        <MessageSquare size={8} />
                                        {transcriptVisible ? 'Hide' : 'Transcript'}
                                      </button>
                                    </>
                                  )}

                                  {/* Source reference (non-transcript) */}
                                  {!isTranscript && disc.source_reference && (
                                    <span style={{ fontSize: '.63rem', color: 'hsl(var(--pencil))', fontStyle: 'italic' }}>
                                      {disc.source_reference}
                                    </span>
                                  )}

                                  {/* Feature 2: Edit button */}
                                  {!isEditing && !isMoving && (
                                    <button
                                      onClick={() => { setEditingPoint({ agendaIdx: ai, discIdx: di }); setEditDraft(disc.point); setMovingPoint(null) }}
                                      title="Edit this point"
                                      style={{
                                        marginLeft: 'auto', display: 'inline-flex', alignItems: 'center', gap: 3,
                                        padding: '1px 6px', borderRadius: 999, cursor: 'pointer',
                                        background: 'transparent', border: '1px solid hsl(var(--border)/.4)',
                                        color: 'hsl(var(--pencil))', fontSize: '.63rem',
                                      }}
                                    >
                                      <Pencil size={8} />
                                    </button>
                                  )}

                                  {/* Feature 3: Move button */}
                                  {!isEditing && !isMoving && (momResult?.meeting?.agendas?.length ?? 0) > 1 && (
                                    <button
                                      onClick={() => { setMovingPoint({ agendaIdx: ai, discIdx: di }); setMoveTarget(null); setEditingPoint(null) }}
                                      title="Move to another agenda"
                                      style={{
                                        display: 'inline-flex', alignItems: 'center', gap: 3,
                                        padding: '1px 6px', borderRadius: 999, cursor: 'pointer',
                                        background: 'transparent', border: '1px solid hsl(var(--border)/.4)',
                                        color: 'hsl(var(--pencil))', fontSize: '.63rem',
                                      }}
                                    >
                                      <ArrowRightLeft size={8} />
                                    </button>
                                  )}
                                </div>

                                {/* Feature 3: Move selector inline */}
                                {isMoving && (
                                  <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6, padding: '.35rem .5rem', borderRadius: 8, background: 'hsl(280,75%,60%/.06)', border: '1px solid hsl(280,75%,60%/.2)' }}>
                                    <ArrowRightLeft size={11} style={{ color: 'hsl(280,75%,65%)', flexShrink: 0 }} />
                                    <select
                                      value={moveTarget ?? ''}
                                      onChange={e => setMoveTarget(Number(e.target.value))}
                                      style={{ flex: 1, fontSize: '.72rem', padding: '2px 4px', borderRadius: 6, border: '1px solid hsl(var(--border)/.4)', background: 'hsl(var(--card))', color: 'hsl(var(--ink))', fontFamily: 'Inter' }}
                                    >
                                      <option value="">Move to agenda…</option>
                                      {(momResult?.meeting?.agendas || []).map((a, idx) =>
                                        idx !== ai ? <option key={idx} value={idx}>{idx + 1}. {a.agenda_topic}</option> : null
                                      )}
                                    </select>
                                    <button
                                      disabled={moveTarget === null}
                                      onClick={() => moveTarget !== null && moveDiscussionPoint(ai, di, moveTarget)}
                                      style={{ display: 'inline-flex', alignItems: 'center', gap: 3, padding: '2px 8px', borderRadius: 6, cursor: moveTarget !== null ? 'pointer' : 'not-allowed', background: 'hsl(140,70%,45%)', color: '#fff', border: 'none', fontSize: '.68rem', fontWeight: 700, opacity: moveTarget !== null ? 1 : 0.45 }}
                                    >
                                      Confirm
                                    </button>
                                    <button
                                      onClick={() => { setMovingPoint(null); setMoveTarget(null) }}
                                      style={{ display: 'inline-flex', padding: '2px 6px', borderRadius: 6, cursor: 'pointer', background: 'none', border: '1px solid hsl(var(--border)/.4)', color: 'hsl(var(--pencil))', fontSize: '.68rem' }}
                                    >
                                      Cancel
                                    </button>
                                  </div>
                                )}

                                {/* Feature 2: Inline edit textarea */}
                                {isEditing ? (
                                  <div style={{ marginBottom: 4 }}>
                                    <textarea
                                      value={editDraft}
                                      onChange={e => setEditDraft(e.target.value)}
                                      rows={3}
                                      style={{ width: '100%', fontSize: '.79rem', color: 'hsl(var(--ink))', fontFamily: 'Inter', lineHeight: 1.55, padding: '.4rem .55rem', borderRadius: 7, border: '1.5px solid hsl(280,75%,60%/.4)', background: 'hsl(var(--paper))', resize: 'vertical', boxSizing: 'border-box' }}
                                      autoFocus
                                    />
                                    <div style={{ display: 'flex', gap: 6, marginTop: 4 }}>
                                      <button
                                        onClick={() => updateDiscussionPoint(ai, di, editDraft)}
                                        style={{ display: 'inline-flex', alignItems: 'center', gap: 4, padding: '3px 10px', borderRadius: 6, cursor: 'pointer', background: 'hsl(140,70%,45%)', color: '#fff', border: 'none', fontSize: '.71rem', fontWeight: 700 }}
                                      >
                                        <Save size={10} /> Save
                                      </button>
                                      <button
                                        onClick={() => { setEditingPoint(null); setEditDraft('') }}
                                        style={{ display: 'inline-flex', alignItems: 'center', gap: 4, padding: '3px 10px', borderRadius: 6, cursor: 'pointer', background: 'none', border: '1px solid hsl(var(--border)/.4)', color: 'hsl(var(--pencil))', fontSize: '.71rem' }}
                                      >
                                        <X size={10} /> Cancel
                                      </button>
                                    </div>
                                  </div>
                                ) : (
                                  /* Point text (normal view) */
                                  <p style={{ margin: 0, fontSize: '.79rem', color: isReference && !isTranscript ? 'hsl(var(--pencil))' : 'hsl(var(--ink))', fontFamily: 'Inter', lineHeight: 1.55, fontStyle: isReference && !isTranscript ? 'italic' : 'normal' }}>
                                    {disc.point}
                                  </p>
                                )}

                                {/* Feature 1: Inline transcript snippet */}
                                {transcriptVisible && (
                                  <div style={{ marginTop: 6, borderRadius: 8, border: '1px solid hsl(280,75%,60%/.2)', background: 'hsl(280,75%,60%/.04)', padding: '.4rem .6rem' }}>
                                    <div style={{ fontSize: '.64rem', fontWeight: 700, color: 'hsl(280,75%,65%)', textTransform: 'uppercase', letterSpacing: '.04em', marginBottom: 4, display: 'flex', alignItems: 'center', gap: 4 }}>
                                      <MessageSquare size={9} /> Source Transcript
                                    </div>
                                    {tSegs.length === 0 ? (
                                      <p style={{ margin: 0, fontSize: '.72rem', color: 'hsl(var(--pencil))', fontStyle: 'italic' }}>No transcript segments found for this timeline window.</p>
                                    ) : (
                                      <div style={{ display: 'flex', flexDirection: 'column', gap: 4, maxHeight: 180, overflowY: 'auto' }}>
                                        {mergeConsecutiveSegments(tSegs).map((block, bi) => (
                                          <div key={bi} style={{ display: 'flex', gap: 8 }}>
                                            <div style={{ flexShrink: 0, paddingTop: 1 }}>
                                              <span style={{ fontSize: '.65rem', fontWeight: 700, color: 'hsl(205,90%,60%)', background: 'hsl(205,90%,55%/.1)', border: '1px solid hsl(205,90%,55%/.25)', padding: '1px 5px', borderRadius: 4, whiteSpace: 'nowrap', fontFamily: 'Inter' }}>
                                                <User size={8} style={{ display: 'inline', marginRight: 2 }} />{block.speaker}
                                              </span>
                                            </div>
                                            <p style={{ margin: 0, fontSize: '.73rem', color: 'hsl(var(--ink))', fontFamily: 'Inter', lineHeight: 1.5 }}>{block.text}</p>
                                          </div>
                                        ))}
                                      </div>
                                    )}
                                  </div>
                                )}

                                {/* Action item */}
                                {disc.action?.description && <p style={{ margin: '3px 0 0', fontSize: '.72rem', color: 'hsl(30,90%,55%)', fontFamily: 'Inter' }}>&#8594; {disc.action.description}{disc.action.owner ? ` (${disc.action.owner})` : ''}{disc.action.deadline ? ` | ${disc.action.deadline}` : ''}</p>}
                              </div>
                            )
                          })
                        ) : (
                          editingRawAgenda === ai ? (
                            <div style={{ padding: '.65rem .85rem', background: 'hsl(var(--card))', border: '1.5px solid hsl(30,90%,55%/.4)', borderRadius: 8, display: 'flex', flexDirection: 'column', gap: 8, margin: '.5rem' }}>
                              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                                <span style={{ fontSize: '.72rem', fontWeight: 700, color: 'hsl(30,90%,55%)', display: 'flex', alignItems: 'center', gap: 4 }}>
                                  <Pencil size={11} /> Edit & Fix JSON Output (Agenda {ai + 1})
                                </span>
                                <span style={{ fontSize: '.65rem', color: 'hsl(var(--pencil))', fontStyle: 'italic' }}>
                                  Paste or edit JSON. If valid JSON, it will load into structured points.
                                </span>
                              </div>
                              <textarea
                                value={rawAgendaDraft}
                                onChange={e => setRawAgendaDraft(e.target.value)}
                                rows={8}
                                style={{
                                  width: '100%', fontSize: '.76rem', fontFamily: 'JetBrains Mono, monospace',
                                  color: 'hsl(var(--ink))', background: 'hsl(var(--paper))',
                                  padding: '.6rem', borderRadius: 6, border: '1px solid hsl(var(--border)/.5)',
                                  lineHeight: 1.45, resize: 'vertical', boxSizing: 'border-box'
                                }}
                              />
                              <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                                <button
                                  disabled={savingRawAgenda}
                                  onClick={() => handleSaveRawAgendaDiscussion(ai, rawAgendaDraft)}
                                  style={{
                                    display: 'inline-flex', alignItems: 'center', gap: 5, padding: '4px 12px',
                                    borderRadius: 6, background: 'hsl(140,70%,45%)', color: '#fff', border: 'none',
                                    fontSize: '.74rem', fontWeight: 700, cursor: savingRawAgenda ? 'not-allowed' : 'pointer'
                                  }}
                                >
                                  {savingRawAgenda ? <Loader size={11} className="spin" /> : <Save size={11} />} Save
                                </button>
                                <button
                                  onClick={() => { setEditingRawAgenda(null); setRawAgendaDraft('') }}
                                  style={{
                                    display: 'inline-flex', alignItems: 'center', gap: 4, padding: '4px 10px',
                                    borderRadius: 6, background: 'none', border: '1px solid hsl(var(--border)/.4)',
                                    color: 'hsl(var(--pencil))', fontSize: '.74rem', cursor: 'pointer'
                                  }}
                                >
                                  <X size={11} /> Cancel
                                </button>
                              </div>
                            </div>
                          ) : (
                            <div style={{ padding: '.65rem .85rem', background: 'hsl(var(--paper)/.5)', border: '1px solid hsl(30,90%,55%/.3)', borderRadius: 8, margin: '.5rem' }}>
                              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6 }}>
                                <span style={{ fontSize: '.68rem', fontWeight: 700, color: 'hsl(30,90%,55%)', textTransform: 'uppercase', letterSpacing: '.04em' }}>
                                  Raw Text Output (Non-JSON)
                                </span>
                                <button
                                  onClick={() => { setEditingRawAgenda(ai); setRawAgendaDraft(String(ag.discussion)) }}
                                  style={{
                                    display: 'inline-flex', alignItems: 'center', gap: 4, padding: '2px 8px',
                                    borderRadius: 6, background: 'hsl(30,90%,55%/.1)', border: '1px solid hsl(30,90%,55%/.3)',
                                    color: 'hsl(30,90%,55%)', fontSize: '.71rem', fontWeight: 600, cursor: 'pointer'
                                  }}
                                >
                                  <Pencil size={10} /> Edit / Fix JSON
                                </button>
                              </div>
                              <pre style={{ margin: 0, fontSize: '.76rem', color: 'hsl(var(--ink))', fontFamily: 'JetBrains Mono, monospace', lineHeight: 1.5, whiteSpace: 'pre-wrap', wordBreak: 'break-word', maxHeight: 200, overflowY: 'auto' }}>
                                {String(ag.discussion)}
                              </pre>
                            </div>
                          )
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* ── Feature 4: Recheck Agenda Points ── */}
              {momResult && (
                <div style={{ borderRadius: 14, border: '1.5px solid hsl(280,75%,60%/.2)', background: 'hsl(var(--card))', overflow: 'hidden' }}>
                  {/* Section header */}
                  <button
                    onClick={() => setRecheckOpen(o => !o)}
                    style={{ display: 'flex', alignItems: 'center', gap: 8, width: '100%', padding: '.65rem 1rem', background: 'linear-gradient(135deg,hsl(280,75%,60%/.07),hsl(260,70%,60%/.04))', border: 'none', cursor: 'pointer', textAlign: 'left' }}
                  >
                    {recheckOpen ? <ChevronDown size={13} style={{ color: 'hsl(280,75%,65%)' }} /> : <ChevronRight size={13} style={{ color: 'hsl(280,75%,65%)' }} />}
                    <BarChart2 size={13} style={{ color: 'hsl(280,75%,65%)' }} />
                    <span style={{ fontSize: '.8rem', fontWeight: 700, color: 'hsl(var(--ink))', flex: 1 }}>Recheck Agenda Points</span>
                    <span style={{ fontSize: '.67rem', color: 'hsl(var(--pencil))', fontStyle: 'italic' }}>Analysis only — no auto-changes</span>
                  </button>

                  {recheckOpen && (
                    <div style={{ padding: '.7rem 1rem', display: 'flex', flexDirection: 'column', gap: '.85rem' }}>
                      {/* Controls */}
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                        <button
                          onClick={runRecheckAnalysis}
                          disabled={recheckLoading}
                          style={{ display: 'inline-flex', alignItems: 'center', gap: 6, padding: '.4rem .9rem', borderRadius: 8, border: 'none', background: 'hsl(280,75%,60%)', color: '#fff', fontSize: '.76rem', fontWeight: 700, cursor: recheckLoading ? 'not-allowed' : 'pointer', opacity: recheckLoading ? 0.6 : 1, fontFamily: 'Inter' }}
                        >
                          {recheckLoading ? <Loader size={12} className="spin" /> : <BarChart2 size={12} />}
                          {recheckLoading ? 'Computing…' : 'Run Analysis'}
                        </button>
                        <span style={{ fontSize: '.72rem', color: 'hsl(var(--pencil))', fontFamily: 'Inter' }}>
                          Threshold: <strong style={{ color: 'hsl(280,75%,65%)' }}>{agendaMoveSimilarityDiff.toFixed(2)}</strong>
                          <span style={{ marginLeft: 4 }}>(adjust in settings panel)</span>
                        </span>
                        {recheckMatrix.length > 0 && (
                          <button
                            onClick={() => { setRecheckMatrix([]); setDismissedSuggestions(new Set()) }}
                            style={{ marginLeft: 'auto', display: 'inline-flex', alignItems: 'center', gap: 4, padding: '3px 8px', borderRadius: 6, border: '1px solid hsl(var(--border)/.4)', background: 'none', color: 'hsl(var(--pencil))', fontSize: '.68rem', cursor: 'pointer' }}
                          >
                            <X size={10} /> Clear
                          </button>
                        )}
                      </div>

                      {/* Similarity Matrix */}
                      {recheckMatrix.length > 0 && (() => {
                        const agendas = momResult.meeting?.agendas || []
                        return (
                          <div>
                            <div style={{ fontSize: '.69rem', fontWeight: 700, color: 'hsl(var(--pencil))', textTransform: 'uppercase', letterSpacing: '.04em', marginBottom: '.4rem' }}>Similarity Matrix</div>
                            <div style={{ overflowX: 'auto', borderRadius: 10, border: '1px solid hsl(var(--border)/.3)' }}>
                              <table style={{ borderCollapse: 'collapse', width: '100%', fontSize: '.69rem', fontFamily: 'JetBrains Mono, monospace' }}>
                                <thead>
                                  <tr style={{ background: 'hsl(var(--muted)/.4)' }}>
                                    <th style={{ padding: '5px 8px', textAlign: 'left', borderBottom: '1px solid hsl(var(--border)/.25)', color: 'hsl(var(--pencil))', fontWeight: 700, whiteSpace: 'nowrap', minWidth: 160 }}>Point</th>
                                    {agendas.map((ag, ai) => (
                                      <th key={ai} style={{ padding: '5px 7px', textAlign: 'center', borderBottom: '1px solid hsl(var(--border)/.25)', color: 'hsl(280,75%,65%)', fontWeight: 700, whiteSpace: 'nowrap', minWidth: 70 }}>
                                        A{ai + 1}
                                      </th>
                                    ))}
                                  </tr>
                                </thead>
                                <tbody>
                                  {recheckMatrix.map((row, ri) => {
                                    const best = Math.max(...row.similarities)
                                    const bestIdx = row.similarities.indexOf(best)
                                    const currentSim = row.similarities[row.agenda_idx] ?? 0
                                    const hasSuggestion = bestIdx !== row.agenda_idx && (best - currentSim) >= agendaMoveSimilarityDiff
                                    const isDismissed = dismissedSuggestions.has(`${row.agenda_idx}_${row.disc_idx}`)
                                    return (
                                      <tr key={ri} style={{ borderBottom: '1px solid hsl(var(--border)/.15)', background: hasSuggestion && !isDismissed ? 'hsl(280,75%,60%/.04)' : 'transparent' }}>
                                        <td style={{ padding: '5px 8px', color: 'hsl(var(--ink))', maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', fontFamily: 'Inter', fontSize: '.7rem' }}
                                          title={row.point}>
                                          <span style={{ fontSize: '.62rem', fontWeight: 700, color: 'hsl(var(--pencil))', marginRight: 4 }}>P{ri + 1}</span>
                                          {row.point.slice(0, 50)}{row.point.length > 50 ? '…' : ''}
                                        </td>
                                        {row.similarities.map((sim, si) => {
                                          const isCurrent = si === row.agenda_idx
                                          const isBest = si === bestIdx
                                          const isSuggested = hasSuggestion && si === bestIdx
                                          let cellBg = 'transparent'
                                          if (isCurrent) cellBg = 'hsl(280,75%,60%/.1)'
                                          if (isBest && !isCurrent) cellBg = 'hsl(140,70%,45%/.1)'
                                          if (isSuggested) cellBg = 'hsl(30,90%,55%/.12)'
                                          return (
                                            <td key={si} style={{
                                              padding: '5px 7px', textAlign: 'center',
                                              background: cellBg,
                                              border: isSuggested ? '1px solid hsl(30,90%,55%/.4)' : isCurrent ? '1px solid hsl(280,75%,60%/.25)' : '1px solid transparent',
                                              color: isBest ? 'hsl(140,70%,45%)' : isCurrent ? 'hsl(280,75%,65%)' : 'hsl(var(--pencil))',
                                              fontWeight: isBest || isCurrent ? 700 : 400,
                                            }}>
                                              {sim.toFixed(3)}
                                              {isCurrent && <span style={{ fontSize: '.58rem', display: 'block', color: 'hsl(280,75%,65%)', lineHeight: 1 }}>curr</span>}
                                              {isSuggested && <span style={{ fontSize: '.58rem', display: 'block', color: 'hsl(30,90%,55%)', lineHeight: 1 }}>→best</span>}
                                            </td>
                                          )
                                        })}
                                      </tr>
                                    )
                                  })}
                                </tbody>
                              </table>
                            </div>
                          </div>
                        )
                      })()}

                      {/* Suggested Moves */}
                      {recheckMatrix.length > 0 && (() => {
                        const agendas = momResult.meeting?.agendas || []
                        const suggestions = recheckMatrix.filter(row => {
                          const current = row.similarities[row.agenda_idx] ?? 0
                          const best = Math.max(...row.similarities)
                          const bestIdx = row.similarities.indexOf(best)
                          return bestIdx !== row.agenda_idx && (best - current) >= agendaMoveSimilarityDiff
                        })
                        if (!suggestions.length) return (
                          <div style={{ padding: '.6rem', borderRadius: 8, background: 'hsl(140,70%,45%/.06)', border: '1px solid hsl(140,70%,45%/.2)', fontSize: '.75rem', color: 'hsl(140,70%,45%)', fontFamily: 'Inter' }}>
                            ✓ All points appear well-placed — no moves suggested at threshold {agendaMoveSimilarityDiff.toFixed(2)}.
                          </div>
                        )
                        const activeSuggestions = suggestions.filter(row => !dismissedSuggestions.has(`${row.agenda_idx}_${row.disc_idx}`))
                        return (
                          <div>
                            <div style={{ fontSize: '.69rem', fontWeight: 700, color: 'hsl(var(--pencil))', textTransform: 'uppercase', letterSpacing: '.04em', marginBottom: '.45rem', display: 'flex', alignItems: 'center', gap: 6 }}>
                              Suggested Moves
                              <span style={{ fontSize: '.65rem', fontWeight: 600, padding: '1px 7px', borderRadius: 999, background: 'hsl(280,75%,60%/.12)', border: '1px solid hsl(280,75%,60%/.25)', color: 'hsl(280,75%,65%)' }}>{activeSuggestions.length} active</span>
                            </div>
                            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                              {suggestions.map((row, si) => {
                                const key = `${row.agenda_idx}_${row.disc_idx}`
                                const isDismissed = dismissedSuggestions.has(key)
                                const best = Math.max(...row.similarities)
                                const bestIdx = row.similarities.indexOf(best)
                                const currentSim = row.similarities[row.agenda_idx] ?? 0
                                const diff = best - currentSim
                                const fromAgenda = agendas[row.agenda_idx]?.agenda_topic || `Agenda ${row.agenda_idx + 1}`
                                const toAgenda = agendas[bestIdx]?.agenda_topic || `Agenda ${bestIdx + 1}`
                                return (
                                  <div key={si} style={{ borderRadius: 10, border: `1px solid ${isDismissed ? 'hsl(var(--border)/.2)' : 'hsl(280,75%,60%/.25)'}`, padding: '.6rem .8rem', background: isDismissed ? 'hsl(var(--muted)/.2)' : 'hsl(280,75%,60%/.04)', opacity: isDismissed ? 0.5 : 1, transition: 'opacity .2s' }}>
                                    <div style={{ display: 'flex', alignItems: 'flex-start', gap: 8 }}>
                                      {/* Dismiss checkbox */}
                                      <button
                                        onClick={() => setDismissedSuggestions(prev => {
                                          const next = new Set(prev)
                                          if (next.has(key)) next.delete(key); else next.add(key)
                                          return next
                                        })}
                                        style={{ flexShrink: 0, background: 'none', border: 'none', cursor: 'pointer', color: isDismissed ? 'hsl(var(--pencil))' : 'hsl(280,75%,65%)', paddingTop: 2 }}
                                        title={isDismissed ? 'Re-include this suggestion' : 'Dismiss this suggestion'}
                                      >
                                        {isDismissed ? <Square size={14} /> : <CheckSquare size={14} />}
                                      </button>
                                      <div style={{ flex: 1 }}>
                                        <p style={{ margin: '0 0 4px', fontSize: '.74rem', color: 'hsl(var(--ink))', fontFamily: 'Inter', lineHeight: 1.45 }}>
                                          {row.point.slice(0, 100)}{row.point.length > 100 ? '…' : ''}
                                        </p>
                                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, fontSize: '.68rem', fontFamily: 'Inter' }}>
                                          <span style={{ padding: '2px 7px', borderRadius: 5, background: 'hsl(var(--muted))', color: 'hsl(var(--pencil))' }}>From: {fromAgenda}</span>
                                          <span style={{ padding: '2px 7px', borderRadius: 5, background: 'hsl(140,70%,45%/.1)', border: '1px solid hsl(140,70%,45%/.3)', color: 'hsl(140,70%,45%)' }}>→ {toAgenda}</span>
                                          <span style={{ padding: '2px 7px', borderRadius: 5, background: 'hsl(var(--muted))', color: 'hsl(var(--pencil))' }}>Curr: {currentSim.toFixed(3)}</span>
                                          <span style={{ padding: '2px 7px', borderRadius: 5, background: 'hsl(140,70%,45%/.1)', color: 'hsl(140,70%,45%)', fontWeight: 700 }}>Best: {best.toFixed(3)}</span>
                                          <span style={{ padding: '2px 7px', borderRadius: 5, background: 'hsl(30,90%,55%/.1)', border: '1px solid hsl(30,90%,55%/.3)', color: 'hsl(30,90%,55%)', fontWeight: 700 }}>+{diff.toFixed(3)}</span>
                                        </div>
                                      </div>
                                    </div>
                                  </div>
                                )
                              })}
                            </div>

                            {/* Apply button */}
                            {activeSuggestions.length > 0 && (
                              <button
                                onClick={applyRecheckMoves}
                                disabled={applyingRecheckMoves}
                                style={{ marginTop: '.65rem', display: 'inline-flex', alignItems: 'center', gap: 6, padding: '.45rem 1.1rem', borderRadius: 8, border: 'none', background: 'hsl(280,75%,60%)', color: '#fff', fontSize: '.76rem', fontWeight: 700, cursor: applyingRecheckMoves ? 'not-allowed' : 'pointer', opacity: applyingRecheckMoves ? 0.6 : 1, fontFamily: 'Inter' }}
                              >
                                {applyingRecheckMoves ? <Loader size={12} className="spin" /> : <CheckCircle size={12} />}
                                Apply {activeSuggestions.length} Selected Change{activeSuggestions.length !== 1 ? 's' : ''}
                              </button>
                            )}
                          </div>
                        )
                      })()}
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* ── TAB 3: FINAL MOM ── */}
            <div style={{ display: activeRightTab === 'final_mom' ? 'flex' : 'none', flexDirection: 'column', gap: '1rem', flex: 1 }}>
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
      </div>
    </div>
  )
}
