import { useState, useCallback, useEffect, useRef } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import {
  ArrowLeft, Sparkles, Download, FileText, Clock, User, UserCheck,
  Loader, Pencil, Upload, Save, X, Play, Plus, Brain,
  Target, List, RefreshCw, FileDown, Layers, Video,
  ChevronUp, ChevronDown, ArrowRightLeft, Tag, Trash2, Sliders,
  RotateCcw, RotateCw, WandSparkles, FileUp, Search, Replace, History,
  Merge, Scissors, Check, CornerDownRight, GitMerge,
  BookOpen, AlertTriangle, CheckCircle, MessageSquare, ListChecks, Zap
} from 'lucide-react'
import api from '../api/client'
import { toast } from 'sonner'
import { getApiErrorDetail } from '../lib/errors'
import { AgendaTimelineControls, AgendaTimelineRange } from '../components/AgendaTimelineControls'
import { Stage2ContextPreviewModal, Stage2ContextGroup } from '../components/Stage2ContextPreviewModal'

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
  technical_terms: string[]
  dates: string[]
  numbers: string[]
  references: string[]
  action_items: string[]
  action_owner?: string | null
  action_owners?: string[]
  window_index: number
  raw_transcript_text?: string
  video_transcript_context?: string
}

const getActionOwnerText = (pt: any): string | null => {
  if (!pt) return null
  let owner = pt.action_owner || pt.action_owners || pt.owner || pt.assignee || pt.action_assignee || pt.assignees

  // If not found at top-level, extract assignees/owners from action_items array (e.g. separate action extraction)
  if (!owner && Array.isArray(pt.action_items) && pt.action_items.length > 0) {
    const itemAssignees: string[] = []
    for (const item of pt.action_items) {
      if (typeof item === 'object' && item !== null) {
        const a = item.assignee || item.owner || item.action_owner || item.action_owners || item.assignees
        if (a) {
          if (Array.isArray(a)) itemAssignees.push(...a.map(String))
          else itemAssignees.push(String(a))
        }
      } else if (typeof item === 'string') {
        const m = item.match(/\((?:Owner|Assignee):\s*([^)]+)\)/i)
        if (m && m[1]) itemAssignees.push(m[1].trim())
      }
    }
    if (itemAssignees.length > 0) {
      owner = itemAssignees
    }
  }

  // Also check if pt is itself an action item object with assignee/owner
  if (!owner && typeof pt === 'object') {
    owner = pt.assignee || pt.owner
  }

  // Also check if discussion_point has embedded (Owner: ...) pattern
  if (!owner && pt.discussion_point && typeof pt.discussion_point === 'string') {
    const m = pt.discussion_point.match(/\((?:Owner|Assignee|Action Owner):\s*([^)]+)\)/i) ||
              pt.discussion_point.match(/\[(?:Owner|Assignee|Action Owner):\s*([^\]]+)\]/i)
    if (m && m[1]) {
      owner = m[1].trim()
    }
  }

  if (!owner) return null
  if (Array.isArray(owner)) {
    const valid = owner
      .map(o => formatItemText(o))
      .filter(s => s && s.toLowerCase() !== 'null' && s.toLowerCase() !== 'none' && s.toLowerCase() !== 'n/a' && s.toLowerCase() !== 'undefined' && s.toLowerCase() !== 'unassigned')
    const unique = Array.from(new Set(valid))
    return unique.length > 0 ? unique.join(', ') : null
  }
  const str = formatItemText(owner)
  if (!str || str.toLowerCase() === 'null' || str.toLowerCase() === 'none' || str.toLowerCase() === 'n/a' || str.toLowerCase() === 'undefined' || str.toLowerCase() === 'unassigned') {
    return null
  }
  return str
}



const formatItemText = (item: any): string => {
  if (item === null || item === undefined) return ''
  if (typeof item === 'string') return item.trim()
  if (typeof item === 'number' || typeof item === 'boolean') return String(item)
  if (typeof item === 'object') {
    const task = item.speaker_text || item.speaker_name || item.speaker || item.name || item.task || item.description || item.item || item.text || item.action || item.decision || item.point || item.val || item.value || item.label || item.real_name
    const assignee = item.assignee || item.owner
    const assigner = item.assigner
    const deadline = item.deadline || item.due || item.date
    const cond = item.conditions || item.condition

    const parts: string[] = []
    if (task) parts.push(String(task).trim())
    if (assigner) parts.push(`(By: ${String(assigner).trim()})`)
    if (assignee) parts.push(`(Owner: ${String(assignee).trim()})`)
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

interface PolishedPoint {
  id: string
  original_point_id: string
  original_point_ids?: string[]
  polished_text: string
  timeline_start: number
  timeline_end: number
  speakers: string[]
  technical_terms: string[]
  dates: string[]
  numbers: string[]
  references: string[]
  action_items: Array<{ task: string; assignee: string | null; deadline: string | null }> | string[]
  context_usage_report?: {
    meeting_context_used: boolean
    global_context_used: boolean
    context_added: boolean
    documents: string[]
  }
  retrieved_context?: {
    meeting_chunks: Array<{ text: string; score: number; filename: string }>
    global_chunks: Array<{ text: string; score: number; filename: string }>
    previous_meeting_chunks?: Array<{
      text: string
      score: number
      meeting_id?: string
      meeting_name?: string
      date?: string
      speakers?: string
      action_owner?: string
    }>
    context_usage_report?: any
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
    discussion_order?: string[]
    agenda_timeline?: Record<string, AgendaTimelineRange>
    enable_agenda_order?: boolean
    enable_agenda_timeline?: boolean
    completed_at: string | null
  } | null
  final_rom: {
    agendas: FinalRomAgenda[]
    speaker_mappings?: Record<string, string>
    last_edited_at?: string | null
    include_agenda_doc_points?: boolean
    date?: string
    time?: string
    members_present?: string[]
  } | null
  final_rom_versions?: {
    long?: {
      agendas: FinalRomAgenda[]
      speaker_mappings?: Record<string, string>
      last_edited_at?: string | null
      include_agenda_doc_points?: boolean
      date?: string
      time?: string
      members_present?: string[]
    }
    short?: {
      agendas: FinalRomAgenda[]
      speaker_mappings?: Record<string, string>
      last_edited_at?: string | null
      include_agenda_doc_points?: boolean
      date?: string
      time?: string
      members_present?: string[]
    }
    medium?: {
      agendas: FinalRomAgenda[]
      speaker_mappings?: Record<string, string>
      last_edited_at?: string | null
      include_agenda_doc_points?: boolean
      date?: string
      time?: string
      members_present?: string[]
    }
  }
  final_rom_active_version?: 'long' | 'short' | 'medium'
  outdated_warnings?: {
    stage1?: string
    stage2?: string
    stage3?: string
    final_rom?: string
  }
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

function Stage1ProgressBanner({ progress }: {
  progress: {
    windows_completed: number
    windows_total: number
    concurrency: number
    eta_seconds: number | null
    elapsed_seconds: number
    status: string
  }
}) {
  const { windows_completed, windows_total, eta_seconds, elapsed_seconds, concurrency } = progress
  const pct = windows_total > 0 ? Math.round((windows_completed / windows_total) * 100) : 0

  const fmtSecs = (s: number | null) => {
    if (s === null || s === undefined) return null
    const sRound = Math.round(s)
    if (sRound < 60) return `${sRound}s`
    const m = Math.floor(sRound / 60)
    const rem = sRound % 60
    return rem > 0 ? `${m}m ${rem}s` : `${m}m`
  }

  const etaLabel = fmtSecs(eta_seconds)
  const elapsedLabel = fmtSecs(elapsed_seconds)

  return (
    <div style={{
      borderRadius: 12,
      border: '1.5px solid hsl(280,75%,65%/.35)',
      background: 'linear-gradient(135deg, hsl(280,75%,65%/.08) 0%, hsl(220,80%,60%/.06) 100%)',
      padding: '1rem 1.25rem',
      display: 'flex',
      flexDirection: 'column',
      gap: '0.65rem',
      backdropFilter: 'blur(8px)',
    }}>
      {/* Header row */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
          <Loader size={13} className="spin" style={{ color: 'hsl(280,75%,65%)' }} />
          <span style={{ fontSize: '.8rem', fontWeight: 700, color: 'hsl(280,75%,65%)', fontFamily: 'Inter' }}>
            Extracting Stage 1
          </span>
          {concurrency > 1 && (
            <span style={{
              fontSize: '.65rem', fontWeight: 600, padding: '1px 6px', borderRadius: 999,
              background: 'hsl(220,80%,60%/.18)', border: '1px solid hsl(220,80%,60%/.3)',
              color: 'hsl(220,80%,60%)', fontFamily: 'Inter'
            }}>
              ×{concurrency} parallel
            </span>
          )}
        </div>
        <span style={{ fontSize: '.78rem', fontWeight: 700, color: 'hsl(var(--ink))', fontFamily: 'Inter' }}>
          Window {windows_completed} / {windows_total}
        </span>
      </div>

      {/* Progress bar */}
      <div style={{ borderRadius: 999, background: 'hsl(var(--border)/.5)', height: 6, overflow: 'hidden' }}>
        <div style={{
          height: '100%',
          width: `${pct}%`,
          borderRadius: 999,
          background: 'linear-gradient(90deg, hsl(280,75%,60%), hsl(220,80%,60%))',
          transition: 'width 0.6s ease',
          boxShadow: '0 0 8px hsl(280,75%,60%/.5)',
        }} />
      </div>

      {/* Stats row */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 16, fontSize: '.72rem', fontFamily: 'Inter', color: 'hsl(var(--pencil))' }}>
        <span style={{ fontWeight: 600, color: 'hsl(var(--ink))' }}>{pct}%</span>
        {elapsedLabel && <span>Elapsed: <strong>{elapsedLabel}</strong></span>}
        {etaLabel && windows_completed > 0 && (
          <span style={{ marginLeft: 'auto', color: 'hsl(220,80%,60%)', fontWeight: 600 }}>
            ~{etaLabel} remaining
          </span>
        )}
        {!etaLabel && windows_completed === 0 && (
          <span style={{ marginLeft: 'auto', color: 'hsl(var(--pencil))', fontStyle: 'italic' }}>
            Estimating...
          </span>
        )}
      </div>
    </div>
  )
}

// ── Stage 1 Window Re-run Modal Component ─────────────────────────────────────

function Stage1WindowRerunModal({
  recordingId,
  windowData,
  transcriptWindowMinutes,
  onClose,
  onAccepted,
}: {
  recordingId: string
  windowData: {
    windowIndex: number
    timelineStart: number
    timelineEnd: number
    points: DiscussionPoint[]
  }
  transcriptWindowMinutes: number
  onClose: () => void
  onAccepted: (newRomData: RomData) => void
}) {
  const [feedback, setFeedback] = useState('')
  const [loading, setLoading] = useState(false)
  const [accepting, setAccepting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<{
    transcript_text: string
    video_context: string
    original_points: DiscussionPoint[]
    regenerated_points: DiscussionPoint[]
  } | null>(null)

  const handleRerun = async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await api.post(`/rom/${recordingId}/stage1/rerun-window`, {
        window_index: windowData.windowIndex,
        user_feedback: feedback,
        transcript_window_minutes: transcriptWindowMinutes,
      })
      if (res.data?.result) {
        setResult(res.data.result)
        toast.success(`Window ${windowData.windowIndex} points regenerated`)
      }
    } catch (e: any) {
      setError(getApiErrorDetail(e) || 'Failed to re-run window')
    } finally {
      setLoading(false)
    }
  }

  const handleAccept = async () => {
    if (!result) return
    setAccepting(true)
    try {
      const res = await api.post(`/rom/${recordingId}/stage1/rerun-window/accept`, {
        window_index: windowData.windowIndex,
        user_feedback: feedback,
        original_points: result.original_points || windowData.points,
        corrected_points: result.regenerated_points,
        transcript_window: result.transcript_text,
      })
      if (res.data?.rom_data) {
        onAccepted(res.data.rom_data)
        toast.success(`Window ${windowData.windowIndex} updated and feedback saved for DSPy training!`)
        onClose()
      }
    } catch (e: any) {
      toast.error(getApiErrorDetail(e) || 'Failed to accept updated points')
    } finally {
      setAccepting(false)
    }
  }

  const sourceTranscript = result?.transcript_text || windowData.points[0]?.raw_transcript_text || ''

  return (
    <div style={{
      position: 'fixed', inset: 0, zIndex: 100,
      background: 'rgba(0,0,0,0.65)', backdropFilter: 'blur(4px)',
      display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '1rem',
    }}>
      <div style={{
        background: 'hsl(var(--card))',
        border: '1.5px solid hsl(280,75%,60%/.4)',
        borderRadius: 14,
        width: '100%',
        maxWidth: 820,
        maxHeight: '90vh',
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden',
        boxShadow: '0 20px 40px rgba(0,0,0,0.4)',
      }}>
        {/* Header */}
        <div style={{
          padding: '.85rem 1.25rem',
          borderBottom: '1px solid hsl(var(--border)/.5)',
          background: 'hsl(280,75%,60%/.08)',
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <div style={{
              width: 28, height: 28, borderRadius: 8,
              background: 'hsl(280,75%,60%/.15)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
            }}>
              <RotateCcw size={15} color="hsl(280,75%,60%)" />
            </div>
            <div>
              <div style={{ fontSize: '.92rem', fontWeight: 700, color: 'hsl(var(--ink))' }}>
                Re-run Window {windowData.windowIndex}
              </div>
              <div style={{ fontSize: '.72rem', color: 'hsl(var(--pencil))', display: 'flex', alignItems: 'center', gap: 4 }}>
                <Clock size={10} /> {fmtTime(windowData.timelineStart)} – {fmtTime(windowData.timelineEnd)}
                <span>•</span>
                <span>{windowData.points.length} extracted point(s)</span>
              </div>
            </div>
          </div>

          <button
            onClick={onClose}
            style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'hsl(var(--pencil))', padding: 4 }}
          >
            <X size={18} />
          </button>
        </div>

        {/* Body */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '1.25rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          {/* Source transcript display */}
          <div style={{ borderRadius: 8, border: '1px solid hsl(var(--border)/.4)', background: 'hsl(var(--muted)/.2)', padding: '.75rem .9rem' }}>
            <div style={{ fontSize: '.72rem', fontWeight: 700, color: 'hsl(var(--pencil))', textTransform: 'uppercase', letterSpacing: '.04em', marginBottom: 4 }}>
              Window Source Transcript
            </div>
            <pre style={{
              margin: 0, maxHeight: 120, overflowY: 'auto', fontSize: '.75rem', lineHeight: 1.45,
              whiteSpace: 'pre-wrap', fontFamily: 'Inter, sans-serif', color: 'hsl(var(--ink))'
            }}>
              {sourceTranscript || 'Source transcript will load when re-running.'}
            </pre>
          </div>

          {/* Current vs Regenerated points */}
          <div style={{ display: 'grid', gridTemplateColumns: result ? '1fr 1fr' : '1fr', gap: '1rem' }}>
            {/* Current extracted points */}
            <div style={{ borderRadius: 8, border: '1px solid hsl(var(--border)/.4)', padding: '.75rem .9rem', background: 'hsl(var(--card))' }}>
              <div style={{ fontSize: '.74rem', fontWeight: 700, color: 'hsl(var(--pencil))', marginBottom: 6, display: 'flex', alignItems: 'center', gap: 4 }}>
                <ListChecks size={13} /> Current Extracted Points ({windowData.points.length})
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6, maxHeight: 200, overflowY: 'auto' }}>
                {windowData.points.map((pt, i) => {
                  const owner = getActionOwnerText(pt)
                  return (
                    <div key={i} style={{ padding: '.45rem .6rem', borderRadius: 6, background: 'hsl(var(--paper)/.4)', border: '1px solid hsl(var(--border)/.3)', fontSize: '.76rem', lineHeight: 1.4 }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 2 }}>
                        <span style={{ fontWeight: 700, color: 'hsl(280,75%,60%)' }}>P{i + 1}</span>
                        {owner ? (
                          <span style={{ background: 'hsl(35,95%,50%/.15)', color: 'hsl(35,95%,40%)', border: '1px solid hsl(35,95%,50%/.35)', padding: '0 5px', borderRadius: 6, fontSize: '.64rem', fontWeight: 700, display: 'inline-flex', alignItems: 'center', gap: 3 }}>
                            <UserCheck size={8} /> Owner: {owner}
                          </span>
                        ) : (
                          <span style={{ background: 'hsl(var(--muted)/.5)', color: 'hsl(var(--pencil))', border: '1px solid hsl(var(--border)/.3)', padding: '0 5px', borderRadius: 6, fontSize: '.64rem', fontWeight: 600, display: 'inline-flex', alignItems: 'center', gap: 3 }}>
                            <UserCheck size={8} /> No action owner
                          </span>
                        )}
                      </div>
                      <div>{formatItemText(pt.discussion_point)}</div>
                    </div>
                  )
                })}
              </div>
            </div>

            {/* Regenerated points */}
            {result && (
              <div style={{ borderRadius: 8, border: '1.5px solid hsl(140,70%,45%/.4)', padding: '.75rem .9rem', background: 'hsl(140,70%,45%/.03)' }}>
                <div style={{ fontSize: '.74rem', fontWeight: 700, color: 'hsl(140,70%,45%)', marginBottom: 6, display: 'flex', alignItems: 'center', gap: 4 }}>
                  <CheckCircle size={13} /> Regenerated Points ({result.regenerated_points.length})
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 6, maxHeight: 200, overflowY: 'auto' }}>
                  {result.regenerated_points.length === 0 ? (
                    <div style={{ color: 'hsl(var(--pencil))', fontSize: '.76rem', fontStyle: 'italic' }}>No points extracted.</div>
                  ) : (
                    result.regenerated_points.map((pt, i) => {
                      const owner = getActionOwnerText(pt)
                      return (
                        <div key={i} style={{ padding: '.45rem .6rem', borderRadius: 6, background: 'hsl(var(--paper))', border: '1px solid hsl(140,70%,45%/.3)', fontSize: '.76rem', lineHeight: 1.4 }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 2 }}>
                            <span style={{ fontWeight: 700, color: 'hsl(140,70%,45%)' }}>New P{i + 1}</span>
                            {owner ? (
                              <span style={{ background: 'hsl(35,95%,50%/.15)', color: 'hsl(35,95%,40%)', border: '1px solid hsl(35,95%,50%/.35)', padding: '0 5px', borderRadius: 6, fontSize: '.64rem', fontWeight: 700, display: 'inline-flex', alignItems: 'center', gap: 3 }}>
                                <UserCheck size={8} /> Owner: {owner}
                              </span>
                            ) : (
                              <span style={{ background: 'hsl(var(--muted)/.5)', color: 'hsl(var(--pencil))', border: '1px solid hsl(var(--border)/.3)', padding: '0 5px', borderRadius: 6, fontSize: '.64rem', fontWeight: 600, display: 'inline-flex', alignItems: 'center', gap: 3 }}>
                                <UserCheck size={8} /> No action owner
                              </span>
                            )}
                          </div>
                          <div>{formatItemText(pt.discussion_point)}</div>
                        </div>
                      )
                    })
                  )}
                </div>
              </div>
            )}
          </div>

          {/* User Feedback & Instructions */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '.4rem' }}>
            <label style={{ fontSize: '.75rem', fontWeight: 700, color: 'hsl(var(--ink))', display: 'flex', alignItems: 'center', gap: 5 }}>
              <MessageSquare size={13} color="hsl(280,75%,60%)" /> User Feedback & Correction Instructions:
            </label>
            <textarea
              value={feedback}
              onChange={e => setFeedback(e.target.value)}
              placeholder="e.g. You didn't extract this point from this window: The team decided to deploy the new release on Sunday night and designated Sarah as the owner."
              rows={3}
              style={{
                width: '100%', padding: '.6rem .75rem', borderRadius: 8,
                border: '1.5px solid hsl(280,75%,60%/.4)', background: 'hsl(var(--background))',
                fontSize: '.8rem', color: 'hsl(var(--ink))', fontFamily: 'Inter',
                outline: 'none', resize: 'vertical', boxSizing: 'border-box',
              }}
            />
            <div style={{ fontSize: '.71rem', color: 'hsl(var(--pencil))' }}>
              Your feedback will be sent directly to the LLM along with the transcript to guide the re-extraction, and saved for DSPy training.
            </div>
          </div>

          {error && (
            <div style={{ padding: '.5rem .75rem', borderRadius: 6, background: 'hsl(0,75%,55%/.1)', border: '1px solid hsl(0,75%,55%/.3)', color: 'hsl(0,75%,55%)', fontSize: '.75rem' }}>
              {error}
            </div>
          )}
        </div>

        {/* Footer */}
        <div style={{
          padding: '.75rem 1.25rem',
          borderTop: '1px solid hsl(var(--border)/.5)',
          background: 'hsl(var(--muted)/.2)',
          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        }}>
          <button
            onClick={onClose}
            style={{
              padding: '.45rem .9rem', borderRadius: 7, border: '1px solid hsl(var(--border))',
              background: 'transparent', color: 'hsl(var(--ink))', fontSize: '.78rem',
              fontWeight: 600, cursor: 'pointer', fontFamily: 'Inter',
            }}
          >
            Cancel
          </button>

          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <button
              onClick={handleRerun}
              disabled={loading}
              style={{
                padding: '.45rem 1rem', borderRadius: 7, border: 'none',
                background: 'hsl(280,75%,60%)', color: 'white', fontSize: '.78rem',
                fontWeight: 700, cursor: loading ? 'not-allowed' : 'pointer',
                fontFamily: 'Inter', display: 'flex', alignItems: 'center', gap: 5,
                opacity: loading ? 0.7 : 1,
              }}
            >
              {loading ? <Loader size={12} className="spin" /> : <Play size={12} />}
              {loading ? 'Regenerating...' : 'Regenerate Window'}
            </button>

            {result && (
              <button
                onClick={handleAccept}
                disabled={accepting}
                style={{
                  padding: '.45rem 1.1rem', borderRadius: 7, border: 'none',
                  background: 'hsl(140,70%,45%)', color: 'white', fontSize: '.78rem',
                  fontWeight: 700, cursor: accepting ? 'not-allowed' : 'pointer',
                  fontFamily: 'Inter', display: 'flex', alignItems: 'center', gap: 5,
                  opacity: accepting ? 0.7 : 1,
                }}
              >
                {accepting ? <Loader size={12} className="spin" /> : <Check size={12} />}
                {accepting ? 'Saving...' : 'Accept & Save Window'}
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

// ── Stage 2 Reference Examples Panel Component ────────────────────────────────

function Stage2ReferenceExamplesPanel({
  examples,
  onChange,
  onSave,
  saving,
}: {
  examples: string[]
  onChange: (newExamples: string[]) => void
  onSave: () => Promise<void>
  saving: boolean
}) {
  const [collapsed, setCollapsed] = useState(false)
  const [savedSuccess, setSavedSuccess] = useState(false)

  const handleSave = async () => {
    await onSave()
    setSavedSuccess(true)
    setTimeout(() => setSavedSuccess(false), 2500)
  }

  const addExample = () => {
    onChange([...examples, ''])
  }

  const removeExample = (idx: number) => {
    onChange(examples.filter((_, i) => i !== idx))
  }

  const updateExample = (idx: number, text: string) => {
    onChange(examples.map((p, i) => (i === idx ? text : p)))
  }

  return (
    <div style={{
      borderRadius: 10,
      border: '1.5px solid hsl(38,92%,50%/.4)',
      background: 'hsl(var(--card))',
      padding: '.85rem 1rem',
      display: 'flex',
      flexDirection: 'column',
      gap: '.75rem',
      boxShadow: '0 2px 8px rgba(0,0,0,0.05)',
    }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div
          onClick={() => setCollapsed(v => !v)}
          style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer', userSelect: 'none' }}
        >
          <div style={{
            width: 26, height: 26, borderRadius: 6,
            background: 'hsl(38,92%,50%/.15)',
            display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0
          }}>
            <BookOpen size={14} color="hsl(38,92%,50%)" />
          </div>
          <div>
            <div style={{ fontSize: '.84rem', fontWeight: 700, color: 'hsl(var(--ink))', display: 'flex', alignItems: 'center', gap: 6 }}>
              Reference Examples — Writing Style Only
              <span style={{ fontSize: '.68rem', fontWeight: 700, padding: '1px 6px', borderRadius: 6, background: 'hsl(38,92%,50%/.15)', color: 'hsl(38,92%,45%)' }}>
                {examples.filter(p => p.trim()).length} examples
              </span>
            </div>
            <div style={{ fontSize: '.72rem', color: 'hsl(var(--pencil))' }}>
              Used strictly for sentence structure, granularity, and tone during Stage 2 generation.
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
          {savedSuccess && (
            <span style={{ display: 'flex', alignItems: 'center', gap: 3, fontSize: '.72rem', color: 'hsl(140,70%,45%)', fontWeight: 700 }}>
              <Check size={12} /> Saved
            </span>
          )}
          <button
            onClick={() => setCollapsed(v => !v)}
            style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'hsl(var(--pencil))', padding: '2px 4px' }}
          >
            {collapsed ? <ChevronDown size={15} /> : <ChevronUp size={15} />}
          </button>
        </div>
      </div>

      {!collapsed && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '.75rem' }}>
          {/* Style-only warning banner */}
          <div style={{
            display: 'flex', gap: 8, padding: '.6rem .75rem', borderRadius: 8,
            background: 'hsl(38,92%,50%/.08)', border: '1px solid hsl(38,92%,50%/.3)',
            fontSize: '.74rem', color: 'hsl(var(--ink))', lineHeight: 1.45,
          }}>
            <AlertTriangle size={15} color="hsl(38,92%,50%)" style={{ flexShrink: 0, marginTop: 1 }} />
            <div>
              <strong>Reference Examples — Writing Style Only:</strong> These examples show how discussion points should be written.
              They are <strong>NOT context, facts, evidence, or information to use in the output</strong>.
              The LLM will use them strictly for style, granularity, and tone, and will never copy their facts.
            </div>
          </div>

          {/* List of examples */}
          {examples.length === 0 ? (
            <div style={{
              textAlign: 'center', padding: '1rem', color: 'hsl(var(--pencil))', fontSize: '.78rem',
              border: '1px dashed hsl(var(--border))', borderRadius: 8, background: 'hsl(var(--muted)/.2)'
            }}>
              No style examples added yet. Click <strong>Add Example</strong> to guide how Stage 2 points are phrased.
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              {examples.map((ex, idx) => (
                <div key={idx} style={{ display: 'flex', gap: 6, alignItems: 'flex-start' }}>
                  <span style={{
                    fontSize: '.68rem', fontWeight: 700, color: 'hsl(38,92%,45%)',
                    background: 'hsl(38,92%,50%/.15)', padding: '2px 6px', borderRadius: 5,
                    fontFamily: 'JetBrains Mono', marginTop: 4, flexShrink: 0,
                  }}>
                    #{idx + 1}
                  </span>
                  <textarea
                    value={ex}
                    onChange={e => updateExample(idx, e.target.value)}
                    placeholder="Enter an example discussion point demonstrating the desired writing style, detail level, and structure..."
                    rows={2}
                    style={{
                      flex: 1, padding: '.45rem .65rem', borderRadius: 6,
                      border: '1px solid hsl(var(--border))', background: 'hsl(var(--background))',
                      fontSize: '.78rem', color: 'hsl(var(--ink))', fontFamily: 'Inter',
                      outline: 'none', resize: 'vertical',
                    }}
                  />
                  <button
                    onClick={() => removeExample(idx)}
                    title="Remove example"
                    style={{
                      background: 'none', border: 'none', cursor: 'pointer',
                      color: 'hsl(var(--destructive))', padding: '4px', marginTop: 3
                    }}
                  >
                    <Trash2 size={13} />
                  </button>
                </div>
              ))}
            </div>
          )}

          {/* Actions */}
          <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end', alignItems: 'center' }}>
            <button
              onClick={addExample}
              style={{
                display: 'inline-flex', alignItems: 'center', gap: 4,
                padding: '.35rem .7rem', borderRadius: 6,
                border: '1px dashed hsl(38,92%,50%/.5)', background: 'hsl(38,92%,50%/.06)',
                color: 'hsl(38,92%,40%)', fontSize: '.74rem', fontWeight: 700,
                cursor: 'pointer', fontFamily: 'Inter',
              }}
            >
              <Plus size={12} /> Add Example
            </button>
            <button
              onClick={handleSave}
              disabled={saving}
              style={{
                display: 'inline-flex', alignItems: 'center', gap: 4,
                padding: '.35rem .85rem', borderRadius: 6,
                border: 'none', background: 'hsl(38,92%,45%)',
                color: 'white', fontSize: '.74rem', fontWeight: 700,
                cursor: saving ? 'not-allowed' : 'pointer', fontFamily: 'Inter',
                opacity: saving ? 0.7 : 1,
              }}
            >
              {saving ? <Loader size={12} className="spin" /> : <Save size={12} />}
              {saving ? 'Saving...' : 'Save Examples'}
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

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
  const [activeTab, setActiveTab] = useState<'stage1' | 'stage2' | 'stage3' | 'final'>('stage1')


  // Stage 1 controls
  const [transcriptWindow, setTranscriptWindow] = useState(2)
  const [stage1Status, setStage1Status] = useState<ProcessState>('idle')
  const [stage1Progress, setStage1Progress] = useState<{
    windows_completed: number
    windows_total: number
    concurrency: number
    eta_seconds: number | null
    elapsed_seconds: number
    status: string
  } | null>(null)
  const stage1PollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // Stage 1 Re-run Window state
  const [rerunModalWindow, setRerunModalWindow] = useState<{
    windowIndex: number
    timelineStart: number
    timelineEnd: number
    points: DiscussionPoint[]
  } | null>(null)

  // Stage 2 controls
  const [meetingTopK, setMeetingTopK] = useState(5)
  const [globalTopK, setGlobalTopK] = useState(3)
  const [discussionWindowSize, setDiscussionWindowSize] = useState(5)
  const [minSimilarityThreshold, setMinSimilarityThreshold] = useState<number | null>(0.80)
  const [processAllTogether, setProcessAllTogether] = useState(false)
  const [stage2Status, setStage2Status] = useState<ProcessState>('idle')
  // Stage 2 meeting documents (uploaded inline, text extracted)
  const [stage2MeetingDocs, setStage2MeetingDocs] = useState<{ id: string; name: string }[]>([])
  const [uploadingStage2Doc, setUploadingStage2Doc] = useState(false)
  const stage2DocInputRef = useRef<HTMLInputElement>(null)

  // Stage 2 Reference Example Points
  const [stage2ReferenceExamples, setStage2ReferenceExamples] = useState<string[]>([])
  const [savingStage2Examples, setSavingStage2Examples] = useState(false)

  // Stage 2 Previous Meeting Context (Dual Mode)
  const [previousMeetingMode, setPreviousMeetingMode] = useState<'auto' | 'select' | 'off'>('auto')
  const [selectedPreviousMeetingId, setSelectedPreviousMeetingId] = useState<string>('')
  const [previousMeetingTopK, setPreviousMeetingTopK] = useState(3)
  const [availablePreviousMeetings, setAvailablePreviousMeetings] = useState<Array<{
    id: string
    name: string
    title: string
    date: string
    stage2_count: number
    has_stage2: boolean
  }>>([])
  const [loadingPreviousMeetings, setLoadingPreviousMeetings] = useState(false)

  // Stage 2 editing features
  const [selectedPointIds, setSelectedPointIds] = useState<Set<string>>(new Set())
  const [showFindReplace, setShowFindReplace] = useState(false)
  const [findText, setFindText] = useState('')
  const [replaceText, setReplaceText] = useState('')
  const [findPreviewCount, setFindPreviewCount] = useState<number | null>(null)
  const [findPreviewPoints, setFindPreviewPoints] = useState<{point_id: string; count: number; point_number: number}[]>([])
  const [applyingFindReplace, setApplyingFindReplace] = useState(false)
  const [mergingPoints, setMergingPoints] = useState(false)
  const [splittingPoint, setSplittingPoint] = useState(false)
  const [deletingText, setDeletingText] = useState(false)
  const [contextMenu, setContextMenu] = useState<{ x: number; y: number; pointId: string; selectedText: string } | null>(null)
  const [showChangeHistory, setShowChangeHistory] = useState(false)
  const [changeHistory, setChangeHistory] = useState<any[]>([])
  const [loadingHistory, setLoadingHistory] = useState(false)
  const [revertingChangeId, setRevertingChangeId] = useState<string | null>(null)
  const [redoingChangeId, setRedoingChangeId] = useState<string | null>(null)
  const [editingStage2PointId, setEditingStage2PointId] = useState<string | null>(null)
  const [stage2EditDraft, setStage2EditDraft] = useState('')
  const [savingStage2Point, setSavingStage2Point] = useState(false)

  // Stage 2 Retrieve Context Preview state
  const [showContextPreview, setShowContextPreview] = useState(false)
  const [contextPreviewGroups, setContextPreviewGroups] = useState<Stage2ContextGroup[]>([])
  const [contextPreviewTotalGroups, setContextPreviewTotalGroups] = useState(0)
  const [contextPreviewTotalPoints, setContextPreviewTotalPoints] = useState(0)
  const [previewingContext, setPreviewingContext] = useState(false)

  useEffect(() => {
    const handleScrollOrClick = () => {
      if (contextMenu) setContextMenu(null)
    }
    window.addEventListener('scroll', handleScrollOrClick, true)
    return () => window.removeEventListener('scroll', handleScrollOrClick, true)
  }, [contextMenu])

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

  // Stage 3 – Skip agenda tracking: { [agendaId]: { skipped: boolean, note: string } }
  const [skippedAgendas, setSkippedAgendas] = useState<Record<string, { skipped: boolean; note: string }>>({})

  // Final ROM – Include Action Points toggle
  const [includeActionPointsInRom, setIncludeActionPointsInRom] = useState(false)
  const [togglingActionPoints, setTogglingActionPoints] = useState(false)

  // Stage 3 – Agenda Discussion Order & Approximate Timeline Optional Controls
  const [discussionOrder, setDiscussionOrder] = useState<string[]>([])
  const [agendaTimeline, setAgendaTimeline] = useState<Record<string, AgendaTimelineRange>>({})
  const [enableAgendaOrder, setEnableAgendaOrder] = useState<boolean>(false)
  const [enableAgendaTimeline, setEnableAgendaTimeline] = useState<boolean>(false)

  // Compute effective meeting duration
  const effectiveMeetingDuration = Math.max(
    recording?.duration && recording.duration > 10
      ? recording.duration
      : Math.max(
          ...((romData?.stage2?.polished_points || []).map(p => p.timeline_end || 0)),
          ...((romData?.stage1?.windows || []).map(w => w.window_end || 0)),
          1800
        ),
    60
  )

  // Synchronize discussionOrder, agendaTimeline, and enable flags with romData.stage3
  useEffect(() => {
    if (typeof romData?.stage3?.enable_agenda_order === 'boolean') {
      setEnableAgendaOrder(romData.stage3.enable_agenda_order)
    }
    if (typeof romData?.stage3?.enable_agenda_timeline === 'boolean') {
      setEnableAgendaTimeline(romData.stage3.enable_agenda_timeline)
    }

    const s3Agendas = romData?.stage3?.agendas || []
    if (s3Agendas.length === 0) {
      setDiscussionOrder([])
      setAgendaTimeline({})
      return
    }

    const currentIds = s3Agendas.map((a, idx) => a.agenda_id || `A${idx + 1}`)
    const savedOrder = romData?.stage3?.discussion_order || []

    let newOrder = savedOrder.filter(id => currentIds.includes(id))
    currentIds.forEach(id => {
      if (!newOrder.includes(id)) newOrder.push(id)
    })
    if (newOrder.length === 0) newOrder = currentIds

    setDiscussionOrder(newOrder)

    const savedTimeline = romData?.stage3?.agenda_timeline || {}
    const hasValidSavedTimeline = newOrder.every(
      id => savedTimeline[id] && typeof savedTimeline[id].start_sec === 'number' && typeof savedTimeline[id].end_sec === 'number'
    )

    if (hasValidSavedTimeline) {
      setAgendaTimeline(savedTimeline)
    } else {
      const count = newOrder.length
      const slice = effectiveMeetingDuration / count
      const initialTimeline: Record<string, AgendaTimelineRange> = {}
      newOrder.forEach((id, idx) => {
        initialTimeline[id] = {
          start_sec: Math.round(idx * slice * 10) / 10,
          end_sec: Math.round((idx + 1) * slice * 10) / 10,
        }
      })
      setAgendaTimeline(initialTimeline)
    }

    // Synchronize skipped agendas from saved data
    const savedSkipped = romData?.stage3?.skipped_agendas
    if (savedSkipped && typeof savedSkipped === 'object') {
      const restored: Record<string, { skipped: boolean; note: string }> = {}
      for (const [aid, info] of Object.entries(savedSkipped)) {
        const noteVal = typeof info === 'object' && info !== null ? (info as any).note || 'Keep this agenda if forward to next meeting' : 'Keep this agenda if forward to next meeting'
        restored[aid] = { skipped: true, note: noteVal }
      }
      setSkippedAgendas(prev => {
        // Only update if different to avoid infinite loops
        const prevKeys = Object.keys(prev).filter(k => prev[k].skipped).sort().join(',')
        const newKeys = Object.keys(restored).sort().join(',')
        if (prevKeys !== newKeys) return restored
        return prev
      })
    }

    // Synchronize include_action_points from final_rom
    if (typeof romData?.final_rom?.include_action_points === 'boolean') {
      setIncludeActionPointsInRom(romData.final_rom.include_action_points)
    }
  }, [romData?.stage3?.agendas, romData?.stage3?.discussion_order, romData?.stage3?.agenda_timeline, romData?.stage3?.enable_agenda_order, romData?.stage3?.enable_agenda_timeline, romData?.stage3?.skipped_agendas, romData?.final_rom?.include_action_points, effectiveMeetingDuration])

  const resetAgendaOrderAndTimeline = () => {
    const s3Agendas = romData?.stage3?.agendas || []
    if (s3Agendas.length === 0) return
    const naturalOrder = s3Agendas.map((a, idx) => a.agenda_id || `A${idx + 1}`)
    setDiscussionOrder(naturalOrder)

    const count = naturalOrder.length
    const slice = effectiveMeetingDuration / count
    const initialTimeline: Record<string, AgendaTimelineRange> = {}
    naturalOrder.forEach((id, idx) => {
      initialTimeline[id] = {
        start_sec: Math.round(idx * slice * 10) / 10,
        end_sec: Math.round((idx + 1) * slice * 10) / 10,
      }
    })
    setAgendaTimeline(initialTimeline)
    toast.info('Reset agenda sequence & timeline to natural order')
  }

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
  const [forceRegenerateAgenda, setForceRegenerateAgenda] = useState(false)
  const [showAgendaTextPreview, setShowAgendaTextPreview] = useState(false)
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
  // ROM Version (Short / Medium / Long)
  const [romVersion, setRomVersion] = useState<'short' | 'medium' | 'long'>('long')
  const DEFAULT_SHORT_PROMPT = "Process ALL points for this agenda together. Produce a SHORT version focused ONLY on action points, decisions, and commitments. Omit background discussions with no actionable outcome. Aim for 2-5 points. Preserve speaker attribution, action owners, dates, numbers. Return a valid JSON array of point strings: [\"point 1\", \"point 2\", ...]"
  const DEFAULT_MEDIUM_PROMPT = "Process ALL points for this agenda together. Produce a MEDIUM version: aggregated, retaining key discussion details alongside all action points and decisions. Merge related/redundant points. Aim for 4-8 points. Preserve speaker attribution, action owners, dates, numbers. Return a valid JSON array of point strings: [\"point 1\", \"point 2\", ...]"
  const [shortRomPrompt, setShortRomPrompt] = useState(DEFAULT_SHORT_PROMPT)
  const [mediumRomPrompt, setMediumRomPrompt] = useState(DEFAULT_MEDIUM_PROMPT)

  // Stage 3 Add Agenda state
  const [showAddAgendaForm, setShowAddAgendaForm] = useState(false)
  const [newAgendaTitle, setNewAgendaTitle] = useState('')
  const [newAgendaDesc, setNewAgendaDesc] = useState('')
  const [newAgendaPresenter, setNewAgendaPresenter] = useState('')
  const [creatingAgenda, setCreatingAgenda] = useState(false)

  // Save edited Stage 3 Agendas to backend
  const saveStage3Agendas = async (updatedAgendas: any[]) => {
    if (!id) return
    setSavingAgendas(true)
    try {
      const res = await api.put(`/rom/${id}/stage3/agendas`, { agendas: updatedAgendas })
      const fullData = res.data.rom_data || res.data
      if (fullData) {
        setRomData(fullData)
      } else {
        setRomData(prev => {
          if (!prev?.stage3) return prev
          return {
            ...prev,
            stage3: { ...prev.stage3, agendas: updatedAgendas }
          }
        })
      }
      toast.success('Agenda updated')
    } catch (e) {
      toast.error(getApiErrorDetail(e) || 'Failed to save agenda')
    } finally {
      setSavingAgendas(false)
    }
  }

  // Create a new Agenda item in Stage 3
  const handleCreateAgenda = async () => {
    if (!newAgendaTitle.trim() || !id) {
      toast.error('Please enter an agenda title')
      return
    }
    setCreatingAgenda(true)
    try {
      const res = await api.post(`/rom/${id}/stage3/agenda`, {
        title: newAgendaTitle.trim(),
        description: newAgendaDesc.trim(),
        presenter: newAgendaPresenter.trim() || undefined,
      })
      const fullData = res.data.rom_data || res.data
      if (fullData) {
        setRomData(fullData)
      } else if (res.data.agenda) {
        setRomData(prev => {
          if (!prev) return prev
          const s3Agendas = [...(prev.stage3?.agendas || []), res.data.agenda]
          return {
            ...prev,
            stage3: { ...(prev.stage3 || {}), agendas: s3Agendas }
          }
        })
      }
      setShowAddAgendaForm(false)
      setNewAgendaTitle('')
      setNewAgendaDesc('')
      setNewAgendaPresenter('')
      toast.success('Agenda item added successfully')
    } catch (e) {
      toast.error(getApiErrorDetail(e) || 'Failed to add agenda')
    } finally {
      setCreatingAgenda(false)
    }
  }

  // Delete an Agenda item from Stage 3
  const handleDeleteAgenda = async (agendaId: string, title: string) => {
    if (!id) return
    if (!window.confirm(`Are you sure you want to delete agenda "${title}" (${agendaId})? Any discussion points assigned to it will be reassigned to General Discussion.`)) {
      return
    }
    try {
      const res = await api.delete(`/rom/${id}/stage3/agenda/${agendaId}`)
      const fullData = res.data.rom_data || res.data
      if (fullData) {
        setRomData(fullData)
      } else {
        setRomData(prev => {
          if (!prev?.stage3) return prev
          return {
            ...prev,
            stage3: {
              ...prev.stage3,
              agendas: prev.stage3.agendas.filter((a: any) => a.agenda_id !== agendaId)
            }
          }
        })
      }
      toast.success(`Agenda ${agendaId} deleted`)
    } catch (e) {
      toast.error(getApiErrorDetail(e) || 'Failed to delete agenda')
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
      setForceRegenerateAgenda(true)
      setShowAgendaTextPreview(true)
    }
    setUploadingAgendaFile(false)
    if (agendaFileInputRef.current) agendaFileInputRef.current.value = ''
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
        enable_discussion_order: enableAgendaOrder,
        enable_agenda_timeline: enableAgendaTimeline,
        discussion_order: enableAgendaOrder && discussionOrder.length > 0 ? discussionOrder : undefined,
        agenda_timeline: enableAgendaTimeline && Object.keys(agendaTimeline).length > 0 ? agendaTimeline : undefined,
        // Build skipped_agendas map: { agendaId: { note: "..." } } for agendas that are marked as skipped
        skipped_agendas: (() => {
          const skipped: Record<string, { note: string }> = {}
          for (const [aid, info] of Object.entries(skippedAgendas)) {
            if (info.skipped) {
              skipped[aid] = { note: info.note }
            }
          }
          return Object.keys(skipped).length > 0 ? skipped : undefined
        })(),
        include_action_points: includeActionPointsInRom,
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
    const currentRom = (
      romVersion === 'long'
        ? (romData?.final_rom_versions?.long || originalFinalRom || romData?.final_rom)
        : (romData?.final_rom_versions?.[romVersion] || romData?.final_rom)
    )
    if (!currentRom) return
    const newFinal = JSON.parse(JSON.stringify(currentRom))
    if (!newFinal.agendas) newFinal.agendas = []

    // Ensure all agendas from stage3 exist in newFinal
    if (romData?.stage3?.agendas) {
      const existingIds = new Set(newFinal.agendas.map((a: any) => a.agenda_id))
      for (const s3a of romData.stage3.agendas) {
        if (!existingIds.has(s3a.agenda_id)) {
          newFinal.agendas.push({ ...s3a, discussion_points: [] })
        }
      }
    }

    const fromAgenda = newFinal.agendas[fromAgendaIdx]
    const toAgenda = newFinal.agendas.find((a: any) => a.agenda_id === toAgendaId)
    if (fromAgenda && toAgenda && fromAgenda.discussion_points) {
      const point = fromAgenda.discussion_points.splice(pointIdx, 1)[0]
      if (point) {
        if (!toAgenda.discussion_points) toAgenda.discussion_points = []
        toAgenda.discussion_points.push(point)
        setRomData(prev => {
          if (!prev) return prev
          return {
            ...prev,
            final_rom: newFinal,
            final_rom_versions: {
              ...(prev.final_rom_versions || {}),
              [romVersion]: newFinal,
            }
          }
        })
        toast.success(`Point moved to ${toAgenda.title || toAgendaId}`)
      }
    }
  }

  // Reorder a discussion point within an agenda
  const reorderPoint = (agendaIdx: number, pointIdx: number, direction: 'up' | 'down') => {
    const currentRom = (
      romVersion === 'long'
        ? (romData?.final_rom_versions?.long || originalFinalRom || romData?.final_rom)
        : (romData?.final_rom_versions?.[romVersion] || romData?.final_rom)
    )
    if (!currentRom?.agendas) return
    const newFinal = JSON.parse(JSON.stringify(currentRom))
    const pts = newFinal.agendas[agendaIdx]?.discussion_points
    if (!pts) return
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

  const fetchPreviousMeetings = useCallback(async () => {
    if (!id) return
    setLoadingPreviousMeetings(true)
    try {
      const res = await api.get(`/rom/${id}/previous-meetings`)
      if (Array.isArray(res.data?.meetings)) {
        setAvailablePreviousMeetings(res.data.meetings)
        if (!selectedPreviousMeetingId) {
          const firstWithStage2 = res.data.meetings.find((m: any) => m.has_stage2)
          if (firstWithStage2) {
            setSelectedPreviousMeetingId(firstWithStage2.id)
          } else if (res.data.meetings.length > 0) {
            setSelectedPreviousMeetingId(res.data.meetings[0].id)
          }
        }
      }
    } catch {
      // ignore
    } finally {
      setLoadingPreviousMeetings(false)
    }
  }, [id, selectedPreviousMeetingId])

  useEffect(() => {
    if (id) {
      loadData()
      fetchPreviousMeetings()
    }
    api.get('/rom/stage2/example-points')
      .then(res => {
        if (Array.isArray(res.data?.points)) {
          setStage2ReferenceExamples(res.data.points)
        }
      })
      .catch(() => {})
    api.get('/settings')
      .then(res => {
        if (res.data) {
          if (typeof res.data.rom_stage2_process_all_together === 'boolean') {
            setProcessAllTogether(res.data.rom_stage2_process_all_together)
          }
          if (typeof res.data.rom_meeting_top_k === 'number') {
            setMeetingTopK(res.data.rom_meeting_top_k)
          }
          if (typeof res.data.rom_global_top_k === 'number') {
            setGlobalTopK(res.data.rom_global_top_k)
          }
          if (typeof res.data.rom_windows_per_batch === 'number') {
            setDiscussionWindowSize(res.data.rom_windows_per_batch)
          }
          if (typeof res.data.rom_min_similarity_threshold === 'number') {
            setMinSimilarityThreshold(res.data.rom_min_similarity_threshold)
          }
        }
      })
      .catch(() => {})
  }, [id, loadData, fetchPreviousMeetings])

  // ── Actions ─────────────────────────────────────────────────────────────────

  const runStage1 = async (): Promise<boolean> => {
    setStage1Status('processing')
    setStage1Progress(null)

    // Start polling progress every 2 seconds
    if (stage1PollRef.current) clearInterval(stage1PollRef.current)
    stage1PollRef.current = setInterval(async () => {
      try {
        const prog = await api.get(`/rom/${id}/stage1/progress`)
        if (prog.data && prog.data.windows_total > 0) {
          setStage1Progress(prog.data)
        }
        if (prog.data?.status === 'done') {
          if (stage1PollRef.current) clearInterval(stage1PollRef.current)
          stage1PollRef.current = null
        }
      } catch {
        // Ignore transient poll errors
      }
    }, 2000)

    try {
      const res = await api.post(`/rom/${id}/stage1/generate`, {
        transcript_window_minutes: transcriptWindow
      })
      // Stop polling on success
      if (stage1PollRef.current) clearInterval(stage1PollRef.current)
      stage1PollRef.current = null
      const fullData = res.data.rom_data || {
        ...(romData || {} as RomData),
        stage1: res.data.stage1 || res.data
      }
      setRomData(fullData)
      setStage1Status('done')
      setStage1Progress(null)
      setActiveTab('stage1')
      const pointCount = fullData.stage1?.discussion_points?.length ?? 0
      toast.success(`Stage 1 complete: ${pointCount} points extracted`)
      return true
    } catch (e) {
      // Stop polling on failure
      if (stage1PollRef.current) clearInterval(stage1PollRef.current)
      stage1PollRef.current = null
      setStage1Status('error')
      setStage1Progress(null)
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
        min_similarity_threshold: minSimilarityThreshold,
        process_all_together: processAllTogether,
        reference_example_points: stage2ReferenceExamples.filter(p => p && p.trim()),
        previous_meeting_mode: previousMeetingMode,
        previous_meeting_id: previousMeetingMode === 'select' ? (selectedPreviousMeetingId || null) : null,
        previous_meeting_top_k: previousMeetingMode !== 'off' ? Math.max(0, Number(previousMeetingTopK) || 0) : 0,
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

  const runPreviewStage2Context = async () => {
    if (!id) return
    setPreviewingContext(true)
    try {
      const res = await api.post(`/rom/${id}/stage2/preview-context`, {
        meeting_context_top_k: Math.max(0, Number(meetingTopK) || 0),
        global_context_top_k: Math.max(0, Number(globalTopK) || 0),
        discussion_window_size: Math.max(3, Number(discussionWindowSize) || 5),
        min_similarity_threshold: minSimilarityThreshold,
        process_all_together: processAllTogether,
        reference_example_points: stage2ReferenceExamples.filter(p => p && p.trim()),
        previous_meeting_mode: previousMeetingMode,
        previous_meeting_id: previousMeetingMode === 'select' ? (selectedPreviousMeetingId || null) : null,
        previous_meeting_top_k: previousMeetingMode !== 'off' ? Math.max(0, Number(previousMeetingTopK) || 0) : 0,
      })
      setContextPreviewGroups(res.data.groups || [])
      setContextPreviewTotalGroups(res.data.total_groups || res.data.groups?.length || 0)
      setContextPreviewTotalPoints(res.data.total_points || 0)
      setShowContextPreview(true)
      const shownCount = res.data.groups?.length || 0
      const totalCount = res.data.total_groups || shownCount
      if (totalCount > shownCount) {
        toast.success(`Retrieved context preview for initial ${shownCount} of ${totalCount} group(s)`)
      } else {
        toast.success(`Retrieved context preview for ${shownCount} group(s)`)
      }
    } catch (e) {
      toast.error(getApiErrorDetail(e) || 'Failed to retrieve context preview')
    } finally {
      setPreviewingContext(false)
    }
  }

  const handleSaveStage2Examples = async () => {
    setSavingStage2Examples(true)
    try {
      await api.post('/rom/stage2/example-points', {
        points: stage2ReferenceExamples.filter(p => p.trim())
      })
      toast.success('Reference style examples saved')
    } catch {
      toast.error('Failed to save example points')
    } finally {
      setSavingStage2Examples(false)
    }
  }

  // ── Stage 2 Editing Handlers ─────────────────────────────────────

  const togglePointSelection = (pointId: string) => {
    setSelectedPointIds(prev => {
      const next = new Set(prev)
      if (next.has(pointId)) next.delete(pointId)
      else next.add(pointId)
      return next
    })
  }

  const handleMergePoints = async () => {
    if (selectedPointIds.size < 2 || !id) return
    setMergingPoints(true)
    try {
      const res = await api.post(`/rom/${id}/stage2/merge-points`, {
        point_ids: Array.from(selectedPointIds),
      })
      setRomData(res.data.rom_data)
      setSelectedPointIds(new Set())
      toast.success('Points merged successfully')
    } catch (e: any) {
      toast.error(getApiErrorDetail(e) || 'Merge failed')
    } finally {
      setMergingPoints(false)
    }
  }

  const handleFindPreview = async () => {
    if (!findText.trim() || !id) return
    try {
      const res = await api.post(`/rom/${id}/stage2/find-replace/preview`, { find_text: findText })
      setFindPreviewCount(res.data.total_occurrences)
      setFindPreviewPoints(res.data.points_affected || [])
    } catch {
      setFindPreviewCount(null)
    }
  }

  const handleFindReplace = async () => {
    if (!findText.trim() || !id) return
    setApplyingFindReplace(true)
    try {
      const res = await api.post(`/rom/${id}/stage2/find-replace`, {
        find_text: findText,
        replace_text: replaceText,
      })
      if (res.data.rom_data) setRomData(res.data.rom_data)
      setShowFindReplace(false)
      setFindText('')
      setReplaceText('')
      setFindPreviewCount(null)
      setFindPreviewPoints([])
      toast.success(`Replaced ${res.data.affected_count} occurrences`)
    } catch (e: any) {
      toast.error(getApiErrorDetail(e) || 'Find & Replace failed')
    } finally {
      setApplyingFindReplace(false)
    }
  }

  const handleContextMenuAction = async (action: 'split' | 'delete_text') => {
    if (!contextMenu || !id) return
    const { pointId, selectedText } = contextMenu
    setContextMenu(null)

    if (action === 'split') {
      setSplittingPoint(true)
      try {
        const res = await api.post(`/rom/${id}/stage2/split-point`, {
          point_id: pointId,
          selected_text: selectedText,
        })
        if (res.data.rom_data) setRomData(res.data.rom_data)
        toast.success('Point split into two')
      } catch (e: any) {
        toast.error(getApiErrorDetail(e) || 'Split failed')
      } finally {
        setSplittingPoint(false)
      }
    } else if (action === 'delete_text') {
      setDeletingText(true)
      try {
        const res = await api.post(`/rom/${id}/stage2/delete-text`, {
          point_id: pointId,
          text_to_delete: selectedText,
        })
        if (res.data.rom_data) setRomData(res.data.rom_data)
        toast.success('Text deleted')
      } catch (e: any) {
        toast.error(getApiErrorDetail(e) || 'Delete failed')
      } finally {
        setDeletingText(false)
      }
    }
  }

  const loadChangeHistory = async () => {
    if (!id) return
    setLoadingHistory(true)
    try {
      const res = await api.get(`/rom/${id}/stage2/edit-history`)
      setChangeHistory(res.data.changes || [])
    } catch {
      toast.error('Failed to load history')
    } finally {
      setLoadingHistory(false)
    }
  }

  const handleRevertChange = async (changeId: string) => {
    if (!id) return
    setRevertingChangeId(changeId)
    try {
      const res = await api.post(`/rom/${id}/stage2/edit-history/${changeId}/revert`)
      if (res.data.rom_data) setRomData(res.data.rom_data)
      // Refresh history
      await loadChangeHistory()
      toast.success('Change reverted')
    } catch (e: any) {
      toast.error(getApiErrorDetail(e) || 'Revert failed')
    } finally {
      setRevertingChangeId(null)
    }
  }

  const handleRedoChange = async (changeId: string) => {
    if (!id) return
    setRedoingChangeId(changeId)
    try {
      const res = await api.post(`/rom/${id}/stage2/edit-history/${changeId}/redo`)
      if (res.data.rom_data) setRomData(res.data.rom_data)
      await loadChangeHistory()
      toast.success('Change re-applied (Redo)')
    } catch (e: any) {
      toast.error(getApiErrorDetail(e) || 'Redo failed')
    } finally {
      setRedoingChangeId(null)
    }
  }

  const handleSaveStage2Point = async (pointId: string) => {
    if (!id || !stage2EditDraft.trim()) return
    setSavingStage2Point(true)
    try {
      const res = await api.post(`/rom/${id}/stage2/point/${pointId}/manual-edit`, {
        polished_text: stage2EditDraft.trim(),
      })
      if (res.data.rom_data) setRomData(res.data.rom_data)
      setEditingStage2PointId(null)
      setStage2EditDraft('')
      toast.success('Discussion point updated')
    } catch (e: any) {
      toast.error(getApiErrorDetail(e) || 'Failed to update point')
    } finally {
      setSavingStage2Point(false)
    }
  }

  const handlePointContextMenu = (e: React.MouseEvent, pointId: string) => {
    const selection = window.getSelection()
    const selectedText = selection?.toString()?.trim() || ''
    if (!selectedText) return
    e.preventDefault()
    setContextMenu({ x: e.clientX, y: e.clientY, pointId, selectedText })
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

  const [generatingVersion, setGeneratingVersion] = useState(false)

  const downloadDocx = async (stage: string) => {
    try {
      const queryParam = stage === 'final'
        ? `?version=${romVersion}&include_action_points=${includeActionPointsInRom}`
        : ''
      const res = await api.get(`/rom/${id}/${stage}/download/docx${queryParam}`, { responseType: 'blob' })
      const url = URL.createObjectURL(new Blob([res.data]))
      const a = document.createElement('a')
      a.href = url
      a.download = stage === 'final' ? `rom_final_${romVersion}_${id}.docx` : `rom_${stage}_${id}.docx`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
      toast.success(`ROM ${stage.toUpperCase()} (${romVersion}) DOCX downloaded`)
    } catch {
      toast.error('Download failed')
    }
  }

  const saveFinalRom = async (updatedFinalRom: any) => {
    if (!id || !updatedFinalRom) return
    setSavingRom(true)
    try {
      const res = await api.put(`/rom/${id}/final`, {
        final_rom: updatedFinalRom,
        version: romVersion
      })
      const serverVersions = res.data?.final_rom_versions
      setRomData(prev => {
        if (!prev) return prev
        const existingVersions = prev.final_rom_versions || {}
        return {
          ...prev,
          final_rom: updatedFinalRom,
          final_rom_versions: serverVersions || {
            ...existingVersions,
            [romVersion]: updatedFinalRom,
          },
          final_rom_active_version: romVersion
        }
      })
      toast.success(`${romVersion.toUpperCase()} ROM saved to database`)
    } catch (e) {
      toast.error(getApiErrorDetail(e) || 'Save failed')
    } finally {
      setSavingRom(false)
    }
  }

  // Final ROM – Toggle Include Action Points handler
  const handleToggleActionPoints = async (enabled: boolean) => {
    setIncludeActionPointsInRom(enabled)
    if (!id) return
    setTogglingActionPoints(true)
    try {
      const res = await api.post(`/rom/${id}/final/action-points`, {
        include_action_points: enabled,
      })
      if (res.data?.rom_data) {
        setRomData(res.data.rom_data)
      } else if (res.data?.final_rom) {
        setRomData(prev => prev ? { ...prev, final_rom: res.data.final_rom } : prev)
      }
      toast.success(enabled ? 'Action points enabled in Final ROM' : 'Action points disabled in Final ROM')
    } catch (e) {
      toast.error(getApiErrorDetail(e) || 'Failed to update Action Points setting')
    } finally {
      setTogglingActionPoints(false)
    }
  }

  // Generate a Short or Medium version of the Final ROM
  const runGenerateRomVersion = async (targetVersion: 'short' | 'medium') => {
    if (!id || !romData?.final_rom) return
    setGeneratingVersion(true)
    // Base is always the Long / Stage 2 points
    const baseRom = romData.final_rom_versions?.long || originalFinalRom || JSON.parse(JSON.stringify(romData.final_rom))
    if (!originalFinalRom) {
      setOriginalFinalRom(baseRom)
    }

    try {
      const res = await api.post(`/rom/${id}/final/generate-version`, {
        version: targetVersion,
        writing_rules: writingRules || '',
        base_final_rom: baseRom,
      })
      const rewrittenFinalRom = res.data.rewritten_final_rom || res.data.final_rom
      if (rewrittenFinalRom && rewrittenFinalRom.agendas?.length) {
        const updatedVersions = res.data.final_rom_versions || {
          ...(romData.final_rom_versions || {}),
          long: baseRom,
          [targetVersion]: rewrittenFinalRom,
        }
        setRomData(prev => ({
          ...(prev as RomData),
          final_rom: rewrittenFinalRom,
          final_rom_versions: updatedVersions,
          final_rom_active_version: targetVersion,
        }))
        setRomVersion(targetVersion)
        const versionLabel = targetVersion === 'short' ? 'Short' : 'Medium'
        toast.success(`${versionLabel} ROM generated and saved successfully!`)
      } else {
        throw new Error('Server returned empty or invalid ROM points')
      }
    } catch (e) {
      toast.error(getApiErrorDetail(e) || `Failed to generate ${targetVersion} ROM`)
    } finally {
      setGeneratingVersion(false)
    }
  }

  // Switch between Long, Short, and Medium without overwriting
  const handleSelectVersion = async (targetVer: 'long' | 'short' | 'medium') => {
    setRomVersion(targetVer)
    const targetRom = targetVer === 'long'
      ? (romData?.final_rom_versions?.long || originalFinalRom || romData?.final_rom)
      : romData?.final_rom_versions?.[targetVer]

    if (targetRom && id) {
      try {
        await api.post(`/rom/${id}/final/select-version`, { version: targetVer })
        setRomData(prev => prev ? ({
          ...prev,
          final_rom: targetRom,
          final_rom_active_version: targetVer
        }) : prev)
      } catch {
        // Non-fatal, local UI already updated
      }
    }
  }

  // Style Rewrite ROM using LLM
  const runRewriteRom = async () => {
    if (!id || !romData?.final_rom) return
    setRewriteStatus('processing')
    const currentRom = (
      romVersion === 'long'
        ? (romData.final_rom_versions?.long || originalFinalRom || romData.final_rom)
        : (romData.final_rom_versions?.[romVersion] || romData.final_rom)
    )
    if (!originalFinalRom) {
      setOriginalFinalRom(JSON.parse(JSON.stringify(currentRom)))
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
        setRomData(prev => {
          if (!prev) return prev
          return {
            ...prev,
            final_rom: rewrittenFinalRom,
            final_rom_versions: {
              ...(prev.final_rom_versions || {}),
              [romVersion]: rewrittenFinalRom,
            },
          }
        })
        setIsRewritten(true)
        setRewriteStatus('done')
        toast.success(`Rewritten ${romVersion.toUpperCase()} ROM ready. Review and click Save Changes to persist.`)
      } else {
        throw new Error('No rewritten ROM returned from server')
      }
    } catch (e) {
      setRewriteStatus('error')
      toast.error(getApiErrorDetail(e) || 'Rewrite failed')
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

            {/* Minimum Similarity Threshold Slider */}
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 2 }}>
                <label style={{ fontSize: '.68rem', color: 'hsl(var(--pencil))' }}>Min Similarity</label>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <label style={{ fontSize: '.64rem', color: 'hsl(var(--pencil))', cursor: 'pointer', display: 'inline-flex', alignItems: 'center', gap: 3 }}>
                    <input
                      type="checkbox"
                      checked={minSimilarityThreshold !== null}
                      onChange={e => setMinSimilarityThreshold(e.target.checked ? 0.80 : null)}
                      style={{ width: 11, height: 11, accentColor: 'hsl(205,90%,55%)' }}
                    />
                    Filter
                  </label>
                  <span style={{ fontSize: '.68rem', fontWeight: 700, color: minSimilarityThreshold !== null ? 'hsl(205,90%,55%)' : 'hsl(var(--pencil))', fontFamily: 'JetBrains Mono' }}>
                    {minSimilarityThreshold !== null ? minSimilarityThreshold.toFixed(2) : 'Off'}
                  </span>
                </div>
              </div>
              {minSimilarityThreshold !== null && (
                <input
                  type="range" min={0.50} max={0.98} step={0.01}
                  value={minSimilarityThreshold}
                  onChange={e => setMinSimilarityThreshold(Number(e.target.value))}
                  style={{ width: '100%', accentColor: 'hsl(205,90%,55%)' }}
                />
              )}
            </div>

            {/* Process All Points Together Toggle */}
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '2px 0' }}>
              <label style={{ fontSize: '.68rem', color: 'hsl(var(--pencil))', cursor: 'pointer', display: 'inline-flex', alignItems: 'center', gap: 5 }}>
                <input
                  type="checkbox"
                  checked={processAllTogether}
                  onChange={e => setProcessAllTogether(e.target.checked)}
                  style={{ width: 12, height: 12, accentColor: 'hsl(205,90%,55%)' }}
                />
                Process All Points Together
              </label>
            </div>

            {/* Previous Meeting Context (Dual Mode) */}
            <div style={{ padding: '.45rem .5rem', borderRadius: 8, background: 'hsl(270,75%,55%/.06)', border: '1px solid hsl(270,75%,55%/.22)', display: 'flex', flexDirection: 'column', gap: '.35rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div style={{ fontSize: '.66rem', fontWeight: 700, color: 'hsl(270,75%,60%)', textTransform: 'uppercase', letterSpacing: '.04em', display: 'flex', alignItems: 'center', gap: 4 }}>
                  ⏮️ Previous Meeting Context
                </div>
                {previousMeetingMode !== 'off' && (
                  <span style={{ fontSize: '.64rem', fontWeight: 700, color: 'hsl(270,75%,60%)', fontFamily: 'JetBrains Mono' }}>
                    K={previousMeetingTopK}
                  </span>
                )}
              </div>

              {/* Mode Pills */}
              <div style={{ display: 'flex', gap: 3, background: 'hsl(var(--muted)/.4)', padding: 2, borderRadius: 6 }}>
                <button
                  type="button"
                  onClick={() => setPreviousMeetingMode('auto')}
                  style={{
                    flex: 1, padding: '3px 0', border: 'none', borderRadius: 4, fontSize: '.64rem', fontWeight: previousMeetingMode === 'auto' ? 700 : 500,
                    background: previousMeetingMode === 'auto' ? 'hsl(270,75%,60%)' : 'transparent',
                    color: previousMeetingMode === 'auto' ? 'white' : 'hsl(var(--pencil))',
                    cursor: 'pointer', transition: 'all .15s ease'
                  }}
                  title="Auto-retrieve most similar Stage 2 points from all previous meetings via ChromaDB"
                >
                  Auto Retrieve
                </button>
                <button
                  type="button"
                  onClick={() => setPreviousMeetingMode('select')}
                  style={{
                    flex: 1, padding: '3px 0', border: 'none', borderRadius: 4, fontSize: '.64rem', fontWeight: previousMeetingMode === 'select' ? 700 : 500,
                    background: previousMeetingMode === 'select' ? 'hsl(270,75%,60%)' : 'transparent',
                    color: previousMeetingMode === 'select' ? 'white' : 'hsl(var(--pencil))',
                    cursor: 'pointer', transition: 'all .15s ease'
                  }}
                  title="Select a specific past meeting to use its Stage 2 points only"
                >
                  Select Meeting
                </button>
                <button
                  type="button"
                  onClick={() => setPreviousMeetingMode('off')}
                  style={{
                    flex: 0.6, padding: '3px 0', border: 'none', borderRadius: 4, fontSize: '.64rem', fontWeight: previousMeetingMode === 'off' ? 700 : 500,
                    background: previousMeetingMode === 'off' ? 'hsl(var(--muted))' : 'transparent',
                    color: previousMeetingMode === 'off' ? 'hsl(var(--ink))' : 'hsl(var(--pencil))',
                    cursor: 'pointer', transition: 'all .15s ease'
                  }}
                >
                  Off
                </button>
              </div>

              {/* Select Meeting Dropdown */}
              {previousMeetingMode === 'select' && (
                <div style={{ marginTop: 2 }}>
                  <label style={{ fontSize: '.64rem', color: 'hsl(var(--pencil))', display: 'block', marginBottom: 2 }}>Target Meeting:</label>
                  {loadingPreviousMeetings ? (
                    <div style={{ fontSize: '.64rem', color: 'hsl(var(--pencil))' }}>Loading meetings...</div>
                  ) : availablePreviousMeetings.length === 0 ? (
                    <div style={{ fontSize: '.64rem', color: 'hsl(var(--destructive))' }}>No other meetings found.</div>
                  ) : (
                    <select
                      value={selectedPreviousMeetingId}
                      onChange={e => setSelectedPreviousMeetingId(e.target.value)}
                      style={{
                        width: '100%', fontSize: '.68rem', padding: '3px 5px', borderRadius: 5,
                        border: '1px solid hsl(270,75%,55%/.3)', background: 'hsl(var(--card))',
                        color: 'hsl(var(--ink))', outline: 'none'
                      }}
                    >
                      {availablePreviousMeetings.map(m => (
                        <option key={m.id} value={m.id}>
                          {m.name} {m.date ? `(${m.date})` : ''} {m.has_stage2 ? `[${m.stage2_count} Stage 2 pts]` : '[No Stage 2 pts]'}
                        </option>
                      ))}
                    </select>
                  )}
                </div>
              )}

              {/* Previous Meeting Top-K slider */}
              {previousMeetingMode !== 'off' && (
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 1 }}>
                    <label style={{ fontSize: '.64rem', color: 'hsl(var(--pencil))' }}>Stage 2 Points (Top-K)</label>
                    <span style={{ fontSize: '.64rem', fontWeight: 700, color: 'hsl(270,75%,60%)', fontFamily: 'JetBrains Mono' }}>
                      {previousMeetingTopK === 0 ? '0 (No Context)' : previousMeetingTopK}
                    </span>
                  </div>
                  <input
                    type="range" min={0} max={10} value={previousMeetingTopK}
                    onChange={e => setPreviousMeetingTopK(Number(e.target.value))}
                    style={{ width: '100%', accentColor: 'hsl(270,75%,60%)' }}
                  />
                </div>
              )}
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

            <div style={{ display: 'flex', gap: 5 }}>
              <button
                type="button"
                onClick={runPreviewStage2Context}
                disabled={previewingContext || stage1Status !== 'done'}
                style={{
                  flex: 1, padding: '4px 6px', borderRadius: 7, fontSize: '.71rem', fontWeight: 600,
                  border: '1px solid hsl(205,90%,55%/.4)', background: 'hsl(205,90%,55%/.08)', color: 'hsl(205,90%,45%)',
                  cursor: (previewingContext || stage1Status !== 'done') ? 'not-allowed' : 'pointer',
                  display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 4, fontFamily: 'Inter'
                }}
                title="Preview retrieved context for all point groups without modifying points"
              >
                {previewingContext ? <Loader size={10} className="spin" /> : <BookOpen size={10} />}
                {previewingContext ? 'Retrieving...' : 'Retrieve Context'}
              </button>

              <button
                type="button"
                onClick={runStage2}
                disabled={stage2Status === 'processing' || stage1Status !== 'done'}
                style={{
                  flex: 1, padding: '4px 6px', borderRadius: 7, fontSize: '.71rem', fontWeight: 600,
                  background: (stage2Status === 'processing' || stage1Status !== 'done') ? 'hsl(var(--muted))' : 'hsl(205,90%,55%)',
                  color: 'white', border: 'none', cursor: (stage2Status === 'processing' || stage1Status !== 'done') ? 'not-allowed' : 'pointer',
                  display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 4, fontFamily: 'Inter'
                }}
              >
                {stage2Status === 'processing' ? <Loader size={10} className="spin" /> : <Sparkles size={10} />}
                {stage2Status === 'processing' ? 'Enhancing...' : 'Enhance Points'}
              </button>
            </div>
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

            {/* Extracted Agenda Text Preview / Editor */}
            {agendaText && (
              <div style={{ borderRadius: 6, border: '1px solid hsl(var(--border)/.5)', background: 'hsl(var(--muted)/.15)', overflow: 'hidden' }}>
                <button
                  onClick={() => setShowAgendaTextPreview(prev => !prev)}
                  type="button"
                  style={{ width: '100%', padding: '4px 8px', background: 'transparent', border: 'none', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '.66rem', fontWeight: 600, color: 'hsl(var(--pencil))' }}
                >
                  <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                    <FileText size={9} style={{ color: 'hsl(140,70%,45%)' }} />
                    Agenda Text ({agendaText.length} chars)
                  </span>
                  <span style={{ fontSize: '.62rem', color: 'hsl(140,70%,45%)' }}>{showAgendaTextPreview ? 'Hide' : 'View / Edit'}</span>
                </button>
                {showAgendaTextPreview && (
                  <div style={{ padding: '0 6px 6px 6px' }}>
                    <textarea
                      value={agendaText}
                      onChange={e => setAgendaText(e.target.value)}
                      placeholder="Extracted agenda text..."
                      rows={4}
                      style={{ width: '100%', fontSize: '.68rem', fontFamily: 'JetBrains Mono, monospace', padding: '4px 6px', borderRadius: 4, border: '1px solid hsl(var(--border)/.6)', background: 'hsl(var(--paper))', color: 'hsl(var(--ink))', resize: 'vertical' }}
                    />
                  </div>
                )}
              </div>
            )}

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

            {/* Regenerate / Bypass Cache toggle */}
            <label style={{ display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer', fontSize: '.7rem', color: 'hsl(var(--ink))', padding: '.2rem 0', userSelect: 'none' }}>
              <input
                type="checkbox"
                checked={forceRegenerateAgenda}
                onChange={e => setForceRegenerateAgenda(e.target.checked)}
                style={{ accentColor: 'hsl(140,70%,45%)', width: 12, height: 12, cursor: 'pointer' }}
              />
              <span style={{ fontWeight: forceRegenerateAgenda ? 700 : 500, color: forceRegenerateAgenda ? 'hsl(140,70%,40%)' : 'hsl(var(--pencil))' }}>
                Regenerate (Bypass Cache)
              </span>
            </label>

            <button
              onClick={() => runCreateAgenda(forceRegenerateAgenda)}
              disabled={stage3AgendaStatus === 'processing' || stage2Status !== 'done'}
              style={{
                width: '100%', padding: '5px 8px', borderRadius: 7, fontSize: '.74rem', fontWeight: 700,
                background: (stage3AgendaStatus === 'processing' || stage2Status !== 'done') ? 'hsl(var(--muted))' : (forceRegenerateAgenda || (romData?.stage3?.agendas?.length ?? 0) > 0 ? 'hsl(140,70%,40%)' : 'hsl(140,70%,45%)'),
                color: 'white', border: 'none', cursor: (stage3AgendaStatus === 'processing' || stage2Status !== 'done') ? 'not-allowed' : 'pointer',
                display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 5, fontFamily: 'Inter'
              }}
            >
              {stage3AgendaStatus === 'processing' ? <Loader size={11} className="spin" /> : (forceRegenerateAgenda || (romData?.stage3?.agendas?.length ?? 0) > 0 ? <RefreshCw size={11} /> : <List size={11} />)}
              {stage3AgendaStatus === 'processing'
                ? 'Creating Agendas...'
                : ((romData?.stage3?.agendas?.length ?? 0) > 0 && forceRegenerateAgenda ? 'Regenerate Agendas' : (romData?.stage3?.agendas?.length ?? 0) > 0 ? 'Update Agendas' : 'Create Agenda')}
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

            {/* Discussion Order & Timeline Guidance Controls */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6, padding: '.45rem .6rem', borderRadius: 7, background: 'hsl(var(--muted)/.25)', border: '1px solid hsl(var(--border)/.5)' }}>
              <label style={{ display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer', fontSize: '.71rem', color: 'hsl(var(--ink))', userSelect: 'none' }}>
                <input
                  type="checkbox"
                  checked={enableAgendaOrder}
                  onChange={e => setEnableAgendaOrder(e.target.checked)}
                  style={{ accentColor: 'hsl(140,70%,45%)', width: 13, height: 13 }}
                />
                <span style={{ fontWeight: 600 }}>Include Discussion Order in LLM</span>
              </label>

              <label style={{ display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer', fontSize: '.71rem', color: 'hsl(var(--ink))', userSelect: 'none' }}>
                <input
                  type="checkbox"
                  checked={enableAgendaTimeline}
                  onChange={e => setEnableAgendaTimeline(e.target.checked)}
                  style={{ accentColor: 'hsl(140,70%,45%)', width: 13, height: 13 }}
                />
                <span style={{ fontWeight: 600 }}>Include Timeline Windows in LLM</span>
              </label>

              {discussionOrder.length > 0 && enableAgendaOrder && (
                <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '.62rem', color: 'hsl(var(--pencil))', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', paddingTop: 2, borderTop: '1px solid hsl(var(--border)/.3)' }}>
                  {discussionOrder.join(' → ')}
                </div>
              )}
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

            {/* Include Action Points in Final ROM checkbox */}
            <label style={{ display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer', fontSize: '.72rem', color: 'hsl(var(--ink))', padding: '.3rem .4rem', borderRadius: 6, background: includeActionPointsInRom ? 'hsl(30,90%,55%/.08)' : 'transparent', border: `1px solid ${includeActionPointsInRom ? 'hsl(30,90%,55%/.3)' : 'hsl(var(--border)/.3)'}`, transition: 'all .15s' }}>
              <input
                type="checkbox"
                checked={includeActionPointsInRom}
                onChange={e => setIncludeActionPointsInRom(e.target.checked)}
                style={{ accentColor: 'hsl(30,90%,55%)', width: 12, height: 12 }}
              />
              <span style={{ fontWeight: 600 }}>Include Action Points</span>
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
            ].map(tab => {
              const isActive = activeTab === tab.id
              const hasWarning =
                (tab.id === 'stage2' && !!romData?.outdated_warnings?.stage2) ||
                (tab.id === 'stage3' && !!romData?.outdated_warnings?.stage3) ||
                (tab.id === 'final' && !!romData?.outdated_warnings?.final_rom)
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
                  {hasWarning && (
                    <span title="Previous stage data changed. This stage may be outdated." style={{ display: 'inline-flex', alignItems: 'center' }}>
                      <AlertTriangle size={13} style={{ color: 'hsl(38,90%,48%)' }} />
                    </span>
                  )}
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
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                  {/* Stage 1 Progress Banner — shown while processing */}
                  {stage1Status === 'processing' && stage1Progress && stage1Progress.windows_total > 0 && (
                    <Stage1ProgressBanner progress={stage1Progress} />
                  )}
                  <div style={{ textAlign: 'center', padding: stage1Status === 'processing' ? '2rem 1.5rem' : '4rem 1.5rem', color: 'hsl(var(--pencil))' }}>
                    <Brain size={42} style={{ margin: '0 auto 1rem', opacity: 0.4, color: 'hsl(280,75%,65%)' }} />
                    <div style={{ fontSize: '1.05rem', fontWeight: 700, color: 'hsl(var(--ink))', marginBottom: '.4rem' }}>
                      {stage1Status === 'processing' ? 'Extracting Discussion Points...' : 'No Stage 1 Points Extracted Yet'}
                    </div>
                    <div style={{ fontSize: '.84rem', maxWidth: 420, margin: '0 auto 1.25rem', lineHeight: 1.45 }}>
                      {stage1Status === 'processing'
                        ? 'Processing transcript windows in parallel. Results will appear when complete.'
                        : <>Click <strong>Generate Points</strong> on the left panel to execute sliding window transcript extraction.</>}
                    </div>
                    {stage1Status !== 'processing' && (
                      <button onClick={runStage1} disabled={stage1Status === 'processing'} className="btn btn-primary" style={{ fontSize: '.8rem', padding: '.45rem 1rem' }}>
                        <Sparkles size={14} /> Run Stage 1 Extraction
                      </button>
                    )}
                  </div>
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                  {/* Progress banner also shown on re-runs when data already exists */}
                  {stage1Status === 'processing' && stage1Progress && stage1Progress.windows_total > 0 && (
                    <Stage1ProgressBanner progress={stage1Progress} />
                  )}
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
                                    whiteSpace: 'nowrap', marginBottom: '.45rem'
                                  }}>
                                    <Clock size={10} /> {fmtTime(group.timelineStart)} – {fmtTime(group.timelineEnd)}
                                  </span>
                                  <div>
                                    <button
                                      onClick={() => setRerunModalWindow(group)}
                                      title="Re-run extraction for this window with custom feedback"
                                      style={{
                                        display: 'inline-flex', alignItems: 'center', gap: 4,
                                        padding: '3px 8px', borderRadius: 6,
                                        border: '1px solid hsl(280,75%,60%/.35)',
                                        background: 'hsl(280,75%,60%/.1)',
                                        color: 'hsl(280,75%,60%)',
                                        fontSize: '.7rem', fontWeight: 700,
                                        cursor: 'pointer', fontFamily: 'Inter',
                                        transition: 'all 0.15s ease'
                                      }}
                                    >
                                      <RotateCcw size={11} /> Re-run
                                    </button>
                                  </div>
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
                                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 6 }}>
                                          <span style={{ fontSize: '.74rem', fontWeight: 700, color: 'hsl(280,75%,65%)', background: 'hsl(280,75%,60%/.08)', padding: '1px 6px', borderRadius: 4 }}>
                                            Point {pIdx + 1}
                                          </span>
                                          <div style={{ display: 'flex', gap: 6, alignItems: 'center', flexWrap: 'wrap' }}>
                                            {pt.speakers && pt.speakers.length > 0 && (
                                              <div style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
                                                {pt.speakers.map((sp, idx) => (
                                                  <span key={idx} style={{
                                                    background: 'hsl(var(--muted)/.6)', color: 'hsl(var(--ink))',
                                                    padding: '1px 7px', borderRadius: 10, fontSize: '.68rem', fontWeight: 600,
                                                    display: 'inline-flex', alignItems: 'center', gap: 3
                                                  }}>
                                                    <User size={8} /> {formatItemText(sp)}
                                                  </span>
                                                ))}
                                              </div>
                                            )}
                                            {(() => {
                                              const owner = getActionOwnerText(pt)
                                              return owner ? (
                                                <span style={{
                                                  background: 'hsl(35,95%,50%/.15)', color: 'hsl(35,95%,40%)',
                                                  border: '1px solid hsl(35,95%,50%/.35)',
                                                  padding: '1px 7px', borderRadius: 10, fontSize: '.68rem', fontWeight: 700,
                                                  display: 'inline-flex', alignItems: 'center', gap: 3
                                                }}>
                                                  <UserCheck size={9} /> Owner: {owner}
                                                </span>
                                              ) : (
                                                <span style={{
                                                  background: 'hsl(var(--muted)/.6)', color: 'hsl(var(--pencil))',
                                                  border: '1px solid hsl(var(--border)/.35)',
                                                  padding: '1px 7px', borderRadius: 10, fontSize: '.68rem', fontWeight: 600,
                                                  display: 'inline-flex', alignItems: 'center', gap: 3
                                                }}>
                                                  <UserCheck size={9} /> No action owner
                                                </span>
                                              )
                                            })()}
                                          </div>
                                        </div>
                                        <p style={{ fontSize: '.88rem', fontWeight: 500, color: 'hsl(var(--ink))', lineHeight: 1.5, margin: 0 }}>
                                          {formatItemText(pt.discussion_point)}
                                        </p>
                                        {((pt.action_items?.length || 0) > 0 || (pt.technical_terms?.length || 0) > 0 || (pt.dates?.length || 0) > 0 || (pt.numbers?.length || 0) > 0 || (pt.references?.length || 0) > 0) && (
                                          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5 }}>
                                            {pt.action_items?.map((a, idx) => (
                                              <span key={`act-${idx}`} style={{ background: 'hsl(35,90%,50%/.12)', color: 'hsl(35,90%,45%)', border: '1px solid hsl(35,90%,50%/.3)', padding: '1px 6px', borderRadius: 8, fontSize: '.66rem', fontWeight: 600 }}>Action: {formatItemText(a)}</span>
                                            ))}
                                            {pt.technical_terms?.map((t, idx) => (
                                              <span key={`tech-${idx}`} style={{ background: 'hsl(280,70%,60%/.12)', color: 'hsl(280,70%,65%)', border: '1px solid hsl(280,70%,60%/.3)', padding: '1px 6px', borderRadius: 8, fontSize: '.66rem', fontWeight: 600 }}>Term: {formatItemText(t)}</span>
                                            ))}
                                            {pt.dates?.map((d, idx) => (
                                              <span key={`date-${idx}`} style={{ background: 'hsl(190,80%,50%/.12)', color: 'hsl(190,85%,45%)', border: '1px solid hsl(190,80%,50%/.3)', padding: '1px 6px', borderRadius: 8, fontSize: '.66rem', fontWeight: 600 }}>Date: {formatItemText(d)}</span>
                                            ))}
                                            {pt.numbers?.map((n, idx) => (
                                              <span key={`num-${idx}`} style={{ background: 'hsl(210,80%,60%/.12)', color: 'hsl(210,80%,60%)', border: '1px solid hsl(210,80%,60%/.3)', padding: '1px 6px', borderRadius: 8, fontSize: '.66rem', fontWeight: 600 }}>Num: {formatItemText(n)}</span>
                                            ))}
                                            {pt.references?.map((r, idx) => (
                                              <span key={`ref-${idx}`} style={{ background: 'hsl(160,70%,45%/.12)', color: 'hsl(160,70%,40%)', border: '1px solid hsl(160,70%,45%/.3)', padding: '1px 6px', borderRadius: 8, fontSize: '.66rem', fontWeight: 600 }}>Ref: {formatItemText(r)}</span>
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

                  {/* Stage 1 Window Re-run Modal */}
                  {rerunModalWindow && id && (
                    <Stage1WindowRerunModal
                      recordingId={id}
                      windowData={rerunModalWindow}
                      transcriptWindowMinutes={transcriptWindow}
                      onClose={() => setRerunModalWindow(null)}
                      onAccepted={(newRom) => setRomData(newRom)}
                    />
                  )}
                </div>
              )
            )}

            {/* ── STAGE 2 TAB ── */}
            {activeTab === 'stage2' && (
              !romData?.stage2?.polished_points || romData.stage2.polished_points.length === 0 ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
                  <Stage2ReferenceExamplesPanel
                    examples={stage2ReferenceExamples}
                    onChange={setStage2ReferenceExamples}
                    onSave={handleSaveStage2Examples}
                    saving={savingStage2Examples}
                  />

                  <div style={{ textAlign: 'center', padding: '3rem 1.5rem', color: 'hsl(var(--pencil))' }}>
                    <Target size={42} style={{ margin: '0 auto 1rem', opacity: 0.4, color: 'hsl(205,90%,55%)' }} />
                    <div style={{ fontSize: '1.05rem', fontWeight: 700, color: 'hsl(var(--ink))', marginBottom: '.4rem' }}>No Stage 2 Enhanced Points Yet</div>
                    <div style={{ fontSize: '.84rem', maxWidth: 460, margin: '0 auto 1.25rem', lineHeight: 1.45 }}>
                      Complete Stage 1 first, then click <strong>Retrieve Context</strong> to preview matched documents/previous points, or click <strong>Run Stage 2 Enhancement</strong> to enhance.
                    </div>
                    <div style={{ display: 'flex', gap: 10, justifyContent: 'center', alignItems: 'center' }}>
                      <button
                        type="button"
                        onClick={runPreviewStage2Context}
                        disabled={previewingContext || stage1Count === 0}
                        style={{
                          display: 'flex', alignItems: 'center', gap: 6,
                          padding: '.45rem 1rem', borderRadius: 8,
                          border: '1.5px solid hsl(205,90%,55%/.4)',
                          background: 'hsl(205,90%,55%/.08)',
                          color: 'hsl(205,90%,45%)',
                          fontSize: '.8rem', fontWeight: 600,
                          cursor: (previewingContext || stage1Count === 0) ? 'not-allowed' : 'pointer',
                          fontFamily: 'Inter',
                        }}
                      >
                        {previewingContext ? <Loader size={14} className="spin" /> : <BookOpen size={14} />}
                        {previewingContext ? 'Retrieving Context...' : 'Retrieve Context (Preview)'}
                      </button>

                      <button
                        onClick={runStage2}
                        disabled={stage2Status === 'processing' || stage1Count === 0}
                        className="btn btn-primary"
                        style={{ fontSize: '.8rem', padding: '.45rem 1rem', display: 'flex', alignItems: 'center', gap: 6 }}
                      >
                        <Sparkles size={14} /> Run Stage 2 Enhancement
                      </button>
                    </div>
                  </div>
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                  <SectionHeader icon={<Target size={14} />} label="Stage 2: RAG Enhanced & Merged Points" count={stage2Count} color="hsl(205,90%,55%)" />

                  {/* Outdated Data Warning Card */}
                  {romData?.outdated_warnings?.stage2 && (
                    <div style={{
                      display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                      gap: 12, padding: '.75rem 1rem', borderRadius: 9,
                      border: '1.5px solid hsl(38,90%,50%/.5)', background: 'hsl(38,90%,50%/.1)',
                      color: 'hsl(38,90%,32%)', marginBottom: '.3rem'
                    }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                        <AlertTriangle size={18} style={{ color: 'hsl(38,90%,45%)', flexShrink: 0 }} />
                        <div>
                          <div style={{ fontWeight: 700, fontSize: '.84rem' }}>Previous Stage Changed (Stage 2 Data May Be Outdated)</div>
                          <div style={{ fontSize: '.75rem', marginTop: 2, opacity: 0.9 }}>
                            {romData.outdated_warnings.stage2} You can continue reviewing existing points or click Regenerate to update.
                          </div>
                        </div>
                      </div>
                      <button
                        onClick={() => runGenerateStage2(true)}
                        disabled={stage2Status === 'processing'}
                        style={{
                          padding: '.4rem .9rem', borderRadius: 7, border: 'none',
                          background: 'hsl(38,90%,45%)', color: 'white', fontWeight: 700,
                          fontSize: '.75rem', cursor: stage2Status === 'processing' ? 'not-allowed' : 'pointer',
                          flexShrink: 0, display: 'flex', alignItems: 'center', gap: 5, fontFamily: 'Inter'
                        }}
                      >
                        <RefreshCw size={12} className={stage2Status === 'processing' ? 'spin' : ''} />
                        Regenerate Stage 2
                      </button>
                    </div>
                  )}

                  {/* ── Reference Example Points (Writing Style Only) ── */}
                  <Stage2ReferenceExamplesPanel
                    examples={stage2ReferenceExamples}
                    onChange={setStage2ReferenceExamples}
                    onSave={handleSaveStage2Examples}
                    saving={savingStage2Examples}
                  />

                  {/* ── Stage 2 Editing Toolbar ── */}
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, alignItems: 'center' }}>
                    <button
                      type="button"
                      onClick={runPreviewStage2Context}
                      disabled={previewingContext || stage1Count === 0}
                      style={{
                        display: 'flex', alignItems: 'center', gap: 5,
                        padding: '.35rem .7rem', borderRadius: 8,
                        border: '1.5px solid hsl(205,90%,55%/.4)',
                        background: 'hsl(205,90%,55%/.08)',
                        color: 'hsl(205,90%,45%)',
                        fontSize: '.74rem', fontWeight: 600,
                        cursor: (previewingContext || stage1Count === 0) ? 'not-allowed' : 'pointer',
                        fontFamily: 'Inter',
                      }}
                      title="Preview retrieved context for all point groups without modifying points"
                    >
                      {previewingContext ? <Loader size={12} className="spin" /> : <BookOpen size={12} />}
                      {previewingContext ? 'Retrieving...' : 'Retrieve Context'}
                    </button>

                    <button
                      onClick={() => setShowFindReplace(v => !v)}
                      style={{
                        display: 'flex', alignItems: 'center', gap: 5,
                        padding: '.35rem .7rem', borderRadius: 8,
                        border: `1.5px solid ${showFindReplace ? 'hsl(205,90%,55%/.5)' : 'hsl(var(--border)/.4)'}`,
                        background: showFindReplace ? 'hsl(205,90%,55%/.08)' : 'hsl(var(--card))',
                        color: showFindReplace ? 'hsl(205,90%,55%)' : 'hsl(var(--ink))',
                        fontSize: '.74rem', fontWeight: 600, cursor: 'pointer', fontFamily: 'Inter',
                      }}
                    >
                      <Search size={12} /> Find & Replace
                    </button>

                    <button
                      onClick={handleMergePoints}
                      disabled={selectedPointIds.size < 2 || mergingPoints}
                      style={{
                        display: 'flex', alignItems: 'center', gap: 5,
                        padding: '.35rem .7rem', borderRadius: 8,
                        border: `1.5px solid ${selectedPointIds.size >= 2 ? 'hsl(280,75%,60%/.5)' : 'hsl(var(--border)/.3)'}`,
                        background: selectedPointIds.size >= 2 ? 'hsl(280,75%,60%/.08)' : 'hsl(var(--muted)/.3)',
                        color: selectedPointIds.size >= 2 ? 'hsl(280,75%,60%)' : 'hsl(var(--pencil))',
                        fontSize: '.74rem', fontWeight: 600, cursor: selectedPointIds.size >= 2 ? 'pointer' : 'not-allowed',
                        fontFamily: 'Inter', opacity: mergingPoints ? 0.6 : 1,
                      }}
                    >
                      {mergingPoints ? <Loader size={12} className="spin" /> : <GitMerge size={12} />}
                      Merge Selected ({selectedPointIds.size})
                    </button>

                    {selectedPointIds.size > 0 && (
                      <button
                        onClick={() => setSelectedPointIds(new Set())}
                        style={{
                          display: 'flex', alignItems: 'center', gap: 4,
                          padding: '.3rem .55rem', borderRadius: 6,
                          border: '1px solid hsl(var(--border)/.3)', background: 'none',
                          color: 'hsl(var(--pencil))', fontSize: '.7rem', fontWeight: 600,
                          cursor: 'pointer', fontFamily: 'Inter',
                        }}
                      >
                        <X size={11} /> Clear
                      </button>
                    )}

                    <div style={{ marginLeft: 'auto' }}>
                      <button
                        onClick={() => { setShowChangeHistory(true); loadChangeHistory() }}
                        style={{
                          display: 'flex', alignItems: 'center', gap: 5,
                          padding: '.35rem .7rem', borderRadius: 8,
                          border: '1.5px solid hsl(var(--border)/.4)', background: 'hsl(var(--card))',
                          color: 'hsl(var(--ink))', fontSize: '.74rem', fontWeight: 600,
                          cursor: 'pointer', fontFamily: 'Inter',
                        }}
                      >
                        <History size={12} /> History
                      </button>
                    </div>
                  </div>

                  {/* ── Find & Replace Panel ── */}
                  {showFindReplace && (
                    <div className="find-replace-panel">
                      <div className="fr-input-row">
                        <label>Find</label>
                        <input
                          value={findText}
                          onChange={e => { setFindText(e.target.value); setFindPreviewCount(null); setFindPreviewPoints([]) }}
                          onKeyDown={e => e.key === 'Enter' && handleFindPreview()}
                          placeholder="Search text..."
                          autoFocus
                        />
                        <button onClick={handleFindPreview} className="btn" style={{ padding: '.3rem .6rem', fontSize: '.72rem' }}>
                          <Search size={12} /> Preview
                        </button>
                      </div>
                      <div className="fr-input-row">
                        <label>Replace</label>
                        <input
                          value={replaceText}
                          onChange={e => setReplaceText(e.target.value)}
                          placeholder="Replacement text..."
                        />
                      </div>
                      <div className="fr-actions">
                        {findPreviewCount !== null && (
                          <span className="fr-preview-count">
                            {findPreviewCount} occurrence{findPreviewCount !== 1 ? 's' : ''} in {findPreviewPoints.length} point{findPreviewPoints.length !== 1 ? 's' : ''}
                          </span>
                        )}
                        <div style={{ display: 'flex', gap: 6, marginLeft: 'auto' }}>
                          <button
                            onClick={() => { setShowFindReplace(false); setFindText(''); setReplaceText(''); setFindPreviewCount(null); setFindPreviewPoints([]) }}
                            className="btn" style={{ padding: '.3rem .6rem', fontSize: '.72rem' }}
                          >
                            Cancel
                          </button>
                          <button
                            onClick={handleFindReplace}
                            disabled={!findText.trim() || applyingFindReplace || findPreviewCount === 0}
                            className="btn btn-primary"
                            style={{ padding: '.3rem .75rem', fontSize: '.72rem', opacity: applyingFindReplace ? 0.6 : 1 }}
                          >
                            {applyingFindReplace ? <Loader size={11} className="spin" /> : <ArrowRightLeft size={11} />}
                            Replace All
                          </button>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* ── Point Cards with Selection ── */}
                  {romData?.stage2?.polished_points?.map((pt, i) => {
                    const isSelected = selectedPointIds.has(pt.id)
                    return (
                    <div key={pt.id || i} style={{
                      borderRadius: 10, border: `1.5px solid ${isSelected ? 'hsl(280,75%,60%/.5)' : 'hsl(var(--border)/.4)'}`,
                      background: isSelected ? 'hsl(280,75%,60%/.04)' : 'hsl(var(--card))', padding: '1rem 1.15rem',
                      transition: 'border-color 0.2s, background 0.2s',
                    }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '.6rem' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                          {/* Merge Checkbox */}
                          <div
                            className={`merge-checkbox ${isSelected ? 'checked' : ''}`}
                            onClick={() => togglePointSelection(pt.id)}
                            title={isSelected ? 'Deselect for merge' : 'Select for merge'}
                          >
                            {isSelected && <Check size={11} />}
                          </div>
                          <span style={{
                            background: 'hsl(205,90%,55%/.12)', color: 'hsl(205,90%,60%)',
                            border: '1px solid hsl(205,90%,55%/.3)',
                            padding: '2px 8px', borderRadius: 12, fontSize: '.72rem', fontWeight: 700,
                            display: 'flex', alignItems: 'center', gap: 4, fontFamily: 'JetBrains Mono'
                          }}>
                            <Clock size={11} /> {fmtTime(pt.timeline_start || 0)} – {fmtTime(pt.timeline_end || 0)}
                          </span>
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                          <span style={{ fontSize: '.68rem', fontWeight: 700, color: 'hsl(205,90%,55%)', background: 'hsl(205,90%,55%/.08)', padding: '1px 6px', borderRadius: 6 }}>
                            Point P{i + 1}
                          </span>
                          <button
                            onClick={() => {
                              setEditingStage2PointId(pt.id)
                              setStage2EditDraft(pt.polished_text || '')
                            }}
                            style={{
                              display: 'flex', alignItems: 'center', gap: 3,
                              padding: '2px 7px', borderRadius: 6,
                              border: '1px solid hsl(var(--border)/.4)', background: 'hsl(var(--card))',
                              color: 'hsl(var(--ink))', fontSize: '.68rem', fontWeight: 600,
                              cursor: 'pointer', fontFamily: 'Inter'
                            }}
                            title="Edit discussion point text"
                          >
                            <Pencil size={10} /> Edit
                          </button>
                        </div>
                      </div>

                      {editingStage2PointId === pt.id ? (
                        <div style={{ marginBottom: '.75rem' }}>
                          <textarea
                            value={stage2EditDraft}
                            onChange={(e) => setStage2EditDraft(e.target.value)}
                            rows={4}
                            style={{
                              width: '100%', padding: '.65rem .8rem', borderRadius: 8,
                              border: '1.5px solid hsl(205,90%,55%/.6)', background: 'hsl(var(--paper))',
                              color: 'hsl(var(--ink))', fontSize: '.92rem', lineHeight: 1.55,
                              fontFamily: 'Inter, sans-serif', outline: 'none', resize: 'vertical'
                            }}
                            autoFocus
                          />
                          <div style={{ display: 'flex', gap: 6, justifyContent: 'flex-end', marginTop: '.4rem' }}>
                            <button
                              onClick={() => setEditingStage2PointId(null)}
                              disabled={savingStage2Point}
                              className="btn"
                              style={{ padding: '.25rem .6rem', fontSize: '.72rem' }}
                            >
                              <X size={11} /> Cancel
                            </button>
                            <button
                              onClick={() => handleSaveStage2Point(pt.id)}
                              disabled={savingStage2Point || !stage2EditDraft.trim()}
                              className="btn btn-primary"
                              style={{ padding: '.25rem .75rem', fontSize: '.72rem', opacity: savingStage2Point ? 0.6 : 1 }}
                            >
                              {savingStage2Point ? <Loader size={11} className="spin" /> : <Save size={11} />} Save
                            </button>
                          </div>
                        </div>
                      ) : (
                        <div
                          onContextMenu={(e) => handlePointContextMenu(e, pt.id)}
                          style={{ padding: '.75rem .9rem', background: 'hsl(var(--muted)/.3)', borderRadius: 8, border: '1px solid hsl(var(--border)/.3)', marginBottom: '.75rem', cursor: 'text', userSelect: 'text' }}
                        >
                          <div style={{ fontSize: '.67rem', fontWeight: 700, color: 'hsl(var(--pencil))', marginBottom: 4, textTransform: 'uppercase', letterSpacing: '.04em' }}>Enhanced Point</div>
                          <p style={{ fontSize: '.92rem', color: 'hsl(var(--ink))', lineHeight: 1.55, margin: 0 }}>
                            {formatItemText(pt.polished_text)}
                          </p>
                        </div>
                      )}

                      {((pt.action_items?.length || 0) > 0 || (pt.technical_terms?.length || 0) > 0 || (pt.dates?.length || 0) > 0 || (pt.numbers?.length || 0) > 0 || (pt.references?.length || 0) > 0) && (
                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5, marginBottom: '.75rem' }}>
                          {pt.action_items?.map((a: any, idx: number) => (
                            <span key={`act-${idx}`} style={{ background: 'hsl(35,90%,50%/.12)', color: 'hsl(35,90%,45%)', border: '1px solid hsl(35,90%,50%/.3)', padding: '1px 6px', borderRadius: 8, fontSize: '.66rem', fontWeight: 600 }}>Action: {formatItemText(a)}</span>
                          ))}
                          {pt.technical_terms?.map((t, idx) => (
                            <span key={`tech-${idx}`} style={{ background: 'hsl(280,70%,60%/.12)', color: 'hsl(280,70%,65%)', border: '1px solid hsl(280,70%,60%/.3)', padding: '1px 6px', borderRadius: 8, fontSize: '.66rem', fontWeight: 600 }}>Term: {formatItemText(t)}</span>
                          ))}
                          {pt.dates?.map((d, idx) => (
                            <span key={`date-${idx}`} style={{ background: 'hsl(190,80%,50%/.12)', color: 'hsl(190,85%,45%)', border: '1px solid hsl(190,80%,50%/.3)', padding: '1px 6px', borderRadius: 8, fontSize: '.66rem', fontWeight: 600 }}>Date: {formatItemText(d)}</span>
                          ))}
                          {pt.numbers?.map((n, idx) => (
                            <span key={`num-${idx}`} style={{ background: 'hsl(210,80%,60%/.12)', color: 'hsl(210,80%,60%)', border: '1px solid hsl(210,80%,60%/.3)', padding: '1px 6px', borderRadius: 8, fontSize: '.66rem', fontWeight: 600 }}>Num: {formatItemText(n)}</span>
                          ))}
                          {pt.references?.map((r, idx) => (
                            <span key={`ref-${idx}`} style={{ background: 'hsl(160,70%,45%/.12)', color: 'hsl(160,70%,40%)', border: '1px solid hsl(160,70%,45%/.3)', padding: '1px 6px', borderRadius: 8, fontSize: '.66rem', fontWeight: 600 }}>Ref: {formatItemText(r)}</span>
                          ))}
                        </div>
                      )}

                      {((pt.retrieved_context?.meeting_chunks?.length || 0) > 0 || (pt.retrieved_context?.global_chunks?.length || 0) > 0 || (pt.retrieved_context?.previous_meeting_chunks?.length || 0) > 0) && (
                        <details style={{ background: 'transparent' }}>
                          <summary style={{ cursor: 'pointer', fontSize: '.76rem', fontWeight: 600, color: 'hsl(205,90%,55%)', outline: 'none' }}>
                            View Context Preview ({((pt.retrieved_context?.meeting_chunks?.length || 0) + (pt.retrieved_context?.global_chunks?.length || 0) + (pt.retrieved_context?.previous_meeting_chunks?.length || 0))})
                          </summary>
                          <div style={{ marginTop: '.5rem', display: 'flex', flexDirection: 'column', gap: 6 }}>
                            {pt.retrieved_context?.previous_meeting_chunks?.map((c, idx) => {
                              const txt = (c.text || (c as any).content || '').trim()
                              return txt ? (
                                <div key={`p-${idx}`} style={{ padding: '.5rem .65rem', border: '1px solid hsl(270,75%,55%/.3)', borderRadius: 7, fontSize: '.76rem', background: 'hsl(270,75%,55%/.04)', color: 'hsl(var(--ink))', lineHeight: 1.45 }}>
                                  <span style={{ fontSize: '.62rem', fontWeight: 700, color: 'hsl(270,75%,60%)', textTransform: 'uppercase', letterSpacing: '.04em', display: 'block', marginBottom: 2 }}>
                                    ⏮️ Previous Meeting · {c.meeting_name || 'Meeting'} {c.date ? `(${c.date})` : ''} {c.speakers ? `· Speakers: ${c.speakers}` : ''}
                                  </span>
                                  {txt}
                                </div>
                              ) : null
                            })}
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
                      {((pt as any).context_usage_report || (pt.retrieved_context as any)?.context_usage_report) && (() => {
                        const rep = (pt as any).context_usage_report || (pt.retrieved_context as any).context_usage_report
                        const badges: { label: string; color: string }[] = []
                        if (rep.previous_meeting_context_used) badges.push({ label: '⏮️ Previous Meeting Stage 2 Used', color: 'hsl(270,75%,60%)' })
                        if (rep.meeting_context_used) badges.push({ label: '📄 Meeting Context Used', color: 'hsl(205,90%,50%)' })
                        if (rep.global_context_used) badges.push({ label: '🌐 Global Context Used', color: 'hsl(30,90%,50%)' })
                        if (rep.context_added) badges.push({ label: '✨ Context Added', color: 'hsl(140,65%,45%)' })
                        if (rep.verified && !rep.meeting_context_used) badges.push({ label: '✅ Verified', color: 'hsl(140,65%,45%)' })
                        if (rep.technical_details_added && !rep.context_added) badges.push({ label: '🔬 Details added', color: 'hsl(205,90%,50%)' })
                        if (rep.terminology_clarified) badges.push({ label: '📘 Terms clarified', color: 'hsl(190,80%,45%)' })

                        const docs: string[] = rep.documents || rep.meeting_context_docs || rep.global_context_docs || []
                        if (badges.length === 0 && docs.length === 0) return null
                        return (
                          <details style={{ marginTop: '.5rem' }}>
                            <summary style={{ cursor: 'pointer', fontSize: '.73rem', fontWeight: 600, color: 'hsl(270,75%,60%)', outline: 'none' }}>
                              Context Usage Report
                            </summary>
                            <div style={{ marginTop: '.45rem', padding: '.55rem .7rem', borderRadius: 8, border: '1px solid hsl(270,75%,55%/.25)', background: 'hsl(270,75%,55%/.05)' }}>
                              {badges.length > 0 && (
                                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginBottom: (docs.length > 0) ? '.5rem' : 0 }}>
                                  {badges.map((b, bi) => (
                                    <span key={bi} style={{
                                      fontSize: '.66rem', fontWeight: 700, padding: '1px 7px', borderRadius: 10,
                                      background: b.color + '18', color: b.color, border: `1px solid ${b.color}33`
                                    }}>{b.label}</span>
                                  ))}
                                </div>
                              )}
                              {docs.length > 0 && (
                                <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                                  {docs.map((d, di) => (
                                    <span key={`doc-${di}`} style={{ fontSize: '.65rem', color: 'hsl(205,90%,55%)', display: 'flex', alignItems: 'center', gap: 3 }}>📄 {d}</span>
                                  ))}
                                </div>
                              )}
                            </div>
                          </details>
                        )
                      })()}
                    </div>
                  )
                })}
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
                  {/* Outdated Data Warning Card */}
                  {romData?.outdated_warnings?.stage3 && (
                    <div style={{
                      display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                      gap: 12, padding: '.75rem 1rem', borderRadius: 9,
                      border: '1.5px solid hsl(38,90%,50%/.5)', background: 'hsl(38,90%,50%/.1)',
                      color: 'hsl(38,90%,32%)', marginBottom: '.3rem'
                    }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                        <AlertTriangle size={18} style={{ color: 'hsl(38,90%,45%)', flexShrink: 0 }} />
                        <div>
                          <div style={{ fontWeight: 700, fontSize: '.84rem' }}>Previous Stage Changed (Stage 3 Data May Be Outdated)</div>
                          <div style={{ fontSize: '.75rem', marginTop: 2, opacity: 0.9 }}>
                            {romData.outdated_warnings.stage3} You can continue working with existing agendas or click Regenerate.
                          </div>
                        </div>
                      </div>
                      <button
                        onClick={() => runCreateAgenda(true)}
                        disabled={stage3AgendaStatus === 'processing'}
                        style={{
                          padding: '.4rem .9rem', borderRadius: 7, border: 'none',
                          background: 'hsl(38,90%,45%)', color: 'white', fontWeight: 700,
                          fontSize: '.75rem', cursor: stage3AgendaStatus === 'processing' ? 'not-allowed' : 'pointer',
                          flexShrink: 0, display: 'flex', alignItems: 'center', gap: 5, fontFamily: 'Inter'
                        }}
                      >
                        <RefreshCw size={12} className={stage3AgendaStatus === 'processing' ? 'spin' : ''} />
                        Regenerate Agendas
                      </button>
                    </div>
                  )}

                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <SectionHeader icon={<List size={14} />} label="Agenda Items" count={romData.stage3.agendas.length} color="hsl(140,70%,50%)" />
                    <div style={{ display: 'flex', gap: 6 }}>
                      <button
                        onClick={() => setShowAddAgendaForm(prev => !prev)}
                        style={{
                          padding: '.3rem .75rem', borderRadius: 7,
                          border: '1.5px solid hsl(140,70%,45%/.5)',
                          background: showAddAgendaForm ? 'hsl(140,70%,45%/.2)' : 'hsl(140,70%,45%/.1)',
                          color: 'hsl(140,70%,35%)', fontSize: '.74rem', fontWeight: 700,
                          display: 'flex', alignItems: 'center', gap: 5, cursor: 'pointer', fontFamily: 'Inter'
                        }}
                      >
                        <Plus size={12} /> {showAddAgendaForm ? 'Cancel Add' : 'Add Agenda'}
                      </button>
                      <button onClick={() => runCreateAgenda(true)} disabled={stage3AgendaStatus === 'processing'} style={{ padding: '.3rem .65rem', borderRadius: 7, border: '1px solid hsl(140,70%,45%/.4)', background: 'hsl(140,70%,45%/.08)', color: 'hsl(140,70%,45%)', fontSize: '.72rem', fontWeight: 600, display: 'flex', alignItems: 'center', gap: 4, cursor: 'pointer', fontFamily: 'Inter' }}>
                        {stage3AgendaStatus === 'processing' ? <Loader size={11} className="spin" /> : <RefreshCw size={11} />}
                        Regenerate
                      </button>
                      <button onClick={() => downloadDocx('stage3')} style={{ padding: '.3rem .65rem', borderRadius: 7, border: '1px solid hsl(var(--border)/.4)', background: 'hsl(var(--muted)/.3)', color: 'hsl(var(--pencil))', fontWeight: 600, fontSize: '.72rem', display: 'flex', alignItems: 'center', gap: 4, cursor: 'pointer', fontFamily: 'Inter' }}>
                        <Download size={11} /> Export
                      </button>
                    </div>
                  </div>

                  {/* ── Add New Agenda Form (Collapsible) ── */}
                  {showAddAgendaForm && (
                    <div style={{
                      borderRadius: 10,
                      border: '2px solid hsl(140,70%,45%/.5)',
                      background: 'hsl(140,70%,45%/.04)',
                      padding: '1rem 1.15rem',
                      display: 'flex',
                      flexDirection: 'column',
                      gap: 10,
                      boxShadow: '0 2px 8px rgba(0,0,0,0.05)'
                    }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <div style={{ fontSize: '.84rem', fontWeight: 800, color: 'hsl(140,70%,35%)', display: 'flex', alignItems: 'center', gap: 6 }}>
                          <Plus size={14} /> Add New Agenda Item
                        </div>
                        <button
                          onClick={() => setShowAddAgendaForm(false)}
                          style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'hsl(var(--pencil))', padding: '2px' }}
                        >
                          <X size={14} />
                        </button>
                      </div>

                      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                        <div>
                          <label style={{ fontSize: '.72rem', fontWeight: 700, color: 'hsl(var(--ink))', display: 'block', marginBottom: 3 }}>
                            Agenda Title <span style={{ color: 'hsl(var(--destructive))' }}>*</span>
                          </label>
                          <input
                            value={newAgendaTitle}
                            onChange={e => setNewAgendaTitle(e.target.value)}
                            placeholder="e.g. Budget Review & Next Quarter Planning"
                            style={{ width: '100%', padding: '6px 10px', borderRadius: 7, border: '1.5px solid hsl(var(--border))', fontSize: '.82rem', fontFamily: 'Inter', color: 'hsl(var(--ink))', background: 'hsl(var(--paper))' }}
                          />
                        </div>

                        <div>
                          <label style={{ fontSize: '.72rem', fontWeight: 700, color: 'hsl(var(--ink))', display: 'block', marginBottom: 3 }}>
                            Description / Discussion Details
                          </label>
                          <textarea
                            value={newAgendaDesc}
                            onChange={e => setNewAgendaDesc(e.target.value)}
                            rows={2}
                            placeholder="Optional overview of topics, context, or expected discussions..."
                            style={{ width: '100%', padding: '6px 10px', borderRadius: 7, border: '1.5px solid hsl(var(--border))', fontSize: '.8rem', fontFamily: 'Inter', color: 'hsl(var(--ink))', background: 'hsl(var(--paper))', resize: 'vertical' }}
                          />
                        </div>

                        <div>
                          <label style={{ fontSize: '.72rem', fontWeight: 700, color: 'hsl(var(--ink))', display: 'block', marginBottom: 3 }}>
                            Presenter / Speaker (Optional)
                          </label>
                          <input
                            value={newAgendaPresenter}
                            onChange={e => setNewAgendaPresenter(e.target.value)}
                            placeholder="e.g. Alice or Speaker_1"
                            style={{ width: '100%', padding: '6px 10px', borderRadius: 7, border: '1.5px solid hsl(var(--border))', fontSize: '.82rem', fontFamily: 'Inter', color: 'hsl(var(--ink))', background: 'hsl(var(--paper))' }}
                          />
                        </div>

                        <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end', marginTop: 4 }}>
                          <button
                            onClick={() => setShowAddAgendaForm(false)}
                            style={{ padding: '.4rem .9rem', borderRadius: 7, border: '1px solid hsl(var(--border))', background: 'transparent', fontSize: '.76rem', fontWeight: 600, cursor: 'pointer', fontFamily: 'Inter' }}
                          >
                            Cancel
                          </button>
                          <button
                            onClick={handleCreateAgenda}
                            disabled={creatingAgenda || !newAgendaTitle.trim()}
                            style={{
                              padding: '.4rem 1.1rem',
                              borderRadius: 7,
                              border: 'none',
                              background: 'hsl(140,70%,45%)',
                              color: 'white',
                              fontSize: '.76rem',
                              fontWeight: 700,
                              cursor: creatingAgenda || !newAgendaTitle.trim() ? 'not-allowed' : 'pointer',
                              display: 'flex',
                              alignItems: 'center',
                              gap: 6,
                              fontFamily: 'Inter'
                            }}
                          >
                            {creatingAgenda ? <Loader size={12} className="spin" /> : <Plus size={12} />}
                            {creatingAgenda ? 'Adding...' : 'Add Agenda'}
                          </button>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* ── Stage 3 Interactive Agenda Discussion Sequence & Approximate Timeline ── */}
                  <AgendaTimelineControls
                    agendas={romData.stage3.agendas}
                    totalDuration={effectiveMeetingDuration}
                    discussionOrder={discussionOrder}
                    agendaTimeline={agendaTimeline}
                    onChangeOrder={(newOrder) => setDiscussionOrder(newOrder)}
                    onChangeTimeline={(newTimeline) => setAgendaTimeline(newTimeline)}
                    onReset={resetAgendaOrderAndTimeline}
                    enableOrder={enableAgendaOrder}
                    onChangeEnableOrder={(enabled) => setEnableAgendaOrder(enabled)}
                    enableTimeline={enableAgendaTimeline}
                    onChangeEnableTimeline={(enabled) => setEnableAgendaTimeline(enabled)}
                  />

                  <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                    {romData.stage3.agendas.map((agenda: any, aIdx: number) => {
                      const agendaId = agenda.agenda_id || `A${aIdx + 1}`
                      const docEntry = romData.stage3?.agenda_doc_points?.[agendaId]
                      const isUploading = agendaDocUploading[agendaId] || false
                      const expanded = romData.stage3?.expanded_agendas?.find((ea: any) => ea.agenda_id === agendaId)
                      const isEditingThisAgenda = editingAgendaId === agendaId
                      const presenterName = agenda.presenter || agenda.speaker || docEntry?.presenter
                      const seqRank = discussionOrder.indexOf(agendaId) + 1
                      const timeRange = agendaTimeline[agendaId]

                      return (
                        <div key={agendaId} style={{ borderRadius: 10, border: '1.5px solid hsl(var(--border)/.4)', background: 'hsl(var(--card))', overflow: 'hidden' }}>
                          {/* Agenda Header */}
                          <div style={{ background: 'hsl(140,70%,50%/.08)', padding: '.75rem 1rem', borderBottom: '1px solid hsl(var(--border)/.3)', display: 'flex', alignItems: 'flex-start', gap: 8 }}>
                            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2, flexShrink: 0, marginTop: 2 }}>
                              <span style={{ background: 'hsl(140,70%,45%)', color: 'white', fontWeight: 700, fontSize: '.72rem', padding: '2px 7px', borderRadius: 5 }}>{agendaId}</span>
                              {seqRank > 0 && (
                                <span style={{ fontSize: '.6rem', color: 'hsl(var(--pencil))', fontWeight: 600 }}>#{seqRank} seq</span>
                              )}
                            </div>
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
                                    <button
                                      onClick={() => handleDeleteAgenda(agendaId, agenda.title)}
                                      title="Delete Agenda Item"
                                      style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'hsl(var(--destructive))', padding: '2px', display: 'flex', alignItems: 'center' }}
                                    >
                                      <Trash2 size={12} />
                                    </button>
                                  </div>
                                  {agenda.description && <div style={{ fontSize: '.73rem', color: 'hsl(var(--pencil))', marginTop: 2, lineHeight: 1.35 }}>{agenda.description}</div>}
                                  <div style={{ marginTop: 4, display: 'flex', flexWrap: 'wrap', gap: 6, alignItems: 'center' }}>
                                    {presenterName && (
                                      <div style={{ display: 'inline-flex', alignItems: 'center', gap: 4, background: 'hsl(280,75%,60%/.1)', color: 'hsl(280,75%,60%)', border: '1px solid hsl(280,75%,60%/.3)', padding: '1px 7px', borderRadius: 6, fontSize: '.68rem', fontWeight: 600 }}>
                                        <User size={10} /> Presenter: {presenterName}
                                      </div>
                                    )}
                                    {timeRange && (
                                      <div style={{ display: 'inline-flex', alignItems: 'center', gap: 4, background: 'hsl(140,70%,45%/.1)', color: 'hsl(140,70%,40%)', border: '1px solid hsl(140,70%,45%/.3)', padding: '1px 7px', borderRadius: 6, fontSize: '.68rem', fontWeight: 600, fontFamily: 'JetBrains Mono, monospace' }}>
                                        <Clock size={10} /> {fmtTime(timeRange.start_sec)} – {fmtTime(timeRange.end_sec)}
                                      </div>
                                    )}
                                  </div>
                                </div>
                              )}
                            </div>
                          </div>

                          {/* Skip Agenda Toggle */}
                          <div style={{ padding: '.5rem 1rem', borderBottom: '1px solid hsl(var(--border)/.2)', display: 'flex', alignItems: 'center', gap: 10, background: skippedAgendas[agendaId]?.skipped ? 'hsl(38,90%,50%/.08)' : 'transparent' }}>
                            <label style={{ display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer', fontSize: '.72rem', color: skippedAgendas[agendaId]?.skipped ? 'hsl(38,90%,40%)' : 'hsl(var(--pencil))', userSelect: 'none', fontWeight: 600 }}>
                              <input
                                type="checkbox"
                                checked={skippedAgendas[agendaId]?.skipped || false}
                                onChange={e => {
                                  const checked = e.target.checked
                                  setSkippedAgendas(prev => ({
                                    ...prev,
                                    [agendaId]: {
                                      skipped: checked,
                                      note: prev[agendaId]?.note || 'Keep this agenda if forward to next meeting'
                                    }
                                  }))
                                }}
                                style={{ accentColor: 'hsl(38,90%,50%)', width: 13, height: 13 }}
                              />
                              Skip this agenda
                            </label>
                            {skippedAgendas[agendaId]?.skipped && (
                              <input
                                value={skippedAgendas[agendaId]?.note || ''}
                                onChange={e => {
                                  const val = e.target.value
                                  setSkippedAgendas(prev => ({
                                    ...prev,
                                    [agendaId]: { ...prev[agendaId], note: val }
                                  }))
                                }}
                                placeholder="Note (e.g. Forward to next meeting)"
                                style={{
                                  flex: 1, padding: '3px 8px', borderRadius: 6,
                                  border: '1px solid hsl(38,90%,50%/.4)',
                                  background: 'hsl(38,90%,50%/.06)',
                                  fontSize: '.72rem', fontFamily: 'Inter',
                                  color: 'hsl(38,90%,35%)',
                                  fontStyle: 'italic'
                                }}
                              />
                            )}
                            {skippedAgendas[agendaId]?.skipped && (
                              <span style={{ fontSize: '.64rem', padding: '1px 6px', borderRadius: 5, background: 'hsl(38,90%,50%/.15)', color: 'hsl(38,90%,40%)', fontWeight: 700, border: '1px solid hsl(38,90%,50%/.35)', whiteSpace: 'nowrap' }}>
                                ⏩ Skipped
                              </span>
                            )}
                          </div>

                          <div style={{ padding: '.85rem 1rem', display: 'flex', flexDirection: 'column', gap: '.75rem', opacity: skippedAgendas[agendaId]?.skipped ? 0.45 : 1, pointerEvents: skippedAgendas[agendaId]?.skipped ? 'none' : 'auto', transition: 'opacity .2s' }}>

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
          {activeTab === 'final' && (() => {
            const baseVersionRom = (
              romVersion === 'long'
                ? (romData?.final_rom_versions?.long || originalFinalRom || romData?.final_rom)
                : romData?.final_rom_versions?.[romVersion]
            )

            // Reconcile with Stage 3 Agendas: ALWAYS include ALL agendas in exact Stage 3 order
            const stage3Agendas = romData?.stage3?.agendas || []
            let activeVersionRom = baseVersionRom
            if (baseVersionRom && stage3Agendas.length > 0) {
              const currentById = new Map<string, any>(
                (baseVersionRom.agendas || []).map((a: any) => [a.agenda_id, a])
              )
              const mergedAgendas = stage3Agendas.map((s3a: any) => {
                const existing = currentById.get(s3a.agenda_id)
                const isSkippedInState = skippedAgendas[s3a.agenda_id]?.skipped
                const skipNoteInState = skippedAgendas[s3a.agenda_id]?.note
                if (existing) {
                  return {
                    ...s3a,
                    ...existing,
                    title: existing.title || s3a.title,
                    description: existing.description || s3a.description,
                    presenter: existing.presenter || s3a.presenter,
                    discussion_points: isSkippedInState ? [] : (existing.discussion_points || []),
                    skipped: isSkippedInState ?? existing.skipped,
                    skip_note: skipNoteInState || existing.skip_note || 'Keep this agenda if forward to next meeting',
                  }
                }
                return {
                  ...s3a,
                  discussion_points: [],
                  skipped: isSkippedInState || false,
                  skip_note: skipNoteInState || 'Keep this agenda if forward to next meeting',
                }
              })
              activeVersionRom = {
                ...baseVersionRom,
                agendas: mergedAgendas,
              }
            }

            const isCurrentVersionGenerated = romVersion === 'long'
              ? !!(activeVersionRom?.agendas?.length)
              : !!(baseVersionRom?.agendas?.some((a: any) => a.discussion_points?.length))

            if (!romData?.final_rom?.agendas || romData.final_rom.agendas.length === 0) {
              return (
                <div style={{ textAlign: 'center', padding: '4rem 1.5rem', color: 'hsl(var(--pencil))' }}>
                  <Sparkles size={42} style={{ margin: '0 auto 1rem', opacity: 0.4, color: 'hsl(30,90%,55%)' }} />
                  <div style={{ fontSize: '1.05rem', fontWeight: 700, color: 'hsl(var(--ink))', marginBottom: '.4rem' }}>Final Record of Meeting Not Generated Yet</div>
                  <div style={{ fontSize: '.84rem', maxWidth: 420, margin: '0 auto 1.25rem', lineHeight: 1.45 }}>
                    Run Stage 3 on the left panel to map discussion points to agendas and generate the Final ROM.
                  </div>
                </div>
              )
            }

            return (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
                {/* Outdated Data Warning Card */}
                {romData?.outdated_warnings?.final_rom && (
                  <div style={{
                    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                    gap: 12, padding: '.75rem 1rem', borderRadius: 9,
                    border: '1.5px solid hsl(38,90%,50%/.5)', background: 'hsl(38,90%,50%/.1)',
                    color: 'hsl(38,90%,32%)', marginBottom: '.1rem'
                  }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                      <AlertTriangle size={18} style={{ color: 'hsl(38,90%,45%)', flexShrink: 0 }} />
                      <div>
                        <div style={{ fontWeight: 700, fontSize: '.84rem' }}>Earlier Stage Data Changed (Final ROM May Be Outdated)</div>
                        <div style={{ fontSize: '.75rem', marginTop: 2, opacity: 0.9 }}>
                          {romData.outdated_warnings.final_rom} Click Re-map Final ROM to update discussion points against agendas.
                        </div>
                      </div>
                    </div>
                    <button
                      onClick={runGenerateFinalRom}
                      disabled={stage3FinalStatus === 'processing'}
                      style={{
                        padding: '.4rem .9rem', borderRadius: 7, border: 'none',
                        background: 'hsl(38,90%,45%)', color: 'white', fontWeight: 700,
                        fontSize: '.75rem', cursor: stage3FinalStatus === 'processing' ? 'not-allowed' : 'pointer',
                        flexShrink: 0, display: 'flex', alignItems: 'center', gap: 5, fontFamily: 'Inter'
                      }}
                    >
                      <RefreshCw size={12} className={stage3FinalStatus === 'processing' ? 'spin' : ''} />
                      Re-map Final ROM
                    </button>
                  </div>
                )}

                {/* Sticky Header Bar for Final ROM */}
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: 'hsl(var(--muted)/.3)', padding: '.65rem 1rem', borderRadius: 9, border: '1px solid hsl(var(--border)/.4)', flexWrap: 'wrap', gap: 10 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
                    <SectionHeader icon={<Sparkles size={14} />} label="Final Record of Meeting" count={activeVersionRom?.agendas?.length || finalCount} color="hsl(30,90%,55%)" />
                    {/* Include Action Points Toggle in Final ROM */}
                    <label style={{
                      display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer',
                      fontSize: '.74rem', fontWeight: 600,
                      color: includeActionPointsInRom ? 'hsl(30,90%,40%)' : 'hsl(var(--ink))',
                      background: includeActionPointsInRom ? 'hsl(30,90%,55%/.12)' : 'hsl(var(--card))',
                      border: `1.5px solid ${includeActionPointsInRom ? 'hsl(30,90%,55%/.45)' : 'hsl(var(--border)/.6)'}`,
                      padding: '.28rem .65rem', borderRadius: 7, transition: 'all .15s ease',
                      userSelect: 'none'
                    }}>
                      <input
                        type="checkbox"
                        checked={includeActionPointsInRom}
                        disabled={togglingActionPoints}
                        onChange={e => handleToggleActionPoints(e.target.checked)}
                        style={{ accentColor: 'hsl(30,90%,55%)', width: 13, height: 13 }}
                      />
                      {togglingActionPoints ? (
                        <Loader size={12} className="spin" style={{ color: 'hsl(30,90%,50%)' }} />
                      ) : (
                        <Zap size={12} style={{ color: includeActionPointsInRom ? 'hsl(30,90%,50%)' : 'hsl(var(--pencil))' }} />
                      )}
                      <span>Include Action Points</span>
                    </label>
                  </div>
                  <div style={{ display: 'flex', gap: 8 }}>
                    <button
                      onClick={() => saveFinalRom(activeVersionRom)}
                      disabled={savingRom || !isCurrentVersionGenerated}
                      style={{
                        padding: '.35rem .85rem', borderRadius: 8,
                        background: 'hsl(280,75%,60%)', color: 'white', fontWeight: 700,
                        fontSize: '.76rem', display: 'flex', alignItems: 'center', gap: 6, cursor: (savingRom || !isCurrentVersionGenerated) ? 'not-allowed' : 'pointer',
                        border: 'none', fontFamily: 'Inter', opacity: isCurrentVersionGenerated ? 1 : 0.6
                      }}
                    >
                      {savingRom ? <Loader size={12} className="spin" /> : <Save size={12} />}
                      {savingRom ? 'Saving...' : 'Save Changes'}
                    </button>
                    <button
                      onClick={() => downloadDocx('final')}
                      disabled={!isCurrentVersionGenerated}
                      style={{
                        padding: '.35rem .85rem', borderRadius: 8,
                        background: 'hsl(205,90%,55%/.1)', color: 'hsl(205,90%,60%)', border: '1px solid hsl(205,90%,55%/.3)',
                        fontWeight: 700, fontSize: '.76rem', display: 'flex', alignItems: 'center', gap: 6, cursor: !isCurrentVersionGenerated ? 'not-allowed' : 'pointer', fontFamily: 'Inter', opacity: isCurrentVersionGenerated ? 1 : 0.6
                      }}
                    >
                      <FileDown size={12} /> Download Final DOCX ({romVersion.toUpperCase()})
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

                {/* ── Version Bar: Long | Short | Medium ── */}
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  background: 'hsl(var(--card))',
                  padding: '.6rem 1rem',
                  borderRadius: 10,
                  border: '1.5px solid hsl(var(--border)/.5)',
                  boxShadow: '0 1px 4px rgba(0,0,0,0.04)',
                  flexWrap: 'wrap',
                  gap: 10
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span style={{ fontSize: '.75rem', fontWeight: 800, color: 'hsl(var(--pencil))', textTransform: 'uppercase', letterSpacing: '.05em' }}>
                      ROM Version:
                    </span>
                    {(['long', 'short', 'medium'] as const).map(ver => {
                      const isActive = romVersion === ver
                      const verRom = ver === 'long'
                        ? (romData?.final_rom_versions?.long || originalFinalRom || romData?.final_rom)
                        : romData?.final_rom_versions?.[ver]
                      const isGen = ver === 'long'
                        ? true
                        : !!verRom?.agendas?.some((a: any) => a.discussion_points?.length)
                      const ptCount = verRom?.agendas?.reduce((sum: number, a: any) => sum + (a.discussion_points?.length || 0), 0) || 0

                      return (
                        <button
                          key={ver}
                          onClick={() => handleSelectVersion(ver)}
                          style={{
                            padding: '.45rem 1.1rem',
                            borderRadius: 8,
                            border: isActive
                              ? '2px solid hsl(200,90%,50%)'
                              : '1.5px solid hsl(var(--border)/.6)',
                            background: isActive
                              ? 'hsl(200,90%,50%/.14)'
                              : 'hsl(var(--muted)/.35)',
                            color: isActive ? 'hsl(200,90%,40%)' : 'hsl(var(--ink))',
                            fontWeight: isActive ? 800 : 600,
                            fontSize: '.82rem',
                            cursor: 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            gap: 7,
                            transition: 'all .15s ease',
                            fontFamily: 'Inter'
                          }}
                        >
                          <span>{ver === 'long' ? 'Long' : ver === 'short' ? 'Short' : 'Medium'}</span>
                          {ver === 'long' ? (
                            <span style={{ fontSize: '.64rem', padding: '1px 6px', borderRadius: 4, background: 'hsl(var(--muted))', color: 'hsl(var(--pencil))', fontWeight: 700 }}>
                              Default ({ptCount} pts)
                            </span>
                          ) : isGen ? (
                            <span style={{ fontSize: '.64rem', padding: '1px 6px', borderRadius: 4, background: 'hsl(142,70%,45%/.18)', color: 'hsl(142,70%,32%)', fontWeight: 800 }}>
                              ✓ Generated ({ptCount} pts)
                            </span>
                          ) : (
                            <span style={{ fontSize: '.64rem', padding: '1px 6px', borderRadius: 4, background: 'hsl(var(--muted))', color: 'hsl(var(--pencil))', fontWeight: 600 }}>
                              Not Generated
                            </span>
                          )}
                        </button>
                      )
                    })}
                  </div>

                  <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                    {romVersion !== 'long' && isCurrentVersionGenerated && (
                      <button
                        onClick={() => runGenerateRomVersion(romVersion)}
                        disabled={generatingVersion}
                        title="Regenerate this version from the base Long ROM points"
                        style={{
                          padding: '.35rem .85rem',
                          borderRadius: 6,
                          border: '1px solid hsl(200,90%,50%/.4)',
                          background: 'hsl(200,90%,50%/.08)',
                          color: 'hsl(200,90%,45%)',
                          fontSize: '.74rem',
                          fontWeight: 700,
                          cursor: generatingVersion ? 'not-allowed' : 'pointer',
                          display: 'flex',
                          alignItems: 'center',
                          gap: 5,
                          fontFamily: 'Inter'
                        }}
                      >
                        {generatingVersion ? <Loader size={11} className="spin" /> : <RotateCcw size={11} />}
                        Regenerate {romVersion.charAt(0).toUpperCase() + romVersion.slice(1)}
                      </button>
                    )}
                    <span style={{ fontSize: '.74rem', color: 'hsl(var(--pencil))' }}>
                      {romVersion === 'long' && 'Stage 2 points with full discussion details'}
                      {romVersion === 'short' && 'Agenda-wise condensed: action points & decisions only'}
                      {romVersion === 'medium' && 'Agenda-wise condensed: aggregated key details & decisions'}
                    </span>
                  </div>
                </div>

                {/* If version is NOT generated, show the prominent Generate button card */}
                {!isCurrentVersionGenerated ? (
                  <div style={{
                    textAlign: 'center',
                    padding: '3.5rem 1.5rem',
                    background: 'hsl(var(--card))',
                    borderRadius: 12,
                    border: '2px dashed hsl(200,90%,50%/.35)',
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    gap: 14,
                    boxShadow: '0 2px 8px rgba(0,0,0,0.03)'
                  }}>
                    <div style={{ width: 52, height: 52, borderRadius: 26, background: 'hsl(200,90%,50%/.12)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'hsl(200,90%,45%)' }}>
                      <Sparkles size={26} />
                    </div>
                    <div style={{ maxWidth: 520 }}>
                      <h3 style={{ fontSize: '1.15rem', fontWeight: 800, color: 'hsl(var(--ink))', marginBottom: '.4rem' }}>
                        {romVersion === 'short' ? 'Short ROM Not Generated Yet' : 'Medium ROM Not Generated Yet'}
                      </h3>
                      <p style={{ fontSize: '.84rem', color: 'hsl(var(--pencil))', lineHeight: 1.55 }}>
                        {romVersion === 'short'
                          ? 'Short ROM processes all points for each agenda together, producing a focused summary of action points, decisions, and outcomes (approx. 2–5 points per agenda).'
                          : 'Medium ROM processes all points for each agenda together into an aggregated version with key discussion details alongside all action points and decisions (approx. 4–8 points per agenda).'
                        }
                      </p>
                    </div>
                    <button
                      onClick={() => runGenerateRomVersion(romVersion as 'short' | 'medium')}
                      disabled={generatingVersion}
                      style={{
                        padding: '.6rem 1.6rem',
                        borderRadius: 9,
                        background: generatingVersion ? 'hsl(var(--muted))' : 'linear-gradient(135deg, hsl(200,90%,45%), hsl(220,90%,50%))',
                        color: 'white',
                        fontWeight: 800,
                        fontSize: '.88rem',
                        border: 'none',
                        cursor: generatingVersion ? 'not-allowed' : 'pointer',
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: 9,
                        fontFamily: 'Inter',
                        boxShadow: '0 3px 12px rgba(0,0,0,0.15)',
                        transition: 'transform .15s'
                      }}
                    >
                      {generatingVersion ? <Loader size={15} className="spin" /> : <WandSparkles size={15} />}
                      {generatingVersion
                        ? `Generating ${romVersion.charAt(0).toUpperCase() + romVersion.slice(1)} ROM...`
                        : `Generate ${romVersion.charAt(0).toUpperCase() + romVersion.slice(1)} ROM`
                      }
                    </button>
                  </div>
                ) : (
                  <>
                    {/* Rewrite ROM Panel (Style / Reference rewriting of active points) */}
                    <div style={{ borderRadius: 9, border: `1.5px solid ${isRewritten ? 'hsl(38,90%,52%/.6)' : 'hsl(var(--border)/.4)'}`, background: isRewritten ? 'hsl(38,90%,52%/.04)' : 'hsl(var(--card))', overflow: 'hidden', transition: 'border-color .2s, background .2s' }}>
                      {/* Panel toggle header */}
                      <button
                        onClick={() => setShowRewritePanel(prev => !prev)}
                        style={{ width: '100%', background: isRewritten ? 'hsl(38,90%,52%/.1)' : 'hsl(var(--muted)/.3)', border: 'none', cursor: 'pointer', padding: '.6rem 1rem', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}
                      >
                        <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '.78rem', fontWeight: 700, color: 'hsl(var(--ink))' }}>
                          <WandSparkles size={13} style={{ color: 'hsl(38,90%,52%)' }} />
                          Style Rewrite ({romVersion.toUpperCase()} ROM)
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
                          <div style={{ fontSize: '.73rem', color: 'hsl(var(--pencil))', lineHeight: 1.5, padding: '.55rem .85rem', borderRadius: 7, background: 'hsl(205,90%,55%/.07)', border: '1px solid hsl(205,90%,55%/.2)' }}>
                            <strong style={{ color: 'hsl(var(--ink))' }}>Style-only rewrite</strong> — improves grammar, phrasing, and formatting of {romVersion.toUpperCase()} points. Facts, speakers, decisions, dates, and action items are <strong style={{ color: 'hsl(var(--ink))' }}>never changed</strong>.
                          </div>

                          {isRewritten && (
                            <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '.5rem .85rem', borderRadius: 7, background: 'hsl(38,90%,52%/.1)', border: '1px solid hsl(38,90%,52%/.35)' }}>
                              <WandSparkles size={12} style={{ color: 'hsl(38,90%,42%)', flexShrink: 0 }} />
                              <span style={{ fontSize: '.74rem', color: 'hsl(38,90%,35%)', fontWeight: 600, flex: 1 }}>
                                Showing rewritten {romVersion.toUpperCase()} version. Click "Save Changes" above to persist, or revert.
                              </span>
                              <button
                                onClick={revertToOriginal}
                                style={{ padding: '.28rem .7rem', borderRadius: 6, background: 'hsl(var(--destructive)/.08)', color: 'hsl(var(--destructive))', border: '1px solid hsl(var(--destructive)/.3)', fontSize: '.71rem', fontWeight: 700, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 4, flexShrink: 0, fontFamily: 'Inter' }}
                              >
                                <RotateCcw size={10} /> Revert
                              </button>
                            </div>
                          )}

                          {/* Rewrite Mode Selector */}
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

                          {/* Window Size */}
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

                          {/* Reference Document Upload */}
                          {rewriteMode === 'reference' && (
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '.65rem', padding: '.7rem .85rem', borderRadius: 8, border: '1.5px dashed hsl(280,75%,60%/.4)', background: 'hsl(280,75%,60%/.04)' }}>
                              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                                <div>
                                  <div style={{ fontSize: '.75rem', fontWeight: 700, color: 'hsl(var(--ink))', display: 'flex', alignItems: 'center', gap: 5 }}>
                                    <FileUp size={13} style={{ color: 'hsl(280,75%,60%)' }} />
                                    Upload Reference Document
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
                                  rows={6}
                                  placeholder="Writing rules will appear here after uploading a reference document. You can also type rules manually."
                                  style={{ width: '100%', padding: '.6rem .75rem', borderRadius: 7, border: `1px solid ${writingRules ? 'hsl(280,75%,60%/.5)' : 'hsl(var(--border))'}`, background: 'hsl(var(--muted)/.25)', fontSize: '.76rem', fontFamily: 'Inter', color: 'hsl(var(--ink))', resize: 'vertical', lineHeight: 1.55, boxSizing: 'border-box', transition: 'border-color .2s' }}
                                />
                                {writingRules && (
                                  <div style={{ fontSize: '.65rem', color: 'hsl(280,75%,55%)', marginTop: 3 }}>
                                    ✓ {writingRules.split('\n').filter(l => l.trim()).length} rule{writingRules.split('\n').filter(l => l.trim()).length !== 1 ? 's' : ''} — will be injected into rewrite calls
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
                              rows={3}
                              style={{ width: '100%', padding: '.6rem .75rem', borderRadius: 7, border: '1px solid hsl(var(--border))', background: 'hsl(var(--muted)/.3)', fontSize: '.78rem', fontFamily: 'Inter', color: 'hsl(var(--ink))', resize: 'vertical', lineHeight: 1.5, boxSizing: 'border-box' }}
                            />
                          </div>

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
                                ) ? 'hsl(var(--muted))' : 'linear-gradient(135deg, hsl(200,90%,45%), hsl(220,90%,50%))',
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
                              {rewriteStatus === 'processing'
                                ? `Rewriting Style (${romVersion.toUpperCase()})...`
                                : `Apply Style Rewrite (${romVersion.toUpperCase()})`
                              }
                            </button>

                            {isRewritten && (
                              <button
                                onClick={revertToOriginal}
                                style={{ padding: '.42rem .95rem', borderRadius: 7, background: 'transparent', color: 'hsl(var(--destructive))', border: '1.5px solid hsl(var(--destructive)/.4)', fontSize: '.78rem', fontWeight: 700, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 5, fontFamily: 'Inter' }}
                              >
                                <RotateCcw size={12} /> Revert to Original
                              </button>
                            )}
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
                      {activeVersionRom.agendas.map((agenda: any, aIdx: number) => {
                        const pts = agenda.discussion_points || []
                        const isSkipped = agenda.skipped === true

                        // Collect all action points belonging to this agenda
                        const agendaActionPoints: any[] = []
                        const seenActionKeys = new Set<string>()

                        if (Array.isArray(agenda.action_points)) {
                          for (const ap of agenda.action_points) {
                            const task = ap.task || ap.item || ap.description || ''
                            const key = task.trim().toLowerCase()
                            if (key && !seenActionKeys.has(key)) {
                              seenActionKeys.add(key)
                              agendaActionPoints.push(ap)
                            }
                          }
                        }

                        for (const pt of pts) {
                          const apList = (pt.action_points && pt.action_points.length > 0)
                            ? pt.action_points
                            : (pt.action_items && pt.action_items.length > 0 ? pt.action_items : [])
                          for (const ap of apList) {
                            const task = ap.task || ap.item || ap.description || ''
                            const key = task.trim().toLowerCase()
                            if (key && !seenActionKeys.has(key)) {
                              seenActionKeys.add(key)
                              agendaActionPoints.push(ap)
                            }
                          }
                        }

                        return (
                          <div key={agenda.agenda_id || aIdx} style={{ borderRadius: 10, border: `1.5px solid ${isSkipped ? 'hsl(38,90%,50%/.4)' : 'hsl(var(--border)/.4)'}`, background: isSkipped ? 'hsl(38,90%,50%/.04)' : 'hsl(var(--card))', overflow: 'hidden' }}>
                            {/* Agenda header */}
                            <div style={{ background: isSkipped ? 'hsl(38,90%,50%/.1)' : 'hsl(30,90%,55%/.08)', padding: '.65rem 1rem', borderBottom: '1px solid hsl(var(--border)/.3)', display: 'flex', alignItems: 'center', gap: 8 }}>
                              <span style={{ background: isSkipped ? 'hsl(38,90%,50%)' : 'hsl(30,90%,55%)', color: 'white', fontWeight: 700, fontSize: '.7rem', padding: '2px 7px', borderRadius: 5, flexShrink: 0, fontFamily: 'JetBrains Mono' }}>{agenda.agenda_id || `A${aIdx + 1}`}</span>
                              <div style={{ flex: 1 }}>
                                <div style={{ fontSize: '.92rem', fontWeight: 700, color: 'hsl(var(--ink))' }}>{agenda.title}</div>
                                {agenda.description && <div style={{ fontSize: '.72rem', color: 'hsl(var(--pencil))', marginTop: 1 }}>{agenda.description}</div>}
                                {(agenda.presenter || agenda.speaker) && (
                                  <div style={{ marginTop: 3, display: 'inline-flex', alignItems: 'center', gap: 4, background: 'hsl(280,75%,60%/.1)', color: 'hsl(280,75%,60%)', border: '1px solid hsl(280,75%,60%/.3)', padding: '1px 6px', borderRadius: 6, fontSize: '.67rem', fontWeight: 600 }}>
                                    <User size={10} /> Presenter: {agenda.presenter || agenda.speaker}
                                  </div>
                                )}
                              </div>
                              {isSkipped ? (
                                <span style={{ fontSize: '.7rem', fontWeight: 700, color: 'hsl(38,90%,45%)', background: 'hsl(38,90%,50%/.15)', padding: '2px 8px', borderRadius: 8, border: '1px solid hsl(38,90%,50%/.35)', display: 'flex', alignItems: 'center', gap: 4 }}>
                                  ⏩ Skipped
                                </span>
                              ) : (
                                <span style={{ fontSize: '.7rem', fontWeight: 700, color: 'hsl(30,90%,55%)', background: 'hsl(30,90%,55%/.1)', padding: '2px 7px', borderRadius: 8 }}>{pts.length} pts</span>
                              )}
                            </div>

                            {/* Skipped agenda note */}
                            {isSkipped ? (
                              <div style={{ padding: '.75rem 1rem' }}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '.6rem .85rem', borderRadius: 8, background: 'hsl(38,90%,50%/.08)', border: '1px solid hsl(38,90%,50%/.25)' }}>
                                  <CornerDownRight size={14} style={{ color: 'hsl(38,90%,45%)', flexShrink: 0 }} />
                                  <div>
                                    <div style={{ fontSize: '.76rem', fontWeight: 700, color: 'hsl(38,90%,38%)' }}>Agenda Skipped</div>
                                    <div style={{ fontSize: '.73rem', color: 'hsl(38,90%,35%)', fontStyle: 'italic', marginTop: 2 }}>
                                      {agenda.skip_note || 'Keep this agenda if forward to next meeting'}
                                    </div>
                                  </div>
                                </div>
                              </div>
                            ) : (
                              /* Points Container */
                              <div style={{ padding: '.75rem 1rem', display: 'flex', flexDirection: 'column', gap: 6 }}>
                                {pts.length === 0 ? (
                                  <div style={{ color: 'hsl(var(--pencil))', fontStyle: 'italic', fontSize: '.78rem' }}>No discussion points mapped to this agenda.</div>
                                ) : (
                                  pts.map((pt: any, pIdx: number) => {
                                    const isEditing = editingPointId === pt.id
                                    const ptText = pt.text || pt.polished_text || ''
                                    const spk = pt.speaker ? formatItemText(pt.speaker) : (pt.speakers?.length ? (Array.isArray(pt.speakers) ? pt.speakers.map((s: any) => formatItemText(s)).join(', ') : formatItemText(pt.speakers)) : '—')
                                    const isDocPoint = pt.is_doc_point

                                    return (
                                      <div key={pt.id || `${agenda.agenda_id || aIdx}-pt-${pIdx}`}>
                                        <div style={{ borderRadius: 8, border: `1px solid ${isDocPoint ? 'hsl(280,75%,60%/.3)' : 'hsl(var(--border)/.3)'}`, background: isDocPoint ? 'hsl(280,75%,60%/.05)' : 'hsl(var(--paper)/.4)', padding: '.6rem .8rem' }}>
                                          <div style={{ display: 'flex', gap: 8, alignItems: 'flex-start' }}>
                                            {/* Bullet marker (no point IDs) */}
                                            <span style={{ color: 'hsl(var(--pencil))', fontSize: '1rem', lineHeight: '1.2rem', flexShrink: 0, marginTop: 1, userSelect: 'none' }}>
                                              •
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
                                                        const newFinal = JSON.parse(JSON.stringify(activeVersionRom))
                                                        newFinal.agendas[aIdx].discussion_points[pIdx].text = editDraft
                                                        if (newFinal.agendas[aIdx].discussion_points[pIdx].polished_text) {
                                                          newFinal.agendas[aIdx].discussion_points[pIdx].polished_text = editDraft
                                                        }
                                                        setRomData(prev => {
                                                          if (!prev) return prev
                                                          return {
                                                            ...prev,
                                                            final_rom: newFinal,
                                                            final_rom_versions: {
                                                              ...(prev.final_rom_versions || {}),
                                                              [romVersion]: newFinal,
                                                            }
                                                          }
                                                        })
                                                        setEditingPointId(null)
                                                        toast.success('Point updated. Click "Save Changes" to persist.')
                                                      }}
                                                      style={{ padding: '2px 8px', fontSize: '.7rem', borderRadius: 4, background: 'hsl(280,75%,60%)', color: 'white', border: 'none', fontWeight: 700, cursor: 'pointer' }}
                                                    >
                                                      Save
                                                    </button>
                                                  </div>
                                                </div>
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
                                                <button onClick={() => { setEditingPointId(pt.id); setEditDraft(ptText) }} title="Edit" style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'hsl(var(--pencil))', padding: '2px' }}><Pencil size={11} /></button>
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
                                                  {activeVersionRom.agendas.filter((a: any) => a.agenda_id !== agenda.agenda_id).map((a: any) => (
                                                    <option key={a.agenda_id} value={a.agenda_id}>{a.agenda_id}: {a.title?.slice(0, 25)} ({a.discussion_points?.length || 0} pts)</option>
                                                  ))}
                                                </select>
                                              </div>
                                            )}
                                          </div>
                                        </div>
                                      </div>
                                    )
                                  })
                                )}

                                {/* Action Points Section – Listed underneath all agenda discussion points */}
                                {includeActionPointsInRom && agendaActionPoints.length > 0 && (
                                  <div style={{ marginTop: '.85rem', display: 'flex', flexDirection: 'column', gap: 6 }}>
                                    <div style={{ fontSize: '.84rem', fontWeight: 700, color: 'hsl(var(--ink))', paddingTop: '.6rem', borderTop: '1px solid hsl(var(--border)/.4)', marginBottom: 2 }}>
                                      Action Points
                                    </div>
                                    {agendaActionPoints.map((ap: any, apIdx: number) => {
                                      const apTask = ap.task || ap.item || ap.description || ''
                                      const apAssignee = ap.assignee || ap.owner || ''
                                      const apDeadline = ap.deadline || ''
                                      return (
                                        <div
                                          key={`agenda-ap-${agenda.agenda_id || aIdx}-${apIdx}`}
                                          style={{
                                            borderRadius: 8,
                                            border: '1px solid hsl(var(--border)/.3)',
                                            background: 'hsl(var(--paper)/.4)',
                                            padding: '.6rem .8rem',
                                            display: 'flex',
                                            gap: 8,
                                            alignItems: 'flex-start'
                                          }}
                                        >
                                          <span style={{ color: 'hsl(var(--pencil))', fontSize: '1rem', lineHeight: '1.2rem', flexShrink: 0, marginTop: 1, userSelect: 'none' }}>•</span>
                                          <div style={{ flex: 1 }}>
                                            <div style={{ color: 'hsl(var(--ink))', lineHeight: 1.45, fontSize: '.83rem' }}>{apTask}</div>
                                            {(apAssignee || (apDeadline && String(apDeadline).toLowerCase() !== 'asap' && String(apDeadline).toLowerCase() !== 'none')) && (
                                              <div style={{ marginTop: 4, display: 'flex', gap: 8, fontSize: '.7rem', color: 'hsl(var(--pencil))' }}>
                                                {apAssignee && <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}><UserCheck size={10} /> {apAssignee}</span>}
                                                {apDeadline && String(apDeadline).toLowerCase() !== 'asap' && String(apDeadline).toLowerCase() !== 'none' && (
                                                  <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}><Clock size={10} /> {apDeadline}</span>
                                                )}
                                              </div>
                                            )}
                                          </div>
                                        </div>
                                      )
                                    })}
                                  </div>
                                )}
                              </div>
                            )}
                          </div>
                        )
                      })}
                    </div>
                  </>
                )}
              </div>
            )
          })()}



        </div>
      </div>
    </div>

      {/* ── Context Menu (right-click on selected text) ── */}
      {contextMenu && (
        <>
          <div
            style={{ position: 'fixed', inset: 0, zIndex: 199 }}
            onClick={() => setContextMenu(null)}
          />
          <div
            className="stage2-context-menu"
            style={{ left: contextMenu.x, top: contextMenu.y }}
          >
            <button
              className="stage2-context-menu-item"
              onClick={() => handleContextMenuAction('split')}
              disabled={splittingPoint}
            >
              {splittingPoint ? <Loader size={14} className="spin" /> : <Scissors size={14} />}
              Create Separate Point
            </button>
            <div className="stage2-context-menu-divider" />
            <button
              className="stage2-context-menu-item danger"
              onClick={() => handleContextMenuAction('delete_text')}
              disabled={deletingText}
            >
              {deletingText ? <Loader size={14} className="spin" /> : <Trash2 size={14} />}
              Delete Selected Text
            </button>
          </div>
        </>
      )}

      {/* ── Change History Drawer ── */}
      {showChangeHistory && (
        <>
          <div className="change-history-drawer-backdrop" onClick={() => setShowChangeHistory(false)} />
          <div className={`change-history-drawer ${showChangeHistory ? 'open' : ''}`}>
            <div className="change-history-header">
              <h3><History size={16} /> Change History</h3>
              <button className="icon-btn" onClick={() => setShowChangeHistory(false)}><X size={16} /></button>
            </div>
            <div className="change-history-list">
              {loadingHistory ? (
                <div style={{ textAlign: 'center', padding: '2rem', color: 'hsl(var(--pencil))' }}>
                  <Loader size={20} className="spin" style={{ margin: '0 auto .5rem' }} />
                  <div style={{ fontSize: '.82rem' }}>Loading history...</div>
                </div>
              ) : changeHistory.length === 0 ? (
                <div style={{ textAlign: 'center', padding: '2rem', color: 'hsl(var(--pencil))', fontSize: '.85rem' }}>
                  No changes recorded yet
                </div>
              ) : (
                changeHistory.map((change: any) => (
                  <div key={change.id} className={`change-history-card ${change.is_reverted ? 'reverted' : ''}`}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '.4rem' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                        <span className={`change-type-badge ${change.change_type}`}>
                          {change.change_type === 'merge' && <GitMerge size={10} />}
                          {change.change_type === 'split' && <Scissors size={10} />}
                          {change.change_type === 'find_replace' && <ArrowRightLeft size={10} />}
                          {change.change_type === 'delete_text' && <Trash2 size={10} />}
                          {change.change_type === 'delete' && <Trash2 size={10} />}
                          {change.change_type === 'manual_edit' && <Pencil size={10} />}
                          {change.change_type?.replace(/_/g, ' ')}
                        </span>
                        {change.is_reverted && <span className="change-type-badge reverted-badge">Reverted</span>}
                      </div>
                      <span style={{ fontSize: '.65rem', color: 'hsl(var(--pencil))' }}>
                        {new Date(change.created_at).toLocaleString()}
                      </span>
                    </div>

                    {/* Before/After Preview */}
                    {change.before_state?.length > 0 && (
                      <details style={{ marginTop: '.35rem' }}>
                        <summary style={{ cursor: 'pointer', fontSize: '.72rem', fontWeight: 600, color: 'hsl(var(--pencil))', outline: 'none' }}>
                          View changes ({change.before_state.length} point{change.before_state.length !== 1 ? 's' : ''})
                        </summary>
                        <div style={{ marginTop: '.35rem', display: 'flex', flexDirection: 'column', gap: '.35rem' }}>
                          <div className="diff-before">
                            <div style={{ fontSize: '.6rem', fontWeight: 700, textTransform: 'uppercase', marginBottom: 2, color: 'hsl(0,75%,55%)' }}>Before</div>
                            {change.before_state.map((bs: any, bi: number) => (
                              <div key={bi} style={{ fontSize: '.75rem', lineHeight: 1.4 }}>
                                {(bs.polished_text || '').substring(0, 200)}{(bs.polished_text || '').length > 200 ? '…' : ''}
                              </div>
                            ))}
                          </div>
                          <div className="diff-after">
                            <div style={{ fontSize: '.6rem', fontWeight: 700, textTransform: 'uppercase', marginBottom: 2, color: 'hsl(130,60%,42%)' }}>After</div>
                            {(change.after_state || []).map((as_: any, ai: number) => (
                              <div key={ai} style={{ fontSize: '.75rem', lineHeight: 1.4 }}>
                                {(as_.polished_text || '').substring(0, 200)}{(as_.polished_text || '').length > 200 ? '…' : ''}
                              </div>
                            ))}
                          </div>
                        </div>
                      </details>
                    )}

                    {/* Revert / Redo Actions */}
                    <div style={{ marginTop: '.5rem', display: 'flex', gap: 6 }}>
                      {!change.is_reverted ? (
                        <button
                          onClick={() => handleRevertChange(change.id)}
                          disabled={revertingChangeId === change.id}
                          style={{
                            display: 'flex', alignItems: 'center', gap: 5,
                            padding: '.3rem .65rem', borderRadius: 6,
                            border: '1px solid hsl(var(--border)/.4)', background: 'hsl(var(--card))',
                            color: 'hsl(var(--ink))', fontSize: '.7rem', fontWeight: 600,
                            cursor: 'pointer', fontFamily: 'Inter',
                            opacity: revertingChangeId === change.id ? 0.5 : 1,
                          }}
                        >
                          {revertingChangeId === change.id ? <Loader size={11} className="spin" /> : <RotateCcw size={11} />}
                          Revert
                        </button>
                      ) : (
                        <button
                          onClick={() => handleRedoChange(change.id)}
                          disabled={redoingChangeId === change.id}
                          style={{
                            display: 'flex', alignItems: 'center', gap: 5,
                            padding: '.3rem .65rem', borderRadius: 6,
                            border: '1px solid hsl(205,90%,55%/.4)', background: 'hsl(205,90%,55%/.1)',
                            color: 'hsl(205,90%,60%)', fontSize: '.7rem', fontWeight: 600,
                            cursor: 'pointer', fontFamily: 'Inter',
                            opacity: redoingChangeId === change.id ? 0.5 : 1,
                          }}
                        >
                          {redoingChangeId === change.id ? <Loader size={11} className="spin" /> : <RotateCw size={11} />}
                          Redo
                        </button>
                      )}
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </>
      )}

      {/* Stage 2 Retrieve Context Preview Modal */}
      <Stage2ContextPreviewModal
        isOpen={showContextPreview}
        onClose={() => setShowContextPreview(false)}
        groups={contextPreviewGroups}
        totalGroups={contextPreviewTotalGroups}
        totalPoints={contextPreviewTotalPoints}
        onEnhance={runStage2}
        isEnhancing={stage2Status === 'processing'}
      />
    </div>
  )
}


