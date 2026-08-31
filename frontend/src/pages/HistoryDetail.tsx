import { useEffect, useState, useRef, useCallback } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { ArrowLeft, Loader, Clock, Users, FileAudio, FileText, Sparkles, RefreshCw, MoreVertical, UserCheck, Video, RotateCcw, AlertTriangle, Replace } from 'lucide-react'
import { toast } from 'sonner'
import TranscriptViewer from '../components/TranscriptViewer'
import VideoTranscriptViewer from '../components/VideoTranscriptViewer'
import AIChatPanel from '../components/AIChatPanel'
import PDFButton from '../components/PDFButton'
import InlineEdit from '../components/InlineEdit'
import api from '../api/client'
import { useJobsStore } from '../store/jobs'
import { useProcessingStore } from '../store/processing'
import type { RecordingDetail, VideoTranscriptBlock } from '../types/recording'

// MoM data shape (mirrors mom_router.py response)
interface MomData {
  title?: string
  introduction?: string
  points_discussed?: string[]
  action_items?: Array<{ task: string; owner: string; deadline: string }>
  conclusion?: string
  participants?: string[]
  date?: string
}

function fmtDuration(s: number) {
  const m = Math.floor(s / 60)
  const sec = Math.floor(s % 60).toString().padStart(2, '0')
  return `${m}m ${sec}s`
}



// Deterministic speaker color (same as History page)
const SPEAKER_COLORS = [
  { bg: 'hsl(14,90%,56%)', border: 'hsl(14,90%,56% / .3)', text: 'hsl(14,90%,30%)' },
  { bg: 'hsl(205,85%,55%)', border: 'hsl(205,85%,55% / .3)', text: 'hsl(205,85%,28%)' },
  { bg: 'hsl(130,60%,45%)', border: 'hsl(130,60%,45% / .3)', text: 'hsl(130,60%,25%)' },
  { bg: 'hsl(280,65%,58%)', border: 'hsl(280,65%,58% / .3)', text: 'hsl(280,65%,30%)' },
  { bg: 'hsl(45,90%,50%)', border: 'hsl(45,90%,50% / .3)', text: 'hsl(45,90%,25%)' },
  { bg: 'hsl(340,75%,58%)', border: 'hsl(340,75%,58% / .3)', text: 'hsl(340,75%,30%)' },
]
function getSpeakerColor(name: string) {
  let hash = 0
  for (const c of name) hash = (hash * 31 + c.charCodeAt(0)) & 0xffff
  return SPEAKER_COLORS[hash % SPEAKER_COLORS.length]
}



export default function HistoryDetail() {
  const [showConfidence, setShowConfidence] = useState(true);
  const { id } = useParams()
  const [rec, setRec] = useState<RecordingDetail | null>(null)
  const [activeTab, setActiveTab] = useState<'audio' | 'video'>('audio')
  const [videoBlocks, setVideoBlocks] = useState<VideoTranscriptBlock[]>([])
  const [loading, setLoading] = useState(true)
  const [chatOpen, setChatOpen] = useState(true)
  const [audioUrl, setAudioUrl] = useState<string | null>(null)
  const [regenerating, setRegenerating] = useState(false)
  const [regenDone, setRegenDone] = useState(false)
  const [menuOpen, setMenuOpen] = useState(false);
  const [momData, setMomData] = useState<MomData | null>(null)
  const [isGeneratingInsights, setIsGeneratingInsights] = useState(false)
  const [reidentifying, setReidentifying] = useState(false)
  const [reidentifyDone, setReidentifyDone] = useState(false)
  const [confirmRerunOpen, setConfirmRerunOpen] = useState(false)
  const [rerunning, setRerunning] = useState(false)
  const [correctMistakeOpen, setCorrectMistakeOpen] = useState(false)
  const [wrongText, setWrongText] = useState('')
  const [correctText, setCorrectText] = useState('')
  const [correctingMistake, setCorrectingMistake] = useState(false)
  const [correctResult, setCorrectResult] = useState<{ count: number; done: boolean } | null>(null)
  const reidentifyPollRef = useRef<ReturnType<typeof setInterval> | null>(null)
  // ── Speaker Tab: scroll-to-segment ──────────────────────────────────────────
  const [highlightSegId, setHighlightSegId] = useState<string | undefined>(undefined)
  const handleScrollToSegment = useCallback((segId: string, _startTime: number) => {
    setHighlightSegId(undefined)  // reset first so useEffect always fires
    requestAnimationFrame(() => setHighlightSegId(segId))
  }, [])
  // â”€â”€ Resizable chat panel â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  const [chatWidth, setChatWidth] = useState<number>(() => {
    const saved = localStorage.getItem('ai-chat-panel-width')
    return saved ? parseInt(saved, 10) : 340
  })
  const isDragging = useRef(false)
  const dragStartX = useRef(0)
  const dragStartWidth = useRef(0)
  const navigate = useNavigate()

  const handleDragStart = useCallback((e: React.MouseEvent) => {
    e.preventDefault()
    isDragging.current = true
    dragStartX.current = e.clientX
    dragStartWidth.current = chatWidth
    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'

    const onMove = (ev: MouseEvent) => {
      if (!isDragging.current) return
      const delta = dragStartX.current - ev.clientX  // dragging left edge = larger delta = bigger panel
      const newW = Math.min(680, Math.max(220, dragStartWidth.current + delta))
      setChatWidth(newW)
    }

    const onUp = () => {
      isDragging.current = false
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
      setChatWidth(w => { localStorage.setItem('ai-chat-panel-width', String(w)); return w })
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
    }

    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
  }, [chatWidth])

  const handleRename = async (newName: string) => {
    await api.patch(`/history/${id}/rename`, { filename: newName })
    setRec((prev) => (prev ? { ...prev, filename: newName } : prev))
  }

  const handleRegenerate = async () => {
    if (!id || regenerating) return
    setRegenerating(true)
    setRegenDone(false)
    try {
      const res = await api.post(`/history/${id}/regenerate-insights`)
      // Live-update the displayed record with fresh Llama output
      setRec((prev) => prev ? {
        ...prev,
        summary: res.data.short_summary,
        short_summary: res.data.short_summary,
        detailed_summary: res.data.detailed_summary,
        key_points: res.data.key_points,
        action_items: res.data.action_items,
      } : prev)
      setRegenDone(true)
      setTimeout(() => setRegenDone(false), 3000)
    } catch (err: unknown) {
      console.error('[Regenerate] Failed:', err)
    } finally {
      setRegenerating(false)
    }
  }

  const handleReidentify = async () => {
    if (!id || reidentifying) return
    setReidentifying(true)
    setReidentifyDone(false)

    // Clear any previous poll
    if (reidentifyPollRef.current) clearInterval(reidentifyPollRef.current)
    reidentifyPollRef.current = null

    try {
      await api.post(`/history/${id}/reidentify-speakers`)
    } catch (err: unknown) {
      console.error('[ReidentifySpeakers] Failed to start:', err)
      setReidentifying(false)
      return
    }

    // Poll /audio/jobs/:id every 2.5 s until done/error
    reidentifyPollRef.current = setInterval(async () => {
      try {
        const r = await api.get(`/audio/jobs/${id}`)
        const { status, result } = r.data
        if (status === 'done' || status === 'transcript_ready') {
          if (reidentifyPollRef.current) clearInterval(reidentifyPollRef.current)
          reidentifyPollRef.current = null
          if (result?.transcript) {
            setRec(prev => prev ? {
              ...prev,
              transcript: result.transcript,
              speakers_detected: result.speakers_detected ?? prev.speakers_detected,
            } : prev)
          }
          setReidentifyDone(true)
          setTimeout(() => setReidentifyDone(false), 4000)
          setReidentifying(false)
        } else if (status === 'error' || status === 'cancelled') {
          if (reidentifyPollRef.current) clearInterval(reidentifyPollRef.current)
          reidentifyPollRef.current = null
          console.error('[ReidentifySpeakers] Job ended with status:', status)
          setReidentifying(false)
        }
      } catch (pollErr) {
        console.warn('[ReidentifySpeakers] Poll failed (will retry):', pollErr)
      }
    }, 2500)
  }

  const handleCorrectMistake = async () => {
    if (!id || !wrongText.trim() || correctingMistake) return
    setCorrectingMistake(true)
    setCorrectResult(null)
    try {
      const res = await api.post(`/history/${id}/correct-mistake`, {
        wrong_text: wrongText.trim(),
        correct_text: correctText,
      })
      const count: number = res.data.occurrences_replaced ?? 0
      setCorrectResult({ count, done: true })
      if (count > 0) {
        // Reload the recording so the corrected transcript is shown immediately
        reloadDetail()
      }
    } catch (err: unknown) {
      console.error('[CorrectMistake] Failed:', err)
      setCorrectResult({ count: -1, done: true })
    } finally {
      setCorrectingMistake(false)
    }
  }

  const handleRerunPipeline = async () => {
    if (!id || rerunning || !rec) return
    setConfirmRerunOpen(false)
    setRerunning(true)
    try {
      await api.post(`/history/${id}/rerun`)
      
      const source = rec.source_type === 'video' ? 'video-upload' : 'upload'
      useJobsStore.getState().addJob({
        jobId: id,
        source: source,
        filename: rec.filename || 'recording',
        startedAt: new Date().toISOString(),
      })
      useProcessingStore.getState().setProcessing(source, 'queued')

      toast.success('Pipeline rerun initiated! Re-processing media...')
      setRec(prev => prev ? { ...prev, status: 'processing' } : prev)
    } catch (err: unknown) {
      console.error('[RerunPipeline] Failed to start:', err)
      toast.error('Failed to initiate pipeline rerun.')
    } finally {
      setRerunning(false)
    }
  }

  /** Generate AI insights on-demand (called from AIChatPanel Generate button) */
  const handleGenerateInsights = useCallback(async (tasks: string[]) => {
    if (!id || isGeneratingInsights) return
    setIsGeneratingInsights(true)
    try {
      const res = await api.post(`/history/${id}/generate-insights`, { tasks })
      setRec((prev) => prev ? {
        ...prev,
        summary: res.data.short_summary ?? prev.summary,
        short_summary: res.data.short_summary ?? prev.short_summary,
        detailed_summary: res.data.detailed_summary ?? prev.detailed_summary,
        key_points: res.data.key_points ?? prev.key_points,
        action_items: res.data.action_items ?? prev.action_items,
      } : prev)
    } catch (err: unknown) {
      console.error('[GenerateInsights] Failed:', err)
    } finally {
      setIsGeneratingInsights(false)
    }
  }, [id, isGeneratingInsights])

  const reloadDetail = useCallback(() => {
    if (!id) return;
    api.get(`/history/${id}`).then((r) => {
      setRec(r.data);
      if (r.data?.video_transcript && Array.isArray(r.data.video_transcript)) {
        setVideoBlocks(r.data.video_transcript);
      }
    });
    api.get(`/mom/${id}`).then((r) => setMomData(r.data)).catch(() => {});
  }, [id]);

  useEffect(() => {
    if (!id) return
    setLoading(true);
    api.get(`/history/${id}`).then((r) => {
      setRec(r.data)
      if (r.data?.video_transcript && Array.isArray(r.data.video_transcript)) {
        setVideoBlocks(r.data.video_transcript)
      }
    }).finally(() => setLoading(false))

    // Fetch video transcript endpoint to ensure latest OCR blocks are loaded
    api.get(`/video/jobs/${id}/video-transcript`)
      .then((r) => {
        if (r.data?.video_transcript && Array.isArray(r.data.video_transcript)) {
          setVideoBlocks(r.data.video_transcript)
        }
      })
      .catch(() => {})

    // Also fetch MoM data (404 means not generated yet — that's fine)
    api.get(`/mom/${id}`)
      .then((r) => setMomData(r.data))
      .catch(() => setMomData(null))
  }, [id])

  useEffect(() => {
    if (!id || !rec) return
    let isActive = true
    let createdUrl: string | null = null

    api.get(`/history/${id}/audio`, { responseType: 'blob' })
      .then((response) => {
        if (!isActive) return
        const url = URL.createObjectURL(response.data)
        createdUrl = url
        setAudioUrl(url)
      })
      .catch((err) => {
        console.warn('[HistoryDetail] Failed to load audio:', err)
        if (isActive) setAudioUrl(null)
      })

    return () => {
      isActive = false
      if (createdUrl) {
        URL.revokeObjectURL(createdUrl)
      }
    }
  }, [id, rec?.id])

  if (loading) return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', flexDirection: 'column', gap: '1rem' }}>
      <Loader size={24} className="spin" style={{ color: 'hsl(var(--accent))' }} />
      <p style={{ fontFamily: 'Inter, sans-serif', fontSize: '.9rem', color: 'hsl(var(--pencil))' }}>Loading recordingâ€¦</p>
    </div>
  )

  if (!rec) return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'hsl(var(--pencil))' }}>
      Recording not found.
    </div>
  )

  const chatW = chatOpen ? `${chatWidth}px` : '48px'
  const speakersFromTranscript = Array.from(new Set(
    (rec.transcript || []).map(s => s.speaker_label || s.speaker).filter((lbl): lbl is string => Boolean(lbl) && lbl !== 'Unknown')
  ))
  const speakers: string[] = Array.from(new Set([
    ...(rec.speakers_detected || []),
    ...speakersFromTranscript
  ]))

  return (
    <div className="workspace-split" style={{
      display: 'grid', gridTemplateColumns: `1fr ${chatW}`,
      position: 'relative', transition: isDragging.current ? 'none' : 'grid-template-columns .25s ease',

    }}>
      <div style={{ display: 'flex', flexDirection: 'column', minHeight: '100%' }}>

        {/* Header */}
        <div className="panel-header history-detail-header">
          <button
            onClick={() => navigate('/dashboard/history')}
            className="icon-btn"
            title="Back to history"
            style={{ flexShrink: 0 }}
          >
            <ArrowLeft size={15} />
          </button>

          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{
              fontWeight: 700, fontSize: '1rem',
              overflow: 'hidden',
              fontFamily: 'Inter, sans-serif', color: 'hsl(var(--ink))',
              marginBottom: '5px',
              width: "30vw"
            }}>
              <InlineEdit
                value={rec.filename}
                onSave={handleRename}
                textStyle={{ fontWeight: 700, fontSize: '1rem', fontFamily: 'Inter, sans-serif', color: 'hsl(var(--ink))' }}
              />
            </div>
            <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', alignItems: 'center' }}>
              <span style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '0.75rem', color: 'hsl(var(--pencil))', fontFamily: 'Inter, sans-serif' }}>
                <Clock size={11} /> {new Date(rec.created_at).toLocaleString()}
              </span>
              {rec.duration > 0 && (
                <span style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '.72rem', fontWeight: 600, background: 'hsl(var(--muted))', color: 'hsl(var(--pencil))', border: '1px solid hsl(var(--ink) / .08)', padding: '.1rem .45rem', borderRadius: '999px', fontFamily: 'JetBrains Mono, monospace' }}>
                  <FileAudio size={10} /> {fmtDuration(rec.duration)}
                </span>
              )}
              {(rec.source_type === 'video' || videoBlocks.length > 0) && (
                <span style={{
                  display: 'inline-flex', alignItems: 'center', gap: '4px',
                  fontSize: '.72rem', fontWeight: 600,
                  background: 'hsl(210,80%,55%/.12)', border: '1px solid hsl(210,80%,55%/.3)',
                  color: 'hsl(210,85%,60%)', padding: '.1rem .5rem', borderRadius: '999px',
                  fontFamily: 'Inter, sans-serif'
                }}>
                  <Video size={10} /> Video Recording ({videoBlocks.length} OCR blocks)
                </span>
              )}
              {/* Speaker chips */}
              {speakers.map((sp, si) => {
                const col = getSpeakerColor(sp)
                return (
                  <span key={si} className="speaker-chip" style={{
                    background: `${col.bg}18`,
                    borderColor: `${col.bg}55`,
                    color: col.text,
                  }}>
                    <span className="speaker-chip-dot" style={{ background: col.bg }}>
                      {sp.charAt(0).toUpperCase()}
                    </span>
                    {sp}
                  </span>
                )
              })}
            </div>
          </div>



          {/* Header Action Buttons */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexShrink: 0 }}>
            <button
              className="btn btn-ghost"
              onClick={() => setConfirmRerunOpen(true)}
              disabled={rerunning || reidentifying || regenerating}
              title="Re-run processing pipeline on this recording"
              id="btn-rerun-pipeline-header"
              style={{ fontSize: '.82rem', padding: '.4rem .85rem', display: 'flex', alignItems: 'center', gap: '6px', color: 'hsl(var(--accent))' }}
            >
              {rerunning ? <Loader size={14} className="spin" /> : <RotateCcw size={14} />}
              <span>Re-Run Pipeline</span>
            </button>

            {/* More Options */}
            <div
              style={{
                position: "relative",
                flexShrink: 0,
              }}
            >
              <button
                className="icon-btn"
                onClick={() => setMenuOpen((open) => !open)}
                aria-label="More recording options"
                aria-expanded={menuOpen}
              >
                <MoreVertical size={18} />
              </button>

              {menuOpen && (
                <div className="header-dropdown" >

                  <button
                    className="dropdown-item"
                    onClick={() => {
                      setConfirmRerunOpen(true);
                      setMenuOpen(false);
                    }}
                    disabled={rerunning || reidentifying}
                    id="btn-rerun-pipeline-dropdown"
                  >
                    <RotateCcw size={14} style={{ color: 'hsl(var(--accent))' }} />
                    Re-Run Pipeline
                  </button>

                  <button
                    className="dropdown-item"
                    onClick={() => {
                      handleRegenerate();
                      setMenuOpen(false);
                    }}
                    disabled={regenerating || reidentifying}
                  >
                    {regenerating ? (
                      <>
                        <Loader size={14} className="spin" />
                        Generating...
                      </>
                    ) : (
                      <>
                        <RefreshCw size={14} />
                        Regenerate AI
                      </>
                    )}
                  </button>

                  <button
                    className="dropdown-item"
                    onClick={() => {
                      handleReidentify();
                      setMenuOpen(false);
                    }}
                    disabled={reidentifying || regenerating}
                    id="btn-reidentify-speakers"
                    aria-label="Re-run speaker identification"
                  >
                    {reidentifying ? (
                      <>
                        <Loader size={14} className="spin" />
                        Re-identifying...
                      </>
                    ) : (
                      <>
                        <UserCheck size={14} />
                        Re-run Speaker ID
                      </>
                    )}
                  </button>

                  <button
                    className="dropdown-item"
                    onClick={() => {
                      navigate(`/dashboard/history/${id}/mom`);
                      setMenuOpen(false);
                    }}
                  >
                    <FileText size={14} />
                    Minutes of Meeting
                  </button>

                  <button
                    className="dropdown-item"
                    onClick={() => {
                      navigate(`/dashboard/history/${id}/rom`);
                      setMenuOpen(false);
                    }}
                  >
                    <FileText size={14} style={{ color: 'hsl(160,70%,45%)' }} />
                    Record of Meeting (ROM)
                  </button>

                  {(rec.source_type === 'video' || videoBlocks.length > 0) && (
                    <button
                      className="dropdown-item"
                      onClick={() => {
                        setActiveTab('video');
                        setMenuOpen(false);
                      }}
                    >
                      <Video size={14} style={{ color: 'hsl(210,85%,60%)' }} />
                      Video OCR Transcript ({videoBlocks.length})
                    </button>
                  )}

                  <button
                    className="dropdown-item"
                    onClick={() => {
                      setCorrectMistakeOpen(true)
                      setCorrectResult(null)
                      setWrongText('')
                      setCorrectText('')
                      setMenuOpen(false)
                    }}
                    id="btn-correct-mistake"
                  >
                    <Replace size={14} style={{ color: 'hsl(45,90%,45%)' }} />
                    Correct Mistake
                  </button>

                  <div className="dropdown-item">
                    <PDFButton
                      recordingId={id}
                      filename={rec.filename}
                      variant="ghost"
                    />
                  </div>

                </div>
              )}
            </div>
          </div>
        </div>

        {/* Re-identification success banner */}
        {reidentifyDone && (
          <div style={{
            display: 'flex', alignItems: 'center', gap: '8px',
            padding: '.45rem 1.25rem',
            background: 'hsl(130,55%,95%)',
            borderBottom: '1px solid hsl(130,55%,80%)',
            fontSize: '.78rem', fontWeight: 600,
            color: 'hsl(130,55%,30%)',
            fontFamily: 'Inter, sans-serif',
            animation: 'fadeIn .25s ease',
          }}>
            <UserCheck size={13} />
            Speaker IDs updated ✓
          </div>
        )}

        {/* Transcript area */}
        <div className="transcript-scroll" style={{
          flex: 1, overflowY: 'auto',
          padding: '0rem 1.5rem',
          background: 'hsl(var(--paper) / .4)',
          minHeight: 0
        }}>
          {/* Tab buttons if video transcript is available */}
          {(rec.source_type === 'video' || videoBlocks.length > 0) && (
            <div style={{
              display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '1.25rem',
              background: 'hsl(var(--muted)/.4)', padding: '4px', borderRadius: '10px',
              width: 'fit-content', border: '1px solid hsl(var(--border)/.4)'
            }}>
              <button
                onClick={() => setActiveTab('audio')}
                style={{
                  display: 'inline-flex', alignItems: 'center', gap: '6px',
                  padding: '.35rem .85rem', borderRadius: '7px', border: 'none',
                  background: activeTab === 'audio' ? 'hsl(var(--card))' : 'transparent',
                  color: activeTab === 'audio' ? 'hsl(var(--ink))' : 'hsl(var(--pencil))',
                  fontWeight: activeTab === 'audio' ? 700 : 500,
                  fontSize: '.8rem', cursor: 'pointer', fontFamily: 'Inter, sans-serif',
                  boxShadow: activeTab === 'audio' ? '0 1px 3px rgba(0,0,0,0.06)' : 'none',
                  transition: 'all .15s ease'
                }}
              >
                <FileAudio size={14} /> Spoken Audio Transcript
              </button>

              <button
                onClick={() => setActiveTab('video')}
                style={{
                  display: 'inline-flex', alignItems: 'center', gap: '6px',
                  padding: '.35rem .85rem', borderRadius: '7px', border: 'none',
                  background: activeTab === 'video' ? 'hsl(var(--card))' : 'transparent',
                  color: activeTab === 'video' ? 'hsl(210,85%,60%)' : 'hsl(var(--pencil))',
                  fontWeight: activeTab === 'video' ? 700 : 500,
                  fontSize: '.8rem', cursor: 'pointer', fontFamily: 'Inter, sans-serif',
                  boxShadow: activeTab === 'video' ? '0 1px 3px rgba(0,0,0,0.06)' : 'none',
                  transition: 'all .15s ease'
                }}
              >
                <Video size={14} /> Video OCR Transcript ({videoBlocks.length})
              </button>
            </div>
          )}

          {activeTab === 'audio' ? (
            <>
              {rec.transcript?.length > 0 && (
                <div className="transcript-subheader">
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <h3 style={{ fontSize: '1rem', fontWeight: 700, color: 'hsl(var(--ink))', fontFamily: 'Inter, sans-serif', letterSpacing: '-.01em', margin: 0 }}>
                      Transcript
                    </h3>
                    <span style={{ fontSize: '.72rem', fontWeight: 600, color: 'hsl(var(--pencil))', background: 'hsl(var(--muted))', padding: '.15rem .5rem', borderRadius: '999px', fontFamily: 'Inter, sans-serif' }}>
                      {rec.transcript.length} segments
                    </span>
                    {speakers.length > 0 && (
                      <span style={{ fontSize: '.72rem', fontWeight: 600, color: 'hsl(var(--pencil))', fontFamily: 'Inter, sans-serif' }}>
                        · <Users size={10} style={{ display: 'inline', verticalAlign: 'middle' }} /> {speakers.length} {speakers.length === 1 ? 'speaker' : 'speakers'}
                      </span>
                    )}
                  </div>
                  <div
                    className="confidence-legend"
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: "16px",
                      flexWrap: "wrap",
                    }}
                  >
                    <span
                      style={{
                        fontSize: ".68rem",
                        color: "hsl(var(--pencil))",
                        textTransform: "uppercase",
                        letterSpacing: ".08em",
                        fontWeight: 700,
                        fontFamily: "Inter, sans-serif",
                      }}
                    >
                      Confidence
                    </span>

                    <span className="confidence-legend-item">
                      <span
                        className="conf-dot"
                        style={{ background: "hsl(var(--sticky-green))" }}
                      />
                      High
                    </span>

                    <span className="confidence-legend-item">
                      <span
                        className="conf-dot"
                        style={{ background: "hsl(45,90%,50%)" }}
                      />
                      Mid
                    </span>

                    <span className="confidence-legend-item">
                      <span
                        className="conf-dot"
                        style={{ background: "hsl(var(--destructive))" }}
                      />
                      Low
                    </span>

                    <div style={{ flex: 1 }} />

                    <label className="confidence-switch">
                      <span>Highlight</span>

                      <input
                        type="checkbox"
                        checked={showConfidence}
                        onChange={(e) => setShowConfidence(e.target.checked)}
                      />

                      <span className="slider" />
                    </label>
                  </div>
                </div>
              )}
              <TranscriptViewer
                segments={rec.transcript || []}
                showConfidence={showConfidence}
                audioUrl={audioUrl || undefined}
                recordingId={id}
                highlightSegId={highlightSegId}
                onSegmentsChange={(updated) => {
                  if (rec) {
                    setRec({ ...rec, transcript: updated });
                  }
                  reloadDetail();
                }}
              />
            </>
          ) : (
            <VideoTranscriptViewer
              blocks={videoBlocks}
              filename={rec.filename}
              recordingId={id}
              onBlocksUpdated={(updated) => setVideoBlocks(updated)}
            />
          )}
        </div>
      </div>

      {/* Drag handle + AI Chat Panel */}
      <div className={`insights-pane ${chatOpen ? 'is-open' : ''}`} style={{ position: 'relative', display: 'flex' }}>
        {/* Drag handle â€” only visible when panel is open */}
        {chatOpen && (
          <div
            onMouseDown={handleDragStart}
            title="Drag to resize"
            style={{
              position: 'absolute', left: 0, top: 0, bottom: 0,
              width: '6px',
              cursor: 'col-resize',
              zIndex: 10,
              background: 'transparent',
              transition: 'background .15s',
            }}
            onMouseEnter={e => (e.currentTarget.style.background = 'hsl(var(--accent) / .25)')}
            onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}
          />
        )}
        <AIChatPanel
          recordingId={id!}
          summary={rec.summary}
          shortSummary={rec.short_summary as string | undefined}
          detailedSummary={rec.detailed_summary as string | undefined}
          keyPoints={rec.key_points}
          actionItems={rec.action_items}
          speakerSummary={rec.speaker_summary}
          momData={momData}
          isOpen={chatOpen}
          onToggle={() => setChatOpen((o) => !o)}
          onGenerateInsights={handleGenerateInsights}
          isGeneratingInsights={isGeneratingInsights}
          onScrollToSegment={handleScrollToSegment}
          onTranscriptChanged={() => reloadDetail()}
        />
      </div>

      {/* ── Correct Mistake Modal ── */}
      {correctMistakeOpen && (
        <div style={{
          position: 'fixed', inset: 0, zIndex: 9999,
          background: 'rgba(0,0,0,0.45)', backdropFilter: 'blur(3px)',
          display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '1rem'
        }}>
          <div style={{
            background: 'hsl(var(--card))',
            border: '2px solid hsl(var(--border))',
            borderRadius: '16px',
            padding: '1.5rem 1.75rem',
            maxWidth: '460px',
            width: '100%',
            boxShadow: '0 20px 40px rgba(0,0,0,0.2)',
            display: 'flex',
            flexDirection: 'column',
            gap: '1.1rem',
          }} className="animate-scale-up">

            {/* Header */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <div style={{
                width: 40, height: 40, borderRadius: '12px',
                background: 'hsl(45,90%,50%/.12)',
                border: '1.5px solid hsl(45,90%,50%/.3)',
                display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0
              }}>
                <Replace size={18} style={{ color: 'hsl(45,90%,45%)' }} />
              </div>
              <div>
                <h3 style={{ fontSize: '1rem', fontWeight: 700, margin: 0, color: 'hsl(var(--ink))' }}>
                  Correct Mistake
                </h3>
                <p style={{ fontSize: '.78rem', color: 'hsl(var(--pencil))', margin: '2px 0 0 0', fontFamily: 'Inter, sans-serif' }}>
                  Find &amp; replace text across the entire meeting
                </p>
              </div>
            </div>

            {/* Inputs */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '.65rem' }}>
              <div>
                <label style={{ fontSize: '.75rem', fontWeight: 700, color: 'hsl(var(--pencil))', fontFamily: 'Inter, sans-serif', display: 'block', marginBottom: '.3rem' }}>
                  Wrong text / word
                </label>
                <input
                  className="input"
                  id="correct-mistake-wrong"
                  placeholder="e.g. Jhon"
                  value={wrongText}
                  autoFocus
                  onChange={e => { setWrongText(e.target.value); setCorrectResult(null) }}
                  onKeyDown={e => { if (e.key === 'Enter' && wrongText.trim()) handleCorrectMistake() }}
                  style={{ fontSize: '.88rem', height: '40px', fontFamily: 'Inter, sans-serif' }}
                />
              </div>
              <div>
                <label style={{ fontSize: '.75rem', fontWeight: 700, color: 'hsl(var(--pencil))', fontFamily: 'Inter, sans-serif', display: 'block', marginBottom: '.3rem' }}>
                  Correct text / word
                </label>
                <input
                  className="input"
                  id="correct-mistake-correct"
                  placeholder="e.g. John"
                  value={correctText}
                  onChange={e => { setCorrectText(e.target.value); setCorrectResult(null) }}
                  onKeyDown={e => { if (e.key === 'Enter' && wrongText.trim()) handleCorrectMistake() }}
                  style={{ fontSize: '.88rem', height: '40px', fontFamily: 'Inter, sans-serif' }}
                />
              </div>
            </div>

            {/* Result banner */}
            {correctResult && correctResult.done && (
              <div style={{
                padding: '.65rem 1rem',
                borderRadius: '10px',
                fontSize: '.82rem',
                fontFamily: 'Inter, sans-serif',
                fontWeight: 600,
                ...(correctResult.count < 0
                  ? { background: 'hsl(0,80%,96%)', border: '1px solid hsl(0,75%,80%)', color: 'hsl(0,65%,40%)' }
                  : correctResult.count === 0
                  ? { background: 'hsl(var(--muted))', border: '1px solid hsl(var(--border))', color: 'hsl(var(--pencil))' }
                  : { background: 'hsl(130,55%,95%)', border: '1px solid hsl(130,55%,75%)', color: 'hsl(130,55%,30%)' })
              }}>
                {correctResult.count < 0
                  ? '✕ An error occurred. Please try again.'
                  : correctResult.count === 0
                  ? '⚠ No matches found. Nothing was changed.'
                  : `✓ Correction applied — ${correctResult.count} occurrence${correctResult.count === 1 ? '' : 's'} replaced.`}
              </div>
            )}

            {/* Buttons */}
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px', marginTop: '.1rem' }}>
              <button
                className="btn btn-ghost"
                onClick={() => { setCorrectMistakeOpen(false); setCorrectResult(null) }}
                style={{ fontSize: '.85rem', padding: '.45rem 1rem' }}
              >
                Close
              </button>
              <button
                className="btn btn-primary"
                onClick={handleCorrectMistake}
                disabled={!wrongText.trim() || correctingMistake}
                id="btn-correct-mistake-apply"
                style={{ fontSize: '.85rem', padding: '.45rem 1.2rem', background: 'hsl(45,90%,45%)', borderColor: 'hsl(45,90%,45%)' }}
              >
                {correctingMistake ? <><Loader size={13} className="spin" /> Applying…</> : <><Replace size={13} /> Apply</>}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Confirmation Modal for Re-Run Pipeline */}
      {confirmRerunOpen && (
        <div style={{
          position: 'fixed', inset: 0, zIndex: 9999,
          background: 'rgba(0, 0, 0, 0.45)', backdropFilter: 'blur(3px)',
          display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '1rem'
        }}>
          <div style={{
            background: 'hsl(var(--card))',
            border: '2px solid hsl(var(--border))',
            borderRadius: '16px',
            padding: '1.5rem 1.75rem',
            maxWidth: '480px',
            width: '100%',
            boxShadow: '0 20px 40px rgba(0, 0, 0, 0.2)',
            display: 'flex',
            flexDirection: 'column',
            gap: '1.25rem',
          }} className="animate-scale-up">
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <div style={{
                width: 40, height: 40, borderRadius: '12px',
                background: 'hsl(var(--destructive) / .12)',
                border: '1.5px solid hsl(var(--destructive) / .3)',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                flexShrink: 0
              }}>
                <RotateCcw size={20} style={{ color: 'hsl(var(--destructive))' }} />
              </div>
              <div>
                <h3 style={{ fontSize: '1.05rem', fontWeight: 700, margin: 0, color: 'hsl(var(--ink))' }}>
                  Re-Run Pipeline Confirmation
                </h3>
                <p style={{ fontSize: '.8rem', color: 'hsl(var(--pencil))', margin: '2px 0 0 0' }}>
                  This will re-process the existing media with your current settings.
                </p>
              </div>
            </div>

            <div style={{
              fontSize: '.85rem', color: 'hsl(var(--ink-soft))', lineHeight: 1.5,
              background: 'hsl(var(--muted) / .5)', padding: '.85rem 1rem', borderRadius: '10px',
              border: '1px dashed hsl(var(--border))'
            }}>
              ⚠️ <strong>Warning:</strong> Existing transcriptions, speaker identifications, Minutes of Meeting (MoM), AI Insights, and ROM outputs will be replaced with newly generated results. The original audio/video file will not be deleted.
            </div>

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px', marginTop: '.25rem' }}>
              <button
                className="btn btn-ghost"
                onClick={() => setConfirmRerunOpen(false)}
                style={{ fontSize: '.85rem', padding: '.45rem 1rem' }}
              >
                Cancel
              </button>
              <button
                className="btn btn-primary"
                onClick={handleRerunPipeline}
                style={{
                  fontSize: '.85rem', padding: '.45rem 1.1rem',
                  background: 'hsl(var(--destructive))', borderColor: 'hsl(var(--destructive))'
                }}
              >
                Re-Run Pipeline
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
