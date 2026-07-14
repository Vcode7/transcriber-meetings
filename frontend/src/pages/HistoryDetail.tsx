import { useEffect, useState, useRef, useCallback } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { ArrowLeft, Loader, Clock, Users, FileAudio, FileText, Sparkles, RefreshCw, MoreVertical, UserCheck } from 'lucide-react'
import TranscriptViewer from '../components/TranscriptViewer'
import AIChatPanel from '../components/AIChatPanel'
import PDFButton from '../components/PDFButton'
import InlineEdit from '../components/InlineEdit'
import api from '../api/client'
import type { RecordingDetail } from '../types/recording'

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
  const reidentifyPollRef = useRef<ReturnType<typeof setInterval> | null>(null)
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

  useEffect(() => {
    if (!id) return
    api.get(`/history/${id}`).then((r) => setRec(r.data)).finally(() => setLoading(false))
    // Also fetch MoM data (404 means not generated yet — that's fine)
    api.get(`/mom/${id}`)
      .then((r) => setMomData(r.data))
      .catch(() => setMomData(null))
  }, [id])

  useEffect(() => {
    if (!id || !rec?.file_path) return
    let isActive = true
    const blobUrlRef = { current: null as string | null }
    api.get(`/history/${id}/audio`, { responseType: 'blob' })
      .then((response) => {
        if (!isActive) return
        const url = URL.createObjectURL(response.data)
        blobUrlRef.current = url
        setAudioUrl(url)
      })
      .catch(() => { if (isActive) setAudioUrl(null) })
    return () => {
      isActive = false
      if (blobUrlRef.current) {
        URL.revokeObjectURL(blobUrlRef.current)
        blobUrlRef.current = null
      }
    }
  }, [id, rec?.file_path])

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
  const speakers: string[] = rec.speakers_detected || []

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

        {/* Transcript */}
        <div className="transcript-scroll" style={{
          flex: 1, overflowY: 'auto',
          padding: '1.25rem 1.5rem',
          background: 'hsl(var(--paper) / .4)',
          minHeight: 0
        }}>
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
                    Â· <Users size={10} style={{ display: 'inline', verticalAlign: 'middle' }} /> {speakers.length} {speakers.length === 1 ? 'speaker' : 'speakers'}
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
            onSegmentsChange={(updated) => {
              if (rec) {
                setRec({ ...rec, transcript: updated });
              }
            }}
          />
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
        />
      </div>
    </div>
  )
}
