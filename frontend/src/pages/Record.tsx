import { useState, useCallback, useEffect, useRef } from 'react'
import { Mic, Square, RotateCcw, Loader, AlertTriangle, Users, Radio, CheckCircle } from 'lucide-react'
import { useAudioRecorder } from '../hooks/useAudioRecorder'
import { useJobPoller } from '../hooks/useJobPoller'
import WaveformVisualizer from '../components/WaveformVisualizer'
import TranscriptViewer from '../components/TranscriptViewer'
import AIChatPanel from '../components/AIChatPanel'
import ProcessingOverlay from '../components/ProcessingOverlay'
import PDFButton from '../components/PDFButton'
import AdvancedOptionsPanel, { type AdvancedOptions } from '../components/AdvancedOptions'
import { useProcessingStore, type ProcessingStage } from '../store/processing'
import api from '../api/client'
import { getApiErrorDetail } from '../lib/errors'
import type { ProcessingResult } from '../types/recording'

type Stage = 'idle' | 'recording' | 'stopped' | 'uploading' | 'processing' | 'transcript_ready' | 'done' | 'error'

const PROGRESS: Record<string, string> = {
  queued: 'Queuedâ€¦',
  transcribing: 'Transcribing audioâ€¦',
  diarizing: 'Identifying speakersâ€¦',
  identifying_speakers: 'Matching voice profilesâ€¦',
  generating_insights: 'Generating AI insightsâ€¦',
}

const OVERLAP_CONFIRM_COUNT = 2
const OVERLAP_COOLDOWN_MS = 4000

export default function RecordPage() {
  const [showConfidence, setShowConfidence] = useState(true);
  const recorder = useAudioRecorder()
  const [stage, setStage] = useState<Stage>('idle')
  const [recordingId, setRecordingId] = useState<string | null>(null)
  const [result, setResult] = useState<ProcessingResult | null>(null)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const [chatOpen, setChatOpen] = useState(true)
  const [overlapAlert, setOverlapAlert] = useState(false)
  const [advancedOpts, setAdvancedOpts] = useState<AdvancedOptions>({
    meetingPrompt: '', expectedSpeakers: null, selectedVoiceIds: [],
    useDictionary: false, useVocabularyInPrompt: false, speakerSummary: false,
  })

  const overlapCountRef = useRef(0)
  const cooldownUntilRef = useRef(0)
  const checkingRef = useRef(false)

  const { setProcessing, updateStage: updateProcStage, clearProcessing, stage: procStage, startedAt, source } = useProcessingStore()
  const [chatWidth, setChatWidth] = useState<number>(() => {
      const saved = localStorage.getItem('ai-chat-panel-width')
      return saved ? parseInt(saved, 10) : 340
    })
    const isDragging = useRef(false)
    const dragStartX = useRef(0)
    const dragStartWidth = useRef(0)
  
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
        const newW = Math.min(580, Math.max(220, dragStartWidth.current + delta))
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
  const onTranscriptReady = useCallback((data: Partial<ProcessingResult>) => {
    // Phase 1 done â€” show transcript immediately, keep polling for AI
    setResult(prev => ({ ...(prev ?? {}), ...data } as ProcessingResult))
    setStage('transcript_ready')
    // Dismiss overlay so the transcript is visible during AI generation
    clearProcessing()
  }, [clearProcessing])

  const onDone = useCallback((data: ProcessingResult) => {
    // Phase 2 done â€” merge AI fields into existing result
    setResult(prev => ({ ...(prev ?? {}), ...data } as ProcessingResult))
    setStage('done')
    clearProcessing()
  }, [clearProcessing])

  // Poll while processing OR while transcript is ready (waiting for AI)
  const isPolling = stage === 'processing' || stage === 'transcript_ready'
  const jobData = useJobPoller(isPolling ? recordingId : null, { onTranscriptReady, onDone })

  // Sync job progress â†’ global processing store
  useEffect(() => {
    if (jobData?.progress) {
      updateProcStage(jobData.progress as ProcessingStage)
    }
    if (jobData?.status === 'error') {
      setStage('error')
      clearProcessing()
    }
  }, [jobData?.progress, jobData?.status, updateProcStage, clearProcessing])

  // Cleanup on unmount
  useEffect(() => {
    return () => clearProcessing()
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // Cross-talk detection
  useEffect(() => {
    if (stage !== 'recording') {
      overlapCountRef.current = 0
      return
    }
    const interval = setInterval(async () => {
      if (checkingRef.current) return
      const chunkBlob = recorder.latestChunkRef.current
      if (!chunkBlob || chunkBlob.size < 500) return
      checkingRef.current = true
      try {
        const form = new FormData()
        form.append('file', chunkBlob, 'chunk.webm')
        const res = await api.post('/api/detect-overlap', form, { timeout: 3000 })
        const isOverlap = res.data.overlap === 1
        if (isOverlap) {
          overlapCountRef.current += 1
        } else {
          overlapCountRef.current = 0
        }
        if (overlapCountRef.current >= OVERLAP_CONFIRM_COUNT && Date.now() > cooldownUntilRef.current) {
          setOverlapAlert(true)
          cooldownUntilRef.current = Date.now() + OVERLAP_COOLDOWN_MS
          overlapCountRef.current = 0
          setTimeout(() => setOverlapAlert(false), 3000)
        }
      } catch (err: unknown) {
        console.warn('[CrossTalk] detect-overlap request failed:', getApiErrorDetail(err, 'Unknown error'))
      } finally {
        checkingRef.current = false
      }
    }, 1200)
    return () => {
      clearInterval(interval)
      overlapCountRef.current = 0
      checkingRef.current = false
    }
  }, [stage, recorder.latestChunkRef])

  const handleStop = () => { recorder.stop(); setStage('stopped') }

  const handleSubmit = async () => {
    if (!recorder.audioBlob) return
    setStage('uploading')
    setUploadError(null)
    setProcessing('record', 'uploading')
    try {
      const form = new FormData()
      form.append('file', recorder.audioBlob, 'recording.webm')
      form.append('meeting_prompt', advancedOpts.meetingPrompt)
      form.append('participant_voice_ids', JSON.stringify(advancedOpts.selectedVoiceIds))
      form.append('use_vocabulary', advancedOpts.useVocabularyInPrompt ? 'true' : 'false')
      form.append('speaker_summary', advancedOpts.speakerSummary ? 'true' : 'false')
      const res = await api.post('/audio/record', form)
      setRecordingId(res.data.recording_id)
      setStage('processing')
      updateProcStage('queued')
    } catch (e: unknown) {
      setUploadError(getApiErrorDetail(e, 'Upload failed.'))
      setStage('error')
      clearProcessing()
    }
  }

  const handleReset = () => {
    recorder.reset(); setStage('idle')
    setRecordingId(null); setResult(null); setUploadError(null)
    setOverlapAlert(false)
    overlapCountRef.current = 0
    cooldownUntilRef.current = 0
    clearProcessing()
  }

  const processing = stage === 'uploading' || stage === 'processing'
  const isProcessingActive = useProcessingStore((s) => s.isProcessing && s.source === 'record')
  const isGeneratingAI = stage === 'transcript_ready'
  const chatW = chatOpen ? `${chatWidth}px` : '48px'
  const isRecording = stage === 'recording'

  return (
    <div className="workspace-split" style={{ display: 'grid', gridTemplateColumns: `1fr ${chatW}`, transition: isDragging.current
    ? 'none'
    : 'grid-template-columns .25s ease' }}>

      {/* â”€â”€ Center panel */}
      <div className="center-panel" style={{ position: 'relative' }}>

        {/* Processing overlay */}
        {processing && (
          <ProcessingOverlay stage={procStage} startedAt={startedAt} source={source} />
        )}

        {/* Header */}
        <div className="panel-header" style={{ position: 'relative', overflow: 'hidden' }}>
          {/* Stage progress bar */}
          <div style={{
            position: 'absolute', top: 0, left: 0, right: 0, height: '3px',
            background: isRecording
              ? 'linear-gradient(90deg, hsl(var(--destructive)), hsl(var(--accent)), hsl(var(--destructive)))'
              : stage === 'done'
              ? 'hsl(var(--success))'
              : 'hsl(var(--accent))',
            backgroundSize: isRecording ? '200% 100%' : '100% 100%',
            animation: isRecording ? 'progress-shimmer 2s linear infinite' : 'none',
            transition: 'background .5s'
          }} />

          <div style={{
            width: '34px', height: '34px', borderRadius: '10px', flexShrink: 0,
            background: result ? 'hsl(var(--success) / .12)' : isRecording ? 'hsl(var(--destructive) / .15)' : 'hsl(var(--accent) / .12)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            border: `2px solid ${result ? 'hsl(var(--success) / .3)' : isRecording ? 'hsl(var(--destructive) / .4)' : 'hsl(var(--accent) / .3)'}`,
            transition: 'all .3s'
          }}>
            {result
              ? <CheckCircle size={16} style={{ color: 'hsl(var(--success))' }} />
              : <Radio
                  size={16}
                  style={{ color: isRecording ? 'hsl(var(--destructive))' : 'hsl(var(--accent))', transition: 'all .3s' }}
                  className={isRecording ? 'animate-pulse-rec' : ''}
                />}
          </div>

          <div style={{ flex: 1, minWidth: 0 }}>
            <h1>{result ? 'Transcript' : 'Record Conversation'}</h1>
            <p style={{ fontSize: '.82rem', color: 'hsl(var(--pencil))', fontFamily: 'Inter, sans-serif', fontWeight: 400, marginTop: '1px' }}>
              {result ? `${result.transcript?.length ?? 0} segments` : 'Record first, then get transcription + speaker ID'}
            </p>
          </div>

          {result?.speakers_detected?.length > 0 && (
            <div className="animate-slide-in-right" style={{
              display: 'flex', alignItems: 'center', gap: '6px',
              fontSize: '.8rem', color: 'hsl(var(--success))',
              padding: '.35rem .8rem',
              background: 'hsl(var(--success) / .12)',
              borderRadius: '999px',
              border: '1.5px solid hsl(var(--success) / .3)',
              fontFamily: 'Inter, sans-serif',
              fontWeight: 600,
              flexShrink: 0
            }}>
              <Users size={13} style={{ color: 'hsl(var(--success))' }} />
              <span>{result.speakers_detected.join(', ')}</span>
            </div>
          )}

          {result && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexShrink: 0 }}>
              <PDFButton
                recordingId={recordingId}
                filename="recording"
                variant="ghost"
              />
              <button
                className="btn btn-ghost animate-bounce-in"
                onClick={handleReset}
                style={{ flexShrink: 0, fontSize: '.82rem', padding: '.4rem .85rem' }}
              >
                <RotateCcw size={14} /> New Recording
              </button>
            </div>
          )}
        </div>

        {/* Recording Section â€” hidden after results arrive */}
        {!result && (<div className="capture-setup" style={{
          padding: '1.75rem 2rem',
          borderBottom: '2px dashed hsl(var(--border))',
          display: 'flex',
          flexDirection: 'column',
          gap: '1.25rem',
          background: isRecording
            ? 'linear-gradient(180deg, hsl(var(--destructive) / .04) 0%, transparent 100%)'
            : 'transparent',
          transition: 'background .4s',
          position: 'relative'
        }}>

          {/* Waveform */}
          <div style={{ width: '100%' }}>
            <WaveformVisualizer
              analyser={recorder.analyser}
              isActive={isRecording}
              height={80}
            />
          </div>

          {/* Timer */}
          <div style={{
            textAlign: 'center',
            fontSize: '2.8rem',
            fontWeight: 700,
            fontFamily: 'JetBrains Mono, monospace',
            color: isRecording ? 'hsl(var(--destructive))' : stage === 'done' ? 'hsl(var(--success))' : 'hsl(var(--pencil))',
            letterSpacing: '.08em',
            lineHeight: 1,
            textShadow: isRecording ? '0 0 28px hsl(var(--destructive) / .4)' : stage === 'done' ? '0 0 20px hsl(var(--success) / .3)' : 'none',
            transition: 'all .3s'
          }}>
            {recorder.formattedDuration}
          </div>

          {/* Status indicators */}
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '10px' }}>

            {isRecording && (
              <div className="animate-bounce-in" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{
                  display: 'flex', alignItems: 'center', gap: '8px',
                  padding: '.4rem 1rem',
                  background: 'hsl(var(--destructive) / .12)',
                  border: '1.5px solid hsl(var(--destructive) / .3)',
                  borderRadius: '999px',
                  fontSize: '.82rem', fontWeight: 600,
                  color: 'hsl(var(--destructive))',
                  fontFamily: 'Inter, sans-serif',
                  boxShadow: '0 0 12px hsl(var(--destructive) / .15)'
                }}>
                  <span style={{
                    width: 8, height: 8, borderRadius: '50%',
                    background: 'hsl(var(--destructive))',
                    display: 'inline-block',
                    boxShadow: '0 0 6px hsl(var(--destructive))'
                  }} className="animate-pulse-rec" />
                  RECORDING
                </span>
              </div>
            )}

            {overlapAlert && (
              <div className="animate-slide-up" style={{
                display: 'flex', alignItems: 'center', gap: '10px',
                padding: '.6rem 1.25rem',
                background: 'hsl(var(--destructive) / .12)',
                border: '1.5px solid hsl(var(--destructive) / .45)',
                borderRadius: '12px',
                fontSize: '.84rem', fontWeight: 600,
                color: 'hsl(var(--destructive))',
                fontFamily: 'Inter, sans-serif',
                maxWidth: '400px',
                boxShadow: '0 0 16px hsl(var(--destructive) / .12)',
              }}>
                <AlertTriangle size={16} style={{ flexShrink: 0 }} />
                <span>âš ï¸ Cross-talk detected â€” please speak one at a time</span>
              </div>
            )}

            {uploadError && (
              <div className="animate-shake" style={{
                display: 'flex', alignItems: 'center', gap: '8px',
                color: 'hsl(var(--destructive))',
                fontSize: '.86rem',
                justifyContent: 'center',
                fontFamily: 'Inter, sans-serif',
                fontWeight: 500
              }}>
                <AlertTriangle size={16} /> {uploadError}
              </div>
            )}
          </div>

          {/* Controls */}
          <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '18px' }}>
            {stage === 'idle' && (
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '10px' }}>
                <button
                  className="record-btn idle"
                  onClick={() => { recorder.start(); setStage('recording') }}
                  id="main-record-btn"
                  title="Start recording"
                >
                  <Mic size={38} color="hsl(var(--accent-foreground))" />
                </button>
                <span style={{ fontSize: '.78rem', color: 'hsl(var(--pencil))', fontFamily: 'Inter, sans-serif' }}>
                  Click to record
                </span>
              </div>
            )}
            {isRecording && (
              <button
                className="record-btn recording"
                onClick={handleStop}
                id="main-stop-btn"
                title="Stop recording"
                style={{ position: 'relative' }}
              >
                {/* Pulsing concentric rings */}
                <span className="record-ring-1" style={{ position: 'absolute', inset: '-18px', borderRadius: '50%', border: '2px solid hsl(var(--destructive) / .3)', animation: 'recording-ring-pulse 2.4s ease-out infinite', animationDelay: '0s' }} />
                <span className="record-ring-2" style={{ position: 'absolute', inset: '-34px', borderRadius: '50%', border: '2px solid hsl(var(--destructive) / .2)', animation: 'recording-ring-pulse 2.4s ease-out infinite', animationDelay: '0.55s' }} />
                <span className="record-ring-3" style={{ position: 'absolute', inset: '-50px', borderRadius: '50%', border: '2px solid hsl(var(--destructive) / .1)', animation: 'recording-ring-pulse 2.4s ease-out infinite', animationDelay: '1.1s' }} />
                <Square size={32} color="hsl(var(--accent-foreground))" fill="hsl(var(--accent-foreground))" />
              </button>
            )}
            {stage === 'stopped' && (
              <div className="animate-slide-up" style={{ display: 'flex', flexDirection: 'column', gap: '12px', alignItems: 'stretch', width: '100%', maxWidth: '520px' }}>
                <AdvancedOptionsPanel onChange={setAdvancedOpts} />
                <div style={{ display: 'flex', gap: '12px', justifyContent: 'center' }}>
                  <button className="btn btn-ghost" onClick={handleReset}>
                    <RotateCcw size={15} /> Discard
                  </button>
                  <button className="btn btn-primary" onClick={handleSubmit} id="submit-btn">
                    Analyse Recording
                  </button>
                </div>
              </div>
            )}
            {processing && (
              <button className="record-btn idle" disabled style={{ opacity: .5, cursor: 'not-allowed' }}>
                <Loader size={32} color="hsl(var(--accent-foreground))" className="spin" />
              </button>
            )}
            {(stage === 'done' || stage === 'error') && (
              <button className="btn btn-ghost animate-bounce-in" onClick={handleReset}>
                <RotateCcw size={15} /> New Recording
              </button>
            )}
          </div>

          {/* Audio preview */}
          {recorder.audioUrl && stage === 'stopped' && (
            <div className="animate-slide-up">
              <audio
                src={recorder.audioUrl}
                controls
                style={{
                  width: '100%',
                  height: 40,
                  accentColor: 'hsl(var(--accent))',
                  borderRadius: '10px',
                }}
              />
            </div>
          )}
        </div>)}

        {/* Transcript */}
        <div className="transcript-scroll" style={{
          flex: 1,
          overflowY: 'auto',
          padding: '1.25rem 1.5rem',
          background: 'hsl(var(--paper) / .4)'
        }}>
          {result?.transcript?.length > 0 && (
            <div className="transcript-subheader animate-slide-up">
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <h3 style={{ fontSize: '1rem', fontWeight: 700, color: 'hsl(var(--ink))', fontFamily: 'Inter, sans-serif', letterSpacing: '-.01em', margin: 0 }}>
                  Transcript
                </h3>
                <span style={{ fontSize: '.72rem', fontWeight: 600, color: 'hsl(var(--pencil))', background: 'hsl(var(--muted))', padding: '.15rem .5rem', borderRadius: '999px', fontFamily: 'Inter, sans-serif' }}>
                  {result.transcript.length} segments
                </span>
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
          <TranscriptViewer segments={result?.transcript || []} showConfidence={showConfidence}/>
        </div>
      </div>

      {/* AI Insights */}
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
        recordingId={recordingId}
        summary={result?.summary}
        shortSummary={result?.short_summary as string | undefined}
        detailedSummary={result?.detailed_summary as string | undefined}
        keyPoints={result?.key_points}
        actionItems={result?.action_items}
        speakerSummary={result?.speaker_summary}
        isOpen={chatOpen}
        onToggle={() => setChatOpen((o) => !o)}
        isGenerating={isGeneratingAI}
      />
      </div>
    </div>
  )
}
