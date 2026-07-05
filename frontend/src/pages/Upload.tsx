import { useState, useRef, useCallback, useEffect } from 'react'
import { Upload, FileAudio, X, Loader, CheckCircle, CloudUpload, RotateCcw } from 'lucide-react'
import { toast } from 'sonner'
import { useJobPoller } from '../hooks/useJobPoller'
import TranscriptViewer from '../components/TranscriptViewer'
import AIChatPanel from '../components/AIChatPanel'
import ProcessingOverlay from '../components/ProcessingOverlay'
import PDFButton from '../components/PDFButton'
import AdvancedOptionsPanel, { type AdvancedOptions } from '../components/AdvancedOptions'
import { useProcessingStore, type ProcessingStage } from '../store/processing'
import api from '../api/client'
import { getApiErrorDetail } from '../lib/errors'
import type { ProcessingResult } from '../types/recording'
import { useJobsStore } from '../store/jobs'

export default function UploadPage() {
  const [showConfidence, setShowConfidence] = useState(true);
  const [file, setFile] = useState<File | null>(null)
  const [dragging, setDragging] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [recordingId, setRecordingId] = useState<string | null>(null)
  const [audioUrl, setAudioUrl] = useState<string | null>(null)
  const [result, setResult] = useState<ProcessingResult | null>(null)
  const [error, setError] = useState('')

  useEffect(() => {
    if (!file) {
      setAudioUrl(null)
      return
    }
    const url = URL.createObjectURL(file)
    setAudioUrl(url)
    return () => {
      URL.revokeObjectURL(url)
    }
  }, [file])

  const [chatOpen, setChatOpen] = useState(true)
  const [isGeneratingMom, setIsGeneratingMom] = useState(false)
  const [momData, setMomData] = useState<Record<string, unknown> | null>(null)
  const [isGeneratingInsights, setIsGeneratingInsights] = useState(false)
  const [advancedOpts, setAdvancedOpts] = useState<AdvancedOptions>({
    meetingPrompt: '', expectedSpeakers: null, selectedVoiceIds: [],
    useDictionary: false, useVocabularyInPrompt: false, speakerSummary: false,
  })
  const fileRef = useRef<HTMLInputElement>(null)

  const { setProcessing, updateStage, clearProcessing, stage, startedAt, source } = useProcessingStore()
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
      const delta = dragStartX.current - ev.clientX
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

  const onTranscriptReady = useCallback((data: Partial<ProcessingResult>) => {
    setResult(prev => ({ ...(prev ?? {}), ...data } as ProcessingResult))
    setIsGeneratingMom(true)

    clearProcessing()
  }, [clearProcessing])






  const onDone = useCallback((data: ProcessingResult) => {
    setResult(prev => ({ ...(prev ?? {}), ...data } as ProcessingResult))
    setIsGeneratingMom(false)
    clearProcessing()
  }, [clearProcessing])

  const shouldPoll = recordingId && (!result || isGeneratingMom)
  const jobData = useJobPoller(shouldPoll ? recordingId : null, { onTranscriptReady, onDone })

  useEffect(() => {
    const priorJob = useJobsStore.getState().jobs.find((j) => j.source === 'upload')
    if (priorJob) {
      setRecordingId(priorJob.jobId)
      if (priorJob.status === 'done') {
        setIsGeneratingMom(false)
      } else if (priorJob.status === 'transcript_ready') {
        setIsGeneratingMom(true)
      }
      if (priorJob.result) {
        setResult(priorJob.result as ProcessingResult)
      }
      setProcessing('upload', priorJob.stage as ProcessingStage || 'queued', new Date(priorJob.startedAt).getTime())
    }
  }, [setProcessing]) // eslint-disable-line react-hooks/exhaustive-deps

  const jobs = useJobsStore((s) => s.jobs)
  const currentJob = jobs.find((j) => j.jobId === recordingId)

  useEffect(() => {
    if (!currentJob) return
    if (currentJob.status === 'cancelled') {
      setFile(null)
      setRecordingId(null)
      setResult(null)
      setError('')
      setIsGeneratingMom(false)
      clearProcessing()
    }
  }, [currentJob, clearProcessing])

  useEffect(() => {
    if (jobData?.progress) {
      updateStage(jobData.progress as ProcessingStage)
    }
    if (jobData?.status === 'error') {
      clearProcessing()
    }
  }, [jobData?.progress, jobData?.status, updateStage, clearProcessing])

  useEffect(() => {
    return () => {
      clearProcessing()
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const handleFile = (f: File) => {
    setFile(f)
    setResult(null)
    setRecordingId(null)
    setError('')
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault(); setDragging(false)
    const f = e.dataTransfer.files[0]
    if (f) handleFile(f)
  }

  const handleCancel = async () => {
    if (!recordingId) return
    try {
      await api.post(`/audio/jobs/${recordingId}/cancel`)
      useJobsStore.getState().removeJob(recordingId)
      setFile(null)
      setRecordingId(null)
      setResult(null)
      setError('')
      setIsGeneratingMom(false)
      clearProcessing()
      toast.success('Processing cancelled successfully')
    } catch (err: unknown) {
      console.error('[Cancel] Failed:', err)
      toast.error('Failed to cancel processing')
    }
  }

  const handleUpload = async () => {
    if (!file) return
    setUploading(true); setError('')
    setProcessing('upload', 'uploading')
    try {
      const form = new FormData()
      form.append('file', file)
      form.append('meeting_prompt', advancedOpts.meetingPrompt)
      form.append('participant_voice_ids', JSON.stringify(advancedOpts.selectedVoiceIds))
      form.append('use_vocabulary', advancedOpts.useVocabularyInPrompt ? 'true' : 'false')
      form.append('speaker_summary', advancedOpts.speakerSummary ? 'true' : 'false')
      const res = await api.post('/audio/upload', form)
      setRecordingId(res.data.recording_id)
      updateStage('queued')
    } catch (e: unknown) {
      setError(getApiErrorDetail(e, 'Upload failed.'))
      clearProcessing()
    } finally {
      setUploading(false)
    }
  }

  // Fetch MoM when the pipeline completes or is already complete
  useEffect(() => {
    if (!recordingId || !result) return
    const isTerminal = currentJob?.status === 'done' || currentJob?.status === 'error'
    if (isTerminal) {
      api.get(`/mom/${recordingId}`)
        .then((r) => setMomData(r.data))
        .catch(() => setMomData(null))
    }
  }, [recordingId, result, currentJob?.status])

  const handleGenerateInsights = useCallback(async (tasks: string[]) => {
    if (!recordingId || isGeneratingInsights) return
    setIsGeneratingInsights(true)
    try {
      const res = await api.post(`/history/${recordingId}/generate-insights`, { tasks })
      setResult((prev) => prev ? {
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
  }, [recordingId, isGeneratingInsights])

  const isProcessingActive = useProcessingStore((s) => s.isProcessing && s.source === 'upload')
  const fmtSize = (b: number) => b > 1024 * 1024 ? `${(b / 1024 / 1024).toFixed(1)} MB` : `${(b / 1024).toFixed(0)} KB`

  const chatW = chatOpen ? `${chatWidth}px` : '48px'

  return (
    <div className="workspace-split" style={{
      display: 'grid',
      gridTemplateColumns: `1fr ${chatW}`,
      transition: isDragging.current ? 'none' : 'grid-template-columns .25s ease',
    }}>
      <div className="center-panel" style={{ position: 'relative' }}>

        {isProcessingActive && (
          <ProcessingOverlay stage={stage} startedAt={startedAt} source={source} onCancel={handleCancel} />
        )}

        <div className="panel-header">
          <div style={{
            width: '34px', height: '34px', borderRadius: '10px', flexShrink: 0,
            background: result ? 'hsl(var(--success) / .12)' : 'hsl(var(--accent) / .12)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            border: `2px solid ${result ? 'hsl(var(--success) / .3)' : 'hsl(var(--accent) / .3)'}`,
            transition: 'all .3s',
          }}>
            {result
              ? <CheckCircle size={16} style={{ color: 'hsl(var(--success))' }} />
              : <CloudUpload size={16} style={{ color: 'hsl(var(--accent))' }} />}
          </div>
          <div style={{ flex: 1, minWidth: 0 }}>
            <h1>{result ? 'Transcript' : 'Upload Audio'}</h1>
            <p style={{ fontSize: '.82rem', color: 'hsl(var(--pencil))', fontFamily: 'Inter, sans-serif', fontWeight: 400, marginTop: '1px' }}>
              {result && file ? file.name : 'WAV · MP3 · MP4 · M4A · OGG · WEBM'}
            </p>
          </div>
          {result && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexShrink: 0 }}>
              <PDFButton
                recordingId={recordingId}
                filename={file?.name}
                variant="ghost"
              />
              <button
                className="btn btn-ghost animate-bounce-in"
                onClick={() => { setFile(null); setRecordingId(null); setResult(null); setError('') }}
                style={{ flexShrink: 0, fontSize: '.82rem', padding: '.4rem .85rem' }}
              >
                <RotateCcw size={14} /> New Upload
              </button>
            </div>
          )}
        </div>

        {!result && (
          <div className="capture-setup" style={{ padding: '1.75rem 2rem', borderBottom: '2px dashed hsl(var(--border))' }}>

            <div
              onDragEnter={() => setDragging(true)}
              onDragOver={(e) => e.preventDefault()}
              onDragLeave={() => setDragging(false)}
              onDrop={handleDrop}
              onClick={() => fileRef.current?.click()}
              style={{
                border: `2px dashed ${dragging ? 'hsl(var(--accent))' : 'hsl(var(--ink) / .2)'}`,
                borderRadius: '14px',
                padding: '3rem 2rem',
                textAlign: 'center',
                cursor: 'pointer',
                background: dragging ? 'hsl(var(--accent) / .06)' : 'hsl(var(--card))',
                transition: 'all 0.25s cubic-bezier(0.4, 0, 0.2, 1)',
                marginBottom: '1.25rem',
                position: 'relative',
                overflow: 'hidden',
              }}
            >
              <div style={{
                position: 'absolute', top: '-50%', left: '-50%',
                width: '200%', height: '200%',
                background: 'radial-gradient(circle, hsl(var(--accent) / .08) 0%, transparent 65%)',
                opacity: dragging ? 1 : 0,
                transition: 'opacity .3s',
                pointerEvents: 'none'
              }} />

              <div style={{
                width: '56px', height: '56px',
                borderRadius: '14px',
                background: dragging ? 'hsl(var(--accent) / .2)' : 'hsl(var(--accent) / .1)',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                margin: '0 auto 1.25rem',
                border: `2px solid ${dragging ? 'hsl(var(--accent) / .5)' : 'hsl(var(--accent) / .2)'}`,
                transition: 'all .25s',
                position: 'relative'
              }}>
                <Upload
                  size={24}
                  style={{ color: 'hsl(var(--accent))', position: 'relative' }}
                  className={dragging ? 'animate-bounce-in' : 'animate-float'}
                />
              </div>

              <p style={{
                fontWeight: 700, marginBottom: '0.4rem', fontSize: '1rem',
                fontFamily: 'Inter, sans-serif', color: 'hsl(var(--ink))', position: 'relative'
              }}>
                {dragging ? 'Drop it here!' : 'Drop audio file here'}
              </p>
              <p style={{ fontSize: '0.85rem', color: 'hsl(var(--pencil))', fontFamily: 'Inter, sans-serif', position: 'relative' }}>
                or <span style={{ color: 'hsl(var(--accent))', fontWeight: 600 }}>click to browse</span>
              </p>
              <input ref={fileRef} type="file" accept="audio/*,video/*" hidden onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])} />
            </div>

            {file && (
              <div className="animate-slide-up" style={{
                display: 'flex', alignItems: 'center', gap: '12px',
                padding: '1rem 1.25rem', marginBottom: '1rem',
                background: 'hsl(var(--card))',
                border: '1.5px solid hsl(var(--ink) / .1)',
                borderLeft: '4px solid hsl(var(--accent))',
                borderRadius: '0 10px 10px 0',
              }}>
                <div style={{
                  width: '40px', height: '40px', borderRadius: '10px',
                  background: 'hsl(var(--accent) / .12)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  border: '1.5px solid hsl(var(--accent) / .25)', flexShrink: 0
                }}>
                  <FileAudio size={18} style={{ color: 'hsl(var(--accent))' }} />
                </div>
                <div style={{ flex: 1, overflow: 'hidden' }}>
                  <div style={{
                    fontWeight: 600, fontSize: '0.9rem',
                    overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                    color: 'hsl(var(--ink))', fontFamily: 'Inter, sans-serif', marginBottom: '3px'
                  }}>
                    {file.name}
                  </div>
                  <div style={{
                    fontSize: '0.78rem', color: 'hsl(var(--pencil))',
                    fontFamily: 'JetBrains Mono, monospace', fontWeight: 500
                  }}>
                    {fmtSize(file.size)}
                  </div>
                </div>
                <button onClick={() => { setFile(null); setRecordingId(null); setResult(null) }} className="icon-btn" style={{ flexShrink: 0 }}>
                  <X size={15} />
                </button>
              </div>
            )}

            {error && (
              <div className="animate-shake" style={{
                color: 'hsl(var(--destructive))', fontSize: '0.87rem', marginBottom: '1rem',
                padding: '.7rem 1rem',
                background: 'hsl(var(--destructive) / .08)',
                border: '1.5px solid hsl(var(--destructive) / .25)',
                borderRadius: '10px', fontFamily: 'Inter, sans-serif', fontWeight: 500,
                display: 'flex', alignItems: 'center', gap: '8px'
              }}>
                ?? {error}
              </div>
            )}

            {file && !recordingId && (
              <>
                <AdvancedOptionsPanel onChange={setAdvancedOpts} />
                <button
                  className="btn btn-primary animate-slide-up"
                  onClick={handleUpload}
                  disabled={uploading}
                  id="upload-btn"
                  style={{ width: '100%', justifyContent: 'center', padding: '.75rem 1.5rem', fontSize: '.95rem', marginTop: '1rem' }}
                >
                  {uploading ? <Loader size={16} className="spin" /> : <CloudUpload size={16} />}
                  {uploading ? 'Uploading…' : 'Process Audio'}
                </button>
              </>
            )}
          </div>
        )}

        <div className="transcript-scroll" style={{ flex: 1, overflowY: 'auto', padding: '1.25rem 1.5rem', background: 'hsl(var(--paper) / .4)' }}>
          {result?.transcript?.length > 0 && (
            <div className="animate-slide-up" style={{ marginBottom: '.75rem', display: 'flex', alignItems: 'center', gap: '10px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <h3 style={{ fontSize: '1rem', fontWeight: 700, color: 'hsl(var(--ink))', fontFamily: 'Inter, sans-serif', letterSpacing: '-.01em', margin: 0 }}>
                  Transcript
                </h3>
                <span style={{ fontSize: '.72rem', fontWeight: 600, color: 'hsl(var(--pencil))', background: 'hsl(var(--muted))', padding: '.15rem .5rem', borderRadius: '999px', fontFamily: 'Inter, sans-serif' }}>
                  {result.transcript.length} segments
                </span>
              </div>
              <div className="confidence-legend" style={{ display: 'flex', alignItems: 'center', gap: '16px', flexWrap: 'wrap' }}>
                <span style={{ fontSize: '.68rem', color: 'hsl(var(--pencil))', textTransform: 'uppercase', letterSpacing: '.08em', fontWeight: 700, fontFamily: 'Inter, sans-serif' }}>
                  Confidence
                </span>
                <span className="confidence-legend-item">
                  <span className="conf-dot" style={{ background: 'hsl(var(--sticky-green))' }} />
                  High
                </span>
                <span className="confidence-legend-item">
                  <span className="conf-dot" style={{ background: 'hsl(45,90%,50%)' }} />
                  Mid
                </span>
                <span className="confidence-legend-item">
                  <span className="conf-dot" style={{ background: 'hsl(var(--destructive))' }} />
                  Low
                </span>
                <div style={{ flex: 1 }} />
                <label className="confidence-switch">
                  <span>Highlight</span>
                  <input type="checkbox" checked={showConfidence} onChange={(e) => setShowConfidence(e.target.checked)} />
                  <span className="slider" />
                </label>
              </div>
            </div>
          )}
          <TranscriptViewer
            segments={result?.transcript || []}
            showConfidence={showConfidence}
            audioUrl={audioUrl || undefined}
            recordingId={recordingId || undefined}
            onSegmentsChange={(updated) => {
              if (result) {
                setResult({ ...result, transcript: updated });
              }
            }}
          />
        </div>
      </div>

      <div className={`insights-pane ${chatOpen ? 'is-open' : ''}`} style={{ position: 'relative', display: 'flex' }}>
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
          momData={momData as any}
          isOpen={chatOpen}
          onToggle={() => setChatOpen((o) => !o)}
          isGeneratingMom={isGeneratingMom}
          onGenerateInsights={handleGenerateInsights}
          isGeneratingInsights={isGeneratingInsights}
        />
      </div>
    </div>
  )
}
