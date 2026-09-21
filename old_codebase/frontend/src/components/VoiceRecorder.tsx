import { useState, useRef, useEffect } from 'react'
import { Mic, Square, RotateCcw, CheckCircle, AlertTriangle, Loader, Upload, FileAudio } from 'lucide-react'
import { useAudioRecorder } from '../hooks/useAudioRecorder'
import WaveformVisualizer from './WaveformVisualizer'
import api from '../api/client'
import { getApiErrorDetail } from '../lib/errors'

interface Props {
  label?: string
  sampleIndex?: number
  onSampleSaved?: (filePath: string, sampleIndex: number) => void
  compact?: boolean
}

export default function VoiceRecorder({ label = 'self', sampleIndex = 0, onSampleSaved, compact = false }: Props) {
  const { state, formattedDuration, audioBlob, audioUrl, analyser, error, start, stop, reset } = useAudioRecorder()
  const [mode, setMode] = useState<'record' | 'upload'>('record')
  
  // Upload mode states
  const [uploadedFile, setUploadedFile] = useState<File | null>(null)
  const [uploadedUrl, setUploadedUrl] = useState<string | null>(null)
  const [dragActive, setDragActive] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const [uploading, setUploading] = useState(false)
  const [saved, setSaved] = useState(false)
  const [uploadError, setUploadError] = useState<string | null>(null)

  useEffect(() => {
    return () => {
      if (uploadedUrl) {
        URL.revokeObjectURL(uploadedUrl)
      }
    }
  }, [uploadedUrl])

  const handleSave = async (fileOrBlob: Blob | File, fileName: string) => {
    setUploading(true)
    setUploadError(null)
    try {
      const form = new FormData()
      form.append('file', fileOrBlob, fileName)
      form.append('label', label)
      form.append('sample_index', String(sampleIndex))

      const res = await api.post('/voice/sample', form, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      setSaved(true)
      onSampleSaved?.(res.data.file_path, sampleIndex)
    } catch (e: unknown) {
      setUploadError(getApiErrorDetail(e, 'Upload failed.'))
    } finally {
      setUploading(false)
    }
  }

  const handleReset = () => {
    if (mode === 'record') {
      reset()
    } else {
      setUploadedFile(null)
      if (uploadedUrl) {
        URL.revokeObjectURL(uploadedUrl)
        setUploadedUrl(null)
      }
    }
    setSaved(false)
    setUploadError(null)
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setUploadedFile(file)
      const url = URL.createObjectURL(file)
      setUploadedUrl(url)
      setSaved(false)
      setUploadError(null)
    }
  }

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true)
    } else if (e.type === "dragleave") {
      setDragActive(false)
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    const file = e.dataTransfer.files?.[0]
    if (file && file.type.startsWith('audio/')) {
      setUploadedFile(file)
      const url = URL.createObjectURL(file)
      setUploadedUrl(url)
      setSaved(false)
      setUploadError(null)
    } else {
      setUploadError("Please drop a valid audio file.")
    }
  }

  const fmtSize = (b: number) => b > 1024 * 1024 ? `${(b / 1024 / 1024).toFixed(1)} MB` : `${(b / 1024).toFixed(0)} KB`

  return (
    <div className="glass" style={{ padding: compact ? '1rem' : '1.5rem', borderRadius: '12px' }}>
      
      {/* Mode Selector */}
      {!saved && state === 'idle' && !uploadedFile && (
        <div style={{
          display: 'flex',
          background: 'hsl(var(--muted))',
          padding: '4px',
          borderRadius: '8px',
          marginBottom: '1rem',
        }}>
          <button
            onClick={() => setMode('record')}
            style={{
              flex: 1,
              padding: '.5rem',
              borderRadius: '6px',
              border: 'none',
              background: mode === 'record' ? 'hsl(var(--card))' : 'transparent',
              color: mode === 'record' ? 'hsl(var(--accent))' : 'hsl(var(--pencil))',
              fontWeight: 600,
              fontSize: '.85rem',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '6px',
              transition: 'all 0.2s',
            }}
          >
            <Mic size={14} /> Record
          </button>
          <button
            onClick={() => setMode('upload')}
            style={{
              flex: 1,
              padding: '.5rem',
              borderRadius: '6px',
              border: 'none',
              background: mode === 'upload' ? 'hsl(var(--card))' : 'transparent',
              color: mode === 'upload' ? 'hsl(var(--accent))' : 'hsl(var(--pencil))',
              fontWeight: 600,
              fontSize: '.85rem',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '6px',
              transition: 'all 0.2s',
            }}
          >
            <Upload size={14} /> Upload File
          </button>
        </div>
      )}

      {mode === 'record' ? (
        <>
          {/* Waveform */}
          <div style={{ marginBottom: '1rem' }}>
            <WaveformVisualizer analyser={analyser} isActive={state === 'recording'} />
          </div>

          {/* Timer */}
          <div style={{
            textAlign: 'center',
            fontSize: '1.6rem', fontWeight: 700,
            fontFamily: 'JetBrains Mono, monospace',
            color: state === 'recording' ? 'hsl(var(--destructive))' : 'hsl(var(--pencil))',
            marginBottom: '1rem'
          }}>
            {formattedDuration}
          </div>

          {/* Controls */}
          <div style={{ display: 'flex', gap: '10px', justifyContent: 'center', flexWrap: 'wrap' }}>
            {state === 'idle' && !saved && (
              <button className="btn btn-primary" onClick={start} id={`record-start-${sampleIndex}`}>
                <Mic size={15} /> Start Recording
              </button>
            )}
            {state === 'recording' && (
              <button className="btn btn-danger" onClick={stop} id={`record-stop-${sampleIndex}`}>
                <Square size={15} /> Stop
              </button>
            )}
            {state === 'stopped' && !saved && (
              <>
                <button className="btn btn-ghost" onClick={handleReset}>
                  <RotateCcw size={15} /> Re-record
                </button>
                <button 
                  className="btn btn-primary" 
                  onClick={() => audioBlob && handleSave(audioBlob, `sample_${sampleIndex}.webm`)} 
                  disabled={uploading}
                >
                  {uploading ? <Loader size={15} className="spin" /> : <CheckCircle size={15} />}
                  {uploading ? 'Saving…' : 'Save Sample'}
                </button>
              </>
            )}
            {saved && (
              <>
                <span className="badge badge-green" style={{ padding: '0.4rem 1rem' }}>
                  <CheckCircle size={13} style={{ marginRight: '4px' }} /> Saved
                </span>
                <button className="btn btn-ghost" onClick={handleReset}>
                  <RotateCcw size={15} /> Re-record
                </button>
              </>
            )}
          </div>

          {/* Audio preview */}
          {audioUrl && state === 'stopped' && (
            <div style={{ marginTop: '1rem' }}>
              <audio
                src={audioUrl}
                controls
                style={{
                  width: '100%', height: '36px',
                  accentColor: 'hsl(var(--accent))',
                  borderRadius: '8px'
                }}
              />
            </div>
          )}
        </>
      ) : (
        <>
          {/* Upload Area */}
          {!uploadedFile && !saved ? (
            <div
              onDragEnter={handleDrag}
              onDragOver={handleDrag}
              onDragLeave={handleDrag}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              style={{
                border: `2px dashed ${dragActive ? 'hsl(var(--accent))' : 'hsl(var(--ink) / .15)'}`,
                borderRadius: '8px',
                padding: '2rem 1.5rem',
                textAlign: 'center',
                cursor: 'pointer',
                background: dragActive ? 'hsl(var(--accent) / .05)' : 'transparent',
                transition: 'all 0.2s',
                marginBottom: '1rem',
              }}
            >
              <Upload size={24} style={{ color: 'hsl(var(--accent))', marginBottom: '.5rem' }} />
              <p style={{ fontWeight: 600, fontSize: '.9rem', color: 'hsl(var(--ink))', marginBottom: '.25rem' }}>
                Drag & drop audio sample or click to browse
              </p>
              <p style={{ fontSize: '.78rem', color: 'hsl(var(--pencil))' }}>
                Supports WAV, MP3, M4A, WEBM
              </p>
              <input
                ref={fileInputRef}
                type="file"
                accept="audio/*"
                hidden
                onChange={handleFileChange}
              />
            </div>
          ) : (
            uploadedFile && (
              <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: '12px',
                padding: '1rem',
                background: 'hsl(var(--card))',
                border: '1.5px solid hsl(var(--ink) / .1)',
                borderRadius: '8px',
                marginBottom: '1rem',
              }}>
                <FileAudio size={20} style={{ color: 'hsl(var(--accent))' }} />
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontWeight: 600, fontSize: '.85rem', color: 'hsl(var(--ink))', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {uploadedFile.name}
                  </div>
                  <div style={{ fontSize: '.75rem', color: 'hsl(var(--pencil))', fontFamily: 'JetBrains Mono, monospace' }}>
                    {fmtSize(uploadedFile.size)}
                  </div>
                </div>
              </div>
            )
          )}

          {/* Controls for upload */}
          <div style={{ display: 'flex', gap: '10px', justifyContent: 'center' }}>
            {uploadedFile && !saved && (
              <>
                <button className="btn btn-ghost" onClick={handleReset}>
                  <RotateCcw size={15} /> Clear
                </button>
                <button 
                  className="btn btn-primary" 
                  onClick={() => uploadedFile && handleSave(uploadedFile, uploadedFile.name)} 
                  disabled={uploading}
                >
                  {uploading ? <Loader size={15} className="spin" /> : <CheckCircle size={15} />}
                  {uploading ? 'Saving…' : 'Save Sample'}
                </button>
              </>
            )}
            {saved && (
              <>
                <span className="badge badge-green" style={{ padding: '0.4rem 1rem' }}>
                  <CheckCircle size={13} style={{ marginRight: '4px' }} /> Saved
                </span>
                <button className="btn btn-ghost" onClick={handleReset}>
                  <RotateCcw size={15} /> Upload Another
                </button>
              </>
            )}
          </div>

          {/* Audio preview for upload */}
          {uploadedUrl && !saved && (
            <div style={{ marginTop: '1rem' }}>
              <audio
                src={uploadedUrl}
                controls
                style={{
                  width: '100%', height: '36px',
                  accentColor: 'hsl(var(--accent))',
                  borderRadius: '8px'
                }}
              />
            </div>
          )}
        </>
      )}

      {/* Errors */}
      {(error || uploadError) && (
        <div style={{
          marginTop: '0.75rem',
          display: 'flex', alignItems: 'center', gap: '6px',
          color: 'hsl(var(--destructive))',
          fontSize: '0.82rem',
          fontFamily: 'Inter, sans-serif',
          fontWeight: 500
        }}>
          <AlertTriangle size={14} /> {error || uploadError}
        </div>
      )}
    </div>
  )
}
