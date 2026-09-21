/**
 * ExportPDFModal.tsx
 * A modal for exporting PDF reports with optional reference image/document attachments.
 */
import { useState, useRef, useCallback, useEffect } from 'react'
import { createPortal } from 'react-dom'
import { FileDown, X, ImageIcon, FileText, Loader, CheckCircle, AlertTriangle, Upload } from 'lucide-react'
import { downloadPdfReport } from '../services/pdfService'
import { getApiErrorDetail } from '../lib/errors'

interface AttachedImage {
  file: File
  preview: string
}

interface AttachedDoc {
  file: File
}

interface Props {
  recordingId: string | null | undefined
  filename?: string
  isOpen: boolean
  onClose: () => void
}

type ExportStatus = 'idle' | 'loading' | 'success' | 'error'

export default function ExportPDFModal({ recordingId, filename, isOpen, onClose }: Props) {
  const [images, setImages] = useState<AttachedImage[]>([])
  const [docs, setDocs] = useState<AttachedDoc[]>([])
  const [status, setStatus] = useState<ExportStatus>('idle')
  const [errorMsg, setErrorMsg] = useState<string | null>(null)
  const [includeTranscription, setIncludeTranscription] = useState(true)

  const imageInputRef = useRef<HTMLInputElement>(null)
  const docInputRef = useRef<HTMLInputElement>(null)
  const statusRef = useRef(status)
  statusRef.current = status

  useEffect(() => {
    if (!isOpen) return

    const previousOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && statusRef.current !== 'loading') onClose()
    }

    document.addEventListener('keydown', handleKeyDown)
    return () => {
      document.body.style.overflow = previousOverflow
      document.removeEventListener('keydown', handleKeyDown)
    }
  }, [isOpen, onClose])

  const handleImageFiles = useCallback((files: FileList | null) => {
    if (!files) return
    Array.from(files).forEach(file => {
      if (!file.type.startsWith('image/')) return
      const reader = new FileReader()
      reader.onload = (e) => {
        setImages(prev => [...prev, { file, preview: e.target?.result as string }])
      }
      reader.readAsDataURL(file)
    })
  }, [])

  const handleDocFiles = useCallback((files: FileList | null) => {
    if (!files) return
    Array.from(files).forEach(file => {
      const lower = file.name.toLowerCase()
      if (lower.endsWith('.pdf') || lower.endsWith('.docx')) {
        setDocs(prev => [...prev, { file }])
      }
    })
  }, [])

  const removeImage = (idx: number) => {
    setImages(prev => {
      const copy = [...prev]
      URL.revokeObjectURL(copy[idx].preview)
      copy.splice(idx, 1)
      return copy
    })
  }

  const removeDoc = (idx: number) => {
    setDocs(prev => {
      const copy = [...prev]
      copy.splice(idx, 1)
      return copy
    })
  }

  const handleExport = async () => {
    if (!recordingId || status === 'loading') return
    setStatus('loading')
    setErrorMsg(null)

    try {
      const formData = new FormData()
      images.forEach(img => formData.append('images', img.file))
      docs.forEach(doc => formData.append('documents', doc.file))
      formData.append('include_transcription', includeTranscription ? 'true' : 'false')

      await downloadPdfReport(recordingId, filename, formData)
      setStatus('success')
      setTimeout(() => {
        setStatus('idle')
        onClose()
      }, 1800)
    } catch (err: unknown) {
      setErrorMsg(getApiErrorDetail(err, 'PDF generation failed'))
      setStatus('error')
      setTimeout(() => { setStatus('idle'); setErrorMsg(null) }, 5000)
    }
  }

  const handleDragOver = (e: React.DragEvent) => { e.preventDefault() }

  const fmtSize = (b: number) => b > 1024 * 1024
    ? `${(b / 1024 / 1024).toFixed(1)} MB`
    : `${(b / 1024).toFixed(0)} KB`

  if (!isOpen) return null

  const overlay: React.CSSProperties = {
    position: 'fixed', inset: 0, zIndex: 9000,
    background: 'hsl(215 60% 6% / .75)',
    backdropFilter: 'blur(6px)',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    padding: 'clamp(.75rem, 3vw, 1.5rem)',
    animation: 'fadeIn .18s ease',
  }

  const modal: React.CSSProperties = {
    width: '100%', maxWidth: '560px',
    maxHeight: 'calc(100dvh - clamp(1.5rem, 6vw, 3rem))',
    background: 'hsl(var(--card))',
    border: '1.5px solid hsl(var(--border) / .3)',
    borderRadius: '16px',
    boxShadow: '0 24px 80px hsl(215 60% 4% / .6)',
    overflow: 'hidden',
    display: 'flex',
    flexDirection: 'column',
    animation: 'slideUp .22s ease',
  }

  return createPortal(
    <div
      style={overlay}
      onClick={(e) => {
        if (e.target === e.currentTarget && status !== 'loading') onClose()
      }}
      role="presentation"
    >
      <div
        style={modal}
        role="dialog"
        aria-modal="true"
        aria-labelledby="export-pdf-title"
        aria-describedby="export-pdf-description"
      >

        {/* Header */}
        <div style={{
          display: 'flex', alignItems: 'center', gap: '12px',
          flexShrink: 0,
          padding: '1rem 1.25rem',
          borderBottom: '1px solid hsl(var(--border) / .2)',
          background: 'linear-gradient(180deg, hsl(var(--card)) 0%, hsl(var(--paper) / .4) 100%)',
        }}>
          <div style={{
            width: '36px', height: '36px', borderRadius: '10px', flexShrink: 0,
            background: 'linear-gradient(135deg, hsl(var(--accent) / .18), hsl(var(--accent) / .06))',
            border: '1.5px solid hsl(var(--accent) / .3)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <FileDown size={18} style={{ color: 'hsl(var(--accent))' }} />
          </div>
          <div style={{ flex: 1, minWidth: 0 }}>
            <div id="export-pdf-title" style={{ fontFamily: 'Inter, sans-serif', fontWeight: 700, fontSize: '.92rem', color: 'hsl(var(--ink))' }}>
              Export Meeting Report
            </div>
            <div id="export-pdf-description" style={{ fontFamily: 'Inter, sans-serif', fontSize: '.68rem', color: 'hsl(var(--pencil))', marginTop: '2px' }}>
              Optionally attach reference images or documents
            </div>
          </div>
          <button
            onClick={onClose}
            className="icon-btn"
            aria-label="Close export PDF dialog"
            disabled={status === 'loading'}
            style={{ width: '32px', height: '32px', flexShrink: 0 }}
          >
            <X size={15} />
          </button>
        </div>

        {/* Body */}
        <div style={{ padding: '1.25rem', display: 'flex', flexDirection: 'column', gap: '16px', flex: 1, minHeight: 0, overflowY: 'auto' }}>

          {/* Include Transcription toggle */}
          <div style={{
            display: 'flex', alignItems: 'center', justifyContent: 'space-between',
            padding: '.75rem 1rem',
            borderRadius: '10px',
            background: includeTranscription ? 'hsl(var(--accent) / .07)' : 'hsl(var(--muted) / .4)',
            border: `1.5px solid ${includeTranscription ? 'hsl(var(--accent) / .3)' : 'hsl(var(--ink) / .1)'}`,
            cursor: 'pointer',
            transition: 'all .15s',
          }}
            onClick={() => setIncludeTranscription(v => !v)}
          >
            <div>
              <div style={{ fontFamily: 'Inter, sans-serif', fontWeight: 600, fontSize: '.85rem', color: 'hsl(var(--ink))' }}>
                Include Transcription
              </div>
              <div style={{ fontFamily: 'Inter, sans-serif', fontSize: '.72rem', color: 'hsl(var(--pencil))', marginTop: '2px' }}>
                {includeTranscription ? 'Full transcript will be included in the PDF' : 'Transcript sections will be excluded from the PDF'}
              </div>
            </div>
            {/* Toggle pill */}
            <div style={{
              width: 40, height: 22, borderRadius: 999, flexShrink: 0,
              background: includeTranscription ? 'hsl(var(--accent))' : 'hsl(var(--muted))',
              position: 'relative',
              transition: 'background .2s',
              border: `1.5px solid ${includeTranscription ? 'hsl(var(--accent) / .6)' : 'hsl(var(--ink) / .15)'}`,
            }}>
              <div style={{
                width: 14, height: 14, borderRadius: '50%',
                background: 'white',
                position: 'absolute',
                top: '50%',
                left: includeTranscription ? 'calc(100% - 17px)' : '3px',
                transform: 'translateY(-50%)',
                transition: 'left .2s',
                boxShadow: '0 1px 4px rgba(0,0,0,.2)',
              }} />
            </div>
          </div>

          {/* Image drop zone */}
          <div>
            <div style={{
              display: 'flex', alignItems: 'center', gap: '6px',
              marginBottom: '8px',
            }}>
              <ImageIcon size={14} style={{ color: 'hsl(var(--accent))' }} />
              <span style={{ fontFamily: 'Inter, sans-serif', fontSize: '.75rem', fontWeight: 700, color: 'hsl(var(--ink))', textTransform: 'uppercase', letterSpacing: '.07em' }}>
                Reference Images
              </span>
              <span style={{ fontFamily: 'Inter, sans-serif', fontSize: '.65rem', color: 'hsl(var(--pencil))' }}>
                (PNG, JPG — optional)
              </span>
            </div>

            <div
              onDragOver={handleDragOver}
              onDrop={(e) => { e.preventDefault(); handleImageFiles(e.dataTransfer.files) }}
              onClick={() => imageInputRef.current?.click()}
              style={{
                border: '1.5px dashed hsl(var(--border) / .4)',
                borderRadius: '10px',
                padding: '1rem',
                cursor: 'pointer',
                background: 'hsl(var(--muted) / .3)',
                textAlign: 'center',
                transition: 'all .2s',
                minHeight: images.length ? 'auto' : '80px',
                display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '6px',
              }}
              onMouseEnter={e => (e.currentTarget.style.borderColor = 'hsl(var(--accent) / .5)')}
              onMouseLeave={e => (e.currentTarget.style.borderColor = 'hsl(var(--border) / .4)')}
            >
              {images.length === 0 ? (
                <>
                  <Upload size={18} style={{ color: 'hsl(var(--pencil))' }} />
                  <span style={{ fontFamily: 'Inter, sans-serif', fontSize: '.75rem', color: 'hsl(var(--pencil))' }}>
                    Click or drag images here
                  </span>
                </>
              ) : (
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(80px, 1fr))', gap: '8px', width: '100%' }}>
                  {images.map((img, i) => (
                    <div key={i} style={{ position: 'relative', borderRadius: '8px', overflow: 'hidden', aspectRatio: '1', background: 'hsl(var(--muted))' }}>
                      <img src={img.preview} alt={img.file.name} style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }} />
                      <button
                        onClick={(e) => { e.stopPropagation(); removeImage(i) }}
                        aria-label={`Remove ${img.file.name}`}
                        style={{
                          position: 'absolute', top: '3px', right: '3px',
                          width: '20px', height: '20px',
                          borderRadius: '50%',
                          background: 'hsl(var(--destructive))',
                          border: 'none', cursor: 'pointer',
                          display: 'flex', alignItems: 'center', justifyContent: 'center',
                        }}
                      >
                        <X size={10} style={{ color: '#fff' }} />
                      </button>
                    </div>
                  ))}
                  <div style={{
                    borderRadius: '8px', aspectRatio: '1', border: '1.5px dashed hsl(var(--border) / .4)',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    background: 'hsl(var(--muted) / .3)',
                  }}>
                    <span style={{ fontSize: '.65rem', color: 'hsl(var(--pencil))' }}>+ Add</span>
                  </div>
                </div>
              )}
            </div>
            <input
              ref={imageInputRef}
              type="file"
              accept="image/png,image/jpeg,image/jpg"
              multiple
              style={{ display: 'none' }}
              onChange={(e) => handleImageFiles(e.target.files)}
            />
          </div>

          {/* Document drop zone */}
          <div>
            <div style={{
              display: 'flex', alignItems: 'center', gap: '6px',
              marginBottom: '8px',
            }}>
              <FileText size={14} style={{ color: 'hsl(var(--accent))' }} />
              <span style={{ fontFamily: 'Inter, sans-serif', fontSize: '.75rem', fontWeight: 700, color: 'hsl(var(--ink))', textTransform: 'uppercase', letterSpacing: '.07em' }}>
                Reference Documents
              </span>
              <span style={{ fontFamily: 'Inter, sans-serif', fontSize: '.65rem', color: 'hsl(var(--pencil))' }}>
                (PDF, DOCX — optional)
              </span>
            </div>

            {docs.length > 0 && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', marginBottom: '8px' }}>
                {docs.map((doc, i) => (
                  <div key={i} style={{
                    display: 'flex', alignItems: 'center', gap: '8px',
                    padding: '.5rem .75rem',
                    background: 'hsl(var(--muted) / .4)',
                    borderRadius: '8px',
                    border: '1px solid hsl(var(--border) / .2)',
                  }}>
                    <FileText size={14} style={{ color: 'hsl(var(--accent))', flexShrink: 0 }} />
                    <span style={{ fontFamily: 'Inter, sans-serif', fontSize: '.76rem', color: 'hsl(var(--ink))', flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {doc.file.name}
                    </span>
                    <span style={{ fontFamily: 'Inter, sans-serif', fontSize: '.65rem', color: 'hsl(var(--pencil))', flexShrink: 0 }}>
                      {fmtSize(doc.file.size)}
                    </span>
                    <button
                      onClick={() => removeDoc(i)}
                      className="icon-btn"
                      aria-label={`Remove ${doc.file.name}`}
                      style={{ width: '24px', height: '24px', flexShrink: 0 }}
                    >
                      <X size={11} />
                    </button>
                  </div>
                ))}
              </div>
            )}

            <div
              onDragOver={handleDragOver}
              onDrop={(e) => { e.preventDefault(); handleDocFiles(e.dataTransfer.files) }}
              onClick={() => docInputRef.current?.click()}
              style={{
                border: '1.5px dashed hsl(var(--border) / .4)',
                borderRadius: '10px',
                padding: '.9rem',
                cursor: 'pointer',
                background: 'hsl(var(--muted) / .3)',
                textAlign: 'center',
                display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px',
                transition: 'all .2s',
              }}
              onMouseEnter={e => (e.currentTarget.style.borderColor = 'hsl(var(--accent) / .5)')}
              onMouseLeave={e => (e.currentTarget.style.borderColor = 'hsl(var(--border) / .4)')}
            >
              <Upload size={14} style={{ color: 'hsl(var(--pencil))' }} />
              <span style={{ fontFamily: 'Inter, sans-serif', fontSize: '.75rem', color: 'hsl(var(--pencil))' }}>
                Click or drag PDF/DOCX files here
              </span>
            </div>
            <input
              ref={docInputRef}
              type="file"
              accept=".pdf,.docx"
              multiple
              style={{ display: 'none' }}
              onChange={(e) => handleDocFiles(e.target.files)}
            />
          </div>

          {/* Summary pill */}
          {(images.length > 0 || docs.length > 0) && (
            <div style={{
              display: 'flex', gap: '8px', flexWrap: 'wrap',
            }}>
              {images.length > 0 && (
                <span style={{
                  display: 'inline-flex', alignItems: 'center', gap: '4px',
                  padding: '.25rem .6rem', borderRadius: '20px',
                  background: 'hsl(var(--accent) / .1)',
                  border: '1px solid hsl(var(--accent) / .25)',
                  fontSize: '.68rem', fontFamily: 'Inter, sans-serif',
                  color: 'hsl(var(--accent))', fontWeight: 600,
                }}>
                  <ImageIcon size={10} />
                  {images.length} image{images.length !== 1 ? 's' : ''} attached
                </span>
              )}
              {docs.length > 0 && (
                <span style={{
                  display: 'inline-flex', alignItems: 'center', gap: '4px',
                  padding: '.25rem .6rem', borderRadius: '20px',
                  background: 'hsl(var(--accent) / .1)',
                  border: '1px solid hsl(var(--accent) / .25)',
                  fontSize: '.68rem', fontFamily: 'Inter, sans-serif',
                  color: 'hsl(var(--accent))', fontWeight: 600,
                }}>
                  <FileText size={10} />
                  {docs.length} document{docs.length !== 1 ? 's' : ''} attached
                </span>
              )}
            </div>
          )}

          {/* Error */}
          {status === 'error' && errorMsg && (
            <div style={{
              display: 'flex', alignItems: 'center', gap: '8px',
              padding: '.6rem .9rem',
              background: 'hsl(var(--destructive) / .1)',
              border: '1px solid hsl(var(--destructive) / .3)',
              borderRadius: '8px',
              fontSize: '.76rem', fontFamily: 'Inter, sans-serif',
              color: 'hsl(var(--destructive))',
            }}>
              <AlertTriangle size={14} />
              {errorMsg}
            </div>
          )}
        </div>

        {/* Footer */}
        <div style={{
          padding: '1rem 1.25rem',
          borderTop: '1px solid hsl(var(--border) / .2)',
          display: 'flex', gap: '10px', justifyContent: 'flex-end',
          flexShrink: 0,
          background: 'hsl(var(--paper) / .3)',
        }}>
          <button className="btn" onClick={onClose} disabled={status === 'loading'}>
            Cancel
          </button>
          <button
            id="pdf-export-confirm-btn"
            className="btn btn-primary"
            onClick={handleExport}
            disabled={!recordingId || status === 'loading'}
            style={{
              gap: '6px',
              ...(status === 'loading' && {
                background: 'linear-gradient(90deg, hsl(var(--muted)) 0%, hsl(var(--card)) 50%, hsl(var(--muted)) 100%)',
                backgroundSize: '200% 100%',
                animation: 'shimmer 1.5s ease-in-out infinite',
                color: 'hsl(var(--pencil))',
              }),
            }}
          >
            {status === 'loading' && <Loader size={14} className="spin" />}
            {status === 'success' && <CheckCircle size={14} style={{ color: 'hsl(var(--success))' }} />}
            {status === 'idle' && <FileDown size={14} />}
            {status === 'loading' ? 'Generating PDF…' : status === 'success' ? 'Downloaded!' : 'Export PDF'}
          </button>
        </div>

      </div>
    </div>,
    document.body,
  )
}
