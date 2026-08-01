import { useState, useRef, useEffect } from 'react'
import { Search, Copy, Download, Video, Check, Clock, RefreshCw, Loader } from 'lucide-react'
import { toast } from 'sonner'
import api from '../api/client'
import type { VideoTranscriptBlock } from '../types/recording'

interface Props {
  blocks: VideoTranscriptBlock[]
  filename?: string
  recordingId?: string
  onBlocksUpdated?: (blocks: VideoTranscriptBlock[]) => void
}

function fmtSec(secs: number): string {
  if (isNaN(secs) || secs < 0) return '0:00'
  const m = Math.floor(secs / 60)
  const s = Math.floor(secs % 60).toString().padStart(2, '0')
  return `${m}:${s}`
}

export default function VideoTranscriptViewer({ blocks, filename, recordingId, onBlocksUpdated }: Props) {
  const [searchQuery, setSearchQuery] = useState('')
  const [copied, setCopied] = useState(false)
  const [isRerunning, setIsRerunning] = useState(false)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const handleRerunOcr = async () => {
    if (!recordingId || isRerunning) return
    setIsRerunning(true)
    toast.info('Video OCR extraction started in background...')

    try {
      await api.post(`/video/jobs/${recordingId}/rerun-ocr`)
    } catch (err: any) {
      console.error('[Re-run Video OCR] Failed to start:', err)
      const msg = err?.response?.data?.detail || 'Failed to start Video OCR re-run'
      toast.error(msg)
      setIsRerunning(false)
      return
    }

    if (pollRef.current) clearInterval(pollRef.current)

    pollRef.current = setInterval(async () => {
      try {
        const res = await api.get(`/video/jobs/${recordingId}/ocr-status`)
        const status = res.data?.ocr_status
        if (status === 'done' || status === 'skipped') {
          if (pollRef.current) clearInterval(pollRef.current)
          pollRef.current = null

          const vtRes = await api.get(`/video/jobs/${recordingId}/video-transcript`)
          const newBlocks: VideoTranscriptBlock[] = vtRes.data?.video_transcript || []

          if (onBlocksUpdated) {
            onBlocksUpdated(newBlocks)
          }

          toast.success(
            status === 'done'
              ? `Video OCR complete! Extracted ${newBlocks.length} frame blocks.`
              : 'Video OCR complete: no text detected in video frames.'
          )
          setIsRerunning(false)
        } else if (status === 'error') {
          if (pollRef.current) clearInterval(pollRef.current)
          pollRef.current = null
          toast.error('Video OCR process failed.')
          setIsRerunning(false)
        }
      } catch (pollErr) {
        console.warn('[Re-run Video OCR] Poll failed (will retry):', pollErr)
      }
    }, 2000)
  }

  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current)
    }
  }, [])

  const filteredBlocks = blocks.filter(b => 
    !searchQuery.trim() || b.text.toLowerCase().includes(searchQuery.toLowerCase())
  )

  const handleCopy = () => {
    const fullText = blocks
      .map(b => `[${fmtSec(b.start)} - ${fmtSec(b.end)}]\n${b.text}`)
      .join('\n\n')
    navigator.clipboard.writeText(fullText)
    setCopied(true)
    toast.success('Video OCR transcript copied to clipboard')
    setTimeout(() => setCopied(false), 2500)
  }

  const handleDownload = () => {
    const fullText = blocks
      .map(b => `[${fmtSec(b.start)} - ${fmtSec(b.end)}]\n${b.text}`)
      .join('\n\n')
    const blob = new Blob([fullText], { type: 'text/plain;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${filename || 'video'}_ocr_transcript.txt`
    a.click()
    URL.revokeObjectURL(url)
    toast.success('Downloaded video transcript')
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', fontFamily: 'Inter, sans-serif' }}>
      {/* Subheader Toolbar */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        flexWrap: 'wrap', gap: '12px', paddingBottom: '.75rem',
        borderBottom: '1.5px solid hsl(var(--border) / .5)'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div style={{
            display: 'inline-flex', alignItems: 'center', gap: 6,
            background: 'hsl(210,80%,55%/.12)', color: 'hsl(210,85%,60%)',
            border: '1px solid hsl(210,80%,55%/.3)', padding: '.25rem .75rem',
            borderRadius: '999px', fontSize: '.8rem', fontWeight: 700
          }}>
            <Video size={14} /> Video Frame Transcription (OCR)
          </div>
          <span style={{
            fontSize: '.74rem', fontWeight: 600, color: 'hsl(var(--pencil))',
            background: 'hsl(var(--muted))', padding: '.15rem .55rem', borderRadius: '999px'
          }}>
            {blocks.length} frame blocks
          </span>
        </div>

        {/* Right side controls */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
          <div style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
            <Search size={13} style={{ position: 'absolute', left: 9, color: 'hsl(var(--pencil))' }} />
            <input
              type="text"
              placeholder="Search on-screen video text..."
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
              style={{
                padding: '.35rem .65rem .35rem 1.8rem', fontSize: '.78rem', borderRadius: 8,
                border: '1px solid hsl(var(--border) / .6)', background: 'hsl(var(--card))',
                color: 'hsl(var(--ink))', outline: 'none', width: 220, fontFamily: 'Inter, sans-serif'
              }}
            />
          </div>

          {recordingId && (
            <button
              onClick={handleRerunOcr}
              disabled={isRerunning}
              id="btn-rerun-video-ocr"
              aria-label="Re-run Video OCR"
              style={{
                display: 'inline-flex', alignItems: 'center', gap: 5,
                padding: '.35rem .75rem', borderRadius: 8,
                border: '1px solid hsl(210 80% 55% / .4)',
                background: 'hsl(210 80% 55% / .12)',
                color: 'hsl(210 85% 60%)', fontSize: '.74rem', fontWeight: 600,
                cursor: isRerunning ? 'not-allowed' : 'pointer',
                opacity: isRerunning ? 0.7 : 1,
                transition: 'all .15s ease'
              }}
              title="Re-run Video OCR frame extraction and text recognition"
            >
              {isRerunning ? <Loader size={12} className="spin" /> : <RefreshCw size={12} />}
              {isRerunning ? 'Re-running OCR...' : 'Re-run Video OCR'}
            </button>
          )}

          <button
            onClick={handleCopy}
            disabled={blocks.length === 0}
            style={{
              display: 'inline-flex', alignItems: 'center', gap: 4,
              padding: '.35rem .65rem', borderRadius: 8,
              border: '1px solid hsl(var(--border) / .6)', background: 'hsl(var(--card))',
              color: 'hsl(var(--ink))', fontSize: '.74rem', fontWeight: 600, cursor: 'pointer'
            }}
            title="Copy video transcript text"
          >
            {copied ? <Check size={12} style={{ color: 'hsl(140,70%,45%)' }} /> : <Copy size={12} />}
            {copied ? 'Copied' : 'Copy'}
          </button>

          <button
            onClick={handleDownload}
            disabled={blocks.length === 0}
            style={{
              display: 'inline-flex', alignItems: 'center', gap: 4,
              padding: '.35rem .65rem', borderRadius: 8,
              border: '1px solid hsl(var(--border) / .6)', background: 'hsl(var(--card))',
              color: 'hsl(var(--ink))', fontSize: '.74rem', fontWeight: 600, cursor: 'pointer'
            }}
            title="Download video OCR transcript text"
          >
            <Download size={12} /> TXT
          </button>
        </div>
      </div>

      {/* Block List */}
      {blocks.length === 0 ? (
        <div style={{ textAlign: 'center', padding: '3rem 1rem', color: 'hsl(var(--pencil))' }}>
          <Video size={36} style={{ margin: '0 auto .75rem', opacity: 0.4, color: 'hsl(210,85%,60%)' }} />
          <div style={{ fontSize: '.95rem', fontWeight: 700, color: 'hsl(var(--ink))', marginBottom: '.3rem' }}>
            No Video Transcription Data
          </div>
          <div style={{ fontSize: '.82rem', maxWidth: 400, margin: '0 auto .9rem' }}>
            No on-screen video frame text was extracted for this recording.
          </div>
          {recordingId && (
            <button
              onClick={handleRerunOcr}
              disabled={isRerunning}
              style={{
                display: 'inline-flex', alignItems: 'center', gap: 6,
                padding: '.45rem 1rem', borderRadius: 8,
                border: '1px solid hsl(210 80% 55% / .4)',
                background: 'hsl(210 80% 55% / .15)',
                color: 'hsl(210 85% 60%)', fontSize: '.8rem', fontWeight: 600,
                cursor: isRerunning ? 'not-allowed' : 'pointer',
                opacity: isRerunning ? 0.7 : 1
              }}
            >
              {isRerunning ? <Loader size={14} className="spin" /> : <RefreshCw size={14} />}
              {isRerunning ? 'Re-running OCR...' : 'Re-run Video OCR Pipeline'}
            </button>
          )}
        </div>
      ) : filteredBlocks.length === 0 ? (
        <div style={{ textAlign: 'center', padding: '2.5rem 1rem', color: 'hsl(var(--pencil))', fontSize: '.84rem' }}>
          No video OCR text matches your search query "{searchQuery}".
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '.85rem' }}>
          {filteredBlocks.map((b, idx) => (
            <div key={idx} style={{
              padding: '.85rem 1rem',
              borderRadius: 10,
              background: 'hsl(var(--paper) / .6)',
              border: '1px solid hsl(210 80% 55% / .2)',
              display: 'flex',
              flexDirection: 'column',
              gap: '.4rem',
              boxShadow: '0 1px 3px rgba(0,0,0,0.02)'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <span style={{
                  display: 'inline-flex', alignItems: 'center', gap: 4,
                  background: 'hsl(210 80% 55% / .12)', color: 'hsl(210 85% 60%)',
                  border: '1px solid hsl(210 80% 55% / .25)',
                  padding: '2px 8px', borderRadius: 8, fontSize: '.7rem', fontWeight: 700,
                  fontFamily: 'JetBrains Mono, monospace'
                }}>
                  <Clock size={10} /> {fmtSec(b.start)} – {fmtSec(b.end)}
                </span>
                <span style={{ fontSize: '.68rem', color: 'hsl(var(--pencil))', fontWeight: 600 }}>
                  Frame Block #{idx + 1}
                </span>
              </div>
              <pre style={{
                margin: 0,
                fontSize: '.82rem',
                lineHeight: 1.5,
                whiteSpace: 'pre-wrap',
                fontFamily: 'Inter, sans-serif',
                color: 'hsl(var(--ink))',
                background: 'hsl(var(--muted) / .25)',
                padding: '.6rem .75rem',
                borderRadius: 6,
                border: '1px solid hsl(var(--border) / .3)'
              }}>
                {b.text}
              </pre>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

