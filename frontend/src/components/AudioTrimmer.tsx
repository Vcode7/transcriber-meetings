/**
 * AudioTrimmer — Interactive audio/video trim component.
 *
 * Renders a waveform canvas using the Web Audio API, two draggable start/end
 * handle bars, a preview play button, and Process / Skip buttons.
 *
 * Props
 * ─────
 *   file          : File | Blob to decode and visualise
 *   fileName      : display name shown in the header
 *   onConfirm(s, e) : called when user clicks "Process Recording" with trim seconds
 *   onSkip()      : called when user clicks "Skip Trim" to use the full recording
 */

import { useState, useRef, useEffect, useCallback } from 'react'
import { Play, Pause, Scissors, SkipForward, Clock, Waveform } from 'lucide-react'

// ── Helpers ──────────────────────────────────────────────────────────────────

function fmtTime(sec: number): string {
  if (!isFinite(sec) || sec < 0) return '0:00'
  const h = Math.floor(sec / 3600)
  const m = Math.floor((sec % 3600) / 60)
  const s = Math.floor(sec % 60)
  if (h > 0) return `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
  return `${m}:${String(s).padStart(2, '0')}`
}

// ── Constants ─────────────────────────────────────────────────────────────────

const HANDLE_W = 14       // handle bar width px
const CANVAS_H = 96       // waveform canvas height px
const PEAK_BINS = 600     // number of amplitude buckets to draw

// ── Component ────────────────────────────────────────────────────────────────

interface AudioTrimmerProps {
  file: File | Blob
  fileName?: string
  onConfirm: (startSec: number, endSec: number) => void
  onSkip: () => void
}

export default function AudioTrimmer({ file, fileName, onConfirm, onSkip }: AudioTrimmerProps) {
  const [duration, setDuration] = useState(0)
  const [trimStart, setTrimStart] = useState(0)
  const [trimEnd, setTrimEnd] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [playHead, setPlayHead] = useState(0)
  const [peaks, setPeaks] = useState<number[]>([])
  const [loading, setLoading] = useState(true)
  const [decodeError, setDecodeError] = useState('')

  const canvasRef = useRef<HTMLCanvasElement>(null)
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const blobUrlRef = useRef<string | null>(null)
  const rafRef = useRef<number>(0)
  const containerRef = useRef<HTMLDivElement>(null)

  // Track dragging state
  const dragging = useRef<'start' | 'end' | null>(null)
  const containerRectRef = useRef<DOMRect | null>(null)

  // ── Decode audio → waveform peaks ─────────────────────────────────────────
  useEffect(() => {
    setLoading(true)
    setDecodeError('')
    let cancelled = false

    const url = URL.createObjectURL(file)
    blobUrlRef.current = url

    // Create audio element for playback
    const audio = new Audio(url)
    audio.preload = 'auto'
    audioRef.current = audio

    audio.addEventListener('loadedmetadata', () => {
      if (cancelled) return
      const dur = audio.duration
      setDuration(dur)
      setTrimStart(0)
      setTrimEnd(dur)
    })

    audio.addEventListener('timeupdate', () => {
      if (cancelled) return
      setPlayHead(audio.currentTime)
      if (audioRef.current && audioRef.current.currentTime >= trimEnd - 0.1) {
        audioRef.current.pause()
        setIsPlaying(false)
      }
    })

    audio.addEventListener('ended', () => {
      if (cancelled) return
      setIsPlaying(false)
    })

    // Decode for waveform via AudioContext
    const ctx = new AudioContext()
    file.arrayBuffer().then(buf => {
      if (cancelled) return
      return ctx.decodeAudioData(buf)
    }).then(decoded => {
      if (!decoded || cancelled) return
      const channel = decoded.getChannelData(0)
      const blockSize = Math.max(1, Math.floor(channel.length / PEAK_BINS))
      const out: number[] = []
      for (let i = 0; i < PEAK_BINS; i++) {
        const start = i * blockSize
        let max = 0
        for (let j = 0; j < blockSize; j++) {
          const v = Math.abs(channel[start + j] || 0)
          if (v > max) max = v
        }
        out.push(max)
      }
      // Normalize
      const globalMax = Math.max(...out, 0.001)
      setPeaks(out.map(v => v / globalMax))
      setLoading(false)
    }).catch(err => {
      if (cancelled) return
      console.warn('[AudioTrimmer] Waveform decode failed:', err)
      // Still show UI with placeholder waveform (all zeros)
      setPeaks(Array(PEAK_BINS).fill(0.05))
      setLoading(false)
    })

    return () => {
      cancelled = true
      audio.pause()
      ctx.close()
      URL.revokeObjectURL(url)
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [file])

  // ── Draw waveform ─────────────────────────────────────────────────────────
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || peaks.length === 0 || duration === 0) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const W = canvas.width
    const H = canvas.height
    const cx = H / 2

    ctx.clearRect(0, 0, W, H)

    // Background
    ctx.fillStyle = 'hsl(220 20% 8%)'
    ctx.fillRect(0, 0, W, H)

    // Clipped region (outside trim) — darker
    const sx = (trimStart / duration) * W
    const ex = (trimEnd / duration) * W

    // Muted region - left
    ctx.fillStyle = 'hsl(220 15% 6%)'
    ctx.fillRect(0, 0, sx, H)
    // Muted region - right
    ctx.fillStyle = 'hsl(220 15% 6%)'
    ctx.fillRect(ex, 0, W - ex, H)

    // Selected region highlight
    ctx.fillStyle = 'hsl(220 70% 50% / 0.08)'
    ctx.fillRect(sx, 0, ex - sx, H)

    // Draw peaks
    const barW = W / peaks.length
    peaks.forEach((amp, i) => {
      const x = i * barW
      const inRange = x >= sx && x <= ex
      const isHead = playHead > 0 && x <= (playHead / duration) * W

      // Color logic
      if (isHead && inRange) {
        ctx.fillStyle = 'hsl(220 90% 70%)'
      } else if (inRange) {
        ctx.fillStyle = 'hsl(220 70% 55%)'
      } else {
        ctx.fillStyle = 'hsl(220 20% 28%)'
      }

      const barH = Math.max(2, amp * (cx - 4))
      ctx.fillRect(x, cx - barH, Math.max(1, barW - 0.5), barH * 2)
    })

    // Trim handles — vertical lines
    ctx.strokeStyle = 'hsl(45 100% 60%)'
    ctx.lineWidth = 2
    ctx.setLineDash([4, 3])
    ctx.beginPath(); ctx.moveTo(sx, 0); ctx.lineTo(sx, H); ctx.stroke()
    ctx.beginPath(); ctx.moveTo(ex, 0); ctx.lineTo(ex, H); ctx.stroke()
    ctx.setLineDash([])

    // Play head
    if (playHead > 0) {
      const ph = (playHead / duration) * W
      ctx.strokeStyle = 'hsl(0 90% 65%)'
      ctx.lineWidth = 1.5
      ctx.beginPath(); ctx.moveTo(ph, 0); ctx.lineTo(ph, H); ctx.stroke()
    }
  }, [peaks, duration, trimStart, trimEnd, playHead])

  // ── Playback ──────────────────────────────────────────────────────────────
  const togglePlay = useCallback(() => {
    const audio = audioRef.current
    if (!audio) return
    if (isPlaying) {
      audio.pause()
      setIsPlaying(false)
    } else {
      if (audio.currentTime < trimStart || audio.currentTime >= trimEnd - 0.1) {
        audio.currentTime = trimStart
      }
      audio.play().catch(() => {})
      setIsPlaying(true)
    }
  }, [isPlaying, trimStart, trimEnd])

  // ── Canvas click → seek ───────────────────────────────────────────────────
  const handleCanvasClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!canvasRef.current || duration === 0) return
    const rect = canvasRef.current.getBoundingClientRect()
    const x = e.clientX - rect.left
    const frac = Math.max(0, Math.min(1, x / rect.width))
    const t = frac * duration
    if (audioRef.current) {
      audioRef.current.currentTime = t
      setPlayHead(t)
    }
  }, [duration])

  // ── Drag handles ──────────────────────────────────────────────────────────
  const getCanvasFrac = useCallback((clientX: number): number => {
    const rect = containerRectRef.current
    if (!rect) return 0
    // The canvas fills the container minus handle padding
    return Math.max(0, Math.min(1, (clientX - rect.left) / rect.width))
  }, [])

  const onMouseDown = useCallback((handle: 'start' | 'end') => (e: React.MouseEvent) => {
    e.preventDefault()
    dragging.current = handle
    if (containerRef.current) {
      containerRectRef.current = containerRef.current.getBoundingClientRect()
    }

    const onMove = (ev: MouseEvent) => {
      if (!dragging.current || duration === 0) return
      const frac = getCanvasFrac(ev.clientX)
      const t = frac * duration
      if (dragging.current === 'start') {
        setTrimStart(Math.min(t, trimEnd - 1))
      } else {
        setTrimEnd(Math.max(t, trimStart + 1))
      }
    }

    const onUp = () => {
      dragging.current = null
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
    }

    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [duration, trimEnd, trimStart, getCanvasFrac])

  // ── Manual time inputs ────────────────────────────────────────────────────
  const parseTime = (v: string): number | null => {
    const parts = v.split(':').map(Number)
    if (parts.some(isNaN)) return null
    if (parts.length === 3) return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if (parts.length === 2) return parts[0] * 60 + parts[1]
    return parts[0]
  }

  // ── Computed handle positions (%) ──────────────────────────────────────────
  const startPct = duration > 0 ? (trimStart / duration) * 100 : 0
  const endPct = duration > 0 ? (trimEnd / duration) * 100 : 100
  const selectedDuration = trimEnd - trimStart

  return (
    <div style={{
      background: 'hsl(220 18% 10%)',
      borderRadius: '16px',
      border: '1.5px solid hsl(220 30% 22%)',
      overflow: 'hidden',
      boxShadow: '0 8px 32px hsl(220 30% 5% / 0.6)',
      fontFamily: 'Inter, sans-serif',
    }}>

      {/* Header */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: '12px',
        padding: '1rem 1.25rem',
        borderBottom: '1px solid hsl(220 25% 18%)',
        background: 'hsl(220 20% 12%)',
      }}>
        <div style={{
          width: '34px', height: '34px', borderRadius: '10px', flexShrink: 0,
          background: 'hsl(220 70% 55% / 0.15)',
          border: '1.5px solid hsl(220 70% 55% / 0.3)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}>
          <Scissors size={16} style={{ color: 'hsl(220 70% 65%)' }} />
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ fontSize: '.9rem', fontWeight: 700, color: 'hsl(220 10% 92%)' }}>
            Edit Recording
          </div>
          <div style={{ fontSize: '.75rem', color: 'hsl(220 15% 55%)', marginTop: '1px' }}>
            {fileName ? `"${fileName}" — ` : ''}
            Trim the audio before processing, or skip to use the full recording
          </div>
        </div>
        <div style={{
          fontSize: '.72rem', fontWeight: 700,
          background: 'hsl(220 70% 55% / 0.1)',
          border: '1px solid hsl(220 70% 55% / 0.25)',
          color: 'hsl(220 70% 70%)',
          padding: '3px 10px', borderRadius: '999px',
          display: 'flex', alignItems: 'center', gap: '5px',
          whiteSpace: 'nowrap',
        }}>
          <Clock size={11} />
          {loading ? '...' : fmtTime(duration)}
        </div>
      </div>

      {/* Waveform area */}
      <div style={{ padding: '1.25rem 1.25rem 0' }}>
        {loading ? (
          <div style={{
            height: `${CANVAS_H}px`,
            background: 'hsl(220 20% 8%)',
            borderRadius: '10px',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            color: 'hsl(220 15% 45%)',
            fontSize: '.82rem',
            border: '1px solid hsl(220 25% 14%)',
          }}>
            <span style={{ animation: 'pulse 1.5s ease-in-out infinite' }}>Decoding waveform…</span>
          </div>
        ) : (
          <div ref={containerRef} style={{ position: 'relative', userSelect: 'none' }}>
            {/* Canvas */}
            <canvas
              ref={canvasRef}
              width={1200}
              height={CANVAS_H * 2}
              style={{
                width: '100%', height: `${CANVAS_H}px`,
                borderRadius: '10px',
                cursor: 'crosshair',
                display: 'block',
              }}
              onClick={handleCanvasClick}
            />

            {/* Start handle */}
            <div
              onMouseDown={onMouseDown('start')}
              style={{
                position: 'absolute',
                top: 0, bottom: 0,
                left: `calc(${startPct}% - ${HANDLE_W / 2}px)`,
                width: `${HANDLE_W}px`,
                cursor: 'ew-resize',
                display: 'flex', flexDirection: 'column', alignItems: 'center',
                justifyContent: 'center',
                zIndex: 10,
              }}
            >
              <div style={{
                width: '4px', height: '100%',
                background: 'hsl(45 100% 60%)',
                borderRadius: '2px',
                boxShadow: '0 0 8px hsl(45 100% 60% / 0.6)',
              }} />
              <div style={{
                position: 'absolute', top: '-24px',
                background: 'hsl(45 100% 60%)',
                color: 'hsl(45 100% 10%)',
                fontSize: '.64rem', fontWeight: 800,
                padding: '2px 6px', borderRadius: '6px',
                whiteSpace: 'nowrap',
              }}>
                {fmtTime(trimStart)}
              </div>
            </div>

            {/* End handle */}
            <div
              onMouseDown={onMouseDown('end')}
              style={{
                position: 'absolute',
                top: 0, bottom: 0,
                left: `calc(${endPct}% - ${HANDLE_W / 2}px)`,
                width: `${HANDLE_W}px`,
                cursor: 'ew-resize',
                display: 'flex', flexDirection: 'column', alignItems: 'center',
                justifyContent: 'center',
                zIndex: 10,
              }}
            >
              <div style={{
                width: '4px', height: '100%',
                background: 'hsl(45 100% 60%)',
                borderRadius: '2px',
                boxShadow: '0 0 8px hsl(45 100% 60% / 0.6)',
              }} />
              <div style={{
                position: 'absolute', top: '-24px',
                background: 'hsl(45 100% 60%)',
                color: 'hsl(45 100% 10%)',
                fontSize: '.64rem', fontWeight: 800,
                padding: '2px 6px', borderRadius: '6px',
                whiteSpace: 'nowrap',
              }}>
                {fmtTime(trimEnd)}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Time inputs row */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: '12px',
        padding: '1rem 1.25rem 0',
      }}>
        {/* Playback */}
        <button
          onClick={togglePlay}
          disabled={loading || duration === 0}
          style={{
            display: 'flex', alignItems: 'center', gap: '7px',
            padding: '0.5rem 1rem',
            borderRadius: '8px',
            background: isPlaying ? 'hsl(0 75% 55% / 0.15)' : 'hsl(220 70% 55% / 0.12)',
            border: `1.5px solid ${isPlaying ? 'hsl(0 75% 55% / 0.4)' : 'hsl(220 70% 55% / 0.3)'}`,
            color: isPlaying ? 'hsl(0 75% 65%)' : 'hsl(220 70% 68%)',
            fontSize: '.82rem', fontWeight: 600,
            cursor: loading ? 'not-allowed' : 'pointer',
            transition: 'all .18s',
          }}
        >
          {isPlaying ? <Pause size={14} /> : <Play size={14} />}
          {isPlaying ? 'Pause' : 'Preview'}
        </button>

        {/* Start time */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flex: 1 }}>
          <span style={{ fontSize: '.75rem', color: 'hsl(45 80% 55%)', fontWeight: 600, whiteSpace: 'nowrap' }}>
            Start
          </span>
          <input
            type="text"
            value={fmtTime(trimStart)}
            onChange={e => {
              const t = parseTime(e.target.value)
              if (t !== null && t >= 0 && t < trimEnd - 0.5) setTrimStart(t)
            }}
            style={{
              flex: 1, minWidth: 0, maxWidth: '80px',
              background: 'hsl(220 20% 8%)',
              border: '1.5px solid hsl(220 25% 22%)',
              color: 'hsl(220 10% 88%)',
              padding: '4px 8px', borderRadius: '6px',
              fontSize: '.8rem', fontFamily: 'JetBrains Mono, monospace',
              textAlign: 'center',
            }}
          />
        </div>

        <span style={{ color: 'hsl(220 15% 40%)', fontSize: '.8rem' }}>→</span>

        {/* End time */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flex: 1 }}>
          <span style={{ fontSize: '.75rem', color: 'hsl(45 80% 55%)', fontWeight: 600, whiteSpace: 'nowrap' }}>
            End
          </span>
          <input
            type="text"
            value={fmtTime(trimEnd)}
            onChange={e => {
              const t = parseTime(e.target.value)
              if (t !== null && t > trimStart + 0.5 && t <= duration) setTrimEnd(t)
            }}
            style={{
              flex: 1, minWidth: 0, maxWidth: '80px',
              background: 'hsl(220 20% 8%)',
              border: '1.5px solid hsl(220 25% 22%)',
              color: 'hsl(220 10% 88%)',
              padding: '4px 8px', borderRadius: '6px',
              fontSize: '.8rem', fontFamily: 'JetBrains Mono, monospace',
              textAlign: 'center',
            }}
          />
        </div>

        {/* Duration badge */}
        <div style={{
          fontSize: '.72rem', fontWeight: 700,
          background: 'hsl(140 60% 45% / 0.1)',
          border: '1px solid hsl(140 60% 45% / 0.25)',
          color: 'hsl(140 60% 55%)',
          padding: '3px 10px', borderRadius: '999px',
          whiteSpace: 'nowrap',
        }}>
          {fmtTime(selectedDuration)} selected
        </div>
      </div>

      {/* Action buttons */}
      <div style={{
        display: 'flex', gap: '10px',
        padding: '1.25rem',
        borderTop: '1px solid hsl(220 25% 14%)',
        marginTop: '1.25rem',
      }}>
        {/* Skip */}
        <button
          onClick={onSkip}
          style={{
            display: 'flex', alignItems: 'center', gap: '7px',
            padding: '0.65rem 1.1rem',
            borderRadius: '10px',
            background: 'transparent',
            border: '1.5px solid hsl(220 25% 24%)',
            color: 'hsl(220 15% 58%)',
            fontSize: '.85rem', fontWeight: 600,
            cursor: 'pointer',
            transition: 'all .18s',
            flexShrink: 0,
          }}
          onMouseEnter={e => {
            (e.currentTarget as HTMLButtonElement).style.borderColor = 'hsl(220 30% 38%)'
            ;(e.currentTarget as HTMLButtonElement).style.color = 'hsl(220 10% 75%)'
          }}
          onMouseLeave={e => {
            (e.currentTarget as HTMLButtonElement).style.borderColor = 'hsl(220 25% 24%)'
            ;(e.currentTarget as HTMLButtonElement).style.color = 'hsl(220 15% 58%)'
          }}
        >
          <SkipForward size={14} />
          Skip Trim / Use Full
        </button>

        {/* Process */}
        <button
          onClick={() => onConfirm(trimStart, trimEnd)}
          disabled={loading || selectedDuration < 1}
          style={{
            flex: 1,
            display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px',
            padding: '0.65rem 1.25rem',
            borderRadius: '10px',
            background: 'linear-gradient(135deg, hsl(220 70% 50%), hsl(260 70% 58%))',
            border: 'none',
            color: '#fff',
            fontSize: '.88rem', fontWeight: 700,
            cursor: loading || selectedDuration < 1 ? 'not-allowed' : 'pointer',
            opacity: loading || selectedDuration < 1 ? 0.55 : 1,
            transition: 'all .18s',
            boxShadow: '0 4px 16px hsl(220 70% 50% / 0.35)',
          }}
        >
          <Scissors size={15} />
          Process Recording
          {selectedDuration > 0 && (
            <span style={{
              fontSize: '.72rem', fontWeight: 600,
              background: 'hsl(0 0% 100% / 0.2)',
              padding: '2px 8px', borderRadius: '999px', marginLeft: '4px',
            }}>
              {fmtTime(selectedDuration)}
            </span>
          )}
        </button>
      </div>
    </div>
  )
}
