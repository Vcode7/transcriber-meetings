import { useEffect, useRef, useState } from 'react'
import { CheckCircle2, Loader, FileText, Users, Sparkles, CloudUpload, Zap } from 'lucide-react'
import { ProcessingStage } from '../store/processing'

// ── Stage definitions ────────────────────────────────────────────────────────
const STEPS: Array<{
  key: ProcessingStage
  label: string
  shortLabel: string
  icon: React.ReactNode
  color: string
  tip: string
}> = [
  {
    key: 'uploading',
    label: 'Uploading',
    shortLabel: 'Upload',
    icon: <CloudUpload size={14} />,
    color: 'hsl(205,90%,55%)',
    tip: 'Sending your audio to the server…',
  },
  {
    key: 'transcribing',
    label: 'Transcribing',
    shortLabel: 'Transcribe',
    icon: <FileText size={14} />,
    color: 'hsl(var(--accent))',
    tip: 'AI is converting speech to text — this is the longest step.',
  },
  {
    key: 'diarizing',
    label: 'Diarizing',
    shortLabel: 'Diarize',
    icon: <Users size={14} />,
    color: 'hsl(280,70%,60%)',
    tip: 'Segmenting audio by speaker turns…',
  },
  {
    key: 'identifying_speakers',
    label: 'Speaker ID',
    shortLabel: 'Speaker ID',
    icon: <Zap size={14} />,
    color: 'hsl(45,90%,50%)',
    tip: 'Matching voice embeddings to known profiles…',
  },
  {
    key: 'generating_insights',
    label: 'AI Insights',
    shortLabel: 'Insights',
    icon: <Sparkles size={14} />,
    color: 'hsl(130,60%,45%)',
    tip: 'Generating summary, key points, and action items…',
  },
]

// Queued / unknown maps to transcribing visually
const STAGE_INDEX: Record<string, number> = {
  uploading: 0,
  queued: 1,
  transcribing: 1,
  diarizing: 2,
  identifying_speakers: 3,
  generating_insights: 4,
  done: 5,
}

const ROTATING_TIPS = [
  'Large files can take 2–5 minutes — grab a coffee ☕',
  'Speaker diarization uses AI voice embeddings.',
  'The more voice samples you add, the better Speaker ID works.',
  'You can chat with your transcript using the AI panel.',
  'Transcripts are saved automatically to your history.',
  'Word-level confidence scores help you spot uncertain transcriptions.',
]

// ── Particle ─────────────────────────────────────────────────────────────────
function Particle({ x, delay, color }: { x: number; delay: number; color: string }) {
  return (
    <div
      style={{
        position: 'absolute',
        left: `${x}%`,
        bottom: '15%',
        width: 6,
        height: 6,
        borderRadius: '50%',
        background: color,
        opacity: 0,
        animation: `particle-rise ${2 + Math.random() * 2}s ease-out ${delay}s infinite`,
      }}
    />
  )
}

// ── Confetti piece ────────────────────────────────────────────────────────────
function ConfettiPiece({ x, color, delay, shape }: { x: number; color: string; delay: number; shape: 'circle' | 'rect' }) {
  return (
    <div
      style={{
        position: 'absolute',
        left: `${x}%`,
        top: '50%',
        width: shape === 'circle' ? 8 : 6,
        height: shape === 'circle' ? 8 : 14,
        borderRadius: shape === 'circle' ? '50%' : 2,
        background: color,
        animation: `confetti-spin 1s ease-out ${delay}s both`,
      }}
    />
  )
}

// ── Mini waveform (shown during transcribing) ─────────────────────────────────
function MiniWaveform() {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 3, height: 28 }}>
      {Array.from({ length: 12 }).map((_, i) => (
        <div
          key={i}
          style={{
            width: 3,
            height: '100%',
            background: 'hsl(var(--accent))',
            borderRadius: 2,
            transformOrigin: 'bottom',
            animation: `bar-dance ${0.6 + (i % 4) * 0.15}s ease-in-out ${i * 0.05}s infinite`,
          }}
        />
      ))}
    </div>
  )
}

// ── Progress ring SVG ─────────────────────────────────────────────────────────
function ProgressRing({ step, total }: { step: number; total: number }) {
  const r = 46
  const circ = 2 * Math.PI * r
  const filled = Math.min(step / total, 1) * circ
  const pct = Math.round((step / total) * 100)

  return (
    <svg width={110} height={110} style={{ transform: 'rotate(-90deg)' }}>
      {/* Track */}
      <circle cx={55} cy={55} r={r} fill="none" stroke="hsl(var(--muted))" strokeWidth={7} />
      {/* Fill */}
      <circle
        cx={55} cy={55} r={r}
        fill="none"
        stroke="url(#ring-grad)"
        strokeWidth={7}
        strokeLinecap="round"
        strokeDasharray={circ}
        strokeDashoffset={circ - filled}
        style={{ transition: 'stroke-dashoffset 0.7s cubic-bezier(0.4,0,0.2,1)' }}
      />
      <defs>
        <linearGradient id="ring-grad" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="hsl(var(--accent))" />
          <stop offset="100%" stopColor="hsl(14,95%,70%)" />
        </linearGradient>
      </defs>
      {/* Label in center (rotated back) */}
      <text
        x={55} y={55}
        textAnchor="middle"
        dominantBaseline="middle"
        style={{
          transform: 'rotate(90deg)',
          transformOrigin: '55px 55px',
          fill: 'hsl(var(--ink))',
          fontFamily: 'JetBrains Mono, monospace',
          fontSize: 18,
          fontWeight: 700,
        }}
      >
        {pct}%
      </text>
    </svg>
  )
}

// ── Main ProcessingOverlay ────────────────────────────────────────────────────
interface Props {
  stage: ProcessingStage
  startedAt: number | null
  source: 'record' | 'upload' | 'tab-audio' | null
}

export default function ProcessingOverlay({ stage, startedAt, source }: Props) {
  const [elapsed, setElapsed] = useState(0)
  const [tipIndex, setTipIndex] = useState(0)
  const [tipKey, setTipKey] = useState(0)
  const [showConfetti, setShowConfetti] = useState(false)
  const prevStageRef = useRef<ProcessingStage>(null)

  // Elapsed timer
  useEffect(() => {
    const t = setInterval(() => {
      setElapsed(startedAt ? Math.floor((Date.now() - startedAt) / 1000) : 0)
    }, 1000)
    return () => clearInterval(t)
  }, [startedAt])

  // Rotating tips
  useEffect(() => {
    const t = setInterval(() => {
      setTipIndex(i => (i + 1) % ROTATING_TIPS.length)
      setTipKey(k => k + 1)
    }, 5000)
    return () => clearInterval(t)
  }, [])

  // Confetti on done
  useEffect(() => {
    if (stage === 'done' && prevStageRef.current !== 'done') {
      setShowConfetti(true)
      const t = setTimeout(() => setShowConfetti(false), 1500)
      return () => clearTimeout(t)
    }
    prevStageRef.current = stage
  }, [stage])

  const stepIdx = STAGE_INDEX[stage ?? 'uploading'] ?? 0
  const currentStep = STEPS[Math.min(stepIdx, STEPS.length - 1)]
  const isDone = stage === 'done'

  const fmtElapsed = (s: number) => {
    const m = Math.floor(s / 60)
    const sec = s % 60
    return m > 0 ? `${m}m ${sec}s` : `${sec}s`
  }

  const CONFETTI_COLORS = [
    'hsl(var(--accent))', 'hsl(45,90%,55%)', 'hsl(130,60%,50%)',
    'hsl(280,70%,60%)', 'hsl(205,90%,55%)', 'hsl(340,80%,65%)',
  ]

  return (
    <div className="processing-overlay animate-fade-in">

      {/* Background particles */}
      {!isDone && Array.from({ length: 14 }).map((_, i) => (
        <Particle
          key={i}
          x={5 + i * 7}
          delay={i * 0.4}
          color={STEPS[i % STEPS.length].color + '80'}
        />
      ))}

      {/* Confetti burst on done */}
      {showConfetti && Array.from({ length: 24 }).map((_, i) => (
        <ConfettiPiece
          key={i}
          x={10 + (i % 10) * 8 + Math.random() * 5}
          color={CONFETTI_COLORS[i % CONFETTI_COLORS.length]}
          delay={i * 0.04}
          shape={i % 3 === 0 ? 'circle' : 'rect'}
        />
      ))}

      {/* ── Content card ── */}
      <div style={{
        position: 'relative', zIndex: 1,
        display: 'flex', flexDirection: 'column', alignItems: 'center',
        gap: '1.5rem',
        padding: '2.25rem 2.5rem',
        background: 'hsl(var(--card))',
        border: '2px solid hsl(var(--ink) / .12)',
        borderRadius: '20px',
        boxShadow: '0 20px 60px hsl(var(--ink) / .08), 4px 4px 0 0 hsl(var(--ink) / .06)',
        maxWidth: 460,
        width: '90%',
      }}>

        {/* Center icon + ring */}
        <div style={{ position: 'relative', width: 110, height: 110 }}>
          {/* Ripple rings */}
          {!isDone && (
            <div style={{ position: 'absolute', inset: '22px' }}>
              <div className="ripple-ring" />
              <div className="ripple-ring" />
              <div className="ripple-ring" />
            </div>
          )}

          {/* Progress ring */}
          <ProgressRing step={Math.max(stepIdx, isDone ? 5 : 0)} total={5} />

          {/* Icon overlay */}
          <div style={{
            position: 'absolute', inset: 0,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            {isDone ? (
              <div className="animate-bounce-in">
                <CheckCircle2 size={28} style={{ color: 'hsl(var(--success))' }} />
              </div>
            ) : (
              <div className="animate-heartbeat" style={{ color: 'hsl(var(--accent))', display: 'flex' }}>
                {currentStep.icon}
              </div>
            )}
          </div>
        </div>

        {/* Title */}
        <div style={{ textAlign: 'center' }}>
          <h2 style={{
            fontFamily: 'Caveat, cursive',
            fontSize: '1.6rem',
            fontWeight: 800,
            color: isDone ? 'hsl(var(--success))' : 'hsl(var(--ink))',
            margin: 0,
            letterSpacing: '-0.02em',
            transition: 'color 0.4s',
          }}>
            {isDone ? '✓ Complete!' : currentStep.label + '…'}
          </h2>
          {!isDone && (
            <p style={{
              fontFamily: 'Inter, sans-serif',
              fontSize: '0.82rem',
              color: 'hsl(var(--pencil))',
              margin: '4px 0 0',
            }}>
              {source === 'upload' ? 'Uploaded file' : source === 'tab-audio' ? 'Tab recording' : 'Live recording'} · {fmtElapsed(elapsed)} elapsed
            </p>
          )}
        </div>

        {/* Mini waveform during transcription */}
        {stage === 'transcribing' && <MiniWaveform />}

        {/* Stage stepper */}
        <div className="proc-stepper" style={{ marginTop: 4 }}>
          {STEPS.map((s, i) => {
            const state = i < stepIdx ? 'done' : i === stepIdx ? 'active' : 'pending'
            const isLast = i === STEPS.length - 1
            return (
              <div key={s.key} style={{ display: 'flex', alignItems: 'flex-start' }}>
                <div className="proc-step">
                  <div
                    className={`proc-step-dot ${state} ${state === 'active' ? 'step-active-anim' : ''}`}
                    title={s.label}
                  >
                    {state === 'done'
                      ? <CheckCircle2 size={14} />
                      : state === 'active'
                      ? <Loader size={14} className="spin" />
                      : <span style={{ fontSize: '0.7rem' }}>{i + 1}</span>
                    }
                  </div>
                  <span className={`proc-step-label ${state}`}>{s.shortLabel}</span>
                </div>
                {!isLast && (
                  <div
                    className={`proc-connector ${
                      i < stepIdx ? 'done' : i === stepIdx ? 'active' : ''
                    }`}
                  />
                )}
              </div>
            )
          })}
        </div>

        {/* Rotating tip */}
        {!isDone && (
          <div
            key={tipKey}
            className="animate-tip-in"
            style={{
              padding: '0.65rem 1rem',
              background: 'hsl(var(--muted))',
              borderRadius: '10px',
              border: '1.5px dashed hsl(var(--ink) / .12)',
              fontFamily: 'Inter, sans-serif',
              fontSize: '0.78rem',
              color: 'hsl(var(--ink-soft))',
              textAlign: 'center',
              width: '100%',
              lineHeight: 1.5,
            }}
          >
            💡 {ROTATING_TIPS[tipIndex]}
          </div>
        )}

        {/* Lock warning */}
        {!isDone && (
          <div style={{
            display: 'flex', alignItems: 'center', gap: 7,
            fontFamily: 'Inter, sans-serif',
            fontSize: '0.74rem',
            color: 'hsl(var(--pencil))',
            fontStyle: 'italic',
          }}>
            <span style={{
              width: 8, height: 8, borderRadius: '50%',
              background: 'hsl(var(--accent))',
              flexShrink: 0,
              boxShadow: '0 0 6px hsl(var(--accent) / .5)',
              display: 'inline-block',
            }} className="animate-pulse-rec" />
            Navigation locked until processing completes
          </div>
        )}
      </div>
    </div>
  )
}
