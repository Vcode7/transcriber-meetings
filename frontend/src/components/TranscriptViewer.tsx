import { useState, useEffect, useRef } from "react";
import { BookOpen } from "lucide-react";
import api from "../api/client";
export interface Word {
  word: string
  start: number
  end: number
  probability: number
}

export interface Segment {
  speaker_label: string
  start: number
  end: number
  text: string
  words: Word[]
  is_overlap: boolean
}

interface Shortcut {
  shortcut: string
  full_form: string
}

interface Props {
  segments: Segment[]
  showConfidence: boolean
  wordConfLow?: number
  wordConfMid?: number
  shortcuts?: Shortcut[]
}

const SPEAKER_COLORS = [
  'hsl(14, 90%, 56%)',    // accent orange-red
  'hsl(205, 90%, 55%)',   // blue
  'hsl(130, 60%, 45%)',   // green
  'hsl(280, 70%, 60%)',   // purple
  'hsl(45, 90%, 50%)',    // yellow
  'hsl(340, 80%, 60%)',   // pink
]

function formatTime(s: number) {
  const m = Math.floor(s / 60)
  const sec = Math.floor(s % 60).toString().padStart(2, '0')
  return `${m}:${sec}`
}

function wordClass(prob: number, low: number, mid: number) {
  if (prob < low) return 'word-low'
  if (prob < mid) return 'word-mid'
  return 'word-hi'
}

function SpeakerAvatar({ label, color }: { label: string; color: string }) {
  const initials = label
    .split(' ')
    .map((w) => w[0])
    .join('')
    .toUpperCase()
    .slice(0, 2)

  return (
    <div
      className="speaker-avatar"
      style={{
        background: `${color}30`,
        border: `1.5px solid ${color}60`,
        color,
        flexShrink: 0,
        marginTop: '1px',
      }}
      title={label}
    >
      {initials}
    </div>
  )
}
function AnalysisItem({
  label,
  value,
  color,
}: {
  label: string;
  value: string | number;
  color?: string;
}) {
  return (
    <div
      style={{
        padding: ".75rem",
        borderRadius: "10px",
        background: "hsl(var(--muted) / .45)",
        border: "1px solid hsl(var(--ink) / .06)",
      }}
    >
      <div
        style={{
          fontSize: ".7rem",
          color: "hsl(var(--pencil))",
          marginBottom: ".35rem",
          textTransform: "uppercase",
          letterSpacing: ".05em",
        }}
      >
        {label}
      </div>

      <div
        style={{
          fontWeight: 700,
          fontSize: "1rem",
          color: color || "hsl(var(--ink))",
        }}
      >
        {value}
      </div>
    </div>
  );
}
// Client-side shortcut expander (mirrors backend transcript_normalizer.py)
function buildExpander(shortcuts: Shortcut[]): (text: string) => string {
  const patterns = shortcuts
    .filter(s => s.shortcut?.trim())
    .map(s => {
      const letters = s.shortcut.trim().toUpperCase().split('')
      const SEP = '[\\s.\\-,]*'
      const inner = letters.map(c => `${c.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\.?`).join(SEP)
      const pattern = new RegExp(`(?<![A-Za-z0-9])${inner}(?![A-Za-z0-9])`, 'gi')
      return { pattern, full_form: s.full_form }
    })
  return (text: string) => {
    let result = text
    for (const { pattern, full_form } of patterns) {
      result = result.replace(pattern, full_form)
    }
    return result
  }
}

export default function TranscriptViewer({ segments, showConfidence = true, wordConfLow = 0.5, wordConfMid = 0.75, shortcuts = [] }: Props) {
  const [useDictExp, setUseDictExp] = useState(false)
  const [loadedShortcuts, setLoadedShortcuts] = useState<Shortcut[]>([])
  const fetchedRef = useRef(false)

  useEffect(() => {
    if (shortcuts && shortcuts.length > 0) {
      setLoadedShortcuts(shortcuts)
    } else if (!shortcuts || shortcuts.length === 0) {
      if (fetchedRef.current) return
      fetchedRef.current = true
      api.get('/dictionary/shortcuts')
        .then(res => {
          if (Array.isArray(res.data)) {
            setLoadedShortcuts(res.data)
          }
        })
        .catch(() => {})
    }
  }, [shortcuts])

  const expander = useDictExp && loadedShortcuts.length > 0
    ? buildExpander(loadedShortcuts)
    : null
  if (!segments || segments.length === 0) {
    return (
      <div style={{
        color: 'hsl(var(--pencil))',
        textAlign: 'center',
        padding: '3.5rem 2rem',
        fontSize: '0.95rem',
        fontFamily: 'Inter, sans-serif',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: '1.25rem',
      }}>
        <div style={{
          width: '72px', height: '72px',
          borderRadius: '50%',
          background: 'hsl(var(--accent) / .08)',
          border: '2px dashed hsl(var(--accent) / .22)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: '2rem',
        }} className="animate-float">🎤</div>
        <div>
          <p style={{ fontWeight: 700, marginBottom: '.4rem', color: 'hsl(var(--ink))', fontSize: '1rem' }}>
            No transcript yet
          </p>
          <p style={{ fontSize: '.85rem', opacity: .65, lineHeight: 1.5 }}>
            Record or upload audio to get started
          </p>
        </div>
      </div>
    )
  }

  // Build speaker → color map
  const speakerColors: Record<string, string> = {}
  let colorIdx = 0
  for (const seg of segments) {
    if (!(seg.speaker_label in speakerColors)) {
      speakerColors[seg.speaker_label] = SPEAKER_COLORS[colorIdx % SPEAKER_COLORS.length]
      colorIdx++
    }
  }

  // Check if any segment has word-level confidence data
  const hasWordConf = segments.some((s) => s.words && s.words.length > 0)
  const totalWords = segments.reduce(
  (sum, seg) => sum + (seg.words?.length || 0),
  0
);

const highWords = segments.reduce(
  (sum, seg) =>
    sum + (seg.words?.filter(w => w.probability >= wordConfMid).length || 0),
  0
);

const midWords = segments.reduce(
  (sum, seg) =>
    sum +
    (seg.words?.filter(
      w => w.probability >= wordConfLow && w.probability < wordConfMid
    ).length || 0),
  0
);

const lowWords = segments.reduce(
  (sum, seg) =>
    sum + (seg.words?.filter(w => w.probability < wordConfLow).length || 0),
  0
);

const avgConfidence =
  totalWords > 0
    ? (
        segments.reduce(
          (sum, seg) =>
            sum +
            (seg.words?.reduce((s, w) => s + w.probability, 0) || 0),
          0
        ) / totalWords
      ) * 100
    : 0;
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0' }}>
      {/* Dictionary expansion toggle */}
      {loadedShortcuts.length > 0 && (
        <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: '8px' }}>
          <button
            id="dict-expansion-toggle"
            onClick={() => setUseDictExp(v => !v)}
            title={useDictExp ? 'Disable dictionary expansion' : 'Expand abbreviations using Dictionary'}
            style={{
              display: 'flex', alignItems: 'center', gap: '6px',
              padding: '.35rem .8rem',
              borderRadius: '999px',
              border: `1.5px solid ${useDictExp ? 'hsl(130,60%,45% / .4)' : 'hsl(var(--ink) / .15)'}`,
              background: useDictExp ? 'hsl(130,60%,45% / .08)' : 'hsl(var(--muted) / .5)',
              color: useDictExp ? 'hsl(130,60%,45%)' : 'hsl(var(--pencil))',
              cursor: 'pointer',
              fontSize: '.74rem',
              fontWeight: 600,
              fontFamily: 'Inter, sans-serif',
              transition: 'all .15s',
            }}
          >
            <BookOpen size={11} />
            {useDictExp ? 'Dictionary ON' : 'Dictionary'}
          </button>
        </div>
      )}
      {segments.map((seg, i) => (
        <div
          key={i}
          className="transcript-segment animate-slide-up "
          style={{
            '--speaker-color': speakerColors[seg.speaker_label],
            animationDelay: `${Math.min(i * 0.04, 0.5)}s`,
            animationFillMode: 'both',
          } as React.CSSProperties}
        >
          <div className="seg-meta">
            {/* Speaker avatar */}
            <SpeakerAvatar label={seg.speaker_label} color={speakerColors[seg.speaker_label]} />

            <span
              className="speaker-name"
              style={{ color: speakerColors[seg.speaker_label] }}
            >
              {seg.speaker_label}
            </span>
            <span className="seg-time">
              {formatTime(seg.start)} → {formatTime(seg.end)}
            </span>
            {seg.is_overlap && (
              <span style={{
                fontSize: '.65rem', fontWeight: 700,
                color: 'hsl(var(--destructive))',
                background: 'hsl(var(--destructive) / .1)',
                border: '1px solid hsl(var(--destructive) / .3)',
                borderRadius: '999px',
                padding: '.1rem .45rem',
                fontFamily: 'Inter, sans-serif',
                letterSpacing: '.04em',
                display: 'inline-flex', alignItems: 'center', gap: '3px',
              }}>
                ⚡ OVERLAP
              </span>
            )}
          </div>
          <div className="seg-text">
           {seg.words && seg.words.length > 0
  ? seg.words.map((w, wi) => {
      const wordText = expander ? expander(w.word) : w.word
      if (showConfidence) {
        return (
          <span
            key={wi}
            className={wordClass(
              w.probability,
              wordConfLow,
              wordConfMid
            )}
            title={`${(w.probability * 100).toFixed(0)}% confidence`}
          >
            {wordText}{" "}
          </span>
        );
      }

      return (
        <span
          key={wi}
          className={w.probability < 0.3 ? "word-underlined" : ""}
          title={
            w.probability < 0.3
              ? `${(w.probability * 100).toFixed(0)}% confidence`
              : ""
          }
        >
          {wordText}{" "}
        </span>
      );
    })
  : expander ? expander(seg.text) : seg.text}
          </div>
        </div>
      ))}
<div
  style={{
    marginTop: "12px",
    background: "hsl(var(--card))",
    border: "1px solid hsl(var(--ink) / .08)",
    borderRadius: "12px",
    padding: "1rem",
  }}
>
  <div
    style={{
      fontSize: ".72rem",
      fontWeight: 700,
      textTransform: "uppercase",
      letterSpacing: ".08em",
      color: "hsl(var(--pencil))",
      marginBottom: ".9rem",
    }}
  >
    Transcript Analysis
  </div>

  <div
    style={{
      display: "grid",
      gridTemplateColumns: "repeat(auto-fit, minmax(150px,1fr))",
      gap: "14px",
    }}
  >
    <AnalysisItem
      label="Total Words"
      value={totalWords}
    />

    <AnalysisItem
      label="High Confidence"
      value={`${highWords} (${((highWords / totalWords) * 100 || 0).toFixed(1)}%)`}
      color="hsl(var(--sticky-green))"
    />

    <AnalysisItem
      label="Medium Confidence"
      value={`${midWords} (${((midWords / totalWords) * 100 || 0).toFixed(1)}%)`}
      color="hsl(45,90%,50%)"
    />

    <AnalysisItem
      label="Low Confidence"
      value={`${lowWords} (${((lowWords / totalWords) * 100 || 0).toFixed(1)}%)`}
      color="hsl(var(--destructive))"
    />

    <AnalysisItem
      label="Average Accuracy"
      value={`${avgConfidence.toFixed(2)}%`}
      color="hsl(var(--accent))"
    />
  </div>
</div>
      {/* Legend — only show if word confidence data exists */}
      {hasWordConf && (
        <div style={{
          display: 'flex',
          gap: '10px',
          padding: '.85rem 1rem',
          fontSize: '0.78rem',
          color: 'hsl(var(--ink-soft))',
          marginTop: '.75rem',
          background: 'hsl(var(--card))',
          borderRadius: '10px',
          border: '1px solid hsl(var(--ink) / .08)',
          flexWrap: 'wrap',
          fontFamily: 'Inter, sans-serif',
          fontWeight: 500,
          alignItems: 'center',
        }}>
          <span style={{ fontSize: '.68rem', color: 'hsl(var(--pencil))', textTransform: 'uppercase', letterSpacing: '.1em', fontWeight: 700 }}>
            Word confidence:
          </span>
          {[
            { label: 'High >75%', cls: 'word-hi' },
            { label: 'Mid 50–75%', cls: 'word-mid' },
            { label: 'Low <50%', cls: 'word-low' },
          ].map(({ label, cls }) => (
            <span key={cls} style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
              <span className={cls} style={{ fontSize: '.72rem', padding: '1px 6px' }}>{label}</span>
            </span>
          ))}
        </div>
      )}

      {/* Speaker legend */}
      {Object.keys(speakerColors).length > 1 && (
        <div style={{
          display: 'flex',
          gap: '8px',
          flexWrap: 'wrap',
          padding: '.75rem 1rem',
          marginTop: hasWordConf ? '6px' : '.75rem',
          background: 'hsl(var(--card))',
          borderRadius: '10px',
          border: '1px solid hsl(var(--ink) / .08)',
          alignItems: 'center',
        }}>
          <span style={{ fontSize: '.68rem', color: 'hsl(var(--pencil))', textTransform: 'uppercase', letterSpacing: '.1em', fontWeight: 700, fontFamily: 'Inter, sans-serif' }}>
            Speakers:
          </span>
          {Object.entries(speakerColors).map(([label, color]) => (
            <span key={label} style={{
              display: 'flex', alignItems: 'center', gap: '5px',
              fontSize: '.75rem', fontFamily: 'Inter, sans-serif',
              fontWeight: 600, color: 'hsl(var(--ink-soft))',
            }}>
              <span style={{ width: 10, height: 10, borderRadius: '50%', background: color, display: 'inline-block', boxShadow: `0 0 4px ${color}60` }} />
              {label}
            </span>
          ))}
        </div>
      )}
    </div>
  )
}
