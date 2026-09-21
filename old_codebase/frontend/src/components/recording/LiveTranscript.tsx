import { useEffect, useRef } from 'react'
import { Mic } from 'lucide-react'

interface TranscriptLine {
  speaker: string
  text: string
  timestamp: string
  color: string
}

interface LiveTranscriptProps {
  lines: TranscriptLine[]
  isRecording: boolean
}

export default function LiveTranscript({ lines, isRecording }: LiveTranscriptProps) {
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (containerRef.current && isRecording) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight
    }
  }, [lines, isRecording])

  if (lines.length === 0) {
    return (
      <div
        style={{
          textAlign: 'center',
          padding: '4rem 1.5rem',
          color: 'hsl(var(--pencil))',
          fontSize: '0.9rem',
        }}
      >
        <Mic 
          size={40} 
          style={{ 
            margin: '0 auto 1.5rem', 
            display: 'block', 
            opacity: 0.3,
            color: 'hsl(var(--accent))'
          }}
          className={isRecording ? 'animate-pulse-rec' : 'animate-float'}
        />
        <p style={{ 
          fontFamily: 'Inter, sans-serif',
          fontWeight: 500,
          marginBottom: '.5rem'
        }}>
          {isRecording ? 'Listening...' : 'Start recording'}
        </p>
        <p style={{ 
          fontSize: '.82rem',
          opacity: .7
        }}>
          {isRecording ? 'Transcript will appear here' : 'Live transcript will appear here'}
        </p>
      </div>
    )
  }

  return (
    <div
      ref={containerRef}
      style={{
        maxHeight: '450px',
        overflowY: 'auto',
        display: 'flex',
        flexDirection: 'column',
        gap: '14px',
        padding: '1.25rem',
        background: 'hsl(var(--card))',
        borderRadius: '14px 18px 16px 20px / 18px 14px 20px 16px',
        border: '2.5px solid hsl(var(--ink) / .2)',
        boxShadow: '3px 3px 0 0 hsl(var(--ink) / .15)',
      }}
    >
      {lines.map((line, idx) => (
        <div 
          key={idx} 
          className="animate-slide-up" 
          style={{ 
            display: 'flex', 
            flexDirection: 'column', 
            gap: '6px',
            padding: '.75rem',
            background: 'hsl(var(--paper) / .5)',
            borderRadius: '10px 14px 12px 16px / 14px 10px 16px 12px',
            border: '2px dashed hsl(var(--ink) / .15)',
            transition: 'all .2s',
            animationDelay: `${idx * 0.05}s`,
            animationFillMode: 'both'
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <span
              className={`badge badge-${line.color}`}
              style={{ 
                fontSize: '0.72rem', 
                fontWeight: 600,
                fontFamily: 'Inter, sans-serif'
              }}
            >
              {line.speaker}
            </span>
            <span 
              className="mono" 
              style={{ 
                fontSize: '0.72rem', 
                color: 'hsl(var(--pencil))',
                fontWeight: 500
              }}
            >
              {line.timestamp}
            </span>
          </div>
          <p style={{ 
            fontSize: '0.9rem', 
            lineHeight: 1.7, 
            color: 'hsl(var(--ink))',
            fontFamily: 'Inter, sans-serif'
          }}>
            {line.text}
          </p>
        </div>
      ))}
      {isRecording && (
        <div 
          className="animate-pulse-rec"
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            padding: '.5rem .75rem',
            fontSize: '.8rem',
            color: 'hsl(var(--accent))',
            fontFamily: 'Inter, sans-serif',
            fontWeight: 500
          }}
        >
          <div style={{
            width: '8px',
            height: '8px',
            borderRadius: '50%',
            background: 'hsl(var(--accent))',
            boxShadow: '0 0 8px hsl(var(--accent))'
          }} />
          Listening...
        </div>
      )}
    </div>
  )
}
