import { Mic, Square, Loader } from 'lucide-react'

interface RecordButtonProps {
  state: 'idle' | 'recording' | 'processing'
  onStart: () => void
  onStop: () => void
  disabled?: boolean
}

export default function RecordButton({ state, onStart, onStop, disabled }: RecordButtonProps) {
  if (state === 'processing') {
    return (
      <button className="record-btn idle" disabled style={{ opacity: 0.6 }}>
        <Loader size={36} color="var(--text-inverse)" className="spin" />
      </button>
    )
  }

  if (state === 'recording') {
    return (
      <button
        className="record-btn recording"
        onClick={onStop}
        disabled={disabled}
        title="Stop recording"
      >
        <Square size={32} color="var(--text-inverse)" fill="var(--text-inverse)" />
      </button>
    )
  }

  return (
    <button
      className="record-btn idle"
      onClick={onStart}
      disabled={disabled}
      title="Start recording"
    >
      <Mic size={38} color="var(--text-inverse)" />
    </button>
  )
}
