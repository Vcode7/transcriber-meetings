import SpeakerLabel from './SpeakerLabel'

interface TranscriptSegmentProps {
  speaker: string
  text: string
  timestamp?: string
  color: string
  onSpeakerRename?: (oldName: string, newName: string) => void
  editable?: boolean
}

export default function TranscriptSegment({
  speaker,
  text,
  timestamp,
  color,
  onSpeakerRename,
  editable = false,
}: TranscriptSegmentProps) {
  return (
    <div className="transcript-segment">
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
        <SpeakerLabel speaker={speaker} color={color} onRename={onSpeakerRename} editable={editable} />
        {timestamp && (
          <span className="mono" style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
            {timestamp}
          </span>
        )}
      </div>
      <p className="seg-text">{text}</p>
    </div>
  )
}
