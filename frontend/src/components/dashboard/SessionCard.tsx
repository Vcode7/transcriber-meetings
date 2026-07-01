import { Clock, Users, ChevronRight, Trash2, Loader } from 'lucide-react'

interface SessionCardProps {
  id: string
  filename: string
  date: string
  duration: string
  speakers: string[]
  summary?: string
  status: string
  hasAI: boolean
  onClick: () => void
  onDelete: (e: React.MouseEvent) => void
  isDeleting?: boolean
}

export default function SessionCard({
  filename,
  date,
  duration,
  speakers,
  summary,
  status,
  hasAI,
  onClick,
  onDelete,
  isDeleting,
}: SessionCardProps) {
  return (
    <div
      className="glass card-hover fade-in"
      onClick={onClick}
      style={{
        padding: '1.25rem',
        cursor: 'pointer',
        display: 'flex',
        alignItems: 'center',
        gap: '16px',
      }}
    >
      <div style={{ flex: 1, minWidth: 0 }}>
        <h3
          style={{
            fontWeight: 700,
            fontSize: '1rem',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
            marginBottom: '8px',
            color: 'hsl(var(--ink))',
            fontFamily: 'Inter, sans-serif'
          }}
        >
          {filename}
        </h3>
        {summary && (
          <p
            style={{
              fontSize: '0.85rem',
              color: 'hsl(var(--pencil))',
              marginBottom: '10px',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
              fontFamily: 'Inter, sans-serif'
            }}
          >
            {summary}
          </p>
        )}
        <div style={{
          display: 'flex', flexWrap: 'wrap', gap: '12px',
          fontSize: '0.8rem',
          color: 'hsl(var(--pencil))',
          fontFamily: 'Inter, sans-serif'
        }}>
          <span style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
            <Clock size={13} /> {date}
          </span>
          {duration && <span>⏱ {duration}</span>}
          {speakers.length > 0 && (
            <span style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
              <Users size={13} /> {speakers.join(', ')}
            </span>
          )}
        </div>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
        <span
          className={`badge ${
            status === 'done' ? 'badge-green' : status === 'error' ? 'badge-red' : 'badge-yellow'
          }`}
        >
          {status}
        </span>
        {hasAI && <span className="badge badge-purple">AI</span>}
        <button
          onClick={onDelete}
          disabled={isDeleting}
          className="icon-btn"
          style={{ width: '32px', height: '32px' }}
        >
          {isDeleting ? <Loader size={16} className="spin" /> : <Trash2 size={16} />}
        </button>
        <ChevronRight size={18} style={{ color: 'hsl(var(--pencil))' }} />
      </div>
    </div>
  )
}
