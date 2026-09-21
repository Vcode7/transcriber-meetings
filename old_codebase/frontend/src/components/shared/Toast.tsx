import { useEffect } from 'react'
import { CheckCircle, XCircle, Info, AlertTriangle, X } from 'lucide-react'

export type ToastType = 'success' | 'error' | 'info' | 'warning'

interface ToastProps {
  message: string
  type?: ToastType
  onClose: () => void
  duration?: number
}

const icons = {
  success: CheckCircle,
  error: XCircle,
  info: Info,
  warning: AlertTriangle,
}

const colors = {
  success: 'var(--success)',
  error: 'var(--danger)',
  info: 'var(--accent)',
  warning: 'var(--warning)',
}

export default function Toast({ message, type = 'info', onClose, duration = 3000 }: ToastProps) {
  const Icon = icons[type]

  useEffect(() => {
    if (duration > 0) {
      const timer = setTimeout(onClose, duration)
      return () => clearTimeout(timer)
    }
  }, [duration, onClose])

  return (
    <div
      className="fade-in"
      style={{
        position: 'fixed',
        bottom: '24px',
        right: '24px',
        background: 'var(--bg-elevated)',
        border: `1px solid ${colors[type]}`,
        borderRadius: 'var(--radius-sm)',
        padding: '12px 16px',
        display: 'flex',
        alignItems: 'center',
        gap: '12px',
        boxShadow: 'var(--shadow-lg)',
        zIndex: 9999,
        maxWidth: '400px',
      }}
    >
      <Icon size={20} style={{ color: colors[type], flexShrink: 0 }} />
      <span style={{ flex: 1, fontSize: '0.875rem', color: 'var(--text-primary)' }}>{message}</span>
      <button
        onClick={onClose}
        style={{
          background: 'none',
          border: 'none',
          cursor: 'pointer',
          color: 'var(--text-muted)',
          padding: '4px',
          display: 'flex',
          alignItems: 'center',
        }}
      >
        <X size={16} />
      </button>
    </div>
  )
}
