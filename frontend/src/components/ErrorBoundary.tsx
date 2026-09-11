import React, { Component, ErrorInfo, ReactNode } from 'react'
import { AlertTriangle, RefreshCw } from 'lucide-react'

interface Props {
  children: ReactNode
  fallback?: ReactNode | ((error: Error, reset: () => void) => ReactNode)
  onCatch?: (error: Error, errorInfo: ErrorInfo) => void
}

interface State {
  hasError: boolean
  error: Error | null
}

export default class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props)
    this.state = { hasError: false, error: null }
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('[ErrorBoundary] Caught unhandled React error:', error, errorInfo)
    if (this.props.onCatch) {
      this.props.onCatch(error, errorInfo)
    }
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null })
  }

  render() {
    if (this.state.hasError) {
      if (typeof this.props.fallback === 'function') {
        return this.props.fallback(this.state.error || new Error('Unknown error'), this.handleReset)
      }
      if (this.props.fallback) {
        return this.props.fallback
      }

      return (
        <div style={{
          padding: '1.5rem',
          margin: '1rem 0',
          borderRadius: '12px',
          background: 'hsl(var(--card))',
          border: '1.5px solid hsl(var(--destructive) / .35)',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: '12px',
          textAlign: 'center',
        }}>
          <div style={{
            width: '40px',
            height: '40px',
            borderRadius: '10px',
            background: 'hsl(var(--destructive) / .12)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: 'hsl(var(--destructive))',
          }}>
            <AlertTriangle size={20} />
          </div>
          <div style={{ fontSize: '0.92rem', fontWeight: 600, color: 'hsl(var(--foreground))' }}>
            Something went wrong in this section
          </div>
          <div style={{ fontSize: '0.78rem', color: 'hsl(var(--muted-foreground))', maxWidth: '400px' }}>
            {this.state.error?.message || 'An unexpected rendering error occurred.'}
          </div>
          <button
            onClick={this.handleReset}
            style={{
              marginTop: '4px',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              padding: '0.45rem 0.9rem',
              borderRadius: '8px',
              background: 'hsl(var(--secondary))',
              border: '1px solid hsl(var(--border))',
              color: 'hsl(var(--foreground))',
              fontSize: '0.8rem',
              fontWeight: 600,
              cursor: 'pointer',
            }}
          >
            <RefreshCw size={13} />
            Try Again
          </button>
        </div>
      )
    }

    return this.props.children
  }
}
