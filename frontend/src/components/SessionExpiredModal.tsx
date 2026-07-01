/**
 * SessionExpiredModal.tsx
 * Shown when the 30-day refresh token expires or is revoked.
 * Blocks all interaction and prompts the user to sign in again.
 */
import { useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { LogIn, ShieldAlert } from 'lucide-react'
import { useAuthStore } from '../store/auth'

export default function SessionExpiredModal() {
  const sessionExpired = useAuthStore((s) => s.sessionExpired)
  const setSessionExpired = useAuthStore((s) => s.setSessionExpired)
  const logout = useAuthStore((s) => s.logout)
  const navigate = useNavigate()

  // Prevent background scroll when modal is open
  useEffect(() => {
    if (sessionExpired) {
      document.body.style.overflow = 'hidden'
    } else {
      document.body.style.overflow = ''
    }
    return () => { document.body.style.overflow = '' }
  }, [sessionExpired])

  if (!sessionExpired) return null

  const handleSignIn = () => {
    setSessionExpired(false)
    logout()
    navigate('/login', { replace: true })
  }

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 9999,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'hsl(var(--background) / .85)',
        backdropFilter: 'blur(12px)',
        WebkitBackdropFilter: 'blur(12px)',
        animation: 'fade-in .25s ease',
      }}
    >
      {/* Card */}
      <div
        className="animate-bounce-in"
        style={{
          background: 'hsl(var(--card))',
          border: '1.5px solid hsl(var(--border))',
          borderTop: '3px solid hsl(var(--accent))',
          borderRadius: '18px',
          padding: '2.5rem 2rem',
          maxWidth: '380px',
          width: '90%',
          textAlign: 'center',
          boxShadow: '0 24px 60px hsl(var(--background) / .6), 0 0 0 1px hsl(var(--border) / .5)',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: '1rem',
        }}
      >
        {/* Icon */}
        <div
          style={{
            width: 56, height: 56,
            borderRadius: '14px',
            background: 'hsl(var(--accent) / .1)',
            border: '2px solid hsl(var(--accent) / .25)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            marginBottom: '.25rem',
          }}
        >
          <ShieldAlert size={26} style={{ color: 'hsl(var(--accent))' }} />
        </div>

        {/* Title */}
        <h2 style={{
          fontSize: '1.15rem',
          fontWeight: 700,
          color: 'hsl(var(--ink))',
          fontFamily: 'Inter, sans-serif',
          letterSpacing: '-.02em',
          margin: 0,
        }}>
          Session Expired
        </h2>

        {/* Body */}
        <p style={{
          fontSize: '.88rem',
          color: 'hsl(var(--pencil))',
          fontFamily: 'Inter, sans-serif',
          lineHeight: 1.6,
          margin: 0,
          maxWidth: '280px',
        }}>
          Your session has expired after 30 days for security. Please sign in again to continue.
        </p>

        {/* CTA */}
        <button
          id="session-expired-signin"
          className="btn btn-primary"
          onClick={handleSignIn}
          style={{
            width: '100%',
            justifyContent: 'center',
            padding: '.75rem 1.5rem',
            fontSize: '.95rem',
            marginTop: '.5rem',
          }}
        >
          <LogIn size={16} />
          Sign In Again
        </button>

        {/* Fine print */}
        <p style={{
          fontSize: '.72rem',
          color: 'hsl(var(--pencil) / .6)',
          fontFamily: 'Inter, sans-serif',
          margin: 0,
        }}>
          Your data is safe — sessions expire after 30 days
        </p>
      </div>
    </div>
  )
}
