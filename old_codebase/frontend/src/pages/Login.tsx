import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { Mail, Lock, Zap, Eye, EyeOff, Loader, Sparkles, AlertCircle } from 'lucide-react'
import api from '../api/client'
import { useAuthStore } from '../store/auth'
import { getApiErrorDetail } from '../lib/errors'

export default function Login() {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [showPw, setShowPw] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const { setAuth } = useAuthStore()
  const navigate = useNavigate()

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')
    setLoading(true)

    try {
      const res = await api.post('/auth/login', { email, password })
      setAuth(res.data.user, res.data.access_token)
      navigate(res.data.user.needs_setup ? '/setup' : '/dashboard')
    } catch (e: unknown) {
      setError(getApiErrorDetail(e, 'Login failed. Please check your credentials.'))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="auth-bg">
      <div className="auth-card animate-bounce-in">

        {/* Logo + Brand */}
        <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '10px',
            marginBottom: '1rem',
          }}>
            <div style={{ position: 'relative' }}>
              <div style={{
                width: 44, height: 44,
                borderRadius: '12px',
                background: 'linear-gradient(135deg, hsl(var(--accent) / .15), hsl(var(--accent) / .05))',
                border: '2px solid hsl(var(--accent) / .35)',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                boxShadow: '0 0 20px hsl(var(--accent) / .15)',
              }}>
                <Zap
                  size={22}
                  style={{ color: 'hsl(var(--accent))', position: 'relative', zIndex: 1 }}
                  fill="currentColor"
                  className="animate-float"
                />
              </div>
              <Sparkles
                size={13}
                style={{
                  position: 'absolute',
                  top: -4, right: -4,
                  color: 'hsl(var(--accent))',
                  animation: 'pulse-rec 2s ease-in-out infinite',
                }}
              />
            </div>
            <span style={{
              fontSize: '1.85rem',
              fontWeight: 800,
              letterSpacing: '-0.03em',
              fontFamily: 'Caveat, cursive',
              lineHeight: 1,
            }}>
              Voice<span style={{ color: 'hsl(var(--accent))' }}>Sum</span>
            </span>
          </div>
          <h1 style={{
            fontSize: '1.15rem',
            fontWeight: 700,
            color: 'hsl(var(--ink))',
            fontFamily: 'Inter, sans-serif',
            marginBottom: '.3rem',
          }}>
            Welcome back
          </h1>
          <p style={{
            fontSize: '.85rem',
            color: 'hsl(var(--pencil))',
            fontFamily: 'Inter, sans-serif',
            lineHeight: 1.5,
          }}>
            Sign in to continue your conversations
          </p>
        </div>

        <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '1.1rem' }}>
          {/* Email */}
          <div className="animate-slide-up" style={{ animationDelay: '0.1s', animationFillMode: 'both' }}>
            <label className="label">Email address</label>
            <div style={{ position: 'relative' }}>
              <Mail size={15} style={{
                position: 'absolute',
                left: '13px',
                top: '50%',
                transform: 'translateY(-50%)',
                color: 'hsl(var(--pencil))',
                zIndex: 1,
                pointerEvents: 'none',
              }} />
              <input
                id="login-email"
                className="input"
                type="email"
                placeholder="you@example.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                style={{ paddingLeft: '2.4rem' }}
              />
            </div>
          </div>

          {/* Password */}
          <div className="animate-slide-up" style={{ animationDelay: '0.17s', animationFillMode: 'both' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '.45rem' }}>
              <label className="label" style={{ marginBottom: 0 }}>Password</label>
              <span style={{
                fontSize: '.78rem',
                color: 'hsl(var(--accent))',
                fontFamily: 'Inter, sans-serif',
                fontWeight: 500,
                cursor: 'pointer',
              }}>
                Forgot password?
              </span>
            </div>
            <div style={{ position: 'relative' }}>
              <Lock size={15} style={{
                position: 'absolute',
                left: '13px',
                top: '50%',
                transform: 'translateY(-50%)',
                color: 'hsl(var(--pencil))',
                zIndex: 1,
                pointerEvents: 'none',
              }} />
              <input
                id="login-password"
                className="input"
                type={showPw ? 'text' : 'password'}
                placeholder="••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                style={{ paddingLeft: '2.4rem', paddingRight: '2.75rem' }}
              />
              <button
                type="button"
                onClick={() => setShowPw(!showPw)}
                style={{
                  position: 'absolute',
                  right: '10px',
                  top: '50%',
                  transform: 'translateY(-50%)',
                  background: 'transparent',
                  border: 'none',
                  cursor: 'pointer',
                  color: 'hsl(var(--pencil))',
                  padding: '4px',
                  display: 'flex', alignItems: 'center',
                  borderRadius: '6px',
                  transition: 'color .15s',
                }}
                onMouseEnter={(e) => (e.currentTarget.style.color = 'hsl(var(--ink))')}
                onMouseLeave={(e) => (e.currentTarget.style.color = 'hsl(var(--pencil))')}
              >
                {showPw ? <EyeOff size={16} /> : <Eye size={16} />}
              </button>
            </div>
          </div>

          {/* Error */}
          {error && (
            <div
              className="animate-shake"
              style={{
                display: 'flex', alignItems: 'center', gap: '9px',
                background: 'hsl(var(--destructive) / 0.08)',
                border: '1.5px solid hsl(var(--destructive) / 0.35)',
                borderRadius: '10px',
                padding: '0.7rem 1rem',
                color: 'hsl(var(--destructive))',
                fontSize: '0.85rem',
                fontFamily: 'Inter, sans-serif',
                fontWeight: 500,
              }}
            >
              <AlertCircle size={15} style={{ flexShrink: 0 }} />
              {error}
            </div>
          )}

          {/* Submit */}
          <button
            id="login-submit"
            className="btn btn-primary animate-slide-up"
            type="submit"
            disabled={loading}
            style={{
              width: '100%',
              justifyContent: 'center',
              padding: '.8rem 1.5rem',
              fontSize: '.95rem',
              marginTop: '.15rem',
              animationDelay: '0.25s',
              animationFillMode: 'both',
            }}
          >
            {loading ? <Loader size={16} className="spin" /> : null}
            {loading ? 'Signing in…' : 'Sign In'}
          </button>
        </form>

        <div className="divider-text" style={{ margin: '1.5rem 0 1.25rem' }}>
          <span>new here?</span>
        </div>

        <p style={{
          textAlign: 'center',
          fontSize: '0.9rem',
          color: 'hsl(var(--pencil))',
          fontFamily: 'Inter, sans-serif',
        }}>
          Don't have an account?{' '}
          <Link
            to="/signup"
            style={{
              color: 'hsl(var(--accent))',
              textDecoration: 'none',
              fontWeight: 700,
            }}
            className="scribble-underline"
          >
            Create one free →
          </Link>
        </p>
      </div>
    </div>
  )
}
