import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { Mail, Lock, User, Zap, Loader, Sparkles, AlertCircle, CheckCircle2 } from 'lucide-react'
import api from '../api/client'
import { useAuthStore } from '../store/auth'
import { getApiErrorDetail } from '../lib/errors'

function PasswordStrength({ password }: { password: string }) {
  const strength = password.length === 0 ? 0
    : password.length < 6 ? 1
    : password.length < 10 || !/[A-Z]/.test(password) || !/[0-9]/.test(password) ? 2
    : 3

  const labels = ['', 'Weak', 'Fair', 'Strong']
  const colors = ['', 'hsl(var(--destructive))', 'hsl(45,90%,50%)', 'hsl(var(--success))']

  if (!password) return null

  return (
    <div style={{ marginTop: '.5rem' }}>
      <div style={{ display: 'flex', gap: '4px', marginBottom: '4px' }}>
        {[1, 2, 3].map((i) => (
          <div
            key={i}
            style={{
              flex: 1, height: 4, borderRadius: 99,
              background: strength >= i ? colors[strength] : 'hsl(var(--muted))',
              transition: 'background .25s',
            }}
          />
        ))}
      </div>
      <span style={{
        fontSize: '.72rem',
        color: colors[strength],
        fontFamily: 'Inter, sans-serif',
        fontWeight: 600,
      }}>
        {labels[strength]} password
      </span>
    </div>
  )
}

export default function Signup() {
  const [name, setName] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const { setAuth } = useAuthStore()
  const navigate = useNavigate()

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (password.length < 6) { setError('Password must be at least 6 characters.'); return }
    setError('')
    setLoading(true)
    try {
      const res = await api.post('/auth/register', { name, email, password })
      setAuth(res.data.user, res.data.access_token)
      navigate('/setup')
    } catch (e: unknown) {
      setError(getApiErrorDetail(e, 'Registration failed. Please try again.'))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="auth-bg">
      <div className="auth-card animate-bounce-in">

        {/* Logo + Brand */}
        <div style={{ textAlign: 'center', marginBottom: '1.75rem' }}>
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
            Create your account
          </h1>
          <p style={{
            fontSize: '.85rem',
            color: 'hsl(var(--pencil))',
            fontFamily: 'Inter, sans-serif',
            lineHeight: 1.5,
          }}>
            Start transcribing conversations with AI
          </p>
        </div>

        <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '1.05rem' }}>

          {/* Name */}
          <div className="animate-slide-up" style={{ animationDelay: '0.08s', animationFillMode: 'both' }}>
            <label className="label">Full name</label>
            <div style={{ position: 'relative' }}>
              <User size={15} style={{
                position: 'absolute', left: '13px', top: '50%',
                transform: 'translateY(-50%)', color: 'hsl(var(--pencil))',
                zIndex: 1, pointerEvents: 'none',
              }} />
              <input
                id="signup-name"
                className="input"
                type="text"
                placeholder="John Doe"
                value={name}
                onChange={(e) => setName(e.target.value)}
                required
                style={{ paddingLeft: '2.4rem' }}
              />
            </div>
          </div>

          {/* Email */}
          <div className="animate-slide-up" style={{ animationDelay: '0.14s', animationFillMode: 'both' }}>
            <label className="label">Email address</label>
            <div style={{ position: 'relative' }}>
              <Mail size={15} style={{
                position: 'absolute', left: '13px', top: '50%',
                transform: 'translateY(-50%)', color: 'hsl(var(--pencil))',
                zIndex: 1, pointerEvents: 'none',
              }} />
              <input
                id="signup-email"
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
          <div className="animate-slide-up" style={{ animationDelay: '0.2s', animationFillMode: 'both' }}>
            <label className="label">Password</label>
            <div style={{ position: 'relative' }}>
              <Lock size={15} style={{
                position: 'absolute', left: '13px', top: '50%',
                transform: 'translateY(-50%)', color: 'hsl(var(--pencil))',
                zIndex: 1, pointerEvents: 'none',
              }} />
              <input
                id="signup-password"
                className="input"
                type="password"
                placeholder="Min. 6 characters"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                style={{ paddingLeft: '2.4rem' }}
              />
            </div>
            <PasswordStrength password={password} />
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

          {/* Features list */}
          <div style={{
            display: 'flex', flexDirection: 'column', gap: '5px',
            padding: '.75rem 1rem',
            background: 'hsl(var(--success) / .07)',
            border: '1px solid hsl(var(--success) / .2)',
            borderRadius: '10px',
            marginTop: '.1rem',
          }}>
            {[
              'Free forever plan included',
              'Speaker diarization powered by AI',
              'No credit card required',
            ].map((item) => (
              <div key={item} style={{
                display: 'flex', alignItems: 'center', gap: '8px',
                fontSize: '.8rem', fontFamily: 'Inter, sans-serif',
                color: 'hsl(var(--ink-soft))',
              }}>
                <CheckCircle2 size={13} style={{ color: 'hsl(var(--success))', flexShrink: 0 }} />
                {item}
              </div>
            ))}
          </div>

          {/* Submit */}
          <button
            id="signup-submit"
            className="btn btn-primary animate-slide-up"
            type="submit"
            disabled={loading}
            style={{
              width: '100%',
              justifyContent: 'center',
              padding: '.8rem 1.5rem',
              fontSize: '.95rem',
              animationDelay: '0.28s',
              animationFillMode: 'both',
            }}
          >
            {loading ? <Loader size={16} className="spin" /> : null}
            {loading ? 'Creating account…' : 'Create Account'}
          </button>
        </form>

        <div className="divider-text" style={{ margin: '1.4rem 0 1.1rem' }}>
          <span>already a member?</span>
        </div>

        <p style={{
          textAlign: 'center',
          fontSize: '0.9rem',
          color: 'hsl(var(--pencil))',
          fontFamily: 'Inter, sans-serif',
        }}>
          Already have an account?{' '}
          <Link
            to="/login"
            style={{
              color: 'hsl(var(--accent))',
              textDecoration: 'none',
              fontWeight: 700,
            }}
            className="scribble-underline"
          >
            Sign in →
          </Link>
        </p>
      </div>
    </div>
  )
}
