import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { CheckCircle, Mic, MessageSquare, Volume2, Loader, ChevronRight } from 'lucide-react'
import VoiceRecorder from '../components/VoiceRecorder'
import { useAuthStore } from '../store/auth'
import api from '../api/client'
import { getApiErrorDetail } from '../lib/errors'

const STEPS = [
  {
    title: 'Script Reading',
    icon: <MessageSquare size={20} />,
    instruction: 'Please read the following text aloud clearly and naturally:',
    script: '"The quick brown fox jumps over the lazy dog. Voice recognition systems need clear audio samples to work accurately. Please speak at your normal pace and volume."',
  },
  {
    title: 'Free Speech',
    icon: <Mic size={20} />,
    instruction: 'Speak freely for 15–30 seconds. Talk about your day, a hobby, or anything you like.',
    script: null,
  },
  {
    title: 'Variation (Optional)',
    icon: <Volume2 size={20} />,
    instruction: 'Speak in a slightly different tone — faster, slower, or in a different environment. This improves recognition accuracy.',
    script: '"Hello, this is my voice profile sample. I am recording this to help the system identify my voice in conversations."',
  },
]

export default function Setup() {
  const [step, setStep] = useState(0)
  const [savedPaths, setSavedPaths] = useState<string[]>([])
  const [label, setLabel] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState('')
  const { user, updateUser } = useAuthStore()
  const navigate = useNavigate()

  const handleSampleSaved = (filePath: string, sampleIndex: number) => {
    setSavedPaths((prev) => {
      const next = [...prev]
      next[sampleIndex] = filePath
      return next
    })
  }

  const handleFinalize = async () => {
    if (savedPaths.filter(Boolean).length === 0) {
      setError('Please record at least one sample.')
      return
    }
    setSubmitting(true)
    setError('')
    try {
      const res = await api.post('/voice/finalize-setup', {
        file_paths: savedPaths.filter(Boolean),
        label: label.trim() || user?.name || 'Me',
      })
      updateUser({ needs_setup: false, own_profile_id: res.data.profile_id })
      navigate('/dashboard')
    } catch (e: unknown) {
      setError(getApiErrorDetail(e, 'Setup failed.'))
    } finally {
      setSubmitting(false)
    }
  }

  const handleSkip = async () => {
    setSubmitting(true)
    setError('')
    try {
      await api.post('/voice/skip-setup')
      updateUser({ needs_setup: false })
      navigate('/dashboard')
    } catch (e: unknown) {
      setError(getApiErrorDetail(e, 'Failed to skip setup.'))
    } finally {
      setSubmitting(false)
    }
  }

  const stepDone = (i: number) => Boolean(savedPaths[i])
  const canFinalize = savedPaths.filter(Boolean).length >= 1

  return (
    <div className="auth-bg" style={{ alignItems: 'flex-start', padding: '2rem 1rem' }}>
      <div style={{ width: '100%', maxWidth: '560px', margin: '0 auto' }} className="fade-in">
        {/* Header */}
        <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
          <h1 style={{
            fontSize: '1.5rem', fontWeight: 800,
            fontFamily: 'Inter, sans-serif',
            color: 'hsl(var(--ink))',
            lineHeight: 1.1, marginBottom: '.4rem'
          }}>
            Set Up Your Voice Profile
          </h1>
          <p style={{
            color: 'hsl(var(--pencil))',
            marginTop: '0.4rem', fontSize: '0.9rem',
            fontFamily: 'Inter, sans-serif'
          }}>
            Record 1–3 voice samples so VoiceSum can identify you in conversations.
          </p>
        </div>

        {/* Step tabs */}
        <div style={{ display: 'flex', gap: '8px', marginBottom: '1.5rem' }}>
          {STEPS.map((s, i) => (
            <button
              key={i}
              onClick={() => setStep(i)}
              style={{
                flex: 1, padding: '0.6rem 0.5rem',
                borderRadius: '10px 14px 12px 16px / 14px 10px 16px 12px',
                border: step === i
                  ? '2px solid hsl(var(--accent))'
                  : stepDone(i)
                  ? '2px solid hsl(130, 60%, 45% / .4)'
                  : '1.5px dashed hsl(var(--ink) / .15)',
                cursor: 'pointer',
                background: step === i
                  ? 'hsl(var(--accent) / .1)'
                  : stepDone(i)
                  ? 'hsl(130, 60%, 45% / .08)'
                  : 'hsl(var(--card))',
                color: step === i
                  ? 'hsl(var(--accent))'
                  : stepDone(i)
                  ? 'hsl(130, 60%, 45%)'
                  : 'hsl(var(--pencil))',
                fontSize: '0.78rem', fontWeight: 600,
                display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px',
                transition: 'all 0.18s',
                position: 'relative',
                fontFamily: 'Inter, sans-serif',
                boxShadow: step === i ? '0 2px 8px hsl(var(--accent) / .12)' : 'none'
              }}
            >
              {stepDone(i) && <CheckCircle size={12} style={{ color: step === i ? 'hsl(var(--accent))' : 'hsl(130, 60%, 45%)' }} />}
              {s.title}
            </button>
          ))}
        </div>

        {/* Current step */}
        <div className="glass" style={{ padding: '1.5rem', marginBottom: '1.5rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '1rem' }}>
            <div style={{ color: 'hsl(var(--accent))' }}>{STEPS[step].icon}</div>
            <h2 style={{
              fontWeight: 700, fontSize: '1.05rem',
              fontFamily: 'Inter, sans-serif',
              color: 'hsl(var(--ink))'
            }}>
              {STEPS[step].title}
            </h2>
            {step === 2 && <span className="badge badge-yellow">Optional</span>}
          </div>
          <p style={{
            color: 'hsl(var(--pencil))',
            fontSize: '0.875rem',
            marginBottom: STEPS[step].script ? '1rem' : '1.25rem',
            fontFamily: 'Inter, sans-serif'
          }}>
            {STEPS[step].instruction}
          </p>
          {STEPS[step].script && (
            <div style={{
              background: 'hsl(var(--muted))',
              border: '1.5px solid hsl(var(--ink) / .1)',
              borderRadius: '10px',
              padding: '1rem', marginBottom: '1.25rem',
              fontStyle: 'italic',
              color: 'hsl(var(--ink))',
              fontSize: '0.9rem', lineHeight: 1.7,
              fontFamily: 'Inter, sans-serif'
            }}>
              {STEPS[step].script}
            </div>
          )}
          <VoiceRecorder
            key={step}
            sampleIndex={step}
            label={label || user?.name || 'Me'}
            onSampleSaved={handleSampleSaved}
          />
        </div>

        {/* Navigation */}
        <div style={{ display: 'flex', gap: '10px', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap' }}>
          <div style={{ display: 'flex', gap: '8px' }}>
            {step < STEPS.length - 1 && (
              <button className="btn btn-ghost" onClick={() => setStep(step + 1)}>
                Next Step <ChevronRight size={15} />
              </button>
            )}
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: '8px' }}>
            {/* Label */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <label style={{ fontSize: '0.8rem', color: 'hsl(var(--pencil))' }}>Profile name:</label>
              <input
                className="input"
                placeholder={user?.name || 'Your Name'}
                value={label}
                onChange={(e) => setLabel(e.target.value)}
                style={{ width: '160px' }}
              />
            </div>
            <button
              className="btn btn-primary"
              onClick={handleFinalize}
              disabled={!canFinalize || submitting}
              id="finalize-setup"
            >
              {submitting ? <Loader size={15} className="spin" /> : <CheckCircle size={15} />}
              {submitting ? 'Saving…' : 'Complete Setup'}
            </button>
            {error && (
              <div style={{
                color: 'hsl(var(--destructive))',
                fontSize: '0.82rem',
                fontFamily: 'Inter, sans-serif',
                fontWeight: 500
              }}>
                {error}
              </div>
            )}
          </div>
        </div>

        {/* Skip */}
        <div style={{ textAlign: 'center', marginTop: '1.5rem' }}>
          <button
            style={{
              background: 'none', border: 'none',
              color: 'hsl(var(--pencil))',
              fontSize: '0.82rem', cursor: 'pointer',
              textDecoration: 'underline',
              fontFamily: 'Inter, sans-serif'
            }}
            onClick={handleSkip}
            disabled={submitting}
          >
            {submitting ? 'Skipping...' : 'Skip for now (speaker identification will be limited)'}
          </button>
        </div>
      </div>
    </div>
  )
}
