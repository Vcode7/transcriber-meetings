import { useState } from 'react'
import { CheckCircle, UserPlus, Loader, AlertTriangle, Sparkles, Mic, ArrowRight } from 'lucide-react'
import VoiceRecorder from '../components/VoiceRecorder'
import api from '../api/client'
import { getApiErrorDetail } from '../lib/errors'

const STEPS = [
  { title: 'Sample 1', instruction: 'Record a clear voice sample (10-30 seconds). Speak slowly and pronounce each word clearly.', num: '1' },
  { title: 'Sample 2', instruction: 'Record another sample in a normal conversational tone, as if talking in a real meeting.', num: '2' },
  { title: 'Sample 3', instruction: 'Optional but recommended -- speak with a slightly different pitch (excited, serious, or calm).', num: '3', optional: true },
]

const SAMPLE_SCRIPTS = [
  "Now I am speaking slowly and clearly. Each word is pronounced properly. This helps the system understand my voice in a clean way.",
  "Now I will speak normally, like in a real conversation. There will be some background noise and I will not be perfectly clear. This helps the system understand my voice in a real-world environment.",
  "Now I will speak with a slightly different pitch. This helps the system recognise my voice when I am excited, happy, calm or serious.",
]

export default function AddVoicePage() {
  const [label, setLabel] = useState('')
  const [savedPaths, setSavedPaths] = useState<(string | null)[]>([null, null, null])
  const [submitting, setSubmitting] = useState(false)
  const [success, setSuccess] = useState(false)
  const [error, setError] = useState('')
  const [step, setStep] = useState(0)

  const handleSampleSaved = (filePath: string, sampleIndex: number) => {
    setSavedPaths((prev) => {
      const next = [...prev]
      next[sampleIndex] = filePath
      return next
    })
  }

  const goNext = () => {
    if (!savedPaths[step]) {
      setError('Please record and save this sample before continuing.')
      return
    }
    setError('')
    setStep((s) => Math.min(s + 1, STEPS.length - 1))
  }

  const handleSave = async () => {
    if (!label.trim()) { setError('Please enter a name for this voice.'); return }
    const paths = savedPaths.filter(Boolean) as string[]
    if (paths.length < 2) { setError('Please record at least 2 samples for better accuracy.'); return }
    setSubmitting(true); setError('')
    try {
      await api.post('/voice/add-profile', { file_paths: paths, label: label.trim() })
      setSuccess(true)
      setLabel(''); setSavedPaths([null, null, null]); setStep(0)
    } catch (e: unknown) {
      setError(getApiErrorDetail(e, 'Failed to save profile.'))
    } finally {
      setSubmitting(false)
    }
  }

  if (success) return (
    <div style={{
      flex: 1, display: 'flex', flexDirection: 'column',
      alignItems: 'center', justifyContent: 'center',
      gap: '2rem', padding: '3rem',
      background: 'hsl(var(--paper) / .4)'
    }}>
      <div className="animate-bounce-in" style={{
        width: '100px', height: '100px', borderRadius: '50%',
        background: 'hsl(var(--success) / .12)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        border: '3px solid hsl(var(--success) / .35)',
        boxShadow: '0 0 0 14px hsl(var(--success) / .06)',
      }}>
        <CheckCircle size={52} style={{ color: 'hsl(var(--success))' }} />
      </div>
      <div style={{ textAlign: 'center' }}>
        <h2 style={{ fontWeight: 800, fontSize: '1.85rem', fontFamily: 'Caveat, cursive', marginBottom: '.65rem', color: 'hsl(var(--ink))', letterSpacing: '-.02em' }}>
          Voice profile saved!
        </h2>
        <p style={{ color: 'hsl(var(--pencil))', fontSize: '.95rem', fontFamily: 'Inter, sans-serif', maxWidth: '380px', lineHeight: 1.65 }}>
          VoiceSum will now recognise this person in future recordings
        </p>
      </div>
      <button
        className="btn btn-success animate-slide-up"
        onClick={() => setSuccess(false)}
        style={{ animationDelay: '0.2s', animationFillMode: 'both', padding: '.75rem 1.75rem', fontSize: '.95rem' }}
      >
        <UserPlus size={16} /> Add Another Voice
      </button>
    </div>
  )

  const doneCount = savedPaths.filter(Boolean).length

  return (
    <div className="page-scroll-root" style={{ display: 'flex', flexDirection: 'column' }}>

      <div className="panel-header">
        <div style={{
          width: '34px', height: '34px', borderRadius: '10px', flexShrink: 0,
          background: 'hsl(var(--accent) / .12)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          border: '2px solid hsl(var(--accent) / .3)',
        }}>
          <UserPlus size={16} style={{ color: 'hsl(var(--accent))' }} />
        </div>
        <div style={{ flex: 1 }}>
          <h1>Add Voice Profile</h1>
          <p style={{ fontSize: '.82rem', color: 'hsl(var(--pencil))', fontFamily: 'Inter, sans-serif', fontWeight: 400, marginTop: '1px' }}>
            Train speaker identification for a new person
          </p>
        </div>
      </div>

      <div className="page-wrapper">

        {/* Name */}
        <div className="animate-slide-up" style={{
          padding: '1.5rem', marginBottom: '1.5rem',
          background: 'hsl(var(--card))', border: '1.5px solid hsl(var(--ink) / .1)',
          borderRadius: '12px', animationDelay: '0.08s', animationFillMode: 'both'
        }}>
          <label className="label" style={{ marginBottom: '.75rem', fontSize: '.88rem' }}>
            Name / Label *
          </label>
          <input
            id="voice-label"
            className="input"
            placeholder="e.g. Alice, John, Boss"
            value={label}
            onChange={(e) => setLabel(e.target.value)}
          />
        </div>

        {/* Sample scripts */}
        <div className="animate-slide-up" style={{
          padding: '1.25rem', marginBottom: '1.5rem',
          background: 'hsl(205,90%,55% / .07)', border: '1.5px solid hsl(205,90%,55% / .25)',
          borderRadius: '12px', animationDelay: '0.12s', animationFillMode: 'both'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '1rem' }}>
            <Sparkles size={16} style={{ color: 'hsl(205,90%,55%)' }} />
            <h3 style={{ fontSize: '.95rem', fontWeight: 700, fontFamily: 'Inter, sans-serif', color: 'hsl(var(--ink))' }}>
              Sample Scripts <span style={{ fontWeight: 400, color: 'hsl(var(--pencil))' }}>(read aloud)</span>
            </h3>
          </div>
          {SAMPLE_SCRIPTS.map((script, idx) => (
            <div key={idx} style={{
              padding: '.85rem 1rem',
              background: step === idx ? 'hsl(var(--accent) / .07)' : 'hsl(var(--card))',
              borderRadius: '8px',
              border: step === idx ? '1.5px solid hsl(var(--accent) / .3)' : '1.5px solid hsl(var(--ink) / .08)',
              marginBottom: idx < SAMPLE_SCRIPTS.length - 1 ? '.75rem' : 0,
              fontSize: '.88rem', lineHeight: 1.7, fontFamily: 'Inter, sans-serif',
              color: 'hsl(var(--ink-soft))', display: 'flex', gap: '.75rem', transition: 'all .2s',
            }}>
              <span style={{ fontWeight: 700, color: step === idx ? 'hsl(var(--accent))' : 'hsl(var(--pencil))', flexShrink: 0, fontSize: '.85rem' }}>
                {idx + 1}.
              </span>
              {script}
            </div>
          ))}
        </div>

        {/* Step tabs */}
        <div className="animate-slide-up" style={{
          display: 'flex', gap: '8px', marginBottom: '1.5rem',
          animationDelay: '0.16s', animationFillMode: 'both'
        }}>
          {STEPS.map((s, i) => {
            const isDone = !!savedPaths[i]
            const isActive = step === i
            return (
              <button
                key={i}
                onClick={() => { setError(''); setStep(i) }}
                style={{
                  flex: 1, padding: '.85rem .5rem', borderRadius: '10px',
                  border: isActive ? '2px solid hsl(var(--accent))' : isDone ? '2px solid hsl(130,60%,45% / .4)' : '1.5px dashed hsl(var(--ink) / .15)',
                  cursor: 'pointer',
                  background: isActive ? 'hsl(var(--accent) / .1)' : isDone ? 'hsl(130,60%,45% / .08)' : 'hsl(var(--card))',
                  color: isActive ? 'hsl(var(--accent))' : isDone ? 'hsl(130,60%,45%)' : 'hsl(var(--pencil))',
                  fontSize: '0.85rem', fontWeight: 600,
                  display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '4px',
                  transition: 'all 0.2s', fontFamily: 'Inter, sans-serif',
                  boxShadow: isActive ? '0 2px 8px hsl(var(--accent) / .15)' : 'none'
                }}
              >
                <div style={{
                  width: '24px', height: '24px', borderRadius: '50%',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  background: isDone ? 'hsl(130,60%,45%)' : isActive ? 'hsl(var(--accent))' : 'hsl(var(--muted))',
                  color: isDone || isActive ? '#fff' : 'hsl(var(--pencil))',
                  fontSize: '.72rem', fontWeight: 700, transition: 'all .2s'
                }}>
                  {isDone ? '\u2713' : s.num}
                </div>
                {s.title}
                {s.optional && (
                  <span style={{ fontSize: '.62rem', fontWeight: 400, color: 'hsl(var(--pencil))', opacity: .75 }}>Optional</span>
                )}
              </button>
            )
          })}
        </div>

        {/* Current recorder -- key={step} forces full remount on tab switch, clearing all recorder state */}
        <div className="animate-slide-up" style={{ marginBottom: '1.25rem', animationDelay: '0.2s', animationFillMode: 'both' }}>
          <div style={{
            padding: '1.5rem', background: 'hsl(var(--card))',
            border: '1.5px solid hsl(var(--ink) / .1)', borderRadius: '12px', marginBottom: '1rem'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '1.1rem' }}>
              <Mic size={16} style={{ color: 'hsl(var(--accent))', flexShrink: 0 }} />
              <p style={{ fontSize: '.9rem', color: 'hsl(var(--ink-soft))', fontFamily: 'Inter, sans-serif', fontWeight: 500 }}>
                {STEPS[step].instruction}
              </p>
            </div>
            <VoiceRecorder
              key={step}
              sampleIndex={step}
              label={label || 'speaker'}
              onSampleSaved={handleSampleSaved}
            />
          </div>

          {/* Navigation */}
          <div style={{ display: 'flex', gap: '8px' }}>
            {step > 0 && (
              <button
                className="btn btn-ghost"
                onClick={() => { setError(''); setStep(step - 1) }}
                style={{ flex: 1, justifyContent: 'center', padding: '.65rem 1.25rem', fontSize: '.9rem' }}
              >
                Previous
              </button>
            )}
            {step < STEPS.length - 1 && (
              <button
                className="btn btn-ghost"
                onClick={goNext}
                style={{ flex: 1, justifyContent: 'center', padding: '.65rem 1.25rem', fontSize: '.9rem' }}
              >
                Next sample <ArrowRight size={14} style={{ marginLeft: '4px' }} />
              </button>
            )}
          </div>
        </div>

        {/* Progress dots */}
        <div style={{
          display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '1.25rem',
          padding: '.75rem 1rem', borderRadius: '10px',
          background: 'hsl(var(--muted) / .5)', border: '1.5px solid hsl(var(--ink) / .08)',
          fontFamily: 'Inter, sans-serif', fontSize: '.82rem', color: 'hsl(var(--pencil))',
        }}>
          <div style={{ display: 'flex', gap: '5px', alignItems: 'center' }}>
            {savedPaths.map((p, i) => (
              <div key={i} style={{
                width: 10, height: 10, borderRadius: '50%',
                background: p ? 'hsl(130,60%,45%)' : 'hsl(var(--muted))',
                border: `1.5px solid ${p ? 'hsl(130,60%,45% / .5)' : 'hsl(var(--ink) / .15)'}`,
                transition: 'all .2s',
              }} />
            ))}
          </div>
          <span>
            {doneCount} of 3 samples recorded
            {doneCount >= 2 ? ' \u2014 ready to save!' : ' \u2014 need at least 2'}
          </span>
        </div>

        {/* Error */}
        {error && (
          <div className="animate-shake" style={{
            display: 'flex', alignItems: 'center', gap: '10px',
            color: 'hsl(var(--destructive))', fontSize: '0.88rem', marginBottom: '1.25rem',
            padding: '.75rem 1rem', background: 'hsl(var(--destructive) / .08)',
            border: '1.5px solid hsl(var(--destructive) / .25)',
            borderRadius: '10px', fontFamily: 'Inter, sans-serif', fontWeight: 500
          }}>
            <AlertTriangle size={15} /> {error}
          </div>
        )}

        {/* Save */}
        <button
          className="btn btn-primary"
          onClick={handleSave}
          disabled={submitting || doneCount < 2}
          id="save-voice-btn"
          style={{ width: '100%', justifyContent: 'center', padding: '.8rem 1.5rem', fontSize: '.95rem', marginTop: '.5rem' }}
        >
          {submitting ? <Loader size={16} className="spin" /> : <CheckCircle size={16} />}
          {submitting ? 'Saving...' : `Save Voice Profile (${doneCount} sample${doneCount !== 1 ? 's' : ''})`}
        </button>
      </div>
    </div>
  )
}
