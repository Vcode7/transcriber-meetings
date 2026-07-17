import { useEffect, useState, useRef } from 'react'
import {
  Settings, Mic, Trash2, Pencil, Save, Loader, Sliders, Sparkles, User, CheckCircle,
  MessageSquare, RotateCcw, Upload, Download, FileText, Code
} from 'lucide-react'
import api from '../api/client'
import { toast } from 'sonner'

interface Profile {
  id: string
  label: string
  sample_count: number
  is_self: boolean
  created_at: string
}

interface UserSettings {
  speaker_similarity_threshold: number
  word_conf_low: number
  word_conf_mid: number
  min_segment_duration: number
}

interface PromptTemplate {
  key: string
  name: string
  category: string
  description: string
  variables: string[]
  template: string
  default_template: string
  is_modified: boolean
  updated_at: string | null
}

export default function SettingsPage() {
  const [profiles, setProfiles] = useState<Profile[]>([])
  const [settings, setSettings] = useState<UserSettings | null>(null)
  const [loadingProfiles, setLoadingProfiles] = useState(true)
  const [editingId, setEditingId] = useState<string | null>(null)
  const [editLabel, setEditLabel] = useState('')
  const [savingLabel, setSavingLabel] = useState(false)
  const [deletingId, setDeletingId] = useState<string | null>(null)
  const [savingSettings, setSavingSettings] = useState(false)
  const [settingsSaved, setSettingsSaved] = useState(false)

  // Global prompt
  const [globalPrompt, setGlobalPrompt] = useState('')
  const [promptSaving, setPromptSaving] = useState(false)
  const [promptSaved, setPromptSaved] = useState(false)
  const promptDebounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  // Prompt Templates state
  const [prompts, setPrompts] = useState<PromptTemplate[]>([])
  const [loadingPrompts, setLoadingPrompts] = useState(true)
  const [activeCategory, setActiveCategory] = useState('MoM')
  const [savingKeys, setSavingKeys] = useState<Set<string>>(new Set())
  const [savedKeys, setSavedKeys] = useState<Set<string>>(new Set())
  const [importing, setImporting] = useState(false)
  const [editingTemplates, setEditingTemplates] = useState<Record<string, string>>({})

  const loadData = async () => {
    try {
      const [pRes, sRes, prRes, ptRes] = await Promise.all([
        api.get('/voice/profiles'),
        api.get('/settings'),
        api.get('/prompt/global'),
        api.get('/prompt-templates'),
      ])
      setProfiles(pRes.data)
      setSettings(sRes.data)
      setGlobalPrompt(prRes.data?.prompt || '')
      setPrompts(ptRes.data)
    } catch (err) {
      toast.error('Failed to load settings data')
    } finally {
      setLoadingProfiles(false)
      setLoadingPrompts(false)
    }
  }
  useEffect(() => { loadData() }, [])

  const handlePromptChange = (value: string) => {
    setGlobalPrompt(value)
    setPromptSaved(false)
    if (promptDebounceRef.current) clearTimeout(promptDebounceRef.current)
    promptDebounceRef.current = setTimeout(async () => {
      setPromptSaving(true)
      try {
        await api.put('/prompt/global', { prompt: value })
        setPromptSaved(true)
        setTimeout(() => setPromptSaved(false), 2500)
      } catch { } finally { setPromptSaving(false) }
    }, 600)
  }

  const handleRename = async (id: string) => {
    if (!editLabel.trim()) return
    setSavingLabel(true)
    await api.put(`/voice/profiles/${id}`, { label: editLabel })
    setProfiles((prev) => prev.map((p) => p.id === id ? { ...p, label: editLabel } : p))
    setEditingId(null)
    setSavingLabel(false)
  }

  const handleDelete = async (id: string) => {
    if (!confirm('Delete this voice profile?')) return
    setDeletingId(id)
    await api.delete(`/voice/profiles/${id}`)
    setProfiles((prev) => prev.filter((p) => p.id !== id))
    setDeletingId(null)
  }

  const handleSaveSettings = async () => {
    if (!settings) return
    setSavingSettings(true)
    await api.put('/settings', settings)
    setSavingSettings(false)
    setSettingsSaved(true)
    setTimeout(() => setSettingsSaved(false), 2000)
  }

  const upd = (key: keyof UserSettings, val: number) =>
    setSettings((prev) => prev ? { ...prev, [key]: val } : prev)

  const handleSavePromptTemplate = async (key: string, template: string) => {
    setSavingKeys(prev => {
      const next = new Set(prev)
      next.add(key)
      return next
    })
    try {
      await api.put(`/prompt-templates/${key}`, { template })
      setPrompts(prev => prev.map(p => p.key === key ? { ...p, template, is_modified: true } : p))
      setSavedKeys(prev => {
        const next = new Set(prev)
        next.add(key)
        return next
      })
      setTimeout(() => {
        setSavedKeys(prev => {
          const next = new Set(prev)
          next.delete(key)
          return next
        })
      }, 2000)
      toast.success('Prompt template saved successfully')
    } catch (err: any) {
      toast.error(err?.response?.data?.detail ?? 'Failed to save prompt template')
    } finally {
      setSavingKeys(prev => {
        const next = new Set(prev)
        next.delete(key)
        return next
      })
    }
  }

  const handleResetPromptTemplate = async (key: string) => {
    if (!confirm('Are you sure you want to reset this prompt template to its default?')) return
    try {
      await api.delete(`/prompt-templates/${key}`)
      const res = await api.get(`/prompt-templates/${key}`)
      setPrompts(prev => prev.map(p => p.key === key ? {
        ...p,
        template: res.data.template,
        is_modified: false,
        updated_at: null
      } : p))
      toast.success('Prompt template reset to default')
    } catch (err: any) {
      toast.error('Failed to reset prompt template')
    }
  }

  const handleResetAllPrompts = async () => {
    if (!confirm('Are you sure you want to reset ALL prompt templates to their defaults? This cannot be undone.')) return
    try {
      await api.delete('/prompt-templates')
      const ptRes = await api.get('/prompt-templates')
      setPrompts(ptRes.data)
      toast.success('All prompt templates reset to defaults')
    } catch (err) {
      toast.error('Failed to reset all prompt templates')
    }
  }

  const handleExportPrompts = async () => {
    try {
      const res = await api.get('/prompt-templates/export')
      const blob = new Blob([JSON.stringify(res.data, null, 2)], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `prompt_templates_export_${new Date().toISOString().split('T')[0]}.json`
      a.click()
      URL.revokeObjectURL(url)
      toast.success('Prompt templates exported successfully')
    } catch (err) {
      toast.error('Failed to export prompt templates')
    }
  }

  const handleImportPrompts = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    setImporting(true)
    try {
      const text = await file.text()
      const data = JSON.parse(text)
      const res = await api.post('/prompt-templates/import', data)
      const ptRes = await api.get('/prompt-templates')
      setPrompts(ptRes.data)
      toast.success(res.data.message || 'Prompt templates imported successfully')
    } catch (err: any) {
      toast.error(err?.response?.data?.detail ?? 'Failed to import prompt templates. Check file format.')
    } finally {
      setImporting(false)
      e.target.value = ''
    }
  }

  const PROFILE_COLORS = [
    'hsl(14, 90%, 56%)',
    'hsl(205, 90%, 55%)',
    'hsl(130, 60%, 45%)',
    'hsl(280, 70%, 60%)',
    'hsl(45, 90%, 50%)',
  ]

  return (
    <div className="page-scroll-root" style={{ display: 'flex', flexDirection: 'column' }}>

      {/* Panel Header */}
      <div className="panel-header">
        <div style={{
          width: '34px', height: '34px', borderRadius: '10px', flexShrink: 0,
          background: 'hsl(var(--accent) / .12)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          border: '2px solid hsl(var(--accent) / .3)',
        }}>
          <Settings size={16} style={{ color: 'hsl(var(--accent))' }} />
        </div>
        <div style={{ flex: 1 }}>
          <h1>Settings</h1>
          <p style={{ fontSize: '.82rem', color: 'hsl(var(--pencil))', fontFamily: 'Inter, sans-serif', fontWeight: 400, marginTop: '1px' }}>
            Voice profiles &amp; recognition thresholds
          </p>
        </div>
      </div>

      {/* Scrollable content */}
      <div className="page-wrapper">

      {/* Page title - REMOVED, now in panel-header */}

      {/* ─── Global Transcription Prompt ─── */}
      <section className="animate-slide-up" style={{ marginBottom: '2.5rem', animationDelay: '0.02s', animationFillMode: 'both' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '1.25rem' }}>
          <div style={{
            width: '36px', height: '36px', borderRadius: '9px',
            background: 'hsl(280,70%,60% / .12)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            border: '1.5px solid hsl(280,70%,60% / .25)',
          }}>
            <MessageSquare size={18} style={{ color: 'hsl(280,70%,60%)' }} />
          </div>
          <h2 style={{ fontSize: '1.15rem', fontWeight: 700, fontFamily: 'Inter, sans-serif', color: 'hsl(var(--ink))' }}>
            Global Transcription Prompt
          </h2>
          <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: '6px' }}>
            {promptSaving && <Loader size={13} className="spin" style={{ color: 'hsl(var(--pencil))' }} />}
            {promptSaved && <span style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '.78rem', color: 'hsl(130,60%,45%)', fontFamily: 'Inter, sans-serif', fontWeight: 600 }}>
              <CheckCircle size={13} /> Saved
            </span>}
          </div>
        </div>

        <div style={{
          padding: '1.25rem',
          background: 'hsl(var(--card))',
          border: '1.5px solid hsl(var(--ink) / .1)',
          borderRadius: '12px',
        }}>
          <textarea
            id="global-prompt-textarea"
            value={globalPrompt}
            onChange={e => handlePromptChange(e.target.value)}
            placeholder="Enter a global system prompt for all transcriptions…\n\nExamples:\n\u2022 This is a technical meeting in the healthcare domain.\n\u2022 The participants speak English with Indian accents.\n\u2022 Use formal language and preserve acronyms as-is."
            rows={5}
            style={{
              width: '100%',
              padding: '.75rem',
              borderRadius: '8px',
              background: 'hsl(var(--muted) / .4)',
              border: '1.5px solid hsl(var(--ink) / .1)',
              color: 'hsl(var(--ink))',
              fontFamily: 'Inter, sans-serif',
              fontSize: '.88rem',
              lineHeight: 1.7,
              resize: 'vertical',
              outline: 'none',
              boxSizing: 'border-box',
              transition: 'border-color .15s',
            }}
            onFocus={e => (e.currentTarget.style.borderColor = 'hsl(280,70%,60% / .5)')}
            onBlur={e => (e.currentTarget.style.borderColor = 'hsl(var(--ink) / .1)')}
          />
          <p style={{ fontSize: '.78rem', color: 'hsl(var(--pencil))', marginTop: '.6rem', fontFamily: 'Inter, sans-serif', lineHeight: 1.6 }}>
            This prompt is automatically included in every Whisper transcription request.
            Keep it under 200 words for best results. Auto-saved as you type.
          </p>
        </div>
      </section>

      {/* ─── Voice Profiles ─── */}
      <section className="animate-slide-up" style={{ marginBottom: '2.5rem', animationDelay: '0.05s', animationFillMode: 'both' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '1.5rem' }}>
          <div style={{
            width: '36px', height: '36px', borderRadius: '9px',
            background: 'hsl(205,90%,55% / .12)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            border: '1.5px solid hsl(205,90%,55% / .25)',
          }}>
            <Mic size={18} style={{ color: 'hsl(205,90%,55%)' }} />
          </div>
          <h2 style={{ fontSize: '1.15rem', fontWeight: 700, fontFamily: 'Inter, sans-serif', color: 'hsl(var(--ink))' }}>
            Voice Profiles
          </h2>
          <span style={{
            fontSize: '.76rem', fontWeight: 600,
            color: 'hsl(235,80%,60%)',
            background: 'hsl(235,80%,60% / .1)',
            padding: '.15rem .55rem', borderRadius: '999px',
            border: '1.5px solid hsl(235,80%,60% / .25)',
            fontFamily: 'Inter, sans-serif'
          }}>
            {profiles.length}
          </span>
        </div>

        {loadingProfiles ? (
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '4rem', flexDirection: 'column', gap: '1rem' }}>
            <Loader size={28} className="spin" style={{ color: 'hsl(var(--accent))' }} />
            <p style={{ fontFamily: 'Inter, sans-serif', fontSize: '.9rem', color: 'hsl(var(--pencil))' }}>Loading profiles…</p>
          </div>
        ) : profiles.length === 0 ? (
          <div style={{
            padding: '3.5rem 2rem', textAlign: 'center',
            background: 'hsl(var(--card))',
            border: '1.5px dashed hsl(var(--ink) / .15)',
            borderRadius: '12px'
          }}>
            <div style={{
              width: '60px', height: '60px', borderRadius: '50%',
              background: 'hsl(var(--accent) / .08)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              margin: '0 auto 1.25rem'
            }}>
              <User size={28} style={{ opacity: 0.3, color: 'hsl(var(--accent))' }} className="animate-float" />
            </div>
            <p style={{ color: 'hsl(var(--pencil))', fontSize: '.95rem', fontFamily: 'Inter, sans-serif' }}>
              No voice profiles saved yet
            </p>
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            {profiles.map((p, idx) => {
              const color = PROFILE_COLORS[idx % PROFILE_COLORS.length]
              const sampleDots = Math.min(p.sample_count, 10)
              return (
                <div
                  key={p.id}
                  className="animate-slide-up sketch-border"
                  style={{
                    display: 'flex', alignItems: 'center', gap: '16px',
                    padding: '1.1rem 1.25rem',
                    background: 'hsl(var(--card))',
                    borderLeft: `4px solid ${color}`,
                    borderRadius: '0 12px 12px 0',
                    animationDelay: `${0.1 + idx * 0.04}s`,
                    animationFillMode: 'both',
                    transition: 'box-shadow .2s, transform .2s',
                  }}
                  onMouseEnter={e => {
                    const el = e.currentTarget as HTMLDivElement
                    el.style.boxShadow = `4px 4px 0 ${color}25, 0 4px 16px ${color}15`
                    el.style.transform = 'translateX(2px)'
                  }}
                  onMouseLeave={e => {
                    const el = e.currentTarget as HTMLDivElement
                    el.style.boxShadow = 'none'
                    el.style.transform = 'none'
                  }}
                >
                  {/* Avatar with hover glow */}
                  <div
                    style={{
                      width: '48px', height: '48px', borderRadius: '50%',
                      background: `${color}20`,
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      color, fontWeight: 800, fontSize: '1.2rem', flexShrink: 0,
                      border: `2px solid ${color}50`,
                      fontFamily: 'Inter, sans-serif',
                      transition: 'transform .3s cubic-bezier(0.34,1.56,.64,1), box-shadow .3s',
                      cursor: 'default',
                    }}
                    onMouseEnter={e => {
                      const el = e.currentTarget as HTMLDivElement
                      el.style.transform = 'rotate(8deg) scale(1.08)'
                      el.style.boxShadow = `0 0 16px ${color}55`
                    }}
                    onMouseLeave={e => {
                      const el = e.currentTarget as HTMLDivElement
                      el.style.transform = 'none'
                      el.style.boxShadow = 'none'
                    }}
                  >
                    {p.label[0]?.toUpperCase()}
                  </div>

                  {/* Info */}
                  <div style={{ flex: 1, minWidth: 0 }}>
                    {editingId === p.id ? (
                      <input
                        className="input"
                        value={editLabel}
                        onChange={(e) => setEditLabel(e.target.value)}
                        onKeyDown={(e) => e.key === 'Enter' && handleRename(p.id)}
                        style={{ padding: '0.5rem 0.75rem', fontSize: '.95rem', height: '36px' }}
                        autoFocus
                      />
                    ) : (
                      <div style={{ fontWeight: 700, fontSize: '1rem', fontFamily: 'Inter, sans-serif', color: 'hsl(var(--ink))', marginBottom: '6px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                        {p.label}
                        {p.is_self && (
                          <span style={{
                            fontSize: '.68rem', fontWeight: 600,
                            color: 'hsl(235,80%,60%)', background: 'hsl(235,80%,60% / .1)',
                            padding: '.1rem .45rem', borderRadius: '999px',
                            border: '1.5px solid hsl(235,80%,60% / .25)',
                            fontFamily: 'Inter, sans-serif',
                            display: 'inline-flex', alignItems: 'center', gap: '3px'
                          }}>
                            <Sparkles size={10} /> You
                          </span>
                        )}
                      </div>
                    )}
                    {/* Sample count as dots */}
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <div style={{ display: 'flex', gap: '3px', alignItems: 'center' }}>
                        {Array.from({ length: sampleDots }).map((_, i) => (
                          <span key={i} style={{ width: '7px', height: '7px', borderRadius: '50%', background: color, opacity: 0.65 + (i / sampleDots) * 0.35 }} />
                        ))}
                        {p.sample_count > 10 && <span style={{ fontSize: '.68rem', color: 'hsl(var(--pencil))', fontFamily: 'Inter, sans-serif' }}>+{p.sample_count - 10}</span>}
                      </div>
                      <span style={{ fontSize: '0.76rem', color: 'hsl(var(--pencil))', fontFamily: 'Inter, sans-serif' }}>
                        {p.sample_count} sample{p.sample_count !== 1 ? 's' : ''} · {new Date(p.created_at).toLocaleDateString()}
                      </span>
                    </div>
                  </div>

                  {/* Actions */}
                  <div style={{ display: 'flex', gap: '8px', flexShrink: 0 }}>
                    {editingId === p.id ? (
                      <button
                        className="btn btn-primary"
                        style={{ padding: '0.5rem 1rem', fontSize: '0.85rem', height: '36px' }}
                        onClick={() => handleRename(p.id)}
                        disabled={savingLabel}
                      >
                        {savingLabel ? <Loader size={13} className="spin" /> : <Save size={13} />} Save
                      </button>
                    ) : (
                      <button
                        className="icon-btn"
                        style={{ width: '36px', height: '36px' }}
                        onClick={() => { setEditingId(p.id); setEditLabel(p.label) }}
                        title="Rename"
                      >
                        <Pencil size={14} />
                      </button>
                    )}
                    <button
                      className="icon-btn"
                      style={{ width: '36px', height: '36px', color: 'hsl(var(--destructive) / .7)', transition: 'color .15s' }}
                      onClick={() => handleDelete(p.id)}
                      disabled={deletingId === p.id}
                      title="Delete"
                      onMouseEnter={e => (e.currentTarget as HTMLButtonElement).style.color = 'hsl(var(--destructive))'}
                      onMouseLeave={e => (e.currentTarget as HTMLButtonElement).style.color = 'hsl(var(--destructive) / .7)'}
                    >
                      {deletingId === p.id ? <Loader size={14} className="spin" /> : <Trash2 size={14} />}
                    </button>
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </section>

      {/* ─── Recognition Thresholds ─── */}
      {settings && (
        <section className="animate-slide-up" style={{ animationDelay: '0.2s', animationFillMode: 'both' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '1.5rem' }}>
            <div style={{
              width: '36px', height: '36px', borderRadius: '9px',
              background: 'hsl(45,90%,50% / .15)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              border: '1.5px solid hsl(45,90%,50% / .3)',
            }}>
              <Sliders size={18} style={{ color: 'hsl(45,90%,50%)' }} />
            </div>
            <h2 style={{ fontSize: '1.15rem', fontWeight: 700, fontFamily: 'Inter, sans-serif', color: 'hsl(var(--ink))' }}>
              Recognition Thresholds
            </h2>
          </div>

          <div style={{
            padding: '1.75rem',
            background: 'hsl(var(--card))',
            border: '1.5px solid hsl(var(--ink) / .1)',
            borderRadius: '12px',
            display: 'flex', flexDirection: 'column', gap: '2rem'
          }}>
            {([
              { key: 'speaker_similarity_threshold', label: 'Speaker Similarity Threshold', min: 0.5, max: 0.99, step: 0.01, desc: 'Minimum cosine similarity to match a known speaker (default: 0.73)' },
              { key: 'word_conf_low', label: 'Low Confidence Threshold', min: 0.3, max: 0.9, step: 0.01, desc: 'Words below this are highlighted red (default: 0.70)' },
              { key: 'word_conf_mid', label: 'Mid Confidence Threshold', min: 0.5, max: 0.99, step: 0.01, desc: 'Words below this are highlighted yellow (default: 0.85)' },
              { key: 'min_segment_duration', label: 'Min. Segment Duration (s)', min: 0.5, max: 5, step: 0.5, desc: 'Segments shorter than this are ignored (default: 1.5s)' },
            ] as const).map(({ key, label, min, max, step, desc }) => {
              return (
                <div key={key}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '10px', alignItems: 'center' }}>
                    <label style={{ fontSize: '.95rem', fontWeight: 600, fontFamily: 'Inter, sans-serif', color: 'hsl(var(--ink))' }}>
                      {label}
                    </label>
                    <input
                      type="number"
                      min={min} max={max} step={step}
                      value={settings[key]}
                      onChange={(e) => {
                        const parsed = parseFloat(e.target.value)
                        upd(key, isNaN(parsed) ? min : parsed)
                      }}
                      style={{
                        fontFamily: 'JetBrains Mono, monospace', fontSize: '.9rem',
                        color: 'hsl(var(--accent))', fontWeight: 700,
                        padding: '.3rem .65rem',
                        background: 'hsl(var(--accent) / .1)',
                        borderRadius: '6px',
                        border: '1.5px solid hsl(var(--accent) / .2)',
                        width: '75px', textAlign: 'center',
                        outline: 'none',
                      }}
                    />
                  </div>

                  <input
                    type="range"
                    min={min} max={max} step={step}
                    value={settings[key]}
                    onChange={(e) => upd(key, parseFloat(e.target.value))}
                    style={{
                      width: '100%',
                      accentColor: 'hsl(var(--accent))',
                      height: '6px',
                      cursor: 'pointer',
                      marginTop: '8px',
                      marginBottom: '8px',
                      outline: 'none',
                    }}
                  />
                  <p style={{ fontSize: '0.82rem', color: 'hsl(var(--pencil))', fontFamily: 'Inter, sans-serif', lineHeight: 1.6 }}>
                    {desc}
                  </p>
                </div>
              )
            })}
          </div>

          <button
            className={`btn ${settingsSaved ? 'btn-success' : 'btn-primary'}`}
            onClick={handleSaveSettings}
            disabled={savingSettings}
            style={{ marginTop: '1.5rem', padding: '.75rem 1.75rem', fontSize: '.95rem' }}
            id="save-settings-btn"
          >
            {savingSettings ? <Loader size={16} className="spin" /> : settingsSaved ? <CheckCircle size={16} /> : <Save size={16} />}
            {settingsSaved ? 'Settings Saved!' : savingSettings ? 'Saving…' : 'Save Settings'}
          </button>
        </section>
      )}

      {/* ─── Prompt Templates Section ─── */}
      <section className="animate-slide-up" style={{ marginTop: '2.5rem', marginBottom: '2.5rem', animationDelay: '0.25s', animationFillMode: 'both' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '12px', marginBottom: '1.5rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <div style={{
              width: '36px', height: '36px', borderRadius: '9px',
              background: 'hsl(260,70%,60% / .12)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              border: '1.5px solid hsl(260,70%,60% / .25)',
            }}>
              <Code size={18} style={{ color: 'hsl(260,70%,60%)' }} />
            </div>
            <div>
              <h2 style={{ fontSize: '1.15rem', fontWeight: 700, fontFamily: 'Inter, sans-serif', color: 'hsl(var(--ink))', margin: 0 }}>
                Prompt Templates
              </h2>
              <p style={{ fontSize: '.76rem', color: 'hsl(var(--pencil))', margin: 0, fontFamily: 'Inter, sans-serif' }}>
                Customize and manage AI prompt templates used throughout the application.
              </p>
            </div>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <button
              onClick={handleExportPrompts}
              className="btn btn-secondary"
              style={{ padding: '.5rem 1rem', fontSize: '.82rem', display: 'flex', alignItems: 'center', gap: '6px', height: '36px' }}
              title="Export all custom and default templates to a JSON file"
            >
              <Download size={14} /> Export
            </button>
            <label
              className="btn btn-secondary"
              style={{ padding: '.5rem 1rem', fontSize: '.82rem', display: 'flex', alignItems: 'center', gap: '6px', cursor: 'pointer', height: '36px', margin: 0 }}
              title="Import templates from a JSON export file"
            >
              <Upload size={14} /> {importing ? 'Importing…' : 'Import'}
              <input type="file" accept=".json" onChange={handleImportPrompts} style={{ display: 'none' }} disabled={importing} />
            </label>
            <button
              onClick={handleResetAllPrompts}
              className="btn btn-secondary"
              style={{ padding: '.5rem 1rem', fontSize: '.82rem', display: 'flex', alignItems: 'center', gap: '6px', color: 'hsl(var(--destructive))', borderColor: 'hsl(var(--destructive) / .3)', height: '36px' }}
              title="Reset all prompt templates to defaults"
            >
              <RotateCcw size={14} /> Reset All
            </button>
          </div>
        </div>

        {loadingPrompts ? (
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '4rem', flexDirection: 'column', gap: '1rem' }}>
            <Loader size={28} className="spin" style={{ color: 'hsl(var(--accent))' }} />
            <p style={{ fontFamily: 'Inter, sans-serif', fontSize: '.9rem', color: 'hsl(var(--pencil))' }}>Loading prompt templates…</p>
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            {/* Category tabs */}
            <div style={{ display: 'flex', gap: '8px', borderBottom: '1px solid hsl(var(--border) / .3)', paddingBottom: '.5rem', overflowX: 'auto' }}>
              {['MoM', 'Raw MoM', 'Summaries', 'Analysis', 'Speaker'].map(cat => {
                const isActive = activeCategory === cat
                const count = prompts.filter(p => p.category === cat).length
                return (
                  <button
                    key={cat}
                    onClick={() => setActiveCategory(cat)}
                    style={{
                      padding: '.5rem 1rem', borderRadius: '8px', border: 'none',
                      background: isActive ? 'hsl(var(--accent) / .12)' : 'transparent',
                      color: isActive ? 'hsl(var(--accent))' : 'hsl(var(--pencil))',
                      fontWeight: isActive ? 700 : 500, fontSize: '.85rem',
                      cursor: 'pointer', transition: 'all .15s', fontFamily: 'Inter',
                      display: 'flex', alignItems: 'center', gap: '6px', whiteSpace: 'nowrap'
                    }}
                  >
                    {cat}
                    <span style={{
                      fontSize: '.72rem', fontWeight: isActive ? 700 : 500,
                      background: isActive ? 'hsl(var(--accent) / .15)' : 'hsl(var(--muted))',
                      color: isActive ? 'hsl(var(--accent))' : 'hsl(var(--pencil))',
                      padding: '1px 6px', borderRadius: '999px',
                    }}>
                      {count}
                    </span>
                  </button>
                )
              })}
            </div>

            {/* Prompt Cards */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
              {prompts.filter(p => p.category === activeCategory).map(p => {
                const currentVal = editingTemplates[p.key] !== undefined ? editingTemplates[p.key] : p.template
                const hasChanges = currentVal !== p.template
                const isModified = p.is_modified
                const isSaving = savingKeys.has(p.key)
                const isSaved = savedKeys.has(p.key)

                return (
                  <div key={p.key} style={{
                    padding: '1.5rem', background: 'hsl(var(--card))',
                    border: '1.5px solid hsl(var(--border) / .4)', borderRadius: '12px',
                    display: 'flex', flexDirection: 'column', gap: '1rem',
                    transition: 'box-shadow .2s',
                  }}
                  className="sketch-border"
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '8px' }}>
                      <div>
                        <h3 style={{ margin: 0, fontSize: '.98rem', fontWeight: 700, color: 'hsl(var(--ink))', fontFamily: 'Inter' }}>
                          {p.name}
                        </h3>
                        <p style={{ margin: '4px 0 0', fontSize: '.8rem', color: 'hsl(var(--pencil))', lineHeight: 1.5, fontFamily: 'Inter, sans-serif' }}>
                          {p.description}
                        </p>
                      </div>
                      <div style={{ display: 'flex', gap: '6px', alignItems: 'center' }}>
                        {isModified && (
                          <span style={{
                            fontSize: '.7rem', fontWeight: 600, color: 'hsl(220,80%,60%)',
                            background: 'hsl(220,80%,60% / .1)', border: '1px solid hsl(220,80%,60% / .25)',
                            padding: '1px 6px', borderRadius: '4px', fontFamily: 'Inter, sans-serif'
                          }}>
                            Customized
                          </span>
                        )}
                        {hasChanges && (
                          <span style={{
                            fontSize: '.7rem', fontWeight: 600, color: 'hsl(35,90%,50%)',
                            background: 'hsl(35,90%,50% / .1)', border: '1px solid hsl(35,90%,50% / .25)',
                            padding: '1px 6px', borderRadius: '4px', fontFamily: 'Inter, sans-serif'
                          }}>
                            Unsaved Changes
                          </span>
                        )}
                      </div>
                    </div>

                    {/* Fills variables badges */}
                    {p.variables?.length > 0 && (
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', alignItems: 'center' }}>
                        <span style={{ fontSize: '.7rem', fontWeight: 700, color: 'hsl(var(--pencil))', textTransform: 'uppercase', letterSpacing: '.03em', fontFamily: 'Inter, sans-serif' }}>Required variables:</span>
                        {p.variables.map(v => (
                          <code key={v} style={{
                            fontFamily: 'JetBrains Mono, monospace', fontSize: '.72rem',
                            color: 'hsl(var(--accent))', background: 'hsl(var(--accent) / .08)',
                            padding: '2px 6px', borderRadius: '4px', border: '1px solid hsl(var(--accent) / .15)'
                          }}>
                            {v}
                          </code>
                        ))}
                      </div>
                    )}

                    {/* Textarea */}
                    <div style={{ position: 'relative' }}>
                      <textarea
                        value={currentVal}
                        onChange={e => setEditingTemplates(prev => ({ ...prev, [p.key]: e.target.value }))}
                        rows={10}
                        style={{
                          width: '100%', padding: '.75rem', borderRadius: '8px',
                          background: 'hsl(var(--muted) / .3)', border: '1.5px solid hsl(var(--border) / .6)',
                          color: 'hsl(var(--ink))', fontFamily: 'JetBrains Mono, monospace',
                          fontSize: '.8rem', lineHeight: 1.6, resize: 'vertical',
                          outline: 'none', boxSizing: 'border-box', transition: 'border-color .15s',
                        }}
                        onFocus={e => (e.currentTarget.style.borderColor = 'hsl(var(--accent) / .5)')}
                        onBlur={e => (e.currentTarget.style.borderColor = 'hsl(var(--border) / .6)')}
                      />
                    </div>

                    {/* Card Actions */}
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <div style={{ fontSize: '.75rem', color: 'hsl(var(--pencil))', fontFamily: 'Inter, sans-serif' }}>
                        {currentVal.length} characters
                      </div>
                      <div style={{ display: 'flex', gap: '8px' }}>
                        {isModified && (
                          <button
                            onClick={() => {
                              handleResetPromptTemplate(p.key)
                              setEditingTemplates(prev => {
                                const next = { ...prev }
                                delete next[p.key]
                                return next
                              })
                            }}
                            className="btn btn-secondary"
                            style={{ padding: '.4rem .85rem', fontSize: '.78rem', display: 'flex', alignItems: 'center', gap: '6px' }}
                          >
                            <RotateCcw size={12} /> Reset to Default
                          </button>
                        )}
                        {hasChanges && (
                          <button
                            onClick={() => {
                              setEditingTemplates(prev => {
                                const next = { ...prev }
                                delete next[p.key]
                                return next
                              })
                            }}
                            className="btn btn-secondary"
                            style={{ padding: '.4rem .85rem', fontSize: '.78rem' }}
                          >
                            Discard
                          </button>
                        )}
                        <button
                          onClick={() => {
                            handleSavePromptTemplate(p.key, currentVal)
                            setEditingTemplates(prev => {
                              const next = { ...prev }
                              delete next[p.key]
                              return next
                            })
                          }}
                          disabled={!hasChanges || isSaving}
                          className={`btn ${isSaved ? 'btn-success' : 'btn-primary'}`}
                          style={{ padding: '.4rem 1.25rem', fontSize: '.78rem', display: 'flex', alignItems: 'center', gap: '6px' }}
                        >
                          {isSaving ? <Loader size={12} className="spin" /> : isSaved ? <CheckCircle size={12} /> : <Save size={12} />}
                          {isSaved ? 'Saved!' : isSaving ? 'Saving…' : 'Save Changes'}
                        </button>
                      </div>
                    </div>

                  </div>
                )
              })}
            </div>
          </div>
        )}
      </section>
      </div>
    </div>
  )
}
