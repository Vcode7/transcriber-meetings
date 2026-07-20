import { useEffect, useState, useRef } from 'react'
import {
  Settings, Mic, Trash2, Pencil, Save, Loader, Sliders, Sparkles, User, CheckCircle,
  MessageSquare, RotateCcw, Upload, Download, FileText, Code, Cpu, Database
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
  use_ollama?: boolean
  ollama_server_url?: string
  ollama_port?: number
  ollama_model_priority?: string
  rag_chunk_size?: number
  rag_chunk_overlap?: number
  rag_retrieval_k_global?: number
  rag_retrieval_k_meeting?: number
  rag_retrieval_k_transcript?: number
  rag_relative_score_cutoff?: number
  generate_mom_auto?: boolean

  // New Ollama settings
  ollama_num_ctx?: number
  ollama_temperature?: number
  ollama_top_p?: number
  ollama_top_k?: number
  ollama_repeat_penalty?: number
  ollama_seed?: number
  ollama_stop?: string
  ollama_keep_alive?: string
  ollama_num_thread?: number
  ollama_num_gpu?: number

  // Task max tokens
  max_tokens_mom?: number
  max_tokens_mom_merge?: number
  max_tokens_raw_mom_to_mom?: number
  max_tokens_raw_mom_extraction?: number
  max_tokens_raw_mom_repair?: number
  max_tokens_agenda_compress?: number
  max_tokens_reference_compress?: number
  max_tokens_agenda_from_summary?: number
  max_tokens_executive_summary?: number
  max_tokens_short_summary?: number
  max_tokens_detailed_summary?: number
  max_tokens_chunk_summary?: number
  max_tokens_key_points?: number
  max_tokens_action_items?: number
  max_tokens_key_decisions?: number
  max_tokens_speaker_summary?: number
  max_tokens_speaker_key_points?: number
  max_tokens_speaker_action_items?: number
  max_tokens_collection_chat?: number
  max_tokens_collection_compare?: number
  max_tokens_collection_topic_growth?: number
  max_tokens_vocab_extractor?: number
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

  // Ollama test state
  const [testingOllama, setTestingOllama] = useState(false)
  const [ollamaTestResult, setOllamaTestResult] = useState<{
    success: boolean
    message: string
    available_models?: string[]
    running_models?: string[]
    error?: string
  } | null>(null)

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

  const handleTestOllamaConnection = async () => {
    setTestingOllama(true)
    setOllamaTestResult(null)
    try {
      const res = await api.post('/settings/test-ollama', {
        server_url: settings?.ollama_server_url || 'http://localhost:11434'
      })
      setOllamaTestResult(res.data)
      if (res.data.success) {
        toast.success(res.data.message || 'Connected to Ollama server!')
      } else {
        toast.error(res.data.error || 'Failed to connect to Ollama server.')
      }
    } catch (err: any) {
      const errorMsg = err.response?.data?.error || err.message || 'Connection test failed.'
      setOllamaTestResult({
        success: false,
        message: 'Connection failed.',
        error: errorMsg,
      })
      toast.error(errorMsg)
    } finally {
      setTestingOllama(false)
    }
  }

  const upd = (key: keyof UserSettings, val: any) =>
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

          {/* ─── Minutes of Meeting (MoM) Automation ─── */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginTop: '2.5rem', marginBottom: '1.5rem' }}>
            <div style={{
              width: '36px', height: '36px', borderRadius: '9px',
              background: 'hsl(var(--accent) / .15)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              border: '1.5px solid hsl(var(--accent) / .3)',
            }}>
              <Sparkles size={18} style={{ color: 'hsl(var(--accent))' }} />
            </div>
            <h2 style={{ fontSize: '1.15rem', fontWeight: 700, fontFamily: 'Inter, sans-serif', color: 'hsl(var(--ink))' }}>
              Minutes of Meeting (MoM) Automation
            </h2>
          </div>

          <div style={{
            padding: '1.75rem',
            background: 'hsl(var(--card))',
            border: '1.5px solid hsl(var(--ink) / .1)',
            borderRadius: '12px',
            display: 'flex', flexDirection: 'column', gap: '2rem'
          }}>
            {/* ── Generate MoM Automatically toggle ── */}
            <div style={{
              display: 'flex', alignItems: 'center', justifyContent: 'space-between',
              padding: '1rem 1.25rem',
              background: (settings.generate_mom_auto ?? true) ? 'hsl(var(--accent) / .06)' : 'hsl(var(--muted) / .3)',
              borderRadius: '10px',
              border: `1.5px solid ${(settings.generate_mom_auto ?? true) ? 'hsl(var(--accent) / .35)' : 'hsl(var(--ink) / .08)'}`,
              transition: 'background .2s, border-color .2s',
            }}>
              <div style={{ flex: 1 }}>
                <div style={{ fontWeight: 700, fontSize: '.95rem', fontFamily: 'Inter, sans-serif', color: 'hsl(var(--ink))', marginBottom: '3px' }}>
                  Generate MoM Automatically
                </div>
                <div style={{ fontSize: '.8rem', color: 'hsl(var(--pencil))', fontFamily: 'Inter, sans-serif', lineHeight: 1.5 }}>
                  {(settings.generate_mom_auto ?? true)
                    ? 'ON — After speaker identification completes, automatically generate the MoM.'
                    : 'OFF — Skip automatic MoM generation. The pipeline will finish after speaker identification without starting MoM generation.'}
                </div>
              </div>
              <button
                id="generate-mom-auto-toggle"
                onClick={() => upd('generate_mom_auto', !(settings.generate_mom_auto ?? true))}
                style={{
                  flexShrink: 0,
                  width: '52px', height: '28px',
                  borderRadius: '999px',
                  border: 'none',
                  background: (settings.generate_mom_auto ?? true) ? 'hsl(var(--accent))' : 'hsl(var(--ink) / .18)',
                  cursor: 'pointer',
                  position: 'relative',
                  transition: 'background .25s',
                  outline: 'none',
                  boxShadow: (settings.generate_mom_auto ?? true) ? '0 0 0 3px hsl(var(--accent) / .2)' : 'none',
                }}
                title={(settings.generate_mom_auto ?? true) ? 'Click to disable automatic MoM generation' : 'Click to enable automatic MoM generation'}
              >
                <span style={{
                  position: 'absolute',
                  top: '3px',
                  left: (settings.generate_mom_auto ?? true) ? '26px' : '3px',
                  width: '22px', height: '22px',
                  borderRadius: '50%',
                  background: 'white',
                  transition: 'left .25s cubic-bezier(.4,0,.2,1)',
                  boxShadow: '0 1px 4px rgba(0,0,0,.25)',
                  display: 'block',
                }} />
              </button>
            </div>
          </div>

          {/* ─── Ollama Settings ─── */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginTop: '2.5rem', marginBottom: '1.5rem' }}>
            <div style={{
              width: '36px', height: '36px', borderRadius: '9px',
              background: 'hsl(180,90%,50% / .15)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              border: '1.5px solid hsl(180,90%,50% / .3)',
            }}>
              <Cpu size={18} style={{ color: 'hsl(180,90%,50%)' }} />
            </div>
            <h2 style={{ fontSize: '1.15rem', fontWeight: 700, fontFamily: 'Inter, sans-serif', color: 'hsl(var(--ink))' }}>
              Ollama Offline Fallback Settings
            </h2>
          </div>

          <div style={{
            padding: '1.75rem',
            background: 'hsl(var(--card))',
            border: '1.5px solid hsl(var(--ink) / .1)',
            borderRadius: '12px',
            display: 'flex', flexDirection: 'column', gap: '2rem'
          }}>

            {/* ── Use Ollama toggle ── */}
            <div style={{
              display: 'flex', alignItems: 'center', justifyContent: 'space-between',
              padding: '1rem 1.25rem',
              background: settings.use_ollama ? 'hsl(180,90%,50% / .06)' : 'hsl(var(--muted) / .3)',
              borderRadius: '10px',
              border: `1.5px solid ${settings.use_ollama ? 'hsl(180,90%,50% / .35)' : 'hsl(var(--ink) / .08)'}`,
              transition: 'background .2s, border-color .2s',
            }}>
              <div style={{ flex: 1 }}>
                <div style={{ fontWeight: 700, fontSize: '.95rem', fontFamily: 'Inter, sans-serif', color: 'hsl(var(--ink))', marginBottom: '3px' }}>
                  Use Ollama
                </div>
                <div style={{ fontSize: '.8rem', color: 'hsl(var(--pencil))', fontFamily: 'Inter, sans-serif', lineHeight: 1.5 }}>
                  {settings.use_ollama
                    ? 'Ollama will be tried first. Falls back to the bundled Qwen model if unavailable.'
                    : 'Disabled — the bundled Qwen model is used directly. Ollama is not contacted.'}
                </div>
              </div>
              {/* Toggle switch */}
              <button
                id="use-ollama-toggle"
                onClick={() => upd('use_ollama', !settings.use_ollama)}
                style={{
                  flexShrink: 0,
                  width: '52px', height: '28px',
                  borderRadius: '999px',
                  border: 'none',
                  background: settings.use_ollama ? 'hsl(180,90%,45%)' : 'hsl(var(--ink) / .18)',
                  cursor: 'pointer',
                  position: 'relative',
                  transition: 'background .25s',
                  outline: 'none',
                  boxShadow: settings.use_ollama ? '0 0 0 3px hsl(180,90%,50% / .2)' : 'none',
                }}
                title={settings.use_ollama ? 'Click to disable Ollama' : 'Click to enable Ollama'}
              >
                <span style={{
                  position: 'absolute',
                  top: '3px',
                  left: settings.use_ollama ? '26px' : '3px',
                  width: '22px', height: '22px',
                  borderRadius: '50%',
                  background: 'white',
                  transition: 'left .25s cubic-bezier(.4,0,.2,1)',
                  boxShadow: '0 1px 4px rgba(0,0,0,.25)',
                  display: 'block',
                }} />
              </button>
            </div>

            {/* Server URL, Port, and Priority — dimmed when Ollama disabled */}
            <div style={{ opacity: settings.use_ollama ? 1 : 0.45, transition: 'opacity .2s', pointerEvents: settings.use_ollama ? 'auto' : 'none' }}>
              
              {/* ── Ollama Server URL Field with Test Connection Button ── */}
              <div style={{ marginBottom: '1.75rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px', alignItems: 'center' }}>
                  <label style={{ fontSize: '.95rem', fontWeight: 600, fontFamily: 'Inter, sans-serif', color: 'hsl(var(--ink))' }}>
                    Ollama Server URL
                  </label>
                  <button
                    type="button"
                    onClick={handleTestOllamaConnection}
                    disabled={testingOllama}
                    style={{
                      display: 'inline-flex', alignItems: 'center', gap: '6px',
                      fontSize: '.85rem', fontWeight: 600, fontFamily: 'Inter, sans-serif',
                      padding: '.35rem .85rem',
                      borderRadius: '7px',
                      background: 'hsl(180,90%,50% / .15)',
                      color: 'hsl(180,90%,35%)',
                      border: '1.5px solid hsl(180,90%,50% / .35)',
                      cursor: testingOllama ? 'not-allowed' : 'pointer',
                      transition: 'all .2s',
                    }}
                  >
                    {testingOllama ? <Loader size={14} className="animate-spin" /> : <Sparkles size={14} />}
                    {testingOllama ? 'Testing...' : 'Test Connection'}
                  </button>
                </div>
                <input
                  type="text"
                  value={settings.ollama_server_url !== undefined ? settings.ollama_server_url : 'http://localhost:11434'}
                  onChange={(e) => {
                    upd('ollama_server_url', e.target.value)
                    setOllamaTestResult(null)
                  }}
                  placeholder="http://localhost:11434"
                  style={{
                    width: '100%',
                    fontFamily: 'JetBrains Mono, monospace', fontSize: '.9rem',
                    color: 'hsl(var(--ink))', fontWeight: 500,
                    padding: '.55rem .85rem',
                    background: 'hsl(var(--background))',
                    borderRadius: '8px',
                    border: '1.5px solid hsl(var(--ink) / .15)',
                    outline: 'none',
                    boxSizing: 'border-box',
                  }}
                />
                <p style={{ fontSize: '0.82rem', color: 'hsl(var(--pencil))', fontFamily: 'Inter, sans-serif', lineHeight: 1.6, marginTop: '6px' }}>
                  Configurable HTTP/HTTPS endpoint for your Ollama instance (default: <code>http://localhost:11434</code>). Accepts <code>http://192.168.x.x:11434</code> or custom domain.
                </p>

                {/* Connection test result banner */}
                {ollamaTestResult && (
                  <div style={{
                    marginTop: '12px',
                    padding: '.85rem 1rem',
                    borderRadius: '8px',
                    fontSize: '.85rem',
                    fontFamily: 'Inter, sans-serif',
                    background: ollamaTestResult.success ? 'hsl(142, 70%, 45% / .12)' : 'hsl(0, 70%, 50% / .12)',
                    border: `1.5px solid ${ollamaTestResult.success ? 'hsl(142, 70%, 45% / .35)' : 'hsl(0, 70%, 50% / .35)'}`,
                    color: ollamaTestResult.success ? 'hsl(142, 70%, 30%)' : 'hsl(0, 70%, 35%)',
                  }}>
                    <div style={{ fontWeight: 700, marginBottom: '4px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                      {ollamaTestResult.success ? <CheckCircle size={15} /> : <span style={{ fontWeight: 900 }}>✕</span>}
                      {ollamaTestResult.success ? 'Connection Successful' : 'Connection Failed'}
                    </div>
                    <div>{ollamaTestResult.message || ollamaTestResult.error}</div>
                    {ollamaTestResult.error && (
                      <div style={{ marginTop: '4px', fontSize: '.8rem', opacity: 0.9, fontFamily: 'JetBrains Mono, monospace' }}>
                        {ollamaTestResult.error}
                      </div>
                    )}
                    {ollamaTestResult.available_models && ollamaTestResult.available_models.length > 0 && (
                      <div style={{ marginTop: '8px', paddingTop: '6px', borderTop: '1px solid hsl(142, 70%, 45% / .2)', fontSize: '.8rem' }}>
                        <strong>Available Models ({ollamaTestResult.available_models.length}):</strong>{' '}
                        {ollamaTestResult.available_models.join(', ')}
                      </div>
                    )}
                  </div>
                )}
              </div>

              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '10px', alignItems: 'center' }}>
                  <label style={{ fontSize: '.95rem', fontWeight: 600, fontFamily: 'Inter, sans-serif', color: 'hsl(var(--ink))' }}>
                    Ollama Port
                  </label>
                  <input
                    type="number"
                    min={1} max={65535}
                    value={settings.ollama_port !== undefined ? settings.ollama_port : 11434}
                    onChange={(e) => {
                      const parsed = parseInt(e.target.value, 10)
                      upd('ollama_port', isNaN(parsed) ? 11434 : parsed)
                    }}
                    style={{
                      fontFamily: 'JetBrains Mono, monospace', fontSize: '.9rem',
                      color: 'hsl(var(--accent))', fontWeight: 700,
                      padding: '.3rem .65rem',
                      background: 'hsl(var(--accent) / .1)',
                      borderRadius: '6px',
                      border: '1.5px solid hsl(var(--accent) / .2)',
                      width: '100px', textAlign: 'center',
                      outline: 'none',
                    }}
                  />
                </div>
                <p style={{ fontSize: '0.82rem', color: 'hsl(var(--pencil))', fontFamily: 'Inter, sans-serif', lineHeight: 1.6 }}>
                  Port of the locally running Ollama server (default: 11434).
                </p>
              </div>

                <div style={{ marginTop: '1.5rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '10px', alignItems: 'center' }}>
                    <label style={{ fontSize: '.95rem', fontWeight: 600, fontFamily: 'Inter, sans-serif', color: 'hsl(var(--ink))' }}>
                      Model Priority List
                    </label>
                    <input
                      type="text"
                      value={settings.ollama_model_priority !== undefined ? settings.ollama_model_priority : 'gemma,qwen,llama,deepseek,mistral'}
                      onChange={(e) => {
                        upd('ollama_model_priority', e.target.value)
                      }}
                      style={{
                        fontFamily: 'JetBrains Mono, monospace', fontSize: '.9rem',
                        color: 'hsl(var(--accent))', fontWeight: 600,
                        padding: '.3rem .65rem',
                        background: 'hsl(var(--accent) / .1)',
                        borderRadius: '6px',
                        border: '1.5px solid hsl(var(--accent) / .2)',
                        width: '280px',
                        outline: 'none',
                      }}
                    />
                  </div>
                  <p style={{ fontSize: '0.82rem', color: 'hsl(var(--pencil))', fontFamily: 'Inter, sans-serif', lineHeight: 1.6 }}>
                    Comma-separated list of model keywords in order of preference (e.g., gemma, qwen, llama, deepseek, mistral).
                  </p>
                </div>

                {/* ── Ollama Advanced Parameters Subgrid ── */}
                <div style={{ marginTop: '2rem', paddingTop: '1.5rem', borderTop: '1px solid hsl(var(--border) / .3)' }}>
                  <h3 style={{ fontSize: '0.92rem', fontWeight: 700, fontFamily: 'Inter, sans-serif', color: 'hsl(var(--ink))', marginBottom: '1.25rem' }}>
                    Ollama Advanced Generation Parameters
                  </h3>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1.5rem' }}>
                    
                    {/* Context Size */}
                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px', alignItems: 'center' }}>
                        <label style={{ fontSize: '.88rem', fontWeight: 600, fontFamily: 'Inter, sans-serif', color: 'hsl(var(--ink))' }}>
                          Context Size (num_ctx)
                        </label>
                        <input
                          type="number"
                          min={512} max={131072} step={512}
                          value={settings.ollama_num_ctx ?? 32768}
                          onChange={(e) => {
                            const parsed = parseInt(e.target.value, 10)
                            upd('ollama_num_ctx', isNaN(parsed) ? 32768 : parsed)
                          }}
                          style={{
                            fontFamily: 'JetBrains Mono, monospace', fontSize: '.85rem',
                            color: 'hsl(var(--accent))', fontWeight: 700,
                            padding: '.25rem .5rem',
                            background: 'hsl(var(--accent) / .1)',
                            borderRadius: '6px',
                            border: '1.5px solid hsl(var(--accent) / .2)',
                            width: '90px', textAlign: 'center',
                            outline: 'none',
                          }}
                        />
                      </div>
                      <p style={{ fontSize: '0.78rem', color: 'hsl(var(--pencil))', margin: 0, lineHeight: 1.4 }}>
                        Context window size (default: 32768). Keep large.
                      </p>
                    </div>

                    {/* Temperature */}
                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px', alignItems: 'center' }}>
                        <label style={{ fontSize: '.88rem', fontWeight: 600, fontFamily: 'Inter, sans-serif', color: 'hsl(var(--ink))' }}>
                          Temperature
                        </label>
                        <input
                          type="number"
                          min={0.0} max={2.0} step={0.1}
                          value={settings.ollama_temperature ?? 0.0}
                          onChange={(e) => {
                            const parsed = parseFloat(e.target.value)
                            upd('ollama_temperature', isNaN(parsed) ? 0.0 : parsed)
                          }}
                          style={{
                            fontFamily: 'JetBrains Mono, monospace', fontSize: '.85rem',
                            color: 'hsl(var(--accent))', fontWeight: 700,
                            padding: '.25rem .5rem',
                            background: 'hsl(var(--accent) / .1)',
                            borderRadius: '6px',
                            border: '1.5px solid hsl(var(--accent) / .2)',
                            width: '90px', textAlign: 'center',
                            outline: 'none',
                          }}
                        />
                      </div>
                      <p style={{ fontSize: '0.78rem', color: 'hsl(var(--pencil))', margin: 0, lineHeight: 1.4 }}>
                        Creative randomness: 0.0 is deterministic.
                      </p>
                    </div>

                    {/* Top P */}
                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px', alignItems: 'center' }}>
                        <label style={{ fontSize: '.88rem', fontWeight: 600, fontFamily: 'Inter, sans-serif', color: 'hsl(var(--ink))' }}>
                          Top P
                        </label>
                        <input
                          type="number"
                          min={0.0} max={1.0} step={0.05}
                          value={settings.ollama_top_p ?? 0.9}
                          onChange={(e) => {
                            const parsed = parseFloat(e.target.value)
                            upd('ollama_top_p', isNaN(parsed) ? 0.9 : parsed)
                          }}
                          style={{
                            fontFamily: 'JetBrains Mono, monospace', fontSize: '.85rem',
                            color: 'hsl(var(--accent))', fontWeight: 700,
                            padding: '.25rem .5rem',
                            background: 'hsl(var(--accent) / .1)',
                            borderRadius: '6px',
                            border: '1.5px solid hsl(var(--accent) / .2)',
                            width: '90px', textAlign: 'center',
                            outline: 'none',
                          }}
                        />
                      </div>
                      <p style={{ fontSize: '0.78rem', color: 'hsl(var(--pencil))', margin: 0, lineHeight: 1.4 }}>
                        Nucleus sampling threshold (default: 0.9).
                      </p>
                    </div>

                    {/* Top K */}
                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px', alignItems: 'center' }}>
                        <label style={{ fontSize: '.88rem', fontWeight: 600, fontFamily: 'Inter, sans-serif', color: 'hsl(var(--ink))' }}>
                          Top K
                        </label>
                        <input
                          type="number"
                          min={0} max={500} step={5}
                          value={settings.ollama_top_k ?? 40}
                          onChange={(e) => {
                            const parsed = parseInt(e.target.value, 10)
                            upd('ollama_top_k', isNaN(parsed) ? 40 : parsed)
                          }}
                          style={{
                            fontFamily: 'JetBrains Mono, monospace', fontSize: '.85rem',
                            color: 'hsl(var(--accent))', fontWeight: 700,
                            padding: '.25rem .5rem',
                            background: 'hsl(var(--accent) / .1)',
                            borderRadius: '6px',
                            border: '1.5px solid hsl(var(--accent) / .2)',
                            width: '90px', textAlign: 'center',
                            outline: 'none',
                          }}
                        />
                      </div>
                      <p style={{ fontSize: '0.78rem', color: 'hsl(var(--pencil))', margin: 0, lineHeight: 1.4 }}>
                        Top-K sampling limit (default: 40).
                      </p>
                    </div>

                    {/* Repeat Penalty */}
                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px', alignItems: 'center' }}>
                        <label style={{ fontSize: '.88rem', fontWeight: 600, fontFamily: 'Inter, sans-serif', color: 'hsl(var(--ink))' }}>
                          Repeat Penalty
                        </label>
                        <input
                          type="number"
                          min={0.0} max={3.0} step={0.05}
                          value={settings.ollama_repeat_penalty ?? 1.15}
                          onChange={(e) => {
                            const parsed = parseFloat(e.target.value)
                            upd('ollama_repeat_penalty', isNaN(parsed) ? 1.15 : parsed)
                          }}
                          style={{
                            fontFamily: 'JetBrains Mono, monospace', fontSize: '.85rem',
                            color: 'hsl(var(--accent))', fontWeight: 700,
                            padding: '.25rem .5rem',
                            background: 'hsl(var(--accent) / .1)',
                            borderRadius: '6px',
                            border: '1.5px solid hsl(var(--accent) / .2)',
                            width: '90px', textAlign: 'center',
                            outline: 'none',
                          }}
                        />
                      </div>
                      <p style={{ fontSize: '0.78rem', color: 'hsl(var(--pencil))', margin: 0, lineHeight: 1.4 }}>
                        Repetition penalty weight (default: 1.15).
                      </p>
                    </div>

                    {/* Seed */}
                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px', alignItems: 'center' }}>
                        <label style={{ fontSize: '.88rem', fontWeight: 600, fontFamily: 'Inter, sans-serif', color: 'hsl(var(--ink))' }}>
                          Random Seed
                        </label>
                        <input
                          type="number"
                          min={-1}
                          value={settings.ollama_seed ?? -1}
                          onChange={(e) => {
                            const parsed = parseInt(e.target.value, 10)
                            upd('ollama_seed', isNaN(parsed) ? -1 : parsed)
                          }}
                          style={{
                            fontFamily: 'JetBrains Mono, monospace', fontSize: '.85rem',
                            color: 'hsl(var(--accent))', fontWeight: 700,
                            padding: '.25rem .5rem',
                            background: 'hsl(var(--accent) / .1)',
                            borderRadius: '6px',
                            border: '1.5px solid hsl(var(--accent) / .2)',
                            width: '90px', textAlign: 'center',
                            outline: 'none',
                          }}
                        />
                      </div>
                      <p style={{ fontSize: '0.78rem', color: 'hsl(var(--pencil))', margin: 0, lineHeight: 1.4 }}>
                        Determinism seed (-1 for random).
                      </p>
                    </div>

                    {/* Threads */}
                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px', alignItems: 'center' }}>
                        <label style={{ fontSize: '.88rem', fontWeight: 600, fontFamily: 'Inter, sans-serif', color: 'hsl(var(--ink))' }}>
                          CPU Threads
                        </label>
                        <input
                          type="number"
                          min={0}
                          value={settings.ollama_num_thread ?? 0}
                          onChange={(e) => {
                            const parsed = parseInt(e.target.value, 10)
                            upd('ollama_num_thread', isNaN(parsed) ? 0 : parsed)
                          }}
                          style={{
                            fontFamily: 'JetBrains Mono, monospace', fontSize: '.85rem',
                            color: 'hsl(var(--accent))', fontWeight: 700,
                            padding: '.25rem .5rem',
                            background: 'hsl(var(--accent) / .1)',
                            borderRadius: '6px',
                            border: '1.5px solid hsl(var(--accent) / .2)',
                            width: '90px', textAlign: 'center',
                            outline: 'none',
                          }}
                        />
                      </div>
                      <p style={{ fontSize: '0.78rem', color: 'hsl(var(--pencil))', margin: 0, lineHeight: 1.4 }}>
                        Number of CPU threads (default: 0 = auto).
                      </p>
                    </div>

                    {/* GPU Layers */}
                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px', alignItems: 'center' }}>
                        <label style={{ fontSize: '.88rem', fontWeight: 600, fontFamily: 'Inter, sans-serif', color: 'hsl(var(--ink))' }}>
                          GPU Layers
                        </label>
                        <input
                          type="number"
                          min={-1}
                          value={settings.ollama_num_gpu ?? -1}
                          onChange={(e) => {
                            const parsed = parseInt(e.target.value, 10)
                            upd('ollama_num_gpu', isNaN(parsed) ? -1 : parsed)
                          }}
                          style={{
                            fontFamily: 'JetBrains Mono, monospace', fontSize: '.85rem',
                            color: 'hsl(var(--accent))', fontWeight: 700,
                            padding: '.25rem .5rem',
                            background: 'hsl(var(--accent) / .1)',
                            borderRadius: '6px',
                            border: '1.5px solid hsl(var(--accent) / .2)',
                            width: '90px', textAlign: 'center',
                            outline: 'none',
                          }}
                        />
                      </div>
                      <p style={{ fontSize: '0.78rem', color: 'hsl(var(--pencil))', margin: 0, lineHeight: 1.4 }}>
                        Layers offloaded to GPU (-1 = auto).
                      </p>
                    </div>

                    {/* Keep Alive */}
                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px', alignItems: 'center' }}>
                        <label style={{ fontSize: '.88rem', fontWeight: 600, fontFamily: 'Inter, sans-serif', color: 'hsl(var(--ink))' }}>
                          Keep Alive
                        </label>
                        <input
                          type="text"
                          value={settings.ollama_keep_alive ?? '5m'}
                          onChange={(e) => upd('ollama_keep_alive', e.target.value)}
                          style={{
                            fontFamily: 'JetBrains Mono, monospace', fontSize: '.85rem',
                            color: 'hsl(var(--accent))', fontWeight: 700,
                            padding: '.25rem .5rem',
                            background: 'hsl(var(--accent) / .1)',
                            borderRadius: '6px',
                            border: '1.5px solid hsl(var(--accent) / .2)',
                            width: '90px', textAlign: 'center',
                            outline: 'none',
                          }}
                        />
                      </div>
                      <p style={{ fontSize: '0.78rem', color: 'hsl(var(--pencil))', margin: 0, lineHeight: 1.4 }}>
                        Time model stays loaded (e.g. 5m, 1h, 0).
                      </p>
                    </div>

                    {/* Stop Sequences */}
                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px', alignItems: 'center' }}>
                        <label style={{ fontSize: '.88rem', fontWeight: 600, fontFamily: 'Inter, sans-serif', color: 'hsl(var(--ink))' }}>
                          Stop Sequences
                        </label>
                        <input
                          type="text"
                          value={settings.ollama_stop ?? ''}
                          onChange={(e) => upd('ollama_stop', e.target.value)}
                          placeholder="e.g. \n,User:"
                          style={{
                            fontFamily: 'JetBrains Mono, monospace', fontSize: '.85rem',
                            color: 'hsl(var(--accent))', fontWeight: 700,
                            padding: '.25rem .5rem',
                            background: 'hsl(var(--accent) / .1)',
                            borderRadius: '6px',
                            border: '1.5px solid hsl(var(--accent) / .2)',
                            width: '90px', textAlign: 'center',
                            outline: 'none',
                          }}
                        />
                      </div>
                      <p style={{ fontSize: '0.78rem', color: 'hsl(var(--pencil))', margin: 0, lineHeight: 1.4 }}>
                        Comma-separated sequence list.
                      </p>
                    </div>

                  </div>
                </div>
              </div>
            </div>

          {/* ─── RAG Configuration Settings ─── */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginTop: '2.5rem', marginBottom: '1.5rem' }}>
            <div style={{
              width: '36px', height: '36px', borderRadius: '9px',
              background: 'hsl(210,90%,50% / .15)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              border: '1.5px solid hsl(210,90%,50% / .3)',
            }}>
              <Database size={18} style={{ color: 'hsl(210,90%,50%)' }} />
            </div>
            <h2 style={{ fontSize: '1.15rem', fontWeight: 700, fontFamily: 'Inter, sans-serif', color: 'hsl(var(--ink))' }}>
              RAG & Retrieval Settings
            </h2>
          </div>

          <div style={{
            padding: '1.75rem',
            background: 'hsl(var(--card))',
            border: '1.5px solid hsl(var(--ink) / .1)',
            borderRadius: '12px',
            display: 'flex', flexDirection: 'column', gap: '2rem'
          }}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '1.5rem' }}>
              {/* Chunk Size */}
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px', alignItems: 'center' }}>
                  <label style={{ fontSize: '.9rem', fontWeight: 600, fontFamily: 'Inter, sans-serif', color: 'hsl(var(--ink))' }}>
                    Chunk Size (words)
                  </label>
                  <input
                    type="number"
                    min={50} max={2000} step={50}
                    value={settings.rag_chunk_size !== undefined ? settings.rag_chunk_size : 400}
                    onChange={(e) => {
                      const parsed = parseInt(e.target.value, 10)
                      upd('rag_chunk_size', isNaN(parsed) ? 400 : parsed)
                    }}
                    style={{
                      fontFamily: 'JetBrains Mono, monospace', fontSize: '.85rem',
                      color: 'hsl(var(--accent))', fontWeight: 700,
                      padding: '.25rem .5rem',
                      background: 'hsl(var(--accent) / .1)',
                      borderRadius: '6px',
                      border: '1.5px solid hsl(var(--accent) / .2)',
                      width: '80px', textAlign: 'center',
                      outline: 'none',
                    }}
                  />
                </div>
                <p style={{ fontSize: '0.78rem', color: 'hsl(var(--pencil))', margin: 0, lineHeight: 1.4 }}>
                  Target number of words per document chunk (default: 400).
                </p>
              </div>

              {/* Chunk Overlap */}
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px', alignItems: 'center' }}>
                  <label style={{ fontSize: '.9rem', fontWeight: 600, fontFamily: 'Inter, sans-serif', color: 'hsl(var(--ink))' }}>
                    Chunk Overlap (words)
                  </label>
                  <input
                    type="number"
                    min={0} max={500} step={10}
                    value={settings.rag_chunk_overlap !== undefined ? settings.rag_chunk_overlap : 50}
                    onChange={(e) => {
                      const parsed = parseInt(e.target.value, 10)
                      upd('rag_chunk_overlap', isNaN(parsed) ? 50 : parsed)
                    }}
                    style={{
                      fontFamily: 'JetBrains Mono, monospace', fontSize: '.85rem',
                      color: 'hsl(var(--accent))', fontWeight: 700,
                      padding: '.25rem .5rem',
                      background: 'hsl(var(--accent) / .1)',
                      borderRadius: '6px',
                      border: '1.5px solid hsl(var(--accent) / .2)',
                      width: '80px', textAlign: 'center',
                      outline: 'none',
                    }}
                  />
                </div>
                <p style={{ fontSize: '0.78rem', color: 'hsl(var(--pencil))', margin: 0, lineHeight: 1.4 }}>
                  Words of overlap between adjacent chunks (default: 50).
                </p>
              </div>

              {/* K Global */}
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px', alignItems: 'center' }}>
                  <label style={{ fontSize: '.9rem', fontWeight: 600, fontFamily: 'Inter, sans-serif', color: 'hsl(var(--ink))' }}>
                    K Global Context
                  </label>
                  <input
                    type="number"
                    min={0} max={20} step={1}
                    value={settings.rag_retrieval_k_global !== undefined ? settings.rag_retrieval_k_global : 2}
                    onChange={(e) => {
                      const parsed = parseInt(e.target.value, 10)
                      upd('rag_retrieval_k_global', isNaN(parsed) ? 2 : parsed)
                    }}
                    style={{
                      fontFamily: 'JetBrains Mono, monospace', fontSize: '.85rem',
                      color: 'hsl(var(--accent))', fontWeight: 700,
                      padding: '.25rem .5rem',
                      background: 'hsl(var(--accent) / .1)',
                      borderRadius: '6px',
                      border: '1.5px solid hsl(var(--accent) / .2)',
                      width: '80px', textAlign: 'center',
                      outline: 'none',
                    }}
                  />
                </div>
                <p style={{ fontSize: '0.78rem', color: 'hsl(var(--pencil))', margin: 0, lineHeight: 1.4 }}>
                  Top-K global context documents to retrieve (default: 2).
                </p>
              </div>

              {/* K Meeting */}
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px', alignItems: 'center' }}>
                  <label style={{ fontSize: '.9rem', fontWeight: 600, fontFamily: 'Inter, sans-serif', color: 'hsl(var(--ink))' }}>
                    K Meeting Attachments
                  </label>
                  <input
                    type="number"
                    min={0} max={20} step={1}
                    value={settings.rag_retrieval_k_meeting !== undefined ? settings.rag_retrieval_k_meeting : 3}
                    onChange={(e) => {
                      const parsed = parseInt(e.target.value, 10)
                      upd('rag_retrieval_k_meeting', isNaN(parsed) ? 3 : parsed)
                    }}
                    style={{
                      fontFamily: 'JetBrains Mono, monospace', fontSize: '.85rem',
                      color: 'hsl(var(--accent))', fontWeight: 700,
                      padding: '.25rem .5rem',
                      background: 'hsl(var(--accent) / .1)',
                      borderRadius: '6px',
                      border: '1.5px solid hsl(var(--accent) / .2)',
                      width: '80px', textAlign: 'center',
                      outline: 'none',
                    }}
                  />
                </div>
                <p style={{ fontSize: '0.78rem', color: 'hsl(var(--pencil))', margin: 0, lineHeight: 1.4 }}>
                  Top-K meeting context attachments to retrieve (default: 3).
                </p>
              </div>

              {/* K Transcript */}
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px', alignItems: 'center' }}>
                  <label style={{ fontSize: '.9rem', fontWeight: 600, fontFamily: 'Inter, sans-serif', color: 'hsl(var(--ink))' }}>
                    K Transcript Chunks
                  </label>
                  <input
                    type="number"
                    min={0} max={30} step={1}
                    value={settings.rag_retrieval_k_transcript !== undefined ? settings.rag_retrieval_k_transcript : 10}
                    onChange={(e) => {
                      const parsed = parseInt(e.target.value, 10)
                      upd('rag_retrieval_k_transcript', isNaN(parsed) ? 10 : parsed)
                    }}
                    style={{
                      fontFamily: 'JetBrains Mono, monospace', fontSize: '.85rem',
                      color: 'hsl(var(--accent))', fontWeight: 700,
                      padding: '.25rem .5rem',
                      background: 'hsl(var(--accent) / .1)',
                      borderRadius: '6px',
                      border: '1.5px solid hsl(var(--accent) / .2)',
                      width: '80px', textAlign: 'center',
                      outline: 'none',
                    }}
                  />
                </div>
                <p style={{ fontSize: '0.78rem', color: 'hsl(var(--pencil))', margin: 0, lineHeight: 1.4 }}>
                  Top-K transcript discussion chunks to retrieve (default: 10).
                </p>
              </div>

              {/* Score Cutoff */}
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px', alignItems: 'center' }}>
                  <label style={{ fontSize: '.9rem', fontWeight: 600, fontFamily: 'Inter, sans-serif', color: 'hsl(var(--ink))' }}>
                    Relative Score Cutoff
                  </label>
                  <input
                    type="number"
                    min={0.001} max={0.5} step={0.001}
                    value={settings.rag_relative_score_cutoff !== undefined ? settings.rag_relative_score_cutoff : 0.01}
                    onChange={(e) => {
                      const parsed = parseFloat(e.target.value)
                      upd('rag_relative_score_cutoff', isNaN(parsed) ? 0.01 : parsed)
                    }}
                    style={{
                      fontFamily: 'JetBrains Mono, monospace', fontSize: '.85rem',
                      color: 'hsl(var(--accent))', fontWeight: 700,
                      padding: '.25rem .5rem',
                      background: 'hsl(var(--accent) / .1)',
                      borderRadius: '6px',
                      border: '1.5px solid hsl(var(--accent) / .2)',
                      width: '80px', textAlign: 'center',
                      outline: 'none',
                    }}
                  />
                </div>
                <p style={{ fontSize: '0.78rem', color: 'hsl(var(--pencil))', margin: 0, lineHeight: 1.4 }}>
                  Similarity score delta threshold to filter low-ranking search results (default: 0.01).
                </p>
              </div>
            </div>
          </div>

          {/* ─── AI Task Output Settings ─── */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginTop: '2.5rem', marginBottom: '1.5rem' }}>
            <div style={{
              width: '36px', height: '36px', borderRadius: '9px',
              background: 'hsl(260,70%,60% / .15)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              border: '1.5px solid hsl(260,70%,60% / .3)',
            }}>
              <FileText size={18} style={{ color: 'hsl(260,70%,60%)' }} />
            </div>
            <h2 style={{ fontSize: '1.15rem', fontWeight: 700, fontFamily: 'Inter, sans-serif', color: 'hsl(var(--ink))' }}>
              AI Task Output Limits (max_new_tokens)
            </h2>
          </div>

          <div style={{
            padding: '1.75rem',
            background: 'hsl(var(--card))',
            border: '1.5px solid hsl(var(--ink) / .1)',
            borderRadius: '12px',
            display: 'flex', flexDirection: 'column', gap: '2rem',
            marginBottom: '2rem'
          }}>
            <p style={{ fontSize: '0.85rem', color: 'hsl(var(--pencil))', fontFamily: 'Inter, sans-serif', margin: 0, lineHeight: 1.6 }}>
              Configure the maximum generation token limit (`max_new_tokens`) for every individual AI task type.
            </p>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '1.5rem' }}>
              {[
                { key: 'max_tokens_mom', label: 'Minutes of Meeting (MoM)', desc: 'Max output tokens for generating the primary Minutes of Meeting.' },
                { key: 'max_tokens_mom_merge', label: 'MoM Merge Consolidation', desc: 'Max tokens used during multi-section MoM merges.' },
                { key: 'max_tokens_raw_mom_to_mom', label: 'Raw MoM to Final MoM', desc: 'Max tokens for compiling final MoM from raw annotations.' },
                { key: 'max_tokens_raw_mom_extraction', label: 'Raw MoM Extraction', desc: 'Max tokens for parsing discussion points per section.' },
                { key: 'max_tokens_raw_mom_repair', label: 'Raw MoM Repair', desc: 'Max tokens used to fix corrupted raw MoM JSON structures.' },
                { key: 'max_tokens_agenda_compress', label: 'Agenda Compression', desc: 'Max tokens for converting long/messy agendas into lists.' },
                { key: 'max_tokens_reference_compress', label: 'Reference Document Compression', desc: 'Max tokens for summarizing attached reference knowledge.' },
                { key: 'max_tokens_agenda_from_summary', label: 'Agenda Generation from Summary', desc: 'Max tokens for creating agenda items based on a summary.' },
                { key: 'max_tokens_executive_summary', label: 'Executive Summary', desc: 'Max tokens for generating executive PDF report summaries.' },
                { key: 'max_tokens_short_summary', label: 'Short Summary', desc: 'Max tokens for generating the ~120-word meeting summary.' },
                { key: 'max_tokens_detailed_summary', label: 'Detailed Summary', desc: 'Max tokens for the detailed section-by-section summaries.' },
                { key: 'max_tokens_chunk_summary', label: 'Chunk Summary', desc: 'Max tokens for single-chunk summaries during RAG/hierarchical passes.' },
                { key: 'max_tokens_key_points', label: 'Key Points', desc: 'Max tokens for generating bullet-point meeting highlights.' },
                { key: 'max_tokens_action_items', label: 'Action Items', desc: 'Max tokens for extracting standard action item lists.' },
                { key: 'max_tokens_key_decisions', label: 'Key Decisions', desc: 'Max tokens for highlighting critical meeting decisions.' },
                { key: 'max_tokens_speaker_summary', label: 'Speaker Summary', desc: 'Max tokens for summarizing a speaker\'s overall contributions.' },
                { key: 'max_tokens_speaker_key_points', label: 'Speaker Key Points', desc: 'Max tokens for extracting speaker-specific highlights.' },
                { key: 'max_tokens_speaker_action_items', label: 'Speaker Action Items', desc: 'Max tokens for extracting tasks assigned to specific speakers.' },
                { key: 'max_tokens_collection_chat', label: 'Collection Chat', desc: 'Max output tokens for collection-level RAG questions.' },
                { key: 'max_tokens_collection_compare', label: 'Collection Comparison', desc: 'Max tokens for comparing two full meetings.' },
                { key: 'max_tokens_collection_topic_growth', label: 'Collection Topic Growth', desc: 'Max tokens for tracking how a topic grows over time.' },
                { key: 'max_tokens_vocab_extractor', label: 'Vocabulary Extractor', desc: 'Max tokens for AI-assisted glossary/vocab extraction.' },
              ].map(({ key, label, desc }) => {
                return (
                  <div key={key}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px', alignItems: 'center' }}>
                      <label style={{ fontSize: '.9rem', fontWeight: 600, fontFamily: 'Inter, sans-serif', color: 'hsl(var(--ink))' }}>
                        {label}
                      </label>
                      <input
                        type="number"
                        min={1} max={65536}
                        value={settings[key as keyof UserSettings] ?? 1024}
                        onChange={(e) => {
                          const parsed = parseInt(e.target.value, 10)
                          upd(key as keyof UserSettings, isNaN(parsed) ? 1024 : parsed)
                        }}
                        style={{
                          fontFamily: 'JetBrains Mono, monospace', fontSize: '.85rem',
                          color: 'hsl(var(--accent))', fontWeight: 700,
                          padding: '.25rem .5rem',
                          background: 'hsl(var(--accent) / .1)',
                          borderRadius: '6px',
                          border: '1.5px solid hsl(var(--accent) / .2)',
                          width: '80px', textAlign: 'center',
                          outline: 'none',
                        }}
                      />
                    </div>
                    <p style={{ fontSize: '0.78rem', color: 'hsl(var(--pencil))', margin: 0, lineHeight: 1.4 }}>
                      {desc}
                    </p>
                  </div>
                )
              })}
            </div>
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

          </div>
        )}
      
      </section>
      </div>
    </div>
  )
}
