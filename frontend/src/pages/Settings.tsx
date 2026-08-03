import { useEffect, useState, useRef } from 'react'
import {
  Settings, Mic, Trash2, Pencil, Save, Loader, Sliders, Sparkles, User, CheckCircle,
  MessageSquare, RotateCcw, Upload, Download, FileText, Code, Cpu, Database, Volume2, Activity,
  Layers, Target, RefreshCw, Zap, Shield, Search, ChevronRight, SlidersHorizontal, Copy, Maximize2, Minimize2, Check
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
  rag_max_collection_context?: number
  rag_relative_score_cutoff?: number
  generate_mom_auto?: boolean
  embedding_model?: string

  // Ollama settings
  ollama_num_ctx?: number
  ollama_dynamic_ctx?: boolean
  ollama_temperature?: number
  ollama_top_p?: number
  ollama_top_k?: number
  ollama_repeat_penalty?: number
  ollama_seed?: number
  ollama_stop?: string
  ollama_keep_alive?: string
  ollama_num_thread?: number
  ollama_num_gpu?: number

  // ROM Pipeline Settings
  rom_transcript_window?: number
  rom_meeting_top_k?: number
  rom_global_top_k?: number
  rom_windows_per_batch?: number
  rom_parallel_window_processing?: number

  // Whisper Settings
  whisper_batch_size?: number

  max_tokens_rom_discussion?: number
  max_tokens_rom_polish?: number
  max_tokens_rom_enhance_window?: number
  max_tokens_rom_deduplicate?: number
  max_tokens_rom_agenda?: number
  max_tokens_rom_mom_expansion?: number
  max_tokens_rom_agenda_assign_batch?: number
  max_tokens_rom_agenda_doc_points?: number

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

  // Low-Volume Speech Transcription Pipeline Enhancements
  enable_vad?: boolean
  enable_transcription_vad?: boolean
  enable_alignment_vad?: boolean
  enable_audio_normalization?: boolean
  norm_target_dbfs?: number
  norm_compression_ratio?: number

  enable_adaptive_vad?: boolean
  vad_speech_threshold?: number
  vad_silence_threshold?: number
  vad_min_speech_ms?: number
  vad_min_silence_ms?: number

  enable_speech_padding?: boolean
  speech_pad_ms?: number

  enable_speech_segment_merging?: boolean
  max_merge_silence_ms?: number

  enable_low_volume_recovery?: boolean
  recovery_energy_threshold?: number
  recovery_min_duration_ms?: number

  enable_audio_validation?: boolean
  min_audio_duration_seconds?: number
  min_audio_rms_threshold?: number
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

interface EmbeddingModelOption {
  id: string
  name: string
  description: string
  dim?: number | null
  quantization?: string
  installed: boolean
  path?: string | null
}

type SettingsTab = 'rom' | 'prompts' | 'voice' | 'llm' | 'rag' | 'audio' | 'tokens'

export default function SettingsPage() {
  const [activeTab, setActiveTab] = useState<SettingsTab>('rom')
  const [profiles, setProfiles] = useState<Profile[]>([])
  const [settings, setSettings] = useState<UserSettings | null>(null)
  const [embeddingModels, setEmbeddingModels] = useState<EmbeddingModelOption[]>([])
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
  const [activePromptCategory, setActivePromptCategory] = useState<string>('All')
  const [promptSearchQuery, setPromptSearchQuery] = useState<string>('')
  const [savingKeys, setSavingKeys] = useState<Set<string>>(new Set())
  const [savedKeys, setSavedKeys] = useState<Set<string>>(new Set())
  const [importing, setImporting] = useState(false)
  const [editingTemplates, setEditingTemplates] = useState<Record<string, string>>({})
  const [expandedPromptKey, setExpandedPromptKey] = useState<string | null>(null)
  const [copiedKey, setCopiedKey] = useState<string | null>(null)

  const loadData = async () => {
    try {
      const [pRes, sRes, prRes, ptRes, emRes] = await Promise.all([
        api.get('/voice/profiles'),
        api.get('/settings'),
        api.get('/prompt/global'),
        api.get('/prompt-templates'),
        api.get('/settings/embedding-models').catch(() => ({ data: { models: [] } })),
      ])
      setProfiles(pRes.data)
      setSettings(sRes.data)
      setGlobalPrompt(prRes.data?.prompt || '')
      setPrompts(ptRes.data)
      if (emRes.data?.models) {
        setEmbeddingModels(emRes.data.models)
      }
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
    try {
      await api.put('/settings', settings)
      setSettingsSaved(true)
      toast.success('Settings saved successfully!')
      setTimeout(() => setSettingsSaved(false), 2000)
    } catch (e: any) {
      toast.error(e?.response?.data?.detail || 'Failed to save settings')
    } finally {
      setSavingSettings(false)
    }
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

  const handleSavePromptTemplate = async (key: string) => {
    const pt = prompts.find(p => p.key === key)
    if (!pt) return
    const template = editingTemplates[key] ?? pt.template
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

  const handleCopyPrompt = (key: string, text: string) => {
    navigator.clipboard.writeText(text)
    setCopiedKey(key)
    toast.success('Prompt text copied to clipboard!')
    setTimeout(() => setCopiedKey(null), 2000)
  }

  const handleInsertVariable = (key: string, variable: string) => {
    const current = editingTemplates[key] ?? prompts.find(p => p.key === key)?.template ?? ''
    const updated = current + ` ${variable}`
    setEditingTemplates(prev => ({ ...prev, [key]: updated }))
    toast.info(`Inserted variable ${variable}`)
  }

  const PROFILE_COLORS = [
    'hsl(14, 90%, 56%)',
    'hsl(205, 90%, 55%)',
    'hsl(130, 60%, 45%)',
    'hsl(280, 70%, 60%)',
    'hsl(35, 90%, 50%)',
    'hsl(330, 75%, 55%)',
  ]

  const promptCategories = ['All', ...Array.from(new Set(prompts.map(p => p.category)))]
  const modifiedCount = prompts.filter(p => p.is_modified).length
  const romPromptsCount = prompts.filter(p => p.category === 'ROM').length

  const filteredPrompts = prompts.filter(p => {
    const matchesCat = activePromptCategory === 'All' || p.category === activePromptCategory
    const q = promptSearchQuery.toLowerCase().trim()
    const matchesSearch = !q || p.name.toLowerCase().includes(q) || p.key.toLowerCase().includes(q) || p.description.toLowerCase().includes(q)
    return matchesCat && matchesSearch
  })

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', minHeight: 0, overflow: 'hidden' }}>

      {/* Top Header Bar */}
      <div className="panel-header" style={{ flexShrink: 0, justifyContent: 'space-between', padding: '.65rem 1.5rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div style={{
            width: '36px', height: '36px', borderRadius: '10px', flexShrink: 0,
            background: 'hsl(var(--accent) / .12)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            border: '2px solid hsl(var(--accent) / .3)',
          }}>
            <Settings size={18} style={{ color: 'hsl(var(--accent))' }} />
          </div>
          <div>
            <h1 style={{ fontSize: '1.1rem', fontWeight: 700, margin: 0 }}>System Settings &amp; Configuration</h1>
            <p style={{ fontSize: '.76rem', color: 'hsl(var(--pencil))', fontFamily: 'Inter, sans-serif', margin: 0 }}>
              ROM pipeline parameters, prompt templates engine, inference tuning &amp; audio processing
            </p>
          </div>
        </div>

        {/* Global Save Button */}
        <button
          onClick={handleSaveSettings}
          disabled={savingSettings}
          className="btn btn-primary"
          style={{
            fontSize: '.78rem', padding: '.45rem 1.05rem', borderRadius: 8,
            fontWeight: 700, display: 'flex', alignItems: 'center', gap: 6,
            background: settingsSaved ? 'hsl(140,70%,45%)' : undefined
          }}
        >
          {savingSettings ? <Loader size={13} className="spin" /> : settingsSaved ? <CheckCircle size={13} /> : <Save size={13} />}
          {savingSettings ? 'Saving...' : settingsSaved ? 'Settings Saved!' : 'Save All Settings'}
        </button>
      </div>

      {/* Stats Metric Strip */}
      <div style={{
        display: 'flex', gap: '1.25rem', padding: '.55rem 1.5rem',
        background: 'hsl(var(--muted)/.25)', borderBottom: '1px solid hsl(var(--border)/.3)',
        fontSize: '.72rem', color: 'hsl(var(--pencil))', alignItems: 'center', flexWrap: 'wrap'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
          <Sparkles size={13} style={{ color: 'hsl(280,75%,60%)' }} />
          <span>ROM Prompts: <strong style={{ color: 'hsl(var(--ink))' }}>{romPromptsCount} Active</strong></span>
        </div>
        <div style={{ width: 1, height: 12, background: 'hsl(var(--border))' }} />
        <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
          <FileText size={13} style={{ color: 'hsl(140,70%,45%)' }} />
          <span>Total Templates: <strong style={{ color: 'hsl(var(--ink))' }}>{prompts.length} Registered</strong></span>
        </div>
        <div style={{ width: 1, height: 12, background: 'hsl(var(--border))' }} />
        <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
          <Pencil size={13} style={{ color: 'hsl(35,90%,50%)' }} />
          <span>Customized Overrides: <strong style={{ color: 'hsl(var(--ink))' }}>{modifiedCount} Override(s)</strong></span>
        </div>
        <div style={{ width: 1, height: 12, background: 'hsl(var(--border))' }} />
        <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
          <Mic size={13} style={{ color: 'hsl(205,90%,55%)' }} />
          <span>Voice Profiles: <strong style={{ color: 'hsl(var(--ink))' }}>{profiles.length} Saved</strong></span>
        </div>
      </div>

      {/* Main Tabbed Layout Container */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0, overflow: 'hidden' }}>

        {/* Navigation Tabs Bar */}
        <div style={{
          display: 'flex', gap: 6, padding: '.65rem 1.5rem',
          background: 'hsl(var(--card))', borderBottom: '1px solid hsl(var(--border)/.4)',
          overflowX: 'auto', flexShrink: 0
        }}>
          {[
            { id: 'rom', label: 'ROM Pipeline', count: '4 Settings', icon: <Sparkles size={14} />, color: 'hsl(280,75%,60%)' },
            { id: 'prompts', label: 'Prompt Templates', count: `${prompts.length}`, icon: <FileText size={14} />, color: 'hsl(140,70%,45%)' },
            { id: 'voice', label: 'Voice & Diarization', count: `${profiles.length} Profiles`, icon: <Mic size={14} />, color: 'hsl(205,90%,55%)' },
            { id: 'llm', label: 'LLM Inference', count: 'Ollama', icon: <Cpu size={14} />, color: 'hsl(35,90%,50%)' },
            { id: 'rag', label: 'RAG & Vectors', count: 'FAISS / BM25', icon: <Database size={14} />, color: 'hsl(200,80%,50%)' },
            { id: 'audio', label: 'VAD & Audio', count: 'dBFS / VAD', icon: <Volume2 size={14} />, color: 'hsl(330,75%,55%)' },
            { id: 'tokens', label: 'Task Token Limits', count: 'Max Tokens', icon: <SlidersHorizontal size={14} />, color: 'hsl(250,70%,60%)' },
          ].map(tab => {
            const active = activeTab === tab.id
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as SettingsTab)}
                style={{
                  display: 'flex', alignItems: 'center', gap: 6,
                  padding: '.5rem .9rem', borderRadius: 8,
                  fontSize: '.78rem', fontWeight: active ? 700 : 500,
                  color: active ? tab.color : 'hsl(var(--pencil))',
                  background: active ? `${tab.color}15` : 'transparent',
                  border: active ? `1.5px solid ${tab.color}40` : '1px solid transparent',
                  cursor: 'pointer', fontFamily: 'Inter', whiteSpace: 'nowrap',
                  transition: 'all .15s ease'
                }}
              >
                {tab.icon}
                <span>{tab.label}</span>
                <span style={{ fontSize: '.65rem', padding: '1px 5px', borderRadius: 6, background: active ? `${tab.color}25` : 'hsl(var(--muted)/.6)', color: active ? tab.color : 'hsl(var(--pencil))', fontWeight: 600 }}>{tab.count}</span>
              </button>
            )
          })}
        </div>

        {/* Scrollable Tab Content View */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '1.25rem 1.5rem 2rem' }}>

          {/* 🚀 TAB: ROM PIPELINE */}
          {activeTab === 'rom' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem', maxWidth: 1000, margin: '0 auto' }}>
              <div style={{ borderRadius: 12, border: '1.5px solid hsl(280,75%,60%/.3)', background: 'hsl(var(--card))', padding: '1.25rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: '1.15rem' }}>
                  <div style={{ width: 36, height: 36, borderRadius: 9, background: 'hsl(280,75%,60%/.12)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'hsl(280,75%,60%)' }}>
                    <Sparkles size={18} />
                  </div>
                  <div>
                    <h3 style={{ fontSize: '1.02rem', fontWeight: 700, margin: 0 }}>ROM Pipeline Parameters</h3>
                    <div style={{ fontSize: '.75rem', color: 'hsl(var(--pencil))' }}>Tune sliding transcript window size, context retrieval K depth, and enhancement window batching</div>
                  </div>
                </div>

                {settings && (
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1rem' }}>
                    <SettingCard
                      title="Stage 1 Window Size (mins)"
                      description="Sliding audio/transcript window duration processed in Stage 1 extraction."
                      value={settings.rom_transcript_window ?? 2.0}
                      min={0.5} max={10.0} step={0.5}
                      onChange={v => setSettings({ ...settings, rom_transcript_window: v })}
                    />
                    <SettingCard
                      title="Stage 2 Meeting Context Top-K"
                      description="Top-K context chunks retrieved from uploaded meeting supporting documents."
                      value={settings.rom_meeting_top_k ?? 5}
                      min={1} max={30} step={1}
                      onChange={v => setSettings({ ...settings, rom_meeting_top_k: Math.round(v) })}
                    />
                    <SettingCard
                      title="Stage 2 Global Context Top-K"
                      description="Top-K context chunks retrieved from organizational global context knowledge."
                      value={settings.rom_global_top_k ?? 3}
                      min={1} max={20} step={1}
                      onChange={v => setSettings({ ...settings, rom_global_top_k: Math.round(v) })}
                    />
                    <SettingCard
                      title="Stage 2 Points per Enhancement Window"
                      description="Number of discussion points grouped into one enhancement window LLM call."
                      value={settings.rom_windows_per_batch ?? 5}
                      min={1} max={20} step={1}
                      onChange={v => setSettings({ ...settings, rom_windows_per_batch: Math.round(v) })}
                    />
                    <SettingCard
                      title="Parallel Window Processing (Stage 1 & 2)"
                      description="Max number of LLM window extraction calls running concurrently (1–5, default 2)."
                      value={settings.rom_parallel_window_processing ?? 2}
                      min={1} max={5} step={1}
                      onChange={v => setSettings({ ...settings, rom_parallel_window_processing: Math.round(v) })}
                    />
                  </div>
                )}
              </div>

              {/* ROM Max Tokens */}
              <div style={{ borderRadius: 12, border: '1.5px solid hsl(var(--border)/.4)', background: 'hsl(var(--card))', padding: '1.25rem' }}>
                <h4 style={{ fontSize: '.92rem', fontWeight: 700, marginBottom: '.85rem', color: 'hsl(var(--ink))', display: 'flex', alignItems: 'center', gap: 6 }}>
                  <Sliders size={14} style={{ color: 'hsl(280,75%,60%)' }} /> ROM Task Token Limits
                </h4>
                {settings && (
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '.85rem' }}>
                    <TokenInput label="Stage 1 Discussion Extraction" val={settings.max_tokens_rom_discussion ?? 4096} onChange={v => setSettings({ ...settings, max_tokens_rom_discussion: v })} />
                    <TokenInput label="Stage 2 Polish & Merge" val={settings.max_tokens_rom_polish ?? 4096} onChange={v => setSettings({ ...settings, max_tokens_rom_polish: v })} />
                    <TokenInput label="Stage 2 Enhance Window" val={settings.max_tokens_rom_enhance_window ?? 4096} onChange={v => setSettings({ ...settings, max_tokens_rom_enhance_window: v })} />
                    <TokenInput label="Stage 2 Deduplication" val={settings.max_tokens_rom_deduplicate ?? 2048} onChange={v => setSettings({ ...settings, max_tokens_rom_deduplicate: v })} />
                    <TokenInput label="Stage 3 Generate Agendas" val={settings.max_tokens_rom_agenda ?? 2048} onChange={v => setSettings({ ...settings, max_tokens_rom_agenda: v })} />
                    <TokenInput label="Stage 3 Previous MoM Expansion" val={settings.max_tokens_rom_mom_expansion ?? 3000} onChange={v => setSettings({ ...settings, max_tokens_rom_mom_expansion: v })} />
                    <TokenInput label="Stage 3 Agenda Batch Mapping" val={settings.max_tokens_rom_agenda_assign_batch ?? 4096} onChange={v => setSettings({ ...settings, max_tokens_rom_agenda_assign_batch: v })} />
                    <TokenInput label="Stage 3 Supporting Doc Points" val={settings.max_tokens_rom_agenda_doc_points ?? 1024} onChange={v => setSettings({ ...settings, max_tokens_rom_agenda_doc_points: v })} />
                  </div>
                )}
              </div>
            </div>
          )}

          {/* 📝 TAB: PROMPT TEMPLATES */}
          {activeTab === 'prompts' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem', maxWidth: 1100, margin: '0 auto' }}>
              
              {/* Prompt Controls & Search Header */}
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, alignItems: 'center', justifyContent: 'space-between', borderRadius: 12, border: '1.5px solid hsl(140,70%,45%/.3)', background: 'hsl(var(--card))', padding: '.85rem 1.15rem' }}>
                
                {/* Search */}
                <div style={{ display: 'flex', alignItems: 'center', gap: 6, background: 'hsl(var(--muted)/.3)', border: '1px solid hsl(var(--border)/.4)', borderRadius: 8, padding: '5px 10px', width: 280 }}>
                  <Search size={14} style={{ color: 'hsl(var(--pencil))' }} />
                  <input
                    type="text"
                    placeholder="Search prompts by name, key, description..."
                    value={promptSearchQuery}
                    onChange={e => setPromptSearchQuery(e.target.value)}
                    style={{ border: 'none', background: 'transparent', outline: 'none', fontSize: '.78rem', width: '100%', color: 'hsl(var(--ink))', fontFamily: 'Inter' }}
                  />
                  {promptSearchQuery && <button onClick={() => setPromptSearchQuery('')} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'hsl(var(--pencil))', padding: 0 }}><RotateCcw size={11} /></button>}
                </div>

                {/* Bulk Actions */}
                <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                  <button onClick={handleResetAllPrompts} className="btn btn-secondary" style={{ fontSize: '.74rem', padding: '.35rem .7rem' }}>
                    <RotateCcw size={12} /> Reset All Defaults
                  </button>
                  <button onClick={handleExportPrompts} className="btn btn-secondary" style={{ fontSize: '.74rem', padding: '.35rem .7rem' }}>
                    <Download size={12} /> Export JSON
                  </button>
                  <label className="btn btn-secondary" style={{ fontSize: '.74rem', padding: '.35rem .7rem', cursor: 'pointer' }}>
                    <Upload size={12} /> {importing ? 'Importing...' : 'Import JSON'}
                    <input type="file" accept=".json" onChange={handleImportPrompts} style={{ display: 'none' }} disabled={importing} />
                  </label>
                </div>
              </div>

              {/* Category Pills */}
              <div style={{ display: 'flex', gap: 6, overflowX: 'auto', paddingBottom: 4 }}>
                {promptCategories.map(cat => {
                  const active = activePromptCategory === cat
                  return (
                    <button
                      key={cat}
                      onClick={() => setActivePromptCategory(cat)}
                      style={{
                        padding: '.35rem .85rem', borderRadius: 20, fontSize: '.76rem', fontWeight: active ? 700 : 500,
                        background: active ? (cat === 'ROM' ? 'hsl(280,75%,60%)' : 'hsl(var(--accent))') : 'hsl(var(--muted)/.4)',
                        color: active ? 'white' : 'hsl(var(--ink))',
                        border: 'none', cursor: 'pointer', fontFamily: 'Inter', flexShrink: 0,
                        boxShadow: active ? '0 2px 8px rgba(0,0,0,0.15)' : 'none'
                      }}
                    >
                      {cat} {cat !== 'All' && `(${prompts.filter(p => p.category === cat).length})`}
                    </button>
                  )
                })}
              </div>

              {/* Global System Prompt Card */}
              <div style={{ borderRadius: 12, border: '1.5px solid hsl(var(--border)/.4)', background: 'hsl(var(--card))', padding: '1rem 1.15rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '.5rem' }}>
                  <div style={{ fontSize: '.84rem', fontWeight: 700, color: 'hsl(var(--ink))' }}>System-Wide Global System Prompt</div>
                  {promptSaving && <span style={{ fontSize: '.7rem', color: 'hsl(var(--pencil))' }}>Saving...</span>}
                  {promptSaved && <span style={{ fontSize: '.7rem', color: 'hsl(140,70%,45%)', fontWeight: 700 }}>Saved!</span>}
                </div>
                <textarea
                  value={globalPrompt}
                  onChange={e => handlePromptChange(e.target.value)}
                  rows={2}
                  placeholder="Enter system-wide instructions injected into all AI calls..."
                  style={{ width: '100%', padding: '.6rem .75rem', borderRadius: 8, border: '1px solid hsl(var(--border)/.4)', background: 'hsl(var(--paper)/.4)', fontSize: '.78rem', fontFamily: 'Inter', lineHeight: 1.45, color: 'hsl(var(--ink))' }}
                />
              </div>

              {/* Prompt Templates List */}
              {loadingPrompts ? (
                <div style={{ textAlign: 'center', padding: '3rem', color: 'hsl(var(--pencil))' }}><Loader size={24} className="spin" /> Loading prompt templates...</div>
              ) : filteredPrompts.length === 0 ? (
                <div style={{ textAlign: 'center', padding: '3rem', color: 'hsl(var(--pencil))' }}>No prompt templates found for this filter.</div>
              ) : (
                filteredPrompts.map(pt => {
                  const isSaving = savingKeys.has(pt.key)
                  const isSaved = savedKeys.has(pt.key)
                  const currentVal = editingTemplates[pt.key] ?? pt.template
                  const isExpanded = expandedPromptKey === pt.key

                  return (
                    <div key={pt.key} style={{ borderRadius: 12, border: pt.is_modified ? '1.5px solid hsl(35,90%,50%/.6)' : '1.5px solid hsl(var(--border)/.4)', background: 'hsl(var(--card))', padding: '1.1rem 1.25rem', display: 'flex', flexDirection: 'column', gap: '.7rem' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 8 }}>
                        <div>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
                            <span style={{ fontSize: '.92rem', fontWeight: 700, color: 'hsl(var(--ink))' }}>{pt.name}</span>
                            <span style={{ fontSize: '.65rem', fontWeight: 700, padding: '1px 6px', borderRadius: 4, background: pt.category === 'ROM' ? 'hsl(280,75%,60%/.15)' : 'hsl(var(--muted))', color: pt.category === 'ROM' ? 'hsl(280,75%,60%)' : 'hsl(var(--pencil))' }}>{pt.category}</span>
                            <span style={{ fontSize: '.65rem', fontFamily: 'JetBrains Mono', color: 'hsl(var(--pencil))' }}>key: {pt.key}</span>
                            {pt.is_modified && <span style={{ fontSize: '.65rem', fontWeight: 700, padding: '1px 6px', borderRadius: 4, background: 'hsl(35,90%,50%/.15)', color: 'hsl(35,90%,45%)', border: '1px solid hsl(35,90%,50%/.3)' }}>Customized</span>}
                          </div>
                          <div style={{ fontSize: '.74rem', color: 'hsl(var(--pencil))', marginTop: 3 }}>{pt.description}</div>
                        </div>

                        {/* Prompt Action Buttons */}
                        <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                          <button
                            onClick={() => handleCopyPrompt(pt.key, currentVal)}
                            title="Copy prompt template text"
                            style={{ padding: '.25rem .55rem', borderRadius: 6, border: '1px solid hsl(var(--border)/.4)', background: 'transparent', fontSize: '.7rem', color: 'hsl(var(--pencil))', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 4 }}
                          >
                            {copiedKey === pt.key ? <Check size={10} style={{ color: 'hsl(140,70%,45%)' }} /> : <Copy size={10} />}
                            {copiedKey === pt.key ? 'Copied' : 'Copy'}
                          </button>
                          <button
                            onClick={() => setExpandedPromptKey(isExpanded ? null : pt.key)}
                            title={isExpanded ? 'Collapse editor' : 'Expand editor'}
                            style={{ padding: '.25rem .55rem', borderRadius: 6, border: '1px solid hsl(var(--border)/.4)', background: 'transparent', fontSize: '.7rem', color: 'hsl(var(--pencil))', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 4 }}
                          >
                            {isExpanded ? <Minimize2 size={10} /> : <Maximize2 size={10} />}
                            {isExpanded ? 'Collapse' : 'Expand'}
                          </button>
                          {pt.is_modified && (
                            <button onClick={() => handleResetPromptTemplate(pt.key)} title="Reset to default" style={{ padding: '.25rem .55rem', borderRadius: 6, border: '1px solid hsl(var(--border)/.4)', background: 'transparent', fontSize: '.7rem', color: 'hsl(var(--pencil))', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 4 }}>
                              <RotateCcw size={10} /> Reset
                            </button>
                          )}
                          <button
                            onClick={() => handleSavePromptTemplate(pt.key)}
                            disabled={isSaving}
                            style={{
                              padding: '.32rem .8rem', borderRadius: 6, fontSize: '.74rem', fontWeight: 700,
                              background: isSaved ? 'hsl(140,70%,45%)' : 'hsl(var(--accent))',
                              color: 'white', border: 'none', cursor: isSaving ? 'not-allowed' : 'pointer',
                              display: 'flex', alignItems: 'center', gap: 4, fontFamily: 'Inter'
                            }}
                          >
                            {isSaving ? <Loader size={10} className="spin" /> : isSaved ? <CheckCircle size={10} /> : <Save size={10} />}
                            {isSaving ? 'Saving' : isSaved ? 'Saved' : 'Save Template'}
                          </button>
                        </div>
                      </div>

                      {/* Variables pill list (Clicking variable inserts it into prompt!) */}
                      {pt.variables?.length > 0 && (
                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5, alignItems: 'center', background: 'hsl(var(--muted)/.2)', padding: '.35rem .65rem', borderRadius: 6 }}>
                          <span style={{ fontSize: '.67rem', color: 'hsl(var(--pencil))', fontWeight: 600 }}>Click variable to insert:</span>
                          {pt.variables.map((v, vi) => (
                            <button
                              key={vi}
                              onClick={() => handleInsertVariable(pt.key, v)}
                              title="Click to insert variable into template"
                              style={{ fontSize: '.66rem', padding: '1px 6px', borderRadius: 4, background: 'hsl(205,90%,55%/.12)', color: 'hsl(205,90%,55%)', border: '1px solid hsl(205,90%,55%/.25)', fontFamily: 'JetBrains Mono', cursor: 'pointer' }}
                            >
                              + {v}
                            </button>
                          ))}
                        </div>
                      )}

                      {/* Template Textarea */}
                      <textarea
                        value={currentVal}
                        onChange={e => setEditingTemplates({ ...editingTemplates, [pt.key]: e.target.value })}
                        rows={isExpanded ? 16 : 7}
                        style={{
                          width: '100%', padding: '.65rem .8rem', borderRadius: 8,
                          border: '1px solid hsl(var(--border)/.4)', background: 'hsl(var(--paper)/.5)',
                          fontSize: '.76rem', fontFamily: 'JetBrains Mono, monospace', lineHeight: 1.45,
                          color: 'hsl(var(--ink))', whiteSpace: 'pre-wrap', transition: 'height .2s ease'
                        }}
                      />
                    </div>
                  )
                })
              )}
            </div>
          )}

          {/* 🎙️ TAB: VOICE & DIARIZATION */}
          {activeTab === 'voice' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem', maxWidth: 950, margin: '0 auto' }}>
              {/* Recognition Thresholds */}
              <div style={{ borderRadius: 12, border: '1.5px solid hsl(205,90%,55%/.3)', background: 'hsl(var(--card))', padding: '1.25rem' }}>
                <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: '1rem', color: 'hsl(var(--ink))', display: 'flex', alignItems: 'center', gap: 8 }}>
                  <Sliders size={16} style={{ color: 'hsl(205,90%,55%)' }} /> Diarization &amp; Speaker Matching
                </h3>
                {settings && (
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '1.1rem' }}>
                    <SettingCard
                      title="Speaker Similarity Threshold"
                      description="Cosine similarity threshold for assigning speaker labels (0.50–0.99)."
                      value={settings.speaker_similarity_threshold}
                      min={0.50} max={0.99} step={0.01}
                      onChange={v => setSettings({ ...settings, speaker_similarity_threshold: v })}
                    />
                    <SettingCard
                      title="Min Segment Duration (secs)"
                      description="Minimum audio segment length to consider for speaker identification."
                      value={settings.min_segment_duration}
                      min={0.1} max={10.0} step={0.1}
                      onChange={v => setSettings({ ...settings, min_segment_duration: v })}
                    />
                    <SettingCard
                      title="Word Confidence Low"
                      description="Low word confidence threshold for transcript highlighting."
                      value={settings.word_conf_low}
                      min={0.1} max={1.0} step={0.05}
                      onChange={v => setSettings({ ...settings, word_conf_low: v })}
                    />
                    <SettingCard
                      title="Word Confidence Mid"
                      description="Mid word confidence threshold for transcript highlighting."
                      value={settings.word_conf_mid}
                      min={0.1} max={1.0} step={0.05}
                      onChange={v => setSettings({ ...settings, word_conf_mid: v })}
                    />
                  </div>
                )}
              </div>

              {/* Voice Profiles List */}
              <div style={{ borderRadius: 12, border: '1.5px solid hsl(var(--border)/.4)', background: 'hsl(var(--card))', padding: '1.25rem' }}>
                <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: '1rem', color: 'hsl(var(--ink))', display: 'flex', alignItems: 'center', gap: 8 }}>
                  <Mic size={16} style={{ color: 'hsl(205,90%,55%)' }} /> Voice Profiles ({profiles.length})
                </h3>
                {loadingProfiles ? (
                  <div style={{ textAlign: 'center', padding: '2rem' }}><Loader size={20} className="spin" /></div>
                ) : profiles.length === 0 ? (
                  <div style={{ textAlign: 'center', padding: '2rem', color: 'hsl(var(--pencil))' }}>No voice profiles saved yet.</div>
                ) : (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                    {profiles.map((p, idx) => {
                      const color = PROFILE_COLORS[idx % PROFILE_COLORS.length]
                      return (
                        <div key={p.id} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '.75rem 1rem', borderRadius: 8, background: 'hsl(var(--paper)/.4)', borderLeft: `4px solid ${color}`, border: '1px solid hsl(var(--border)/.3)' }}>
                          <div>
                            {editingId === p.id ? (
                              <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                                <input value={editLabel} onChange={e => setEditLabel(e.target.value)} style={{ padding: '3px 7px', borderRadius: 6, fontSize: '.8rem', border: '1px solid hsl(var(--border))' }} />
                                <button onClick={() => handleRename(p.id)} disabled={savingLabel} style={{ padding: '3px 8px', borderRadius: 6, background: 'hsl(var(--accent))', color: 'white', border: 'none', fontSize: '.74rem', cursor: 'pointer' }}>{savingLabel ? 'Saving' : 'Save'}</button>
                              </div>
                            ) : (
                              <div style={{ fontWeight: 700, fontSize: '.86rem', color: 'hsl(var(--ink))' }}>{p.label} {p.is_self && <span style={{ fontSize: '.65rem', background: 'hsl(205,90%,55%/.15)', color: 'hsl(205,90%,55%)', padding: '1px 6px', borderRadius: 4, marginLeft: 4 }}>You</span>}</div>
                            )}
                            <div style={{ fontSize: '.71rem', color: 'hsl(var(--pencil))', marginTop: 2 }}>{p.sample_count} audio samples recorded</div>
                          </div>

                          <div style={{ display: 'flex', gap: 6 }}>
                            <button onClick={() => { setEditingId(p.id); setEditLabel(p.label) }} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'hsl(var(--pencil))' }}><Pencil size={12} /></button>
                            <button onClick={() => handleDelete(p.id)} disabled={deletingId === p.id} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'hsl(var(--destructive))' }}><Trash2 size={12} /></button>
                          </div>
                        </div>
                      )
                    })}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* ⚡ TAB: LLM INFERENCE */}
          {activeTab === 'llm' && settings && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem', maxWidth: 950, margin: '0 auto' }}>
              <div style={{ borderRadius: 12, border: `1.5px solid ${settings.use_ollama ? 'hsl(35,90%,50%/.5)' : 'hsl(var(--border)/.4)'}`, background: 'hsl(var(--card))', padding: '1.25rem', transition: 'all .2s' }}>
                <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: '1rem', color: 'hsl(var(--ink))', display: 'flex', alignItems: 'center', gap: 8 }}>
                  <Cpu size={16} style={{ color: 'hsl(35,90%,50%)' }} /> Ollama Inference Server &amp; Tuning
                </h3>

                {/* Enable / Disable Ollama Master Toggle */}
                <div style={{
                  marginBottom: '1.25rem', padding: '1rem 1.15rem', borderRadius: 10,
                  background: settings.use_ollama ? 'hsl(35,90%,50%/.08)' : 'hsl(var(--paper)/.5)',
                  border: `1.5px solid ${settings.use_ollama ? 'hsl(35,90%,50%/.4)' : 'hsl(var(--border)/.5)'}`,
                  display: 'flex', justifyContent: 'space-between', alignItems: 'center', transition: 'all .2s'
                }}>
                  <div>
                    <div style={{ fontSize: '.9rem', fontWeight: 700, color: 'hsl(var(--ink))', display: 'flex', alignItems: 'center', gap: 8 }}>
                      Enable Ollama Local Inference
                      {settings.use_ollama ? (
                        <span style={{ fontSize: '.65rem', padding: '2px 8px', borderRadius: 6, background: 'hsl(35,90%,50%/.2)', color: 'hsl(35,90%,40%)', border: '1px solid hsl(35,90%,50%/.4)', fontWeight: 700 }}>
                          ACTIVE (ON)
                        </span>
                      ) : (
                        <span style={{ fontSize: '.65rem', padding: '2px 8px', borderRadius: 6, background: 'hsl(var(--muted))', color: 'hsl(var(--pencil))', border: '1px solid hsl(var(--border))', fontWeight: 600 }}>
                          DISABLED (OFF)
                        </span>
                      )}
                    </div>
                    <div style={{ fontSize: '.74rem', color: 'hsl(var(--pencil))', marginTop: 3, lineHeight: 1.4 }}>
                      When enabled, AI features prioritize your local Ollama server before falling back to local PyTorch models. Turn off to force local PyTorch model usage exclusively.
                    </div>
                  </div>
                  <input
                    type="checkbox"
                    checked={settings.use_ollama ?? false}
                    onChange={e => setSettings({ ...settings, use_ollama: e.target.checked })}
                    style={{ width: 20, height: 20, cursor: 'pointer', accentColor: 'hsl(35,90%,50%)', flexShrink: 0 }}
                  />
                </div>
                
                {/* Connection & Priority Controls */}
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '1rem', marginBottom: '1rem' }}>
                  <SettingInput label="Ollama Server URL" val={settings.ollama_server_url ?? 'http://localhost:11434'} onChange={v => setSettings({ ...settings, ollama_server_url: v })} />
                  <SettingCard title="Ollama Port" description="Standard port 11434" value={settings.ollama_port ?? 11434} min={1} max={65535} step={1} onChange={v => setSettings({ ...settings, ollama_port: Math.round(v) })} />
                  <SettingInput label="Model Priority List (comma-separated)" val={settings.ollama_model_priority ?? 'gemma,qwen,llama,deepseek,mistral'} onChange={v => setSettings({ ...settings, ollama_model_priority: v })} />
                </div>

                <button onClick={handleTestOllamaConnection} disabled={testingOllama} className="btn btn-secondary" style={{ fontSize: '.78rem', padding: '.45rem 1rem', marginBottom: '1rem' }}>
                  {testingOllama ? <Loader size={12} className="spin" /> : <Zap size={12} />} {testingOllama ? 'Testing Connection...' : 'Test Ollama Connection'}
                </button>

                {ollamaTestResult && (
                  <div style={{ padding: '.75rem 1rem', borderRadius: 8, background: ollamaTestResult.success ? 'hsl(140,70%,45%/.1)' : 'hsl(0,80%,50%/.1)', border: `1px solid ${ollamaTestResult.success ? 'hsl(140,70%,45%/.3)' : 'hsl(0,80%,50%/.3)'}`, fontSize: '.78rem', marginBottom: '1rem' }}>
                    <div style={{ fontWeight: 700, color: ollamaTestResult.success ? 'hsl(140,70%,40%)' : 'hsl(0,80%,45%)' }}>{ollamaTestResult.message}</div>
                    {ollamaTestResult.available_models?.length ? <div style={{ marginTop: 4, color: 'hsl(var(--ink))' }}>Available Models: {ollamaTestResult.available_models.join(', ')}</div> : null}
                    {ollamaTestResult.running_models?.length ? <div style={{ marginTop: 2, color: 'hsl(var(--pencil))' }}>Running Models: {ollamaTestResult.running_models.join(', ')}</div> : null}
                  </div>
                )}

                {/* Hyperparameters */}
                <div style={{ borderTop: '1px solid hsl(var(--border)/.3)', paddingTop: '1rem' }}>
                  <h4 style={{ fontSize: '.86rem', fontWeight: 700, marginBottom: '.85rem', color: 'hsl(var(--ink))' }}>Ollama Generation Hyperparameters</h4>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1rem' }}>
                    <SettingToggle label="Dynamic Context Allocation" checked={settings.ollama_dynamic_ctx ?? true} onChange={v => setSettings({ ...settings, ollama_dynamic_ctx: v })} />
                    <SettingCard title="Context Window (Num Ctx)" description="Maximum context tokens passed to Ollama (default: 32768)." value={settings.ollama_num_ctx ?? 32768} min={512} max={131072} step={1024} onChange={v => setSettings({ ...settings, ollama_num_ctx: Math.round(v) })} />
                    <SettingCard title="Temperature" description="Sampling temperature (0.0 = deterministic)." value={settings.ollama_temperature ?? 0.0} min={0.0} max={2.0} step={0.05} onChange={v => setSettings({ ...settings, ollama_temperature: v })} />
                    <SettingCard title="Top-P" description="Nucleus sampling threshold (0.0–1.0)." value={settings.ollama_top_p ?? 0.9} min={0.0} max={1.0} step={0.05} onChange={v => setSettings({ ...settings, ollama_top_p: v })} />
                    <SettingCard title="Top-K" description="Top-K token sampling filter (default: 40)." value={settings.ollama_top_k ?? 40} min={0} max={200} step={5} onChange={v => setSettings({ ...settings, ollama_top_k: Math.round(v) })} />
                    <SettingCard title="Repeat Penalty" description="Penalty for repeating identical tokens." value={settings.ollama_repeat_penalty ?? 1.15} min={0.0} max={3.0} step={0.05} onChange={v => setSettings({ ...settings, ollama_repeat_penalty: v })} />
                    <SettingInput label="Keep Alive Duration" val={settings.ollama_keep_alive ?? '5m'} onChange={v => setSettings({ ...settings, ollama_keep_alive: v })} />
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* 📚 TAB: RAG & VECTOR SEARCH */}
          {activeTab === 'rag' && settings && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem', maxWidth: 950, margin: '0 auto' }}>
              <div style={{ borderRadius: 12, border: '1.5px solid hsl(200,80%,50%/.3)', background: 'hsl(var(--card))', padding: '1.25rem' }}>
                <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: '1rem', color: 'hsl(var(--ink))', display: 'flex', alignItems: 'center', gap: 8 }}>
                  <Database size={16} style={{ color: 'hsl(200,80%,50%)' }} /> RAG Chunking &amp; Vector Search
                </h3>

                {/* Embedding Model selector */}
                {embeddingModels.length > 0 && (
                  <div style={{ marginBottom: '1.25rem', padding: '.85rem 1rem', borderRadius: 8, background: 'hsl(var(--paper)/.4)', border: '1px solid hsl(var(--border)/.4)' }}>
                    <div style={{ fontSize: '.8rem', fontWeight: 700, marginBottom: 4, color: 'hsl(var(--ink))' }}>Embedding Model</div>
                    <select
                      value={settings.embedding_model ?? 'Qwen3-Embedding-0.6B'}
                      onChange={e => setSettings({ ...settings, embedding_model: e.target.value })}
                      style={{ width: '100%', padding: '6px 10px', borderRadius: 6, border: '1px solid hsl(var(--border))', background: 'hsl(var(--card))', fontSize: '.78rem', color: 'hsl(var(--ink))', fontFamily: 'Inter' }}
                    >
                      {embeddingModels.map(m => (
                        <option key={m.id} value={m.id}>{m.name} ({m.description})</option>
                      ))}
                    </select>
                  </div>
                )}

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1rem' }}>
                  <SettingCard title="RAG Chunk Size" description="Token chunk size for document vectorization." value={settings.rag_chunk_size ?? 400} min={50} max={2000} step={50} onChange={v => setSettings({ ...settings, rag_chunk_size: Math.round(v) })} />
                  <SettingCard title="RAG Chunk Overlap" description="Overlapping tokens between consecutive chunks." value={settings.rag_chunk_overlap ?? 50} min={0} max={500} step={10} onChange={v => setSettings({ ...settings, rag_chunk_overlap: Math.round(v) })} />
                  <SettingCard title="Retrieval K (Global Context)" description="Global organizational documents retrieved per search." value={settings.rag_retrieval_k_global ?? 2} min={0} max={20} step={1} onChange={v => setSettings({ ...settings, rag_retrieval_k_global: Math.round(v) })} />
                  <SettingCard title="Retrieval K (Meeting Context)" description="Meeting supporting documents retrieved per search." value={settings.rag_retrieval_k_meeting ?? 3} min={0} max={20} step={1} onChange={v => setSettings({ ...settings, rag_retrieval_k_meeting: Math.round(v) })} />
                  <SettingCard title="Retrieval K (Transcript)" description="Audio transcript chunks retrieved per query." value={settings.rag_retrieval_k_transcript ?? 10} min={0} max={50} step={1} onChange={v => setSettings({ ...settings, rag_retrieval_k_transcript: Math.round(v) })} />
                  <SettingCard title="Score Cutoff Threshold" description="Minimum relative similarity score cutoff (0.00–0.50)." value={settings.rag_relative_score_cutoff ?? 0.01} min={0.0} max={0.5} step={0.01} onChange={v => setSettings({ ...settings, rag_relative_score_cutoff: v })} />
                </div>
              </div>
            </div>
          )}

          {/* 🎛️ TAB: VAD & AUDIO */}
          {activeTab === 'audio' && settings && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem', maxWidth: 950, margin: '0 auto' }}>

              {/* VAD Pipeline Mode Controls */}
              <div style={{ borderRadius: 12, border: '1.5px solid hsl(330,75%,55%/.3)', background: 'hsl(var(--card))', padding: '1.25rem' }}>
                <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: '.35rem', color: 'hsl(var(--ink))', display: 'flex', alignItems: 'center', gap: 8 }}>
                  <Volume2 size={16} style={{ color: 'hsl(330,75%,55%)' }} /> Voice Activity Detection (VAD) Pipeline Controls
                </h3>
                <div style={{ fontSize: '.75rem', color: 'hsl(var(--pencil))', marginBottom: '1rem' }}>
                  Enable or disable VAD independently for transcription, forced alignment, and adaptive energy filtering.
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '1rem' }}>
                  <SettingToggle label="Master Voice Activity Detection (VAD)" checked={settings.enable_vad ?? true} onChange={v => setSettings({ ...settings, enable_vad: v })} />
                  <SettingToggle label="Transcription VAD (Speech Region Post-Processing)" checked={settings.enable_transcription_vad ?? true} onChange={v => setSettings({ ...settings, enable_transcription_vad: v })} />
                  <SettingToggle label="Alignment VAD (Region-Chunked WhisperX Alignment)" checked={settings.enable_alignment_vad ?? true} onChange={v => setSettings({ ...settings, enable_alignment_vad: v })} />
                  <SettingToggle label="Adaptive Energy VAD (Quantile Thresholding)" checked={settings.enable_adaptive_vad ?? true} onChange={v => setSettings({ ...settings, enable_adaptive_vad: v })} />
                </div>
              </div>

              {/* VAD Thresholds & Timing Fine-Tuning */}
              <div style={{ borderRadius: 12, border: '1.5px solid hsl(var(--border)/.4)', background: 'hsl(var(--card))', padding: '1.25rem' }}>
                <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: '1rem', color: 'hsl(var(--ink))', display: 'flex', alignItems: 'center', gap: 8 }}>
                  <Sliders size={16} style={{ color: 'hsl(280,75%,60%)' }} /> VAD Thresholds &amp; Duration Parameters
                </h3>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1rem' }}>
                  <SettingCard title="Adaptive VAD Speech Threshold" description="Probability quantile threshold to classify audio as speech (0.01–0.99)." value={settings.vad_speech_threshold ?? 0.15} min={0.01} max={0.99} step={0.01} onChange={v => setSettings({ ...settings, vad_speech_threshold: v })} />
                  <SettingCard title="Adaptive VAD Silence Threshold" description="Probability threshold below which frames are classified as silence." value={settings.vad_silence_threshold ?? 0.10} min={0.01} max={0.99} step={0.01} onChange={v => setSettings({ ...settings, vad_silence_threshold: v })} />
                  <SettingCard title="Min Speech Duration (ms)" description="Minimum speech segment duration in milliseconds (default 250ms)." value={settings.vad_min_speech_ms ?? 250} min={50} max={2000} step={50} onChange={v => setSettings({ ...settings, vad_min_speech_ms: Math.round(v) })} />
                  <SettingCard title="Min Silence Duration (ms)" description="Minimum silence duration between speech bursts (default 400ms)." value={settings.vad_min_silence_ms ?? 400} min={50} max={3000} step={50} onChange={v => setSettings({ ...settings, vad_min_silence_ms: Math.round(v) })} />
                </div>
              </div>

              {/* Speech Padding, Merging & Low-Volume Recovery */}
              <div style={{ borderRadius: 12, border: '1.5px solid hsl(var(--border)/.4)', background: 'hsl(var(--card))', padding: '1.25rem' }}>
                <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: '1rem', color: 'hsl(var(--ink))', display: 'flex', alignItems: 'center', gap: 8 }}>
                  <Activity size={16} style={{ color: 'hsl(205,90%,55%)' }} /> Segment Padding, Merging &amp; Quiet Region Recovery
                </h3>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '1rem' }}>
                  <SettingToggle label="Enable Speech Segment Padding" checked={settings.enable_speech_padding ?? true} onChange={v => setSettings({ ...settings, enable_speech_padding: v })} />
                  <SettingCard title="Speech Padding (ms)" description="Padding added before &amp; after speech segment boundaries." value={settings.speech_pad_ms ?? 400} min={0} max={2000} step={50} onChange={v => setSettings({ ...settings, speech_pad_ms: Math.round(v) })} />
                  <SettingToggle label="Enable Speech Segment Merging" checked={settings.enable_speech_segment_merging ?? true} onChange={v => setSettings({ ...settings, enable_speech_segment_merging: v })} />
                  <SettingCard title="Max Merge Silence (ms)" description="Merges speech segments separated by less than this silence duration." value={settings.max_merge_silence_ms ?? 500} min={0} max={5000} step={50} onChange={v => setSettings({ ...settings, max_merge_silence_ms: Math.round(v) })} />
                  <SettingToggle label="Low-Volume Recovery Pass" checked={settings.enable_low_volume_recovery ?? true} onChange={v => setSettings({ ...settings, enable_low_volume_recovery: v })} />
                  <SettingCard title="Recovery Energy Threshold (dBFS)" description="Energy threshold to detect and re-process quiet speech regions." value={settings.recovery_energy_threshold ?? -45.0} min={-80.0} max={0.0} step={0.5} onChange={v => setSettings({ ...settings, recovery_energy_threshold: v })} />
                </div>
              </div>

              {/* Normalization & Whisper Batching */}
              <div style={{ borderRadius: 12, border: '1.5px solid hsl(var(--border)/.4)', background: 'hsl(var(--card))', padding: '1.25rem' }}>
                <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: '1rem', color: 'hsl(var(--ink))', display: 'flex', alignItems: 'center', gap: 8 }}>
                  <Zap size={16} style={{ color: 'hsl(140,70%,45%)' }} /> Audio Normalization &amp; Whisper Performance
                </h3>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '1rem' }}>
                  <SettingToggle label="Audio Normalization" checked={settings.enable_audio_normalization ?? true} onChange={v => setSettings({ ...settings, enable_audio_normalization: v })} />
                  <SettingCard title="Target Normalization dBFS" description="Target dBFS signal level for low-volume audio (default -3.0 dBFS)." value={settings.norm_target_dbfs ?? -3.0} min={-30.0} max={0.0} step={0.5} onChange={v => setSettings({ ...settings, norm_target_dbfs: v })} />
                  <SettingCard title="Whisper Batch Size" description="Batch size for Whisper speech recognition (1–32). Higher values increase throughput but use more GPU VRAM." value={settings.whisper_batch_size ?? 8} min={1} max={32} step={1} onChange={v => setSettings({ ...settings, whisper_batch_size: Math.round(v) })} />
                </div>
              </div>

              {/* Audio Validation & Pre-Check Settings */}
              <div style={{ borderRadius: 12, border: '1.5px solid hsl(205,90%,55%/.3)', background: 'hsl(var(--card))', padding: '1.25rem' }}>
                <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: '.35rem', color: 'hsl(var(--ink))', display: 'flex', alignItems: 'center', gap: 8 }}>
                  <Shield size={16} style={{ color: 'hsl(205,90%,55%)' }} /> Audio Upload &amp; Recording Validation
                </h3>
                <div style={{ fontSize: '.75rem', color: 'hsl(var(--pencil))', marginBottom: '1rem' }}>
                  Validate uploaded files &amp; live recordings for audio duration and signal amplitude thresholds before queuing pipelines.
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '1rem' }}>
                  <SettingToggle
                    label="Enable Audio Validation Pre-Check"
                    checked={settings.enable_audio_validation ?? true}
                    onChange={v => setSettings({ ...settings, enable_audio_validation: v })}
                  />
                  <SettingCard
                    title="Minimum Audio Duration (secs)"
                    description="Rejects audio files shorter than this minimum threshold (e.g. 2.0s)."
                    value={settings.min_audio_duration_seconds ?? 2.0}
                    min={0.1} max={30.0} step={0.5}
                    onChange={v => setSettings({ ...settings, min_audio_duration_seconds: v })}
                  />
                  <SettingCard
                    title="Minimum Audio RMS Threshold"
                    description="Minimum peak 1-second RMS signal strength required (default 0.003)."
                    value={settings.min_audio_rms_threshold ?? 0.003}
                    min={0.0001} max={0.05} step={0.0005}
                    onChange={v => setSettings({ ...settings, min_audio_rms_threshold: v })}
                  />
                </div>
              </div>
            </div>
          )}

          {/* 📊 TAB: TASK TOKEN LIMITS */}
          {activeTab === 'tokens' && settings && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem', maxWidth: 1000, margin: '0 auto' }}>
              <div style={{ borderRadius: 12, border: '1.5px solid hsl(250,70%,60%/.3)', background: 'hsl(var(--card))', padding: '1.25rem' }}>
                <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: '1rem', color: 'hsl(var(--ink))', display: 'flex', alignItems: 'center', gap: 8 }}>
                  <SlidersHorizontal size={16} style={{ color: 'hsl(250,70%,60%)' }} /> Maximum Token Output Limits
                </h3>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '.85rem' }}>
                  <TokenInput label="MoM Final Generation" val={settings.max_tokens_mom ?? 1500} onChange={v => setSettings({ ...settings, max_tokens_mom: v })} />
                  <TokenInput label="MoM Section Merge" val={settings.max_tokens_mom_merge ?? 3072} onChange={v => setSettings({ ...settings, max_tokens_mom_merge: v })} />
                  <TokenInput label="Raw MoM → Final MoM" val={settings.max_tokens_raw_mom_to_mom ?? 3000} onChange={v => setSettings({ ...settings, max_tokens_raw_mom_to_mom: v })} />
                  <TokenInput label="Raw MoM Extraction" val={settings.max_tokens_raw_mom_extraction ?? 1024} onChange={v => setSettings({ ...settings, max_tokens_raw_mom_extraction: v })} />
                  <TokenInput label="Executive Summary" val={settings.max_tokens_executive_summary ?? 700} onChange={v => setSettings({ ...settings, max_tokens_executive_summary: v })} />
                  <TokenInput label="Detailed Summary" val={settings.max_tokens_detailed_summary ?? 3000} onChange={v => setSettings({ ...settings, max_tokens_detailed_summary: v })} />
                  <TokenInput label="Short Summary" val={settings.max_tokens_short_summary ?? 120} onChange={v => setSettings({ ...settings, max_tokens_short_summary: v })} />
                  <TokenInput label="Key Points" val={settings.max_tokens_key_points ?? 1028} onChange={v => setSettings({ ...settings, max_tokens_key_points: v })} />
                  <TokenInput label="Action Items" val={settings.max_tokens_action_items ?? 1028} onChange={v => setSettings({ ...settings, max_tokens_action_items: v })} />
                  <TokenInput label="Key Decisions" val={settings.max_tokens_key_decisions ?? 1028} onChange={v => setSettings({ ...settings, max_tokens_key_decisions: v })} />
                  <TokenInput label="Collection Chat Response" val={settings.max_tokens_collection_chat ?? 1500} onChange={v => setSettings({ ...settings, max_tokens_collection_chat: v })} />
                  <TokenInput label="Vocab Extractor" val={settings.max_tokens_vocab_extractor ?? 512} onChange={v => setSettings({ ...settings, max_tokens_vocab_extractor: v })} />
                </div>
              </div>
            </div>
          )}

        </div>
      </div>
    </div>
  )
}

// Helper components for settings controls
function SettingCard({ title, description, value, min, max, step, onChange }: { title: string; description: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void }) {
  return (
    <div style={{ padding: '.85rem 1rem', borderRadius: 8, background: 'hsl(var(--paper)/.4)', border: '1px solid hsl(var(--border)/.4)', display: 'flex', flexDirection: 'column', gap: 6 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span style={{ fontSize: '.78rem', fontWeight: 700, color: 'hsl(var(--ink))' }}>{title}</span>
        <span style={{ fontSize: '.76rem', fontWeight: 700, color: 'hsl(var(--accent))', background: 'hsl(var(--accent)/.1)', padding: '1px 6px', borderRadius: 4, fontFamily: 'JetBrains Mono' }}>{value}</span>
      </div>
      <div style={{ fontSize: '.68rem', color: 'hsl(var(--pencil))', lineHeight: 1.35 }}>{description}</div>
      <input
        type="range"
        min={min} max={max} step={step} value={value}
        onChange={e => onChange(parseFloat(e.target.value))}
        style={{ width: '100%', accentColor: 'hsl(var(--accent))', cursor: 'pointer', marginTop: 4 }}
      />
    </div>
  )
}

function SettingInput({ label, val, onChange }: { label: string; val: string; onChange: (v: string) => void }) {
  return (
    <div style={{ padding: '.85rem 1rem', borderRadius: 8, background: 'hsl(var(--paper)/.4)', border: '1px solid hsl(var(--border)/.4)', display: 'flex', flexDirection: 'column', gap: 4 }}>
      <span style={{ fontSize: '.78rem', fontWeight: 700, color: 'hsl(var(--ink))' }}>{label}</span>
      <input
        type="text"
        value={val}
        onChange={e => onChange(e.target.value)}
        style={{ width: '100%', padding: '5px 8px', borderRadius: 6, border: '1px solid hsl(var(--border))', background: 'hsl(var(--card))', fontSize: '.78rem', color: 'hsl(var(--ink))', fontFamily: 'Inter' }}
      />
    </div>
  )
}

function SettingToggle({ label, checked, onChange }: { label: string; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <div style={{ padding: '.85rem 1rem', borderRadius: 8, background: 'hsl(var(--paper)/.4)', border: '1px solid hsl(var(--border)/.4)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
      <span style={{ fontSize: '.78rem', fontWeight: 700, color: 'hsl(var(--ink))' }}>{label}</span>
      <input
        type="checkbox"
        checked={checked}
        onChange={e => onChange(e.target.checked)}
        style={{ width: 16, height: 16, cursor: 'pointer', accentColor: 'hsl(var(--accent))' }}
      />
    </div>
  )
}

function TokenInput({ label, val, onChange }: { label: string; val: number; onChange: (v: number) => void }) {
  return (
    <div style={{ padding: '.65rem .8rem', borderRadius: 8, background: 'hsl(var(--paper)/.4)', border: '1px solid hsl(var(--border)/.3)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
      <span style={{ fontSize: '.74rem', fontWeight: 600, color: 'hsl(var(--ink))' }}>{label}</span>
      <input
        type="number"
        value={val}
        onChange={e => onChange(parseInt(e.target.value) || 1024)}
        style={{ width: 75, padding: '3px 6px', borderRadius: 6, border: '1px solid hsl(var(--border))', background: 'hsl(var(--card))', fontSize: '.74rem', fontWeight: 700, color: 'hsl(var(--ink))', textAlign: 'right', fontFamily: 'JetBrains Mono' }}
      />
    </div>
  )
}
