import { useEffect, useState, useRef } from 'react'
import {
  Settings, Mic, Trash2, Pencil, Save, Loader, Sliders, Sparkles, User, CheckCircle,
  MessageSquare, RotateCcw, Upload, Download, FileText, Code, Cpu, Database, Volume2, Activity,
  Layers, Target, RefreshCw, Zap, Shield, Search, ChevronRight, SlidersHorizontal, Copy, Maximize2, Minimize2, Check,
  AlertTriangle, CheckCircle2, XCircle, Clock, Loader2, Info, Play, Pause, Plus, PlusCircle
} from 'lucide-react'
import api from '../api/client'
import { toast } from 'sonner'
import { useJobsStore } from '../store/jobs'
import { useProcessingStore } from '../store/processing'

interface ProfileRecording {
  id: string
  filename: string
  duration: number
  created_at: string
}

interface Profile {
  id: string
  label: string
  sample_count: number
  is_self: boolean
  created_at: string
  updated_at?: string
  has_ecapa?: boolean
  has_eres2net?: boolean
  has_audio?: boolean
  recordings?: ProfileRecording[]
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
  speaker_embedding_model?: string

  // Ollama settings
  ollama_num_ctx?: number
  ollama_dynamic_ctx?: boolean
  ollama_think?: boolean
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
  rom_min_similarity_threshold?: number | null
  rom_windows_per_batch?: number
  rom_parallel_window_processing?: number
  rom_separate_action_extraction?: boolean
  rom_action_generation_chunk_size?: number
  rom_pipeline_mode?: string

  // Whisper Settings
  whisper_batch_size?: number

  max_tokens_rom_discussion?: number
  max_tokens_rom_discussion_no_actions?: number
  max_tokens_rom_action_extraction?: number
  max_tokens_mom_extract_actions?: number
  max_tokens_stage1_json_repair?: number
  max_tokens_mom_action_regen?: number
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

  // Whisper & Parallel Pipeline Settings
  whisper_parallel_processing?: number
  whisper_parallel_chunk_minutes?: number
  parallel_transcription_diarization?: boolean

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

  // Missing Segment Recovery
  missing_transcript_recovery_enabled?: boolean
  missing_segment_min_duration_sec?: number
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

type SettingsTab = 'processing' | 'rom' | 'prompts' | 'voice' | 'llm' | 'rag' | 'audio' | 'tokens'

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

  // Voice Recording & Profile Management states
  const [playingRecKey, setPlayingRecKey] = useState<string | null>(null)
  const audioPlayerRef = useRef<HTMLAudioElement | null>(null)
  const [uploadingRecProfileId, setUploadingRecProfileId] = useState<string | null>(null)
  const [deletingRecId, setDeletingRecId] = useState<string | null>(null)
  const [showAddProfileModal, setShowAddProfileModal] = useState(false)
  const [newProfileLabel, setNewProfileLabel] = useState('')
  const [newProfileFile, setNewProfileFile] = useState<File | null>(null)
  const [addingProfile, setAddingProfile] = useState(false)
  const [addProfileError, setAddProfileError] = useState<string | null>(null)

  // Jobs & Processing states
  const jobs = useJobsStore((s) => s.jobs)
  const clearAllLocalJobs = useJobsStore((s) => s.clearAllLocalJobs)
  const reconcileJobs = useJobsStore((s) => s.reconcile)
  const updateJobStore = useJobsStore((s) => s.updateJob)
  const isProcessing = useProcessingStore((s) => s.isProcessing)
  const clearProcessing = useProcessingStore((s) => s.clearProcessing)

  const [refreshingJobs, setRefreshingJobs] = useState(false)
  const [showClearConfirmModal, setShowClearConfirmModal] = useState(false)
  const [cancellingJobId, setCancellingJobId] = useState<string | null>(null)
  const [copiedJobId, setCopiedJobId] = useState<string | null>(null)

  const activeJobsList = jobs.filter((j) => j.status === 'processing' || j.status === 'pending' || j.status === 'transcript_ready')

  const handleRefreshStatus = async () => {
    setRefreshingJobs(true)
    try {
      const res = await api.get('/audio/jobs')
      const backendJobs = (res.data.jobs || []).map((j: any) => ({
        jobId: j.job_id,
        filename: j.filename,
        status: j.status,
        stage: j.progress || null,
        startedAt: j.created_at,
        source: 'upload' as const,
      }))
      reconcileJobs(backendJobs)

      const hasActive = backendJobs.some((j: any) => j.status === 'pending' || j.status === 'processing')
      if (!hasActive) {
        clearProcessing()
      }
      toast.success('Job statuses refreshed and synchronized with backend.')
    } catch (err: any) {
      toast.error(err?.response?.data?.detail || 'Failed to refresh job statuses')
    } finally {
      setRefreshingJobs(false)
    }
  }

  const handleConfirmClearAllLocal = () => {
    clearProcessing()
    clearAllLocalJobs()
    setShowClearConfirmModal(false)
    toast.success('All local job statuses and processing flags cleared.')
  }

  const handleCancelJob = async (jobId: string, filename?: string) => {
    if (!window.confirm(`Are you sure you want to cancel processing for "${filename || jobId}"?`)) {
      return
    }
    setCancellingJobId(jobId)
    try {
      await api.post(`/audio/jobs/${jobId}/cancel`)
      updateJobStore(jobId, { status: 'cancelled', stage: 'Cancelled by user' })
      toast.success(`Job "${filename || jobId}" cancelled successfully.`)
      await handleRefreshStatus()
    } catch (err: any) {
      toast.error(err?.response?.data?.detail || 'Failed to cancel job')
    } finally {
      setCancellingJobId(null)
    }
  }

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

  useEffect(() => {
    return () => {
      if (audioPlayerRef.current) {
        audioPlayerRef.current.pause()
      }
    }
  }, [])

  const handlePlayRecording = (profileId: string, recordingId: string) => {
    const key = `${profileId}_${recordingId}`
    if (playingRecKey === key) {
      if (audioPlayerRef.current) {
        audioPlayerRef.current.pause()
      }
      setPlayingRecKey(null)
      return
    }

    if (audioPlayerRef.current) {
      audioPlayerRef.current.pause()
    }

    const audioUrl = `/voice/profiles/${profileId}/recordings/${recordingId}/audio`
    const audio = new Audio(audioUrl)
    audioPlayerRef.current = audio
    setPlayingRecKey(key)

    audio.play().catch(() => {
      toast.error('Failed to play recording audio')
      setPlayingRecKey(null)
    })

    audio.onended = () => {
      setPlayingRecKey(null)
    }
    audio.onerror = () => {
      toast.error('Could not load audio file')
      setPlayingRecKey(null)
    }
  }

  const handleRemoveRecording = async (profileId: string, recordingId: string) => {
    if (!confirm('Remove this voice recording?')) return
    setDeletingRecId(recordingId)
    try {
      await api.delete(`/voice/profiles/${profileId}/recordings/${recordingId}`)
      toast.success('Recording removed')
      await loadData()
    } catch (err: any) {
      toast.error(err?.response?.data?.detail || 'Failed to remove recording')
    } finally {
      setDeletingRecId(null)
    }
  }

  const handleAddMoreRecording = async (profileId: string, file: File) => {
    if (!file) return
    setUploadingRecProfileId(profileId)
    const formData = new FormData()
    formData.append('file', file)
    try {
      await api.post(`/voice/profiles/${profileId}/recordings`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      toast.success('Voice recording added with ECAPA & ERes2Net embeddings!')
      await loadData()
    } catch (err: any) {
      toast.error(err?.response?.data?.detail || 'Failed to upload voice recording')
    } finally {
      setUploadingRecProfileId(null)
    }
  }

  const handleCreateProfile = async () => {
    setAddProfileError(null)
    if (!newProfileLabel.trim()) {
      setAddProfileError('Speaker name is required.')
      return
    }
    if (!newProfileFile) {
      setAddProfileError('An audio file is required to create a voice profile. Please select an audio file.')
      return
    }

    setAddingProfile(true)
    try {
      const sampleData = new FormData()
      sampleData.append('file', newProfileFile)
      sampleData.append('label', newProfileLabel.trim())
      sampleData.append('sample_index', '0')

      const sRes = await api.post('/voice/sample', sampleData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      const filePath = sRes.data.file_path

      await api.post('/voice/add-profile', {
        label: newProfileLabel.trim(),
        file_paths: [filePath],
      })

      toast.success(`Voice profile "${newProfileLabel}" created with ECAPA & ERes2Net embeddings!`)
      setNewProfileLabel('')
      setNewProfileFile(null)
      setShowAddProfileModal(false)
      await loadData()
    } catch (err: any) {
      setAddProfileError(err?.response?.data?.detail || 'Failed to create voice profile')
    } finally {
      setAddingProfile(false)
    }
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
      setEditingTemplates(prev => ({ ...prev, [key]: template }))
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
      setEditingTemplates(prev => {
        const next = { ...prev }
        delete next[key]
        return next
      })
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
      setEditingTemplates({})
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
            { id: 'processing', label: 'Processing & Active Jobs', count: activeJobsList.length > 0 ? `${activeJobsList.length} Active` : 'Idle', icon: <Activity size={14} className={activeJobsList.length > 0 ? 'spin' : ''} />, color: activeJobsList.length > 0 ? 'hsl(35,90%,50%)' : 'hsl(160,80%,45%)' },
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

          {/* ⚡ TAB: PROCESSING & ACTIVE JOBS */}
          {activeTab === 'processing' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem', maxWidth: 1000, margin: '0 auto' }}>

              {/* Header Card & Control Buttons */}
              <div style={{ borderRadius: 12, border: '1.5px solid hsl(160,80%,45%/.3)', background: 'hsl(var(--card))', padding: '1.25rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '1rem', marginBottom: '1.15rem' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                    <div style={{
                      width: 38, height: 38, borderRadius: 10,
                      background: activeJobsList.length > 0 ? 'hsl(35,90%,50%/.15)' : 'hsl(140,70%,45%/.15)',
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      color: activeJobsList.length > 0 ? 'hsl(35,90%,50%)' : 'hsl(140,70%,45%)'
                    }}>
                      <Activity size={20} className={activeJobsList.length > 0 ? 'spin' : ''} />
                    </div>
                    <div>
                      <h3 style={{ fontSize: '1.05rem', fontWeight: 700, margin: 0, display: 'flex', alignItems: 'center', gap: 8 }}>
                        Background Processing & Job Manager
                        {activeJobsList.length > 0 ? (
                          <span style={{ fontSize: '.72rem', padding: '2px 8px', borderRadius: 12, background: 'hsl(35,90%,50%/.2)', color: 'hsl(35,90%,45%)', fontWeight: 700 }}>
                            {activeJobsList.length} Job{activeJobsList.length > 1 ? 's' : ''} Active
                          </span>
                        ) : (
                          <span style={{ fontSize: '.72rem', padding: '2px 8px', borderRadius: 12, background: 'hsl(140,70%,45%/.2)', color: 'hsl(140,70%,40%)', fontWeight: 700 }}>
                            System Idle
                          </span>
                        )}
                      </h3>
                      <div style={{ fontSize: '.76rem', color: 'hsl(var(--pencil))', marginTop: 2 }}>
                        Monitor active background worker jobs, re-sync status with backend, or clear unresolved frontend flags.
                      </div>
                    </div>
                  </div>

                  {/* Actions Buttons */}
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <button
                      onClick={handleRefreshStatus}
                      disabled={refreshingJobs}
                      className="btn secondary"
                      style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '.8rem', padding: '.45rem .9rem', borderRadius: 8 }}
                      title="Poll backend for active jobs and update local status"
                    >
                      <RefreshCw size={14} className={refreshingJobs ? 'spin' : ''} />
                      <span>{refreshingJobs ? 'Syncing...' : 'Refresh Status'}</span>
                    </button>

                    <button
                      onClick={() => setShowClearConfirmModal(true)}
                      className="btn danger-outline"
                      style={{
                        display: 'flex', alignItems: 'center', gap: 6, fontSize: '.8rem', padding: '.45rem .9rem', borderRadius: 8,
                        color: 'hsl(0,80%,60%)', border: '1.5px solid hsl(0,80%,60%/.3)', background: 'hsl(0,80%,60%/.08)', cursor: 'pointer'
                      }}
                      title="Clear local job state and processing overlay without cancelling backend tasks"
                    >
                      <Trash2 size={14} />
                      <span>Clear All Job Statuses</span>
                    </button>
                  </div>
                </div>

                {/* System Status Banner */}
                <div style={{
                  padding: '.85rem 1rem', borderRadius: 8,
                  background: activeJobsList.length > 0 ? 'hsl(35,90%,50%/.08)' : 'hsl(140,70%,45%/.08)',
                  border: `1px solid ${activeJobsList.length > 0 ? 'hsl(35,90%,50%/.25)' : 'hsl(140,70%,45%/.25)'}`,
                  display: 'flex', alignItems: 'center', gap: 10, fontSize: '.82rem'
                }}>
                  {activeJobsList.length > 0 ? (
                    <>
                      <Loader2 size={16} className="spin" style={{ color: 'hsl(35,90%,50%)', flexShrink: 0 }} />
                      <div>
                        <strong>Active Tasks Running:</strong> {activeJobsList.map(j => `"${j.filename || j.jobId}" (${j.stage || j.status})`).join(', ')}
                      </div>
                    </>
                  ) : (
                    <>
                      <CheckCircle2 size={16} style={{ color: 'hsl(140,70%,45%)', flexShrink: 0 }} />
                      <div>
                        <strong>No Backend Jobs Running:</strong> System is idle. If your screen is showing "Processing" anywhere, click <em>Clear All Job Statuses</em> to clear local flags.
                      </div>
                    </>
                  )}
                </div>
              </div>

              {/* Detailed Active Processes List */}
              <div style={{ borderRadius: 12, border: '1px solid hsl(var(--border)/.4)', background: 'hsl(var(--card))', padding: '1.25rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
                  <h4 style={{ fontSize: '.92rem', fontWeight: 700, margin: 0, display: 'flex', alignItems: 'center', gap: 8 }}>
                    <Activity size={16} style={{ color: 'hsl(var(--accent))' }} />
                    Active & Recent Background Processes ({jobs.length})
                  </h4>
                  {jobs.length > 0 && (
                    <span style={{ fontSize: '.72rem', color: 'hsl(var(--pencil))' }}>
                      Re-synced with backend status APIs
                    </span>
                  )}
                </div>

                {jobs.length === 0 ? (
                  <div style={{ padding: '2.5rem 1rem', textAlign: 'center', background: 'hsl(var(--muted)/.2)', borderRadius: 10, border: '1px dashed hsl(var(--border))' }}>
                    <CheckCircle2 size={36} style={{ color: 'hsl(140,70%,45%)', margin: '0 auto .75rem' }} />
                    <div style={{ fontSize: '.9rem', fontWeight: 700, color: 'hsl(var(--ink))' }}>No Job History or Active Processes</div>
                    <div style={{ fontSize: '.78rem', color: 'hsl(var(--pencil))', marginTop: 4, maxWidth: 460, margin: '4px auto 0' }}>
                      No active processing jobs are currently registered. Start a new recording or upload media to launch background tasks.
                    </div>
                  </div>
                ) : (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '.85rem' }}>
                    {jobs.map((job) => {
                      const isActive = job.status === 'processing' || job.status === 'pending' || job.status === 'transcript_ready'
                      const isCancelling = cancellingJobId === job.jobId
                      const isCopied = copiedJobId === job.jobId

                      return (
                        <div
                          key={job.jobId}
                          style={{
                            padding: '1rem 1.15rem', borderRadius: 10,
                            background: isActive ? 'hsl(var(--card))' : 'hsl(var(--muted)/.2)',
                            border: isActive ? '1.5px solid hsl(var(--accent)/.3)' : '1px solid hsl(var(--border)/.5)',
                            display: 'flex', flexDirection: 'column', gap: '.65rem',
                            transition: 'all .15s ease'
                          }}
                        >
                          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '.5rem' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                              <span style={{
                                fontSize: '.68rem', fontWeight: 800, padding: '2px 8px', borderRadius: 6,
                                background: job.source === 'record' ? 'hsl(0,85%,60%/.15)' : job.source === 'tab-audio' ? 'hsl(280,75%,60%/.15)' : 'hsl(205,90%,55%/.15)',
                                color: job.source === 'record' ? 'hsl(0,85%,55%)' : job.source === 'tab-audio' ? 'hsl(280,75%,60%)' : 'hsl(205,90%,55%)',
                                textTransform: 'uppercase', letterSpacing: '.5px'
                              }}>
                                {job.source}
                              </span>

                              <span style={{ fontSize: '.88rem', fontWeight: 700, color: 'hsl(var(--ink))' }}>
                                {job.filename || 'Recording Session'}
                              </span>

                              {/* Job ID Copy Pill */}
                              <button
                                onClick={() => {
                                  navigator.clipboard.writeText(job.jobId)
                                  setCopiedJobId(job.jobId)
                                  setTimeout(() => setCopiedJobId(null), 1500)
                                }}
                                style={{
                                  display: 'inline-flex', alignItems: 'center', gap: 4, padding: '2px 6px',
                                  borderRadius: 4, background: 'hsl(var(--muted)/.5)', border: 'none',
                                  fontSize: '.7rem', fontFamily: 'monospace', color: 'hsl(var(--pencil))', cursor: 'pointer'
                                }}
                                title="Click to copy full Job ID"
                              >
                                {isCopied ? <Check size={10} color="hsl(140,70%,45%)" /> : <Copy size={10} />}
                                {job.jobId.slice(0, 8)}...
                              </button>
                            </div>

                            {/* Status Pill & Cancel Action */}
                            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                              <span style={{
                                fontSize: '.72rem', fontWeight: 700, padding: '3px 10px', borderRadius: 12,
                                display: 'inline-flex', alignItems: 'center', gap: 5,
                                background: job.status === 'processing' ? 'hsl(205,90%,55%/.18)' :
                                            job.status === 'pending' ? 'hsl(35,90%,50%/.18)' :
                                            job.status === 'transcript_ready' ? 'hsl(180,80%,40%/.18)' :
                                            job.status === 'done' ? 'hsl(140,70%,45%/.18)' : 'hsl(0,80%,60%/.18)',
                                color: job.status === 'processing' ? 'hsl(205,90%,50%)' :
                                       job.status === 'pending' ? 'hsl(35,90%,45%)' :
                                       job.status === 'transcript_ready' ? 'hsl(180,80%,35%)' :
                                       job.status === 'done' ? 'hsl(140,70%,40%)' : 'hsl(0,80%,55%)'
                              }}>
                                {job.status === 'processing' && <Loader2 size={12} className="spin" />}
                                {job.status === 'done' && <CheckCircle2 size={12} />}
                                {job.status === 'cancelled' && <XCircle size={12} />}
                                {job.status.toUpperCase()}
                              </span>

                              {isActive && (
                                <button
                                  onClick={() => handleCancelJob(job.jobId, job.filename)}
                                  disabled={isCancelling}
                                  style={{
                                    display: 'inline-flex', alignItems: 'center', gap: 4,
                                    padding: '3px 9px', borderRadius: 6, fontSize: '.74rem', fontWeight: 600,
                                    color: 'hsl(0,80%,60%)', background: 'hsl(0,80%,60%/.1)',
                                    border: '1px solid hsl(0,80%,60%/.3)', cursor: 'pointer',
                                    transition: 'all .15s ease'
                                  }}
                                  title="Cancel and terminate this backend job"
                                >
                                  {isCancelling ? <Loader2 size={11} className="spin" /> : <XCircle size={11} />}
                                  {isCancelling ? 'Cancelling...' : 'Cancel Job'}
                                </button>
                              )}
                            </div>
                          </div>

                          {/* Stage / Progress Detail Line */}
                          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '.78rem', color: 'hsl(var(--pencil))' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                              <Activity size={13} style={{ color: 'hsl(var(--accent))' }} />
                              <span>Current Stage: <strong style={{ color: 'hsl(var(--ink))' }}>{job.stage || job.status}</strong></span>
                            </div>
                            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                              <Clock size={12} />
                              <span>Started: {job.startedAt ? new Date(job.startedAt).toLocaleTimeString() : 'N/A'}</span>
                            </div>
                          </div>
                        </div>
                      )
                    })}
                  </div>
                )}
              </div>

            </div>
          )}

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
                      title="Stage 2 Min Similarity Threshold"
                      description="Minimum cosine similarity threshold (0.50 - 0.98) required for context chunks. Chunks below this score are ignored."
                      value={settings.rom_min_similarity_threshold ?? 0.80}
                      min={0.50} max={0.98} step={0.01}
                      onChange={v => setSettings({ ...settings, rom_min_similarity_threshold: Number(v.toFixed(2)) })}
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
                    <SettingCard
                      title="Action Extraction Chunk Size (MOM)"
                      description="Number of Stage 2 discussion points passed per LLM call when generating action points for MoM (default 10)."
                      value={settings.rom_action_generation_chunk_size ?? 10}
                      min={1} max={50} step={1}
                      onChange={v => setSettings({ ...settings, rom_action_generation_chunk_size: Math.round(v) })}
                    />
                  </div>
                )}

                {/* Separate Action Extraction Toggle */}
                {settings && (
                  <div style={{ marginTop: '1rem', padding: '1rem 1.15rem', borderRadius: 10, border: '1.5px solid hsl(280,75%,60%/.2)', background: 'hsl(280,75%,60%/.04)', display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: '1.5rem' }}>
                    <div style={{ flex: 1 }}>
                      <div style={{ fontSize: '.88rem', fontWeight: 700, color: 'hsl(var(--ink))', marginBottom: 3 }}>Use Separate Action Point Extraction (Stage 1)</div>
                      <div style={{ fontSize: '.74rem', color: 'hsl(var(--pencil))', lineHeight: 1.5 }}>
                        When enabled, Stage 1 runs two parallel LLM calls per window: one focused exclusively on discussion points and one dedicated to extracting rich, self-contained action items. Action items are attached to the first discussion point in each window. Disable to use the standard single-call extraction (default).
                      </div>
                    </div>
                    <div style={{ flexShrink: 0, display: 'flex', alignItems: 'center', gap: 10 }}>
                      <span style={{ fontSize: '.74rem', color: 'hsl(var(--pencil))' }}>{settings.rom_separate_action_extraction ? 'Enabled' : 'Disabled'}</span>
                      <button
                        onClick={() => setSettings({ ...settings, rom_separate_action_extraction: !settings.rom_separate_action_extraction })}
                        style={{
                          width: 44, height: 24, borderRadius: 12,
                          background: settings.rom_separate_action_extraction ? 'hsl(280,75%,60%)' : 'hsl(var(--muted))',
                          border: 'none', cursor: 'pointer', position: 'relative', transition: 'background .2s',
                          flexShrink: 0
                        }}
                        aria-label="Toggle Separate Action Point Extraction"
                      >
                        <span style={{
                          position: 'absolute', top: 3,
                          left: settings.rom_separate_action_extraction ? 23 : 3,
                          width: 18, height: 18, borderRadius: '50%', background: '#fff',
                          transition: 'left .2s', boxShadow: '0 1px 3px rgba(0,0,0,.25)'
                        }} />
                      </button>
                    </div>
                  </div>
                )}

                {/* ROM Pipeline Variant Selector */}
                {settings && (
                  <div style={{ marginTop: '1rem', padding: '1rem 1.15rem', borderRadius: 10, border: '1.5px solid hsl(205,85%,55%/.25)', background: 'hsl(205,85%,55%/.04)' }}>
                    <div style={{ fontSize: '.88rem', fontWeight: 700, color: 'hsl(var(--ink))', marginBottom: 4, display: 'flex', alignItems: 'center', gap: 6 }}>
                      <Cpu size={14} style={{ color: 'hsl(205,85%,55%)' }} /> ROM Pipeline Variant
                    </div>
                    <div style={{ fontSize: '.74rem', color: 'hsl(var(--pencil))', lineHeight: 1.5, marginBottom: '.75rem' }}>
                      <strong>Base Mode</strong> always uses the default prompts for all ROM stages.<br />
                      <strong>DSPy Mode</strong> routes Stage 2 enhancement through trained DSPy variants when one is active (falls back to base if none exist).
                    </div>
                    <div style={{ display: 'flex', gap: '.6rem' }}>
                      {(['base', 'dspy'] as const).map(mode => {
                        const isActive = (settings.rom_pipeline_mode || 'base') === mode
                        const label = mode === 'base' ? '⚡ Base Mode' : '🧠 DSPy Mode'
                        const desc = mode === 'base' ? 'Standard prompts (default)' : 'Trained DSPy variants'
                        return (
                          <button
                            key={mode}
                            onClick={() => setSettings({ ...settings, rom_pipeline_mode: mode })}
                            style={{
                              flex: 1, padding: '.55rem .75rem', borderRadius: 8,
                              border: isActive ? '2px solid hsl(205,85%,55%)' : '1.5px solid hsl(var(--border)/.5)',
                              background: isActive ? 'hsl(205,85%,55%/.12)' : 'hsl(var(--muted)/.2)',
                              color: isActive ? 'hsl(205,85%,45%)' : 'hsl(var(--ink))',
                              cursor: 'pointer', textAlign: 'left', transition: 'all .15s',
                              fontFamily: 'Inter'
                            }}
                          >
                            <div style={{ fontSize: '.8rem', fontWeight: 700 }}>{label}</div>
                            <div style={{ fontSize: '.68rem', color: isActive ? 'hsl(205,85%,50%)' : 'hsl(var(--pencil))', marginTop: 2 }}>{desc}</div>
                          </button>
                        )
                      })}
                    </div>
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
                    <TokenInput label="Stage 1 Discussion Only (No Actions)" val={settings.max_tokens_rom_discussion_no_actions ?? 4096} onChange={v => setSettings({ ...settings, max_tokens_rom_discussion_no_actions: v })} />
                    <TokenInput label="Stage 1 Separate Action Extraction" val={settings.max_tokens_rom_action_extraction ?? 2048} onChange={v => setSettings({ ...settings, max_tokens_rom_action_extraction: v })} />
                    <TokenInput label="MoM Extract Action Points from ROM" val={settings.max_tokens_mom_extract_actions ?? 4096} onChange={v => setSettings({ ...settings, max_tokens_mom_extract_actions: v })} />
                    <TokenInput label="Stage 1 JSON Repair" val={settings.max_tokens_stage1_json_repair ?? 5048} onChange={v => setSettings({ ...settings, max_tokens_stage1_json_repair: v })} />
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
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
                    {/* Speaker Embedding Model Selector */}
                    <div style={{
                      padding: '1rem 1.25rem',
                      borderRadius: 10,
                      background: 'hsl(var(--paper)/.5)',
                      border: '1px solid hsl(var(--border)/.6)',
                      display: 'flex',
                      flexDirection: 'column',
                      gap: 8,
                    }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 8 }}>
                        <div>
                          <div style={{ fontSize: '.84rem', fontWeight: 700, color: 'hsl(var(--ink))', display: 'flex', alignItems: 'center', gap: 6 }}>
                            Speaker Embedding Model
                            <span style={{
                              fontSize: '.68rem',
                              fontWeight: 700,
                              padding: '2px 8px',
                              borderRadius: 12,
                              background: (settings.speaker_embedding_model || 'ecapa') === 'ecapa' ? 'hsl(205,90%,55%/.15)' : 'hsl(142,70%,45%/.15)',
                              color: (settings.speaker_embedding_model || 'ecapa') === 'ecapa' ? 'hsl(205,90%,55%)' : 'hsl(142,70%,45%)',
                            }}>
                              {(settings.speaker_embedding_model || 'ecapa') === 'ecapa' ? '192-d ECAPA-TDNN' : '293-d ERes2Net'}
                            </span>
                          </div>
                          <div style={{ fontSize: '.72rem', color: 'hsl(var(--pencil))', marginTop: 2 }}>
                            Neural model used for voice embedding extraction and speaker recognition.
                          </div>
                        </div>

                        <select
                          id="speaker-embedding-model-select"
                          value={settings.speaker_embedding_model || 'ecapa'}
                          onChange={e => {
                            const newModel = e.target.value
                            setSettings({
                              ...settings,
                              speaker_embedding_model: newModel,
                            })
                          }}
                          style={{
                            padding: '6px 14px',
                            borderRadius: 8,
                            border: '1.5px solid hsl(var(--border))',
                            background: 'hsl(var(--card))',
                            fontSize: '.82rem',
                            fontWeight: 600,
                            color: 'hsl(var(--ink))',
                            cursor: 'pointer',
                            outline: 'none',
                            fontFamily: 'Inter, sans-serif',
                            minWidth: 220,
                          }}
                        >
                          <option value="ecapa">ECAPA-TDNN (192-d, Default)</option>
                          <option value="eres2net_large">ERes2Net-Large (293-d, 3D-Speaker)</option>
                        </select>
                      </div>

                      <div style={{ fontSize: '.68rem', color: 'hsl(var(--pencil))', borderTop: '1px solid hsl(var(--border)/.3)', paddingTop: 6, display: 'flex', gap: 16, flexWrap: 'wrap' }}>
                        <span>• <strong>ECAPA-TDNN:</strong> Fast &amp; robust, recommended similarity threshold 0.72.</span>
                        <span>• <strong>ERes2Net-Large:</strong> High-precision 3D-Speaker model, recommended similarity threshold 0.55.</span>
                      </div>
                    </div>

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
                  </div>
                )}
              </div>

              {/* Voice Profiles List & Management */}
              <div style={{ borderRadius: 12, border: '1.5px solid hsl(var(--border)/.4)', background: 'hsl(var(--card))', padding: '1.25rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem', flexWrap: 'wrap', gap: '0.75rem' }}>
                  <h3 style={{ fontSize: '1rem', fontWeight: 700, margin: 0, color: 'hsl(var(--ink))', display: 'flex', alignItems: 'center', gap: 8 }}>
                    <Mic size={16} style={{ color: 'hsl(205,90%,55%)' }} /> Voice Profiles ({profiles.length})
                  </h3>
                  <button
                    onClick={() => { setShowAddProfileModal(true); setAddProfileError(null); }}
                    style={{
                      display: 'inline-flex', alignItems: 'center', gap: 6,
                      padding: '.4rem .85rem', borderRadius: 8,
                      background: 'hsl(205,90%,55%)', color: '#fff',
                      border: 'none', fontSize: '.78rem', fontWeight: 700,
                      cursor: 'pointer', transition: 'all .15s'
                    }}
                  >
                    <Plus size={13} /> Add Voice Profile
                  </button>
                </div>

                {/* Add Voice Profile Modal / Form */}
                {showAddProfileModal && (
                  <div style={{
                    marginBottom: '1.25rem', padding: '1.15rem', borderRadius: 10,
                    background: 'hsl(var(--paper)/.7)', border: '1.5px solid hsl(205,90%,55%/.4)',
                    boxShadow: '0 4px 16px rgba(0,0,0,0.1)'
                  }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '.85rem' }}>
                      <div style={{ fontWeight: 700, fontSize: '.9rem', color: 'hsl(var(--ink))', display: 'flex', alignItems: 'center', gap: 6 }}>
                        <User size={15} style={{ color: 'hsl(205,90%,55%)' }} /> Create Speaker Voice Profile
                      </div>
                      <button
                        onClick={() => { setShowAddProfileModal(false); setAddProfileError(null); }}
                        style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'hsl(var(--pencil))', fontSize: '.9rem' }}
                      >
                        ✕
                      </button>
                    </div>

                    {addProfileError && (
                      <div style={{
                        padding: '.55rem .85rem', borderRadius: 6, marginBottom: '.85rem',
                        background: 'hsl(0,80%,50%/.1)', border: '1px solid hsl(0,80%,50%/.3)',
                        fontSize: '.76rem', color: 'hsl(0,80%,45%)', display: 'flex', alignItems: 'center', gap: 6
                      }}>
                        <AlertTriangle size={13} style={{ flexShrink: 0 }} />
                        <span>{addProfileError}</span>
                      </div>
                    )}

                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1rem', marginBottom: '1rem' }}>
                      <div>
                        <label style={{ display: 'block', fontSize: '.74rem', fontWeight: 600, color: 'hsl(var(--pencil))', marginBottom: 4 }}>
                          Speaker Name / Label *
                        </label>
                        <input
                          type="text"
                          placeholder="e.g. Vikas, Alice, Team Lead"
                          value={newProfileLabel}
                          onChange={(e) => setNewProfileLabel(e.target.value)}
                          style={{
                            width: '100%', padding: '.45rem .75rem', borderRadius: 6,
                            border: '1px solid hsl(var(--border))', background: 'hsl(var(--card))',
                            color: 'hsl(var(--ink))', fontSize: '.82rem'
                          }}
                        />
                      </div>

                      <div>
                        <label style={{ display: 'block', fontSize: '.74rem', fontWeight: 600, color: 'hsl(var(--pencil))', marginBottom: 4 }}>
                          Voice Audio Recording (Required) *
                        </label>
                        <input
                          type="file"
                          accept="audio/*"
                          onChange={(e) => {
                            const f = e.target.files?.[0] || null
                            setNewProfileFile(f)
                            if (f) setAddProfileError(null)
                          }}
                          style={{
                            width: '100%', padding: '.35rem', borderRadius: 6,
                            border: '1px solid hsl(var(--border))', background: 'hsl(var(--card))',
                            color: 'hsl(var(--ink))', fontSize: '.78rem'
                          }}
                        />
                      </div>
                    </div>

                    <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
                      <button
                        onClick={() => { setShowAddProfileModal(false); setAddProfileError(null); }}
                        style={{
                          padding: '.4rem .85rem', borderRadius: 6, border: '1px solid hsl(var(--border))',
                          background: 'transparent', color: 'hsl(var(--pencil))', fontSize: '.76rem', cursor: 'pointer'
                        }}
                      >
                        Cancel
                      </button>
                      <button
                        onClick={handleCreateProfile}
                        disabled={addingProfile}
                        style={{
                          display: 'inline-flex', alignItems: 'center', gap: 6,
                          padding: '.4rem 1rem', borderRadius: 6, border: 'none',
                          background: 'hsl(205,90%,55%)', color: '#fff', fontSize: '.76rem',
                          fontWeight: 700, cursor: addingProfile ? 'not-allowed' : 'pointer'
                        }}
                      >
                        {addingProfile ? <Loader size={12} className="spin" /> : <Save size={12} />}
                        {addingProfile ? 'Processing & Generating Dual Embeddings...' : 'Save Voice Profile'}
                      </button>
                    </div>
                  </div>
                )}

                {loadingProfiles ? (
                  <div style={{ textAlign: 'center', padding: '2rem' }}><Loader size={20} className="spin" /></div>
                ) : profiles.length === 0 ? (
                  <div style={{ textAlign: 'center', padding: '2rem', color: 'hsl(var(--pencil))' }}>No voice profiles saved yet. Click "Add Voice Profile" above to create one.</div>
                ) : (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                    {profiles.map((p, idx) => {
                      const color = PROFILE_COLORS[idx % PROFILE_COLORS.length]
                      return (
                        <div
                          key={p.id}
                          style={{
                            padding: '.85rem 1.15rem', borderRadius: 10,
                            background: 'hsl(var(--paper)/.4)', borderLeft: `4px solid ${color}`,
                            border: '1px solid hsl(var(--border)/.3)'
                          }}
                        >
                          {/* Profile Header Row */}
                          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 8 }}>
                            <div>
                              {editingId === p.id ? (
                                <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                                  <input
                                    value={editLabel}
                                    onChange={e => setEditLabel(e.target.value)}
                                    style={{ padding: '3px 7px', borderRadius: 6, fontSize: '.8rem', border: '1px solid hsl(var(--border))' }}
                                  />
                                  <button
                                    onClick={() => handleRename(p.id)}
                                    disabled={savingLabel}
                                    style={{ padding: '3px 8px', borderRadius: 6, background: 'hsl(var(--accent))', color: 'white', border: 'none', fontSize: '.74rem', cursor: 'pointer' }}
                                  >
                                    {savingLabel ? 'Saving' : 'Save'}
                                  </button>
                                </div>
                              ) : (
                                <div style={{ fontWeight: 700, fontSize: '.9rem', color: 'hsl(var(--ink))', display: 'flex', alignItems: 'center', gap: 6 }}>
                                  <span>{p.label}</span>
                                  {p.is_self && (
                                    <span style={{ fontSize: '.65rem', background: 'hsl(205,90%,55%/.15)', color: 'hsl(205,90%,55%)', padding: '1px 6px', borderRadius: 4 }}>
                                      You
                                    </span>
                                  )}
                                </div>
                              )}
                              <div style={{ fontSize: '.71rem', color: 'hsl(var(--pencil))', marginTop: 2 }}>
                                {p.sample_count} voice sample{p.sample_count !== 1 ? 's' : ''} enrolled
                              </div>
                            </div>

                            {/* Embedding Status Tags & Actions */}
                            <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                              <div style={{ display: 'flex', gap: 5 }}>
                                {/* ECAPA tag */}
                                <span style={{
                                  fontSize: '.66rem', fontWeight: 700, padding: '2px 7px', borderRadius: 6,
                                  background: p.has_ecapa ? 'hsl(140,70%,45%/.15)' : 'hsl(var(--muted)/.6)',
                                  color: p.has_ecapa ? 'hsl(140,70%,40%)' : 'hsl(var(--pencil))',
                                  border: `1px solid ${p.has_ecapa ? 'hsl(140,70%,45%/.3)' : 'hsl(var(--border)/.4)'}`
                                }}>
                                  ECAPA {p.has_ecapa ? '✓' : '✗'}
                                </span>

                                {/* ERes2Net-Large tag */}
                                <span style={{
                                  fontSize: '.66rem', fontWeight: 700, padding: '2px 7px', borderRadius: 6,
                                  background: p.has_eres2net ? 'hsl(280,75%,60%/.15)' : 'hsl(var(--muted)/.6)',
                                  color: p.has_eres2net ? 'hsl(280,75%,55%)' : 'hsl(var(--pencil))',
                                  border: `1px solid ${p.has_eres2net ? 'hsl(280,75%,60%/.3)' : 'hsl(var(--border)/.4)'}`
                                }}>
                                  ERes2Net-Large {p.has_eres2net ? '✓' : '✗'}
                                </span>
                              </div>

                              <div style={{ display: 'flex', gap: 4, marginLeft: 6 }}>
                                <button
                                  onClick={() => { setEditingId(p.id); setEditLabel(p.label) }}
                                  title="Rename speaker profile"
                                  style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'hsl(var(--pencil))', padding: '4px' }}
                                >
                                  <Pencil size={13} />
                                </button>
                                <button
                                  onClick={() => handleDelete(p.id)}
                                  disabled={deletingId === p.id}
                                  title="Delete profile"
                                  style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'hsl(var(--destructive))', padding: '4px' }}
                                >
                                  {deletingId === p.id ? <Loader size={13} className="spin" /> : <Trash2 size={13} />}
                                </button>
                              </div>
                            </div>
                          </div>

                          {/* Warning for legacy profiles without saved audio */}
                          {p.has_audio === false && (
                            <div style={{
                              marginTop: 10, padding: '6px 10px', borderRadius: 6,
                              background: 'hsl(35,90%,50%/.1)', border: '1px solid hsl(35,90%,50%/.3)',
                              fontSize: '.72rem', color: 'hsl(35,90%,45%)', display: 'flex', alignItems: 'center', gap: 6
                            }}>
                              <AlertTriangle size={13} style={{ flexShrink: 0 }} />
                              <span>⚠ Original voice recording unavailable — this profile was created before voice recordings were stored.</span>
                            </div>
                          )}

                          {/* Saved Recordings List */}
                          {p.recordings && p.recordings.length > 0 && (
                            <div style={{ marginTop: 10, display: 'flex', flexDirection: 'column', gap: 6 }}>
                              <div style={{ fontSize: '.72rem', fontWeight: 600, color: 'hsl(var(--pencil))', display: 'flex', alignItems: 'center', gap: 5 }}>
                                <Volume2 size={12} /> Saved Voice Recordings ({p.recordings.length}):
                              </div>
                              {p.recordings.map((rec, recIdx) => {
                                const isPlaying = playingRecKey === `${p.id}_${rec.id}`
                                return (
                                  <div
                                    key={rec.id || recIdx}
                                    style={{
                                      display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                                      padding: '5px 10px', borderRadius: 6, background: 'hsl(var(--card)/.7)',
                                      border: '1px solid hsl(var(--border)/.25)', fontSize: '.76rem'
                                    }}
                                  >
                                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, minWidth: 0 }}>
                                      <button
                                        onClick={() => handlePlayRecording(p.id, rec.id)}
                                        title={isPlaying ? "Pause audio" : "Play recording"}
                                        style={{
                                          width: 24, height: 24, borderRadius: '50%', border: 'none',
                                          background: isPlaying ? 'hsl(205,90%,55%)' : 'hsl(var(--muted)/.8)',
                                          color: isPlaying ? '#fff' : 'hsl(var(--ink))',
                                          display: 'flex', alignItems: 'center', justifyContent: 'center',
                                          cursor: 'pointer', flexShrink: 0
                                        }}
                                      >
                                        {isPlaying ? <Pause size={11} /> : <Play size={11} style={{ marginLeft: 2 }} />}
                                      </button>
                                      <div style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                        <span style={{ fontWeight: 600, color: 'hsl(var(--ink))' }}>{rec.filename}</span>
                                        {rec.duration > 0 && (
                                          <span style={{ marginLeft: 6, fontSize: '.68rem', color: 'hsl(var(--pencil))' }}>
                                            ({rec.duration}s)
                                          </span>
                                        )}
                                      </div>
                                    </div>

                                    <button
                                      onClick={() => handleRemoveRecording(p.id, rec.id)}
                                      disabled={deletingRecId === rec.id}
                                      title="Remove this recording"
                                      style={{
                                        background: 'none', border: 'none', cursor: 'pointer',
                                        color: 'hsl(var(--destructive))', padding: '4px',
                                        display: 'flex', alignItems: 'center'
                                      }}
                                    >
                                      {deletingRecId === rec.id ? <Loader size={11} className="spin" /> : <Trash2 size={11} />}
                                    </button>
                                  </div>
                                )
                              })}
                            </div>
                          )}

                          {/* Action Bar: Add More Recordings */}
                          <div style={{ marginTop: 10, display: 'flex', alignItems: 'center', gap: 8 }}>
                            <label style={{
                              display: 'inline-flex', alignItems: 'center', gap: 5,
                              padding: '3px 9px', borderRadius: 6,
                              background: 'hsl(var(--muted)/.5)', border: '1px solid hsl(var(--border)/.4)',
                              fontSize: '.72rem', fontWeight: 600, color: 'hsl(var(--ink))',
                              cursor: uploadingRecProfileId === p.id ? 'not-allowed' : 'pointer',
                              transition: 'all .15s'
                            }}>
                              {uploadingRecProfileId === p.id ? (
                                <>
                                  <Loader size={11} className="spin" />
                                  <span>Generating Embeddings...</span>
                                </>
                              ) : (
                                <>
                                  <Plus size={11} />
                                  <span>Add More Audio</span>
                                </>
                              )}
                              <input
                                type="file"
                                accept="audio/*"
                                disabled={uploadingRecProfileId === p.id}
                                style={{ display: 'none' }}
                                onChange={(e) => {
                                  const f = e.target.files?.[0]
                                  if (f) {
                                    handleAddMoreRecording(p.id, f)
                                    e.target.value = ''
                                  }
                                }}
                              />
                            </label>
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
                    <SettingToggle label="Model Thinking / Reasoning (think mode)" checked={settings.ollama_think ?? false} onChange={v => setSettings({ ...settings, ollama_think: v })} />
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

              {/* ⚡ Parallel Whisper Processing (Multi-Process Chunking) */}
              <div style={{ borderRadius: 12, border: '1.5px solid hsl(38 92% 50% / .35)', background: 'hsl(var(--card))', padding: '1.25rem' }}>
                <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: '.35rem', color: 'hsl(var(--ink))', display: 'flex', alignItems: 'center', gap: 8 }}>
                  <Zap size={16} style={{ color: 'hsl(38 92% 50%)' }} /> Parallel Whisper Transcription (Multi-Process Chunking)
                </h3>
                <div style={{ fontSize: '.75rem', color: 'hsl(var(--pencil))', marginBottom: '1rem' }}>
                  Splits long audio recordings into parallel chunks for faster multi-process Whisper transcription. 1 = Sequential processing (default). Values &gt; 1 run parallel subprocesses.
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '1rem' }}>
                  <SettingCard
                    title="Parallel Whisper Workers (WHISPER_PARALLEL_PROCESSING)"
                    description="Number of parallel Whisper transcription worker processes (1 = sequential / lowest VRAM). Higher values speed up transcription but multiply GPU VRAM usage."
                    value={settings.whisper_parallel_processing ?? 1}
                    min={1} max={8} step={1}
                    onChange={v => setSettings({ ...settings, whisper_parallel_processing: Math.round(v) })}
                  />
                  <SettingCard
                    title="Parallel Chunk Duration (WHISPER_PARALLEL_CHUNK_MINUTES)"
                    description="Duration in minutes of each audio chunk when parallel processing is active (default 10 min, range 1–60 min)."
                    value={settings.whisper_parallel_chunk_minutes ?? 10}
                    min={1} max={60} step={1}
                    onChange={v => setSettings({ ...settings, whisper_parallel_chunk_minutes: Math.round(v) })}
                  />
                </div>
              </div>

              {/* ⚡ Parallel Transcription + Diarization (High VRAM GPU) */}
              <div style={{ borderRadius: 12, border: '1.5px solid hsl(280 70% 55% / .35)', background: 'hsl(var(--card))', padding: '1.25rem' }}>
                <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: '.35rem', color: 'hsl(var(--ink))', display: 'flex', alignItems: 'center', gap: 8 }}>
                  <Zap size={16} style={{ color: 'hsl(280 70% 55%)' }} /> Parallel Transcription &amp; Diarization (High VRAM)
                </h3>
                <div style={{ fontSize: '.75rem', color: 'hsl(var(--pencil))', marginBottom: '1rem' }}>
                  Run Whisper transcription and Pyannote diarization simultaneously on the GPU. Requires 16GB+ VRAM. Default: OFF (sequential processing for lower VRAM usage).
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '1rem' }}>
                  <SettingToggle
                    label="Run Transcription & Diarization in Parallel"
                    checked={settings.parallel_transcription_diarization ?? false}
                    onChange={v => setSettings({ ...settings, parallel_transcription_diarization: v })}
                  />
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

              {/* Missing Segment Recovery */}
              <div style={{ borderRadius: 12, border: '1.5px solid hsl(280,75%,60%/.35)', background: 'hsl(var(--card))', padding: '1.25rem' }}>
                <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: '.35rem', color: 'hsl(var(--ink))', display: 'flex', alignItems: 'center', gap: 8 }}>
                  <Sparkles size={16} style={{ color: 'hsl(280,75%,60%)' }} /> Missing Segment Recovery
                </h3>
                <div style={{ fontSize: '.75rem', color: 'hsl(var(--pencil))', marginBottom: '1rem' }}>
                  Automatically detects untranscribed audio gaps and recovers missed speech using secondary Whisper passes with adaptive loudness enhancement, merging recovered segments without overlap.
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1rem' }}>
                  <SettingToggle
                    label="Enable Missing Segment Recovery"
                    checked={settings.missing_transcript_recovery_enabled ?? false}
                    onChange={v => setSettings({ ...settings, missing_transcript_recovery_enabled: v })}
                  />
                  <SettingCard
                    title="Minimum Segment Length Threshold (secs)"
                    description="Only missing audio gaps equal to or longer than this threshold (default: 2.0s) are evaluated for recovery. Shorter segments are ignored."
                    value={settings.missing_segment_min_duration_sec ?? 2.0}
                    min={0.1}
                    max={30.0}
                    step={0.1}
                    onChange={v => {
                      const num = typeof v === 'number' && !isNaN(v) && v > 0 ? Math.round(v * 10) / 10 : 2.0
                      setSettings({ ...settings, missing_segment_min_duration_sec: num })
                    }}
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
                  <TokenInput label="View MoM Action Point Regeneration" val={settings.max_tokens_mom_action_regen ?? 4048} onChange={v => setSettings({ ...settings, max_tokens_mom_action_regen: v })} />
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
                  <TokenInput label="ROM Stage 1 Discussion Extraction" val={settings.max_tokens_rom_discussion ?? 4096} onChange={v => setSettings({ ...settings, max_tokens_rom_discussion: v })} />
                  <TokenInput label="ROM Stage 1 Discussion (No Actions)" val={settings.max_tokens_rom_discussion_no_actions ?? 4096} onChange={v => setSettings({ ...settings, max_tokens_rom_discussion_no_actions: v })} />
                  <TokenInput label="ROM Stage 1 Separate Action Extraction" val={settings.max_tokens_rom_action_extraction ?? 2048} onChange={v => setSettings({ ...settings, max_tokens_rom_action_extraction: v })} />
                  <TokenInput label="MoM Extract Action Points from ROM" val={settings.max_tokens_mom_extract_actions ?? 4096} onChange={v => setSettings({ ...settings, max_tokens_mom_extract_actions: v })} />
                  <TokenInput label="ROM Stage 1 JSON Repair" val={settings.max_tokens_stage1_json_repair ?? 5048} onChange={v => setSettings({ ...settings, max_tokens_stage1_json_repair: v })} />
                  <TokenInput label="ROM Stage 2 Polish & Merge" val={settings.max_tokens_rom_polish ?? 4096} onChange={v => setSettings({ ...settings, max_tokens_rom_polish: v })} />
                  <TokenInput label="ROM Stage 2 Enhance Window" val={settings.max_tokens_rom_enhance_window ?? 4096} onChange={v => setSettings({ ...settings, max_tokens_rom_enhance_window: v })} />
                  <TokenInput label="ROM Stage 2 Deduplication" val={settings.max_tokens_rom_deduplicate ?? 2048} onChange={v => setSettings({ ...settings, max_tokens_rom_deduplicate: v })} />
                </div>
              </div>
            </div>
          )}

        </div>
      </div>

      {/* ── Confirmation Modal for Clear All Job Statuses ── */}
      {showClearConfirmModal && (
        <div style={{
          position: 'fixed', inset: 0, zIndex: 9999,
          background: 'rgba(0, 0, 0, 0.65)', backdropFilter: 'blur(4px)',
          display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '1rem'
        }}>
          <div style={{
            background: 'hsl(var(--card))', borderRadius: 14,
            border: '1.5px solid hsl(var(--border))', maxWidth: 480, width: '100%',
            padding: '1.5rem', boxShadow: '0 20px 50px rgba(0,0,0,0.3)',
            display: 'flex', flexDirection: 'column', gap: '1.15rem',
            animation: 'scaleUp 0.15s ease-out'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <div style={{
                width: 40, height: 40, borderRadius: 10,
                background: 'hsl(35,90%,50%/.15)', display: 'flex', alignItems: 'center', justifyContent: 'center',
                color: 'hsl(35,90%,50%)', flexShrink: 0
              }}>
                <AlertTriangle size={22} />
              </div>
              <div>
                <h3 style={{ fontSize: '1.05rem', fontWeight: 700, margin: 0, color: 'hsl(var(--ink))' }}>
                  Clear All Local Job Statuses?
                </h3>
                <div style={{ fontSize: '.76rem', color: 'hsl(var(--pencil))' }}>
                  Frontend-only recovery action
                </div>
              </div>
            </div>

            <div style={{
              fontSize: '.82rem', lineHeight: '1.45', color: 'hsl(var(--ink))',
              background: 'hsl(var(--muted)/.4)', padding: '.9rem 1rem', borderRadius: 8,
              border: '1px solid hsl(var(--border)/.6)'
            }}>
              This action will clear all locally stored job states, progress counters, active processing flags, and cached job data in your browser.
              <br /><br />
              <strong style={{ color: 'hsl(var(--accent))' }}>Note:</strong> This will <em>not</em> cancel any actual backend tasks running on the server or modify database records. Use this to recover if your UI is stuck showing "Processing" when no job is actually running.
            </div>

            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: 10 }}>
              <button
                onClick={() => setShowClearConfirmModal(false)}
                className="btn secondary"
                style={{ fontSize: '.82rem', padding: '.5rem 1rem', borderRadius: 8 }}
              >
                Cancel
              </button>
              <button
                onClick={handleConfirmClearAllLocal}
                className="btn danger"
                style={{
                  fontSize: '.82rem', padding: '.5rem 1rem', borderRadius: 8,
                  background: 'hsl(0,80%,55%)', color: '#fff', border: 'none', cursor: 'pointer',
                  fontWeight: 700, display: 'flex', alignItems: 'center', gap: 6
                }}
              >
                <Trash2 size={14} />
                Confirm Clear All
              </button>
            </div>
          </div>
        </div>
      )}
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
