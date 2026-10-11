import React, { useState, useEffect, useRef, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Bot, Send, Loader, Trash2, Copy, Check,
  MessageSquare, FileText, ChevronRight, Sparkles, X,
  Search, ChevronDown, Zap, RotateCcw,
  Plus, Edit3, PanelLeftClose, PanelLeftOpen,
  Square, Clock, MessageCircle, MoreVertical,
  UploadCloud, FileUp,
} from 'lucide-react'
import type { AIChatMeeting, AIChatMessage, AIChatSession } from '../types/recording'
import {
  streamAIChat, getAIChatMeetings, getAIChatHistory, clearAIChatHistory,
  getAIChatSessions, createAIChatSession, updateAIChatSession, deleteAIChatSession,
} from '../api/aiChat'
import MeetingResyncModal from '../components/MeetingResyncModal'
import { ManualMomLibraryModal } from '../components/ManualMomLibraryModal'

const STORAGE_KEY_SESSIONS = 'ai_chat_sessions_v2'
const STORAGE_KEY_ACTIVE = 'ai_chat_active_session_id'
const STORAGE_KEY_SIDEBAR = 'ai_chat_sidebar_open'

function generateId(): string {
  return 'chat-' + Date.now() + '-' + Math.random().toString(36).slice(2, 8)
}

function createEmptySession(title = 'New Chat', selectedMeetingId = ''): AIChatSession {
  const now = new Date().toISOString()
  return {
    id: generateId(),
    title,
    selected_meeting_id: selectedMeetingId,
    messages: [],
    currentSources: [],
    currentQueryType: '',
    created_at: now,
    updated_at: now,
  }
}

export default function AIChat() {
  const navigate = useNavigate()

  // ═══════ Sessions State ═══════
  const [sessions, setSessions] = useState<AIChatSession[]>([])
  const [activeSessionId, setActiveSessionId] = useState<string>('')
  const [sessionSearchQuery, setSessionSearchQuery] = useState('')
  const [editingSessionId, setEditingSessionId] = useState<string | null>(null)
  const [editingTitle, setEditingTitle] = useState('')
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null)

  // ═══════ UI Layout State ═══════
  const [sidebarOpen, setSidebarOpen] = useState<boolean>(() => {
    const saved = localStorage.getItem(STORAGE_KEY_SIDEBAR)
    return saved !== null ? saved === 'true' : true
  })
  const [rightPanelOpen, setRightPanelOpen] = useState<boolean>(true)

  // ═══════ Active Chat State ═══════
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(true)
  const [streaming, setStreaming] = useState(false)
  const [streamedText, setStreamedText] = useState('')

  // ═══════ Meeting Selector State ═══════
  const [availableMeetings, setAvailableMeetings] = useState<AIChatMeeting[]>([])
  const [meetingDropdownOpen, setMeetingDropdownOpen] = useState(false)
  const [meetingSearchQuery, setMeetingSearchQuery] = useState('')
  const [onlyStage2, setOnlyStage2] = useState(false)
  const [meetingFilterType, setMeetingFilterType] = useState<'all' | 'recorded' | 'manual'>('all')
  const [resyncModalOpen, setResyncModalOpen] = useState(false)
  const [manualMomModalOpen, setManualMomModalOpen] = useState(false)
  const dropdownRef = useRef<HTMLDivElement>(null)

  // ═══════ Misc State ═══════
  const [copied, setCopied] = useState<string | null>(null)
  const [confirmClearActive, setConfirmClearActive] = useState(false)

  // Refs
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const chatScrollRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const editTitleInputRef = useRef<HTMLInputElement>(null)
  const abortRef = useRef<AbortController | null>(null)

  // Save sidebar state to localStorage
  const handleToggleSidebar = () => {
    setSidebarOpen(prev => {
      const next = !prev
      localStorage.setItem(STORAGE_KEY_SIDEBAR, String(next))
      return next
    })
  }

  // Active session object helper
  const activeSession = sessions.find(s => s.id === activeSessionId) || sessions[0]
  const currentMessages = activeSession?.messages || []
  const selectedMeetingId = activeSession?.selected_meeting_id || ''
  const currentSources = activeSession?.currentSources || []
  const currentQueryType = activeSession?.currentQueryType || ''

  // ═══════ Bootstrap Sessions & Data ═══════
  useEffect(() => {
    let cancelled = false

    const init = async () => {
      try {
        // Load available meetings
        const meetings = await getAIChatMeetings().catch(() => [])
        if (cancelled) return
        setAvailableMeetings(meetings)

        // Try reading local sessions first for instant UI
        let initialSessions: AIChatSession[] = []
        try {
          const rawLocal = localStorage.getItem(STORAGE_KEY_SESSIONS)
          if (rawLocal) {
            const parsed = JSON.parse(rawLocal)
            if (Array.isArray(parsed) && parsed.length > 0) {
              initialSessions = parsed
            }
          }
        } catch {
          // ignore corrupted local storage
        }

        // Try fetching backend sessions
        let remoteSessions: any[] = []
        try {
          remoteSessions = await getAIChatSessions()
        } catch (err) {
          console.warn('Failed to fetch remote sessions:', err)
        }

        if (Array.isArray(remoteSessions) && remoteSessions.length > 0) {
          const localMap = new Map(initialSessions.map(ls => [ls.id, ls]))

          // Merge: remote sessions list is authoritative
          const merged: AIChatSession[] = remoteSessions.map(rs => {
            const localMatch = localMap.get(rs.id)
            return {
              id: rs.id,
              title: rs.title || localMatch?.title || 'Chat',
              selected_meeting_id: rs.selected_meeting_id || localMatch?.selected_meeting_id || '',
              messages: localMatch?.messages || [],
              currentSources: localMatch?.currentSources || [],
              currentQueryType: localMatch?.currentQueryType || '',
              created_at: rs.created_at || localMatch?.created_at || new Date().toISOString(),
              updated_at: rs.updated_at || localMatch?.updated_at || new Date().toISOString(),
            }
          })

          // Preserve any local sessions not yet indexed
          for (const localSess of initialSessions) {
            if (!remoteSessions.some(rs => rs.id === localSess.id)) {
              merged.push(localSess)
              createAIChatSession(localSess.title, localSess.selected_meeting_id, localSess.id).catch(() => {})
            }
          }

          initialSessions = merged
        } else if (initialSessions.length === 0) {
          const legacyHistory = await getAIChatHistory().catch(() => [])
          const defaultSession = createEmptySession('Default Chat')
          if (legacyHistory && legacyHistory.length > 0) {
            defaultSession.messages = legacyHistory
            const lastAssistant = [...legacyHistory].reverse().find(m => m.role === 'assistant')
            if (lastAssistant?.metadata?.meeting_sources) {
              defaultSession.currentSources = lastAssistant.metadata.meeting_sources
              defaultSession.currentQueryType = lastAssistant.metadata.query_type || ''
            }
          }
          initialSessions = [defaultSession]
          createAIChatSession(defaultSession.title, undefined, defaultSession.id).catch(() => {})
        }

        if (cancelled) return

        // Determine active session
        const savedActiveId = localStorage.getItem(STORAGE_KEY_ACTIVE)
        const targetActiveId = initialSessions.some(s => s.id === savedActiveId)
          ? (savedActiveId as string)
          : initialSessions[0].id

        setActiveSessionId(targetActiveId)

        // Always query the database for the active session's full chat history on reload
        try {
          const hist = await getAIChatHistory(targetActiveId)
          if (!cancelled && hist && hist.length > 0) {
            initialSessions = initialSessions.map(s => {
              if (s.id === targetActiveId) {
                const lastAssistant = [...hist].reverse().find(m => m.role === 'assistant')
                return {
                  ...s,
                  messages: hist,
                  currentSources: lastAssistant?.metadata?.meeting_sources || s.currentSources || [],
                  currentQueryType: lastAssistant?.metadata?.query_type || s.currentQueryType || '',
                }
              }
              return s
            })
          }
        } catch {
          // ignore
        }

        if (cancelled) return
        setSessions(initialSessions)
      } catch (err) {
        console.error('Failed to initialize AI Chat:', err)
      } finally {
        if (!cancelled) setLoading(false)
      }
    }

    init()
    return () => { cancelled = true }
  }, [])

  // Sync sessions to localStorage whenever sessions change
  useEffect(() => {
    if (sessions.length > 0) {
      try {
        localStorage.setItem(STORAGE_KEY_SESSIONS, JSON.stringify(sessions))
      } catch (err) {
        console.warn('Failed to save sessions to localStorage:', err)
      }
    }
  }, [sessions])

  // Sync activeSessionId to localStorage
  useEffect(() => {
    if (activeSessionId) {
      localStorage.setItem(STORAGE_KEY_ACTIVE, activeSessionId)
    }
  }, [activeSessionId])

  // Focus edit title input when editing begins
  useEffect(() => {
    if (editingSessionId && editTitleInputRef.current) {
      editTitleInputRef.current.focus()
      editTitleInputRef.current.select()
    }
  }, [editingSessionId])

  // Auto-scroll messages smoothly
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [currentMessages, streamedText])

  // Close dropdown on click outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setMeetingDropdownOpen(false)
      }
    }
    if (meetingDropdownOpen) {
      document.addEventListener('mousedown', handleClickOutside)
    }
    return () => {
      document.removeEventListener('mousedown', handleClickOutside)
    }
  }, [meetingDropdownOpen])

  // Refresh meetings
  const refreshMeetings = useCallback(async () => {
    try {
      const updated = await getAIChatMeetings()
      setAvailableMeetings(updated)
    } catch (err) {
      console.error('Failed to refresh AI Chat meetings:', err)
    }
  }, [])

  // ═══════ Session Management Handlers ═══════
  const handleNewChat = useCallback(async () => {
    if (streaming) return

    const newSession = createEmptySession('New Chat', '')
    setSessions(prev => [newSession, ...prev])
    setActiveSessionId(newSession.id)
    setInput('')
    setStreamedText('')
    setConfirmClearActive(false)

    // Notify backend with the exact session ID so client and DB stay 100% in sync
    createAIChatSession('New Chat', undefined, newSession.id).catch(() => {})

    setTimeout(() => {
      inputRef.current?.focus()
    }, 50)
  }, [streaming])

  const handleSelectSession = useCallback(async (sessionId: string) => {
    if (sessionId === activeSessionId) return
    if (streaming && abortRef.current) {
      abortRef.current.abort()
      abortRef.current = null
      setStreaming(false)
      setStreamedText('')
    }

    setActiveSessionId(sessionId)
    setInput('')
    setStreamedText('')
    setConfirmClearActive(false)

    // Check if we need to load history for this session from backend
    const targetSession = sessions.find(s => s.id === sessionId)
    if (!targetSession || targetSession.messages.length === 0) {
      try {
        const hist = await getAIChatHistory(sessionId)
        if (hist && hist.length > 0) {
          setSessions(prev => prev.map(s => {
            if (s.id === sessionId) {
              const lastAssistant = [...hist].reverse().find(m => m.role === 'assistant')
              return {
                ...s,
                messages: hist,
                currentSources: lastAssistant?.metadata?.meeting_sources || s.currentSources || [],
                currentQueryType: lastAssistant?.metadata?.query_type || s.currentQueryType || '',
              }
            }
            return s
          }))
        }
      } catch {
        // ignore
      }
    }

    setTimeout(() => {
      inputRef.current?.focus()
    }, 50)
  }, [activeSessionId, streaming, sessions])

  const handleStartRename = (session: AIChatSession, e: React.MouseEvent) => {
    e.stopPropagation()
    setEditingSessionId(session.id)
    setEditingTitle(session.title)
  }

  const handleSaveRename = (sessionId: string) => {
    const finalTitle = editingTitle.trim() || 'Chat'
    setSessions(prev => prev.map(s => s.id === sessionId ? { ...s, title: finalTitle, updated_at: new Date().toISOString() } : s))
    setEditingSessionId(null)
    updateAIChatSession(sessionId, { title: finalTitle }).catch(() => {})
  }

  const handleCancelRename = () => {
    setEditingSessionId(null)
    setEditingTitle('')
  }

  const handleDeleteSession = async (sessionId: string, e: React.MouseEvent) => {
    e.stopPropagation()
    const remaining = sessions.filter(s => s.id !== sessionId)

    if (remaining.length === 0) {
      // If deleting the last session, create a clean replacement
      const replacement = createEmptySession('New Chat', '')
      setSessions([replacement])
      setActiveSessionId(replacement.id)
      createAIChatSession('New Chat', undefined, replacement.id).catch(() => {})
    } else {
      setSessions(remaining)
      if (sessionId === activeSessionId) {
        setActiveSessionId(remaining[0].id)
      }
    }

    setDeleteConfirmId(null)
    deleteAIChatSession(sessionId).catch(() => {})
  }

  // Update focused meeting for active session
  const handleSetMeetingForActiveSession = (meetingId: string) => {
    if (!activeSessionId) return
    setSessions(prev => prev.map(s => {
      if (s.id === activeSessionId) {
        return { ...s, selected_meeting_id: meetingId, updated_at: new Date().toISOString() }
      }
      return s
    }))
    updateAIChatSession(activeSessionId, { selected_meeting_id: meetingId }).catch(() => {})
  }

  // Clear current active session history
  const handleClearActiveHistory = useCallback(async () => {
    if (!activeSessionId) return
    try {
      await clearAIChatHistory(activeSessionId)
      setSessions(prev => prev.map(s => {
        if (s.id === activeSessionId) {
          return {
            ...s,
            messages: [],
            currentSources: [],
            currentQueryType: '',
            updated_at: new Date().toISOString(),
          }
        }
        return s
      }))
      setConfirmClearActive(false)
    } catch (err) {
      console.error('Failed to clear history:', err)
    }
  }, [activeSessionId])

  // Stop streaming response
  const handleStopStreaming = () => {
    if (abortRef.current) {
      abortRef.current.abort()
      abortRef.current = null
    }
    if (streamedText && activeSessionId) {
      const assistantMsg: AIChatMessage = {
        id: `resp-${Date.now()}`,
        role: 'assistant',
        content: streamedText + ' [Response stopped]',
        metadata: {},
        created_at: new Date().toISOString(),
      }
      setSessions(prev => prev.map(s => s.id === activeSessionId ? { ...s, messages: [...s.messages, assistantMsg] } : s))
    }
    setStreamedText('')
    setStreaming(false)
  }

  // Send message
  const handleSend = useCallback(async (textToSend?: string) => {
    const rawMsg = textToSend !== undefined ? textToSend : input
    const msg = rawMsg.trim()
    if (!msg || streaming || !activeSessionId) return

    setInput('')
    setStreaming(true)
    setStreamedText('')

    const currentSessId = activeSessionId
    const currentMeetingFocus = selectedMeetingId

    const userMsg: AIChatMessage = {
      id: `temp-${Date.now()}`,
      role: 'user',
      content: msg,
      metadata: {},
      created_at: new Date().toISOString(),
    }

    // Determine if session should be auto-named
    let autoTitle: string | null = null
    const currentSess = sessions.find(s => s.id === currentSessId)
    if (currentSess && (currentSess.title === 'New Chat' || currentSess.title === 'Default Chat' || !currentSess.title.trim())) {
      autoTitle = msg.length > 36 ? msg.slice(0, 36) + '...' : msg
    }

    // Append user message immediately
    setSessions(prev => prev.map(s => {
      if (s.id === currentSessId) {
        return {
          ...s,
          title: autoTitle || s.title,
          messages: [...s.messages, userMsg],
          updated_at: new Date().toISOString(),
        }
      }
      return s
    }))

    const controller = new AbortController()
    abortRef.current = controller
    let accumulated = ''
    let responseMeta: any = {}

    try {
      await streamAIChat(
        msg,
        (chunk) => {
          accumulated += chunk
          setStreamedText(accumulated)
        },
        (meta) => {
          responseMeta = meta
          setSessions(prev => prev.map(s => {
            if (s.id === currentSessId) {
              return {
                ...s,
                currentSources: meta.meeting_sources || s.currentSources || [],
                currentQueryType: meta.query_type || s.currentQueryType || '',
              }
            }
            return s
          }))
        },
        (error) => {
          accumulated = error
          setStreamedText(error)
        },
        () => {
          if (accumulated) {
            const assistantMsg: AIChatMessage = {
              id: `resp-${Date.now()}`,
              role: 'assistant',
              content: accumulated,
              metadata: responseMeta,
              created_at: new Date().toISOString(),
            }
            setSessions(prev => prev.map(s => {
              if (s.id === currentSessId) {
                return {
                  ...s,
                  messages: [...s.messages, assistantMsg],
                  currentSources: responseMeta.meeting_sources || s.currentSources || [],
                  currentQueryType: responseMeta.query_type || s.currentQueryType || '',
                  updated_at: new Date().toISOString(),
                }
              }
              return s
            }))

            // Sync with backend DB in background to ensure canonical database IDs and state
            getAIChatHistory(currentSessId).then(dbHist => {
              if (dbHist && dbHist.length > 0) {
                setSessions(prev => prev.map(s => {
                  if (s.id === currentSessId) {
                    const lastAssistant = [...dbHist].reverse().find(m => m.role === 'assistant')
                    return {
                      ...s,
                      messages: dbHist,
                      currentSources: lastAssistant?.metadata?.meeting_sources || s.currentSources || [],
                      currentQueryType: lastAssistant?.metadata?.query_type || s.currentQueryType || '',
                    }
                  }
                  return s
                }))
              }
            }).catch(() => {})
          }
          setStreamedText('')
          setStreaming(false)
        },
        controller.signal,
        currentMeetingFocus || undefined,
        undefined,
        currentSessId,
      )
    } catch (err: any) {
      if (err.name !== 'AbortError') {
        const errorMsg: AIChatMessage = {
          id: `err-${Date.now()}`,
          role: 'assistant',
          content: '⚠️ Failed to get AI response. Please verify connection and try again.',
          metadata: {},
          created_at: new Date().toISOString(),
        }
        setSessions(prev => prev.map(s => {
          if (s.id === currentSessId) {
            return {
              ...s,
              messages: [...s.messages, errorMsg],
              updated_at: new Date().toISOString(),
            }
          }
          return s
        }))
      }
      setStreamedText('')
      setStreaming(false)
    }
  }, [input, streaming, activeSessionId, selectedMeetingId, sessions])

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const handleCopy = useCallback((text: string, id: string) => {
    navigator.clipboard.writeText(text)
    setCopied(id)
    setTimeout(() => setCopied(null), 2000)
  }, [])

  // ═══════ Formatters & Helpers ═══════
  const fmtDate = (iso: string) => {
    if (!iso) return ''
    try {
      const d = new Date(iso)
      return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })
    } catch { return iso }
  }

  const fmtRelativeTime = (iso: string) => {
    if (!iso) return ''
    try {
      const d = new Date(iso)
      const now = new Date()
      const diffMs = now.getTime() - d.getTime()
      const diffMins = Math.floor(diffMs / (1000 * 60))
      if (diffMins < 1) return 'Just now'
      if (diffMins < 60) return `${diffMins}m ago`
      const diffHours = Math.floor(diffMins / 60)
      if (diffHours < 24) return `${diffHours}h ago`
      const diffDays = Math.floor(diffHours / 24)
      if (diffDays === 1) return 'Yesterday'
      if (diffDays < 7) return `${diffDays}d ago`
      return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
    } catch {
      return ''
    }
  }

  const renderMarkdown = (text: string) => {
    const lines = text.split('\n')
    return lines.map((line, i) => {
      if (line.startsWith('### ')) return <h4 key={i} style={{ fontSize: '.95rem', fontWeight: 700, margin: '1rem 0 .5rem', color: 'hsl(var(--ink))' }}>{line.slice(4)}</h4>
      if (line.startsWith('## ')) return <h3 key={i} style={{ fontSize: '1.05rem', fontWeight: 700, margin: '1.25rem 0 .5rem', color: 'hsl(var(--accent))', borderBottom: '1px solid hsl(var(--accent) / .15)', paddingBottom: '.35rem' }}>{line.slice(3)}</h3>
      if (line.startsWith('# ')) return <h2 key={i} style={{ fontSize: '1.15rem', fontWeight: 700, margin: '1.5rem 0 .75rem', color: 'hsl(var(--ink))' }}>{line.slice(2)}</h2>
      if (line.match(/^[-•]\s/)) {
        const content = line.replace(/^[-•]\s/, '')
        return (
          <div key={i} style={{ display: 'flex', gap: '.5rem', margin: '.25rem 0', lineHeight: 1.6 }}>
            <span style={{ color: 'hsl(var(--accent))', flexShrink: 0, marginTop: '2px' }}>•</span>
            <span dangerouslySetInnerHTML={{ __html: boldify(content) }} />
          </div>
        )
      }
      if (!line.trim()) return <div key={i} style={{ height: '.5rem' }} />
      return <p key={i} style={{ margin: '.25rem 0', lineHeight: 1.6 }} dangerouslySetInnerHTML={{ __html: boldify(line) }} />
    })
  }

  const boldify = (text: string) => {
    return text
      .replace(/\*\*(.+?)\*\*/g, '<strong style="color:hsl(var(--ink));font-weight:700">$1</strong>')
      .replace(/\[Meeting[:\s]*([^\]]+)\]/g, '<span style="background:hsl(var(--accent)/.1);color:hsl(var(--accent));padding:0.1rem 0.4rem;border-radius:4px;font-size:.8rem;font-weight:600">📎 $1</span>')
  }

  // Filtered sessions
  const filteredSessions = sessions.filter(s => {
    if (!sessionSearchQuery.trim()) return true
    return s.title.toLowerCase().includes(sessionSearchQuery.toLowerCase())
  })

  // Filtered meetings for selector
  const filteredMeetings = availableMeetings.filter(m => {
    if (onlyStage2 && !m.has_stage2) return false
    if (meetingFilterType === 'recorded' && m.is_manual_mom) return false
    if (meetingFilterType === 'manual' && !m.is_manual_mom) return false
    if (!meetingSearchQuery.trim()) return true
    const q = meetingSearchQuery.toLowerCase()
    const nameMatch = m.name.toLowerCase().includes(q)
    const dateMatch = fmtDate(m.date).toLowerCase().includes(q)
    const tagMatch = m.is_manual_mom && ('manual'.includes(q) || 'mom'.includes(q))
    return nameMatch || dateMatch || tagMatch
  })

  const stage2Count = availableMeetings.filter(m => m.has_stage2).length
  const manualMomCount = availableMeetings.filter(m => m.is_manual_mom).length
  const selectedMeeting = availableMeetings.find(m => m.id === selectedMeetingId)

  // Suggested questions when a new chat starts
  const suggestedQueries = [
    { label: 'Summarize recent discussions', prompt: 'Summarize the key discussions and major outcomes across all recent meetings.' },
    { label: 'Action items & assignees', prompt: 'What action items and pending tasks were agreed upon in recent meetings?' },
    { label: 'Key decisions made', prompt: 'What were the most important decisions and conclusions reached recently?' },
    { label: 'Risks & open concerns', prompt: 'Were there any notable blockers, risks, or concerns highlighted in recent meetings?' },
  ]

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>

      {/* ══════════════════════════════════════════════════════════════════
          TOP HEADER
      ══════════════════════════════════════════════════════════════════ */}
      <div style={{
        padding: '0.85rem 1.5rem',
        borderBottom: '2px solid hsl(var(--border) / .1)',
        background: 'hsl(var(--card) / .65)',
        backdropFilter: 'blur(12px)',
        flexShrink: 0,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        position: 'relative',
        zIndex: 20,
        gap: '1rem',
      }}>
        {/* Left: Sidebar Toggle + Title */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '.75rem', minWidth: 0 }}>
          <button
            onClick={handleToggleSidebar}
            title={sidebarOpen ? 'Hide chat history' : 'Show chat history'}
            className="icon-btn"
            style={{
              width: '36px',
              height: '36px',
              borderRadius: '9px',
              border: '1px solid hsl(var(--border) / .2)',
              background: sidebarOpen ? 'hsl(var(--paper))' : 'hsl(var(--accent) / .1)',
              color: sidebarOpen ? 'hsl(var(--pencil))' : 'hsl(var(--accent))',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              cursor: 'pointer',
              transition: 'all .15s ease',
            }}
          >
            {sidebarOpen ? <PanelLeftClose size={18} /> : <PanelLeftOpen size={18} />}
          </button>

          <div style={{
            width: '38px',
            height: '38px',
            borderRadius: '11px',
            background: 'linear-gradient(135deg, hsl(270 80% 60% / .2), hsl(200 80% 60% / .2))',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            border: '1.5px solid hsl(270 60% 60% / .3)',
            flexShrink: 0,
          }}>
            <MessageSquare size={19} style={{ color: 'hsl(270 60% 60%)' }} />
          </div>

          <div style={{ minWidth: 0 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '.5rem' }}>
              <h2 style={{ fontSize: '1.15rem', fontWeight: 700, margin: 0, color: 'hsl(var(--ink))', whiteSpace: 'nowrap' }}>
                AI Chat
              </h2>
              {activeSession && (
                <span style={{
                  fontSize: '.74rem',
                  fontWeight: 600,
                  padding: '2px 8px',
                  borderRadius: '6px',
                  background: 'hsl(var(--accent) / .1)',
                  color: 'hsl(var(--accent))',
                  maxWidth: '180px',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                }}
                title={activeSession.title}
                >
                  {activeSession.title}
                </span>
              )}
            </div>
            <p style={{ fontSize: '.76rem', color: 'hsl(var(--pencil))', margin: 0, fontWeight: 500 }}>
              Ask questions about your meetings
            </p>
          </div>
        </div>

        {/* Right: Controls & Context Selectors */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '.65rem', flexShrink: 0 }}>

          {/* Upload Meeting MoM Button */}
          <button
            onClick={() => setManualMomModalOpen(true)}
            style={{
              height: '36px',
              fontSize: '.82rem',
              fontWeight: 700,
              padding: '0 .9rem',
              borderRadius: '9px',
              background: 'linear-gradient(135deg, hsl(280 80% 55%), hsl(260 80% 60%))',
              color: '#fff',
              border: 'none',
              display: 'flex',
              alignItems: 'center',
              gap: '.45rem',
              cursor: 'pointer',
              boxShadow: '0 3px 10px hsl(280 80% 55% / .25)',
              transition: 'all .15s ease',
            }}
            title="Upload multiple MoM files and view the MoM Library"
          >
            <UploadCloud size={15} />
            <span>Upload Meeting MoM</span>
            {manualMomCount > 0 && (
              <span style={{
                fontSize: '.66rem',
                fontWeight: 700,
                background: 'rgba(255, 255, 255, 0.25)',
                padding: '1px 6px',
                borderRadius: '10px',
                marginLeft: '1px',
              }}>
                {manualMomCount}
              </span>
            )}
          </button>

          {/* Searchable Meeting Selector */}
          <div ref={dropdownRef} style={{ position: 'relative' }}>
            <button
              className="btn btn-ghost"
              onClick={() => setMeetingDropdownOpen(prev => !prev)}
              style={{
                height: '36px',
                fontSize: '.83rem',
                padding: '0 .75rem',
                minWidth: '200px',
                maxWidth: '300px',
                borderRadius: '9px',
                border: selectedMeetingId
                  ? '2px solid hsl(var(--accent) / .5)'
                  : '1.5px solid hsl(var(--border) / .25)',
                background: selectedMeetingId
                  ? 'hsl(var(--accent) / .08)'
                  : 'hsl(var(--paper))',
                color: selectedMeetingId ? 'hsl(var(--ink))' : 'hsl(var(--pencil))',
                display: 'flex',
                alignItems: 'center',
                gap: '.5rem',
                justifyContent: 'space-between',
                cursor: 'pointer',
                textAlign: 'left',
              }}
              title={selectedMeeting ? `Focused on: ${selectedMeeting.name}` : 'Select a meeting to focus context for this chat'}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '.4rem', minWidth: 0, flex: 1 }}>
                <FileText size={14} style={{ color: selectedMeetingId ? 'hsl(var(--accent))' : 'hsl(var(--pencil))', flexShrink: 0 }} />
                <span style={{
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                  fontWeight: selectedMeetingId ? 600 : 500,
                  fontSize: '.82rem',
                  color: selectedMeetingId ? 'hsl(var(--ink))' : 'hsl(var(--pencil))',
                }}>
                  {selectedMeeting ? selectedMeeting.name : 'All Meetings'}
                </span>
                {selectedMeeting && selectedMeeting.is_manual_mom ? (
                  <span style={{
                    fontSize: '.62rem',
                    fontWeight: 700,
                    padding: '1px 5px',
                    borderRadius: '4px',
                    background: 'hsl(280 80% 50% / .18)',
                    color: 'hsl(280 80% 40%)',
                    flexShrink: 0,
                  }}>
                    Manual MoM
                  </span>
                ) : selectedMeeting && selectedMeeting.has_stage2 ? (
                  <span style={{
                    fontSize: '.62rem',
                    fontWeight: 700,
                    padding: '1px 5px',
                    borderRadius: '4px',
                    background: 'hsl(142 70% 45% / .18)',
                    color: 'hsl(142 70% 35%)',
                    flexShrink: 0,
                  }}>
                    Stage 2
                  </span>
                ) : null}
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '.25rem', flexShrink: 0 }}>
                {selectedMeetingId && (
                  <span
                    onClick={(e) => {
                      e.stopPropagation()
                      handleSetMeetingForActiveSession('')
                    }}
                    title="Clear meeting focus"
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      width: '18px',
                      height: '18px',
                      borderRadius: '50%',
                      background: 'hsl(var(--ink) / .08)',
                      cursor: 'pointer',
                    }}
                  >
                    <X size={11} style={{ color: 'hsl(var(--pencil))' }} />
                  </span>
                )}
                <ChevronDown
                  size={14}
                  style={{
                    color: 'hsl(var(--pencil))',
                    transform: meetingDropdownOpen ? 'rotate(180deg)' : 'none',
                    transition: 'transform .15s ease',
                  }}
                />
              </div>
            </button>

            {/* Dropdown Popover */}
            {meetingDropdownOpen && (
              <div style={{
                position: 'absolute',
                top: 'calc(100% + 6px)',
                right: 0,
                width: '350px',
                borderRadius: '12px',
                background: 'hsl(var(--card))',
                border: '1.5px solid hsl(var(--border) / .25)',
                boxShadow: '0 12px 36px hsl(var(--ink) / .16)',
                zIndex: 100,
                display: 'flex',
                flexDirection: 'column',
                overflow: 'hidden',
              }}>
                {/* Search Bar */}
                <div style={{
                  padding: '.6rem .75rem .45rem',
                  borderBottom: '1px solid hsl(var(--border) / .12)',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '.5rem',
                  background: 'hsl(var(--paper))',
                }}>
                  <Search size={14} style={{ color: 'hsl(var(--pencil))', flexShrink: 0 }} />
                  <input
                    type="text"
                    value={meetingSearchQuery}
                    onChange={(e) => setMeetingSearchQuery(e.target.value)}
                    placeholder="Search meetings by name or date..."
                    autoFocus
                    style={{
                      flex: 1,
                      border: 'none',
                      background: 'transparent',
                      outline: 'none',
                      fontSize: '.82rem',
                      color: 'hsl(var(--ink))',
                      fontFamily: 'Inter, sans-serif',
                    }}
                  />
                  {meetingSearchQuery && (
                    <button
                      onClick={() => setMeetingSearchQuery('')}
                      style={{
                        background: 'none',
                        border: 'none',
                        cursor: 'pointer',
                        padding: '2px',
                        color: 'hsl(var(--pencil))',
                        display: 'flex',
                      }}
                    >
                      <X size={13} />
                    </button>
                  )}
                </div>

                {/* Filter toggle bar */}
                <div style={{
                  padding: '.45rem .75rem',
                  background: 'hsl(var(--paper-deep) / .5)',
                  borderBottom: '1px solid hsl(var(--border) / .12)',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '.35rem',
                }}>
                  {/* Category Pills */}
                  <div style={{ display: 'flex', alignItems: 'center', gap: '.3rem' }}>
                    <button
                      onClick={() => setMeetingFilterType('all')}
                      style={{
                        padding: '2px 7px',
                        borderRadius: '5px',
                        border: 'none',
                        background: meetingFilterType === 'all' ? 'hsl(var(--accent) / .2)' : 'transparent',
                        color: meetingFilterType === 'all' ? 'hsl(var(--accent))' : 'hsl(var(--pencil))',
                        fontWeight: meetingFilterType === 'all' ? 700 : 500,
                        fontSize: '.72rem',
                        cursor: 'pointer',
                      }}
                    >
                      All ({availableMeetings.length})
                    </button>
                    <button
                      onClick={() => setMeetingFilterType('recorded')}
                      style={{
                        padding: '2px 7px',
                        borderRadius: '5px',
                        border: 'none',
                        background: meetingFilterType === 'recorded' ? 'hsl(var(--accent) / .2)' : 'transparent',
                        color: meetingFilterType === 'recorded' ? 'hsl(var(--accent))' : 'hsl(var(--pencil))',
                        fontWeight: meetingFilterType === 'recorded' ? 700 : 500,
                        fontSize: '.72rem',
                        cursor: 'pointer',
                      }}
                    >
                      Recorded ({availableMeetings.length - manualMomCount})
                    </button>
                    <button
                      onClick={() => setMeetingFilterType('manual')}
                      style={{
                        padding: '2px 7px',
                        borderRadius: '5px',
                        border: 'none',
                        background: meetingFilterType === 'manual' ? 'hsl(280 80% 50% / .2)' : 'transparent',
                        color: meetingFilterType === 'manual' ? 'hsl(280 80% 45%)' : 'hsl(var(--pencil))',
                        fontWeight: meetingFilterType === 'manual' ? 700 : 500,
                        fontSize: '.72rem',
                        cursor: 'pointer',
                      }}
                    >
                      Manual MoM ({manualMomCount})
                    </button>
                  </div>

                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '.74rem' }}>
                    <label
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '.45rem',
                        cursor: 'pointer',
                        userSelect: 'none',
                        color: onlyStage2 ? 'hsl(var(--accent))' : 'hsl(var(--ink))',
                        fontWeight: onlyStage2 ? 600 : 500,
                      }}
                    >
                      <input
                        type="checkbox"
                        checked={onlyStage2}
                        onChange={(e) => setOnlyStage2(e.target.checked)}
                        style={{
                          accentColor: 'hsl(var(--accent))',
                          cursor: 'pointer',
                          width: '13px',
                          height: '13px',
                        }}
                      />
                      <span style={{ display: 'flex', alignItems: 'center', gap: '.3rem' }}>
                        <Zap size={11} style={{ color: onlyStage2 ? 'hsl(var(--accent))' : 'hsl(var(--pencil))' }} />
                        Only Stage 2 ready
                      </span>
                    </label>
                    <span style={{
                      fontSize: '.68rem',
                      color: 'hsl(var(--pencil))',
                      background: 'hsl(var(--paper))',
                      padding: '1px 5px',
                      borderRadius: '4px',
                      border: '1px solid hsl(var(--border) / .15)',
                    }}>
                      {stage2Count} ready
                    </span>
                  </div>
                </div>

                {/* List items */}
                <div style={{
                  maxHeight: '250px',
                  overflowY: 'auto',
                  padding: '.35rem',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '2px',
                }}>
                  {/* Option: All Meetings */}
                  {(!meetingSearchQuery || 'all meetings'.includes(meetingSearchQuery.toLowerCase())) && (
                    <div
                      onClick={() => {
                        handleSetMeetingForActiveSession('')
                        setMeetingDropdownOpen(false)
                      }}
                      style={{
                        padding: '.55rem .75rem',
                        borderRadius: '8px',
                        cursor: 'pointer',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        background: !selectedMeetingId ? 'hsl(var(--accent) / .1)' : 'transparent',
                        color: !selectedMeetingId ? 'hsl(var(--accent))' : 'hsl(var(--ink))',
                        fontWeight: !selectedMeetingId ? 600 : 500,
                        fontSize: '.82rem',
                        transition: 'background .15s',
                      }}
                    >
                      <div style={{ display: 'flex', flexDirection: 'column' }}>
                        <span style={{ display: 'flex', alignItems: 'center', gap: '.4rem' }}>
                          <Sparkles size={13} style={{ color: 'hsl(var(--accent))' }} />
                          All Meetings
                        </span>
                        <span style={{ fontSize: '.7rem', color: 'hsl(var(--pencil))', marginLeft: '1.2rem', marginTop: '1px' }}>
                          Search across all meetings in workspace
                        </span>
                      </div>
                      {!selectedMeetingId && <Check size={14} style={{ color: 'hsl(var(--accent))', flexShrink: 0 }} />}
                    </div>
                  )}

                  {/* Filtered Meeting Items */}
                  {filteredMeetings.length > 0 ? (
                    filteredMeetings.map((m) => {
                      const isSelected = m.id === selectedMeetingId
                      return (
                        <div
                          key={m.id}
                          onClick={() => {
                            handleSetMeetingForActiveSession(m.id)
                            setMeetingDropdownOpen(false)
                          }}
                          style={{
                            padding: '.55rem .75rem',
                            borderRadius: '8px',
                            cursor: 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'space-between',
                            gap: '.5rem',
                            background: isSelected ? 'hsl(var(--accent) / .1)' : 'transparent',
                            transition: 'background .15s',
                          }}
                        >
                          <div style={{ minWidth: 0, flex: 1 }}>
                            <div style={{
                              fontSize: '.82rem',
                              fontWeight: isSelected ? 600 : 500,
                              color: isSelected ? 'hsl(var(--accent))' : 'hsl(var(--ink))',
                              overflow: 'hidden',
                              textOverflow: 'ellipsis',
                              whiteSpace: 'nowrap',
                            }}>
                              {m.name}
                            </div>
                            <div style={{
                              display: 'flex',
                              alignItems: 'center',
                              gap: '.5rem',
                              marginTop: '2px',
                            }}>
                              {m.date && (
                                <span style={{ fontSize: '.7rem', color: 'hsl(var(--pencil))' }}>
                                  {fmtDate(m.date)}
                                </span>
                              )}
                              {m.is_manual_mom ? (
                                <span style={{
                                  fontSize: '.62rem',
                                  fontWeight: 700,
                                  padding: '1px 5px',
                                  borderRadius: '4px',
                                  background: 'hsl(280 80% 50% / .18)',
                                  color: 'hsl(280 80% 40%)',
                                }}>
                                  Manual MoM
                                </span>
                              ) : m.has_stage2 ? (
                                <span style={{
                                  fontSize: '.62rem',
                                  fontWeight: 700,
                                  padding: '1px 5px',
                                  borderRadius: '4px',
                                  background: 'hsl(142 70% 45% / .15)',
                                  color: 'hsl(142 70% 35%)',
                                }}>
                                  ⚡ Stage 2 Ready
                                </span>
                              ) : (
                                <span style={{
                                  fontSize: '.62rem',
                                  fontWeight: 500,
                                  padding: '1px 5px',
                                  borderRadius: '4px',
                                  background: 'hsl(var(--muted) / .5)',
                                  color: 'hsl(var(--pencil))',
                                }}>
                                  Recorded
                                </span>
                              )}
                            </div>
                          </div>
                          {isSelected && <Check size={14} style={{ color: 'hsl(var(--accent))', flexShrink: 0 }} />}
                        </div>
                      )
                    })
                  ) : (
                    <div style={{
                      padding: '1.25rem 1rem',
                      textAlign: 'center',
                      color: 'hsl(var(--pencil))',
                      fontSize: '.78rem',
                    }}>
                      No meetings matched your search.
                    </div>
                  )}
                </div>

                {/* Dropdown footer info */}
                <div style={{
                  padding: '.45rem .75rem',
                  borderTop: '1px solid hsl(var(--border) / .1)',
                  background: 'hsl(var(--paper))',
                  fontSize: '.7rem',
                  color: 'hsl(var(--pencil))',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  gap: '.5rem',
                }}>
                  <span>{filteredMeetings.length} of {availableMeetings.length} meetings</span>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '.65rem' }}>
                    <button
                      onClick={() => {
                        setMeetingDropdownOpen(false)
                        setManualMomModalOpen(true)
                      }}
                      style={{
                        background: 'none',
                        border: 'none',
                        color: 'hsl(280 80% 50%)',
                        cursor: 'pointer',
                        padding: 0,
                        fontSize: '.7rem',
                        fontWeight: 700,
                        display: 'flex',
                        alignItems: 'center',
                        gap: '.25rem',
                      }}
                    >
                      <UploadCloud size={11} />
                      MoM Library ({manualMomCount})
                    </button>
                    <button
                      onClick={() => {
                        setMeetingDropdownOpen(false)
                        setResyncModalOpen(true)
                      }}
                      style={{
                        background: 'none',
                        border: 'none',
                        color: 'hsl(var(--accent))',
                        cursor: 'pointer',
                        padding: 0,
                        fontSize: '.7rem',
                        fontWeight: 600,
                        display: 'flex',
                        alignItems: 'center',
                        gap: '.25rem',
                      }}
                    >
                      <RotateCcw size={10} />
                      Resync Stage 2
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Resync Meeting Button */}
          <button
            onClick={() => setResyncModalOpen(true)}
            title="Resync meetings with Stage 2 points into ChromaDB"
            style={{
              height: '36px',
              fontSize: '.8rem',
              fontWeight: 600,
              padding: '0 .75rem',
              borderRadius: '9px',
              border: '1.5px solid hsl(var(--border) / .3)',
              background: 'hsl(var(--card))',
              color: 'hsl(var(--ink))',
              display: 'flex',
              alignItems: 'center',
              gap: '.4rem',
              cursor: 'pointer',
              transition: 'all .15s ease',
            }}
          >
            <RotateCcw size={13} style={{ color: 'hsl(var(--accent))' }} />
            <span>Resync</span>
          </button>

          {/* Clear Current Chat History */}
          {currentMessages.length > 0 && (
            confirmClearActive ? (
              <div style={{
                display: 'flex', alignItems: 'center', gap: '.3rem',
                background: 'hsl(var(--card))', padding: '.2rem .4rem',
                borderRadius: '8px', border: '1px solid hsl(var(--destructive) / .3)',
                fontSize: '.75rem',
              }}>
                <span style={{ fontWeight: 600 }}>Clear chat?</span>
                <button
                  className="btn"
                  onClick={handleClearActiveHistory}
                  style={{ padding: '.15rem .45rem', background: 'hsl(var(--destructive))', color: 'white', minHeight: 0, height: 'auto', fontSize: '.72rem' }}
                >
                  Yes
                </button>
                <button
                  className="btn btn-ghost"
                  onClick={() => setConfirmClearActive(false)}
                  style={{ padding: '.15rem .45rem', minHeight: 0, height: 'auto', fontSize: '.72rem' }}
                >
                  No
                </button>
              </div>
            ) : (
              <button
                className="icon-btn"
                onClick={() => setConfirmClearActive(true)}
                title="Clear messages in this chat"
                style={{ width: '36px', height: '36px', borderRadius: '9px', color: 'hsl(var(--pencil))' }}
              >
                <Trash2 size={15} />
              </button>
            )
          )}

          {/* Toggle Right Sources Panel */}
          <button
            onClick={() => setRightPanelOpen(prev => !prev)}
            title={rightPanelOpen ? 'Hide sources sidebar' : 'Show sources sidebar'}
            className="icon-btn"
            style={{
              width: '36px',
              height: '36px',
              borderRadius: '9px',
              border: '1px solid hsl(var(--border) / .2)',
              background: rightPanelOpen ? 'hsl(var(--accent) / .1)' : 'hsl(var(--paper))',
              color: rightPanelOpen ? 'hsl(var(--accent))' : 'hsl(var(--pencil))',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              cursor: 'pointer',
            }}
          >
            <FileText size={16} />
          </button>
        </div>
      </div>

      {/* ══════════════════════════════════════════════════════════════════
          MAIN BODY (3-PANE LAYOUT: SIDEBAR + CHAT + CONTEXT)
      ══════════════════════════════════════════════════════════════════ */}
      <div style={{ flex: 1, display: 'flex', overflow: 'hidden', minHeight: 0 }}>

        {/* ────────────────────────────────────────────────────────────
            1. COLLAPSIBLE LEFT SIDEBAR (CHAT SESSIONS)
        ──────────────────────────────────────────────────────────── */}
        <div style={{
          width: sidebarOpen ? '280px' : '0px',
          minWidth: sidebarOpen ? '280px' : '0px',
          maxWidth: '280px',
          transition: 'width 0.22s cubic-bezier(0.4, 0, 0.2, 1), min-width 0.22s cubic-bezier(0.4, 0, 0.2, 1)',
          overflow: 'hidden',
          display: 'flex',
          flexDirection: 'column',
          borderRight: sidebarOpen ? '2px solid hsl(var(--border) / .12)' : 'none',
          background: 'hsl(var(--card) / .5)',
          backdropFilter: 'blur(10px)',
          flexShrink: 0,
          zIndex: 10,
        }}>
          {/* Sidebar Top: Header & New Chat button */}
          <div style={{ padding: '1rem', borderBottom: '1px solid hsl(var(--border) / .1)', display: 'flex', flexDirection: 'column', gap: '.75rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <span style={{ fontSize: '.84rem', fontWeight: 700, color: 'hsl(var(--ink))', display: 'flex', alignItems: 'center', gap: '.4rem' }}>
                <Clock size={14} style={{ color: 'hsl(var(--accent))' }} />
                Chat History
              </span>
              <span style={{
                fontSize: '.7rem',
                color: 'hsl(var(--pencil))',
                background: 'hsl(var(--paper))',
                padding: '2px 7px',
                borderRadius: '6px',
                border: '1px solid hsl(var(--border) / .15)',
              }}>
                {sessions.length}
              </span>
            </div>

            {/* New Chat Button */}
            <button
              onClick={handleNewChat}
              disabled={streaming}
              style={{
                width: '100%',
                height: '38px',
                borderRadius: '10px',
                border: '1.5px solid hsl(var(--accent) / .4)',
                background: 'linear-gradient(135deg, hsl(var(--accent)), hsl(var(--accent) / .85))',
                color: 'white',
                fontSize: '.85rem',
                fontWeight: 600,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '.5rem',
                cursor: streaming ? 'not-allowed' : 'pointer',
                opacity: streaming ? 0.6 : 1,
                boxShadow: '0 2px 8px hsl(var(--accent) / .25)',
                transition: 'all .15s ease',
              }}
              onMouseEnter={e => {
                if (!streaming) (e.currentTarget as HTMLElement).style.filter = 'brightness(1.08)'
              }}
              onMouseLeave={e => {
                (e.currentTarget as HTMLElement).style.filter = 'none'
              }}
            >
              <Plus size={16} />
              <span>New Chat</span>
            </button>

            {/* Session Search Input */}
            {sessions.length > 3 && (
              <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: '.4rem',
                background: 'hsl(var(--paper))',
                borderRadius: '8px',
                padding: '4px 8px',
                border: '1px solid hsl(var(--border) / .15)',
              }}>
                <Search size={12} style={{ color: 'hsl(var(--pencil))' }} />
                <input
                  type="text"
                  placeholder="Filter chats..."
                  value={sessionSearchQuery}
                  onChange={e => setSessionSearchQuery(e.target.value)}
                  style={{
                    border: 'none',
                    background: 'transparent',
                    outline: 'none',
                    fontSize: '.75rem',
                    color: 'hsl(var(--ink))',
                    width: '100%',
                    fontFamily: 'Inter, sans-serif',
                  }}
                />
                {sessionSearchQuery && (
                  <button
                    onClick={() => setSessionSearchQuery('')}
                    style={{ background: 'none', border: 'none', cursor: 'pointer', padding: 0, color: 'hsl(var(--pencil))' }}
                  >
                    <X size={12} />
                  </button>
                )}
              </div>
            )}
          </div>

          {/* Sessions List */}
          <div style={{
            flex: 1,
            overflowY: 'auto',
            padding: '.6rem .65rem',
            display: 'flex',
            flexDirection: 'column',
            gap: '.35rem',
          }}>
            {filteredSessions.length === 0 ? (
              <div style={{ textAlign: 'center', padding: '2rem 1rem', color: 'hsl(var(--pencil))', fontSize: '.78rem' }}>
                {sessionSearchQuery ? 'No chats matched your search.' : 'No chat history yet.'}
              </div>
            ) : (
              filteredSessions.map(sess => {
                const isActive = sess.id === activeSessionId
                const isEditing = sess.id === editingSessionId
                const isDeleting = sess.id === deleteConfirmId

                return (
                  <div
                    key={sess.id}
                    onClick={() => !isEditing && handleSelectSession(sess.id)}
                    style={{
                      position: 'relative',
                      padding: '.6rem .7rem',
                      borderRadius: '10px',
                      cursor: 'pointer',
                      border: isActive
                        ? '1.5px solid hsl(var(--accent) / .4)'
                        : '1px solid transparent',
                      background: isActive
                        ? 'hsl(var(--accent) / .1)'
                        : 'transparent',
                      transition: 'all .15s ease',
                      display: 'flex',
                      flexDirection: 'column',
                      gap: '.25rem',
                    }}
                    onMouseEnter={e => {
                      if (!isActive) {
                        (e.currentTarget as HTMLElement).style.background = 'hsl(var(--paper-deep) / .6)'
                      }
                    }}
                    onMouseLeave={e => {
                      if (!isActive) {
                        (e.currentTarget as HTMLElement).style.background = 'transparent'
                      }
                    }}
                  >
                    {/* Title or Edit Input */}
                    {isEditing ? (
                      <div
                        onClick={e => e.stopPropagation()}
                        style={{ display: 'flex', alignItems: 'center', gap: '.3rem' }}
                      >
                        <input
                          ref={editTitleInputRef}
                          type="text"
                          value={editingTitle}
                          onChange={e => setEditingTitle(e.target.value)}
                          onKeyDown={e => {
                            if (e.key === 'Enter') handleSaveRename(sess.id)
                            if (e.key === 'Escape') handleCancelRename()
                          }}
                          style={{
                            flex: 1,
                            fontSize: '.8rem',
                            fontWeight: 600,
                            padding: '2px 6px',
                            borderRadius: '5px',
                            border: '1px solid hsl(var(--accent))',
                            background: 'hsl(var(--card))',
                            color: 'hsl(var(--ink))',
                            outline: 'none',
                          }}
                        />
                        <button
                          onClick={() => handleSaveRename(sess.id)}
                          title="Save title"
                          style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'hsl(var(--accent))', padding: '2px' }}
                        >
                          <Check size={14} />
                        </button>
                        <button
                          onClick={handleCancelRename}
                          title="Cancel"
                          style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'hsl(var(--pencil))', padding: '2px' }}
                        >
                          <X size={14} />
                        </button>
                      </div>
                    ) : (
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '.4rem' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '.45rem', minWidth: 0, flex: 1 }}>
                          <MessageCircle
                            size={14}
                            style={{
                              color: isActive ? 'hsl(var(--accent))' : 'hsl(var(--pencil))',
                              flexShrink: 0,
                            }}
                          />
                          <span style={{
                            fontSize: '.82rem',
                            fontWeight: isActive ? 600 : 500,
                            color: isActive ? 'hsl(var(--ink))' : 'hsl(var(--ink) / .85)',
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                            whiteSpace: 'nowrap',
                          }}>
                            {sess.title || 'Untitled Chat'}
                          </span>
                        </div>

                        {/* Actions on hover or active */}
                        <div
                          style={{
                            display: 'flex',
                            alignItems: 'center',
                            gap: '.2rem',
                            opacity: isActive ? 1 : 0.6,
                            flexShrink: 0,
                          }}
                          onClick={e => e.stopPropagation()}
                        >
                          <button
                            onClick={e => handleStartRename(sess, e)}
                            title="Rename chat"
                            style={{
                              background: 'none',
                              border: 'none',
                              cursor: 'pointer',
                              padding: '2px 4px',
                              color: 'hsl(var(--pencil))',
                              borderRadius: '4px',
                            }}
                          >
                            <Edit3 size={12} />
                          </button>

                          {isDeleting ? (
                            <div style={{ display: 'flex', alignItems: 'center', gap: '2px' }}>
                              <button
                                onClick={e => handleDeleteSession(sess.id, e)}
                                title="Confirm delete"
                                style={{
                                  background: 'hsl(var(--destructive))',
                                  color: 'white',
                                  border: 'none',
                                  borderRadius: '4px',
                                  padding: '1px 5px',
                                  fontSize: '.65rem',
                                  fontWeight: 700,
                                  cursor: 'pointer',
                                }}
                              >
                                Del
                              </button>
                              <button
                                onClick={e => {
                                  e.stopPropagation()
                                  setDeleteConfirmId(null)
                                }}
                                title="Cancel"
                                style={{
                                  background: 'none',
                                  border: 'none',
                                  color: 'hsl(var(--pencil))',
                                  padding: '1px',
                                  cursor: 'pointer',
                                }}
                              >
                                <X size={11} />
                              </button>
                            </div>
                          ) : (
                            <button
                              onClick={e => {
                                e.stopPropagation()
                                setDeleteConfirmId(sess.id)
                              }}
                              title="Delete chat"
                              style={{
                                background: 'none',
                                border: 'none',
                                cursor: 'pointer',
                                padding: '2px 4px',
                                color: 'hsl(var(--pencil))',
                                borderRadius: '4px',
                              }}
                            >
                              <Trash2 size={12} />
                            </button>
                          )}
                        </div>
                      </div>
                    )}

                    {/* Metadata Subtitle: Time & Meeting pill */}
                    <div style={{ display: 'flex', alignItems: 'center', gap: '.4rem', fontSize: '.7rem', color: 'hsl(var(--pencil))' }}>
                      <span>{fmtRelativeTime(sess.updated_at || sess.created_at)}</span>
                      {sess.messages.length > 0 && (
                        <span>• {sess.messages.length} msg{sess.messages.length > 1 ? 's' : ''}</span>
                      )}
                      {sess.selected_meeting_id && (
                        <span style={{
                          marginLeft: 'auto',
                          background: 'hsl(var(--paper-deep))',
                          padding: '1px 5px',
                          borderRadius: '4px',
                          fontSize: '.65rem',
                          color: 'hsl(var(--accent))',
                          fontWeight: 600,
                          maxWidth: '90px',
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap',
                        }}>
                          📎 {availableMeetings.find(m => m.id === sess.selected_meeting_id)?.name || 'Meeting'}
                        </span>
                      )}
                    </div>
                  </div>
                )
              })
            )}
          </div>
        </div>

        {/* ────────────────────────────────────────────────────────────
            2. CENTER CHAT AREA (INDEPENDENT SMOOTH SCROLL + FIXED BOTTOM INPUT)
        ──────────────────────────────────────────────────────────── */}
        <div style={{
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          minWidth: 0,
          height: '100%',
          overflow: 'hidden',
          background: 'linear-gradient(170deg, hsl(var(--paper-deep)), hsl(var(--paper)))',
        }}>

          {/* ═══════ Scrollable Messages Container ═══════ */}
          <div
            ref={chatScrollRef}
            style={{
              flex: 1,
              overflowY: 'auto',
              minHeight: 0,
              padding: '1.5rem 2rem',
              display: 'flex',
              flexDirection: 'column',
              gap: '.75rem',
              scrollBehavior: 'smooth',
            }}
          >
            {loading ? (
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', flex: 1 }}>
                <Loader size={26} className="spin" style={{ color: 'hsl(var(--accent))' }} />
              </div>
            ) : currentMessages.length === 0 && !streaming ? (
              <div style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                flex: 1,
                gap: '1.25rem',
                textAlign: 'center',
                padding: '2rem 1rem',
                maxWidth: '640px',
                margin: 'auto',
              }}>
                <div style={{
                  width: '68px',
                  height: '68px',
                  borderRadius: '50%',
                  background: 'linear-gradient(135deg, hsl(270 80% 60% / .15), hsl(200 80% 60% / .15))',
                  border: '2px dashed hsl(270 60% 60% / .3)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}>
                  <Sparkles size={30} style={{ color: 'hsl(270 60% 60%)' }} />
                </div>

                <div>
                  <h3 style={{ fontSize: '1.2rem', fontWeight: 700, color: 'hsl(var(--ink))', margin: '0 0 .35rem' }}>
                    Ask anything about your meetings
                  </h3>
                  <p style={{ fontSize: '.88rem', color: 'hsl(var(--pencil))', lineHeight: 1.5, margin: 0 }}>
                    AI searches discussion points and meeting summaries to provide synthesized answers with source citations.
                  </p>
                  {selectedMeetingId && (
                    <div style={{
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: '.4rem',
                      fontSize: '.82rem',
                      color: 'hsl(var(--accent))',
                      fontWeight: 600,
                      marginTop: '.75rem',
                      background: 'hsl(var(--accent) / .1)',
                      padding: '.25rem .75rem',
                      borderRadius: '8px',
                    }}>
                      <FileText size={13} />
                      Focused on: {selectedMeeting?.name || 'Selected meeting'}
                    </div>
                  )}
                </div>

                {/* Suggested prompt chips */}
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
                  gap: '.6rem',
                  width: '100%',
                  marginTop: '.5rem',
                }}>
                  {suggestedQueries.map((item, idx) => (
                    <div
                      key={idx}
                      onClick={() => handleSend(item.prompt)}
                      style={{
                        padding: '.75rem .9rem',
                        borderRadius: '10px',
                        background: 'hsl(var(--card))',
                        border: '1.5px solid hsl(var(--border) / .2)',
                        textAlign: 'left',
                        cursor: 'pointer',
                        transition: 'all .15s ease',
                      }}
                      onMouseEnter={e => {
                        (e.currentTarget as HTMLElement).style.borderColor = 'hsl(var(--accent) / .5)';
                        (e.currentTarget as HTMLElement).style.transform = 'translateY(-1px)'
                      }}
                      onMouseLeave={e => {
                        (e.currentTarget as HTMLElement).style.borderColor = 'hsl(var(--border) / .2)';
                        (e.currentTarget as HTMLElement).style.transform = 'translateY(0)'
                      }}
                    >
                      <div style={{ fontSize: '.82rem', fontWeight: 600, color: 'hsl(var(--ink))', marginBottom: '.2rem' }}>
                        {item.label}
                      </div>
                      <div style={{ fontSize: '.74rem', color: 'hsl(var(--pencil))', lineHeight: 1.4, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        "{item.prompt}"
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <>
                {currentMessages.map(msg => (
                  <div
                    key={msg.id}
                    style={{
                      display: 'flex',
                      justifyContent: msg.role === 'user' ? 'flex-end' : 'flex-start',
                      gap: '.5rem',
                    }}
                  >
                    {msg.role === 'assistant' && (
                      <div style={{
                        width: '32px',
                        height: '32px',
                        borderRadius: '10px',
                        flexShrink: 0,
                        background: 'linear-gradient(135deg, hsl(270 70% 55% / .18), hsl(200 70% 55% / .18))',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        border: '1.5px solid hsl(270 60% 55% / .25)',
                        marginTop: '2px',
                      }}>
                        <Bot size={16} style={{ color: 'hsl(270 60% 55%)' }} />
                      </div>
                    )}

                    <div style={{
                      maxWidth: msg.role === 'user' ? '75%' : '85%',
                      padding: msg.role === 'user' ? '.65rem 1rem' : '.85rem 1.25rem',
                      borderRadius: msg.role === 'user' ? '14px 14px 4px 14px' : '14px 14px 14px 4px',
                      background: msg.role === 'user'
                        ? 'linear-gradient(135deg, hsl(var(--accent)), hsl(var(--accent) / .85))'
                        : 'hsl(var(--card))',
                      color: msg.role === 'user' ? 'white' : 'hsl(var(--ink))',
                      border: msg.role === 'user' ? 'none' : '1px solid hsl(var(--border) / .15)',
                      fontSize: '.88rem',
                      lineHeight: 1.6,
                      boxShadow: msg.role === 'user'
                        ? '0 2px 8px hsl(var(--accent) / .3)'
                        : '0 1px 3px hsl(var(--ink) / .04)',
                    }}>
                      {msg.role === 'assistant' ? (
                        <div>
                          <div style={{ fontSize: '.88rem' }}>{renderMarkdown(msg.content)}</div>
                          <div style={{
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'space-between',
                            marginTop: '.6rem',
                            paddingTop: '.5rem',
                            borderTop: '1px solid hsl(var(--border) / .1)',
                          }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '.4rem' }}>
                              {msg.metadata?.meeting_sources && msg.metadata.meeting_sources.length > 0 ? (
                                <button
                                  className="btn btn-ghost"
                                  onClick={() => {
                                    if (activeSessionId) {
                                      setSessions(prev => prev.map(s => {
                                        if (s.id === activeSessionId) {
                                          return {
                                            ...s,
                                            currentSources: msg.metadata?.meeting_sources || [],
                                            currentQueryType: msg.metadata?.query_type || '',
                                          }
                                        }
                                        return s
                                      }))
                                    }
                                    setRightPanelOpen(true)
                                  }}
                                  style={{
                                    padding: '2px 8px',
                                    fontSize: '.72rem',
                                    height: 'auto',
                                    minHeight: 0,
                                    borderRadius: '6px',
                                    background: 'hsl(var(--accent) / .1)',
                                    color: 'hsl(var(--accent))',
                                    fontWeight: 600,
                                  }}
                                  title="View cited sources in right context panel"
                                >
                                  📎 {msg.metadata.meeting_sources.length} meeting{msg.metadata.meeting_sources.length > 1 ? 's' : ''} cited
                                </button>
                              ) : msg.metadata?.query_type === 'direct' ? (
                                <span style={{
                                  fontSize: '.72rem',
                                  color: 'hsl(var(--pencil))',
                                  padding: '2px 6px',
                                  borderRadius: '4px',
                                  background: 'hsl(var(--muted) / .5)',
                                }}>
                                  ⚡ Direct answer
                                </span>
                              ) : null}
                            </div>
                            <button
                              className="icon-btn"
                              onClick={() => handleCopy(msg.content, msg.id)}
                              title="Copy to clipboard"
                              style={{ width: '28px', height: '28px', color: 'hsl(var(--pencil))' }}
                            >
                              {copied === msg.id ? <Check size={12} style={{ color: 'hsl(var(--success))' }} /> : <Copy size={12} />}
                            </button>
                          </div>
                        </div>
                      ) : (
                        <span dangerouslySetInnerHTML={{ __html: boldify(msg.content) }} />
                      )}
                    </div>
                  </div>
                ))}

                {/* Streaming Assistant Response */}
                {streaming && streamedText && (
                  <div style={{ display: 'flex', gap: '.5rem' }}>
                    <div style={{
                      width: '32px',
                      height: '32px',
                      borderRadius: '10px',
                      flexShrink: 0,
                      background: 'linear-gradient(135deg, hsl(270 70% 55% / .18), hsl(200 70% 55% / .18))',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      border: '1.5px solid hsl(270 60% 55% / .25)',
                      marginTop: '2px',
                    }}>
                      <Bot size={16} style={{ color: 'hsl(270 60% 55%)' }} />
                    </div>
                    <div style={{
                      maxWidth: '85%',
                      padding: '.85rem 1.25rem',
                      borderRadius: '14px 14px 14px 4px',
                      background: 'hsl(var(--card))',
                      border: '1px solid hsl(var(--border) / .15)',
                      fontSize: '.88rem',
                      lineHeight: 1.6,
                    }}>
                      <div>{renderMarkdown(streamedText)}</div>
                      <span className="spin" style={{
                        display: 'inline-block',
                        width: '8px',
                        height: '8px',
                        borderRadius: '50%',
                        background: 'hsl(var(--accent))',
                        marginLeft: '.3rem',
                        animation: 'pulse 1s infinite',
                      }} />
                    </div>
                  </div>
                )}

                {/* Loading State Before Stream Begins */}
                {streaming && !streamedText && (
                  <div style={{ display: 'flex', gap: '.5rem', alignItems: 'center' }}>
                    <div style={{
                      width: '32px',
                      height: '32px',
                      borderRadius: '10px',
                      flexShrink: 0,
                      background: 'linear-gradient(135deg, hsl(270 70% 55% / .18), hsl(200 70% 55% / .18))',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      border: '1.5px solid hsl(270 60% 55% / .25)',
                    }}>
                      <Bot size={16} style={{ color: 'hsl(270 60% 55%)' }} />
                    </div>
                    <div style={{
                      padding: '.65rem 1rem',
                      borderRadius: '14px 14px 14px 4px',
                      background: 'hsl(var(--card))',
                      border: '1px solid hsl(var(--border) / .15)',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '.6rem',
                      fontSize: '.85rem',
                      color: 'hsl(var(--pencil))',
                    }}>
                      <Loader size={14} className="spin" style={{ color: 'hsl(var(--accent))' }} />
                      Searching meetings & generating answer...
                    </div>
                  </div>
                )}
              </>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* ═══════ FIXED BOTTOM INPUT / PROMPT BAR ═══════ */}
          <div style={{
            flexShrink: 0,
            borderTop: '2px solid hsl(var(--border) / .1)',
            background: 'hsl(var(--card) / .7)',
            backdropFilter: 'blur(12px)',
            padding: '0.85rem 2rem 1rem',
            position: 'relative',
            zIndex: 10,
          }}>
            {/* Meeting Focus Indicator Pill */}
            {selectedMeetingId && (
              <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: '.5rem',
                marginBottom: '.5rem',
                padding: '.3rem .75rem',
                borderRadius: '8px',
                background: selectedMeeting?.has_stage2 ? 'hsl(var(--accent) / .08)' : 'hsl(45 90% 50% / .1)',
                border: selectedMeeting?.has_stage2 ? '1px solid hsl(var(--accent) / .2)' : '1px solid hsl(45 90% 50% / .3)',
                fontSize: '.76rem',
                color: selectedMeeting?.has_stage2 ? 'hsl(var(--accent))' : 'hsl(45 80% 40%)',
                fontWeight: 600,
              }}>
                <FileText size={12} />
                <span>Focused: <strong>{selectedMeeting?.name || 'Selected meeting'}</strong></span>
                {selectedMeeting && selectedMeeting.has_stage2 ? (
                  <span style={{ fontSize: '.65rem', background: 'hsl(142 70% 45% / .2)', color: 'hsl(142 70% 30%)', padding: '1px 5px', borderRadius: '4px' }}>
                    Stage 2 Ready
                  </span>
                ) : (
                  <span style={{ fontSize: '.68rem', fontWeight: 500, opacity: 0.9 }}>
                    (No Stage 2 points indexed for this meeting)
                  </span>
                )}
                <button
                  onClick={() => handleSetMeetingForActiveSession('')}
                  style={{ marginLeft: 'auto', background: 'none', border: 'none', cursor: 'pointer', color: 'inherit', padding: '2px' }}
                  title="Clear meeting filter"
                >
                  <X size={12} />
                </button>
              </div>
            )}

            {/* Input Row */}
            <div style={{ display: 'flex', gap: '.5rem', alignItems: 'center' }}>
              <div style={{
                flex: 1,
                position: 'relative',
                background: 'hsl(var(--paper))',
                borderRadius: '12px',
                border: '2px solid hsl(var(--border) / .2)',
                transition: 'border-color .2s',
              }}>
                <input
                  ref={inputRef}
                  type="text"
                  value={input}
                  onChange={e => setInput(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder={
                    selectedMeetingId
                      ? `Ask questions about ${selectedMeeting?.name || 'this meeting'}...`
                      : 'Ask questions about your meetings...'
                  }
                  style={{
                    width: '100%',
                    border: 'none',
                    background: 'transparent',
                    padding: '.75rem 1rem',
                    fontSize: '.9rem',
                    color: 'hsl(var(--ink))',
                    outline: 'none',
                    fontFamily: 'Inter, sans-serif',
                  }}
                />
              </div>

              {/* Action Button: Send or Stop */}
              {streaming ? (
                <button
                  onClick={handleStopStreaming}
                  className="btn"
                  title="Stop generating"
                  style={{
                    width: '42px',
                    height: '42px',
                    padding: 0,
                    borderRadius: '12px',
                    flexShrink: 0,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    background: 'hsl(var(--destructive))',
                    color: 'white',
                    border: 'none',
                    cursor: 'pointer',
                  }}
                >
                  <Square size={15} fill="currentColor" />
                </button>
              ) : (
                <button
                  onClick={() => handleSend()}
                  disabled={!input.trim()}
                  className="btn btn-primary"
                  title="Send message (Enter)"
                  style={{
                    width: '42px',
                    height: '42px',
                    padding: 0,
                    borderRadius: '12px',
                    flexShrink: 0,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    opacity: !input.trim() ? 0.45 : 1,
                    cursor: !input.trim() ? 'not-allowed' : 'pointer',
                  }}
                >
                  <Send size={16} />
                </button>
              )}
            </div>

            {/* Helper Micro-copy */}
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              marginTop: '.45rem',
              fontSize: '.68rem',
              color: 'hsl(var(--pencil))',
              padding: '0 .25rem',
            }}>
              <span>Press <strong>Enter</strong> to send • AI searches Stage 2 indexed points</span>
              <span>{sessions.length} active session{sessions.length > 1 ? 's' : ''}</span>
            </div>
          </div>
        </div>

        {/* ────────────────────────────────────────────────────────────
            3. RIGHT CONTEXT PANEL (MEETING SOURCES)
        ──────────────────────────────────────────────────────────── */}
        <div style={{
          width: rightPanelOpen ? '280px' : '0px',
          minWidth: rightPanelOpen ? '280px' : '0px',
          maxWidth: '280px',
          transition: 'width 0.22s cubic-bezier(0.4, 0, 0.2, 1), min-width 0.22s cubic-bezier(0.4, 0, 0.2, 1)',
          borderLeft: rightPanelOpen ? '2px solid hsl(var(--border) / .12)' : 'none',
          background: 'hsl(var(--card) / .45)',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
          flexShrink: 0,
        }}>
          {/* Sources Header */}
          <div style={{
            padding: '1rem 1.25rem .75rem',
            borderBottom: '1px solid hsl(var(--border) / .1)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
          }}>
            <h3 style={{ fontSize: '.86rem', fontWeight: 700, color: 'hsl(var(--ink))', margin: 0, display: 'flex', alignItems: 'center', gap: '.4rem' }}>
              <FileText size={14} style={{ color: 'hsl(var(--accent))' }} />
              Meeting Sources
            </h3>
            {currentSources.length > 0 && (
              <span style={{
                fontSize: '.68rem',
                fontWeight: 600,
                color: 'hsl(var(--accent))',
                background: 'hsl(var(--accent) / .1)',
                padding: '1px 6px',
                borderRadius: '4px',
              }}>
                {currentSources.length} cited
              </span>
            )}
          </div>

          {/* Sources List */}
          <div style={{ flex: 1, overflowY: 'auto', padding: '1rem 1.15rem' }}>
            {currentSources.length > 0 ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '.65rem' }}>
                {currentSources.map(source => (
                  <div
                    key={source.id}
                    style={{
                      padding: '.8rem .95rem',
                      borderRadius: '10px',
                      background: 'hsl(var(--card))',
                      border: '1.5px solid hsl(var(--border) / .15)',
                      transition: 'all .2s',
                      cursor: 'pointer',
                    }}
                    onClick={() => {
                      if (source.is_manual_mom || source.id?.startsWith('mom_')) {
                        setManualMomModalOpen(true)
                      } else {
                        navigate(`/dashboard/history/${source.id}`)
                      }
                    }}
                    onMouseEnter={e => {
                      (e.currentTarget as HTMLElement).style.borderColor = 'hsl(var(--accent) / .4)';
                      (e.currentTarget as HTMLElement).style.transform = 'translateY(-1px)'
                    }}
                    onMouseLeave={e => {
                      (e.currentTarget as HTMLElement).style.borderColor = 'hsl(var(--border) / .15)';
                      (e.currentTarget as HTMLElement).style.transform = 'translateY(0)'
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'flex-start', gap: '.6rem' }}>
                      <div style={{
                        width: '28px',
                        height: '28px',
                        borderRadius: '8px',
                        flexShrink: 0,
                        background: (source.is_manual_mom || source.id?.startsWith('mom_'))
                          ? 'linear-gradient(135deg, hsl(280 80% 55% / .2), hsl(280 80% 55% / .1))'
                          : 'linear-gradient(135deg, hsl(var(--accent) / .15), hsl(var(--accent) / .08))',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        border: (source.is_manual_mom || source.id?.startsWith('mom_'))
                          ? '1.5px solid hsl(280 80% 55% / .3)'
                          : '1.5px solid hsl(var(--accent) / .25)',
                      }}>
                        <FileText
                          size={13}
                          style={{
                            color: (source.is_manual_mom || source.id?.startsWith('mom_'))
                              ? 'hsl(280 80% 55%)'
                              : 'hsl(var(--accent))',
                          }}
                        />
                      </div>
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '.35rem', flexWrap: 'wrap' }}>
                          <span style={{
                            fontSize: '.82rem',
                            fontWeight: 600,
                            color: 'hsl(var(--ink))',
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                            whiteSpace: 'nowrap',
                          }}>
                            {source.name}
                          </span>
                          {(source.is_manual_mom || source.id?.startsWith('mom_')) && (
                            <span style={{
                              fontSize: '.58rem',
                              fontWeight: 700,
                              padding: '1px 5px',
                              borderRadius: '4px',
                              background: 'hsl(280 80% 50% / .18)',
                              color: 'hsl(280 80% 40%)',
                              flexShrink: 0,
                            }}>
                              Manual MoM
                            </span>
                          )}
                        </div>
                        {source.date && (
                          <div style={{ fontSize: '.71rem', color: 'hsl(var(--pencil))', marginTop: '.2rem', fontWeight: 500 }}>
                            {fmtDate(source.date)}
                          </div>
                        )}
                      </div>
                      <ChevronRight size={14} style={{ color: 'hsl(var(--accent))', flexShrink: 0, marginTop: '2px' }} />
                    </div>
                  </div>
                ))}
              </div>
            ) : currentQueryType === 'direct' ? (
              <div style={{
                textAlign: 'center',
                padding: '2rem 1rem',
                color: 'hsl(var(--pencil))',
                fontSize: '.82rem',
                lineHeight: 1.5,
              }}>
                <Bot size={24} style={{ color: 'hsl(var(--pencil))', opacity: 0.4, marginBottom: '.75rem' }} />
                <p style={{ fontWeight: 600, marginBottom: '.25rem' }}>Direct answer</p>
                <p style={{ opacity: 0.7, margin: 0 }}>This response was generated directly without needing meeting context.</p>
              </div>
            ) : (
              <div style={{
                textAlign: 'center',
                padding: '2rem 1rem',
                color: 'hsl(var(--pencil))',
                fontSize: '.82rem',
                lineHeight: 1.5,
              }}>
                <Sparkles size={24} style={{ color: 'hsl(var(--pencil))', opacity: 0.3, marginBottom: '.75rem' }} />
                <p style={{ fontWeight: 600, marginBottom: '.25rem' }}>Meeting sources</p>
                <p style={{ opacity: 0.7, margin: 0 }}>Citations from relevant meetings will appear here when questions are answered.</p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* ═══════ Resync Modal ═══════ */}
      <MeetingResyncModal
        isOpen={resyncModalOpen}
        onClose={() => setResyncModalOpen(false)}
        onSyncCompleted={refreshMeetings}
      />

      {/* ═══════ Manual MoM Library & Upload Modal ═══════ */}
      <ManualMomLibraryModal
        isOpen={manualMomModalOpen}
        onClose={() => setManualMomModalOpen(false)}
        onSelectMeetingForChat={(mid) => handleSetMeetingForActiveSession(mid)}
        onMoMsChanged={refreshMeetings}
      />
    </div>
  )
}
