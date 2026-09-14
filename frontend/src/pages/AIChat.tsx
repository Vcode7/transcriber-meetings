import React, { useState, useEffect, useRef, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Bot, Send, Loader, Trash2, Copy, Check, Download,
  MessageSquare, FileText, ChevronRight, ExternalLink, Sparkles, X,
  Search, ChevronDown, Filter, Zap,
} from 'lucide-react'
import type { AIChatMeeting, AIChatMessage } from '../types/recording'
import {
  streamAIChat, getAIChatMeetings, getAIChatHistory, clearAIChatHistory,
} from '../api/aiChat'

export default function AIChat() {
  const navigate = useNavigate()
  const [messages, setMessages] = useState<AIChatMessage[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(true)
  const [streaming, setStreaming] = useState(false)
  const [streamedText, setStreamedText] = useState('')

  // Meeting selector
  const [availableMeetings, setAvailableMeetings] = useState<AIChatMeeting[]>([])
  const [selectedMeetingId, setSelectedMeetingId] = useState<string>('')
  const [meetingDropdownOpen, setMeetingDropdownOpen] = useState(false)
  const [meetingSearchQuery, setMeetingSearchQuery] = useState('')
  const [onlyStage2, setOnlyStage2] = useState(false)
  const dropdownRef = useRef<HTMLDivElement>(null)

  // Sources sidebar
  const [currentSources, setCurrentSources] = useState<AIChatMeeting[]>([])
  const [currentQueryType, setCurrentQueryType] = useState<string>('')

  // Misc
  const [copied, setCopied] = useState<string | null>(null)
  const [confirmClear, setConfirmClear] = useState(false)

  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const abortRef = useRef<AbortController | null>(null)

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

  // Load meetings and chat history on mount
  useEffect(() => {
    let cancelled = false
    const load = async () => {
      try {
        const [history, meetings] = await Promise.all([
          getAIChatHistory(),
          getAIChatMeetings(),
        ])
        if (!cancelled) {
          setMessages(history)
          setAvailableMeetings(meetings)

          // Set current sources from the last assistant message
          const lastAssistant = [...history].reverse().find(m => m.role === 'assistant')
          if (lastAssistant?.metadata?.meeting_sources) {
            setCurrentSources(lastAssistant.metadata.meeting_sources)
            setCurrentQueryType(lastAssistant.metadata.query_type || '')
          }
        }
      } catch (err) {
        console.error('Failed to load AI Chat data:', err)
      } finally {
        if (!cancelled) setLoading(false)
      }
    }
    load()
    return () => { cancelled = true }
  }, [])

  // Auto-scroll
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, streamedText])

  const handleCopy = useCallback((text: string, id: string) => {
    navigator.clipboard.writeText(text)
    setCopied(id)
    setTimeout(() => setCopied(null), 2000)
  }, [])

  const handleClearHistory = useCallback(async () => {
    try {
      await clearAIChatHistory()
      setMessages([])
      setCurrentSources([])
      setCurrentQueryType('')
      setConfirmClear(false)
    } catch (err) {
      console.error('Failed to clear history:', err)
    }
  }, [])

  const handleSend = useCallback(async () => {
    const msg = input.trim()
    if (!msg || streaming) return

    setInput('')
    setStreaming(true)
    setStreamedText('')

    const userMsg: AIChatMessage = {
      id: `temp-${Date.now()}`,
      role: 'user',
      content: msg,
      metadata: {},
      created_at: new Date().toISOString(),
    }
    setMessages(prev => [...prev, userMsg])

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
          if (meta.meeting_sources) {
            setCurrentSources(meta.meeting_sources)
          }
          setCurrentQueryType(meta.query_type || '')
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
            setMessages(prev => [...prev, assistantMsg])
          }
          setStreamedText('')
          setStreaming(false)
        },
        controller.signal,
        selectedMeetingId || undefined,
      )
    } catch (err: any) {
      if (err.name !== 'AbortError') {
        setMessages(prev => [...prev, {
          id: `err-${Date.now()}`, role: 'assistant',
          content: '⚠️ Failed to get AI response. Please try again.',
          metadata: {}, created_at: new Date().toISOString(),
        }])
      }
      setStreamedText('')
      setStreaming(false)
    }
  }, [input, streaming, selectedMeetingId])

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  // Simple markdown renderer (bold, headers, bullets)
  const renderMarkdown = (text: string) => {
    const lines = text.split('\n')
    return lines.map((line, i) => {
      if (line.startsWith('### ')) return <h4 key={i} style={{ fontSize: '.95rem', fontWeight: 700, margin: '1rem 0 .5rem', color: 'hsl(var(--ink))' }}>{line.slice(4)}</h4>
      if (line.startsWith('## ')) return <h3 key={i} style={{ fontSize: '1.05rem', fontWeight: 700, margin: '1.25rem 0 .5rem', color: 'hsl(var(--accent))', borderBottom: '1px solid hsl(var(--accent) / .15)', paddingBottom: '.35rem' }}>{line.slice(3)}</h3>
      if (line.startsWith('# ')) return <h2 key={i} style={{ fontSize: '1.15rem', fontWeight: 700, margin: '1.5rem 0 .75rem', color: 'hsl(var(--ink))' }}>{line.slice(2)}</h2>
      if (line.match(/^[-•]\s/)) {
        const content = line.replace(/^[-•]\s/, '')
        return <div key={i} style={{ display: 'flex', gap: '.5rem', margin: '.25rem 0', lineHeight: 1.6 }}>
          <span style={{ color: 'hsl(var(--accent))', flexShrink: 0, marginTop: '2px' }}>•</span>
          <span dangerouslySetInnerHTML={{ __html: boldify(content) }} />
        </div>
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

  const fmtDate = (iso: string) => {
    if (!iso) return ''
    try {
      const d = new Date(iso)
      return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })
    } catch { return iso }
  }

  const filteredMeetings = availableMeetings.filter(m => {
    if (onlyStage2 && !m.has_stage2) return false
    if (!meetingSearchQuery.trim()) return true
    const q = meetingSearchQuery.toLowerCase()
    const nameMatch = m.name.toLowerCase().includes(q)
    const dateMatch = fmtDate(m.date).toLowerCase().includes(q)
    return nameMatch || dateMatch
  })

  const stage2Count = availableMeetings.filter(m => m.has_stage2).length
  const selectedMeeting = availableMeetings.find(m => m.id === selectedMeetingId)

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>
      {/* ═══════ Header ═══════ */}
      <div style={{
        padding: '1.25rem 2rem',
        borderBottom: '2px solid hsl(var(--border) / .1)',
        background: 'hsl(var(--card) / .6)',
        backdropFilter: 'blur(12px)',
        flexShrink: 0,
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        position: 'relative',
        zIndex: 20,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '.75rem' }}>
          <div style={{
            width: '40px', height: '40px', borderRadius: '12px',
            background: 'linear-gradient(135deg, hsl(270 80% 60% / .2), hsl(200 80% 60% / .2))',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            border: '1.5px solid hsl(270 60% 60% / .3)',
          }}>
            <MessageSquare size={20} style={{ color: 'hsl(270 60% 60%)' }} />
          </div>
          <div>
            <h2 style={{ fontSize: '1.25rem', fontWeight: 700, margin: 0, color: 'hsl(var(--ink))' }}>
              AI Chat
            </h2>
            <p style={{ fontSize: '.78rem', color: 'hsl(var(--pencil))', margin: 0, fontWeight: 500 }}>
              Ask questions about your meetings
            </p>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '.75rem' }}>
          {/* Searchable Meeting Selector */}
          <div ref={dropdownRef} style={{ position: 'relative' }}>
            <button
              className="btn btn-ghost"
              onClick={() => setMeetingDropdownOpen(prev => !prev)}
              style={{
                height: '38px',
                fontSize: '.85rem',
                padding: '0 .75rem',
                minWidth: '220px',
                maxWidth: '340px',
                borderRadius: '10px',
                border: selectedMeetingId
                  ? '2px solid hsl(var(--accent) / .5)'
                  : '2px solid hsl(var(--border) / .25)',
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
              title={selectedMeeting ? `Focused on: ${selectedMeeting.name}` : 'Select a meeting to focus context'}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '.4rem', minWidth: 0, flex: 1 }}>
                <FileText size={15} style={{ color: selectedMeetingId ? 'hsl(var(--accent))' : 'hsl(var(--pencil))', flexShrink: 0 }} />
                <span style={{
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                  fontWeight: selectedMeetingId ? 600 : 500,
                  fontSize: '.84rem',
                  color: selectedMeetingId ? 'hsl(var(--ink))' : 'hsl(var(--pencil))',
                }}>
                  {selectedMeeting ? selectedMeeting.name : 'All Meetings'}
                </span>
                {selectedMeeting && selectedMeeting.has_stage2 && (
                  <span style={{
                    fontSize: '.65rem',
                    fontWeight: 700,
                    padding: '1px 5px',
                    borderRadius: '4px',
                    background: 'hsl(142 70% 45% / .18)',
                    color: 'hsl(142 70% 35%)',
                    flexShrink: 0,
                  }}>
                    Stage 2
                  </span>
                )}
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '.25rem', flexShrink: 0 }}>
                {selectedMeetingId && (
                  <span
                    onClick={(e) => {
                      e.stopPropagation()
                      setSelectedMeetingId('')
                    }}
                    title="Clear selection"
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
                width: '360px',
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
                  padding: '.65rem .75rem .45rem',
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
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  fontSize: '.76rem',
                }}>
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
                        width: '14px',
                        height: '14px',
                      }}
                    />
                    <span style={{ display: 'flex', alignItems: 'center', gap: '.3rem' }}>
                      <Zap size={12} style={{ color: onlyStage2 ? 'hsl(var(--accent))' : 'hsl(var(--pencil))' }} />
                      Only show Stage 2 points
                    </span>
                  </label>
                  <span style={{
                    fontSize: '.7rem',
                    color: 'hsl(var(--pencil))',
                    background: 'hsl(var(--paper))',
                    padding: '1px 6px',
                    borderRadius: '4px',
                    border: '1px solid hsl(var(--border) / .15)',
                  }}>
                    {stage2Count} ready
                  </span>
                </div>

                {/* List items */}
                <div style={{
                  maxHeight: '260px',
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
                        setSelectedMeetingId('')
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
                      onMouseEnter={(e) => {
                        if (selectedMeetingId) (e.currentTarget as HTMLElement).style.background = 'hsl(var(--paper-deep))'
                      }}
                      onMouseLeave={(e) => {
                        if (selectedMeetingId) (e.currentTarget as HTMLElement).style.background = 'transparent'
                      }}
                    >
                      <div style={{ display: 'flex', flexDirection: 'column' }}>
                        <span style={{ display: 'flex', alignItems: 'center', gap: '.4rem' }}>
                          <Sparkles size={13} style={{ color: 'hsl(var(--accent))' }} />
                          All Meetings
                        </span>
                        <span style={{ fontSize: '.7rem', color: 'hsl(var(--pencil))', marginLeft: '1.2rem', marginTop: '1px' }}>
                          Global search & auto-triage across all meetings
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
                            setSelectedMeetingId(m.id)
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
                          onMouseEnter={(e) => {
                            if (!isSelected) (e.currentTarget as HTMLElement).style.background = 'hsl(var(--paper-deep))'
                          }}
                          onMouseLeave={(e) => {
                            if (!isSelected) (e.currentTarget as HTMLElement).style.background = 'transparent'
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
                              {m.has_stage2 ? (
                                <span style={{
                                  fontSize: '.64rem',
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
                                  fontSize: '.64rem',
                                  fontWeight: 500,
                                  padding: '1px 5px',
                                  borderRadius: '4px',
                                  background: 'hsl(var(--muted) / .5)',
                                  color: 'hsl(var(--pencil))',
                                }}>
                                  No Stage 2
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
                      padding: '1.5rem 1rem',
                      textAlign: 'center',
                      color: 'hsl(var(--pencil))',
                      fontSize: '.78rem',
                    }}>
                      {onlyStage2 && stage2Count === 0 ? (
                        <div>
                          <p style={{ fontWeight: 600, margin: '0 0 .25rem' }}>No Stage 2 points generated</p>
                          <p style={{ margin: 0, opacity: 0.8 }}>Run Stage 2 on meetings in History to index them.</p>
                        </div>
                      ) : (
                        <div>
                          <p style={{ fontWeight: 600, margin: '0 0 .25rem' }}>No meetings found</p>
                          <p style={{ margin: 0, opacity: 0.8 }}>Try a different search or uncheck the filter.</p>
                        </div>
                      )}
                    </div>
                  )}
                </div>

                {/* Dropdown footer info */}
                <div style={{
                  padding: '.4rem .75rem',
                  borderTop: '1px solid hsl(var(--border) / .1)',
                  background: 'hsl(var(--paper))',
                  fontSize: '.7rem',
                  color: 'hsl(var(--pencil))',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                }}>
                  <span>
                    Showing {filteredMeetings.length} of {availableMeetings.length} meetings
                  </span>
                  {onlyStage2 && (
                    <button
                      onClick={() => setOnlyStage2(false)}
                      style={{
                        background: 'none',
                        border: 'none',
                        color: 'hsl(var(--accent))',
                        cursor: 'pointer',
                        padding: 0,
                        fontSize: '.7rem',
                        fontWeight: 600,
                      }}
                    >
                      Show all
                    </button>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Clear History */}
          {messages.length > 0 && (
            confirmClear ? (
              <div style={{
                display: 'flex', alignItems: 'center', gap: '.3rem',
                background: 'hsl(var(--card))', padding: '.2rem .4rem',
                borderRadius: '8px', border: '1px solid hsl(var(--destructive) / .3)',
                fontSize: '.78rem',
              }}>
                <span style={{ fontWeight: 600 }}>Clear?</span>
                <button className="btn" onClick={handleClearHistory}
                  style={{ padding: '.15rem .45rem', background: 'hsl(var(--destructive))', color: 'white', minHeight: 0, height: 'auto', fontSize: '.75rem' }}>
                  Yes
                </button>
                <button className="btn btn-ghost" onClick={() => setConfirmClear(false)}
                  style={{ padding: '.15rem .45rem', minHeight: 0, height: 'auto', fontSize: '.75rem' }}>
                  No
                </button>
              </div>
            ) : (
              <button className="icon-btn" onClick={() => setConfirmClear(true)}
                title="Clear chat history"
                style={{ width: '36px', height: '36px', color: 'hsl(var(--pencil))' }}>
                <Trash2 size={16} />
              </button>
            )
          )}
        </div>
      </div>

      {/* ═══════ Main Content ═══════ */}
      <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>

        {/* ═══════ Chat Area ═══════ */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>

          {/* Messages */}
          <div style={{
            flex: 1, overflowY: 'auto', padding: '1.5rem 2rem',
            display: 'flex', flexDirection: 'column', gap: '.75rem',
            minHeight: 0,
            background: 'linear-gradient(170deg, hsl(var(--paper-deep)), hsl(var(--paper)))',
          }}>
            {loading ? (
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', flex: 1 }}>
                <Loader size={24} className="spin" style={{ color: 'hsl(var(--accent))' }} />
              </div>
            ) : messages.length === 0 && !streaming ? (
              <div style={{
                display: 'flex', flexDirection: 'column', alignItems: 'center',
                justifyContent: 'center', flex: 1, gap: '1rem', textAlign: 'center',
                padding: '2rem',
              }}>
                <div style={{
                  width: '72px', height: '72px', borderRadius: '50%',
                  background: 'linear-gradient(135deg, hsl(270 80% 60% / .1), hsl(200 80% 60% / .1))',
                  border: '2px dashed hsl(270 60% 60% / .25)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                }}>
                  <Sparkles size={32} style={{ color: 'hsl(270 60% 60%)', opacity: 0.6 }} />
                </div>
                <div>
                  <h3 style={{ fontSize: '1.15rem', fontWeight: 700, color: 'hsl(var(--ink))', margin: '0 0 .35rem' }}>
                    Ask anything about your meetings
                  </h3>
                  <p style={{ fontSize: '.9rem', color: 'hsl(var(--pencil))', lineHeight: 1.5, maxWidth: '400px', margin: '0 auto' }}>
                    AI will search your meeting discussion points to find relevant information and answer your questions.
                  </p>
                  {selectedMeetingId && (
                    <p style={{ fontSize: '.82rem', color: 'hsl(var(--accent))', fontWeight: 600, marginTop: '.75rem' }}>
                      🎯 Focused on: {availableMeetings.find(m => m.id === selectedMeetingId)?.name || 'Selected meeting'}
                    </p>
                  )}
                </div>
              </div>
            ) : (
              <>
                {messages.map(msg => (
                  <div key={msg.id} style={{ display: 'flex', justifyContent: msg.role === 'user' ? 'flex-end' : 'flex-start', gap: '.5rem' }}>
                    {msg.role === 'assistant' && (
                      <div style={{
                        width: '30px', height: '30px', borderRadius: '10px', flexShrink: 0,
                        background: 'linear-gradient(135deg, hsl(270 70% 55% / .15), hsl(200 70% 55% / .15))',
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                        border: '1.5px solid hsl(270 60% 55% / .25)', marginTop: '2px',
                      }}>
                        <Bot size={15} style={{ color: 'hsl(270 60% 55%)' }} />
                      </div>
                    )}
                    <div style={{
                      maxWidth: msg.role === 'user' ? '75%' : '85%',
                      padding: msg.role === 'user' ? '.65rem 1rem' : '.75rem 1.25rem',
                      borderRadius: msg.role === 'user' ? '14px 14px 4px 14px' : '14px 14px 14px 4px',
                      background: msg.role === 'user'
                        ? 'linear-gradient(135deg, hsl(var(--accent)), hsl(var(--accent) / .85))'
                        : 'hsl(var(--card))',
                      color: msg.role === 'user' ? 'white' : 'hsl(var(--ink))',
                      border: msg.role === 'user' ? 'none' : '1px solid hsl(var(--border) / .15)',
                      fontSize: '.88rem', lineHeight: 1.6,
                      boxShadow: msg.role === 'user'
                        ? '0 2px 8px hsl(var(--accent) / .3)'
                        : '0 1px 3px hsl(var(--ink) / .04)',
                    }}>
                      {msg.role === 'assistant' ? (
                        <div>
                          <div style={{ fontSize: '.88rem' }}>{renderMarkdown(msg.content)}</div>
                          <div style={{
                            display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                            marginTop: '.6rem', paddingTop: '.5rem',
                            borderTop: '1px solid hsl(var(--border) / .1)',
                          }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '.4rem' }}>
                              {msg.metadata?.meeting_sources && msg.metadata.meeting_sources.length > 0 ? (
                                <button
                                  className="btn btn-ghost"
                                  onClick={() => {
                                    setCurrentSources(msg.metadata.meeting_sources || [])
                                    setCurrentQueryType(msg.metadata.query_type || '')
                                  }}
                                  style={{
                                    padding: '2px 8px', fontSize: '.72rem', height: 'auto', minHeight: 0,
                                    borderRadius: '6px', background: 'hsl(var(--accent) / .1)',
                                    color: 'hsl(var(--accent))', fontWeight: 600,
                                  }}
                                  title="View sources for this answer in sidebar"
                                >
                                  📎 {msg.metadata.meeting_sources.length} meeting{msg.metadata.meeting_sources.length > 1 ? 's' : ''} cited
                                </button>
                              ) : msg.metadata?.query_type === 'direct' ? (
                                <span style={{
                                  fontSize: '.72rem', color: 'hsl(var(--pencil))',
                                  padding: '2px 6px', borderRadius: '4px',
                                  background: 'hsl(var(--muted) / .5)',
                                }}>
                                  ⚡ Direct answer
                                </span>
                              ) : null}
                            </div>
                            <button className="icon-btn"
                              onClick={() => handleCopy(msg.content, msg.id)}
                              title="Copy to clipboard"
                              style={{ width: '28px', height: '28px', color: 'hsl(var(--pencil))' }}>
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

                {/* Streaming indicator */}
                {streaming && streamedText && (
                  <div style={{ display: 'flex', gap: '.5rem' }}>
                    <div style={{
                      width: '30px', height: '30px', borderRadius: '10px', flexShrink: 0,
                      background: 'linear-gradient(135deg, hsl(270 70% 55% / .15), hsl(200 70% 55% / .15))',
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      border: '1.5px solid hsl(270 60% 55% / .25)', marginTop: '2px',
                    }}>
                      <Bot size={15} style={{ color: 'hsl(270 60% 55%)' }} />
                    </div>
                    <div style={{
                      maxWidth: '85%', padding: '.75rem 1.25rem', borderRadius: '14px 14px 14px 4px',
                      background: 'hsl(var(--card))', border: '1px solid hsl(var(--border) / .15)',
                      fontSize: '.88rem', lineHeight: 1.6,
                    }}>
                      <div>{renderMarkdown(streamedText)}</div>
                      <span className="spin" style={{
                        display: 'inline-block', width: '8px', height: '8px',
                        borderRadius: '50%', background: 'hsl(var(--accent))',
                        marginLeft: '.3rem', animation: 'pulse 1s infinite',
                      }} />
                    </div>
                  </div>
                )}

                {/* Loading (before streaming starts) */}
                {streaming && !streamedText && (
                  <div style={{ display: 'flex', gap: '.5rem', alignItems: 'center' }}>
                    <div style={{
                      width: '30px', height: '30px', borderRadius: '10px', flexShrink: 0,
                      background: 'linear-gradient(135deg, hsl(270 70% 55% / .15), hsl(200 70% 55% / .15))',
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      border: '1.5px solid hsl(270 60% 55% / .25)',
                    }}>
                      <Bot size={15} style={{ color: 'hsl(270 60% 55%)' }} />
                    </div>
                    <div style={{
                      padding: '.65rem 1rem', borderRadius: '14px 14px 14px 4px',
                      background: 'hsl(var(--card))', border: '1px solid hsl(var(--border) / .15)',
                      display: 'flex', alignItems: 'center', gap: '.6rem',
                      fontSize: '.85rem', color: 'hsl(var(--pencil))',
                    }}>
                      <Loader size={14} className="spin" style={{ color: 'hsl(var(--accent))' }} />
                      Thinking...
                    </div>
                  </div>
                )}
              </>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* ═══════ Input Area ═══════ */}
          <div style={{
            borderTop: '2px solid hsl(var(--border) / .1)',
            background: 'hsl(var(--card) / .6)',
            backdropFilter: 'blur(12px)',
            padding: '1rem 2rem',
            flexShrink: 0,
          }}>
            {selectedMeetingId && (
              <div style={{
                display: 'flex', alignItems: 'center', gap: '.5rem', marginBottom: '.6rem',
                padding: '.35rem .75rem', borderRadius: '8px',
                background: selectedMeeting?.has_stage2 ? 'hsl(var(--accent) / .08)' : 'hsl(45 90% 50% / .1)',
                border: selectedMeeting?.has_stage2 ? '1px solid hsl(var(--accent) / .2)' : '1px solid hsl(45 90% 50% / .3)',
                fontSize: '.78rem', color: selectedMeeting?.has_stage2 ? 'hsl(var(--accent))' : 'hsl(45 80% 40%)', fontWeight: 600,
              }}>
                <FileText size={12} />
                Focused: {selectedMeeting?.name || 'Selected meeting'}
                {selectedMeeting && selectedMeeting.has_stage2 ? (
                  <span style={{ fontSize: '.68rem', background: 'hsl(142 70% 45% / .2)', color: 'hsl(142 70% 30%)', padding: '1px 5px', borderRadius: '4px' }}>
                    Stage 2 Ready
                  </span>
                ) : (
                  <span style={{ fontSize: '.72rem', fontWeight: 500, opacity: 0.9 }}>
                    (Note: No Stage 2 points indexed for this meeting)
                  </span>
                )}
                <button
                  onClick={() => setSelectedMeetingId('')}
                  style={{ marginLeft: 'auto', background: 'none', border: 'none', cursor: 'pointer', color: 'inherit', padding: '2px' }}
                  title="Clear meeting filter"
                >
                  <X size={12} />
                </button>
              </div>
            )}
            <div style={{ display: 'flex', gap: '.5rem', alignItems: 'flex-end' }}>
              <div style={{
                flex: 1, position: 'relative',
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
                  placeholder={selectedMeetingId ? 'Ask about this meeting...' : 'Ask about your meetings...'}
                  disabled={streaming}
                  style={{
                    width: '100%', border: 'none', background: 'transparent',
                    padding: '.75rem 1rem', fontSize: '.9rem',
                    color: 'hsl(var(--ink))', outline: 'none',
                    fontFamily: 'Inter, sans-serif',
                  }}
                />
              </div>
              <button
                onClick={handleSend}
                disabled={!input.trim() || streaming}
                className="btn btn-primary"
                style={{
                  width: '44px', height: '44px', padding: 0,
                  borderRadius: '12px', flexShrink: 0,
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  opacity: !input.trim() || streaming ? 0.5 : 1,
                }}
              >
                {streaming ? <Loader size={16} className="spin" /> : <Send size={16} />}
              </button>
            </div>
          </div>
        </div>

        {/* ═══════ Sources Sidebar ═══════ */}
        <div style={{
          width: '280px', flexShrink: 0,
          borderLeft: '2px solid hsl(var(--border) / .1)',
          background: 'hsl(var(--card) / .4)',
          display: 'flex', flexDirection: 'column',
          overflow: 'hidden',
        }}>
          <div style={{
            padding: '1.25rem 1.25rem .75rem',
            borderBottom: '1px solid hsl(var(--border) / .1)',
          }}>
            <h3 style={{ fontSize: '.88rem', fontWeight: 700, color: 'hsl(var(--ink))', margin: 0, display: 'flex', alignItems: 'center', gap: '.4rem' }}>
              <FileText size={14} style={{ color: 'hsl(var(--accent))' }} />
              Meeting Sources
            </h3>
          </div>

          <div style={{ flex: 1, overflowY: 'auto', padding: '1rem 1.25rem' }}>
            {currentSources.length > 0 ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '.75rem' }}>
                {currentSources.map(source => (
                  <div
                    key={source.id}
                    style={{
                      padding: '.85rem 1rem',
                      borderRadius: '10px',
                      background: 'hsl(var(--card))',
                      border: '1.5px solid hsl(var(--border) / .15)',
                      transition: 'all .2s',
                      cursor: 'pointer',
                    }}
                    onClick={() => navigate(`/dashboard/history/${source.id}`)}
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
                        width: '28px', height: '28px', borderRadius: '8px', flexShrink: 0,
                        background: 'linear-gradient(135deg, hsl(var(--accent) / .15), hsl(var(--accent) / .08))',
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                        border: '1.5px solid hsl(var(--accent) / .25)',
                      }}>
                        <FileText size={13} style={{ color: 'hsl(var(--accent))' }} />
                      </div>
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{
                          fontSize: '.82rem', fontWeight: 600, color: 'hsl(var(--ink))',
                          overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                        }}>
                          {source.name}
                        </div>
                        {source.date && (
                          <div style={{ fontSize: '.72rem', color: 'hsl(var(--pencil))', marginTop: '.2rem', fontWeight: 500 }}>
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
                textAlign: 'center', padding: '2rem 1rem',
                color: 'hsl(var(--pencil))', fontSize: '.82rem', lineHeight: 1.5,
              }}>
                <Bot size={24} style={{ color: 'hsl(var(--pencil))', opacity: 0.4, marginBottom: '.75rem' }} />
                <p style={{ fontWeight: 600, marginBottom: '.25rem' }}>No meeting context used</p>
                <p style={{ opacity: 0.7 }}>This answer was generated directly without searching meetings.</p>
              </div>
            ) : (
              <div style={{
                textAlign: 'center', padding: '2rem 1rem',
                color: 'hsl(var(--pencil))', fontSize: '.82rem', lineHeight: 1.5,
              }}>
                <Sparkles size={24} style={{ color: 'hsl(var(--pencil))', opacity: 0.3, marginBottom: '.75rem' }} />
                <p style={{ fontWeight: 600, marginBottom: '.25rem' }}>Ask a question</p>
                <p style={{ opacity: 0.7 }}>Meeting sources will appear here after the AI responds.</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
