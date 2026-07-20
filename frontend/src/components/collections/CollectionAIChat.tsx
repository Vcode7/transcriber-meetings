import React, { useState, useEffect, useRef, useCallback } from 'react'
import {
  MessageSquare, GitCompare, TrendingUp, Send, Loader, Bot,
  Trash2, Copy, Check, Download, ChevronDown, Sparkles, X
} from 'lucide-react'
import type { CollectionMeeting, ChatMessage } from '../../types/recording'
import {
  streamChat, streamComparison, streamTopicGrowth,
  getChatHistory, clearChatHistory, exportReport,
} from '../../api/collectionAI'

interface CollectionAIChatProps {
  collectionId: string
  meetings: CollectionMeeting[]
  onClose: () => void
}

type ChatMode = 'chat' | 'compare' | 'topic_growth'

export default function CollectionAIChat({ collectionId, meetings, onClose }: CollectionAIChatProps) {
  const [mode, setMode] = useState<ChatMode>('chat')
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [streaming, setStreaming] = useState(false)
  const [streamedText, setStreamedText] = useState('')
  const [historyLoading, setHistoryLoading] = useState(true)
  const [copied, setCopied] = useState<string | null>(null)

  // Compare mode state
  const [meetingA, setMeetingA] = useState('')
  const [meetingB, setMeetingB] = useState('')

  // Topic growth state
  const [topic, setTopic] = useState('')

  // Clear confirmation
  const [confirmClear, setConfirmClear] = useState(false)

  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const abortRef = useRef<AbortController | null>(null)

  // Load chat history on mount
  useEffect(() => {
    let cancelled = false
    const load = async () => {
      try {
        const history = await getChatHistory(collectionId)
        if (!cancelled) setMessages(history)
      } catch (err) {
        console.error('Failed to load chat history:', err)
      } finally {
        if (!cancelled) setHistoryLoading(false)
      }
    }
    load()
    return () => { cancelled = true }
  }, [collectionId])

  // Auto-scroll on new messages or streaming
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, streamedText])

  const handleCopy = useCallback((text: string, id: string) => {
    navigator.clipboard.writeText(text)
    setCopied(id)
    setTimeout(() => setCopied(null), 2000)
  }, [])

  const handleExport = useCallback(async (content: string) => {
    try {
      await exportReport(collectionId, content, `ai-report-${Date.now()}`)
    } catch (err) {
      console.error('Export failed:', err)
    }
  }, [collectionId])

  const handleClearHistory = useCallback(async () => {
    try {
      await clearChatHistory(collectionId)
      setMessages([])
      setConfirmClear(false)
    } catch (err) {
      console.error('Failed to clear history:', err)
    }
  }, [collectionId])

  const handleSendChat = useCallback(async () => {
    const msg = input.trim()
    if (!msg || streaming) return

    setInput('')
    setStreaming(true)
    setStreamedText('')

    // Add user message optimistically
    const userMsg: ChatMessage = {
      id: `temp-${Date.now()}`,
      role: 'user',
      content: msg,
      message_type: 'chat',
      metadata: {},
      created_at: new Date().toISOString(),
    }
    setMessages(prev => [...prev, userMsg])

    const controller = new AbortController()
    abortRef.current = controller
    let accumulated = ''

    try {
      await streamChat(
        collectionId,
        msg,
        (chunk) => {
          accumulated += chunk
          setStreamedText(accumulated)
        },
        (meta) => {
          // metadata received — will be included in saved message
        },
        (error) => {
          accumulated = error
          setStreamedText(error)
        },
        () => {
          // Done — add assistant message
          if (accumulated) {
            const assistantMsg: ChatMessage = {
              id: `resp-${Date.now()}`,
              role: 'assistant',
              content: accumulated,
              message_type: 'chat',
              metadata: {},
              created_at: new Date().toISOString(),
            }
            setMessages(prev => [...prev, assistantMsg])
          }
          setStreamedText('')
          setStreaming(false)
        },
        controller.signal,
      )
    } catch (err: any) {
      if (err.name !== 'AbortError') {
        const errMsg: ChatMessage = {
          id: `err-${Date.now()}`,
          role: 'assistant',
          content: '⚠️ Failed to get AI response. Please try again.',
          message_type: 'chat',
          metadata: {},
          created_at: new Date().toISOString(),
        }
        setMessages(prev => [...prev, errMsg])
      }
      setStreamedText('')
      setStreaming(false)
    }
  }, [input, streaming, collectionId])

  const handleCompare = useCallback(async () => {
    if (!meetingA || !meetingB || streaming) return

    setStreaming(true)
    setStreamedText('')

    const nameA = meetings.find(m => m.id === meetingA)?.filename || 'Meeting A'
    const nameB = meetings.find(m => m.id === meetingB)?.filename || 'Meeting B'

    const userMsg: ChatMessage = {
      id: `temp-${Date.now()}`,
      role: 'user',
      content: `📊 Compare: **${nameA}** vs **${nameB}**`,
      message_type: 'comparison',
      metadata: {},
      created_at: new Date().toISOString(),
    }
    setMessages(prev => [...prev, userMsg])

    const controller = new AbortController()
    abortRef.current = controller
    let accumulated = ''

    try {
      await streamComparison(
        collectionId, meetingA, meetingB,
        (chunk) => { accumulated += chunk; setStreamedText(accumulated) },
        undefined,
        (error) => { accumulated = error; setStreamedText(error) },
        () => {
          if (accumulated) {
            setMessages(prev => [...prev, {
              id: `resp-${Date.now()}`, role: 'assistant', content: accumulated,
              message_type: 'comparison', metadata: {}, created_at: new Date().toISOString(),
            }])
          }
          setStreamedText('')
          setStreaming(false)
        },
        controller.signal,
      )
    } catch (err: any) {
      if (err.name !== 'AbortError') {
        setMessages(prev => [...prev, {
          id: `err-${Date.now()}`, role: 'assistant', content: '⚠️ Comparison failed.',
          message_type: 'comparison', metadata: {}, created_at: new Date().toISOString(),
        }])
      }
      setStreamedText('')
      setStreaming(false)
    }
  }, [meetingA, meetingB, streaming, collectionId, meetings])

  const handleTopicGrowth = useCallback(async () => {
    const t = topic.trim()
    if (!t || streaming) return

    setStreaming(true)
    setStreamedText('')

    const userMsg: ChatMessage = {
      id: `temp-${Date.now()}`,
      role: 'user',
      content: `📈 Topic Growth: **${t}**`,
      message_type: 'topic_growth',
      metadata: {},
      created_at: new Date().toISOString(),
    }
    setMessages(prev => [...prev, userMsg])

    const controller = new AbortController()
    abortRef.current = controller
    let accumulated = ''

    try {
      await streamTopicGrowth(
        collectionId, t,
        (chunk) => { accumulated += chunk; setStreamedText(accumulated) },
        undefined,
        (error) => { accumulated = error; setStreamedText(error) },
        () => {
          if (accumulated) {
            setMessages(prev => [...prev, {
              id: `resp-${Date.now()}`, role: 'assistant', content: accumulated,
              message_type: 'topic_growth', metadata: {}, created_at: new Date().toISOString(),
            }])
          }
          setStreamedText('')
          setStreaming(false)
        },
        controller.signal,
      )
    } catch (err: any) {
      if (err.name !== 'AbortError') {
        setMessages(prev => [...prev, {
          id: `err-${Date.now()}`, role: 'assistant', content: '⚠️ Topic growth analysis failed.',
          message_type: 'topic_growth', metadata: {}, created_at: new Date().toISOString(),
        }])
      }
      setStreamedText('')
      setStreaming(false)
    }
  }, [topic, streaming, collectionId])

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      if (mode === 'chat') handleSendChat()
    }
  }

  const modes: { key: ChatMode; label: string; icon: typeof MessageSquare }[] = [
    { key: 'chat', label: 'Chat', icon: MessageSquare },
    { key: 'compare', label: 'Compare', icon: GitCompare },
    { key: 'topic_growth', label: 'Topic Growth', icon: TrendingUp },
  ]

  const renderMarkdown = (text: string) => {
    // Simple markdown renderer: bold, headers, bullets, code
    const lines = text.split('\n')
    return lines.map((line, i) => {
      // Headers
      if (line.startsWith('### ')) return <h4 key={i} style={{ fontSize: '.95rem', fontWeight: 700, margin: '1rem 0 .5rem', color: 'hsl(var(--ink))' }}>{line.slice(4)}</h4>
      if (line.startsWith('## ')) return <h3 key={i} style={{ fontSize: '1.05rem', fontWeight: 700, margin: '1.25rem 0 .5rem', color: 'hsl(var(--accent))', borderBottom: '1px solid hsl(var(--accent) / .15)', paddingBottom: '.35rem' }}>{line.slice(3)}</h3>
      if (line.startsWith('# ')) return <h2 key={i} style={{ fontSize: '1.15rem', fontWeight: 700, margin: '1.5rem 0 .75rem', color: 'hsl(var(--ink))' }}>{line.slice(2)}</h2>

      // Bullets
      if (line.match(/^[-•]\s/)) {
        const content = line.replace(/^[-•]\s/, '')
        return <div key={i} style={{ display: 'flex', gap: '.5rem', margin: '.25rem 0', lineHeight: 1.6 }}>
          <span style={{ color: 'hsl(var(--accent))', flexShrink: 0, marginTop: '2px' }}>•</span>
          <span dangerouslySetInnerHTML={{ __html: boldify(content) }} />
        </div>
      }

      // Empty line
      if (!line.trim()) return <div key={i} style={{ height: '.5rem' }} />

      // Regular text
      return <p key={i} style={{ margin: '.25rem 0', lineHeight: 1.6 }} dangerouslySetInnerHTML={{ __html: boldify(line) }} />
    })
  }

  const boldify = (text: string) => {
    return text
      .replace(/\*\*(.+?)\*\*/g, '<strong style="color:hsl(var(--ink));font-weight:700">$1</strong>')
      .replace(/\[Meeting[:\s]*([^\]]+)\]/g, '<span style="background:hsl(var(--accent)/.1);color:hsl(var(--accent));padding:0.1rem 0.4rem;border-radius:4px;font-size:.8rem;font-weight:600">📎 $1</span>')
  }

  return (
    <div style={{
      display: 'flex', flexDirection: 'column', height: '100%',
      background: 'linear-gradient(170deg, hsl(var(--paper-deep)), hsl(var(--paper)))',
      borderLeft: '2px solid hsl(var(--border) / .12)',
      overflow: 'hidden',
    }}>
      {/* ═══════ Header ═══════ */}
      <div style={{
        padding: '1rem 1.25rem',
        borderBottom: '2px solid hsl(var(--border) / .1)',
        background: 'hsl(var(--card) / .6)',
        backdropFilter: 'blur(12px)',
        flexShrink: 0,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '.75rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '.6rem' }}>
            <div style={{
              width: '36px', height: '36px', borderRadius: '10px',
              background: 'linear-gradient(135deg, hsl(270 80% 60% / .2), hsl(200 80% 60% / .2))',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              border: '1.5px solid hsl(270 60% 60% / .3)',
            }}>
              <Bot size={18} style={{ color: 'hsl(270 60% 60%)' }} />
            </div>
            <div>
              <h3 style={{ fontSize: '1rem', fontWeight: 700, margin: 0, color: 'hsl(var(--ink))' }}>
                Collection AI
              </h3>
              <p style={{ fontSize: '.72rem', color: 'hsl(var(--pencil))', margin: 0, fontWeight: 500 }}>
                Powered by local LLM • {meetings.length} meetings
              </p>
            </div>
          </div>
          <div style={{ display: 'flex', gap: '.4rem', alignItems: 'center' }}>
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
                  style={{ width: '32px', height: '32px', color: 'hsl(var(--pencil))' }}>
                  <Trash2 size={14} />
                </button>
              )
            )}
            <button className="icon-btn" onClick={onClose}
              style={{ width: '32px', height: '32px', color: 'hsl(var(--pencil))' }}>
              <X size={16} />
            </button>
          </div>
        </div>

        {/* ═══════ Mode Tabs ═══════ */}
        <div style={{
          display: 'flex', gap: '2px',
          background: 'hsl(var(--muted))',
          borderRadius: '10px', padding: '3px',
        }}>
          {modes.map(m => {
            const active = mode === m.key
            const Icon = m.icon
            return (
              <button
                key={m.key}
                onClick={() => setMode(m.key)}
                style={{
                  flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center',
                  gap: '.35rem', padding: '.45rem .5rem',
                  fontSize: '.78rem', fontWeight: 600,
                  borderRadius: '8px', border: 'none', cursor: 'pointer',
                  background: active ? 'hsl(var(--card))' : 'transparent',
                  color: active ? 'hsl(var(--accent))' : 'hsl(var(--pencil))',
                  boxShadow: active ? '0 1px 3px hsl(var(--ink) / .08)' : 'none',
                  transition: 'all .2s ease',
                }}
              >
                <Icon size={13} />
                {m.label}
              </button>
            )
          })}
        </div>
      </div>

      {/* ═══════ Messages Area ═══════ */}
      <div style={{
        flex: 1, overflowY: 'auto', padding: '1rem 1rem',
        display: 'flex', flexDirection: 'column', gap: '.75rem',
        minHeight: 0,
      }}>
        {historyLoading ? (
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
              width: '64px', height: '64px', borderRadius: '50%',
              background: 'linear-gradient(135deg, hsl(270 80% 60% / .1), hsl(200 80% 60% / .1))',
              border: '2px dashed hsl(270 60% 60% / .25)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
            }}>
              <Sparkles size={28} style={{ color: 'hsl(270 60% 60%)', opacity: 0.6 }} />
            </div>
            <div>
              <h4 style={{ fontSize: '1rem', fontWeight: 700, color: 'hsl(var(--ink))', margin: '0 0 .35rem' }}>
                {mode === 'chat' ? 'Ask anything about your meetings'
                  : mode === 'compare' ? 'Compare two meetings'
                  : 'Track a topic across meetings'}
              </h4>
              <p style={{ fontSize: '.85rem', color: 'hsl(var(--pencil))', lineHeight: 1.5, maxWidth: '300px', margin: '0 auto' }}>
                {mode === 'chat'
                  ? 'The AI uses RAG to search all meetings in this collection and answer your questions with citations.'
                  : mode === 'compare'
                  ? 'Select two meetings below to generate a detailed comparison report.'
                  : 'Enter a topic to see how it evolved across meetings over time.'}
              </p>
            </div>
          </div>
        ) : (
          <>
            {messages.map(msg => (
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
                    width: '28px', height: '28px', borderRadius: '8px', flexShrink: 0,
                    background: 'linear-gradient(135deg, hsl(270 70% 55% / .15), hsl(200 70% 55% / .15))',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    border: '1.5px solid hsl(270 60% 55% / .25)', marginTop: '2px',
                  }}>
                    <Bot size={14} style={{ color: 'hsl(270 60% 55%)' }} />
                  </div>
                )}
                <div style={{
                  maxWidth: msg.role === 'user' ? '80%' : '90%',
                  padding: msg.role === 'user' ? '.65rem 1rem' : '.75rem 1rem',
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
                  position: 'relative',
                }}>
                  {msg.role === 'assistant' ? (
                    <div>
                      <div style={{ fontSize: '.88rem' }}>{renderMarkdown(msg.content)}</div>
                      <div style={{
                        display: 'flex', gap: '.3rem', marginTop: '.6rem',
                        paddingTop: '.5rem', borderTop: '1px solid hsl(var(--border) / .1)',
                      }}>
                        <button className="icon-btn"
                          onClick={() => handleCopy(msg.content, msg.id)}
                          title="Copy to clipboard"
                          style={{ width: '26px', height: '26px', color: 'hsl(var(--pencil))' }}>
                          {copied === msg.id ? <Check size={12} style={{ color: 'hsl(var(--success))' }} /> : <Copy size={12} />}
                        </button>
                        <button className="icon-btn"
                          onClick={() => handleExport(msg.content)}
                          title="Download as Markdown"
                          style={{ width: '26px', height: '26px', color: 'hsl(var(--pencil))' }}>
                          <Download size={12} />
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
                  width: '28px', height: '28px', borderRadius: '8px', flexShrink: 0,
                  background: 'linear-gradient(135deg, hsl(270 70% 55% / .15), hsl(200 70% 55% / .15))',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  border: '1.5px solid hsl(270 60% 55% / .25)', marginTop: '2px',
                }}>
                  <Bot size={14} style={{ color: 'hsl(270 60% 55%)' }} />
                </div>
                <div style={{
                  maxWidth: '90%', padding: '.75rem 1rem', borderRadius: '14px 14px 14px 4px',
                  background: 'hsl(var(--card))', border: '1px solid hsl(var(--border) / .15)',
                  fontSize: '.88rem', lineHeight: 1.6,
                  boxShadow: '0 1px 3px hsl(var(--ink) / .04)',
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

            {/* Loading spinner (before streaming starts) */}
            {streaming && !streamedText && (
              <div style={{ display: 'flex', gap: '.5rem', alignItems: 'center' }}>
                <div style={{
                  width: '28px', height: '28px', borderRadius: '8px', flexShrink: 0,
                  background: 'linear-gradient(135deg, hsl(270 70% 55% / .15), hsl(200 70% 55% / .15))',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  border: '1.5px solid hsl(270 60% 55% / .25)',
                }}>
                  <Bot size={14} style={{ color: 'hsl(270 60% 55%)' }} />
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
        padding: '1rem 1rem',
        flexShrink: 0,
      }}>
        {mode === 'chat' && (
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
                placeholder="Ask about your meetings..."
                disabled={streaming}
                style={{
                  width: '100%', border: 'none', background: 'transparent',
                  padding: '.7rem 1rem', fontSize: '.88rem',
                  color: 'hsl(var(--ink))', outline: 'none',
                  fontFamily: 'Inter, sans-serif',
                }}
              />
            </div>
            <button
              onClick={handleSendChat}
              disabled={!input.trim() || streaming}
              className="btn btn-primary"
              style={{
                width: '42px', height: '42px', padding: 0,
                borderRadius: '12px', flexShrink: 0,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                opacity: !input.trim() || streaming ? 0.5 : 1,
              }}
            >
              {streaming ? <Loader size={16} className="spin" /> : <Send size={16} />}
            </button>
          </div>
        )}

        {mode === 'compare' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '.6rem' }}>
            <div style={{ display: 'flex', gap: '.5rem', alignItems: 'center' }}>
              <span style={{ fontSize: '.78rem', fontWeight: 700, color: 'hsl(var(--accent))', width: '16px', textAlign: 'center' }}>A</span>
              <select
                className="input"
                value={meetingA}
                onChange={e => setMeetingA(e.target.value)}
                style={{ flex: 1, height: '36px', fontSize: '.85rem', padding: '0 .75rem' }}
              >
                <option value="">Select Meeting A...</option>
                {meetings.map(m => (
                  <option key={m.id} value={m.id} disabled={m.id === meetingB}>{m.filename}</option>
                ))}
              </select>
            </div>
            <div style={{ display: 'flex', gap: '.5rem', alignItems: 'center' }}>
              <span style={{ fontSize: '.78rem', fontWeight: 700, color: 'hsl(var(--accent))', width: '16px', textAlign: 'center' }}>B</span>
              <select
                className="input"
                value={meetingB}
                onChange={e => setMeetingB(e.target.value)}
                style={{ flex: 1, height: '36px', fontSize: '.85rem', padding: '0 .75rem' }}
              >
                <option value="">Select Meeting B...</option>
                {meetings.map(m => (
                  <option key={m.id} value={m.id} disabled={m.id === meetingA}>{m.filename}</option>
                ))}
              </select>
            </div>
            <button
              onClick={handleCompare}
              disabled={!meetingA || !meetingB || streaming}
              className="btn btn-primary"
              style={{ width: '100%', gap: '.5rem', opacity: !meetingA || !meetingB || streaming ? 0.5 : 1 }}
            >
              {streaming ? <><Loader size={14} className="spin" /> Generating...</> : <><GitCompare size={14} /> Generate Comparison</>}
            </button>
          </div>
        )}

        {mode === 'topic_growth' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '.6rem' }}>
            <div style={{
              display: 'flex', gap: '.5rem', alignItems: 'center',
            }}>
              <div style={{
                flex: 1, position: 'relative',
                background: 'hsl(var(--paper))',
                borderRadius: '12px',
                border: '2px solid hsl(var(--border) / .2)',
              }}>
                <input
                  type="text"
                  value={topic}
                  onChange={e => setTopic(e.target.value)}
                  onKeyDown={e => { if (e.key === 'Enter') handleTopicGrowth() }}
                  placeholder="Enter a topic (e.g. Budget, Authentication)..."
                  disabled={streaming}
                  style={{
                    width: '100%', border: 'none', background: 'transparent',
                    padding: '.7rem 1rem', fontSize: '.88rem',
                    color: 'hsl(var(--ink))', outline: 'none',
                    fontFamily: 'Inter, sans-serif',
                  }}
                />
              </div>
            </div>
            <button
              onClick={handleTopicGrowth}
              disabled={!topic.trim() || streaming}
              className="btn btn-primary"
              style={{ width: '100%', gap: '.5rem', opacity: !topic.trim() || streaming ? 0.5 : 1 }}
            >
              {streaming ? <><Loader size={14} className="spin" /> Analyzing...</> : <><TrendingUp size={14} /> Track Topic Growth</>}
            </button>
          </div>
        )}
      </div>
    </div>
  )
}
