import { useState, useEffect } from 'react'
import { Bot, Sparkles, FileText, ListChecks, ChevronRight, ChevronLeft, Copy, Check, Loader, Users } from 'lucide-react'


interface Props {
  recordingId: string | null
  summary?: string
  shortSummary?: string
  detailedSummary?: string
  keyPoints?: string[]
  actionItems?: string[]
  speakerSummary?: Record<string, { summary: string; key_points: string[]; action_items: string[] }> | null
  isOpen: boolean
  onToggle: () => void
  /** True while Llama is generating AI insights (transcript is ready but AI not done) */
  isGenerating?: boolean
}

// ── Minimal markdown parser ───────────────────────────────────
function parseMarkdown(text: string): string {
  return text
    .replace(/^## (.+)$/gm, '<h3 style="font-size:1rem;font-weight:700;color:hsl(var(--ink));margin:1rem 0 .4rem;font-family:Inter,sans-serif">$1</h3>')
    .replace(/^### (.+)$/gm, '<h4 style="font-size:.82rem;font-weight:700;color:hsl(var(--ink));margin:.8rem 0 .3rem;font-family:Inter,sans-serif">$1</h4>')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/^[-*] (.+)$/gm, '<li>$1</li>')
    .replace(/^(\d+)\. (.+)$/gm, '<li>$2</li>')
    .replace(/(<li>.*<\/li>(\n|$))+/g, (match) => `<ul style="margin:.4rem 0;padding-left:1.2rem">${match}</ul>`)
    .replace(/`([^`]+)`/g, '<code style="background:hsl(var(--muted));padding:.1rem .3rem;border-radius:4px;font-size:.8em">$1</code>')
    .split(/\n{2,}/)
    .map(block => block.startsWith('<') ? block : `<p style="margin:.4rem 0;line-height:1.7">${block.replace(/\n/g, '<br/>')}</p>`)
    .join('')
}

function MarkdownContent({ content }: { content: string }) {
  return (
    <div
      style={{ fontSize: '.84rem', color: 'hsl(var(--ink-soft))', fontFamily: 'Inter, sans-serif' }}
      dangerouslySetInnerHTML={{ __html: parseMarkdown(content) }}
    />
  )
}

function LocalAIBadge() {
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: '3px',
      fontSize: '.58rem', fontWeight: 700,
      background: 'hsl(var(--success) / .12)',
      color: 'hsl(var(--success))',
      border: '1px solid hsl(var(--success) / .3)',
      borderRadius: '4px', padding: '.05rem .32rem',
      letterSpacing: '.06em', textTransform: 'uppercase',
    }}>
      <Bot size={8} /> Local AI
    </span>
  )
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false)
  const copy = () => {
    navigator.clipboard.writeText(text)
    setCopied(true)
    setTimeout(() => setCopied(false), 1800)
  }
  return (
    <button
      onClick={copy}
      title="Copy"
      style={{
        background: 'none', border: 'none', cursor: 'pointer',
        color: 'hsl(var(--pencil))', padding: '3px 6px', borderRadius: '5px',
        display: 'flex', alignItems: 'center', gap: '4px',
        fontSize: '.68rem', fontFamily: 'Inter, sans-serif',
        transition: 'all .2s', opacity: 0.55,
      }}
      onMouseEnter={(e) => {
        const el = e.currentTarget as HTMLButtonElement
        el.style.opacity = '1'
        el.style.background = 'hsl(var(--muted))'
      }}
      onMouseLeave={(e) => {
        const el = e.currentTarget as HTMLButtonElement
        el.style.opacity = '0.55'
        el.style.background = 'none'
      }}
    >
      <span style={{ display: 'inline-block', transition: 'transform .2s', transform: copied ? 'rotateY(180deg)' : 'rotateY(0deg)' }}>
        {copied ? <Check size={11} style={{ color: 'hsl(var(--success))' }} /> : <Copy size={11} />}
      </span>
      {copied ? 'Copied!' : 'Copy'}
    </button>
  )
}

// ── AI Generating skeleton ────────────────────────────────────
const GENERATING_MESSAGES = [
  'Analyzing transcript…',
  'Generating AI insights…',
  'Creating summary…',
  'Extracting key points…',
  'Identifying action items…',
  'Almost there…',
]

function AIGeneratingSkeleton() {
  const [msgIdx, setMsgIdx] = useState(0)
  useEffect(() => {
    const t = setInterval(() => setMsgIdx(i => (i + 1) % GENERATING_MESSAGES.length), 2200)
    return () => clearInterval(t)
  }, [])
  return (
    <div style={{ padding: '1.25rem 1rem', display: 'flex', flexDirection: 'column', gap: '16px' }}>
      {/* Status row */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: '10px',
        padding: '.7rem .9rem',
        background: 'hsl(var(--accent) / .07)',
        borderRadius: '10px',
        border: '1px solid hsl(var(--accent) / .18)',
      }}>
        <Loader size={14} className="spin" style={{ color: 'hsl(var(--accent))', flexShrink: 0 }} />
        <span style={{
          fontSize: '.78rem', fontFamily: 'Inter, sans-serif',
          color: 'hsl(var(--accent))', fontWeight: 600,
          transition: 'opacity .4s',
        }}>
          {GENERATING_MESSAGES[msgIdx]}
        </span>
      </div>

      {/* Skeleton blocks */}
      {[{ w: '100%', h: 14 }, { w: '88%', h: 14 }, { w: '95%', h: 14 }, { w: '72%', h: 14 }].map((s, i) => (
        <div key={i} style={{
          height: `${s.h}px`, width: s.w, borderRadius: '6px',
          background: 'linear-gradient(90deg, hsl(var(--muted)) 0%, hsl(var(--card)) 50%, hsl(var(--muted)) 100%)',
          backgroundSize: '200% 100%',
          animation: `shimmer 1.8s ease-in-out infinite`,
          animationDelay: `${i * 0.15}s`,
        }} />
      ))}

      <div style={{ display: 'flex', gap: '8px', marginTop: '4px' }}>
        {[60, 45, 55].map((w, i) => (
          <div key={i} style={{
            height: '28px', width: `${w}%`, borderRadius: '6px',
            background: 'linear-gradient(90deg, hsl(var(--muted)) 0%, hsl(var(--card)) 50%, hsl(var(--muted)) 100%)',
            backgroundSize: '200% 100%',
            animation: `shimmer 1.8s ease-in-out infinite`,
            animationDelay: `${i * 0.2 + 0.6}s`,
          }} />
        ))}
      </div>

      {[{ w: '100%' }, { w: '90%' }, { w: '80%' }].map((s, i) => (
        <div key={i} style={{
          height: '12px', width: s.w, borderRadius: '4px',
          background: 'linear-gradient(90deg, hsl(var(--muted)) 0%, hsl(var(--card)) 50%, hsl(var(--muted)) 100%)',
          backgroundSize: '200% 100%',
          animation: `shimmer 1.8s ease-in-out infinite`,
          animationDelay: `${i * 0.12 + 1.0}s`,
        }} />
      ))}

      <p style={{
        fontSize: '.67rem', fontFamily: 'Inter, sans-serif',
        color: 'hsl(var(--pencil))', textAlign: 'center',
        marginTop: '8px', letterSpacing: '.03em',
      }}>
        Qwen-AI · Running locally · Please wait
      </p>
    </div>
  )
}

export default function AIChatPanel({ recordingId, summary, shortSummary, detailedSummary, keyPoints, actionItems, speakerSummary, isOpen, onToggle, isGenerating = false }: Props) {
  const [summaryTab, setSummaryTab] = useState<'short' | 'detailed' | 'speaker'>('short')

  // Resolve summary content — prefer new fields, fall back to legacy `summary`
  const resolvedShort = shortSummary || summary || ''
  const resolvedDetailed = detailedSummary || ''

  const hasContent = resolvedShort || resolvedDetailed || keyPoints?.length || actionItems?.length || (speakerSummary && Object.keys(speakerSummary).length > 0)

  /* ── Collapsed strip */
  if (!isOpen) {
    return (
      <div style={{
        display: 'flex', flexDirection: 'column', alignItems: 'center',
        paddingTop: '.75rem',
        background: 'hsl(var(--card))',
        borderLeft: '1.5px solid hsl(var(--border) / .2)',
      }}>
        <button
          className="icon-btn"
          onClick={onToggle}
          title="Open AI Insights"
          style={{ marginBottom: '12px' }}
        >
          <ChevronLeft size={16} />
        </button>
        <div style={{
          width: '34px', height: '34px', borderRadius: '10px',
          background: 'linear-gradient(135deg, hsl(var(--accent) / .15), hsl(var(--accent) / .05))',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          border: '1.5px solid hsl(var(--accent) / .3)',
          boxShadow: '0 0 12px hsl(var(--accent) / .1)',
        }}>
          <Sparkles size={17} style={{ color: 'hsl(var(--accent))' }} />
        </div>
        {/* Indicator dot — green if has content, orange-pulse if generating */}
        {(hasContent || isGenerating) && (
          <div style={{
            marginTop: '10px',
            width: '8px', height: '8px',
            borderRadius: '50%',
            background: isGenerating ? 'hsl(var(--accent))' : 'hsl(var(--success))',
            boxShadow: isGenerating
              ? '0 0 8px hsl(var(--accent) / .6)'
              : '0 0 8px hsl(var(--success) / .6)',
          }} className="animate-pulse-rec" />
        )}
      </div>
    )
  }

  return (
    <div style={{
      display: 'flex', flexDirection: 'column', height: '100%',
      background: 'hsl(var(--card))',
      borderLeft: '1.5px solid hsl(var(--border) / .2)',
      overflow: 'hidden',
    }}>

      {/* Header */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: '10px',
        padding: '.9rem 1rem',
        borderBottom: '1.5px solid hsl(var(--border) / .2)',
        background: 'linear-gradient(180deg, hsl(var(--card)) 0%, hsl(var(--card) / .9) 100%)',
        backdropFilter: 'blur(8px)',
        flexShrink: 0,
        minHeight: '58px',
      }}>
        <div style={{
          width: '32px', height: '32px', borderRadius: '9px', flexShrink: 0,
          background: 'linear-gradient(135deg, hsl(var(--accent) / .18), hsl(var(--accent) / .06))',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          border: '1.5px solid hsl(var(--accent) / .35)',
          boxShadow: '0 0 14px hsl(var(--accent) / .12)',
        }}>
          <Sparkles size={16} style={{ color: 'hsl(var(--accent))' }} />
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{
            fontSize: '.88rem', fontWeight: 700, fontFamily: 'Inter, sans-serif',
            color: 'hsl(var(--ink))', lineHeight: 1.2,
            display: 'flex', alignItems: 'center', gap: '6px',
          }}>
            AI Insights
            <LocalAIBadge />
          </div>
          <div style={{
            fontSize: '.65rem', color: 'hsl(var(--pencil))',
            fontFamily: 'Inter, sans-serif',
            marginTop: '2px',
          }}>
            Powered by Llama 3.2 · Fully offline
          </div>
        </div>
        <button className="icon-btn" onClick={onToggle} title="Collapse panel" style={{ width: '30px', height: '30px', flexShrink: 0 }}>
          <ChevronRight size={14} />
        </button>
      </div>

      {/* Content area */}
      <div style={{ flex: 1, overflowY: 'auto', minHeight: 0 }}>

        {/* No recording selected */}
        {!recordingId && (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', padding: '2rem', gap: '14px', textAlign: 'center' }}>
            <div style={{
              width: '56px', height: '56px', borderRadius: '16px',
              background: 'linear-gradient(135deg, hsl(var(--accent) / .12), hsl(var(--accent) / .04))',
              border: '1.5px solid hsl(var(--accent) / .25)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              boxShadow: '0 0 20px hsl(var(--accent) / .08)',
            }}>
              <Sparkles size={24} style={{ color: 'hsl(var(--accent))' }} className="animate-float" />
            </div>
            <div>
              <p style={{ fontFamily: 'Inter, sans-serif', fontSize: '.88rem', color: 'hsl(var(--ink))', fontWeight: 600, marginBottom: '.3rem' }}>
                AI Insights
              </p>
              <p style={{ fontFamily: 'Inter, sans-serif', fontSize: '.78rem', color: 'hsl(var(--pencil))', lineHeight: 1.5, maxWidth: '380px', margin: '0 auto' }}>
                Process a recording to see AI-generated summaries and insights
              </p>
            </div>
          </div>
        )}

        {/* Generating skeleton — shown when transcript is ready but AI is still running */}
        {recordingId && isGenerating && !hasContent && (
          <AIGeneratingSkeleton />
        )}

        {/* Recording selected but no content yet (and not generating) */}
        {recordingId && !hasContent && !isGenerating && (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', padding: '2rem', gap: '14px', textAlign: 'center' }}>
            <div style={{
              width: '56px', height: '56px', borderRadius: '16px',
              background: 'linear-gradient(135deg, hsl(var(--muted) / .3), hsl(var(--muted) / .1))',
              border: '1.5px solid hsl(var(--border) / .3)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
            }}>
              <Bot size={24} style={{ color: 'hsl(var(--pencil))' }} />
            </div>
            <p style={{ fontFamily: 'Inter, sans-serif', fontSize: '.82rem', color: 'hsl(var(--pencil))', lineHeight: 1.5, maxWidth: '380px', margin: '0 auto' }}>
              AI insights will appear here once processing is complete
            </p>
          </div>
        )}

        {/* ── Tabbed Summary Card ── */}
        {recordingId && (resolvedShort || resolvedDetailed || (speakerSummary && Object.keys(speakerSummary).length > 0)) && (
          <div style={{
            borderBottom: '1.5px solid hsl(var(--border) / .2)',
            background: 'hsl(var(--paper) / .4)',
          }}>
            {/* Tab strip */}
            <div style={{
              display: 'flex', gap: 0,
              borderBottom: '1px solid hsl(var(--border) / .18)',
              padding: '0 .75rem',
            }}>
              {([
                ...(resolvedShort ? [{ id: 'short', label: 'Short', icon: <Sparkles size={10} /> }] : []),
                ...(resolvedDetailed ? [{ id: 'detailed', label: 'Detailed', icon: <FileText size={10} /> }] : []),
                ...(speakerSummary && Object.keys(speakerSummary).length > 0
                  ? [{ id: 'speaker', label: 'Speakers', icon: <Users size={10} /> }]
                  : []),
              ] as const).map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setSummaryTab(tab.id as any)}
                  style={{
                    background: 'none', border: 'none', cursor: 'pointer',
                    padding: '.55rem .75rem',
                    fontSize: '.72rem', fontWeight: 700,
                    fontFamily: 'Inter, sans-serif',
                    textTransform: 'uppercase', letterSpacing: '.07em',
                    color: summaryTab === tab.id ? 'hsl(var(--accent))' : 'hsl(var(--pencil))',
                    borderBottom: summaryTab === tab.id
                      ? '2px solid hsl(var(--accent))'
                      : '2px solid transparent',
                    marginBottom: '-1px',
                    transition: 'color .18s',
                    display: 'flex', alignItems: 'center', gap: '5px',
                  }}
                >
                  {tab.icon}
                  {tab.label}
                </button>
              ))}
            </div>

            {/* Short summary tab */}
            {summaryTab === 'short' && resolvedShort && (
              <div style={{ padding: '.85rem 1rem' }}>
                <div style={{
                  background: 'linear-gradient(135deg, hsl(var(--accent) / .06), hsl(var(--accent) / .02))',
                  border: '1px solid hsl(var(--accent) / .2)',
                  borderLeft: '3px solid hsl(var(--accent))',
                  borderRadius: '8px',
                  padding: '.8rem 1rem',
                }}>
                  <p style={{
                    fontSize: '.85rem', color: 'hsl(var(--ink-soft))',
                    fontFamily: 'Inter, sans-serif', lineHeight: 1.7,
                    margin: 0,
                  }}>
                    {resolvedShort}
                  </p>
                </div>
                <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '4px' }}>
                  <CopyButton text={resolvedShort} />
                </div>
              </div>
            )}

            {/* Detailed summary tab */}
            {summaryTab === 'detailed' && (
              <div style={{ padding: '.85rem 1rem' }}>
                {resolvedDetailed ? (
                  <>
                    <MarkdownContent content={resolvedDetailed} />
                    <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '6px' }}>
                      <CopyButton text={resolvedDetailed} />
                    </div>
                  </>
                ) : (
                  <p style={{
                    fontSize: '.82rem', color: 'hsl(var(--pencil))',
                    fontFamily: 'Inter, sans-serif', fontStyle: 'italic',
                    margin: 0, textAlign: 'center', padding: '1rem 0',
                  }}>
                    Detailed summary available for newly processed recordings.
                  </p>
                )}
              </div>
            )}

            {/* Speaker summary tab */}
            {summaryTab === 'speaker' && speakerSummary && (
              <div style={{ padding: '.85rem 1rem', display: 'flex', flexDirection: 'column', gap: '14px' }}>
                {Object.entries(speakerSummary).map(([speaker, spData], idx) => (
                  <div
                    key={speaker}
                    style={{
                      background: 'hsl(var(--card))',
                      border: '1.5px solid hsl(var(--border) / .25)',
                      borderRadius: '8px',
                      padding: '.8rem 1rem',
                      display: 'flex',
                      flexDirection: 'column',
                      gap: '8px',
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <div
                        style={{
                          width: '22px',
                          height: '22px',
                          borderRadius: '50%',
                          background: `hsl(${(idx * 137) % 360}, 65%, 50% / .15)`,
                          color: `hsl(${(idx * 137) % 360}, 65%, 50%)`,
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          fontSize: '.65rem',
                          fontWeight: 700,
                          fontFamily: 'Inter, sans-serif',
                        }}
                      >
                        {speaker.slice(0, 2).toUpperCase()}
                      </div>
                      <span style={{ fontWeight: 700, fontSize: '.85rem', color: 'hsl(var(--ink))', fontFamily: 'Inter, sans-serif' }}>
                        {speaker}
                      </span>
                    </div>

                    {spData.summary && (
                      <p style={{ fontSize: '.8rem', color: 'hsl(var(--ink-soft))', margin: 0, lineHeight: 1.6, fontFamily: 'Inter, sans-serif' }}>
                        {spData.summary}
                      </p>
                    )}

                    {spData.key_points && spData.key_points.length > 0 && (
                      <div style={{ marginTop: '4px' }}>
                        <div style={{ fontSize: '.65rem', fontWeight: 750, color: 'hsl(var(--pencil))', textTransform: 'uppercase', letterSpacing: '.04em', marginBottom: '4px', fontFamily: 'Inter, sans-serif' }}>
                          Key Points
                        </div>
                        {spData.key_points.map((kp, kIdx) => (
                          <div key={kIdx} style={{ fontSize: '.78rem', color: 'hsl(var(--ink-soft))', paddingLeft: '.75rem', position: 'relative', marginBottom: '2px', lineHeight: 1.4, fontFamily: 'Inter, sans-serif' }}>
                            <span style={{ position: 'absolute', left: 0, color: 'hsl(var(--accent))' }}>•</span>
                            {kp}
                          </div>
                        ))}
                      </div>
                    )}

                    {spData.action_items && spData.action_items.length > 0 && (
                      <div style={{ marginTop: '4px' }}>
                        <div style={{ fontSize: '.65rem', fontWeight: 750, color: 'hsl(var(--pencil))', textTransform: 'uppercase', letterSpacing: '.04em', marginBottom: '4px', fontFamily: 'Inter, sans-serif' }}>
                          Action Items
                        </div>
                        {spData.action_items.map((ai, aIdx) => (
                          <div key={aIdx} style={{ fontSize: '.78rem', color: 'hsl(var(--ink-soft))', paddingLeft: '.75rem', position: 'relative', marginBottom: '2px', lineHeight: 1.4, fontFamily: 'Inter, sans-serif' }}>
                            <span style={{ position: 'absolute', left: 0, color: 'hsl(var(--success))' }}>•</span>
                            {ai}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Key points + action items */}
        {recordingId && (keyPoints?.length || actionItems?.length) ? (
          <div style={{
            padding: '.9rem',
            display: 'flex', flexDirection: 'column', gap: '10px',
          }}>
            {keyPoints && keyPoints.length > 0 && (
              <div className="animate-slide-up" style={{
                padding: '.9rem 1rem',
                background: 'hsl(var(--card))',
                borderRadius: '10px',
                border: '1px solid hsl(var(--success) / .2)',
                borderLeft: '3px solid hsl(var(--success))',
              }}>
                <div style={{
                  fontSize: '.68rem', fontWeight: 700,
                  color: 'hsl(var(--success))',
                  textTransform: 'uppercase', letterSpacing: '.09em',
                  marginBottom: '8px',
                  display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                  fontFamily: 'Inter, sans-serif',
                }}>
                  <span style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                    <FileText size={11} /> Key Points
                  </span>
                  <span style={{ background: 'hsl(var(--success) / .1)', padding: '.08rem .4rem', borderRadius: '999px', fontSize: '.62rem' }}>
                    {keyPoints.length}
                  </span>
                </div>
                {keyPoints.map((k, i) => (
                  <div key={i} style={{
                    fontSize: '.83rem', color: 'hsl(var(--ink-soft))',
                    marginBottom: '5px', paddingLeft: '.9rem', position: 'relative',
                    lineHeight: 1.65, fontFamily: 'Inter, sans-serif',
                  }}>
                    <span style={{ position: 'absolute', left: 0, color: 'hsl(var(--success))', fontWeight: 700 }}>•</span>
                    {k}
                  </div>
                ))}
              </div>
            )}

            {actionItems && actionItems.length > 0 && (
              <div className="animate-slide-up" style={{
                padding: '.9rem 1rem',
                background: 'hsl(var(--card))',
                borderRadius: '10px',
                border: '1px solid hsl(var(--accent) / .2)',
                borderLeft: '3px solid hsl(var(--accent))',
              }}>
                <div style={{
                  fontSize: '.68rem', fontWeight: 700,
                  color: 'hsl(var(--accent))',
                  textTransform: 'uppercase', letterSpacing: '.09em',
                  marginBottom: '8px',
                  display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                  fontFamily: 'Inter, sans-serif',
                }}>
                  <span style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                    <ListChecks size={11} /> Action Items
                  </span>
                  <span style={{ background: 'hsl(var(--accent) / .1)', padding: '.08rem .4rem', borderRadius: '999px', fontSize: '.62rem' }}>
                    {actionItems.length}
                  </span>
                </div>
                {actionItems.map((a, i) => (
                  <div key={i} style={{
                    fontSize: '.83rem', color: 'hsl(var(--ink-soft))',
                    marginBottom: '5px', paddingLeft: '1.25rem', position: 'relative',
                    lineHeight: 1.65, fontFamily: 'Inter, sans-serif',
                  }}>
                    <span style={{ position: 'absolute', left: 0, color: 'hsl(var(--pencil))', fontSize: '.9rem' }}>☐</span>
                    {a}
                  </div>
                ))}
              </div>
            )}
          </div>
        ) : null}

      </div>{/* end scroll container */}

      {/* Footer — offline indicator */}
      <div style={{
        padding: '.55rem 1rem',
        borderTop: '1px solid hsl(var(--border) / .15)',
        display: 'flex', alignItems: 'center', gap: '6px',
        flexShrink: 0,
        background: 'hsl(var(--card))',
      }}>
        <div style={{
          width: '7px', height: '7px', borderRadius: '50%',
          background: 'hsl(var(--success))',
          boxShadow: '0 0 6px hsl(var(--success) / .6)',
          flexShrink: 0,
        }} />
        <span style={{
          fontSize: '.65rem', color: 'hsl(var(--pencil))',
          fontFamily: 'Inter, sans-serif',
        }}>
          Qwen-AI · Running locally · No internet required
        </span>
      </div>
    </div>
  )
}
