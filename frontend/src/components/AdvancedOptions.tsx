import { useState, useEffect, useCallback } from 'react'
import {
  ChevronDown, ChevronUp, MessageSquare, Users, Mic, BookOpen, Sparkles,
  Search, CheckSquare, Square,
} from 'lucide-react'
import api from '../api/client'

export interface AdvancedOptions {
  meetingPrompt: string
  expectedSpeakers: number | null
  selectedVoiceIds: string[]
  useDictionary: boolean
  useVocabularyInPrompt: boolean
  speakerSummary: boolean
}

interface VoiceProfile {
  id: string
  label: string
  sample_count: number
  is_self: boolean
}

interface Props {
  onChange: (opts: AdvancedOptions) => void
  defaults?: Partial<AdvancedOptions>
}

const DEFAULT_OPTS: AdvancedOptions = {
  meetingPrompt: '',
  expectedSpeakers: null,
  selectedVoiceIds: [],
  useDictionary: false,
  useVocabularyInPrompt: false,
  speakerSummary: false,
}

export default function AdvancedOptions({ onChange, defaults }: Props) {
  const [open, setOpen] = useState(false)
  const [opts, setOpts] = useState<AdvancedOptions>({ ...DEFAULT_OPTS, ...defaults })
  const [profiles, setProfiles] = useState<VoiceProfile[]>([])
  const [profileSearch, setProfileSearch] = useState('')
  const [loadingProfiles, setLoadingProfiles] = useState(false)

  // Load voice profiles lazily when accordion opens
  useEffect(() => {
    if (!open || profiles.length > 0) return
    setLoadingProfiles(true)
    api.get('/voice/profiles')
      .then(res => setProfiles(res.data || []))
      .catch(() => {})
      .finally(() => setLoadingProfiles(false))
  }, [open])

  const update = useCallback(<K extends keyof AdvancedOptions>(key: K, value: AdvancedOptions[K]) => {
    setOpts(prev => {
      const next = { ...prev, [key]: value }
      onChange(next)
      return next
    })
  }, [onChange])

  const toggleVoiceId = (id: string) => {
    const next = opts.selectedVoiceIds.includes(id)
      ? opts.selectedVoiceIds.filter(v => v !== id)
      : [...opts.selectedVoiceIds, id]
    update('selectedVoiceIds', next)
  }

  const filteredProfiles = profiles.filter(p =>
    p.label.toLowerCase().includes(profileSearch.toLowerCase())
  )

  return (
    <div
      className="animate-slide-up sketch-border"
      style={{
        borderRadius: '12px',
        background: 'hsl(var(--card))',
        border: '1.5px solid hsl(var(--ink) / .1)',
        overflow: 'hidden',
        marginTop: '1rem',
      }}
    >
      {/* Accordion header */}
      <button
        onClick={() => setOpen(o => !o)}
        style={{
          width: '100%',
          display: 'flex',
          alignItems: 'center',
          gap: '10px',
          padding: '1rem 1.25rem',
          background: 'transparent',
          border: 'none',
          cursor: 'pointer',
          color: 'hsl(var(--ink))',
          fontFamily: 'Inter, sans-serif',
          fontWeight: 600,
          fontSize: '.9rem',
          transition: 'background .15s',
        }}
        onMouseEnter={e => (e.currentTarget.style.background = 'hsl(var(--muted) / .5)')}
        onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}
        id="advanced-options-toggle"
        aria-expanded={open}
      >
        <div style={{
          width: 28, height: 28, borderRadius: '8px',
          background: 'hsl(var(--accent) / .1)',
          border: '1.5px solid hsl(var(--accent) / .25)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          flexShrink: 0,
        }}>
          <Sparkles size={13} style={{ color: 'hsl(var(--accent))' }} />
        </div>
        <span style={{ flex: 1, textAlign: 'left' }}>Advanced Options</span>
        {(opts.meetingPrompt || opts.selectedVoiceIds.length > 0 || opts.useDictionary || opts.useVocabularyInPrompt || opts.speakerSummary) && (
          <span style={{
            fontSize: '.7rem', fontWeight: 700,
            color: 'hsl(var(--accent))',
            background: 'hsl(var(--accent) / .1)',
            border: '1.5px solid hsl(var(--accent) / .2)',
            padding: '.15rem .5rem',
            borderRadius: '999px',
          }}>Active</span>
        )}
        {open
          ? <ChevronUp size={15} style={{ color: 'hsl(var(--pencil))' }} />
          : <ChevronDown size={15} style={{ color: 'hsl(var(--pencil))' }} />
        }
      </button>

      {/* Accordion body */}
      {open && (
        <div style={{
          padding: '0 1.25rem 1.25rem',
          display: 'flex',
          flexDirection: 'column',
          gap: '1.5rem',
          borderTop: '1px solid hsl(var(--ink) / .08)',
        }}>

          {/* ── Meeting Prompt ─────────────────────────────────── */}
          <div style={{ paddingTop: '1rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '.6rem' }}>
              <MessageSquare size={14} style={{ color: 'hsl(var(--accent))' }} />
              <label style={{ fontSize: '.85rem', fontWeight: 600, fontFamily: 'Inter, sans-serif', color: 'hsl(var(--ink))' }}>
                Meeting Prompt
              </label>
              <span style={{
                fontSize: '.68rem', color: 'hsl(var(--pencil))',
                background: 'hsl(var(--muted))', borderRadius: '999px',
                padding: '.1rem .5rem', fontFamily: 'Inter, sans-serif',
              }}>optional</span>
            </div>
            <textarea
              id="meeting-prompt-input"
              value={opts.meetingPrompt}
              onChange={e => update('meetingPrompt', e.target.value)}
              placeholder="Describe the meeting: agenda, project, client, technologies being discussed, or any important context…"
              rows={3}
              style={{
                width: '100%',
                padding: '.75rem',
                borderRadius: '8px',
                background: 'hsl(var(--muted) / .5)',
                border: '1.5px solid hsl(var(--ink) / .1)',
                color: 'hsl(var(--ink))',
                fontFamily: 'Inter, sans-serif',
                fontSize: '.85rem',
                lineHeight: 1.6,
                resize: 'vertical',
                outline: 'none',
                boxSizing: 'border-box',
                transition: 'border-color .15s',
              }}
              onFocus={e => (e.currentTarget.style.borderColor = 'hsl(var(--accent) / .5)')}
              onBlur={e => (e.currentTarget.style.borderColor = 'hsl(var(--ink) / .1)')}
            />
            <p style={{ fontSize: '.75rem', color: 'hsl(var(--pencil))', marginTop: '.35rem', fontFamily: 'Inter, sans-serif' }}>
              Appended after the Global Prompt to guide Whisper transcription.
            </p>
          </div>

          {/* ── Expected Participants ──────────────────────────── */}
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '.6rem' }}>
              <Users size={14} style={{ color: 'hsl(205,90%,55%)' }} />
              <label style={{ fontSize: '.85rem', fontWeight: 600, fontFamily: 'Inter, sans-serif', color: 'hsl(var(--ink))' }}>
                Expected Participants
              </label>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '.85rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{ fontSize: '.82rem', color: 'hsl(var(--pencil))', fontFamily: 'Inter, sans-serif' }}>
                  Expected speakers:
                </span>
                <input
                  id="expected-speakers-input"
                  type="number"
                  min={1}
                  max={20}
                  value={opts.expectedSpeakers ?? ''}
                  onChange={e => update('expectedSpeakers', e.target.value ? parseInt(e.target.value) : null)}
                  placeholder="Auto"
                  style={{
                    width: '72px',
                    padding: '.35rem .6rem',
                    borderRadius: '8px',
                    border: '1.5px solid hsl(var(--ink) / .15)',
                    background: 'hsl(var(--muted) / .5)',
                    color: 'hsl(var(--ink))',
                    fontFamily: 'Inter, sans-serif',
                    fontSize: '.85rem',
                    outline: 'none',
                  }}
                />
              </div>
            </div>

            {/* Voice profile selector */}
            <div style={{
              border: '1.5px solid hsl(var(--ink) / .1)',
              borderRadius: '10px',
              overflow: 'hidden',
            }}>
              <div style={{
                padding: '.6rem .75rem',
                background: 'hsl(var(--muted) / .4)',
                borderBottom: '1px solid hsl(var(--ink) / .08)',
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
              }}>
                <Mic size={12} style={{ color: 'hsl(var(--pencil))' }} />
                <span style={{ fontSize: '.75rem', fontWeight: 600, color: 'hsl(var(--pencil))', fontFamily: 'Inter, sans-serif', textTransform: 'uppercase', letterSpacing: '.05em' }}>
                  Select voice profiles for identification
                </span>
                {opts.selectedVoiceIds.length > 0 && (
                  <span style={{
                    marginLeft: 'auto', fontSize: '.7rem', fontWeight: 700,
                    color: 'hsl(205,90%,55%)',
                    background: 'hsl(205,90%,55% / .1)',
                    border: '1px solid hsl(205,90%,55% / .25)',
                    padding: '.1rem .5rem', borderRadius: '999px',
                  }}>
                    {opts.selectedVoiceIds.length} selected
                  </span>
                )}
              </div>

              {profiles.length > 4 && (
                <div style={{ padding: '.5rem .75rem', borderBottom: '1px solid hsl(var(--ink) / .06)' }}>
                  <div style={{ position: 'relative' }}>
                    <Search size={12} style={{ position: 'absolute', left: 8, top: '50%', transform: 'translateY(-50%)', color: 'hsl(var(--pencil))' }} />
                    <input
                      type="text"
                      value={profileSearch}
                      onChange={e => setProfileSearch(e.target.value)}
                      placeholder="Search profiles…"
                      style={{
                        width: '100%',
                        paddingLeft: '26px',
                        padding: '.3rem .5rem .3rem 26px',
                        borderRadius: '6px',
                        border: '1.5px solid hsl(var(--ink) / .1)',
                        background: 'hsl(var(--card))',
                        color: 'hsl(var(--ink))',
                        fontFamily: 'Inter, sans-serif',
                        fontSize: '.8rem',
                        outline: 'none',
                        boxSizing: 'border-box',
                      }}
                    />
                  </div>
                </div>
              )}

              <div style={{ maxHeight: '200px', overflowY: 'auto' }}>
                {loadingProfiles ? (
                  <div style={{ padding: '1.5rem', textAlign: 'center', color: 'hsl(var(--pencil))', fontSize: '.82rem', fontFamily: 'Inter, sans-serif' }}>
                    Loading profiles…
                  </div>
                ) : filteredProfiles.length === 0 ? (
                  <div style={{ padding: '1.5rem', textAlign: 'center', color: 'hsl(var(--pencil))', fontSize: '.82rem', fontFamily: 'Inter, sans-serif' }}>
                    {profiles.length === 0 ? 'No voice profiles enrolled yet' : 'No profiles match search'}
                  </div>
                ) : (
                  filteredProfiles.map(p => {
                    const selected = opts.selectedVoiceIds.includes(p.id)
                    return (
                      <button
                        key={p.id}
                        onClick={() => toggleVoiceId(p.id)}
                        style={{
                          width: '100%',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '10px',
                          padding: '.6rem .75rem',
                          background: selected ? 'hsl(205,90%,55% / .07)' : 'transparent',
                          border: 'none',
                          borderBottom: '1px solid hsl(var(--ink) / .05)',
                          cursor: 'pointer',
                          transition: 'background .12s',
                          textAlign: 'left',
                        }}
                        onMouseEnter={e => !selected && (e.currentTarget.style.background = 'hsl(var(--muted) / .5)')}
                        onMouseLeave={e => !selected && (e.currentTarget.style.background = 'transparent')}
                      >
                        {selected
                          ? <CheckSquare size={15} style={{ color: 'hsl(205,90%,55%)', flexShrink: 0 }} />
                          : <Square size={15} style={{ color: 'hsl(var(--pencil))', flexShrink: 0 }} />
                        }
                        <div style={{
                          width: 28, height: 28, borderRadius: '50%',
                          background: 'hsl(var(--accent) / .12)',
                          display: 'flex', alignItems: 'center', justifyContent: 'center',
                          fontSize: '.75rem', fontWeight: 800, color: 'hsl(var(--accent))',
                          flexShrink: 0, fontFamily: 'Inter, sans-serif',
                        }}>
                          {p.label[0]?.toUpperCase()}
                        </div>
                        <div style={{ flex: 1, minWidth: 0 }}>
                          <div style={{ fontWeight: 600, fontSize: '.85rem', color: 'hsl(var(--ink))', fontFamily: 'Inter, sans-serif', display: 'flex', alignItems: 'center', gap: '6px' }}>
                            {p.label}
                            {p.is_self && (
                              <span style={{ fontSize: '.65rem', color: 'hsl(235,80%,60%)', background: 'hsl(235,80%,60% / .1)', padding: '.1rem .4rem', borderRadius: '999px', border: '1px solid hsl(235,80%,60% / .2)' }}>
                                You
                              </span>
                            )}
                          </div>
                          <div style={{ fontSize: '.72rem', color: 'hsl(var(--pencil))', fontFamily: 'Inter, sans-serif' }}>
                            {p.sample_count} sample{p.sample_count !== 1 ? 's' : ''}
                          </div>
                        </div>
                      </button>
                    )
                  })
                )}
              </div>
              {opts.selectedVoiceIds.length > 0 && (
                <div style={{ padding: '.5rem .75rem', borderTop: '1px solid hsl(var(--ink) / .06)', display: 'flex', justifyContent: 'flex-end' }}>
                  <button
                    onClick={() => update('selectedVoiceIds', [])}
                    style={{
                      fontSize: '.75rem', color: 'hsl(var(--destructive) / .7)',
                      background: 'none', border: 'none', cursor: 'pointer',
                      fontFamily: 'Inter, sans-serif', padding: '.2rem .5rem',
                      borderRadius: '6px',
                    }}
                    onMouseEnter={e => (e.currentTarget.style.color = 'hsl(var(--destructive))')}
                    onMouseLeave={e => (e.currentTarget.style.color = 'hsl(var(--destructive) / .7)')}
                  >
                    Clear selection
                  </button>
                </div>
              )}
            </div>
          </div>

          {/* ── Dictionary Toggles ─────────────────────────────── */}
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '.75rem' }}>
              <BookOpen size={14} style={{ color: 'hsl(130,60%,45%)' }} />
              <span style={{ fontSize: '.85rem', fontWeight: 600, fontFamily: 'Inter, sans-serif', color: 'hsl(var(--ink))' }}>
                Dictionary Options
              </span>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '.65rem' }}>
              {([
                {
                  key: 'useDictionary' as const,
                  label: 'Use Dictionary for Transcription',
                  desc: 'Apply shortcut expansion to the transcript text after processing.',
                  color: 'hsl(130,60%,45%)',
                },
                {
                  key: 'useVocabularyInPrompt' as const,
                  label: 'Inject Technical Vocabulary into Whisper Prompt',
                  desc: 'Technical words are added to the initial_prompt so Whisper recognises domain-specific terms.',
                  color: 'hsl(var(--accent))',
                },
                {
                  key: 'speakerSummary' as const,
                  label: 'Speaker Summary',
                  desc: 'Generate per-speaker summary, key points and action items in addition to the overall meeting summary.',
                  color: 'hsl(260,75%,60%)',
                },
              ] as const).map(({ key, label, desc, color }) => (
                <button
                  key={key}
                  id={`toggle-${key}`}
                  onClick={() => update(key, !opts[key])}
                  style={{
                    display: 'flex',
                    alignItems: 'flex-start',
                    gap: '12px',
                    padding: '.75rem',
                    borderRadius: '10px',
                    background: opts[key] ? `${color.replace(')', ' / .07)')}` : 'hsl(var(--muted) / .3)',
                    border: `1.5px solid ${opts[key] ? color.replace(')', ' / .25)') : 'hsl(var(--ink) / .1)'}`,
                    cursor: 'pointer',
                    textAlign: 'left',
                    transition: 'all .15s',
                    width: '100%',
                  }}
                >
                  {/* Toggle pill */}
                  <div style={{
                    width: 36, height: 20, borderRadius: 999, flexShrink: 0,
                    background: opts[key] ? color : 'hsl(var(--muted))',
                    position: 'relative',
                    transition: 'background .2s',
                    marginTop: '1px',
                    border: `1.5px solid ${opts[key] ? color.replace(')', ' / .5)') : 'hsl(var(--ink) / .15)'}`,
                  }}>
                    <div style={{
                      width: 13, height: 13, borderRadius: '50%',
                      background: 'white',
                      position: 'absolute',
                      top: '50%',
                      left: opts[key] ? 'calc(100% - 15px)' : '2px',
                      transform: 'translateY(-50%)',
                      transition: 'left .2s',
                      boxShadow: '0 1px 4px rgba(0,0,0,.2)',
                    }} />
                  </div>
                  <div>
                    <div style={{ fontWeight: 600, fontSize: '.85rem', color: 'hsl(var(--ink))', fontFamily: 'Inter, sans-serif' }}>
                      {label}
                    </div>
                    <div style={{ fontSize: '.75rem', color: 'hsl(var(--pencil))', fontFamily: 'Inter, sans-serif', marginTop: '2px', lineHeight: 1.5 }}>
                      {desc}
                    </div>
                  </div>
                </button>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
