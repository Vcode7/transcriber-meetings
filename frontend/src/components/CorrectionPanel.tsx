import { useState, useEffect, useCallback } from 'react'
import {
  CheckSquare, Square, ChevronDown, ChevronRight, Loader,
  BookOpen, AlertCircle, Edit2, Check, X, RotateCcw, Zap, Save
} from 'lucide-react'
import { toast } from 'sonner'
import api from '../api/client'

// =============================================================================
// Types
// =============================================================================

interface CorrectionEntry {
  wrong: string
  correct: string
  user_correct?: string
  enabled: boolean
}

interface AcronymEntry {
  acronym: string
  full_form: string | null
  user_full_form?: string
  is_known: boolean
  enabled: boolean
  not_an_acronym: boolean
  first_occurrence?: number
}

interface CorrectionData {
  proposed: Array<{ wrong: string; correct: string }>
  decisions: CorrectionEntry[]
  applied: boolean
  has_data: boolean
}

interface AcronymData {
  detected: AcronymEntry[]
  decisions: AcronymEntry[]
  applied: boolean
  has_data: boolean
}

interface CorrectionPanelProps {
  recordingId: string
  onTranscriptChanged: () => void
}

// =============================================================================
// Helpers
// =============================================================================

function mergeCorrectionDecisions(
  proposed: Array<{ wrong: string; correct: string }>,
  decisions: CorrectionEntry[]
): CorrectionEntry[] {
  const decMap = new Map(decisions.map(d => [d.wrong, d]))
  return proposed.map(p => {
    const saved = decMap.get(p.wrong)
    return saved ?? { wrong: p.wrong, correct: p.correct, enabled: true }
  })
}

function mergeAcronymDecisions(
  detected: AcronymEntry[],
  decisions: AcronymEntry[]
): AcronymEntry[] {
  const decMap = new Map(decisions.map(d => [d.acronym, d]))
  return detected.map(d => {
    const saved = decMap.get(d.acronym)
    return saved ?? { ...d, enabled: true, not_an_acronym: false }
  })
}

// =============================================================================
// Sub-components
// =============================================================================

function SectionHeader({ title, count, color }: { title: string; count: number; color: string }) {
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: '8px',
      padding: '.45rem .75rem',
      background: `${color}12`,
      borderRadius: '8px',
      border: `1px solid ${color}30`,
      marginBottom: '.5rem',
    }}>
      <span style={{ fontWeight: 700, fontSize: '.78rem', color, fontFamily: 'Inter, sans-serif' }}>{title}</span>
      <span style={{
        fontSize: '.68rem', fontWeight: 600,
        background: `${color}22`, color,
        padding: '.1rem .45rem', borderRadius: '999px',
        fontFamily: 'JetBrains Mono, monospace',
      }}>{count}</span>
    </div>
  )
}

function ToggleSwitch({ value, onChange }: { value: boolean; onChange: (v: boolean) => void }) {
  return (
    <button
      onClick={() => onChange(!value)}
      style={{
        width: 32, height: 18, borderRadius: 999, border: 'none', cursor: 'pointer',
        background: value ? 'hsl(160,70%,45%)' : 'hsl(var(--muted))',
        position: 'relative', flexShrink: 0, transition: 'background .2s',
        boxShadow: 'inset 0 1px 3px rgba(0,0,0,0.15)',
      }}
      aria-label={value ? 'Disable' : 'Enable'}
    >
      <span style={{
        position: 'absolute', top: 2, left: value ? 16 : 2,
        width: 14, height: 14, borderRadius: '50%',
        background: 'white', transition: 'left .2s',
        boxShadow: '0 1px 3px rgba(0,0,0,0.2)',
      }} />
    </button>
  )
}

// =============================================================================
// Main component
// =============================================================================

export default function CorrectionPanel({ recordingId, onTranscriptChanged }: CorrectionPanelProps) {
  const [activeTab, setActiveTab] = useState<'corrections' | 'acronyms'>('corrections')
  const [loading, setLoading] = useState(true)
  const [running, setRunning] = useState(false)
  const [applying, setApplying] = useState(false)
  const [reverting, setReverting] = useState(false)

  const [corrData, setCorrData] = useState<CorrectionData | null>(null)
  const [acrData, setAcrData] = useState<AcronymData | null>(null)

  // Editable decisions
  const [corrDecisions, setCorrDecisions] = useState<CorrectionEntry[]>([])
  const [acrDecisions, setAcrDecisions] = useState<AcronymEntry[]>([])

  // Track which correction is being edited
  const [editingIdx, setEditingIdx] = useState<number | null>(null)
  const [editValue, setEditValue] = useState('')

  // Unknown acronym input state
  const [acrInputs, setAcrInputs] = useState<Record<string, string>>({})
  const [savingDict, setSavingDict] = useState<Record<string, boolean>>({})

  // Decisions dirty flag
  const [dirty, setDirty] = useState(false)

  // ── Load data on mount ────────────────────────────────────────────────────
  const loadData = useCallback(async () => {
    setLoading(true)
    try {
      const res = await api.get(`/corrections/${recordingId}`)
      const { corrections, acronyms } = res.data
      setCorrData(corrections)
      setAcrData(acronyms)
      // Merge proposals with saved decisions
      setCorrDecisions(mergeCorrectionDecisions(corrections.proposed, corrections.decisions))
      setAcrDecisions(mergeAcronymDecisions(acronyms.detected, acronyms.decisions))
      setDirty(false)
    } catch {
      setCorrData(null)
      setAcrData(null)
    } finally {
      setLoading(false)
    }
  }, [recordingId])

  useEffect(() => { loadData() }, [loadData])

  // ── Run detection (on-demand) ─────────────────────────────────────────────
  const handleRun = async () => {
    if (running) return
    setRunning(true)
    try {
      await api.post(`/corrections/${recordingId}/run?force=true`)
      toast.success('Correction detection complete')
      await loadData()
    } catch (err: unknown) {
      const msg = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ?? 'Detection failed'
      toast.error(msg)
    } finally {
      setRunning(false)
    }
  }

  // ── Save decisions (auto-saves) ────────────────────────────────────────────
  const saveDecisions = useCallback(async (cd: CorrectionEntry[], ad: AcronymEntry[]) => {
    try {
      await api.put(`/corrections/${recordingId}/decisions`, {
        correction_decisions: cd,
        acronym_decisions: ad,
      })
      setDirty(false)
    } catch {
      // non-critical
    }
  }, [recordingId])

  // ── Apply to transcript ────────────────────────────────────────────────────
  const handleApply = async () => {
    if (applying) return
    setApplying(true)
    try {
      // Save latest decisions first
      await saveDecisions(corrDecisions, acrDecisions)
      const res = await api.post(`/corrections/${recordingId}/apply`)
      toast.success(`Corrections applied — ${res.data.corrections_applied} replacement(s)`)
      await loadData()
      onTranscriptChanged()
    } catch (err: unknown) {
      const msg = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ?? 'Apply failed'
      toast.error(msg)
    } finally {
      setApplying(false)
    }
  }

  // ── Revert ────────────────────────────────────────────────────────────────
  const handleRevert = async () => {
    if (reverting) return
    setReverting(true)
    try {
      await api.post(`/corrections/${recordingId}/revert`)
      toast.success('Original transcript restored')
      await loadData()
      onTranscriptChanged()
    } catch (err: unknown) {
      const msg = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ?? 'Revert failed'
      toast.error(msg)
    } finally {
      setReverting(false)
    }
  }

  // ── Correction decision helpers ───────────────────────────────────────────
  const toggleCorrection = (idx: number) => {
    const updated = corrDecisions.map((d, i) => i === idx ? { ...d, enabled: !d.enabled } : d)
    setCorrDecisions(updated)
    setDirty(true)
    saveDecisions(updated, acrDecisions)
  }

  const acceptAllCorrections = () => {
    const updated = corrDecisions.map(d => ({ ...d, enabled: true }))
    setCorrDecisions(updated)
    setDirty(true)
    saveDecisions(updated, acrDecisions)
  }

  const rejectAllCorrections = () => {
    const updated = corrDecisions.map(d => ({ ...d, enabled: false }))
    setCorrDecisions(updated)
    setDirty(true)
    saveDecisions(updated, acrDecisions)
  }

  const startEdit = (idx: number) => {
    setEditingIdx(idx)
    setEditValue(corrDecisions[idx].user_correct ?? corrDecisions[idx].correct)
  }

  const commitEdit = (idx: number) => {
    const val = editValue.trim()
    if (val) {
      const updated = corrDecisions.map((d, i) => i === idx ? { ...d, user_correct: val } : d)
      setCorrDecisions(updated)
      setDirty(true)
      saveDecisions(updated, acrDecisions)
    }
    setEditingIdx(null)
    setEditValue('')
  }

  // ── Acronym decision helpers ──────────────────────────────────────────────
  const toggleAcronym = (idx: number) => {
    const updated = acrDecisions.map((d, i) => i === idx ? { ...d, enabled: !d.enabled } : d)
    setAcrDecisions(updated)
    setDirty(true)
    saveDecisions(corrDecisions, updated)
  }

  const markNotAcronym = (idx: number) => {
    const updated = acrDecisions.map((d, i) =>
      i === idx ? { ...d, not_an_acronym: true, enabled: false } : d
    )
    setAcrDecisions(updated)
    setDirty(true)
    saveDecisions(corrDecisions, updated)
  }

  const saveToDict = async (acr: string, idx: number) => {
    const fullForm = (acrInputs[acr] || '').trim()
    if (!fullForm) { toast.error('Enter a full form first'); return }
    setSavingDict(prev => ({ ...prev, [acr]: true }))
    try {
      await api.post('/corrections/acronym-dictionary', { acronym: acr, full_form: fullForm })
      // Update local decision with the user-entered full form
      const updated = acrDecisions.map((d, i) =>
        i === idx ? { ...d, user_full_form: fullForm, is_known: true, enabled: true } : d
      )
      setAcrDecisions(updated)
      setDirty(true)
      saveDecisions(corrDecisions, updated)
      toast.success(`Saved "${acr} → ${fullForm}" to dictionary`)
    } catch {
      toast.error('Failed to save to dictionary')
    } finally {
      setSavingDict(prev => ({ ...prev, [acr]: false }))
    }
  }

  const isApplied = corrData?.applied || acrData?.applied

  // =============================================================================
  // Render
  // =============================================================================

  const panelStyle: React.CSSProperties = {
    display: 'flex', flexDirection: 'column', height: '100%',
    background: 'hsl(var(--card))',
    borderLeft: '1px solid hsl(var(--border)/.6)',
    fontFamily: 'Inter, sans-serif',
    overflow: 'hidden',
  }

  const tabStyle = (active: boolean): React.CSSProperties => ({
    flex: 1, padding: '.45rem .6rem', border: 'none', cursor: 'pointer',
    background: active ? 'hsl(var(--accent)/.15)' : 'transparent',
    color: active ? 'hsl(var(--accent))' : 'hsl(var(--pencil))',
    fontWeight: active ? 700 : 500, fontSize: '.78rem',
    borderBottom: active ? '2px solid hsl(var(--accent))' : '2px solid transparent',
    transition: 'all .15s ease', fontFamily: 'Inter, sans-serif',
  })

  if (loading) return (
    <div style={{ ...panelStyle, alignItems: 'center', justifyContent: 'center', gap: '10px' }}>
      <Loader size={18} className="spin" style={{ color: 'hsl(var(--accent))' }} />
      <span style={{ fontSize: '.8rem', color: 'hsl(var(--pencil))' }}>Loading…</span>
    </div>
  )

  const hasNoData = !corrData?.has_data && !acrData?.has_data
  const enabledCorrCount = corrDecisions.filter(d => d.enabled).length
  const enabledAcrCount = acrDecisions.filter(d => d.enabled && !d.not_an_acronym && (d.user_full_form || d.full_form)).length

  return (
    <div style={panelStyle}>
      {/* Header */}
      <div style={{
        padding: '.75rem 1rem .5rem', borderBottom: '1px solid hsl(var(--border)/.5)',
        display: 'flex', flexDirection: 'column', gap: '8px',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <div style={{
              width: 28, height: 28, borderRadius: '8px',
              background: 'hsl(220,80%,60%/.15)', border: '1px solid hsl(220,80%,60%/.3)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
            }}>
              <Zap size={14} style={{ color: 'hsl(220,80%,60%)' }} />
            </div>
            <span style={{ fontWeight: 700, fontSize: '.88rem', color: 'hsl(var(--ink))' }}>
              Corrections
            </span>
          </div>
          <button
            onClick={handleRun}
            disabled={running}
            style={{
              display: 'flex', alignItems: 'center', gap: '4px',
              fontSize: '.72rem', fontWeight: 600,
              padding: '.3rem .65rem', borderRadius: '7px',
              background: 'hsl(220,80%,60%/.12)', border: '1px solid hsl(220,80%,60%/.3)',
              color: 'hsl(220,80%,60%)', cursor: 'pointer',
            }}
            title="Re-run LLM correction detection"
          >
            {running ? <Loader size={11} className="spin" /> : <RotateCcw size={11} />}
            {running ? 'Detecting…' : 'Re-detect'}
          </button>
        </div>

        {/* Status banner */}
        {isApplied && (
          <div style={{
            fontSize: '.72rem', fontWeight: 600, padding: '.3rem .65rem', borderRadius: '7px',
            background: 'hsl(160,60%,45%/.12)', border: '1px solid hsl(160,60%,45%/.3)',
            color: 'hsl(160,60%,35%)', display: 'flex', alignItems: 'center', gap: '5px',
          }}>
            <Check size={11} /> Corrections applied to transcript
          </div>
        )}
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', borderBottom: '1px solid hsl(var(--border)/.4)' }}>
        <button style={tabStyle(activeTab === 'corrections')} onClick={() => setActiveTab('corrections')}>
          Corrections
          {corrDecisions.length > 0 && (
            <span style={{
              marginLeft: '5px', fontSize: '.65rem', fontWeight: 700,
              background: activeTab === 'corrections' ? 'hsl(var(--accent)/.2)' : 'hsl(var(--muted))',
              color: activeTab === 'corrections' ? 'hsl(var(--accent))' : 'hsl(var(--pencil))',
              padding: '0 .35rem', borderRadius: '999px',
            }}>{corrDecisions.length}</span>
          )}
        </button>
        <button style={tabStyle(activeTab === 'acronyms')} onClick={() => setActiveTab('acronyms')}>
          Acronyms
          {acrDecisions.length > 0 && (
            <span style={{
              marginLeft: '5px', fontSize: '.65rem', fontWeight: 700,
              background: activeTab === 'acronyms' ? 'hsl(var(--accent)/.2)' : 'hsl(var(--muted))',
              color: activeTab === 'acronyms' ? 'hsl(var(--accent))' : 'hsl(var(--pencil))',
              padding: '0 .35rem', borderRadius: '999px',
            }}>{acrDecisions.length}</span>
          )}
        </button>
      </div>

      {/* Content */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '.75rem 1rem' }}>

        {hasNoData ? (
          <div style={{
            display: 'flex', flexDirection: 'column', alignItems: 'center',
            gap: '12px', padding: '2rem 1rem', textAlign: 'center',
          }}>
            <div style={{
              width: 44, height: 44, borderRadius: '12px',
              background: 'hsl(var(--muted))', border: '1px solid hsl(var(--border))',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
            }}>
              <AlertCircle size={20} style={{ color: 'hsl(var(--pencil))' }} />
            </div>
            <p style={{ fontSize: '.8rem', color: 'hsl(var(--pencil))', margin: 0 }}>
              No correction data yet. Click <strong>Re-detect</strong> to analyse this transcript.
            </p>
          </div>
        ) : activeTab === 'corrections' ? (
          /* ── Corrections Tab ───────────────────────────────── */
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            {corrDecisions.length === 0 ? (
              <p style={{ fontSize: '.8rem', color: 'hsl(var(--pencil))', textAlign: 'center', marginTop: '1.5rem' }}>
                ✓ No spelling corrections detected.
              </p>
            ) : (
              <>
                <SectionHeader title="Detected Corrections" count={corrDecisions.length} color="hsl(45,90%,50%)" />
                {/* Bulk actions */}
                <div style={{ display: 'flex', gap: '6px', marginBottom: '.25rem' }}>
                  <button
                    onClick={acceptAllCorrections}
                    style={{
                      fontSize: '.7rem', fontWeight: 600, padding: '.25rem .6rem', borderRadius: '6px',
                      background: 'hsl(160,60%,45%/.12)', border: '1px solid hsl(160,60%,45%/.3)',
                      color: 'hsl(160,60%,35%)', cursor: 'pointer',
                    }}
                  >Accept All</button>
                  <button
                    onClick={rejectAllCorrections}
                    style={{
                      fontSize: '.7rem', fontWeight: 600, padding: '.25rem .6rem', borderRadius: '6px',
                      background: 'hsl(0,65%,55%/.1)', border: '1px solid hsl(0,65%,55%/.25)',
                      color: 'hsl(0,65%,45%)', cursor: 'pointer',
                    }}
                  >Reject All</button>
                </div>

                {corrDecisions.map((dec, idx) => (
                  <div key={idx} style={{
                    padding: '.6rem .75rem', borderRadius: '10px',
                    background: dec.enabled ? 'hsl(var(--card))' : 'hsl(var(--muted)/.5)',
                    border: `1px solid ${dec.enabled ? 'hsl(45,90%,50%/.3)' : 'hsl(var(--border)/.4)'}`,
                    display: 'flex', flexDirection: 'column', gap: '6px',
                    opacity: dec.enabled ? 1 : 0.6,
                    transition: 'all .15s',
                  }}>
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flex: 1, minWidth: 0 }}>
                        {/* wrong */}
                        <span style={{
                          fontFamily: 'JetBrains Mono, monospace', fontSize: '.78rem',
                          background: 'hsl(0,70%,55%/.12)', color: 'hsl(0,65%,45%)',
                          padding: '.1rem .45rem', borderRadius: '5px', flexShrink: 0,
                          textDecoration: 'line-through',
                        }}>{dec.wrong}</span>
                        <span style={{ color: 'hsl(var(--pencil))', fontSize: '.72rem', flexShrink: 0 }}>→</span>
                        {/* correct (editable) */}
                        {editingIdx === idx ? (
                          <div style={{ display: 'flex', alignItems: 'center', gap: '4px', flex: 1 }}>
                            <input
                              value={editValue}
                              onChange={e => setEditValue(e.target.value)}
                              onKeyDown={e => { if (e.key === 'Enter') commitEdit(idx); if (e.key === 'Escape') { setEditingIdx(null) } }}
                              autoFocus
                              style={{
                                flex: 1, minWidth: 0, fontFamily: 'JetBrains Mono, monospace',
                                fontSize: '.78rem', padding: '.15rem .4rem', borderRadius: '5px',
                                border: '1.5px solid hsl(var(--accent))',
                                background: 'hsl(var(--paper))', color: 'hsl(var(--ink))',
                              }}
                            />
                            <button onClick={() => commitEdit(idx)} style={{ border: 'none', background: 'none', cursor: 'pointer', color: 'hsl(160,60%,40%)' }}><Check size={13} /></button>
                            <button onClick={() => setEditingIdx(null)} style={{ border: 'none', background: 'none', cursor: 'pointer', color: 'hsl(var(--pencil))' }}><X size={13} /></button>
                          </div>
                        ) : (
                          <div style={{ display: 'flex', alignItems: 'center', gap: '4px', flex: 1, minWidth: 0 }}>
                            <span style={{
                              fontFamily: 'JetBrains Mono, monospace', fontSize: '.78rem',
                              background: 'hsl(160,60%,45%/.12)', color: 'hsl(160,55%,35%)',
                              padding: '.1rem .45rem', borderRadius: '5px', flexShrink: 0,
                            }}>{dec.user_correct || dec.correct}</span>
                            <button
                              onClick={() => startEdit(idx)}
                              style={{ border: 'none', background: 'none', cursor: 'pointer', color: 'hsl(var(--pencil))' }}
                              title="Edit correction"
                            ><Edit2 size={11} /></button>
                          </div>
                        )}
                      </div>
                      <ToggleSwitch value={dec.enabled} onChange={() => toggleCorrection(idx)} />
                    </div>
                  </div>
                ))}
              </>
            )}
          </div>
        ) : (
          /* ── Acronyms Tab ──────────────────────────────────── */
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            {/* Known acronyms */}
            {(() => {
              const known = acrDecisions.filter(d => (d.is_known || d.user_full_form) && !d.not_an_acronym)
              if (known.length === 0) return null
              return (
                <div>
                  <SectionHeader title="Known Full Forms" count={known.length} color="hsl(160,60%,45%)" />
                  {known.map((dec, realIdx) => {
                    const idx = acrDecisions.indexOf(dec)
                    return (
                      <div key={realIdx} style={{
                        padding: '.55rem .75rem', borderRadius: '10px', marginBottom: '6px',
                        background: dec.enabled ? 'hsl(var(--card))' : 'hsl(var(--muted)/.4)',
                        border: `1px solid ${dec.enabled ? 'hsl(160,60%,45%/.3)' : 'hsl(var(--border)/.4)'}`,
                        display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '8px',
                        opacity: dec.enabled ? 1 : 0.6, transition: 'all .15s',
                      }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flex: 1, minWidth: 0 }}>
                          <span style={{
                            fontFamily: 'JetBrains Mono, monospace', fontSize: '.8rem', fontWeight: 700,
                            color: 'hsl(220,80%,60%)', flexShrink: 0,
                          }}>{dec.acronym}</span>
                          <span style={{ color: 'hsl(var(--pencil))', fontSize: '.7rem', flexShrink: 0 }}>→</span>
                          <span style={{
                            fontSize: '.75rem', color: 'hsl(var(--ink))', fontFamily: 'Inter, sans-serif',
                            overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                          }}>{dec.user_full_form || dec.full_form}</span>
                        </div>
                        <ToggleSwitch value={dec.enabled} onChange={() => toggleAcronym(idx)} />
                      </div>
                    )
                  })}
                </div>
              )
            })()}

            {/* Unknown acronyms */}
            {(() => {
              const unknown = acrDecisions.filter(d => !d.is_known && !d.user_full_form && !d.not_an_acronym)
              if (unknown.length === 0) return null
              return (
                <div>
                  <SectionHeader title="Unknown Acronyms" count={unknown.length} color="hsl(45,90%,50%)" />
                  {unknown.map((dec, realIdx) => {
                    const idx = acrDecisions.indexOf(dec)
                    const inputVal = acrInputs[dec.acronym] || ''
                    const isSaving = savingDict[dec.acronym]
                    return (
                      <div key={realIdx} style={{
                        padding: '.65rem .75rem', borderRadius: '10px', marginBottom: '6px',
                        background: 'hsl(var(--card))',
                        border: '1px solid hsl(45,90%,50%/.25)',
                        display: 'flex', flexDirection: 'column', gap: '8px',
                      }}>
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <span style={{
                              fontFamily: 'JetBrains Mono, monospace', fontSize: '.82rem', fontWeight: 700,
                              color: 'hsl(220,80%,60%)',
                            }}>{dec.acronym}</span>
                            <span style={{
                              fontSize: '.7rem', color: 'hsl(45,90%,45%)',
                              background: 'hsl(45,90%,50%/.1)', border: '1px solid hsl(45,90%,50%/.25)',
                              padding: '.1rem .4rem', borderRadius: '5px', fontWeight: 600,
                            }}>Full form not found</span>
                          </div>
                          <button
                            onClick={() => markNotAcronym(idx)}
                            style={{
                              fontSize: '.68rem', fontWeight: 600, padding: '.2rem .5rem', borderRadius: '6px',
                              background: 'hsl(var(--muted))', border: '1px solid hsl(var(--border))',
                              color: 'hsl(var(--pencil))', cursor: 'pointer',
                            }}
                            title="Mark as not an acronym"
                          >Not Acronym</button>
                        </div>
                        <div style={{ display: 'flex', gap: '6px' }}>
                          <input
                            placeholder="Enter full form…"
                            value={inputVal}
                            onChange={e => setAcrInputs(prev => ({ ...prev, [dec.acronym]: e.target.value }))}
                            onKeyDown={e => { if (e.key === 'Enter') saveToDict(dec.acronym, idx) }}
                            style={{
                              flex: 1, fontSize: '.78rem', padding: '.3rem .55rem',
                              borderRadius: '7px', border: '1px solid hsl(var(--border))',
                              background: 'hsl(var(--paper))', color: 'hsl(var(--ink))',
                              fontFamily: 'Inter, sans-serif',
                            }}
                          />
                          <button
                            onClick={() => saveToDict(dec.acronym, idx)}
                            disabled={!inputVal.trim() || isSaving}
                            style={{
                              display: 'flex', alignItems: 'center', gap: '4px',
                              fontSize: '.72rem', fontWeight: 600, padding: '.3rem .6rem', borderRadius: '7px',
                              background: 'hsl(220,80%,60%/.12)', border: '1px solid hsl(220,80%,60%/.3)',
                              color: 'hsl(220,80%,55%)', cursor: 'pointer', flexShrink: 0,
                            }}
                          >
                            {isSaving ? <Loader size={11} className="spin" /> : <Save size={11} />}
                            Save
                          </button>
                        </div>
                      </div>
                    )
                  })}
                </div>
              )
            })()}

            {/* Not-acronyms section */}
            {(() => {
              const dismissed = acrDecisions.filter(d => d.not_an_acronym)
              if (dismissed.length === 0) return null
              return (
                <div>
                  <SectionHeader title="Not Acronyms (Dismissed)" count={dismissed.length} color="hsl(var(--pencil))" />
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '5px' }}>
                    {dismissed.map((d, i) => (
                      <span key={i} style={{
                        fontFamily: 'JetBrains Mono, monospace', fontSize: '.72rem',
                        padding: '.15rem .5rem', borderRadius: '6px',
                        background: 'hsl(var(--muted))', color: 'hsl(var(--pencil))',
                        border: '1px solid hsl(var(--border))',
                        textDecoration: 'line-through', opacity: 0.6,
                      }}>{d.acronym}</span>
                    ))}
                  </div>
                </div>
              )
            })()}

            {acrDecisions.length === 0 && (
              <p style={{ fontSize: '.8rem', color: 'hsl(var(--pencil))', textAlign: 'center', marginTop: '1.5rem' }}>
                ✓ No acronyms detected in this transcript.
              </p>
            )}
          </div>
        )}
      </div>

      {/* Footer actions */}
      {!hasNoData && (
        <div style={{
          padding: '.75rem 1rem', borderTop: '1px solid hsl(var(--border)/.5)',
          display: 'flex', flexDirection: 'column', gap: '8px',
        }}>
          <div style={{ display: 'flex', gap: '8px' }}>
            <button
              onClick={handleApply}
              disabled={applying || reverting}
              style={{
                flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px',
                padding: '.5rem', borderRadius: '9px', border: 'none', cursor: 'pointer',
                fontWeight: 700, fontSize: '.8rem', fontFamily: 'Inter, sans-serif',
                background: 'hsl(220,80%,60%)',
                color: 'white',
                boxShadow: '0 2px 8px hsl(220,80%,60%/.3)',
                transition: 'opacity .15s',
                opacity: applying ? 0.7 : 1,
              }}
              id="btn-apply-corrections"
            >
              {applying ? <Loader size={13} className="spin" /> : <Check size={13} />}
              {applying ? 'Applying…' : `Apply to Transcript`}
            </button>
            {isApplied && (
              <button
                onClick={handleRevert}
                disabled={reverting || applying}
                style={{
                  display: 'flex', alignItems: 'center', gap: '4px',
                  padding: '.5rem .75rem', borderRadius: '9px',
                  background: 'hsl(var(--muted))', border: '1px solid hsl(var(--border))',
                  color: 'hsl(var(--pencil))', cursor: 'pointer', fontSize: '.78rem', fontWeight: 600,
                }}
                title="Revert to original transcript"
              >
                {reverting ? <Loader size={12} className="spin" /> : <RotateCcw size={12} />}
              </button>
            )}
          </div>
          {(enabledCorrCount > 0 || enabledAcrCount > 0) && (
            <p style={{ fontSize: '.68rem', color: 'hsl(var(--pencil))', margin: 0, textAlign: 'center' }}>
              {enabledCorrCount > 0 && `${enabledCorrCount} correction(s)`}
              {enabledCorrCount > 0 && enabledAcrCount > 0 && ' + '}
              {enabledAcrCount > 0 && `${enabledAcrCount} acronym expansion(s)`}
              {' '}will be applied
            </p>
          )}
        </div>
      )}
    </div>
  )
}
