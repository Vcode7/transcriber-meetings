/**
 * Stage 1 and Stage 2 option controls of the ROM pipeline.
 * Shared by the ROM page (left panel) and the ROM settings panel on the
 * Upload / Record / Tab Recording pages, so both always offer the same controls.
 */
import type { CSSProperties } from 'react'
import { Sparkles, History, Filter, Layers, Check } from 'lucide-react'

export type PreviousMeetingMode = 'auto' | 'select' | 'off'

export interface Stage2OptionsValue {
  meetingTopK: number
  globalTopK: number
  discussionWindowSize: number
  minSimilarityThreshold: number | null
  processAllTogether: boolean
  previousMeetingMode: PreviousMeetingMode
  previousMeetingId: string
  previousMeetingTopK: number
}

export interface PreviousMeetingOption {
  id: string
  name: string
  date?: string
  stage2_count?: number
  has_stage2?: boolean
}

export function Stage1Options({ transcriptWindow, onChange }: {
  transcriptWindow: number
  onChange: (minutes: number) => void
}) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <label style={{ fontSize: '.78rem', fontWeight: 600, color: 'hsl(var(--ink))', fontFamily: 'Inter, sans-serif' }}>
            Transcript Window
          </label>
          <div style={{ fontSize: '.7rem', color: 'hsl(var(--pencil))' }}>
            Time chunk for discussion extraction
          </div>
        </div>
        <span style={{
          fontSize: '.75rem', fontWeight: 700,
          color: 'hsl(280, 75%, 65%)',
          fontFamily: 'JetBrains Mono, monospace',
          padding: '2px 8px', borderRadius: '6px',
          background: 'hsl(280, 75%, 60% / .12)',
          border: '1px solid hsl(280, 75%, 60% / .25)',
        }}>
          {transcriptWindow}m
        </span>
      </div>

      <input
        type="range"
        min={0.5}
        max={15}
        step={0.5}
        value={transcriptWindow}
        onChange={e => onChange(Number(e.target.value))}
        className="modern-range-slider"
        style={{ color: 'hsl(280, 75%, 60%)' }}
      />

      {/* Quick Presets */}
      <div style={{ display: 'flex', gap: 6, marginTop: 2 }}>
        {[1, 2, 5, 10].map(mins => (
          <button
            key={mins}
            type="button"
            onClick={() => onChange(mins)}
            style={{
              flex: 1,
              padding: '3px 0',
              borderRadius: 6,
              fontSize: '.68rem',
              fontWeight: transcriptWindow === mins ? 700 : 500,
              fontFamily: 'JetBrains Mono, monospace',
              border: `1px solid ${transcriptWindow === mins ? 'hsl(280, 75%, 60% / .4)' : 'hsl(var(--border) / .3)'}`,
              background: transcriptWindow === mins ? 'hsl(280, 75%, 60% / .15)' : 'hsl(var(--muted) / .3)',
              color: transcriptWindow === mins ? 'hsl(280, 75%, 65%)' : 'hsl(var(--pencil))',
              cursor: 'pointer',
              transition: 'all .15s ease',
            }}
          >
            {mins}m
          </button>
        ))}
      </div>
    </div>
  )
}

const modePillStyle = (active: boolean, flex = 1, off = false): CSSProperties => ({
  flex,
  padding: '5px 8px',
  border: 'none',
  borderRadius: 6,
  fontSize: '.72rem',
  fontWeight: active ? 700 : 500,
  background: active ? (off ? 'hsl(var(--muted))' : 'hsl(270, 75%, 60%)') : 'transparent',
  color: active ? (off ? 'hsl(var(--ink))' : '#ffffff') : 'hsl(var(--pencil))',
  cursor: 'pointer',
  transition: 'all .18s cubic-bezier(0.4, 0, 0.2, 1)',
  boxShadow: active && !off ? '0 1px 4px rgba(120, 60, 220, 0.25)' : 'none',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  gap: 4,
})

export function Stage2Options({ value, onChange, previousMeetings, loadingPreviousMeetings = false }: {
  value: Stage2OptionsValue
  onChange: (patch: Partial<Stage2OptionsValue>) => void
  previousMeetings: PreviousMeetingOption[]
  loadingPreviousMeetings?: boolean
}) {
  const {
    meetingTopK, globalTopK, discussionWindowSize, minSimilarityThreshold, processAllTogether,
    previousMeetingMode, previousMeetingId, previousMeetingTopK,
  } = value

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
      {/* Meeting K and Global K */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <label style={{ fontSize: '.74rem', fontWeight: 600, color: 'hsl(var(--ink))' }}>Meeting K</label>
            <span style={{
              fontSize: '.72rem', fontWeight: 700, color: 'hsl(140, 70%, 45%)',
              fontFamily: 'JetBrains Mono, monospace', padding: '1px 6px',
              borderRadius: 5, background: 'hsl(140, 70%, 45% / .1)',
              border: '1px solid hsl(140, 70%, 45% / .25)',
            }}>
              {meetingTopK}
            </span>
          </div>
          <input
            type="range"
            min={0}
            max={20}
            value={meetingTopK}
            onChange={e => onChange({ meetingTopK: Number(e.target.value) })}
            className="modern-range-slider"
            style={{ color: 'hsl(140, 70%, 45%)' }}
          />
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <label style={{ fontSize: '.74rem', fontWeight: 600, color: 'hsl(var(--ink))' }}>Global K</label>
            <span style={{
              fontSize: '.72rem', fontWeight: 700, color: 'hsl(30, 90%, 50%)',
              fontFamily: 'JetBrains Mono, monospace', padding: '1px 6px',
              borderRadius: 5, background: 'hsl(30, 90%, 50% / .1)',
              border: '1px solid hsl(30, 90%, 50% / .25)',
            }}>
              {globalTopK}
            </span>
          </div>
          <input
            type="range"
            min={0}
            max={20}
            value={globalTopK}
            onChange={e => onChange({ globalTopK: Number(e.target.value) })}
            className="modern-range-slider"
            style={{ color: 'hsl(30, 90%, 50%)' }}
          />
        </div>
      </div>

      {/* Discussion Window Size slider */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <label style={{ fontSize: '.74rem', fontWeight: 600, color: 'hsl(var(--ink))' }}>Discussion Window</label>
            <span style={{ fontSize: '.68rem', color: 'hsl(var(--pencil))', marginLeft: 6 }}>points per context block</span>
          </div>
          <span style={{
            fontSize: '.72rem', fontWeight: 700, color: 'hsl(270, 80%, 65%)',
            fontFamily: 'JetBrains Mono, monospace', padding: '1px 6px',
            borderRadius: 5, background: 'hsl(270, 80%, 65% / .1)',
            border: '1px solid hsl(270, 80%, 65% / .25)',
          }}>
            {discussionWindowSize} pts
          </span>
        </div>
        <input
          type="range"
          min={3}
          max={20}
          step={1}
          value={discussionWindowSize}
          onChange={e => onChange({ discussionWindowSize: Number(e.target.value) })}
          className="modern-range-slider"
          style={{ color: 'hsl(270, 80%, 65%)' }}
        />
      </div>

      {/* Minimum Similarity Threshold Slider & Toggle */}
      <div style={{
        padding: '.6rem .75rem',
        borderRadius: 8,
        background: 'hsl(var(--muted) / .25)',
        border: '1px solid hsl(var(--border) / .35)',
        display: 'flex',
        flexDirection: 'column',
        gap: 6,
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <Filter size={13} style={{ color: 'hsl(205, 90%, 55%)' }} />
            <label style={{ fontSize: '.74rem', fontWeight: 600, color: 'hsl(var(--ink))' }}>
              Min Similarity Filter
            </label>
          </div>
          <label style={{
            display: 'flex',
            alignItems: 'center',
            gap: 6,
            cursor: 'pointer',
            userSelect: 'none',
          }}>
            <input
              type="checkbox"
              checked={minSimilarityThreshold !== null}
              onChange={e => onChange({ minSimilarityThreshold: e.target.checked ? 0.60 : null })}
              style={{
                width: 15,
                height: 15,
                borderRadius: 4,
                accentColor: 'hsl(205, 90%, 55%)',
                cursor: 'pointer',
              }}
            />
            <span style={{
              fontSize: '.72rem',
              fontWeight: 700,
              color: minSimilarityThreshold !== null ? 'hsl(205, 90%, 55%)' : 'hsl(var(--pencil))',
              fontFamily: 'JetBrains Mono, monospace',
            }}>
              {minSimilarityThreshold !== null ? minSimilarityThreshold.toFixed(2) : 'Off'}
            </span>
          </label>
        </div>

        {minSimilarityThreshold !== null && (
          <div style={{ marginTop: 2 }}>
            <input
              type="range"
              min={0.50}
              max={0.98}
              step={0.01}
              value={minSimilarityThreshold}
              onChange={e => onChange({ minSimilarityThreshold: Number(e.target.value) })}
              className="modern-range-slider"
              style={{ color: 'hsl(205, 90%, 55%)' }}
            />
          </div>
        )}
      </div>

      {/* Process All Points Together Toggle */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '.45rem .75rem',
        borderRadius: 8,
        background: 'hsl(var(--muted) / .25)',
        border: '1px solid hsl(var(--border) / .35)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <Layers size={13} style={{ color: 'hsl(205, 90%, 55%)' }} />
          <div>
            <div style={{ fontSize: '.74rem', fontWeight: 600, color: 'hsl(var(--ink))' }}>
              Process All Points Together
            </div>
            <div style={{ fontSize: '.68rem', color: 'hsl(var(--pencil))' }}>
              Single batch LLM analysis rather than per-cluster
            </div>
          </div>
        </div>
        <input
          type="checkbox"
          checked={processAllTogether}
          onChange={e => onChange({ processAllTogether: e.target.checked })}
          style={{
            width: 16,
            height: 16,
            borderRadius: 4,
            accentColor: 'hsl(205, 90%, 55%)',
            cursor: 'pointer',
          }}
        />
      </div>

      {/* Previous Meeting Context Card */}
      <div style={{
        padding: '.65rem .8rem',
        borderRadius: 9,
        background: 'linear-gradient(135deg, hsl(270, 75%, 60% / .06), hsl(280, 80%, 55% / .02))',
        border: '1px solid hsl(270, 75%, 60% / .22)',
        display: 'flex',
        flexDirection: 'column',
        gap: '.55rem',
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div style={{
            fontSize: '.72rem',
            fontWeight: 700,
            color: 'hsl(270, 75%, 65%)',
            textTransform: 'uppercase',
            letterSpacing: '.04em',
            display: 'flex',
            alignItems: 'center',
            gap: 6,
          }}>
            <History size={13} /> Previous Meeting Context
          </div>
          {previousMeetingMode !== 'off' && (
            <span style={{
              fontSize: '.7rem',
              fontWeight: 700,
              color: 'hsl(270, 75%, 60%)',
              fontFamily: 'JetBrains Mono, monospace',
              background: 'hsl(270, 75%, 60% / .12)',
              padding: '1px 6px',
              borderRadius: 4,
            }}>
              K={previousMeetingTopK}
            </span>
          )}
        </div>

        {/* Mode Pills */}
        <div style={{
          display: 'flex',
          gap: 3,
          background: 'hsl(var(--muted) / .5)',
          padding: 3,
          borderRadius: 8,
          border: '1px solid hsl(var(--border) / .3)',
        }}>
          <button
            type="button"
            onClick={() => onChange({ previousMeetingMode: 'auto' })}
            style={modePillStyle(previousMeetingMode === 'auto')}
            title="Auto-retrieve most similar Stage 2 points from all previous meetings via ChromaDB"
          >
            Auto Retrieve
          </button>
          <button
            type="button"
            onClick={() => onChange({ previousMeetingMode: 'select' })}
            style={modePillStyle(previousMeetingMode === 'select')}
            title="Select a specific past meeting to use its Stage 2 points only"
          >
            Select Meeting
          </button>
          <button
            type="button"
            onClick={() => onChange({ previousMeetingMode: 'off' })}
            style={modePillStyle(previousMeetingMode === 'off', 0.6, true)}
          >
            Off
          </button>
        </div>

        {/* Select Meeting Dropdown */}
        {previousMeetingMode === 'select' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            <label style={{ fontSize: '.7rem', color: 'hsl(var(--pencil))', fontWeight: 600 }}>
              Target Past Meeting:
            </label>
            {loadingPreviousMeetings ? (
              <div style={{ fontSize: '.72rem', color: 'hsl(var(--pencil))', padding: '4px 0' }}>
                Loading previous meetings…
              </div>
            ) : previousMeetings.length === 0 ? (
              <div style={{
                fontSize: '.72rem',
                color: 'hsl(var(--destructive))',
                padding: '4px 8px',
                borderRadius: 5,
                background: 'hsl(var(--destructive) / .1)',
              }}>
                No prior meetings found with Stage 2 vectors.
              </div>
            ) : (
              <select
                value={previousMeetingId}
                onChange={e => onChange({ previousMeetingId: e.target.value })}
                style={{
                  width: '100%',
                  fontSize: '.76rem',
                  padding: '6px 8px',
                  borderRadius: 6,
                  border: '1px solid hsl(270, 75%, 55% / .3)',
                  background: 'hsl(var(--card))',
                  color: 'hsl(var(--ink))',
                  outline: 'none',
                  cursor: 'pointer',
                  fontFamily: 'Inter, sans-serif',
                }}
              >
                {previousMeetings.map(m => (
                  <option key={m.id} value={m.id}>
                    {m.name} {m.date ? `(${m.date})` : ''} {m.has_stage2 ? `[${m.stage2_count} pts]` : '[No pts]'}
                  </option>
                ))}
              </select>
            )}
          </div>
        )}

        {/* Previous Meeting Top-K slider */}
        {previousMeetingMode !== 'off' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4, marginTop: 2 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <label style={{ fontSize: '.72rem', color: 'hsl(var(--pencil))', fontWeight: 500 }}>
                Points to retrieve (Top-K)
              </label>
              <span style={{
                fontSize: '.72rem',
                fontWeight: 700,
                color: 'hsl(270, 75%, 65%)',
                fontFamily: 'JetBrains Mono, monospace',
              }}>
                {previousMeetingTopK === 0 ? '0 (Off)' : previousMeetingTopK}
              </span>
            </div>
            <input
              type="range"
              min={0}
              max={10}
              value={previousMeetingTopK}
              onChange={e => onChange({ previousMeetingTopK: Number(e.target.value) })}
              className="modern-range-slider"
              style={{ color: 'hsl(270, 75%, 60%)' }}
            />
          </div>
        )}
      </div>
    </div>
  )
}
