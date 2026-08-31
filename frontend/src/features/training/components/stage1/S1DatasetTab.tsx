import { useState } from 'react';
import { Calendar, Clock, Play, CheckSquare, Square, Layers, FileText, RefreshCw, ChevronRight, ChevronDown, Users } from 'lucide-react';
import { useStage1Store } from '../../store/stage1Store';
import { prepareTranscriptWindows } from '../../api/stage1Api';

const S = {
  card: {
    background: 'hsl(var(--card))',
    borderRadius: '12px',
    border: '1px solid hsl(var(--border))',
    padding: '1.25rem',
  } as React.CSSProperties,
  label: {
    display: 'block',
    fontSize: '0.8rem',
    fontWeight: 600,
    color: 'hsl(var(--muted-foreground))',
    marginBottom: '0.4rem',
    textTransform: 'uppercase' as const,
    letterSpacing: '0.04em',
  },
  input: {
    width: '100%',
    padding: '0.6rem 0.8rem',
    borderRadius: '8px',
    border: '1px solid hsl(var(--border))',
    background: 'hsl(var(--background))',
    color: 'hsl(var(--foreground))',
    fontSize: '0.9rem',
    outline: 'none',
    boxSizing: 'border-box' as const,
  },
  btn: (variant: 'primary' | 'ghost' = 'primary') => ({
    display: 'inline-flex',
    alignItems: 'center',
    gap: '6px',
    padding: '0.55rem 1.1rem',
    borderRadius: '8px',
    border: 'none',
    cursor: 'pointer',
    fontSize: '0.85rem',
    fontWeight: 600,
    transition: 'all 0.15s ease',
    background: variant === 'primary' ? 'hsl(var(--accent))' : 'hsl(var(--muted))',
    color: variant === 'primary' ? 'white' : 'hsl(var(--foreground))',
  } as React.CSSProperties),
};

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

export default function S1DatasetTab() {
  const {
    meetings,
    datasetMeetingId,
    setDatasetMeetingId,
    datasetMeetingFilename,
    setDatasetMeetingFilename,
    windowSizeMinutes,
    setWindowSizeMinutes,
    windowOverlapSeconds,
    setWindowOverlapSeconds,
    generatedWindows,
    setGeneratedWindows,
    windowsLoading,
    setWindowsLoading,
    toggleWindowSelected,
    setWindowRole,
    selectAllWindows,
  } = useStage1Store();

  const [expandedWindowId, setExpandedWindowId] = useState<string | null>(null);

  const selectedCount = generatedWindows.filter((w) => w.selected).length;
  const trainCount = generatedWindows.filter((w) => w.selected && w.role !== 'val').length;
  const valCount = generatedWindows.filter((w) => w.selected && w.role === 'val').length;

  const handleSelectMeeting = (id: string, filename: string) => {
    setDatasetMeetingId(id);
    setDatasetMeetingFilename(filename);
    setGeneratedWindows([]);
  };

  const handlePrepareWindows = async () => {
    if (!datasetMeetingId) return;
    setWindowsLoading(true);
    try {
      const res = await prepareTranscriptWindows(
        datasetMeetingId,
        windowSizeMinutes * 60,
        windowOverlapSeconds,
      );
      setGeneratedWindows(res.windows);
    } catch (err) {
      console.error('Failed to prepare windows:', err);
    } finally {
      setWindowsLoading(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', width: '100%' }}>
      {/* Section: Meeting Selection */}
      <div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '0.75rem' }}>
          <div style={{
            width: 28, height: 28, borderRadius: '8px',
            background: 'hsl(var(--accent) / .15)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <FileText size={14} color="hsl(var(--accent))" />
          </div>
          <h3 style={{ margin: 0, fontSize: '0.95rem', fontWeight: 700 }}>Training Meeting</h3>
        </div>

        {meetings.length === 0 ? (
          <div style={{ ...S.card, textAlign: 'center', color: 'hsl(var(--muted-foreground))', padding: '2rem' }}>
            No processed meetings found. Process a recording first.
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', maxHeight: '260px', overflowY: 'auto', paddingRight: '4px' }}>
            {meetings.map((m) => {
              const isSelected = datasetMeetingId === m.id;
              return (
                <div
                  key={m.id}
                  onClick={() => handleSelectMeeting(m.id, m.filename)}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    padding: '0.75rem 1rem',
                    borderRadius: '10px',
                    border: `1.5px solid ${isSelected ? 'hsl(var(--accent))' : 'hsl(var(--border) / 0.6)'}`,
                    background: isSelected ? 'hsl(var(--accent) / 0.06)' : 'hsl(var(--card))',
                    cursor: 'pointer',
                    transition: 'all 0.12s ease',
                  }}
                >
                  <div>
                    <div style={{ fontWeight: 600, fontSize: '0.875rem', color: 'hsl(var(--foreground))' }}>
                      {m.filename}
                    </div>
                    <div style={{ display: 'flex', gap: '1rem', fontSize: '0.75rem', color: 'hsl(var(--muted-foreground))', marginTop: '3px' }}>
                      <span style={{ display: 'flex', alignItems: 'center', gap: '3px' }}>
                        <Calendar size={11} /> {new Date(m.created_at).toLocaleDateString()}
                      </span>
                      <span style={{ display: 'flex', alignItems: 'center', gap: '3px' }}>
                        <Clock size={11} /> {Math.round(m.duration / 60)} min
                      </span>
                    </div>
                  </div>
                  <div style={{ display: 'flex', gap: '4px' }}>
                    {m.stages_available.includes('stage_1') && (
                      <span style={{
                        fontSize: '0.68rem', fontWeight: 700, padding: '2px 7px', borderRadius: '8px',
                        background: 'hsl(var(--accent) / 0.12)', color: 'hsl(var(--accent))',
                      }}>
                        Stage 1
                      </span>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Section: Window Configuration */}
      {datasetMeetingId && (
        <div style={{ ...S.card, display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <h3 style={{ margin: 0, fontSize: '0.9rem', fontWeight: 700, color: 'hsl(var(--foreground))' }}>
            Window Configuration
          </h3>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
            <div>
              <label style={S.label}>Transcript Window Size</label>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <input
                  type="range" min={0.5} max={10} step={0.5}
                  value={windowSizeMinutes}
                  onChange={(e) => setWindowSizeMinutes(parseFloat(e.target.value))}
                  style={{ flex: 1 }}
                />
                <span style={{
                  minWidth: '60px', textAlign: 'right', fontSize: '0.9rem', fontWeight: 700,
                  color: 'hsl(var(--accent))',
                }}>
                  {windowSizeMinutes} min
                </span>
              </div>
              <div style={{ fontSize: '0.72rem', color: 'hsl(var(--muted-foreground))', marginTop: '4px' }}>
                Each training window = {windowSizeMinutes * 60}s of transcript
              </div>
            </div>
            <div>
              <label style={S.label}>Window Overlap</label>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <input
                  type="range" min={0} max={Math.min(windowSizeMinutes * 60 * 0.8, 120)} step={5}
                  value={windowOverlapSeconds}
                  onChange={(e) => setWindowOverlapSeconds(parseFloat(e.target.value))}
                  style={{ flex: 1 }}
                />
                <span style={{
                  minWidth: '60px', textAlign: 'right', fontSize: '0.9rem', fontWeight: 700,
                  color: 'hsl(var(--accent))',
                }}>
                  {windowOverlapSeconds}s
                </span>
              </div>
              <div style={{ fontSize: '0.72rem', color: 'hsl(var(--muted-foreground))', marginTop: '4px' }}>
                Overlap between consecutive windows
              </div>
            </div>
          </div>

          <button
            onClick={handlePrepareWindows}
            disabled={windowsLoading}
            style={{
              ...S.btn('primary'),
              alignSelf: 'flex-start',
              opacity: windowsLoading ? 0.7 : 1,
            }}
          >
            {windowsLoading ? <RefreshCw size={14} style={{ animation: 'spin 1s linear infinite' }} /> : <Play size={14} />}
            {windowsLoading ? 'Preparing...' : 'Prepare Training Windows'}
          </button>
        </div>
      )}

      {/* Section: Generated Windows */}
      {generatedWindows.length > 0 && (
        <div>
          {/* Summary bar */}
          <div style={{
            display: 'flex', alignItems: 'center', justifyContent: 'space-between',
            marginBottom: '0.75rem',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <div style={{
                width: 28, height: 28, borderRadius: '8px',
                background: 'hsl(var(--accent) / .15)',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
              }}>
                <Layers size={14} color="hsl(var(--accent))" />
              </div>
              <h3 style={{ margin: 0, fontSize: '0.95rem', fontWeight: 700 }}>
                Generated Windows
              </h3>
              <span style={{
                fontSize: '0.75rem', fontWeight: 600, padding: '2px 8px', borderRadius: '6px',
                background: 'hsl(var(--accent) / 0.1)', color: 'hsl(var(--accent))',
              }}>
                {generatedWindows.length} total
              </span>
            </div>
            <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
              <button onClick={() => selectAllWindows(true)} style={S.btn('ghost')}>
                Select All
              </button>
              <button onClick={() => selectAllWindows(false)} style={S.btn('ghost')}>
                Deselect All
              </button>
            </div>
          </div>

          {/* Stats row */}
          <div style={{
            display: 'flex', gap: '1rem', marginBottom: '0.75rem',
            padding: '0.7rem 1rem', borderRadius: '10px',
            background: 'hsl(var(--card))', border: '1px solid hsl(var(--border))',
          }}>
            {[
              { label: 'Total Windows', value: generatedWindows.length, color: 'hsl(var(--foreground))' },
              { label: 'Selected', value: selectedCount, color: 'hsl(var(--accent))' },
              { label: 'Training', value: trainCount, color: 'hsl(142 71% 45%)' },
              { label: 'Validation', value: valCount, color: 'hsl(45 93% 47%)' },
            ].map((item) => (
              <div key={item.label} style={{ textAlign: 'center', flex: 1 }}>
                <div style={{ fontSize: '1.2rem', fontWeight: 800, color: item.color }}>{item.value}</div>
                <div style={{ fontSize: '0.7rem', color: 'hsl(var(--muted-foreground))', fontWeight: 500 }}>{item.label}</div>
              </div>
            ))}
          </div>

          {/* Window list */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem', maxHeight: '420px', overflowY: 'auto', paddingRight: '4px' }}>
            {generatedWindows.map((win) => {
              const isExpanded = expandedWindowId === win.window_id;
              return (
                <div
                  key={win.window_id}
                  style={{
                    borderRadius: '10px',
                    border: `1.5px solid ${win.selected ? 'hsl(var(--accent) / 0.4)' : 'hsl(var(--border) / 0.5)'}`,
                    background: win.selected ? 'hsl(var(--accent) / 0.03)' : 'hsl(var(--card))',
                    overflow: 'hidden',
                    transition: 'all 0.12s ease',
                  }}
                >
                  <div style={{
                    display: 'flex', alignItems: 'center', gap: '10px',
                    padding: '0.65rem 0.9rem', cursor: 'pointer',
                  }}>
                    {/* Checkbox */}
                    <button
                      onClick={() => toggleWindowSelected(win.window_id)}
                      style={{ background: 'none', border: 'none', cursor: 'pointer', padding: 0, flexShrink: 0, display: 'flex', alignItems: 'center' }}
                    >
                      {win.selected
                        ? <CheckSquare size={18} color="hsl(var(--accent))" />
                        : <Square size={18} color="hsl(var(--muted-foreground))" />}
                    </button>

                    {/* Window index badge */}
                    <div style={{
                      minWidth: '32px', height: '22px', borderRadius: '6px',
                      background: 'hsl(var(--muted))', display: 'flex', alignItems: 'center',
                      justifyContent: 'center', fontSize: '0.72rem', fontWeight: 700,
                      color: 'hsl(var(--muted-foreground))',
                    }}>
                      W{win.window_index + 1}
                    </div>

                    {/* Time range */}
                    <div style={{ flex: 1, display: 'flex', gap: '12px', fontSize: '0.82rem', alignItems: 'center' }}>
                      <span style={{ fontWeight: 600, color: 'hsl(var(--foreground))' }}>
                        {formatTime(win.start_time)} – {formatTime(win.end_time)}
                      </span>
                      <span style={{ display: 'flex', alignItems: 'center', gap: '3px', color: 'hsl(var(--muted-foreground))', fontSize: '0.75rem' }}>
                        <Users size={11} /> {win.speakers?.length || 0} speakers
                      </span>
                      <span style={{ color: 'hsl(var(--muted-foreground))', fontSize: '0.75rem' }}>
                        {win.segment_count} segments
                      </span>
                    </div>

                    {/* Role selector */}
                    {win.selected && (
                      <select
                        value={win.role || ''}
                        onChange={(e) => setWindowRole(win.window_id, (e.target.value as 'train' | 'val' | '') || undefined)}
                        onClick={(e) => e.stopPropagation()}
                        style={{
                          fontSize: '0.72rem', fontWeight: 600, padding: '2px 6px', borderRadius: '6px',
                          border: '1px solid hsl(var(--border))', background: 'hsl(var(--background))',
                          color: win.role === 'val' ? 'hsl(45 93% 40%)' : 'hsl(142 71% 40%)',
                          cursor: 'pointer',
                        }}
                      >
                        <option value="">Auto (train)</option>
                        <option value="train">Training</option>
                        <option value="val">Validation</option>
                      </select>
                    )}

                    {/* Expand toggle */}
                    <button
                      onClick={() => setExpandedWindowId(isExpanded ? null : win.window_id)}
                      style={{ background: 'none', border: 'none', cursor: 'pointer', padding: 0, display: 'flex', alignItems: 'center', color: 'hsl(var(--muted-foreground))' }}
                    >
                      {isExpanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
                    </button>
                  </div>

                  {/* Expanded transcript preview */}
                  {isExpanded && (
                    <div style={{
                      padding: '0.75rem 0.9rem 0.75rem 2.9rem',
                      borderTop: '1px solid hsl(var(--border) / 0.4)',
                      background: 'hsl(var(--background))',
                    }}>
                      <div style={{
                        fontSize: '0.78rem', color: 'hsl(var(--muted-foreground))',
                        fontFamily: 'monospace', whiteSpace: 'pre-wrap',
                        maxHeight: '160px', overflowY: 'auto', lineHeight: '1.5',
                      }}>
                        {win.transcript_text || 'No transcript text available.'}
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          {selectedCount > 0 && (
            <div style={{
              marginTop: '0.75rem',
              padding: '0.65rem 1rem',
              borderRadius: '8px',
              background: 'hsl(var(--accent) / 0.07)',
              border: '1px solid hsl(var(--accent) / 0.2)',
              fontSize: '0.82rem',
              color: 'hsl(var(--foreground))',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
            }}>
              <CheckSquare size={14} color="hsl(var(--accent))" />
              <strong>{selectedCount} windows selected</strong> — {trainCount} for training, {valCount} for validation.
              Go to <strong>Training Loop</strong> to start optimization.
            </div>
          )}
        </div>
      )}

      <style>{`@keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}
