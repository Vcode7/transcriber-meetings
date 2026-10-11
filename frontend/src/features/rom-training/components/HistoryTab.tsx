import { useState, useEffect } from 'react';
import { History, Trash2, Loader, ChevronDown, ChevronUp, Check, AlertCircle, Brain, RotateCcw, Maximize2 } from 'lucide-react';
import { listVariants, deleteVariant, rerunVariantTest } from '../api/romTrainingApi';
import type { RomVariant, TestResult } from '../types/romTrainingTypes';
import TestComparisonModal from './TestComparisonModal';

export default function HistoryTab() {
  const [variants, setVariants] = useState<RomVariant[]>([]);
  const [loading, setLoading] = useState(true);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [deleting, setDeleting] = useState<string | null>(null);
  const [retestingId, setRetestingId] = useState<string | null>(null);
  const [retestMsg, setRetestMsg] = useState<{ id: string; msg: string; isError?: boolean } | null>(null);
  const [disableThinking, setDisableThinking] = useState(true);
  const [modalState, setModalState] = useState<{
    isOpen: boolean;
    results: TestResult[];
    initialIndex: number;
    variantName: string;
    accuracy: string | null;
  }>({
    isOpen: false,
    results: [],
    initialIndex: 0,
    variantName: '',
    accuracy: null,
  });

  const load = () => {
    setLoading(true);
    listVariants()
      .then(r => setVariants(r.variants || []))
      .catch(() => {})
      .finally(() => setLoading(false));
  };

  useEffect(() => { load(); }, []);

  const handleRerunTest = async (id: string) => {
    setRetestingId(id);
    setRetestMsg(null);
    try {
      const res = await rerunVariantTest(id, undefined, disableThinking);
      setVariants(prev => prev.map(v => v.id === id ? {
        ...v,
        test_results: res.test_results,
        accuracy: res.accuracy,
        test_size: res.test_size,
        status: 'done',
        error: null,
        has_checkpoint: true,
      } : v));
      setExpandedId(id);
      setRetestMsg({ id, msg: `Test complete! ${res.accuracy || ''}` });
    } catch (e: any) {
      const detail = e?.response?.data?.detail || 'Failed to rerun test.';
      setRetestMsg({ id, msg: detail, isError: true });
    } finally {
      setRetestingId(null);
    }
  };

  const handleDelete = async (id: string) => {
    if (!confirm('Delete this variant and its trained adapter files?')) return;
    setDeleting(id);
    try {
      await deleteVariant(id);
      setVariants(prev => prev.filter(v => v.id !== id));
    } catch {
    } finally {
      setDeleting(null);
    }
  };

  const statusColor = (s: string) => {
    if (s === 'done') return 'hsl(142 70% 35%)';
    if (s === 'error') return 'hsl(var(--destructive))';
    if (s === 'training') return 'hsl(var(--accent))';
    return 'hsl(var(--muted-foreground))';
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 8 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <History size={18} style={{ color: 'hsl(var(--accent))' }} />
          <span style={{ fontWeight: 700, fontSize: '1rem' }}>Training History</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <label
            style={{
              display: 'flex', alignItems: 'center', gap: 6,
              fontSize: '0.74rem', cursor: 'pointer',
              userSelect: 'none', color: 'hsl(var(--foreground))',
              background: 'hsl(var(--muted) / .35)',
              padding: '0.25rem 0.65rem', borderRadius: 6,
              border: '1px solid hsl(var(--border))',
              fontWeight: 500,
            }}
            title="When checked, suppresses <think> reasoning tokens during test evaluation for clean, direct MoM output"
          >
            <input
              type="checkbox"
              checked={disableThinking}
              onChange={e => setDisableThinking(e.target.checked)}
              style={{ cursor: 'pointer', accentColor: 'hsl(var(--accent))' }}
            />
            <span>Disable Thinking</span>
          </label>
          <span style={{ fontSize: '0.78rem', color: 'hsl(var(--muted-foreground))' }}>
            {variants.length} variant{variants.length !== 1 ? 's' : ''}
          </span>
        </div>
      </div>

      {loading && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: 'hsl(var(--muted-foreground))', padding: '2rem' }}>
          <Loader size={16} className="spin" /> Loading…
        </div>
      )}

      {!loading && variants.length === 0 && (
        <div style={{
          textAlign: 'center', padding: '3rem',
          background: 'hsl(var(--card))', borderRadius: 12,
          border: '1px solid hsl(var(--border))',
          color: 'hsl(var(--muted-foreground))', fontSize: '0.85rem',
        }}>
          <Brain size={32} style={{ marginBottom: 12, opacity: 0.4 }} />
          <p>No training history yet.</p>
          <p style={{ fontSize: '0.75rem' }}>Use the Training tab to train your first model.</p>
        </div>
      )}

      {!loading && variants.map(v => (
        <div key={v.id} style={{
          background: 'hsl(var(--card))', borderRadius: 12,
          border: '1px solid hsl(var(--border))', overflow: 'hidden',
        }}>
          <div
            style={{
              padding: '0.85rem 1rem',
              display: 'flex', alignItems: 'center', justifyContent: 'space-between',
              cursor: 'pointer',
            }}
            onClick={() => setExpandedId(expandedId === v.id ? null : v.id)}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              {expandedId === v.id ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
              <div>
                <div style={{ fontWeight: 700, fontSize: '0.9rem' }}>{v.name}</div>
                <div style={{ fontSize: '0.72rem', color: 'hsl(var(--muted-foreground))', marginTop: 2 }}>
                  {v.training_type === 'upgrade' ? '↑ Upgrade' : '✦ New'} · {v.base_model} · {v.created_at?.slice(0, 10)}
                </div>
              </div>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{
                fontSize: '0.72rem', fontWeight: 700, padding: '2px 8px', borderRadius: 6,
                background: `${statusColor(v.status)}20`,
                color: statusColor(v.status),
              }}>
                {v.status === 'training' && <Loader size={10} className="spin" style={{ verticalAlign: 'middle', marginRight: 3 }} />}
                {v.status.toUpperCase()}
              </span>
              {v.has_checkpoint ? (
                <button
                  style={{
                    display: 'flex', alignItems: 'center', gap: 4,
                    padding: '0.3rem 0.65rem', borderRadius: 6, border: 'none',
                    background: 'hsl(var(--accent) / .12)',
                    color: 'hsl(var(--accent))', cursor: 'pointer',
                    fontSize: '0.72rem', fontWeight: 600,
                    opacity: retestingId === v.id ? 0.7 : 1,
                  }}
                  onClick={e => { e.stopPropagation(); handleRerunTest(v.id); }}
                  disabled={retestingId === v.id}
                  title="Rerun test evaluation using the saved checkpoint adapter"
                >
                  {retestingId === v.id ? <Loader size={11} className="spin" /> : <RotateCcw size={11} />}
                  {retestingId === v.id ? 'Testing…' : 'Rerun Test'}
                </button>
              ) : v.status === 'done' ? (
                <span
                  style={{ fontSize: '0.68rem', color: 'hsl(var(--muted-foreground))', fontStyle: 'italic', padding: '0.2rem 0.4rem' }}
                  title="Checkpoint adapter files not found on disk"
                >
                  (No checkpoint on disk)
                </span>
              ) : null}
              <button
                style={{
                  display: 'flex', alignItems: 'center', gap: 4,
                  padding: '0.3rem 0.6rem', borderRadius: 6, border: 'none',
                  background: 'hsl(var(--destructive) / .1)',
                  color: 'hsl(var(--destructive))', cursor: 'pointer',
                  fontSize: '0.72rem', fontWeight: 600,
                }}
                onClick={e => { e.stopPropagation(); handleDelete(v.id); }}
                disabled={deleting === v.id || retestingId === v.id}
              >
                {deleting === v.id ? <Loader size={11} className="spin" /> : <Trash2 size={11} />}
                Delete
              </button>
            </div>
          </div>

          {expandedId === v.id && (
            <div style={{ padding: '0 1rem 1rem', borderTop: '1px solid hsl(var(--border))' }}>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '0.75rem', marginTop: '0.75rem' }}>
                <div>
                  <div style={{ fontSize: '0.68rem', fontWeight: 700, color: 'hsl(var(--muted-foreground))', marginBottom: 4 }}>BASE MODEL</div>
                  <div style={{ fontSize: '0.78rem' }}>{v.base_model}</div>
                </div>
                <div>
                  <div style={{ fontSize: '0.68rem', fontWeight: 700, color: 'hsl(var(--muted-foreground))', marginBottom: 4 }}>DATASET</div>
                  <div style={{ fontSize: '0.78rem' }}>{v.dataset_size} train · {v.test_size} test</div>
                </div>
                <div>
                  <div style={{ fontSize: '0.68rem', fontWeight: 700, color: 'hsl(var(--muted-foreground))', marginBottom: 4 }}>COMPLETED</div>
                  <div style={{ fontSize: '0.78rem' }}>{v.completed_at?.slice(0, 16) || 'N/A'}</div>
                </div>
              </div>

              {retestMsg && retestMsg.id === v.id && (
                <div style={{
                  marginTop: '0.75rem', padding: '0.6rem 0.85rem', borderRadius: 8,
                  background: retestMsg.isError ? 'hsl(var(--destructive) / .08)' : 'hsl(142 70% 45% / .08)',
                  border: `1px solid ${retestMsg.isError ? 'hsl(var(--destructive) / .2)' : 'hsl(142 70% 45% / .2)'}`,
                  fontSize: '0.78rem', color: retestMsg.isError ? 'hsl(var(--destructive))' : 'hsl(142 70% 35%)', fontWeight: 600,
                }}>
                  {retestMsg.isError ? (
                    <AlertCircle size={12} style={{ verticalAlign: 'middle', marginRight: 4 }} />
                  ) : (
                    <Check size={12} style={{ verticalAlign: 'middle', marginRight: 4 }} />
                  )}
                  {retestMsg.msg}
                </div>
              )}

              {v.accuracy && (
                <div style={{
                  marginTop: '0.75rem', padding: '0.6rem 0.85rem', borderRadius: 8,
                  background: 'hsl(142 70% 45% / .08)',
                  border: '1px solid hsl(142 70% 45% / .2)',
                  fontSize: '0.78rem', color: 'hsl(142 70% 35%)', fontWeight: 600,
                }}>
                  <Check size={12} style={{ verticalAlign: 'middle', marginRight: 4 }} />
                  {v.accuracy}
                </div>
              )}

              {v.error && (
                <div style={{
                  marginTop: '0.75rem', padding: '0.6rem 0.85rem', borderRadius: 8,
                  background: 'hsl(var(--destructive) / .08)',
                  border: '1px solid hsl(var(--destructive) / .2)',
                  fontSize: '0.78rem', color: 'hsl(var(--destructive))',
                }}>
                  <AlertCircle size={12} style={{ verticalAlign: 'middle', marginRight: 4 }} />
                  {v.error}
                </div>
              )}

              {/* Config summary */}
              {v.config && Object.keys(v.config).length > 0 && (
                <div style={{ marginTop: '0.75rem' }}>
                  <div style={{ fontSize: '0.7rem', fontWeight: 700, color: 'hsl(var(--muted-foreground))', marginBottom: 6 }}>TRAINING CONFIG</div>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                    {Object.entries(v.config).filter(([k]) => !['bnb_4bit_compute_dtype'].includes(k)).map(([k, val]) => (
                      <span key={k} style={{
                        fontSize: '0.68rem', padding: '2px 7px', borderRadius: 5,
                        background: 'hsl(var(--muted))', color: 'hsl(var(--foreground))',
                        fontFamily: 'monospace',
                      }}>
                        {k}: {String(val)}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* If no test results recorded */}
              {v.test_results.length === 0 && (
                <div style={{
                  marginTop: '0.75rem', padding: '0.75rem', borderRadius: 8,
                  background: 'hsl(var(--muted) / .3)', border: '1px solid hsl(var(--border))',
                  display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                }}>
                  <span style={{ fontSize: '0.75rem', color: 'hsl(var(--muted-foreground))' }}>
                    {v.has_checkpoint
                      ? 'Checkpoint adapter is saved, but no test results are recorded yet.'
                      : v.status === 'done'
                        ? 'Checkpoint adapter not found on disk at configured path.'
                        : 'Training failed before a checkpoint could be saved.'}
                  </span>
                  {v.has_checkpoint && (
                    <button
                      style={{
                        padding: '0.35rem 0.75rem', borderRadius: 6, border: 'none',
                        background: 'hsl(var(--accent))', color: '#fff',
                        fontSize: '0.72rem', fontWeight: 600, cursor: 'pointer',
                        display: 'flex', alignItems: 'center', gap: 4,
                      }}
                      onClick={() => handleRerunTest(v.id)}
                      disabled={retestingId === v.id}
                    >
                      {retestingId === v.id ? <Loader size={11} className="spin" /> : <RotateCcw size={11} />}
                      Run Test Inference
                    </button>
                  )}
                </div>
              )}

              {/* Test results preview */}
              {v.test_results.length > 0 && (
                <div style={{ marginTop: '0.75rem' }}>
                  <div style={{
                    fontSize: '0.7rem', fontWeight: 700, color: 'hsl(var(--muted-foreground))',
                    marginBottom: 8, display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                    flexWrap: 'wrap', gap: 6,
                  }}>
                    <span>TEST RESULTS ({v.test_results.length} sample{v.test_results.length !== 1 ? 's' : ''})</span>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <button
                        type="button"
                        style={{
                          display: 'flex', alignItems: 'center', gap: 5,
                          padding: '0.28rem 0.65rem', borderRadius: 6,
                          border: '1px solid hsl(var(--accent) / .4)',
                          background: 'hsl(var(--accent) / .12)',
                          color: 'hsl(var(--accent))', cursor: 'pointer',
                          fontSize: '0.7rem', fontWeight: 700,
                          transition: 'all 0.15s ease',
                        }}
                        onClick={() => setModalState({
                          isOpen: true,
                          results: v.test_results,
                          initialIndex: 0,
                          variantName: v.name,
                          accuracy: v.accuracy,
                        })}
                        title="Open full-screen side-by-side comparison modal with complete untruncated outputs"
                      >
                        <Maximize2 size={11} /> View Full Comparison
                      </button>
                      {v.has_checkpoint && (
                        <button
                          style={{
                            border: 'none', background: 'transparent',
                            color: 'hsl(var(--accent))', cursor: 'pointer',
                            fontSize: '0.7rem', fontWeight: 600,
                            display: 'flex', alignItems: 'center', gap: 4,
                          }}
                          onClick={() => handleRerunTest(v.id)}
                          disabled={retestingId === v.id}
                        >
                          <RotateCcw size={10} /> Rerun evaluation
                        </button>
                      )}
                    </div>
                  </div>
                  {v.test_results.map((r, i) => (
                    <div key={i} style={{
                      marginBottom: 8, padding: '0.65rem 0.85rem', borderRadius: 8,
                      border: '1px solid hsl(var(--border))', fontSize: '0.72rem',
                      background: 'hsl(var(--card))',
                    }}>
                      <div style={{
                        fontWeight: 700, marginBottom: 6,
                        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                        gap: 8,
                      }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 6, overflow: 'hidden' }}>
                          <span style={{
                            padding: '1px 6px', borderRadius: 4,
                            background: 'hsl(var(--muted))',
                            fontSize: '0.65rem', color: 'hsl(var(--muted-foreground))',
                            fontFamily: 'monospace',
                          }}>
                            #{i + 1}
                          </span>
                          <span style={{ textOverflow: 'ellipsis', whiteSpace: 'nowrap', overflow: 'hidden' }}>
                            {r.agenda_title || `Sample ${i + 1}`}
                          </span>
                          {r.recording_title && (
                            <span style={{ fontSize: '0.68rem', fontWeight: 400, color: 'hsl(var(--muted-foreground))' }}>
                              ({r.recording_title})
                            </span>
                          )}
                        </div>
                        <button
                          type="button"
                          style={{
                            border: '1px solid hsl(var(--border))',
                            background: 'hsl(var(--muted) / .5)',
                            color: 'hsl(var(--foreground))',
                            borderRadius: 5,
                            padding: '2px 8px',
                            cursor: 'pointer',
                            fontSize: '0.68rem',
                            fontWeight: 600,
                            display: 'flex',
                            alignItems: 'center',
                            gap: 4,
                            flexShrink: 0,
                          }}
                          onClick={() => setModalState({
                            isOpen: true,
                            results: v.test_results,
                            initialIndex: i,
                            variantName: v.name,
                            accuracy: v.accuracy,
                          })}
                          title="Open this sample in the full comparison modal"
                        >
                          <Maximize2 size={10} /> Full View
                        </button>
                      </div>
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8 }}>
                        <div style={{
                          background: 'hsl(var(--muted) / .25)',
                          padding: '0.45rem',
                          borderRadius: 6,
                          border: '1px solid hsl(var(--border) / .6)',
                        }}>
                          <div style={{ fontSize: '0.65rem', fontWeight: 700, color: 'hsl(0 0% 50%)' }}>BEFORE (Base)</div>
                          <div style={{ marginTop: 3, whiteSpace: 'pre-wrap', lineHeight: 1.4, color: 'hsl(var(--muted-foreground))' }}>
                            {r.before ? (r.before.slice(0, 260) + (r.before.length > 260 ? '…' : '')) : 'N/A'}
                          </div>
                        </div>
                        <div style={{
                          background: 'hsl(142 70% 45% / .04)',
                          padding: '0.45rem',
                          borderRadius: 6,
                          border: '1px solid hsl(142 70% 45% / .2)',
                        }}>
                          <div style={{ fontSize: '0.65rem', fontWeight: 700, color: 'hsl(142 70% 35%)' }}>AFTER (Trained)</div>
                          <div style={{ marginTop: 3, whiteSpace: 'pre-wrap', lineHeight: 1.4 }}>
                            {r.after ? (r.after.slice(0, 260) + (r.after.length > 260 ? '…' : '')) : 'N/A'}
                          </div>
                        </div>
                        <div style={{
                          background: 'hsl(280 80% 55% / .04)',
                          padding: '0.45rem',
                          borderRadius: 6,
                          border: '1px solid hsl(280 80% 55% / .2)',
                        }}>
                          <div style={{ fontSize: '0.65rem', fontWeight: 700, color: 'hsl(280 80% 55%)' }}>MANUAL (Target)</div>
                          <div style={{ marginTop: 3, whiteSpace: 'pre-wrap', lineHeight: 1.4 }}>
                            {r.manual ? (r.manual.slice(0, 260) + (r.manual.length > 260 ? '…' : '')) : 'N/A'}
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      ))}

      {/* Full Comparison Modal */}
      <TestComparisonModal
        isOpen={modalState.isOpen}
        onClose={() => setModalState(prev => ({ ...prev, isOpen: false }))}
        results={modalState.results}
        initialIndex={modalState.initialIndex}
        variantName={modalState.variantName}
        accuracySummary={modalState.accuracy}
      />
    </div>
  );
}
