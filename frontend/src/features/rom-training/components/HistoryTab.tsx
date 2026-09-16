import { useState, useEffect } from 'react';
import { History, Trash2, Loader, ChevronDown, ChevronUp, Check, AlertCircle, Brain } from 'lucide-react';
import { listVariants, deleteVariant } from '../api/romTrainingApi';
import type { RomVariant, TestResult } from '../types/romTrainingTypes';

export default function HistoryTab() {
  const [variants, setVariants] = useState<RomVariant[]>([]);
  const [loading, setLoading] = useState(true);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [deleting, setDeleting] = useState<string | null>(null);

  const load = () => {
    setLoading(true);
    listVariants()
      .then(r => setVariants(r.variants || []))
      .catch(() => {})
      .finally(() => setLoading(false));
  };

  useEffect(() => { load(); }, []);

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
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <History size={18} style={{ color: 'hsl(var(--accent))' }} />
          <span style={{ fontWeight: 700, fontSize: '1rem' }}>Training History</span>
        </div>
        <span style={{ fontSize: '0.78rem', color: 'hsl(var(--muted-foreground))' }}>
          {variants.length} variant{variants.length !== 1 ? 's' : ''}
        </span>
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
              <button
                style={{
                  display: 'flex', alignItems: 'center', gap: 4,
                  padding: '0.3rem 0.6rem', borderRadius: 6, border: 'none',
                  background: 'hsl(var(--destructive) / .1)',
                  color: 'hsl(var(--destructive))', cursor: 'pointer',
                  fontSize: '0.72rem', fontWeight: 600,
                }}
                onClick={e => { e.stopPropagation(); handleDelete(v.id); }}
                disabled={deleting === v.id}
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

              {/* Test results preview */}
              {v.test_results.length > 0 && (
                <div style={{ marginTop: '0.75rem' }}>
                  <div style={{ fontSize: '0.7rem', fontWeight: 700, color: 'hsl(var(--muted-foreground))', marginBottom: 6 }}>
                    TEST RESULTS ({v.test_results.length} samples)
                  </div>
                  {v.test_results.slice(0, 2).map((r, i) => (
                    <div key={i} style={{
                      marginBottom: 8, padding: '0.5rem 0.75rem', borderRadius: 8,
                      border: '1px solid hsl(var(--border))', fontSize: '0.72rem',
                    }}>
                      <div style={{ fontWeight: 700, marginBottom: 4 }}>{r.agenda_title || `Sample ${i + 1}`}</div>
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                        <div>
                          <div style={{ fontSize: '0.65rem', fontWeight: 700, color: 'hsl(var(--muted-foreground))' }}>AFTER (Trained)</div>
                          <div style={{ marginTop: 2, whiteSpace: 'pre-wrap' }}>{r.after.slice(0, 200)}{r.after.length > 200 ? '…' : ''}</div>
                        </div>
                        <div>
                          <div style={{ fontSize: '0.65rem', fontWeight: 700, color: 'hsl(280 80% 55%)' }}>MANUAL</div>
                          <div style={{ marginTop: 2, whiteSpace: 'pre-wrap' }}>{r.manual.slice(0, 200)}{r.manual.length > 200 ? '…' : ''}</div>
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
    </div>
  );
}
