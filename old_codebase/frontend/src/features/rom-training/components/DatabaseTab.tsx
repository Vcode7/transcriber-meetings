import { useState, useEffect } from 'react';
import { Database, Trash2, Loader, ChevronDown, ChevronUp, AlertCircle } from 'lucide-react';
import { listTrainingData, deleteTrainingData } from '../api/romTrainingApi';
import type { TrainingDataEntry } from '../types/romTrainingTypes';

export default function DatabaseTab() {
  const [data, setData] = useState<TrainingDataEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [deleting, setDeleting] = useState<string | null>(null);

  const load = () => {
    setLoading(true);
    listTrainingData()
      .then(r => setData(r.data || []))
      .catch(() => {})
      .finally(() => setLoading(false));
  };

  useEffect(() => { load(); }, []);

  const handleDelete = async (id: string) => {
    if (!confirm('Delete this training data entry?')) return;
    setDeleting(id);
    try {
      await deleteTrainingData(id);
      setData(prev => prev.filter(d => d.id !== id));
    } catch {
    } finally {
      setDeleting(null);
    }
  };

  // Group by recording
  const grouped: Record<string, TrainingDataEntry[]> = {};
  for (const entry of data) {
    const key = entry.recording_id;
    if (!grouped[key]) grouped[key] = [];
    grouped[key].push(entry);
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <Database size={18} style={{ color: 'hsl(var(--accent))' }} />
          <span style={{ fontWeight: 700, fontSize: '1rem' }}>Training Database</span>
        </div>
        <span style={{ fontSize: '0.78rem', color: 'hsl(var(--muted-foreground))' }}>
          {data.length} entr{data.length !== 1 ? 'ies' : 'y'}
        </span>
      </div>

      {loading && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: 'hsl(var(--muted-foreground))', padding: '2rem' }}>
          <Loader size={16} className="spin" /> Loading…
        </div>
      )}

      {!loading && data.length === 0 && (
        <div style={{
          textAlign: 'center', padding: '3rem',
          background: 'hsl(var(--card))', borderRadius: 12,
          border: '1px solid hsl(var(--border))',
          color: 'hsl(var(--muted-foreground))', fontSize: '0.85rem',
        }}>
          <Database size={32} style={{ marginBottom: 12, opacity: 0.4 }} />
          <p>No training data yet.</p>
          <p style={{ fontSize: '0.75rem' }}>Use the Data Creation tab to add meeting data.</p>
        </div>
      )}

      {!loading && Object.entries(grouped).map(([recId, entries]) => {
        const title = entries[0]?.recording_title || recId;
        return (
          <div key={recId} style={{
            background: 'hsl(var(--card))',
            borderRadius: 12, border: '1px solid hsl(var(--border))',
            overflow: 'hidden',
          }}>
            <div style={{
              padding: '0.75rem 1rem',
              background: 'hsl(var(--muted) / .3)',
              borderBottom: '1px solid hsl(var(--border))',
              display: 'flex', alignItems: 'center', justifyContent: 'space-between',
            }}>
              <div style={{ fontWeight: 700, fontSize: '0.85rem' }}>{title}</div>
              <span style={{
                fontSize: '0.7rem', fontWeight: 700, padding: '2px 8px',
                borderRadius: 6, background: 'hsl(var(--accent) / .12)',
                color: 'hsl(var(--accent))',
              }}>
                {entries.length} agenda{entries.length !== 1 ? 's' : ''}
              </span>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column' }}>
              {entries.map(entry => (
                <div key={entry.id} style={{ borderBottom: '1px solid hsl(var(--border) / .5)' }}>
                  <div
                    style={{
                      display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                      padding: '0.6rem 1rem', cursor: 'pointer',
                    }}
                    onClick={() => setExpandedId(expandedId === entry.id ? null : entry.id)}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      {expandedId === entry.id ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                      <span style={{ fontSize: '0.82rem', fontWeight: 600 }}>{entry.agenda_title}</span>
                      <span style={{ fontSize: '0.7rem', color: 'hsl(var(--muted-foreground))' }}>
                        {entry.long_rom_points.length} ROM pts · {entry.manual_mom_points.length} MoM pts
                      </span>
                    </div>
                    <button
                      style={{
                        display: 'flex', alignItems: 'center', gap: 4,
                        padding: '0.3rem 0.6rem', borderRadius: 6, border: 'none',
                        background: 'hsl(var(--destructive) / .1)',
                        color: 'hsl(var(--destructive))', cursor: 'pointer',
                        fontSize: '0.72rem', fontWeight: 600,
                      }}
                      onClick={e => { e.stopPropagation(); handleDelete(entry.id); }}
                      disabled={deleting === entry.id}
                    >
                      {deleting === entry.id ? <Loader size={11} className="spin" /> : <Trash2 size={11} />}
                      Delete
                    </button>
                  </div>
                  {expandedId === entry.id && (
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 0, padding: '0 1rem 0.75rem' }}>
                      <div style={{ paddingRight: '0.75rem', borderRight: '1px solid hsl(var(--border))' }}>
                        <div style={{ fontSize: '0.7rem', fontWeight: 700, color: 'hsl(var(--accent))', marginBottom: 6 }}>LONG ROM POINTS</div>
                        {entry.long_rom_points.slice(0, 8).map((pt: any, i: number) => (
                          <div key={i} style={{ fontSize: '0.72rem', marginBottom: 3, paddingLeft: 8, borderLeft: '2px solid hsl(var(--accent) / .3)' }}>
                            {pt.polished_text || pt.text || String(pt)}
                          </div>
                        ))}
                        {entry.long_rom_points.length > 8 && (
                          <div style={{ fontSize: '0.68rem', color: 'hsl(var(--muted-foreground))' }}>+ {entry.long_rom_points.length - 8} more</div>
                        )}
                      </div>
                      <div style={{ paddingLeft: '0.75rem' }}>
                        <div style={{ fontSize: '0.7rem', fontWeight: 700, color: 'hsl(280 80% 60%)', marginBottom: 6 }}>MANUAL MoM POINTS</div>
                        {(typeof entry.manual_mom_points[0] === 'string'
                          ? entry.manual_mom_points as string[]
                          : entry.manual_mom_points.map((p: any) => p.text || String(p))
                        ).slice(0, 8).map((pt: string, i: number) => (
                          <div key={i} style={{ fontSize: '0.72rem', marginBottom: 3, paddingLeft: 8, borderLeft: '2px solid hsl(280 80% 60% / .3)' }}>
                            {pt}
                          </div>
                        ))}
                        {entry.manual_mom_points.length > 8 && (
                          <div style={{ fontSize: '0.68rem', color: 'hsl(var(--muted-foreground))' }}>+ {entry.manual_mom_points.length - 8} more</div>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}
