import { useState, useEffect } from 'react';
import { Database, CheckSquare, Square, Loader2, Sparkles, FileText, ArrowRight } from 'lucide-react';
import { toast } from 'sonner';
import api from '../../../api/client';
import { fetchEditTrainingData, generateTrainingFromEdits } from '../api/trainingApi';

interface EditItem {
  change_id: string;
  change_type: string;
  original_text: string;
  final_text: string;
  metadata: any;
  created_at: string;
}

interface MeetingItem {
  id: string;
  filename: string;
  created_at: string;
  duration: number;
}

export default function EditTrainingPanel() {
  const [meetings, setMeetings] = useState<MeetingItem[]>([]);
  const [loadingMeetings, setLoadingMeetings] = useState(false);
  const [selectedMeetingId, setSelectedMeetingId] = useState<string | null>(null);
  const [edits, setEdits] = useState<EditItem[]>([]);
  const [loadingEdits, setLoadingEdits] = useState(false);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [generating, setGenerating] = useState(false);
  const [result, setResult] = useState<{ dataset_id: string; sample_count: number; samples_preview: any[] } | null>(null);

  useEffect(() => {
    loadMeetings();
  }, []);

  const loadMeetings = async () => {
    setLoadingMeetings(true);
    try {
      const res = await api.get('/api/training/meetings');
      setMeetings(res.data?.meetings || []);
    } catch {
      toast.error('Failed to load meetings');
    } finally {
      setLoadingMeetings(false);
    }
  };

  const loadEdits = async (meetingId: string) => {
    setLoadingEdits(true);
    setEdits([]);
    setSelectedIds(new Set());
    setResult(null);
    try {
      const data = await fetchEditTrainingData(meetingId);
      setEdits(data.edits || []);
    } catch {
      toast.error('Failed to load edit history');
    } finally {
      setLoadingEdits(false);
    }
  };

  const handleSelectMeeting = (id: string) => {
    setSelectedMeetingId(id);
    loadEdits(id);
  };

  const toggleSelect = (id: string) => {
    setSelectedIds(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const selectAll = () => setSelectedIds(new Set(edits.map(e => e.change_id)));
  const deselectAll = () => setSelectedIds(new Set());

  const handleGenerate = async () => {
    if (!selectedMeetingId || selectedIds.size === 0) return;
    setGenerating(true);
    try {
      const data = await generateTrainingFromEdits(selectedMeetingId, Array.from(selectedIds));
      setResult(data);
      toast.success(`Generated ${data.sample_count} training samples`);
    } catch (e: any) {
      toast.error(e?.response?.data?.detail || 'Failed to generate training data');
    } finally {
      setGenerating(false);
    }
  };

  const changeTypeLabel: Record<string, string> = {
    merge: 'Merge',
    split: 'Split',
    find_replace: 'Find & Replace',
    delete_text: 'Delete Text',
    manual_edit: 'Manual Edit',
    delete: 'Delete Point',
  };

  const changeTypeColor: Record<string, string> = {
    merge: 'hsl(280,75%,60%)',
    split: 'hsl(140,65%,45%)',
    find_replace: 'hsl(205,90%,55%)',
    delete_text: 'hsl(0,75%,55%)',
    manual_edit: 'hsl(30,90%,50%)',
    delete: 'hsl(0,75%,55%)',
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
      {/* Header */}
      <div style={{
        background: 'hsl(var(--card))',
        borderRadius: 12,
        border: '1.5px solid hsl(var(--border) / .4)',
        padding: '1.25rem 1.5rem',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: '.75rem' }}>
          <div style={{ padding: '0.4rem', borderRadius: 8, background: 'hsl(var(--accent) / .1)', color: 'hsl(var(--accent))' }}>
            <Database size={18} />
          </div>
          <div>
            <h3 style={{ margin: 0, fontWeight: 700, fontSize: '1.05rem' }}>Train on Stage 2 Edits</h3>
            <p style={{ margin: '2px 0 0', fontSize: '.78rem', color: 'hsl(var(--pencil))' }}>
              Use your Stage 2 edit history to create training examples that teach the model your preferred style.
            </p>
          </div>
        </div>

        {/* Meeting Selector */}
        <div style={{ marginTop: '.75rem' }}>
          <label style={{ display: 'block', fontSize: '.78rem', fontWeight: 600, color: 'hsl(var(--pencil))', marginBottom: '.4rem' }}>
            Select Meeting
          </label>
          <select
            value={selectedMeetingId || ''}
            onChange={e => e.target.value && handleSelectMeeting(e.target.value)}
            style={{
              width: '100%',
              padding: '0.5rem 0.75rem',
              borderRadius: 8,
              border: '1.5px solid hsl(var(--border) / .4)',
              background: 'hsl(var(--background))',
              color: 'hsl(var(--ink))',
              fontSize: '.85rem',
              fontFamily: 'Inter, sans-serif',
            }}
          >
            <option value="">— Choose a meeting —</option>
            {meetings.map(m => (
              <option key={m.id} value={m.id}>{m.filename} ({new Date(m.created_at).toLocaleDateString()})</option>
            ))}
          </select>
        </div>
      </div>

      {/* Loading state */}
      {loadingEdits && (
        <div style={{ textAlign: 'center', padding: '2rem', color: 'hsl(var(--pencil))' }}>
          <Loader2 size={24} className="spin" style={{ margin: '0 auto 0.5rem' }} />
          <div style={{ fontSize: '.85rem' }}>Loading edit history...</div>
        </div>
      )}

      {/* No edits state */}
      {selectedMeetingId && !loadingEdits && edits.length === 0 && (
        <div style={{
          textAlign: 'center', padding: '2.5rem 1.5rem',
          background: 'hsl(var(--card))', borderRadius: 12,
          border: '1.5px solid hsl(var(--border) / .4)',
          color: 'hsl(var(--pencil))',
        }}>
          <FileText size={32} style={{ margin: '0 auto 0.75rem', opacity: 0.4 }} />
          <div style={{ fontSize: '.92rem', fontWeight: 600 }}>No Stage 2 edits found</div>
          <div style={{ fontSize: '.8rem', marginTop: '.25rem' }}>Edit discussion points in the Stage 2 tab first, then return here to generate training data.</div>
        </div>
      )}

      {/* Edit list */}
      {edits.length > 0 && (
        <div style={{
          background: 'hsl(var(--card))',
          borderRadius: 12,
          border: '1.5px solid hsl(var(--border) / .4)',
          padding: '1rem 1.25rem',
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '.75rem' }}>
            <span style={{ fontSize: '.85rem', fontWeight: 600, color: 'hsl(var(--ink))' }}>
              {edits.length} edits available · {selectedIds.size} selected
            </span>
            <div style={{ display: 'flex', gap: '.4rem' }}>
              <button onClick={selectAll} style={{
                fontSize: '.72rem', fontWeight: 600, padding: '3px 8px', borderRadius: 6,
                border: '1px solid hsl(var(--border) / .4)', background: 'none',
                color: 'hsl(var(--accent))', cursor: 'pointer',
              }}>Select All</button>
              <button onClick={deselectAll} style={{
                fontSize: '.72rem', fontWeight: 600, padding: '3px 8px', borderRadius: 6,
                border: '1px solid hsl(var(--border) / .4)', background: 'none',
                color: 'hsl(var(--pencil))', cursor: 'pointer',
              }}>Deselect All</button>
            </div>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '.5rem', maxHeight: 400, overflowY: 'auto' }}>
            {edits.map(edit => {
              const isSelected = selectedIds.has(edit.change_id);
              const color = changeTypeColor[edit.change_type] || 'hsl(var(--pencil))';
              return (
                <div
                  key={edit.change_id}
                  onClick={() => toggleSelect(edit.change_id)}
                  style={{
                    padding: '.7rem .85rem',
                    borderRadius: 8,
                    border: `1.5px solid ${isSelected ? 'hsl(var(--accent) / .5)' : 'hsl(var(--border) / .3)'}`,
                    background: isSelected ? 'hsl(var(--accent) / .04)' : 'transparent',
                    cursor: 'pointer',
                    transition: 'all 0.15s',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: '.4rem' }}>
                    {isSelected
                      ? <CheckSquare size={15} style={{ color: 'hsl(var(--accent))', flexShrink: 0 }} />
                      : <Square size={15} style={{ color: 'hsl(var(--pencil))', flexShrink: 0 }} />
                    }
                    <span className={`change-type-badge ${edit.change_type}`}>
                      {changeTypeLabel[edit.change_type] || edit.change_type}
                    </span>
                    <span style={{ fontSize: '.68rem', color: 'hsl(var(--pencil))', marginLeft: 'auto' }}>
                      {new Date(edit.created_at).toLocaleString()}
                    </span>
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr auto 1fr', gap: 8, alignItems: 'start' }}>
                    <div style={{ fontSize: '.75rem', color: 'hsl(var(--ink) / .7)', lineHeight: 1.4 }}>
                      <div style={{ fontSize: '.62rem', fontWeight: 700, color: 'hsl(0,75%,55%)', textTransform: 'uppercase', marginBottom: 2 }}>Before</div>
                      {(edit.original_text || '').substring(0, 120)}{(edit.original_text || '').length > 120 ? '…' : ''}
                    </div>
                    <ArrowRight size={14} style={{ color: 'hsl(var(--pencil))', marginTop: 12, flexShrink: 0 }} />
                    <div style={{ fontSize: '.75rem', color: 'hsl(var(--ink) / .7)', lineHeight: 1.4 }}>
                      <div style={{ fontSize: '.62rem', fontWeight: 700, color: 'hsl(140,65%,45%)', textTransform: 'uppercase', marginBottom: 2 }}>After</div>
                      {(edit.final_text || '').substring(0, 120)}{(edit.final_text || '').length > 120 ? '…' : ''}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Generate Button */}
          <div style={{ marginTop: '1rem', display: 'flex', justifyContent: 'flex-end' }}>
            <button
              onClick={handleGenerate}
              disabled={selectedIds.size === 0 || generating}
              style={{
                display: 'flex', alignItems: 'center', gap: 8,
                padding: '.6rem 1.25rem', borderRadius: 8,
                background: selectedIds.size > 0 ? 'linear-gradient(135deg, hsl(var(--accent)), hsl(14, 95%, 62%))' : 'hsl(var(--muted))',
                color: selectedIds.size > 0 ? 'white' : 'hsl(var(--pencil))',
                border: 'none', cursor: selectedIds.size > 0 ? 'pointer' : 'not-allowed',
                fontSize: '.82rem', fontWeight: 700, fontFamily: 'Inter, sans-serif',
                opacity: generating ? 0.7 : 1,
              }}
            >
              {generating ? <Loader2 size={15} className="spin" /> : <Sparkles size={15} />}
              {generating ? 'Generating...' : `Generate Training Data (${selectedIds.size})`}
            </button>
          </div>
        </div>
      )}

      {/* Results */}
      {result && (
        <div style={{
          background: 'hsl(var(--card))',
          borderRadius: 12,
          border: '1.5px solid hsl(140 65% 45% / .3)',
          padding: '1rem 1.25rem',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: '.75rem' }}>
            <div style={{ width: 8, height: 8, borderRadius: '50%', background: 'hsl(140,65%,45%)' }} />
            <span style={{ fontSize: '.9rem', fontWeight: 700, color: 'hsl(140,65%,45%)' }}>
              {result.sample_count} training samples generated
            </span>
            <span style={{ fontSize: '.72rem', color: 'hsl(var(--pencil))', marginLeft: 'auto' }}>
              Dataset: {result.dataset_id.substring(0, 8)}...
            </span>
          </div>
          {result.samples_preview?.length > 0 && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '.5rem' }}>
              {result.samples_preview.map((s: any, i: number) => (
                <details key={i} style={{ fontSize: '.78rem' }}>
                  <summary style={{ cursor: 'pointer', fontWeight: 600, color: 'hsl(var(--ink))' }}>
                    Sample {i + 1} — {s.change_id?.substring(0, 8)}
                  </summary>
                  <div style={{ marginTop: '.4rem', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '.5rem' }}>
                    <div className="diff-before">
                      <div style={{ fontSize: '.62rem', fontWeight: 700, marginBottom: 2, textTransform: 'uppercase' }}>Input</div>
                      <pre style={{ margin: 0, whiteSpace: 'pre-wrap', fontSize: '.72rem' }}>{JSON.stringify(s.inputs, null, 2)}</pre>
                    </div>
                    <div className="diff-after">
                      <div style={{ fontSize: '.62rem', fontWeight: 700, marginBottom: 2, textTransform: 'uppercase' }}>Target</div>
                      <pre style={{ margin: 0, whiteSpace: 'pre-wrap', fontSize: '.72rem' }}>{s.target}</pre>
                    </div>
                  </div>
                </details>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
