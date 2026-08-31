import { useEffect } from 'react';
import { History, RefreshCw, CheckCircle, XCircle, Clock, TrendingUp } from 'lucide-react';
import { useStage1Store } from '../../store/stage1Store';
import { listStage1History } from '../../api/stage1Api';

function StatusBadge({ status }: { status: string }) {
  const cfg = {
    done: { color: 'hsl(142 71% 45%)', bg: 'hsl(142 71% 45% / 0.1)', icon: <CheckCircle size={11} />, label: 'Done' },
    error: { color: 'hsl(0 75% 55%)', bg: 'hsl(0 75% 55% / 0.1)', icon: <XCircle size={11} />, label: 'Error' },
    running: { color: 'hsl(45 93% 47%)', bg: 'hsl(45 93% 47% / 0.1)', icon: <Clock size={11} />, label: 'Running' },
  }[status] ?? { color: 'hsl(var(--muted-foreground))', bg: 'hsl(var(--muted))', icon: null, label: status };

  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: '4px',
      fontSize: '0.68rem', fontWeight: 700, padding: '2px 8px', borderRadius: '10px',
      background: cfg.bg, color: cfg.color,
    }}>
      {cfg.icon} {cfg.label}
    </span>
  );
}

export default function S1HistoryTab() {
  const { history, setHistory, historyLoading, setHistoryLoading } = useStage1Store();

  const loadHistory = async () => {
    setHistoryLoading(true);
    try {
      const h = await listStage1History();
      setHistory(h);
    } catch (e) {
      console.error(e);
    } finally {
      setHistoryLoading(false);
    }
  };

  useEffect(() => {
    loadHistory();
  }, []);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div style={{
            width: 28, height: 28, borderRadius: '8px',
            background: 'hsl(var(--accent) / 0.15)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <History size={14} color="hsl(var(--accent))" />
          </div>
          <h3 style={{ margin: 0, fontSize: '0.95rem', fontWeight: 700 }}>Training History</h3>
          <span style={{
            fontSize: '0.72rem', fontWeight: 600, padding: '2px 8px', borderRadius: '6px',
            background: 'hsl(var(--muted))', color: 'hsl(var(--muted-foreground))',
          }}>
            {history.length} runs
          </span>
        </div>
        <button
          onClick={loadHistory}
          disabled={historyLoading}
          style={{
            display: 'flex', alignItems: 'center', gap: '5px',
            padding: '5px 12px', borderRadius: '8px', border: '1px solid hsl(var(--border))',
            background: 'transparent', cursor: 'pointer', fontSize: '0.78rem', color: 'hsl(var(--muted-foreground))',
          }}
        >
          <RefreshCw size={12} style={{ animation: historyLoading ? 'spin 1s linear infinite' : 'none' }} />
          Refresh
        </button>
      </div>

      {historyLoading && history.length === 0 ? (
        <div style={{ textAlign: 'center', padding: '3rem', color: 'hsl(var(--muted-foreground))' }}>
          <RefreshCw size={24} style={{ animation: 'spin 1s linear infinite', display: 'block', margin: '0 auto 10px' }} />
          Loading history...
        </div>
      ) : history.length === 0 ? (
        <div style={{
          padding: '3rem', textAlign: 'center', borderRadius: '12px',
          border: '1px dashed hsl(var(--border))', color: 'hsl(var(--muted-foreground))',
          background: 'hsl(var(--card))',
        }}>
          <History size={32} style={{ marginBottom: '12px', opacity: 0.4, display: 'block', margin: '0 auto 12px' }} />
          <div style={{ fontWeight: 600, marginBottom: '6px' }}>No training runs yet</div>
          <div style={{ fontSize: '0.82rem' }}>
            Training runs will appear here after you start optimization in the <strong>Training Loop</strong> tab.
          </div>
        </div>
      ) : (
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'separate', borderSpacing: '0 4px' }}>
            <thead>
              <tr>
                {[
                  'Run ID', 'Variant Created', 'Meeting', 'Windows',
                  'Optimizer', 'Model', 'Score Before', 'Score After',
                  'Feedback Rounds', 'Date', 'Status',
                ].map((h) => (
                  <th key={h} style={{
                    padding: '0.5rem 0.75rem', textAlign: 'left', fontSize: '0.68rem',
                    fontWeight: 700, color: 'hsl(var(--muted-foreground))',
                    textTransform: 'uppercase', letterSpacing: '0.04em',
                    borderBottom: '1px solid hsl(var(--border))',
                    whiteSpace: 'nowrap',
                  }}>
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {history.map((entry, idx) => {
                const improvement = entry.final_score - entry.baseline_score;
                return (
                  <tr key={entry.run_id} style={{
                    background: idx % 2 === 0 ? 'hsl(var(--card))' : 'transparent',
                  }}>
                    <td style={{ padding: '0.6rem 0.75rem', fontSize: '0.75rem', fontFamily: 'monospace', color: 'hsl(var(--muted-foreground))' }}>
                      {entry.run_id.slice(0, 8)}…
                    </td>
                    <td style={{ padding: '0.6rem 0.75rem', fontSize: '0.8rem', fontWeight: 600 }}>
                      {entry.variant_label}
                    </td>
                    <td style={{ padding: '0.6rem 0.75rem', fontSize: '0.78rem', maxWidth: '140px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {entry.meeting_id.slice(0, 12)}…
                    </td>
                    <td style={{ padding: '0.6rem 0.75rem', fontSize: '0.8rem', textAlign: 'center' }}>
                      <span style={{ fontWeight: 600 }}>{entry.training_window_count}</span>
                      <span style={{ color: 'hsl(var(--muted-foreground))', fontSize: '0.72rem' }}> / {entry.validation_window_count}</span>
                    </td>
                    <td style={{ padding: '0.6rem 0.75rem', fontSize: '0.78rem' }}>{entry.optimizer}</td>
                    <td style={{ padding: '0.6rem 0.75rem', fontSize: '0.78rem', maxWidth: '100px', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                      {entry.model || 'N/A'}
                    </td>
                    <td style={{ padding: '0.6rem 0.75rem', fontSize: '0.8rem', textAlign: 'center', color: 'hsl(var(--muted-foreground))' }}>
                      {entry.baseline_score > 0 ? `${entry.baseline_score}%` : '—'}
                    </td>
                    <td style={{ padding: '0.6rem 0.75rem', textAlign: 'center' }}>
                      <span style={{ fontSize: '0.85rem', fontWeight: 700, color: entry.final_score >= 80 ? 'hsl(142 71% 45%)' : entry.final_score >= 60 ? 'hsl(45 93% 47%)' : 'hsl(0 75% 55%)' }}>
                        {entry.final_score}%
                      </span>
                      {improvement > 0 && (
                        <span style={{ fontSize: '0.68rem', color: 'hsl(142 71% 45%)', marginLeft: '4px', display: 'inline-flex', alignItems: 'center', gap: '1px' }}>
                          <TrendingUp size={10} />+{improvement.toFixed(1)}
                        </span>
                      )}
                    </td>
                    <td style={{ padding: '0.6rem 0.75rem', fontSize: '0.8rem', textAlign: 'center' }}>
                      {entry.feedback_rounds}
                    </td>
                    <td style={{ padding: '0.6rem 0.75rem', fontSize: '0.75rem', color: 'hsl(var(--muted-foreground))', whiteSpace: 'nowrap' }}>
                      {new Date(entry.created_at).toLocaleDateString()}
                    </td>
                    <td style={{ padding: '0.6rem 0.75rem' }}>
                      <StatusBadge status={entry.status} />
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      <style>{`@keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}
