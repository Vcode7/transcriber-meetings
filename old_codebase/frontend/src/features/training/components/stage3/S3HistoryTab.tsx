import { useEffect } from 'react';
import { History, CheckCircle, AlertTriangle, Clock } from 'lucide-react';
import { useStage3Store } from '../../store/stage3Store';
import { listStage3History } from '../../api/stage3Api';

const S = {
  card: {
    background: 'hsl(var(--card))',
    borderRadius: '12px',
    border: '1px solid hsl(var(--border))',
    padding: '1.25rem',
  } as React.CSSProperties,
  th: {
    padding: '0.65rem 0.85rem',
    textAlign: 'left' as const,
    fontSize: '0.72rem',
    fontWeight: 700,
    color: 'hsl(var(--muted-foreground))',
    textTransform: 'uppercase' as const,
    letterSpacing: '0.04em',
    borderBottom: '1px solid hsl(var(--border))',
  },
  td: {
    padding: '0.75rem 0.85rem',
    fontSize: '0.8rem',
    borderBottom: '1px solid hsl(var(--border) / 0.5)',
    verticalAlign: 'middle' as const,
  },
};

export default function S3HistoryTab() {
  const { history, setHistory, historyLoading, setHistoryLoading } = useStage3Store();

  useEffect(() => {
    setHistoryLoading(true);
    listStage3History()
      .then((res) => setHistory(res.history || []))
      .catch(console.warn)
      .finally(() => setHistoryLoading(false));
  }, []);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', width: '100%' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        <div style={{
          width: 28, height: 28, borderRadius: '8px',
          background: 'hsl(var(--accent) / .15)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}>
          <History size={14} color="hsl(var(--accent))" />
        </div>
        <div>
          <h3 style={{ margin: 0, fontSize: '0.95rem', fontWeight: 700 }}>Stage 3 Training Run History</h3>
          <p style={{ margin: '2px 0 0', fontSize: '0.76rem', color: 'hsl(var(--muted-foreground))' }}>
            Chronological audit log of all Stage 3 DSPy optimization and retraining runs.
          </p>
        </div>
      </div>

      <div style={{ ...S.card, overflowX: 'auto', padding: 0 }}>
        {history.length === 0 ? (
          <div style={{ textAlign: 'center', padding: '2.5rem', color: 'hsl(var(--muted-foreground))', fontSize: '0.85rem' }}>
            No Stage 3 training runs recorded yet.
          </div>
        ) : (
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ background: 'hsl(var(--muted) / 0.3)' }}>
                <th style={S.th}>Run ID</th>
                <th style={S.th}>Timestamp</th>
                <th style={S.th}>Variant Created</th>
                <th style={S.th}>Feedback Used</th>
                <th style={S.th}>Accuracy</th>
                <th style={S.th}>Overall Score</th>
                <th style={S.th}>Status</th>
              </tr>
            </thead>
            <tbody>
              {history.map((h, i) => {
                const scores = h.scores || { overall: 0, accuracy: 0 };
                return (
                  <tr key={h.run_id || i}>
                    <td style={S.td}>
                      <span style={{ fontFamily: 'monospace', fontSize: '0.75rem', fontWeight: 700 }}>
                        {h.run_id}
                      </span>
                    </td>
                    <td style={S.td}>
                      <span style={{ fontSize: '0.75rem', color: 'hsl(var(--muted-foreground))' }}>
                        {new Date(h.timestamp).toLocaleString()}
                      </span>
                    </td>
                    <td style={S.td}>
                      <span style={{ fontWeight: 600 }}>{h.variant_id || '—'}</span>
                    </td>
                    <td style={S.td}>
                      <span>{h.feedback_count ?? 0}</span>
                    </td>
                    <td style={S.td}>
                      <span style={{ fontWeight: 700, color: 'hsl(140 70% 45%)' }}>
                        {Math.round((scores.accuracy || 0) * 100)}%
                      </span>
                    </td>
                    <td style={S.td}>
                      <span style={{ fontWeight: 800, color: 'hsl(var(--accent))' }}>
                        {Math.round((scores.overall || 0) * 100)}%
                      </span>
                    </td>
                    <td style={S.td}>
                      <span style={{
                        fontSize: '0.72rem', fontWeight: 700, padding: '2px 7px', borderRadius: '5px',
                        background: h.status === 'done' ? 'hsl(140 70% 45% / 0.15)' : 'hsl(0 75% 55% / 0.15)',
                        color: h.status === 'done' ? 'hsl(140 70% 45%)' : 'hsl(0 75% 55%)',
                      }}>
                        {h.status}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
