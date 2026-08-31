import { useState } from 'react';
import { useTrainingStore } from '../store/trainingStore';
import { Eye, ExternalLink } from 'lucide-react';
import JobDetailModal from './JobDetailModal';

export default function TrainingHistoryTable() {
  const { history, historyArtifacts } = useTrainingStore();
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);

  if (!history || history.length === 0) {
    return <div style={{ color: 'hsl(var(--muted-foreground))' }}>No training history available.</div>;
  }

  return (
    <div>
      <div style={{ background: 'hsl(var(--card))', borderRadius: '12px', border: '1px solid hsl(var(--border))', overflow: 'hidden' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.85rem' }}>
          <thead>
            <tr style={{ background: 'hsl(var(--muted) / 0.3)', borderBottom: '1px solid hsl(var(--border))' }}>
              <th style={{ padding: '1rem', fontWeight: 600 }}>Date</th>
              <th style={{ padding: '1rem', fontWeight: 600 }}>Job ID</th>
              <th style={{ padding: '1rem', fontWeight: 600 }}>Stages</th>
              <th style={{ padding: '1rem', fontWeight: 600 }}>Method</th>
              <th style={{ padding: '1rem', fontWeight: 600 }}>Status</th>
              <th style={{ padding: '1rem', fontWeight: 600, textAlign: 'right' }}>Actions</th>
            </tr>
          </thead>
          <tbody>
            {history.map(run => {
              const date = new Date(run.created_at).toLocaleDateString();
              const isSuccess = run.status === 'done';
              
              return (
                <tr key={run.job_id} style={{ borderBottom: '1px solid hsl(var(--border))' }}>
                  <td style={{ padding: '1rem' }}>{date}</td>
                  <td style={{ padding: '1rem', fontFamily: 'monospace' }}>{run.job_id.slice(0,8)}...</td>
                  <td style={{ padding: '1rem' }}>
                    <div style={{ display: 'flex', gap: '0.25rem', flexWrap: 'wrap' }}>
                      {run.stages.map(s => (
                        <span key={s} style={{ background: 'hsl(var(--primary) / 0.1)', color: 'hsl(var(--primary))', padding: '0.1rem 0.4rem', borderRadius: '4px', fontSize: '0.7rem' }}>
                          {s.replace('stage_', 'S')}
                        </span>
                      ))}
                    </div>
                  </td>
                  <td style={{ padding: '1rem' }}>{run.methods.join(', ').toUpperCase()}</td>
                  <td style={{ padding: '1rem' }}>
                    <span style={{ 
                      padding: '0.2rem 0.6rem', 
                      borderRadius: '12px', 
                      fontSize: '0.75rem', 
                      fontWeight: 500,
                      background: isSuccess ? 'hsl(142 71% 45% / 0.1)' : 'hsl(var(--destructive) / 0.1)',
                      color: isSuccess ? 'hsl(142 71% 45%)' : 'hsl(var(--destructive))'
                    }}>
                      {run.status}
                    </span>
                  </td>
                  <td style={{ padding: '1rem', textAlign: 'right' }}>
                    <button 
                      onClick={() => setSelectedJobId(run.job_id)}
                      style={{ background: 'transparent', border: '1px solid hsl(var(--border))', color: 'hsl(var(--foreground))', padding: '0.3rem 0.75rem', borderRadius: '6px', cursor: 'pointer', display: 'inline-flex', alignItems: 'center', gap: '0.4rem' }}
                    >
                      <Eye size={14} /> View
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {selectedJobId && (
        <JobDetailModal jobId={selectedJobId} onClose={() => setSelectedJobId(null)} />
      )}
    </div>
  );
}
