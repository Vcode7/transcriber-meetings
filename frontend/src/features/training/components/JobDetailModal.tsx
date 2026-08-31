import { X, Download, Activity } from 'lucide-react';
import { useTrainingStore } from '../store/trainingStore';

interface Props {
  jobId: string;
  onClose: () => void;
}

export default function JobDetailModal({ jobId, onClose }: Props) {
  const { history } = useTrainingStore();
  const job = history.find(h => h.job_id === jobId);

  if (!job) return null;

  return (
    <div style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, background: 'rgba(0,0,0,0.6)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 100, padding: '2rem' }}>
      <div style={{ background: 'hsl(var(--background))', borderRadius: '12px', width: '100%', maxWidth: '800px', maxHeight: '90vh', display: 'flex', flexDirection: 'column', border: '1px solid hsl(var(--border))', boxShadow: '0 20px 40px rgba(0,0,0,0.4)' }}>
        
        <div style={{ padding: '1.5rem', borderBottom: '1px solid hsl(var(--border))', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <h2 style={{ margin: 0, fontWeight: 600, fontSize: '1.25rem' }}>Job Details</h2>
            <div style={{ fontSize: '0.85rem', color: 'hsl(var(--muted-foreground))', marginTop: '0.25rem', fontFamily: 'monospace' }}>{job.job_id}</div>
          </div>
          <button onClick={onClose} style={{ background: 'transparent', border: 'none', color: 'hsl(var(--muted-foreground))', cursor: 'pointer', padding: '0.5rem' }}>
            <X size={20} />
          </button>
        </div>

        <div style={{ padding: '1.5rem', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '2rem' }}>
          
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem' }}>
            <div style={{ background: 'hsl(var(--card))', padding: '1rem', borderRadius: '8px', border: '1px solid hsl(var(--border))' }}>
              <div style={{ fontSize: '0.75rem', color: 'hsl(var(--muted-foreground))', textTransform: 'uppercase' }}>Date</div>
              <div style={{ fontWeight: 500 }}>{new Date(job.created_at).toLocaleString()}</div>
            </div>
            <div style={{ background: 'hsl(var(--card))', padding: '1rem', borderRadius: '8px', border: '1px solid hsl(var(--border))' }}>
              <div style={{ fontSize: '0.75rem', color: 'hsl(var(--muted-foreground))', textTransform: 'uppercase' }}>Status</div>
              <div style={{ fontWeight: 500, color: job.status === 'done' ? 'hsl(142 71% 45%)' : 'hsl(var(--destructive))', textTransform: 'capitalize' }}>{job.status}</div>
            </div>
            <div style={{ background: 'hsl(var(--card))', padding: '1rem', borderRadius: '8px', border: '1px solid hsl(var(--border))' }}>
              <div style={{ fontSize: '0.75rem', color: 'hsl(var(--muted-foreground))', textTransform: 'uppercase' }}>Models</div>
              <div style={{ fontWeight: 500 }}>{job.models.join(', ') || 'N/A'}</div>
            </div>
          </div>

          {job.eval_results && Object.keys(job.eval_results).length > 0 && (
            <div>
              <h3 style={{ margin: '0 0 1rem 0', fontWeight: 600, fontSize: '1.1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}><Activity size={18} /> Evaluation Results</h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                {Object.entries(job.eval_results).map(([stage, res]) => (
                  <div key={stage} style={{ background: 'hsl(var(--card))', padding: '1rem', borderRadius: '8px', border: '1px solid hsl(var(--border))', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div style={{ fontWeight: 500 }}>{stage.replace('_', ' ').toUpperCase()}</div>
                    <div style={{ display: 'flex', gap: '2rem' }}>
                      <div style={{ textAlign: 'right' }}>
                        <div style={{ fontSize: '0.7rem', color: 'hsl(var(--muted-foreground))' }}>BEFORE</div>
                        <div style={{ fontWeight: 600 }}>{res.score_before.toFixed(2)}</div>
                      </div>
                      <div style={{ textAlign: 'right' }}>
                        <div style={{ fontSize: '0.7rem', color: 'hsl(var(--muted-foreground))' }}>AFTER</div>
                        <div style={{ fontWeight: 600, color: res.score_after > res.score_before ? 'hsl(142 71% 45%)' : 'hsl(var(--foreground))' }}>{res.score_after.toFixed(2)}</div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {job.artifacts && job.artifacts.length > 0 && (
            <div>
              <h3 style={{ margin: '0 0 1rem 0', fontWeight: 600, fontSize: '1.1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}><Download size={18} /> Generated Artifacts</h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                {job.artifacts.map(a => (
                  <div key={a.artifact_id} style={{ background: 'hsl(var(--card))', padding: '1rem', borderRadius: '8px', border: '1px solid hsl(var(--border))', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div>
                      <div style={{ fontWeight: 500, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        {a.stage.replace('_', ' ').toUpperCase()} Artifact
                        {a.is_active && <span style={{ background: 'hsl(var(--primary) / 0.1)', color: 'hsl(var(--primary))', padding: '0.1rem 0.4rem', borderRadius: '4px', fontSize: '0.65rem', fontWeight: 600 }}>ACTIVE</span>}
                      </div>
                      <div style={{ fontSize: '0.8rem', color: 'hsl(var(--muted-foreground))', fontFamily: 'monospace' }}>{a.artifact_id}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

        </div>
      </div>
    </div>
  );
}
