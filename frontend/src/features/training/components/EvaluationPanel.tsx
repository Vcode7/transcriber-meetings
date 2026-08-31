import { useState } from 'react';
import { useTrainingStore } from '../store/trainingStore';
import { ArrowUpRight, ArrowDownRight, Minus, Check } from 'lucide-react';
import ActivationPanel from './ActivationPanel';

export default function EvaluationPanel() {
  const { activeJob } = useTrainingStore();
  const [stageToActivate, setStageToActivate] = useState<string | null>(null);

  if (!activeJob) {
    return <div style={{ color: 'hsl(var(--muted-foreground))' }}>No active or recently completed job.</div>;
  }
  
  if (activeJob.status !== 'done' || !activeJob.eval_results) {
    return <div style={{ color: 'hsl(var(--muted-foreground))' }}>Job is not done or evaluation results are not available yet. (Current status: {activeJob.status})</div>;
  }

  const results = activeJob.eval_results;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      
      {Object.entries(results).map(([stage, res]) => {
        const diff = res.score_after - res.score_before;
        const isBetter = diff > 0;
        const isSame = diff === 0;

        return (
          <div key={stage} style={{ background: 'hsl(var(--card))', borderRadius: '12px', border: '1px solid hsl(var(--border))', overflow: 'hidden' }}>
            <div style={{ padding: '1.5rem', borderBottom: '1px solid hsl(var(--border))', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <h3 style={{ margin: 0, fontWeight: 600, fontSize: '1.1rem' }}>{stage.replace('_', ' ').toUpperCase()} Results</h3>
                <div style={{ fontSize: '0.85rem', color: 'hsl(var(--muted-foreground))', marginTop: '0.25rem' }}>Method: {res.method.toUpperCase()}</div>
              </div>
              
              <div style={{ display: 'flex', gap: '2rem' }}>
                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: '0.75rem', color: 'hsl(var(--muted-foreground))', textTransform: 'uppercase' }}>Before</div>
                  <div style={{ fontSize: '1.5rem', fontWeight: 600 }}>{res.score_before.toFixed(2)}</div>
                </div>
                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: '0.75rem', color: 'hsl(var(--muted-foreground))', textTransform: 'uppercase' }}>After</div>
                  <div style={{ fontSize: '1.5rem', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.25rem', color: isBetter ? 'hsl(142 71% 45%)' : isSame ? 'hsl(var(--foreground))' : 'hsl(var(--destructive))' }}>
                    {res.score_after.toFixed(2)}
                    {isBetter && <ArrowUpRight size={20} />}
                    {!isBetter && !isSame && <ArrowDownRight size={20} />}
                    {isSame && <Minus size={20} />}
                  </div>
                </div>
              </div>
            </div>
            
            <div style={{ padding: '1.5rem', background: 'hsl(var(--accent) / 0.02)' }}>
              <h4 style={{ margin: '0 0 1rem 0', fontSize: '0.9rem', fontWeight: 600 }}>Sample Comparison</h4>
              
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                {res.sample_comparisons?.slice(0, 2).map((comp: any, idx) => (
                  <div key={idx} style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                    <div style={{ background: 'hsl(var(--background))', padding: '1rem', borderRadius: '8px', border: '1px solid hsl(var(--border))' }}>
                      <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'hsl(var(--muted-foreground))', marginBottom: '0.5rem' }}>EXPECTED</div>
                      <div style={{ fontSize: '0.85rem', whiteSpace: 'pre-wrap' }}>{comp.expected}</div>
                    </div>
                    <div style={{ background: 'hsl(var(--background))', padding: '1rem', borderRadius: '8px', border: `1px solid ${isBetter ? 'hsl(142 71% 45% / 0.5)' : 'hsl(var(--border))'}` }}>
                      <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'hsl(var(--muted-foreground))', marginBottom: '0.5rem' }}>PREDICTED</div>
                      <div style={{ fontSize: '0.85rem', whiteSpace: 'pre-wrap' }}>{comp.predicted}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div style={{ padding: '1rem 1.5rem', display: 'flex', justifyContent: 'flex-end', gap: '1rem', borderTop: '1px solid hsl(var(--border))' }}>
              <button style={{ background: 'transparent', border: '1px solid hsl(var(--border))', color: 'hsl(var(--foreground))', padding: '0.5rem 1rem', borderRadius: '6px', cursor: 'pointer', fontSize: '0.85rem' }}>
                View Full Results
              </button>
              <button 
                onClick={() => setStageToActivate(stage)}
                style={{ background: 'hsl(var(--primary))', color: 'white', border: 'none', padding: '0.5rem 1.25rem', borderRadius: '6px', cursor: 'pointer', fontSize: '0.85rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}
              >
                <Check size={14} /> Activate for {stage.replace('_', ' ').toUpperCase()}
              </button>
            </div>
          </div>
        );
      })}

      {stageToActivate && (
        <ActivationPanel stage={stageToActivate} onClose={() => setStageToActivate(null)} />
      )}
    </div>
  );
}
