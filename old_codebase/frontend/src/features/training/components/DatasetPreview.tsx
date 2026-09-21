import { ChevronDown, ChevronUp } from 'lucide-react';
import { useState } from 'react';
import { useTrainingStore } from '../store/trainingStore';

export default function DatasetPreview() {
  const { datasetPreview, datasetSamplesByStage } = useTrainingStore();
  const [expandedIndex, setExpandedIndex] = useState<number | null>(null);

  if (!datasetPreview || datasetPreview.length === 0) {
    return <div style={{ color: 'hsl(var(--muted-foreground))' }}>No dataset built yet.</div>;
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', marginBottom: '2rem' }}>
      <h3 style={{ margin: 0, fontWeight: 600 }}>Dataset Preview</h3>
      
      <div style={{ display: 'flex', gap: '1rem' }}>
        {Object.entries(datasetSamplesByStage).map(([stage, count]) => (
          <div key={stage} style={{ padding: '1rem', background: 'hsl(var(--card))', border: '1px solid hsl(var(--border))', borderRadius: '8px', minWidth: '150px' }}>
            <div style={{ fontSize: '0.75rem', color: 'hsl(var(--muted-foreground))', textTransform: 'uppercase' }}>{stage.replace('_', ' ')} Samples</div>
            <div style={{ fontSize: '1.5rem', fontWeight: 700 }}>{count}</div>
          </div>
        ))}
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
        {datasetPreview.slice(0, 5).map((sample: any, idx) => {
          const isExpanded = expandedIndex === idx;
          return (
            <div key={idx} style={{ background: 'hsl(var(--card))', border: '1px solid hsl(var(--border))', borderRadius: '8px', overflow: 'hidden' }}>
              <div 
                onClick={() => setExpandedIndex(isExpanded ? null : idx)}
                style={{ padding: '1rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center', cursor: 'pointer', background: 'hsl(var(--accent) / 0.05)' }}
              >
                <div style={{ fontWeight: 500, fontSize: '0.9rem' }}>Sample {idx + 1} ({sample.stage})</div>
                {isExpanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
              </div>
              
              {isExpanded && (
                <div style={{ padding: '1rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                  <div>
                    <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'hsl(var(--muted-foreground))', marginBottom: '0.5rem' }}>INPUTS</div>
                    <pre style={{ margin: 0, background: 'hsl(var(--background))', padding: '1rem', borderRadius: '6px', fontSize: '0.8rem', overflowX: 'auto', whiteSpace: 'pre-wrap' }}>
                      {JSON.stringify(sample.inputs, null, 2)}
                    </pre>
                  </div>
                  <div>
                    <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'hsl(var(--muted-foreground))', marginBottom: '0.5rem' }}>TARGET</div>
                    <pre style={{ margin: 0, background: 'hsl(var(--background))', padding: '1rem', borderRadius: '6px', fontSize: '0.8rem', overflowX: 'auto', whiteSpace: 'pre-wrap' }}>
                      {sample.target}
                    </pre>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
