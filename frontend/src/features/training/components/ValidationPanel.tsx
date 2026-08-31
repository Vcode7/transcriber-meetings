import { CheckCircle, XCircle } from 'lucide-react';
import { useTrainingStore } from '../store/trainingStore';
import { validateConfig } from '../api/trainingApi';

export default function ValidationPanel() {
  const { selectedStages, stageConfigs, validationResults, setValidationResult } = useTrainingStore();

  const handleRunAll = async () => {
    for (const stage of selectedStages) {
      const config = stageConfigs[stage];
      try {
        const res = await validateConfig({
          model_name: config.model_name,
          method: config.method,
          stage: config.stage,
          hf_model_path: config.hf_model_path,
        });
        setValidationResult(stage, res);
      } catch (err) {
        console.warn(`Validation error for ${stage}`);
      }
    }
  };

  const allValid = selectedStages.every(s => validationResults[s]?.valid);

  return (
    <div style={{ background: 'hsl(var(--card))', borderRadius: '12px', padding: '1.5rem', border: '1px solid hsl(var(--border))', marginBottom: '2rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
        <h3 style={{ margin: 0, fontWeight: 600 }}>Validation Summary</h3>
        <button
          onClick={handleRunAll}
          style={{ background: 'hsl(var(--primary))', color: 'white', border: 'none', padding: '0.5rem 1rem', borderRadius: '6px', cursor: 'pointer', fontSize: '0.85rem' }}
        >
          Run All Validations
        </button>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
        {selectedStages.map(stage => {
          const res = validationResults[stage];
          return (
            <div key={stage} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '1rem', background: 'hsl(var(--background))', borderRadius: '8px', border: '1px solid hsl(var(--border))' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                <span style={{ fontWeight: 600, fontSize: '0.9rem' }}>{stage.replace('_', ' ').toUpperCase()}</span>
                <span style={{ fontSize: '0.85rem', color: 'hsl(var(--muted-foreground))' }}>{stageConfigs[stage].model_name || 'No model selected'}</span>
              </div>
              
              {res ? (
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: res.valid ? 'hsl(var(--primary))' : 'hsl(var(--destructive))', fontSize: '0.85rem' }}>
                  {res.valid ? <CheckCircle size={16} /> : <XCircle size={16} />}
                  {res.valid ? 'Ready' : res.reason}
                </div>
              ) : (
                <div style={{ fontSize: '0.85rem', color: 'hsl(var(--muted-foreground))' }}>Not validated</div>
              )}
            </div>
          );
        })}
      </div>

      {allValid && selectedStages.length > 0 && (
        <div style={{ marginTop: '1rem', padding: '1rem', background: 'hsl(var(--primary) / 0.1)', color: 'hsl(var(--primary))', borderRadius: '8px', textAlign: 'center', fontWeight: 500 }}>
          All stages are valid and ready for training!
        </div>
      )}
    </div>
  );
}
