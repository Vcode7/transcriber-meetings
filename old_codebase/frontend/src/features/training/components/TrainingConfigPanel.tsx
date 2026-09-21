import { useTrainingStore } from '../store/trainingStore';

export default function TrainingConfigPanel() {
  const { selectedStages, stageConfigs, updateStageConfig } = useTrainingStore();

  if (selectedStages.length === 0) {
    return <div style={{ color: 'hsl(var(--muted-foreground))' }}>Please select at least one stage first.</div>;
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      {selectedStages.map(stage => {
        const config = stageConfigs[stage];
        const isDspy = config.method === 'dspy';
        
        return (
          <div key={stage} style={{ background: 'hsl(var(--card))', borderRadius: '12px', padding: '1.5rem', border: '1px solid hsl(var(--border))' }}>
            <h3 style={{ margin: '0 0 1rem 0', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <span style={{ padding: '0.1rem 0.5rem', borderRadius: '4px', background: 'hsl(var(--primary) / 0.1)', color: 'hsl(var(--primary))', fontSize: '0.75rem' }}>
                {stage.replace('_', ' ').toUpperCase()}
              </span>
              Hyperparameters ({config.method.toUpperCase()})
            </h3>
            
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1.5rem' }}>
              {isDspy ? (
                <>
                  <div>
                    <label style={{ display: 'block', fontSize: '0.85rem', marginBottom: '0.5rem', color: 'hsl(var(--muted-foreground))' }}>Optimizer</label>
                    <select
                      value={config.dspy_optimizer}
                      onChange={(e) => updateStageConfig(stage, { dspy_optimizer: e.target.value as any })}
                      style={{ width: '100%', padding: '0.75rem', borderRadius: '8px', border: '1px solid hsl(var(--border))', background: 'hsl(var(--background))', color: 'hsl(var(--foreground))' }}
                    >
                      <option value="BootstrapFewShot">BootstrapFewShot</option>
                      <option value="MIPROv2">MIPROv2</option>
                    </select>
                  </div>
                  <div>
                    <label style={{ display: 'block', fontSize: '0.85rem', marginBottom: '0.5rem', color: 'hsl(var(--muted-foreground))' }}>Bootstrapped Demos</label>
                    <input type="number" min="0" value={config.dspy_max_bootstrapped_demos} onChange={(e) => updateStageConfig(stage, { dspy_max_bootstrapped_demos: parseInt(e.target.value) || 0 })} style={{ width: '100%', padding: '0.75rem', borderRadius: '8px', border: '1px solid hsl(var(--border))', background: 'hsl(var(--background))', color: 'hsl(var(--foreground))' }} />
                  </div>
                  <div>
                    <label style={{ display: 'block', fontSize: '0.85rem', marginBottom: '0.5rem', color: 'hsl(var(--muted-foreground))' }}>Labeled Demos</label>
                    <input type="number" min="0" value={config.dspy_max_labeled_demos} onChange={(e) => updateStageConfig(stage, { dspy_max_labeled_demos: parseInt(e.target.value) || 0 })} style={{ width: '100%', padding: '0.75rem', borderRadius: '8px', border: '1px solid hsl(var(--border))', background: 'hsl(var(--background))', color: 'hsl(var(--foreground))' }} />
                  </div>
                  <div>
                    <label style={{ display: 'block', fontSize: '0.85rem', marginBottom: '0.5rem', color: 'hsl(var(--muted-foreground))' }}>Trials</label>
                    <input type="number" min="1" value={config.dspy_num_trials} onChange={(e) => updateStageConfig(stage, { dspy_num_trials: parseInt(e.target.value) || 1 })} style={{ width: '100%', padding: '0.75rem', borderRadius: '8px', border: '1px solid hsl(var(--border))', background: 'hsl(var(--background))', color: 'hsl(var(--foreground))' }} />
                  </div>
                </>
              ) : (
                <>
                  <div>
                    <label style={{ display: 'block', fontSize: '0.85rem', marginBottom: '0.5rem', color: 'hsl(var(--muted-foreground))' }}>Epochs</label>
                    <input type="number" min="1" value={config.num_train_epochs} onChange={(e) => updateStageConfig(stage, { num_train_epochs: parseInt(e.target.value) || 1 })} style={{ width: '100%', padding: '0.75rem', borderRadius: '8px', border: '1px solid hsl(var(--border))', background: 'hsl(var(--background))', color: 'hsl(var(--foreground))' }} />
                  </div>
                  <div>
                    <label style={{ display: 'block', fontSize: '0.85rem', marginBottom: '0.5rem', color: 'hsl(var(--muted-foreground))' }}>Batch Size</label>
                    <input type="number" min="1" value={config.per_device_train_batch_size} onChange={(e) => updateStageConfig(stage, { per_device_train_batch_size: parseInt(e.target.value) || 1 })} style={{ width: '100%', padding: '0.75rem', borderRadius: '8px', border: '1px solid hsl(var(--border))', background: 'hsl(var(--background))', color: 'hsl(var(--foreground))' }} />
                  </div>
                  <div>
                    <label style={{ display: 'block', fontSize: '0.85rem', marginBottom: '0.5rem', color: 'hsl(var(--muted-foreground))' }}>Learning Rate</label>
                    <input type="number" step="0.0001" value={config.learning_rate} onChange={(e) => updateStageConfig(stage, { learning_rate: parseFloat(e.target.value) || 0.0001 })} style={{ width: '100%', padding: '0.75rem', borderRadius: '8px', border: '1px solid hsl(var(--border))', background: 'hsl(var(--background))', color: 'hsl(var(--foreground))' }} />
                  </div>
                  <div>
                    <label style={{ display: 'block', fontSize: '0.85rem', marginBottom: '0.5rem', color: 'hsl(var(--muted-foreground))' }}>LoRA r</label>
                    <input type="number" min="1" value={config.lora_r} onChange={(e) => updateStageConfig(stage, { lora_r: parseInt(e.target.value) || 8 })} style={{ width: '100%', padding: '0.75rem', borderRadius: '8px', border: '1px solid hsl(var(--border))', background: 'hsl(var(--background))', color: 'hsl(var(--foreground))' }} />
                  </div>
                  <div>
                    <label style={{ display: 'block', fontSize: '0.85rem', marginBottom: '0.5rem', color: 'hsl(var(--muted-foreground))' }}>LoRA Alpha</label>
                    <input type="number" min="1" value={config.lora_alpha} onChange={(e) => updateStageConfig(stage, { lora_alpha: parseInt(e.target.value) || 32 })} style={{ width: '100%', padding: '0.75rem', borderRadius: '8px', border: '1px solid hsl(var(--border))', background: 'hsl(var(--background))', color: 'hsl(var(--foreground))' }} />
                  </div>
                </>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}
