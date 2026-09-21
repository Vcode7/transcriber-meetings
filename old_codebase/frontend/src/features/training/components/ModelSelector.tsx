import { RefreshCw, CheckCircle, XCircle } from 'lucide-react';
import { useTrainingStore } from '../store/trainingStore';
import { fetchOllamaModels, validateConfig } from '../api/trainingApi';

export default function ModelSelector() {
  const { selectedStages, stageConfigs, updateStageConfig, ollamaModels, modelsLoading, setOllamaModels, setModelsLoading, validationResults, setValidationResult } = useTrainingStore();

  const handleRefresh = async () => {
    setModelsLoading(true);
    try {
      const models = await fetchOllamaModels();
      setOllamaModels(models);
    } catch (err) {
      console.warn('Failed to load models');
    }
    setModelsLoading(false);
  };

  const handleValidate = async (stage: string) => {
    const config = stageConfigs[stage as keyof typeof stageConfigs];
    try {
      const res = await validateConfig({
        model_name: config.model_name,
        method: config.method,
        stage: config.stage,
        hf_model_path: config.hf_model_path,
      });
      setValidationResult(stage as keyof typeof stageConfigs, res);
    } catch (err) {
      console.warn('Validation error');
    }
  };

  if (selectedStages.length === 0) {
    return <div style={{ color: 'hsl(var(--muted-foreground))' }}>Please select at least one stage first.</div>;
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
        <button onClick={handleRefresh} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', background: 'transparent', border: '1px solid hsl(var(--border))', padding: '0.5rem 1rem', borderRadius: '6px', color: 'hsl(var(--foreground))', cursor: 'pointer' }}>
          <RefreshCw size={14} className={modelsLoading ? 'animate-spin' : ''} />
          Refresh Models
        </button>
      </div>

      {selectedStages.map(stage => {
        const config = stageConfigs[stage];
        const validation = validationResults[stage];
        const isHf = config.method === 'lora' || config.method === 'qlora';
        
        return (
          <div key={stage} style={{ background: 'hsl(var(--card))', borderRadius: '12px', padding: '1.5rem', border: '1px solid hsl(var(--border))' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
              <h3 style={{ margin: 0, fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <span style={{ padding: '0.1rem 0.5rem', borderRadius: '4px', background: 'hsl(var(--primary) / 0.1)', color: 'hsl(var(--primary))', fontSize: '0.75rem' }}>
                  {stage.replace('_', ' ').toUpperCase()}
                </span>
                Model Selection
              </h3>
              
              {validation && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.25rem', fontSize: '0.8rem', color: validation.valid ? 'hsl(var(--primary))' : 'hsl(var(--destructive))' }}>
                  {validation.valid ? <CheckCircle size={14} /> : <XCircle size={14} />}
                  {validation.valid ? 'Ready' : 'Validation Failed'}
                </div>
              )}
            </div>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              <div>
                <label style={{ display: 'block', fontSize: '0.85rem', marginBottom: '0.5rem', color: 'hsl(var(--muted-foreground))' }}>Ollama Base Model</label>
                <select
                  value={config.model_name}
                  onChange={(e) => updateStageConfig(stage, { model_name: e.target.value })}
                  style={{ width: '100%', padding: '0.75rem', borderRadius: '8px', border: '1px solid hsl(var(--border))', background: 'hsl(var(--background))', color: 'hsl(var(--foreground))' }}
                >
                  <option value="">Select a model...</option>
                  {ollamaModels.map(m => (
                    <option key={m.name} value={m.name}>{m.name} ({(m.size / 1024 / 1024 / 1024).toFixed(1)} GB)</option>
                  ))}
                </select>
              </div>

              {isHf && (
                <div>
                  <label style={{ display: 'block', fontSize: '0.85rem', marginBottom: '0.5rem', color: 'hsl(var(--muted-foreground))' }}>HuggingFace Model Path (Required for LoRA/QLoRA)</label>
                  <input
                    type="text"
                    value={config.hf_model_path || ''}
                    onChange={(e) => updateStageConfig(stage, { hf_model_path: e.target.value })}
                    placeholder="e.g. meta-llama/Llama-2-7b-hf"
                    style={{ width: '100%', padding: '0.75rem', borderRadius: '8px', border: '1px solid hsl(var(--border))', background: 'hsl(var(--background))', color: 'hsl(var(--foreground))' }}
                  />
                </div>
              )}

              {validation && !validation.valid && (
                <div style={{ padding: '0.75rem', borderRadius: '8px', background: 'hsl(var(--destructive) / 0.1)', color: 'hsl(var(--destructive))', fontSize: '0.85rem' }}>
                  {validation.reason}
                </div>
              )}

              <div style={{ marginTop: '0.5rem' }}>
                <button
                  onClick={() => handleValidate(stage)}
                  disabled={!config.model_name || (isHf && !config.hf_model_path)}
                  style={{ background: 'hsl(var(--accent))', color: 'white', border: 'none', padding: '0.5rem 1rem', borderRadius: '6px', cursor: 'pointer', opacity: (!config.model_name || (isHf && !config.hf_model_path)) ? 0.5 : 1 }}
                >
                  Validate Selection
                </button>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}
