import { Brain, Cpu, Zap } from 'lucide-react';
import { useTrainingStore } from '../store/trainingStore';
import type { TrainingMethod, TrainingStage } from '../types/training';

const METHODS: { id: TrainingMethod; title: string; icon: React.ElementType; desc: string; tag?: string }[] = [
  { id: 'dspy', title: 'DSPy Prompt Optimization', icon: Brain, desc: 'Optimizes prompts automatically without weight updates.', tag: 'Recommended for Ollama' },
  { id: 'lora', title: 'LoRA Fine-tuning', icon: Cpu, desc: 'Parameter-efficient fine-tuning via low-rank adaptation.', tag: 'Requires HF Model' },
  { id: 'qlora', title: 'QLoRA Fine-tuning', icon: Zap, desc: 'Quantized LoRA for highly memory-efficient training.', tag: 'Requires HF Model' },
];

export default function MethodSelector() {
  const { selectedStages, stageConfigs, updateStageConfig } = useTrainingStore();

  if (selectedStages.length === 0) {
    return <div style={{ color: 'hsl(var(--muted-foreground))' }}>Please select at least one stage first.</div>;
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      {selectedStages.map(stage => {
        const config = stageConfigs[stage];
        if (!config) return null;
        
        return (
          <div key={stage} style={{ background: 'hsl(var(--card))', borderRadius: '12px', padding: '1.5rem', border: '1px solid hsl(var(--border))' }}>
            <h3 style={{ margin: '0 0 1rem 0', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <span style={{ padding: '0.1rem 0.5rem', borderRadius: '4px', background: 'hsl(var(--primary) / 0.1)', color: 'hsl(var(--primary))', fontSize: '0.75rem' }}>
                {stage.replace('_', ' ').toUpperCase()}
              </span>
              Select Method
            </h3>
            
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '1rem' }}>
              {METHODS.map(method => {
                const isSelected = config.method === method.id;
                const Icon = method.icon;
                return (
                  <div
                    key={method.id}
                    onClick={() => updateStageConfig(stage, { method: method.id })}
                    style={{
                      padding: '1rem',
                      borderRadius: '8px',
                      border: `2px solid ${isSelected ? 'hsl(var(--accent))' : 'hsl(var(--border))'}`,
                      background: isSelected ? 'hsl(var(--accent) / 0.05)' : 'transparent',
                      cursor: 'pointer',
                      transition: 'all 0.2s',
                      position: 'relative'
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
                      <Icon size={18} color={isSelected ? 'hsl(var(--accent))' : 'hsl(var(--muted-foreground))'} />
                      <h4 style={{ margin: 0, fontWeight: 600, fontSize: '0.95rem' }}>{method.title}</h4>
                    </div>
                    <p style={{ margin: 0, fontSize: '0.8rem', color: 'hsl(var(--muted-foreground))' }}>
                      {method.desc}
                    </p>
                    {method.tag && (
                      <div style={{ marginTop: '0.75rem', fontSize: '0.7rem', display: 'inline-block', padding: '0.1rem 0.4rem', borderRadius: '4px', background: method.id === 'dspy' ? 'hsl(var(--primary) / 0.1)' : 'hsl(var(--destructive) / 0.1)', color: method.id === 'dspy' ? 'hsl(var(--primary))' : 'hsl(var(--destructive))' }}>
                        {method.tag}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        );
      })}
    </div>
  );
}
