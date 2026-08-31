import { FileText, List, CheckSquare, Layers } from 'lucide-react';
import { useTrainingStore } from '../store/trainingStore';
import type { TrainingStage } from '../types/training';

const STAGES: { id: TrainingStage; title: string; icon: React.ElementType; color: string; desc: string }[] = [
  { id: 'stage_1', title: 'Stage 1 — Context Planning', icon: FileText, color: 'hsl(var(--primary))', desc: 'Trains model to extract basic entities and structure.' },
  { id: 'stage_2', title: 'Stage 2 — Discussion Point Enhancement', icon: List, color: 'hsl(var(--accent))', desc: 'Improves detail and synthesis of discussion points.' },
  { id: 'stage_3', title: 'Stage 3 — Agenda Matching', icon: CheckSquare, color: 'hsl(var(--destructive))', desc: 'Aligns points to agenda items accurately.' },
];

export default function StageSelector() {
  const { selectedStages, toggleStage, datasetSamplesByStage } = useTrainingStore();

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      <div>
        <h2 style={{ fontSize: '1.2rem', fontWeight: 600, marginBottom: '0.5rem', color: 'hsl(var(--foreground))' }}>
          Select Training Stages
        </h2>
        <p style={{ margin: 0, fontSize: '0.85rem', color: 'hsl(var(--muted-foreground))' }}>
          Choose which stages of the MoM pipeline you want to optimize.
        </p>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
        {STAGES.map((stage) => {
          const isSelected = selectedStages.includes(stage.id);
          const Icon = stage.icon;
          const samples = datasetSamplesByStage[stage.id] || 0;
          
          return (
            <div
              key={stage.id}
              onClick={() => toggleStage(stage.id)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '1rem',
                padding: '1.25rem',
                borderRadius: '12px',
                border: `2px solid ${isSelected ? stage.color : 'hsl(var(--border))'}`,
                background: isSelected ? `${stage.color.replace(')', ' / 0.05)')}` : 'hsl(var(--card))',
                cursor: 'pointer',
                transition: 'all 0.2s',
              }}
            >
              <div style={{
                width: 24, height: 24,
                borderRadius: '4px',
                border: `2px solid ${isSelected ? stage.color : 'hsl(var(--muted-foreground))'}`,
                background: isSelected ? stage.color : 'transparent',
                display: 'flex', alignItems: 'center', justifyContent: 'center'
              }}>
                {isSelected && <Layers size={14} color="white" />}
              </div>
              
              <div style={{ padding: '0.5rem', borderRadius: '8px', background: `${stage.color.replace(')', ' / 0.1)')}`, color: stage.color }}>
                <Icon size={24} />
              </div>
              
              <div style={{ flex: 1 }}>
                <h3 style={{ margin: 0, fontWeight: 600, fontSize: '1rem' }}>{stage.title}</h3>
                <p style={{ margin: '0.25rem 0 0', fontSize: '0.85rem', color: 'hsl(var(--muted-foreground))' }}>
                  {stage.desc}
                </p>
              </div>
              
              <div style={{ textAlign: 'right' }}>
                <div style={{ fontSize: '0.75rem', color: 'hsl(var(--muted-foreground))', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                  Est. Samples
                </div>
                <div style={{ fontSize: '1.2rem', fontWeight: 700, color: 'hsl(var(--foreground))' }}>
                  {samples > 0 ? samples : '-'}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
