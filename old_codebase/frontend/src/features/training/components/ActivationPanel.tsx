import { AlertTriangle } from 'lucide-react';
import { useTrainingStore } from '../store/trainingStore';
import { activateArtifact } from '../api/trainingApi';
import type { TrainingStage } from '../types/training';

interface Props {
  stage: string;
  onClose: () => void;
}

export default function ActivationPanel({ stage, onClose }: Props) {
  const { activeJob } = useTrainingStore();

  const handleActivate = async () => {
    if (!activeJob) return;
    const artifact = activeJob.artifacts.find(a => a.stage === stage);
    if (!artifact) {
      alert('No artifact found for this stage.');
      return;
    }
    
    try {
      await activateArtifact(activeJob.job_id, {
        artifact_id: artifact.artifact_id,
        stage: stage as TrainingStage
      });
      alert('Activation successful.');
      onClose();
    } catch (err) {
      console.error('Activation failed', err);
      alert('Activation failed.');
    }
  };

  return (
    <div style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, background: 'rgba(0,0,0,0.5)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 100 }}>
      <div style={{ background: 'hsl(var(--card))', borderRadius: '12px', padding: '2rem', maxWidth: '400px', width: '100%', border: '1px solid hsl(var(--border))', boxShadow: '0 10px 30px rgba(0,0,0,0.3)' }}>
        <h3 style={{ margin: '0 0 1rem 0', fontWeight: 600 }}>Activate {stage.replace('_', ' ').toUpperCase()} Model</h3>
        
        <div style={{ background: 'hsl(var(--warning) / 0.1)', color: 'hsl(var(--warning))', padding: '1rem', borderRadius: '8px', display: 'flex', gap: '0.75rem', marginBottom: '1.5rem', border: '1px solid hsl(var(--warning) / 0.2)' }}>
          <AlertTriangle size={20} style={{ flexShrink: 0, marginTop: '2px' }} />
          <div style={{ fontSize: '0.85rem' }}>
            The existing pipeline will continue to use its current configuration until you restart the backend server. The configuration file will be updated immediately.
          </div>
        </div>

        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '1rem' }}>
          <button onClick={onClose} style={{ background: 'transparent', border: '1px solid hsl(var(--border))', padding: '0.5rem 1rem', borderRadius: '6px', cursor: 'pointer', color: 'hsl(var(--foreground))' }}>
            Cancel
          </button>
          <button onClick={handleActivate} style={{ background: 'hsl(var(--primary))', color: 'white', border: 'none', padding: '0.5rem 1rem', borderRadius: '6px', cursor: 'pointer' }}>
            Confirm Activation
          </button>
        </div>
      </div>
    </div>
  );
}
