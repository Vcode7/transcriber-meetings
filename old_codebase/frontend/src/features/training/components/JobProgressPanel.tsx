import { Play, Loader2, StopCircle, RotateCcw } from 'lucide-react';
import { useTrainingStore } from '../store/trainingStore';
import { buildDataset, startTrainingJob, cancelJob } from '../api/trainingApi';
import ValidationPanel from './ValidationPanel';
import DatasetPreview from './DatasetPreview';

export default function JobProgressPanel() {
  const { 
    selectedStages, stageConfigs, 
    sourceType, selectedMeetingId, manualMom,
    datasetBuilding, setDatasetBuilding,
    builtDatasetId, setBuiltDatasetId, setDatasetPreview, setDatasetSamplesByStage,
    activeJobId, setActiveJobId, activeJob,
    validationResults
  } = useTrainingStore();

  const handleBuildDataset = async () => {
    setDatasetBuilding(true);
    try {
      const res = await buildDataset({
        source_type: sourceType,
        meeting_id: selectedMeetingId || undefined,
        stages: selectedStages,
        manual_mom: manualMom
      });
      if (res.dataset_id) {
        setBuiltDatasetId(res.dataset_id);
        setDatasetPreview(res.samples_preview || []);
        setDatasetSamplesByStage(res.samples_by_stage || {});
      }
    } catch (err) {
      console.error('Failed to build dataset', err);
    }
    setDatasetBuilding(false);
  };

  const handleStartTraining = async () => {
    if (!builtDatasetId) return;
    try {
      const configsToRun = selectedStages.map(stage => stageConfigs[stage]);
      const res = await startTrainingJob({
        dataset_id: builtDatasetId,
        stage_configs: configsToRun,
        description: 'Frontend initiated run'
      });
      setActiveJobId(res.job_id);
    } catch (err) {
      console.error('Failed to start training', err);
    }
  };

  const handleCancel = async () => {
    if (activeJobId) {
      await cancelJob(activeJobId);
    }
  };

  const handleResetJob = () => {
    setActiveJobId(null);
    setActiveJob(null);
  };

  const handleRerun = async () => {
    handleResetJob();
    if (builtDatasetId) {
      await handleStartTraining();
    }
  };

  const allValid = selectedStages.length > 0 && selectedStages.every(s => validationResults[s]?.valid);
  const isJobFinished = activeJob?.status === 'done' || activeJob?.status === 'error' || activeJob?.status === 'cancelled';
  const canStart = builtDatasetId && allValid && (!activeJobId || isJobFinished);

  return (
    <div style={{ maxWidth: '800px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      
      {/* Step 1: Build Dataset */}
      <div style={{ background: 'hsl(var(--card))', padding: '1.5rem', borderRadius: '12px', border: '1px solid hsl(var(--border))' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
          <div>
            <h3 style={{ margin: 0, fontWeight: 600 }}>1. Build Dataset</h3>
            <p style={{ margin: '0.25rem 0 0', fontSize: '0.85rem', color: 'hsl(var(--muted-foreground))' }}>Extract and format data for selected stages.</p>
          </div>
          <button
            onClick={handleBuildDataset}
            disabled={datasetBuilding || (!selectedMeetingId && sourceType === 'meeting')}
            style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', background: 'hsl(var(--accent))', color: 'white', border: 'none', padding: '0.5rem 1.5rem', borderRadius: '6px', cursor: 'pointer', opacity: datasetBuilding ? 0.7 : 1 }}
          >
            {datasetBuilding ? <Loader2 size={16} className="animate-spin" /> : <Play size={16} />}
            Build
          </button>
        </div>
        {builtDatasetId && <div style={{ fontSize: '0.85rem', color: 'hsl(var(--primary))' }}>Dataset {builtDatasetId} built successfully.</div>}
      </div>

      {/* Step 2: Preview & Validation */}
      {builtDatasetId && (
        <>
          <DatasetPreview />
          <ValidationPanel />
        </>
      )}

      {/* Step 3: Train */}
      <div style={{ background: 'hsl(var(--card))', padding: '1.5rem', borderRadius: '12px', border: '1px solid hsl(var(--border))' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.5rem' }}>
          <div>
            <h3 style={{ margin: 0, fontWeight: 600 }}>2. Execute Training</h3>
            <p style={{ margin: '0.25rem 0 0', fontSize: '0.85rem', color: 'hsl(var(--muted-foreground))' }}>Launch the optimization/fine-tuning process.</p>
          </div>
          <div style={{ display: 'flex', gap: '0.75rem' }}>
            {isJobFinished && (
              <>
                <button
                  onClick={handleRerun}
                  disabled={!builtDatasetId}
                  style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', background: 'hsl(var(--accent))', color: 'white', border: 'none', padding: '0.75rem 1.5rem', borderRadius: '6px', cursor: 'pointer', fontWeight: 600 }}
                >
                  <RotateCcw size={16} />
                  Rerun Training
                </button>
                <button
                  onClick={handleResetJob}
                  style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', background: 'transparent', color: 'hsl(var(--muted-foreground))', border: '1px solid hsl(var(--border))', padding: '0.75rem 1rem', borderRadius: '6px', cursor: 'pointer' }}
                >
                  Reset Status
                </button>
              </>
            )}
            {(!activeJobId || (!isJobFinished && !activeJob)) && (
              <button
                onClick={handleStartTraining}
                disabled={!canStart}
                style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', background: 'hsl(var(--primary))', color: 'white', border: 'none', padding: '0.75rem 2rem', borderRadius: '6px', cursor: 'pointer', opacity: !canStart ? 0.5 : 1, fontWeight: 600 }}
              >
                Start Training
              </button>
            )}
            {activeJobId && !isJobFinished && (
              <button
                onClick={handleCancel}
                style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', background: 'hsl(var(--destructive))', color: 'white', border: 'none', padding: '0.5rem 1rem', borderRadius: '6px', cursor: 'pointer' }}
              >
                <StopCircle size={16} />
                Cancel Job
              </button>
            )}
          </div>
        </div>

        {activeJob && (
          <div style={{ background: 'hsl(var(--background))', border: `1px solid ${activeJob.status === 'error' ? 'hsl(var(--destructive))' : 'hsl(var(--border))'}`, borderRadius: '8px', padding: '1.5rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '1rem', alignItems: 'center' }}>
              <div style={{ fontWeight: 600 }}>Status: <span style={{ color: activeJob.status === 'error' ? 'hsl(var(--destructive))' : 'hsl(var(--accent))', textTransform: 'capitalize' }}>{activeJob.status.replace('_', ' ')}</span></div>
              <div style={{ fontSize: '0.85rem', color: 'hsl(var(--muted-foreground))' }}>Job ID: {activeJob.job_id}</div>
            </div>

            {activeJob.error && (
              <div style={{ background: 'rgba(239, 68, 68, 0.1)', border: '1px solid hsl(var(--destructive))', color: 'hsl(var(--destructive))', padding: '1rem', borderRadius: '6px', marginBottom: '1rem', fontSize: '0.85rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                  <strong>Training Error:</strong> {activeJob.error}
                </div>
                <button
                  onClick={handleRerun}
                  style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', background: 'hsl(var(--destructive))', color: 'white', border: 'none', padding: '0.4rem 1rem', borderRadius: '4px', cursor: 'pointer', fontSize: '0.8rem', fontWeight: 600, flexShrink: 0, marginLeft: '1rem' }}
                >
                  <RotateCcw size={14} />
                  Rerun Now
                </button>
              </div>
            )}
            
            {activeJob.progress && (
              <div style={{ marginBottom: '1.5rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '0.5rem' }}>
                  <span>{activeJob.progress.message}</span>
                  <span>{Math.round(activeJob.progress.percent)}%</span>
                </div>
                <div style={{ width: '100%', height: '8px', background: 'hsl(var(--muted))', borderRadius: '4px', overflow: 'hidden' }}>
                  <div style={{ height: '100%', width: `${activeJob.progress.percent}%`, background: 'hsl(var(--accent))', transition: 'width 0.3s ease' }} />
                </div>
              </div>
            )}

            <div style={{ background: '#1e1e1e', color: '#00ff00', padding: '1rem', borderRadius: '6px', minHeight: '200px', maxHeight: '300px', overflowY: 'auto', fontSize: '0.8rem', fontFamily: 'monospace', whiteSpace: 'pre-wrap' }}>
              {activeJob.logs.length > 0 ? activeJob.logs.join('\n') : 'Waiting for logs...'}
            </div>
          </div>
        )}
      </div>

    </div>
  );
}
