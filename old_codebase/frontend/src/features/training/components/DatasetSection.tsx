import { useState } from 'react';
import { Database, FileText, Sparkles, Loader2, CheckCircle2, ArrowRight } from 'lucide-react';
import { useTrainingStore } from '../store/trainingStore';
import { buildDataset } from '../api/trainingApi';
import { toast } from 'sonner';
import ManualMomEditor from './ManualMomEditor';
import MeetingSelector from './MeetingSelector';
import FileUploadPanel from './FileUploadPanel';

export default function DatasetSection() {
  const {
    inputSourceType,
    setInputSourceType,
    selectedMeetingId,
    selectedStages,
    manualMom,
    useEditedStage2,
    uploadedTranscript,
    uploadedContext,
    uploadedAgenda,
    extractedMomPoints,
    setBuiltDatasetId,
    setDatasetPreview,
    setDatasetSamplesByStage,
    datasetBuilding,
    setDatasetBuilding,
  } = useTrainingStore();

  const [builtResult, setBuiltResult] = useState<{
    dataset_id: string;
    total_samples: number;
    samples_by_stage: Record<string, number>;
    samples_preview: any[];
  } | null>(null);

  const handleBuildDataset = async () => {
    if (selectedStages.length === 0) {
      toast.error('Please select at least one stage to train (in Stage Selection tab).');
      return;
    }

    if (inputSourceType === 'meeting' && !selectedMeetingId) {
      toast.error('Please select an existing meeting on the right panel.');
      return;
    }

    setDatasetBuilding(true);
    setBuiltResult(null);

    try {
      // If extracted points are available, format them as JSON target
      const targetMomText = extractedMomPoints.length > 0
        ? JSON.stringify(extractedMomPoints)
        : manualMom;

      const data = await buildDataset({
        source_type: inputSourceType === 'meeting' ? 'meeting' : 'upload',
        meeting_id: selectedMeetingId || undefined,
        stages: selectedStages,
        manual_mom: targetMomText || undefined,
        use_edited_stage2: useEditedStage2,
        transcript_text: uploadedTranscript || undefined,
        context_text: uploadedContext || undefined,
        agenda_text: uploadedAgenda || undefined,
      });

      setBuiltDatasetId(data.dataset_id || null);
      setDatasetPreview(data.samples_preview || []);
      setDatasetSamplesByStage(data.samples_by_stage || {});
      setBuiltResult({
        dataset_id: data.dataset_id || '',
        total_samples: data.total_samples || 0,
        samples_by_stage: data.samples_by_stage || {},
        samples_preview: data.samples_preview || [],
      });

      toast.success(`Successfully built training dataset with ${data.total_samples} samples!`);
    } catch (err: any) {
      toast.error(err?.response?.data?.detail || 'Failed to build training dataset.');
    } finally {
      setDatasetBuilding(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      {/* 2-Column Main Layout */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem', alignItems: 'stretch' }}>
        {/* Left Column: Manual MoM Upload */}
        <div style={{ display: 'flex', flexDirection: 'column' }}>
          <ManualMomEditor />
        </div>

        {/* Right Column: Input Source */}
        <div style={{
          background: 'hsl(var(--card))',
          borderRadius: '12px',
          padding: '1.25rem 1.5rem',
          border: '1.5px solid hsl(var(--border) / 0.8)',
          display: 'flex',
          flexDirection: 'column',
          gap: '1.25rem',
        }}>
          <div>
            <h2 style={{ fontSize: '1.05rem', fontWeight: 700, margin: '0 0 0.25rem 0', color: 'hsl(var(--foreground))' }}>
              Input Source
            </h2>
            <p style={{ margin: 0, fontSize: '0.8rem', color: 'hsl(var(--muted-foreground))' }}>
              Select how training inputs (transcripts, context, agendas) are provided.
            </p>
          </div>

          {/* Option Selector Cards */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem' }}>
            <div
              onClick={() => setInputSourceType('context')}
              style={{
                padding: '1rem',
                borderRadius: '10px',
                border: `2px solid ${inputSourceType === 'context' ? 'hsl(var(--accent))' : 'hsl(var(--border) / 0.6)'}`,
                background: inputSourceType === 'context' ? 'hsl(var(--accent) / 0.06)' : 'hsl(var(--background))',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'flex-start',
                gap: '0.75rem',
                transition: 'all 0.15s ease',
              }}
            >
              <div style={{ padding: '0.4rem', borderRadius: '8px', background: 'hsl(var(--primary) / 0.1)', color: 'hsl(var(--primary))' }}>
                <FileText size={18} />
              </div>
              <div>
                <h4 style={{ margin: 0, fontWeight: 700, fontSize: '0.88rem', color: 'hsl(var(--foreground))' }}>
                  1. Upload Context Files
                </h4>
                <p style={{ margin: '3px 0 0', fontSize: '0.75rem', color: 'hsl(var(--muted-foreground))', lineHeight: '1.35' }}>
                  Upload transcription context, agenda files & meeting context.
                </p>
              </div>
            </div>

            <div
              onClick={() => setInputSourceType('meeting')}
              style={{
                padding: '1rem',
                borderRadius: '10px',
                border: `2px solid ${inputSourceType === 'meeting' ? 'hsl(var(--accent))' : 'hsl(var(--border) / 0.6)'}`,
                background: inputSourceType === 'meeting' ? 'hsl(var(--accent) / 0.06)' : 'hsl(var(--background))',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'flex-start',
                gap: '0.75rem',
                transition: 'all 0.15s ease',
              }}
            >
              <div style={{ padding: '0.4rem', borderRadius: '8px', background: 'hsl(var(--primary) / 0.1)', color: 'hsl(var(--primary))' }}>
                <Database size={18} />
              </div>
              <div>
                <h4 style={{ margin: 0, fontWeight: 700, fontSize: '0.88rem', color: 'hsl(var(--foreground))' }}>
                  2. Select Meeting
                </h4>
                <p style={{ margin: '3px 0 0', fontSize: '0.75rem', color: 'hsl(var(--muted-foreground))', lineHeight: '1.35' }}>
                  Select an existing processed meeting as training input.
                </p>
              </div>
            </div>
          </div>

          {/* Active Input Source View */}
          <div style={{
            padding: '1rem',
            background: 'hsl(var(--background))',
            borderRadius: '10px',
            border: '1px solid hsl(var(--border) / 0.5)',
            flex: 1,
          }}>
            {inputSourceType === 'meeting' ? <MeetingSelector /> : <FileUploadPanel />}
          </div>
        </div>
      </div>

      {/* Build Dataset Action Bar */}
      <div style={{
        background: 'linear-gradient(135deg, hsl(var(--card)) 0%, hsl(var(--background)) 100%)',
        borderRadius: '12px',
        padding: '1.25rem 1.5rem',
        border: '1.5px solid hsl(var(--border) / 0.8)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: '1rem',
      }}>
        <div>
          <div style={{ fontSize: '0.95rem', fontWeight: 700, color: 'hsl(var(--foreground))' }}>
            Ready to generate training dataset
          </div>
          <div style={{ fontSize: '0.78rem', color: 'hsl(var(--muted-foreground))', marginTop: '2px' }}>
            Stages selected: {selectedStages.map((s) => s.replace('_', ' ')).join(', ')} ·
            Input Source: {inputSourceType === 'meeting' ? 'Existing Meeting' : 'Context Files'}
            {useEditedStage2 && ' · Stage 2 Target: Edited Version'}
          </div>
        </div>

        <button
          onClick={handleBuildDataset}
          disabled={datasetBuilding}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            padding: '0.75rem 1.75rem',
            borderRadius: '10px',
            background: 'linear-gradient(135deg, hsl(var(--accent)), hsl(262 80% 65%))',
            color: 'white',
            border: 'none',
            fontWeight: 800,
            fontSize: '0.9rem',
            cursor: datasetBuilding ? 'not-allowed' : 'pointer',
            opacity: datasetBuilding ? 0.7 : 1,
            boxShadow: '0 0 20px hsl(var(--accent) / 0.35)',
            transition: 'all 0.15s ease',
          }}
        >
          {datasetBuilding ? <Loader2 size={18} className="spin" /> : <Sparkles size={18} />}
          {datasetBuilding ? 'Building Dataset...' : 'Build Training Dataset'}
        </button>
      </div>

      {/* Dataset Generation Result & Samples Preview */}
      {builtResult && (
        <div style={{
          background: 'hsl(var(--card))',
          borderRadius: '12px',
          border: '1.5px solid hsl(140 65% 45% / 0.4)',
          padding: '1.25rem',
          display: 'flex',
          flexDirection: 'column',
          gap: '1rem',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <CheckCircle2 size={20} style={{ color: 'hsl(140 65% 45%)' }} />
              <span style={{ fontSize: '1rem', fontWeight: 700, color: 'hsl(140 65% 40%)' }}>
                Dataset Built Successfully! ({builtResult.total_samples} samples)
              </span>
            </div>
            <span style={{ fontSize: '0.78rem', color: 'hsl(var(--muted-foreground))' }}>
              Dataset ID: {builtResult.dataset_id.substring(0, 8)}...
            </span>
          </div>

          <div style={{ display: 'flex', gap: '1rem', fontSize: '0.8rem' }}>
            {Object.entries(builtResult.samples_by_stage).map(([st, cnt]) => (
              <span key={st} style={{ padding: '4px 10px', borderRadius: '12px', background: 'hsl(var(--background))', border: '1px solid hsl(var(--border))', fontWeight: 600 }}>
                {st.replace('_', ' ')}: {cnt} samples
              </span>
            ))}
          </div>

          {builtResult.samples_preview?.length > 0 && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              <span style={{ fontSize: '0.8rem', fontWeight: 700, color: 'hsl(var(--foreground))' }}>Sample Target Preview:</span>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem', maxHeight: '180px', overflowY: 'auto' }}>
                {builtResult.samples_preview.slice(0, 3).map((sample: any, idx: number) => (
                  <div key={idx} style={{ padding: '0.6rem 0.8rem', borderRadius: '8px', background: 'hsl(var(--background))', border: '1px solid hsl(var(--border) / 0.5)', fontSize: '0.75rem' }}>
                    <div style={{ fontWeight: 700, color: 'hsl(var(--accent))', marginBottom: '2px' }}>
                      Stage: {sample.stage} | ID: {sample.sample_id?.substring(0, 8)} {sample.metadata?.target_version && `[Target: ${sample.metadata.target_version}]`}
                    </div>
                    <div style={{ color: 'hsl(var(--foreground))', fontFamily: 'monospace' }}>
                      Target: {typeof sample.target === 'string' ? sample.target.substring(0, 150) : JSON.stringify(sample.target).substring(0, 150)}...
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
