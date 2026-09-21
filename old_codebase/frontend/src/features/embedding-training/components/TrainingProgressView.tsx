import React, { useState, useEffect, useRef } from 'react';
import {
  Brain, Play, XCircle, CheckCircle2, AlertCircle,
  Terminal, ArrowLeft, ArrowRight, Activity, Clock,
  Layers, Sparkles, TrendingDown
} from 'lucide-react';
import {
  ExtractionResult,
  TrainingJobStatus,
  embeddingTrainingApi
} from '../../../services/embeddingTrainingApi';

interface TrainingProgressViewProps {
  baseModelName: string;
  extractionResult: ExtractionResult;
  onTrainingFinished: (jobStatus: TrainingJobStatus) => void;
  onBack: () => void;
  onNext: () => void;
}

export default function TrainingProgressView({
  baseModelName,
  extractionResult,
  onTrainingFinished,
  onBack,
  onNext,
}: TrainingProgressViewProps) {
  // Configuration
  const [variantName, setVariantName] = useState('ADA-Technical-v1');
  const [epochs, setEpochs] = useState(3);
  const [batchSize, setBatchSize] = useState(4);
  const [learningRate, setLearningRate] = useState(0.00002);

  // Job State
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<TrainingJobStatus | null>(null);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [cancelling, setCancelling] = useState(false);
  const [showLogs, setShowLogs] = useState(true);

  const logsEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll logs to bottom
  useEffect(() => {
    if (showLogs && logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [jobStatus?.logs, showLogs]);

  // Polling loop for active training job
  useEffect(() => {
    if (!jobId) return;

    let timer: NodeJS.Timeout;
    const pollStatus = async () => {
      try {
        const data = await embeddingTrainingApi.getTrainingStatus(jobId);
        setJobStatus(data);

        if (data.status === 'completed') {
          onTrainingFinished(data);
        } else if (data.status === 'failed') {
          setError(data.error || 'Training failed on server.');
        } else if (data.status === 'cancelled') {
          setError('Training was cancelled.');
        } else {
          // Continue polling while in progress
          timer = setTimeout(pollStatus, 1500);
        }
      } catch (err: any) {
        console.error('Error polling training status:', err);
        timer = setTimeout(pollStatus, 3000);
      }
    };

    pollStatus();

    return () => {
      clearTimeout(timer);
    };
  }, [jobId]);

  const handleStartTraining = async () => {
    if (!variantName.trim()) {
      setError('Please provide a variant name.');
      return;
    }

    setStarting(true);
    setError(null);

    try {
      const res = await embeddingTrainingApi.startTraining({
        base_model_name: baseModelName,
        variant_name: variantName.trim(),
        session_id: extractionResult.session_id,
        epochs,
        batch_size: batchSize,
        learning_rate: learningRate,
      });
      setJobId(res.job_id);
    } catch (err: any) {
      setError(err?.response?.data?.detail || err?.message || 'Failed to start domain training.');
    } finally {
      setStarting(false);
    }
  };

  const handleCancelTraining = async () => {
    if (!jobId) return;
    setCancelling(true);
    try {
      await embeddingTrainingApi.cancelTraining(jobId);
    } catch (err: any) {
      console.error('Error cancelling training:', err);
    } finally {
      setCancelling(false);
    }
  };

  const isTrainingActive = jobStatus && (jobStatus.status === 'training' || jobStatus.status === 'initializing');
  const isCompleted = jobStatus?.status === 'completed';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      <div>
        <h2 style={{ fontSize: '1.15rem', fontWeight: 700, margin: '0 0 0.4rem', color: 'hsl(var(--foreground))' }}>
          Step 4: Embedding Domain Training
        </h2>
        <p style={{ fontSize: '0.85rem', color: 'hsl(var(--muted-foreground))', margin: 0 }}>
          Unsupervised domain adaptation aligns the embedding space with ADA technical vocabulary, acronyms, and recurring project relations.
        </p>
      </div>

      {/* Configuration Section (shown before or during training) */}
      {!jobId && (
        <div style={{
          background: 'hsl(var(--card))', borderRadius: '16px',
          border: '1px solid hsl(var(--border))', padding: '1.5rem',
          display: 'flex', flexDirection: 'column', gap: '1.25rem',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <div style={{
              width: 38, height: 38, borderRadius: '10px',
              background: 'linear-gradient(135deg, hsl(210 100% 60%) 0%, hsl(var(--accent)) 100%)',
              display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'white',
            }}>
              <Brain size={20} />
            </div>
            <div>
              <h3 style={{ margin: 0, fontSize: '1rem', fontWeight: 700, color: 'hsl(var(--foreground))' }}>
                Training Configuration
              </h3>
              <p style={{ margin: 0, fontSize: '0.78rem', color: 'hsl(var(--muted-foreground))' }}>
                Corpus: {extractionResult.successful_documents} Documents ({extractionResult.total_chunks} Passages) | Base: {baseModelName}
              </p>
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '1rem' }}>
            {/* Variant Name */}
            <div>
              <label style={{ display: 'block', fontSize: '0.8rem', fontWeight: 600, marginBottom: '0.4rem', color: 'hsl(var(--foreground))' }}>
                Variant Name <span style={{ color: 'hsl(var(--accent))' }}>*</span>
              </label>
              <input
                type="text"
                value={variantName}
                onChange={(e) => setVariantName(e.target.value)}
                placeholder="e.g. ADA-Technical-v1"
                style={{
                  width: '100%', padding: '0.6rem 0.85rem', borderRadius: '8px',
                  border: '1px solid hsl(var(--border))', background: 'hsl(var(--background))',
                  color: 'hsl(var(--foreground))', fontSize: '0.85rem', fontWeight: 600,
                  outline: 'none', fontFamily: 'Inter, sans-serif',
                }}
              />
              <span style={{ fontSize: '0.72rem', color: 'hsl(var(--muted-foreground))', marginTop: '4px', display: 'block' }}>
                Saved as a separate model variant. Base model will never be overwritten.
              </span>
            </div>

            {/* Epochs */}
            <div>
              <label style={{ display: 'block', fontSize: '0.8rem', fontWeight: 600, marginBottom: '0.4rem', color: 'hsl(var(--foreground))' }}>
                Training Epochs: <span style={{ color: 'hsl(var(--accent))' }}>{epochs}</span>
              </label>
              <input
                type="range"
                min={1}
                max={10}
                value={epochs}
                onChange={(e) => setEpochs(Number(e.target.value))}
                style={{ width: '100%', accentColor: 'hsl(var(--accent))' }}
              />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', color: 'hsl(var(--muted-foreground))' }}>
                <span>1 (Fast)</span>
                <span>3 (Recommended)</span>
                <span>10 (Deep)</span>
              </div>
            </div>

            {/* Batch Size */}
            <div>
              <label style={{ display: 'block', fontSize: '0.8rem', fontWeight: 600, marginBottom: '0.4rem', color: 'hsl(var(--foreground))' }}>
                Batch Size: <span style={{ color: 'hsl(var(--accent))' }}>{batchSize}</span>
              </label>
              <select
                value={batchSize}
                onChange={(e) => setBatchSize(Number(e.target.value))}
                style={{
                  width: '100%', padding: '0.6rem 0.85rem', borderRadius: '8px',
                  border: '1px solid hsl(var(--border))', background: 'hsl(var(--background))',
                  color: 'hsl(var(--foreground))', fontSize: '0.85rem', outline: 'none',
                }}
              >
                <option value={2}>2 (Minimal Memory)</option>
                <option value={4}>4 (Balanced)</option>
                <option value={8}>8 (High Throughput)</option>
                <option value={16}>16 (GPU 16GB+)</option>
              </select>
            </div>

            {/* Learning Rate */}
            <div>
              <label style={{ display: 'block', fontSize: '0.8rem', fontWeight: 600, marginBottom: '0.4rem', color: 'hsl(var(--foreground))' }}>
                Learning Rate
              </label>
              <select
                value={learningRate}
                onChange={(e) => setLearningRate(Number(e.target.value))}
                style={{
                  width: '100%', padding: '0.6rem 0.85rem', borderRadius: '8px',
                  border: '1px solid hsl(var(--border))', background: 'hsl(var(--background))',
                  color: 'hsl(var(--foreground))', fontSize: '0.85rem', outline: 'none',
                }}
              >
                <option value={0.00001}>1e-5 (Conservative)</option>
                <option value={0.00002}>2e-5 (Standard SimCSE)</option>
                <option value={0.00005}>5e-5 (Aggressive)</option>
              </select>
            </div>
          </div>

          <div style={{
            background: 'hsl(var(--accent) / 0.07)', border: '1px solid hsl(var(--accent) / 0.2)',
            borderRadius: '10px', padding: '0.85rem 1rem', display: 'flex', alignItems: 'center', gap: '10px',
          }}>
            <Sparkles size={18} style={{ color: 'hsl(var(--accent))', flexShrink: 0 }} />
            <span style={{ fontSize: '0.8rem', color: 'hsl(var(--foreground))', lineHeight: 1.4 }}>
              <strong>Unsupervised Domain Adaptation:</strong> The model adapts to ADA terms, acronyms, and semantic relationships directly from your document corpus using contrastive representation learning.
            </span>
          </div>

          <button
            onClick={handleStartTraining}
            disabled={starting}
            style={{
              display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px',
              background: 'hsl(var(--accent))', color: 'white', border: 'none',
              padding: '0.8rem', borderRadius: '10px', fontWeight: 700, fontSize: '0.92rem',
              cursor: starting ? 'not-allowed' : 'pointer', marginTop: '0.5rem',
              boxShadow: '0 4px 15px hsl(var(--accent) / 0.3)',
            }}
          >
            {starting ? (
              <>
                <div className="spin" style={{
                  width: 16, height: 16, border: '2px solid white',
                  borderTop: '2px solid transparent', borderRadius: '50%',
                }} />
                Initializing Training...
              </>
            ) : (
              <>
                <Play size={16} />
                Start Domain Training
              </>
            )}
          </button>
        </div>
      )}

      {error && (
        <div style={{
          display: 'flex', alignItems: 'center', gap: '10px',
          background: 'hsl(0 80% 50% / 0.1)', color: 'hsl(0 80% 65%)',
          border: '1px solid hsl(0 80% 50% / 0.25)', padding: '1rem',
          borderRadius: '12px', fontSize: '0.875rem',
        }}>
          <AlertCircle size={18} />
          <div>{error}</div>
        </div>
      )}

      {/* Live Training Dashboard */}
      {jobStatus && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          {/* Header & Status */}
          <div style={{
            background: 'hsl(var(--card))', borderRadius: '14px',
            border: '1px solid hsl(var(--border))', padding: '1.25rem 1.5rem',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <h3 style={{ margin: 0, fontSize: '1.1rem', fontWeight: 800, color: 'hsl(var(--foreground))' }}>
                    {jobStatus.variant_name}
                  </h3>
                  {isTrainingActive && (
                    <span style={{
                      display: 'inline-flex', alignItems: 'center', gap: '5px',
                      background: 'hsl(210 100% 50% / 0.15)', color: 'hsl(210 100% 60%)',
                      padding: '2px 8px', borderRadius: '20px', fontSize: '0.72rem', fontWeight: 700,
                    }}>
                      <span style={{
                        width: 7, height: 7, borderRadius: '50%', background: 'hsl(210 100% 60%)',
                        display: 'inline-block', animation: 'pulse 1.5s infinite',
                      }} />
                      Training Active
                    </span>
                  )}
                  {isCompleted && (
                    <span style={{
                      display: 'inline-flex', alignItems: 'center', gap: '5px',
                      background: 'hsl(150 75% 45% / 0.15)', color: 'hsl(150 75% 45%)',
                      padding: '2px 8px', borderRadius: '20px', fontSize: '0.72rem', fontWeight: 700,
                    }}>
                      <CheckCircle2 size={12} /> Completed
                    </span>
                  )}
                </div>
                <div style={{ fontSize: '0.78rem', color: 'hsl(var(--muted-foreground))', marginTop: '2px' }}>
                  Base Model: {jobStatus.base_model_name}
                </div>
              </div>

              {isTrainingActive && (
                <button
                  onClick={handleCancelTraining}
                  disabled={cancelling}
                  style={{
                    display: 'flex', alignItems: 'center', gap: '6px',
                    background: 'hsl(0 80% 50% / 0.12)', color: 'hsl(0 80% 65%)',
                    border: '1px solid hsl(0 80% 50% / 0.3)', padding: '0.45rem 0.9rem',
                    borderRadius: '8px', fontSize: '0.8rem', fontWeight: 600, cursor: 'pointer',
                  }}
                >
                  <XCircle size={14} />
                  {cancelling ? 'Cancelling...' : 'Cancel Training'}
                </button>
              )}
            </div>

            {/* Progress bar */}
            <div style={{ marginBottom: '1rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', fontWeight: 600, marginBottom: '6px' }}>
                <span style={{ color: 'hsl(var(--foreground))' }}>
                  {jobStatus.status === 'completed'
                    ? 'Training Complete'
                    : `Epoch ${jobStatus.current_epoch} of ${jobStatus.total_epochs} (Step ${jobStatus.current_step}/${jobStatus.total_steps})`}
                </span>
                <span style={{ color: 'hsl(var(--accent))' }}>
                  {jobStatus.progress_percent}%
                </span>
              </div>
              <div style={{
                width: '100%', height: '8px', background: 'hsl(var(--muted) / 0.5)',
                borderRadius: '10px', overflow: 'hidden',
              }}>
                <div style={{
                  width: `${jobStatus.progress_percent}%`,
                  height: '100%',
                  background: isCompleted ? 'hsl(150 75% 45%)' : 'hsl(var(--accent))',
                  borderRadius: '10px',
                  transition: 'width 0.3s ease',
                }} />
              </div>
            </div>

            {/* Key Metrics */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: '0.75rem' }}>
              <div style={{
                background: 'hsl(var(--background))', padding: '0.75rem',
                borderRadius: '8px', border: '1px solid hsl(var(--border) / 0.7)',
              }}>
                <div style={{ fontSize: '0.72rem', color: 'hsl(var(--muted-foreground))', display: 'flex', alignItems: 'center', gap: '4px' }}>
                  <TrendingDown size={12} /> Current Loss
                </div>
                <div style={{ fontSize: '1.15rem', fontWeight: 800, color: 'hsl(var(--foreground))', marginTop: '2px' }}>
                  {jobStatus.current_loss !== null ? jobStatus.current_loss : '—'}
                </div>
              </div>

              {jobStatus.final_loss !== null && (
                <div style={{
                  background: 'hsl(var(--background))', padding: '0.75rem',
                  borderRadius: '8px', border: '1px solid hsl(var(--border) / 0.7)',
                }}>
                  <div style={{ fontSize: '0.72rem', color: 'hsl(var(--muted-foreground))' }}>
                    Final Average Loss
                  </div>
                  <div style={{ fontSize: '1.15rem', fontWeight: 800, color: 'hsl(150 75% 45%)', marginTop: '2px' }}>
                    {jobStatus.final_loss}
                  </div>
                </div>
              )}

              <div style={{
                background: 'hsl(var(--background))', padding: '0.75rem',
                borderRadius: '8px', border: '1px solid hsl(var(--border) / 0.7)',
              }}>
                <div style={{ fontSize: '0.72rem', color: 'hsl(var(--muted-foreground))', display: 'flex', alignItems: 'center', gap: '4px' }}>
                  <Clock size={12} /> Elapsed
                </div>
                <div style={{ fontSize: '1.15rem', fontWeight: 800, color: 'hsl(var(--foreground))', marginTop: '2px' }}>
                  {jobStatus.elapsed_seconds}s
                </div>
              </div>

              {isTrainingActive && (
                <div style={{
                  background: 'hsl(var(--background))', padding: '0.75rem',
                  borderRadius: '8px', border: '1px solid hsl(var(--border) / 0.7)',
                }}>
                  <div style={{ fontSize: '0.72rem', color: 'hsl(var(--muted-foreground))' }}>
                    Estimated ETA
                  </div>
                  <div style={{ fontSize: '1.15rem', fontWeight: 800, color: 'hsl(var(--accent))', marginTop: '2px' }}>
                    {jobStatus.eta_seconds}s
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Real-Time Training Log Terminal */}
          <div style={{
            background: '#0d1117', borderRadius: '12px',
            border: '1px solid #30363d', overflow: 'hidden',
          }}>
            <div style={{
              display: 'flex', alignItems: 'center', justifyContent: 'space-between',
              padding: '0.65rem 1rem', background: '#161b22', borderBottom: '1px solid #30363d',
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#8b949e', fontSize: '0.78rem' }}>
                <Terminal size={14} />
                <span style={{ fontWeight: 600, color: '#c9d1d9' }}>Training Execution Logs</span>
                <span>({jobStatus.logs.length} lines)</span>
              </div>
              <button
                onClick={() => setShowLogs(!showLogs)}
                style={{
                  background: 'transparent', border: 'none', color: '#8b949e',
                  fontSize: '0.75rem', cursor: 'pointer',
                }}
              >
                {showLogs ? 'Collapse' : 'Expand'}
              </button>
            </div>

            {showLogs && (
              <div style={{
                maxHeight: '220px', overflowY: 'auto', padding: '0.85rem 1rem',
                fontFamily: 'SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace',
                fontSize: '0.78rem', color: '#58a6ff', lineHeight: 1.6,
              }}>
                {jobStatus.logs.map((log, index) => (
                  <div key={index} style={{
                    color: log.includes('ERROR')
                      ? '#f85149'
                      : log.includes('complete') || log.includes('successfully')
                      ? '#3fb950'
                      : '#c9d1d9',
                  }}>
                    {log}
                  </div>
                ))}
                <div ref={logsEndRef} />
              </div>
            )}
          </div>
        </div>
      )}

      {/* Navigation buttons */}
      <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '1rem' }}>
        <button
          onClick={onBack}
          disabled={isTrainingActive}
          style={{
            display: 'flex', alignItems: 'center', gap: '8px',
            background: 'hsl(var(--muted) / 0.5)', color: 'hsl(var(--foreground))',
            border: '1px solid hsl(var(--border))', padding: '0.65rem 1.25rem',
            borderRadius: '8px', fontWeight: 600, fontSize: '0.875rem',
            cursor: isTrainingActive ? 'not-allowed' : 'pointer',
            opacity: isTrainingActive ? 0.5 : 1,
          }}
        >
          <ArrowLeft size={16} />
          Back: Extraction
        </button>

        <button
          onClick={onNext}
          disabled={!isCompleted}
          style={{
            display: 'flex', alignItems: 'center', gap: '8px',
            background: isCompleted ? 'hsl(var(--accent))' : 'hsl(var(--muted))',
            color: isCompleted ? 'white' : 'hsl(var(--muted-foreground))',
            border: 'none', padding: '0.65rem 1.5rem', borderRadius: '8px',
            fontWeight: 600, fontSize: '0.875rem',
            cursor: isCompleted ? 'pointer' : 'not-allowed',
          }}
        >
          Next: Save Variant
          <ArrowRight size={16} />
        </button>
      </div>
    </div>
  );
}
