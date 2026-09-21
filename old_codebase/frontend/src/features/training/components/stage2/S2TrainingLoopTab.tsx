import { useState, useEffect, useRef } from 'react';
import {
  Play, RefreshCw, ChevronRight, ChevronDown, CheckCircle, XCircle,
  AlertCircle, Sparkles, MessageSquare, Send, FileText, Copy, Check,
  Layers, User, CheckSquare, CornerDownRight, ArrowRight, Zap, FolderOpen,
  Sliders, ShieldCheck, ShieldAlert, Cpu
} from 'lucide-react';
import { useStage2Store } from '../../store/stage2Store';
import {
  startStage2Optimization,
  getStage2RunStatus,
  runStage2Validation,
  submitStage2Feedback,
  submitStage2FeedbackAndRetrain,
  listStage2Variants,
  previewStage2Groups,
  fetchStage1PointsForMeeting,
  validateStage2ModelPath,
} from '../../api/stage2Api';
import {
  STAGE2_FEEDBACK_LABELS,
  STAGE2_METRIC_LABELS,
  STAGE2_METRIC_ORDER,
} from '../../types/stage2Types';
import type { Stage2FeedbackCategory, Stage2Group } from '../../types/stage2Types';

const S = {
  card: {
    background: 'hsl(var(--card))',
    borderRadius: '12px',
    border: '1px solid hsl(var(--border))',
    padding: '1.25rem',
  } as React.CSSProperties,
  sectionHeader: (color = 'hsl(var(--accent))') => ({
    display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '0.9rem',
  } as React.CSSProperties),
  iconBadge: (bg = 'hsl(var(--accent) / .15)') => ({
    width: 28, height: 28, borderRadius: '8px', background: bg,
    display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0,
  } as React.CSSProperties),
  label: {
    display: 'block', fontSize: '0.75rem', fontWeight: 600,
    color: 'hsl(var(--muted-foreground))', marginBottom: '0.35rem',
    textTransform: 'uppercase' as const, letterSpacing: '0.04em',
  },
  select: {
    width: '100%', padding: '0.55rem 0.8rem', borderRadius: '8px',
    border: '1px solid hsl(var(--border))', background: 'hsl(var(--background))',
    color: 'hsl(var(--foreground))', fontSize: '0.88rem', outline: 'none',
  } as React.CSSProperties,
  input: {
    width: '100%', padding: '0.55rem 0.8rem', borderRadius: '8px',
    border: '1px solid hsl(var(--border))', background: 'hsl(var(--background))',
    color: 'hsl(var(--foreground))', fontSize: '0.88rem', outline: 'none',
    boxSizing: 'border-box' as const,
  } as React.CSSProperties,
  btn: (variant: 'primary' | 'ghost' | 'danger' | 'success' | 'secondary' = 'primary', disabled = false) => ({
    display: 'inline-flex', alignItems: 'center', gap: '6px',
    padding: '0.55rem 1.1rem', borderRadius: '8px', border: 'none',
    cursor: disabled ? 'not-allowed' : 'pointer',
    fontSize: '0.85rem', fontWeight: 600, transition: 'all 0.15s ease',
    opacity: disabled ? 0.6 : 1,
    background:
      variant === 'primary' ? 'hsl(var(--accent))' :
      variant === 'success' ? 'hsl(142 76% 36%)' :
      variant === 'danger'  ? 'hsl(0 75% 55%)' :
      variant === 'secondary' ? 'hsl(var(--secondary))' :
      'hsl(var(--muted))',
    color: variant === 'ghost' ? 'hsl(var(--foreground))' : 'white',
  } as React.CSSProperties),
};

function ScoreBar({ value, label }: { value: number; label: string }) {
  const pct = Math.round(value);
  const color = pct >= 80 ? 'hsl(142 71% 45%)' : pct >= 60 ? 'hsl(45 93% 47%)' : 'hsl(0 75% 55%)';
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.78rem' }}>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{
          color: 'hsl(var(--muted-foreground))', marginBottom: '3px',
          whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis'
        }}>
          {label}
        </div>
        <div style={{ height: '6px', borderRadius: '3px', background: 'hsl(var(--muted))', overflow: 'hidden' }}>
          <div style={{ height: '100%', width: `${Math.min(pct, 100)}%`, borderRadius: '3px', background: color, transition: 'width 0.5s ease' }} />
        </div>
      </div>
      <div style={{ fontWeight: 700, color, minWidth: '38px', textAlign: 'right' }}>{pct}%</div>
    </div>
  );
}

export default function S2TrainingLoopTab() {
  const {
    groups, settings, setSettings, variants, setVariants,
    currentRunId, setCurrentRunId, currentRunStatus, setCurrentRunStatus,
    trainingLogs, appendTrainingLog, clearTrainingLogs,
    meetings,
    datasetMeetingId, referencePoints, referenceSource,
    hfModelPathValidation, setHfModelPathValidation,
    hfModelPathValidating, setHfModelPathValidating,
    valMeetingId, setValMeetingId,
    valGroupIndex, setValGroupIndex,
    valGroups, setValGroups,
    valVariantId, setValVariantId,
    validationOutput, setValidationOutput, validationLoading, setValidationLoading,
    feedbackCategories, toggleFeedbackCategory, feedbackComment, setFeedbackComment,
    feedbackSubmitting, setFeedbackSubmitting, feedbackSubmitted, setFeedbackSubmitted,
    clearFeedback,
  } = useStage2Store();

  const [showRawJson, setShowRawJson] = useState(false);
  const [jsonCopied, setJsonCopied] = useState(false);
  const [showFeedback, setShowFeedback] = useState(false);
  const [loadingValGroups, setLoadingValGroups] = useState(false);
  const [showAdvancedParams, setShowAdvancedParams] = useState(false);
  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const isLoRAorQLoRA = settings.training_mode === 'lora' || settings.training_mode === 'qlora';
  const isRunning = currentRunStatus?.status === 'running' || currentRunStatus?.status === 'starting';
  const isDone = currentRunStatus?.status === 'done';
  const isError = currentRunStatus?.status === 'error';

  // Poll optimization run status
  useEffect(() => {
    if (currentRunId && isRunning) {
      pollingRef.current = setInterval(async () => {
        try {
          const status = await getStage2RunStatus(currentRunId);
          setCurrentRunStatus(status);
          if (status.message) appendTrainingLog(status.message);
          if (status.status === 'done' || status.status === 'error') {
            clearInterval(pollingRef.current!);
            const freshVariants = await listStage2Variants();
            setVariants(freshVariants.variants || []);
          }
        } catch (e) {
          clearInterval(pollingRef.current!);
        }
      }, 2000);
    }
    return () => { if (pollingRef.current) clearInterval(pollingRef.current); };
  }, [currentRunId, isRunning]);

  // Set default validation meeting if not set
  useEffect(() => {
    if (!valMeetingId && datasetMeetingId) {
      setValMeetingId(datasetMeetingId);
      setValGroups(groups);
    }
  }, [datasetMeetingId, groups]);

  const handleValidateModel = async () => {
    if (!settings.hf_model_path || !settings.hf_model_path.trim()) {
      setHfModelPathValidation({
        valid: false,
        reason: 'Please enter a local Hugging Face model directory path.',
        warnings: [],
      });
      return;
    }
    setHfModelPathValidating(true);
    try {
      const res = await validateStage2ModelPath({
        hf_model_path: settings.hf_model_path,
        method: settings.training_mode || 'lora',
      });
      setHfModelPathValidation(res);
    } catch (e: any) {
      setHfModelPathValidation({
        valid: false,
        reason: e?.response?.data?.detail || e?.message || 'Model path validation failed',
        warnings: [],
      });
    } finally {
      setHfModelPathValidating(false);
    }
  };

  const handleStartTraining = async () => {
    if (!groups.length) return;

    if (isLoRAorQLoRA) {
      if (!settings.hf_model_path || !settings.hf_model_path.trim()) {
        appendTrainingLog('Error: A local Hugging Face model directory path is required for LoRA/QLoRA training.');
        return;
      }
      // Trigger validation if not done
      if (!hfModelPathValidation || !hfModelPathValidation.valid) {
        setHfModelPathValidating(true);
        try {
          const valRes = await validateStage2ModelPath({
            hf_model_path: settings.hf_model_path,
            method: settings.training_mode || 'lora',
          });
          setHfModelPathValidation(valRes);
          if (!valRes.valid) {
            appendTrainingLog(`Model validation failed: ${valRes.reason}`);
            setHfModelPathValidating(false);
            return;
          }
        } catch (e: any) {
          appendTrainingLog(`Model validation error: ${e?.message || e}`);
          setHfModelPathValidating(false);
          return;
        }
        setHfModelPathValidating(false);
      }
    }

    clearTrainingLogs();
    setCurrentRunStatus(null);
    try {
      const modeLabel = settings.training_mode === 'qlora' ? 'QLoRA' : settings.training_mode === 'lora' ? 'LoRA' : 'DSPy';
      appendTrainingLog(`Starting Stage 2 ${modeLabel} training...`);
      const res = await startStage2Optimization({
        meeting_id: datasetMeetingId || '',
        groups,
        reference_points: referencePoints,
        settings,
        parent_variant_id: 'default',
      });
      setCurrentRunId(res.run_id);
      setCurrentRunStatus({
        run_id: res.run_id,
        status: 'starting',
        progress: 0,
        message: `Initializing Stage 2 ${modeLabel} training...`,
        variant_id: null,
        error: null,
      });
    } catch (err: any) {
      appendTrainingLog(`Error: ${err?.message || 'Unknown error'}`);
    }
  };

  const handleLoadValGroups = async (meetingId: string) => {
    setLoadingValGroups(true);
    try {
      const data = await fetchStage1PointsForMeeting(meetingId);
      if (data.stage1_points && data.stage1_points.length > 0) {
        const groupRes = await previewStage2Groups({
          meeting_id: meetingId,
          points_per_group: settings.points_per_group || 5,
          context_retrieval: settings.context_retrieval !== false,
        });
        setValGroups(groupRes.groups || []);
        setValGroupIndex(0);
      }
    } catch (e) {
      console.error(e);
    } finally {
      setLoadingValGroups(false);
    }
  };

  const handleRunValidation = async () => {
    const activeGroup = valGroups[valGroupIndex];
    if (!activeGroup) return;
    setValidationLoading(true);
    setValidationOutput(null);
    clearFeedback();
    setShowFeedback(false);
    try {
      const res = await runStage2Validation({
        variant_id: valVariantId,
        meeting_id: valMeetingId || '',
        group_index: valGroupIndex,
        stage1_points: activeGroup.points,
        global_context: activeGroup.global_context,
        meeting_context: activeGroup.meeting_context,
        reference_points: referencePoints,
      });
      setValidationOutput(res);
    } catch (err: any) {
      console.error('Validation error:', err);
    } finally {
      setValidationLoading(false);
    }
  };

  const handleSubmitFeedbackOnly = async () => {
    if (!validationOutput) return;
    const activeGroup = valGroups[valGroupIndex];
    setFeedbackSubmitting(true);
    try {
      await submitStage2Feedback({
        meeting_id: valMeetingId || '',
        group_index: valGroupIndex,
        variant_id: valVariantId,
        stage1_points_input: JSON.stringify(activeGroup?.points || []),
        model_output: validationOutput.raw_output,
        categories: feedbackCategories,
        comment: feedbackComment,
        global_context: activeGroup?.global_context,
        meeting_context: activeGroup?.meeting_context,
      });
      setFeedbackSubmitted(true);
    } catch (err: any) {
      console.error(err);
    } finally {
      setFeedbackSubmitting(false);
    }
  };

  const handleSubmitFeedbackAndRetrain = async () => {
    if (!validationOutput) return;
    const activeGroup = valGroups[valGroupIndex];
    setFeedbackSubmitting(true);
    try {
      const feedbackPayload = {
        meeting_id: valMeetingId || '',
        group_index: valGroupIndex,
        variant_id: valVariantId,
        stage1_points_input: JSON.stringify(activeGroup?.points || []),
        model_output: validationOutput.raw_output,
        categories: feedbackCategories,
        comment: feedbackComment,
        global_context: activeGroup?.global_context,
        meeting_context: activeGroup?.meeting_context,
      };
      const res = await submitStage2FeedbackAndRetrain({
        feedback: feedbackPayload,
        parent_variant_id: valVariantId,
        meeting_id: valMeetingId || '',
        groups: valGroups.length > 0 ? valGroups : groups,
        reference_points: referencePoints,
        settings,
      });
      setCurrentRunId(res.run_id);
      setCurrentRunStatus({
        run_id: res.run_id,
        status: 'starting',
        progress: 0,
        message: 'Feedback submitted. Starting Stage 2 retraining...',
        variant_id: null,
        error: null,
      });
      setFeedbackSubmitted(true);
    } catch (err: any) {
      console.error(err);
    } finally {
      setFeedbackSubmitting(false);
    }
  };

  const handleCopyJson = () => {
    if (!validationOutput?.raw_output) return;
    navigator.clipboard.writeText(validationOutput.raw_output);
    setJsonCopied(true);
    setTimeout(() => setJsonCopied(false), 2000);
  };

  const currentGroup = valGroups[valGroupIndex] || groups[0];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', width: '100%' }}>
      {/* ── Section 1: Optimization & Fine-Tuning Setup ── */}
      <div style={{ ...S.card, display: 'flex', flexDirection: 'column', gap: '1.1rem' }}>
        <div style={S.sectionHeader()}>
          <div style={S.iconBadge()}>
            <Sparkles size={15} color="hsl(var(--accent))" />
          </div>
          <div>
            <h3 style={{ margin: 0, fontSize: '0.95rem', fontWeight: 700 }}>Stage 2 Training Setup</h3>
            <p style={{ margin: 0, fontSize: '0.76rem', color: 'hsl(var(--muted-foreground))' }}>
              Choose whether to optimize consolidation prompts with DSPy or fine-tune local models using LoRA/QLoRA.
            </p>
          </div>
        </div>

        {/* Training Mode Selector Tabs */}
        <div>
          <label style={S.label}>Training Mode</label>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '8px' }}>
            {[
              { id: 'dspy', title: 'DSPy (Ollama)', sub: 'Prompt optimization & few-shot bootstrapping' },
              { id: 'lora', title: 'LoRA Fine-Tuning', sub: 'Adapter training on local Hugging Face model' },
              { id: 'qlora', title: 'QLoRA (4-bit)', sub: 'Quantized 4-bit PEFT for low VRAM GPUs' },
            ].map((m) => {
              const active = (settings.training_mode || 'dspy') === m.id;
              return (
                <button
                  key={m.id}
                  type="button"
                  onClick={() => {
                    setSettings({ ...settings, training_mode: m.id as any });
                    setHfModelPathValidation(null);
                  }}
                  style={{
                    padding: '0.75rem 0.9rem',
                    borderRadius: '10px',
                    border: `1.5px solid ${active ? 'hsl(var(--accent))' : 'hsl(var(--border))'}`,
                    background: active ? 'hsl(var(--accent) / 0.08)' : 'hsl(var(--background))',
                    cursor: 'pointer',
                    textAlign: 'left',
                    transition: 'all 0.15s ease',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '3px' }}>
                    <span style={{ fontWeight: 700, fontSize: '0.85rem', color: active ? 'hsl(var(--accent))' : 'hsl(var(--foreground))' }}>
                      {m.title}
                    </span>
                    {active && <CheckCircle size={14} color="hsl(var(--accent))" />}
                  </div>
                  <div style={{ fontSize: '0.7rem', color: 'hsl(var(--muted-foreground))', lineHeight: 1.3 }}>
                    {m.sub}
                  </div>
                </button>
              );
            })}
          </div>
        </div>

        {/* LoRA / QLoRA Local Model Configuration & Validation */}
        {isLoRAorQLoRA && (
          <div style={{
            padding: '1rem', borderRadius: '10px',
            background: 'hsl(var(--muted) / 0.25)', border: '1px solid hsl(var(--border))',
            display: 'flex', flexDirection: 'column', gap: '0.85rem',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Zap size={16} color="hsl(var(--accent))" />
              <span style={{ fontWeight: 700, fontSize: '0.85rem', color: 'hsl(var(--foreground))' }}>
                Local Hugging Face Model Required (No Ollama)
              </span>
            </div>

            <div style={{ fontSize: '0.76rem', color: 'hsl(var(--muted-foreground))', lineHeight: 1.4 }}>
              LoRA/QLoRA trains parameter-efficient adapters directly on unquantized base weights.
              Please provide the absolute path to a downloaded Hugging Face model directory containing <code>config.json</code> and weights.
            </div>

            {/* Hugging Face Model Path Input + Validate Button */}
            <div>
              <label style={S.label}>Local Hugging Face Model Directory</label>
              <div style={{ display: 'flex', gap: '8px' }}>
                <div style={{ position: 'relative', flex: 1 }}>
                  <input
                    type="text"
                    placeholder="e.g. C:\models\Qwen2.5-7B-Instruct or /data/models/Llama-3-8B"
                    value={settings.hf_model_path || ''}
                    onChange={(e) => {
                      setSettings({ ...settings, hf_model_path: e.target.value });
                      setHfModelPathValidation(null);
                    }}
                    style={S.input}
                  />
                </div>
                <button
                  type="button"
                  onClick={handleValidateModel}
                  disabled={hfModelPathValidating || !settings.hf_model_path}
                  style={S.btn('primary', hfModelPathValidating || !settings.hf_model_path)}
                >
                  {hfModelPathValidating ? <RefreshCw size={13} className="animate-spin" /> : <ShieldCheck size={14} />}
                  {hfModelPathValidating ? 'Validating...' : 'Validate Model'}
                </button>
              </div>
            </div>

            {/* Model Validation Result Callout */}
            {hfModelPathValidation && (
              <div style={{
                padding: '0.75rem 0.9rem',
                borderRadius: '8px',
                border: `1px solid ${hfModelPathValidation.valid ? 'hsl(142 76% 36% / 0.4)' : 'hsl(0 75% 55% / 0.4)'}`,
                background: hfModelPathValidation.valid ? 'hsl(142 76% 36% / 0.08)' : 'hsl(0 75% 55% / 0.08)',
                display: 'flex', flexDirection: 'column', gap: '4px',
                fontSize: '0.78rem',
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontWeight: 600, color: hfModelPathValidation.valid ? 'hsl(142 76% 36%)' : 'hsl(0 75% 55%)' }}>
                  {hfModelPathValidation.valid ? <CheckCircle size={14} /> : <AlertCircle size={14} />}
                  {hfModelPathValidation.reason}
                </div>
                {hfModelPathValidation.details && (
                  <div style={{ display: 'flex', gap: '12px', fontSize: '0.72rem', color: 'hsl(var(--muted-foreground))', marginTop: '2px' }}>
                    {hfModelPathValidation.details.model_type && (
                      <span>Architecture: <strong>{hfModelPathValidation.details.model_type}</strong></span>
                    )}
                    {hfModelPathValidation.details.weights_count !== undefined && (
                      <span>Weight files: <strong>{hfModelPathValidation.details.weights_count}</strong></span>
                    )}
                    {hfModelPathValidation.details.has_tokenizer !== undefined && (
                      <span>Tokenizer: <strong>{hfModelPathValidation.details.has_tokenizer ? 'Found' : 'Missing'}</strong></span>
                    )}
                  </div>
                )}
                {hfModelPathValidation.warnings && hfModelPathValidation.warnings.length > 0 && (
                  <div style={{ color: 'hsl(45 93% 47%)', fontSize: '0.72rem', marginTop: '2px' }}>
                    Warning: {hfModelPathValidation.warnings.join(' | ')}
                  </div>
                )}
              </div>
            )}

            {/* Hyperparameters Toggle & Grid */}
            <div style={{ borderTop: '1px solid hsl(var(--border) / 0.6)', paddingTop: '0.6rem' }}>
              <div
                onClick={() => setShowAdvancedParams(!showAdvancedParams)}
                style={{
                  display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                  cursor: 'pointer', fontSize: '0.78rem', fontWeight: 700, color: 'hsl(var(--accent))',
                }}
              >
                <span style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                  <Sliders size={13} /> {settings.training_mode === 'qlora' ? 'QLoRA' : 'LoRA'} Training Hyperparameters
                </span>
                {showAdvancedParams ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
              </div>

              {showAdvancedParams && (
                <div style={{
                  display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
                  gap: '0.75rem', marginTop: '0.75rem',
                }}>
                  <div>
                    <label style={S.label}>LoRA Rank (r)</label>
                    <select
                      value={settings.lora_r ?? 8}
                      onChange={(e) => setSettings({ ...settings, lora_r: Number(e.target.value) })}
                      style={S.select}
                    >
                      <option value={4}>r = 4 (Lightweight)</option>
                      <option value={8}>r = 8 (Recommended)</option>
                      <option value={16}>r = 16 (High capacity)</option>
                      <option value={32}>r = 32</option>
                      <option value={64}>r = 64</option>
                    </select>
                  </div>

                  <div>
                    <label style={S.label}>LoRA Alpha</label>
                    <select
                      value={settings.lora_alpha ?? 32}
                      onChange={(e) => setSettings({ ...settings, lora_alpha: Number(e.target.value) })}
                      style={S.select}
                    >
                      <option value={16}>α = 16</option>
                      <option value={32}>α = 32 (Recommended)</option>
                      <option value={64}>α = 64</option>
                    </select>
                  </div>

                  <div>
                    <label style={S.label}>Training Epochs</label>
                    <select
                      value={settings.num_train_epochs ?? 3}
                      onChange={(e) => setSettings({ ...settings, num_train_epochs: Number(e.target.value) })}
                      style={S.select}
                    >
                      <option value={1}>1 Epoch</option>
                      <option value={2}>2 Epochs</option>
                      <option value={3}>3 Epochs (Standard)</option>
                      <option value={5}>5 Epochs</option>
                    </select>
                  </div>

                  <div>
                    <label style={S.label}>Learning Rate</label>
                    <select
                      value={settings.learning_rate ?? 0.0002}
                      onChange={(e) => setSettings({ ...settings, learning_rate: Number(e.target.value) })}
                      style={S.select}
                    >
                      <option value={0.0003}>3e-4 (Fast)</option>
                      <option value={0.0002}>2e-4 (Standard)</option>
                      <option value={0.0001}>1e-4</option>
                      <option value={0.00005}>5e-5 (Fine)</option>
                    </select>
                  </div>

                  <div>
                    <label style={S.label}>Gradient Accumulation</label>
                    <select
                      value={settings.gradient_accumulation_steps ?? 4}
                      onChange={(e) => setSettings({ ...settings, gradient_accumulation_steps: Number(e.target.value) })}
                      style={S.select}
                    >
                      <option value={1}>1 Step</option>
                      <option value={2}>2 Steps</option>
                      <option value={4}>4 Steps (Recommended)</option>
                      <option value={8}>8 Steps</option>
                    </select>
                  </div>

                  <div>
                    <label style={S.label}>Max Sequence Length</label>
                    <select
                      value={settings.max_seq_length ?? 2048}
                      onChange={(e) => setSettings({ ...settings, max_seq_length: Number(e.target.value) })}
                      style={S.select}
                    >
                      <option value={1024}>1024 tokens</option>
                      <option value={2048}>2048 tokens</option>
                      <option value={4096}>4096 tokens</option>
                    </select>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Config pill summary */}
        <div style={{
          display: 'flex', flexWrap: 'wrap', gap: '8px',
          padding: '0.75rem 1rem', borderRadius: '8px',
          background: 'hsl(var(--muted) / 0.3)', border: '1px solid hsl(var(--border))',
          fontSize: '0.78rem', color: 'hsl(var(--muted-foreground))',
        }}>
          <span>Mode: <strong style={{ color: 'hsl(var(--foreground))' }}>{settings.training_mode?.toUpperCase() || 'DSPY'}</strong></span>
          <span>•</span>
          <span>Groups: <strong style={{ color: 'hsl(var(--foreground))' }}>{groups.length}</strong></span>
          <span>•</span>
          <span>Batch Size: <strong style={{ color: 'hsl(var(--foreground))' }}>{settings.points_per_group || 5} pts</strong></span>
          <span>•</span>
          <span>Context: <strong style={{ color: 'hsl(var(--foreground))' }}>{settings.context_retrieval ? 'On' : 'Off'}</strong></span>
          <span>•</span>
          {isLoRAorQLoRA ? (
            <span>HF Model: <strong style={{ color: 'hsl(var(--foreground))' }}>{settings.hf_model_path ? settings.hf_model_path.split(/[\\/]/).pop() : 'Not configured'}</strong></span>
          ) : (
            <span>Ollama Model: <strong style={{ color: 'hsl(var(--foreground))' }}>{settings.model_name || 'pipeline default'}</strong></span>
          )}
          <span>•</span>
          <span>Reference: <strong style={{ color: 'hsl(var(--foreground))' }}>{referenceSource} ({referencePoints.length} pts)</strong></span>
        </div>

        {/* Start button & progress */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', flexWrap: 'wrap' }}>
          <button
            onClick={handleStartTraining}
            disabled={
              isRunning || groups.length === 0 || referencePoints.length === 0 ||
              (isLoRAorQLoRA && (!settings.hf_model_path || !settings.hf_model_path.trim()))
            }
            style={S.btn('primary', isRunning || groups.length === 0 || referencePoints.length === 0 || (isLoRAorQLoRA && (!settings.hf_model_path || !settings.hf_model_path.trim())))}
          >
            {isRunning ? <RefreshCw size={14} className="animate-spin" /> : <Play size={14} />}
            {isRunning
              ? `Training ${settings.training_mode?.toUpperCase() || 'Stage 2'}...`
              : `Start Stage 2 ${settings.training_mode === 'qlora' ? 'QLoRA' : settings.training_mode === 'lora' ? 'LoRA' : 'DSPy'} Training`}
          </button>

          {groups.length === 0 && (
            <span style={{ fontSize: '0.78rem', color: 'hsl(var(--muted-foreground))' }}>
              Select a meeting in the <strong>Dataset</strong> tab first.
            </span>
          )}
          {groups.length > 0 && referencePoints.length === 0 && (
            <span style={{ fontSize: '0.78rem', color: 'hsl(0 75% 55%)' }}>
              No reference points available. Upload a MoM in Dataset tab.
            </span>
          )}
          {isLoRAorQLoRA && (!settings.hf_model_path || !settings.hf_model_path.trim()) && (
            <span style={{ fontSize: '0.78rem', color: 'hsl(var(--muted-foreground))' }}>
              Enter and validate a local Hugging Face model directory.
            </span>
          )}
        </div>

        {/* Live progress and status */}
        {currentRunStatus && (
          <div style={{
            padding: '0.9rem', borderRadius: '8px',
            border: `1px solid ${isDone ? 'hsl(142 76% 36% / 0.4)' : isError ? 'hsl(0 75% 55% / 0.4)' : 'hsl(var(--accent) / 0.3)'}`,
            background: isDone ? 'hsl(142 76% 36% / 0.05)' : isError ? 'hsl(0 75% 55% / 0.05)' : 'hsl(var(--accent) / 0.04)',
            display: 'flex', flexDirection: 'column', gap: '0.5rem',
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.82rem', fontWeight: 600 }}>
              <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                {isDone && <CheckCircle size={14} color="hsl(142 76% 36%)" />}
                {isError && <XCircle size={14} color="hsl(0 75% 55%)" />}
                {isRunning && <RefreshCw size={14} color="hsl(var(--accent))" className="animate-spin" />}
                {currentRunStatus.message}
              </span>
              <span style={{ color: 'hsl(var(--muted-foreground))' }}>{currentRunStatus.progress}%</span>
            </div>
            <div style={{ height: '5px', borderRadius: '3px', background: 'hsl(var(--muted))', overflow: 'hidden' }}>
              <div style={{
                height: '100%', width: `${currentRunStatus.progress}%`,
                background: isDone ? 'hsl(142 76% 36%)' : isError ? 'hsl(0 75% 55%)' : 'hsl(var(--accent))',
                transition: 'width 0.3s ease',
              }} />
            </div>
          </div>
        )}

        {/* Logs tail */}
        {trainingLogs.length > 0 && (
          <div style={{
            background: 'hsl(var(--background))',
            borderRadius: '8px',
            border: '1px solid hsl(var(--border))',
            padding: '0.6rem 0.8rem',
            maxHeight: '130px',
            overflowY: 'auto',
            fontFamily: 'monospace',
            fontSize: '0.72rem',
            color: 'hsl(var(--muted-foreground))',
            display: 'flex',
            flexDirection: 'column',
            gap: '2px',
          }}>
            {trainingLogs.map((log, i) => (
              <div key={i}>{log}</div>
            ))}
          </div>
        )}
      </div>

      {/* ── Section 2: Interactive Validation Workspace ── */}
      <div style={{ ...S.card, display: 'flex', flexDirection: 'column', gap: '1.2rem' }}>
        <div style={S.sectionHeader()}>
          <div style={S.iconBadge('hsl(217 91% 60% / 0.15)')}>
            <CheckSquare size={15} color="hsl(217 91% 60%)" />
          </div>
          <div>
            <h3 style={{ margin: 0, fontSize: '0.92rem', fontWeight: 700 }}>Interactive Validation Workspace</h3>
            <p style={{ margin: 0, fontSize: '0.76rem', color: 'hsl(var(--muted-foreground))' }}>
              Run and inspect any Stage 2 variant (DSPy, LoRA, or QLoRA) on any group of Stage 1 points side-by-side.
            </p>
          </div>
        </div>

        {/* 3-Column Selector Bar */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr auto', gap: '0.75rem', alignItems: 'flex-end' }}>
          {/* Meeting Selector */}
          <div>
            <label style={S.label}>Validation Meeting</label>
            <select
              value={valMeetingId || ''}
              onChange={(e) => {
                setValMeetingId(e.target.value);
                handleLoadValGroups(e.target.value);
              }}
              style={S.select}
            >
              <option value="">Select Meeting...</option>
              {meetings.map((m) => (
                <option key={m.id} value={m.id}>{m.filename}</option>
              ))}
            </select>
          </div>

          {/* Group Selector */}
          <div>
            <label style={S.label}>Stage 1 Point Group</label>
            <select
              value={valGroupIndex}
              onChange={(e) => setValGroupIndex(Number(e.target.value))}
              disabled={valGroups.length === 0}
              style={S.select}
            >
              {valGroups.map((grp, idx) => (
                <option key={idx} value={idx}>
                  Group {idx + 1} ({grp.point_count} Stage 1 points)
                </option>
              ))}
              {valGroups.length === 0 && <option value="0">No groups loaded</option>}
            </select>
          </div>

          {/* Variant Selector */}
          <div>
            <label style={S.label}>Stage 2 Variant</label>
            <select
              value={valVariantId}
              onChange={(e) => setValVariantId(e.target.value)}
              style={S.select}
            >
              {variants.map((v) => (
                <option key={v.variant_id} value={v.variant_id}>
                  {v.label} {v.method ? `[${v.method.toUpperCase()}]` : ''} {v.is_default ? '(Baseline)' : v.scores ? `(${v.scores.overall}%)` : ''}
                </option>
              ))}
            </select>
          </div>

          {/* Run Button */}
          <button
            onClick={handleRunValidation}
            disabled={validationLoading || valGroups.length === 0}
            style={S.btn('primary', validationLoading || valGroups.length === 0)}
          >
            {validationLoading ? <RefreshCw size={13} className="animate-spin" /> : <Play size={13} />}
            {validationLoading ? 'Running...' : 'Run'}
          </button>
        </div>

        {/* Side-by-side Workspace Layout */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', minHeight: '380px' }}>
          {/* Left Panel: STAGE 2 INPUT */}
          <div style={{
            border: '1px solid hsl(var(--border))',
            borderRadius: '10px',
            background: 'hsl(var(--background))',
            display: 'flex',
            flexDirection: 'column',
            overflow: 'hidden',
          }}>
            <div style={{
              padding: '0.65rem 0.9rem',
              background: 'hsl(var(--muted) / 0.4)',
              borderBottom: '1px solid hsl(var(--border))',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
            }}>
              <span style={{ fontSize: '0.78rem', fontWeight: 700, color: 'hsl(var(--foreground))' }}>
                STAGE 2 INPUT — GROUP {(valGroupIndex + 1)}
              </span>
              <span style={{
                fontSize: '0.7rem', fontWeight: 600, padding: '2px 6px', borderRadius: '6px',
                background: 'hsl(var(--accent) / 0.15)', color: 'hsl(var(--accent))',
              }}>
                {currentGroup?.points?.length || 0} Stage 1 Points
              </span>
            </div>

            <div style={{ padding: '0.9rem', flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
              {/* Stage 1 Points */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
                <div style={{ ...S.label, marginBottom: '0.2rem' }}>Stage 1 Discussion Points:</div>
                {(!currentGroup?.points || currentGroup.points.length === 0) ? (
                  <div style={{ fontSize: '0.8rem', color: 'hsl(var(--muted-foreground))', padding: '1rem 0' }}>
                    Select a meeting to view Stage 1 points.
                  </div>
                ) : (
                  currentGroup.points.map((pt: any, i: number) => {
                    const text = typeof pt === 'object' ? (pt.discussion_point || pt.point || JSON.stringify(pt)) : String(pt);
                    const speakers = typeof pt === 'object' ? (pt.speakers || pt.speaker || []) : [];
                    return (
                      <div
                        key={i}
                        style={{
                          padding: '0.5rem 0.7rem',
                          borderRadius: '7px',
                          background: 'hsl(var(--muted) / 0.3)',
                          border: '1px solid hsl(var(--border) / 0.5)',
                          fontSize: '0.8rem',
                          lineHeight: '1.45',
                        }}
                      >
                        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '2px' }}>
                          <span style={{ fontWeight: 700, color: 'hsl(var(--accent))', fontSize: '0.72rem' }}>
                            #{i + 1}
                          </span>
                          {speakers.map((sp: string, sIdx: number) => (
                            <span
                              key={sIdx}
                              style={{
                                fontSize: '0.66rem', fontWeight: 600, padding: '1px 5px', borderRadius: '4px',
                                background: 'hsl(var(--muted))', color: 'hsl(var(--foreground))',
                              }}
                            >
                              {sp}
                            </span>
                          ))}
                        </div>
                        <div style={{ color: 'hsl(var(--foreground))' }}>{text}</div>
                      </div>
                    );
                  })
                )}
              </div>

              {/* Context Summary */}
              {currentGroup?.global_context && (
                <div>
                  <div style={{ ...S.label, marginBottom: '0.2rem' }}>Retrieved Global Context:</div>
                  <div style={{
                    fontSize: '0.75rem', color: 'hsl(var(--muted-foreground))',
                    background: 'hsl(var(--muted) / 0.2)', padding: '0.45rem 0.6rem',
                    borderRadius: '6px', maxHeight: '80px', overflowY: 'auto',
                    border: '1px solid hsl(var(--border) / 0.5)',
                  }}>
                    {currentGroup.global_context}
                  </div>
                </div>
              )}

              {/* Meeting Context */}
              {currentGroup?.meeting_context && (
                <div>
                  <div style={{ ...S.label, marginBottom: '0.2rem' }}>Retrieved Meeting Context:</div>
                  <div style={{
                    fontSize: '0.75rem', color: 'hsl(var(--muted-foreground))',
                    background: 'hsl(var(--muted) / 0.2)', padding: '0.45rem 0.6rem',
                    borderRadius: '6px', maxHeight: '80px', overflowY: 'auto',
                    border: '1px solid hsl(var(--border) / 0.5)',
                  }}>
                    {currentGroup.meeting_context}
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Right Panel: DSPy STAGE 2 OUTPUT */}
          <div style={{
            border: '1px solid hsl(var(--border))',
            borderRadius: '10px',
            background: 'hsl(var(--background))',
            display: 'flex',
            flexDirection: 'column',
            overflow: 'hidden',
          }}>
            <div style={{
              padding: '0.65rem 0.9rem',
              background: 'hsl(var(--muted) / 0.4)',
              borderBottom: '1px solid hsl(var(--border))',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
            }}>
              <span style={{ fontSize: '0.78rem', fontWeight: 700, color: 'hsl(var(--foreground))' }}>
                STAGE 2 DSPy OUTPUT
              </span>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                {validationOutput && (
                  <button
                    onClick={handleCopyJson}
                    style={{
                      display: 'flex', alignItems: 'center', gap: '3px',
                      padding: '2px 7px', borderRadius: '5px', border: '1px solid hsl(var(--border))',
                      background: 'transparent', color: 'hsl(var(--muted-foreground))', cursor: 'pointer',
                      fontSize: '0.72rem',
                    }}
                  >
                    {jsonCopied ? <Check size={11} color="hsl(142 76% 36%)" /> : <Copy size={11} />}
                    {jsonCopied ? 'Copied' : 'Copy JSON'}
                  </button>
                )}
                <button
                  onClick={() => setShowRawJson(!showRawJson)}
                  style={{
                    padding: '2px 7px', borderRadius: '5px', border: '1px solid hsl(var(--border))',
                    background: showRawJson ? 'hsl(var(--accent) / 0.15)' : 'transparent',
                    color: showRawJson ? 'hsl(var(--accent))' : 'hsl(var(--muted-foreground))',
                    cursor: 'pointer', fontSize: '0.72rem', fontWeight: 600,
                  }}
                >
                  {showRawJson ? 'Formatted View' : 'Raw JSON'}
                </button>
              </div>
            </div>

            <div style={{ padding: '0.9rem', flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
              {validationLoading ? (
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', gap: '8px', color: 'hsl(var(--muted-foreground))' }}>
                  <RefreshCw size={22} className="animate-spin" />
                  <span style={{ fontSize: '0.82rem' }}>Running DSPy inference on Group {valGroupIndex + 1}...</span>
                </div>
              ) : !validationOutput ? (
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', gap: '8px', color: 'hsl(var(--muted-foreground))', textAlign: 'center' }}>
                  <Sparkles size={24} style={{ opacity: 0.3 }} />
                  <span style={{ fontSize: '0.82rem' }}>Click <strong>Run</strong> above to test this variant on Group {valGroupIndex + 1}.</span>
                </div>
              ) : showRawJson ? (
                <pre style={{
                  margin: 0, padding: '0.6rem', borderRadius: '6px',
                  background: 'hsl(var(--muted) / 0.3)', fontSize: '0.75rem',
                  fontFamily: 'monospace', whiteSpace: 'pre-wrap', wordBreak: 'break-word',
                  color: 'hsl(var(--foreground))',
                }}>
                  {validationOutput.raw_output}
                </pre>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
                  {validationOutput.parsed_output?.points && validationOutput.parsed_output.points.length > 0 ? (
                    validationOutput.parsed_output.points.map((pt, pIdx) => (
                      <div
                        key={pIdx}
                        style={{
                          padding: '0.65rem 0.85rem',
                          borderRadius: '8px',
                          border: '1px solid hsl(var(--border))',
                          background: 'hsl(var(--card))',
                          display: 'flex',
                          flexDirection: 'column',
                          gap: '0.35rem',
                        }}
                      >
                        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                          <span style={{
                            fontSize: '0.68rem', fontWeight: 700, padding: '1px 5px', borderRadius: '4px',
                            background: 'hsl(142 76% 36% / 0.15)', color: 'hsl(142 76% 36%)',
                          }}>
                            Point {pIdx + 1}
                          </span>
                        </div>
                        <div style={{ fontSize: '0.83rem', color: 'hsl(var(--foreground))', lineHeight: '1.45' }}>
                          {pt.point}
                        </div>

                        {/* Speaker & Action Owner Tags */}
                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', marginTop: '2px' }}>
                          {pt.speaker && pt.speaker.length > 0 && (
                            <div style={{ display: 'flex', alignItems: 'center', gap: '3px', fontSize: '0.7rem', color: 'hsl(var(--muted-foreground))' }}>
                              <User size={11} />
                              <span>Speakers: <strong>{pt.speaker.join(', ')}</strong></span>
                            </div>
                          )}
                          {pt.action_owner && pt.action_owner.length > 0 && (
                            <div style={{
                              display: 'flex', alignItems: 'center', gap: '3px', fontSize: '0.7rem',
                              color: 'hsl(217 91% 60%)', fontWeight: 600,
                            }}>
                              <CornerDownRight size={11} />
                              <span>Owner: {pt.action_owner.join(', ')}</span>
                            </div>
                          )}
                        </div>
                      </div>
                    ))
                  ) : (
                    <div style={{ fontSize: '0.8rem', color: 'hsl(var(--muted-foreground))' }}>
                      {validationOutput.raw_output || 'No points returned.'}
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Score mini-dashboard at bottom */}
            {validationOutput && (
              <div style={{
                padding: '0.75rem 0.9rem',
                borderTop: '1px solid hsl(var(--border))',
                background: 'hsl(var(--muted) / 0.2)',
                display: 'grid',
                gridTemplateColumns: 'repeat(3, 1fr)',
                gap: '0.6rem',
              }}>
                <ScoreBar value={validationOutput.eval_scores.overall} label="Overall Score" />
                <ScoreBar value={validationOutput.eval_scores.point_coverage} label="Point Coverage" />
                <ScoreBar value={validationOutput.eval_scores.speaker_preservation} label="Speaker Preserved" />
                <ScoreBar value={validationOutput.eval_scores.action_owner_preservation} label="Action Owner" />
                <ScoreBar value={validationOutput.eval_scores.hallucination_detection} label="No Hallucination" />
                <ScoreBar value={validationOutput.eval_scores.duplicate_detection} label="No Duplicates" />
              </div>
            )}
          </div>
        </div>

        {/* ── Section 3: Feedback & Retraining ── */}
        {validationOutput && (
          <div style={{
            borderRadius: '10px',
            border: '1px solid hsl(var(--border))',
            background: 'hsl(var(--card))',
            padding: '1rem',
            display: 'flex',
            flexDirection: 'column',
            gap: '0.9rem',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <MessageSquare size={16} color="hsl(var(--accent))" />
                <span style={{ fontWeight: 700, fontSize: '0.88rem', color: 'hsl(var(--foreground))' }}>
                  Provide Feedback on this Output & Retrain
                </span>
              </div>
              <span style={{ fontSize: '0.74rem', color: 'hsl(var(--muted-foreground))' }}>
                Retraining builds an immutable new variant (e.g. Variant 002) without modifying existing ones.
              </span>
            </div>

            {/* Category Chips */}
            <div>
              <div style={{ ...S.label, marginBottom: '0.4rem' }}>Error Categories:</div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                {(Object.keys(STAGE2_FEEDBACK_LABELS) as Stage2FeedbackCategory[]).map((cat) => {
                  const isSelected = feedbackCategories.includes(cat);
                  return (
                    <button
                      key={cat}
                      onClick={() => toggleFeedbackCategory(cat)}
                      style={{
                        padding: '3px 9px',
                        borderRadius: '6px',
                        border: `1px solid ${isSelected ? 'hsl(var(--accent))' : 'hsl(var(--border))'}`,
                        background: isSelected ? 'hsl(var(--accent) / 0.15)' : 'hsl(var(--background))',
                        color: isSelected ? 'hsl(var(--accent))' : 'hsl(var(--foreground))',
                        fontSize: '0.75rem',
                        fontWeight: isSelected ? 700 : 500,
                        cursor: 'pointer',
                        transition: 'all 0.12s ease',
                      }}
                    >
                      {STAGE2_FEEDBACK_LABELS[cat]}
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Comment textarea */}
            <div>
              <label style={S.label}>Detailed Correction / Comment:</label>
              <textarea
                value={feedbackComment}
                onChange={(e) => setFeedbackComment(e.target.value)}
                placeholder="Describe what needs improvement or write the ideal consolidated point..."
                rows={3}
                style={{
                  width: '100%',
                  padding: '0.55rem 0.75rem',
                  borderRadius: '8px',
                  border: '1px solid hsl(var(--border))',
                  background: 'hsl(var(--background))',
                  color: 'hsl(var(--foreground))',
                  fontSize: '0.82rem',
                  outline: 'none',
                  boxSizing: 'border-box',
                  resize: 'vertical',
                }}
              />
            </div>

            {/* Actions */}
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div style={{ display: 'flex', gap: '8px' }}>
                <button
                  onClick={handleSubmitFeedbackAndRetrain}
                  disabled={feedbackSubmitting || (feedbackCategories.length === 0 && !feedbackComment)}
                  style={S.btn('primary', feedbackSubmitting || (feedbackCategories.length === 0 && !feedbackComment))}
                >
                  {feedbackSubmitting ? <RefreshCw size={13} className="animate-spin" /> : <Send size={13} />}
                  Submit Feedback & Retrain
                </button>
                <button
                  onClick={handleSubmitFeedbackOnly}
                  disabled={feedbackSubmitting || (feedbackCategories.length === 0 && !feedbackComment)}
                  style={S.btn('ghost', feedbackSubmitting || (feedbackCategories.length === 0 && !feedbackComment))}
                >
                  Save Feedback Only
                </button>
              </div>

              {feedbackSubmitted && (
                <span style={{ fontSize: '0.78rem', color: 'hsl(142 76% 36%)', fontWeight: 600 }}>
                  ✓ Feedback submitted!
                </span>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
