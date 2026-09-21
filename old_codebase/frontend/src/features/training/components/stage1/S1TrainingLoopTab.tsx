import { useState, useEffect, useRef } from 'react';
import {
  Play, RefreshCw, ChevronRight, ChevronDown, CheckCircle, XCircle,
  AlertCircle, Sparkles, MessageSquare, Send, FileText, Copy, Check,
} from 'lucide-react';
import { useStage1Store } from '../../store/stage1Store';
import {
  startStage1Optimization,
  getOptimizationRunStatus,
  runStage1Validation,
  submitFeedbackAndRetrain,
  prepareTranscriptWindows,
  listStage1Variants,
} from '../../api/stage1Api';
import { FEEDBACK_CATEGORY_LABELS } from '../../types/stage1Types';
import type { FeedbackCategory } from '../../types/stage1Types';

// ── Shared styles ────────────────────────────────────────────────────────────

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
  btn: (variant: 'primary' | 'ghost' | 'danger' = 'primary', disabled = false) => ({
    display: 'inline-flex', alignItems: 'center', gap: '6px',
    padding: '0.55rem 1.1rem', borderRadius: '8px', border: 'none',
    cursor: disabled ? 'not-allowed' : 'pointer',
    fontSize: '0.85rem', fontWeight: 600, transition: 'all 0.15s ease',
    opacity: disabled ? 0.6 : 1,
    background: variant === 'primary'
      ? 'hsl(var(--accent))'
      : variant === 'danger'
      ? 'hsl(0 75% 55%)'
      : 'hsl(var(--muted))',
    color: variant === 'ghost' ? 'hsl(var(--foreground))' : 'white',
  } as React.CSSProperties),
};

// ── Score bar component ───────────────────────────────────────────────────────

function ScoreBar({ value, label }: { value: number; label: string }) {
  const pct = Math.round(value);
  const color = pct >= 80 ? 'hsl(142 71% 45%)' : pct >= 60 ? 'hsl(45 93% 47%)' : 'hsl(0 75% 55%)';
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.78rem' }}>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ color: 'hsl(var(--muted-foreground))', marginBottom: '3px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{label}</div>
        <div style={{ height: '6px', borderRadius: '3px', background: 'hsl(var(--muted))', overflow: 'hidden' }}>
          <div style={{ height: '100%', width: `${Math.min(pct, 100)}%`, borderRadius: '3px', background: color, transition: 'width 0.5s ease' }} />
        </div>
      </div>
      <div style={{ fontWeight: 700, color, minWidth: '38px', textAlign: 'right' }}>{pct}%</div>
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────

export default function S1TrainingLoopTab() {
  const {
    generatedWindows, settings, variants, setVariants,
    currentRunId, setCurrentRunId, currentRunStatus, setCurrentRunStatus,
    trainingLogs, appendTrainingLog, clearTrainingLogs,
    meetings,
    valMeetingId, setValMeetingId,
    valWindowId, setValWindowId,
    valWindows, setValWindows,
    valVariantId, setValVariantId,
    validationOutput, setValidationOutput, validationLoading, setValidationLoading,
    feedbackCategories, toggleFeedbackCategory, feedbackComment, setFeedbackComment,
    feedbackSubmitting, setFeedbackSubmitting, feedbackSubmitted, setFeedbackSubmitted,
    clearFeedback, datasetMeetingId,
  } = useStage1Store();

  const [showRawJson, setShowRawJson] = useState(false);
  const [jsonCopied, setJsonCopied] = useState(false);
  const [showFeedback, setShowFeedback] = useState(false);
  const [loadingValWindows, setLoadingValWindows] = useState(false);
  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const trainingWindows = generatedWindows.filter((w) => w.selected);
  const isRunning = currentRunStatus?.status === 'running' || currentRunStatus?.status === 'starting';
  const isDone = currentRunStatus?.status === 'done';
  const isError = currentRunStatus?.status === 'error';

  // Poll optimization run status
  useEffect(() => {
    if (currentRunId && (isRunning)) {
      pollingRef.current = setInterval(async () => {
        try {
          const status = await getOptimizationRunStatus(currentRunId);
          setCurrentRunStatus(status);
          if (status.message) appendTrainingLog(status.message);
          if (status.status === 'done' || status.status === 'error') {
            clearInterval(pollingRef.current!);
            // Refresh variants list
            const freshVariants = await listStage1Variants();
            setVariants(freshVariants);
          }
        } catch (e) {
          clearInterval(pollingRef.current!);
        }
      }, 2000);
    }
    return () => { if (pollingRef.current) clearInterval(pollingRef.current); };
  }, [currentRunId, isRunning]);

  const handleStartTraining = async () => {
    if (!trainingWindows.length) return;
    clearTrainingLogs();
    setCurrentRunStatus(null);
    try {
      appendTrainingLog('Starting Stage 1 optimization...');
      const res = await startStage1Optimization({
        meeting_id: datasetMeetingId || '',
        windows: trainingWindows,
        settings,
        parent_variant_id: 'default',
      });
      setCurrentRunId(res.run_id);
      setCurrentRunStatus({ run_id: res.run_id, status: 'starting', progress: 0, message: 'Initializing...', variant_id: null, error: null });
    } catch (err: any) {
      appendTrainingLog(`Error: ${err?.response?.data?.detail || err.message || 'Unknown error'}`);
    }
  };

  const handleLoadValWindows = async (meetingId: string) => {
    setLoadingValWindows(true);
    try {
      const res = await prepareTranscriptWindows(meetingId, settings.default_window_size_minutes * 60, settings.default_window_overlap_seconds);
      setValWindows(res.windows);
    } catch (e) {
      console.error(e);
    } finally {
      setLoadingValWindows(false);
    }
  };

  const handleRunValidation = async () => {
    const selectedWin = valWindows.find((w) => w.window_id === valWindowId);
    if (!selectedWin) return;
    setValidationLoading(true);
    setValidationOutput(null);
    clearFeedback();
    setShowFeedback(false);
    try {
      const res = await runStage1Validation({
        variant_id: valVariantId,
        meeting_id: valMeetingId || '',
        window_id: valWindowId || undefined,
        transcript_text: selectedWin.transcript_text,
        context_summary: selectedWin.context_summary,
        agenda_summary: selectedWin.agenda_summary,
      });
      setValidationOutput(res);
    } catch (err: any) {
      console.error(err);
    } finally {
      setValidationLoading(false);
    }
  };

  const handleSubmitFeedbackAndRetrain = async () => {
    const selectedWin = valWindows.find((w) => w.window_id === valWindowId);
    if (!selectedWin || !validationOutput) return;
    setFeedbackSubmitting(true);
    try {
      const feedback = {
        meeting_id: valMeetingId || '',
        window_id: valWindowId || '',
        variant_id: valVariantId,
        transcript_window: selectedWin.transcript_text,
        model_output: validationOutput.raw_output,
        categories: feedbackCategories,
        comment: feedbackComment,
        context_summary: selectedWin.context_summary,
        agenda_summary: selectedWin.agenda_summary,
      };
      const res = await submitFeedbackAndRetrain({
        feedback,
        parent_variant_id: valVariantId,
        meeting_id: valMeetingId || '',
        windows: trainingWindows.length ? trainingWindows : valWindows,
        settings,
      });
      clearTrainingLogs();
      setCurrentRunId(res.run_id);
      setCurrentRunStatus({ run_id: res.run_id, status: 'starting', progress: 0, message: 'Retraining from feedback...', variant_id: null, error: null });
      setFeedbackSubmitted(true);
      setShowFeedback(false);
    } catch (err) {
      console.error(err);
    } finally {
      setFeedbackSubmitting(false);
    }
  };

  const copyJson = () => {
    if (validationOutput?.raw_output) {
      navigator.clipboard.writeText(validationOutput.raw_output);
      setJsonCopied(true);
      setTimeout(() => setJsonCopied(false), 1500);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      {/* ── Initial Optimization ─────────────────────────────────────────────── */}
      <div style={S.card}>
        <div style={S.sectionHeader()}>
          <div style={S.iconBadge()}>
            <Sparkles size={14} color="hsl(var(--accent))" />
          </div>
          <h3 style={{ margin: 0, fontSize: '0.95rem', fontWeight: 700 }}>Initial Optimization</h3>
          {trainingWindows.length > 0 && (
            <span style={{
              fontSize: '0.72rem', fontWeight: 600, padding: '2px 8px', borderRadius: '6px',
              background: 'hsl(var(--accent) / 0.1)', color: 'hsl(var(--accent))',
            }}>
              {trainingWindows.length} windows ready
            </span>
          )}
        </div>

        {trainingWindows.length === 0 ? (
          <div style={{
            padding: '1.5rem', borderRadius: '10px', textAlign: 'center',
            background: 'hsl(var(--muted) / 0.4)', border: '1px dashed hsl(var(--border))',
            color: 'hsl(var(--muted-foreground))', fontSize: '0.85rem',
          }}>
            ← Go to the <strong>Dataset</strong> tab to select a meeting and prepare transcript windows first.
          </div>
        ) : (
          <>
            <div style={{ marginBottom: '1rem', fontSize: '0.82rem', color: 'hsl(var(--muted-foreground))', lineHeight: '1.5' }}>
              Runs DSPy optimization on <strong>{trainingWindows.filter(w => w.role !== 'val').length}</strong> training
              windows from the Dataset tab. Creates an immutable Stage 1 variant.
              Starting variant: <strong>Default Text Prompt</strong>.
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
              <button
                onClick={handleStartTraining}
                disabled={isRunning}
                style={S.btn('primary', isRunning)}
              >
                {isRunning
                  ? <><RefreshCw size={14} style={{ animation: 'spin 1s linear infinite' }} /> Optimizing...</>
                  : <><Play size={14} /> Start Initial Training</>}
              </button>
              {isDone && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: 'hsl(142 71% 45%)', fontSize: '0.85rem', fontWeight: 600 }}>
                  <CheckCircle size={16} /> Variant created: {currentRunStatus?.variant_id?.slice(0, 8)}...
                </div>
              )}
              {isError && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: 'hsl(0 75% 55%)', fontSize: '0.85rem' }}>
                  <XCircle size={16} /> {currentRunStatus?.error || 'Optimization failed'}
                </div>
              )}
            </div>
          </>
        )}

        {/* Progress */}
        {(isRunning || isDone || isError) && currentRunStatus && (
          <div style={{ marginTop: '1rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '6px' }}>
              <span style={{ fontSize: '0.78rem', color: 'hsl(var(--muted-foreground))' }}>
                {currentRunStatus.message}
              </span>
              <span style={{ fontSize: '0.78rem', fontWeight: 700, color: 'hsl(var(--accent))' }}>
                {currentRunStatus.progress}%
              </span>
            </div>
            <div style={{ height: '6px', borderRadius: '3px', background: 'hsl(var(--muted))' }}>
              <div style={{
                height: '100%', borderRadius: '3px',
                background: isError ? 'hsl(0 75% 55%)' : 'hsl(var(--accent))',
                width: `${currentRunStatus.progress}%`, transition: 'width 0.4s ease',
              }} />
            </div>
            {/* Log tail */}
            {trainingLogs.length > 0 && (
              <div style={{
                marginTop: '0.6rem', background: 'hsl(var(--background))', borderRadius: '8px',
                padding: '0.6rem 0.8rem', maxHeight: '100px', overflowY: 'auto',
                fontFamily: 'monospace', fontSize: '0.72rem', color: 'hsl(var(--muted-foreground))',
                border: '1px solid hsl(var(--border) / 0.5)',
              }}>
                {trainingLogs.map((log, i) => <div key={i}>{log}</div>)}
              </div>
            )}
          </div>
        )}
      </div>

      {/* ── Validation Workspace ──────────────────────────────────────────────── */}
      <div style={S.card}>
        <div style={S.sectionHeader()}>
          <div style={S.iconBadge('hsl(262 80% 65% / 0.15)')}>
            <Play size={14} color="hsl(262 80% 65%)" />
          </div>
          <h3 style={{ margin: 0, fontSize: '0.95rem', fontWeight: 700 }}>Interactive Validation</h3>
        </div>

        {/* Controls row */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '0.75rem', marginBottom: '1rem' }}>
          {/* Meeting selector */}
          <div>
            <label style={S.label}>Meeting</label>
            <select
              value={valMeetingId || ''}
              onChange={(e) => {
                const id = e.target.value;
                setValMeetingId(id || null);
                if (id) handleLoadValWindows(id);
              }}
              style={S.select}
            >
              <option value="">Select meeting...</option>
              {meetings.map((m) => (
                <option key={m.id} value={m.id}>{m.filename}</option>
              ))}
            </select>
          </div>

          {/* Window selector */}
          <div>
            <label style={S.label}>Transcript Window</label>
            <select
              value={valWindowId || ''}
              onChange={(e) => setValWindowId(e.target.value || null)}
              style={S.select}
              disabled={!valMeetingId || loadingValWindows}
            >
              <option value="">{loadingValWindows ? 'Loading windows...' : 'Select window...'}</option>
              {valWindows.map((w) => (
                <option key={w.window_id} value={w.window_id}>
                  Window {w.window_index + 1} ({Math.floor(w.start_time / 60)}m–{Math.floor(w.end_time / 60)}m)
                </option>
              ))}
            </select>
          </div>

          {/* Variant selector */}
          <div>
            <label style={S.label}>DSPy Variant</label>
            <select
              value={valVariantId}
              onChange={(e) => setValVariantId(e.target.value)}
              style={S.select}
            >
              {variants.map((v) => (
                <option key={v.variant_id} value={v.variant_id}>
                  {v.label}{v.is_best ? ' ★' : ''}
                </option>
              ))}
              {variants.length === 0 && <option value="default">Default Text Prompt</option>}
            </select>
          </div>
        </div>

        <button
          onClick={handleRunValidation}
          disabled={!valMeetingId || !valWindowId || validationLoading}
          style={S.btn('primary', !valMeetingId || !valWindowId || validationLoading)}
        >
          {validationLoading
            ? <><RefreshCw size={14} style={{ animation: 'spin 1s linear infinite' }} /> Running...</>
            : <><Play size={14} /> Run</>}
        </button>

        {/* Split layout result */}
        {validationOutput && (
          <div style={{ marginTop: '1.25rem', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', minHeight: '300px' }}>
            {/* Left: Transcript */}
            <div style={{
              borderRadius: '10px', border: '1px solid hsl(var(--border))',
              display: 'flex', flexDirection: 'column', overflow: 'hidden',
            }}>
              <div style={{
                padding: '0.6rem 0.9rem', borderBottom: '1px solid hsl(var(--border))',
                background: 'hsl(var(--muted) / 0.5)', display: 'flex', alignItems: 'center', gap: '6px',
              }}>
                <FileText size={13} color="hsl(var(--muted-foreground))" />
                <span style={{ fontSize: '0.78rem', fontWeight: 700, color: 'hsl(var(--muted-foreground))' }}>
                  Transcript Window
                </span>
              </div>
              <div style={{
                flex: 1, overflowY: 'auto', padding: '0.75rem',
                fontFamily: 'monospace', fontSize: '0.78rem', color: 'hsl(var(--foreground))',
                lineHeight: '1.6', whiteSpace: 'pre-wrap',
              }}>
                {valWindows.find((w) => w.window_id === valWindowId)?.transcript_text || ''}
              </div>
            </div>

            {/* Right: Output */}
            <div style={{
              borderRadius: '10px', border: '1px solid hsl(var(--accent) / 0.3)',
              display: 'flex', flexDirection: 'column', overflow: 'hidden',
            }}>
              <div style={{
                padding: '0.6rem 0.9rem', borderBottom: '1px solid hsl(var(--border))',
                background: 'hsl(var(--accent) / 0.05)', display: 'flex', alignItems: 'center', justifyContent: 'space-between',
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <Sparkles size={13} color="hsl(var(--accent))" />
                  <span style={{ fontSize: '0.78rem', fontWeight: 700, color: 'hsl(var(--accent))' }}>
                    Stage 1 DSPy Output
                  </span>
                </div>
                <div style={{ display: 'flex', gap: '6px' }}>
                  <button
                    onClick={() => setShowRawJson(!showRawJson)}
                    style={{ ...S.btn('ghost'), padding: '2px 8px', fontSize: '0.72rem' }}
                  >
                    {showRawJson ? 'Formatted' : 'Raw JSON'}
                  </button>
                  <button onClick={copyJson} style={{ ...S.btn('ghost'), padding: '2px 8px', fontSize: '0.72rem' }}>
                    {jsonCopied ? <Check size={12} /> : <Copy size={12} />}
                  </button>
                </div>
              </div>
              <div style={{ flex: 1, overflowY: 'auto', padding: '0.75rem' }}>
                {validationOutput.error ? (
                  <div style={{ color: 'hsl(0 75% 55%)', fontSize: '0.82rem' }}>
                    Error: {validationOutput.error}
                  </div>
                ) : showRawJson ? (
                  <pre style={{ margin: 0, fontFamily: 'monospace', fontSize: '0.72rem', whiteSpace: 'pre-wrap', color: 'hsl(var(--foreground))' }}>
                    {validationOutput.raw_output || 'No output'}
                  </pre>
                ) : validationOutput.parsed_output ? (
                  <DiscussionPointsList output={validationOutput.parsed_output} />
                ) : (
                  <div style={{ color: 'hsl(var(--muted-foreground))', fontSize: '0.82rem' }}>
                    Could not parse JSON output.
                  </div>
                )}
              </div>

              {/* Eval scores mini-bar */}
              {validationOutput.eval_scores && (
                <div style={{
                  padding: '0.6rem 0.75rem', borderTop: '1px solid hsl(var(--border))',
                  background: 'hsl(var(--background))',
                }}>
                  <div style={{ fontSize: '0.72rem', fontWeight: 700, color: 'hsl(var(--muted-foreground))', marginBottom: '6px' }}>
                    EVAL SCORES
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px' }}>
                    {[
                      ['Overall', 'overall'],
                      ['Coverage', 'transcript_coverage'],
                      ['Speakers', 'speaker_preservation'],
                      ['No Hallucinations', 'hallucination_detection'],
                    ].map(([label, key]) => (
                      <ScoreBar
                        key={key}
                        label={label}
                        value={(validationOutput.eval_scores as any)[key] * 100 || 0}
                      />
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* ── Feedback & Retraining ─────────────────────────────────────────────── */}
      {validationOutput && !feedbackSubmitted && (
        <div style={S.card}>
          <div style={{
            display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.9rem',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <div style={S.iconBadge('hsl(45 93% 47% / 0.15)')}>
                <MessageSquare size={14} color="hsl(45 93% 47%)" />
              </div>
              <h3 style={{ margin: 0, fontSize: '0.95rem', fontWeight: 700 }}>Feedback & Retraining</h3>
            </div>
            <button
              onClick={() => setShowFeedback(!showFeedback)}
              style={{ ...S.btn('ghost'), padding: '4px 10px', fontSize: '0.78rem' }}
            >
              {showFeedback ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
              {showFeedback ? 'Hide' : 'Give Feedback'}
            </button>
          </div>

          {showFeedback && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              <div>
                <div style={{ fontSize: '0.8rem', fontWeight: 600, color: 'hsl(var(--muted-foreground))', marginBottom: '0.5rem' }}>
                  Error categories (select all that apply):
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                  {(Object.entries(FEEDBACK_CATEGORY_LABELS) as [FeedbackCategory, string][]).map(([key, label]) => {
                    const isSelected = feedbackCategories.includes(key);
                    return (
                      <button
                        key={key}
                        onClick={() => toggleFeedbackCategory(key)}
                        style={{
                          padding: '4px 10px', borderRadius: '20px', fontSize: '0.75rem',
                          fontWeight: isSelected ? 700 : 500, cursor: 'pointer',
                          border: `1.5px solid ${isSelected ? 'hsl(var(--accent))' : 'hsl(var(--border))'}`,
                          background: isSelected ? 'hsl(var(--accent) / 0.1)' : 'transparent',
                          color: isSelected ? 'hsl(var(--accent))' : 'hsl(var(--muted-foreground))',
                          transition: 'all 0.12s ease',
                        }}
                      >
                        {label}
                      </button>
                    );
                  })}
                </div>
              </div>

              <div>
                <label style={S.label}>Additional comments</label>
                <textarea
                  value={feedbackComment}
                  onChange={(e) => setFeedbackComment(e.target.value)}
                  placeholder="Describe the issue in detail. E.g., 'The system merged two unrelated topics — Sprint planning and budget review should be separate points.'"
                  rows={3}
                  style={{
                    width: '100%', padding: '0.6rem 0.8rem', borderRadius: '8px',
                    border: '1px solid hsl(var(--border))', background: 'hsl(var(--background))',
                    color: 'hsl(var(--foreground))', fontSize: '0.85rem', resize: 'vertical',
                    fontFamily: 'inherit', outline: 'none', boxSizing: 'border-box',
                  }}
                />
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                <button
                  onClick={handleSubmitFeedbackAndRetrain}
                  disabled={feedbackSubmitting || feedbackCategories.length === 0}
                  style={S.btn('primary', feedbackSubmitting || feedbackCategories.length === 0)}
                >
                  {feedbackSubmitting
                    ? <><RefreshCw size={14} style={{ animation: 'spin 1s linear infinite' }} /> Retraining...</>
                    : <><Send size={14} /> Submit Feedback & Retrain</>}
                </button>
                <div style={{ fontSize: '0.75rem', color: 'hsl(var(--muted-foreground))' }}>
                  Creates a new immutable DSPy variant — does not overwrite the existing one.
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {feedbackSubmitted && (
        <div style={{
          padding: '0.8rem 1rem', borderRadius: '10px',
          background: 'hsl(142 71% 45% / 0.08)', border: '1px solid hsl(142 71% 45% / 0.3)',
          display: 'flex', alignItems: 'center', gap: '8px',
          fontSize: '0.85rem', color: 'hsl(142 71% 35%)', fontWeight: 600,
        }}>
          <CheckCircle size={16} />
          Feedback saved! Retraining started — a new variant is being created. Check the Evaluation / Variants tab when done.
        </div>
      )}

      <style>{`@keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}

// ── Discussion point renderer ─────────────────────────────────────────────────

function DiscussionPointsList({ output }: { output: Record<string, unknown> }) {
  const points: any[] = Array.isArray((output as any).discussion_points)
    ? (output as any).discussion_points
    : [];

  if (points.length === 0) {
    return (
      <div style={{ color: 'hsl(var(--muted-foreground))', fontSize: '0.82rem', display: 'flex', alignItems: 'center', gap: '6px' }}>
        <AlertCircle size={14} /> No discussion points found in output.
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.7rem' }}>
      {points.map((dp: any, i: number) => (
        <div key={i} style={{
          padding: '0.7rem 0.8rem', borderRadius: '8px',
          background: 'hsl(var(--background))', border: '1px solid hsl(var(--border) / 0.6)',
        }}>
          <div style={{ fontSize: '0.83rem', fontWeight: 600, color: 'hsl(var(--foreground))', marginBottom: '6px', lineHeight: '1.4' }}>
            {i + 1}. {dp.discussion_point || '(No text)'}
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
            {(dp.speakers || []).map((s: string) => (
              <Pill key={s} value={s} color="hsl(142 71% 45%)" />
            ))}
            {(() => {
              const owner = dp.action_owner || dp.action_owners || dp.owner || dp.assignee
              const ownerText = Array.isArray(owner) ? owner.filter(Boolean).join(', ') : (owner && String(owner).trim() !== '' && String(owner).toLowerCase() !== 'null' && String(owner).toLowerCase() !== 'none' ? String(owner) : null)
              return ownerText ? (
                <Pill value={`Owner: ${ownerText}`} color="hsl(35 95% 45%)" />
              ) : (
                <Pill value="No action owner" color="hsl(var(--muted-foreground))" />
              )
            })()}
            {(dp.technical_terms || []).map((t: string) => (
              <Pill key={t} value={t} color="hsl(var(--accent))" />
            ))}
            {(dp.dates || []).map((d: string) => (
              <Pill key={d} value={d} color="hsl(45 93% 47%)" />
            ))}
            {(dp.numbers || []).map((n: string) => (
              <Pill key={n} value={String(n)} color="hsl(200 80% 55%)" />
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

function Pill({ value, color }: { value: string; color: string }) {
  return (
    <span style={{
      fontSize: '0.68rem', fontWeight: 600, padding: '1px 7px', borderRadius: '10px',
      background: `${color}1a`, color, border: `1px solid ${color}33`,
    }}>
      {value}
    </span>
  );
}
