import { useState, useEffect } from 'react';
import {
  Play,
  CheckCircle,
  AlertTriangle,
  Layers,
  ArrowRight,
  MoveRight,
  Sparkles,
  Loader,
  RefreshCw,
  MessageSquare,
  ShieldCheck,
  Check,
  BarChart2,
  FileText,
  CornerDownRight,
  User,
  Sliders,
} from 'lucide-react';
import { useStage3Store } from '../../store/stage3Store';
import {
  startStage3Optimization,
  getStage3RunStatus,
  runStage3Validation,
  submitStage3Feedback,
  submitStage3FeedbackAndRetrain,
  listStage3Variants,
} from '../../api/stage3Api';
import type { Stage3Agenda, Stage3Assignment, Stage3ValidationOutput } from '../../types/training';

const S = {
  card: {
    background: 'hsl(var(--card))',
    borderRadius: '12px',
    border: '1px solid hsl(var(--border))',
    padding: '1.25rem',
  } as React.CSSProperties,
  btn: (variant: 'primary' | 'ghost' | 'success' | 'danger' = 'primary', disabled = false) => ({
    display: 'inline-flex',
    alignItems: 'center',
    gap: '6px',
    padding: '0.55rem 1.1rem',
    borderRadius: '8px',
    border: 'none',
    cursor: disabled ? 'not-allowed' : 'pointer',
    fontSize: '0.85rem',
    fontWeight: 600,
    transition: 'all 0.15s ease',
    opacity: disabled ? 0.6 : 1,
    background:
      variant === 'primary'
        ? 'hsl(var(--accent))'
        : variant === 'success'
        ? 'hsl(140 70% 45%)'
        : variant === 'danger'
        ? 'hsl(var(--destructive))'
        : 'hsl(var(--muted))',
    color: variant === 'ghost' ? 'hsl(var(--foreground))' : 'white',
  } as React.CSSProperties),
};

function formatScore(val?: number): string {
  if (val === undefined || val === null) return '0%';
  const num = val <= 1.0 ? val * 100 : val;
  return `${Math.round(num)}%`;
}

export default function S3TrainingLoopTab() {
  const {
    datasetMeetingId,
    datasetMeetingFilename,
    batches,
    agendas,
    settings,
    variants,
    setVariants,
    currentRunId,
    setCurrentRunId,
    currentRunStatus,
    setCurrentRunStatus,
    trainingLogs,
    appendTrainingLog,
    valMeetingId,
    setValMeetingId,
    valBatchIndex,
    setValBatchIndex,
    valVariantId,
    setValVariantId,
    validationOutput,
    setValidationOutput,
    validationLoading,
    setValidationLoading,
  } = useStage3Store();

  const [feedbackCount, setFeedbackCount] = useState(0);
  const [movingPointId, setMovingPointId] = useState<string | null>(null);
  const [lastFeedbackMessage, setLastFeedbackMessage] = useState<string | null>(null);
  const [retraining, setRetraining] = useState(false);

  // Poll active run
  useEffect(() => {
    if (!currentRunId) return;
    if (currentRunStatus?.status === 'done' || currentRunStatus?.status === 'error') return;

    const interval = setInterval(async () => {
      try {
        const res = await getStage3RunStatus(currentRunId);
        setCurrentRunStatus(res);
        if (res.message) appendTrainingLog(res.message);

        if (res.status === 'done') {
          clearInterval(interval);
          const varRes = await listStage3Variants();
          setVariants(varRes.variants || []);
          if (res.variant_id) setValVariantId(res.variant_id);
        } else if (res.status === 'error') {
          clearInterval(interval);
        }
      } catch (err) {
        console.error('Failed to poll Stage 3 run status:', err);
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [currentRunId, currentRunStatus?.status]);

  // Set default validation meeting
  useEffect(() => {
    if (datasetMeetingId && !valMeetingId) {
      setValMeetingId(datasetMeetingId);
    }
  }, [datasetMeetingId]);

  const handleStartOptimization = async () => {
    if (!datasetMeetingId || batches.length === 0 || agendas.length === 0) return;
    appendTrainingLog(`Starting Stage 3 DSPy optimization for meeting ${datasetMeetingFilename}...`);
    try {
      const res = await startStage3Optimization({
        meeting_id: datasetMeetingId,
        batches,
        agendas,
        settings,
        parent_variant_id: valVariantId || 'default',
      });
      setCurrentRunId(res.run_id);
      setCurrentRunStatus({
        run_id: res.run_id,
        status: 'starting',
        progress: 10,
        message: 'Initializing Stage 3 DSPy optimizer...',
      });
    } catch (err: any) {
      appendTrainingLog(`Optimization start error: ${err?.message || 'Unknown error'}`);
    }
  };

  const handleRunValidation = async () => {
    if (!datasetMeetingId || batches.length === 0 || agendas.length === 0) return;
    const activeBatch = batches[valBatchIndex] || batches[0];
    setValidationLoading(true);
    setValidationOutput(null);
    try {
      const res = await runStage3Validation({
        variant_id: valVariantId,
        meeting_id: datasetMeetingId,
        batch_index: valBatchIndex,
        batch_points: activeBatch.points,
        agendas,
      });
      setValidationOutput(res);
    } catch (err: any) {
      console.error('Validation error:', err);
    } finally {
      setValidationLoading(false);
    }
  };

  const handleMovePoint = async (
    point: Stage3Assignment,
    currentAgendaId: string,
    targetAgendaId: string
  ) => {
    if (currentAgendaId === targetAgendaId) return;
    setMovingPointId(point.point_id);
    try {
      const targetAgenda = agendas.find((a) => a.agenda_id === targetAgendaId);
      await submitStage3Feedback({
        meeting_id: datasetMeetingId || '',
        point_id: point.point_id,
        point_text: point.enhanced_point || '',
        original_agenda_id: currentAgendaId,
        correct_agenda_id: targetAgendaId,
        reason: `Manually reassigned to ${targetAgenda?.title || targetAgendaId}`,
      });

      setFeedbackCount((c) => c + 1);
      setLastFeedbackMessage(`Point ${point.point_id} reassigned to [${targetAgendaId}] and saved as DSPy feedback.`);
      setTimeout(() => setLastFeedbackMessage(null), 4000);

      // Relocate the point locally in validationOutput grouped_by_agenda
      if (validationOutput) {
        const updatedGroups = validationOutput.grouped_by_agenda.map((group) => {
          if (group.agenda_id === currentAgendaId) {
            return {
              ...group,
              points: group.points.filter((p) => p.point_id !== point.point_id),
            };
          }
          if (group.agenda_id === targetAgendaId) {
            return {
              ...group,
              points: [
                ...group.points,
                {
                  ...point,
                  assigned_agenda_id: targetAgendaId,
                  agenda_title: targetAgenda?.title || `Agenda ${targetAgendaId}`,
                  confidence: 'high',
                  reason: 'User manual reassignment (Ground Truth)',
                },
              ],
            };
          }
          return group;
        });

        setValidationOutput({
          ...validationOutput,
          grouped_by_agenda: updatedGroups,
        });
      }
    } catch (err: any) {
      console.error('Failed to submit feedback on move:', err);
    } finally {
      setMovingPointId(null);
    }
  };

  const handleRetrainWithFeedback = async () => {
    if (!datasetMeetingId || batches.length === 0 || agendas.length === 0) return;
    setRetraining(true);
    appendTrainingLog(`Triggering Stage 3 DSPy retraining with collected feedback...`);
    try {
      const res = await submitStage3FeedbackAndRetrain({
        feedback: { meeting_id: datasetMeetingId, timestamp: new Date().toISOString() },
        parent_variant_id: valVariantId || 'default',
        meeting_id: datasetMeetingId,
        batches,
        agendas,
        settings,
      });
      setCurrentRunId(res.run_id);
      setCurrentRunStatus({
        run_id: res.run_id,
        status: 'starting',
        progress: 10,
        message: 'Retraining Stage 3 with feedback examples...',
      });
    } catch (err: any) {
      console.error('Retrain error:', err);
    } finally {
      setRetraining(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', width: '100%' }}>
      {/* ── Section 1: Optimization Action & Status Banner ── */}
      <div style={{ ...S.card, display: 'flex', flexDirection: 'column', gap: '1rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div>
            <h3 style={{ margin: 0, fontSize: '0.95rem', fontWeight: 700 }}>
              Stage 3 DSPy Optimization Loop
            </h3>
            <p style={{ margin: '2px 0 0', fontSize: '0.78rem', color: 'hsl(var(--muted-foreground))' }}>
              Optimizes point → agenda assignment decisions and incorporates user corrections.
            </p>
          </div>

          <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
            {feedbackCount > 0 && (
              <button
                disabled={retraining || currentRunStatus?.status === 'running'}
                onClick={handleRetrainWithFeedback}
                style={S.btn('success', retraining)}
              >
                {retraining ? <Loader size={13} className="spin" /> : <Sparkles size={13} />}
                Retrain with Feedback ({feedbackCount})
              </button>
            )}

            <button
              disabled={
                !datasetMeetingId ||
                batches.length === 0 ||
                agendas.length === 0 ||
                currentRunStatus?.status === 'running'
              }
              onClick={handleStartOptimization}
              style={S.btn(
                'primary',
                !datasetMeetingId || batches.length === 0 || currentRunStatus?.status === 'running'
              )}
            >
              {currentRunStatus?.status === 'running' ? (
                <Loader size={13} className="spin" />
              ) : (
                <Play size={13} />
              )}
              {currentRunStatus?.status === 'running' ? 'Optimizing...' : 'Start Stage 3 Optimization'}
            </button>
          </div>
        </div>

        {/* Live status banner */}
        {currentRunStatus && currentRunStatus.status !== 'not_found' && (
          <div style={{
            borderRadius: '10px',
            border: `1.5px solid ${
              currentRunStatus.status === 'done'
                ? 'hsl(140 70% 45% / 0.4)'
                : currentRunStatus.status === 'error'
                ? 'hsl(0 75% 55% / 0.4)'
                : 'hsl(var(--accent) / 0.4)'
            }`,
            background:
              currentRunStatus.status === 'done'
                ? 'hsl(140 70% 45% / 0.08)'
                : currentRunStatus.status === 'error'
                ? 'hsl(0 75% 55% / 0.08)'
                : 'hsl(var(--accent) / 0.08)',
            padding: '0.85rem 1rem',
            display: 'flex',
            flexDirection: 'column',
            gap: '6px',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.84rem', fontWeight: 700 }}>
                {currentRunStatus.status === 'running' && <Loader size={14} className="spin" color="hsl(var(--accent))" />}
                {currentRunStatus.status === 'done' && <CheckCircle size={14} color="hsl(140 70% 45%)" />}
                {currentRunStatus.status === 'error' && <AlertTriangle size={14} color="hsl(0 75% 55%)" />}
                <span>{currentRunStatus.message}</span>
              </div>
              <span style={{ fontSize: '0.78rem', fontWeight: 800 }}>{currentRunStatus.progress}%</span>
            </div>

            {/* Progress bar */}
            <div style={{ borderRadius: '999px', background: 'hsl(var(--border) / 0.5)', height: '5px', overflow: 'hidden' }}>
              <div style={{
                height: '100%',
                width: `${currentRunStatus.progress}%`,
                background: currentRunStatus.status === 'done' ? 'hsl(140 70% 45%)' : 'hsl(var(--accent))',
                transition: 'width 0.4s ease',
              }} />
            </div>
          </div>
        )}

        {/* Training Log console */}
        {trainingLogs.length > 0 && (
          <div style={{
            borderRadius: '8px', background: 'hsl(var(--background))',
            border: '1px solid hsl(var(--border) / 0.5)', padding: '0.65rem 0.85rem',
            maxHeight: '90px', overflowY: 'auto', fontSize: '0.72rem', fontFamily: 'monospace',
            color: 'hsl(var(--muted-foreground))', lineHeight: 1.4,
          }}>
            {trainingLogs.map((log, i) => (
              <div key={i}>{log}</div>
            ))}
          </div>
        )}
      </div>

      {/* ── Section 2: Validation Workspace Controls ── */}
      <div style={{ ...S.card, display: 'flex', flexDirection: 'column', gap: '1rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div>
            <h3 style={{ margin: 0, fontSize: '0.92rem', fontWeight: 700 }}>
              Point → Agenda Validation Workspace
            </h3>
            <p style={{ margin: '2px 0 0', fontSize: '0.76rem', color: 'hsl(var(--muted-foreground))' }}>
              Run batch assignment on Stage 2 points, review assignments, and reassign points to give ground-truth feedback.
            </p>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            {/* Variant selector */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span style={{ fontSize: '0.76rem', fontWeight: 600, color: 'hsl(var(--muted-foreground))' }}>
                Variant:
              </span>
              <select
                value={valVariantId}
                onChange={(e) => setValVariantId(e.target.value)}
                style={{
                  padding: '0.35rem 0.65rem', borderRadius: '6px',
                  border: '1px solid hsl(var(--border))', background: 'hsl(var(--background))',
                  fontSize: '0.8rem', color: 'hsl(var(--foreground))',
                }}
              >
                {variants.map((v) => (
                  <option key={v.variant_id} value={v.variant_id}>
                    {v.label} {v.is_best ? '⭐' : ''}
                  </option>
                ))}
              </select>
            </div>

            {/* Batch selector */}
            {batches.length > 1 && (
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <span style={{ fontSize: '0.76rem', fontWeight: 600, color: 'hsl(var(--muted-foreground))' }}>
                  Batch:
                </span>
                <select
                  value={valBatchIndex}
                  onChange={(e) => setValBatchIndex(Number(e.target.value))}
                  style={{
                    padding: '0.35rem 0.65rem', borderRadius: '6px',
                    border: '1px solid hsl(var(--border))', background: 'hsl(var(--background))',
                    fontSize: '0.8rem', color: 'hsl(var(--foreground))',
                  }}
                >
                  {batches.map((b, i) => (
                    <option key={i} value={i}>
                      Batch {i + 1} ({b.point_count} pts)
                    </option>
                  ))}
                </select>
              </div>
            )}

            <button
              disabled={validationLoading || batches.length === 0}
              onClick={handleRunValidation}
              style={S.btn('primary', validationLoading || batches.length === 0)}
            >
              {validationLoading ? <Loader size={13} className="spin" /> : <Play size={13} />}
              {validationLoading ? 'Assigning...' : 'Run Validation'}
            </button>
          </div>
        </div>

        {/* Feedback confirmation banner */}
        {lastFeedbackMessage && (
          <div style={{
            padding: '0.65rem 0.85rem', borderRadius: '8px',
            background: 'hsl(140 70% 45% / 0.12)', border: '1px solid hsl(140 70% 45% / 0.3)',
            fontSize: '0.78rem', color: 'hsl(140 70% 40%)', display: 'flex', alignItems: 'center', gap: '6px',
          }}>
            <Check size={14} /> {lastFeedbackMessage}
          </div>
        )}

        {/* Validation scores & interactive assignment groups */}
        {validationOutput && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem', marginTop: '0.5rem' }}>
            {/* Score cards */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '8px' }}>
              <div style={{ padding: '0.65rem', borderRadius: '8px', background: 'hsl(var(--muted) / 0.3)', textAlign: 'center' }}>
                <div style={{ fontSize: '0.7rem', color: 'hsl(var(--muted-foreground))', fontWeight: 600 }}>OVERALL</div>
                <div style={{ fontSize: '1.15rem', fontWeight: 800, color: 'hsl(var(--accent))' }}>
                  {formatScore(validationOutput.scores.overall)}
                </div>
              </div>
              <div style={{ padding: '0.65rem', borderRadius: '8px', background: 'hsl(var(--muted) / 0.3)', textAlign: 'center' }}>
                <div style={{ fontSize: '0.7rem', color: 'hsl(var(--muted-foreground))', fontWeight: 600 }}>ACCURACY</div>
                <div style={{ fontSize: '1.15rem', fontWeight: 800, color: 'hsl(140 70% 45%)' }}>
                  {formatScore(validationOutput.scores.accuracy)}
                </div>
              </div>
              <div style={{ padding: '0.65rem', borderRadius: '8px', background: 'hsl(var(--muted) / 0.3)', textAlign: 'center' }}>
                <div style={{ fontSize: '0.7rem', color: 'hsl(var(--muted-foreground))', fontWeight: 600 }}>CANDIDATE MATCH</div>
                <div style={{ fontSize: '1.15rem', fontWeight: 800, color: 'hsl(210 80% 55%)' }}>
                  {formatScore(validationOutput.scores.candidate_validity)}
                </div>
              </div>
              <div style={{ padding: '0.65rem', borderRadius: '8px', background: 'hsl(var(--muted) / 0.3)', textAlign: 'center' }}>
                <div style={{ fontSize: '0.7rem', color: 'hsl(var(--muted-foreground))', fontWeight: 600 }}>SCHEMA VALIDITY</div>
                <div style={{ fontSize: '1.15rem', fontWeight: 800, color: 'hsl(var(--foreground))' }}>
                  {formatScore(validationOutput.scores.schema_validity)}
                </div>
              </div>
              <div style={{ padding: '0.65rem', borderRadius: '8px', background: 'hsl(var(--muted) / 0.3)', textAlign: 'center' }}>
                <div style={{ fontSize: '0.7rem', color: 'hsl(var(--muted-foreground))', fontWeight: 600 }}>JSON VALIDITY</div>
                <div style={{ fontSize: '1.15rem', fontWeight: 800, color: 'hsl(var(--foreground))' }}>
                  {formatScore(validationOutput.scores.json_validity)}
                </div>
              </div>
            </div>

            {/* Interactive Agenda Groups */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              <div style={{ fontSize: '0.84rem', fontWeight: 700, color: 'hsl(var(--foreground))', display: 'flex', alignItems: 'center', gap: '6px' }}>
                <Layers size={15} color="hsl(var(--accent))" />
                Assigned Discussion Points by Agenda (Use dropdown to move points):
              </div>

              {validationOutput.grouped_by_agenda.map((group) => (
                <div
                  key={group.agenda_id}
                  style={{
                    borderRadius: '10px',
                    border: '1.5px solid hsl(var(--border))',
                    background: 'hsl(var(--card))',
                    overflow: 'hidden',
                  }}
                >
                  {/* Group header */}
                  <div style={{
                    padding: '0.65rem 0.9rem',
                    background: 'hsl(var(--muted) / 0.35)',
                    borderBottom: '1px solid hsl(var(--border) / 0.6)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                  }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <span style={{
                        fontSize: '0.74rem', fontWeight: 800,
                        background: 'hsl(210 80% 55% / 0.15)', color: 'hsl(210 80% 55%)',
                        padding: '2px 7px', borderRadius: '5px',
                      }}>
                        [{group.agenda_id}]
                      </span>
                      <span style={{ fontSize: '0.84rem', fontWeight: 700, color: 'hsl(var(--foreground))' }}>
                        {group.title}
                      </span>
                    </div>

                    <span style={{ fontSize: '0.72rem', fontWeight: 600, color: 'hsl(var(--muted-foreground))' }}>
                      {group.points.length} Point(s)
                    </span>
                  </div>

                  {/* Group points */}
                  <div style={{ padding: '0.85rem', display: 'flex', flexDirection: 'column', gap: '0.65rem' }}>
                    {group.points.length === 0 ? (
                      <div style={{ fontSize: '0.75rem', color: 'hsl(var(--muted-foreground))', fontStyle: 'italic', padding: '4px' }}>
                        No discussion points assigned to this agenda.
                      </div>
                    ) : (
                      group.points.map((pt) => (
                        <div
                          key={pt.point_id}
                          style={{
                            padding: '0.65rem 0.85rem',
                            borderRadius: '8px',
                            background: 'hsl(var(--background))',
                            border: '1px solid hsl(var(--border) / 0.5)',
                            display: 'flex',
                            flexDirection: 'column',
                            gap: '6px',
                          }}
                        >
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                              <span style={{ fontSize: '0.72rem', fontWeight: 700, color: 'hsl(var(--accent))' }}>
                                {pt.point_id}
                              </span>
                              <span style={{
                                fontSize: '0.65rem', fontWeight: 700, padding: '1px 6px', borderRadius: '4px',
                                background: pt.confidence === 'high' ? 'hsl(140 70% 45% / 0.15)' : 'hsl(38 92% 50% / 0.15)',
                                color: pt.confidence === 'high' ? 'hsl(140 70% 45%)' : 'hsl(38 92% 40%)',
                              }}>
                                Confidence: {pt.confidence}
                              </span>
                            </div>

                            {/* Move point dropdown */}
                            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                              <span style={{ fontSize: '0.7rem', fontWeight: 600, color: 'hsl(var(--muted-foreground))', display: 'flex', alignItems: 'center', gap: '3px' }}>
                                <MoveRight size={11} /> Move to:
                              </span>
                              <select
                                value={group.agenda_id}
                                disabled={movingPointId === pt.point_id}
                                onChange={(e) => handleMovePoint(pt, group.agenda_id, e.target.value)}
                                style={{
                                  padding: '2px 6px', borderRadius: '5px',
                                  border: '1px solid hsl(var(--accent) / 0.4)', background: 'hsl(var(--card))',
                                  fontSize: '0.72rem', fontWeight: 600, color: 'hsl(var(--foreground))',
                                  cursor: 'pointer',
                                }}
                              >
                                {agendas.map((a) => (
                                  <option key={a.agenda_id} value={a.agenda_id}>
                                    [{a.agenda_id}] {a.title}
                                  </option>
                                ))}
                              </select>
                            </div>
                          </div>

                          <p style={{ margin: 0, fontSize: '0.78rem', color: 'hsl(var(--foreground))', lineHeight: 1.45 }}>
                            {pt.enhanced_point}
                          </p>

                          {pt.reason && (
                            <div style={{ fontSize: '0.7rem', color: 'hsl(var(--muted-foreground))', fontStyle: 'italic', display: 'flex', gap: '4px' }}>
                              <span>Reason:</span> <span>{pt.reason}</span>
                            </div>
                          )}

                          {/* Candidate agendas similarity chips */}
                          {pt.candidate_agendas && pt.candidate_agendas.length > 0 && (
                            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px', marginTop: '2px', alignItems: 'center' }}>
                              <span style={{ fontSize: '0.65rem', color: 'hsl(var(--muted-foreground))' }}>Candidates:</span>
                              {pt.candidate_agendas.map((cand, cIdx) => (
                                <span
                                  key={cIdx}
                                  style={{
                                    fontSize: '0.66rem', padding: '1px 5px', borderRadius: '4px',
                                    background: cand.agenda_id === group.agenda_id ? 'hsl(var(--accent) / 0.15)' : 'hsl(var(--muted))',
                                    color: cand.agenda_id === group.agenda_id ? 'hsl(var(--accent))' : 'hsl(var(--muted-foreground))',
                                  }}
                                >
                                  [{cand.agenda_id}] {cand.agenda_title} ({Math.round(cand.score * 100)}%)
                                </span>
                              ))}
                            </div>
                          )}
                        </div>
                      ))
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
