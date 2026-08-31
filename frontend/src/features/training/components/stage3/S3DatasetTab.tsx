import { useState, useRef } from 'react';
import {
  FileText,
  Upload,
  Layers,
  Sparkles,
  CheckCircle,
  AlertTriangle,
  Clock,
  User,
  Tag,
  Loader,
  Calendar,
  ChevronRight,
  ChevronDown,
  CheckSquare,
  FilePlus,
  Trash2,
  Sliders,
  RefreshCw,
  Eye,
} from 'lucide-react';
import { useStage3Store } from '../../store/stage3Store';
import {
  fetchStage3MeetingData,
  previewStage3Batches,
  extractStage3File,
  generateStage3Agendas,
} from '../../api/stage3Api';
import type { Stage3Agenda, Stage3Batch } from '../../types/training';

const S = {
  card: {
    background: 'hsl(var(--card))',
    borderRadius: '12px',
    border: '1px solid hsl(var(--border))',
    padding: '1.25rem',
  } as React.CSSProperties,
  label: {
    display: 'block',
    fontSize: '0.8rem',
    fontWeight: 600,
    color: 'hsl(var(--muted-foreground))',
    marginBottom: '0.4rem',
    textTransform: 'uppercase' as const,
    letterSpacing: '0.04em',
  },
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

export default function S3DatasetTab() {
  const {
    meetings,
    datasetMeetingId,
    setDatasetMeetingId,
    datasetMeetingFilename,
    setDatasetMeetingFilename,
    stage2Points,
    setStage2Points,
    agendas,
    setAgendas,
    hasStage2,
    setHasStage2,
    hasAgendas,
    setHasAgendas,
    setGroundTruthMappings,
    pointsPerBatch,
    setPointsPerBatch,
    batches,
    setBatches,
    batchesLoading,
    setBatchesLoading,
  } = useStage3Store();

  const [loadingMeetingData, setLoadingMeetingData] = useState(false);
  const [expandedBatchIdx, setExpandedBatchIdx] = useState<number | null>(0);

  // Agenda / MoM Generation (Stage 3 Step 1)
  const [showAgendaCreator, setShowAgendaCreator] = useState(false);
  const [agendaText, setAgendaText] = useState('');
  const [agendaFileName, setAgendaFileName] = useState('');
  const [extractingAgenda, setExtractingAgenda] = useState(false);
  const [previousMoms, setPreviousMoms] = useState<{ filename: string; text: string }[]>([]);
  const [extractingMom, setExtractingMom] = useState(false);
  const [useGlobalContext, setUseGlobalContext] = useState(true);
  const [globalContextTopK, setGlobalContextTopK] = useState(5);
  const [meetingContextTopK, setMeetingContextTopK] = useState(5);
  const [generatingAgendas, setGeneratingAgendas] = useState(false);
  const [agendaGenError, setAgendaGenError] = useState<string | null>(null);

  const agendaFileInputRef = useRef<HTMLInputElement>(null);
  const momFileInputRef = useRef<HTMLInputElement>(null);

  const handleSelectMeeting = async (meetingId: string, filename: string) => {
    setDatasetMeetingId(meetingId);
    setDatasetMeetingFilename(filename);
    setLoadingMeetingData(true);
    setAgendaGenError(null);
    setShowAgendaCreator(false);
    setAgendaText('');
    setAgendaFileName('');
    setPreviousMoms([]);

    try {
      const data = await fetchStage3MeetingData(meetingId);
      setStage2Points(data.stage2_points || []);
      setAgendas(data.agendas || []);
      setHasStage2(data.has_stage2);
      setHasAgendas(data.has_agendas);
      setGroundTruthMappings(data.ground_truth_mappings || {});

      if (!data.has_agendas) {
        setShowAgendaCreator(true);
      }

      // Auto preview batches if both stage 2 points and agendas are ready
      if (data.has_stage2 && data.has_agendas) {
        setBatchesLoading(true);
        try {
          const batchRes = await previewStage3Batches({
            meeting_id: meetingId,
            batch_size: pointsPerBatch,
          });
          setBatches(batchRes.batches || []);
        } catch (bErr) {
          console.error('Failed to auto-preview batches:', bErr);
        } finally {
          setBatchesLoading(false);
        }
      } else {
        setBatches([]);
      }
    } catch (err: any) {
      console.error('Failed to load meeting data for Stage 3:', err);
    } finally {
      setLoadingMeetingData(false);
    }
  };

  // Extract agenda text from uploaded file
  const handleAgendaFileSelect = async (file: File) => {
    setExtractingAgenda(true);
    setAgendaGenError(null);
    try {
      const res = await extractStage3File(file);
      setAgendaText(res.extracted_text);
      setAgendaFileName(res.filename);
    } catch (err: any) {
      setAgendaGenError(err?.response?.data?.detail || err?.message || 'Failed to extract text from agenda file.');
    } finally {
      setExtractingAgenda(false);
    }
  };

  // Extract previous MoM text from uploaded file
  const handleMomFileSelect = async (file: File) => {
    setExtractingMom(true);
    setAgendaGenError(null);
    try {
      const res = await extractStage3File(file);
      setPreviousMoms((prev) => [...prev, { filename: res.filename, text: res.extracted_text }]);
    } catch (err: any) {
      setAgendaGenError(err?.response?.data?.detail || err?.message || 'Failed to extract text from MoM file.');
    } finally {
      setExtractingMom(false);
    }
  };

  // Generate Agendas with Global Context (Stage 3 Step 1)
  const handleGenerateAgendas = async () => {
    if (!datasetMeetingId || !agendaText.trim()) return;
    setGeneratingAgendas(true);
    setAgendaGenError(null);
    try {
      const res = await generateStage3Agendas({
        meeting_id: datasetMeetingId,
        agenda_text: agendaText,
        previous_mom_texts: previousMoms.map((m) => m.text),
        use_global_context: useGlobalContext,
        global_context_top_k: globalContextTopK,
        meeting_context_top_k: meetingContextTopK,
      });

      setAgendas(res.agendas || []);
      setHasAgendas((res.agendas || []).length > 0);
      setShowAgendaCreator(false);

      // Auto preview batches with Stage 2 points
      if (stage2Points.length > 0 && res.agendas.length > 0) {
        setBatchesLoading(true);
        const batchRes = await previewStage3Batches({
          meeting_id: datasetMeetingId,
          batch_size: pointsPerBatch,
        });
        setBatches(batchRes.batches || []);
        setBatchesLoading(false);
      }
    } catch (err: any) {
      setAgendaGenError(err?.response?.data?.detail || err?.message || 'Failed to generate agendas.');
    } finally {
      setGeneratingAgendas(false);
    }
  };

  const handlePreviewBatches = async () => {
    if (!datasetMeetingId) return;
    setBatchesLoading(true);
    try {
      const res = await previewStage3Batches({
        meeting_id: datasetMeetingId,
        batch_size: pointsPerBatch,
      });
      setBatches(res.batches || []);
    } catch (err: any) {
      console.error('Failed to preview Stage 3 batches:', err);
    } finally {
      setBatchesLoading(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', width: '100%' }}>
      {/* ── Section 1: Meeting Selection ── */}
      <div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '0.75rem' }}>
          <div style={{
            width: 28, height: 28, borderRadius: '8px',
            background: 'hsl(var(--accent) / .15)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <FileText size={14} color="hsl(var(--accent))" />
          </div>
          <h3 style={{ margin: 0, fontSize: '0.95rem', fontWeight: 700 }}>Select Meeting for Stage 3 Training</h3>
        </div>

        {meetings.length === 0 ? (
          <div style={{ ...S.card, textAlign: 'center', color: 'hsl(var(--muted-foreground))', padding: '2rem' }}>
            No processed meetings found. Process a recording first.
          </div>
        ) : (
          <div style={{
            display: 'flex', flexDirection: 'column', gap: '0.5rem',
            maxHeight: '240px', overflowY: 'auto', paddingRight: '4px',
          }}>
            {meetings.map((m: any) => {
              const meetingId = m.id || m.recording_id;
              const isSelected = datasetMeetingId === meetingId;
              return (
                <div
                  key={meetingId}
                  onClick={() => handleSelectMeeting(meetingId, m.filename)}
                  style={{
                    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                    padding: '0.75rem 1rem', borderRadius: '10px',
                    border: `1.5px solid ${isSelected ? 'hsl(var(--accent))' : 'hsl(var(--border))'}`,
                    background: isSelected ? 'hsl(var(--accent) / 0.06)' : 'hsl(var(--card))',
                    cursor: 'pointer', transition: 'all 0.15s ease',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px', minWidth: 0 }}>
                    <div style={{
                      width: 32, height: 32, borderRadius: '8px',
                      background: isSelected ? 'hsl(var(--accent))' : 'hsl(var(--muted))',
                      color: isSelected ? 'white' : 'hsl(var(--muted-foreground))',
                      display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0,
                    }}>
                      <Calendar size={15} />
                    </div>
                    <div style={{ minWidth: 0 }}>
                      <div style={{
                        fontSize: '0.88rem', fontWeight: isSelected ? 700 : 500,
                        color: 'hsl(var(--foreground))', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                      }}>
                        {m.filename}
                      </div>
                      <div style={{ fontSize: '0.74rem', color: 'hsl(var(--muted-foreground))', display: 'flex', gap: '8px' }}>
                        <span>Segments: {m.transcript_segments || m.segment_count || '—'}</span>
                        <span>•</span>
                        <span>Stage 2 Points: {m.stage2_points ?? (m.stages_available?.includes('stage_2') ? 'Available' : 'None')}</span>
                        <span>•</span>
                        <span>Agendas: {m.stage3_agendas ?? (m.stages_available?.includes('stage_3') ? 'Available' : 'None')}</span>
                      </div>
                    </div>
                  </div>

                  <div style={{ display: 'flex', gap: '6px', alignItems: 'center' }}>
                    {m.has_stage2 || m.stages_available?.includes('stage_2') ? (
                      <span style={{
                        fontSize: '0.68rem', fontWeight: 700, padding: '2px 7px', borderRadius: '6px',
                        background: 'hsl(140 70% 45% / 0.15)', color: 'hsl(140 70% 45%)',
                      }}>
                        Stage 2 Ready
                      </span>
                    ) : (
                      <span style={{
                        fontSize: '0.68rem', fontWeight: 700, padding: '2px 7px', borderRadius: '6px',
                        background: 'hsl(var(--muted))', color: 'hsl(var(--muted-foreground))',
                      }}>
                        No Stage 2
                      </span>
                    )}

                    {m.has_agenda || m.has_rom_data ? (
                      <span style={{
                        fontSize: '0.68rem', fontWeight: 700, padding: '2px 7px', borderRadius: '6px',
                        background: 'hsl(210 80% 55% / 0.15)', color: 'hsl(210 80% 55%)',
                      }}>
                        Agendas Available
                      </span>
                    ) : (
                      <span style={{
                        fontSize: '0.68rem', fontWeight: 700, padding: '2px 7px', borderRadius: '6px',
                        background: 'hsl(38 92% 50% / 0.15)', color: 'hsl(38 92% 40%)',
                      }}>
                        Upload Agenda
                      </span>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* ── Section 2: Stage 2 Points & Agenda Status ── */}
      {datasetMeetingId && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
          {/* Stage 2 Points Card */}
          <div style={{ ...S.card, display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <Sparkles size={15} color="hsl(var(--accent))" />
                <span style={{ fontSize: '0.85rem', fontWeight: 700 }}>Stage 2 Enhanced Points</span>
              </div>
              <span style={{
                fontSize: '0.75rem', fontWeight: 700, padding: '2px 8px', borderRadius: '6px',
                background: stage2Points.length > 0 ? 'hsl(140 70% 45% / 0.15)' : 'hsl(0 75% 55% / 0.15)',
                color: stage2Points.length > 0 ? 'hsl(140 70% 45%)' : 'hsl(0 75% 55%)',
              }}>
                {stage2Points.length} Point(s)
              </span>
            </div>

            {stage2Points.length === 0 ? (
              <div style={{
                padding: '0.85rem', borderRadius: '8px', background: 'hsl(0 75% 55% / 0.08)',
                border: '1px solid hsl(0 75% 55% / 0.25)', fontSize: '0.78rem', color: 'hsl(var(--foreground))',
                lineHeight: 1.45,
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontWeight: 700, color: 'hsl(0 75% 55%)', marginBottom: '3px' }}>
                  <AlertTriangle size={13} /> Stage 2 Points Missing
                </div>
                This meeting has no Stage 2 enhanced discussion points. Run Stage 2 on the main processing page before training.
              </div>
            ) : (
              <div style={{
                fontSize: '0.76rem', color: 'hsl(var(--muted-foreground))',
                maxHeight: '140px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '4px',
              }}>
                {stage2Points.slice(0, 4).map((p, i) => (
                  <div key={i} style={{ padding: '4px 6px', borderRadius: '4px', background: 'hsl(var(--muted) / 0.4)' }}>
                    <strong>P{i + 1}:</strong> {p.polished_text || p.point || p.discussion_point}
                  </div>
                ))}
                {stage2Points.length > 4 && (
                  <div style={{ fontStyle: 'italic', textAlign: 'center', marginTop: '2px' }}>
                    + {stage2Points.length - 4} more point(s) loaded
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Agenda Card */}
          <div style={{ ...S.card, display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <Layers size={15} color="hsl(210, 80%, 55%)" />
                <span style={{ fontSize: '0.85rem', fontWeight: 700 }}>Meeting Agendas</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <span style={{
                  fontSize: '0.75rem', fontWeight: 700, padding: '2px 8px', borderRadius: '6px',
                  background: agendas.length > 0 ? 'hsl(140 70% 45% / 0.15)' : 'hsl(38 92% 50% / 0.15)',
                  color: agendas.length > 0 ? 'hsl(140 70% 45%)' : 'hsl(38 92% 40%)',
                }}>
                  {agendas.length} Agenda(s)
                </span>
                {agendas.length > 0 && (
                  <button
                    onClick={() => setShowAgendaCreator(!showAgendaCreator)}
                    style={{
                      padding: '2px 7px', borderRadius: '5px', border: '1px solid hsl(var(--border))',
                      background: 'hsl(var(--background))', fontSize: '0.72rem', cursor: 'pointer',
                    }}
                  >
                    {showAgendaCreator ? 'Hide Creator' : 'Re-extract / Upload'}
                  </button>
                )}
              </div>
            </div>

            {agendas.length > 0 && !showAgendaCreator ? (
              <div style={{
                fontSize: '0.76rem', color: 'hsl(var(--foreground))',
                maxHeight: '140px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '4px',
              }}>
                {agendas.map((a, i) => (
                  <div key={i} style={{ padding: '4px 6px', borderRadius: '4px', background: 'hsl(var(--muted) / 0.4)', display: 'flex', gap: '6px' }}>
                    <span style={{ fontWeight: 700, color: 'hsl(var(--accent))' }}>[{a.agenda_id}]:</span>
                    <span>{a.title}</span>
                  </div>
                ))}
              </div>
            ) : (
              <div style={{ fontSize: '0.78rem', color: 'hsl(var(--muted-foreground))' }}>
                {agendas.length === 0 && (
                  <div style={{ display: 'flex', alignItems: 'center', gap: '5px', fontWeight: 700, color: 'hsl(38 92% 40%)', marginBottom: '4px' }}>
                    <AlertTriangle size={13} /> Agenda File Required
                  </div>
                )}
                Upload an agenda file, optional previous MoMs, and enhance details using Global Context.
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── Section 3: Stage 3 Step 1 — Agenda File, MoM Upload & Global Context Enrichment ── */}
      {datasetMeetingId && (showAgendaCreator || agendas.length === 0) && (
        <div style={{ ...S.card, border: '1.5px solid hsl(var(--accent) / 0.4)', display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div>
              <h3 style={{ margin: 0, fontSize: '0.92rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '6px' }}>
                <FilePlus size={16} color="hsl(var(--accent))" />
                Step 1: Upload Agenda, MoM &amp; Enhance with Global Context
              </h3>
              <p style={{ margin: '2px 0 0', fontSize: '0.76rem', color: 'hsl(var(--muted-foreground))' }}>
                Extracts agendas, enriches details using Global Context RAG and previous meeting MoMs (matches Stage 3 Step 1 pipeline).
              </p>
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
            {/* Left: Agenda File Upload & Text */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
              <label style={S.label}>1. Agenda File / Source Text</label>

              <div style={{ display: 'flex', gap: '6px', alignItems: 'center' }}>
                <input
                  ref={agendaFileInputRef}
                  type="file"
                  accept=".pdf,.docx,.txt"
                  style={{ display: 'none' }}
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    if (f) handleAgendaFileSelect(f);
                  }}
                />
                <button
                  type="button"
                  disabled={extractingAgenda}
                  onClick={() => agendaFileInputRef.current?.click()}
                  style={S.btn('primary', extractingAgenda)}
                >
                  {extractingAgenda ? <Loader size={13} className="spin" /> : <Upload size={13} />}
                  {extractingAgenda ? 'Extracting File...' : 'Upload Agenda File (PDF/DOCX/TXT)'}
                </button>
                {agendaFileName && (
                  <span style={{ fontSize: '0.74rem', color: 'hsl(var(--muted-foreground))', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {agendaFileName}
                  </span>
                )}
              </div>

              <textarea
                value={agendaText}
                onChange={(e) => setAgendaText(e.target.value)}
                placeholder="Paste or edit extracted agenda text here (e.g. 1. Architecture Review\n2. Q3 Roadmap\n3. Budget Approval)..."
                rows={5}
                style={{
                  width: '100%', borderRadius: '8px', border: '1px solid hsl(var(--border))',
                  background: 'hsl(var(--background))', color: 'hsl(var(--foreground))',
                  fontSize: '0.78rem', padding: '0.6rem', fontFamily: 'inherit', resize: 'vertical',
                }}
              />
            </div>

            {/* Right: Previous MoM Upload & Context RAG */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
              <div>
                <label style={S.label}>2. Previous MoM Files (Optional Context)</label>
                <input
                  ref={momFileInputRef}
                  type="file"
                  accept=".pdf,.docx,.txt"
                  style={{ display: 'none' }}
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    if (f) handleMomFileSelect(f);
                  }}
                />
                <button
                  type="button"
                  disabled={extractingMom || previousMoms.length >= 5}
                  onClick={() => momFileInputRef.current?.click()}
                  style={{ ...S.btn('ghost', extractingMom || previousMoms.length >= 5), padding: '0.45rem 0.85rem' }}
                >
                  {extractingMom ? <Loader size={12} className="spin" /> : <Upload size={12} />}
                  {extractingMom ? 'Extracting MoM...' : `Upload Previous MoM (${previousMoms.length}/5)`}
                </button>

                {previousMoms.length > 0 && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', marginTop: '6px' }}>
                    {previousMoms.map((m, idx) => (
                      <div key={idx} style={{
                        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                        padding: '3px 8px', borderRadius: '5px', background: 'hsl(var(--muted) / 0.5)',
                        fontSize: '0.72rem',
                      }}>
                        <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{m.filename}</span>
                        <button
                          type="button"
                          onClick={() => setPreviousMoms((prev) => prev.filter((_, i) => i !== idx))}
                          style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'hsl(var(--destructive))', padding: '2px' }}
                        >
                          <Trash2 size={12} />
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Global Context Enhancement Toggle */}
              <div style={{
                padding: '0.75rem', borderRadius: '8px', background: 'hsl(var(--muted) / 0.3)',
                border: '1px solid hsl(var(--border) / 0.6)', display: 'flex', flexDirection: 'column', gap: '6px',
              }}>
                <label style={{ display: 'flex', alignItems: 'center', gap: '6px', cursor: 'pointer', fontSize: '0.78rem', fontWeight: 600 }}>
                  <input
                    type="checkbox"
                    checked={useGlobalContext}
                    onChange={(e) => setUseGlobalContext(e.target.checked)}
                  />
                  Enhance Agendas using Global Context &amp; Meeting Context
                </label>

                {useGlobalContext && (
                  <div style={{ display: 'flex', gap: '12px', fontSize: '0.72rem', color: 'hsl(var(--muted-foreground))', marginTop: '2px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                      <span>Global Top-K:</span>
                      <select
                        value={globalContextTopK}
                        onChange={(e) => setGlobalContextTopK(Number(e.target.value))}
                        style={{ padding: '1px 4px', borderRadius: '4px', border: '1px solid hsl(var(--border))', background: 'hsl(var(--background))' }}
                      >
                        <option value={3}>3</option>
                        <option value={5}>5</option>
                        <option value={8}>8</option>
                      </select>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                      <span>Meeting Top-K:</span>
                      <select
                        value={meetingContextTopK}
                        onChange={(e) => setMeetingContextTopK(Number(e.target.value))}
                        style={{ padding: '1px 4px', borderRadius: '4px', border: '1px solid hsl(var(--border))', background: 'hsl(var(--background))' }}
                      >
                        <option value={3}>3</option>
                        <option value={5}>5</option>
                        <option value={8}>8</option>
                      </select>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>

          {agendaGenError && (
            <div style={{
              padding: '0.65rem 0.85rem', borderRadius: '8px',
              background: 'hsl(0 75% 55% / 0.1)', border: '1px solid hsl(0 75% 55% / 0.3)',
              color: 'hsl(0 75% 50%)', fontSize: '0.78rem',
            }}>
              {agendaGenError}
            </div>
          )}

          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '8px' }}>
            {agendas.length > 0 && (
              <button
                type="button"
                onClick={() => setShowAgendaCreator(false)}
                style={S.btn('ghost')}
              >
                Cancel
              </button>
            )}
            <button
              type="button"
              disabled={generatingAgendas || !agendaText.trim()}
              onClick={handleGenerateAgendas}
              style={S.btn('success', generatingAgendas || !agendaText.trim())}
            >
              {generatingAgendas ? <Loader size={13} className="spin" /> : <Sparkles size={13} />}
              {generatingAgendas ? 'Enhancing & Generating Agendas...' : 'Enhance & Generate Agendas (Step 1)'}
            </button>
          </div>
        </div>
      )}

      {/* ── Section 4: Batch Configuration & Candidate Preview ── */}
      {datasetMeetingId && hasStage2 && hasAgendas && (
        <div style={{ ...S.card, display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div>
              <h3 style={{ margin: 0, fontSize: '0.92rem', fontWeight: 700 }}>
                Step 2: Point → Agenda Batches (10–20 Points per Batch)
              </h3>
              <p style={{ margin: '2px 0 0', fontSize: '0.76rem', color: 'hsl(var(--muted-foreground))' }}>
                Cosine similarity matches each Stage 2 point with the Top-3 candidate agendas.
              </p>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <span style={{ fontSize: '0.78rem', fontWeight: 600, color: 'hsl(var(--muted-foreground))' }}>
                  Batch Size:
                </span>
                <select
                  value={pointsPerBatch}
                  onChange={(e) => setPointsPerBatch(Number(e.target.value))}
                  style={{
                    padding: '0.35rem 0.65rem', borderRadius: '6px',
                    border: '1px solid hsl(var(--border))', background: 'hsl(var(--background))',
                    fontSize: '0.8rem', color: 'hsl(var(--foreground))',
                  }}
                >
                  <option value={10}>10 points</option>
                  <option value={15}>15 points</option>
                  <option value={20}>20 points</option>
                </select>
              </div>

              <button
                disabled={batchesLoading}
                onClick={handlePreviewBatches}
                style={S.btn('primary', batchesLoading)}
              >
                {batchesLoading ? <Loader size={13} className="spin" /> : <Sparkles size={13} />}
                {batchesLoading ? 'Computing...' : 'Recalculate Batches'}
              </button>
            </div>
          </div>

          {/* Batches Preview */}
          {batches.length === 0 ? (
            <div style={{ textAlign: 'center', padding: '1.5rem', color: 'hsl(var(--muted-foreground))', fontSize: '0.82rem' }}>
              Click <strong>Recalculate Batches</strong> to compute cosine similarity and generate assignment batches.
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
              {batches.map((b, bIdx) => {
                const isExpanded = expandedBatchIdx === bIdx;
                return (
                  <div
                    key={bIdx}
                    style={{
                      borderRadius: '10px',
                      border: '1px solid hsl(var(--border) / 0.7)',
                      background: 'hsl(var(--background))',
                      overflow: 'hidden',
                    }}
                  >
                    {/* Batch header */}
                    <div
                      onClick={() => setExpandedBatchIdx(isExpanded ? null : bIdx)}
                      style={{
                        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                        padding: '0.75rem 1rem', cursor: 'pointer', background: 'hsl(var(--muted) / 0.25)',
                      }}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <span style={{
                          fontSize: '0.75rem', fontWeight: 800, color: 'hsl(var(--accent))',
                          background: 'hsl(var(--accent) / 0.12)', padding: '2px 7px', borderRadius: '5px',
                        }}>
                          Batch {bIdx + 1}
                        </span>
                        <span style={{ fontSize: '0.82rem', fontWeight: 600, color: 'hsl(var(--foreground))' }}>
                          {b.point_count} Discussion Point(s)
                        </span>
                      </div>

                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <span style={{ fontSize: '0.72rem', color: 'hsl(var(--muted-foreground))' }}>
                          Top-3 candidate agendas calculated
                        </span>
                        {isExpanded ? <ChevronDown size={15} /> : <ChevronRight size={15} />}
                      </div>
                    </div>

                    {/* Batch points & candidates */}
                    {isExpanded && (
                      <div style={{ padding: '0.85rem 1rem', display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                        {b.points.map((pt, pIdx) => (
                          <div
                            key={pt.point_id || pIdx}
                            style={{
                              padding: '0.7rem 0.85rem',
                              borderRadius: '8px',
                              background: 'hsl(var(--card))',
                              border: '1px solid hsl(var(--border) / 0.5)',
                              display: 'flex',
                              flexDirection: 'column',
                              gap: '6px',
                            }}
                          >
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                              <span style={{ fontSize: '0.74rem', fontWeight: 700, color: 'hsl(var(--accent))' }}>
                                Point #{pIdx + 1} ({pt.point_id})
                              </span>
                              {pt.speakers && pt.speakers.length > 0 && (
                                <div style={{ display: 'flex', gap: '4px' }}>
                                  {pt.speakers.map((sp, idx) => (
                                    <span key={idx} style={{
                                      fontSize: '0.68rem', padding: '1px 6px', borderRadius: '6px',
                                      background: 'hsl(var(--muted))', color: 'hsl(var(--muted-foreground))',
                                      display: 'inline-flex', alignItems: 'center', gap: '3px',
                                    }}>
                                      <User size={9} /> {sp}
                                    </span>
                                  ))}
                                </div>
                              )}
                            </div>

                            <p style={{ margin: 0, fontSize: '0.8rem', color: 'hsl(var(--foreground))', lineHeight: 1.45 }}>
                              {pt.enhanced_point}
                            </p>

                            {/* Candidate agendas chips */}
                            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', alignItems: 'center', marginTop: '2px' }}>
                              <span style={{ fontSize: '0.7rem', fontWeight: 700, color: 'hsl(var(--muted-foreground))' }}>
                                Top-3 Similarity Matches:
                              </span>
                              {pt.candidate_agendas.map((cand, cIdx) => (
                                <span
                                  key={cIdx}
                                  style={{
                                    fontSize: '0.7rem', fontWeight: 600,
                                    padding: '2px 7px', borderRadius: '6px',
                                    background: cIdx === 0 ? 'hsl(var(--accent) / 0.15)' : 'hsl(var(--muted))',
                                    color: cIdx === 0 ? 'hsl(var(--accent))' : 'hsl(var(--foreground))',
                                    border: `1px solid ${cIdx === 0 ? 'hsl(var(--accent) / 0.3)' : 'hsl(var(--border))'}`,
                                    display: 'inline-flex', alignItems: 'center', gap: '4px',
                                  }}
                                >
                                  <strong>[{cand.agenda_id}]</strong> {cand.agenda_title}
                                  <span style={{ opacity: 0.7, fontSize: '0.65rem' }}>({Math.round(cand.score * 100)}%)</span>
                                </span>
                              ))}
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}

          {batches.length > 0 && (
            <div style={{
              padding: '0.75rem 1rem', borderRadius: '8px',
              background: 'hsl(var(--accent) / 0.08)', border: '1px solid hsl(var(--accent) / 0.25)',
              fontSize: '0.8rem', color: 'hsl(var(--foreground))', display: 'flex', alignItems: 'center', gap: '8px',
            }}>
              <CheckSquare size={16} color="hsl(var(--accent))" />
              <div>
                <strong>{batches.length} batch(es) prepared ({stage2Points.length} total points).</strong> Switch to the <strong>Training Loop</strong> tab to train or validate.
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
