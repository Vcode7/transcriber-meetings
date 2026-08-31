import { useState, useEffect } from 'react';
import {
  Calendar, Clock, FileText, RefreshCw, ChevronRight, ChevronDown,
  Layers, Upload, CheckCircle2, AlertCircle, FileCheck, Sliders,
  Sparkles, ShieldCheck, ArrowRight
} from 'lucide-react';
import { useStage2Store } from '../../store/stage2Store';
import {
  fetchStage1PointsForMeeting,
  previewStage2Groups,
  uploadMomForStage2
} from '../../api/stage2Api';

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
  btn: (variant: 'primary' | 'ghost' | 'success' = 'primary') => ({
    display: 'inline-flex',
    alignItems: 'center',
    gap: '6px',
    padding: '0.55rem 1.1rem',
    borderRadius: '8px',
    border: 'none',
    cursor: 'pointer',
    fontSize: '0.85rem',
    fontWeight: 600,
    transition: 'all 0.15s ease',
    background:
      variant === 'primary' ? 'hsl(var(--accent))' :
      variant === 'success' ? 'hsl(142 76% 36%)' :
      'hsl(var(--muted))',
    color: variant === 'ghost' ? 'hsl(var(--foreground))' : 'white',
  } as React.CSSProperties),
};

export default function S2DatasetTab() {
  const {
    meetings,
    datasetMeetingId,
    setDatasetMeetingId,
    datasetMeetingFilename,
    setDatasetMeetingFilename,
    stage1Points,
    setStage1Points,
    stage1PointsLoading,
    setStage1PointsLoading,
    pointsPerGroup,
    setPointsPerGroup,
    contextRetrieval,
    setContextRetrieval,
    groups,
    setGroups,
    groupsLoading,
    setGroupsLoading,
    hasStage2Edits,
    setHasStage2Edits,
    referenceSource,
    setReferenceSource,
    referencePoints,
    setReferencePoints,
    momFilename,
    setMomFilename,
    setMomExtractedText,
    momPoints,
    setMomPoints,
    momUploading,
    setMomUploading,
    setActiveTab,
  } = useStage2Store();

  const [expandedGroupIdx, setExpandedGroupIdx] = useState<number | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  // When meeting is selected, load Stage 1 points & reference info
  const handleSelectMeeting = async (id: string, filename: string) => {
    setDatasetMeetingId(id);
    setDatasetMeetingFilename(filename);
    setStage1PointsLoading(true);
    setUploadError(null);

    try {
      const data = await fetchStage1PointsForMeeting(id);
      setStage1Points(data.stage1_points || []);
      setHasStage2Edits(data.has_stage2_edits || false);
      setReferenceSource(data.reference_source || 'none');
      setReferencePoints(data.reference_points || []);

      // Auto-preview groups if Stage 1 points exist
      if (data.stage1_points && data.stage1_points.length > 0) {
        setGroupsLoading(true);
        const groupRes = await previewStage2Groups({
          meeting_id: id,
          points_per_group: pointsPerGroup,
          context_retrieval: contextRetrieval,
        });
        setGroups(groupRes.groups || []);
      }
    } catch (err: any) {
      console.error('Failed to load meeting data for Stage 2:', err);
    } finally {
      setStage1PointsLoading(false);
      setGroupsLoading(false);
    }
  };

  // Re-run group preview when parameters change
  const handlePreviewGroups = async () => {
    if (!datasetMeetingId) return;
    setGroupsLoading(true);
    try {
      const res = await previewStage2Groups({
        meeting_id: datasetMeetingId,
        points_per_group: pointsPerGroup,
        context_retrieval: contextRetrieval,
      });
      setGroups(res.groups || []);
    } catch (err: any) {
      console.error('Failed to preview groups:', err);
    } finally {
      setGroupsLoading(false);
    }
  };

  // Handle manual MoM file upload
  const handleFileUpload = async (file: File) => {
    setMomUploading(true);
    setUploadError(null);
    try {
      const res = await uploadMomForStage2(file);
      setMomFilename(res.filename);
      setMomExtractedText(res.extracted_text);
      setMomPoints(res.points);
      setReferencePoints(res.points);
      setReferenceSource('manual_mom');
    } catch (err: any) {
      setUploadError(err.message || 'File upload failed');
    } finally {
      setMomUploading(false);
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
          <h3 style={{ margin: 0, fontSize: '0.95rem', fontWeight: 700 }}>Select Meeting for Stage 2</h3>
        </div>

        {meetings.length === 0 ? (
          <div style={{ ...S.card, textAlign: 'center', color: 'hsl(var(--muted-foreground))', padding: '2rem' }}>
            No processed meetings found. Process a recording first.
          </div>
        ) : (
          <div style={{
            display: 'flex', flexDirection: 'column', gap: '0.5rem',
            maxHeight: '260px', overflowY: 'auto', paddingRight: '4px'
          }}>
            {meetings.map((m) => {
              const isSelected = datasetMeetingId === m.id;
              return (
                <div
                  key={m.id}
                  onClick={() => handleSelectMeeting(m.id, m.filename)}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    padding: '0.75rem 1rem',
                    borderRadius: '10px',
                    border: `1.5px solid ${isSelected ? 'hsl(var(--accent))' : 'hsl(var(--border) / 0.6)'}`,
                    background: isSelected ? 'hsl(var(--accent) / 0.06)' : 'hsl(var(--card))',
                    cursor: 'pointer',
                    transition: 'all 0.12s ease',
                  }}
                >
                  <div>
                    <div style={{ fontWeight: 600, fontSize: '0.875rem', color: 'hsl(var(--foreground))' }}>
                      {m.filename}
                    </div>
                    <div style={{ display: 'flex', gap: '1rem', fontSize: '0.75rem', color: 'hsl(var(--muted-foreground))', marginTop: '3px' }}>
                      <span style={{ display: 'flex', alignItems: 'center', gap: '3px' }}>
                        <Calendar size={11} /> {new Date(m.created_at).toLocaleDateString()}
                      </span>
                      <span style={{ display: 'flex', alignItems: 'center', gap: '3px' }}>
                        <Clock size={11} /> {Math.round(m.duration / 60)} min
                      </span>
                    </div>
                  </div>
                  <div style={{ display: 'flex', gap: '6px' }}>
                    {m.stages_available.includes('stage_1') && (
                      <span style={{
                        fontSize: '0.68rem', fontWeight: 700, padding: '2px 7px', borderRadius: '8px',
                        background: 'hsl(var(--accent) / 0.12)', color: 'hsl(var(--accent))',
                      }}>
                        Stage 1 Ready
                      </span>
                    )}
                    {m.stages_available.includes('stage_2') && (
                      <span style={{
                        fontSize: '0.68rem', fontWeight: 700, padding: '2px 7px', borderRadius: '8px',
                        background: 'hsl(142 76% 36% / 0.12)', color: 'hsl(142 76% 36%)',
                      }}>
                        Stage 2 Ready
                      </span>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* ── Section 2: Input Summary & Parameters ── */}
      {datasetMeetingId && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          {/* Status summary banner */}
          <div style={{
            ...S.card,
            background: 'hsl(var(--accent) / 0.04)',
            borderColor: 'hsl(var(--accent) / 0.25)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '1rem 1.25rem',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <div style={{
                width: 36, height: 36, borderRadius: '10px',
                background: 'hsl(var(--accent) / 0.15)',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
              }}>
                <Sparkles size={18} color="hsl(var(--accent))" />
              </div>
              <div>
                <div style={{ fontWeight: 700, fontSize: '0.92rem', color: 'hsl(var(--foreground))' }}>
                  {datasetMeetingFilename}
                </div>
                <div style={{ fontSize: '0.8rem', color: 'hsl(var(--muted-foreground))', marginTop: '2px' }}>
                  {stage1PointsLoading ? (
                    <span>Loading Stage 1 points...</span>
                  ) : (
                    <span><strong>{stage1Points.length}</strong> Stage 1 discussion points loaded • Primary Input A</span>
                  )}
                </div>
              </div>
            </div>

            <button
              onClick={handlePreviewGroups}
              disabled={groupsLoading || stage1PointsLoading || stage1Points.length === 0}
              style={S.btn(groups.length > 0 ? 'ghost' : 'primary')}
            >
              <RefreshCw size={13} className={groupsLoading ? 'animate-spin' : ''} />
              {groupsLoading ? 'Grouping...' : 'Refresh Groups'}
            </button>
          </div>

          {/* Configurable Parameters */}
          <div style={{ ...S.card, display: 'flex', flexDirection: 'column', gap: '1.2rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Sliders size={15} color="hsl(var(--accent))" />
              <h3 style={{ margin: 0, fontSize: '0.9rem', fontWeight: 700, color: 'hsl(var(--foreground))' }}>
                Stage 2 Consolidation Parameters
              </h3>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
              {/* Points per Group */}
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.4rem' }}>
                  <label style={{ ...S.label, marginBottom: 0 }}>Points per Group (Batch Size)</label>
                  <span style={{
                    fontSize: '0.85rem', fontWeight: 700,
                    background: 'hsl(var(--accent) / 0.15)', color: 'hsl(var(--accent))',
                    padding: '2px 8px', borderRadius: '6px',
                  }}>
                    {pointsPerGroup} points
                  </span>
                </div>
                <input
                  type="range"
                  min={1}
                  max={20}
                  step={1}
                  value={pointsPerGroup}
                  onChange={(e) => setPointsPerGroup(Number(e.target.value))}
                  style={{ width: '100%', accentColor: 'hsl(var(--accent))', cursor: 'pointer' }}
                />
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', color: 'hsl(var(--muted-foreground))', marginTop: '3px' }}>
                  <span>1 (individual)</span>
                  <span>5 (recommended)</span>
                  <span>20 (large batch)</span>
                </div>
              </div>

              {/* Context Retrieval Toggle */}
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.4rem' }}>
                  <label style={{ ...S.label, marginBottom: 0 }}>Context Retrieval (Input B)</label>
                  <span style={{
                    fontSize: '0.75rem', fontWeight: 700,
                    color: contextRetrieval ? 'hsl(142 76% 36%)' : 'hsl(var(--muted-foreground))',
                  }}>
                    {contextRetrieval ? 'ENABLED' : 'DISABLED'}
                  </span>
                </div>
                <div style={{
                  display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                  padding: '0.5rem 0.8rem', borderRadius: '8px',
                  background: 'hsl(var(--muted) / 0.5)',
                  border: '1px solid hsl(var(--border))',
                }}>
                  <span style={{ fontSize: '0.82rem', color: 'hsl(var(--foreground))' }}>
                    Retrieve Global & Meeting Context
                  </span>
                  <input
                    type="checkbox"
                    checked={contextRetrieval}
                    onChange={(e) => setContextRetrieval(e.target.checked)}
                    style={{ width: '18px', height: '18px', accentColor: 'hsl(var(--accent))', cursor: 'pointer' }}
                  />
                </div>
                <span style={{ fontSize: '0.72rem', color: 'hsl(var(--muted-foreground))', display: 'block', marginTop: '3px' }}>
                  Fetches context summary and agenda to enrich point consolidation.
                </span>
              </div>
            </div>
          </div>

          {/* ── Section 3: Reference Output Selection ── */}
          <div style={{ ...S.card, display: 'flex', flexDirection: 'column', gap: '1.2rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <ShieldCheck size={16} color="hsl(var(--accent))" />
                <h3 style={{ margin: 0, fontSize: '0.9rem', fontWeight: 700, color: 'hsl(var(--foreground))' }}>
                  Reference Output Target
                </h3>
              </div>

              {/* Active source badge */}
              {referenceSource === 'edited_stage2' && (
                <span style={{
                  fontSize: '0.75rem', fontWeight: 700, padding: '3px 9px', borderRadius: '8px',
                  background: 'hsl(142 76% 36% / 0.15)', color: 'hsl(142 76% 36%)',
                  display: 'flex', alignItems: 'center', gap: '4px',
                }}>
                  <CheckCircle2 size={12} /> Using Edited Stage 2 Output ({referencePoints.length} points)
                </span>
              )}
              {referenceSource === 'existing_stage2' && (
                <span style={{
                  fontSize: '0.75rem', fontWeight: 700, padding: '3px 9px', borderRadius: '8px',
                  background: 'hsl(var(--accent) / 0.12)', color: 'hsl(var(--accent))',
                  display: 'flex', alignItems: 'center', gap: '4px',
                }}>
                  <FileCheck size={12} /> Using Stage 2 Output ({referencePoints.length} points)
                </span>
              )}
              {referenceSource === 'manual_mom' && (
                <span style={{
                  fontSize: '0.75rem', fontWeight: 700, padding: '3px 9px', borderRadius: '8px',
                  background: 'hsl(217 91% 60% / 0.15)', color: 'hsl(217 91% 60%)',
                  display: 'flex', alignItems: 'center', gap: '4px',
                }}>
                  <FileCheck size={12} /> Using Uploaded MoM ({referencePoints.length} points)
                </span>
              )}
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
              {/* Option B: Existing edited output (Preferred) */}
              <div style={{
                padding: '1rem',
                borderRadius: '10px',
                border: `1.5px solid ${hasStage2Edits ? 'hsl(142 76% 36% / 0.5)' : 'hsl(var(--border))'}`,
                background: hasStage2Edits ? 'hsl(142 76% 36% / 0.04)' : 'hsl(var(--muted) / 0.2)',
                display: 'flex', flexDirection: 'column', gap: '0.5rem',
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <span style={{ fontSize: '0.8rem', fontWeight: 700, color: 'hsl(var(--foreground))' }}>
                    Option B: Edited Stage 2 Output
                  </span>
                  {hasStage2Edits && (
                    <span style={{
                      fontSize: '0.68rem', fontWeight: 700, padding: '1px 6px', borderRadius: '6px',
                      background: 'hsl(142 76% 36% / 0.2)', color: 'hsl(142 76% 36%)',
                    }}>
                      RECOMMENDED
                    </span>
                  )}
                </div>
                <p style={{ margin: 0, fontSize: '0.78rem', color: 'hsl(var(--muted-foreground))', lineHeight: '1.4' }}>
                  Uses human-edited Stage 2 discussion points stored from this meeting as the optimization target.
                </p>
                <div style={{ marginTop: 'auto', paddingTop: '0.5rem', fontSize: '0.75rem', fontWeight: 600 }}>
                  {hasStage2Edits ? (
                    <span style={{ color: 'hsl(142 76% 36%)' }}>
                      ✓ Edits found • Ready as training reference
                    </span>
                  ) : (
                    <span style={{ color: 'hsl(var(--muted-foreground))' }}>
                      No manual edits recorded for this meeting.
                    </span>
                  )}
                </div>
              </div>

              {/* Option A: Manual MoM Upload */}
              <div
                onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)}
                onDrop={(e) => {
                  e.preventDefault();
                  setDragOver(false);
                  const file = e.dataTransfer.files[0];
                  if (file) handleFileUpload(file);
                }}
                style={{
                  padding: '1rem',
                  borderRadius: '10px',
                  border: `1.5px dashed ${dragOver ? 'hsl(var(--accent))' : 'hsl(var(--border))'}`,
                  background: dragOver ? 'hsl(var(--accent) / 0.05)' : 'hsl(var(--muted) / 0.2)',
                  display: 'flex', flexDirection: 'column', gap: '0.5rem', alignItems: 'center',
                  textAlign: 'center',
                  cursor: 'pointer',
                  position: 'relative',
                }}
              >
                <input
                  type="file"
                  accept=".pdf,.docx,.doc,.txt,image/*"
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file) handleFileUpload(file);
                  }}
                  style={{
                    position: 'absolute', inset: 0, opacity: 0, cursor: 'pointer', width: '100%', height: '100%',
                  }}
                />
                <Upload size={20} color="hsl(var(--accent))" />
                <div>
                  <div style={{ fontSize: '0.8rem', fontWeight: 700, color: 'hsl(var(--foreground))' }}>
                    Option A: Upload Manual MoM
                  </div>
                  <div style={{ fontSize: '0.72rem', color: 'hsl(var(--muted-foreground))', marginTop: '2px' }}>
                    PDF, DOCX, TXT, or Image (OCR)
                  </div>
                </div>
                {momUploading && (
                  <span style={{ fontSize: '0.72rem', color: 'hsl(var(--accent))', fontWeight: 600 }}>
                    Extracting points with LLM...
                  </span>
                )}
                {momFilename && !momUploading && (
                  <span style={{ fontSize: '0.72rem', color: 'hsl(142 76% 36%)', fontWeight: 600 }}>
                    ✓ {momFilename} ({momPoints.length} points extracted)
                  </span>
                )}
                {uploadError && (
                  <span style={{ fontSize: '0.72rem', color: 'hsl(0 84% 60%)' }}>
                    {uploadError}
                  </span>
                )}
              </div>
            </div>
          </div>

          {/* ── Section 4: Group Preview ── */}
          {groups.length > 0 && (
            <div style={{ ...S.card, display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Layers size={15} color="hsl(var(--accent))" />
                  <h3 style={{ margin: 0, fontSize: '0.9rem', fontWeight: 700, color: 'hsl(var(--foreground))' }}>
                    Stage 2 Training Groups Preview ({groups.length} groups)
                  </h3>
                </div>
                <span style={{ fontSize: '0.78rem', color: 'hsl(var(--muted-foreground))' }}>
                  {stage1Points.length} Stage 1 points grouped by {pointsPerGroup}
                </span>
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
                {groups.map((grp) => {
                  const isExpanded = expandedGroupIdx === grp.group_index;
                  return (
                    <div
                      key={grp.group_index}
                      style={{
                        borderRadius: '8px',
                        border: '1px solid hsl(var(--border))',
                        background: 'hsl(var(--background))',
                        overflow: 'hidden',
                      }}
                    >
                      {/* Header */}
                      <div
                        onClick={() => setExpandedGroupIdx(isExpanded ? null : grp.group_index)}
                        style={{
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'space-between',
                          padding: '0.65rem 0.9rem',
                          cursor: 'pointer',
                          background: isExpanded ? 'hsl(var(--muted) / 0.5)' : 'transparent',
                          transition: 'background 0.12s ease',
                        }}
                      >
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <span style={{
                            fontSize: '0.72rem', fontWeight: 700, padding: '2px 7px', borderRadius: '6px',
                            background: 'hsl(var(--accent) / 0.15)', color: 'hsl(var(--accent))',
                          }}>
                            Group {grp.group_index + 1}
                          </span>
                          <span style={{ fontSize: '0.82rem', fontWeight: 600, color: 'hsl(var(--foreground))' }}>
                            {grp.point_count} Stage 1 points
                          </span>
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                          {contextRetrieval && (
                            <span style={{ fontSize: '0.7rem', color: 'hsl(var(--muted-foreground))' }}>
                              Context attached
                            </span>
                          )}
                          {isExpanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                        </div>
                      </div>

                      {/* Expanded Content */}
                      {isExpanded && (
                        <div style={{ padding: '0.9rem', borderTop: '1px solid hsl(var(--border))', display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                          {/* Stage 1 Points List */}
                          <div>
                            <div style={{ ...S.label, marginBottom: '0.3rem' }}>Stage 1 Discussion Points:</div>
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
                              {grp.points.map((pt, pIdx) => {
                                const text = typeof pt === 'object' ? (pt.discussion_point || pt.point || JSON.stringify(pt)) : String(pt);
                                const speakers = typeof pt === 'object' ? (pt.speakers || pt.speaker || []) : [];
                                return (
                                  <div
                                    key={pIdx}
                                    style={{
                                      padding: '0.45rem 0.65rem',
                                      borderRadius: '6px',
                                      background: 'hsl(var(--muted) / 0.3)',
                                      fontSize: '0.8rem',
                                      lineHeight: '1.4',
                                    }}
                                  >
                                    <span style={{ fontWeight: 600, color: 'hsl(var(--accent))', marginRight: '6px' }}>
                                      #{pIdx + 1}
                                    </span>
                                    {text}
                                    {speakers.length > 0 && (
                                      <span style={{ fontSize: '0.7rem', color: 'hsl(var(--muted-foreground))', marginLeft: '6px' }}>
                                        [{speakers.join(', ')}]
                                      </span>
                                    )}
                                  </div>
                                );
                              })}
                            </div>
                          </div>

                          {/* Retrieved Context Preview */}
                          {contextRetrieval && (
                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem' }}>
                              <div>
                                <div style={{ ...S.label, marginBottom: '0.2rem' }}>Global Context:</div>
                                <div style={{
                                  fontSize: '0.75rem', color: 'hsl(var(--muted-foreground))',
                                  background: 'hsl(var(--muted) / 0.2)', padding: '0.4rem 0.6rem',
                                  borderRadius: '6px', maxHeight: '70px', overflowY: 'auto',
                                }}>
                                  {grp.global_context || 'None available'}
                                </div>
                              </div>
                              <div>
                                <div style={{ ...S.label, marginBottom: '0.2rem' }}>Meeting Context:</div>
                                <div style={{
                                  fontSize: '0.75rem', color: 'hsl(var(--muted-foreground))',
                                  background: 'hsl(var(--muted) / 0.2)', padding: '0.4rem 0.6rem',
                                  borderRadius: '6px', maxHeight: '70px', overflowY: 'auto',
                                }}>
                                  {grp.meeting_context || 'None available'}
                                </div>
                              </div>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>

              {/* Bottom navigation hint */}
              <div style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '0.75rem 1rem',
                borderRadius: '8px',
                background: 'hsl(var(--accent) / 0.08)',
                border: '1px solid hsl(var(--accent) / 0.2)',
                marginTop: '0.5rem',
              }}>
                <span style={{ fontSize: '0.82rem', fontWeight: 600, color: 'hsl(var(--foreground))' }}>
                  {groups.length} groups ready with {referencePoints.length > 0 ? `${referencePoints.length} reference targets` : 'baseline reference'}
                </span>
                <button
                  onClick={() => setActiveTab('training_loop')}
                  style={S.btn('primary')}
                >
                  Go to Training Loop <ArrowRight size={13} />
                </button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
