import { useState } from 'react';
import { useTrainingStore } from '../store/trainingStore';
import { fetchMeetingData } from '../api/trainingApi';
import { Check, Calendar, Clock, Layers, Pencil, FileText, ChevronDown, ChevronUp } from 'lucide-react';

export default function MeetingSelector() {
  const {
    meetings,
    selectedMeetingId,
    setSelectedMeetingId,
    meetingDataSummary,
    setMeetingDataSummary,
    useEditedStage2,
    setUseEditedStage2,
  } = useTrainingStore();

  const [loadingSummary, setLoadingSummary] = useState(false);
  const [showPreviewComparison, setShowPreviewComparison] = useState(false);

  const handleSelect = async (id: string) => {
    setSelectedMeetingId(id);
    setLoadingSummary(true);
    try {
      const data = await fetchMeetingData(id);
      setMeetingDataSummary(data);
      // Auto-enable useEditedStage2 if edits are available
      if (data.has_stage2_edits) {
        setUseEditedStage2(true);
      } else {
        setUseEditedStage2(false);
      }
    } catch (err) {
      console.warn('Failed to fetch meeting data', err);
    } finally {
      setLoadingSummary(false);
    }
  };

  const selectedMeeting = meetings.find((m) => m.id === selectedMeetingId);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
      <div>
        <h3 style={{ margin: '0 0 0.25rem 0', fontWeight: 700, fontSize: '1rem', color: 'hsl(var(--foreground))' }}>
          Select Meeting Input Source
        </h3>
        <p style={{ margin: 0, fontSize: '0.8rem', color: 'hsl(var(--muted-foreground))' }}>
          Choose a previously processed meeting to use its transcript, ROM points, and agendas as training inputs.
        </p>
      </div>

      {meetings.length === 0 ? (
        <div style={{
          padding: '2rem',
          textAlign: 'center',
          color: 'hsl(var(--muted-foreground))',
          background: 'hsl(var(--card))',
          borderRadius: '10px',
          border: '1px solid hsl(var(--border))',
        }}>
          No processed meetings available for training.
        </div>
      ) : (
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          gap: '0.6rem',
          maxHeight: '320px',
          overflowY: 'auto',
          paddingRight: '0.25rem',
        }}>
          {meetings.map((meeting) => {
            const isSelected = selectedMeetingId === meeting.id;
            const hasEdits = meeting.has_stage2_edits || (meeting.stage2_edit_count && meeting.stage2_edit_count > 0);
            return (
              <div
                key={meeting.id}
                onClick={() => handleSelect(meeting.id)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  padding: '0.85rem 1rem',
                  borderRadius: '10px',
                  border: `1.5px solid ${isSelected ? 'hsl(var(--accent))' : 'hsl(var(--border) / 0.6)'}`,
                  background: isSelected ? 'hsl(var(--accent) / 0.06)' : 'hsl(var(--card))',
                  cursor: 'pointer',
                  transition: 'all 0.15s ease',
                }}
              >
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.3rem' }}>
                  <div style={{ fontWeight: 600, fontSize: '0.9rem', display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'hsl(var(--foreground))' }}>
                    {meeting.filename}
                    {isSelected && <Check size={16} color="hsl(var(--accent))" />}
                    {hasEdits && (
                      <span style={{
                        fontSize: '0.68rem',
                        fontWeight: 700,
                        padding: '2px 7px',
                        borderRadius: '10px',
                        background: 'hsl(30 90% 50% / 0.15)',
                        color: 'hsl(30 90% 45%)',
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: '3px',
                      }}>
                        <Pencil size={10} />
                        Stage 2 Edits Available
                      </span>
                    )}
                  </div>
                  <div style={{ display: 'flex', gap: '1rem', fontSize: '0.78rem', color: 'hsl(var(--muted-foreground))' }}>
                    <span style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                      <Calendar size={12} /> {new Date(meeting.created_at).toLocaleDateString()}
                    </span>
                    <span style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                      <Clock size={12} /> {Math.round(meeting.duration / 60)} min
                    </span>
                  </div>
                </div>

                <div style={{ display: 'flex', gap: '0.4rem' }}>
                  {meeting.stages_available.map((stage) => (
                    <span
                      key={stage}
                      style={{
                        fontSize: '0.68rem',
                        fontWeight: 600,
                        padding: '2px 8px',
                        borderRadius: '10px',
                        background: 'hsl(var(--primary) / 0.1)',
                        color: 'hsl(var(--primary))',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '0.25rem',
                      }}
                    >
                      <Layers size={10} />
                      {stage.replace('_', ' ')}
                    </span>
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Selected Meeting Summary & Training Output Options */}
      {selectedMeetingId && meetingDataSummary && (
        <div style={{
          background: 'hsl(var(--card))',
          borderRadius: '12px',
          padding: '1.25rem',
          border: '1.5px solid hsl(var(--border) / 0.8)',
          display: 'flex',
          flexDirection: 'column',
          gap: '1rem',
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '0.88rem', fontWeight: 700, color: 'hsl(var(--foreground))', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <FileText size={16} style={{ color: 'hsl(var(--accent))' }} />
              Training Data Available for {selectedMeeting?.filename}
            </span>
            <span style={{ fontSize: '0.75rem', color: 'hsl(var(--muted-foreground))' }}>
              {meetingDataSummary.transcript_segments} transcript segments
            </span>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '0.75rem' }}>
            <div style={{ padding: '0.6rem 0.8rem', borderRadius: '8px', background: 'hsl(var(--background))', border: '1px solid hsl(var(--border) / 0.4)' }}>
              <div style={{ fontSize: '0.7rem', color: 'hsl(var(--muted-foreground))' }}>Stage 1 Points</div>
              <div style={{ fontSize: '1rem', fontWeight: 700, marginTop: '2px' }}>{meetingDataSummary.stage1_points}</div>
            </div>
            <div style={{ padding: '0.6rem 0.8rem', borderRadius: '8px', background: 'hsl(var(--background))', border: '1px solid hsl(var(--border) / 0.4)' }}>
              <div style={{ fontSize: '0.7rem', color: 'hsl(var(--muted-foreground))' }}>Stage 2 Points</div>
              <div style={{ fontSize: '1rem', fontWeight: 700, marginTop: '2px' }}>{meetingDataSummary.stage2_points}</div>
            </div>
            <div style={{ padding: '0.6rem 0.8rem', borderRadius: '8px', background: 'hsl(var(--background))', border: '1px solid hsl(var(--border) / 0.4)' }}>
              <div style={{ fontSize: '0.7rem', color: 'hsl(var(--muted-foreground))' }}>Stage 3 Agendas</div>
              <div style={{ fontSize: '1rem', fontWeight: 700, marginTop: '2px' }}>{meetingDataSummary.stage3_agendas}</div>
            </div>
          </div>

          {/* Training Target Output Option: Stage 2 Edits */}
          {meetingDataSummary.has_stage2_edits ? (
            <div style={{
              background: useEditedStage2 ? 'hsl(30 90% 50% / 0.08)' : 'hsl(var(--background))',
              border: `1.5px solid ${useEditedStage2 ? 'hsl(30 90% 50% / 0.5)' : 'hsl(var(--border))'}`,
              borderRadius: '10px',
              padding: '1rem',
              transition: 'all 0.2s ease',
            }}>
              <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: '1rem' }}>
                <div style={{ display: 'flex', alignItems: 'flex-start', gap: '0.75rem' }}>
                  <div style={{
                    padding: '0.4rem',
                    borderRadius: '8px',
                    background: useEditedStage2 ? 'hsl(30 90% 50% / 0.2)' : 'hsl(var(--muted))',
                    color: useEditedStage2 ? 'hsl(30 90% 45%)' : 'hsl(var(--muted-foreground))',
                    marginTop: '2px',
                  }}>
                    <Pencil size={18} />
                  </div>
                  <div>
                    <div style={{ fontWeight: 700, fontSize: '0.9rem', color: 'hsl(var(--foreground))', display: 'flex', alignItems: 'center', gap: '8px' }}>
                      Use Edited Stage 2 Version for Training
                      <span style={{
                        fontSize: '0.65rem',
                        fontWeight: 700,
                        padding: '2px 6px',
                        borderRadius: '8px',
                        background: 'hsl(140 65% 45% / 0.15)',
                        color: 'hsl(140 65% 40%)',
                      }}>
                        {meetingDataSummary.stage2_edit_count || 1} manual edits
                      </span>
                    </div>
                    <p style={{ margin: '4px 0 0', fontSize: '0.78rem', color: 'hsl(var(--muted-foreground))', lineHeight: '1.4' }}>
                      When enabled, the training target will use your manually edited Stage 2 discussion points instead of the original auto-generated Stage 2 output.
                    </p>
                  </div>
                </div>

                <label style={{ position: 'relative', display: 'inline-block', width: '44px', height: '24px', flexShrink: 0, cursor: 'pointer' }}>
                  <input
                    type="checkbox"
                    checked={useEditedStage2}
                    onChange={(e) => setUseEditedStage2(e.target.checked)}
                    style={{ opacity: 0, width: 0, height: 0 }}
                  />
                  <span style={{
                    position: 'absolute',
                    top: 0, left: 0, right: 0, bottom: 0,
                    backgroundColor: useEditedStage2 ? 'hsl(30 90% 50%)' : 'hsl(var(--muted))',
                    transition: '0.2s',
                    borderRadius: '24px',
                  }}>
                    <span style={{
                      position: 'absolute',
                      content: '""',
                      height: '18px', width: '18px',
                      left: useEditedStage2 ? '22px' : '3px',
                      bottom: '3px',
                      backgroundColor: 'white',
                      transition: '0.2s',
                      borderRadius: '50%',
                    }} />
                  </span>
                </label>
              </div>

              {/* Version Comparison Toggle */}
              {(meetingDataSummary.original_stage2_points?.length || 0) > 0 && (
                <div style={{ marginTop: '0.85rem', paddingTop: '0.75rem', borderTop: '1px solid hsl(var(--border) / 0.4)' }}>
                  <button
                    onClick={() => setShowPreviewComparison(!showPreviewComparison)}
                    style={{
                      background: 'none',
                      border: 'none',
                      color: 'hsl(var(--accent))',
                      fontSize: '0.75rem',
                      fontWeight: 600,
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '4px',
                      padding: 0,
                    }}
                  >
                    {showPreviewComparison ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                    {showPreviewComparison ? 'Hide Stage 2 Output Comparison' : 'Compare Original vs Edited Stage 2 Version'}
                  </button>

                  {showPreviewComparison && (
                    <div style={{ marginTop: '0.6rem', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem', fontSize: '0.75rem' }}>
                      <div style={{ background: 'hsl(var(--background))', padding: '0.6rem 0.8rem', borderRadius: '8px', border: '1px solid hsl(var(--border))' }}>
                        <div style={{ fontWeight: 700, color: 'hsl(var(--muted-foreground))', marginBottom: '4px', textTransform: 'uppercase', fontSize: '0.65rem' }}>
                          Original Auto-Generated Output
                        </div>
                        <div style={{ color: 'hsl(var(--foreground))', maxHeight: '120px', overflowY: 'auto' }}>
                          {meetingDataSummary.original_stage2_points?.slice(0, 3).map((p: any, idx: number) => (
                            <div key={idx} style={{ marginBottom: '4px' }}>• {p.polished_text || p.discussion_point || JSON.stringify(p)}</div>
                          ))}
                        </div>
                      </div>
                      <div style={{ background: 'hsl(var(--background))', padding: '0.6rem 0.8rem', borderRadius: '8px', border: '1.5px solid hsl(30 90% 50% / 0.4)' }}>
                        <div style={{ fontWeight: 700, color: 'hsl(30 90% 45%)', marginBottom: '4px', textTransform: 'uppercase', fontSize: '0.65rem' }}>
                          Manually Edited Version (Target)
                        </div>
                        <div style={{ color: 'hsl(var(--foreground))', maxHeight: '120px', overflowY: 'auto' }}>
                          {meetingDataSummary.edited_stage2_points?.slice(0, 3).map((p: any, idx: number) => (
                            <div key={idx} style={{ marginBottom: '4px' }}>• {p.polished_text || p.discussion_point || JSON.stringify(p)}</div>
                          ))}
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          ) : (
            <div style={{ fontSize: '0.75rem', color: 'hsl(var(--muted-foreground))', fontStyle: 'italic' }}>
              Original Stage 2 output will be used as the target (no manual Stage 2 edits recorded for this meeting).
            </div>
          )}
        </div>
      )}
    </div>
  );
}
