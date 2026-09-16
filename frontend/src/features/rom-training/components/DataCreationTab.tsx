import { useState, useEffect, useRef } from 'react';
import {
  Upload, FileText, Check, AlertCircle, ChevronDown,
  Loader, ExternalLink, Database, RefreshCw, X,
} from 'lucide-react';
import {
  listMeetings, getLongRomPoints, extractMom,
  matchAgendas, saveTrainingData, getDataForMeeting,
} from '../api/romTrainingApi';
import type { Meeting, LongRomAgenda, MomAgenda, AgendaMatch } from '../types/romTrainingTypes';
import { useNavigate } from 'react-router-dom';

const S = {
  card: {
    background: 'hsl(var(--card))',
    borderRadius: '12px',
    border: '1px solid hsl(var(--border))',
    padding: '1.25rem',
  } as React.CSSProperties,
  label: {
    display: 'block',
    fontSize: '0.78rem',
    fontWeight: 700,
    color: 'hsl(var(--foreground))',
    marginBottom: '0.35rem',
  } as React.CSSProperties,
  btn: (primary = true) => ({
    display: 'inline-flex', alignItems: 'center', gap: 6,
    padding: '0.5rem 1rem',
    borderRadius: '8px',
    border: 'none',
    cursor: 'pointer',
    fontSize: '0.82rem',
    fontWeight: 600,
    fontFamily: 'Inter, sans-serif',
    background: primary ? 'hsl(var(--accent))' : 'hsl(var(--muted))',
    color: primary ? 'white' : 'hsl(var(--foreground))',
    transition: 'opacity 0.15s',
  } as React.CSSProperties),
};

type Step = 'select-meeting' | 'upload-mom' | 'match-agendas' | 'preview';

export default function DataCreationTab() {
  const navigate = useNavigate();
  const [step, setStep] = useState<Step>('select-meeting');

  // Meeting selection
  const [meetings, setMeetings] = useState<Meeting[]>([]);
  const [loadingMeetings, setLoadingMeetings] = useState(false);
  const [selectedMeeting, setSelectedMeeting] = useState<Meeting | null>(null);
  const [longRomData, setLongRomData] = useState<{ available: boolean; agendas: LongRomAgenda[] } | null>(null);
  const [loadingRom, setLoadingRom] = useState(false);
  const [existingData, setExistingData] = useState<any[]>([]);

  // MoM upload
  const [uploading, setUploading] = useState(false);
  const [momAgendas, setMomAgendas] = useState<MomAgenda[]>([]);
  const [uploadError, setUploadError] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Agenda matching
  const [matches, setMatches] = useState<AgendaMatch[]>([]);
  const [matchingLoading, setMatchingLoading] = useState(false);
  // manual override: mom_agenda_idx -> meeting_agenda_id
  const [manualOverrides, setManualOverrides] = useState<Record<number, { id: string; title: string }>>({});

  // Save
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [saveError, setSaveError] = useState('');

  useEffect(() => {
    setLoadingMeetings(true);
    listMeetings()
      .then(d => setMeetings(d.meetings || []))
      .catch((err) => {
        console.error('Failed to load meetings for ROM Training:', err);
      })
      .finally(() => setLoadingMeetings(false));
  }, []);

  const handleSelectMeeting = async (meeting: Meeting) => {
    setSelectedMeeting(meeting);
    setLongRomData(null);
    setExistingData([]);
    setMomAgendas([]);
    setMatches([]);
    setManualOverrides({});
    setSaved(false);
    setStep('select-meeting');

    setLoadingRom(true);
    try {
      const [romRes, dataRes] = await Promise.all([
        getLongRomPoints(meeting.id),
        getDataForMeeting(meeting.id),
      ]);
      setLongRomData(romRes);
      setExistingData(dataRes.data || []);
    } catch (err) {
      console.error('Failed to load ROM data for meeting:', err);
    } finally {
      setLoadingRom(false);
    }
  };

  const handleFileUpload = async (file: File) => {
    setUploading(true);
    setUploadError('');
    setMomAgendas([]);
    setMatches([]);
    setManualOverrides({});
    try {
      const res = await extractMom(file);
      setMomAgendas(res.agendas || []);
      setStep('match-agendas');
      // Auto-match
      if (longRomData?.agendas) {
        await handleAutoMatch(res.agendas || []);
      }
    } catch (e: any) {
      setUploadError(e?.response?.data?.detail || 'Failed to extract MoM');
    } finally {
      setUploading(false);
    }
  };

  const handleAutoMatch = async (mom: MomAgenda[]) => {
    if (!longRomData?.agendas || !mom.length) return;
    setMatchingLoading(true);
    try {
      const res = await matchAgendas(
        mom,
        longRomData.agendas.map(a => ({ agenda_id: a.agenda_id, agenda_title: a.agenda_title })),
      );
      setMatches(res.matches || []);
    } catch {
    } finally {
      setMatchingLoading(false);
    }
  };

  const allMatched = matches.length > 0 && matches.every(m => {
    const override = manualOverrides[m.mom_agenda_idx];
    return (override?.id) || m.meeting_agenda_id;
  });

  const handleSave = async () => {
    if (!selectedMeeting || !allMatched) return;
    setSaving(true);
    setSaveError('');
    try {
      const pairs = matches.map(m => {
        const override = manualOverrides[m.mom_agenda_idx];
        const meetingAgendaId = override?.id || m.meeting_agenda_id;
        const meetingAgendaTitle = override?.title || m.meeting_agenda_title || '';
        const longRomAgenda = longRomData?.agendas.find(a => a.agenda_id === meetingAgendaId);
        const momAgenda = momAgendas[m.mom_agenda_idx];
        return {
          agenda_id: meetingAgendaId,
          agenda_title: meetingAgendaTitle,
          long_rom_points: longRomAgenda?.discussion_points || [],
          manual_mom_points: momAgenda?.points || [],
        };
      });

      await saveTrainingData({
        recording_id: selectedMeeting.id,
        recording_title: selectedMeeting.title,
        pairs,
      });
      setSaved(true);
      setStep('select-meeting');
      // Refresh existing data
      const dataRes = await getDataForMeeting(selectedMeeting.id);
      setExistingData(dataRes.data || []);
    } catch (e: any) {
      setSaveError(e?.response?.data?.detail || 'Failed to save');
    } finally {
      setSaving(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      {/* Main two-column layout */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem', minHeight: 0 }}>

        {/* LEFT PANEL */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>

          {/* Meeting Selector */}
          <div style={S.card}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: '0.75rem' }}>
              <FileText size={16} style={{ color: 'hsl(var(--accent))' }} />
              <span style={{ fontWeight: 700, fontSize: '0.9rem' }}>Select Meeting</span>
            </div>
            {loadingMeetings ? (
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: 'hsl(var(--muted-foreground))' }}>
                <Loader size={14} className="spin" /> Loading meetings…
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', maxHeight: 220, overflowY: 'auto' }}>
                {meetings.map(m => (
                  <button
                    key={m.id}
                    onClick={() => handleSelectMeeting(m)}
                    style={{
                      display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                      padding: '0.6rem 0.85rem',
                      borderRadius: '8px',
                      border: selectedMeeting?.id === m.id ? '1.5px solid hsl(var(--accent))' : '1px solid hsl(var(--border))',
                      background: selectedMeeting?.id === m.id ? 'hsl(var(--accent) / .08)' : 'hsl(var(--background))',
                      cursor: 'pointer', textAlign: 'left',
                      fontFamily: 'Inter, sans-serif',
                    }}
                  >
                    <div>
                      <div style={{ fontSize: '0.82rem', fontWeight: 600, color: 'hsl(var(--foreground))' }}>{m.title}</div>
                      <div style={{ fontSize: '0.7rem', color: 'hsl(var(--muted-foreground))' }}>{m.created_at?.slice(0, 10)}</div>
                    </div>
                    <div style={{ display: 'flex', gap: 4 }}>
                      {m.has_long_rom && (
                        <span style={{
                          fontSize: '0.65rem', fontWeight: 700, padding: '2px 6px',
                          borderRadius: 6, background: 'hsl(142 70% 45% / .15)',
                          color: 'hsl(142 70% 35%)',
                        }}>ROM ✓</span>
                      )}
                    </div>
                  </button>
                ))}
                {meetings.length === 0 && (
                  <div style={{ fontSize: '0.8rem', color: 'hsl(var(--muted-foreground))', textAlign: 'center', padding: '1rem' }}>
                    No meetings found
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Existing training data badge */}
          {selectedMeeting && existingData.length > 0 && (
            <div style={{
              padding: '0.75rem', borderRadius: '10px',
              background: 'hsl(var(--accent) / .08)',
              border: '1px solid hsl(var(--accent) / .2)',
              display: 'flex', alignItems: 'center', gap: 8,
            }}>
              <Database size={14} style={{ color: 'hsl(var(--accent))' }} />
              <span style={{ fontSize: '0.78rem', color: 'hsl(var(--accent))', fontWeight: 600 }}>
                {existingData.length} agenda{existingData.length !== 1 ? 's' : ''} already in database for this meeting
              </span>
            </div>
          )}

          {/* MoM Upload */}
          {selectedMeeting && longRomData?.available && (
            <div style={S.card}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: '0.75rem' }}>
                <Upload size={16} style={{ color: 'hsl(var(--accent))' }} />
                <span style={{ fontWeight: 700, fontSize: '0.9rem' }}>Upload Manual MoM</span>
              </div>
              <p style={{ fontSize: '0.75rem', color: 'hsl(var(--muted-foreground))', marginBottom: '0.75rem' }}>
                Upload a manually written Minutes of Meeting (PDF, DOCX, or TXT).
                Points will be extracted exactly as written — no modification.
              </p>
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf,.docx,.doc,.txt"
                style={{ display: 'none' }}
                onChange={e => e.target.files?.[0] && handleFileUpload(e.target.files[0])}
              />
              <button
                style={S.btn()}
                onClick={() => fileInputRef.current?.click()}
                disabled={uploading}
              >
                {uploading ? <Loader size={14} className="spin" /> : <Upload size={14} />}
                {uploading ? 'Extracting…' : 'Choose File'}
              </button>
              {uploadError && (
                <div style={{ marginTop: 8, fontSize: '0.75rem', color: 'hsl(var(--destructive))' }}>
                  <AlertCircle size={12} style={{ verticalAlign: 'middle', marginRight: 4 }} />{uploadError}
                </div>
              )}
              {momAgendas.length > 0 && (
                <div style={{ marginTop: 10, fontSize: '0.78rem', color: 'hsl(142 70% 35%)' }}>
                  <Check size={12} style={{ verticalAlign: 'middle', marginRight: 4 }} />
                  Extracted {momAgendas.length} agenda section{momAgendas.length !== 1 ? 's' : ''}
                </div>
              )}
            </div>
          )}

          {/* Agenda Matching */}
          {matches.length > 0 && (
            <div style={S.card}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: '0.75rem' }}>
                <RefreshCw size={16} style={{ color: 'hsl(var(--accent))' }} />
                <span style={{ fontWeight: 700, fontSize: '0.9rem' }}>Agenda Matching</span>
                {matchingLoading && <Loader size={12} className="spin" style={{ color: 'hsl(var(--muted-foreground))' }} />}
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {matches.map((m, idx) => {
                  const override = manualOverrides[m.mom_agenda_idx];
                  const matchedId = override?.id || m.meeting_agenda_id;
                  const matchedTitle = override?.title || m.meeting_agenda_title;
                  const isMatched = Boolean(matchedId);
                  return (
                    <div key={idx} style={{
                      padding: '0.6rem', borderRadius: 8,
                      border: isMatched ? '1px solid hsl(142 70% 45% / .3)' : '1px solid hsl(var(--destructive) / .3)',
                      background: isMatched ? 'hsl(142 70% 45% / .05)' : 'hsl(var(--destructive) / .05)',
                    }}>
                      <div style={{ fontSize: '0.75rem', fontWeight: 700, marginBottom: 4 }}>
                        MoM: "{m.mom_agenda_title}"
                      </div>
                      {isMatched ? (
                        <div style={{ fontSize: '0.72rem', color: 'hsl(142 70% 35%)' }}>
                          <Check size={11} style={{ verticalAlign: 'middle', marginRight: 3 }} />
                          Matched → "{matchedTitle}"
                          {m.confidence > 0 && !override && (
                            <span style={{ marginLeft: 6, opacity: 0.7 }}>(conf: {(m.confidence * 100).toFixed(0)}%)</span>
                          )}
                        </div>
                      ) : (
                        <div style={{ fontSize: '0.72rem', color: 'hsl(var(--destructive))', marginBottom: 4 }}>
                          <AlertCircle size={11} style={{ verticalAlign: 'middle', marginRight: 3 }} />
                          Could not auto-match — select manually:
                        </div>
                      )}
                      {/* Manual override dropdown */}
                      <select
                        value={matchedId || ''}
                        onChange={e => {
                          const sel = e.target.value;
                          if (!sel) {
                            const next = { ...manualOverrides };
                            delete next[m.mom_agenda_idx];
                            setManualOverrides(next);
                          } else {
                            const ag = longRomData?.agendas.find(a => a.agenda_id === sel);
                            setManualOverrides(prev => ({
                              ...prev,
                              [m.mom_agenda_idx]: { id: sel, title: ag?.agenda_title || sel },
                            }));
                          }
                        }}
                        style={{
                          marginTop: 4, width: '100%', padding: '0.3rem 0.5rem',
                          borderRadius: 6, border: '1px solid hsl(var(--border))',
                          background: 'hsl(var(--background))', color: 'hsl(var(--foreground))',
                          fontSize: '0.72rem', fontFamily: 'Inter, sans-serif',
                        }}
                      >
                        <option value="">— Select meeting agenda —</option>
                        {longRomData?.agendas.map(a => (
                          <option key={a.agenda_id} value={a.agenda_id}>{a.agenda_title}</option>
                        ))}
                      </select>
                    </div>
                  );
                })}
              </div>

              {/* Save button */}
              {allMatched && (
                <div style={{ marginTop: '1rem' }}>
                  {saved && (
                    <div style={{ fontSize: '0.78rem', color: 'hsl(142 70% 35%)', marginBottom: 8 }}>
                      <Check size={12} style={{ verticalAlign: 'middle', marginRight: 4 }} />
                      Saved to database successfully!
                    </div>
                  )}
                  {saveError && (
                    <div style={{ fontSize: '0.75rem', color: 'hsl(var(--destructive))', marginBottom: 8 }}>
                      {saveError}
                    </div>
                  )}
                  <button
                    style={S.btn()}
                    onClick={handleSave}
                    disabled={saving}
                  >
                    {saving ? <Loader size={14} className="spin" /> : <Database size={14} />}
                    {saving ? 'Saving…' : 'Add to Database'}
                  </button>
                </div>
              )}
            </div>
          )}
        </div>

        {/* RIGHT PANEL: Long ROM Points */}
        <div style={S.card}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: '0.75rem' }}>
            <FileText size={16} style={{ color: 'hsl(var(--accent))' }} />
            <span style={{ fontWeight: 700, fontSize: '0.9rem' }}>Long ROM Points</span>
          </div>

          {!selectedMeeting && (
            <div style={{ color: 'hsl(var(--muted-foreground))', fontSize: '0.82rem', textAlign: 'center', padding: '2rem' }}>
              Select a meeting to view its Long ROM points
            </div>
          )}

          {selectedMeeting && loadingRom && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: 'hsl(var(--muted-foreground))' }}>
              <Loader size={14} className="spin" /> Loading ROM data…
            </div>
          )}

          {selectedMeeting && !loadingRom && longRomData && !longRomData.available && (
            <div style={{ textAlign: 'center', padding: '2rem' }}>
              <AlertCircle size={32} style={{ color: 'hsl(var(--muted-foreground))', marginBottom: 12 }} />
              <p style={{ fontSize: '0.82rem', color: 'hsl(var(--muted-foreground))', marginBottom: '1rem' }}>
                Long ROM Points are not available for this meeting.
              </p>
              <button
                style={S.btn()}
                onClick={() => navigate(`/dashboard/history/${selectedMeeting.id}/rom`)}
              >
                <ExternalLink size={14} /> Generate ROM
              </button>
            </div>
          )}

          {selectedMeeting && !loadingRom && longRomData?.available && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', overflowY: 'auto', maxHeight: 'calc(100vh - 320px)' }}>
              {longRomData.agendas.map(ag => (
                <div key={ag.agenda_id} style={{
                  borderRadius: 8,
                  border: '1px solid hsl(var(--border))',
                  overflow: 'hidden',
                }}>
                  <div style={{
                    padding: '0.5rem 0.75rem',
                    background: 'hsl(var(--muted) / .5)',
                    fontWeight: 700, fontSize: '0.8rem',
                    borderBottom: '1px solid hsl(var(--border))',
                  }}>
                    {ag.agenda_title}
                    <span style={{ marginLeft: 8, fontWeight: 400, fontSize: '0.7rem', color: 'hsl(var(--muted-foreground))' }}>
                      ({ag.discussion_points.length} point{ag.discussion_points.length !== 1 ? 's' : ''})
                    </span>
                  </div>
                  <div style={{ padding: '0.5rem 0.75rem', display: 'flex', flexDirection: 'column', gap: 6 }}>
                    {ag.discussion_points.slice(0, 10).map((pt: any, i: number) => (
                      <div key={i} style={{
                        fontSize: '0.75rem', color: 'hsl(var(--foreground))',
                        padding: '0.3rem 0.5rem',
                        borderLeft: '2px solid hsl(var(--accent) / .4)',
                        background: 'hsl(var(--accent) / .04)',
                        borderRadius: '0 4px 4px 0',
                      }}>
                        {pt.polished_text || pt.text || JSON.stringify(pt)}
                      </div>
                    ))}
                    {ag.discussion_points.length > 10 && (
                      <div style={{ fontSize: '0.72rem', color: 'hsl(var(--muted-foreground))', textAlign: 'center' }}>
                        + {ag.discussion_points.length - 10} more points
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Preview: side-by-side when we have matches */}
      {matches.length > 0 && longRomData?.available && (
        <div style={S.card}>
          <div style={{ fontWeight: 700, fontSize: '0.9rem', marginBottom: '1rem' }}>Data Preview</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            {matches.map((m, idx) => {
              const override = manualOverrides[m.mom_agenda_idx];
              const meetingAgendaId = override?.id || m.meeting_agenda_id;
              const longAg = longRomData.agendas.find(a => a.agenda_id === meetingAgendaId);
              const momAg = momAgendas[m.mom_agenda_idx];
              if (!longAg || !momAg) return null;
              return (
                <div key={idx} style={{ borderRadius: 8, border: '1px solid hsl(var(--border))', overflow: 'hidden' }}>
                  <div style={{
                    padding: '0.5rem 0.75rem',
                    background: 'hsl(var(--muted) / .5)',
                    fontWeight: 700, fontSize: '0.8rem',
                    borderBottom: '1px solid hsl(var(--border))',
                  }}>
                    {longAg.agenda_title}
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 0 }}>
                    <div style={{ padding: '0.75rem', borderRight: '1px solid hsl(var(--border))' }}>
                      <div style={{ fontSize: '0.7rem', fontWeight: 700, color: 'hsl(var(--accent))', marginBottom: 8 }}>LONG ROM POINTS</div>
                      {longAg.discussion_points.slice(0, 5).map((pt: any, i: number) => (
                        <div key={i} style={{ fontSize: '0.72rem', marginBottom: 4, paddingLeft: 8, borderLeft: '2px solid hsl(var(--accent) / .3)' }}>
                          {pt.polished_text || pt.text || ''}
                        </div>
                      ))}
                    </div>
                    <div style={{ padding: '0.75rem' }}>
                      <div style={{ fontSize: '0.7rem', fontWeight: 700, color: 'hsl(280 80% 60%)', marginBottom: 8 }}>MANUAL MoM POINTS</div>
                      {momAg.points.slice(0, 5).map((pt, i) => (
                        <div key={i} style={{ fontSize: '0.72rem', marginBottom: 4, paddingLeft: 8, borderLeft: '2px solid hsl(280 80% 60% / .3)' }}>
                          {pt}
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
