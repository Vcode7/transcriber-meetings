import { useState, useEffect, useRef, useMemo } from 'react';
import {
  Upload, FileText, Check, AlertCircle, Loader,
  ExternalLink, Database, RefreshCw, X,
  ChevronDown, ChevronRight, Info, BookOpen, Layers,
  ArrowRight, Plus, Sparkles, Tag,
} from 'lucide-react';
import {
  listMeetings, getLongRomPoints, extractMom,
  matchAgendas, saveTrainingData, getDataForMeeting,
} from '../api/romTrainingApi';
import type { Meeting, LongRomAgenda, MomAgenda, AgendaMatch } from '../types/romTrainingTypes';
import { useNavigate } from 'react-router-dom';

// ── Color / Design Tokens ─────────────────────────────────────────────────────
const C = {
  accent: 'hsl(var(--accent))',
  accentSoft: 'hsl(var(--accent) / .1)',
  accentBorder: 'hsl(var(--accent) / .35)',
  border: 'hsl(var(--border))',
  muted: 'hsl(var(--muted))',
  mutedFg: 'hsl(var(--muted-foreground))',
  card: 'hsl(var(--card))',
  bg: 'hsl(var(--background))',
  fg: 'hsl(var(--foreground))',
  green: 'hsl(142 70% 40%)',
  greenSoft: 'hsl(142 70% 45% / .12)',
  greenBorder: 'hsl(142 70% 45% / .3)',
  amber: 'hsl(38 90% 48%)',
  amberSoft: 'hsl(38 90% 50% / .1)',
  amberBorder: 'hsl(38 90% 50% / .3)',
  purple: 'hsl(280 75% 60%)',
  purpleSoft: 'hsl(280 75% 60% / .1)',
  purpleBorder: 'hsl(280 75% 60% / .3)',
  red: 'hsl(var(--destructive))',
  redSoft: 'hsl(var(--destructive) / .08)',
};

export default function DataCreationTab() {
  const navigate = useNavigate();

  // ── Meeting state ──────────────────────────────────────────────────────────
  const [meetings, setMeetings] = useState<Meeting[]>([]);
  const [loadingMeetings, setLoadingMeetings] = useState(false);
  const [selectedMeeting, setSelectedMeeting] = useState<Meeting | null>(null);
  const [showMeetingPicker, setShowMeetingPicker] = useState(false);
  const [meetingFilter, setMeetingFilter] = useState('');
  const meetingPickerRef = useRef<HTMLDivElement>(null);

  // ── Long ROM state ─────────────────────────────────────────────────────────
  const [longRomData, setLongRomData] = useState<{ available: boolean; agendas: LongRomAgenda[] } | null>(null);
  const [loadingRom, setLoadingRom] = useState(false);
  const [existingData, setExistingData] = useState<any[]>([]);

  // ── Upload & Extraction state ──────────────────────────────────────────────
  const [uploading, setUploading] = useState(false);
  const [uploadedFileName, setUploadedFileName] = useState('');
  const [uploadError, setUploadError] = useState('');
  const [momAgendas, setMomAgendas] = useState<MomAgenda[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // ── Agenda Mapping state: meeting_agenda_id -> array of momAgenda indices ───
  const [agendaMapping, setAgendaMapping] = useState<Record<string, number[]>>({});
  const [matching, setMatching] = useState(false);
  const [autoMatches, setAutoMatches] = useState<AgendaMatch[]>([]);

  // ── Accordion / Collapsed state ────────────────────────────────────────────
  const [collapsedAgendas, setCollapsedAgendas] = useState<Set<string>>(new Set());
  const [showUnmappedDrawer, setShowUnmappedDrawer] = useState(true);

  // ── Save state ─────────────────────────────────────────────────────────────
  const [saving, setSaving] = useState(false);
  const [saveSuccess, setSaveSuccess] = useState(false);
  const [saveError, setSaveError] = useState('');

  // ── Lifecycle: Load meetings ───────────────────────────────────────────────
  useEffect(() => {
    setLoadingMeetings(true);
    listMeetings()
      .then(d => setMeetings(d.meetings || []))
      .catch(err => console.error('Failed to load meetings:', err))
      .finally(() => setLoadingMeetings(false));
  }, []);

  // Close meeting picker on click outside
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (meetingPickerRef.current && !meetingPickerRef.current.contains(e.target as Node)) {
        setShowMeetingPicker(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  // ── Meeting selection handler ──────────────────────────────────────────────
  const handleSelectMeeting = async (meeting: Meeting) => {
    setSelectedMeeting(meeting);
    setShowMeetingPicker(false);
    setMeetingFilter('');
    setLongRomData(null);
    setMomAgendas([]);
    setAgendaMapping({});
    setAutoMatches([]);
    setUploadedFileName('');
    setUploadError('');
    setSaveSuccess(false);
    setSaveError('');
    setLoadingRom(true);

    try {
      const [romRes, dataRes] = await Promise.all([
        getLongRomPoints(meeting.id),
        getDataForMeeting(meeting.id),
      ]);
      setLongRomData(romRes);
      setExistingData(dataRes.data || []);
    } catch (err) {
      console.error('Failed to load ROM data:', err);
    } finally {
      setLoadingRom(false);
    }
  };

  // ── File upload & extraction handler ───────────────────────────────────────
  const handleFileUpload = async (file: File) => {
    setUploading(true);
    setUploadError('');
    setMomAgendas([]);
    setAgendaMapping({});
    setAutoMatches([]);
    setUploadedFileName(file.name);
    setSaveSuccess(false);
    setSaveError('');

    try {
      const res = await extractMom(file);
      const extracted: MomAgenda[] = res.agendas || [];
      setMomAgendas(extracted);

      // Auto-match agendas immediately after extraction
      if (longRomData?.agendas?.length && extracted.length) {
        setMatching(true);
        try {
          const matchRes = await matchAgendas(
            extracted,
            longRomData.agendas.map(a => ({ agenda_id: a.agenda_id, agenda_title: a.agenda_title })),
          );
          const matches: AgendaMatch[] = matchRes.matches || [];
          setAutoMatches(matches);

          // Build initial mapping: meeting_agenda_id -> [momAgendaIdx]
          const initialMap: Record<string, number[]> = {};
          for (const ag of longRomData.agendas) {
            initialMap[ag.agenda_id] = [];
          }

          for (const m of matches) {
            if (m.meeting_agenda_id && m.confidence >= 0.2) {
              if (!initialMap[m.meeting_agenda_id]) {
                initialMap[m.meeting_agenda_id] = [];
              }
              if (!initialMap[m.meeting_agenda_id].includes(m.mom_agenda_idx)) {
                initialMap[m.meeting_agenda_id].push(m.mom_agenda_idx);
              }
            }
          }
          setAgendaMapping(initialMap);
        } catch (e) {
          console.error('Agenda auto-matching failed:', e);
        } finally {
          setMatching(false);
        }
      }
    } catch (e: any) {
      setUploadError(e?.response?.data?.detail || 'Failed to extract text from uploaded file');
    } finally {
      setUploading(false);
    }
  };

  // ── Agenda mapping helpers ─────────────────────────────────────────────────
  const assignManualSection = (meetingAgendaId: string, momIdx: number) => {
    setAgendaMapping(prev => {
      const current = prev[meetingAgendaId] || [];
      if (current.includes(momIdx)) return prev;
      return { ...prev, [meetingAgendaId]: [...current, momIdx] };
    });
  };

  const removeManualSection = (meetingAgendaId: string, momIdx: number) => {
    setAgendaMapping(prev => {
      const current = prev[meetingAgendaId] || [];
      return { ...prev, [meetingAgendaId]: current.filter(i => i !== momIdx) };
    });
  };

  const setSingleManualSection = (meetingAgendaId: string, momIdxStr: string) => {
    setAgendaMapping(prev => {
      if (momIdxStr === '' || momIdxStr === '__unmapped__') {
        return { ...prev, [meetingAgendaId]: [] };
      }
      const momIdx = parseInt(momIdxStr, 10);
      return { ...prev, [meetingAgendaId]: [momIdx] };
    });
  };

  // ── Derived statistics & helpers ───────────────────────────────────────────
  const totalLongPoints = useMemo(
    () => longRomData?.agendas.reduce((s, ag) => s + ag.discussion_points.length, 0) ?? 0,
    [longRomData]
  );

  const totalManualPoints = useMemo(
    () => momAgendas.reduce((s, ag) => s + ag.points.length, 0),
    [momAgendas]
  );

  // Set of momAgenda indices that are mapped to at least one meeting agenda
  const mappedMomIndices = useMemo(() => {
    const s = new Set<number>();
    for (const indices of Object.values(agendaMapping)) {
      for (const idx of indices) s.add(idx);
    }
    return s;
  }, [agendaMapping]);

  // Unmapped manual sections
  const unmappedMomSections = useMemo(
    () => momAgendas.map((ag, i) => ({ ...ag, idx: i })).filter(ag => !mappedMomIndices.has(ag.idx)),
    [momAgendas, mappedMomIndices]
  );

  // Number of meeting agendas with at least 1 manual section mapped
  const mappedAgendasCount = useMemo(() => {
    if (!longRomData?.agendas) return 0;
    return longRomData.agendas.filter(ag => (agendaMapping[ag.agenda_id] || []).length > 0).length;
  }, [longRomData, agendaMapping]);

  const filteredMeetings = useMemo(() => {
    if (!meetingFilter.trim()) return meetings;
    const q = meetingFilter.toLowerCase();
    return meetings.filter(m => m.title.toLowerCase().includes(q));
  }, [meetings, meetingFilter]);

  // ── Save mapped agendas to database ────────────────────────────────────────
  const handleSave = async () => {
    if (!selectedMeeting || !longRomData || mappedAgendasCount === 0) return;
    setSaving(true);
    setSaveError('');
    setSaveSuccess(false);

    try {
      const pairs: any[] = [];
      for (const ag of longRomData.agendas) {
        const momIndices = agendaMapping[ag.agenda_id] || [];
        if (momIndices.length === 0) continue;

        // Gather all manual points under mapped sections
        const manualPoints: string[] = [];
        for (const idx of momIndices) {
          const mAg = momAgendas[idx];
          if (mAg && Array.isArray(mAg.points)) {
            for (const pt of mAg.points) {
              if (pt && pt.trim()) manualPoints.push(pt.trim());
            }
          }
        }

        if (manualPoints.length > 0) {
          pairs.push({
            agenda_id: ag.agenda_id,
            agenda_title: ag.agenda_title,
            long_rom_points: ag.discussion_points,
            manual_mom_points: manualPoints,
          });
        }
      }

      if (pairs.length === 0) {
        setSaveError('No valid agenda mappings found. Assign at least one Manual ROM section to an agenda.');
        return;
      }

      await saveTrainingData({
        recording_id: selectedMeeting.id,
        recording_title: selectedMeeting.title,
        pairs,
      });

      setSaveSuccess(true);
      const dataRes = await getDataForMeeting(selectedMeeting.id);
      setExistingData(dataRes.data || []);
    } catch (e: any) {
      setSaveError(e?.response?.data?.detail || 'Failed to save training data to database');
    } finally {
      setSaving(false);
    }
  };

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>

      {/* ════ TOP CONTROL BAR: Step 1 (Meeting) + Step 2 (Upload) ════════════ */}
      <div style={{ display: 'flex', gap: '0.85rem', alignItems: 'stretch', flexWrap: 'wrap' }}>

        {/* Meeting picker */}
        <div style={{ flex: '1 1 320px', position: 'relative' }} ref={meetingPickerRef}>
          <div style={{ fontSize: '0.68rem', fontWeight: 700, color: C.mutedFg, marginBottom: 5, letterSpacing: '0.05em', textTransform: 'uppercase' }}>
            Step 1 — Select Meeting
          </div>
          <button
            onClick={() => setShowMeetingPicker(v => !v)}
            style={{
              width: '100%', display: 'flex', alignItems: 'center', gap: 8,
              padding: '0.6rem 0.9rem', borderRadius: 10,
              border: `1.5px solid ${selectedMeeting ? C.accentBorder : C.border}`,
              background: selectedMeeting ? C.accentSoft : C.bg,
              cursor: 'pointer', fontFamily: 'Inter, sans-serif', color: C.fg,
            }}
          >
            {loadingRom
              ? <Loader size={14} className="spin" color={C.mutedFg} />
              : <FileText size={14} color={selectedMeeting ? C.accent : C.mutedFg} />
            }
            <span style={{ flex: 1, textAlign: 'left', fontSize: '0.82rem', fontWeight: selectedMeeting ? 600 : 400, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              {selectedMeeting ? selectedMeeting.title : 'Choose a meeting…'}
            </span>
            {selectedMeeting && !loadingRom && longRomData?.available && (
              <span style={{ fontSize: '0.62rem', color: C.mutedFg, flexShrink: 0 }}>
                {longRomData.agendas.length} agendas · {totalLongPoints} pts
              </span>
            )}
            <ChevronDown size={13} color={C.mutedFg} style={{ flexShrink: 0 }} />
          </button>

          {showMeetingPicker && (
            <div style={{
              position: 'absolute', top: 'calc(100% + 4px)', left: 0, right: 0, zIndex: 200,
              background: C.card, border: `1px solid ${C.border}`, borderRadius: 10,
              boxShadow: '0 10px 30px rgba(0,0,0,.2)', overflow: 'hidden',
            }}>
              <div style={{ padding: '8px 8px 4px' }}>
                <input
                  autoFocus
                  type="text"
                  placeholder="Search meetings…"
                  value={meetingFilter}
                  onChange={e => setMeetingFilter(e.target.value)}
                  style={{
                    width: '100%', padding: '0.4rem 0.65rem',
                    border: `1px solid ${C.border}`, borderRadius: 7,
                    background: C.bg, color: C.fg,
                    fontSize: '0.78rem', fontFamily: 'Inter, sans-serif', outline: 'none',
                  }}
                />
              </div>
              <div style={{ maxHeight: 260, overflowY: 'auto' }}>
                {loadingMeetings && (
                  <div style={{ padding: '1.2rem', textAlign: 'center', color: C.mutedFg, fontSize: '0.8rem' }}>
                    <Loader size={14} className="spin" style={{ verticalAlign: 'middle', marginRight: 6 }} />Loading…
                  </div>
                )}
                {filteredMeetings.map(m => (
                  <button
                    key={m.id}
                    onClick={() => handleSelectMeeting(m)}
                    style={{
                      width: '100%', display: 'flex', alignItems: 'center', gap: 8,
                      padding: '0.55rem 0.9rem',
                      background: selectedMeeting?.id === m.id ? C.accentSoft : 'transparent',
                      border: 'none', cursor: 'pointer', textAlign: 'left', fontFamily: 'Inter, sans-serif',
                    }}
                  >
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{ fontSize: '0.8rem', fontWeight: 600, color: C.fg, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{m.title}</div>
                      <div style={{ fontSize: '0.66rem', color: C.mutedFg }}>{m.created_at?.slice(0, 10)}</div>
                    </div>
                    {m.has_long_rom && (
                      <span style={{ fontSize: '0.6rem', fontWeight: 700, padding: '1px 5px', borderRadius: 5, background: C.greenSoft, color: C.green, flexShrink: 0 }}>ROM ✓</span>
                    )}
                  </button>
                ))}
                {filteredMeetings.length === 0 && !loadingMeetings && (
                  <div style={{ padding: '1.2rem', textAlign: 'center', color: C.mutedFg, fontSize: '0.78rem' }}>No meetings found</div>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Upload manual ROM */}
        <div style={{ flex: '1 1 320px' }}>
          <div style={{ fontSize: '0.68rem', fontWeight: 700, color: C.mutedFg, marginBottom: 5, letterSpacing: '0.05em', textTransform: 'uppercase' }}>
            Step 2 — Upload Manual ROM
          </div>
          <div style={{
            display: 'flex', alignItems: 'center', gap: 8,
            padding: '0.6rem 0.9rem', borderRadius: 10,
            border: `1.5px solid ${momAgendas.length > 0 ? C.purpleBorder : C.border}`,
            background: momAgendas.length > 0 ? C.purpleSoft : C.bg,
          }}>
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf,.docx,.doc,.txt,.md"
              style={{ display: 'none' }}
              onChange={e => e.target.files?.[0] && handleFileUpload(e.target.files[0])}
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={!selectedMeeting || !longRomData?.available || uploading}
              style={{
                display: 'inline-flex', alignItems: 'center', gap: 5,
                padding: '0.38rem 0.85rem', borderRadius: 7, border: 'none',
                cursor: !selectedMeeting || !longRomData?.available || uploading ? 'not-allowed' : 'pointer',
                background: C.accent, color: 'white',
                fontSize: '0.76rem', fontWeight: 600, fontFamily: 'Inter, sans-serif',
                opacity: !selectedMeeting || !longRomData?.available ? 0.45 : 1,
                flexShrink: 0,
              }}
            >
              {uploading ? <Loader size={12} className="spin" /> : <Upload size={12} />}
              {uploading ? 'Extracting…' : 'Choose File'}
            </button>

            <div style={{ flex: 1, minWidth: 0 }}>
              {(uploading || matching) && (
                <div style={{ fontSize: '0.73rem', color: C.mutedFg, display: 'flex', alignItems: 'center', gap: 5 }}>
                  <Loader size={11} className="spin" />
                  {uploading ? 'Extracting document text & structure…' : 'Matching agendas…'}
                </div>
              )}
              {!uploading && !matching && momAgendas.length > 0 && (
                <>
                  <div style={{ fontSize: '0.76rem', fontWeight: 600, color: C.purple, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    <Check size={11} style={{ verticalAlign: 'middle', marginRight: 3 }} />{uploadedFileName}
                  </div>
                  <div style={{ fontSize: '0.66rem', color: C.mutedFg }}>
                    {momAgendas.length} section{momAgendas.length !== 1 ? 's' : ''} · {totalManualPoints} content points
                  </div>
                </>
              )}
              {!uploading && !matching && !momAgendas.length && !uploadError && (
                <div style={{ fontSize: '0.73rem', color: C.mutedFg }}>
                  {!selectedMeeting ? 'Select a meeting first'
                    : !longRomData?.available ? 'Generate Long ROM for this meeting first'
                    : 'PDF, DOCX, DOC, or TXT'}
                </div>
              )}
              {uploadError && (
                <div style={{ fontSize: '0.7rem', color: C.red }}>
                  <AlertCircle size={11} style={{ verticalAlign: 'middle', marginRight: 3 }} />{uploadError}
                </div>
              )}
            </div>

            {momAgendas.length > 0 && (
              <button
                onClick={() => fileInputRef.current?.click()}
                style={{ background: 'none', border: 'none', cursor: 'pointer', padding: 2, color: C.mutedFg, flexShrink: 0 }}
                title="Re-upload a different document"
              >
                <RefreshCw size={13} />
              </button>
            )}
          </div>
        </div>

        {/* Existing saved badge */}
        {existingData.length > 0 && (
          <div style={{
            display: 'flex', alignItems: 'center', gap: 6,
            padding: '0 0.85rem', borderRadius: 10,
            border: `1px solid ${C.greenBorder}`, background: C.greenSoft,
            alignSelf: 'flex-end', height: 42,
            fontSize: '0.73rem', fontWeight: 600, color: C.green,
          }}>
            <Database size={13} color={C.green} />
            {existingData.length} agenda{existingData.length !== 1 ? 's' : ''} saved in database
          </div>
        )}
      </div>

      {/* ── Warning if meeting has no Long ROM ─────────────────────────────── */}
      {selectedMeeting && !loadingRom && longRomData && !longRomData.available && (
        <div style={{
          display: 'flex', alignItems: 'center', gap: 10, padding: '0.85rem 1.1rem',
          borderRadius: 10, border: `1px solid ${C.amberBorder}`, background: C.amberSoft,
        }}>
          <AlertCircle size={16} color={C.amber} style={{ flexShrink: 0 }} />
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: '0.82rem', fontWeight: 600, color: C.amber }}>Long ROM not available for this meeting</div>
            <div style={{ fontSize: '0.72rem', color: C.mutedFg }}>Generate the Long ROM first, then return here to map with manual ROM.</div>
          </div>
          <button
            onClick={() => navigate(`/dashboard/history/${selectedMeeting.id}/rom`)}
            style={{
              display: 'inline-flex', alignItems: 'center', gap: 5, padding: '0.38rem 0.8rem',
              borderRadius: 7, border: 'none', cursor: 'pointer',
              background: C.accent, color: 'white', fontSize: '0.74rem', fontWeight: 600, fontFamily: 'Inter, sans-serif',
            }}
          >
            <ExternalLink size={12} /> Go to ROM
          </button>
        </div>
      )}

      {/* ════ MAIN WORKSPACE: Side-by-Side Agenda Mapping ═════════════════════ */}
      {selectedMeeting && longRomData?.available && (
        <>
          {/* Section column title headers */}
          <div style={{
            display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.85rem',
            padding: '0 0.2rem',
          }}>
            <div style={{
              display: 'flex', alignItems: 'center', gap: 7,
              padding: '0.5rem 0.75rem', borderRadius: 8,
              background: C.accentSoft, border: `1px solid ${C.accentBorder}`,
            }}>
              <BookOpen size={14} color={C.accent} />
              <span style={{ fontSize: '0.82rem', fontWeight: 700, color: C.accent, flex: 1 }}>Generated Long Points</span>
              <span style={{ fontSize: '0.67rem', color: C.mutedFg }}>
                {longRomData.agendas.length} agendas · {totalLongPoints} total points
              </span>
            </div>

            <div style={{
              display: 'flex', alignItems: 'center', gap: 7,
              padding: '0.5rem 0.75rem', borderRadius: 8,
              background: C.purpleSoft, border: `1px solid ${C.purpleBorder}`,
            }}>
              <Layers size={14} color={C.purple} />
              <span style={{ fontSize: '0.82rem', fontWeight: 700, color: C.purple, flex: 1 }}>Manual ROM</span>
              <span style={{ fontSize: '0.67rem', color: C.mutedFg }}>
                {momAgendas.length > 0 ? `${momAgendas.length} sections · ${totalManualPoints} points` : 'Upload file above'}
              </span>
            </div>
          </div>

          {/* Agenda-by-Agenda Rows */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
            {longRomData.agendas.map((ag, agIdx) => {
              const mappedIndices = agendaMapping[ag.agenda_id] || [];
              const isMapped = mappedIndices.length > 0;
              const isCollapsed = collapsedAgendas.has(ag.agenda_id);

              // Find auto match info for badge
              const matchInfo = autoMatches.find(m => m.meeting_agenda_id === ag.agenda_id);

              return (
                <div
                  key={ag.agenda_id}
                  style={{
                    border: `1.5px solid ${isMapped ? C.greenBorder : C.border}`,
                    borderRadius: 12, overflow: 'hidden', background: C.card,
                    boxShadow: '0 1px 3px rgba(0,0,0,0.04)',
                    transition: 'border-color 0.15s',
                  }}
                >
                  {/* ── Unified Agenda Header ───────────────────────────────── */}
                  <div style={{
                    padding: '0.6rem 0.9rem',
                    background: isMapped ? C.greenSoft : 'hsl(var(--muted) / .4)',
                    borderBottom: `1px solid ${isMapped ? C.greenBorder : C.border}`,
                    display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap',
                  }}>
                    {/* Expand/Collapse toggle */}
                    <button
                      onClick={() => setCollapsedAgendas(prev => {
                        const n = new Set(prev);
                        n.has(ag.agenda_id) ? n.delete(ag.agenda_id) : n.add(ag.agenda_id);
                        return n;
                      })}
                      style={{ background: 'none', border: 'none', cursor: 'pointer', padding: 0, color: C.fg, display: 'flex', alignItems: 'center' }}
                    >
                      {isCollapsed ? <ChevronRight size={14} /> : <ChevronDown size={14} />}
                    </button>

                    {/* Agenda index & title */}
                    <div style={{ flex: 1, minWidth: 200, display: 'flex', alignItems: 'center', gap: 7 }}>
                      <span style={{
                        fontSize: '0.67rem', fontWeight: 800, padding: '2px 6px',
                        borderRadius: 5, background: C.bg, border: `1px solid ${C.border}`,
                        color: C.mutedFg, letterSpacing: '0.04em',
                      }}>
                        AGENDA {agIdx + 1}
                      </span>
                      <span style={{ fontSize: '0.84rem', fontWeight: 700, color: C.fg }}>
                        {ag.agenda_title}
                      </span>
                      <span style={{ fontSize: '0.68rem', color: C.mutedFg }}>
                        ({ag.discussion_points.length} long point{ag.discussion_points.length !== 1 ? 's' : ''})
                      </span>
                    </div>

                    {/* Mapping control dropdown */}
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                      {momAgendas.length > 0 && (
                        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                          <span style={{ fontSize: '0.68rem', fontWeight: 600, color: C.mutedFg }}>
                            Manual Agenda:
                          </span>
                          <select
                            value={mappedIndices.length === 1 ? mappedIndices[0].toString() : mappedIndices.length > 1 ? '__multiple__' : ''}
                            onChange={e => setSingleManualSection(ag.agenda_id, e.target.value)}
                            style={{
                              fontSize: '0.74rem', padding: '0.28rem 0.55rem', borderRadius: 7,
                              border: `1.5px solid ${isMapped ? C.green : C.border}`,
                              background: C.bg, color: C.fg, fontFamily: 'Inter, sans-serif',
                              cursor: 'pointer', outline: 'none', maxWidth: 240,
                            }}
                          >
                            <option value="">— Unmapped (None) —</option>
                            {momAgendas.map((mAg, mIdx) => (
                              <option key={mIdx} value={mIdx.toString()}>
                                {mAg.agenda_title} ({mAg.points.length} pts)
                              </option>
                            ))}
                          </select>
                        </div>
                      )}

                      {/* Status indicator badge */}
                      {isMapped ? (
                        <span style={{
                          fontSize: '0.65rem', fontWeight: 700, padding: '2px 7px',
                          borderRadius: 6, background: C.green, color: 'white',
                          display: 'flex', alignItems: 'center', gap: 4, flexShrink: 0,
                        }}>
                          <Check size={10} /> Mapped
                        </span>
                      ) : (
                        <span style={{
                          fontSize: '0.65rem', fontWeight: 600, padding: '2px 7px',
                          borderRadius: 6, background: C.amberSoft, color: C.amber,
                          border: `1px solid ${C.amberBorder}`, flexShrink: 0,
                        }}>
                          Unmapped
                        </span>
                      )}

                      {matchInfo && matchInfo.confidence > 0 && (
                        <span style={{ fontSize: '0.62rem', color: C.mutedFg }}>
                          {(matchInfo.confidence * 100).toFixed(0)}% auto-confidence
                        </span>
                      )}
                    </div>
                  </div>

                  {/* ── Side-by-Side Body ───────────────────────────────────── */}
                  {!isCollapsed && (
                    <div style={{
                      display: 'grid', gridTemplateColumns: '1fr 1fr',
                      borderTop: 'none',
                    }}>
                      {/* Left: Generated Long Points */}
                      <div style={{
                        padding: '0.75rem 1rem',
                        borderRight: `1px solid ${C.border}`,
                        display: 'flex', flexDirection: 'column', gap: 7,
                        background: 'hsl(var(--muted) / .08)',
                      }}>
                        <div style={{ fontSize: '0.68rem', fontWeight: 700, color: C.accent, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                          Generated Discussion Points ({ag.discussion_points.length})
                        </div>

                        {ag.discussion_points.length === 0 ? (
                          <div style={{ fontSize: '0.75rem', color: C.mutedFg, fontStyle: 'italic', padding: '0.5rem 0' }}>
                            No discussion points in this agenda.
                          </div>
                        ) : (
                          ag.discussion_points.map((pt, ptIdx) => {
                            const ptText = pt.polished_text || pt.text || (typeof pt === 'string' ? pt : JSON.stringify(pt));
                            const speaker = pt.speaker || pt.speakers?.[0] || null;

                            return (
                              <div
                                key={ptIdx}
                                style={{
                                  display: 'flex', alignItems: 'flex-start', gap: 8,
                                  fontSize: '0.76rem', color: C.fg, lineHeight: 1.55,
                                  padding: '0.35rem 0.5rem', borderRadius: 6,
                                  background: 'hsl(var(--card))',
                                  border: `1px solid hsl(var(--border) / .6)`,
                                }}
                              >
                                <span style={{ color: C.accent, fontWeight: 700, marginTop: 1 }}>•</span>
                                <div style={{ flex: 1, minWidth: 0 }}>
                                  {speaker && (
                                    <span style={{
                                      fontSize: '0.62rem', fontWeight: 700,
                                      color: C.mutedFg, marginRight: 6,
                                      textTransform: 'uppercase', letterSpacing: '0.04em',
                                    }}>
                                      [{speaker}]
                                    </span>
                                  )}
                                  <span>{ptText}</span>
                                </div>
                              </div>
                            );
                          })
                        )}
                      </div>

                      {/* Right: Manual ROM Content */}
                      <div style={{
                        padding: '0.75rem 1rem',
                        display: 'flex', flexDirection: 'column', gap: 7,
                        background: isMapped ? C.purpleSoft : 'hsl(var(--muted) / .04)',
                      }}>
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                          <span style={{ fontSize: '0.68rem', fontWeight: 700, color: C.purple, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                            Manual ROM Content
                          </span>

                          {/* Multi-section selector if needed */}
                          {momAgendas.length > 0 && (
                            <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                              <select
                                value=""
                                onChange={e => {
                                  if (e.target.value) {
                                    assignManualSection(ag.agenda_id, parseInt(e.target.value, 10));
                                  }
                                }}
                                style={{
                                  fontSize: '0.64rem', padding: '1px 5px', borderRadius: 5,
                                  border: `1px solid ${C.border}`, background: C.bg, color: C.fg,
                                  fontFamily: 'Inter, sans-serif', cursor: 'pointer', outline: 'none',
                                }}
                              >
                                <option value="">+ Add manual section</option>
                                {momAgendas.map((mAg, mIdx) => (
                                  <option key={mIdx} value={mIdx.toString()} disabled={mappedIndices.includes(mIdx)}>
                                    {mAg.agenda_title}
                                  </option>
                                ))}
                              </select>
                            </div>
                          )}
                        </div>

                        {/* Mapped section tags */}
                        {mappedIndices.length > 0 && (
                          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5, marginBottom: 2 }}>
                            {mappedIndices.map(mIdx => {
                              const mAg = momAgendas[mIdx];
                              if (!mAg) return null;
                              return (
                                <span
                                  key={mIdx}
                                  style={{
                                    display: 'inline-flex', alignItems: 'center', gap: 4,
                                    fontSize: '0.67rem', fontWeight: 600, padding: '2px 7px',
                                    borderRadius: 6, background: C.purple, color: 'white',
                                  }}
                                >
                                  <Tag size={10} />
                                  {mAg.agenda_title}
                                  <button
                                    onClick={() => removeManualSection(ag.agenda_id, mIdx)}
                                    style={{
                                      background: 'none', border: 'none', cursor: 'pointer',
                                      color: 'white', padding: 0, marginLeft: 2,
                                      display: 'inline-flex', alignItems: 'center',
                                    }}
                                    title="Unmap this section"
                                  >
                                    <X size={11} />
                                  </button>
                                </span>
                              );
                            })}
                          </div>
                        )}

                        {/* Mapped points content */}
                        {mappedIndices.length === 0 ? (
                          <div style={{
                            display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
                            padding: '2rem 1rem', textAlign: 'center', color: C.mutedFg, gap: 8,
                          }}>
                            <Layers size={22} style={{ opacity: 0.3 }} />
                            <div style={{ fontSize: '0.76rem' }}>
                              {momAgendas.length > 0
                                ? 'No manual ROM section mapped to this agenda.'
                                : 'Upload a manual ROM file above to view and map content.'}
                            </div>
                          </div>
                        ) : (
                          mappedIndices.map(mIdx => {
                            const mAg = momAgendas[mIdx];
                            if (!mAg) return null;

                            return (
                              <div key={mIdx} style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                                {mappedIndices.length > 1 && (
                                  <div style={{ fontSize: '0.67rem', fontWeight: 700, color: C.purple }}>
                                    {mAg.agenda_title}:
                                  </div>
                                )}
                                {mAg.points.map((pointText, pIdx) => (
                                  <div
                                    key={pIdx}
                                    style={{
                                      display: 'flex', alignItems: 'flex-start', gap: 8,
                                      fontSize: '0.76rem', color: C.fg, lineHeight: 1.55,
                                      padding: '0.35rem 0.5rem', borderRadius: 6,
                                      background: 'hsl(var(--card))',
                                      border: `1px solid ${C.purpleBorder}`,
                                    }}
                                  >
                                    <span style={{ color: C.purple, fontWeight: 700, marginTop: 1 }}>•</span>
                                    <span style={{ flex: 1 }}>{pointText}</span>
                                  </div>
                                ))}
                              </div>
                            );
                          })
                        )}
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          {/* ── Unmapped Manual Sections Drawer ──────────────────────────────── */}
          {momAgendas.length > 0 && unmappedMomSections.length > 0 && (
            <div style={{
              border: `1px solid ${C.amberBorder}`, borderRadius: 10,
              background: C.amberSoft, overflow: 'hidden',
            }}>
              <div
                onClick={() => setShowUnmappedDrawer(v => !v)}
                style={{
                  padding: '0.6rem 0.9rem', display: 'flex', alignItems: 'center', gap: 8,
                  cursor: 'pointer', userSelect: 'none',
                }}
              >
                <AlertCircle size={14} color={C.amber} />
                <span style={{ fontSize: '0.78rem', fontWeight: 700, color: C.amber, flex: 1 }}>
                  Unmapped Manual ROM Sections ({unmappedMomSections.length})
                </span>
                <span style={{ fontSize: '0.68rem', color: C.mutedFg }}>
                  These sections are not yet mapped to any generated agenda
                </span>
                {showUnmappedDrawer ? <ChevronDown size={13} color={C.amber} /> : <ChevronRight size={13} color={C.amber} />}
              </div>

              {showUnmappedDrawer && (
                <div style={{
                  padding: '0.7rem 0.9rem', display: 'flex', flexDirection: 'column', gap: 8,
                  borderTop: `1px solid ${C.amberBorder}`, background: C.card,
                }}>
                  {unmappedMomSections.map(sec => (
                    <div
                      key={sec.idx}
                      style={{
                        padding: '0.5rem 0.75rem', borderRadius: 8,
                        border: `1px solid ${C.border}`, background: C.bg,
                        display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 12,
                      }}
                    >
                      <div style={{ flex: 1 }}>
                        <div style={{ fontSize: '0.78rem', fontWeight: 700, color: C.fg, marginBottom: 4 }}>
                          {sec.agenda_title}
                        </div>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                          {sec.points.slice(0, 3).map((p, pi) => (
                            <div key={pi} style={{ fontSize: '0.72rem', color: C.mutedFg }}>
                              • {p}
                            </div>
                          ))}
                          {sec.points.length > 3 && (
                            <div style={{ fontSize: '0.68rem', color: C.mutedFg, fontStyle: 'italic' }}>
                              + {sec.points.length - 3} more items…
                            </div>
                          )}
                        </div>
                      </div>

                      {/* Quick assign dropdown */}
                      <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexShrink: 0 }}>
                        <span style={{ fontSize: '0.68rem', color: C.mutedFg }}>Map to:</span>
                        <select
                          value=""
                          onChange={e => {
                            if (e.target.value) assignManualSection(e.target.value, sec.idx);
                          }}
                          style={{
                            fontSize: '0.72rem', padding: '0.25rem 0.5rem', borderRadius: 6,
                            border: `1px solid ${C.border}`, background: C.bg, color: C.fg,
                            fontFamily: 'Inter, sans-serif', cursor: 'pointer', outline: 'none',
                          }}
                        >
                          <option value="">— Select Agenda —</option>
                          {longRomData.agendas.map(ag => (
                            <option key={ag.agenda_id} value={ag.agenda_id}>
                              {ag.agenda_title}
                            </option>
                          ))}
                        </select>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* ════ BOTTOM SAVE BAR ═════════════════════════════════════════════ */}
          <div style={{
            display: 'flex', alignItems: 'center', gap: 12,
            padding: '0.75rem 1.1rem', borderRadius: 10,
            border: `1px solid ${C.border}`, background: C.card,
            boxShadow: '0 2px 8px rgba(0,0,0,0.04)',
          }}>
            {/* Stats */}
            <div style={{ flex: 1, display: 'flex', gap: 16, flexWrap: 'wrap', alignItems: 'center' }}>
              <span style={{ fontSize: '0.76rem', color: C.mutedFg }}>
                <strong style={{ color: C.fg }}>{mappedAgendasCount}</strong> of{' '}
                <strong style={{ color: C.fg }}>{longRomData.agendas.length}</strong> agendas mapped
              </span>
              {saveSuccess && (
                <span style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: '0.76rem', color: C.green, fontWeight: 600 }}>
                  <Check size={14} /> Training data saved to database successfully!
                </span>
              )}
              {saveError && (
                <span style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: '0.76rem', color: C.red }}>
                  <AlertCircle size={14} /> {saveError}
                </span>
              )}
            </div>

            {/* Save to Database Button */}
            <button
              id="rom-training-save-btn"
              onClick={handleSave}
              disabled={saving || mappedAgendasCount === 0}
              style={{
                display: 'inline-flex', alignItems: 'center', gap: 7,
                padding: '0.55rem 1.3rem', borderRadius: 9, border: 'none',
                cursor: saving || mappedAgendasCount === 0 ? 'not-allowed' : 'pointer',
                background: mappedAgendasCount > 0 ? C.accent : C.muted,
                color: mappedAgendasCount > 0 ? 'white' : C.mutedFg,
                fontSize: '0.84rem', fontWeight: 700, fontFamily: 'Inter, sans-serif',
                opacity: saving ? 0.7 : 1, transition: 'all 0.15s',
                boxShadow: mappedAgendasCount > 0 ? `0 0 16px hsl(var(--accent) / .3)` : 'none',
              }}
            >
              {saving ? <Loader size={14} className="spin" /> : <Database size={14} />}
              {saving ? 'Saving…' : `Save ${mappedAgendasCount} Mapped Agenda${mappedAgendasCount !== 1 ? 's' : ''} to Database`}
            </button>
          </div>
        </>
      )}

      {/* ── Empty State: No meeting selected ──────────────────────────────── */}
      {!selectedMeeting && (
        <div style={{
          display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
          padding: '4.5rem 2rem', gap: 16, color: C.mutedFg,
          border: `1px dashed ${C.border}`, borderRadius: 12, background: C.card,
        }}>
          <BookOpen size={44} style={{ opacity: 0.2 }} />
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '0.96rem', fontWeight: 700, color: C.fg, marginBottom: 6 }}>
              Select a Meeting to Begin
            </div>
            <div style={{ fontSize: '0.78rem', maxWidth: 420, lineHeight: 1.6 }}>
              Choose a meeting with generated Long ROM points, then upload your manually written ROM document
              to map agendas and save training data.
            </div>
          </div>
          <div style={{ display: 'flex', gap: 24, marginTop: 6 }}>
            {[
              { n: '1', label: 'Select Meeting' },
              { n: '2', label: 'Upload Manual ROM' },
              { n: '3', label: 'Review Agenda Mapping' },
              { n: '4', label: 'Save to Database' },
            ].map(step => (
              <div key={step.n} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 5 }}>
                <div style={{
                  width: 28, height: 28, borderRadius: '50%',
                  border: `1.5px solid ${C.accentBorder}`, color: C.accent,
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  fontSize: '0.78rem', fontWeight: 700,
                }}>
                  {step.n}
                </div>
                <div style={{ fontSize: '0.68rem', color: C.mutedFg, textAlign: 'center' }}>{step.label}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
