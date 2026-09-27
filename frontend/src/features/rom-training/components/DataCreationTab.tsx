import { useState, useEffect, useRef, useMemo } from 'react';
import {
  Upload, FileText, Check, AlertCircle, Loader,
  ExternalLink, Database, RefreshCw, X,
  ChevronDown, ChevronRight, Info, BookOpen, Layers,
  ArrowRight, Plus, Sparkles, Tag, ArrowRightLeft,
  Trash2, Edit2, RotateCcw, CheckSquare, Square,
  GitMerge, PlusCircle, ClipboardPaste, MoveRight,
} from 'lucide-react';
import {
  listMeetings, getLongRomPoints, extractMom,
  matchAgendas, saveTrainingData, getDataForMeeting,
} from '../api/romTrainingApi';
import type { Meeting, LongRomAgenda, AgendaMatch } from '../types/romTrainingTypes';
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

// ── Types for Local State (guarantees DB data remains untouched) ─────────────
interface LocalLongRomAgenda {
  agenda_id: string;
  agenda_title: string;
  discussion_points: any[];
}

interface LocalMomAgenda {
  id: string;
  agenda_title: string;
  points: string[];
}

export default function DataCreationTab() {
  const navigate = useNavigate();

  // ── Meeting state ──────────────────────────────────────────────────────────
  const [meetings, setMeetings] = useState<Meeting[]>([]);
  const [loadingMeetings, setLoadingMeetings] = useState(false);
  const [selectedMeeting, setSelectedMeeting] = useState<Meeting | null>(null);
  const [showMeetingPicker, setShowMeetingPicker] = useState(false);
  const [meetingFilter, setMeetingFilter] = useState('');
  const meetingPickerRef = useRef<HTMLDivElement>(null);

  // ── Custom / Manual session mode (when not starting from a meeting) ─────────
  const [isManualSession, setIsManualSession] = useState(false);
  const [manualSessionTitle, setManualSessionTitle] = useState('');

  // ── Long ROM state & Local Agendas ─────────────────────────────────────────
  const [longRomAvailable, setLongRomAvailable] = useState<boolean | null>(null);
  const [originalAgendas, setOriginalAgendas] = useState<LocalLongRomAgenda[]>([]);
  const [localAgendas, setLocalAgendas] = useState<LocalLongRomAgenda[]>([]);
  const [loadingRom, setLoadingRom] = useState(false);
  const [existingData, setExistingData] = useState<any[]>([]);

  // ── Local Agendas Editing & Grouping controls ──────────────────────────────
  const [selectedAgendaIds, setSelectedAgendaIds] = useState<Set<string>>(new Set());
  const [showMergeModal, setShowMergeModal] = useState(false);
  const [mergeTargetTitle, setMergeTargetTitle] = useState('');
  const [movingPointInfo, setMovingPointInfo] = useState<{ agendaId: string; pointIdx: number } | null>(null);
  const [editingPoint, setEditingPoint] = useState<{ agendaId: string; pointIdx: number; text: string } | null>(null);
  const [addingPointAgendaId, setAddingPointAgendaId] = useState<string | null>(null);
  const [newPointInputText, setNewPointInputText] = useState('');
  const [showAddAgendaModal, setShowAddAgendaModal] = useState(false);
  const [newAgendaTitleInput, setNewAgendaTitleInput] = useState('');
  const [editingAgendaTitleId, setEditingAgendaTitleId] = useState<string | null>(null);
  const [agendaTitleEditText, setAgendaTitleEditText] = useState('');

  // ── Manual MOM state (File upload OR Direct Creation) ──────────────────────
  const [uploading, setUploading] = useState(false);
  const [uploadedFileName, setUploadedFileName] = useState('');
  const [uploadError, setUploadError] = useState('');
  const [momAgendas, setMomAgendas] = useState<LocalMomAgenda[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Direct Manual MOM input states
  const [showCreateManualModal, setShowCreateManualModal] = useState(false);
  const [newManualAgendaTitle, setNewManualAgendaTitle] = useState('');
  const [pasteBoxText, setPasteBoxText] = useState<Record<string, string>>({});
  const [quickPointText, setQuickPointText] = useState<Record<string, string>>({});
  const [editingManualPoint, setEditingManualPoint] = useState<{ momId: string; pointIdx: number; text: string } | null>(null);
  const [editingMomTitleId, setEditingMomTitleId] = useState<string | null>(null);
  const [momTitleEditText, setMomTitleEditText] = useState('');

  // ── Agenda Mapping state: meeting_agenda_id -> array of LocalMomAgenda.id ───
  const [agendaMapping, setAgendaMapping] = useState<Record<string, string[]>>({});
  const [matching, setMatching] = useState(false);
  const [autoMatches, setAutoMatches] = useState<AgendaMatch[]>([]);

  // ── Accordion / Collapsed state ────────────────────────────────────────────
  const [collapsedAgendas, setCollapsedAgendas] = useState<Set<string>>(new Set());
  const [showUnmappedDrawer, setShowUnmappedDrawer] = useState(true);

  // ── Save state ("Create Training Dataset") ──────────────────────────────────
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
    setIsManualSession(false);
    setShowMeetingPicker(false);
    setMeetingFilter('');
    setOriginalAgendas([]);
    setLocalAgendas([]);
    setSelectedAgendaIds(new Set());
    setLongRomAvailable(null);
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

      const isAvail = !!romRes?.available && Array.isArray(romRes?.agendas) && romRes.agendas.length > 0;
      setLongRomAvailable(isAvail);

      if (isAvail) {
        const loaded: LocalLongRomAgenda[] = romRes.agendas.map((a: any, idx: number) => ({
          agenda_id: a.agenda_id || `ag_${idx}_${Date.now()}`,
          agenda_title: a.agenda_title || `Agenda ${idx + 1}`,
          discussion_points: Array.isArray(a.discussion_points) ? [...a.discussion_points] : [],
        }));
        // Store pristine copy from database & working local copy
        setOriginalAgendas(JSON.parse(JSON.stringify(loaded)));
        setLocalAgendas(JSON.parse(JSON.stringify(loaded)));
      }

      setExistingData(dataRes.data || []);
    } catch (err) {
      console.error('Failed to load ROM data:', err);
      setLongRomAvailable(false);
    } finally {
      setLoadingRom(false);
    }
  };

  // ── Start Manual Session (without pre-selecting a meeting) ─────────────────
  const handleStartManualSession = () => {
    setSelectedMeeting(null);
    setIsManualSession(true);
    setManualSessionTitle('Custom Manual Training Session');
    setOriginalAgendas([]);
    setLocalAgendas([]);
    setLongRomAvailable(true);
    setSelectedAgendaIds(new Set());
    setExistingData([]);
  };

  // ── File upload & extraction handler ───────────────────────────────────────
  const handleFileUpload = async (file: File) => {
    setUploading(true);
    setUploadError('');
    setUploadedFileName(file.name);
    setSaveSuccess(false);
    setSaveError('');

    try {
      const res = await extractMom(file);
      const extractedAgendas: any[] = res.agendas || [];

      // Convert extracted agendas to LocalMomAgenda with unique IDs
      const newMomAgendas: LocalMomAgenda[] = extractedAgendas.map((ag: any, idx: number) => ({
        id: `upload_mom_${Date.now()}_${idx}`,
        agenda_title: ag.agenda_title || `Uploaded Section ${idx + 1}`,
        points: Array.isArray(ag.points) ? ag.points.map((p: any) => String(p).trim()).filter(Boolean) : [],
      }));

      // Append to or replace momAgendas
      setMomAgendas(prev => [...prev, ...newMomAgendas]);

      // Auto-match agendas if meeting agendas exist
      if (localAgendas.length > 0 && newMomAgendas.length > 0) {
        setMatching(true);
        try {
          const matchRes = await matchAgendas(
            newMomAgendas.map(m => ({ agenda_title: m.agenda_title, points: m.points })),
            localAgendas.map(a => ({ agenda_id: a.agenda_id, agenda_title: a.agenda_title })),
          );
          const matches: AgendaMatch[] = matchRes.matches || [];
          setAutoMatches(matches);

          // Build mapping additions
          setAgendaMapping(prev => {
            const next = { ...prev };
            for (const ag of localAgendas) {
              if (!next[ag.agenda_id]) next[ag.agenda_id] = [];
            }
            for (const m of matches) {
              if (m.meeting_agenda_id && m.confidence >= 0.2) {
                const matchedMom = newMomAgendas[m.mom_agenda_idx];
                if (matchedMom && next[m.meeting_agenda_id]) {
                  if (!next[m.meeting_agenda_id].includes(matchedMom.id)) {
                    next[m.meeting_agenda_id].push(matchedMom.id);
                  }
                }
              }
            }
            return next;
          });
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

  // ── Manual MOM: Direct Agenda & Point Creation ─────────────────────────────
  const handleCreateManualMomAgenda = () => {
    const title = newManualAgendaTitle.trim();
    if (!title) return;

    const newId = `manual_mom_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`;
    const newMom: LocalMomAgenda = {
      id: newId,
      agenda_title: title,
      points: [],
    };

    setMomAgendas(prev => [...prev, newMom]);
    setNewManualAgendaTitle('');
    setShowCreateManualModal(false);
  };

  const handleParseAndAddPoints = (momId: string) => {
    const raw = pasteBoxText[momId] || '';
    if (!raw.trim()) return;

    const parsed = raw
      .split(/\r?\n/)
      .map(line => line.trim())
      // Strip leading bullets: •, -, *, –, or numbered sub-points like 1. or 1)
      .map(line => line.replace(/^(?:[\-\*•–]|\d+[\.\)]|[a-z][\.\)])\s+/, '').trim())
      .filter(line => line.length > 0);

    if (parsed.length === 0) return;

    setMomAgendas(prev => prev.map(m => {
      if (m.id === momId) {
        return { ...m, points: [...m.points, ...parsed] };
      }
      return m;
    }));

    setPasteBoxText(prev => ({ ...prev, [momId]: '' }));
  };

  const handleAddSingleManualPoint = (momId: string) => {
    const text = (quickPointText[momId] || '').trim();
    if (!text) return;

    setMomAgendas(prev => prev.map(m => {
      if (m.id === momId) {
        return { ...m, points: [...m.points, text] };
      }
      return m;
    }));

    setQuickPointText(prev => ({ ...prev, [momId]: '' }));
  };

  const handleDeleteManualPoint = (momId: string, pointIdx: number) => {
    setMomAgendas(prev => prev.map(m => {
      if (m.id === momId) {
        return { ...m, points: m.points.filter((_, idx) => idx !== pointIdx) };
      }
      return m;
    }));
  };

  const handleSaveEditManualPoint = (momId: string, pointIdx: number, newText: string) => {
    if (!newText.trim()) return;
    setMomAgendas(prev => prev.map(m => {
      if (m.id === momId) {
        const pts = [...m.points];
        pts[pointIdx] = newText.trim();
        return { ...m, points: pts };
      }
      return m;
    }));
    setEditingManualPoint(null);
  };

  const handleDeleteManualAgenda = (momId: string) => {
    setMomAgendas(prev => prev.filter(m => m.id !== momId));
    // Remove from agendaMapping
    setAgendaMapping(prev => {
      const next: Record<string, string[]> = {};
      for (const [k, v] of Object.entries(prev)) {
        next[k] = v.filter(id => id !== momId);
      }
      return next;
    });
  };

  const handleSaveMomTitle = (momId: string) => {
    if (!momTitleEditText.trim()) return;
    setMomAgendas(prev => prev.map(m => m.id === momId ? { ...m, agenda_title: momTitleEditText.trim() } : m));
    setEditingMomTitleId(null);
    setMomTitleEditText('');
  };

  // ── Meeting Long Points: Local Editing / Grouping ───────────────────────────

  // 1. Move Point
  const handleMovePoint = (sourceAgendaId: string, pointIdx: number, targetAgendaId: string) => {
    if (sourceAgendaId === targetAgendaId) {
      setMovingPointInfo(null);
      return;
    }

    setLocalAgendas(prev => {
      const sourceAgenda = prev.find(a => a.agenda_id === sourceAgendaId);
      const targetAgenda = prev.find(a => a.agenda_id === targetAgendaId);
      if (!sourceAgenda || !targetAgenda) return prev;

      const pointToMove = sourceAgenda.discussion_points[pointIdx];
      if (pointToMove === undefined) return prev;

      return prev.map(ag => {
        if (ag.agenda_id === sourceAgendaId) {
          return {
            ...ag,
            discussion_points: ag.discussion_points.filter((_, idx) => idx !== pointIdx),
          };
        }
        if (ag.agenda_id === targetAgendaId) {
          return {
            ...ag,
            discussion_points: [...ag.discussion_points, pointToMove],
          };
        }
        return ag;
      });
    });

    setMovingPointInfo(null);
  };

  // 2. Merge Selected Agendas
  const handleMergeSelectedAgendas = () => {
    const selectedIds = Array.from(selectedAgendaIds);
    if (selectedIds.length < 2) return;

    const mergedPoints: any[] = [];
    const mergedMomIds = new Set<string>();

    localAgendas.forEach(ag => {
      if (selectedIds.includes(ag.agenda_id)) {
        mergedPoints.push(...ag.discussion_points);
        (agendaMapping[ag.agenda_id] || []).forEach(id => mergedMomIds.add(id));
      }
    });

    const newAgendaId = `merged_${Date.now()}`;
    const targetTitle = mergeTargetTitle.trim() ||
      selectedIds.map(id => localAgendas.find(a => a.agenda_id === id)?.agenda_title || 'Agenda').join(' + ');

    const newMergedAgenda: LocalLongRomAgenda = {
      agenda_id: newAgendaId,
      agenda_title: targetTitle,
      discussion_points: mergedPoints,
    };

    // Replace the first selected agenda with newMergedAgenda and remove other selected agendas
    const firstIdx = localAgendas.findIndex(ag => selectedIds.includes(ag.agenda_id));
    const newAgendas = localAgendas.filter(ag => !selectedIds.includes(ag.agenda_id));
    newAgendas.splice(firstIdx >= 0 ? firstIdx : 0, 0, newMergedAgenda);

    setLocalAgendas(newAgendas);

    // Update agendaMapping
    setAgendaMapping(prev => {
      const next = { ...prev };
      selectedIds.forEach(id => delete next[id]);
      next[newAgendaId] = Array.from(mergedMomIds);
      return next;
    });

    setSelectedAgendaIds(new Set());
    setShowMergeModal(false);
    setMergeTargetTitle('');
  };

  // 3. Merge All Agendas
  const handleMergeAllAgendas = () => {
    if (localAgendas.length <= 1) return;
    const confirmText = 'Merge all agendas into a single consolidated group? This change is local to Data Creation only and never alters the database meeting points.';
    if (!window.confirm(confirmText)) return;

    const allPoints: any[] = [];
    const allMomIds = new Set<string>();

    localAgendas.forEach(ag => {
      allPoints.push(...ag.discussion_points);
      (agendaMapping[ag.agenda_id] || []).forEach(id => allMomIds.add(id));
    });

    const masterId = `all_merged_${Date.now()}`;
    const masterTitle = selectedMeeting?.title
      ? `${selectedMeeting.title} (All Topics Merged)`
      : 'Full Meeting Discussion (Merged All Agendas)';

    const masterAgenda: LocalLongRomAgenda = {
      agenda_id: masterId,
      agenda_title: masterTitle,
      discussion_points: allPoints,
    };

    setLocalAgendas([masterAgenda]);
    setAgendaMapping({ [masterId]: Array.from(allMomIds) });
    setSelectedAgendaIds(new Set());
  };

  // 4. Reset to Original DB Meeting Points
  const handleResetAgendas = () => {
    if (originalAgendas.length === 0) return;
    if (!window.confirm('Reset all local moves and merges back to original meeting points from database?')) return;

    setLocalAgendas(JSON.parse(JSON.stringify(originalAgendas)));
    setSelectedAgendaIds(new Set());
    setMovingPointInfo(null);
  };

  // 5. Add Local Agenda
  const handleAddLocalAgenda = () => {
    const title = newAgendaTitleInput.trim();
    if (!title) return;

    const newAgenda: LocalLongRomAgenda = {
      agenda_id: `custom_agenda_${Date.now()}`,
      agenda_title: title,
      discussion_points: [],
    };

    setLocalAgendas(prev => [...prev, newAgenda]);
    setNewAgendaTitleInput('');
    setShowAddAgendaModal(false);
  };

  // 6. Inline Edit Agenda Title
  const handleSaveAgendaTitle = (agendaId: string) => {
    if (!agendaTitleEditText.trim()) return;
    setLocalAgendas(prev => prev.map(ag => ag.agenda_id === agendaId ? { ...ag, agenda_title: agendaTitleEditText.trim() } : ag));
    setEditingAgendaTitleId(null);
    setAgendaTitleEditText('');
  };

  // 7. Delete Local Agenda
  const handleDeleteLocalAgenda = (agendaId: string) => {
    if (!window.confirm('Remove this agenda group from local dataset?')) return;
    setLocalAgendas(prev => prev.filter(ag => ag.agenda_id !== agendaId));
    setAgendaMapping(prev => {
      const next = { ...prev };
      delete next[agendaId];
      return next;
    });
    setSelectedAgendaIds(prev => {
      const next = new Set(prev);
      next.delete(agendaId);
      return next;
    });
  };

  // 8. Add Point to Local Agenda
  const handleAddPointToAgenda = (agendaId: string) => {
    const text = newPointInputText.trim();
    if (!text) return;

    setLocalAgendas(prev => prev.map(ag => {
      if (ag.agenda_id === agendaId) {
        return {
          ...ag,
          discussion_points: [
            ...ag.discussion_points,
            { polished_text: text, text },
          ],
        };
      }
      return ag;
    }));

    setNewPointInputText('');
    setAddingPointAgendaId(null);
  };

  // 9. Edit Point in Local Agenda
  const handleSaveEditPoint = (agendaId: string, pointIdx: number, newText: string) => {
    if (!newText.trim()) return;
    setLocalAgendas(prev => prev.map(ag => {
      if (ag.agenda_id === agendaId) {
        const pts = [...ag.discussion_points];
        const oldPt = pts[pointIdx];
        if (typeof oldPt === 'object' && oldPt !== null) {
          pts[pointIdx] = { ...oldPt, polished_text: newText.trim(), text: newText.trim() };
        } else {
          pts[pointIdx] = newText.trim();
        }
        return { ...ag, discussion_points: pts };
      }
      return ag;
    }));
    setEditingPoint(null);
  };

  // 10. Delete Point from Local Agenda
  const handleDeletePoint = (agendaId: string, pointIdx: number) => {
    setLocalAgendas(prev => prev.map(ag => {
      if (ag.agenda_id === agendaId) {
        return {
          ...ag,
          discussion_points: ag.discussion_points.filter((_, idx) => idx !== pointIdx),
        };
      }
      return ag;
    }));
  };

  // ── Agenda mapping helpers ─────────────────────────────────────────────────
  const assignManualSection = (meetingAgendaId: string, momId: string) => {
    setAgendaMapping(prev => {
      const current = prev[meetingAgendaId] || [];
      if (current.includes(momId)) return prev;
      return { ...prev, [meetingAgendaId]: [...current, momId] };
    });
  };

  const removeManualSection = (meetingAgendaId: string, momId: string) => {
    setAgendaMapping(prev => {
      const current = prev[meetingAgendaId] || [];
      return { ...prev, [meetingAgendaId]: current.filter(id => id !== momId) };
    });
  };

  const setSingleManualSection = (meetingAgendaId: string, momId: string) => {
    setAgendaMapping(prev => {
      if (!momId || momId === '__unmapped__') {
        return { ...prev, [meetingAgendaId]: [] };
      }
      return { ...prev, [meetingAgendaId]: [momId] };
    });
  };

  // ── Derived statistics & helpers ───────────────────────────────────────────
  const totalLongPoints = useMemo(
    () => localAgendas.reduce((s, ag) => s + ag.discussion_points.length, 0),
    [localAgendas]
  );

  const totalManualPoints = useMemo(
    () => momAgendas.reduce((s, ag) => s + ag.points.length, 0),
    [momAgendas]
  );

  // Set of momAgenda IDs that are mapped to at least one meeting agenda
  const mappedMomIds = useMemo(() => {
    const s = new Set<string>();
    for (const ids of Object.values(agendaMapping)) {
      for (const id of ids) s.add(id);
    }
    return s;
  }, [agendaMapping]);

  // Unmapped manual sections
  const unmappedMomSections = useMemo(
    () => momAgendas.filter(ag => !mappedMomIds.has(ag.id)),
    [momAgendas, mappedMomIds]
  );

  // Number of meeting agendas with at least 1 manual section mapped
  const mappedAgendasCount = useMemo(() => {
    return localAgendas.filter(ag => (agendaMapping[ag.agenda_id] || []).length > 0).length;
  }, [localAgendas, agendaMapping]);

  const filteredMeetings = useMemo(() => {
    if (!meetingFilter.trim()) return meetings;
    const q = meetingFilter.toLowerCase();
    return meetings.filter(m => m.title.toLowerCase().includes(q));
  }, [meetings, meetingFilter]);

  const isLocallyModified = useMemo(() => {
    if (originalAgendas.length === 0) return false;
    return JSON.stringify(originalAgendas) !== JSON.stringify(localAgendas);
  }, [originalAgendas, localAgendas]);

  // Helper to extract text from a point item
  const getPointText = (pt: any): string => {
    if (!pt) return '';
    if (typeof pt === 'string') return pt;
    return pt.polished_text || pt.text || JSON.stringify(pt);
  };

  const getPointSpeaker = (pt: any): string | null => {
    if (!pt || typeof pt !== 'object') return null;
    return pt.speaker || pt.speakers?.[0] || null;
  };

  // ── Save action ("Create Training Dataset") ────────────────────────────────
  const handleCreateTrainingDataset = async () => {
    if (mappedAgendasCount === 0) return;

    setSaving(true);
    setSaveError('');
    setSaveSuccess(false);

    try {
      const pairs: any[] = [];
      for (const ag of localAgendas) {
        const momIds = agendaMapping[ag.agenda_id] || [];
        if (momIds.length === 0) continue;

        // Gather all manual points under mapped sections
        const manualPoints: string[] = [];
        for (const mId of momIds) {
          const mAg = momAgendas.find(m => m.id === mId);
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
        setSaveError('No valid agenda mappings found. Assign at least one Manual MOM section to an agenda.');
        setSaving(false);
        return;
      }

      const recId = selectedMeeting?.id || `manual_session_${Date.now()}`;
      const recTitle = selectedMeeting?.title || manualSessionTitle.trim() || 'Manual Training Dataset';

      await saveTrainingData({
        recording_id: recId,
        recording_title: recTitle,
        pairs,
      });

      setSaveSuccess(true);
      if (selectedMeeting) {
        const dataRes = await getDataForMeeting(selectedMeeting.id);
        setExistingData(dataRes.data || []);
      }
    } catch (e: any) {
      setSaveError(e?.response?.data?.detail || 'Failed to create training dataset in database');
    } finally {
      setSaving(false);
    }
  };

  const hasActiveWorkspace = (selectedMeeting && longRomAvailable) || isManualSession || localAgendas.length > 0;

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>

      {/* ════ TOP CONTROL BAR: Step 1 (Meeting / Manual Session) + Step 2 (Upload / Create Manual MOM) ════ */}
      <div style={{ display: 'flex', gap: '0.85rem', alignItems: 'stretch', flexWrap: 'wrap' }}>

        {/* Step 1: Meeting Selection or Manual Session */}
        <div style={{ flex: '1 1 340px', position: 'relative' }} ref={meetingPickerRef}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 5 }}>
            <span style={{ fontSize: '0.68rem', fontWeight: 700, color: C.mutedFg, letterSpacing: '0.05em', textTransform: 'uppercase' }}>
              Step 1 — Meeting / Session
            </span>
            {!selectedMeeting && !isManualSession && (
              <button
                onClick={handleStartManualSession}
                style={{
                  background: 'none', border: 'none', cursor: 'pointer',
                  fontSize: '0.68rem', color: C.accent, fontWeight: 600, padding: 0,
                  textDecoration: 'underline',
                }}
              >
                Or start manual session
              </button>
            )}
            {isManualSession && (
              <span style={{ fontSize: '0.64rem', color: C.purple, fontWeight: 700, background: C.purpleSoft, padding: '1px 6px', borderRadius: 4 }}>
                Manual Session
              </span>
            )}
          </div>

          {isManualSession ? (
            <div style={{
              display: 'flex', alignItems: 'center', gap: 8,
              padding: '0.5rem 0.85rem', borderRadius: 10,
              border: `1.5px solid ${C.purpleBorder}`, background: C.purpleSoft,
            }}>
              <FileText size={14} color={C.purple} />
              <input
                type="text"
                value={manualSessionTitle}
                onChange={e => setManualSessionTitle(e.target.value)}
                placeholder="Dataset session title…"
                style={{
                  flex: 1, border: 'none', background: 'transparent',
                  fontSize: '0.82rem', fontWeight: 600, color: C.fg,
                  outline: 'none', fontFamily: 'Inter, sans-serif',
                }}
              />
              <button
                onClick={() => {
                  setIsManualSession(false);
                  setLocalAgendas([]);
                }}
                style={{ background: 'none', border: 'none', cursor: 'pointer', padding: 2, color: C.mutedFg }}
                title="Switch back to meeting selection"
              >
                <X size={13} />
              </button>
            </div>
          ) : (
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
              {selectedMeeting && !loadingRom && longRomAvailable && (
                <span style={{ fontSize: '0.62rem', color: C.mutedFg, flexShrink: 0 }}>
                  {localAgendas.length} agendas · {totalLongPoints} pts
                </span>
              )}
              <ChevronDown size={13} color={C.mutedFg} style={{ flexShrink: 0 }} />
            </button>
          )}

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
              <div style={{ borderTop: `1px solid ${C.border}`, padding: '6px 8px', background: 'hsl(var(--muted) / .2)' }}>
                <button
                  onClick={() => {
                    handleStartManualSession();
                    setShowMeetingPicker(false);
                  }}
                  style={{
                    width: '100%', padding: '0.4rem', border: 'none', borderRadius: 6,
                    background: 'transparent', cursor: 'pointer', textAlign: 'center',
                    fontSize: '0.74rem', color: C.accent, fontWeight: 600,
                  }}
                >
                  + Start Direct Manual Session without Meeting
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Step 2: Manual MOM (Document Upload OR Direct Creation) */}
        <div style={{ flex: '2 1 480px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 5 }}>
            <span style={{ fontSize: '0.68rem', fontWeight: 700, color: C.mutedFg, letterSpacing: '0.05em', textTransform: 'uppercase' }}>
              Step 2 — Manual MOM (Upload or Create Directly)
            </span>
            {momAgendas.length > 0 && (
              <span style={{ fontSize: '0.66rem', color: C.purple, fontWeight: 700 }}>
                {momAgendas.length} section{momAgendas.length !== 1 ? 's' : ''} · {totalManualPoints} points
              </span>
            )}
          </div>

          <div style={{
            display: 'flex', alignItems: 'center', gap: 8,
            padding: '0.45rem 0.85rem', borderRadius: 10,
            border: `1.5px solid ${momAgendas.length > 0 ? C.purpleBorder : C.border}`,
            background: momAgendas.length > 0 ? C.purpleSoft : C.bg,
            flexWrap: 'wrap',
          }}>
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf,.docx,.doc,.txt,.md"
              style={{ display: 'none' }}
              onChange={e => e.target.files?.[0] && handleFileUpload(e.target.files[0])}
            />

            {/* Document Upload Button */}
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={uploading}
              style={{
                display: 'inline-flex', alignItems: 'center', gap: 5,
                padding: '0.36rem 0.75rem', borderRadius: 7, border: 'none',
                cursor: uploading ? 'not-allowed' : 'pointer',
                background: C.accent, color: 'white',
                fontSize: '0.74rem', fontWeight: 600, fontFamily: 'Inter, sans-serif',
                flexShrink: 0,
              }}
            >
              {uploading ? <Loader size={12} className="spin" /> : <Upload size={12} />}
              {uploading ? 'Extracting…' : 'Upload File'}
            </button>

            {/* Create Agenda Directly Button */}
            <button
              onClick={() => setShowCreateManualModal(true)}
              style={{
                display: 'inline-flex', alignItems: 'center', gap: 5,
                padding: '0.36rem 0.75rem', borderRadius: 7,
                border: `1px solid ${C.purpleBorder}`,
                background: C.card, color: C.purple,
                fontSize: '0.74rem', fontWeight: 600, fontFamily: 'Inter, sans-serif',
                cursor: 'pointer', flexShrink: 0,
              }}
            >
              <Plus size={12} /> Create Agenda
            </button>

            <div style={{ flex: 1, minWidth: 160 }}>
              {(uploading || matching) && (
                <div style={{ fontSize: '0.72rem', color: C.mutedFg, display: 'flex', alignItems: 'center', gap: 5 }}>
                  <Loader size={11} className="spin" />
                  {uploading ? 'Extracting text…' : 'Auto-matching…'}
                </div>
              )}
              {!uploading && !matching && momAgendas.length > 0 && (
                <div style={{ fontSize: '0.74rem', color: C.fg }}>
                  {uploadedFileName ? (
                    <span style={{ fontWeight: 600, color: C.purple }}>
                      <Check size={11} style={{ verticalAlign: 'middle', marginRight: 3 }} />{uploadedFileName}
                    </span>
                  ) : (
                    <span style={{ fontWeight: 600, color: C.purple }}>
                      Directly Created Dataset
                    </span>
                  )}
                </div>
              )}
              {!uploading && !matching && !momAgendas.length && !uploadError && (
                <div style={{ fontSize: '0.72rem', color: C.mutedFg }}>
                  Upload document or click <strong>Create Agenda</strong> to type/paste points
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
                title="Upload another file"
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
            {existingData.length} agenda{existingData.length !== 1 ? 's' : ''} saved in DB
          </div>
        )}
      </div>

      {/* ── Warning if selected meeting has no Long ROM ────────────────────── */}
      {selectedMeeting && !loadingRom && longRomAvailable === false && !isManualSession && (
        <div style={{
          display: 'flex', alignItems: 'center', gap: 10, padding: '0.85rem 1.1rem',
          borderRadius: 10, border: `1px solid ${C.amberBorder}`, background: C.amberSoft,
        }}>
          <AlertCircle size={16} color={C.amber} style={{ flexShrink: 0 }} />
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: '0.82rem', fontWeight: 600, color: C.amber }}>Long ROM not generated for this meeting</div>
            <div style={{ fontSize: '0.72rem', color: C.mutedFg }}>
              You can generate Long ROM first, or click <strong>+ Add Agenda</strong> to build the dataset locally.
            </div>
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            <button
              onClick={() => {
                setIsManualSession(true);
                setLongRomAvailable(true);
              }}
              style={{
                display: 'inline-flex', alignItems: 'center', gap: 5, padding: '0.38rem 0.8rem',
                borderRadius: 7, border: `1px solid ${C.amberBorder}`, cursor: 'pointer',
                background: C.card, color: C.amber, fontSize: '0.74rem', fontWeight: 600, fontFamily: 'Inter, sans-serif',
              }}
            >
              + Create Agendas Locally
            </button>
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
        </div>
      )}

      {/* ── Local modifications banner & Local Dataset Guarantees ─────────── */}
      {hasActiveWorkspace && (
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          padding: '0.45rem 0.85rem', borderRadius: 8,
          background: 'hsl(var(--muted) / .25)', border: `1px solid ${C.border}`,
          fontSize: '0.72rem', color: C.mutedFg, flexWrap: 'wrap', gap: 8,
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <Info size={13} color={C.accent} />
            <span>
              <strong>Local Dataset Controls Active:</strong> All point moves, merges, and additions are local to this Data Creation workspace.
              <strong style={{ color: C.fg }}> The original meeting points in the database are never modified.</strong>
            </span>
          </div>
          {isLocallyModified && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ color: C.amber, fontWeight: 700 }}>● Locally Modified</span>
              <button
                onClick={handleResetAgendas}
                style={{
                  display: 'inline-flex', alignItems: 'center', gap: 4,
                  padding: '2px 8px', borderRadius: 6,
                  border: `1px solid ${C.border}`, background: C.card,
                  fontSize: '0.68rem', fontWeight: 600, color: C.fg, cursor: 'pointer',
                }}
              >
                <RotateCcw size={10} /> Reset to DB Original
              </button>
            </div>
          )}
        </div>
      )}

      {/* ════ MAIN WORKSPACE: Side-by-Side with Local Grouping Controls ═════ */}
      {hasActiveWorkspace && (
        <>
          {/* Section column title headers & Action Toolbars */}
          <div style={{
            display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.85rem',
            padding: '0 0.2rem',
          }}>
            {/* Left Header: Generated Long Points & Local Controls */}
            <div style={{
              display: 'flex', flexDirection: 'column', gap: 6,
              padding: '0.6rem 0.8rem', borderRadius: 8,
              background: C.accentSoft, border: `1px solid ${C.accentBorder}`,
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
                <BookOpen size={14} color={C.accent} />
                <span style={{ fontSize: '0.82rem', fontWeight: 700, color: C.accent, flex: 1 }}>
                  Meeting Long Points
                </span>
                <span style={{ fontSize: '0.67rem', color: C.mutedFg }}>
                  {localAgendas.length} agendas · {totalLongPoints} total points
                </span>
              </div>

              {/* Local Controls: Merge Selected, Merge All, Add Agenda */}
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap', paddingTop: 2 }}>
                {/* Merge Selected Button */}
                {selectedAgendaIds.size >= 2 ? (
                  <button
                    onClick={() => {
                      const titles = Array.from(selectedAgendaIds)
                        .map(id => localAgendas.find(a => a.agenda_id === id)?.agenda_title || 'Agenda')
                        .join(' + ');
                      setMergeTargetTitle(titles);
                      setShowMergeModal(true);
                    }}
                    style={{
                      display: 'inline-flex', alignItems: 'center', gap: 4,
                      padding: '3px 8px', borderRadius: 6, border: 'none',
                      background: C.accent, color: 'white',
                      fontSize: '0.7rem', fontWeight: 700, cursor: 'pointer',
                    }}
                  >
                    <GitMerge size={11} /> Merge Selected ({selectedAgendaIds.size})
                  </button>
                ) : null}

                {/* Merge All Button */}
                {localAgendas.length > 1 && (
                  <button
                    onClick={handleMergeAllAgendas}
                    style={{
                      display: 'inline-flex', alignItems: 'center', gap: 4,
                      padding: '3px 8px', borderRadius: 6,
                      border: `1px solid ${C.accentBorder}`, background: C.card,
                      color: C.accent, fontSize: '0.7rem', fontWeight: 600, cursor: 'pointer',
                    }}
                    title="Merge all agendas into a single consolidated group"
                  >
                    <Layers size={11} /> Merge All ({localAgendas.length})
                  </button>
                )}

                {/* Add Agenda Button */}
                <button
                  onClick={() => setShowAddAgendaModal(true)}
                  style={{
                    display: 'inline-flex', alignItems: 'center', gap: 4,
                    padding: '3px 8px', borderRadius: 6,
                    border: `1px solid ${C.border}`, background: C.card,
                    color: C.fg, fontSize: '0.7rem', fontWeight: 600, cursor: 'pointer',
                  }}
                >
                  <Plus size={11} /> Add Agenda
                </button>

                {selectedAgendaIds.size > 0 && (
                  <button
                    onClick={() => setSelectedAgendaIds(new Set())}
                    style={{
                      background: 'none', border: 'none', cursor: 'pointer',
                      fontSize: '0.67rem', color: C.mutedFg, padding: '2px 4px',
                    }}
                  >
                    Clear selection
                  </button>
                )}
              </div>
            </div>

            {/* Right Header: Manual MOM & Direct Creation */}
            <div style={{
              display: 'flex', flexDirection: 'column', gap: 6,
              padding: '0.6rem 0.8rem', borderRadius: 8,
              background: C.purpleSoft, border: `1px solid ${C.purpleBorder}`,
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
                <Layers size={14} color={C.purple} />
                <span style={{ fontSize: '0.82rem', fontWeight: 700, color: C.purple, flex: 1 }}>
                  Manual MOM Sections
                </span>
                <span style={{ fontSize: '0.67rem', color: C.mutedFg }}>
                  {momAgendas.length} sections · {totalManualPoints} points
                </span>
              </div>

              {/* Direct Manual MOM Quick Controls */}
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap', paddingTop: 2 }}>
                <button
                  onClick={() => setShowCreateManualModal(true)}
                  style={{
                    display: 'inline-flex', alignItems: 'center', gap: 4,
                    padding: '3px 8px', borderRadius: 6,
                    border: 'none', background: C.purple,
                    color: 'white', fontSize: '0.7rem', fontWeight: 700, cursor: 'pointer',
                  }}
                >
                  <Plus size={11} /> New Manual Agenda
                </button>

                <button
                  onClick={() => fileInputRef.current?.click()}
                  style={{
                    display: 'inline-flex', alignItems: 'center', gap: 4,
                    padding: '3px 8px', borderRadius: 6,
                    border: `1px solid ${C.purpleBorder}`, background: C.card,
                    color: C.purple, fontSize: '0.7rem', fontWeight: 600, cursor: 'pointer',
                  }}
                >
                  <Upload size={11} /> Upload Document
                </button>
              </div>
            </div>
          </div>

          {/* ── Empty Agendas Placeholder if none exist ─────────────────────── */}
          {localAgendas.length === 0 && (
            <div style={{
              display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
              padding: '3rem 1rem', border: `1px dashed ${C.border}`, borderRadius: 12, background: C.card,
              gap: 12, textAlign: 'center', color: C.mutedFg,
            }}>
              <BookOpen size={32} style={{ opacity: 0.3 }} />
              <div>
                <div style={{ fontSize: '0.88rem', fontWeight: 700, color: C.fg, marginBottom: 4 }}>
                  No Agendas in Workspace Yet
                </div>
                <div style={{ fontSize: '0.76rem', maxWidth: 400 }}>
                  Select a meeting with Long ROM points, or click <strong>+ Add Agenda</strong> to create custom agendas locally.
                </div>
              </div>
              <button
                onClick={() => setShowAddAgendaModal(true)}
                style={{
                  display: 'inline-flex', alignItems: 'center', gap: 6,
                  padding: '0.45rem 1rem', borderRadius: 8, border: 'none',
                  background: C.accent, color: 'white',
                  fontSize: '0.78rem', fontWeight: 700, cursor: 'pointer',
                }}
              >
                <Plus size={13} /> Add First Agenda Group
              </button>
            </div>
          )}

          {/* ── Agenda-by-Agenda Rows ────────────────────────────────────────── */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
            {localAgendas.map((ag, agIdx) => {
              const mappedIds = agendaMapping[ag.agenda_id] || [];
              const isMapped = mappedIds.length > 0;
              const isCollapsed = collapsedAgendas.has(ag.agenda_id);
              const isSelectedForMerge = selectedAgendaIds.has(ag.agenda_id);
              const matchInfo = autoMatches.find(m => m.meeting_agenda_id === ag.agenda_id);

              return (
                <div
                  key={ag.agenda_id}
                  style={{
                    border: `1.5px solid ${isSelectedForMerge ? C.accent : isMapped ? C.greenBorder : C.border}`,
                    borderRadius: 12, overflow: 'hidden', background: C.card,
                    boxShadow: isSelectedForMerge ? `0 0 10px ${C.accentSoft}` : '0 1px 3px rgba(0,0,0,0.04)',
                    transition: 'all 0.15s',
                  }}
                >
                  {/* ── Unified Agenda Header ───────────────────────────────── */}
                  <div style={{
                    padding: '0.55rem 0.85rem',
                    background: isSelectedForMerge ? C.accentSoft : isMapped ? C.greenSoft : 'hsl(var(--muted) / .4)',
                    borderBottom: `1px solid ${isSelectedForMerge ? C.accentBorder : isMapped ? C.greenBorder : C.border}`,
                    display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap',
                  }}>
                    {/* Checkbox for Merge */}
                    <button
                      onClick={() => {
                        setSelectedAgendaIds(prev => {
                          const next = new Set(prev);
                          next.has(ag.agenda_id) ? next.delete(ag.agenda_id) : next.add(ag.agenda_id);
                          return next;
                        });
                      }}
                      style={{ background: 'none', border: 'none', cursor: 'pointer', padding: 0, color: isSelectedForMerge ? C.accent : C.mutedFg, display: 'flex', alignItems: 'center' }}
                      title={isSelectedForMerge ? 'Deselect agenda' : 'Select agenda to merge'}
                    >
                      {isSelectedForMerge ? <CheckSquare size={16} /> : <Square size={16} />}
                    </button>

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

                    {/* Agenda index badge */}
                    <span style={{
                      fontSize: '0.65rem', fontWeight: 800, padding: '2px 6px',
                      borderRadius: 5, background: C.bg, border: `1px solid ${C.border}`,
                      color: C.mutedFg, letterSpacing: '0.04em',
                    }}>
                      AGENDA {agIdx + 1}
                    </span>

                    {/* Title with inline edit */}
                    <div style={{ flex: 1, minWidth: 200, display: 'flex', alignItems: 'center', gap: 6 }}>
                      {editingAgendaTitleId === ag.agenda_id ? (
                        <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                          <input
                            autoFocus
                            type="text"
                            value={agendaTitleEditText}
                            onChange={e => setAgendaTitleEditText(e.target.value)}
                            onKeyDown={e => {
                              if (e.key === 'Enter') handleSaveAgendaTitle(ag.agenda_id);
                              if (e.key === 'Escape') setEditingAgendaTitleId(null);
                            }}
                            style={{
                              fontSize: '0.82rem', padding: '2px 6px', borderRadius: 5,
                              border: `1px solid ${C.accent}`, background: C.bg, color: C.fg,
                              fontFamily: 'Inter, sans-serif',
                            }}
                          />
                          <button
                            onClick={() => handleSaveAgendaTitle(ag.agenda_id)}
                            style={{ padding: '2px 6px', borderRadius: 4, background: C.accent, color: 'white', border: 'none', cursor: 'pointer', fontSize: '0.7rem' }}
                          >
                            Save
                          </button>
                          <button
                            onClick={() => setEditingAgendaTitleId(null)}
                            style={{ padding: '2px 6px', borderRadius: 4, background: C.muted, color: C.fg, border: 'none', cursor: 'pointer', fontSize: '0.7rem' }}
                          >
                            Cancel
                          </button>
                        </div>
                      ) : (
                        <>
                          <span style={{ fontSize: '0.84rem', fontWeight: 700, color: C.fg }}>
                            {ag.agenda_title}
                          </span>
                          <button
                            onClick={() => {
                              setEditingAgendaTitleId(ag.agenda_id);
                              setAgendaTitleEditText(ag.agenda_title);
                            }}
                            style={{ background: 'none', border: 'none', cursor: 'pointer', padding: 2, color: C.mutedFg }}
                            title="Rename agenda"
                          >
                            <Edit2 size={11} />
                          </button>
                          <span style={{ fontSize: '0.68rem', color: C.mutedFg }}>
                            ({ag.discussion_points.length} point{ag.discussion_points.length !== 1 ? 's' : ''})
                          </span>
                        </>
                      )}
                    </div>

                    {/* Mapping control dropdown */}
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                      {momAgendas.length > 0 && (
                        <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                          <span style={{ fontSize: '0.68rem', fontWeight: 600, color: C.mutedFg }}>
                            Manual MOM:
                          </span>
                          <select
                            value={mappedIds.length === 1 ? mappedIds[0] : mappedIds.length > 1 ? '__multiple__' : ''}
                            onChange={e => setSingleManualSection(ag.agenda_id, e.target.value)}
                            style={{
                              fontSize: '0.74rem', padding: '0.24rem 0.55rem', borderRadius: 7,
                              border: `1.5px solid ${isMapped ? C.green : C.border}`,
                              background: C.bg, color: C.fg, fontFamily: 'Inter, sans-serif',
                              cursor: 'pointer', outline: 'none', maxWidth: 220,
                            }}
                          >
                            <option value="">— Unmapped (None) —</option>
                            {momAgendas.map(mAg => (
                              <option key={mAg.id} value={mAg.id}>
                                {mAg.agenda_title} ({mAg.points.length} pts)
                              </option>
                            ))}
                          </select>
                        </div>
                      )}

                      {/* Status badge */}
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

                      {/* Delete local agenda */}
                      <button
                        onClick={() => handleDeleteLocalAgenda(ag.agenda_id)}
                        style={{ background: 'none', border: 'none', cursor: 'pointer', padding: 2, color: C.mutedFg }}
                        title="Delete agenda from local dataset"
                      >
                        <Trash2 size={12} />
                      </button>
                    </div>
                  </div>

                  {/* ── Side-by-Side Body ───────────────────────────────────── */}
                  {!isCollapsed && (
                    <div style={{
                      display: 'grid', gridTemplateColumns: '1fr 1fr',
                      borderTop: 'none',
                    }}>
                      {/* Left: Meeting Long Points with Move/Edit/Delete controls */}
                      <div style={{
                        padding: '0.75rem 1rem',
                        borderRight: `1px solid ${C.border}`,
                        display: 'flex', flexDirection: 'column', gap: 7,
                        background: 'hsl(var(--muted) / .08)',
                      }}>
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                          <span style={{ fontSize: '0.68rem', fontWeight: 700, color: C.accent, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                            Discussion Points ({ag.discussion_points.length})
                          </span>
                          <button
                            onClick={() => {
                              setAddingPointAgendaId(ag.agenda_id);
                              setNewPointInputText('');
                            }}
                            style={{
                              display: 'inline-flex', alignItems: 'center', gap: 3,
                              fontSize: '0.67rem', color: C.accent, background: 'none',
                              border: 'none', cursor: 'pointer', fontWeight: 600,
                            }}
                          >
                            <Plus size={11} /> Add Point
                          </button>
                        </div>

                        {/* Inline Add Point Input */}
                        {addingPointAgendaId === ag.agenda_id && (
                          <div style={{
                            display: 'flex', flexDirection: 'column', gap: 5,
                            padding: '0.5rem', borderRadius: 7, background: C.bg,
                            border: `1px solid ${C.accentBorder}`,
                          }}>
                            <textarea
                              autoFocus
                              rows={2}
                              value={newPointInputText}
                              onChange={e => setNewPointInputText(e.target.value)}
                              placeholder="Type new discussion point here…"
                              style={{
                                width: '100%', fontSize: '0.76rem', padding: '0.3rem',
                                border: `1px solid ${C.border}`, borderRadius: 5,
                                background: C.card, color: C.fg, resize: 'vertical',
                                fontFamily: 'Inter, sans-serif', outline: 'none',
                              }}
                            />
                            <div style={{ display: 'flex', gap: 6, justifyContent: 'flex-end' }}>
                              <button
                                onClick={() => handleAddPointToAgenda(ag.agenda_id)}
                                style={{
                                  padding: '2px 8px', borderRadius: 5, background: C.accent,
                                  color: 'white', border: 'none', cursor: 'pointer', fontSize: '0.7rem', fontWeight: 600,
                                }}
                              >
                                Add Point
                              </button>
                              <button
                                onClick={() => setAddingPointAgendaId(null)}
                                style={{
                                  padding: '2px 8px', borderRadius: 5, background: C.muted,
                                  color: C.fg, border: 'none', cursor: 'pointer', fontSize: '0.7rem',
                                }}
                              >
                                Cancel
                              </button>
                            </div>
                          </div>
                        )}

                        {ag.discussion_points.length === 0 ? (
                          <div style={{ fontSize: '0.75rem', color: C.mutedFg, fontStyle: 'italic', padding: '0.5rem 0' }}>
                            No discussion points in this agenda. Click "+ Add Point" or move points here.
                          </div>
                        ) : (
                          ag.discussion_points.map((pt, ptIdx) => {
                            const ptText = getPointText(pt);
                            const speaker = getPointSpeaker(pt);
                            const isMoving = movingPointInfo?.agendaId === ag.agenda_id && movingPointInfo?.pointIdx === ptIdx;
                            const isEditing = editingPoint?.agendaId === ag.agenda_id && editingPoint?.pointIdx === ptIdx;

                            return (
                              <div
                                key={ptIdx}
                                style={{
                                  display: 'flex', flexDirection: 'column', gap: 4,
                                  fontSize: '0.76rem', color: C.fg, lineHeight: 1.5,
                                  padding: '0.45rem 0.6rem', borderRadius: 7,
                                  background: 'hsl(var(--card))',
                                  border: `1px solid ${isMoving ? C.accent : 'hsl(var(--border) / .7)'}`,
                                }}
                              >
                                {isEditing ? (
                                  <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                                    <textarea
                                      rows={2}
                                      value={editingPoint.text}
                                      onChange={e => setEditingPoint({ ...editingPoint, text: e.target.value })}
                                      style={{
                                        width: '100%', fontSize: '0.76rem', padding: '0.3rem',
                                        border: `1px solid ${C.accent}`, borderRadius: 5,
                                        background: C.bg, color: C.fg, resize: 'vertical',
                                        fontFamily: 'Inter, sans-serif',
                                      }}
                                    />
                                    <div style={{ display: 'flex', gap: 5, justifyContent: 'flex-end' }}>
                                      <button
                                        onClick={() => handleSaveEditPoint(ag.agenda_id, ptIdx, editingPoint.text)}
                                        style={{ padding: '2px 7px', borderRadius: 4, background: C.accent, color: 'white', border: 'none', cursor: 'pointer', fontSize: '0.68rem', fontWeight: 600 }}
                                      >
                                        Save
                                      </button>
                                      <button
                                        onClick={() => setEditingPoint(null)}
                                        style={{ padding: '2px 7px', borderRadius: 4, background: C.muted, color: C.fg, border: 'none', cursor: 'pointer', fontSize: '0.68rem' }}
                                      >
                                        Cancel
                                      </button>
                                    </div>
                                  </div>
                                ) : (
                                  <div style={{ display: 'flex', alignItems: 'flex-start', gap: 7 }}>
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

                                    {/* Action Icons: Move, Edit, Delete */}
                                    <div style={{ display: 'flex', alignItems: 'center', gap: 3, flexShrink: 0, opacity: 0.85 }}>
                                      {/* Move Point Trigger */}
                                      <button
                                        onClick={() => setMovingPointInfo(isMoving ? null : { agendaId: ag.agenda_id, pointIdx: ptIdx })}
                                        style={{
                                          background: isMoving ? C.accentSoft : 'none',
                                          border: isMoving ? `1px solid ${C.accentBorder}` : 'none',
                                          borderRadius: 4, cursor: 'pointer', padding: '2px 4px',
                                          color: isMoving ? C.accent : C.mutedFg, display: 'flex', alignItems: 'center', gap: 2,
                                        }}
                                        title="Move point to another agenda"
                                      >
                                        <ArrowRightLeft size={11} />
                                        <span style={{ fontSize: '0.62rem', fontWeight: 600 }}>Move</span>
                                      </button>

                                      <button
                                        onClick={() => setEditingPoint({ agendaId: ag.agenda_id, pointIdx: ptIdx, text: ptText })}
                                        style={{ background: 'none', border: 'none', cursor: 'pointer', padding: 2, color: C.mutedFg }}
                                        title="Edit point locally"
                                      >
                                        <Edit2 size={11} />
                                      </button>

                                      <button
                                        onClick={() => handleDeletePoint(ag.agenda_id, ptIdx)}
                                        style={{ background: 'none', border: 'none', cursor: 'pointer', padding: 2, color: C.mutedFg }}
                                        title="Delete point from local dataset"
                                      >
                                        <Trash2 size={11} />
                                      </button>
                                    </div>
                                  </div>
                                )}

                                {/* Destination Selector for Move Point */}
                                {isMoving && (
                                  <div style={{
                                    display: 'flex', alignItems: 'center', gap: 6,
                                    marginTop: 4, padding: '4px 6px', borderRadius: 6,
                                    background: C.accentSoft, border: `1px solid ${C.accentBorder}`,
                                  }}>
                                    <MoveRight size={12} color={C.accent} />
                                    <span style={{ fontSize: '0.68rem', fontWeight: 600, color: C.accent }}>
                                      Move to:
                                    </span>
                                    <select
                                      autoFocus
                                      defaultValue=""
                                      onChange={e => {
                                        if (e.target.value) {
                                          handleMovePoint(ag.agenda_id, ptIdx, e.target.value);
                                        }
                                      }}
                                      style={{
                                        flex: 1, fontSize: '0.72rem', padding: '2px 5px',
                                        borderRadius: 5, border: `1px solid ${C.border}`,
                                        background: C.bg, color: C.fg, cursor: 'pointer', outline: 'none',
                                      }}
                                    >
                                      <option value="">— Select Target Agenda —</option>
                                      {localAgendas
                                        .filter(target => target.agenda_id !== ag.agenda_id)
                                        .map(target => (
                                          <option key={target.agenda_id} value={target.agenda_id}>
                                            {target.agenda_title}
                                          </option>
                                        ))
                                      }
                                    </select>
                                    <button
                                      onClick={() => setMovingPointInfo(null)}
                                      style={{ background: 'none', border: 'none', cursor: 'pointer', color: C.mutedFg, padding: 2 }}
                                    >
                                      <X size={11} />
                                    </button>
                                  </div>
                                )}
                              </div>
                            );
                          })
                        )}
                      </div>

                      {/* Right: Manual MOM Content */}
                      <div style={{
                        padding: '0.75rem 1rem',
                        display: 'flex', flexDirection: 'column', gap: 7,
                        background: isMapped ? C.purpleSoft : 'hsl(var(--muted) / .04)',
                      }}>
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                          <span style={{ fontSize: '0.68rem', fontWeight: 700, color: C.purple, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                            Manual MOM Content
                          </span>

                          {/* Multi-section selector */}
                          {momAgendas.length > 0 && (
                            <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                              <select
                                value=""
                                onChange={e => {
                                  if (e.target.value) {
                                    assignManualSection(ag.agenda_id, e.target.value);
                                  }
                                }}
                                style={{
                                  fontSize: '0.64rem', padding: '1px 5px', borderRadius: 5,
                                  border: `1px solid ${C.border}`, background: C.bg, color: C.fg,
                                  fontFamily: 'Inter, sans-serif', cursor: 'pointer', outline: 'none',
                                }}
                              >
                                <option value="">+ Add manual section</option>
                                {momAgendas.map(mAg => (
                                  <option key={mAg.id} value={mAg.id} disabled={mappedIds.includes(mAg.id)}>
                                    {mAg.agenda_title} ({mAg.points.length} pts)
                                  </option>
                                ))}
                              </select>
                            </div>
                          )}
                        </div>

                        {/* Mapped section tags */}
                        {mappedIds.length > 0 && (
                          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5, marginBottom: 2 }}>
                            {mappedIds.map(mId => {
                              const mAg = momAgendas.find(m => m.id === mId);
                              if (!mAg) return null;
                              return (
                                <span
                                  key={mId}
                                  style={{
                                    display: 'inline-flex', alignItems: 'center', gap: 4,
                                    fontSize: '0.67rem', fontWeight: 600, padding: '2px 7px',
                                    borderRadius: 6, background: C.purple, color: 'white',
                                  }}
                                >
                                  <Tag size={10} />
                                  {mAg.agenda_title}
                                  <button
                                    onClick={() => removeManualSection(ag.agenda_id, mId)}
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
                        {mappedIds.length === 0 ? (
                          <div style={{
                            display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
                            padding: '2rem 1rem', textAlign: 'center', color: C.mutedFg, gap: 8,
                          }}>
                            <Layers size={22} style={{ opacity: 0.3 }} />
                            <div style={{ fontSize: '0.76rem' }}>
                              {momAgendas.length > 0
                                ? 'No manual MOM section mapped to this agenda.'
                                : 'Upload a manual MOM file or click "Create Agenda" to add and map content.'}
                            </div>
                          </div>
                        ) : (
                          mappedIds.map(mId => {
                            const mAg = momAgendas.find(m => m.id === mId);
                            if (!mAg) return null;

                            return (
                              <div key={mId} style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                                {mappedIds.length > 1 && (
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

          {/* ════ DIRECT MANUAL MOM AGENDAS WORKSPACE (Cards for typing/pasting) ════ */}
          {momAgendas.length > 0 && (
            <div style={{
              display: 'flex', flexDirection: 'column', gap: 10,
              padding: '0.9rem', borderRadius: 12,
              background: 'hsl(var(--card))', border: `1px solid ${C.border}`,
            }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <Layers size={16} color={C.purple} />
                  <span style={{ fontSize: '0.86rem', fontWeight: 700, color: C.fg }}>
                    Manual MOM Agendas & Points ({momAgendas.length})
                  </span>
                  <span style={{ fontSize: '0.7rem', color: C.mutedFg }}>
                    Type or paste MOM points directly into each agenda below
                  </span>
                </div>
                <button
                  onClick={() => setShowCreateManualModal(true)}
                  style={{
                    display: 'inline-flex', alignItems: 'center', gap: 5,
                    padding: '0.35rem 0.75rem', borderRadius: 7, border: 'none',
                    background: C.purple, color: 'white',
                    fontSize: '0.74rem', fontWeight: 700, cursor: 'pointer',
                  }}
                >
                  <Plus size={12} /> Create Another Agenda
                </button>
              </div>

              {/* Grid of Manual MOM Agendas */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                {momAgendas.map(mAg => {
                  const isMappedSomewhere = mappedMomIds.has(mAg.id);

                  return (
                    <div
                      key={mAg.id}
                      style={{
                        borderRadius: 10, border: `1px solid ${C.purpleBorder}`,
                        background: 'hsl(var(--muted) / .12)', overflow: 'hidden',
                      }}
                    >
                      {/* Manual Agenda Header */}
                      <div style={{
                        padding: '0.55rem 0.85rem', background: C.purpleSoft,
                        borderBottom: `1px solid ${C.purpleBorder}`,
                        display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap',
                      }}>
                        {editingMomTitleId === mAg.id ? (
                          <div style={{ display: 'flex', alignItems: 'center', gap: 4, flex: 1 }}>
                            <input
                              autoFocus
                              type="text"
                              value={momTitleEditText}
                              onChange={e => setMomTitleEditText(e.target.value)}
                              onKeyDown={e => {
                                if (e.key === 'Enter') handleSaveMomTitle(mAg.id);
                                if (e.key === 'Escape') setEditingMomTitleId(null);
                              }}
                              style={{
                                fontSize: '0.8rem', padding: '2px 6px', borderRadius: 5,
                                border: `1px solid ${C.purple}`, background: C.bg, color: C.fg,
                              }}
                            />
                            <button
                              onClick={() => handleSaveMomTitle(mAg.id)}
                              style={{ padding: '2px 6px', borderRadius: 4, background: C.purple, color: 'white', border: 'none', cursor: 'pointer', fontSize: '0.68rem' }}
                            >
                              Save
                            </button>
                            <button
                              onClick={() => setEditingMomTitleId(null)}
                              style={{ padding: '2px 6px', borderRadius: 4, background: C.muted, color: C.fg, border: 'none', cursor: 'pointer', fontSize: '0.68rem' }}
                            >
                              Cancel
                            </button>
                          </div>
                        ) : (
                          <div style={{ display: 'flex', alignItems: 'center', gap: 6, flex: 1 }}>
                            <span style={{ fontSize: '0.84rem', fontWeight: 700, color: C.fg }}>
                              {mAg.agenda_title}
                            </span>
                            <button
                              onClick={() => {
                                setEditingMomTitleId(mAg.id);
                                setMomTitleEditText(mAg.agenda_title);
                              }}
                              style={{ background: 'none', border: 'none', cursor: 'pointer', padding: 2, color: C.mutedFg }}
                              title="Rename agenda"
                            >
                              <Edit2 size={11} />
                            </button>
                            <span style={{ fontSize: '0.68rem', color: C.mutedFg }}>
                              ({mAg.points.length} point{mAg.points.length !== 1 ? 's' : ''})
                            </span>
                          </div>
                        )}

                        {/* Map to Meeting Agenda Dropdown */}
                        {localAgendas.length > 0 && (
                          <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                            <span style={{ fontSize: '0.68rem', color: C.mutedFg }}>Map to:</span>
                            <select
                              value=""
                              onChange={e => {
                                if (e.target.value) assignManualSection(e.target.value, mAg.id);
                              }}
                              style={{
                                fontSize: '0.72rem', padding: '2px 6px', borderRadius: 6,
                                border: `1px solid ${C.border}`, background: C.bg, color: C.fg,
                                cursor: 'pointer', outline: 'none',
                              }}
                            >
                              <option value="">— Choose Meeting Agenda —</option>
                              {localAgendas.map(lag => (
                                <option key={lag.agenda_id} value={lag.agenda_id}>
                                  {lag.agenda_title}
                                </option>
                              ))}
                            </select>
                          </div>
                        )}

                        <span style={{
                          fontSize: '0.64rem', fontWeight: 600, padding: '1px 6px', borderRadius: 5,
                          background: isMappedSomewhere ? C.greenSoft : C.amberSoft,
                          color: isMappedSomewhere ? C.green : C.amber,
                        }}>
                          {isMappedSomewhere ? 'Mapped' : 'Unmapped'}
                        </span>

                        <button
                          onClick={() => handleDeleteManualAgenda(mAg.id)}
                          style={{ background: 'none', border: 'none', cursor: 'pointer', padding: 2, color: C.mutedFg }}
                          title="Delete manual agenda"
                        >
                          <Trash2 size={12} />
                        </button>
                      </div>

                      {/* Manual Agenda Body: Large Paste Box & Points List */}
                      <div style={{ padding: '0.75rem', display: 'flex', flexDirection: 'column', gap: 8 }}>
                        {/* Large Textbox for pasting/typing multiple points */}
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                            <span style={{ fontSize: '0.68rem', fontWeight: 700, color: C.mutedFg, letterSpacing: '0.04em', textTransform: 'uppercase' }}>
                              Paste / Type MOM Points:
                            </span>
                            <span style={{ fontSize: '0.65rem', color: C.mutedFg }}>
                              Supports bullets (•, -), numbered lists, or newlines
                            </span>
                          </div>
                          <textarea
                            rows={3}
                            value={pasteBoxText[mAg.id] || ''}
                            onChange={e => setPasteBoxText({ ...pasteBoxText, [mAg.id]: e.target.value })}
                            placeholder="Paste multiple bullet points, numbered notes, or paragraph text here…"
                            style={{
                              width: '100%', fontSize: '0.76rem', padding: '0.4rem 0.6rem',
                              border: `1px solid ${C.border}`, borderRadius: 7,
                              background: C.bg, color: C.fg, resize: 'vertical',
                              fontFamily: 'Inter, sans-serif', outline: 'none',
                            }}
                          />
                          <div style={{ display: 'flex', gap: 6, justifyContent: 'space-between', alignItems: 'center' }}>
                            {/* Single point quick add */}
                            <div style={{ display: 'flex', alignItems: 'center', gap: 5, flex: 1, maxWidth: 380 }}>
                              <input
                                type="text"
                                value={quickPointText[mAg.id] || ''}
                                onChange={e => setQuickPointText({ ...quickPointText, [mAg.id]: e.target.value })}
                                onKeyDown={e => {
                                  if (e.key === 'Enter') handleAddSingleManualPoint(mAg.id);
                                }}
                                placeholder="Or type a single point…"
                                style={{
                                  flex: 1, fontSize: '0.74rem', padding: '0.25rem 0.5rem',
                                  border: `1px solid ${C.border}`, borderRadius: 6,
                                  background: C.bg, color: C.fg, outline: 'none',
                                }}
                              />
                              <button
                                onClick={() => handleAddSingleManualPoint(mAg.id)}
                                disabled={!quickPointText[mAg.id]?.trim()}
                                style={{
                                  padding: '0.25rem 0.6rem', borderRadius: 6, border: 'none',
                                  background: C.purple, color: 'white', fontSize: '0.7rem',
                                  fontWeight: 600, cursor: 'pointer', opacity: quickPointText[mAg.id]?.trim() ? 1 : 0.5,
                                }}
                              >
                                + Add
                              </button>
                            </div>

                            <button
                              onClick={() => handleParseAndAddPoints(mAg.id)}
                              disabled={!pasteBoxText[mAg.id]?.trim()}
                              style={{
                                display: 'inline-flex', alignItems: 'center', gap: 4,
                                padding: '0.3rem 0.75rem', borderRadius: 6, border: 'none',
                                background: C.purple, color: 'white',
                                fontSize: '0.72rem', fontWeight: 700, cursor: 'pointer',
                                opacity: pasteBoxText[mAg.id]?.trim() ? 1 : 0.5,
                              }}
                            >
                              <ClipboardPaste size={12} /> Parse & Add Points
                            </button>
                          </div>
                        </div>

                        {/* List of current points in this manual agenda */}
                        {mAg.points.length > 0 && (
                          <div style={{ display: 'flex', flexDirection: 'column', gap: 5, marginTop: 4 }}>
                            <div style={{ fontSize: '0.66rem', fontWeight: 700, color: C.mutedFg, textTransform: 'uppercase' }}>
                              Current Points ({mAg.points.length}):
                            </div>
                            {mAg.points.map((pt, pIdx) => {
                              const isEditingPt = editingManualPoint?.momId === mAg.id && editingManualPoint?.pointIdx === pIdx;

                              return (
                                <div
                                  key={pIdx}
                                  style={{
                                    display: 'flex', alignItems: 'flex-start', gap: 6,
                                    fontSize: '0.75rem', padding: '0.35rem 0.55rem', borderRadius: 6,
                                    background: C.card, border: `1px solid ${C.purpleBorder}`,
                                  }}
                                >
                                  <span style={{ color: C.purple, fontWeight: 700, marginTop: 1 }}>•</span>
                                  {isEditingPt ? (
                                    <div style={{ flex: 1, display: 'flex', gap: 5, alignItems: 'center' }}>
                                      <input
                                        autoFocus
                                        type="text"
                                        value={editingManualPoint.text}
                                        onChange={e => setEditingManualPoint({ ...editingManualPoint, text: e.target.value })}
                                        onKeyDown={e => {
                                          if (e.key === 'Enter') handleSaveEditManualPoint(mAg.id, pIdx, editingManualPoint.text);
                                          if (e.key === 'Escape') setEditingManualPoint(null);
                                        }}
                                        style={{
                                          flex: 1, fontSize: '0.75rem', padding: '2px 5px',
                                          border: `1px solid ${C.purple}`, borderRadius: 4,
                                          background: C.bg, color: C.fg,
                                        }}
                                      />
                                      <button
                                        onClick={() => handleSaveEditManualPoint(mAg.id, pIdx, editingManualPoint.text)}
                                        style={{ padding: '2px 6px', borderRadius: 4, background: C.purple, color: 'white', border: 'none', cursor: 'pointer', fontSize: '0.67rem' }}
                                      >
                                        Save
                                      </button>
                                      <button
                                        onClick={() => setEditingManualPoint(null)}
                                        style={{ padding: '2px 6px', borderRadius: 4, background: C.muted, color: C.fg, border: 'none', cursor: 'pointer', fontSize: '0.67rem' }}
                                      >
                                        Cancel
                                      </button>
                                    </div>
                                  ) : (
                                    <>
                                      <span style={{ flex: 1, color: C.fg }}>{pt}</span>
                                      <button
                                        onClick={() => setEditingManualPoint({ momId: mAg.id, pointIdx: pIdx, text: pt })}
                                        style={{ background: 'none', border: 'none', cursor: 'pointer', padding: 2, color: C.mutedFg }}
                                        title="Edit point"
                                      >
                                        <Edit2 size={11} />
                                      </button>
                                      <button
                                        onClick={() => handleDeleteManualPoint(mAg.id, pIdx)}
                                        style={{ background: 'none', border: 'none', cursor: 'pointer', padding: 2, color: C.mutedFg }}
                                        title="Delete point"
                                      >
                                        <Trash2 size={11} />
                                      </button>
                                    </>
                                  )}
                                </div>
                              );
                            })}
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

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
                  Unmapped Manual MOM Sections ({unmappedMomSections.length})
                </span>
                <span style={{ fontSize: '0.68rem', color: C.mutedFg }}>
                  These sections are not yet mapped to any agenda group
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
                      key={sec.id}
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
                            if (e.target.value) assignManualSection(e.target.value, sec.id);
                          }}
                          style={{
                            fontSize: '0.72rem', padding: '0.25rem 0.5rem', borderRadius: 6,
                            border: `1px solid ${C.border}`, background: C.bg, color: C.fg,
                            fontFamily: 'Inter, sans-serif', cursor: 'pointer', outline: 'none',
                          }}
                        >
                          <option value="">— Select Agenda —</option>
                          {localAgendas.map(ag => (
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

          {/* ════ BOTTOM SAVE BAR ("Create Training Dataset") ══════════════════ */}
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
                <strong style={{ color: C.fg }}>{localAgendas.length}</strong> agendas mapped
              </span>
              {saveSuccess && (
                <span style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: '0.76rem', color: C.green, fontWeight: 600 }}>
                  <Check size={14} /> Training dataset created successfully! (DB meeting points remain untouched)
                </span>
              )}
              {saveError && (
                <span style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: '0.76rem', color: C.red }}>
                  <AlertCircle size={14} /> {saveError}
                </span>
              )}
            </div>

            {/* Create Training Dataset Button */}
            <button
              id="rom-training-save-btn"
              onClick={handleCreateTrainingDataset}
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
              {saving ? 'Creating Dataset…' : `Create Training Dataset (${mappedAgendasCount} Mapped)`}
            </button>
          </div>
        </>
      )}

      {/* ── Empty State: Neither meeting selected nor manual session active ─ */}
      {!hasActiveWorkspace && (
        <div style={{
          display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
          padding: '4.5rem 2rem', gap: 16, color: C.mutedFg,
          border: `1px dashed ${C.border}`, borderRadius: 12, background: C.card,
        }}>
          <BookOpen size={44} style={{ opacity: 0.2 }} />
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '0.96rem', fontWeight: 700, color: C.fg, marginBottom: 6 }}>
              Start ROM Training Data Creation
            </div>
            <div style={{ fontSize: '0.78rem', maxWidth: 460, lineHeight: 1.6 }}>
              Choose a meeting with generated Long ROM points to group and map with Manual MOM,
              or start directly with Manual MOM creation.
            </div>
          </div>

          <div style={{ display: 'flex', gap: 12 }}>
            <button
              onClick={() => setShowMeetingPicker(true)}
              style={{
                display: 'inline-flex', alignItems: 'center', gap: 6,
                padding: '0.5rem 1.1rem', borderRadius: 8, border: 'none',
                background: C.accent, color: 'white',
                fontSize: '0.78rem', fontWeight: 700, cursor: 'pointer',
              }}
            >
              <FileText size={13} /> Select Meeting
            </button>

            <button
              onClick={handleStartManualSession}
              style={{
                display: 'inline-flex', alignItems: 'center', gap: 6,
                padding: '0.5rem 1.1rem', borderRadius: 8,
                border: `1.5px solid ${C.purpleBorder}`, background: C.purpleSoft,
                color: C.purple, fontSize: '0.78rem', fontWeight: 700, cursor: 'pointer',
              }}
            >
              <Plus size={13} /> Manual MOM Direct Creation
            </button>
          </div>

          <div style={{ display: 'flex', gap: 24, marginTop: 10 }}>
            {[
              { n: '1', label: 'Select or Create Meeting Agendas' },
              { n: '2', label: 'Locally Move / Merge Points' },
              { n: '3', label: 'Upload or Type Manual MOM' },
              { n: '4', label: 'Create Training Dataset' },
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
                <div style={{ fontSize: '0.68rem', color: C.mutedFg, textAlign: 'center', maxWidth: 110 }}>{step.label}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── MODAL: Merge Selected Agendas ─────────────────────────────────── */}
      {showMergeModal && (
        <div style={{
          position: 'fixed', inset: 0, zIndex: 999,
          background: 'rgba(0,0,0,0.5)', display: 'flex', alignItems: 'center', justifyContent: 'center',
          backdropFilter: 'blur(2px)',
        }}>
          <div style={{
            width: '100%', maxWidth: 460, background: C.card,
            borderRadius: 12, border: `1px solid ${C.border}`,
            boxShadow: '0 20px 40px rgba(0,0,0,0.3)', padding: '1.25rem',
            display: 'flex', flexDirection: 'column', gap: 12,
          }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
                <GitMerge size={16} color={C.accent} />
                <span style={{ fontSize: '0.9rem', fontWeight: 700, color: C.fg }}>
                  Merge {selectedAgendaIds.size} Agendas
                </span>
              </div>
              <button
                onClick={() => setShowMergeModal(false)}
                style={{ background: 'none', border: 'none', cursor: 'pointer', color: C.mutedFg }}
              >
                <X size={15} />
              </button>
            </div>

            <div style={{ fontSize: '0.76rem', color: C.mutedFg, lineHeight: 1.5 }}>
              All discussion points and mapped manual sections from these {selectedAgendaIds.size} agendas will be combined into a single agenda group in your local dataset.
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              <label style={{ fontSize: '0.72rem', fontWeight: 700, color: C.fg }}>
                Merged Agenda Title:
              </label>
              <input
                autoFocus
                type="text"
                value={mergeTargetTitle}
                onChange={e => setMergeTargetTitle(e.target.value)}
                placeholder="Enter merged agenda title…"
                style={{
                  width: '100%', padding: '0.45rem 0.65rem', borderRadius: 7,
                  border: `1px solid ${C.accent}`, background: C.bg, color: C.fg,
                  fontSize: '0.8rem', fontFamily: 'Inter, sans-serif', outline: 'none',
                }}
              />
            </div>

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, marginTop: 4 }}>
              <button
                onClick={() => setShowMergeModal(false)}
                style={{
                  padding: '0.45rem 0.9rem', borderRadius: 7,
                  border: `1px solid ${C.border}`, background: C.bg,
                  color: C.fg, fontSize: '0.76rem', fontWeight: 600, cursor: 'pointer',
                }}
              >
                Cancel
              </button>
              <button
                onClick={handleMergeSelectedAgendas}
                style={{
                  padding: '0.45rem 1rem', borderRadius: 7,
                  border: 'none', background: C.accent,
                  color: 'white', fontSize: '0.76rem', fontWeight: 700, cursor: 'pointer',
                }}
              >
                Confirm Merge
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── MODAL: Create Manual Agenda Directly ──────────────────────────── */}
      {showCreateManualModal && (
        <div style={{
          position: 'fixed', inset: 0, zIndex: 999,
          background: 'rgba(0,0,0,0.5)', display: 'flex', alignItems: 'center', justifyContent: 'center',
          backdropFilter: 'blur(2px)',
        }}>
          <div style={{
            width: '100%', maxWidth: 440, background: C.card,
            borderRadius: 12, border: `1px solid ${C.purpleBorder}`,
            boxShadow: '0 20px 40px rgba(0,0,0,0.3)', padding: '1.25rem',
            display: 'flex', flexDirection: 'column', gap: 12,
          }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
                <Layers size={16} color={C.purple} />
                <span style={{ fontSize: '0.9rem', fontWeight: 700, color: C.fg }}>
                  Create Manual MOM Agenda
                </span>
              </div>
              <button
                onClick={() => setShowCreateManualModal(false)}
                style={{ background: 'none', border: 'none', cursor: 'pointer', color: C.mutedFg }}
              >
                <X size={15} />
              </button>
            </div>

            <div style={{ fontSize: '0.76rem', color: C.mutedFg, lineHeight: 1.5 }}>
              Enter an agenda topic name. You can immediately paste or type MOM points inside it.
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              <label style={{ fontSize: '0.72rem', fontWeight: 700, color: C.fg }}>
                Agenda Topic / Title:
              </label>
              <input
                autoFocus
                type="text"
                value={newManualAgendaTitle}
                onChange={e => setNewManualAgendaTitle(e.target.value)}
                onKeyDown={e => {
                  if (e.key === 'Enter') handleCreateManualMomAgenda();
                }}
                placeholder="e.g. Architecture Decisions, Action Items, Timeline…"
                style={{
                  width: '100%', padding: '0.45rem 0.65rem', borderRadius: 7,
                  border: `1px solid ${C.purple}`, background: C.bg, color: C.fg,
                  fontSize: '0.8rem', fontFamily: 'Inter, sans-serif', outline: 'none',
                }}
              />
            </div>

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, marginTop: 4 }}>
              <button
                onClick={() => setShowCreateManualModal(false)}
                style={{
                  padding: '0.45rem 0.9rem', borderRadius: 7,
                  border: `1px solid ${C.border}`, background: C.bg,
                  color: C.fg, fontSize: '0.76rem', fontWeight: 600, cursor: 'pointer',
                }}
              >
                Cancel
              </button>
              <button
                onClick={handleCreateManualMomAgenda}
                disabled={!newManualAgendaTitle.trim()}
                style={{
                  padding: '0.45rem 1rem', borderRadius: 7,
                  border: 'none', background: C.purple,
                  color: 'white', fontSize: '0.76rem', fontWeight: 700, cursor: 'pointer',
                  opacity: newManualAgendaTitle.trim() ? 1 : 0.5,
                }}
              >
                Create Agenda
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── MODAL: Add Local Agenda (Meeting side) ────────────────────────── */}
      {showAddAgendaModal && (
        <div style={{
          position: 'fixed', inset: 0, zIndex: 999,
          background: 'rgba(0,0,0,0.5)', display: 'flex', alignItems: 'center', justifyContent: 'center',
          backdropFilter: 'blur(2px)',
        }}>
          <div style={{
            width: '100%', maxWidth: 440, background: C.card,
            borderRadius: 12, border: `1px solid ${C.accentBorder}`,
            boxShadow: '0 20px 40px rgba(0,0,0,0.3)', padding: '1.25rem',
            display: 'flex', flexDirection: 'column', gap: 12,
          }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
                <PlusCircle size={16} color={C.accent} />
                <span style={{ fontSize: '0.9rem', fontWeight: 700, color: C.fg }}>
                  Add Local Agenda Group
                </span>
              </div>
              <button
                onClick={() => setShowAddAgendaModal(false)}
                style={{ background: 'none', border: 'none', cursor: 'pointer', color: C.mutedFg }}
              >
                <X size={15} />
              </button>
            </div>

            <div style={{ fontSize: '0.76rem', color: C.mutedFg, lineHeight: 1.5 }}>
              Create an agenda group in your local training dataset. You can move existing discussion points into it or add points directly.
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              <label style={{ fontSize: '0.72rem', fontWeight: 700, color: C.fg }}>
                Agenda Name:
              </label>
              <input
                autoFocus
                type="text"
                value={newAgendaTitleInput}
                onChange={e => setNewAgendaTitleInput(e.target.value)}
                onKeyDown={e => {
                  if (e.key === 'Enter') handleAddLocalAgenda();
                }}
                placeholder="e.g. Budget & Financial Review…"
                style={{
                  width: '100%', padding: '0.45rem 0.65rem', borderRadius: 7,
                  border: `1px solid ${C.accent}`, background: C.bg, color: C.fg,
                  fontSize: '0.8rem', fontFamily: 'Inter, sans-serif', outline: 'none',
                }}
              />
            </div>

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, marginTop: 4 }}>
              <button
                onClick={() => setShowAddAgendaModal(false)}
                style={{
                  padding: '0.45rem 0.9rem', borderRadius: 7,
                  border: `1px solid ${C.border}`, background: C.bg,
                  color: C.fg, fontSize: '0.76rem', fontWeight: 600, cursor: 'pointer',
                }}
              >
                Cancel
              </button>
              <button
                onClick={handleAddLocalAgenda}
                disabled={!newAgendaTitleInput.trim()}
                style={{
                  padding: '0.45rem 1rem', borderRadius: 7,
                  border: 'none', background: C.accent,
                  color: 'white', fontSize: '0.76rem', fontWeight: 700, cursor: 'pointer',
                  opacity: newAgendaTitleInput.trim() ? 1 : 0.5,
                }}
              >
                Add Agenda
              </button>
            </div>
          </div>
        </div>
      )}

    </div>
  );
}
