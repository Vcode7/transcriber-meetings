import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import { BookOpen, Pencil, Check, X as XIcon, Play, Pause, Mic2, Search, ChevronLeft, ChevronRight } from "lucide-react";
import api from "../api/client";
import VoiceTrainingModal from "./VoiceTrainingModal";

export interface Word {
  word: string;
  start: number;
  end: number;
  probability: number;
  score?: number;  // raw WhisperX field — aliased to probability in pipeline but may appear in fallback paths
}

export interface OverlapRegion {
  start: number;
  end: number;
  speakers: string[];   // resolved names after speaker identification
}

export interface Segment {
  speaker_label: string;
  start: number;
  end: number;
  text: string;
  words: Word[];
  is_overlap: boolean;
  overlap_regions?: OverlapRegion[];
}

interface Shortcut {
  shortcut: string;
  full_form: string;
}

interface Props {
  segments: Segment[];
  showConfidence: boolean;
  wordConfLow?: number;
  wordConfMid?: number;
  shortcuts?: Shortcut[];
  // New optional props
  audioUrl?: string;       // blob URL or HTTP URL for the audio player
  recordingId?: string;    // needed for transcript editing + voice training
  onSegmentsChange?: (segs: Segment[]) => void;
  /** When set, the segment with this seg_id is scrolled into view and briefly highlighted */
  highlightSegId?: string;
}

const SPEAKER_COLORS = [
  "hsl(14, 90%, 56%)",
  "hsl(205, 90%, 55%)",
  "hsl(130, 60%, 45%)",
  "hsl(280, 70%, 60%)",
  "hsl(45, 90%, 50%)",
  "hsl(340, 80%, 60%)",
];

/**
 * Minimum average word-level confidence to include a segment in MoM/AI Insights.
 * Must match backend config.MIN_AVG_SEGMENT_CONFIDENCE (default 0.35).
 */
const LOW_CONF_THRESHOLD = 0.40;

/** Safe confidence accessor: accepts both `probability` and `score` fields. */
function wordConf(w: Word): number {
  const v = w.probability ?? w.score;
  return typeof v === 'number' ? v : 1.0;
}

/** Compute the average word-level confidence for a segment. Returns null when no word data. */
function segmentAvgConf(seg: Segment): number | null {
  const words = seg.words;
  if (!words || words.length === 0) return null;
  return words.reduce((sum, w) => sum + wordConf(w), 0) / words.length;
}

function formatTime(s: number) {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60).toString().padStart(2, "0");
  return `${m}:${sec}`;
}

function wordClass(prob: number, low: number, mid: number) {
  if (prob < low) return "word-low";
  if (prob < mid) return "word-mid";
  return "word-hi";
}


function SpeakerAvatar({ label, color }: { label: string; color: string }) {
  const initials = label.split(" ").map((w) => w[0]).join("").toUpperCase().slice(0, 2);
  return (
    <div
      className="speaker-avatar"
      style={{ background: `${color}30`, border: `1.5px solid ${color}60`, color, flexShrink: 0, marginTop: "1px" }}
      title={label}
    >
      {initials}
    </div>
  );
}

function AnalysisItem({ label, value, color }: { label: string; value: string | number; color?: string }) {
  return (
    <div style={{ padding: ".75rem", borderRadius: "10px", background: "hsl(var(--muted) / .45)", border: "1px solid hsl(var(--ink) / .06)" }}>
      <div style={{ fontSize: ".7rem", color: "hsl(var(--pencil))", marginBottom: ".35rem", textTransform: "uppercase", letterSpacing: ".05em" }}>{label}</div>
      <div style={{ fontWeight: 700, fontSize: "1rem", color: color || "hsl(var(--ink))" }}>{value}</div>
    </div>
  );
}

function buildExpander(shortcuts: Shortcut[]): (text: string) => string {
  const patterns = shortcuts
    .filter((s) => s.shortcut?.trim())
    .map((s) => {
      const letters = s.shortcut.trim().toUpperCase().split("");
      const SEP = "[\\s.\\-,]*";
      const inner = letters.map((c) => `${c.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\.?`).join(SEP);
      const pattern = new RegExp(`(?<![A-Za-z0-9])${inner}(?![A-Za-z0-9])`, "gi");
      return { pattern, full_form: s.full_form };
    });
  return (text: string) => {
    let result = text;
    for (const { pattern, full_form } of patterns) result = result.replace(pattern, full_form);
    return result;
  };
}

// ── Embedded Audio Player ────────────────────────────────────────────────────
function EmbeddedPlayer({
  src,
  audioRef,
}: {
  src: string;
  audioRef: React.RefObject<HTMLAudioElement>;
}) {
  const [playing, setPlaying] = useState(false);
  const [progress, setProgress] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [error, setError] = useState(false);
  const scrubRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;
    setError(false);

    const updateMetadata = () => {
      if (Number.isFinite(audio.duration) && !isNaN(audio.duration)) {
        setDuration(audio.duration);
      }
    };

    const onTime = () => {
      setCurrentTime(audio.currentTime);
      const dur = Number.isFinite(audio.duration) && !isNaN(audio.duration) ? audio.duration : 0;
      setProgress(dur > 0 ? (audio.currentTime / dur) * 100 : 0);
    };

    const onEnd = () => setPlaying(false);
    const onPlay = () => setPlaying(true);
    const onPause = () => setPlaying(false);
    const onError = () => {
      console.warn("[EmbeddedPlayer] Failed to load audio stream");
      setError(true);
      setPlaying(false);
    };

    audio.addEventListener("timeupdate", onTime);
    audio.addEventListener("durationchange", updateMetadata);
    audio.addEventListener("loadedmetadata", updateMetadata);
    audio.addEventListener("ended", onEnd);
    audio.addEventListener("play", onPlay);
    audio.addEventListener("pause", onPause);
    audio.addEventListener("error", onError);

    if (Number.isFinite(audio.duration) && !isNaN(audio.duration) && audio.duration > 0) {
      setDuration(audio.duration);
    }

    return () => {
      audio.removeEventListener("timeupdate", onTime);
      audio.removeEventListener("durationchange", updateMetadata);
      audio.removeEventListener("loadedmetadata", updateMetadata);
      audio.removeEventListener("ended", onEnd);
      audio.removeEventListener("play", onPlay);
      audio.removeEventListener("pause", onPause);
      audio.removeEventListener("error", onError);
    };
  }, [audioRef, src]);

  const togglePlay = () => {
    const audio = audioRef.current;
    if (!audio || error) return;
    if (playing) {
      audio.pause();
    } else {
      audio.play().catch((err) => {
        console.warn("[EmbeddedPlayer] Play failure:", err);
        setPlaying(false);
      });
    }
  };

  const handleScrub = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      const audio = audioRef.current;
      const bar = scrubRef.current;
      const dur = Number.isFinite(audio?.duration) ? (audio?.duration || 0) : 0;
      if (!audio || !bar || dur <= 0) return;
      const pct = Math.max(0, Math.min(1, (e.clientX - bar.getBoundingClientRect().left) / bar.getBoundingClientRect().width));
      audio.currentTime = pct * dur;
    },
    [audioRef]
  );

  if (error) {
    return (
      <div className="custom-audio-player" style={{ margin: "0 0 12px", opacity: 0.8 }}>
        <span style={{ fontSize: "0.8rem", color: "hsl(var(--destructive))", padding: "0.4rem 0.6rem" }}>
          Audio stream unavailable for playback
        </span>
      </div>
    );
  }

  return (
    <div className="custom-audio-player" style={{ margin: "0 0 12px" }}>
      <audio ref={audioRef} src={src} preload="metadata" />
      <button className="audio-play-btn" onClick={togglePlay} title={playing ? "Pause" : "Play"}>
        {playing ? <Pause size={13} /> : <Play size={13} style={{ marginLeft: 1 }} />}
      </button>
      <div ref={scrubRef} className="audio-scrub" onClick={handleScrub} title="Click to seek">
        <div className="audio-scrub-fill" style={{ width: `${progress}%` }} />
      </div>
      <span className="audio-time">
        {formatTime(currentTime)} / {formatTime(duration)}
      </span>
    </div>
  );
}

// ── Main Component ───────────────────────────────────────────────────────────
export default function TranscriptViewer({
  segments: initialSegments,
  showConfidence = true,
  wordConfLow = 0.5,
  wordConfMid = 0.75,
  shortcuts = [],
  audioUrl,
  recordingId,
  onSegmentsChange,
  highlightSegId,
}: Props) {
  const [segments, setSegments] = useState<Segment[]>(initialSegments);
  const [useDictExp, setUseDictExp] = useState(false);
  const [loadedShortcuts, setLoadedShortcuts] = useState<Shortcut[]>([]);
  const [activeSegIdx, setActiveSegIdx] = useState<number>(-1);
  const [editingIdx, setEditingIdx] = useState<number>(-1);
  const [editText, setEditText] = useState("");
  const [savingIdx, setSavingIdx] = useState<number>(-1);
  const [showTrainModal, setShowTrainModal] = useState(false);
  // ── Find / Search ──────────────────────────────────────────────────────────
  const [findOpen, setFindOpen] = useState(false);
  const [findQuery, setFindQuery] = useState("");
  const [findCurrentIdx, setFindCurrentIdx] = useState(0);
  const findInputRef = useRef<HTMLInputElement>(null);
  const segmentRefs = useRef<Record<number, HTMLDivElement | null>>({});
  const fetchedRef = useRef(false);
  const audioRef = useRef<HTMLAudioElement>(null);
  // ── Highlight flash state for Speaker Tab jump ──────────────────────────────
  const [highlightFlashId, setHighlightFlashId] = useState<string | null>(null);

  // Scroll to and briefly flash the highlighted segment
  useEffect(() => {
    if (!highlightSegId) return;
    const el = document.querySelector<HTMLElement>(`[data-segid="${highlightSegId}"]`);
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'center' });
      setHighlightFlashId(highlightSegId);
      const t = setTimeout(() => setHighlightFlashId(null), 2000);
      return () => clearTimeout(t);
    }
  }, [highlightSegId]);

  // Keep local segments in sync when prop changes
  useEffect(() => {
    setSegments(initialSegments);
  }, [initialSegments]);

  // Load dictionary shortcuts
  useEffect(() => {
    if (shortcuts && shortcuts.length > 0) {
      setLoadedShortcuts(shortcuts);
    } else if (!shortcuts || shortcuts.length === 0) {
      if (fetchedRef.current) return;
      fetchedRef.current = true;
      api.get("/dictionary/shortcuts").then((res) => {
        if (Array.isArray(res.data)) setLoadedShortcuts(res.data);
      }).catch(() => {});
    }
  }, [shortcuts]);

  // Track playing segment from audio time
  useEffect(() => {
    if (!audioUrl) return;
    const audio = audioRef.current;
    if (!audio) return;
    const onTime = () => {
      const t = audio.currentTime;
      const idx = segments.findIndex((s) => t >= s.start && t < s.end);
      setActiveSegIdx(idx);
    };
    audio.addEventListener("timeupdate", onTime);
    return () => audio.removeEventListener("timeupdate", onTime);
  }, [audioUrl, segments]);

  const seekTo = useCallback((start: number) => {
    const audio = audioRef.current;
    if (!audio) return;

    const playAndSeek = () => {
      try {
        audio.currentTime = start;
      } catch (e) {
        console.warn("[TranscriptViewer] Seek failed:", e);
      }
      audio.play().catch((err) => {
        console.warn("[TranscriptViewer] Play failed:", err);
      });
    };

    if (audio.readyState >= 1) {
      playAndSeek();
    } else {
      const onLoaded = () => {
        playAndSeek();
        audio.removeEventListener("loadedmetadata", onLoaded);
      };
      audio.addEventListener("loadedmetadata", onLoaded);
      audio.load();
    }
  }, []);

  const handleEditStart = (idx: number) => {
    setEditingIdx(idx);
    setEditText(segments[idx].text);
  };

  const handleEditCancel = () => {
    setEditingIdx(-1);
    setEditText("");
  };

  const handleEditSave = async (idx: number) => {
    const newText = editText.trim();
    if (!newText || !recordingId) { handleEditCancel(); return; }
    setSavingIdx(idx);
    try {
      const res = await api.patch(`/history/${recordingId}/transcript`, { segment_index: idx, text: newText });
      const updated = segments.map((s, i) => (i === idx ? res.data.segment : s));
      setSegments(updated);
      onSegmentsChange?.(updated);
    } catch (e) {
      console.error("[TranscriptEdit] Save failed:", e);
    } finally {
      setSavingIdx(-1);
      setEditingIdx(-1);
      setEditText("");
    }
  };

  const handleTrainingComplete = (_oldLabel: string, _newLabel: string, updatedSegs: Segment[]) => {
    setSegments(updatedSegs);
    onSegmentsChange?.(updatedSegs);
  };

  // ── Find: compute matches whenever query or segments change ───────────────
  interface FindMatch { segIdx: number; charStart: number; charEnd: number }

  const findMatches = useMemo<FindMatch[]>(() => {
    const q = findQuery.trim();
    if (!q) return [];
    const qLower = q.toLowerCase();
    const results: FindMatch[] = [];

    segments.forEach((seg, segIdx) => {
      // Build the full visible text for this segment
      const segText = (seg.words && seg.words.length > 0)
        ? seg.words.map(w => w.word).join(" ")
        : seg.text;

      const lower = segText.toLowerCase();
      let cursor = 0;
      while (cursor < lower.length) {
        const pos = lower.indexOf(qLower, cursor);
        if (pos === -1) break;
        results.push({ segIdx, charStart: pos, charEnd: pos + q.length });
        cursor = pos + q.length;
      }
    });

    return results;
  }, [findQuery, segments]);

  // Reset current match index when query/matches change
  useEffect(() => {
    setFindCurrentIdx(0);
  }, [findQuery]);

  // Auto-scroll to the segment that contains the current match
  useEffect(() => {
    if (findMatches.length === 0) return;
    const match = findMatches[findCurrentIdx] ?? findMatches[0];
    const el = segmentRefs.current[match.segIdx];
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }, [findCurrentIdx, findMatches]);

  const findNext = useCallback(() => {
    if (findMatches.length === 0) return;
    setFindCurrentIdx(i => (i + 1) % findMatches.length);
  }, [findMatches.length]);

  const findPrev = useCallback(() => {
    if (findMatches.length === 0) return;
    setFindCurrentIdx(i => (i - 1 + findMatches.length) % findMatches.length);
  }, [findMatches.length]);

  const openFind = useCallback(() => {
    setFindOpen(true);
    setTimeout(() => findInputRef.current?.focus(), 50);
  }, []);

  const closeFind = useCallback(() => {
    setFindOpen(false);
    setFindQuery("");
    setFindCurrentIdx(0);
  }, []);

  // ── Helper: render a segment's text with find-highlights ──────────────────
  function renderTextWithHighlights(
    text: string,
    segIdx: number,
    matchesInSeg: { charStart: number; charEnd: number; globalIdx: number }[]
  ) {
    if (!matchesInSeg.length) return <>{text}</>;

    const parts: React.ReactNode[] = [];
    let cursor = 0;
    matchesInSeg.forEach(({ charStart, charEnd, globalIdx }, mIdx) => {
      if (cursor < charStart) parts.push(<span key={`p-${segIdx}-${mIdx}-pre`}>{text.slice(cursor, charStart)}</span>);
      const isActive = globalIdx === findCurrentIdx;
      parts.push(
        <mark
          key={`m-${segIdx}-${mIdx}`}
          style={{
            backgroundColor: isActive ? 'hsl(var(--accent))' : 'hsl(45,95%,60%/.5)',
            color: isActive ? 'hsl(var(--accent-foreground))' : 'inherit',
            borderRadius: '3px',
            padding: '0 1px',
            fontWeight: isActive ? 700 : 'inherit',
          }}
        >
          {text.slice(charStart, charEnd)}
        </mark>
      );
      cursor = charEnd;
    });
    if (cursor < text.length) parts.push(<span key={`p-${segIdx}-tail`}>{text.slice(cursor)}</span>);
    return <>{parts}</>;
  }

  const expander = useDictExp && loadedShortcuts.length > 0 ? buildExpander(loadedShortcuts) : null;

  if (!segments || segments.length === 0) {
    return (
      <div style={{ display: "flex", flexDirection: "column", gap: "0" }}>
        {audioUrl && <EmbeddedPlayer src={audioUrl} audioRef={audioRef} />}
        <div style={{ color: "hsl(var(--pencil))", textAlign: "center", padding: "3.5rem 2rem", fontSize: "0.95rem", fontFamily: "Inter, sans-serif", display: "flex", flexDirection: "column", alignItems: "center", gap: "1.25rem" }}>
          <div style={{ width: "72px", height: "72px", borderRadius: "50%", background: "hsl(var(--accent) / .08)", border: "2px dashed hsl(var(--accent) / .22)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: "2rem" }} className="animate-float">
            🎤
          </div>
          <div>
            <p style={{ fontWeight: 700, marginBottom: ".4rem", color: "hsl(var(--ink))", fontSize: "1rem" }}>No transcript yet</p>
            <p style={{ fontSize: ".85rem", opacity: 0.65, lineHeight: 1.5 }}>Record or upload audio to get started</p>
          </div>
        </div>
      </div>
    );
  }

  // Speaker color map
  const speakerColors: Record<string, string> = {};
  let colorIdx = 0;
  for (const seg of segments) {
    if (!(seg.speaker_label in speakerColors)) {
      speakerColors[seg.speaker_label] = SPEAKER_COLORS[colorIdx % SPEAKER_COLORS.length];
      colorIdx++;
    }
  }

  const hasWordConf = segments.some((s) => s.words && s.words.length > 0);
  const totalWords = segments.reduce((sum, seg) => sum + (seg.words?.length || 0), 0);
  const highWords = segments.reduce((sum, seg) => sum + (seg.words?.filter((w) => wordConf(w) >= wordConfMid).length || 0), 0);
  const midWords = segments.reduce((sum, seg) => sum + (seg.words?.filter((w) => wordConf(w) >= wordConfLow && wordConf(w) < wordConfMid).length || 0), 0);
  const lowWords = segments.reduce((sum, seg) => sum + (seg.words?.filter((w) => wordConf(w) < wordConfLow).length || 0), 0);
  const avgConfidence = totalWords > 0 ? (segments.reduce((sum, seg) => sum + (seg.words?.reduce((s, w) => s + wordConf(w), 0) || 0), 0) / totalWords) * 100 : 0;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "0" }}>

      {/* Embedded audio player */}
      {audioUrl && <EmbeddedPlayer src={audioUrl} audioRef={audioRef} />}

      {/* Toolbar row: dict toggle + train button + find button */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: findOpen ? "0" : "8px", flexWrap: "wrap", gap: "6px" }}>
        {loadedShortcuts.length > 0 && (
          <button
            id="dict-expansion-toggle"
            onClick={() => setUseDictExp((v) => !v)}
            title={useDictExp ? "Disable dictionary expansion" : "Expand abbreviations using Dictionary"}
            style={{
              display: "flex", alignItems: "center", gap: "6px",
              padding: ".35rem .8rem", borderRadius: "999px",
              border: `1.5px solid ${useDictExp ? "hsl(130,60%,45% / .4)" : "hsl(var(--ink) / .15)"}`,
              background: useDictExp ? "hsl(130,60%,45% / .08)" : "hsl(var(--muted) / .5)",
              color: useDictExp ? "hsl(130,60%,45%)" : "hsl(var(--pencil))",
              cursor: "pointer", fontSize: ".74rem", fontWeight: 600, fontFamily: "Inter, sans-serif", transition: "all .15s",
            }}
          >
            <BookOpen size={11} />
            {useDictExp ? "Dictionary ON" : "Dictionary"}
          </button>
        )}
        {!loadedShortcuts.length && <div />}

        {/* Right side: Train + Find buttons */}
        <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
          {recordingId && (
            <button
              id="train-voice-btn"
              onClick={() => setShowTrainModal(true)}
              style={{
                display: "flex", alignItems: "center", gap: "6px",
                padding: ".35rem .9rem", borderRadius: "999px",
                border: "1.5px solid hsl(var(--accent) / .35)",
                background: "hsl(var(--accent) / .08)",
                color: "hsl(var(--accent))",
                cursor: "pointer", fontSize: ".74rem", fontWeight: 700, fontFamily: "Inter, sans-serif", transition: "all .15s",
              }}
            >
              <Mic2 size={11} />
              Train Voice Samples
            </button>
          )}

          {/* Find button */}
          <button
            id="transcript-find-btn"
            onClick={openFind}
            title="Find in transcript (Ctrl+F)"
            style={{
              display: "flex", alignItems: "center", gap: "5px",
              padding: ".35rem .8rem", borderRadius: "999px",
              border: `1.5px solid ${findOpen ? "hsl(var(--accent) / .4)" : "hsl(var(--ink) / .15)"}`,
              background: findOpen ? "hsl(var(--accent) / .1)" : "hsl(var(--muted) / .5)",
              color: findOpen ? "hsl(var(--accent))" : "hsl(var(--pencil))",
              cursor: "pointer", fontSize: ".74rem", fontWeight: 600, fontFamily: "Inter, sans-serif", transition: "all .15s",
            }}
          >
            <Search size={11} />
            Find
          </button>
        </div>
      </div>

      {/* Find bar */}
      {findOpen && (
        <div style={{
          position: "sticky",
          top: "0px",
          zIndex: 40,
          display: "flex", alignItems: "center", gap: "6px",
          padding: ".45rem .75rem",
          margin: "6px 0 8px",
          background: "hsl(var(--card))",
          border: "1.5px solid hsl(var(--accent) / .4)",
          borderRadius: "10px",
          boxShadow: "0 4px 16px rgba(0, 0, 0, 0.12)",
        }}>
          <Search size={13} style={{ color: "hsl(var(--accent))", flexShrink: 0 }} />
          <input
            ref={findInputRef}
            id="transcript-find-input"
            type="text"
            placeholder="Search transcript…"
            value={findQuery}
            onChange={e => setFindQuery(e.target.value)}
            onKeyDown={e => {
              if (e.key === 'Enter') { e.shiftKey ? findPrev() : findNext(); }
              if (e.key === 'Escape') closeFind();
            }}
            style={{
              flex: 1, background: "transparent", border: "none", outline: "none",
              fontSize: ".83rem", fontFamily: "Inter, sans-serif", color: "hsl(var(--ink))",
              minWidth: 0,
            }}
          />

          {/* Match counter */}
          <span style={{
            fontSize: ".72rem", fontWeight: 700, fontFamily: "JetBrains Mono, monospace",
            color: findMatches.length === 0 && findQuery.trim()
              ? "hsl(var(--destructive))"
              : "hsl(var(--pencil))",
            whiteSpace: "nowrap", flexShrink: 0,
          }}>
            {findQuery.trim()
              ? (findMatches.length === 0
                ? "No matches"
                : `${findCurrentIdx + 1} of ${findMatches.length}`)
              : ""}
          </span>

          {/* Prev */}
          <button
            id="transcript-find-prev"
            onClick={findPrev}
            disabled={findMatches.length === 0}
            title="Previous match (Shift+Enter)"
            style={{
              background: "none", border: "none", cursor: findMatches.length ? "pointer" : "default",
              color: "hsl(var(--pencil))", display: "flex", alignItems: "center", padding: "2px 4px", borderRadius: "4px",
              opacity: findMatches.length === 0 ? 0.35 : 1, transition: "opacity .15s",
            }}
          >
            <ChevronLeft size={14} />
          </button>

          {/* Next */}
          <button
            id="transcript-find-next"
            onClick={findNext}
            disabled={findMatches.length === 0}
            title="Next match (Enter)"
            style={{
              background: "none", border: "none", cursor: findMatches.length ? "pointer" : "default",
              color: "hsl(var(--pencil))", display: "flex", alignItems: "center", padding: "2px 4px", borderRadius: "4px",
              opacity: findMatches.length === 0 ? 0.35 : 1, transition: "opacity .15s",
            }}
          >
            <ChevronRight size={14} />
          </button>

          {/* Close */}
          <button
            id="transcript-find-close"
            onClick={closeFind}
            title="Close find bar (Esc)"
            style={{
              background: "none", border: "none", cursor: "pointer",
              color: "hsl(var(--pencil))", display: "flex", alignItems: "center", padding: "2px 4px", borderRadius: "4px",
            }}
          >
            <XIcon size={13} />
          </button>
        </div>
      )}

      {/* Segments */}
      {segments.map((seg, i) => {
        const color = speakerColors[seg.speaker_label];
        const isActive = i === activeSegIdx;
        const isEditing = i === editingIdx;
        const isSaving = i === savingIdx;
        const avgConf = segmentAvgConf(seg);
        const isLowConf = avgConf !== null && avgConf < LOW_CONF_THRESHOLD;

        // ── Collect find-matches for this segment ────────────────────────────
        const segMatchesRaw = findQuery.trim()
          ? findMatches
              .map((m, gi) => ({ ...m, globalIdx: gi }))
              .filter(m => m.segIdx === i)
          : [];

        const isFlashing = highlightFlashId != null && seg.seg_id === highlightFlashId;

        return (
          <div
            key={i}
            ref={el => { segmentRefs.current[i] = el; }}
            data-segid={seg.seg_id || undefined}
            className="transcript-segment animate-slide-up"
            title={isLowConf
              ? `⚠ Low confidence transcription (avg ${(avgConf! * 100).toFixed(0)}% < ${(LOW_CONF_THRESHOLD * 100).toFixed(0)}%). Excluded from MoM & AI Insights.`
              : undefined
            }
            style={{
              "--speaker-color": color,
              scrollMarginTop: "60px",
              animationDelay: `${Math.min(i * 0.04, 0.5)}s`,
              animationFillMode: "both",
              background: isFlashing
                ? `${color}28`
                : isLowConf
                  ? "hsl(0, 80%, 96%)"
                  : isActive ? `${color}0d` : undefined,
              borderLeft: isFlashing
                ? `3px solid ${color}`
                : isActive ? `2.5px solid ${color}` : isLowConf ? "2.5px solid hsl(0,75%,65%)" : "2.5px solid transparent",
              transition: "background .4s, border-color .4s",
              outline: isFlashing ? `2px solid ${color}50` : isLowConf ? "1px solid hsl(0,75%,85%)" : undefined,
              outlineOffset: "-1px",
            } as React.CSSProperties}
          >
            <div className="seg-meta">
              <SpeakerAvatar label={seg.speaker_label} color={color} />
              <span className="speaker-name" style={{ color }}>{seg.speaker_label}</span>

              {/* Clickable timestamp */}
              <button
                onClick={() => seekTo(seg.start)}
                title={audioUrl ? `Seek to ${formatTime(seg.start)}` : "No audio loaded"}
                disabled={!audioUrl}
                style={{
                  display: "flex", alignItems: "center", gap: "3px",
                  background: "none", border: "none", cursor: audioUrl ? "pointer" : "default",
                  padding: "2px 5px", borderRadius: "5px",
                  fontSize: ".72rem", fontFamily: "JetBrains Mono, monospace",
                  color: audioUrl ? "hsl(var(--accent))" : "hsl(var(--pencil))",
                  opacity: audioUrl ? 1 : 0.6,
                  transition: "background .12s",
                }}
                onMouseEnter={(e) => { if (audioUrl) (e.currentTarget as HTMLElement).style.background = "hsl(var(--accent) / .1)"; }}
                onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.background = "none"; }}
              >
                {audioUrl && <Play size={8} style={{ flexShrink: 0 }} />}
                {formatTime(seg.start)} → {formatTime(seg.end)}
              </button>

              {seg.is_overlap && (() => {
                // Collect all unique speaker names from overlap_regions
                const regions = seg.overlap_regions ?? [];
                const speakerSet = new Set<string>();
                regions.forEach(r => r.speakers.forEach(sp => speakerSet.add(sp)));
                const speakerNames = Array.from(speakerSet);
                const label = speakerNames.length > 0
                  ? `Overlap: ${speakerNames.join(", ")}`
                  : "Overlap";
                return (
                  <span style={{ fontSize: ".65rem", fontWeight: 700, color: "hsl(var(--destructive))", background: "hsl(var(--destructive) / .1)", border: "1px solid hsl(var(--destructive) / .3)", borderRadius: "999px", padding: ".1rem .5rem", fontFamily: "Inter, sans-serif", letterSpacing: ".04em", display: "inline-flex", alignItems: "center", gap: "3px" }}>
                    ⚡ {label}
                  </span>
                );
              })()}

              {/* Low-confidence segment badge */}
              {isLowConf && (
                <span
                  title={`Average confidence ${(avgConf! * 100).toFixed(0)}% — below the ${(LOW_CONF_THRESHOLD * 100).toFixed(0)}% threshold. This segment is excluded from MoM & AI Insights.`}
                  style={{
                    fontSize: ".62rem", fontWeight: 700,
                    color: "hsl(0,65%,45%)",
                    background: "hsl(0,80%,96%)",
                    border: "1px solid hsl(0,75%,75%)",
                    borderRadius: "999px",
                    padding: ".1rem .45rem",
                    fontFamily: "Inter, sans-serif",
                    letterSpacing: ".04em",
                    display: "inline-flex",
                    alignItems: "center",
                    gap: "3px",
                    cursor: "help",
                  }}
                >
                  ⚠ Low Confidence
                </span>
              )}

              {/* Edit button — only if recordingId supplied */}
              {recordingId && !isEditing && (
                <button
                  onClick={() => handleEditStart(i)}
                  title="Edit this segment"
                  className="seg-edit-btn"
                  style={{
                    marginLeft: "auto",
                    background: "none", border: "none", cursor: "pointer",
                    display: "flex", alignItems: "center", gap: "3px",
                    padding: "2px 6px", borderRadius: "5px",
                    fontSize: ".7rem", color: "hsl(var(--pencil))", opacity: 0,
                    transition: "opacity .15s",
                  }}
                >
                  <Pencil size={10} />
                </button>
              )}
            </div>

            <div className="seg-text">
              {isEditing ? (
                <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
                  <textarea
                    autoFocus
                    value={editText}
                    onChange={(e) => setEditText(e.target.value)}
                    rows={3}
                    style={{
                      width: "100%", boxSizing: "border-box",
                      padding: ".5rem .7rem",
                      borderRadius: "8px",
                      border: "1.5px solid hsl(var(--accent) / .4)",
                      background: "hsl(var(--muted) / .6)",
                      color: "hsl(var(--ink))",
                      fontSize: ".88rem", fontFamily: "Inter, sans-serif",
                      resize: "vertical", outline: "none",
                    }}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) handleEditSave(i);
                      if (e.key === "Escape") handleEditCancel();
                    }}
                  />
                  <div style={{ display: "flex", gap: "6px", alignItems: "center" }}>
                    <button
                      onClick={() => handleEditSave(i)}
                      disabled={isSaving}
                      style={{
                        display: "flex", alignItems: "center", gap: "4px",
                        padding: ".3rem .75rem", borderRadius: "6px", border: "none",
                        background: "hsl(var(--accent))", color: "hsl(var(--accent-foreground))",
                        fontSize: ".76rem", fontWeight: 700, fontFamily: "Inter, sans-serif",
                        cursor: isSaving ? "wait" : "pointer",
                      }}
                    >
                      <Check size={11} /> {isSaving ? "Saving…" : "Save"}
                    </button>
                    <button
                      onClick={handleEditCancel}
                      style={{
                        display: "flex", alignItems: "center", gap: "4px",
                        padding: ".3rem .75rem", borderRadius: "6px",
                        border: "1px solid hsl(var(--ink) / .15)",
                        background: "hsl(var(--muted) / .5)", color: "hsl(var(--pencil))",
                        fontSize: ".76rem", fontWeight: 600, fontFamily: "Inter, sans-serif",
                        cursor: "pointer",
                      }}
                    >
                      <XIcon size={11} /> Cancel
                    </button>
                    <span style={{ fontSize: ".68rem", color: "hsl(var(--pencil))", fontFamily: "Inter, sans-serif", opacity: 0.7 }}>
                      Ctrl+Enter to save · Esc to cancel
                    </span>
                  </div>
                </div>
              ) : (
                <>
                  {seg.words && seg.words.length > 0
                    ? (() => {
                        // Build full word text for offset calculation
                        const wordTexts = seg.words.map(w => expander ? expander(w.word) : w.word);
                        let charOffset = 0;
                        return seg.words.map((w, wi) => {
                          const wordText = wordTexts[wi];
                          const wordStart = charOffset;
                          const wordEnd = charOffset + wordText.length;
                          charOffset += wordText.length + 1; // +1 for space

                          const conf = wordConf(w);

                          // Find matches that overlap this word
                          const matchesHere = segMatchesRaw.filter(
                            m => m.charStart < wordEnd && m.charEnd > wordStart
                          );

                          let inner: React.ReactNode;
                          if (matchesHere.length > 0) {
                            // Map global char positions to local word positions
                            const localMatches = matchesHere.map(m => ({
                              charStart: Math.max(0, m.charStart - wordStart),
                              charEnd: Math.min(wordText.length, m.charEnd - wordStart),
                              globalIdx: m.globalIdx,
                            }));
                            inner = renderTextWithHighlights(wordText, i, localMatches);
                          } else {
                            inner = wordText;
                          }

                          if (showConfidence) {
                            return (
                              <span key={wi} className={wordClass(conf, wordConfLow, wordConfMid)} title={`${(conf * 100).toFixed(0)}% confidence`}>
                                {inner}{" "}
                              </span>
                            );
                          }
                          return (
                            <span key={wi} className={conf < 0.3 ? "word-underlined" : ""} title={conf < 0.3 ? `${(conf * 100).toFixed(0)}% confidence` : ""}>
                              {inner}{" "}
                            </span>
                          );
                        });
                      })()
                    : (() => {
                        const rawText = expander ? expander(seg.text) : seg.text;
                        if (segMatchesRaw.length > 0) {
                          return renderTextWithHighlights(rawText, i, segMatchesRaw);
                        }
                        return rawText;
                      })()}
                </>
              )}
            </div>
          </div>
        );
      })}

      {/* Analysis box */}
      <div style={{ marginTop: "12px", background: "hsl(var(--card))", border: "1px solid hsl(var(--ink) / .08)", borderRadius: "12px", padding: "1rem" }}>
        <div style={{ fontSize: ".72rem", fontWeight: 700, textTransform: "uppercase", letterSpacing: ".08em", color: "hsl(var(--pencil))", marginBottom: ".9rem" }}>
          Transcript Analysis
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px,1fr))", gap: "14px" }}>
          <AnalysisItem label="Total Words" value={totalWords} />
          <AnalysisItem label="High Confidence" value={`${highWords} (${((highWords / totalWords) * 100 || 0).toFixed(1)}%)`} color="hsl(var(--sticky-green))" />
          <AnalysisItem label="Medium Confidence" value={`${midWords} (${((midWords / totalWords) * 100 || 0).toFixed(1)}%)`} color="hsl(45,90%,50%)" />
          <AnalysisItem label="Low Confidence" value={`${lowWords} (${((lowWords / totalWords) * 100 || 0).toFixed(1)}%)`} color="hsl(var(--destructive))" />
          <AnalysisItem label="Average Accuracy" value={`${avgConfidence.toFixed(2)}%`} color="hsl(var(--accent))" />
        </div>
      </div>

      {/* Word confidence legend */}
      {hasWordConf && (
        <div style={{ display: "flex", gap: "10px", padding: ".85rem 1rem", fontSize: "0.78rem", color: "hsl(var(--ink-soft))", marginTop: ".75rem", background: "hsl(var(--card))", borderRadius: "10px", border: "1px solid hsl(var(--ink) / .08)", flexWrap: "wrap", fontFamily: "Inter, sans-serif", fontWeight: 500, alignItems: "center" }}>
          <span style={{ fontSize: ".68rem", color: "hsl(var(--pencil))", textTransform: "uppercase", letterSpacing: ".1em", fontWeight: 700 }}>Word confidence:</span>
          {[{ label: "High >75%", cls: "word-hi" }, { label: "Mid 50–75%", cls: "word-mid" }, { label: "Low <50%", cls: "word-low" }].map(({ label, cls }) => (
            <span key={cls} style={{ display: "flex", alignItems: "center", gap: "5px" }}>
              <span className={cls} style={{ fontSize: ".72rem", padding: "1px 6px" }}>{label}</span>
            </span>
          ))}
        </div>
      )}

      {/* Speaker legend */}
      {Object.keys(speakerColors).length > 1 && (
        <div style={{ display: "flex", gap: "8px", flexWrap: "wrap", padding: ".75rem 1rem", marginTop: hasWordConf ? "6px" : ".75rem", background: "hsl(var(--card))", borderRadius: "10px", border: "1px solid hsl(var(--ink) / .08)", alignItems: "center" }}>
          <span style={{ fontSize: ".68rem", color: "hsl(var(--pencil))", textTransform: "uppercase", letterSpacing: ".1em", fontWeight: 700, fontFamily: "Inter, sans-serif" }}>Speakers:</span>
          {Object.entries(speakerColors).map(([label, color]) => (
            <span key={label} style={{ display: "flex", alignItems: "center", gap: "5px", fontSize: ".75rem", fontFamily: "Inter, sans-serif", fontWeight: 600, color: "hsl(var(--ink-soft))" }}>
              <span style={{ width: 10, height: 10, borderRadius: "50%", background: color, display: "inline-block", boxShadow: `0 0 4px ${color}60` }} />
              {label}
            </span>
          ))}
        </div>
      )}

      {/* Voice Training Modal */}
      {showTrainModal && recordingId && (
        <VoiceTrainingModal
          recordingId={recordingId}
          segments={segments}
          onClose={() => setShowTrainModal(false)}
          onTrainingComplete={handleTrainingComplete}
        />
      )}
    </div>
  );
}
