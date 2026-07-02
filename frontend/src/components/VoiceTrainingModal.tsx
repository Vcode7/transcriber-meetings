import { useState, useEffect, useRef, useCallback } from "react";
import { X, Mic2, Trash2, Play, Pause, Loader, CheckCircle, AlertCircle, User } from "lucide-react";
import api from "../api/client";
import type { Segment } from "./TranscriptViewer";

interface Sample {
  file_path: string;
  start: number;
  end: number;
  duration: number;
  segment_text: string;
}

interface Props {
  recordingId: string;
  segments: Segment[];
  onClose: () => void;
  onTrainingComplete: (oldLabel: string, newLabel: string, updatedSegments: Segment[]) => void;
}

const SPEAKER_COLORS = [
  "hsl(14, 90%, 56%)",
  "hsl(205, 90%, 55%)",
  "hsl(130, 60%, 45%)",
  "hsl(280, 70%, 60%)",
  "hsl(45, 90%, 50%)",
  "hsl(340, 80%, 60%)",
];

function fmtDur(s: number) {
  return `${s.toFixed(1)}s`;
}

// Audio playback row for a single sample
function SampleRow({ sample, index, onDelete }: { sample: Sample; index: number; onDelete: () => void }) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [playing, setPlaying] = useState(false);
  const [progress, setProgress] = useState(0);
  const [blobUrl, setBlobUrl] = useState<string | null>(null);

  useEffect(() => {
    let url: string | null = null;
    api
      .get("/voice/sample-audio", { params: { file_path: sample.file_path }, responseType: "blob" })
      .then((res) => {
        url = URL.createObjectURL(res.data);
        setBlobUrl(url);
      })
      .catch(() => {});
    return () => { if (url) URL.revokeObjectURL(url); };
  }, [sample.file_path]);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;
    const onTime = () => setProgress(audio.duration ? (audio.currentTime / audio.duration) * 100 : 0);
    const onEnd = () => { setPlaying(false); setProgress(0); };
    audio.addEventListener("timeupdate", onTime);
    audio.addEventListener("ended", onEnd);
    return () => { audio.removeEventListener("timeupdate", onTime); audio.removeEventListener("ended", onEnd); };
  }, [blobUrl]);

  const toggle = () => {
    const audio = audioRef.current;
    if (!audio) return;
    if (playing) { audio.pause(); setPlaying(false); }
    else { audio.play(); setPlaying(true); }
  };

  return (
    <div style={{
      display: "flex", alignItems: "center", gap: "10px",
      padding: ".65rem .9rem",
      background: "hsl(var(--muted) / .45)",
      border: "1px solid hsl(var(--ink) / .07)",
      borderRadius: "10px",
    }}>
      {blobUrl && <audio ref={audioRef} src={blobUrl} preload="metadata" />}
      <button
        onClick={toggle}
        disabled={!blobUrl}
        style={{
          width: 28, height: 28, borderRadius: "50%", border: "none",
          background: blobUrl ? "hsl(var(--accent))" : "hsl(var(--muted))",
          color: "hsl(var(--accent-foreground))",
          display: "flex", alignItems: "center", justifyContent: "center",
          cursor: blobUrl ? "pointer" : "not-allowed", flexShrink: 0,
        }}
      >
        {playing ? <Pause size={12} /> : <Play size={12} style={{ marginLeft: 1 }} />}
      </button>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ height: 4, borderRadius: 2, background: "hsl(var(--ink) / .1)", overflow: "hidden", marginBottom: 4 }}>
          <div style={{ height: "100%", width: `${progress}%`, background: "hsl(var(--accent))", transition: "width .1s" }} />
        </div>
        <div style={{ fontSize: ".72rem", color: "hsl(var(--pencil))", fontFamily: "Inter, sans-serif", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
          Sample {index + 1} · {fmtDur(sample.duration)}
          {sample.segment_text && ` · "${sample.segment_text.slice(0, 50)}${sample.segment_text.length > 50 ? "…" : ""}"`}
        </div>
      </div>
      <button
        onClick={onDelete}
        style={{
          width: 26, height: 26, borderRadius: "6px", border: "none",
          background: "hsl(var(--destructive) / .1)", color: "hsl(var(--destructive))",
          display: "flex", alignItems: "center", justifyContent: "center",
          cursor: "pointer", flexShrink: 0,
        }}
        title="Remove sample"
      >
        <Trash2 size={12} />
      </button>
    </div>
  );
}

export default function VoiceTrainingModal({ recordingId, segments, onClose, onTrainingComplete }: Props) {
  const speakers = Array.from(new Set(segments.map((s) => s.speaker_label))).filter(Boolean);

  const [selectedSpeaker, setSelectedSpeaker] = useState(speakers[0] || "");
  const [nameInput, setNameInput] = useState(speakers[0] || "");
  const [nameError, setNameError] = useState("");
  const [nameChecking, setNameChecking] = useState(false);

  const [samples, setSamples] = useState<Sample[]>([]);
  const [loadingSamples, setLoadingSamples] = useState(false);
  const [samplesError, setSamplesError] = useState("");

  const [training, setTraining] = useState(false);
  const [trained, setTrained] = useState<string | null>(null);
  const [trainError, setTrainError] = useState("");

  const checkTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const nameValidRef = useRef(true);

  useEffect(() => {
    if (!selectedSpeaker) return;
    setNameInput(selectedSpeaker);
    setNameError("");
    setTrained(null);
    setTrainError("");
    fetchSamples(selectedSpeaker);
  }, [selectedSpeaker]); // eslint-disable-line react-hooks/exhaustive-deps

  const fetchSamples = async (speaker: string) => {
    setLoadingSamples(true);
    setSamplesError("");
    setSamples([]);
    try {
      const res = await api.post("/voice/extract-samples", {
        recording_id: recordingId,
        speaker_label: speaker,
        max_samples: 5,
      });
      setSamples(res.data.samples || []);
    } catch (e: unknown) {
      const msg = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail || "Could not extract samples.";
      setSamplesError(msg);
    } finally {
      setLoadingSamples(false);
    }
  };

  const handleNameChange = useCallback((val: string) => {
    setNameInput(val);
    setNameError("");
    if (checkTimerRef.current) clearTimeout(checkTimerRef.current);
    if (!val.trim() || val.trim() === selectedSpeaker) {
      nameValidRef.current = true;
      return;
    }
    setNameChecking(true);
    checkTimerRef.current = setTimeout(async () => {
      try {
        const res = await api.get("/voice/check-label", { params: { label: val.trim() } });
        if (res.data.exists) {
          setNameError(`"${val.trim()}" is already used by another profile.`);
          nameValidRef.current = false;
        } else {
          nameValidRef.current = true;
        }
      } catch { nameValidRef.current = true; }
      finally { setNameChecking(false); }
    }, 450);
  }, [selectedSpeaker]);

  const handleDeleteSample = (idx: number) => setSamples((prev) => prev.filter((_, i) => i !== idx));

  const handleTrain = async () => {
    const label = nameInput.trim();
    if (!label) { setNameError("Name cannot be empty."); return; }
    if (nameError) return;
    if (samples.length === 0) { setTrainError("At least one sample is required."); return; }

    setTraining(true);
    setTrainError("");
    try {
      const res = await api.post("/voice/train-from-transcript", {
        recording_id: recordingId,
        speaker_label: selectedSpeaker,
        new_label: label,
        sample_paths: samples.map((s) => s.file_path),
      });
      const { new_label, updated_segment_count } = res.data;
      const updated = segments.map((seg) =>
        seg.speaker_label === selectedSpeaker ? { ...seg, speaker_label: new_label } : seg
      );
      setTrained(`"${selectedSpeaker}" → "${new_label}" · ${updated_segment_count} segments updated`);
      setSamples([]);
      onTrainingComplete(selectedSpeaker, new_label, updated);
    } catch (e: unknown) {
      const msg = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail || "Training failed.";
      setTrainError(msg);
    } finally {
      setTraining(false);
    }
  };

  const speakerColor = (sp: string) => SPEAKER_COLORS[speakers.indexOf(sp) % SPEAKER_COLORS.length];
  const canTrain = !training && !nameError && samples.length > 0 && !loadingSamples;

  return (
    <div
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
      style={{
        position: "fixed", inset: 0, zIndex: 1000,
        background: "rgba(0,0,0,.55)",
        display: "flex", alignItems: "center", justifyContent: "center",
        padding: "1rem", backdropFilter: "blur(4px)",
      }}
    >
      <div style={{
        background: "hsl(var(--card))",
        border: "1px solid hsl(var(--ink) / .12)",
        borderRadius: "16px",
        width: "100%", maxWidth: "580px", maxHeight: "90vh",
        display: "flex", flexDirection: "column",
        boxShadow: "0 24px 64px rgba(0,0,0,.45)",
        overflow: "hidden",
      }}>

        {/* Header */}
        <div style={{
          display: "flex", alignItems: "center", gap: "10px",
          padding: "1.1rem 1.4rem", borderBottom: "1px solid hsl(var(--ink) / .08)", flexShrink: 0,
        }}>
          <div style={{
            width: 32, height: 32, borderRadius: "8px",
            background: "hsl(var(--accent) / .12)", border: "1.5px solid hsl(var(--accent) / .3)",
            display: "flex", alignItems: "center", justifyContent: "center",
          }}>
            <Mic2 size={15} style={{ color: "hsl(var(--accent))" }} />
          </div>
          <div style={{ flex: 1 }}>
            <div style={{ fontWeight: 700, fontSize: ".95rem", color: "hsl(var(--ink))", fontFamily: "Inter, sans-serif" }}>Train Voice Samples</div>
            <div style={{ fontSize: ".72rem", color: "hsl(var(--pencil))", fontFamily: "Inter, sans-serif" }}>Assign names and build voice profiles from this recording</div>
          </div>
          <button onClick={onClose} style={{ width: 28, height: 28, borderRadius: "6px", border: "none", background: "hsl(var(--muted))", color: "hsl(var(--pencil))", display: "flex", alignItems: "center", justifyContent: "center", cursor: "pointer" }}>
            <X size={14} />
          </button>
        </div>

        {/* Body */}
        <div style={{ flex: 1, overflowY: "auto", padding: "1.2rem 1.4rem", display: "flex", flexDirection: "column", gap: "1.1rem" }}>

          {/* Speaker chips */}
          <div>
            <div style={{ fontSize: ".68rem", fontWeight: 700, textTransform: "uppercase", letterSpacing: ".08em", color: "hsl(var(--pencil))", fontFamily: "Inter, sans-serif", marginBottom: ".55rem" }}>
              Detected Speakers
            </div>
            <div style={{ display: "flex", gap: "8px", overflowX: "auto", paddingBottom: "4px" }}>
              {speakers.map((sp) => {
                const col = speakerColor(sp);
                const active = sp === selectedSpeaker;
                return (
                  <button
                    key={sp}
                    onClick={() => setSelectedSpeaker(sp)}
                    style={{
                      display: "flex", alignItems: "center", gap: "6px",
                      padding: ".35rem .85rem", borderRadius: "999px",
                      border: `1.5px solid ${active ? col : "hsl(var(--ink) / .12)"}`,
                      background: active ? `${col}22` : "hsl(var(--muted) / .5)",
                      color: active ? col : "hsl(var(--pencil))",
                      fontFamily: "Inter, sans-serif", fontSize: ".78rem", fontWeight: 600,
                      cursor: "pointer", flexShrink: 0, transition: "all .15s",
                    }}
                  >
                    <User size={11} /> {sp}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Name input */}
          <div>
            <div style={{ fontSize: ".68rem", fontWeight: 700, textTransform: "uppercase", letterSpacing: ".08em", color: "hsl(var(--pencil))", fontFamily: "Inter, sans-serif", marginBottom: ".45rem" }}>
              Assign Name to "{selectedSpeaker}"
            </div>
            <div style={{ position: "relative" }}>
              <input
                id="voice-train-name-input"
                type="text"
                value={nameInput}
                onChange={(e) => handleNameChange(e.target.value)}
                placeholder="Enter a name for this speaker…"
                style={{
                  width: "100%", boxSizing: "border-box",
                  padding: ".55rem .85rem", paddingRight: "2.2rem",
                  borderRadius: "8px",
                  border: `1.5px solid ${nameError ? "hsl(var(--destructive))" : "hsl(var(--ink) / .15)"}`,
                  background: "hsl(var(--muted) / .5)", color: "hsl(var(--ink))",
                  fontSize: ".88rem", fontFamily: "Inter, sans-serif", outline: "none",
                }}
              />
              {nameChecking && <Loader size={13} className="spin" style={{ position: "absolute", right: 10, top: "50%", transform: "translateY(-50%)", color: "hsl(var(--pencil))" }} />}
            </div>
            {nameError && (
              <div style={{ display: "flex", alignItems: "center", gap: "5px", marginTop: "5px", fontSize: ".75rem", color: "hsl(var(--destructive))", fontFamily: "Inter, sans-serif" }}>
                <AlertCircle size={11} /> {nameError}
              </div>
            )}
          </div>

          {/* Samples */}
          <div>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: ".55rem" }}>
              <div style={{ fontSize: ".68rem", fontWeight: 700, textTransform: "uppercase", letterSpacing: ".08em", color: "hsl(var(--pencil))", fontFamily: "Inter, sans-serif" }}>
                Voice Samples
              </div>
              {samples.length > 0 && <span style={{ fontSize: ".72rem", color: "hsl(var(--pencil))", fontFamily: "Inter, sans-serif" }}>{samples.length} sample{samples.length !== 1 ? "s" : ""} · click ▶ to preview</span>}
            </div>

            {loadingSamples && (
              <div style={{ display: "flex", alignItems: "center", gap: "8px", padding: "1rem", color: "hsl(var(--pencil))", fontSize: ".84rem", fontFamily: "Inter, sans-serif" }}>
                <Loader size={14} className="spin" style={{ color: "hsl(var(--accent))" }} /> Extracting best quality samples…
              </div>
            )}

            {samplesError && !loadingSamples && (
              <div style={{ display: "flex", alignItems: "center", gap: "7px", padding: ".75rem 1rem", borderRadius: "8px", background: "hsl(var(--destructive) / .08)", border: "1px solid hsl(var(--destructive) / .2)", color: "hsl(var(--destructive))", fontSize: ".8rem", fontFamily: "Inter, sans-serif" }}>
                <AlertCircle size={13} /> {samplesError}
              </div>
            )}

            {!loadingSamples && samples.length > 0 && (
              <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
                {samples.map((s, i) => (
                  <SampleRow key={s.file_path} sample={s} index={i} onDelete={() => handleDeleteSample(i)} />
                ))}
              </div>
            )}

            {!loadingSamples && !samplesError && samples.length === 0 && (
              <div style={{ padding: ".75rem 1rem", borderRadius: "8px", background: "hsl(var(--muted) / .4)", color: "hsl(var(--pencil))", fontSize: ".82rem", fontFamily: "Inter, sans-serif", textAlign: "center" }}>
                No samples available.
              </div>
            )}
          </div>

          {/* Success/error banners */}
          {trained && (
            <div style={{ display: "flex", alignItems: "center", gap: "8px", padding: ".75rem 1rem", borderRadius: "10px", background: "hsl(var(--success) / .1)", border: "1px solid hsl(var(--success) / .3)", color: "hsl(var(--success))", fontSize: ".82rem", fontFamily: "Inter, sans-serif", fontWeight: 600 }}>
              <CheckCircle size={14} /> {trained}
            </div>
          )}
          {trainError && (
            <div style={{ display: "flex", alignItems: "center", gap: "8px", padding: ".75rem 1rem", borderRadius: "10px", background: "hsl(var(--destructive) / .08)", border: "1px solid hsl(var(--destructive) / .25)", color: "hsl(var(--destructive))", fontSize: ".82rem", fontFamily: "Inter, sans-serif" }}>
              <AlertCircle size={13} /> {trainError}
            </div>
          )}
        </div>

        {/* Footer */}
        <div style={{ display: "flex", justifyContent: "flex-end", gap: "10px", padding: "1rem 1.4rem", borderTop: "1px solid hsl(var(--ink) / .08)", flexShrink: 0 }}>
          <button
            onClick={onClose}
            style={{ padding: ".5rem 1.1rem", borderRadius: "8px", border: "1px solid hsl(var(--ink) / .15)", background: "hsl(var(--muted) / .5)", color: "hsl(var(--pencil))", fontSize: ".84rem", fontWeight: 600, fontFamily: "Inter, sans-serif", cursor: "pointer" }}
          >
            {trained ? "Close" : "Cancel"}
          </button>
          <button
            id="voice-train-done-btn"
            onClick={handleTrain}
            disabled={!canTrain}
            style={{
              display: "flex", alignItems: "center", gap: "7px",
              padding: ".5rem 1.25rem", borderRadius: "8px", border: "none",
              background: canTrain ? "hsl(var(--accent))" : "hsl(var(--muted))",
              color: canTrain ? "hsl(var(--accent-foreground))" : "hsl(var(--pencil))",
              fontSize: ".84rem", fontWeight: 700, fontFamily: "Inter, sans-serif",
              cursor: canTrain ? "pointer" : "not-allowed", transition: "all .15s",
            }}
          >
            {training ? <><Loader size={13} className="spin" /> Training…</> : <><Mic2 size={13} /> Done — Train</>}
          </button>
        </div>
      </div>
    </div>
  );
}
