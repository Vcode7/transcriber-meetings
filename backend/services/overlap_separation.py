"""
Overlap re-transcription via speech separation — 100% offline, opt-in.

Whisper transcribes a mixture: where two people talk at once it usually keeps
the louder speaker and drops the other, and exclusive diarization then gives
every remaining word to one speaker. This module recovers the second speaker:

  1. Collect overlap windows from diarization (pad, merge neighbours, keep ≥ min).
  2. Embed every diarization speaker once (overlap-masked clean speech).
  3. Run pyannote/speech-separation-ami-1.0 (ToTaToNet) on a context crop around
     each window → one separated source per local speaker.
  4. Match each source to one of the window's diarization speakers by embedding.
  5. Unload separation, then Whisper + align each matched source inside its window.
  6. Return the recovered words (raw speaker id, global timestamps,
     ``from_separation=True``); ``merge_recovered_words`` adds the ones the main
     transcript does not already contain as their own overlap turns.

Models are loaded one at a time and unloaded after use (6 GB GPU budget). A
missing model or any failure logs a warning and returns no words — it never fails
the job. Enable with OVERLAP_SEPARATION_ENABLED; model download:
``python tools/download_speaker_models.py --with-separation``.
"""
import gc
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config import settings

logger = logging.getLogger(__name__)

MODEL_NAME = "speech_separation"
_separation_pipeline = None
_missing_logged = False


# ── Model management ─────────────────────────────────────────────────────────

def _model_dir():
    from services.model_loader import ModelLoader
    # _try_load_plain: same lookup as get_model_path without its "not found" warning
    # (this is polled by the settings page; the missing-model warning is logged once below).
    path = ModelLoader._try_load_plain(MODEL_NAME)
    if path is None or not (path / "config.yaml").exists():
        return None
    return path


def is_separation_available() -> bool:
    """True when the offline separation snapshot is present in MODELS_DIR."""
    return _model_dir() is not None


def is_separation_enabled(user_settings: Optional[Dict[str, Any]] = None) -> bool:
    """Per-user override (user_settings.overlap_separation_enabled) or the global setting."""
    value = (user_settings or {}).get("overlap_separation_enabled")
    if value is None:
        value = getattr(settings, "OVERLAP_SEPARATION_ENABLED", False)
    return bool(value)


def _load_pipeline():
    global _separation_pipeline, _missing_logged
    if _separation_pipeline is not None:
        return _separation_pipeline
    path = _model_dir()
    if path is None:
        if not _missing_logged:
            logger.warning(
                f"[Separation] {MODEL_NAME}/ not found in MODELS_DIR ({settings.MODELS_DIR}) — "
                "overlap re-transcription skipped. Run tools/download_speaker_models.py --with-separation."
            )
            _missing_logged = True
        return None
    from services.compat import apply_compatibility_patches
    apply_compatibility_patches()
    from services.device_utils import get_torch_device, log_gpu_memory
    from pyannote.audio import Pipeline
    log_gpu_memory("Pre-load Separation")
    pipeline = Pipeline.from_pretrained(str(path))
    _separation_pipeline = pipeline.to(get_torch_device())
    log_gpu_memory("Post-load Separation")
    logger.info(f"[Separation] speech-separation-ami-1.0 ready from {path}")
    return _separation_pipeline


def unload_separation_pipeline() -> None:
    global _separation_pipeline
    if _separation_pipeline is None:
        return
    _separation_pipeline = None
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    logger.info("[Separation] pipeline unloaded.")


# ── Window selection ─────────────────────────────────────────────────────────

def select_overlap_windows(
    diar_segs: List[Dict[str, Any]],
    duration: float,
    min_sec: Optional[float] = None,
    pad_sec: Optional[float] = None,
    max_windows: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Overlap windows worth separating: padded, merged when they touch, at least
    ``min_sec`` long (before padding) and capped at ``max_windows`` (longest first,
    returned in time order). Each window: {start, end, core_start, core_end, speakers}.
    """
    from services.diarization import collect_overlap_windows

    min_sec = float(getattr(settings, "OVERLAP_SEPARATION_MIN_SEC", 0.5) if min_sec is None else min_sec)
    pad_sec = float(getattr(settings, "OVERLAP_SEPARATION_PAD_SEC", 0.3) if pad_sec is None else pad_sec)
    max_windows = int(getattr(settings, "OVERLAP_SEPARATION_MAX_WINDOWS", 200) if max_windows is None else max_windows)

    merged: List[Dict[str, Any]] = []
    for w in collect_overlap_windows(diar_segs):
        if w["end"] - w["start"] < min_sec:
            continue
        start, end = max(0.0, w["start"] - pad_sec), min(duration, w["end"] + pad_sec)
        if merged and start <= merged[-1]["end"]:
            last = merged[-1]
            last["end"] = max(last["end"], end)
            last["core_end"] = max(last["core_end"], w["end"])
            last["speakers"] = sorted(set(last["speakers"]) | set(w.get("speakers", [])))
        else:
            merged.append({
                "start": start, "end": end,
                "core_start": w["start"], "core_end": w["end"],
                "speakers": sorted(set(w.get("speakers", []))),
            })
    merged = [w for w in merged if len(w["speakers"]) >= 2]
    if len(merged) > max_windows:
        logger.warning(f"[Separation] {len(merged)} overlap windows — keeping the {max_windows} longest.")
        merged = sorted(merged, key=lambda w: w["end"] - w["start"], reverse=True)[:max_windows]
        merged.sort(key=lambda w: w["start"])
    return merged


# ── Separation + source matching ─────────────────────────────────────────────

def _separate_crop(pipeline, crop: np.ndarray, sr: int, n_speakers: int) -> Optional[np.ndarray]:
    """Run separation on a crop; returns (num_samples, num_sources) or None."""
    import torch
    waveform = torch.from_numpy(crop.astype(np.float32)).unsqueeze(0)
    output = pipeline({"waveform": waveform, "sample_rate": sr}, min_speakers=min(2, n_speakers))
    sources = output[1] if isinstance(output, tuple) else getattr(output, "sources", None)
    if sources is None:
        return None
    data = np.asarray(getattr(sources, "data", sources), dtype=np.float32)
    if data.ndim == 1:
        data = data[:, None]
    return data


def _match_sources(
    source_embs: List[Optional[np.ndarray]],
    window_speakers: List[str],
    speaker_refs: Dict[str, Optional[np.ndarray]],
    min_sim: float = 0.3,
) -> Dict[int, Tuple[str, float]]:
    """One-to-one source → diarization speaker matching (Hungarian), above ``min_sim``."""
    from scipy.optimize import linear_sum_assignment
    from services.embedding import cosine_similarity

    speakers = [s for s in window_speakers if speaker_refs.get(s) is not None]
    sources = [i for i, e in enumerate(source_embs) if e is not None]
    if not speakers or not sources:
        return {}
    sims = np.array([[cosine_similarity(source_embs[i], speaker_refs[s]) for s in speakers] for i in sources])
    rows, cols = linear_sum_assignment(-sims)
    return {
        sources[r]: (speakers[c], float(sims[r, c]))
        for r, c in zip(rows, cols)
        if sims[r, c] >= min_sim
    }


def _separate_windows(
    audio: np.ndarray,
    sr: int,
    windows: List[Dict[str, Any]],
    speaker_refs: Dict[str, Optional[np.ndarray]],
    embedding_model: str,
) -> List[Tuple[Dict[str, Any], str, np.ndarray]]:
    """Separate every window; returns [(window, raw speaker id, source audio for window)]."""
    from services.identification import _embed_audio

    pipeline = _load_pipeline()
    if pipeline is None:
        return []
    context = float(getattr(settings, "OVERLAP_SEPARATION_CONTEXT_SEC", 5.0))
    total = len(audio) / sr
    results = []
    try:
        for w in windows:
            extra = max(0.0, (context - (w["end"] - w["start"])) / 2)
            c_start, c_end = max(0.0, w["start"] - extra), min(total, w["end"] + extra)
            crop = audio[int(c_start * sr):int(c_end * sr)]
            try:
                sources = _separate_crop(pipeline, crop, sr, len(w["speakers"]))
            except Exception as e:
                logger.warning(f"[Separation] window {w['start']:.2f}-{w['end']:.2f}s failed: {e}")
                continue
            if sources is None or sources.shape[1] < 2:
                continue
            # Embed each source over the whole crop (more speech → steadier match).
            source_embs = [_embed_audio(sources[:, k], sr, embedding_model) for k in range(sources.shape[1])]
            matches = _match_sources(source_embs, w["speakers"], speaker_refs)
            s_idx, e_idx = int((w["start"] - c_start) * sr), int((w["end"] - c_start) * sr)
            for k, (raw_id, sim) in matches.items():
                results.append((w, raw_id, sources[s_idx:e_idx, k].copy()))
                logger.debug(
                    f"[Separation] {w['start']:.2f}-{w['end']:.2f}s source {k} → {raw_id} (sim={sim:.3f})"
                )
    finally:
        unload_separation_pipeline()
    return results


# ── Re-transcription ─────────────────────────────────────────────────────────

def _transcribe_sources(
    separated: List[Tuple[Dict[str, Any], str, np.ndarray]],
    sr: int,
    language: Optional[str],
    initial_prompt: Optional[str],
) -> List[Dict[str, Any]]:
    """Whisper + forced alignment on each separated source; words inside the core window."""
    import whisperx
    from dataclasses import is_dataclass, replace
    from services import transcription as tx

    language = tx.resolve_language(language)
    model = tx.get_whisperx_model()
    device, _ = tx._resolve_device()
    if hasattr(model, "options") and is_dataclass(model.options):
        model.options = replace(model.options, initial_prompt=(initial_prompt or "").strip() or None)

    align_cache: Dict[str, Any] = {}
    words_out: List[Dict[str, Any]] = []
    for w, raw_id, source in separated:
        if len(source) < int(0.3 * sr):
            continue
        peak = float(np.max(np.abs(source))) or 1.0
        source = (source / peak * 0.9).astype(np.float32)  # separated sources come out at arbitrary gain
        try:
            kwargs = {"batch_size": 1}
            if language:
                kwargs["language"] = language
            result = model.transcribe(source, **kwargs)
            segs = result.get("segments", [])
            if not segs:
                continue
            lang = result.get("language", language or "en")
            if lang not in align_cache:
                model_dir, cache_only = tx._resolve_align_model_dir()
                align_cache[lang] = whisperx.load_align_model(
                    language_code=lang, device=device, model_name="facebook/wav2vec2-base",
                    model_dir=model_dir, model_cache_only=cache_only,
                )
            model_a, metadata = align_cache[lang]
            aligned = whisperx.align(segs, model_a, metadata, source, device, return_char_alignments=False)
        except Exception as e:
            logger.warning(f"[Separation] re-transcription {w['start']:.2f}-{w['end']:.2f}s failed: {e}")
            continue
        for seg in aligned.get("segments", []):
            for word in seg.get("words", []):
                if "start" not in word or "end" not in word:
                    continue
                g_start = round(w["start"] + float(word["start"]), 3)
                g_end = round(w["start"] + float(word["end"]), 3)
                mid = (g_start + g_end) / 2
                if not (w["core_start"] - 0.1 <= mid <= w["core_end"] + 0.1):
                    continue  # padding/context speech belongs to the main transcript
                words_out.append({
                    "word": str(word.get("word", "")).strip(),
                    "start": g_start,
                    "end": g_end,
                    "probability": round(float(word.get("score", word.get("probability", 0.0)) or 0.0), 4),
                    "speaker": raw_id,
                    "from_separation": True,
                })
    align_cache.clear()
    return words_out


def recover_overlap_words(
    file_path: str,
    diar_segs: List[Dict[str, Any]],
    language: Optional[str] = None,
    initial_prompt: Optional[str] = None,
    embedding_model: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Separate overlapping speech and re-transcribe each speaker. Returns recovered
    words (raw diarization speaker ids); [] when disabled upstream, unavailable or
    on failure. Leaves the separation, Whisper and alignment models unloaded.
    """
    if not diar_segs or not is_separation_available():
        if diar_segs and not is_separation_available():
            _load_pipeline()  # logs the missing-model warning once
        return []
    from services import embedding_router
    from services.identification import compute_speaker_embeddings
    from utils.audio_utils import load_audio

    try:
        audio, sr = load_audio(file_path, target_sr=16000)
        windows = select_overlap_windows(diar_segs, duration=len(audio) / sr)
        if not windows:
            logger.info("[Separation] no overlap windows to separate.")
            return []
        model = embedding_model or embedding_router.get_active_model()
        speaker_refs, _ = compute_speaker_embeddings(audio, sr, diar_segs, model)
        logger.info(f"[Separation] separating {len(windows)} overlap windows...")
        separated = _separate_windows(audio, sr, windows, speaker_refs, model)
        if not separated:
            return []
        words = _transcribe_sources(separated, sr, language, initial_prompt)
        logger.info(
            f"[Separation] recovered {len(words)} words from {len(separated)} separated sources "
            f"in {len(windows)} windows"
        )
        return words
    except Exception as e:
        logger.warning(f"[Separation] overlap re-transcription failed (non-fatal): {e}", exc_info=True)
        return []
    finally:
        unload_separation_pipeline()
        try:
            from services.transcription import unload_whisperx_model, unload_align_model
            unload_whisperx_model()
            unload_align_model()
        except Exception:
            pass


# ── Merge into the transcript ────────────────────────────────────────────────

def _norm(word: str) -> str:
    return re.sub(r"[^\w']+", "", word.lower())


def merge_recovered_words(
    speaker_segments: List[Dict[str, Any]],
    recovered_words: List[Dict[str, Any]],
    identified_segs: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Add recovered overlap words as their own turns.

    A recovered word is dropped when the transcript already has the same word
    overlapping it in time (the main pass caught it). The rest are grouped into
    contiguous same-speaker runs (gap ≤ 1 s) and inserted as segments marked
    ``is_overlap`` and ``from_separation``, labelled via the identification result.
    """
    stats = {"recovered": len(recovered_words), "duplicates": 0, "added_words": 0, "added_segments": 0}
    if not recovered_words:
        return speaker_segments, stats

    id_to_label: Dict[str, Tuple[str, Optional[str]]] = {}
    for s in identified_segs:
        raw = s.get("speaker")
        if raw and raw not in id_to_label:
            id_to_label[raw] = (s.get("speaker_label") or raw, s.get("speaker_profile_id"))

    existing = [
        (float(w.get("start", 0.0)), float(w.get("end", 0.0)), _norm(str(w.get("word", ""))))
        for seg in speaker_segments for w in seg.get("words", []) if isinstance(w, dict)
    ]
    existing.sort()
    starts = np.array([e[0] for e in existing]) if existing else np.array([])

    def _is_duplicate(word: Dict[str, Any]) -> bool:
        text = _norm(word["word"])
        if not text or not existing:
            return not text
        lo = int(np.searchsorted(starts, word["start"] - 2.0))
        for s, e, t in existing[lo:]:
            if s > word["end"] + 0.2:
                break
            if t == text and min(e, word["end"] + 0.2) > max(s, word["start"] - 0.2):
                return True
        return False

    kept = []
    for w in sorted(recovered_words, key=lambda x: (x["speaker"], x["start"])):
        if _is_duplicate(w):
            stats["duplicates"] += 1
        else:
            kept.append(w)

    new_segments: List[Dict[str, Any]] = []
    run: List[Dict[str, Any]] = []

    def _flush():
        if not run:
            return
        raw = run[0]["speaker"]
        label, profile_id = id_to_label.get(raw, (raw, None))
        words = [{**w, "speaker_label": label} for w in run]
        start, end = words[0]["start"], words[-1]["end"]
        overlapping = sorted({
            s.get("speaker_label") for s in speaker_segments
            if s.get("speaker_label") and s.get("speaker_label") != label
            and max(start, s["start"]) < min(end, s["end"])
        })
        new_segments.append({
            "start": start,
            "end": end,
            "text": " ".join(w["word"] for w in words if w["word"]),
            "words": words,
            "avg_logprob": 0.0,
            "speaker_label": label,
            "speaker_profile_id": profile_id,
            "speaker": raw,
            "is_overlap": True,
            "overlap_regions": [{"start": start, "end": end, "speakers": sorted({label, *overlapping})}],
            "overlap_speakers": overlapping,
            "from_separation": True,
        })
        stats["added_words"] += len(words)
        run.clear()

    for w in kept:
        if run and (w["speaker"] != run[-1]["speaker"] or w["start"] - run[-1]["end"] > 1.0):
            _flush()
        run.append(w)
    _flush()

    stats["added_segments"] = len(new_segments)
    if not new_segments:
        return speaker_segments, stats
    merged = sorted(speaker_segments + new_segments, key=lambda s: (s["start"], s["end"]))
    logger.info(
        f"[Separation] merged {stats['added_words']} recovered words as {stats['added_segments']} overlap turns "
        f"({stats['duplicates']} already in transcript)"
    )
    return merged, stats
