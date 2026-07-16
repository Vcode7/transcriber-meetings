"""
Speaker identification service.

Architecture change (speaker-level identification):
----------------------------------------------------
Previously, embeddings were extracted and matched per diarization segment,
which caused a single diarized speaker (e.g. SPEAKER_00) to receive different
human-readable labels across its segments — incorrect and noisy.

The new workflow:
  1. Group all diarization segments by their raw diarization ID (SPEAKER_00, etc.)
  2. For each unique speaker, extract an ECAPA-TDNN embedding from every segment.
  3. Average those embeddings (L2-normalised) to get one robust embedding per speaker.
  4. Perform profile matching ONCE per unique diarization speaker.
  5. Apply the resolved label to all segments for that speaker.

This guarantees a single, stable human-readable label per diarization speaker.

Profile scoring (multi-sample centroid):
-----------------------------------------
When a voice profile has multiple stored embeddings, this service:
  - Computes a **centroid** (L2-normalised mean) of all valid same-dimension
    embeddings in the profile.
  - Compares the query embedding against the centroid (single cosine distance).
  - Falls back to best_match_similarity() (max over all samples) if the profile
    has only one embedding or if centroid computation fails.

This produces more stable, noise-robust matching than max-over-samples alone.

Threshold behaviour:
--------------------
- If the caller provides an explicit ``similarity_threshold`` the value is used as-is.
- If ``use_model_default=True`` (the default) *and* the user threshold matches the
  legacy Resemblyzer value (0.75), the model-specific ECAPA-TDNN default from
  ``settings.SPEAKER_SIMILARITY_THRESHOLD_ECAPA_TDNN`` is used instead.
- This avoids silently inheriting a threshold calibrated for a different model.

Embedding compatibility note
-----------------------------
This service now uses SpeechBrain ECAPA-TDNN embeddings (192-d).
Voice profiles enrolled with the previous CAM++ model (512-d) or Resemblyzer
(256-d) will produce a dimension-mismatch warning and be skipped — they will
NOT cause a crash. Users must re-enroll (re-record) their voice profiles after
upgrading.
"""
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from services.embedding import (
    extract_embedding, best_match_similarity, average_embeddings, EMBEDDING_DIM,
    vad_extract_speaker_embedding,
)
from utils.audio_utils import load_audio

logger = logging.getLogger(__name__)

# One-time warning flag to avoid log spam for stale profiles
_stale_profile_warning_logged = False

# Resemblyzer default threshold — used to detect "unset / legacy" thresholds
_LEGACY_THRESHOLD = 0.75


def _check_embedding_dim(stored_embeddings: list, profile_label: str) -> bool:
    """
    Return True if the stored embeddings have the expected dimension.
    Log a one-time warning if they appear to be from an older model.
    """
    global _stale_profile_warning_logged
    if not stored_embeddings:
        return False
    first = stored_embeddings[0]
    stored_dim = len(first) if hasattr(first, "__len__") else 0
    if stored_dim != EMBEDDING_DIM:
        if not _stale_profile_warning_logged:
            logger.warning(
                f"[Identify] Profile '{profile_label}' has {stored_dim}-d embeddings, "
                f"but the current model (SpeechBrain ECAPA-TDNN) produces {EMBEDDING_DIM}-d embeddings. "
                "This profile was enrolled with an older model and is incompatible. "
                "Please re-enroll all voice profiles to restore speaker identification. "
                "(This warning is shown once.)"
            )
            _stale_profile_warning_logged = True
        return False
    return True


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    """Return L2-normalised vector; return zeros on zero-norm input."""
    norm = np.linalg.norm(vec)
    if norm < 1e-10:
        return vec
    return vec / norm


def _is_multi_speaker_label(label: str) -> bool:
    """
    Return True if the speaker label represents multiple speakers and should
    be excluded from ECAPA embedding / centroid computation.

    Patterns detected:
    - Contains '+' separator (e.g. 'SPEAKER_00+SPEAKER_01')
    - Contains 'Multiple Speaker' (case-insensitive)
    - Starts with '[' and ends with ']' (e.g. '[Multiple Speaker]')
    """
    if not label:
        return False
    if '+' in label:
        return True
    if 'multiple speaker' in label.lower():
        return True
    if label.startswith('[') and label.endswith(']'):
        return True
    return False



def _average_embeddings(embeddings: List[np.ndarray]) -> Optional[np.ndarray]:
    """
    Compute the L2-normalised mean of a list of embeddings.
    Returns None if the list is empty.
    """
    if not embeddings:
        return None
    stacked = np.stack(embeddings, axis=0)          # (N, dim)
    mean_emb = np.mean(stacked, axis=0)             # (dim,)
    return _l2_normalize(mean_emb)



def _compute_profile_centroid(
    stored_embeddings: list,
    expected_dim: int = EMBEDDING_DIM,
) -> Tuple[Optional[np.ndarray], str]:
    """
    Compute the L2-normalised centroid (mean embedding) for a voice profile.

    Filters out any stored embeddings that don't match ``expected_dim`` to guard
    against mixed-dimension profiles.

    Returns
    -------
    (centroid, scoring_method)
        centroid        : np.ndarray of shape (expected_dim,), or None on failure.
        scoring_method  : "centroid" | "single_sample" | "failed"
    """
    if not stored_embeddings:
        return None, "failed"

    valid = []
    for raw in stored_embeddings:
        try:
            arr = np.array(raw, dtype=np.float32)
            if arr.shape == (expected_dim,):
                valid.append(arr)
        except Exception:
            continue

    if not valid:
        return None, "failed"

    if len(valid) == 1:
        # Single sample — no real centroid to compute; just return it
        centroid = _l2_normalize(valid[0])
        return centroid, "single_sample"

    # Multiple samples — compute mean centroid
    centroid = _average_embeddings(valid)
    if centroid is None:
        return None, "failed"
    return centroid, "centroid"


def _get_effective_threshold(
    similarity_threshold: float,
    use_model_default: bool = True,
) -> Tuple[float, str]:
    """
    Determine the effective similarity threshold and log the reason.

    If ``use_model_default=True`` and the provided threshold equals the legacy
    Resemblyzer value (0.75), return the ECAPA-TDNN-specific default instead.

    Returns
    -------
    (effective_threshold, reason_string)
    """
    try:
        from config import settings
        ecapa_default = settings.SPEAKER_SIMILARITY_THRESHOLD_ECAPA_TDNN
    except Exception:
        ecapa_default = 0.75

    if use_model_default and abs(similarity_threshold - _LEGACY_THRESHOLD) < 1e-6:
        # The caller passed the raw legacy default — substitute the ECAPA-TDNN default
        return ecapa_default, (
            f"model-specific ECAPA-TDNN default ({ecapa_default}) "
            f"[legacy threshold {_LEGACY_THRESHOLD} overridden]"
        )

    return similarity_threshold, f"caller-provided ({similarity_threshold})"


def identify_speakers(
    file_path: str,
    diarization_segments: List[Dict[str, Any]],
    voice_profiles: List[Dict],          # from DB, each has "label", "embeddings"
    similarity_threshold: float = 0.75,
    use_model_default_threshold: bool = True,
) -> List[Dict[str, Any]]:
    """
    Perform speaker identification at the SPEAKER level (not per-segment).

    Workflow:
      1. Resolve the effective similarity threshold (ECAPA-TDNN default or caller value).
      2. Group all segments by raw diarization speaker ID.
      3. For each unique diarization speaker, extract ECAPA-TDNN embeddings from
         every non-overlap segment and compute the L2-normalised mean embedding.
      4. For each voice profile compute its centroid (or use single sample).
      5. Match the speaker's mean embedding against each profile's centroid.
      6. Propagate the resolved label to all segments for that speaker.

    Args:
        file_path:              Path to the audio file.
        diarization_segments:   Output of the diarization service.
        voice_profiles:         Voice profile dicts from the DB.
                                Each must have "label" and "embeddings" keys.
        similarity_threshold:   Minimum cosine similarity to accept a match.
                                If this equals the legacy Resemblyzer default
                                (0.75) AND use_model_default_threshold=True,
                                the ECAPA-TDNN default is substituted automatically.
        use_model_default_threshold:
                                When True (default), auto-substitute a better
                                threshold when the legacy value is detected.

    Returns:
        Enriched segments:
        [{
            "start", "end", "speaker" (raw diarization id),
            "speaker_label" (human name or "Speaker N"),
            "speaker_profile_id" (or None),
            "is_overlap", "similarity",
            "scoring_method"  (new: "centroid" | "single_sample" | "no_match")
        }]
    """
    # ── Step 0: Resolve effective threshold ──────────────────────────────────
    effective_threshold, threshold_reason = _get_effective_threshold(
        similarity_threshold, use_model_default=use_model_default_threshold
    )
    logger.info(
        f"[Identify] Similarity threshold: {effective_threshold:.4f} "
        f"(reason: {threshold_reason})"
    )

    audio, sr = load_audio(file_path, target_sr=16000)

    # ── Step 1: Group segments by raw diarization speaker ID ─────────────────
    speaker_to_segs: Dict[str, List[Dict]] = {}
    for seg in diarization_segments:
        if seg.get("is_overlap"):
            continue
        raw_id = seg["speaker"]
        speaker_to_segs.setdefault(raw_id, []).append(seg)

    # ── Step 2: Build a mean ECAPA-TDNN embedding per diarization speaker ────────
    speaker_to_embedding: Dict[str, Optional[np.ndarray]] = {}
    for raw_id, segs in speaker_to_segs.items():
        seg_embeddings = []
        for seg in segs:
            start = seg["start"]
            end = seg["end"]
            s_idx = int(start * sr)
            e_idx = int(end * sr)
            seg_audio = audio[s_idx:e_idx]
            if len(seg_audio) < sr * 0.5:
                # Too short for even a basic embedding — skip cheaply before VAD
                continue
            emb = vad_extract_speaker_embedding(seg_audio, sr=sr)
            if emb is not None:
                seg_embeddings.append(emb)
        mean_emb = _average_embeddings(seg_embeddings) if seg_embeddings else None
        speaker_to_embedding[raw_id] = mean_emb
        logger.info(
            f"[Identify] Speaker '{raw_id}': built mean embedding from "
            f"{len(seg_embeddings)}/{len(segs)} segments "
            f"({'OK' if mean_emb is not None else 'FAILED — too short'})"
        )

    # ── Step 3: Pre-compute profile centroids ────────────────────────────────
    # This is done once per profile, outside the speaker loop, so it's O(P)
    # rather than O(P * S).
    profile_centroids: Dict[str, Tuple[Optional[np.ndarray], str]] = {}
    for profile in voice_profiles:
        label = profile.get("label", "?")
        stored = profile.get("embeddings", [])
        if not stored:
            logger.warning(f"[Identify] Profile '{label}' has no embeddings — skipping")
            profile_centroids[label] = (None, "failed")
            continue
        if not _check_embedding_dim(stored, label):
            profile_centroids[label] = (None, "failed")
            continue
        centroid, method = _compute_profile_centroid(stored)
        profile_centroids[label] = (centroid, method)
        logger.info(
            f"[Identify] Profile '{label}': "
            f"{len(stored)} stored embeddings → scoring method = {method}"
        )

    # ── Step 4: Match each unique speaker against voice profiles ─────────────
    sorted_speakers = sorted(speaker_to_segs.keys())
    generic_counter_map: Dict[str, int] = {
        spk: idx + 1 for idx, spk in enumerate(sorted_speakers)
    }

    # ── Step 4: Match speakers against voice profiles (Greedy Global Matching) ──
    # Result: raw diarization id → (human_label, profile_id, similarity, scoring_method)
    speaker_resolution: Dict[str, tuple] = {}

    # 1. Compute similarity for all speaker-profile pairs
    candidates = []

    for raw_id, mean_emb in speaker_to_embedding.items():
        if mean_emb is None:
            logger.warning(f"[Identify] Speaker '{raw_id}' has no usable embedding — generic label.")
            continue
        if not voice_profiles:
            continue

        for profile in voice_profiles:
            label = profile.get("label", "?")
            centroid, method = profile_centroids.get(label, (None, "failed"))
            profile_id = str(profile.get("_id", profile.get("id", "")))

            if centroid is None:
                stored = profile.get("embeddings", [])
                if stored and _check_embedding_dim(stored, label):
                    sim = best_match_similarity(mean_emb, stored)
                    used_method = "sample_max_fallback"
                else:
                    continue
            else:
                from services.embedding import cosine_similarity
                sim = cosine_similarity(mean_emb, centroid)
                used_method = method

            logger.info(
                f"[Identify] Speaker '{raw_id}' vs profile '{label}': "
                f"similarity={sim:.4f} (method={used_method}, "
                f"threshold={effective_threshold:.4f})"
            )

            candidates.append({
                "similarity": sim,
                "raw_id": raw_id,
                "label": label,
                "profile_id": profile_id,
                "method": used_method,
            })

    # 2. Sort candidates in descending order of similarity
    candidates.sort(key=lambda x: x["similarity"], reverse=True)

    # 3. Greedy selection of non-overlapping matches above threshold
    resolved_speakers = set()
    resolved_profiles = set()

    for cand in candidates:
        sim = cand["similarity"]
        raw_id = cand["raw_id"]
        label = cand["label"]
        profile_id = cand["profile_id"]
        method = cand["method"]

        # Once a speaker or profile is matched, skip other options for them
        if raw_id in resolved_speakers:
            continue
        profile_key = profile_id or label
        if profile_key in resolved_profiles:
            continue

        # Match check
        if sim >= effective_threshold:
            speaker_resolution[raw_id] = (label, profile_id, round(sim, 4), method)
            resolved_speakers.add(raw_id)
            resolved_profiles.add(profile_key)
            logger.info(
                f"[Identify] Speaker '{raw_id}' → MATCHED '{label}' via Greedy Matching "
                f"(sim={sim:.4f}, method={method})"
            )

    # 4. Resolve remaining speakers to generic labels
    for raw_id in speaker_to_segs.keys():
        if raw_id not in resolved_speakers:
            generic_n = generic_counter_map.get(raw_id, 1)
            resolved_label = f"Speaker {generic_n}"

            # Find the best similarity this speaker had (even if below threshold/skipped)
            best_sim = 0.0
            for cand in candidates:
                if cand["raw_id"] == raw_id:
                    best_sim = max(best_sim, cand["similarity"])

            speaker_resolution[raw_id] = (
                resolved_label, None, round(best_sim, 4), "no_match"
            )
            logger.info(
                f"[Identify] Speaker '{raw_id}' → NO MATCH (best_sim={best_sim:.4f}) "
                f"→ assigned '{resolved_label}'"
            )


    # ── Step 5: Build enriched segment list ──────────────────────────────────
    all_raw_ids = {seg["speaker"] for seg in diarization_segments}
    for raw_id in all_raw_ids:
        if raw_id not in speaker_resolution:
            generic_n = generic_counter_map.get(raw_id, len(generic_counter_map) + 1)
            speaker_resolution[raw_id] = (f"Speaker {generic_n}", None, 0.0, "no_match")

    def _resolve_overlap_regions(
        regions: List[Dict], resolution: Dict
    ) -> List[Dict]:
        """Translate raw speaker IDs in overlap_regions to resolved labels."""
        resolved = []
        for region in regions:
            speakers = [
                resolution.get(sp, (sp, None, 0.0, "no_match"))[0]
                for sp in region.get("speakers", [])
            ]
            resolved.append({
                "start": region["start"],
                "end": region["end"],
                "speakers": sorted(set(speakers)),
            })
        return resolved

    identified = []
    for seg in diarization_segments:
        raw_id = seg["speaker"]
        label, profile_id, sim, scoring_method = speaker_resolution.get(
            raw_id, (f"Speaker ?", None, 0.0, "no_match")
        )
        resolved_regions = _resolve_overlap_regions(
            seg.get("overlap_regions", []), speaker_resolution
        )
        identified.append({
            **seg,
            "speaker_label": label,
            "speaker_profile_id": profile_id,
            "similarity": sim,
            "scoring_method": scoring_method,
            "overlap_regions": resolved_regions,
        })

    return identified


def merge_transcript_with_speakers(
    transcript_segments: List[Dict],
    speaker_segments: List[Dict],
) -> List[Dict]:
    """
    Assign speaker labels to transcript segments by temporal overlap.
    Each transcript segment gets the speaker whose diarization window overlaps most.
    """
    result = []
    for t_seg in transcript_segments:
        t_start = t_seg["start"]
        t_end = t_seg["end"]
        best_overlap = 0.0
        best_speaker_label = t_seg.get("speaker", "Speaker 1")
        best_profile_id = None
        is_overlap = False
        overlap_regions: List[Dict] = []

        for s_seg in speaker_segments:
            overlap_start = max(t_start, s_seg["start"])
            overlap_end = min(t_end, s_seg["end"])
            overlap = max(0.0, overlap_end - overlap_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker_label = s_seg.get("speaker_label") or s_seg.get("speaker") or best_speaker_label
                best_profile_id = s_seg.get("speaker_profile_id")
                is_overlap = s_seg.get("is_overlap", False)
                overlap_regions = s_seg.get("overlap_regions", [])

        result.append({
            **t_seg,
            "speaker_label": best_speaker_label,
            "speaker_profile_id": best_profile_id,
            "is_overlap": is_overlap,
            "overlap_regions": overlap_regions,
        })

    return result


def refine_transcript_speakers_with_ecapa(
    file_path: str,
    speaker_segments: List[Dict[str, Any]],
    voice_profiles: List[Dict],
    similarity_threshold: float = 0.75,
    use_model_default_threshold: bool = True,
) -> List[Dict[str, Any]]:
    """
    Perform a high-confidence ECAPA segment-level refinement pass on the final re-segmented transcript segments.
    """
    if not speaker_segments:
        return speaker_segments

    # Resolve speaker refinement margin
    try:
        from config import settings
        speaker_refinement_margin = settings.speaker_refinement_margin
    except Exception:
        speaker_refinement_margin = 0.10

    # 1. Resolve effective threshold (for logging Considered but Rejected)
    effective_threshold, _ = _get_effective_threshold(
        similarity_threshold, use_model_default=use_model_default_threshold
    )

    # 2. Pre-compute profile centroids
    profile_centroids: Dict[str, Tuple[Optional[np.ndarray], str]] = {}
    if voice_profiles:
        for profile in voice_profiles:
            label = profile.get("label", "?")
            stored = profile.get("embeddings", [])
            if not stored or not _check_embedding_dim(stored, label):
                profile_centroids[label] = (None, "failed")
                continue
            centroid, method = _compute_profile_centroid(stored)
            profile_centroids[label] = (centroid, method)

    # 3. Load audio once
    try:
        audio, sr = load_audio(file_path, target_sr=16000)
    except Exception as e:
        logger.error(f"[Refine] Failed to load audio for override pass: {e}")
        return speaker_segments

    # 4. Group all transcript segments by assigned speaker and compute conversation centroids
    speaker_to_embs: Dict[str, List[np.ndarray]] = {}
    speaker_key_to_meta: Dict[str, Tuple[str, Optional[str]]] = {}
    segment_embeddings: Dict[int, np.ndarray] = {}

    logger.info("[Refine] Extracting segment embeddings for conversation centroids...")
    for i, seg in enumerate(speaker_segments):
        if seg.get("is_overlap"):
            continue
        spk_label = seg.get("speaker_label") or seg.get("speaker") or "Speaker 1"
        if _is_multi_speaker_label(spk_label):
            logger.debug(f"[Refine] Skipping multi-speaker segment for centroid: '{spk_label}'")
            continue
        start = seg["start"]
        end = seg["end"]
        s_idx = int(start * sr)
        e_idx = int(end * sr)
        seg_audio = audio[s_idx:e_idx]

        segment_emb = vad_extract_speaker_embedding(seg_audio, sr=sr)
        if segment_emb is None:
            segment_emb = extract_embedding(seg_audio, sr=sr)

        if segment_emb is not None:
            segment_embeddings[i] = segment_emb
            spk_label = seg.get("speaker_label") or seg.get("speaker") or "Speaker 1"
            spk_profile_id = seg.get("speaker_profile_id")
            spk_key = spk_profile_id if spk_profile_id else spk_label

            speaker_to_embs.setdefault(spk_key, []).append(segment_emb)
            if spk_key not in speaker_key_to_meta:
                speaker_key_to_meta[spk_key] = (spk_label, spk_profile_id)

    conversation_centroids: Dict[str, np.ndarray] = {}
    for spk_key, embs in speaker_to_embs.items():
        centroid = _average_embeddings(embs)
        if centroid is not None:
            conversation_centroids[spk_key] = centroid
            logger.info(f"[Refine] Computed conversation centroid for '{spk_key}' using {len(embs)} segments.")

    # 5. Process each segment
    logger.info(f"[Refine] Processing {len(speaker_segments)} segments...")
    for i, seg in enumerate(speaker_segments):
        if seg.get("is_overlap"):
            continue

        original_label = seg.get("speaker_label") or seg.get("speaker") or "Speaker 1"
        if _is_multi_speaker_label(original_label):
            logger.debug(f"[Refine] Skipping multi-speaker segment for override: '{original_label}'")
            continue

        segment_emb = segment_embeddings.get(i)
        if segment_emb is None:
            continue

        start = seg["start"]
        end = seg["end"]

        # Compare against all enrolled profiles and conversation centroids
        best_profile_sim = -1.0
        best_profile_label = None
        best_profile_id = None
        match_source = "none"

        original_profile_id = seg.get("speaker_profile_id")
        original_spk_key = original_profile_id if original_profile_id else original_label

        # Calculate original_label_sim against original representation
        original_label_sim = 1.0
        found_original_sim = False

        # Try to find original similarity from saved profiles
        if voice_profiles:
            for profile in voice_profiles:
                label = profile.get("label", "?")
                centroid, method = profile_centroids.get(label, (None, "failed"))
                profile_id = str(profile.get("_id", profile.get("id", "")))

                if (label == original_label) or (original_profile_id and profile_id == original_profile_id):
                    if centroid is None:
                        stored = profile.get("embeddings", [])
                        if stored and _check_embedding_dim(stored, label):
                            original_label_sim = best_match_similarity(segment_emb, stored)
                            found_original_sim = True
                    else:
                        from services.embedding import cosine_similarity
                        original_label_sim = cosine_similarity(segment_emb, centroid)
                        found_original_sim = True
                    break

        # Fallback to conversation centroid of the original speaker if not matching profiles
        if not found_original_sim:
            if original_spk_key in conversation_centroids:
                from services.embedding import cosine_similarity
                original_label_sim = cosine_similarity(segment_emb, conversation_centroids[original_spk_key])
                found_original_sim = True

        # 1. Compare against saved profiles
        if voice_profiles:
            for profile in voice_profiles:
                label = profile.get("label", "?")
                centroid, method = profile_centroids.get(label, (None, "failed"))
                profile_id = str(profile.get("_id", profile.get("id", "")))

                if centroid is None:
                    stored = profile.get("embeddings", [])
                    if stored and _check_embedding_dim(stored, label):
                        sim_val = best_match_similarity(segment_emb, stored)
                    else:
                        continue
                else:
                    from services.embedding import cosine_similarity
                    sim_val = cosine_similarity(segment_emb, centroid)

                if sim_val > best_profile_sim:
                    best_profile_sim = sim_val
                    best_profile_label = label
                    best_profile_id = profile_id
                    match_source = "saved_profile"

        # 2. Compare against conversation speaker centroids
        for spk_key, centroid in conversation_centroids.items():
            from services.embedding import cosine_similarity
            sim_val = cosine_similarity(segment_emb, centroid)

            if sim_val > best_profile_sim:
                best_profile_sim = sim_val
                lbl, pid = speaker_key_to_meta.get(spk_key, (spk_key, None))
                best_profile_label = lbl
                best_profile_id = pid
                match_source = "conversation_speaker"

        if best_profile_label is None:
            continue

        is_different = (original_profile_id != best_profile_id) or (original_label != best_profile_label)
       
        if is_different:
            if (
                best_profile_sim >= 0.82
                or (
                    best_profile_sim > 0.20
                    and (best_profile_sim - original_label_sim) > speaker_refinement_margin
                )
            ):
                # Update only speaker_label, speaker_profile_id, and similarity
                seg["speaker_label"] = best_profile_label
                seg["speaker_profile_id"] = best_profile_id
                seg["similarity"] = round(best_profile_sim, 4)

                logger.info(
                    f"[Identify] Override Applied: Time: {start:.2f} - {end:.2f} | {original_label} -> {best_profile_label} | Similarity: {best_profile_sim:.4f}"
                )
            else:
                logger.info(
                    f"[Identify] Rejected: Time: {start:.2f} - {end:.2f} | {original_label} -> {best_profile_label} | Similarity: {best_profile_sim:.4f} (similarity < 0.82 and margin check <= {speaker_refinement_margin:.2f})"
                )

    return speaker_segments
