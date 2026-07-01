"""
Speaker identification service.

For each diarization segment, extracts a resemblyzer embedding and compares
against stored voice profiles to assign a human-readable label.
"""
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from services.embedding import extract_embedding, best_match_similarity
from utils.audio_utils import load_audio

logger = logging.getLogger(__name__)


def identify_speakers(
    file_path: str,
    diarization_segments: List[Dict[str, Any]],
    voice_profiles: List[Dict],          # from MongoDB, each has "label", "embeddings"
    similarity_threshold: float = 0.75,
) -> List[Dict[str, Any]]:
    """
    For each diarization segment, extract embedding and match to voice profiles.

    Returns enriched segments:
    [{
        "start", "end", "speaker" (raw diarization id),
        "speaker_label" (human name or "Speaker N"),
        "speaker_profile_id" (or None),
        "is_overlap", "similarity"
    }]
    """
    audio, sr = load_audio(file_path, target_sr=16000)
    duration = len(audio) / sr

    # Map diarization speaker IDs → consistent "Speaker N" labels
    diar_id_to_generic: Dict[str, str] = {}
    generic_counter = 1

    identified = []

    for seg in diarization_segments:
        start = seg["start"]
        end = seg["end"]
        raw_speaker = seg["speaker"]
        is_overlap = seg.get("is_overlap", False)

        if is_overlap:
            identified.append({
                **seg,
                "speaker_label": "[Multiple Speakers]",
                "speaker_profile_id": None,
                "similarity": 0.0,
            })
            continue

        # slice audio for this segment
        s_idx = int(start * sr)
        e_idx = int(end * sr)
        seg_audio = audio[s_idx:e_idx]

        if len(seg_audio) < sr * 0.5:
            # too short to embed
            label = _get_generic_label(raw_speaker, diar_id_to_generic, generic_counter)
            if raw_speaker not in diar_id_to_generic:
                diar_id_to_generic[raw_speaker] = label
                generic_counter += 1
            identified.append({
                **seg,
                "speaker_label": label,
                "speaker_profile_id": None,
                "similarity": 0.0,
            })
            continue

        embedding = extract_embedding(seg_audio, sr=sr)

        best_label: Optional[str] = None
        best_profile_id: Optional[str] = None
        best_sim = 0.0

        if embedding is None:
            logger.warning(f"[Identify] Embedding is None for seg {start:.2f}-{end:.2f} (audio len={len(seg_audio)})")
        elif not voice_profiles:
            logger.info("[Identify] No voice profiles stored — all speakers will be generic.")
        else:
            for profile in voice_profiles:
                stored_embeddings = profile.get("embeddings", [])
                if not stored_embeddings:
                    logger.warning(f"[Identify] Profile '{profile.get('label')}' has no embeddings!")
                    continue
                sim = best_match_similarity(embedding, stored_embeddings)
                logger.info(
                    f"[Identify] seg {start:.2f}-{end:.2f} vs '{profile.get('label')}': "
                    f"similarity={sim:.4f} (threshold={similarity_threshold})"
                )
                if sim > best_sim:
                    best_sim = sim
                    best_label = profile["label"]
                    best_profile_id = str(profile.get("_id", ""))

        if best_sim >= similarity_threshold and best_label:
            speaker_label = best_label
            profile_id = best_profile_id
            logger.info(f"[Identify] seg {start:.2f}-{end:.2f} → MATCHED '{speaker_label}' (sim={best_sim:.4f})")
        else:
            # assign generic label, consistent per diarization speaker id
            if raw_speaker not in diar_id_to_generic:
                diar_id_to_generic[raw_speaker] = f"Speaker {generic_counter}"
                generic_counter += 1
            speaker_label = diar_id_to_generic[raw_speaker]
            profile_id = None
            if best_sim > 0:
                logger.info(
                    f"[Identify] seg {start:.2f}-{end:.2f} → NO MATCH (best={best_sim:.4f} < threshold={similarity_threshold}) "
                    f"→ assigned '{speaker_label}'"
                )

        identified.append({
            **seg,
            "speaker_label": speaker_label,
            "speaker_profile_id": profile_id,
            "similarity": round(best_sim, 4),
        })

    return identified


def _get_generic_label(raw_id: str, mapping: Dict[str, str], counter: int) -> str:
    if raw_id in mapping:
        return mapping[raw_id]
    return f"Speaker {counter}"


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
        best_speaker_label = "Unknown"
        best_profile_id = None
        is_overlap = False

        for s_seg in speaker_segments:
            overlap_start = max(t_start, s_seg["start"])
            overlap_end = min(t_end, s_seg["end"])
            overlap = max(0.0, overlap_end - overlap_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker_label = s_seg.get("speaker_label", "Unknown")
                best_profile_id = s_seg.get("speaker_profile_id")
                is_overlap = s_seg.get("is_overlap", False)

        result.append({
            **t_seg,
            "speaker_label": best_speaker_label,
            "speaker_profile_id": best_profile_id,
            "is_overlap": is_overlap,
        })

    return result
