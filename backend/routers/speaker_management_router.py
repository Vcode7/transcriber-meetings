"""
Speaker Management Router — Speaker Tab Backend

Provides isolated CRUD endpoints for the Speaker Tab in AIChatPanel.
Does NOT touch the diarization/transcription pipeline.

Segment addressing:
  Each transcript segment receives a stable `seg_id` (UUID string) on the
  first call to GET /speakers/{recording_id}. This field is persisted back
  to the database so subsequent calls use the same IDs.

Endpoints:
  GET    /speakers/{recording_id}                   - list speakers + segments
  POST   /speakers/{recording_id}/move-segment      - reassign segment speaker
  DELETE /speakers/{recording_id}/segment/{seg_id}  - remove a segment
  DELETE /speakers/{recording_id}/speaker           - remove speaker + segments
  POST   /speakers/{recording_id}/dissolve-preview  - calculate reassignment proposals
  POST   /speakers/{recording_id}/dissolve-confirm  - apply dissolve assignments
"""
import logging
import uuid
import os
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import text

from database import get_db, to_json, from_json
from routers.auth import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/speakers", tags=["speakers"])


# ── Pydantic models ────────────────────────────────────────────────────────────

class MoveSegmentBody(BaseModel):
    seg_id: str
    new_speaker_label: str


class DeleteSpeakerBody(BaseModel):
    speaker_label: str
    action: str  # "delete_segments"


class DissolvePreviewBody(BaseModel):
    speaker_label: str


class DissolveAssignment(BaseModel):
    seg_id: str
    new_speaker_label: str


class DissolveConfirmBody(BaseModel):
    speaker_label: str
    assignments: List[DissolveAssignment]


# ── Internal helpers ───────────────────────────────────────────────────────────

async def _get_recording_for_user(recording_id: str, user_id: str, db) -> dict:
    """Fetch the recording row and verify ownership. Raises 404/403."""
    r = await db.execute(
        text("SELECT * FROM recordings WHERE id = :rid"),
        {"rid": recording_id},
    )
    row = r.mappings().fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Recording not found")
    if str(row["user_id"]) != str(user_id):
        raise HTTPException(status_code=403, detail="Access denied")
    return dict(row)


def _ensure_seg_ids(segments: List[Dict[str, Any]]) -> bool:
    """
    Add a `seg_id` UUID to every segment that is missing one.
    Returns True if any segment was modified.
    """
    modified = False
    for seg in segments:
        if not seg.get("seg_id"):
            seg["seg_id"] = str(uuid.uuid4())
            modified = True
    return modified


def _group_by_speaker(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Group segments by speaker_label.
    Returns a list of speaker objects ordered by first appearance.
    """
    order: List[str] = []
    groups: Dict[str, List[Dict]] = {}

    for seg in segments:
        label = seg.get("speaker_label") or "Unknown"
        if label not in groups:
            order.append(label)
            groups[label] = []
        groups[label].append({
            "seg_id": seg.get("seg_id", ""),
            "text": seg.get("text", ""),
            "start": seg.get("start", 0.0),
            "end": seg.get("end", 0.0),
            "speaker_label": label,
            "similarity": seg.get("similarity"),
            "scoring_method": seg.get("scoring_method"),
            "is_overlap": seg.get("is_overlap", False),
        })

    return [
        {"speaker_label": lbl, "segments": groups[lbl]}
        for lbl in order
    ]


def _cosine_sim_numpy(a: List[float], b: List[float]) -> float:
    """Pure-numpy cosine similarity."""
    import numpy as np
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def _compute_speaker_embedding_from_audio(
    file_path: str,
    segments: List[Dict[str, Any]],
) -> Optional[List[float]]:
    """
    Extract a mean ECAPA-TDNN embedding for a speaker using their segments.
    Returns a Python list of floats, or None on failure.
    """
    try:
        from utils.audio_utils import load_audio
        from services.embedding import vad_extract_speaker_embedding, average_embeddings

        audio, sr = load_audio(file_path, target_sr=16000)
        embeddings = []
        for seg in segments:
            start = seg.get("start", 0.0)
            end = seg.get("end", start)
            if seg.get("is_overlap"):
                continue
            duration = end - start
            if duration < 0.5:
                continue
            s_idx = int(start * sr)
            e_idx = int(end * sr)
            seg_audio = audio[s_idx:e_idx]
            emb = vad_extract_speaker_embedding(seg_audio, sr=sr)
            if emb is not None:
                embeddings.append(emb)

        if not embeddings:
            return None
        mean_emb = average_embeddings(embeddings)
        return mean_emb.tolist()
    except Exception as e:
        logger.warning(f"[SpeakerMgr] _compute_speaker_embedding_from_audio failed: {e}")
        return None


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.get("/{recording_id}")
async def get_speakers(
    recording_id: str,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_db),
):
    """
    Return all speakers in the meeting together with their transcript segments.
    Lazily stamps `seg_id` onto any segments that are missing one.
    """
    user_id = current_user["id"]
    rec = await _get_recording_for_user(recording_id, user_id, db)

    segments: List[Dict] = from_json(rec.get("transcript"), default=[])
    if not isinstance(segments, list):
        segments = []

    # Stamp seg_id where missing, persist if needed
    if _ensure_seg_ids(segments):
        await db.execute(
            text("UPDATE recordings SET transcript = :t WHERE id = :rid"),
            {"t": to_json(segments), "rid": recording_id},
        )
        await db.commit()

    speakers = _group_by_speaker(segments)
    return {
        "recording_id": recording_id,
        "speakers": speakers,
        "total_segments": len(segments),
    }


@router.post("/{recording_id}/move-segment")
async def move_segment(
    recording_id: str,
    body: MoveSegmentBody,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_db),
):
    """Move a segment from its current speaker to a new speaker label."""
    user_id = current_user["id"]
    rec = await _get_recording_for_user(recording_id, user_id, db)

    segments: List[Dict] = from_json(rec.get("transcript"), default=[])
    if not isinstance(segments, list):
        segments = []

    _ensure_seg_ids(segments)

    target = next((s for s in segments if s.get("seg_id") == body.seg_id), None)
    if not target:
        raise HTTPException(status_code=404, detail="Segment not found")

    old_label = target.get("speaker_label", "")
    target["speaker_label"] = body.new_speaker_label
    # Also update words speaker_label if present
    for word in target.get("words", []):
        if isinstance(word, dict):
            word["speaker_label"] = body.new_speaker_label

    # Update speakers_detected list (add new speaker if not present)
    speakers_detected: List[str] = from_json(rec.get("speakers_detected"), default=[])
    if not isinstance(speakers_detected, list):
        speakers_detected = []
    if body.new_speaker_label not in speakers_detected:
        speakers_detected.append(body.new_speaker_label)

    # Remove old speaker if no segments remain for it
    remaining_old = [s for s in segments if s.get("speaker_label") == old_label]
    if not remaining_old and old_label in speakers_detected:
        speakers_detected.remove(old_label)

    await db.execute(
        text(
            "UPDATE recordings SET transcript = :t, speakers_detected = :sd WHERE id = :rid"
        ),
        {"t": to_json(segments), "sd": to_json(speakers_detected), "rid": recording_id},
    )
    await db.commit()

    return {"success": True, "seg_id": body.seg_id, "new_speaker_label": body.new_speaker_label}


@router.delete("/{recording_id}/segment/{seg_id}")
async def delete_segment(
    recording_id: str,
    seg_id: str,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_db),
):
    """Permanently delete a single transcript segment."""
    user_id = current_user["id"]
    rec = await _get_recording_for_user(recording_id, user_id, db)

    segments: List[Dict] = from_json(rec.get("transcript"), default=[])
    if not isinstance(segments, list):
        segments = []

    _ensure_seg_ids(segments)

    removed = next((s for s in segments if s.get("seg_id") == seg_id), None)
    if not removed:
        raise HTTPException(status_code=404, detail="Segment not found")

    old_label = removed.get("speaker_label", "")
    segments = [s for s in segments if s.get("seg_id") != seg_id]

    # Remove speaker from speakers_detected if no segments remain
    speakers_detected: List[str] = from_json(rec.get("speakers_detected"), default=[])
    if not isinstance(speakers_detected, list):
        speakers_detected = []
    remaining = [s for s in segments if s.get("speaker_label") == old_label]
    if not remaining and old_label in speakers_detected:
        speakers_detected.remove(old_label)

    await db.execute(
        text(
            "UPDATE recordings SET transcript = :t, speakers_detected = :sd WHERE id = :rid"
        ),
        {"t": to_json(segments), "sd": to_json(speakers_detected), "rid": recording_id},
    )
    await db.commit()

    return {"success": True, "deleted_seg_id": seg_id}


@router.delete("/{recording_id}/speaker")
async def delete_speaker(
    recording_id: str,
    body: DeleteSpeakerBody,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_db),
):
    """
    Delete a speaker. action='delete_segments' removes the speaker and all
    their segments from the transcript.
    """
    user_id = current_user["id"]
    rec = await _get_recording_for_user(recording_id, user_id, db)

    if body.action != "delete_segments":
        raise HTTPException(status_code=400, detail="action must be 'delete_segments'")

    segments: List[Dict] = from_json(rec.get("transcript"), default=[])
    if not isinstance(segments, list):
        segments = []

    deleted_count = sum(1 for s in segments if s.get("speaker_label") == body.speaker_label)
    segments = [s for s in segments if s.get("speaker_label") != body.speaker_label]

    speakers_detected: List[str] = from_json(rec.get("speakers_detected"), default=[])
    if not isinstance(speakers_detected, list):
        speakers_detected = []
    speakers_detected = [s for s in speakers_detected if s != body.speaker_label]

    await db.execute(
        text(
            "UPDATE recordings SET transcript = :t, speakers_detected = :sd WHERE id = :rid"
        ),
        {"t": to_json(segments), "sd": to_json(speakers_detected), "rid": recording_id},
    )
    await db.commit()

    return {
        "success": True,
        "speaker_label": body.speaker_label,
        "deleted_segments": deleted_count,
    }


@router.post("/{recording_id}/dissolve-preview")
async def dissolve_preview(
    recording_id: str,
    body: DissolvePreviewBody,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_db),
):
    """
    Calculate proposed speaker reassignments for dissolving a speaker.

    For each segment belonging to the speaker being dissolved:
      1. Try to extract an ECAPA-TDNN embedding from the audio file.
      2. Compare against mean embeddings of remaining speakers.
      3. Select the best-scoring remaining speaker.
      4. Fall back to stored similarity field if audio is unavailable.

    Returns a list of {seg_id, text, start, proposed_speaker, similarity, method}
    for the frontend to display in a confirmation dialog.
    """
    user_id = current_user["id"]
    rec = await _get_recording_for_user(recording_id, user_id, db)

    segments: List[Dict] = from_json(rec.get("transcript"), default=[])
    if not isinstance(segments, list):
        segments = []

    _ensure_seg_ids(segments)

    dissolve_label = body.speaker_label
    dissolve_segs = [s for s in segments if s.get("speaker_label") == dissolve_label]
    if not dissolve_segs:
        raise HTTPException(status_code=404, detail=f"Speaker '{dissolve_label}' not found")

    remaining_labels = list({
        s.get("speaker_label")
        for s in segments
        if s.get("speaker_label") != dissolve_label
    })

    if not remaining_labels:
        raise HTTPException(
            status_code=400,
            detail="Cannot dissolve: no other speakers to reassign to",
        )

    # --- Try embedding-based reassignment ---
    file_path: str = rec.get("file_path", "")
    audio_available = bool(file_path and os.path.isfile(file_path))

    # Build mean embeddings for each remaining speaker (from audio if available)
    remaining_embeddings: Dict[str, Optional[List[float]]] = {}
    if audio_available:
        for lbl in remaining_labels:
            lbl_segs = [s for s in segments if s.get("speaker_label") == lbl]
            emb = _compute_speaker_embedding_from_audio(file_path, lbl_segs)
            remaining_embeddings[lbl] = emb

    # Compute proposals
    proposals = []
    for seg in dissolve_segs:
        proposed_label: Optional[str] = None
        proposed_sim: Optional[float] = None
        method = "stored_similarity"

        if audio_available:
            # Try to extract embedding for this segment
            try:
                from utils.audio_utils import load_audio
                from services.embedding import vad_extract_speaker_embedding

                audio, sr = load_audio(file_path, target_sr=16000)
                start = seg.get("start", 0.0)
                end = seg.get("end", start)
                if not seg.get("is_overlap") and (end - start) >= 0.4:
                    s_idx = int(start * sr)
                    e_idx = int(end * sr)
                    seg_audio = audio[s_idx:e_idx]
                    seg_emb = vad_extract_speaker_embedding(seg_audio, sr=sr)

                    if seg_emb is not None:
                        best_sim = -1.0
                        best_lbl = None
                        for lbl, rem_emb in remaining_embeddings.items():
                            if rem_emb is None:
                                continue
                            sim = _cosine_sim_numpy(seg_emb.tolist(), rem_emb)
                            if sim > best_sim:
                                best_sim = sim
                                best_lbl = lbl
                        if best_lbl is not None:
                            proposed_label = best_lbl
                            proposed_sim = round(best_sim, 4)
                            method = "embedding"
            except Exception as e:
                logger.debug(f"[SpeakerMgr] dissolve embedding extraction failed: {e}")

        if proposed_label is None:
            # Fallback: assign to the speaker with the highest average stored similarity
            best_stored_sim = -1.0
            best_stored_lbl: Optional[str] = None
            for lbl in remaining_labels:
                lbl_sims = [
                    s.get("similarity", 0.0) or 0.0
                    for s in segments
                    if s.get("speaker_label") == lbl and not s.get("is_overlap")
                ]
                avg_sim = sum(lbl_sims) / len(lbl_sims) if lbl_sims else 0.0
                if avg_sim > best_stored_sim:
                    best_stored_sim = avg_sim
                    best_stored_lbl = lbl
            proposed_label = best_stored_lbl or remaining_labels[0]
            proposed_sim = round(float(seg.get("similarity") or 0.0), 4)

        proposals.append({
            "seg_id": seg.get("seg_id", ""),
            "text": seg.get("text", ""),
            "start": seg.get("start", 0.0),
            "end": seg.get("end", 0.0),
            "proposed_speaker": proposed_label,
            "similarity": proposed_sim,
            "method": method,
        })

    return {
        "dissolving_speaker": dissolve_label,
        "remaining_speakers": remaining_labels,
        "audio_available": audio_available,
        "proposals": proposals,
    }


@router.post("/{recording_id}/dissolve-confirm")
async def dissolve_confirm(
    recording_id: str,
    body: DissolveConfirmBody,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_db),
):
    """
    Apply the approved dissolve assignments:
      - Reassign each seg_id to its new speaker.
      - Remove the dissolved speaker from speakers_detected.
      - Persist to DB.
    """
    user_id = current_user["id"]
    rec = await _get_recording_for_user(recording_id, user_id, db)

    segments: List[Dict] = from_json(rec.get("transcript"), default=[])
    if not isinstance(segments, list):
        segments = []

    _ensure_seg_ids(segments)

    assignment_map: Dict[str, str] = {a.seg_id: a.new_speaker_label for a in body.assignments}

    new_speakers_added: List[str] = []
    for seg in segments:
        sid = seg.get("seg_id", "")
        if sid in assignment_map:
            new_lbl = assignment_map[sid]
            seg["speaker_label"] = new_lbl
            for word in seg.get("words", []):
                if isinstance(word, dict):
                    word["speaker_label"] = new_lbl
            if new_lbl not in new_speakers_added:
                new_speakers_added.append(new_lbl)

    # Update speakers_detected
    speakers_detected: List[str] = from_json(rec.get("speakers_detected"), default=[])
    if not isinstance(speakers_detected, list):
        speakers_detected = []

    # Remove dissolved speaker
    speakers_detected = [s for s in speakers_detected if s != body.speaker_label]

    # Add any newly-referenced speakers
    for lbl in new_speakers_added:
        if lbl not in speakers_detected:
            speakers_detected.append(lbl)

    # Final safety: ensure no segment still references the dissolved speaker
    remaining_dissolve = [
        s for s in segments if s.get("speaker_label") == body.speaker_label
    ]
    if remaining_dissolve:
        fallback_label = speakers_detected[0] if speakers_detected else "Unknown"
        for seg in remaining_dissolve:
            seg["speaker_label"] = fallback_label

    await db.execute(
        text(
            "UPDATE recordings SET transcript = :t, speakers_detected = :sd WHERE id = :rid"
        ),
        {"t": to_json(segments), "sd": to_json(speakers_detected), "rid": recording_id},
    )
    await db.commit()

    return {
        "success": True,
        "dissolved_speaker": body.speaker_label,
        "reassigned_count": len(body.assignments),
    }
