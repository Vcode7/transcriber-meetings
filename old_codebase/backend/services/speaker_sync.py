"""
Speaker Synchronization Service — Unified Speaker Label Rename.

All resolved speaker labels (e.g. "Speaker 1", "John") are now written into
every LLM payload during the pipeline, so the LLM never sees raw Pyannote IDs
(e.g. SPEAKER_00). This means only a single mapping is needed at rename time:

    Label Map  {display_label → new_name}
    Applied to fields that store speaker_label values:
      transcript[].speaker_label
      transcript[].words[].speaker_label
      speakers_detected[]
      speaker_summary dict keys
      final_rom.participants[]
      minutes_of_meeting.participants[]
      summary / short_summary / detailed_summary text
      stage1/stage2/stage3/final_rom speaker fields (now all label-based)

Legacy Track 2 — Raw ID Map  {raw_diarization_id → new_name}
    Kept for backward-compatibility only. Recordings processed before the
    speaker-label unification fix may still have raw Pyannote IDs (SPEAKER_XX)
    stored in ROM stage speaker fields. The raw_id_map derived from scanning
    `transcript[].speaker` covers those cases so old recordings can still be
    renamed correctly.

New recordings no longer need Track 2 because Stage 1 ROM extraction now
reads `speaker_label` instead of `speaker` when building window text.

No alias expansion. No regex. No speaker-number extraction.
Each identifier is matched and replaced exactly as it is stored.
"""
import json
import logging
import re
from typing import Dict, Any, List, Set, Optional
from sqlalchemy import text
from database import to_json, from_json

logger = logging.getLogger(__name__)


# ── Public helpers (kept for backward compatibility with rom_service.py) ───────

def replace_speaker_in_text(text_val: str, mapping: Dict[str, str]) -> str:
    """Replace speaker identifiers in a text string using case-insensitive and format-flexible word matching."""
    if not isinstance(text_val, str) or not text_val or not mapping:
        return text_val

    result = text_val
    for old in sorted(mapping, key=len, reverse=True):
        new = mapping[old]
        if old in result:
            result = result.replace(old, new)
        old_clean = str(old).strip()
        if old_clean:
            # Match variations like SPEAKER_01, Speaker_01, speaker_01, Speaker 01, etc.
            escaped = re.escape(old_clean).replace(r'\_', r'[\s\_]?').replace(r'\ ', r'[\s\_]?')
            pattern = r'\b' + escaped + r'\b'
            try:
                result = re.sub(pattern, new, result, flags=re.IGNORECASE)
            except Exception:
                pass
    return result


def replace_deep_speaker_names(obj: Any, mapping: Dict[str, str]) -> Any:
    """Recursively apply mapping to all strings inside lists and dicts."""
    if not mapping:
        return obj
    if isinstance(obj, str):
        return replace_speaker_in_text(obj, mapping)
    elif isinstance(obj, list):
        return [replace_deep_speaker_names(item, mapping) for item in obj]
    elif isinstance(obj, dict):
        return {k: replace_deep_speaker_names(v, mapping) for k, v in obj.items()}
    return obj


def build_clean_speaker_mappings(speaker_mappings: Dict[str, Any]) -> Dict[str, str]:
    """
    Compatibility shim. Returns exact mappings provided.
    """
    if not speaker_mappings or not isinstance(speaker_mappings, dict):
        return {}
    return {
        str(k).strip(): str(v).strip()
        for k, v in speaker_mappings.items()
        if str(k).strip() and str(v).strip() and str(k).strip() != str(v).strip()
    }


# ── Core Two-Track Helpers ─────────────────────────────────────────────────────

def collect_raw_ids_for_label(
    transcript_list: List[Dict],
    display_label: str,
    target_name: Optional[str] = None,
    min_duration_ratio: float = 0.20,
) -> Set[str]:
    """
    Scan the transcript and collect primary raw diarization IDs for display_label or target_name.
    """
    if not transcript_list or not display_label:
        return set()

    matching_labels = {display_label}
    if target_name:
        matching_labels.add(target_name)

    durations: Dict[str, float] = {}
    for seg in transcript_list:
        if isinstance(seg, dict) and not seg.get("is_overlap"):
            lbl = seg.get("speaker_label")
            if lbl in matching_labels:
                raw_id = str(seg.get("speaker", "") or "").strip()
                if raw_id:
                    start = float(seg.get("start", 0.0))
                    end = float(seg.get("end", start))
                    dur = max(0.0, end - start)
                    durations[raw_id] = durations.get(raw_id, 0.0) + dur

    if not durations:
        return set()

    total_dur = sum(durations.values())
    if total_dur <= 0:
        counts: Dict[str, int] = {}
        for seg in transcript_list:
            if isinstance(seg, dict) and not seg.get("is_overlap"):
                lbl = seg.get("speaker_label")
                if lbl in matching_labels:
                    raw_id = str(seg.get("speaker", "") or "").strip()
                    if raw_id:
                        counts[raw_id] = counts.get(raw_id, 0) + 1
        if counts:
            max_count = max(counts.values())
            return {r for r, c in counts.items() if c >= max_count * min_duration_ratio}
        return set()

    valid_raw_ids = {
        raw_id for raw_id, dur in durations.items()
        if (dur / total_dur) >= min_duration_ratio
    }

    if not valid_raw_ids and durations:
        valid_raw_ids = set(durations.keys())

    return valid_raw_ids


def _map_val(v: Any, mapping: Dict[str, str]) -> Any:
    """Return mapped value if v (or any case/underscore variation of v) is in mapping."""
    if not isinstance(v, str) or not v or not mapping:
        return v
    v_str = str(v).strip()
    if v_str in mapping:
        return mapping[v_str]
    # Flexible case/underscore matching (e.g., Speaker_01 -> SPEAKER_01)
    v_norm = re.sub(r'[\s_]+', '_', v_str).upper()
    for k, target in mapping.items():
        k_norm = re.sub(r'[\s_]+', '_', str(k).strip()).upper()
        if v_norm == k_norm:
            return target
    return v


def _map_list(lst: Any, mapping: Dict[str, str]) -> Any:
    """Map each string element of a list through _map_val."""
    if not isinstance(lst, list):
        return lst
    return [_map_val(x, mapping) for x in lst]


# ── Track 1: Label Rename ──────────────────────────────────────────────────────

def _apply_label_track(
    rec: Dict[str, Any],
    label_map: Dict[str, str],
    raw_id_map: Optional[Dict[str, str]] = None,
    replace_all: bool = False,
) -> Dict[str, Any]:
    """
    Track 1: Apply display-label renames to all speaker_label-based fields.
    """
    if not label_map and not raw_id_map and not replace_all:
        return rec

    updated = dict(rec)
    combined = {**(raw_id_map or {}), **label_map}

    # 1. Transcript speaker_label
    raw_t = updated.get("transcript")
    if raw_t:
        t_list = from_json(raw_t) if isinstance(raw_t, str) else list(raw_t)
        if isinstance(t_list, list):
            for seg in t_list:
                if isinstance(seg, dict):
                    lbl = seg.get("speaker_label")
                    if lbl and lbl in label_map:
                        seg["speaker_label"] = label_map[lbl]
                    for w in (seg.get("words") or []):
                        if isinstance(w, dict):
                            w_lbl = w.get("speaker_label")
                            if w_lbl and w_lbl in label_map:
                                w["speaker_label"] = label_map[w_lbl]
            updated["transcript"] = t_list

    # 2. speakers_detected (mapped using combined map to cover both label and raw ID)
    raw_sd = updated.get("speakers_detected")
    if raw_sd:
        sd = from_json(raw_sd) if isinstance(raw_sd, str) else list(raw_sd)
        if isinstance(sd, list):
            seen: Set[str] = set()
            new_sd = []
            for s in sd:
                mapped = combined.get(str(s), str(s))
                if mapped not in seen and mapped not in ("Unknown", "null", "None"):
                    new_sd.append(mapped)
                    seen.add(mapped)
            # Ensure any labels now in the (updated) transcript are also included
            t = updated.get("transcript")
            if isinstance(t, list):
                for seg in t:
                    if isinstance(seg, dict):
                        lbl = seg.get("speaker_label")
                        if lbl and str(lbl).strip() and lbl not in seen \
                                and lbl not in ("Unknown", "null", "None"):
                            new_sd.append(lbl)
                            seen.add(lbl)
            updated["speakers_detected"] = new_sd

    # 3. speaker_summary — rename dict keys
    raw_ss = updated.get("speaker_summary")
    if raw_ss:
        ss = from_json(raw_ss) if isinstance(raw_ss, str) else dict(raw_ss)
        if isinstance(ss, dict):
            new_ss: Dict[str, Any] = {}
            for k, v in ss.items():
                new_key = combined.get(str(k), str(k))
                if new_key in new_ss:
                    existing = new_ss[new_key]
                    if isinstance(existing, dict) and isinstance(v, dict):
                        existing["summary"] = (
                            existing.get("summary", "") + "\n" + v.get("summary", "")
                        ).strip()
                        existing["key_points"] = list(
                            dict.fromkeys(existing.get("key_points", []) + v.get("key_points", []))
                        )
                        existing["action_items"] = list(
                            dict.fromkeys(existing.get("action_items", []) + v.get("action_items", []))
                        )
                    else:
                        new_ss[new_key] = v
                else:
                    new_ss[new_key] = v
            updated["speaker_summary"] = new_ss

    # 8. Persist canonical speaker_mappings (label → name only)
    if replace_all:
        canonical = dict(label_map)
    else:
        existing_sm = updated.get("speaker_mappings")
        sm = from_json(existing_sm) if isinstance(existing_sm, str) else (existing_sm or {})
        canonical = dict(sm) if isinstance(sm, dict) else {}
        canonical.update(label_map)
    updated["speaker_mappings"] = canonical

    # 4. Final ROM participants & speaker_mappings (speaker_label-based)
    raw_rom = updated.get("rom_data")
    if raw_rom:
        rom = from_json(raw_rom) if isinstance(raw_rom, str) else dict(raw_rom)
        if isinstance(rom, dict):
            final_rom = rom.get("final_rom")
            if isinstance(final_rom, dict):
                parts = final_rom.get("participants")
                if isinstance(parts, list):
                    seen_p: Set[str] = set()
                    new_parts = []
                    for p in parts:
                        mapped = combined.get(p, p)
                        if mapped not in seen_p:
                            new_parts.append(mapped)
                            seen_p.add(mapped)
                    final_rom["participants"] = new_parts
                final_rom["speaker_mappings"] = dict(canonical)
            updated["rom_data"] = rom

    # 5. Text summary fields
    for field in ("summary", "short_summary", "detailed_summary"):
        val = updated.get(field)
        if isinstance(val, str):
            updated[field] = replace_speaker_in_text(val, combined)

    # 6. key_points / action_items (top-level lists — string replacement in content)
    for field in ("key_points", "action_items"):
        val = updated.get(field)
        if val:
            items = from_json(val) if isinstance(val, str) else list(val)
            if isinstance(items, list):
                updated[field] = replace_deep_speaker_names(items, combined)

    return updated


# ── Track 2: Raw ID Rename ─────────────────────────────────────────────────────

def _apply_raw_id_track(rec: Dict[str, Any], raw_id_map: Dict[str, str]) -> Dict[str, Any]:
    """
    Track 2: Apply raw diarization ID renames to all ROM stage speaker fields.
    """
    if not raw_id_map:
        return rec

    updated = dict(rec)

    raw_rom = updated.get("rom_data")
    if raw_rom:
        rom = from_json(raw_rom) if isinstance(raw_rom, str) else dict(raw_rom)
        if isinstance(rom, dict):

            # Stage 1
            stage1 = rom.get("stage1")
            if isinstance(stage1, dict):
                pts = stage1.get("discussion_points")
                if isinstance(pts, list):
                    for pt in pts:
                        if isinstance(pt, dict):
                            if "speaker" in pt:
                                pt["speaker"] = _map_val(pt["speaker"], raw_id_map)
                            pt["speakers"] = _map_list(pt.get("speakers", []), raw_id_map)
                            if "action_owner" in pt:
                                pt["action_owner"] = _map_val(pt["action_owner"], raw_id_map)
                            for tf in ("discussion_point", "text", "polished_text"):
                                if isinstance(pt.get(tf), str):
                                    pt[tf] = replace_speaker_in_text(pt[tf], raw_id_map)

            # Stage 2
            stage2 = rom.get("stage2")
            if isinstance(stage2, dict):
                pts = stage2.get("polished_points")
                if isinstance(pts, list):
                    for pt in pts:
                        if isinstance(pt, dict):
                            if "speaker" in pt:
                                pt["speaker"] = _map_val(pt["speaker"], raw_id_map)
                            pt["speakers"] = _map_list(pt.get("speakers", []), raw_id_map)
                            if "action_owner" in pt:
                                pt["action_owner"] = _map_val(pt["action_owner"], raw_id_map)
                            if isinstance(pt.get("action_items"), list):
                                for act in pt["action_items"]:
                                    if isinstance(act, dict) and act.get("assignee"):
                                        act["assignee"] = _map_val(act["assignee"], raw_id_map)
                            for tf in ("polished_text", "text"):
                                if isinstance(pt.get(tf), str):
                                    pt[tf] = replace_speaker_in_text(pt[tf], raw_id_map)

            # Stage 3
            stage3 = rom.get("stage3")
            if isinstance(stage3, dict):
                agendas = stage3.get("agendas")
                if isinstance(agendas, list):
                    for a in agendas:
                        if isinstance(a, dict):
                            if "speaker" in a:
                                a["speaker"] = _map_val(a["speaker"], raw_id_map)
                            if "presenter" in a:
                                a["presenter"] = _map_val(a["presenter"], raw_id_map)
                            dpts = a.get("discussion_points")
                            if isinstance(dpts, list):
                                for dp in dpts:
                                    if isinstance(dp, dict):
                                        if "speaker" in dp:
                                            dp["speaker"] = _map_val(dp["speaker"], raw_id_map)
                                        dp["speakers"] = _map_list(dp.get("speakers", []), raw_id_map)
                                        if "action_owner" in dp:
                                            dp["action_owner"] = _map_val(dp["action_owner"], raw_id_map)
                                        for tf in ("polished_text", "text"):
                                            if isinstance(dp.get(tf), str):
                                                dp[tf] = replace_speaker_in_text(dp[tf], raw_id_map)

                doc_pts = stage3.get("agenda_doc_points")
                if isinstance(doc_pts, dict):
                    for aid, entry in doc_pts.items():
                        if isinstance(entry, dict):
                            if "speaker" in entry:
                                entry["speaker"] = _map_val(entry["speaker"], raw_id_map)

            # Final ROM
            final_rom = rom.get("final_rom")
            if isinstance(final_rom, dict):
                agendas = final_rom.get("agendas")
                if isinstance(agendas, list):
                    for fa in agendas:
                        if isinstance(fa, dict):
                            if "presenter" in fa:
                                fa["presenter"] = _map_val(fa["presenter"], raw_id_map)
                            if "speaker" in fa:
                                fa["speaker"] = _map_val(fa["speaker"], raw_id_map)
                            dpts = fa.get("discussion_points")
                            if isinstance(dpts, list):
                                for dp in dpts:
                                    if isinstance(dp, dict):
                                        dp["speakers"] = _map_list(dp.get("speakers", []), raw_id_map)
                                        if "action_owner" in dp:
                                            dp["action_owner"] = _map_val(dp["action_owner"], raw_id_map)
                                        if "speaker" in dp:
                                            dp["speaker"] = _map_val(dp["speaker"], raw_id_map)
                                        for tf in ("polished_text", "text"):
                                            if isinstance(dp.get(tf), str):
                                                dp[tf] = replace_speaker_in_text(dp[tf], raw_id_map)
                            acts = fa.get("action_items")
                            if isinstance(acts, list):
                                for ai in acts:
                                    if isinstance(ai, dict):
                                        if "owner" in ai:
                                            ai["owner"] = _map_val(ai["owner"], raw_id_map)
                                        if "speaker" in ai:
                                            ai["speaker"] = _map_val(ai["speaker"], raw_id_map)

            updated["rom_data"] = rom


    return updated


# ── MoM row helper ─────────────────────────────────────────────────────────────

def _apply_tracks_to_mom(
    mom_dict: Dict[str, Any],
    label_map: Dict[str, str],
    raw_id_map: Dict[str, str],
) -> Dict[str, Any]:
    """
    Apply both tracks to a minutes_of_meeting row.

    participants[] → label_map (speaker_label values)
    action_items[].owner → raw_id_map first, then label_map fallback
    points_discussed, introduction, conclusion → both maps as text replacement
    """
    if not label_map and not raw_id_map:
        return mom_dict

    updated = dict(mom_dict)
    combined = {**raw_id_map, **label_map}  # label takes precedence on conflict

    # Safely deserialize list fields if they are JSON strings
    for list_field in ("participants", "action_items", "points_discussed"):
        val = updated.get(list_field)
        if isinstance(val, str):
            try:
                parsed = from_json(val, [])
                if isinstance(parsed, list):
                    updated[list_field] = parsed
            except Exception:
                pass

    # participants — speaker_label based
    parts = updated.get("participants")
    if parts:
        p_list = from_json(parts) if isinstance(parts, str) else (list(parts) if isinstance(parts, list) else [])
        if isinstance(p_list, list):
            seen: Set[str] = set()
            new_p = []
            for p in p_list:
                mapped = label_map.get(str(p), str(p))
                if mapped not in seen:
                    new_p.append(mapped)
                    seen.add(mapped)
            updated["participants"] = new_p

    # action_items — owner may be raw ID or display label
    actions = updated.get("action_items")
    if actions:
        a_list = from_json(actions) if isinstance(actions, str) else (list(actions) if isinstance(actions, list) else [])
        if isinstance(a_list, list):
            new_a = []
            for item in a_list:
                if isinstance(item, dict):
                    item_copy = dict(item)
                    owner = item_copy.get("owner")
                    if owner in raw_id_map:
                        item_copy["owner"] = raw_id_map[owner]
                    elif owner in label_map:
                        item_copy["owner"] = label_map[owner]
                    new_a.append(replace_deep_speaker_names(item_copy, combined))
                elif isinstance(item, str):
                    new_a.append(replace_speaker_in_text(item, combined))
                else:
                    new_a.append(item)
            updated["action_items"] = new_a

    # Text fields — replace with combined map
    for field in ("points_discussed", "introduction", "conclusion", "discussion_summary"):
        val = updated.get(field)
        if val:
            if isinstance(val, str):
                updated[field] = replace_speaker_in_text(val, combined)
            elif isinstance(val, list):
                updated[field] = replace_deep_speaker_names(val, combined)

    return updated


# ── Public API compatibility shims ─────────────────────────────────────────────

def apply_speaker_mappings_to_recording_dict(
    rec: Dict[str, Any],
    speaker_mappings: Dict[str, Any],
    _already_resolved: bool = False,
) -> Dict[str, Any]:
    """
    Public API shim used by history.py and tests.

    Applies Track 1 (label_map) for display labels and Track 2 (raw_id_map)
    for raw diarization IDs found in the transcript or matching label keys.
    """
    label_map = build_clean_speaker_mappings(speaker_mappings)
    if not label_map:
        return rec

    raw_t = rec.get("transcript")
    t_list = from_json(raw_t, []) if isinstance(raw_t, str) else (list(raw_t) if raw_t else [])

    raw_id_map: Dict[str, str] = {}
    for label, name in label_map.items():
        raw_id_map[label] = name
        for raw_id in collect_raw_ids_for_label(t_list, label):
            raw_id_map[raw_id] = name

    updated = _apply_label_track(rec, label_map, raw_id_map)
    updated = _apply_raw_id_track(updated, raw_id_map)
    return updated


def apply_speaker_mappings_to_mom_dict(
    mom_dict: Dict[str, Any],
    speaker_mappings: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Public API shim used by rom_router.py and mom_router.py.

    Applies speaker_mappings as a label_map to the MoM row.
    No raw_id_map is available in this context (no transcript).
    """
    label_map = build_clean_speaker_mappings(speaker_mappings)
    if not label_map or not isinstance(mom_dict, dict):
        return mom_dict
    return _apply_tracks_to_mom(mom_dict, label_map, {})


# ── Legacy shim (no longer used internally, kept for external callers) ─────────

def resolve_recording_speaker_mappings(
    rec: Dict[str, Any],
    initial_mappings: Dict[str, Any],
) -> Dict[str, str]:
    """
    Legacy shim. Returns only exact mappings provided (no expansion).
    New code should use collect_raw_ids_for_label + two-track approach instead.
    """
    return build_clean_speaker_mappings(initial_mappings)


# ── Main entry point ───────────────────────────────────────────────────────────

async def sync_global_speaker_rename(
    db,
    recording_id: str,
    user_id: str,
    new_mappings: Dict[str, Any],
    replace_all: bool = False,
) -> Dict[str, Any]:
    """
    Rename speakers across all recording data structures using two separate tracks.
    If replace_all=True, overwrites speaker_mappings with new_mappings instead of merging.
    """
    # Build Track 1 exact label → name map
    label_map: Dict[str, str] = {}
    for k, v in new_mappings.items():
        k_s = str(k).strip()
        v_s = str(v).strip()
        if k_s and v_s and k_s != v_s:
            label_map[k_s] = v_s

    if not label_map and not replace_all:
        logger.info(f"[SpeakerSync] No valid speaker mappings for {recording_id}")
        return {}

    # Load recording
    r = await db.execute(
        text("SELECT * FROM recordings WHERE id = :id AND user_id = :uid"),
        {"id": recording_id, "uid": user_id},
    )
    row = r.mappings().fetchone()
    if not row:
        logger.warning(f"[SpeakerSync] Recording {recording_id} not found for user {user_id}")
        return {}

    rec_dict = dict(row)

    # Merge existing DB mappings if not replace_all
    existing_sm_raw = rec_dict.get("speaker_mappings")
    existing_sm = from_json(existing_sm_raw, {}) if isinstance(existing_sm_raw, str) else (existing_sm_raw or {})
    if not isinstance(existing_sm, dict):
        existing_sm = {}

    if replace_all:
        combined_label_map = dict(label_map)
    else:
        combined_label_map = dict(existing_sm)
        combined_label_map.update(label_map)

    # Build Track 2 raw_id_map by scanning transcript across all cumulative mappings
    raw_t = rec_dict.get("transcript")
    t_list = from_json(raw_t, []) if isinstance(raw_t, str) else (list(raw_t) if raw_t else [])

    raw_id_map: Dict[str, str] = {}
    for label, name in combined_label_map.items():
        for raw_id in collect_raw_ids_for_label(t_list, label, target_name=name):
            raw_id_map[raw_id] = name

    logger.info(
        f"[SpeakerSync] [{recording_id}] Track 1 label→name: {combined_label_map}\n"
        f"[SpeakerSync] [{recording_id}] Track 2 raw_id→name: {raw_id_map} (replace_all={replace_all})"
    )

    # Apply Track 1: display label fields
    updated_rec = _apply_label_track(rec_dict, combined_label_map, raw_id_map, replace_all=replace_all)

    # Apply Track 2: raw ID fields in ROM stages
    updated_rec = _apply_raw_id_track(updated_rec, raw_id_map)

    # Persist recording row
    await db.execute(
        text("""
            UPDATE recordings
            SET transcript = :transcript,
                speakers_detected = :speakers_detected,
                speaker_summary = :speaker_summary,
                rom_data = :rom_data,
                summary = :summary,
                short_summary = :short_summary,
                detailed_summary = :detailed_summary,
                key_points = :key_points,
                action_items = :action_items,
                speaker_mappings = :speaker_mappings
            WHERE id = :id AND user_id = :uid
        """),
        {
            "transcript": to_json(updated_rec.get("transcript", [])),
            "speakers_detected": to_json(updated_rec.get("speakers_detected", [])),
            "speaker_summary": to_json(updated_rec.get("speaker_summary")),
            "rom_data": to_json(updated_rec.get("rom_data", {})),
            "summary": updated_rec.get("summary"),
            "short_summary": updated_rec.get("short_summary"),
            "detailed_summary": updated_rec.get("detailed_summary"),
            "key_points": to_json(updated_rec.get("key_points", [])),
            "action_items": to_json(updated_rec.get("action_items", [])),
            "speaker_mappings": to_json(updated_rec.get("speaker_mappings", {})),
            "id": recording_id,
            "uid": user_id,
        },
    )

    # Persist minutes_of_meeting rows
    m_res = await db.execute(
        text("SELECT * FROM minutes_of_meeting WHERE recording_id = :rid AND user_id = :uid"),
        {"rid": recording_id, "uid": user_id},
    )
    mom_rows = m_res.mappings().fetchall()
    for mom_row in mom_rows:
        mom_dict = dict(mom_row)
        updated_mom = _apply_tracks_to_mom(mom_dict, label_map, raw_id_map)
        await db.execute(
            text("""
                UPDATE minutes_of_meeting
                SET participants = :participants,
                    action_items = :action_items,
                    points_discussed = :points_discussed,
                    introduction = :introduction,
                    conclusion = :conclusion,
                    updated_at = :updated_at
                WHERE id = :id AND user_id = :uid
            """),
            {
                "participants": to_json(updated_mom.get("participants", [])),
                "action_items": to_json(updated_mom.get("action_items", [])),
                "points_discussed": to_json(updated_mom.get("points_discussed", [])),
                "introduction": updated_mom.get("introduction"),
                "conclusion": updated_mom.get("conclusion"),
                "updated_at": updated_mom.get("updated_at"),
                "id": mom_dict["id"],
                "uid": user_id,
            },
        )

    await db.commit()
    logger.info(
        f"[SpeakerSync] [{recording_id}] Done — "
        f"label_map={label_map}, raw_id_map={raw_id_map}"
    )
    return updated_rec
