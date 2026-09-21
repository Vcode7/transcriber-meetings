import json
import logging
import uuid
import urllib.request
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import text

from database import get_db_context, from_json, to_json
from routers.auth import get_current_user
from services import prompt_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/arena", tags=["arena"])

# ── Pydantic Models ────────────────────────────────────────────────────────────

class ArenaTestRequest(BaseModel):
    prompt_key: str
    recording_id: Optional[str] = None
    meeting_id: Optional[str] = None
    model_name: Optional[str] = None
    model: Optional[str] = None
    prompt_text: str
    window_index: Optional[int] = None

class GenerateRuleRequest(BaseModel):
    description: str
    model_name: Optional[str] = None
    model: Optional[str] = None
    context: str = ""

class SaveDraftRequest(BaseModel):
    prompt_key: str
    draft_name: Optional[str] = ""
    name: Optional[str] = ""
    template: str
    rule_states: Optional[dict] = {}

# ── Helper Functions ───────────────────────────────────────────────────────────

def safe_parse_json(val: Any, default: Any = None) -> Any:
    """Deserialize JSON data, unwrapping nested stringified JSON, None, or 'null' strings."""
    if val is None:
        return default
    if isinstance(val, (dict, list)):
        return val
    curr = val
    for _ in range(5):
        if not isinstance(curr, str):
            break
        trimmed = curr.strip()
        if not trimmed or trimmed.lower() == "null":
            return default
        try:
            curr = json.loads(trimmed)
        except Exception:
            break
    return curr if curr is not None else default

def safe_parse_dict(val: Any) -> dict:
    res = safe_parse_json(val, default={})
    return res if isinstance(res, dict) else {}

def safe_parse_list(val: Any) -> list:
    res = safe_parse_json(val, default=[])
    return res if isinstance(res, list) else []

async def get_recording_rom_data(recording_id: str, user_id: str, db) -> dict:
    """Fetch rom_data checking both rom_metadata and recordings tables, guaranteed to return a dict."""
    # 1. Try rom_metadata first (where current pipeline persists rom_data)
    try:
        res = await db.execute(
            text("SELECT rom_data FROM rom_metadata WHERE recording_id = :id AND user_id = :uid"),
            {"id": recording_id, "uid": user_id}
        )
        rm_row = res.mappings().fetchone()
        if rm_row and rm_row.get("rom_data"):
            d = safe_parse_dict(rm_row["rom_data"])
            if d:
                return d
    except Exception as e:
        logger.debug(f"Error querying rom_metadata for {recording_id}: {e}")

    # 2. Fall back to recordings table
    try:
        res = await db.execute(
            text("SELECT rom_data FROM recordings WHERE id = :id AND user_id = :uid"),
            {"id": recording_id, "uid": user_id}
        )
        rec_row = res.mappings().fetchone()
        if rec_row and rec_row.get("rom_data"):
            d = safe_parse_dict(rec_row["rom_data"])
            if d:
                return d
    except Exception as e:
        logger.debug(f"Error querying recordings.rom_data for {recording_id}: {e}")

    return {}

def get_sliding_windows(transcript: Any, window_minutes: float) -> List[List[Dict]]:
    windows = []
    if not transcript or not isinstance(transcript, list):
        return windows

    clean_segments = [s for s in transcript if isinstance(s, dict)]
    if not clean_segments:
        return windows

    current_window = []
    current_start_time = clean_segments[0].get('start', 0.0)
    window_seconds = window_minutes * 60.0
    
    for segment in clean_segments:
        seg_start = segment.get('start', 0.0)
        if current_window and (seg_start - current_start_time) >= window_seconds:
            windows.append(current_window)
            current_window = [segment]
            current_start_time = seg_start
        else:
            current_window.append(segment)
            
    if current_window:
        windows.append(current_window)
        
    return windows

def format_window_text(window: List[Dict]) -> str:
    window_text = ""
    if not window or not isinstance(window, list):
        return window_text
    for seg in window:
        if not isinstance(seg, dict):
            continue
        speaker = seg.get('speaker_label') or seg.get('speaker') or 'Unknown'
        start = seg.get('start', 0.0)
        end = seg.get('end', 0.0)
        text_content = seg.get('text', '').strip()
        window_text += f"[{start:.1f}-{end:.1f}] {speaker}: {text_content}\n"
    return window_text

def normalize_ollama_url(url: str) -> str:
    url = url.strip()
    if not url.startswith("http"):
        url = "http://" + url
    return url.rstrip("/")

# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/ollama-models")
async def list_ollama_models(current_user: dict = Depends(get_current_user)):
    user_id = current_user["id"]
    async with get_db_context() as db:
        r = await db.execute(
            text("SELECT ollama_server_url FROM user_settings WHERE user_id = :uid"),
            {"uid": user_id},
        )
        row = r.mappings().fetchone()
        url_to_test = row["ollama_server_url"] if row and row.get("ollama_server_url") else "http://localhost:11434"

    server_url = normalize_ollama_url(url_to_test)
    running_models = []
    available_models = []

    try:
        req_tags = urllib.request.Request(f"{server_url}/api/tags")
        with urllib.request.urlopen(req_tags, timeout=5.0) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                models = data.get("models", [])
                available_models = [m.get("name") for m in models if isinstance(m, dict) and m.get("name")]
    except Exception as e:
        return {
            "success": False,
            "server_url": server_url,
            "error": f"Could not connect to Ollama server at {server_url}: {str(e)}",
            "message": "Connection failed.",
        }

    try:
        req_ps = urllib.request.Request(f"{server_url}/api/ps")
        with urllib.request.urlopen(req_ps, timeout=3.0) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                models = data.get("models", [])
                running_models = [m.get("name") for m in models if isinstance(m, dict) and m.get("name")]
    except Exception:
        pass

    return {
        "success": True,
        "server_url": server_url,
        "available_models": available_models,
        "running_models": running_models,
        "models": available_models,
        "running": running_models,
    }

@router.get("/meetings")
async def list_arena_meetings(current_user: dict = Depends(get_current_user)):
    user_id = current_user["id"]
    async with get_db_context() as db:
        query = text("""
            SELECT r.id, r.filename, r.duration, r.created_at, r.transcript, r.rom_data, rm.rom_data as rm_rom_data
            FROM recordings r
            LEFT JOIN rom_metadata rm ON rm.recording_id = r.id AND rm.user_id = r.user_id
            WHERE r.user_id = :uid AND r.status = 'done' AND r.transcript IS NOT NULL
            ORDER BY r.created_at DESC
        """)
        result = await db.execute(query, {"uid": user_id})
        
        meetings = []
        for row in result.mappings():
            rom_data = safe_parse_dict(row.get("rm_rom_data"))
            if not rom_data:
                rom_data = safe_parse_dict(row.get("rom_data"))
            transcript = safe_parse_list(row.get("transcript"))
            meetings.append({
                "id": row["id"],
                "filename": row["filename"],
                "duration": row["duration"],
                "created_at": row["created_at"],
                "has_transcript": bool(transcript),
                "has_rom_data": bool(rom_data)
            })
            
    return meetings

@router.get("/meetings/{recording_id}/windows")
async def get_meeting_windows(
    recording_id: str,
    window_minutes: float = Query(default=2.0),
    current_user: dict = Depends(get_current_user)
):
    async with get_db_context() as db:
        query = text("SELECT transcript FROM recordings WHERE id = :rid AND user_id = :uid")
        result = await db.execute(query, {"rid": recording_id, "uid": current_user["id"]})
        row = result.mappings().fetchone()
        
        if not row or not row.get("transcript"):
            raise HTTPException(status_code=404, detail="Recording or transcript not found")
            
        transcript = safe_parse_list(row["transcript"])
        
    windows = get_sliding_windows(transcript, window_minutes)
    
    window_info = []
    for i, window in enumerate(windows):
        if not window:
            continue
        start_time = window[0].get('start', 0.0) if isinstance(window[0], dict) else 0.0
        end_time = window[-1].get('end', 0.0) if isinstance(window[-1], dict) else 0.0
        
        # Build preview text
        preview = ""
        for seg in window:
            if not isinstance(seg, dict):
                continue
            speaker = seg.get('speaker_label') or seg.get('speaker') or 'Unknown'
            preview += f"{speaker}: {seg.get('text', '')} "
            if len(preview) > 200:
                preview = preview[:200] + "..."
                break
                
        # Unique speakers
        speakers = list({(seg.get('speaker_label') or seg.get('speaker') or 'Unknown') for seg in window if isinstance(seg, dict)})
        
        window_info.append({
            "index": i,
            "start_time": start_time,
            "end_time": end_time,
            "segment_count": len(window),
            "speakers": speakers,
            "preview": preview
        })
        
    return window_info

@router.get("/meetings/{recording_id}/stage-data")
async def get_meeting_stage_data(
    recording_id: str,
    current_user: dict = Depends(get_current_user)
):
    async with get_db_context() as db:
        query = text("SELECT transcript FROM recordings WHERE id = :rid AND user_id = :uid")
        result = await db.execute(query, {"rid": recording_id, "uid": current_user["id"]})
        row = result.mappings().fetchone()
        
        if not row:
            raise HTTPException(status_code=404, detail="Recording not found")
            
        transcript = safe_parse_list(row["transcript"])
        rom_data = await get_recording_rom_data(recording_id, current_user["id"], db)
        
    stage1_dict = rom_data.get("stage1") if isinstance(rom_data.get("stage1"), dict) else {}
    stage1_points = (
        stage1_dict.get("discussion_points") or 
        stage1_dict.get("points") or 
        rom_data.get("discussion_points") or 
        []
    )
    if not isinstance(stage1_points, list):
        stage1_points = []

    stage2_dict = rom_data.get("stage2") if isinstance(rom_data.get("stage2"), dict) else {}
    stage2_points = (
        stage2_dict.get("polished_points") or 
        stage2_dict.get("enhanced_discussion_points") or 
        stage2_dict.get("discussion_points") or 
        stage2_dict.get("points") or 
        rom_data.get("polished_points") or 
        rom_data.get("enhanced_discussion_points") or 
        []
    )
    if not isinstance(stage2_points, list):
        stage2_points = []

    stage3_dict = rom_data.get("stage3") if isinstance(rom_data.get("stage3"), dict) else {}
    final_rom_versions = rom_data.get("final_rom_versions") if isinstance(rom_data.get("final_rom_versions"), dict) else {}
    final_rom = rom_data.get("final_rom") if isinstance(rom_data.get("final_rom"), dict) else {}
    stage3_agendas = (
        final_rom_versions.get("long", {}).get("agendas") or
        final_rom_versions.get("medium", {}).get("agendas") or
        final_rom_versions.get("short", {}).get("agendas") or
        final_rom.get("agendas") or 
        stage3_dict.get("agendas") or 
        stage3_dict.get("expanded_agendas") or 
        rom_data.get("agendas") or 
        []
    )
    if not isinstance(stage3_agendas, list):
        stage3_agendas = []

    has_final_rom = (
        bool(final_rom.get("agendas")) or
        bool(final_rom_versions.get("long", {}).get("agendas")) or
        bool(final_rom_versions.get("medium", {}).get("agendas")) or
        bool(final_rom_versions.get("short", {}).get("agendas")) or
        bool(final_rom)
    )
    
    return {
        "has_transcript": bool(transcript),
        "has_stage1_points": len(stage1_points) > 0,
        "stage1_point_count": len(stage1_points),
        "has_stage2_points": len(stage2_points) > 0,
        "stage2_point_count": len(stage2_points),
        "has_stage3_agendas": len(stage3_agendas) > 0,
        "stage3_agenda_count": len(stage3_agendas),
        "has_final_rom": has_final_rom
    }

@router.post("/test")
async def test_arena_prompt(
    body: ArenaTestRequest,
    current_user: dict = Depends(get_current_user)
):
    if body.prompt_key not in prompt_service.VALID_KEYS:
        return {"success": False, "error": f"Invalid prompt key: {body.prompt_key}"}
        
    rec_id = body.recording_id or body.meeting_id
    if not rec_id:
        return {"success": False, "error": "Recording ID is required"}

    async with get_db_context() as db:
        query = text("""
            SELECT filename, transcript, video_transcript, created_at
            FROM recordings 
            WHERE id = :rid AND user_id = :uid
        """)
        result = await db.execute(query, {"rid": rec_id, "uid": current_user["id"]})
        row = result.mappings().fetchone()
        
        if not row:
            return {"success": False, "error": "Recording not found"}
            
        settings_query = text("SELECT * FROM user_settings WHERE user_id = :uid")
        settings_result = await db.execute(settings_query, {"uid": current_user["id"]})
        user_settings = settings_result.mappings().fetchone()

        rom_data = await get_recording_rom_data(rec_id, current_user["id"], db)
        
    if not user_settings:
        return {"success": False, "error": "User settings not found"}

    transcript = safe_parse_list(row["transcript"])
    video_transcript = safe_parse_list(row["video_transcript"])
    
    # ── 1. Gather all transcript & text representations ─────────────────────────
    transcript_segments = [s for s in transcript if isinstance(s, dict)]
    raw_lines = [seg.get("text", "").strip() for seg in transcript_segments if seg.get("text")]
    transcript_raw_text = "\n".join(raw_lines)
    full_transcript_text = "\n".join([
        f"[{seg.get('start', 0.0):.1f}-{seg.get('end', 0.0):.1f}] "
        f"{seg.get('speaker_label') or seg.get('speaker') or 'Speaker'}: "
        f"{seg.get('text', '').strip()}"
        for seg in transcript_segments if seg.get("text")
    ])
    speakers = list({
        (seg.get('speaker_label') or seg.get('speaker') or 'Speaker 1')
        for seg in transcript_segments
    })
    primary_speaker = speakers[0] if speakers else "Speaker 1"

    # ── 2. Sliding windows for windowed prompts ────────────────────────────────
    windows = get_sliding_windows(transcript, window_minutes=2.0)
    idx = body.window_index if body.window_index is not None and 0 <= body.window_index < len(windows) else 0
    window = windows[idx] if windows else []
    window_info = {
        "index": idx,
        "start": window[0].get('start', 0.0) if window and isinstance(window[0], dict) else 0.0,
        "end": window[-1].get('end', 0.0) if window and isinstance(window[-1], dict) else 0.0
    }
    window_text = format_window_text(window)
    previous_context = (
        "PREVIOUS TRANSCRIPT WINDOW:\n" + format_window_text(windows[idx-1])
        if idx > 0 and len(windows) > idx - 1 else ""
    )
    video_context_section = (
        "VIDEO OCR TRANSCRIPT (supplementary — audio transcript is primary):\n[Placeholder for OCR]"
        if video_transcript else ""
    )

    # ── 3. Extract points across all stages & pipelines ─────────────────────────
    stage1_dict = rom_data.get("stage1") if isinstance(rom_data.get("stage1"), dict) else {}
    stage1_points = (
        stage1_dict.get("discussion_points") or 
        stage1_dict.get("points") or 
        rom_data.get("discussion_points") or 
        []
    )
    if not isinstance(stage1_points, list):
        stage1_points = []

    stage2_dict = rom_data.get("stage2") if isinstance(rom_data.get("stage2"), dict) else {}
    stage2_points = (
        stage2_dict.get("polished_points") or 
        stage2_dict.get("enhanced_discussion_points") or 
        stage2_dict.get("discussion_points") or 
        stage2_dict.get("points") or 
        rom_data.get("polished_points") or 
        rom_data.get("enhanced_discussion_points") or 
        []
    )
    if not isinstance(stage2_points, list):
        stage2_points = []

    stage3_dict = rom_data.get("stage3") if isinstance(rom_data.get("stage3"), dict) else {}
    final_rom_versions = rom_data.get("final_rom_versions") if isinstance(rom_data.get("final_rom_versions"), dict) else {}
    final_rom = rom_data.get("final_rom") if isinstance(rom_data.get("final_rom"), dict) else {}

    # Find the best candidate agenda list that has discussion_points populated
    agendas_candidates = [
        final_rom_versions.get("long", {}).get("agendas"),
        final_rom_versions.get("medium", {}).get("agendas"),
        final_rom_versions.get("short", {}).get("agendas"),
        final_rom.get("agendas"),
        stage3_dict.get("agendas"),
        stage3_dict.get("expanded_agendas"),
        rom_data.get("agendas"),
    ]
    resolved_agendas = []
    for cand in agendas_candidates:
        if isinstance(cand, list) and cand:
            if any(isinstance(a, dict) and (a.get("discussion_points") or a.get("points")) for a in cand):
                resolved_agendas = cand
                break
    if not resolved_agendas:
        for cand in agendas_candidates:
            if isinstance(cand, list) and cand:
                resolved_agendas = cand
                break

    # Gather all discussion points into a consolidated pool
    all_points = []
    for ag in resolved_agendas:
        if isinstance(ag, dict):
            ag_pts = ag.get("discussion_points") or ag.get("points")
            if isinstance(ag_pts, list) and ag_pts:
                all_points.extend(ag_pts)
    if not all_points and stage2_points:
        all_points.extend(stage2_points)
    if not all_points and stage1_points:
        all_points.extend(stage1_points)
    if not all_points and transcript_segments:
        for i, seg in enumerate(transcript_segments[:10]):
            txt = seg.get("text", "").strip()
            if txt:
                spk = seg.get("speaker_label") or seg.get("speaker") or "Speaker 1"
                all_points.append({
                    "id": f"P{i+1}",
                    "point_id": f"P{i+1}",
                    "speaker": spk,
                    "discussion_point": txt,
                    "polished_text": txt,
                    "text": txt,
                    "action_owner": None,
                })

    # Pick the primary agenda
    first_agenda = None
    for ag in resolved_agendas:
        if isinstance(ag, dict) and (ag.get("discussion_points") or ag.get("points")):
            first_agenda = ag
            break
    if not first_agenda and resolved_agendas and isinstance(resolved_agendas[0], dict):
        first_agenda = dict(resolved_agendas[0])
        first_agenda["discussion_points"] = all_points[:10]
    if not first_agenda:
        first_agenda = {
            "agenda_id": "A1",
            "title": "General Discussion",
            "description": "General meeting discussion topics and decisions",
            "discussion_points": all_points[:10]
        }

    agenda_pts = first_agenda.get("discussion_points") or first_agenda.get("points") or all_points[:10]
    if not isinstance(agenda_pts, list) or not agenda_pts:
        agenda_pts = all_points[:10]

    # Clean structured point objects: Point ID, Speaker, Discussion, Action Owner
    clean_pts = []
    for i, pt in enumerate(agenda_pts):
        if isinstance(pt, dict):
            pt_id = pt.get("id") or pt.get("point_id") or f"P{i+1}"
            spk = pt.get("speaker") or (pt.get("speakers") or [None])[0] or primary_speaker
            disc = (pt.get("polished_text") or pt.get("discussion_point") or pt.get("text") or "").strip()
            act = pt.get("action_owner") or "N/A"
            clean_pts.append({
                "id": pt_id,
                "speaker": spk,
                "discussion_point": disc,
                "action_owner": act
            })
        elif isinstance(pt, str) and pt.strip():
            clean_pts.append({
                "id": f"P{i+1}",
                "speaker": primary_speaker,
                "discussion_point": pt.strip(),
                "action_owner": "N/A"
            })

    if not clean_pts and all_points:
        for i, pt in enumerate(all_points[:10]):
            if isinstance(pt, dict):
                pt_id = pt.get("id") or pt.get("point_id") or f"P{i+1}"
                spk = pt.get("speaker") or (pt.get("speakers") or [None])[0] or primary_speaker
                disc = (pt.get("polished_text") or pt.get("discussion_point") or pt.get("text") or "").strip()
                act = pt.get("action_owner") or "N/A"
                clean_pts.append({
                    "id": pt_id,
                    "speaker": spk,
                    "discussion_point": disc,
                    "action_owner": act
                })

    # Action items pool
    extracted_actions = []
    for p in all_points:
        if isinstance(p, dict) and p.get("action_items") and isinstance(p["action_items"], list):
            extracted_actions.extend(p["action_items"])
    if not extracted_actions:
        for p in clean_pts[:5]:
            if p.get("action_owner") and p.get("action_owner") != "N/A":
                extracted_actions.append({
                    "task": p.get("discussion_point", "")[:120],
                    "owner": p.get("action_owner"),
                    "deadline": "End of week"
                })
    if not extracted_actions:
        extracted_actions = [{
            "task": f"Follow up on action commitments from {row.get('filename', 'meeting')}",
            "owner": primary_speaker,
            "deadline": "Next Meeting"
        }]

    # Clean agenda summaries
    agendas_simple = []
    for ag in resolved_agendas:
        if isinstance(ag, dict):
            agendas_simple.append({
                "agenda_id": ag.get("agenda_id") or f"A{len(agendas_simple)+1}",
                "title": ag.get("title") or "Discussion Topic",
                "description": ag.get("description") or "Meeting agenda section"
            })
    if not agendas_simple:
        agendas_simple = [{
            "agenda_id": "A1",
            "title": first_agenda.get("title", "General Discussion"),
            "description": first_agenda.get("description", "General meeting discussion and topics")
        }]

    # ── 4. Stage dependency checks ──────────────────────────────────────────────
    stage1_keys = ["rom_discussion_embedded", "rom_discussion", "rom_discussion_no_actions", "rom_action_extraction", "stage1_json_repair"]
    stage2_keys = ["rom_enhance_window", "rom_enhance_all_together", "rom_deduplicate", "rom_polish"]
    stage3_keys = ["rom_agenda", "rom_agenda_assign_batch", "rom_mom_expansion", "rom_agenda_doc_points"]
    final_rom_keys = ["rom_version_short", "rom_version_medium", "rom_ai_edit_points", "rom_ai_chat"]
    mom_keys = ["mom", "mom_extract_actions", "mom_merge", "mom_action_dedup", "mom_action_regen"]

    if body.prompt_key in stage1_keys:
        if not transcript:
            return {"success": False, "error": "missing_dependency", "missing_dependency": "transcript", "required_stage": "Stage 0 (Transcription)", "reason": "Transcript segments are required for Stage 1 discussion extraction"}
    elif body.prompt_key in stage2_keys:
        if not stage1_points and not transcript:
            return {"success": False, "error": "missing_dependency", "missing_dependency": "stage1_points", "required_stage": "Stage 1 (Discussion Points)", "reason": "Stage 1 discussion points are required for Stage 2 point enhancement"}
    elif body.prompt_key in stage3_keys:
        if not all_points and not transcript:
            return {"success": False, "error": "missing_dependency", "missing_dependency": "stage2_points", "required_stage": "Stage 2 (Enhanced Points)", "reason": "Discussion points are required for Stage 3 agenda mapping"}
    elif body.prompt_key in final_rom_keys:
        if not resolved_agendas and not all_points and not transcript:
            return {"success": False, "error": "missing_dependency", "missing_dependency": "final_rom_agendas", "required_stage": "Stage 3 (Agendas)", "reason": "Stage 3 Agendas or discussion points are required for Final ROM generation"}
    elif body.prompt_key in mom_keys:
        if not all_points and not transcript:
            return {"success": False, "error": "missing_dependency", "missing_dependency": "enhanced_points", "required_stage": "Stage 2 (Enhanced Points)", "reason": "Discussion points or transcript are required for MoM generation"}

    # ── 5. Build comprehensive variable dictionary covering all prompt keys ─────
    filename = row.get("filename", "Meeting")
    meeting_date = str(row.get("created_at") or "2026-09-01")[:10]

    arena_important_points = rom_data.get("important_points", [])
    if body.prompt_key in ("rom_version_short", "rom_version_medium"):
        for pt in clean_pts:
            pt_id = pt.get("id", "")
            disc = pt.get("discussion_point", "")
            is_imp = any(
                (imp.get("point_id") == pt_id or (imp.get("text") and imp.get("text").lower() in disc.lower()))
                if isinstance(imp, dict) else (str(imp).lower() in disc.lower())
                for imp in arena_important_points
            )
            if is_imp:
                pt["is_very_important"] = True
                pt["status"] = "VERY IMPORTANT - PRESERVE INTACT"

        mandatory_section_str = ""
        if arena_important_points:
            mandatory_section_str = "\n\nMANDATORY VERY IMPORTANT POINTS (MUST BE INCLUDED VERBATIM WITHOUT ANY SUMMARIZATION):\n"
            for idx, imp in enumerate(arena_important_points, 1):
                imp_text = imp.get("text", "") if isinstance(imp, dict) else str(imp)
                mandatory_section_str += f"- Point {idx}:\n  \"{imp_text}\"\n"
            mandatory_section_str += (
                "STRICT RULES FOR VERY IMPORTANT POINTS:\n"
                "1. EVERY point marked as 'Very Important' above MUST be included in the output JSON array.\n"
                "2. You MUST preserve the COMPLETE ORIGINAL CONTENT of each marked point fully intact.\n"
                "3. Do NOT summarize, shorten, merge, paraphrase, or omit any Very Important point.\n"
            )
    else:
        mandatory_section_str = ""
    version_points_str = json.dumps(clean_pts, indent=2)

    variables = {
        # Final ROM & Rewrite variables
        "{agenda_title}": first_agenda.get("title", "General Discussion"),
        "{agenda_description}": first_agenda.get("description", "General meeting discussion topics and decisions"),
        "{points_json}": version_points_str,
        "{rules_section}": "",
        "{mandatory_section}": mandatory_section_str,
        "{agenda_context}": first_agenda.get("title", "General Discussion"),
        "{selected_points}": json.dumps(clean_pts[:5], indent=2),
        "{user_prompt}": "Refine tone and ensure clear action ownership.",
        "{chat_history_section}": "",
        "{rom_context}": json.dumps(clean_pts[:10], indent=2),
        "{chat_history}": "[]",
        "{user_message}": "Summary of key decisions made.",

        # Stage 1 variables
        "{window_text}": window_text,
        "{previous_context}": previous_context,
        "{video_context_section}": video_context_section,
        "{invalid_json}": (
            json.dumps({"discussion_points": clean_pts[:2]}, indent=2)[:-15]
            if clean_pts else '{"discussion_points": [{"id": "p1", "text": "Incomplete discussion point'
        ),

        # Stage 2 variables
        "{window_json}": json.dumps(clean_pts[:5], indent=2),
        "{candidate_points}": json.dumps(clean_pts[:4], indent=2),
        "{original_points}": json.dumps(clean_pts[:10], indent=2),
        "{meeting_context}": f"[Meeting Context: {filename} — {transcript_raw_text[:250]}...]",
        "{global_context}": "[Global Context: Organizational project objectives and standard execution procedures]",
        "{previous_meeting_context}": "[Previous Meeting Context: Status updates from preceding review milestone]",
        "{retrieved_context}": f"[Retrieved Context: Relevant excerpts from project documentation for {filename}]",
        "{reference_examples_section}": "",

        # Stage 3 variables
        "{agenda_input}": "\n".join([f"- {p.get('discussion_point', '')}" for p in clean_pts[:10]]),
        "{context}": f"[Meeting Context: {filename}]\n{transcript_raw_text[:400]}",
        "{previous_mom}": "[Previous MoM: Reviewed preliminary project plan and confirmed milestone roadmap.]",
        "{previous_mom_docs}": "[Previous MoM Documents: Summary of prior commitments, decisions, and sign-offs.]",
        "{agenda_list}": json.dumps(agendas_simple, indent=2),
        "{agenda_reference}": json.dumps(agendas_simple, indent=2),
        "{discussion_order_guidance}": "Process discussion points in chronological order of appearance.",
        "{timeline_guidance}": "Group discussion items by meeting timeline and topic transitions.",
        "{batch_json}": json.dumps(clean_pts[:15], indent=2),
        "{document_text}": transcript_raw_text[:1200],

        # MoM variables
        "{transcript}": full_transcript_text or transcript_raw_text,
        "{agenda_section}": "\n".join([f"- {a.get('title', 'General Discussion')}" for a in agendas_simple]),
        "{reference_section}": "Enterprise Meeting Standards: Action items require an explicit owner and deadline.",
        "{action_items_json}": json.dumps(extracted_actions, indent=2),
        "{partial_moms_json}": json.dumps([
            {
                "title": first_agenda.get("title", "General Discussion"),
                "action_items": extracted_actions[:2],
                "discussion": transcript_raw_text[:300]
            }
        ], indent=2),

        # Summaries & Analysis variables
        "{text}": transcript_raw_text[:2000],
        "{summary}": f"Executive meeting overview of {filename}: Key discussions covered project milestones and deliverables.",
        "{chunk}": transcript_raw_text[:1500],
        "{speaker}": primary_speaker,

        # Collection AI variables
        "{conversation_history}": "[]",
        "{question}": "What were the key decisions and action owners established during this meeting?",
        "{meeting_a_name}": filename,
        "{meeting_a_date}": meeting_date,
        "{meeting_a_context}": transcript_raw_text[:800],
        "{meeting_b_name}": "Previous Project Review",
        "{meeting_b_date}": "2026-08-20",
        "{meeting_b_context}": "Initial scoping session establishing roadmap milestones and resource allocation.",
        "{topic}": "Project Milestones & Deliverables",
        "{meetings_context}": f"Meeting 1 ({filename}):\n{transcript_raw_text[:600]}\n\nMeeting 2 (Follow-up):\nTeam confirmed milestones and finalized delivery schedules.",
    }

    # Assemble final prompt text by replacing all variable placeholders
    final_prompt = body.prompt_text
    for var_name, var_value in variables.items():
        final_prompt = final_prompt.replace(var_name, var_value)
        
    # Model resolution
    model_name = body.model_name or body.model
    if not model_name:
        model_name = user_settings.get("ollama_model_priority", "gemma").split(",")[0].strip()

    # Call Ollama
    server_url = normalize_ollama_url(user_settings.get("ollama_server_url") or "http://localhost:11434")
    
    # Try to grab max_tokens for this prompt key, fallback to 4096
    max_tokens = user_settings.get(f"max_tokens_{body.prompt_key}") or 4096
    
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": "You are an expert enterprise meeting analyst. Follow instructions exactly. Preserve all technical terminology."
            },
            {"role": "user", "content": final_prompt}
        ],
        "options": {
            "num_predict": max_tokens,
            "temperature": float(user_settings.get("ollama_temperature", 0.3)),
            "num_ctx": int(user_settings.get("ollama_num_ctx", 32768)),
        },
        "stream": False
    }

    try:
        url = f"{server_url}/api/chat"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode('utf-8'),
            headers={"Content-Type": "application/json"}
        )
        # 30 minutes (1800 seconds) timeout to allow heavy LLM generation without premature timeouts
        with urllib.request.urlopen(req, timeout=1800) as response:
            if response.status == 200:
                resp_data = json.loads(response.read().decode('utf-8'))
                output = resp_data.get("message", {}).get("content", "")
                
                timing = {
                    "total_duration": resp_data.get("total_duration"),
                    "prompt_eval_count": resp_data.get("prompt_eval_count"),
                    "eval_count": resp_data.get("eval_count")
                }
                
                history_id = str(uuid.uuid4())
                created_at = datetime.now(timezone.utc).isoformat()
                
                result_payload = {
                    "id": history_id,
                    "history_id": history_id,
                    "created_at": created_at,
                    "prompt_key": body.prompt_key,
                    "meeting_id": rec_id,
                    "success": True,
                    "model": model_name,
                    "server_url": server_url,
                    "recording_name": row["filename"],
                    "window_info": window_info,
                    "final_prompt": final_prompt,
                    "prompt": final_prompt,
                    "input_context": variables,
                    "context": variables,
                    "output": output,
                    "timing": timing,
                    "duration_ms": round(timing["total_duration"] / 1e6, 1) if timing.get("total_duration") else 0,
                    "tokens_eval": timing.get("eval_count", 0),
                    "warnings": []
                }

                try:
                    async with get_db_context() as db:
                        history_query = text("""
                            INSERT INTO arena_history (id, user_id, prompt_key, meeting_id, recording_name, model_name, window_index, result_json, created_at)
                            VALUES (:id, :uid, :prompt_key, :meeting_id, :recording_name, :model_name, :window_index, :result_json, :created_at)
                        """)
                        await db.execute(history_query, {
                            "id": history_id,
                            "uid": current_user["id"],
                            "prompt_key": body.prompt_key,
                            "meeting_id": rec_id,
                            "recording_name": row["filename"] or "",
                            "model_name": model_name,
                            "window_index": body.window_index,
                            "result_json": json.dumps(result_payload),
                            "created_at": created_at
                        })
                        await db.commit()
                except Exception as db_err:
                    logger.warning(f"Failed to auto-persist arena history: {db_err}")

                return result_payload
            else:
                return {"success": False, "error": f"Ollama returned status {response.status}"}
    except Exception as e:
        logger.error(f"Arena Ollama call failed: {e}")
        return {"success": False, "error": f"Ollama call failed: {str(e)}"}

@router.post("/generate-rule")
async def generate_arena_rule(
    body: GenerateRuleRequest,
    current_user: dict = Depends(get_current_user)
):
    async with get_db_context() as db:
        settings_query = text("SELECT ollama_server_url, ollama_model_priority FROM user_settings WHERE user_id = :uid")
        settings_result = await db.execute(settings_query, {"uid": current_user["id"]})
        user_settings = settings_result.mappings().fetchone()
        
    if not user_settings:
        return {"success": False, "error": "User settings not found"}
        
    server_url = normalize_ollama_url(user_settings.get("ollama_server_url") or "http://localhost:11434")
    
    priority_list = user_settings.get("ollama_model_priority", "gemma,qwen,llama,deepseek,mistral")
    
    # Strictly use the default model saved in Settings (the model used by the main pipeline).
    # Do NOT use the Ollama model selected in Model Arena.
    from services.ai_provider import QwenProvider
    default_pipeline_model = QwenProvider._detect_ollama_model(server_url, priority_list)
    if not default_pipeline_model:
        priorities = [p.strip() for p in priority_list.split(",") if p.strip()]
        default_pipeline_model = priorities[0] if priorities else "qwen2.5"

    model_name = default_pipeline_model
    
    meta_prompt = f"""
    Based on the following natural language description, generate a well-written, concise instruction rule for an LLM prompt.
    Do not include any conversational filler, just return the exact rule string.
    
    Description: {body.description}
    Context/Prompt: {body.context}
    """
    
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a prompt engineering assistant."},
            {"role": "user", "content": meta_prompt}
        ],
        "options": {"num_predict": 256, "temperature": 0.2},
        "stream": False
    }

    try:
        url = f"{server_url}/api/chat"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode('utf-8'),
            headers={"Content-Type": "application/json"}
        )
        # 30 minutes (1800 seconds) timeout for AI rule generation
        with urllib.request.urlopen(req, timeout=1800) as response:
            if response.status == 200:
                resp_data = json.loads(response.read().decode('utf-8'))
                rule = resp_data.get("message", {}).get("content", "").strip()
                return {"rule": rule, "model_used": model_name}
            else:
                raise HTTPException(status_code=500, detail="Failed to generate rule")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/drafts")
async def list_arena_drafts(current_user: dict = Depends(get_current_user)):
    try:
        async with get_db_context() as db:
            query = text("SELECT id, prompt_key, draft_name, template, rule_states, created_at, updated_at FROM arena_drafts WHERE user_id = :uid ORDER BY created_at DESC")
            result = await db.execute(query, {"uid": current_user["id"]})
            drafts = []
            for row in result.mappings():
                drafts.append({
                    "id": row["id"],
                    "prompt_key": row["prompt_key"],
                    "draft_name": row["draft_name"],
                    "name": row["draft_name"],
                    "template": row["template"],
                    "rule_states": safe_parse_dict(row["rule_states"]) if row["rule_states"] else {},
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"]
                })
            return drafts
    except Exception as e:
        logger.warning(f"Error fetching arena drafts (table might not exist): {e}")
        return []

@router.post("/drafts")
async def save_arena_draft(
    body: SaveDraftRequest,
    current_user: dict = Depends(get_current_user)
):
    draft_id = str(uuid.uuid4())
    draft_title = body.draft_name or body.name or "Untitled Draft"
    try:
        async with get_db_context() as db:
            query = text("""
                INSERT INTO arena_drafts (id, user_id, prompt_key, draft_name, template, rule_states, created_at, updated_at)
                VALUES (:id, :uid, :prompt_key, :draft_name, :template, :rule_states, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """)
            await db.execute(query, {
                "id": draft_id,
                "uid": current_user["id"],
                "prompt_key": body.prompt_key,
                "draft_name": draft_title,
                "template": body.template,
                "rule_states": to_json(body.rule_states or {})
            })
            await db.commit()
        return {"success": True, "id": draft_id}
    except Exception as e:
        logger.error(f"Error saving draft: {e}")
        raise HTTPException(status_code=500, detail="Failed to save draft")

@router.delete("/drafts/{draft_id}")
async def delete_arena_draft(
    draft_id: str,
    current_user: dict = Depends(get_current_user)
):
    try:
        async with get_db_context() as db:
            query = text("DELETE FROM arena_drafts WHERE id = :id AND user_id = :uid")
            await db.execute(query, {"id": draft_id, "uid": current_user["id"]})
            await db.commit()
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to delete draft")

@router.get("/history")
async def list_arena_history(
    prompt_key: Optional[str] = None,
    meeting_id: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """
    List history entries for the current user.
    Optionally filter by prompt_key and/or meeting_id.
    Returns entries in reverse chronological order (newest first).
    """
    try:
        async with get_db_context() as db:
            where_clauses = ["user_id = :uid"]
            params: dict = {"uid": current_user["id"]}
            if prompt_key:
                where_clauses.append("prompt_key = :prompt_key")
                params["prompt_key"] = prompt_key
            if meeting_id:
                where_clauses.append("meeting_id = :meeting_id")
                params["meeting_id"] = meeting_id

            where_str = " AND ".join(where_clauses)
            query = text(f"""
                SELECT id, prompt_key, meeting_id, recording_name, model_name, window_index, result_json, created_at
                FROM arena_history
                WHERE {where_str}
                ORDER BY created_at DESC
                LIMIT 100
            """)
            result = await db.execute(query, params)
            entries = []
            for row in result.mappings():
                res_data = safe_parse_dict(row["result_json"]) if row["result_json"] else {}
                entries.append({
                    "id": row["id"],
                    "history_id": row["id"],
                    "prompt_key": row["prompt_key"],
                    "meeting_id": row["meeting_id"],
                    "recording_name": row["recording_name"],
                    "model_name": row["model_name"],
                    "window_index": row["window_index"],
                    "created_at": row["created_at"],
                    "data": res_data
                })
            return entries
    except Exception as e:
        logger.warning(f"Error fetching arena history: {e}")
        return []

@router.delete("/history/{history_id}")
async def delete_arena_history_entry(
    history_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Delete a single history entry."""
    try:
        async with get_db_context() as db:
            query = text("DELETE FROM arena_history WHERE id = :id AND user_id = :uid")
            await db.execute(query, {"id": history_id, "uid": current_user["id"]})
            await db.commit()
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to delete history entry")

@router.delete("/history")
async def clear_arena_history(
    prompt_key: Optional[str] = None,
    meeting_id: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Clear history entries for the user, optionally filtered by prompt_key and meeting_id."""
    try:
        async with get_db_context() as db:
            where_clauses = ["user_id = :uid"]
            params: dict = {"uid": current_user["id"]}
            if prompt_key:
                where_clauses.append("prompt_key = :prompt_key")
                params["prompt_key"] = prompt_key
            if meeting_id:
                where_clauses.append("meeting_id = :meeting_id")
                params["meeting_id"] = meeting_id

            where_str = " AND ".join(where_clauses)
            query = text(f"DELETE FROM arena_history WHERE {where_str}")
            await db.execute(query, params)
            await db.commit()
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to clear history")

@router.get("/prompt-groups")
async def get_prompt_groups(current_user: dict = Depends(get_current_user)):
    GROUPS = [
        {"group": "Stage 1", "keys": ["rom_discussion_embedded", "rom_discussion", "rom_discussion_no_actions", "rom_action_extraction", "stage1_json_repair"]},
        {"group": "Stage 2", "keys": ["rom_enhance_window", "rom_enhance_all_together", "rom_deduplicate", "rom_polish"]},
        {"group": "Stage 3", "keys": ["rom_agenda", "rom_mom_expansion", "rom_agenda_assign_batch", "rom_agenda_doc_points"]},
        {"group": "Final ROM", "keys": ["rom_version_short", "rom_version_medium", "rom_ai_edit_points", "rom_ai_chat"]},
        {"group": "Action Point from Enhanced Point", "keys": ["mom_extract_actions"]},
        {"group": "MoM", "keys": ["mom", "mom_merge", "mom_action_dedup", "mom_action_regen"]},
        {"group": "Summaries", "keys": ["executive_summary", "short_summary", "detailed_summary", "chunk_summary"]},
        {"group": "Analysis", "keys": ["key_points", "action_items", "key_decisions"]},
        {"group": "Speaker", "keys": ["speaker_summary", "speaker_key_points", "speaker_action_items"]},
        {"group": "Raw MoM", "keys": ["agenda_compress", "reference_compress", "agenda_from_summary", "agenda_compress_with_context"]},
        {"group": "Collection AI", "keys": ["collection_planning", "collection_chat", "collection_compare", "collection_topic_growth"]},
    ]
    
    meta_dict = {p["key"]: p for p in prompt_service.PROMPT_META}
    
    enriched_groups = []
    for g in GROUPS:
        enriched_keys = []
        for k in g["keys"]:
            meta = meta_dict.get(k, {})
            enriched_keys.append({
                "key": k,
                "name": meta.get("name", k),
                "description": meta.get("description", "")
            })
        enriched_groups.append({
            "group": g["group"],
            "keys": enriched_keys
        })
        
    return enriched_groups
