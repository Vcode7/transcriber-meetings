import json
import logging
import uuid
import urllib.request
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
    recording_id: str
    model_name: str
    prompt_text: str
    window_index: Optional[int] = None

class GenerateRuleRequest(BaseModel):
    description: str
    model_name: str
    context: str = ""

class SaveDraftRequest(BaseModel):
    prompt_key: str
    draft_name: str = ""
    template: str
    rule_states: dict = {}

# ── Helper Functions ───────────────────────────────────────────────────────────

def get_sliding_windows(transcript: List[Dict], window_minutes: float) -> List[List[Dict]]:
    windows = []
    current_window = []
    
    if not transcript:
        return windows

    current_start_time = transcript[0].get('start', 0.0)
    window_seconds = window_minutes * 60.0
    
    for segment in transcript:
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
    for seg in window:
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
    }

@router.get("/meetings")
async def list_arena_meetings(current_user: dict = Depends(get_current_user)):
    user_id = current_user["id"]
    async with get_db_context() as db:
        query = text("""
            SELECT id, filename, duration, created_at, transcript, rom_data 
            FROM recordings 
            WHERE user_id = :uid AND status = 'done' AND transcript IS NOT NULL
            ORDER BY created_at DESC
        """)
        result = await db.execute(query, {"uid": user_id})
        
        meetings = []
        for row in result.mappings():
            rom_data = from_json(row.get("rom_data")) or {}
            meetings.append({
                "id": row["id"],
                "filename": row["filename"],
                "duration": row["duration"],
                "created_at": row["created_at"],
                "has_transcript": row["transcript"] is not None,
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
        
        if not row or not row["transcript"]:
            raise HTTPException(status_code=404, detail="Recording or transcript not found")
            
        transcript = from_json(row["transcript"])
        
    windows = get_sliding_windows(transcript, window_minutes)
    
    window_info = []
    for i, window in enumerate(windows):
        if not window:
            continue
        start_time = window[0].get('start', 0.0)
        end_time = window[-1].get('end', 0.0)
        
        # Build preview text
        preview = ""
        for seg in window:
            speaker = seg.get('speaker_label') or seg.get('speaker') or 'Unknown'
            preview += f"{speaker}: {seg.get('text', '')} "
            if len(preview) > 200:
                preview = preview[:200] + "..."
                break
                
        # Unique speakers
        speakers = list({(seg.get('speaker_label') or seg.get('speaker') or 'Unknown') for seg in window})
        
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
        query = text("SELECT transcript, rom_data FROM recordings WHERE id = :rid AND user_id = :uid")
        result = await db.execute(query, {"rid": recording_id, "uid": current_user["id"]})
        row = result.mappings().fetchone()
        
        if not row:
            raise HTTPException(status_code=404, detail="Recording not found")
            
        transcript = row["transcript"]
        rom_data = from_json(row["rom_data"]) or {}
        
    stage1_points = rom_data.get("discussion_points", [])
    stage2_points = rom_data.get("enhanced_discussion_points", [])
    stage3_agendas = rom_data.get("agendas", [])
    final_rom = rom_data.get("final_rom", {})
    
    return {
        "has_transcript": transcript is not None,
        "has_stage1_points": len(stage1_points) > 0,
        "stage1_point_count": len(stage1_points),
        "has_stage2_points": len(stage2_points) > 0,
        "stage2_point_count": len(stage2_points),
        "has_stage3_agendas": len(stage3_agendas) > 0,
        "stage3_agenda_count": len(stage3_agendas),
        "has_final_rom": bool(final_rom)
    }

@router.post("/test")
async def test_arena_prompt(
    body: ArenaTestRequest,
    current_user: dict = Depends(get_current_user)
):
    if body.prompt_key not in prompt_service.VALID_KEYS:
        return {"success": False, "error": f"Invalid prompt key: {body.prompt_key}"}
        
    async with get_db_context() as db:
        query = text("""
            SELECT filename, transcript, rom_data, video_transcript 
            FROM recordings 
            WHERE id = :rid AND user_id = :uid
        """)
        result = await db.execute(query, {"rid": body.recording_id, "uid": current_user["id"]})
        row = result.mappings().fetchone()
        
        if not row:
            return {"success": False, "error": "Recording not found"}
            
        settings_query = text("SELECT * FROM user_settings WHERE user_id = :uid")
        settings_result = await db.execute(settings_query, {"uid": current_user["id"]})
        user_settings = settings_result.mappings().fetchone()
        
    if not user_settings:
        return {"success": False, "error": "User settings not found"}

    transcript = from_json(row["transcript"])
    rom_data = from_json(row["rom_data"]) or {}
    video_transcript = from_json(row["video_transcript"])
    
    # Check dependencies and construct variables
    variables = {}
    window_info = None
    
    stage1_keys = ["rom_discussion_embedded", "rom_discussion", "rom_discussion_no_actions", "rom_action_extraction"]
    stage2_keys = ["rom_enhance_window", "rom_enhance_all_together"]
    stage3_keys = ["rom_agenda", "rom_agenda_assign_batch"]
    final_rom_keys = ["rom_version_short", "rom_version_medium"]
    mom_keys = ["mom", "mom_extract_actions"]
    
    if body.prompt_key in stage1_keys:
        if not transcript:
            return {"success": False, "error": "missing_dependency", "missing_dependency": "transcript", "required_stage": "0", "reason": "Transcript is required for Stage 1"}
            
        windows = get_sliding_windows(transcript, window_minutes=2.0)
        idx = body.window_index if body.window_index is not None and 0 <= body.window_index < len(windows) else 0
        
        window = windows[idx]
        window_info = {"index": idx, "start": window[0].get('start', 0.0) if window else 0.0, "end": window[-1].get('end', 0.0) if window else 0.0}
        
        variables["{window_text}"] = format_window_text(window)
        variables["{previous_context}"] = ""
        if idx > 0:
            variables["{previous_context}"] = "PREVIOUS TRANSCRIPT WINDOW:\n" + format_window_text(windows[idx-1])
            
        variables["{video_context_section}"] = ""
        if video_transcript:
            variables["{video_context_section}"] = "VIDEO OCR TRANSCRIPT (supplementary — audio transcript is primary):\n[Placeholder for OCR]"
            
    elif body.prompt_key in stage2_keys:
        stage1_points = rom_data.get("discussion_points", [])
        if not stage1_points:
            return {"success": False, "error": "missing_dependency", "missing_dependency": "stage1_points", "required_stage": "1", "reason": "Stage 1 discussion points are required for Stage 2"}
            
        if body.prompt_key == "rom_enhance_window":
            idx = body.window_index or 0
            batch = stage1_points[idx:idx+5]
            variables["{window_json}"] = json.dumps(batch, indent=2)
            variables["{meeting_context}"] = "[Arena Placeholder: Full FAISS RAG context omitted in test mode]"
            variables["{global_context}"] = "[Arena Placeholder: Full Global FAISS RAG context omitted in test mode]"
        else:
            variables["{points_json}"] = json.dumps(stage1_points, indent=2)
            variables["{meeting_context}"] = "[Arena Placeholder: Full FAISS RAG context omitted in test mode]"
            variables["{global_context}"] = "[Arena Placeholder: Full Global FAISS RAG context omitted in test mode]"
            
    elif body.prompt_key in final_rom_keys:
        agendas = rom_data.get("final_rom", {}).get("agendas", [])
        if not agendas:
            return {"success": False, "error": "missing_dependency", "missing_dependency": "final_rom_agendas", "required_stage": "3", "reason": "Stage 3 Agendas are required for Final ROM"}
            
        first_agenda = agendas[0]
        variables["{agenda_title}"] = first_agenda.get("title", "Agenda")
        variables["{points_json}"] = json.dumps(first_agenda.get("discussion_points", []), indent=2)
        variables["{rules_section}"] = ""
        variables["{mandatory_section}"] = ""
        
    elif body.prompt_key in mom_keys:
        stage2_points = rom_data.get("enhanced_discussion_points", [])
        if not stage2_points:
            return {"success": False, "error": "missing_dependency", "missing_dependency": "enhanced_points", "required_stage": "2", "reason": "Stage 2 enhanced points are required for MoM actions"}
        variables["{points_json}"] = json.dumps(stage2_points, indent=2)
        variables["{transcript}"] = "\n".join([seg.get("text", "") for seg in transcript]) if transcript else ""
        
    else:
        variables["{transcript}"] = "\n".join([seg.get("text", "") for seg in transcript]) if transcript else ""

    # Assemble final prompt text
    final_prompt = body.prompt_text
    for var_name, var_value in variables.items():
        final_prompt = final_prompt.replace(var_name, var_value)
        
    # Call Ollama
    server_url = normalize_ollama_url(user_settings.get("ollama_server_url") or "http://localhost:11434")
    
    # Try to grab max_tokens for this prompt key, fallback to 4096
    max_tokens = user_settings.get(f"max_tokens_{body.prompt_key}") or 4096
    
    payload = {
        "model": body.model_name,
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
        with urllib.request.urlopen(req, timeout=120) as response:
            if response.status == 200:
                resp_data = json.loads(response.read().decode('utf-8'))
                output = resp_data.get("message", {}).get("content", "")
                
                timing = {
                    "total_duration": resp_data.get("total_duration"),
                    "prompt_eval_count": resp_data.get("prompt_eval_count"),
                    "eval_count": resp_data.get("eval_count")
                }
                
                return {
                    "success": True,
                    "model": body.model_name,
                    "recording_name": row["filename"],
                    "window_info": window_info,
                    "final_prompt": final_prompt,
                    "input_context": variables,
                    "output": output,
                    "timing": timing,
                    "warnings": []
                }
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
        settings_query = text("SELECT ollama_server_url FROM user_settings WHERE user_id = :uid")
        settings_result = await db.execute(settings_query, {"uid": current_user["id"]})
        user_settings = settings_result.mappings().fetchone()
        
    if not user_settings:
        return {"success": False, "error": "User settings not found"}
        
    server_url = normalize_ollama_url(user_settings.get("ollama_server_url") or "http://localhost:11434")
    
    meta_prompt = f"""
    Based on the following natural language description, generate a well-written, concise instruction rule for an LLM prompt.
    Do not include any conversational filler, just return the exact rule string.
    
    Description: {body.description}
    Context/Prompt: {body.context}
    """
    
    payload = {
        "model": body.model_name,
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
        with urllib.request.urlopen(req, timeout=30) as response:
            if response.status == 200:
                resp_data = json.loads(response.read().decode('utf-8'))
                rule = resp_data.get("message", {}).get("content", "").strip()
                return {"rule": rule}
            else:
                raise HTTPException(status_code=500, detail="Failed to generate rule")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/drafts")
async def list_arena_drafts(current_user: dict = Depends(get_current_user)):
    # Assuming there's a table arena_drafts. The user requested this, 
    # but we might need to create it or assume it exists in DB schema.
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
                    "template": row["template"],
                    "rule_states": from_json(row["rule_states"]) if row["rule_states"] else {},
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
                "draft_name": body.draft_name,
                "template": body.template,
                "rule_states": to_json(body.rule_states)
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

@router.get("/prompt-groups")
async def get_prompt_groups(current_user: dict = Depends(get_current_user)):
    GROUPS = [
        {"group": "Stage 1", "keys": ["rom_discussion_embedded", "rom_discussion", "rom_discussion_no_actions", "rom_action_extraction", "stage1_json_repair"]},
        {"group": "Stage 2", "keys": ["rom_enhance_window", "rom_enhance_all_together", "rom_deduplicate"]},
        {"group": "Stage 3", "keys": ["rom_agenda", "rom_mom_expansion", "rom_agenda_assign_batch", "rom_agenda_doc_points"]},
        {"group": "Final ROM", "keys": ["rom_version_short", "rom_version_medium"]},
        {"group": "Action Point from Enhanced Point", "keys": ["mom_extract_actions"]},
        {"group": "MoM", "keys": ["mom", "mom_merge", "mom_action_dedup"]},
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
