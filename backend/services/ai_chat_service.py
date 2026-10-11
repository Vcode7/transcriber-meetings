"""
ai_chat_service.py — Standalone AI Chat orchestration service.

Handles query triage, Stage 2 ChromaDB retrieval, prompt construction,
and LLM inference for the standalone AI Chat tab.

Retrieval source: ONLY Stage 2 Long Changes points from ChromaDB.
This module does NOT import from collection_ai_service.py.
"""
from __future__ import annotations

import os
import json
import logging
import re
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple, Set

logger = logging.getLogger(__name__)


# ── Prompt Templates ─────────────────────────────────────────────────────────

AI_CHAT_TRIAGE_PROMPT = """\
You are an intelligent AI triage agent for a meeting analysis platform.
Analyze the user's question and recent conversation history, then decide what response strategy is required.
Do NOT assume any specific meeting list at this stage.

You MUST choose one of three response categories:

Category 1 — "none" (NO meeting context needed):
- The question is a greeting (e.g. "Hello"), general chat, meta-question about your capabilities, or common knowledge.
- Response format:
```json
{{
  "context_required": false,
  "retrieval_mode": "none",
  "detail": "Polite, helpful final answer to the user's question."
}}
```

Category 2 — "direct_meeting" (User is asking about a SPECIFIC single meeting):
- The user is asking about a specific meeting by name, topic, date, or relative reference (e.g. "Summarize the Sprint 42 Retrospective", "What did John say in yesterday's sync?", "What happened in the meeting with Acme Corp?", "Give me action items from the kickoff call", "What decisions were made in the latest architecture review?").
- Response format:
```json
{{
  "context_required": true,
  "retrieval_mode": "direct_meeting",
  "meeting_reference": "Extracted name, title, keywords, topic, or date reference of the meeting (e.g. 'Sprint 42 Retrospective', 'Acme Corp kickoff', 'yesterday sync')",
  "detail": "Concise summary of what the user wants to know from that meeting."
}}
```

Category 3 — "semantic_search" (User is asking a broader question across MULTIPLE meetings or general topic):
- The question asks about a topic, keyword, or decision that could span across multiple meetings, or general research across meetings (e.g. "What was discussed about budget?", "Find all mentions of Kubernetes", "What are all action items across recent meetings?").
- Response format:
```json
{{
  "context_required": true,
  "retrieval_mode": "semantic_search",
  "detail": "Targeted search keywords and entities for vector similarity search."
}}
```

{conversation_history}

USER QUESTION: {question}

JSON RESPONSE:"""


AI_CHAT_MEETING_SELECT_PROMPT = """\
You are an AI meeting identification assistant.
The user wants to retrieve context from a specific meeting.

USER QUESTION: {question}
MEETING REFERENCE SOUGHT: "{meeting_reference}"

Here is the list of candidate meetings:
{candidate_meetings_text}

Task:
Identify which single meeting from the candidate list corresponds to what the user is referring to.
- If a candidate clearly matches the user's reference, set "matched": true and provide its exact ID and name.
- If NO candidate matches, set "matched": false.

You MUST respond with valid JSON strictly adhering to this schema:
```json
{{
  "matched": true | false,
  "meeting_id": "<exact_id_from_list>" | null,
  "meeting_name": "<exact_name_from_list>" | null,
  "reason": "Brief reason for selection or why none matched"
}}
```

JSON RESPONSE:"""


AI_CHAT_ANSWER_PROMPT = """\
You are an expert meeting analyst assistant. Answer the user's question using ONLY the meeting discussion points provided below.

Rules:
- Base your answer ONLY on the provided context. Do NOT make up information.
- When referencing information from a specific meeting, cite it: [Meeting: <meeting_name>]
- If the context does not contain enough information to answer, say so clearly.
- Preserve all technical terms, project names, and acronyms exactly.
- Format your response in clear Markdown with headers, bullets, and bold text where appropriate.

{conversation_history}

MEETING DISCUSSION POINTS:
{context}

USER QUESTION: {question}

Answer:"""


# ── Direct Ollama Configuration & Invocation ─────────────────────────────────

def get_ai_chat_ollama_config() -> dict:
    """
    Load standalone AI Chat LLM configuration directly from environment / settings.
    Bypasses SQLite user_settings table to ensure Ollama is never skipped due to
    multi-user or client settings overwriting use_ollama.
    """
    import os
    import urllib.parse
    import re
    from config import settings

    def _bool_val(key: str, default: bool) -> bool:
        v = os.getenv(key)
        if v is None:
            return bool(getattr(settings, key, default))
        return v.strip().lower() in ("true", "1", "yes")

    def _str_val(key: str, default: str) -> str:
        v = os.getenv(key)
        if v is None:
            return str(getattr(settings, key, default))
        return v.strip()

    def _int_val(key: str, default: int) -> int:
        v = os.getenv(key)
        if v is None:
            return int(getattr(settings, key, default))
        try:
            return int(v.strip())
        except ValueError:
            return default

    def _float_val(key: str, default: float) -> float:
        v = os.getenv(key)
        if v is None:
            return float(getattr(settings, key, default))
        try:
            return float(v.strip())
        except ValueError:
            return default

    server_url = _str_val("AI_CHAT_OLLAMA_URL", "http://localhost:11434")
    if not server_url.startswith("http://") and not server_url.startswith("https://"):
        server_url = f"http://{server_url}"
    server_url = server_url.rstrip("/")
    try:
        parsed = urllib.parse.urlparse(server_url)
        if not parsed.port and (
            parsed.hostname in ("localhost", "127.0.0.1")
            or (parsed.hostname and re.match(r"^\d+\.\d+\.\d+\.\d+$", parsed.hostname))
        ):
            server_url = f"{server_url}:11434"
    except Exception:
        pass

    return {
        "use_ollama": _bool_val("AI_CHAT_USE_OLLAMA", True),
        "ollama_server_url": server_url,
        "ollama_model_priority": _str_val(
            "AI_CHAT_OLLAMA_MODEL_PRIORITY",
            "hf.co/prithivMLmods/gemma-4-E4B-it-qat-GGUF:Q3_K_S,qwen2.5:7b,qwen2.5:14b,llama3.1:8b,mistral:7b",
        ),
        "ollama_dynamic_ctx": _bool_val("AI_CHAT_OLLAMA_DYNAMIC_CTX", True),
        "ollama_num_ctx": _int_val("AI_CHAT_OLLAMA_NUM_CTX", 32768),
        "ollama_temperature": _float_val("AI_CHAT_OLLAMA_TEMPERATURE", 0.2),
        "ollama_repeat_penalty": 1.1,
        "ollama_top_p": 0.9,
        "ollama_top_k": 40,
        "ollama_seed": None,
        "ollama_stop": None,
        "ollama_num_thread": None,
        "ollama_num_gpu": None,
        "ollama_keep_alive": "5m",
        "ollama_think": False,
    }


def _call_ollama_direct(
    prompt: str,
    max_new_tokens: int = 512,
    system_content: Optional[str] = None,
    cfg: Optional[dict] = None,
) -> Optional[str]:
    """Call Ollama directly via HTTP without checking SQLite database."""
    import urllib.request
    from services.ai_provider import QwenProvider, calculate_dynamic_num_ctx, strip_thinking

    if cfg is None:
        cfg = get_ai_chat_ollama_config()

    if not cfg.get("use_ollama"):
        return None

    server_url = cfg["ollama_server_url"]
    priority_list = cfg["ollama_model_priority"]

    model_name = QwenProvider._detect_ollama_model(server_url, priority_list)
    if not model_name:
        logger.warning(
            f"[AIChat] Direct Ollama enabled, but no model found at {server_url} with priority '{priority_list}'"
        )
        return None

    dynamic_enabled = bool(cfg.get("ollama_dynamic_ctx", True))
    manual_num_ctx = int(cfg.get("ollama_num_ctx", 32768))

    est_input_tokens, calculated_num_ctx = calculate_dynamic_num_ctx(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        safety_buffer=512,
        system_content=system_content or "",
        tokenizer=QwenProvider._tokenizer,
    )
    selected_num_ctx = calculated_num_ctx if dynamic_enabled else manual_num_ctx

    options = {
        "num_predict": max_new_tokens,
        "temperature": float(cfg.get("ollama_temperature", 0.2)),
        "num_ctx": selected_num_ctx,
        "repeat_penalty": float(cfg.get("ollama_repeat_penalty", 1.1)),
        "top_p": float(cfg.get("ollama_top_p", 0.9)),
        "top_k": int(cfg.get("ollama_top_k", 40)),
        "think": bool(cfg.get("ollama_think", False)),
    }

    messages = []
    if system_content:
        messages.append({"role": "system", "content": system_content})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model_name,
        "messages": messages,
        "options": options,
        "stream": False,
        "think": bool(cfg.get("ollama_think", False)),
    }

    base_url = server_url.rstrip("/")
    url = f"{base_url}/api/chat"

    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120.0) as resp:
            if resp.status == 200:
                resp_data = json.loads(resp.read().decode("utf-8"))
                content = resp_data.get("message", {}).get("content", "")
                logger.info(
                    f"[AIChat] Direct Ollama inference successful ({len(content)} chars) with model '{model_name}'"
                )
                return strip_thinking(content).strip()
            else:
                logger.warning(f"[AIChat] Direct Ollama returned HTTP status {resp.status}")
    except Exception as e:
        logger.warning(f"[AIChat] Direct Ollama inference failed: {e}")
        return None

    return None


# ── Triage Result & Helpers ──────────────────────────────────────────────────

class TriageResult:
    """
    Structured outcome of the query triage process.
    Supports unpacking as a 2-tuple (context_required, detail) for 100% backward compatibility
    with existing tests and callers, while exposing rich context routing attributes.
    """
    def __init__(
        self,
        context_required: bool,
        detail: str,
        retrieval_mode: str = "semantic_search",  # "direct_meeting" | "semantic_search" | "none"
        target_meeting_id: Optional[str] = None,
        target_meeting_name: Optional[str] = None,
    ):
        self.context_required = context_required
        self.detail = detail
        self.retrieval_mode = retrieval_mode
        self.target_meeting_id = target_meeting_id
        self.target_meeting_name = target_meeting_name

    def __iter__(self):
        yield self.context_required
        yield self.detail

    def __getitem__(self, index: int):
        if index == 0:
            return self.context_required
        elif index == 1:
            return self.detail
        raise IndexError("TriageResult only supports index 0 (context_required) and 1 (detail)")

    def __len__(self) -> int:
        return 2

    def __repr__(self) -> str:
        return (
            f"TriageResult(context_required={self.context_required}, "
            f"retrieval_mode='{self.retrieval_mode}', "
            f"target_meeting_id={self.target_meeting_id!r}, "
            f"target_meeting_name={self.target_meeting_name!r}, "
            f"detail='{self.detail[:40]}...')"
        )


def format_candidate_meetings_for_prompt(candidates: List[Dict[str, Any]]) -> str:
    """Format candidate meetings for the second LLM meeting selection call."""
    if not candidates:
        return "(No candidate meetings available)"
    lines = []
    for m in candidates:
        mid = m.get("id", "")
        name = m.get("name") or m.get("title") or "Meeting"
        date = m.get("date", "")
        date_str = f" | Date: {date}" if date else ""
        lines.append(f'- ID: "{mid}" | Name: "{name}"{date_str}')
    return "\n".join(lines)


def filter_candidate_meetings(
    reference: str,
    question: str,
    available_meetings: Optional[List[Dict[str, Any]]],
    max_candidates: int = 30,
) -> List[Dict[str, Any]]:
    """
    Filter available meetings (which could be up to 1000) down to the most relevant
    candidates for the meeting selection LLM call.
    """
    if not available_meetings:
        return []

    if len(available_meetings) <= max_candidates:
        return available_meetings

    ref_lower = (reference or "").lower()
    q_lower = (question or "").lower()
    ref_tokens = set(re.findall(r"\b[a-zA-Z0-9_\-]+\b", ref_lower + " " + q_lower))

    scored = []
    for idx, m in enumerate(available_meetings):
        mid = str(m.get("id", "")).lower()
        mname = str(m.get("name") or m.get("title") or "").strip().lower()
        mdate = str(m.get("date", "")).strip().lower()

        score = 0.0
        # Exact ID match
        if mid and (mid in ref_lower or mid in q_lower):
            score += 100.0

        # Exact title substring match
        if mname and len(mname) >= 3:
            if mname in ref_lower or mname in q_lower:
                score += 50.0
            elif ref_lower in mname:
                score += 30.0

            # Token overlap
            m_tokens = set(re.findall(r"\b[a-zA-Z0-9_\-]+\b", mname))
            overlap = ref_tokens.intersection(m_tokens)
            score += len(overlap) * 5.0

        # Date match
        if mdate:
            simple_date = mdate.split("T")[0] if "T" in mdate else mdate.split(" ")[0]
            if simple_date in ref_lower or simple_date in q_lower:
                score += 20.0

        # Recency bonus for the top 5 newest meetings
        if idx < 5:
            score += (5 - idx) * 0.5

        scored.append((score, idx, m))

    scored.sort(key=lambda x: (x[0], -x[1]), reverse=True)
    return [m for _, _, m in scored[:max_candidates]]


def select_meeting_from_list(
    question: str,
    meeting_reference: str,
    available_meetings: Optional[List[Dict[str, Any]]],
) -> Tuple[Optional[str], Optional[str]]:
    """
    Second LLM call: Select the exact meeting from the candidate meetings list
    based on the user's question and extracted meeting reference.
    Returns (meeting_id, meeting_name) or (None, None).
    """
    from services.ai_provider import get_provider

    if not available_meetings:
        return None, None

    # Filter down to top candidates (at most 30) so 1000 meetings never blow up prompt tokens
    candidates = filter_candidate_meetings(
        reference=meeting_reference,
        question=question,
        available_meetings=available_meetings,
        max_candidates=30,
    )

    if not candidates:
        return None, None

    # Check for direct deterministic match first as a fast-path
    exact_match = find_matching_meeting(meeting_reference or question, candidates)
    if exact_match and len(candidates) == 1:
        return exact_match.get("id"), (exact_match.get("name") or exact_match.get("title"))

    cand_text = format_candidate_meetings_for_prompt(candidates)
    prompt = AI_CHAT_MEETING_SELECT_PROMPT.format(
        question=question,
        meeting_reference=meeting_reference or question,
        candidate_meetings_text=cand_text,
    )

    try:
        raw_resp = _call_ollama_direct(
            prompt,
            max_new_tokens=200,
            system_content="You are an AI meeting selector. You MUST respond with valid JSON strictly according to instructions.",
        )
        if not raw_resp:
            provider = get_provider()
            raw_resp = provider._infer(prompt, max_new_tokens=200)

        if raw_resp:
            clean = raw_resp.strip()
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', clean, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
            else:
                first = clean.find('{')
                last = clean.rfind('}')
                json_str = clean[first:last + 1] if (first != -1 and last > first) else clean

            data = json.loads(json_str)
            if data.get("matched"):
                sel_id = data.get("meeting_id")
                sel_name = data.get("meeting_name")
                for c in candidates:
                    if str(c.get("id")) == str(sel_id):
                        logger.info(f"[AIChat Meeting Select] LLM selected meeting '{c.get('name')}' (ID: {sel_id})")
                        return str(sel_id), (c.get("name") or c.get("title"))
                if sel_name:
                    for c in candidates:
                        c_title = c.get("name") or c.get("title") or ""
                        if sel_name.lower() in c_title.lower() or c_title.lower() in sel_name.lower():
                            logger.info(f"[AIChat Meeting Select] Matched meeting by name '{c_title}' (ID: {c.get('id')})")
                            return str(c.get("id")), c_title
    except Exception as e:
        logger.warning(f"[AIChat Meeting Select] LLM meeting selection call failed: {e}")

    # Fallback to deterministic matcher if LLM selection was inconclusive
    if exact_match:
        logger.info(f"[AIChat Meeting Select] Fallback to deterministic match: {exact_match.get('name')} ({exact_match.get('id')})")
        return exact_match.get("id"), (exact_match.get("name") or exact_match.get("title"))

    return None, None


def is_meeting_listing_query(question: str) -> bool:
    """Detect if the user is asking to view or list their available meetings."""
    q = question.lower().strip()
    patterns = [
        r"\b(list|show|view|see|display|get)\b.*?\b(meetings?|recordings?|sessions?)\b",
        r"\bwhat\b.*?\bmeetings?\b.*?\b(available|recorded|do (i|we|you) have|exist|there)\b",
        r"\bwhich\b.*?\bmeetings?\b.*?\b(available|recorded|do (i|we|you) have|exist)\b",
        r"\bhow many\b.*?\bmeetings?\b",
        r"\bwhat are (my|the|our) meetings\b",
        r"\bmeetings?\s+(list|overview)\b",
        r"\bdo you have any meetings\b",
    ]
    for pat in patterns:
        if re.search(pat, q):
            return True
    return False


def format_meeting_list_response(meetings: Optional[List[Dict[str, Any]]]) -> str:
    """Generate a clean markdown list of meetings when user asks to see/list meetings."""
    if not meetings:
        return (
            "You don't have any recorded meetings yet. Once you record or upload a meeting, "
            "its discussion points will be analyzed and ready for AI Chat!"
        )

    total_count = len(meetings)
    display_limit = 20
    display_meetings = meetings[:display_limit]

    lines = [f"Here are your available meetings (showing {len(display_meetings)} of {total_count}):\n"]
    for idx, m in enumerate(display_meetings, start=1):
        name = m.get("name") or m.get("title") or "Meeting"
        date = m.get("date", "")
        date_str = f" *({date})*" if date else ""
        status_note = " *(Stage 2 points indexed)*" if m.get("has_stage2") else ""
        lines.append(f"{idx}. **{name}**{date_str}{status_note}")

    if total_count > display_limit:
        lines.append(f"\n*(...and {total_count - display_limit} more meetings recorded)*")

    lines.append(
        "\n💡 **How to query:**\n"
        "- Ask specific questions about any meeting (e.g. *\"Summarize key decisions in "
        f"{display_meetings[0].get('name', 'your recent meeting')}\"*)\n"
        "- Or ask general topic questions across all your meetings (e.g. *\"What action items were assigned to engineering?\"*)."
    )
    return "\n".join(lines)


def find_matching_meeting(
    question: str,
    available_meetings: Optional[List[Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:
    """
    Identify if the user explicitly mentions a specific meeting from their available list.
    Checks exact ID, full title, title keywords, and explicit date references.
    """
    if not available_meetings:
        return None

    q_lower = question.lower()
    q_tokens = set(re.findall(r"\b[a-zA-Z0-9_\-]+\b", q_lower))

    best_match = None
    best_score = 0.0

    # Common generic filler words in meeting titles to downweight
    generic_words = {
        "meeting", "discussion", "call", "sync", "daily", "weekly",
        "monthly", "review", "session", "1-on-1", "catchup", "the",
        "a", "an", "and", "or", "to", "for", "with"
    }

    for m in available_meetings:
        mid = str(m.get("id", "")).lower()
        mname = str(m.get("name") or m.get("title") or "").strip()
        mdate = str(m.get("date", "")).strip()

        # 1. Exact ID in query
        if mid and mid in q_lower:
            return m

        if not mname:
            continue

        name_lower = mname.lower()
        # 2. Exact or substring title match in query
        if len(name_lower) >= 4 and name_lower in q_lower:
            score = 100.0 + len(name_lower)
            if score > best_score:
                best_score = score
                best_match = m
            continue

        # 3. Clean title without common generic terms
        clean_name = re.sub(r"[^\w\s]", " ", name_lower)
        name_words = [w for w in clean_name.split() if w and w not in generic_words]

        if name_words:
            matched_words = [w for w in name_words if w in q_tokens or (len(w) >= 4 and w in q_lower)]
            match_ratio = len(matched_words) / len(name_words)

            if match_ratio == 1.0 or (len(matched_words) >= 2 and match_ratio >= 0.5):
                score = 50.0 + (len(matched_words) * 10.0)
                if score > best_score:
                    best_score = score
                    best_match = m

        # 4. Explicit date match if query mentions "meeting on YYYY-MM-DD" or similar
        if mdate:
            simple_date = mdate.split("T")[0] if "T" in mdate else mdate.split(" ")[0]
            if len(simple_date) >= 8 and simple_date in q_lower and ("meeting" in q_lower or "call" in q_lower):
                score = 40.0
                if score > best_score:
                    best_score = score
                    best_match = m

    return best_match


def _safe_parse_rom_data(raw: Any) -> dict:
    """Safely parse ROM data from dict, JSON string, or nested JSON string."""
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, str):
            try:
                parsed = json.loads(parsed)
            except Exception:
                return {}
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _extract_stage2_points(rom_data: dict) -> List[dict]:
    """Extract valid Stage 2 discussion points from parsed rom_data."""
    if not isinstance(rom_data, dict):
        return []
    stage2 = rom_data.get("stage2")
    raw_points = []
    if isinstance(stage2, dict):
        raw_points = (
            stage2.get("polished_points")
            or stage2.get("discussion_points")
            or stage2.get("points")
            or []
        )
    elif isinstance(stage2, list):
        raw_points = stage2

    valid_points = []
    if isinstance(raw_points, list):
        for p in raw_points:
            if isinstance(p, dict):
                text = (p.get("polished_text") or p.get("discussion_point") or "").strip()
                if text:
                    valid_points.append(p)
            elif isinstance(p, str) and p.strip():
                valid_points.append({"polished_text": p.strip()})
    return valid_points


def is_detailed_query(query: str) -> bool:
    """
    Check if the user explicitly asks for detailed information, full discussion,
    or Stage 2 level points for a meeting/topic.
    """
    if not query:
        return False
    q = query.lower()
    detail_keywords = [
        "detail", "detailed", "in-depth", "in depth", "deep dive", "exhaustive",
        "stage 2", "stage2", "long change", "long point", "verbatim", "comprehensive",
        "step by step", "step-by-step", "minute by minute", "minute-by-minute",
        "full discussion", "raw points", "raw discussion", "transcript quote",
        "transcript quotes", "more details", "additional detail", "give me details"
    ]
    return any(kw in q for kw in detail_keywords)


def _extract_final_rom_points(rom_dict: Optional[Dict]) -> List[Dict[str, Any]]:
    """Extract flattened discussion points from Final ROM agendas."""
    if not isinstance(rom_dict, dict):
        return []
    final_rom = rom_dict.get("final_rom") or {}
    agendas = final_rom.get("agendas") or []
    flat = []
    for a_idx, ag in enumerate(agendas):
        if not isinstance(ag, dict):
            continue
        ag_id = str(ag.get("agenda_id") or f"agenda_{a_idx}")
        ag_title = str(ag.get("title") or ag.get("agenda_title") or "").strip()
        pts = ag.get("discussion_points") or []
        for p in pts:
            if isinstance(p, dict):
                text = (p.get("text") or p.get("polished_text") or p.get("discussion_point") or "").strip()
                if text:
                    flat.append({
                        "id": p.get("id"),
                        "text": text,
                        "agenda_id": ag_id,
                        "agenda_title": ag_title,
                        "speakers": p.get("speakers") or p.get("speaker"),
                        "action_owner": p.get("action_owner"),
                    })
            elif isinstance(p, str) and p.strip():
                flat.append({
                    "id": str(uuid.uuid4()),
                    "text": p.strip(),
                    "agenda_id": ag_id,
                    "agenda_title": ag_title,
                    "speakers": None,
                    "action_owner": None,
                })
    return flat


def get_direct_meeting_context(
    user_id: str,
    meeting_id: str,
    max_results: int = 35,
    max_total_chars: int = 25000,
    is_detailed: bool = False,
) -> List[Dict[str, Any]]:
    """
    Directly retrieve discussion points from a specific meeting WITHOUT similarity search.

    Strict Retrieval Priority:
    1. Priority 1: If the meeting has Final ROM points and is_detailed is False,
       retrieve Final ROM points as primary context.
    2. Priority 2 / Detailed: If the meeting has no Final ROM points OR is_detailed is True,
       retrieve Stage 2 points.
    3. Handles SQLite fallback and auto-indexing into ChromaDB if missing.
    """
    from services.vector_store import get_final_rom_points_store, get_stage2_points_store
    results: List[Dict[str, Any]] = []

    if not meeting_id or not user_id:
        return []

    # Priority 1: Check Final ROM points first (unless detailed query requested)
    if not is_detailed:
        try:
            store = get_final_rom_points_store(user_id)
            if store.exists():
                store.load_or_create()
                if store._collection is not None:
                    raw = store._collection.get(
                        where={"meeting_id": str(meeting_id)},
                        include=["documents", "metadatas"],
                    )
                    docs = raw.get("documents") or []
                    metas = raw.get("metadatas") or []
                    if docs and len(docs) > 0:
                        paired = []
                        for i in range(len(docs)):
                            doc_text = (docs[i] or "").strip()
                            if not doc_text:
                                continue
                            meta = dict(metas[i]) if i < len(metas) and metas[i] else {}
                            chunk_idx = meta.get("chunk_index", i)
                            try:
                                chunk_idx = int(chunk_idx)
                            except (ValueError, TypeError):
                                chunk_idx = i

                            clean_pt = meta.get("clean_text") or doc_text
                            agenda_title = meta.get("agenda_title", "")
                            spk = meta.get("speakers") or ""
                            if isinstance(spk, str) and spk:
                                spk = [s.strip() for s in spk.split(",") if s.strip()]
                            elif not isinstance(spk, list):
                                spk = []

                            is_manual = bool(
                                meta.get("source") == "manual_mom"
                                or str(meta.get("is_manual_mom", "")).lower() == "true"
                                or meta.get("is_manual_mom") is True
                                or str(meeting_id).startswith("mom_")
                            )
                            source_tag = "manual_mom" if is_manual else (meta.get("source") or "final_roim")
                            entry = {
                                **meta,
                                "_text": doc_text,
                                "clean_text": clean_pt,
                                "agenda_title": agenda_title,
                                "meeting_id": str(meeting_id),
                                "meeting_name": meta.get("meeting_name") or "Meeting",
                                "date": meta.get("date") or "",
                                "chunk_index": chunk_idx,
                                "speakers": spk,
                                "stage": "manual_mom" if is_manual else "final_roim",
                                "source": source_tag,
                                "is_manual_mom": is_manual,
                                "score": 1.0,
                                "retrieval_mode": "direct_meeting",
                            }
                            paired.append(entry)

                        paired.sort(key=lambda x: x.get("chunk_index", 0))

                        total_chars = 0
                        for item in paired:
                            tlen = len(item["_text"])
                            if total_chars + tlen > max_total_chars:
                                continue
                            results.append(item)
                            total_chars += tlen
                            if len(results) >= max_results:
                                break

                        if results:
                            logger.info(
                                f"[AIChat] Priority 1: Direct Final ROM retrieval for meeting {meeting_id}: "
                                f"{len(results)} points ({total_chars} chars)"
                            )
                            return results
        except Exception as e:
            logger.warning(f"[AIChat] Direct Final ROM retrieval failed for meeting {meeting_id}: {e}")

    # Fallback / Detailed: Fetch Stage 2 points from ChromaDB
    try:
        store = get_stage2_points_store(user_id)
        if store.exists():
            store.load_or_create()
            if store._collection is not None:
                raw = store._collection.get(
                    where={"meeting_id": str(meeting_id)},
                    include=["documents", "metadatas"],
                )
                docs = raw.get("documents") or []
                metas = raw.get("metadatas") or []

                if docs and len(docs) > 0:
                    paired = []
                    for i in range(len(docs)):
                        doc_text = (docs[i] or "").strip()
                        if not doc_text:
                            continue
                        meta = dict(metas[i]) if i < len(metas) and metas[i] else {}
                        chunk_idx = meta.get("chunk_index", i)
                        try:
                            chunk_idx = int(chunk_idx)
                        except (ValueError, TypeError):
                            chunk_idx = i

                        spk = meta.get("speakers") or ""
                        if isinstance(spk, str) and spk:
                            spk = [s.strip() for s in spk.split(",") if s.strip()]
                        elif not isinstance(spk, list):
                            spk = []

                        entry = {
                            **meta,
                            "_text": doc_text,
                            "clean_text": doc_text,
                            "meeting_id": str(meeting_id),
                            "meeting_name": meta.get("meeting_name") or "Meeting",
                            "date": meta.get("date") or "",
                            "chunk_index": chunk_idx,
                            "speakers": spk,
                            "stage": "stage2",
                            "source": "stage2",
                            "score": 1.0,
                            "retrieval_mode": "direct_meeting",
                        }
                        paired.append(entry)

                    paired.sort(key=lambda x: x.get("chunk_index", 0))

                    total_chars = 0
                    for item in paired:
                        tlen = len(item["_text"])
                        if total_chars + tlen > max_total_chars:
                            continue
                        results.append(item)
                        total_chars += tlen
                        if len(results) >= max_results:
                            break

                    if results:
                        logger.info(
                            f"[AIChat] Priority 2: Direct Stage 2 retrieval for meeting {meeting_id}: "
                            f"{len(results)} points ({total_chars} chars)"
                        )
                        return results
    except Exception as e:
        logger.warning(f"[AIChat] Direct Stage 2 retrieval failed for meeting {meeting_id}: {e}")

    # Fallback to SQLite recordings / rom_metadata table
    try:
        import sqlite3
        from config import settings
        db_url = settings.DATABASE_URL
        db_path = db_url.replace("sqlite+aiosqlite:///", "").replace("sqlite:///", "")
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT r.id, r.filename, r.title, r.created_at, "
                "       COALESCE(rm.rom_data, r.rom_data) as rom_data "
                "FROM recordings r "
                "LEFT JOIN rom_metadata rm ON rm.recording_id = r.id "
                "WHERE r.id = ? AND r.user_id = ?",
                (str(meeting_id), str(user_id)),
            )
            row = cursor.fetchone()
            if not row and str(meeting_id).startswith("mom_"):
                cursor.execute(
                    "SELECT id, filename, title, meeting_date, chunks_json, raw_text "
                    "FROM manual_moms WHERE id = ? AND user_id = ?",
                    (str(meeting_id), str(user_id)),
                )
                mom_row = cursor.fetchone()
                if mom_row:
                    m_id, filename, title, meeting_date, chunks_raw, raw_txt = mom_row
                    m_name = title or filename or "Manual MoM"
                    m_date = meeting_date or ""
                    chunks = []
                    try:
                        chunks = json.loads(chunks_raw) if chunks_raw else []
                    except Exception:
                        chunks = []
                    if not chunks and raw_txt:
                        chunks = [{"text": p.strip()} for p in raw_txt.split("\n\n") if p.strip()]

                    total_chars = 0
                    for idx, ch in enumerate(chunks):
                        txt = (ch.get("text") if isinstance(ch, dict) else str(ch)) or ""
                        txt = txt.strip()
                        if not txt or total_chars + len(txt) > max_total_chars:
                            continue
                        results.append({
                            "_text": txt,
                            "clean_text": txt,
                            "agenda_title": ch.get("agenda_title", "") if isinstance(ch, dict) else "",
                            "meeting_id": str(m_id),
                            "meeting_name": m_name,
                            "date": m_date,
                            "chunk_index": idx,
                            "speakers": ch.get("speakers", []) if isinstance(ch, dict) else [],
                            "stage": "manual_mom",
                            "source": "manual_mom",
                            "score": 1.0,
                            "retrieval_mode": "direct_meeting",
                        })
                        total_chars += len(txt)
                        if len(results) >= max_results:
                            break

                    if results:
                        logger.info(
                            f"[AIChat] Direct SQLite Manual MoM retrieval for {meeting_id}: "
                            f"{len(results)} points ({total_chars} chars)"
                        )
                        conn.close()
                        return results

            conn.close()

            if row:
                m_id, filename, title, created_at, rom_raw = row
                m_name = title or filename or "Meeting"
                m_date = created_at or ""
                rom_dict = _safe_parse_rom_data(rom_raw)

                # If not detailed, try extracting Final ROM points first
                if not is_detailed:
                    final_pts = _extract_final_rom_points(rom_dict)
                    if final_pts:
                        # Auto-index into ChromaDB so it's ready for future calls
                        try:
                            from services.rom_service import rom_service
                            rom_service.index_final_rom_points_in_chromadb(
                                recording_id=str(meeting_id),
                                user_id=str(user_id),
                                final_rom=rom_dict.get("final_rom", {}),
                                meeting_name=m_name,
                                meeting_date=m_date,
                            )
                        except Exception as ex:
                            logger.warning(f"[AIChat] Auto-indexing Final ROM from SQLite failed: {ex}")

                        total_chars = 0
                        for idx, pt in enumerate(final_pts):
                            pt_text = pt["text"]
                            ag_title = pt.get("agenda_title", "")
                            doc_text = f"[{ag_title}] {pt_text}" if ag_title else pt_text
                            if total_chars + len(doc_text) > max_total_chars:
                                continue
                            spk = pt.get("speakers") or []
                            if isinstance(spk, str):
                                spk = [s.strip() for s in spk.split(",") if s.strip()]

                            results.append({
                                "_text": doc_text,
                                "clean_text": pt_text,
                                "agenda_title": ag_title,
                                "meeting_id": str(m_id),
                                "meeting_name": m_name,
                                "date": m_date,
                                "chunk_index": idx,
                                "speakers": spk,
                                "stage": "final_roim",
                                "source": "final_roim",
                                "score": 1.0,
                                "retrieval_mode": "direct_meeting",
                            })
                            total_chars += len(doc_text)
                            if len(results) >= max_results:
                                break

                        if results:
                            logger.info(
                                f"[AIChat] Direct SQLite Final ROM retrieval for {meeting_id}: "
                                f"{len(results)} points ({total_chars} chars)"
                            )
                            return results

                # Fallback to Stage 2 points in SQLite
                pts = _extract_stage2_points(rom_dict)
                if pts:
                    # Auto-index into ChromaDB
                    try:
                        from services.rom_service import rom_service
                        rom_service.index_stage2_points_in_chromadb(
                            recording_id=str(meeting_id),
                            user_id=str(user_id),
                            points=pts,
                            meeting_name=m_name,
                            meeting_date=m_date,
                        )
                    except Exception as ex:
                        logger.warning(f"[AIChat] Auto-indexing Stage 2 from SQLite failed: {ex}")

                    total_chars = 0
                    for idx, pt in enumerate(pts):
                        pt_text = (pt.get("polished_text") or pt.get("discussion_point") or "").strip()
                        if not pt_text:
                            continue
                        if total_chars + len(pt_text) > max_total_chars:
                            continue
                        spk = pt.get("speakers") or []
                        if isinstance(spk, str):
                            spk = [s.strip() for s in spk.split(",") if s.strip()]

                        results.append({
                            "_text": pt_text,
                            "clean_text": pt_text,
                            "meeting_id": str(m_id),
                            "meeting_name": m_name,
                            "date": m_date,
                            "chunk_index": idx,
                            "speakers": spk,
                            "stage": "stage2",
                            "source": "stage2",
                            "score": 1.0,
                            "retrieval_mode": "direct_meeting",
                        })
                        total_chars += len(pt_text)
                        if len(results) >= max_results:
                            break

                    if results:
                        logger.info(
                            f"[AIChat] Direct SQLite Stage 2 retrieval for {meeting_id}: "
                            f"{len(results)} points ({total_chars} chars)"
                        )
                        return results
        else:
            conn.close()
    except Exception as e:
        logger.warning(f"[AIChat] Direct SQLite retrieval failed for meeting {meeting_id}: {e}")

    if not results:
        logger.warning(
            f"[AIChat Context Warning] Direct meeting context retrieval yielded 0 discussion points for meeting ID '{meeting_id}' (user: {user_id}). "
            f"ChromaDB stores had no matching points and SQLite database had no valid ROM/Stage 2 discussion points for this meeting."
        )

    return results


# ── Triage ───────────────────────────────────────────────────────────────────

def triage_query(
    question: str,
    chat_history: List[Dict],
    available_meetings: Optional[List[Dict[str, Any]]] = None,
    user_id: Optional[str] = None,
) -> TriageResult:
    """
    Send the user's question to the LLM to determine context retrieval strategy.

    Returns a TriageResult object:
    - Supports unpacking as (context_required, detail) for backward compatibility
    - Contains attributes:
        .context_required: bool
        .retrieval_mode: 'direct_meeting' | 'semantic_search' | 'none'
        .target_meeting_id: Optional[str]
        .target_meeting_name: Optional[str]
        .detail: str
    """
    from services.ai_provider import get_provider

    # If available meetings not supplied but user_id is, load available meetings
    if available_meetings is None and user_id:
        try:
            available_meetings = get_available_meetings(user_id)
        except Exception as e:
            logger.warning(f"[AIChat Triage] Failed to load available meetings: {e}")
            available_meetings = []

    # 1. Check if user is asking to view/list available meetings
    if is_meeting_listing_query(question):
        list_response = format_meeting_list_response(available_meetings)
        logger.info("[AIChat Triage] Detected meeting listing query. Returning direct list response.")
        return TriageResult(
            context_required=False,
            detail=list_response,
            retrieval_mode="none",
        )

    # 2. Check for explicit meeting mention in question
    explicit_match = find_matching_meeting(question, available_meetings)
    if explicit_match:
        logger.info(
            f"[AIChat Triage] Explicit meeting mention detected: "
            f"'{explicit_match.get('name')}' (ID: {explicit_match.get('id')})"
        )

    # 3. Format history and available meetings
    history_text = ""
    if chat_history:
        recent = chat_history[-4:]
        lines = ["PREVIOUS CONVERSATION:"]
        for msg in recent:
            role = "User" if msg.get("role") == "user" else "Assistant"
            content = msg.get("content", "")
            if len(content) > 300:
                content = content[:300] + "..."
            lines.append(f"{role}: {content}")
        history_text = "\n".join(lines)

    prompt = AI_CHAT_TRIAGE_PROMPT.format(
        conversation_history=history_text,
        question=question,
    )

    try:
        # 1. First Call: Lightweight intent & retrieval strategy triage (NO meeting list)
        raw_resp = _call_ollama_direct(
            prompt,
            max_new_tokens=300,
            system_content=(
                "You are an AI assistant for a meeting analysis platform. "
                "You MUST respond with valid JSON strictly according to instructions."
            ),
        )
        if not raw_resp:
            logger.info("[AIChat Triage] Direct Ollama not returned, falling back to provider._infer...")
            provider = get_provider()
            raw_resp = provider._infer(prompt, max_new_tokens=300)

        if raw_resp:
            clean = raw_resp.strip()
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', clean, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
            else:
                first = clean.find('{')
                last = clean.rfind('}')
                if first != -1 and last > first:
                    json_str = clean[first:last + 1]
                else:
                    json_str = clean

            try:
                data = json.loads(json_str)
                ctx_req = bool(data.get("context_required", True))
                retrieval_mode = str(data.get("retrieval_mode", "semantic_search")).strip().lower()
                meeting_ref = str(data.get("meeting_reference") or "").strip()
                detail = str(data.get("detail", "")).strip()

                # If no context is required (greeting, general talk, meta-question)
                if not ctx_req or retrieval_mode == "none":
                    return TriageResult(
                        context_required=False,
                        detail=detail or question.strip(),
                        retrieval_mode="none",
                    )

                # Check if user question explicitly mentions an available meeting
                if explicit_match:
                    retrieval_mode = "direct_meeting"
                    if not meeting_ref:
                        meeting_ref = explicit_match.get("name") or explicit_match.get("title") or question

                # 2. Second Call: If specific meeting is required, select meeting from available list
                if retrieval_mode == "direct_meeting":
                    if available_meetings is None and user_id:
                        available_meetings = get_available_meetings(user_id)

                    target_mid, target_name = select_meeting_from_list(
                        question=question,
                        meeting_reference=meeting_ref or question,
                        available_meetings=available_meetings,
                    )

                    if target_mid:
                        logger.info(
                            f"[AIChat Triage] Additional meeting selection call resolved: "
                            f"'{target_name}' (ID: {target_mid})"
                        )
                        return TriageResult(
                            context_required=True,
                            detail=detail or question.strip(),
                            retrieval_mode="direct_meeting",
                            target_meeting_id=target_mid,
                            target_meeting_name=target_name,
                        )
                    else:
                        logger.info(
                            f"[AIChat Triage] Meeting '{meeting_ref}' could not be matched from list. "
                            "Falling back to semantic search."
                        )
                        return TriageResult(
                            context_required=True,
                            detail=detail or question.strip(),
                            retrieval_mode="semantic_search",
                        )

                # 3. Semantic search across meetings
                return TriageResult(
                    context_required=True,
                    detail=detail or question.strip(),
                    retrieval_mode="semantic_search",
                )

            except Exception:
                logger.warning(f"[AIChat Triage] JSON parse failed: '{clean[:100]}'")

            if explicit_match:
                return TriageResult(
                    context_required=True,
                    detail=question.strip(),
                    retrieval_mode="direct_meeting",
                    target_meeting_id=explicit_match.get("id"),
                    target_meeting_name=explicit_match.get("name") or explicit_match.get("title"),
                )

            return TriageResult(
                context_required=True,
                detail=clean,
                retrieval_mode="semantic_search",
            )
    except Exception as e:
        logger.warning(f"[AIChat Triage] Failed: {e}. Using raw query.")

    if explicit_match:
        return TriageResult(
            context_required=True,
            detail=question.strip(),
            retrieval_mode="direct_meeting",
            target_meeting_id=explicit_match.get("id"),
            target_meeting_name=explicit_match.get("name") or explicit_match.get("title"),
        )

    return TriageResult(
        context_required=True,
        detail=question.strip(),
        retrieval_mode="semantic_search",
    )



# ── ChromaDB Search ──────────────────────────────────────────────────────────

def _get_embedder():
    """Lazy-import the text embedder."""
    from services.text_embedding_service import get_text_embedder
    embedder = get_text_embedder()
    embedder.load()
    return embedder


def search_meeting_points(
    query: str,
    user_id: str,
    meeting_id: Optional[str] = None,
    max_results: int = 8,
    max_total_chars: int = 15000,
    is_detailed: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    """
    Search discussion points across meetings in ChromaDB adhering to strict retrieval priority:
    - Priority 1: If a meeting has Final ROM points, use Final ROM points as primary context.
    - Priority 2: If a meeting does not have Final ROM points, fall back to its Stage 2 points.
    - Detailed Meeting Queries: If the user explicitly asks for detailed information about a
      particular meeting/topic, Stage 2 points may be retrieved even when Final ROM points exist.
    - Do not use Stage 2 points by default when Final ROM points are available.
    """
    from services.vector_store import get_final_rom_points_store, get_stage2_points_store

    if is_detailed is None:
        is_detailed = is_detailed_query(query)

    embedder = _get_embedder()
    dim = embedder.embedding_dim()
    query_embedding = embedder.encode(query)

    where_filter = {"meeting_id": str(meeting_id)} if meeting_id else None

    # ── Specific Meeting Requested ──────────────────────────────────────────
    if meeting_id:
        if not is_detailed:
            final_store = get_final_rom_points_store(user_id, dim)
            if final_store.exists():
                final_results = final_store.search(
                    query_embedding,
                    k=max_results * 2,
                    score_threshold=0.0,
                    where=where_filter,
                )
                if final_results:
                    filtered = []
                    total_chars = 0
                    for r in final_results:
                        t = (r.get("clean_text") or r.get("_text", "")).strip()
                        if not t or total_chars + len(t) > max_total_chars:
                            continue
                        filtered.append(r)
                        total_chars += len(t)
                        if len(filtered) >= max_results:
                            break
                    if filtered:
                        logger.info(
                            f"[AIChat] Priority 1: Retrieved {len(filtered)} Final ROM points for meeting {meeting_id}"
                        )
                        return filtered

        # Fallback to Stage 2 for this specific meeting
        s2_store = get_stage2_points_store(user_id, dim)
        if s2_store.exists():
            s2_results = s2_store.search(
                query_embedding,
                k=max_results * 2,
                score_threshold=0.0,
                where=where_filter,
            )
            filtered = []
            total_chars = 0
            for r in s2_results:
                t = (r.get("clean_text") or r.get("_text", "")).strip()
                if not t or total_chars + len(t) > max_total_chars:
                    continue
                filtered.append(r)
                total_chars += len(t)
                if len(filtered) >= max_results:
                    break
            logger.info(
                f"[AIChat] Priority 2: Retrieved {len(filtered)} Stage 2 points for meeting {meeting_id} "
                f"(is_detailed={is_detailed})"
            )
            if filtered:
                return filtered
        logger.warning(
            f"[AIChat Context Warning] Similarity search on specific meeting '{meeting_id}' yielded 0 points. "
            f"Query: '{query[:80]}', is_detailed={is_detailed}. "
            f"ChromaDB stage2_store exists={s2_store.exists()}."
        )
        return []

    # ── Cross-Meeting Search Across All Meetings ─────────────────────────────
    final_store = get_final_rom_points_store(user_id, dim)
    s2_store = get_stage2_points_store(user_id, dim)

    final_exists = final_store.exists()
    s2_exists = s2_store.exists()
    if not final_exists and not s2_exists:
        logger.warning(
            f"[AIChat Context Warning] ChromaDB vector collections do not exist or are empty for user '{user_id}'. "
            f"Neither final_rom_{user_id} nor stage2_points_{user_id} contains indexed documents."
        )

    final_mids = get_final_rom_meeting_ids(user_id)

    final_results = []
    if final_exists:
        final_results = final_store.search(
            query_embedding,
            k=max_results * 2,
            score_threshold=0.0,
        )

    s2_results = []
    if s2_exists:
        s2_results = s2_store.search(
            query_embedding,
            k=max_results * 2,
            score_threshold=0.0,
        )

    # Filter Stage 2 candidates:
    # If a meeting already has Final ROM points, do not use its Stage 2 points UNLESS
    # the user explicitly asked for detailed information.
    eligible_s2 = []
    for r in s2_results:
        mid = str(r.get("meeting_id") or r.get("recording_id", ""))
        if not is_detailed and mid in final_mids:
            # Exclude: Final ROM exists for this meeting
            continue
        eligible_s2.append(r)

    # Combine candidates, prioritizing higher score
    combined = final_results + eligible_s2
    combined.sort(key=lambda x: x.get("score", 0.0), reverse=True)

    filtered = []
    total_chars = 0
    seen_points = set()

    for r in combined:
        pid = r.get("point_id") or r.get("id") or (r.get("_text", "")[:30])
        if pid in seen_points:
            continue
        seen_points.add(pid)
        text = (r.get("clean_text") or r.get("_text", "")).strip()
        if not text:
            continue
        if total_chars + len(text) > max_total_chars:
            continue
        filtered.append(r)
        total_chars += len(text)
        if len(filtered) >= max_results:
            break

    if not filtered:
        logger.warning(
            f"[AIChat Context Warning] Priority cross-meeting search returned 0 points for query: '{query[:80]}' (user: '{user_id}'). "
            f"Final ROM candidates: {len(final_results)}, Stage 2 candidates: {len(s2_results)}, Total combined before filter: {len(combined)}."
        )
    else:
        logger.info(
            f"[AIChat] Priority Search retrieved {len(filtered)} points "
            f"({total_chars} chars, is_detailed={is_detailed}) for query: '{query[:60]}...'"
        )
    return filtered


def search_stage2_points(
    query: str,
    user_id: str,
    meeting_id: Optional[str] = None,
    max_results: int = 8,
    max_total_chars: int = 15000,
) -> List[Dict[str, Any]]:
    """
    Search points adhering to the retrieval priority pipeline (Priority 1: Final ROM,
    Priority 2: Stage 2, Detailed override). Retained for full backward compatibility.
    """
    return search_meeting_points(
        query=query,
        user_id=user_id,
        meeting_id=meeting_id,
        max_results=max_results,
        max_total_chars=max_total_chars,
    )


def extract_meeting_sources(results: List[Dict]) -> List[Dict[str, Any]]:
    """
    Extract unique meeting-level sources from search results.
    Returns deduplicated list of {id, name, date, is_manual_mom}.
    """
    seen = {}
    for r in results:
        mid = r.get("meeting_id") or r.get("recording_id", "")
        if not mid or mid in seen:
            continue
        is_manual = bool(
            r.get("source") == "manual_mom"
            or str(r.get("is_manual_mom", "")).lower() == "true"
            or r.get("is_manual_mom") is True
            or str(mid).startswith("mom_")
        )
        seen[mid] = {
            "id": mid,
            "name": r.get("meeting_name", "Unknown Meeting"),
            "date": r.get("date", ""),
            "is_manual_mom": is_manual,
        }
    return list(seen.values())


def get_final_rom_meeting_ids(user_id: str) -> Set[str]:
    """
    Return the set of all meeting_ids that have Final ROM points indexed in ChromaDB.
    Uses dim=0 to query collection metadata directly without loading the text embedding model into VRAM.
    """
    from services.vector_store import get_final_rom_points_store

    store = get_final_rom_points_store(user_id, dim=0)

    if not store.exists():
        return set()

    try:
        store.load_or_create()
        if store._collection is None:
            return set()

        results = store._collection.get(include=["metadatas"])
        metadatas = results.get("metadatas", [])

        mids: Set[str] = set()
        for meta in metadatas:
            if not meta:
                continue
            mid = meta.get("meeting_id") or meta.get("recording_id", "")
            if mid:
                mids.add(str(mid))

        return mids
    except Exception as e:
        logger.error(f"[AIChat] Failed to get Final ROM meeting IDs: {e}")
        return set()


def get_stage2_meeting_ids(user_id: str) -> Set[str]:
    """
    Return the set of all meeting_ids that have Stage 2 points indexed in ChromaDB.
    Uses dim=0 to query collection metadata directly without loading the text embedding model into VRAM.
    """
    from services.vector_store import get_stage2_points_store

    store = get_stage2_points_store(user_id, dim=0)

    if not store.exists():
        return set()

    try:
        store.load_or_create()
        if store._collection is None:
            return set()

        results = store._collection.get(include=["metadatas"])
        metadatas = results.get("metadatas", [])

        mids: Set[str] = set()
        for meta in metadatas:
            if not meta:
                continue
            mid = meta.get("meeting_id") or meta.get("recording_id", "")
            if mid:
                mids.add(str(mid))

        return mids
    except Exception as e:
        logger.error(f"[AIChat] Failed to get Stage 2 meeting IDs: {e}")
        return set()


def get_available_meetings(user_id: str) -> List[Dict[str, Any]]:
    """
    Get all meetings that have Final ROM or Stage 2 points available for AI Chat.
    Returns list of {id, name, date, has_final_rom, has_stage2} sorted by date descending.
    Uses dim=0 to query collection metadata without loading the heavy embedding model into VRAM.
    """
    from services.vector_store import get_final_rom_points_store, get_stage2_points_store

    meetings_map: Dict[str, Dict[str, Any]] = {}

    # 1. Final ROM store
    try:
        f_store = get_final_rom_points_store(user_id, dim=0)
        if f_store.exists():
            f_store.load_or_create()
            if f_store._collection is not None:
                res = f_store._collection.get(include=["metadatas"])
                for meta in (res.get("metadatas") or []):
                    if meta:
                        mid = meta.get("meeting_id") or meta.get("recording_id")
                        if mid:
                            mid_str = str(mid)
                            is_manual = bool(
                                meta.get("source") == "manual_mom"
                                or str(meta.get("is_manual_mom", "")).lower() == "true"
                                or meta.get("is_manual_mom") is True
                                or mid_str.startswith("mom_")
                            )
                            if mid_str not in meetings_map:
                                meetings_map[mid_str] = {
                                    "id": mid_str,
                                    "name": meta.get("meeting_name") or "Meeting",
                                    "date": meta.get("date") or "",
                                    "has_final_rom": True,
                                    "has_stage2": False,
                                    "is_manual_mom": is_manual,
                                }
                            else:
                                meetings_map[mid_str]["has_final_rom"] = True
                                if is_manual:
                                    meetings_map[mid_str]["is_manual_mom"] = True
    except Exception as e:
        logger.warning(f"[AIChat] Failed to get available meetings from Final ROM store: {e}")

    # 2. Stage 2 store
    try:
        s_store = get_stage2_points_store(user_id, dim=0)
        if s_store.exists():
            s_store.load_or_create()
            if s_store._collection is not None:
                res = s_store._collection.get(include=["metadatas"])
                for meta in (res.get("metadatas") or []):
                    if meta:
                        mid = meta.get("meeting_id") or meta.get("recording_id")
                        if mid:
                            mid_str = str(mid)
                            is_manual = bool(
                                meta.get("source") == "manual_mom"
                                or str(meta.get("is_manual_mom", "")).lower() == "true"
                                or meta.get("is_manual_mom") is True
                                or mid_str.startswith("mom_")
                            )
                            if mid_str not in meetings_map:
                                meetings_map[mid_str] = {
                                    "id": mid_str,
                                    "name": meta.get("meeting_name") or "Meeting",
                                    "date": meta.get("date") or "",
                                    "has_final_rom": False,
                                    "has_stage2": True,
                                    "is_manual_mom": is_manual,
                                }
                            else:
                                meetings_map[mid_str]["has_stage2"] = True
                                if is_manual:
                                    meetings_map[mid_str]["is_manual_mom"] = True
    except Exception as e:
        logger.warning(f"[AIChat] Failed to get available meetings from Stage 2 store: {e}")

    # 3. Fallback to SQLite recordings if ChromaDB is currently unindexed
    if not meetings_map:
        try:
            import sqlite3
            from config import settings
            db_url = settings.DATABASE_URL
            db_path = db_url.replace("sqlite+aiosqlite:///", "").replace("sqlite:///", "")
            if os.path.exists(db_path):
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT r.id, r.filename, r.title, r.created_at, "
                    "       COALESCE(rm.rom_data, r.rom_data) as rom_data "
                    "FROM recordings r "
                    "LEFT JOIN rom_metadata rm ON rm.recording_id = r.id "
                    "WHERE r.user_id = ? ORDER BY r.created_at DESC",
                    (str(user_id),),
                )
                rows = cursor.fetchall()
                conn.close()
                for r in rows:
                    m_id, filename, title, created_at, rom_raw = r
                    rom_dict = _safe_parse_rom_data(rom_raw)
                    has_f = bool(rom_dict.get("final_rom", {}).get("agendas"))
                    has_s = bool(rom_dict.get("stage2", {}).get("polished_points"))
                    if has_f or has_s:
                        meetings_map[str(m_id)] = {
                            "id": str(m_id),
                            "name": title or filename or "Meeting",
                            "date": created_at or "",
                            "has_final_rom": has_f,
                            "has_stage2": has_s,
                        }
        except Exception as e:
            logger.warning(f"[AIChat] Failed to query meetings from SQLite: {e}")

    meetings = list(meetings_map.values())
    meetings.sort(key=lambda m: m.get("date", ""), reverse=True)
    logger.info(f"[AIChat] Found {len(meetings)} meetings with ROM/Stage2 data.")
    return meetings


# ── Context Formatting ───────────────────────────────────────────────────────

def format_context(results: List[Dict]) -> str:
    """Format retrieved points into a rich context string grouped by meeting and source."""
    if not results:
        return "(No relevant meeting discussion points found.)"

    sections: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        meeting = r.get("meeting_name", "Unknown Meeting")
        if meeting not in sections:
            sections[meeting] = []
        sections[meeting].append(r)

    parts = []
    for name, items in sections.items():
        is_manual = any(
            i.get("source") == "manual_mom"
            or str(i.get("is_manual_mom", "")).lower() == "true"
            or i.get("is_manual_mom") is True
            or str(i.get("meeting_id", "")).startswith("mom_")
            for i in items
        )
        has_final = any(
            i.get("source") in ("final_roim", "final_rom") or i.get("stage") in ("final_roim", "final_rom")
            for i in items
        )
        if is_manual:
            tag = "Manual MoM"
        elif has_final:
            tag = "Final ROM"
        else:
            tag = "Stage 2"
        parts.append(f"--- {name} [{tag}] ---")
        for item in items:
            clean = (item.get("clean_text") or item.get("_text", "")).strip()
            agenda = item.get("agenda_title", "").strip()
            speakers = item.get("speakers", "")
            if isinstance(speakers, list):
                speakers = ", ".join(speakers)
            speaker_info = f" [Speakers: {speakers}]" if speakers else ""
            if agenda and not clean.startswith(f"[{agenda}]"):
                parts.append(f"• [{agenda}] {clean}{speaker_info}")
            else:
                parts.append(f"• {clean}{speaker_info}")
        parts.append("")

    return "\n".join(parts)



# ── Prompt Building ──────────────────────────────────────────────────────────

def build_answer_prompt(
    question: str,
    context_results: List[Dict],
    chat_history: List[Dict],
) -> str:
    """Build the final LLM prompt with retrieved Stage 2 context."""
    history_text = ""
    if chat_history:
        recent = chat_history[-6:]
        lines = ["PREVIOUS CONVERSATION:"]
        for msg in recent:
            role = "User" if msg.get("role") == "user" else "Assistant"
            content = msg.get("content", "")
            if len(content) > 500:
                content = content[:500] + "..."
            lines.append(f"{role}: {content}")
        history_text = "\n".join(lines)

    context_text = format_context(context_results)

    return AI_CHAT_ANSWER_PROMPT.format(
        conversation_history=history_text,
        context=context_text,
        question=question,
    )


# ── LLM Streaming Inference ──────────────────────────────────────────────────

def stream_inference(
    prompt: str,
    max_new_tokens: int = 1500,
    chunk_callback: Optional[Callable[[str], None]] = None,
) -> str:
    """
    Run LLM inference with streaming support.
    Uses direct Ollama settings from environment / config (bypasses SQLite user_settings).
    Falls back to local QwenProvider only if Ollama is unavailable or disabled.
    """
    from services.ai_provider import get_provider, QwenProvider

    # Load direct AI Chat settings from environment/config
    cfg = get_ai_chat_ollama_config()
    use_ollama = cfg["use_ollama"]
    server_url = cfg["ollama_server_url"]
    priority_list = cfg["ollama_model_priority"]

    # Try Ollama directly if enabled
    if use_ollama:
        model_name = QwenProvider._detect_ollama_model(server_url, priority_list)
        if model_name:
            try:
                if chunk_callback:
                    return _stream_ollama_chat(
                        server_url, model_name, prompt,
                        max_new_tokens, chunk_callback, cfg,
                    )
                else:
                    direct_resp = _call_ollama_direct(
                        prompt, max_new_tokens=max_new_tokens, cfg=cfg
                    )
                    if direct_resp is not None:
                        return direct_resp
            except Exception as e:
                logger.warning(f"[AIChat] Direct Ollama inference failed: {e}. Falling back to local.")
        else:
            logger.warning(f"[AIChat] Direct Ollama enabled but no model detected at {server_url}. Falling back to local.")
    else:
        logger.info("[AIChat] Direct Ollama disabled via AI_CHAT_USE_OLLAMA. Using local inference.")

    # Fallback: local inference
    provider = get_provider()
    try:
        full_response = provider._infer(prompt, max_new_tokens=max_new_tokens)
        full_response = full_response.strip()

        if chunk_callback and full_response:
            words = full_response.split(" ")
            buffer = []
            for i, word in enumerate(words):
                buffer.append(word)
                if len(buffer) >= 3 or i == len(words) - 1:
                    chunk_callback(" ".join(buffer) + (" " if i < len(words) - 1 else ""))
                    buffer = []

        return full_response
    except Exception as e:
        logger.error(f"[AIChat] LLM inference failed: {e}", exc_info=True)
        error_msg = f"⚠️ AI generation failed: {str(e)}"
        if chunk_callback:
            chunk_callback(error_msg)
        return error_msg
    finally:
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _stream_ollama_chat(
    server_url: str,
    model: str,
    prompt: str,
    max_new_tokens: int,
    chunk_callback: Callable[[str], None],
    cfg: dict,
) -> str:
    """Stream from Ollama's /api/chat endpoint."""
    import urllib.request

    from services.ai_provider import QwenProvider, calculate_dynamic_num_ctx, strip_thinking

    system_content = (
        "You are an expert meeting analyst assistant. "
        "Answer questions accurately using provided meeting context. "
        "Preserve all technical terminology."
    )

    dynamic_enabled = bool(cfg.get("ollama_dynamic_ctx", True))
    manual_num_ctx = int(cfg.get("ollama_num_ctx", 32768))

    est_input_tokens, calculated_num_ctx = calculate_dynamic_num_ctx(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        safety_buffer=512,
        system_content=system_content,
        tokenizer=QwenProvider._tokenizer,
    )

    selected_num_ctx = calculated_num_ctx if dynamic_enabled else manual_num_ctx
    think_enabled = bool(cfg.get("ollama_think", False))

    options = {
        "num_predict": max_new_tokens,
        "temperature": float(cfg.get("ollama_temperature", 0.2)),
        "num_ctx": selected_num_ctx,
        "repeat_penalty": float(cfg.get("ollama_repeat_penalty", 1.1)),
        "top_p": float(cfg.get("ollama_top_p", 0.9)),
        "top_k": int(cfg.get("ollama_top_k", 40)),
        "think": think_enabled,
    }
    if cfg.get("ollama_seed") is not None and cfg.get("ollama_seed", -1) >= 0:
        options["seed"] = int(cfg["ollama_seed"])
    if cfg.get("ollama_stop"):
        stop_seqs = [s.strip() for s in str(cfg["ollama_stop"]).split(",") if s.strip()]
        if stop_seqs:
            options["stop"] = stop_seqs
    if cfg.get("ollama_num_thread") is not None and cfg.get("ollama_num_thread", 0) > 0:
        options["num_thread"] = int(cfg["ollama_num_thread"])
    if cfg.get("ollama_num_gpu") is not None and cfg.get("ollama_num_gpu", -1) >= 0:
        options["num_gpu"] = int(cfg["ollama_num_gpu"])

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
        ],
        "options": options,
        "think": think_enabled,
        "stream": True,
    }
    if cfg.get("ollama_keep_alive") is not None:
        try:
            payload["keep_alive"] = int(cfg["ollama_keep_alive"])
        except (ValueError, TypeError):
            payload["keep_alive"] = str(cfg["ollama_keep_alive"])

    base_url = server_url.rstrip("/")
    url = f"{base_url}/api/chat"

    logger.info(
        f"[AIChat] Ollama streaming: model={model}, num_ctx={selected_num_ctx}, "
        f"prompt_chars={len(prompt)}"
    )

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    full_response = []
    with urllib.request.urlopen(req, timeout=120.0) as response:
        for line in response:
            if not line:
                continue
            try:
                data = json.loads(line.decode("utf-8"))
                content = data.get("message", {}).get("content", "")
                if content:
                    full_response.append(content)
                    chunk_callback(content)
                if data.get("done", False):
                    break
            except json.JSONDecodeError:
                continue

    return strip_thinking("".join(full_response))
