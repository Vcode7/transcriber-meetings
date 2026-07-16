"""
AI Provider — 100% offline Qwen3 4B Instruct (4-bit quantized).

Architecture:
  - Single provider: QwenProvider
  - Model is loaded ONCE at startup and kept resident in memory
  - GPU (CUDA) when available, CPU fallback otherwise
  - Hierarchical chunked summarization for long meetings
  - Dedicated prompt templates per task (no shared generic prompt)

Removed:
  - LocalSummarizationProvider (DistilBART)
  - GroqProvider (cloud Groq API)
  - AIProviderFactory with cloud routing
  - LlamaProvider (replaced by QwenProvider)
"""
from fastapi import datastructures
from dns import entropy
from __future__ import annotations

import json
import logging
import re
import textwrap
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# Prompt Templates — one per task, purpose-built for Qwen3
# ══════════════════════════════════════════════════════════════
from dateparser.search import search_dates
import re


def extract_dates_from_text(text: str):
    """
    Extract dates from free text and return:
    [
        {
            "value": "...",
            "purpose": ""
        }
    ]
    """
    if not text:
        return []

    try:
        matches = search_dates(
            text,
            settings={
                "PREFER_DATES_FROM": "past",
                "RETURN_AS_TIMEZONE_AWARE": False,
            },
        )
    except Exception:
        matches = None

    if not matches:
        return []

    seen = set()
    dates = []

    for original, parsed in matches:
        value = parsed.strftime("%B %d, %Y")

        key = value.lower()
        if key in seen:
            continue
        seen.add(key)

        purpose = ""

        lower = text.lower()

        if re.search(r"\bdue\b", lower):
            purpose = "Due date"
        elif re.search(r"\bdeadline\b", lower):
            purpose = "Deadline"
        elif re.search(r"\bexpire", lower):
            purpose = "Expiration date"
        elif re.search(r"\bapproval\b", lower):
            purpose = "Approval date"
        elif re.search(r"\bmeeting\b", lower):
            purpose = "Meeting date"
        elif re.search(r"\bacquired\b", lower):
            purpose = "Acquisition date"
        elif re.search(r"\bentered\b", lower):
            purpose = "Agreement date"
        elif re.search(r"\bamend", lower):
            purpose = "Amendment date"
        elif re.search(r"\bmilestone\b|\bphase\b", lower):
            purpose = "Project milestone"
        elif re.search(r"\bdesign review\b|\bpdr\b", lower):
            purpose = "Preliminary Design Review"
        elif re.search(r"\bcdr\b|\bcritical design review\b", lower):
            purpose = "Critical Design Review"
        elif re.search(r"\btest\b|\btesting\b|\bqualification\b", lower):
            purpose = "Test schedule"
        elif re.search(r"\bvalidation\b|\bverification\b", lower):
            purpose = "Verification milestone"
        elif re.search(r"\bintegration\b", lower):
            purpose = "Integration milestone"
        elif re.search(r"\brelease\b|\bdeployment\b", lower):
            purpose = "Release date"
        elif re.search(r"\bprototype\b", lower):
            purpose = "Prototype milestone"
        elif re.search(r"\bmanufactur", lower):
            purpose = "Manufacturing milestone"
        elif re.search(r"\bflight\b", lower):
            purpose = "Flight schedule"

        elif re.search(r"\btrial\b", lower):
            purpose = "Trial schedule"

        elif re.search(r"\bground test\b", lower):
            purpose = "Ground test schedule"

        elif re.search(r"\bacceptance\b", lower):
            purpose = "Acceptance milestone"

        elif re.search(r"\bdelivery\b|\bdeliverable\b", lower):
            purpose = "Delivery milestone"

        elif re.search(r"\bdemo\b|\bdemonstration\b", lower):
            purpose = "Demonstration date"

        elif re.search(r"\bcertification\b|\bclearance\b", lower):
            purpose = "Certification milestone"

        elif re.search(r"\binspection\b", lower):
            purpose = "Inspection date"

        elif re.search(r"\bproduction\b", lower):
            purpose = "Production milestone"

        elif re.search(r"\bcommission\b", lower):
            purpose = "Commissioning date"

        else:
            purpose = "Referenced date"
        dates.append(
            {
                "value": value,
                "purpose": purpose,
            }
        )

    return dates

EXECUTIVE_SUMMARY_PROMPT = """\
You are an expert enterprise meeting analyst. Analyze the following meeting transcript and write a professional Executive Summary.

Answer each of the following questions concisely and professionally:

1. What was the main agenda of the meeting?
2. What problems or topics were discussed?
3. What work was completed or achieved?
4. What decisions were taken?
5. What conclusions were reached?
6. What are the next expected steps?

Format your response EXACTLY as follows (use these section headers):

MEETING PURPOSE:
[2–5 sentences answering question 1]

MAIN DISCUSSION POINTS:
- [point derived from question 2]
- [point derived from question 2]
- up to 5 or more (based on the transcript)

OUTCOMES:
- [outcome from questions 3 and 4]
- [outcome from questions 3 and 4]
- up to 5 or more (based on the transcript)

NEXT STEPS:
- [step from questions 5 and 6]
- [step from questions 5 and 6]
- up to 5 or more (based on the transcript)

Rules:
- Be concise and professional. Suitable for senior management.
- Do NOT repeat verbatim sentences from the transcript.
- Do NOT attribute statements to specific speakers.
- Do NOT add information not present in the transcript.
- Preserve all technical terms, acronyms, and project names exactly as spoken.

TRANSCRIPT:
{transcript}

Now write the Executive Summary:"""


SHORT_SUMMARY_PROMPT = """\
You are an expert meeting analyst. Summarize the following meeting transcript in exactly 120 words or fewer.

Write as a single professional paragraph. Do NOT use bullet points. Focus on what was discussed and decided.
Do NOT mention who said what. Do NOT repeat transcript sentences. Preserve all technical terms exactly.

TRANSCRIPT:
{transcript}

120-word summary:"""


DETAILED_SUMMARY_PROMPT = """\
You are an expert meeting analyst writing a comprehensive meeting report.

Analyze the transcript and write a detailed summary with SEPARATE SECTIONS for each major discussion topic.

For each topic section:
- Give the section a clear title (use ## Title format)
- Explain the topic and context
- Summarize the discussion details and technical observations
- State conclusions or outcomes reached for that topic

After all topic sections, add:

## Key Decisions
- [decision 1]
- [decision 2]

## Action Items
- [action item with owner if mentioned]

Rules:
- Do NOT attribute statements to specific speakers.
- Preserve all technical terms, acronyms, aircraft names, project names exactly.
- Be thorough. Reading this summary alone should give full meeting understanding.
- Do NOT repeat verbatim transcript sentences.

TRANSCRIPT:
{transcript}

Comprehensive meeting report:"""

MOM_PROMPT = """\
You are an experienced Executive Assistant responsible for producing professional, comprehensive, and factual Minutes of Meeting (MoM).

Your task is to analyze the complete meeting transcript and generate a structured Minutes of Meeting that captures ALL meaningful discussion while remaining concise and free of repetition.

Return ONLY a valid JSON object.
Do NOT wrap the response in markdown.
Do NOT include ```json.
Do NOT include explanations or any text outside the JSON.

The output MUST strictly follow this schema:

{{
  "title": "Professional meeting title",
  "introduction": "A professional overview describing the meeting purpose, agenda, participants' objectives, and overall context.",
  "points_discussed": [
    "Each item describes one meaningful discussion topic."
  ],
  "action_items": [
    {{
      "task": "Task description",
      "owner": "Speaker name or Unassigned",
      "deadline": "Deadline if mentioned, otherwise ASAP or null"
    }}
  ],
  "general_action_items": [
    "Action item without a specific owner"
  ],
  "conclusion": "Professional conclusion summarizing decisions, agreements, pending items, risks, and next steps.",
  "actual_start_time": "HH:MM if explicitly mentioned, otherwise null",
  "next_meeting_date": "Date/time if explicitly mentioned, otherwise null"
}}

Generation Guidelines

TITLE
- Generate a concise, professional meeting title.
- Reflect the primary objective of the meeting.
{agenda_section}

INTRODUCTION
- Write a well-structured executive introduction.
- Clearly explain:
  - meeting purpose
  - overall agenda
  - major subjects discussed
  - project/context
- Do not simply repeat the transcript.

POINTS DISCUSSED
- Capture EVERY important discussion topic.
- There is NO maximum or minimum number of points.
- Create one point for every meaningful topic discussed.
- Merge duplicate or repetitive discussions into a single comprehensive point.
- Ignore greetings, filler conversations, acknowledgements, interruptions, and casual chit-chat.
- Preserve important technical details, architecture discussions, implementation decisions, requirements, risks, blockers, design changes, timelines, dependencies, and stakeholder decisions.
- Each point should be:
  - self-contained
  - factually accurate
  - written as a complete sentence
  - professional
- Maintain the chronological flow whenever practical.

ACTION ITEMS
- Extract EVERY actionable task discussed.
- Include:
  - implementation tasks
  - follow-up work
  - reviews
  - testing
  - documentation
  - approvals
  - bug fixes
  - investigations
  - deployments
- Infer the owner only when clearly supported by the transcript.
- Otherwise use "Unassigned".
- Preserve deadlines exactly if mentioned.
- Never invent deadlines.

GENERAL ACTION ITEMS
- Include only actionable tasks that have no identifiable owner.
- Do not duplicate items already listed in action_items.

CONCLUSION
- Write a professional executive summary.
- Include:
  - key outcomes
  - decisions made
  - agreements reached
  - unresolved issues
  - risks
  - future work
  - next steps
- Do not introduce information not present in the transcript.

EXTRACTION RULES
- Never hallucinate facts.
- Never invent attendees, owners, dates, deadlines, or decisions.
- Preserve all technical terminology exactly:
  - API names
  - model names
  - class names
  - filenames
  - project names
  - architecture
  - frameworks
  - libraries
  - database names
  - commands
  - code identifiers
  - acronyms
- If conflicting information exists, report only what was actually discussed.
- Ignore repeated statements and produce a clean consolidated MoM.
- If a field has no relevant information:
  - arrays → []
  - string/date fields → null

The final output must be valid JSON and nothing else.
{reference_section}
TRANSCRIPT:
{transcript}

JSON:
"""
MOM_MERGE_PROMPT = """\
You are an experienced Executive Assistant responsible for producing professional, comprehensive, and consolidated Minutes of Meeting (MoM).

You will be given multiple partial Minutes of Meeting (MoM) generated from different sections of the same meeting. Your task is to merge and consolidate these partial MoMs into a single, unified, and cohesive final MoM.

Return ONLY a valid JSON object matching the exact schema.
Do NOT wrap the response in markdown.
Do NOT include ```json.
Do NOT include explanations or any text outside the JSON.

The output MUST strictly follow this schema:
{{
  "title": "Consolidated professional meeting title",
  "introduction": "A unified professional overview describing the meeting purpose, agenda, participants' objectives, and overall context.",
  "points_discussed": [
    "Each item describes one meaningful discussion topic."
  ],
  "action_items": [
    {{
      "task": "Task description",
      "owner": "Speaker name or Unassigned",
      "deadline": "Deadline if mentioned, otherwise ASAP or null"
    }}
  ],
  "general_action_items": [
    "Action item without a specific owner"
  ],
  "conclusion": "Unified professional conclusion summarizing decisions, agreements, pending items, risks, and next steps."
}}

Guidelines for Consolidation:
1. TITLE: Choose the most professional and representative title across the partial MoMs.
2. INTRODUCTION: Write a unified introduction. Combine the purpose and objectives from the partial introductions into a single cohesive paragraph.
3. POINTS DISCUSSED:
   - Merge all discussion points.
   - Consolidate repetitive or overlapping discussions across different sections into a single comprehensive point.
   - Preserve all unique technical details, architecture descriptions, requirements, risks, and decisions.
   - Do not lose unique details. Enforce clean, professional sentences.
4. ACTION ITEMS & GENERAL ACTION ITEMS:
   - Combine all action items and general action items.
   - Remove duplicate action items.
   - Keep owners and deadlines intact.
5. CONCLUSION:
   - Synthesize a final unified conclusion summarizing the overall decisions, agreements, and next steps for the entire meeting.

PARTIAL MOMs TO CONSOLIDATE:
{partial_moms_json}

Now generate the consolidated final MoM JSON:"""


AGENDA_COMPRESS_PROMPT = """\
You are an expert meeting agenda parser.

Your task is to extract ONLY the actual meeting agenda items from the document.

IMPORTANT:
Do NOT immediately start extracting agenda items.

Instead, complete the task in the following stages.

STEP 1 - IDENTIFY THE AGENDA SECTION
First identify where the actual meeting agenda begins and ends.

Agenda sections commonly appear after introductory information such as:

- Organization name
- Meeting title
- Meeting date
- Meeting time
- Meeting location
- Committee members
- Staff information
- Contact details
- Public comment instructions
- Zoom/Webinar information
- Accessibility information
- General meeting instructions
- Headers
- Footers
- Page numbers

Ignore all of the above.

The agenda section usually contains discussion topics that will actually be covered during the meeting.

Examples include:

- Call to Order
- Roll Call
- Approval of Minutes
- Public Comment
- Consent Agenda
- Discussion Items
- Briefings
- Presentations
- Reports
- Proposed Ordinances
- Proposed Motions
- Project Updates
- Technical Reviews
- Design Reviews
- Other Business
- Adjournment

STEP 2 - DETECT THE AGENDA STRUCTURE
Agenda items may appear in many different formats.

Examples:

1.
2.
3.

A.
B.
C.

•
•

Roman numerals

Tables

Indented lists

Nested sections

Paragraphs

Do NOT assume a specific layout.

Understand the document structure before extracting.

If the agenda is presented in a table, identify the columns first.

Typical columns include:

- Agenda ID
- Item Number
- Agenda Title
- Topic
- Subject
- Presenter
- Presented By
- Speaker
- Lead
- Owner
- Responsible Person
- Department
- Duration

If an agenda item contains a title followed by one or more descriptive
paragraphs, they belong to the SAME agenda item.

The title and its description should be merged into a single topic.

Do not split a single agenda item into multiple topics.

STEP 3 - IDENTIFY THE PRESENTER
For every agenda item determine whether a presenter is explicitly associated with it.

Possible labels include (but are not limited to):

- Presenter
- Presented By
- Speaker
- Agenda Presenter
- Lead
- Owner
- Responsible
- Presented by
- Council Staff
- Staff
- Department Lead
- Project Lead
- Facilitator
- Chair
- Vice Chair

The presenter's name may appear:

- in the same row
- below the agenda item
- above the agenda item
- in an adjacent table column
- immediately after the topic

When extracting an agenda item, search the nearby text before and after
the topic.

If a presenter, council staff, presenter, presented by, project lead,
staff lead, briefing by or similar information is found immediately
before or after the agenda item, associate that person (it would be mention near by) with the agenda item.

If none of them is explicitly associated with the agenda item, return:

null

Do NOT guess.

Do NOT use committee members listed at the beginning of the document unless they are explicitly associated with that agenda item.


STEP 4 - EXTRACT THE AGENDA 

After identifying the agenda section:

Extract every agenda item.

Rules:

- Copy the agenda topic exactly as written whenever possible.
- Do NOT summarize.
- Do NOT rewrite.
- Do NOT shorten.
- If the title spans multiple lines, combine them into one topic.
- Do NOT include administrative information.
- Do NOT include contact information.
- Do NOT include webinar instructions.
- Do NOT include meeting metadata.
- Do NOT include organization information.
- Do NOT include committee member lists.
- Do NOT include page headers or page footers.

If no agenda section exists, return:

[]

also extract

details
- Include ALL remaining information that belongs to this agenda item but is not part of the topic.
- Examples include:
    - agenda description
    - ordinance description
    - motion description
    - proposal description
    - objective
    - supporting information
    - sponsor
    - department
    - project information
    - references
    - document numbers
    - any explanatory text directly associated with this agenda item
- Preserve the wording as closely as possible.
- Do NOT summarize.
- Do NOT remove technical details.
- Do NOT include information belonging to another agenda item.
- If there are multiple paragraphs describing the same agenda item, combine them into a single details string.

If an agenda item consists of a title followed by descriptive paragraphs, they represent ONE agenda item.


For every agenda item extract:

{{
    "topic": "...",
    "speaker": "..."
    "details": "..."
}}

Rules:

1. Copy the agenda topic exactly as written whenever possible.
2. Do not summarize.
3. Do not merge multiple agenda items.
4. If an agenda item spans multiple lines, combine them into a single topic.
5. If a presenter, owner, lead, chair, council staff, presenter, speaker or responsible person is explicitly associated with that agenda item, extract their name.
6. If no presenter is explicitly associated with the agenda item, return null.
7. Never invent a speaker.
8. Return ONLY valid JSON.
9. If no agenda items are found, return an empty JSON array [].


AGENDA DOCUMENT:
{text}

JSON:"""


REFERENCE_COMPRESS_PROMPT = """\
You are a technical analyst preparing background knowledge for a meeting. Given the reference/context document below, extract the most relevant information in concise bullet points.

Focus on:
- Key facts, figures, decisions, or specifications mentioned
- Background knowledge relevant to understanding the meeting topic
- Project status, system descriptions, technical context
- Any constraints, requirements, or policies mentioned

Rules:
- Output ONLY bullet points starting with -
- Be concise. Each bullet should be one complete sentence.
- Preserve technical terms, project names, and system names exactly.
- Do NOT include irrelevant or generic content.

REFERENCE DOCUMENT:
{text}

Knowledge summary:"""


# ══════════════════════════════════════════════════════════════
# RAG Pipeline Prompts — used by Raw MoM pipeline ONLY
# These prompts are completely independent of the MoM pipeline above.
# ══════════════════════════════════════════════════════════════



RAW_MOM_EXTRACTION_PROMPT = """\
You are a precise information extractor for meeting records. Your ONLY job is structured extraction — not summarization.

You will receive evidence retrieved from the transcript, presentation slides, and organizational documents for ONE agenda topic.

Before extracting information, first understand the meaning and purpose of the agenda topic.

The retrieved evidence may contain unrelated information because semantic retrieval is not perfect. Do NOT assume that every retrieved sentence belongs to the current agenda.

Your task is to extract ONLY the factual information that is directly relevant to the current agenda topic.

- If the agenda is procedural (for example: Call to Order, Roll Call, Introductions, Approval of Minutes, Opening Remarks, Adjournment, etc.), extract only information related to that procedure. These agenda items often contain very little discussion, so it is acceptable to return only a few discussion entries.
- If the agenda is a discussion, project, technical, financial, planning, or decision-making topic, extract all relevant facts, discussions, decisions, action items, risks, milestones, dependencies, technical details, dates, and responsibilities related to that topic.
- Ignore evidence that clearly belongs to another agenda item, even if it appears in the retrieved context.

Your task:
- Extract ALL factual information relevant to the current agenda topic.
- Preserve EVERY important fact: dates, deadlines, numbers, decisions, action items, responsibilities, technical values, risks, milestones, dependencies.
- Do NOT compress, summarize, or omit relevant information.
- Do NOT include facts unrelated to the current agenda topic.
- Do NOT generate an introduction, conclusion, or meeting summary.
- Do NOT hallucinate facts not present in the evidence.
- Do NOT rewrite or polish the content.

Return ONLY a valid JSON object matching this exact schema:

{{
  "agenda_topic": "{agenda_topic}",
  "agenda_speaker": "{agenda_speaker}",
  "discussion": [
    {{
      "type": "decision|action|discussion|clarification|risk|milestone|dependency",
      "speaker": "Speaker name or null if unknown",
      "point": "Exact fact or statement extracted from evidence",
      "dates": [
        {{"value": "date/time value", "purpose": "what this date is for"}}
      ],
      "action": {{
        "owner": "person responsible or null",
        "description": "what needs to be done or null",
        "deadline": "deadline date or null",
        "status": "open|in_progress|completed|null"
      }}
    }}
  ]
}}

Rules for discussion entries:
- One entry per distinct fact, decision, action item, or risk.
- "dates" array: include ONLY if the entry involves a specific date/deadline/milestone.
- "action": fill ONLY if the entry is an action item; set all fields to null otherwise.
- "type" must be one of: decision, action, discussion, clarification, risk, milestone, dependency.
- Preserve all technical terms, acronyms, numbers, system names, and project names exactly.

AGENDA TOPIC: {agenda_topic}
SPEAKER: {agenda_speaker}

RETRIEVED EVIDENCE:
{evidence}

JSON:"""




KEY_POINTS_PROMPT = """\
You are an expert meeting analyst. Extract the key discussion points from the following meeting transcript.

For each key point, provide:
• [Topic Name]
[One to three sentences explanation of what was discussed about this topic]

Extract all key points. Cover all major topics discussed.

Rules:
- Preserve all technical terms, acronyms, and project names exactly as spoken.
- Do NOT attribute to speakers.
- Do NOT repeat verbatim transcript sentences.

TRANSCRIPT:
{transcript}

Key discussion points:"""


ACTION_ITEMS_PROMPT = """\
You are an expert meeting analyst. Extract all action items from the following meeting transcript.

Group them into sections:

General Action Items
- [items not clearly assigned to any specific speaker]

[Speaker Name or Role if items were assigned to them]
- [their specific action items]

Rules:
- Only include a speaker section if that speaker has action items.
- Do NOT include empty sections.
- Do NOT include Owner or Deadline fields.
- Do NOT number items. Use - prefix for each item.
- If there are no action items, return: None identified.
- Preserve all technical terms and project names exactly.

TRANSCRIPT:
{transcript}

Action items (grouped by speaker):"""


KEY_DECISIONS_PROMPT = """\
You are an expert meeting analyst. Extract all concrete decisions that were made or agreed upon in the following meeting.

Return a numbered list of decisions only. Each decision should be one clear sentence.
If there are no decisions, return: None identified.

Rules:
- Only include decisions explicitly stated or agreed upon (not proposals or suggestions).
- Preserve all technical terms exactly.

TRANSCRIPT:
{transcript}

Decisions made:"""


CHUNK_SUMMARY_PROMPT = """\
Summarize the following meeting excerpt in 3–7 sentences. Focus on the main topics discussed and any decisions or outcomes.
Preserve all technical terms, acronyms, and project names exactly. Do not attribute to speakers.

EXCERPT:
{chunk}

Summary:"""


SPEAKER_SUMMARY_PROMPT = """\
You are an expert meeting analyst. Summarize only {speaker}'s contributions from the transcript below in 2–4 sentences.

Focus on what {speaker} specifically discussed, proposed, or decided. Do NOT repeat verbatim sentences.
Preserve all technical terms, acronyms, and project names exactly.

TRANSCRIPT (only {speaker}'s lines):
{transcript}

{speaker}'s summary:"""


SPEAKER_KEY_POINTS_PROMPT = """\
You are an expert meeting analyst. Extract 3–6 key points from {speaker}'s contributions in the transcript below.

Format:
• [Key point title]: [One sentence explanation]

Preserve all technical terms exactly. Only include what {speaker} actually said.

TRANSCRIPT (only {speaker}'s lines):
{transcript}

{speaker}'s key points:"""


SPEAKER_ACTION_ITEMS_PROMPT = """\
You are an expert meeting analyst. Extract any action items that {speaker} committed to or was assigned in the transcript below.

Return a numbered list. If none, return: None identified.

TRANSCRIPT (only {speaker}'s lines):
{transcript}

{speaker}'s action items:"""


# ══════════════════════════════════════════════════════════════
# Shared utilities
# ══════════════════════════════════════════════════════════════

def _format_transcript(transcript: List[Dict]) -> str:
    """Convert transcript segments into readable dialogue string (Speaker: text)."""
    lines = []
    for seg in transcript:
        label = seg.get("speaker_label", "Unknown")
        text = seg.get("text", "").strip()
        if text:
            lines.append(f"{label}: {text}")
    return "\n".join(lines)


def _format_for_summary(transcript: List[Dict]) -> str:
    """Convert transcript into plain text with no speaker labels — for prose summaries."""
    parts = []
    for seg in transcript:
        text = seg.get("text", "").strip()
        if text:
            parts.append(text)
    return " ".join(parts)


def _clean_list(raw: str) -> List[str]:
    """Parse a numbered/bulleted list from LLM output."""
    lines = [l.strip() for l in raw.split("\n") if l.strip()]
    cleaned = []
    for line in lines:
        if "none identified" in line.lower():
            continue
        if line and line[0].isdigit() and ". " in line:
            cleaned.append(line.split(". ", 1)[1])
        elif line.startswith("- "):
            cleaned.append(line[2:])
        elif line.startswith("• "):
            cleaned.append(line[2:])
        else:
            cleaned.append(line)
    return [c for c in cleaned if c and len(c) > 3]


def _split_sentences(text: str) -> List[str]:
    """Simple sentence splitter."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def _chunk_text(text: str, max_words: int = 700) -> List[str]:
    """Split text into word-count-bounded chunks, breaking on sentence boundaries."""
    sentences = _split_sentences(text)
    chunks = []
    current_words = []
    current_count = 0

    for sent in sentences:
        word_count = len(sent.split())
        if current_count + word_count > max_words and current_words:
            chunks.append(" ".join(current_words))
            current_words = [sent]
            current_count = word_count
        else:
            current_words.append(sent)
            current_count += word_count

    if current_words:
        chunks.append(" ".join(current_words))

    return [c for c in chunks if len(c.split()) >= 30]


def _empty_mom(recording_meta: dict) -> dict:
    return {
        "title": recording_meta.get("filename", "Meeting Notes"),
        "date": recording_meta.get("created_at", ""),
        "duration": recording_meta.get("duration", 0),
        "planned_start_time": "",
        "actual_start_time": "",
        "participants": recording_meta.get("speakers_detected", []),
        "introduction": "Failed to generate meeting introduction.",
        "points_discussed": [],
        "action_items": [],
        "conclusion": "",
    }


# ══════════════════════════════════════════════════════════════
# Abstract base
# ══════════════════════════════════════════════════════════════

class AIProvider(ABC):
    """Abstract interface for all AI providers."""

    @abstractmethod
    def build_context_summary(self, transcript: List[Dict]) -> str: ...

    @abstractmethod
    def compress_agenda(self, text: str) -> str: ...

    @abstractmethod
    def compress_reference(self, text: str) -> str: ...

    @abstractmethod
    def generate_summary(self, transcript: List[Dict], context: Optional[str] = None) -> str: ...

    @abstractmethod
    def generate_key_points(self, transcript: List[Dict], context: Optional[str] = None) -> List[str]: ...

    @abstractmethod
    def generate_action_items(self, transcript: List[Dict], context: Optional[str] = None) -> List[str]: ...

    @abstractmethod
    def generate_key_decisions(self, transcript: List[Dict], context: Optional[str] = None) -> List[str]: ...

    @abstractmethod
    def generate_mom(self, transcript: List[Dict], recording_meta: dict, context: Optional[str] = None) -> dict: ...

    @abstractmethod
    def generate_executive_summary(self, transcript: List[Dict], context: Optional[str] = None) -> dict: ...

    @abstractmethod
    def generate_short_summary(self, transcript: List[Dict], context: Optional[str] = None) -> str: ...

    @abstractmethod
    def generate_detailed_summary(self, transcript: List[Dict], context: Optional[str] = None) -> str: ...

    @abstractmethod
    def generate_speaker_summaries(self, transcript: List[Dict]) -> Dict[str, Dict]: ...


# ══════════════════════════════════════════════════════════════
# Qwen3 4B Instruct Provider — single offline AI engine
# ══════════════════════════════════════════════════════════════

class QwenProvider(AIProvider):
    """
    100% offline summarization and document generation using
    Qwen/Qwen3-4B-Instruct (4-bit BitsAndBytes quantization).

    Model is loaded ONCE at startup and kept resident in memory.
    GPU (CUDA) acceleration when available; CPU fallback otherwise.
    Optimized for RTX 4050 6GB VRAM.

    Qwen3 specifics:
      - Thinking mode is disabled (enable_thinking=False) for structured
        document generation tasks — ensures clean, deterministic output
        without <think>...</think> preamble blocks.
      - Uses the standard chat template via apply_chat_template.
    """

    _model = None
    _tokenizer = None
    _pipeline = None
    _load_attempted = False
    _device = None

    @classmethod
    def clean_up_other_models(cls):
        """Ensure other AI models are fully unloaded and references are cleared to free VRAM/RAM."""
        logger.info("[QwenAI] Cleaning up other AI models from memory...")
        try:
            from services.transcription import unload_whisperx_model, unload_align_model
            unload_whisperx_model()
            unload_align_model()
        except Exception as e:
            logger.warning(f"[QwenAI] Failed to unload transcription/alignment models: {e}")

        try:
            from services.diarization import unload_diarization_pipeline
            unload_diarization_pipeline()
        except Exception as e:
            logger.warning(f"[QwenAI] Failed to unload diarization pipeline: {e}")

        try:
            from services.embedding import unload_encoder
            unload_encoder()
        except Exception as e:
            logger.warning(f"[QwenAI] Failed to unload speaker encoder: {e}")

        try:
            from main import unload_overlap_model
            unload_overlap_model()
        except Exception:
            pass

        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                logger.info(f"[QwenAI] VRAM cleaned up. Allocated memory: {torch.cuda.memory_allocated() / (1024 * 1024):.2f} MB")
        except Exception:
            pass

    @classmethod
    def _load_model_impl(cls):
        """Actual model loading code block (tokenizer, model, pipeline)."""
        import torch
        from transformers import (
            AutoTokenizer,
            AutoModelForCausalLM,
            BitsAndBytesConfig,
            pipeline,
        )
        from config import settings
        from services.model_loader import ModelLoader

        model_id = getattr(settings, "QWEN_MODEL_ID", "Qwen/Qwen3-4B")
        load_in_4bit = getattr(settings, "QWEN_LOAD_IN_4BIT", True)

        local_path = ModelLoader.get_model_path("nlp_engine")
        if local_path is not None:
            load_from = str(local_path)
            local_files_only = True
            logger.info(f"[QwenAI] Loading from local path: {load_from}")
        else:
            raise FileNotFoundError(
                f"[QwenAI] nlp_engine model not found in MODELS_DIR ({settings.MODELS_DIR}) "
                "or runtime/nlp_engine/. "
                "Copy the Qwen3-4B model folder to 'Application/runtime/nlp_engine/' and restart."
            )

        use_cuda = torch.cuda.is_available()
        cls._device = "cuda" if use_cuda else "cpu"
        logger.info(f"[QwenAI] Loading {load_from} on {cls._device} ...")

        bnb_config = None
        if load_in_4bit and use_cuda:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            logger.info("[QwenAI] 4-bit NF4 quantization enabled (RTX 4050 optimized)")
        elif use_cuda:
            logger.info("[QwenAI] Loading in float16 on CUDA (no quantization)")
        else:
            logger.info("[QwenAI] Loading on CPU (float32, slower inference)")

        hf_token = getattr(settings, "HF_TOKEN", None) or None
        cls._tokenizer = AutoTokenizer.from_pretrained(
            load_from,
            local_files_only=local_files_only,
            token=hf_token if not local_files_only else None,
        )

        model_kwargs = {
            "local_files_only": local_files_only,
            "device_map": "cuda:0" if use_cuda else None,
            "torch_dtype": torch.float16 if use_cuda else torch.float32,
        }
        if not local_files_only:
            model_kwargs["token"] = hf_token
        if bnb_config:
            model_kwargs["quantization_config"] = bnb_config

        cls._model = AutoModelForCausalLM.from_pretrained(load_from, **model_kwargs)
        cls._model.eval()

        pipe = pipeline(
            "text-generation",
            model=cls._model,
            tokenizer=cls._tokenizer,
        )
        logger.info(f"[QwenAI] Qwen3 4B Instruct ready on {cls._device} ✓")
        return pipe

    @classmethod
    def _get_pipeline(cls):
        """Return cached pipeline, loading model once if needed."""
        if cls._pipeline is not None:
            return cls._pipeline
        if cls._load_attempted:
            return None

        from services.device_utils import log_gpu_memory
        logger.info("[QwenAI] Initializing model load. Running pre-load memory cleanup...")
        cls.clean_up_other_models()
        log_gpu_memory("Pre-load QwenLLM")

        try:
            cls._pipeline = cls._load_model_impl()
            if cls._pipeline is not None:
                logger.info("[QwenAI] Model loaded successfully on first attempt.")
                log_gpu_memory("Post-load QwenLLM (Success)")
                return cls._pipeline
        except Exception as e:
            logger.warning(f"[QwenAI] First loading attempt failed: {e}. Recovering VRAM and retrying load once...", exc_info=True)

        # First load failed. Run recovery sequence and retry once.
        cls.clean_up_other_models()
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except Exception:
            pass

        log_gpu_memory("Pre-retry load QwenLLM")

        try:
            cls._pipeline = cls._load_model_impl()
            if cls._pipeline is not None:
                logger.info("[QwenAI] Model loaded successfully on retry.")
                log_gpu_memory("Post-load QwenLLM (Success on retry)")
                return cls._pipeline
        except Exception as retry_err:
            logger.error(f"[QwenAI] Loading attempt retry failed: {retry_err}. Disabling Qwen LLM.", exc_info=True)
            cls._pipeline = None

        # Mark load attempted to prevent further retry loops
        cls._load_attempted = True
        log_gpu_memory("Post-load QwenLLM (Permanent Failure)")
        return None

    @classmethod
    def unload_model(cls):
        """Unload Qwen model from VRAM and CPU memory."""
        from services.device_utils import log_gpu_memory
        log_gpu_memory("Pre-unload QwenLLM")
        logger.info("[QwenAI] Unloading Qwen3 model...")
        cls._pipeline = None
        cls._model = None
        cls._tokenizer = None
        cls._load_attempted = False
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info("[QwenAI] Qwen3 model unloaded.")
        log_gpu_memory("Post-unload QwenLLM")


    def _infer(self, prompt: str, max_new_tokens: int = 512) -> str:
        """Run inference with the loaded Qwen3 model."""
        pipe = self._get_pipeline()
        if pipe is None:
            logger.warning("[QwenAI] Model not available. Returning empty result.")
            return ""

        try:
            # Use chat template for Qwen3 Instruct
            # enable_thinking=False suppresses the <think>...</think> preamble
            # that Qwen3 emits in reasoning mode — keeps output clean for structured docs.
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an expert enterprise meeting analyst. "
                        "Follow instructions exactly. Preserve all technical terminology."
                    ),
                },
                {"role": "user", "content": prompt},
            ]

            # Apply Qwen3 chat template with thinking disabled
            tokenizer = self._tokenizer
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,  # Qwen3: disable chain-of-thought preamble
            )

            output = pipe(
                text,
                max_new_tokens=max_new_tokens,
                do_sample=False,          # deterministic — enterprise doc quality
                temperature=1.0,          # ignored when do_sample=False
                repetition_penalty=1.15,  # reduce repetitive output
                return_full_text=False,
            )

            # Extract generated text
            result = output[0]["generated_text"]
            if isinstance(result, list):
                # Chat pipeline returns list of messages
                for msg in reversed(result):
                    if msg.get("role") == "assistant":
                        return msg.get("content", "").strip()
            return str(result).strip()

        except Exception as e:
            err_msg = str(e).lower()
            if "out of memory" in err_msg or "oom" in err_msg or "cuda memory" in err_msg:
                raise e
            logger.error(f"[QwenAI] Inference failed: {e}", exc_info=True)
            return ""

    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 1.0) -> str:
        """
        Public generate method — used by vocab_extractor for AI-assisted extraction.
        Delegates to _infer.
        """
        return self._infer(prompt, max_new_tokens=max_new_tokens)

    def _hierarchical_summarize(self, text: str) -> str:
        """
        Hierarchical chunked summarization for long transcripts.
        Splits → summarizes each chunk → merges → returns combined summary text.
        """
        word_count = len(text.split())
        if word_count <= 1500:
            return text  # Short enough to use directly

        logger.info(f"[QwenAI] Long transcript ({word_count} words) — applying hierarchical chunking")
        chunks = _chunk_text(text, max_words=700)
        logger.info(f"[QwenAI] Split into {len(chunks)} chunks")

        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            logger.info(f"[QwenAI] Summarizing chunk {i+1}/{len(chunks)}")
            summary = self._infer(
                CHUNK_SUMMARY_PROMPT.format(chunk=chunk),
                max_new_tokens=256,
            )
            if summary:
                chunk_summaries.append(summary)

        merged = "\n\n".join(chunk_summaries)
        logger.info(f"[QwenAI] Merged {len(chunk_summaries)} chunk summaries ({len(merged.split())} words)")
        return merged

    def build_context_summary(self, transcript: List[Dict]) -> str:
        """
        Build the compressed context summary for a transcript.

        This is the SINGLE entry point for hierarchical summarization.
        Call this once, store the result in the DB, and pass it as the
        `context` parameter to all downstream AI methods to avoid
        redundant LLM inference.

        For transcripts <= 1500 words, returns the raw formatted dialogue
        (no inference needed). For longer transcripts, runs chunked
        summarization (N inference calls, one per ~700-word chunk).
        """
        dialogue = _format_transcript(transcript)
        if not dialogue.strip():
            return ""
        return self._hierarchical_summarize(dialogue)

    def compress_agenda(self, text: str) -> str:
        """Compress raw agenda/objectives text into a structured numbered list."""
        if not text or not text.strip():
            return ""
        words = text.split()
        if len(words) > 2500:
            chunks = _chunk_text(text, max_words=1200)
            summaries = []
            for chunk in chunks:
                out = self._infer(AGENDA_COMPRESS_PROMPT.format(text=chunk), max_new_tokens=2000)
                if out:
                    summaries.append(out.strip())
            text = "\n".join(summaries)
        result = self._infer(AGENDA_COMPRESS_PROMPT.format(text=text), max_new_tokens=2000)
        return result.strip() if result else ""

    def compress_reference(self, text: str) -> str:
        """Compress reference/context document text into a concise knowledge summary."""
        if not text or not text.strip():
            return ""
        words = text.split()
        if len(words) > 1500:
            chunks = _chunk_text(text, max_words=1200)
            summaries = []
            for chunk in chunks:
                out = self._infer(REFERENCE_COMPRESS_PROMPT.format(text=chunk), max_new_tokens=2000)
                if out:
                    summaries.append(out.strip())
            text = "\n".join(summaries)
        result = self._infer(REFERENCE_COMPRESS_PROMPT.format(text=text), max_new_tokens=2000)
        return result.strip() if result else ""

    # ── AIProvider interface ──────────────────────────────────

    def generate_summary(self, transcript: List[Dict], context: Optional[str] = None) -> str:
        """Generate a short meeting summary (backward compat alias for short_summary)."""
        return self.generate_short_summary(transcript, context=context)

    def generate_short_summary(self, transcript: List[Dict], context: Optional[str] = None) -> str:
        """Generate a concise ~120-word professional summary. Uses pre-computed context if supplied."""
        if context is None:
            plain = _format_for_summary(transcript)
            if not plain.strip():
                return "No transcript available to summarize."
            context = self._hierarchical_summarize(plain)
        elif not context.strip():
            return "No transcript available to summarize."
        result = self._infer(
            SHORT_SUMMARY_PROMPT.format(transcript=context),
            max_new_tokens=120,
        )
        return result or "Unable to generate summary."

    def generate_detailed_summary(self, transcript: List[Dict], context: Optional[str] = None) -> str:
        """Generate a comprehensive sectioned meeting report. Uses pre-computed context if supplied."""
        if context is None:
            dialogue = _format_transcript(transcript)
            if not dialogue.strip():
                return "No transcript available to summarize."
            context = self._hierarchical_summarize(dialogue)
        elif not context.strip():
            return "No transcript available to summarize."
        result = self._infer(
            DETAILED_SUMMARY_PROMPT.format(transcript=context),
            max_new_tokens=3000,
        )
        return result or "Unable to generate detailed summary."

    def generate_key_points(self, transcript: List[Dict], context: Optional[str] = None) -> List[str]:
        """Extract key discussion points. Uses pre-computed context if supplied."""
        if context is None:
            dialogue = _format_transcript(transcript)
            if not dialogue.strip():
                return []
            context = self._hierarchical_summarize(dialogue)
        elif not context.strip():
            return []
        raw = self._infer(
            KEY_POINTS_PROMPT.format(transcript=context),
            max_new_tokens=1028,
        )
        if not raw:
            return []

        # Parse bullet-format key points
        points = []
        current_topic = None
        for line in raw.split("\n"):
            line = line.strip()
            if not line:
                continue
            if line.startswith("•"):
                current_topic = line.lstrip("• ").rstrip(":")
            elif current_topic and not line.startswith("•"):
                points.append(f"{current_topic}: {line}")
                current_topic = None
            elif not current_topic and len(line) > 10:
                points.append(line)

        return points if points else _clean_list(raw)

    def generate_action_items(self, transcript: List[Dict], context: Optional[str] = None) -> List[str]:
        """Extract action items grouped by General / Speaker sections. Uses pre-computed context if supplied."""
        if context is None:
            dialogue = _format_transcript(transcript)
            if not dialogue.strip():
                return []
            context = self._hierarchical_summarize(dialogue)
        elif not context.strip():
            return []
        raw = self._infer(
            ACTION_ITEMS_PROMPT.format(transcript=context),
            max_new_tokens=1028,
        )
        if not raw or "none identified" in raw.lower():
            return []

        # Clean/split Qwen output
        lines = []
        for line in raw.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            # Remove leading numbers/bullets
            stripped = re.sub(r'^\d+\.\s*', '', stripped)
            lines.append(stripped)
        return lines if lines else []

    def generate_key_decisions(self, transcript: List[Dict], context: Optional[str] = None) -> List[str]:
        """Extract concrete decisions made during the meeting. Uses pre-computed context if supplied."""
        if context is None:
            dialogue = _format_transcript(transcript)
            if not dialogue.strip():
                return []
            context = self._hierarchical_summarize(dialogue)
        elif not context.strip():
            return []
        raw = self._infer(
            KEY_DECISIONS_PROMPT.format(transcript=context),
            max_new_tokens=1028,
        )
        if not raw or "none identified" in raw.lower():
            return []
        return _clean_list(raw)

    def generate_executive_summary(self, transcript: List[Dict], context: Optional[str] = None) -> dict:
        """Generate structured executive summary for PDF reports. Uses pre-computed context if supplied."""
        if context is None:
            dialogue = _format_transcript(transcript)
            if not dialogue.strip():
                return {
                    "purpose": "No transcript available.",
                    "discussion_points": [], "outcomes": [], "next_steps": [],
                }
            context = self._hierarchical_summarize(dialogue)
        elif not context.strip():
            return {
                "purpose": "No transcript available.",
                "discussion_points": [], "outcomes": [], "next_steps": [],
            }
        raw = self._infer(
            EXECUTIVE_SUMMARY_PROMPT.format(transcript=context),
            max_new_tokens=700,
        )

        if not raw:
            return {
                "purpose": "Unable to generate executive summary.",
                "discussion_points": [],
                "outcomes": [],
                "next_steps": [],
            }

        def _extract_section(text: str, header: str) -> List[str]:
            lines = text.split("\n")
            collecting = False
            items = []
            headers = ["MEETING PURPOSE:", "MAIN DISCUSSION POINTS:", "OUTCOMES:", "NEXT STEPS:"]
            for line in lines:
                if header.upper() in line.upper():
                    collecting = True
                    continue
                if collecting:
                    stripped = line.strip()
                    if stripped.startswith("-"):
                        items.append(stripped[1:].strip())
                    elif stripped and any(h in stripped.upper() for h in headers) and stripped != header:
                        break
            return [i for i in items if i]

        def _extract_paragraph(text: str, header: str) -> str:
            lines = text.split("\n")
            collecting = False
            parts = []
            for line in lines:
                if header.upper() in line.upper():
                    collecting = True
                    continue
                if collecting:
                    stripped = line.strip()
                    if stripped and any(h in stripped.upper() for h in
                                        ["MAIN DISCUSSION POINTS:", "OUTCOMES:", "NEXT STEPS:"]):
                        break
                    if stripped and not stripped.startswith("-"):
                        parts.append(stripped)
            return " ".join(parts).strip()

        return {
            "purpose": _extract_paragraph(raw, "MEETING PURPOSE:") or "See transcript for details.",
            "discussion_points": _extract_section(raw, "MAIN DISCUSSION POINTS:"),
            "outcomes": _extract_section(raw, "OUTCOMES:"),
            "next_steps": _extract_section(raw, "NEXT STEPS:"),
        }

    def generate_speaker_summaries(self, transcript: List[Dict]) -> Dict[str, Dict]:
        """
        Generate per-speaker summary, key points, and action items.

        Groups transcript segments by speaker_label, then runs three separate
        Qwen3 inference calls per speaker. Returns:
        {
            "SpeakerName": {
                "summary": str,
                "key_points": [str, ...],
                "action_items": [str, ...],
            },
            ...
        }
        """
        # Group segments by speaker
        speaker_segments: Dict[str, List[Dict]] = {}
        for seg in transcript:
            label = seg.get("speaker_label", "Unknown")
            if label in ("Unknown",) or seg.get("is_overlap"):
                continue
            speaker_segments.setdefault(label, []).append(seg)

        if not speaker_segments:
            logger.info("[QwenAI] No named speakers found — skipping per-speaker summaries")
            return {}

        logger.info(f"[QwenAI] Generating per-speaker summaries for: {list(speaker_segments.keys())}")
        results: Dict[str, Dict] = {}

        for speaker, segs in speaker_segments.items():
            # Build plain-text transcript for this speaker
            lines = [seg.get("text", "").strip() for seg in segs if seg.get("text", "").strip()]
            if not lines:
                continue
            speaker_text = "\n".join(lines)
            # Truncate to avoid context overflow
            if len(speaker_text.split()) > 1200:
                # Summarize speaker chunks first
                speaker_text = self._hierarchical_summarize(speaker_text)
            speaker_text = speaker_text[:3000]

            logger.info(f"[QwenAI] Summarizing speaker: {speaker} ({len(lines)} segments)")

            # Summary
            summary_raw = self._infer(
                SPEAKER_SUMMARY_PROMPT.format(speaker=speaker, transcript=speaker_text),
                max_new_tokens=200,
            )

            # Key points
            kp_raw = self._infer(
                SPEAKER_KEY_POINTS_PROMPT.format(speaker=speaker, transcript=speaker_text),
                max_new_tokens=350,
            )
            key_points = []
            if kp_raw:
                topic = None
                for line in kp_raw.split("\n"):
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("•"):
                        topic = line.lstrip("• ").rstrip(":")
                    elif topic and not line.startswith("•"):
                        key_points.append(f"{topic}: {line}")
                        topic = None
                    elif len(line) > 8:
                        key_points.append(line)
                if not key_points:
                    key_points = _clean_list(kp_raw)[:6]

            # Action items
            ai_raw = self._infer(
                SPEAKER_ACTION_ITEMS_PROMPT.format(speaker=speaker, transcript=speaker_text),
                max_new_tokens=250,
            )
            action_items = [] if (not ai_raw or "none identified" in ai_raw.lower()) else _clean_list(ai_raw)

            results[speaker] = {
                "summary": summary_raw or "",
                "key_points": key_points[:6],
                "action_items": action_items[:8],
            }
            logger.info(f"[QwenAI] Speaker '{speaker}' done: {len(key_points)} kp, {len(action_items)} ai")

        return results

    # ── RAG Pipeline Methods (Raw MoM only) ──────────────────────────────────
    # These methods are ONLY used by the Raw MoM pipeline.
    # They do NOT affect the existing generate_mom() method.

    def parse_agenda_items(self, agenda_text: str) -> List[Dict]:
        """
        Parse an agenda document into a list of {topic, speaker} dicts.

        The topic is copied VERBATIM from the agenda — no rewriting or shortening.
        The speaker is the presenter/owner if explicitly mentioned, else None.

        Returns
        -------
        List of dicts: [{"topic": str, "speaker": str|None}]
        Empty list if parsing fails or no items found.
        """
        if not agenda_text or not agenda_text.strip():
            return []

        raw = self._infer(
            AGENDA_COMPRESS_PROMPT.format(text=agenda_text.strip()),
            max_new_tokens=1024,
        )
        if not raw:
            return []

        # Strip markdown fences
        raw = raw.strip()
        if raw.startswith("```json"):
            raw = raw[7:]
        elif raw.startswith("```"):
            raw = raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

        try:
            items = json.loads(raw)
            if not isinstance(items, list):
                raise ValueError("Expected JSON array")
            result = []
            for item in items:
                if isinstance(item, dict) and item.get("topic"):
                    result.append({
                        "topic": str(item["topic"]).strip(),
                        "speaker": str(item["speaker"]).strip() if item.get("speaker") else None,
                    })
            logger.info(f"[QwenAI] Parsed {len(result)} agenda items")
            return result
        except Exception as e:
            # Attempt to extract JSON array from the raw output
            match = re.search(r'\[.*\]', raw, re.DOTALL)
            if match:
                try:
                    items = json.loads(match.group())
                    if isinstance(items, list):
                        return [
                            {
                                "topic": str(i.get("topic", "")).strip(),
                                "speaker": str(i.get("speaker", "")).strip() or None,
                            }
                            for i in items
                            if isinstance(i, dict) and i.get("topic")
                        ]
                except Exception:
                    pass
            logger.warning(f"[QwenAI] Agenda parsing failed: {e} — raw: {raw[:200]}")
            return []

    def extract_raw_mom_for_agenda(
        self,
        agenda_topic: str,
        agenda_speaker: Optional[str],
        evidence: str,
    ) -> Dict:
        """
        Extract structured Raw MoM facts for ONE agenda item using retrieved evidence.

        The LLM performs ONLY extraction — no summarization, no introduction,
        no conclusion. All facts are preserved verbatim.

        Parameters
        ----------
        agenda_topic   : Exact topic text from the agenda (verbatim).
        agenda_speaker : Presenter/owner from the agenda, or None.
        evidence       : Merged retrieved context from all three FAISS stores.

        Returns
        -------
        Dict matching the raw_mom agenda schema:
        {
            "agenda_topic": str,
            "agenda_speaker": str|None,
            "discussion": [...]
        }
        Empty discussion list on failure.
        """
        if not evidence or not evidence.strip():
            return {
                "agenda_topic": agenda_topic,
                "agenda_speaker": agenda_speaker,
                "discussion": [],
            }

        raw = self._infer(
            RAW_MOM_EXTRACTION_PROMPT.format(
                agenda_topic=agenda_topic,
                agenda_speaker=agenda_speaker or "Not specified",
                evidence=evidence.strip(),
            ),
            max_new_tokens=1024,
        )

        if not raw:
            return {
                "agenda_topic": agenda_topic,
                "agenda_speaker": agenda_speaker,
                "discussion": [],
            }

        # Strip markdown fences
        logger.info("raw_mom", raw)
        raw = raw.strip()
        if raw.startswith("```json"):
            raw = raw[7:]
        elif raw.startswith("```"):
            raw = raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

        try:
            data = json.loads(raw)
            print(data)
            if not isinstance(data, dict):
                raise ValueError("Expected JSON object")

            discussion = data.get("discussion", [])
            if not isinstance(discussion, list):
                discussion = []

            # Normalize each discussion entry
            normalized = []
            for entry in discussion:
                if not isinstance(entry, dict):
                    continue
                dates = entry.get("dates", [])
                if not isinstance(dates, list):
                    dates = []

                # Preserve LLM extracted dates
                normalized_dates = []
                seen = set()

                for d in dates:
                    if not isinstance(d, dict):
                        continue

                    value = str(d.get("value", "")).strip()
                    purpose = str(d.get("purpose", "")).strip()

                    if value:
                        normalized_dates.append(
                            {
                                "value": value,
                                "purpose": purpose,
                            }
                        )
                        seen.add(value.lower())

                # Deterministically extract any missing dates
                text_to_scan = f"{entry.get('point', '')}"

                action = entry.get("action", {})
                if isinstance(action, dict):
                    text_to_scan += " " + str(action.get("description") or "")

                for d in extract_dates_from_text(text_to_scan):
                    if d["value"].lower() not in seen:
                        normalized_dates.append(d)
                        seen.add(d["value"].lower())

                action = entry.get("action", {})
                if not isinstance(action, dict):
                    action = {}

                normalized.append({
                    "type": str(entry.get("type", "discussion")),
                    "speaker": entry.get("speaker") or None,
                    "point": str(entry.get("point", "")).strip(),
                    "dates": normalized_dates,
                    "action": {
                        "owner": action.get("owner") or None,
                        "description": action.get("description") or None,
                        "deadline": action.get("deadline") or None,
                        "status": action.get("status") or None,
                    },
                })

            return {
                "agenda_topic": agenda_topic,
                "agenda_speaker": agenda_speaker,
                "discussion": normalized,
            }

        except Exception as e:
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                    if isinstance(data, dict):
                        return {
                            "agenda_topic": agenda_topic,
                            "agenda_speaker": agenda_speaker,
                            "discussion": data.get("discussion", []),
                        }
                except Exception:
                    pass
            logger.warning(
                f"[QwenAI] Raw MoM extraction failed for '{agenda_topic[:50]}': {e}"
            )
            return {
                "agenda_topic": agenda_topic,
                "agenda_speaker": agenda_speaker,
                "discussion": [],
            }

    def _parse_mom_json(self, raw: str, recording_meta: dict) -> Optional[dict]:

        """Utility to parse and clean MoM JSON output from Qwen3."""
        if not raw or not raw.strip():
            return None

        raw = raw.strip()
        if raw.startswith("```json"):
            raw = raw[7:]
        elif raw.startswith("```"):
            raw = raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except Exception:
                    logger.error("[QwenAI] Failed to parse MoM JSON from Qwen3 output")
                    return None
            else:
                logger.error("[QwenAI] No JSON found in Qwen3 MoM output")
                return None

        # Merge action_items + general_action_items
        ai_list = data.get("action_items", []) or []
        for g in (data.get("general_action_items", []) or []):
            if isinstance(g, str) and g.strip():
                ai_list.append({"task": g, "owner": "Unassigned", "deadline": "ASAP"})

        action_items = [
            {
                "task": (a.get("task", a) if isinstance(a, dict) else str(a)),
                "owner": (a.get("owner", "Unassigned") if isinstance(a, dict) else "Unassigned"),
                "deadline": (a.get("deadline", "ASAP") if isinstance(a, dict) else "ASAP"),
            }
            for a in ai_list
        ]

        points = data.get("points_discussed", []) or []
        if isinstance(points, list):
            points = [str(p) for p in points if p]
        else:
            points = []

        return {
            "title": data.get("title") or recording_meta.get("filename", "Meeting Notes"),
            "date": recording_meta.get("created_at", ""),
            "duration": recording_meta.get("duration", 0),
            "planned_start_time": "",
            "actual_start_time": data.get("actual_start_time") or "",
            "participants": recording_meta.get("speakers_detected", []),
            "introduction": data.get("introduction") or "",
            "points_discussed": points,
            "action_items": action_items,
            "conclusion": data.get("conclusion") or "",
        }

    def _split_context_into_sections(self, context: str, max_tokens: int = 4000) -> List[str]:
        """Split final meeting context into sections using token count, preserving line/sentence boundaries."""
        tokenizer = self._tokenizer
        lines = context.split("\n")
        sections = []
        current_lines = []
        current_tokens = 0

        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            if tokenizer:
                line_tokens = len(tokenizer.encode(line + "\n"))
            else:
                line_tokens = int(len(line_stripped) / 4) + 1
            if current_tokens + line_tokens > max_tokens and current_lines:
                sections.append("\n".join(current_lines))
                current_lines = [line]
                current_tokens = line_tokens
            else:
                current_lines.append(line)
                current_tokens += line_tokens

        if current_lines:
            sections.append("\n".join(current_lines))

        return sections

    def _generate_mom_section_wise(
        self,
        transcript: List[Dict],
        recording_meta: dict,
        context: str,
        agenda_section: str,
        reference_section: str,
    ) -> dict:
        """Sequential Section-wise MoM Generation pipeline with manual fallback support."""
        logger.info("[QwenAI] Starting Section-wise MoM Generation pipeline...")
        
        # 1. Split context by token count
        from config import settings
        sections = self._split_context_into_sections(context, max_tokens=settings.MOM_CONTEXT_TOKEN_THRESHOLD)
        logger.info(f"[QwenAI] Context split into {len(sections)} sections (threshold: {settings.MOM_CONTEXT_TOKEN_THRESHOLD}).")

        # 2. Generate partial MoM for each section
        partial_moms = []
        for idx, section in enumerate(sections):
            logger.info(f"[QwenAI] Generating partial MoM for section {idx+1}/{len(sections)}...")
            try:
                raw_sec = self._infer(
                    MOM_PROMPT.format(
                        transcript=section,
                        agenda_section=agenda_section,
                        reference_section=reference_section,
                    ),
                    max_new_tokens=1500,
                )
                sec_data = self._parse_mom_json(raw_sec, recording_meta)
                if sec_data:
                    partial_moms.append(sec_data)
            except Exception as sec_err:
                logger.warning(f"[QwenAI] Section {idx+1} generation failed: {sec_err}. Continuing with remaining sections...")
            finally:
                # Release temporary tensors
                import gc
                gc.collect()
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if not partial_moms:
            logger.error("[QwenAI] All partial MoMs failed to generate.")
            return _empty_mom(recording_meta)

        if len(partial_moms) == 1:
            logger.info("[QwenAI] Single partial MoM generated successfully. Skipping merge pass.")
            mom_data = partial_moms[0]
            return mom_data

        # 3. Merge partial MoMs using the LLM
        logger.info(f"[QwenAI] Merging {len(partial_moms)} partial MoMs using LLM...")
        partial_moms_json = json.dumps(partial_moms, indent=2)
        try:
            raw_merge = self._infer(
                MOM_MERGE_PROMPT.format(partial_moms_json=partial_moms_json),
                max_new_tokens=3072,
            )
            final_data = self._parse_mom_json(raw_merge, recording_meta)
            if final_data:
                logger.info("[QwenAI] MoM consolidation complete.")
                return final_data
        except Exception as merge_err:
            logger.error(f"[QwenAI] LLM MoM merge failed: {merge_err}. Using manual consolidation fallback.", exc_info=True)

        # 4. Fallback manual merge in Python
        merged_points = []
        merged_action_items = []
        seen_tasks = set()
        introductions = []
        conclusions = []

        for p_mom in partial_moms:
            if p_mom.get("introduction"):
                introductions.append(p_mom["introduction"])
            if p_mom.get("conclusion"):
                conclusions.append(p_mom["conclusion"])
            for pt in p_mom.get("points_discussed", []):
                if pt not in merged_points:
                    merged_points.append(pt)
            for item in p_mom.get("action_items", []):
                task_key = (item.get("task", "") or "").lower().strip()
                if task_key and task_key not in seen_tasks:
                    seen_tasks.add(task_key)
                    merged_action_items.append(item)

        return {
            "title": partial_moms[0].get("title") or recording_meta.get("filename", "Meeting Notes"),
            "date": recording_meta.get("created_at", ""),
            "duration": recording_meta.get("duration", 0),
            "planned_start_time": "",
            "actual_start_time": partial_moms[0].get("actual_start_time") or "",
            "participants": recording_meta.get("speakers_detected", []),
            "introduction": " ".join(introductions)[:1000],
            "points_discussed": merged_points,
            "action_items": merged_action_items,
            "conclusion": " ".join(conclusions)[:1000],
        }

    def generate_mom(
        self,
        transcript: List[Dict],
        recording_meta: dict,
        context: Optional[str] = None,
        agenda_summary: Optional[str] = None,
        reference_summary: Optional[str] = None,
    ) -> dict:
        """Generate full enterprise-grade Minutes of Meeting with Section-wise OOM fallback."""
        if context is None:
            dialogue = _format_transcript(transcript)
            if not dialogue.strip():
                return _empty_mom(recording_meta)
            context = self._hierarchical_summarize(dialogue)
        elif not context.strip():
            return _empty_mom(recording_meta)

        # Build optional agenda / reference inject sections
        if agenda_summary and agenda_summary.strip():
            agenda_section = (
                "\nMEETING AGENDA (use this to frame the title and introduction):\n"
                + agenda_summary.strip()
            )
        else:
            agenda_section = ""

        if reference_summary and reference_summary.strip():
            reference_section = (
                "\nREFERENCE KNOWLEDGE (background context - use to enrich the MoM):\n"
                + reference_summary.strip()
                + "\n\n"
            )
        else:
            reference_section = "\n"

        # Calculate final context token length and decide generation strategy
        from config import settings
        self._get_pipeline()  # Ensure model/tokenizer is loaded
        if self._tokenizer:
            context_tokens = len(self._tokenizer.encode(context))
        else:
            # Heuristic fallback if tokenizer is unavailable
            context_tokens = int(len(context) / 4)
        logger.info(f"[QwenAI] Final context length ({context_tokens} tokens).")


        if context_tokens > settings.MOM_CONTEXT_TOKEN_THRESHOLD:
            logger.info(
                f"[QwenAI] Final context length ({context_tokens} tokens) exceeds "
                f"threshold ({settings.MOM_CONTEXT_TOKEN_THRESHOLD} tokens). "
                "Automatically switching to Section-wise MoM Generation."
            )
            return self._generate_mom_section_wise(
                transcript=transcript,
                recording_meta=recording_meta,
                context=context,
                agenda_section=agenda_section,
                reference_section=reference_section,
            )

        # Try single-pass generation first
        try:
            raw = self._infer(
                MOM_PROMPT.format(
                    transcript=context,
                    agenda_section=agenda_section,
                    reference_section=reference_section,
                ),
                max_new_tokens=1500,
            )
            data = self._parse_mom_json(raw, recording_meta)
            if data and data.get("points_discussed"):
                return data
            raise ValueError("[QwenAI] Single-pass MoM returned empty points or invalid data")

        except Exception as e:
            err_msg = str(e).lower()
            is_oom = "out of memory" in err_msg or "oom" in err_msg or "cuda memory" in err_msg
            if is_oom:
                logger.warning(
                    f"[QwenAI] Single-pass MoM generation encountered OOM: {e}. "
                    "Triggering Section-wise MoM Generation pipeline fallback..."
                )
                return self._generate_mom_section_wise(
                    transcript=transcript,
                    recording_meta=recording_meta,
                    context=context,
                    agenda_section=agenda_section,
                    reference_section=reference_section,
                )
            else:
                logger.error(f"[QwenAI] Single-pass MoM generation failed: {e}", exc_info=True)
                return _empty_mom(recording_meta)


# ══════════════════════════════════════════════════════════════
# Provider singleton accessor
# ══════════════════════════════════════════════════════════════

_provider_instance: Optional[QwenProvider] = None


def get_provider() -> QwenProvider:
    """Return the singleton QwenProvider instance."""
    global _provider_instance
    if _provider_instance is None:
        _provider_instance = QwenProvider()
    return _provider_instance


def warm_up_model():
    """
    Pre-loading is disabled to optimize memory lifecycle.
    Models are loaded lazily on demand and unloaded immediately after use.
    """
    logger.info("[QwenAI] Model warm-up deferred (load-on-demand enabled).")


# ══════════════════════════════════════════════════════════════
# Backward-compatibility alias
# ══════════════════════════════════════════════════════════════

# Keep LlamaProvider as an alias so any import from external code
# that still references it does not crash immediately.
LlamaProvider = QwenProvider

