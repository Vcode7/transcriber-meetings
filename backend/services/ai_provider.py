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
You are an expert executive assistant generating professional Minutes of Meeting (MoM).

Analyze the transcript and return ONLY a valid JSON object. Do not include markdown, ```json, or any other text outside the JSON.

JSON schema (STRICTLY follow this — do not add or rename keys):
{{
  "title": "Concise professional meeting title",
  "introduction": "One paragraph (3-5 sentences) that clearly defines the meeting agenda, purpose, and topics discussed. This should read as a formal introduction to the meeting.",
  "points_discussed": [
    "Point 1 — a complete sentence describing what was discussed (minimum 3, maximum 10)",
    "Point 2 — another key discussion point"
  ],
  "action_items": [
    {{"task": "Task description", "owner": "Speaker name or Unassigned", "deadline": "Date or ASAP"}}
  ],
  "general_action_items": ["Action item not assigned to any specific speaker"],
  "conclusion": "One to two paragraphs summarizing outcomes, agreements reached, and the overall conclusion of the meeting.",
  "actual_start_time": "HH:MM if mentioned in transcript or null",
  "next_meeting_date": "Date/time if mentioned or null"
}}

Rules:
- introduction: Write exactly ONE paragraph. Define the meeting's agenda and all topics covered.
- points_discussed: List 3 to 10 points. Each point is a single complete sentence. Cover every major topic.
- action_items: Include speaker name as owner wherever possible. If general, use 'Unassigned'.
- general_action_items: Only items with no clear owner.
- conclusion: Summarize what was concluded, agreed upon, and any next steps.
- Do NOT hallucinate. Only report what is in the transcript.
- Preserve all technical terms, acronyms, project names exactly.
- If a field has no relevant content, use [] for arrays or null for strings.

TRANSCRIPT:
{transcript}

JSON MoM:"""



KEY_POINTS_PROMPT = """\
You are an expert meeting analyst. Extract the key discussion points from the following meeting transcript.

For each key point, provide:
• [Topic Name]
[One to three sentences explanation of what was discussed about this topic]

Extract 5–10 key points. Cover all major topics discussed.

Rules:
- Preserve all technical terms, acronyms, and project names exactly as spoken.
- Do NOT attribute to speakers.
- Do NOT repeat verbatim transcript sentences.

TRANSCRIPT:
{transcript}

Key discussion points:"""


ACTION_ITEMS_PROMPT = """\
You are an expert meeting analyst. Extract all action items from the following meeting transcript.

Return a numbered list. For each action item include:
1. [Task description] — Owner: [name or Unassigned] — Deadline: [date or ASAP]

If there are no action items, return: None identified.

Rules:
- Preserve all technical terms and project names exactly.
- Only include explicitly stated tasks, commitments, or follow-ups.

TRANSCRIPT:
{transcript}

Action items:"""


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
    def generate_summary(self, transcript: List[Dict]) -> str: ...

    @abstractmethod
    def generate_key_points(self, transcript: List[Dict]) -> List[str]: ...

    @abstractmethod
    def generate_action_items(self, transcript: List[Dict]) -> List[str]: ...

    @abstractmethod
    def generate_key_decisions(self, transcript: List[Dict]) -> List[str]: ...

    @abstractmethod
    def generate_mom(self, transcript: List[Dict], recording_meta: dict) -> dict: ...

    @abstractmethod
    def generate_executive_summary(self, transcript: List[Dict]) -> dict: ...

    @abstractmethod
    def generate_short_summary(self, transcript: List[Dict]) -> str: ...

    @abstractmethod
    def generate_detailed_summary(self, transcript: List[Dict]) -> str: ...

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
    def _get_pipeline(cls):
        """Return cached pipeline, loading model once if needed."""
        if cls._pipeline is not None:
            return cls._pipeline
        if cls._load_attempted:
            return None
        cls._load_attempted = True

        try:
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

            # ── Resolve model path ────────────────────────────────
            # In production the Qwen model is shipped unencrypted in
            # runtime/nlp-engine/ (plain directory, not a .dat).
            # ModelLoader._try_load_plain() will find it there.
            # In development mode it falls back to the HuggingFace Hub ID.
            local_path = ModelLoader.get_model_path("nlp_engine")
            if local_path is not None:
                load_from = str(local_path)
                local_files_only = True
                logger.info(f"[QwenAI] Loading from local path: {load_from}")
            else:
                load_from = model_id
                local_files_only = False
                logger.warning(
                    "[QwenAI] Local nlp-engine model not found in runtime/. "
                    "Falling back to HuggingFace Hub — requires internet access."
                )

            # Determine device
            use_cuda = torch.cuda.is_available()
            cls._device = "cuda" if use_cuda else "cpu"
            logger.info(f"[QwenAI] Loading {load_from} on {cls._device} ...")

            # 4-bit quantization config (saves ~50% VRAM)
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

            # Load tokenizer
            hf_token = getattr(settings, "HF_TOKEN", None) or None
            cls._tokenizer = AutoTokenizer.from_pretrained(
                load_from,
                local_files_only=local_files_only,
                token=hf_token if not local_files_only else None,
            )

            # Load model
            model_kwargs: Dict[str, Any] = {
                "local_files_only": local_files_only,
                "device_map": "auto" if use_cuda else None,
                "torch_dtype": torch.float16 if use_cuda else torch.float32,
            }
            if not local_files_only:
                model_kwargs["token"] = hf_token
            if bnb_config:
                model_kwargs["quantization_config"] = bnb_config

            cls._model = AutoModelForCausalLM.from_pretrained(load_from, **model_kwargs)
            cls._model.eval()

            # Build pipeline
            cls._pipeline = pipeline(
                "text-generation",
                model=cls._model,
                tokenizer=cls._tokenizer,
                device_map="auto" if use_cuda else None,
            )

            logger.info(f"[QwenAI] Qwen3 4B Instruct ready on {cls._device} ✓")

        except ImportError as e:
            logger.error(
                f"[QwenAI] Missing dependency: {e}. "
                "Run: pip install transformers bitsandbytes accelerate sentencepiece"
            )
            cls._pipeline = None
        except Exception as e:
            logger.error(f"[QwenAI] Failed to load Qwen3 model: {e}", exc_info=True)
            cls._pipeline = None

        return cls._pipeline

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
                max_new_tokens=200,
            )
            if summary:
                chunk_summaries.append(summary)

        merged = "\n\n".join(chunk_summaries)
        logger.info(f"[QwenAI] Merged {len(chunk_summaries)} chunk summaries ({len(merged.split())} words)")
        return merged

    # ── AIProvider interface ──────────────────────────────────

    def generate_summary(self, transcript: List[Dict]) -> str:
        """Generate a short meeting summary (backward compat alias for short_summary)."""
        return self.generate_short_summary(transcript)

    def generate_short_summary(self, transcript: List[Dict]) -> str:
        """Generate a concise ~60-word professional summary."""
        plain = _format_for_summary(transcript)
        if not plain.strip():
            return "No transcript available to summarize."

        # For very long transcripts, first compress
        compressed = self._hierarchical_summarize(plain)
        result = self._infer(
            SHORT_SUMMARY_PROMPT.format(transcript=compressed[:3000]),
            max_new_tokens=120,
        )
        return result or "Unable to generate summary."

    def generate_detailed_summary(self, transcript: List[Dict]) -> str:
        """Generate a comprehensive sectioned meeting report."""
        dialogue = _format_transcript(transcript)
        if not dialogue.strip():
            return "No transcript available to summarize."

        # For long transcripts use hierarchical approach
        compressed = self._hierarchical_summarize(dialogue)

        result = self._infer(
            DETAILED_SUMMARY_PROMPT.format(transcript=compressed[:4000]),
            max_new_tokens=1200,
        )
        return result or "Unable to generate detailed summary."

    def generate_key_points(self, transcript: List[Dict]) -> List[str]:
        """Extract key discussion points with Topic: Explanation format."""
        dialogue = _format_transcript(transcript)
        if not dialogue.strip():
            return []

        compressed = self._hierarchical_summarize(dialogue)
        raw = self._infer(
            KEY_POINTS_PROMPT.format(transcript=compressed[:3500]),
            max_new_tokens=600,
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
                # New topic
                current_topic = line.lstrip("• ").rstrip(":")
            elif current_topic and not line.startswith("•"):
                # Explanation line — combine with topic
                points.append(f"{current_topic}: {line}")
                current_topic = None
            elif not current_topic and len(line) > 10:
                points.append(line)

        return points[:10] if points else _clean_list(raw)[:8]

    def generate_action_items(self, transcript: List[Dict]) -> List[str]:
        """Extract action items with owner and deadline."""
        dialogue = _format_transcript(transcript)
        if not dialogue.strip():
            return []

        compressed = self._hierarchical_summarize(dialogue)
        raw = self._infer(
            ACTION_ITEMS_PROMPT.format(transcript=compressed[:3500]),
            max_new_tokens=400,
        )
        if not raw or "none identified" in raw.lower():
            return []
        return _clean_list(raw)

    def generate_key_decisions(self, transcript: List[Dict]) -> List[str]:
        """Extract concrete decisions made during the meeting."""
        dialogue = _format_transcript(transcript)
        if not dialogue.strip():
            return []

        compressed = self._hierarchical_summarize(dialogue)
        raw = self._infer(
            KEY_DECISIONS_PROMPT.format(transcript=compressed[:3500]),
            max_new_tokens=300,
        )
        if not raw or "none identified" in raw.lower():
            return []
        return _clean_list(raw)

    def generate_executive_summary(self, transcript: List[Dict]) -> dict:
        """Generate structured executive summary for PDF reports."""
        dialogue = _format_transcript(transcript)
        if not dialogue.strip():
            return {
                "purpose": "No transcript available.",
                "discussion_points": [],
                "outcomes": [],
                "next_steps": [],
            }

        compressed = self._hierarchical_summarize(dialogue)
        raw = self._infer(
            EXECUTIVE_SUMMARY_PROMPT.format(transcript=compressed[:4000]),
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
            if label in ("Unknown", "[Multiple Speakers]"):
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

    def generate_mom(self, transcript: List[Dict], recording_meta: dict) -> dict:
        """Generate full enterprise-grade Minutes of Meeting."""
        dialogue = _format_transcript(transcript)
        if not dialogue.strip():
            return _empty_mom(recording_meta)

        compressed = self._hierarchical_summarize(dialogue)

        raw = self._infer(
            MOM_PROMPT.format(transcript=compressed[:4500]),
            max_new_tokens=1500,
        )

        if not raw:
            return _empty_mom(recording_meta)

        # Strip markdown code fences if present
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
            # Try to extract JSON object from surrounding text
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except Exception:
                    logger.error("[QwenAI] Failed to parse MoM JSON from Qwen3 output")
                    return _empty_mom(recording_meta)
            else:
                logger.error("[QwenAI] No JSON found in Qwen3 MoM output")
                return _empty_mom(recording_meta)

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

        # points_discussed: enforce 3-10 items
        points = data.get("points_discussed", []) or []
        if isinstance(points, list):
            points = [str(p) for p in points if p][:10]
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
    Pre-load the Qwen3 4B model during application startup.
    Call this once from the FastAPI lifespan to avoid cold-start delays.
    """
    logger.info("[QwenAI] Warming up Qwen3 4B Instruct model...")
    provider = get_provider()
    pipe = provider._get_pipeline()
    if pipe is not None:
        logger.info("[QwenAI] Model warm-up complete ✓")
    else:
        logger.warning("[QwenAI] Model warm-up failed — inference will be unavailable")


# ══════════════════════════════════════════════════════════════
# Backward-compatibility alias
# ══════════════════════════════════════════════════════════════

# Keep LlamaProvider as an alias so any import from external code
# that still references it does not crash immediately.
LlamaProvider = QwenProvider
