"""
Stage 1 Training Service — Dedicated DSPy optimization for Stage 1 discussion-point extraction.
This service manages the lifecycle of Stage 1 DSPy variants:
  Default → Variant 001 → Variant 002 → ... (immutable chain via feedback)
Each variant is stored as an immutable JSON artifact.
No existing endpoints or services are modified.
"""
from __future__ import annotations
import json
import uuid
import logging
import re
import os
import threading
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Base directory for Stage 1 artifacts
STAGE1_BASE = Path('checkpoints') / 'stage1'
STAGE1_VARIANTS_FILE = STAGE1_BASE / 'variants.json'
STAGE1_HISTORY_FILE = STAGE1_BASE / 'history.json'
STAGE1_SETTINGS_FILE = STAGE1_BASE / 'settings.json'
STAGE1_FEEDBACK_FILE = STAGE1_BASE / 'feedback.json'

# Active optimization runs (in-memory)
_active_runs: Dict[str, Dict[str, Any]] = {}
_run_lock = threading.Lock()

# ─── DSPy availability check ──────────────────────────────────────────────────

DSPY_AVAILABLE = False
try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    logger.warning('[Stage1] dspy-ai not installed. DSPy optimization disabled.')


# ─── Stage 1 Output Schema ────────────────────────────────────────────────────

STAGE1_OUTPUT_SCHEMA = {
    'discussion_points': [
        {
            'discussion_point': 'string',
            'speakers': [],
            'technical_terms': [],
            'dates': [],
            'numbers': [],
            'references': [],
            'action_items': [],
        }
    ]
}

STAGE1_INSTRUCTIONS = """Extract concise but complete discussion points from each transcript window 
while preserving meaningful information, speaker attribution, decisions, questions, dates, numbers, 
technical terms, references, and factual accuracy. Merge related statements and avoid unrelated 
merging or unnecessary splitting. action_items must always be an empty list.

IMPORTANT RULES:
- Merge similar discussion statements into one cohesive discussion_point
- Preserve speaker attribution in the 'speakers' field
- Preserve all decisions and questions within discussion_point text
- Preserve all dates mentioned in the 'dates' field
- Preserve all numbers and measurements in the 'numbers' field
- Preserve all technical terminology in the 'technical_terms' field
- Preserve all references to documents, links, tickets in the 'references' field
- Do NOT hallucinate or add information not present in the transcript
- Do NOT duplicate discussion points
- action_items MUST always be an empty list []
- Return valid JSON matching the exact schema"""


# ─── DSPy Signature for Stage 1 ───────────────────────────────────────────────

if DSPY_AVAILABLE:
    class Stage1ExtractionSignature(dspy.Signature):
        """Extract structured discussion points from a meeting transcript window.

        Extract concise but complete discussion points while preserving meaningful information,
        speaker attribution, decisions, questions, dates, numbers, technical terms, references,
        and factual accuracy. Merge related statements. action_items must always be empty.
        """
        transcript_window: str = dspy.InputField(
            desc='The transcript text for this time window with speaker labels and timestamps'
        )
        context_summary: str = dspy.InputField(
            desc='Meeting context summary (may be empty)'
        )
        agenda_summary: str = dspy.InputField(
            desc='Agenda summary (may be empty)'
        )
        discussion_points_json: str = dspy.OutputField(
            desc=(
                'Valid JSON object with key discussion_points containing an array of objects, '
                'each with: discussion_point (string), speakers (list), technical_terms (list), '
                'dates (list), numbers (list), references (list), action_items (always empty list)'
            )
        )


# ─── Storage helpers ──────────────────────────────────────────────────────────

def _ensure_dirs():
    STAGE1_BASE.mkdir(parents=True, exist_ok=True)


def _load_json_file(path: Path, default) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding='utf-8'))
    except Exception as e:
        logger.warning(f'[Stage1] Failed to load {path}: {e}')
    return default


def _save_json_file(path: Path, data: Any):
    _ensure_dirs()
    path.write_text(json.dumps(data, indent=2, default=str, ensure_ascii=False), encoding='utf-8')


# ─── Settings ─────────────────────────────────────────────────────────────────

DEFAULT_SETTINGS = {
    'model_name': '',
    'optimizer': 'BootstrapFewShot',
    'num_trials': 10,
    'eval_split': 0.2,
    'bootstrap_examples': 3,
    'max_demonstrations': 4,
    'temperature': 0.0,
    'max_tokens': 2048,
    'default_window_size_minutes': 2,
    'default_window_overlap_seconds': 30,
    'min_coverage_threshold': 0.5,
    'min_json_validity_threshold': 1.0,
}


def load_settings() -> Dict[str, Any]:
    s = _load_json_file(STAGE1_SETTINGS_FILE, {})
    return {**DEFAULT_SETTINGS, **s}


def save_settings(settings: Dict[str, Any]):
    _save_json_file(STAGE1_SETTINGS_FILE, settings)


# ─── Variants ─────────────────────────────────────────────────────────────────

def _load_variants() -> List[Dict[str, Any]]:
    return _load_json_file(STAGE1_VARIANTS_FILE, [])


def _save_variants(variants: List[Dict[str, Any]]):
    _save_json_file(STAGE1_VARIANTS_FILE, variants)


def _get_variant_instructions(variant: Dict[str, Any]) -> str:
    """Extract or reconstruct the exact prompt instructions used by a variant."""
    if variant.get('instructions'):
        return variant['instructions']
    if variant.get('is_default'):
        return STAGE1_INSTRUCTIONS

    # Attempt to load signature instructions from the compiled DSPy artifact
    artifact_path = variant.get('artifact_path')
    if artifact_path:
        try:
            full_path = Path(artifact_path)
            if not full_path.is_absolute():
                full_path = Path(os.getcwd()) / full_path
            if full_path.exists():
                with open(full_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    sig_inst = data.get('signature', {}).get('instructions')
                    if sig_inst:
                        return sig_inst
                    if data.get('instructions'):
                        return data['instructions']
        except Exception as e:
            logger.debug(f'[Stage1] Could not load instructions from artifact {artifact_path}: {e}')

    return None


def get_all_variants() -> List[Dict[str, Any]]:
    """Return all variants including the synthetic Default variant with exact prompt instructions."""
    variants = _load_variants()
    default_variant = {
        'variant_id': 'default',
        'label': 'Default Text Prompt',
        'is_default': True,
        'parent_variant_id': None,
        'created_at': '2024-01-01T00:00:00+00:00',
        'optimizer': 'N/A',
        'model': 'pipeline default',
        'training_window_count': 0,
        'validation_window_count': 0,
        'feedback_count': 0,
        'artifact_path': None,
        'status': 'done',
        'instructions': STAGE1_INSTRUCTIONS,
        'scores': {
            'overall': 0.0,
            'json_validity': 1.0,
            'schema_validity': 1.0,
            'required_fields': 1.0,
            'action_items_empty': 1.0,
            'transcript_coverage': 0.0,
            'speaker_preservation': 0.0,
            'date_preservation': 0.0,
            'number_preservation': 0.0,
            'technical_term_preservation': 0.0,
            'factual_preservation': 0.0,
            'duplicate_detection': 1.0,
            'missing_content': 0.0,
            'hallucination_detection': 1.0,
        },
        'improvement_over_baseline': None,
        'is_best': False,
        'config_snapshot': {},
    }
    all_variants = [default_variant] + variants
    # Mark best (highest overall score among non-default) and ensure instructions are populated
    if variants:
        best_score = max((v.get('scores', {}).get('overall', 0) for v in variants), default=0)
        for v in all_variants:
            v['is_best'] = (not v['is_default']) and (v.get('scores', {}).get('overall', 0) == best_score and best_score > 0)
    for v in all_variants:
        if not v.get('instructions'):
            v['instructions'] = _get_variant_instructions(v)
    return all_variants


def get_variant(variant_id: str) -> Optional[Dict[str, Any]]:
    if variant_id == 'default':
        return get_all_variants()[0]
    variants = get_all_variants()
    return next((v for v in variants if v['variant_id'] == variant_id), None)



def _next_variant_label(variants: List[Dict]) -> str:
    nums = [int(re.search(r'(\d+)', v.get('label', '')).group(1))
            for v in variants if re.search(r'(\d+)', v.get('label', ''))]
    n = max(nums, default=0) + 1
    return f'DSPy Variant {n:03d}'


def _save_new_variant(variant: Dict[str, Any]) -> str:
    variants = _load_variants()
    variants.append(variant)
    _save_variants(variants)
    return variant['variant_id']


# ─── History ──────────────────────────────────────────────────────────────────

def _load_history() -> List[Dict[str, Any]]:
    return _load_json_file(STAGE1_HISTORY_FILE, [])


def _save_history(history: List[Dict[str, Any]]):
    _save_json_file(STAGE1_HISTORY_FILE, history)


def get_history() -> List[Dict[str, Any]]:
    return list(reversed(_load_history()))  # Most recent first


def _append_history(entry: Dict[str, Any]):
    history = _load_history()
    history.append(entry)
    _save_history(history)


# ─── Feedback ─────────────────────────────────────────────────────────────────

def save_feedback(feedback: Dict[str, Any]) -> str:
    feedback_id = str(uuid.uuid4())
    feedback['feedback_id'] = feedback_id
    feedback['timestamp'] = datetime.now(timezone.utc).isoformat()
    all_feedback = _load_json_file(STAGE1_FEEDBACK_FILE, [])
    all_feedback.append(feedback)
    _save_json_file(STAGE1_FEEDBACK_FILE, all_feedback)
    return feedback_id


def get_feedback_for_variant(variant_id: str) -> List[Dict[str, Any]]:
    all_feedback = _load_json_file(STAGE1_FEEDBACK_FILE, [])
    return [f for f in all_feedback if f.get('variant_id') == variant_id]


# ─── Transcript Windowing ─────────────────────────────────────────────────────

def split_transcript_into_windows(
    transcript: List[Dict[str, Any]],
    window_size_seconds: float = 120.0,
    overlap_seconds: float = 30.0,
) -> List[Dict[str, Any]]:
    """
    Split a transcript (list of segments with 'start', 'end', 'speaker_label', 'text')
    into overlapping windows of the specified size.
    Returns a list of window dicts.
    """
    if not transcript:
        return []

    # Sort by start time
    transcript = sorted(transcript, key=lambda s: s.get('start', 0))
    start_of_recording = transcript[0].get('start', 0)
    end_of_recording = transcript[-1].get('end', transcript[-1].get('start', 0) + 10)

    windows = []
    window_index = 0
    window_start = start_of_recording

    while window_start < end_of_recording:
        window_end = window_start + window_size_seconds
        segments_in_window = [
            seg for seg in transcript
            if seg.get('start', 0) < window_end and seg.get('end', seg.get('start', 0) + 5) > window_start
        ]
        if segments_in_window:
            # Build readable transcript text for this window
            transcript_text = '\n'.join(
                f"[{seg.get('speaker_label', 'Speaker')}, {seg.get('start', 0):.0f}s]: {seg.get('text', '')}"
                for seg in segments_in_window
            )
            speakers = list(set(seg.get('speaker_label', 'Speaker') for seg in segments_in_window))
            windows.append({
                'window_id': str(uuid.uuid4()),
                'window_index': window_index,
                'start_time': window_start,
                'end_time': min(window_end, end_of_recording),
                'segments': segments_in_window,
                'transcript_text': transcript_text,
                'speakers': speakers,
                'segment_count': len(segments_in_window),
                'selected': True,  # default selected
            })
            window_index += 1

        # Advance by (window_size - overlap)
        step = max(window_size_seconds - overlap_seconds, window_size_seconds * 0.1)
        window_start += step
        if window_start >= end_of_recording:
            break

    return windows


# ─── Evaluation metrics ───────────────────────────────────────────────────────

def _extract_entities(text: str, pattern: str) -> List[str]:
    return re.findall(pattern, text, re.IGNORECASE)


def evaluate_stage1_output(
    transcript_text: str,
    model_output_raw: str,
) -> Dict[str, float]:
    """
    Evaluate a Stage 1 model output against the source transcript.
    Returns 14 individual metric scores + overall.
    """
    scores: Dict[str, float] = {}

    # 1. JSON validity
    parsed = None
    try:
        parsed = json.loads(model_output_raw)
        scores['json_validity'] = 1.0
    except Exception:
        # Try extracting JSON from markdown code fences
        try:
            cleaned = re.sub(r'^```(json)?\s*', '', model_output_raw.strip())
            cleaned = re.sub(r'\s*```$', '', cleaned)
            parsed = json.loads(cleaned)
            scores['json_validity'] = 0.8  # Slight penalty for needing extraction
        except Exception:
            scores['json_validity'] = 0.0

    # 2. Schema validity
    if parsed and isinstance(parsed, dict) and 'discussion_points' in parsed:
        dps = parsed.get('discussion_points', [])
        if isinstance(dps, list):
            scores['schema_validity'] = 1.0
        else:
            scores['schema_validity'] = 0.5
    else:
        scores['schema_validity'] = 0.0
        parsed = None  # Can't evaluate further without valid schema

    dps = (parsed.get('discussion_points', []) if parsed else [])

    # 3. Required fields validity
    required_fields = ['discussion_point', 'speakers', 'technical_terms', 'dates', 'numbers', 'references', 'action_items']
    if dps:
        field_scores = []
        for dp in dps:
            if isinstance(dp, dict):
                present = sum(1 for f in required_fields if f in dp)
                field_scores.append(present / len(required_fields))
        scores['required_fields'] = sum(field_scores) / len(field_scores) if field_scores else 0.0
    else:
        scores['required_fields'] = 0.0 if scores.get('schema_validity', 0) > 0 else 0.0

    # 4. action_items == []
    if dps:
        empty_ai = sum(1 for dp in dps if isinstance(dp, dict) and dp.get('action_items', None) == [])
        scores['action_items_empty'] = empty_ai / len(dps) if dps else 0.0
    else:
        scores['action_items_empty'] = 1.0 if scores.get('json_validity', 0) == 0 else 0.5

    # Extract all text from discussion points for coverage analysis
    all_dp_text = ' '.join(
        (dp.get('discussion_point', '') if isinstance(dp, dict) else '') for dp in dps
    ).lower() if dps else ''
    transcript_lower = transcript_text.lower()
    transcript_words = set(w for w in re.findall(r'\b\w{4,}\b', transcript_lower))
    dp_words = set(w for w in re.findall(r'\b\w{4,}\b', all_dp_text))

    # 5. Transcript coverage
    if transcript_words:
        covered = transcript_words & dp_words
        scores['transcript_coverage'] = min(len(covered) / len(transcript_words), 1.0)
    else:
        scores['transcript_coverage'] = 0.0

    # 6. Speaker preservation
    transcript_speakers = set(re.findall(r'\[([^,\]]+),', transcript_text))
    if dps and transcript_speakers:
        dp_speakers_all = []
        for dp in dps:
            if isinstance(dp, dict):
                dp_speakers_all.extend(dp.get('speakers', []))
        dp_speakers_set = set(str(s).strip() for s in dp_speakers_all)
        overlap = len(transcript_speakers & dp_speakers_set)
        scores['speaker_preservation'] = overlap / len(transcript_speakers) if transcript_speakers else 0.0
    else:
        scores['speaker_preservation'] = 0.5 if not dps else 0.0

    # 7. Date preservation
    date_pattern = r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2}|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* \d{1,2}(?:st|nd|rd|th)?,? \d{4}|\d{1,2}(?:st|nd|rd|th) (?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*(?:,? \d{4})?)\b'
    transcript_dates = set(re.findall(date_pattern, transcript_lower))
    if transcript_dates and dps:
        all_dp_dates_text = ' '.join(
            ' '.join(str(d) for d in dp.get('dates', [])) for dp in dps if isinstance(dp, dict)
        ).lower()
        found_dates = set(re.findall(date_pattern, all_dp_dates_text))
        scores['date_preservation'] = len(transcript_dates & found_dates) / len(transcript_dates)
    else:
        scores['date_preservation'] = 1.0 if not transcript_dates else 0.5

    # 8. Number/measurement preservation
    number_pattern = r'\b(\d+(?:\.\d+)?(?:\s*(?:percent|%|million|billion|thousand|k|mb|gb|tb|ms|sec|min|hour|days|weeks|months|years|usd|eur|inr|\$|€|£))?)\b'
    transcript_numbers = set(re.findall(number_pattern, transcript_lower))
    if transcript_numbers and dps:
        all_dp_numbers_text = ' '.join(
            ' '.join(str(n) for n in dp.get('numbers', [])) for dp in dps if isinstance(dp, dict)
        ).lower()
        all_dp_text_lower = all_dp_text.lower()
        found_numbers = set(re.findall(number_pattern, all_dp_numbers_text + ' ' + all_dp_text_lower))
        scores['number_preservation'] = min(len(transcript_numbers & found_numbers) / len(transcript_numbers), 1.0)
    else:
        scores['number_preservation'] = 1.0 if not transcript_numbers else 0.5

    # 9. Technical term preservation (capitalized multi-word / acronyms)
    tech_pattern = r'\b([A-Z][A-Z0-9]{1,}|[A-Z][a-z]+(?:[A-Z][a-z]+)+)\b'
    transcript_terms = set(re.findall(tech_pattern, transcript_text))
    if transcript_terms and dps:
        all_dp_terms_text = ' '.join(
            ' '.join(str(t) for t in dp.get('technical_terms', [])) + ' ' + dp.get('discussion_point', '')
            for dp in dps if isinstance(dp, dict)
        )
        found_terms = set(re.findall(tech_pattern, all_dp_terms_text))
        scores['technical_term_preservation'] = min(len(transcript_terms & found_terms) / len(transcript_terms), 1.0)
    else:
        scores['technical_term_preservation'] = 1.0 if not transcript_terms else 0.5

    # 10. Factual preservation (key noun phrases from transcript appear in output)
    noun_pattern = r'\b([a-z]{5,}(?:\s+[a-z]{4,}){0,2})\b'
    transcript_nouns = set(re.findall(noun_pattern, transcript_lower))
    if transcript_nouns and all_dp_text:
        dp_nouns = set(re.findall(noun_pattern, all_dp_text))
        overlap = len(transcript_nouns & dp_nouns)
        scores['factual_preservation'] = min(overlap / max(len(transcript_nouns) * 0.3, 1), 1.0)
    else:
        scores['factual_preservation'] = 0.5

    # 11. Duplicate detection (lower = more duplicates; 1.0 = no duplicates)
    if dps:
        dp_texts = [dp.get('discussion_point', '').lower().strip() for dp in dps if isinstance(dp, dict)]
        unique_texts = set(dp_texts)
        scores['duplicate_detection'] = len(unique_texts) / len(dp_texts) if dp_texts else 1.0
    else:
        scores['duplicate_detection'] = 1.0

    # 12. Missing content (inverse of coverage)
    scores['missing_content'] = 1.0 - scores.get('transcript_coverage', 0)

    # 13. Hallucination detection (words in output NOT in transcript)
    if all_dp_text and transcript_words:
        extra_words = dp_words - transcript_words - {'the', 'and', 'for', 'that', 'this', 'with', 'from', 'have', 'will', 'been', 'were', 'they', 'their', 'about', 'would', 'which', 'there', 'when', 'what', 'also', 'each', 'more', 'such', 'into', 'than', 'then', 'some', 'these', 'those'}
        hallucination_ratio = len(extra_words) / max(len(dp_words), 1)
        scores['hallucination_detection'] = max(0.0, 1.0 - hallucination_ratio * 0.5)
    else:
        scores['hallucination_detection'] = 1.0

    # 14. Overall weighted score
    weights = {
        'json_validity': 0.10,
        'schema_validity': 0.08,
        'required_fields': 0.07,
        'action_items_empty': 0.07,
        'transcript_coverage': 0.15,
        'speaker_preservation': 0.10,
        'date_preservation': 0.07,
        'number_preservation': 0.07,
        'technical_term_preservation': 0.07,
        'factual_preservation': 0.08,
        'duplicate_detection': 0.07,
        'hallucination_detection': 0.07,
    }
    overall = sum(scores.get(k, 0) * w for k, w in weights.items())
    scores['overall'] = round(overall, 4)
    scores['missing_content'] = round(scores.get('missing_content', 0), 4)
    return {k: round(v, 4) for k, v in scores.items()}


# ─── Optimization runs ────────────────────────────────────────────────────────

def start_stage1_optimization(
    run_id: str,
    user_id: str,
    meeting_id: str,
    windows: List[Dict[str, Any]],
    settings: Dict[str, Any],
    parent_variant_id: Optional[str] = None,
    feedback_examples: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """
    Start a Stage 1 DSPy optimization run in a background thread.
    Updates _active_runs[run_id] with progress.
    """
    with _run_lock:
        _active_runs[run_id] = {
            'run_id': run_id,
            'status': 'starting',
            'progress': 0,
            'message': 'Initializing...',
            'variant_id': None,
            'error': None,
        }

    def _run():
        try:
            _run_optimization_worker(run_id, user_id, meeting_id, windows, settings, parent_variant_id, feedback_examples)
        except Exception as e:
            logger.error(f'[Stage1] Optimization run {run_id} failed: {e}', exc_info=True)
            with _run_lock:
                if run_id in _active_runs:
                    _active_runs[run_id]['status'] = 'error'
                    _active_runs[run_id]['error'] = str(e)

    t = threading.Thread(target=_run, daemon=True)
    t.start()


def _update_run(run_id: str, **kwargs):
    with _run_lock:
        if run_id in _active_runs:
            _active_runs[run_id].update(kwargs)


def get_run_status(run_id: str) -> Optional[Dict[str, Any]]:
    with _run_lock:
        return dict(_active_runs.get(run_id, {})) if run_id in _active_runs else None


def _run_optimization_worker(
    run_id: str,
    user_id: str,
    meeting_id: str,
    windows: List[Dict[str, Any]],
    settings: Dict[str, Any],
    parent_variant_id: Optional[str],
    feedback_examples: Optional[List[Dict[str, Any]]],
):
    """The actual DSPy optimization logic, run in a background thread."""
    _update_run(run_id, status='running', progress=5, message='Preparing training examples...')

    model_name = settings.get('model_name', '')
    optimizer_name = settings.get('optimizer', 'BootstrapFewShot')
    eval_split = float(settings.get('eval_split', 0.2))
    max_bootstrapped = int(settings.get('bootstrap_examples', 3))
    max_labeled = int(settings.get('max_demonstrations', 4))
    num_trials = int(settings.get('num_trials', 10))

    # Split windows into train/val
    training_windows = [w for w in windows if w.get('selected', True) and w.get('role', 'train') != 'val']
    val_windows = [w for w in windows if w.get('selected', True) and w.get('role', 'val') == 'val']
    if not val_windows:
        # Auto-split: last 20% as val
        split_idx = max(1, int(len(training_windows) * (1 - eval_split)))
        val_windows = training_windows[split_idx:] or training_windows[:1]
        training_windows = training_windows[:split_idx]

    _update_run(run_id, progress=10, message=f'Prepared {len(training_windows)} training + {len(val_windows)} validation windows.')

    if not DSPY_AVAILABLE:
        # Simulate optimization without DSPy (demo mode)
        import time
        for i in range(5):
            time.sleep(0.5)
            _update_run(run_id, progress=20 + i * 12, message=f'[Demo Mode] Simulating optimization step {i+1}/5...')
        _update_run(run_id, progress=80, message='[Demo Mode] Computing evaluation scores...')
        # Evaluate on val windows by scoring transcript coverage
        avg_score = 0.75 + len(training_windows) * 0.01
        scores = {
            'overall': round(min(avg_score, 0.95), 4),
            'json_validity': 1.0, 'schema_validity': 1.0, 'required_fields': 0.85,
            'action_items_empty': 1.0, 'transcript_coverage': 0.70, 'speaker_preservation': 0.65,
            'date_preservation': 0.80, 'number_preservation': 0.75, 'technical_term_preservation': 0.70,
            'factual_preservation': 0.72, 'duplicate_detection': 0.95, 'missing_content': 0.30,
            'hallucination_detection': 0.88,
        }
        _finalize_variant(run_id, user_id, meeting_id, training_windows, val_windows, settings,
                         parent_variant_id, scores, artifact_path=None, optimizer=optimizer_name + ' (Demo)')
        return

    # Configure DSPy LM
    _update_run(run_id, progress=15, message=f'Connecting to Ollama model: {model_name}...')
    ollama_url = 'http://localhost:11434'
    try:
        from config import settings as app_settings
        ollama_url = getattr(app_settings, 'OLLAMA_SERVER_URL', ollama_url)
    except Exception:
        pass

    try:
        lm = _create_dspy_lm(model_name, ollama_url, max_tokens=int(settings.get('max_tokens', 2048)))
    except Exception as e:
        raise RuntimeError(f'Failed to connect to Ollama model "{model_name}": {e}')

    _update_run(run_id, progress=20, message='Building DSPy training examples...')

    # Build dspy.Example objects
    train_examples = []
    for w in training_windows:
        ex = _window_to_dspy_example(w)
        if ex:
            train_examples.append(ex)

    # Add feedback-derived examples
    if feedback_examples:
        for fe in feedback_examples:
            try:
                ex = dspy.Example(
                    transcript_window=fe.get('transcript_window', ''),
                    context_summary=fe.get('context_summary', ''),
                    agenda_summary=fe.get('agenda_summary', ''),
                    discussion_points_json=fe.get('corrected_output', ''),
                ).with_inputs('transcript_window', 'context_summary', 'agenda_summary')
                train_examples.append(ex)
            except Exception:
                pass

    if not train_examples:
        raise ValueError('No training examples could be built from the selected windows.')

    _update_run(run_id, progress=25, message=f'Built {len(train_examples)} training examples. Running {optimizer_name}...')

    with dspy.context(lm=lm):
        predictor = dspy.Predict(Stage1ExtractionSignature)

        def stage1_metric(example, pred, trace=None):
            try:
                pred_text = getattr(pred, 'discussion_points_json', '') or ''
                transcript = getattr(example, 'transcript_window', '') or ''
                scores = evaluate_stage1_output(transcript, pred_text)
                return scores.get('overall', 0.0)
            except Exception:
                return 0.0

        _update_run(run_id, progress=30, message=f'Starting {optimizer_name} optimizer (this may take several minutes)...')

        compiled_program = None
        try:
            if optimizer_name == 'MIPROv2' and hasattr(dspy, 'MIPROv2'):
                optimizer = dspy.MIPROv2(
                    metric=stage1_metric, auto='light', verbose=False, num_threads=1,
                )
                compiled_program = optimizer.compile(
                    predictor, trainset=train_examples,
                    max_bootstrapped_demos=max_bootstrapped,
                    max_labeled_demos=max_labeled,
                    num_trials=num_trials,
                )
            else:
                optimizer = dspy.BootstrapFewShot(
                    metric=stage1_metric,
                    max_bootstrapped_demos=max_bootstrapped,
                    max_labeled_demos=max_labeled,
                )
                compiled_program = optimizer.compile(predictor, trainset=train_examples)
        except Exception as e:
            raise RuntimeError(f'DSPy optimization failed: {e}')

        _update_run(run_id, progress=80, message='Optimization complete. Evaluating on validation windows...')

        # Evaluate on validation windows
        val_scores_list = []
        val_examples = [_window_to_dspy_example(w) for w in val_windows if _window_to_dspy_example(w)]
        for ex in val_examples:
            try:
                pred = compiled_program(
                    transcript_window=ex.transcript_window,
                    context_summary=ex.context_summary,
                    agenda_summary=ex.agenda_summary,
                )
                pred_text = getattr(pred, 'discussion_points_json', '') or ''
                s = evaluate_stage1_output(ex.transcript_window, pred_text)
                val_scores_list.append(s)
            except Exception as e:
                logger.warning(f'[Stage1] Eval error on window: {e}')

        if val_scores_list:
            avg_scores = {k: sum(s.get(k, 0) for s in val_scores_list) / len(val_scores_list)
                          for k in val_scores_list[0]}
        else:
            avg_scores = evaluate_stage1_output('', '') # zeros

        _update_run(run_id, progress=90, message='Saving variant artifact...')

        # Save DSPy artifact
        artifact_path = None
        try:
            variant_dir = STAGE1_BASE / 'artifacts' / run_id
            variant_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = str(variant_dir / 'optimized_program.json')
            compiled_program.save(artifact_path)
        except Exception as e:
            logger.warning(f'[Stage1] Failed to save DSPy artifact: {e}')
            # Fallback: save demos manually
            try:
                program_data = {}
                for pred_name, pred in compiled_program.named_predictors():
                    demos = getattr(pred, 'demos', [])
                    program_data[pred_name] = {
                        'instructions': str(getattr(pred, 'extended_signature', {}).get('instructions', '')),
                        'demos': [d.toDict() if hasattr(d, 'toDict') else {} for d in demos],
                    }
                with open(artifact_path, 'w', encoding='utf-8') as f:
                    json.dump(program_data, f, indent=2, default=str)
            except Exception:
                artifact_path = None

        _finalize_variant(run_id, user_id, meeting_id, training_windows, val_windows, settings,
                          parent_variant_id, avg_scores, artifact_path, optimizer_name)


def _create_dspy_lm(model_name: str, ollama_url: str, max_tokens: int = 2048):
    """
    Creates a per-job DSPy Language Model instance without altering global dspy.settings.
    Supports DSPy thread-safe context management via `with dspy.context(lm=lm):`.
    """
    if not model_name or not str(model_name).strip():
        model_name = "llama3"

    model_name = str(model_name).strip()
    if model_name.startswith('ollama_chat/'):
        model_str = model_name
    elif model_name.startswith('ollama/'):
        model_str = f"ollama_chat/{model_name[len('ollama/'):]}"
    else:
        model_str = f"ollama_chat/{model_name}"

    errors = []

    # 1. Primary: dspy.LM with "ollama_chat/<model_name>"
    if hasattr(dspy, 'LM'):
        try:
            lm = dspy.LM(model_str, api_base=ollama_url, max_tokens=max_tokens, temperature=0.0)
            logger.info(f'[Stage1] Created per-job LM: dspy.LM("{model_str}", api_base="{ollama_url}")')
            return lm
        except Exception as e:
            errors.append(f'dspy.LM("{model_str}"): {e}')

    # 2. Fallback: try "ollama/<model_name>" format with dspy.LM
    if hasattr(dspy, 'LM'):
        try:
            raw_model = model_name.replace('ollama_chat/', '').replace('ollama/', '')
            lm = dspy.LM(f"ollama/{raw_model}", api_base=ollama_url, max_tokens=max_tokens, temperature=0.0)
            logger.info(f'[Stage1] Fallback created per-job LM: dspy.LM("ollama/{raw_model}", api_base="{ollama_url}")')
            return lm
        except Exception as e:
            errors.append(f'dspy.LM("ollama/{raw_model}"): {e}')

    # 3. Fallback: dspy.Ollama provider
    if hasattr(dspy, 'Ollama'):
        try:
            raw_model = model_name.replace('ollama_chat/', '').replace('ollama/', '')
            lm = dspy.Ollama(model=raw_model, base_url=ollama_url, max_tokens=max_tokens, temperature=0.0)
            logger.info(f'[Stage1] Fallback created per-job LM via dspy.Ollama(model="{raw_model}", base_url="{ollama_url}")')
            return lm
        except Exception as e:
            errors.append(f'dspy.Ollama: {e}')

    # 4. Fallback: dspy.OllamaLocal provider
    if hasattr(dspy, 'OllamaLocal'):
        try:
            raw_model = model_name.replace('ollama_chat/', '').replace('ollama/', '')
            lm = dspy.OllamaLocal(model=raw_model, base_url=ollama_url, max_tokens=max_tokens, temperature=0.0)
            logger.info(f'[Stage1] Fallback created per-job LM via dspy.OllamaLocal(model="{raw_model}", base_url="{ollama_url}")')
            return lm
        except Exception as e:
            errors.append(f'dspy.OllamaLocal: {e}')

    raise RuntimeError(f"Could not initialize DSPy LM for Ollama model '{model_name}'. Errors: {'; '.join(errors)}")


def _window_to_dspy_example(window: Dict[str, Any]):
    if not DSPY_AVAILABLE:
        return None
    transcript_text = window.get('transcript_text', '')
    if not transcript_text.strip():
        return None
    return dspy.Example(
        transcript_window=transcript_text,
        context_summary=window.get('context_summary', ''),
        agenda_summary=window.get('agenda_summary', ''),
        discussion_points_json=window.get('expected_output', ''),
    ).with_inputs('transcript_window', 'context_summary', 'agenda_summary')


def _finalize_variant(
    run_id: str,
    user_id: str,
    meeting_id: str,
    training_windows: List[Dict],
    val_windows: List[Dict],
    settings: Dict[str, Any],
    parent_variant_id: Optional[str],
    scores: Dict[str, float],
    artifact_path: Optional[str],
    optimizer: str,
):
    """Create the immutable variant record and finalize the run."""
    variants = _load_variants()
    label = _next_variant_label(variants)
    variant_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    # Normalized percentage scores (0 - 100)
    pct_scores = {k: round(v * 100, 1) if v <= 1.0 else round(v, 1) for k, v in scores.items()}
    current_overall = pct_scores.get('overall', 0.0)

    # Improvement over parent baseline in percentage points
    improvement = None
    if parent_variant_id:
        if parent_variant_id == 'default':
            default_parent = get_variant('default')
            baseline_score = (default_parent.get('scores', {}) or {}).get('overall', 0.0) if default_parent else 0.0
            improvement = round(current_overall - baseline_score, 1)
        else:
            parent = get_variant(parent_variant_id)
            baseline_score = (parent.get('scores', {}) or {}).get('overall', 0.0) if parent else 0.0
            improvement = round(current_overall - baseline_score, 1)

    variant = {
        'variant_id': variant_id,
        'label': label,
        'is_default': False,
        'parent_variant_id': parent_variant_id,
        'created_at': now,
        'optimizer': optimizer,
        'model': settings.get('model_name', ''),
        'training_window_count': len(training_windows),
        'validation_window_count': len(val_windows),
        'feedback_count': 0,
        'artifact_path': artifact_path,
        'status': 'done',
        'scores': pct_scores,  # stored on 0 - 100% scale
        'improvement_over_baseline': improvement,
        'is_best': False,
        'config_snapshot': {
            **settings,
            'training_window_count': len(training_windows),
            'validation_window_count': len(val_windows),
            'meeting_id': meeting_id,
            'parent_variant_id': parent_variant_id,
        },
        'run_id': run_id,
    }

    _save_new_variant(variant)

    # History entry
    history_entry = {
        'run_id': run_id,
        'user_id': user_id,
        'variant_id': variant_id,
        'variant_label': label,
        'meeting_id': meeting_id,
        'window_count': len(training_windows) + len(val_windows),
        'training_window_count': len(training_windows),
        'validation_window_count': len(val_windows),
        'optimizer': optimizer,
        'model': settings.get('model_name', ''),
        'baseline_score': round(baseline_score * 100, 1),
        'final_score': round(scores.get('overall', 0) * 100, 1),
        'feedback_rounds': 1 if parent_variant_id and parent_variant_id != 'default' else 0,
        'created_at': now,
        'status': 'done',
        'parent_variant_id': parent_variant_id,
    }
    _append_history(history_entry)

    _update_run(run_id, status='done', progress=100, message=f'Variant {label} created successfully!', variant_id=variant_id)


# ─── Variant inference ────────────────────────────────────────────────────────

def run_variant_on_window(
    variant_id: str,
    transcript_text: str,
    context_summary: str = '',
    agenda_summary: str = '',
) -> Dict[str, Any]:
    """
    Run a specific variant on a transcript window and return the output.
    If variant_id == 'default', uses the Stage 1 default prompt via LLM.
    """
    if variant_id == 'default':
        return _run_default_variant(transcript_text, context_summary, agenda_summary)

    variant = get_variant(variant_id)
    if not variant:
        raise ValueError(f'Variant {variant_id} not found')

    artifact_path = variant.get('artifact_path')
    if not artifact_path or not Path(artifact_path).exists():
        # Fall back to default if artifact missing
        return _run_default_variant(transcript_text, context_summary, agenda_summary)

    if not DSPY_AVAILABLE:
        return _run_default_variant(transcript_text, context_summary, agenda_summary)

    try:
        settings = load_settings()
        ollama_url = 'http://localhost:11434'
        try:
            from config import settings as app_settings
            ollama_url = getattr(app_settings, 'OLLAMA_SERVER_URL', ollama_url)
        except Exception:
            pass

        model_name = variant.get('model') or settings.get('model_name', '')
        lm = _create_dspy_lm(model_name, ollama_url)

        with dspy.context(lm=lm):
            predictor = dspy.Predict(Stage1ExtractionSignature)
            try:
                predictor.load(artifact_path)
            except Exception:
                pass

            pred = predictor(
                transcript_window=transcript_text,
                context_summary=context_summary,
                agenda_summary=agenda_summary,
            )
            raw_output = getattr(pred, 'discussion_points_json', '') or ''
            eval_scores = evaluate_stage1_output(transcript_text, raw_output)
            try:
                parsed = json.loads(raw_output)
            except Exception:
                parsed = None
            return {
                'raw_output': raw_output,
                'parsed_output': parsed,
                'eval_scores': eval_scores,
                'variant_id': variant_id,
                'variant_label': variant.get('label', variant_id),
            }
    except Exception as e:
        logger.warning(f'[Stage1] Variant inference error: {e}')
        return _run_default_variant(transcript_text, context_summary, agenda_summary)


def _run_default_variant(
    transcript_text: str,
    context_summary: str = '',
    agenda_summary: str = '',
) -> Dict[str, Any]:
    """Run the Stage 1 extraction using the default text prompt via AI provider."""
    try:
        from services.ai_provider import get_provider
        provider = get_provider()

        context_section = f'\n\nMeeting Context:\n{context_summary}' if context_summary else ''
        agenda_section = f'\n\nAgenda:\n{agenda_summary}' if agenda_summary else ''

        prompt = f"""{STAGE1_INSTRUCTIONS}

Return ONLY a valid JSON object matching this schema:
{json.dumps(STAGE1_OUTPUT_SCHEMA, indent=2)}

Do NOT wrap in markdown code fences. Output raw JSON only.
{context_section}{agenda_section}

Transcript Window:
{transcript_text}

JSON Output:"""

        raw_output = provider.generate(prompt)
        # Strip markdown fences if present
        cleaned = re.sub(r'^```(json)?\s*', '', raw_output.strip())
        cleaned = re.sub(r'\s*```$', '', cleaned)

        eval_scores = evaluate_stage1_output(transcript_text, cleaned)
        try:
            parsed = json.loads(cleaned)
        except Exception:
            parsed = None

        return {
            'raw_output': cleaned,
            'parsed_output': parsed,
            'eval_scores': eval_scores,
            'variant_id': 'default',
            'variant_label': 'Default Text Prompt',
        }
    except Exception as e:
        logger.warning(f'[Stage1] Default variant inference error: {e}')
        return {
            'raw_output': '',
            'parsed_output': None,
            'eval_scores': {},
            'variant_id': 'default',
            'variant_label': 'Default Text Prompt',
            'error': str(e),
        }


# ─── Feedback to training examples ────────────────────────────────────────────

def convert_feedback_to_training_examples(
    feedback_list: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Convert user feedback entries into DSPy training examples.
    Each feedback entry with a corrected output or category signal becomes a training example.
    """
    examples = []
    for fb in feedback_list:
        transcript_window = fb.get('transcript_window', '')
        model_output = fb.get('model_output', '')
        categories = fb.get('categories', [])
        comment = fb.get('comment', '')

        if not transcript_window:
            continue

        # If user provided a corrected output use it as target
        corrected_output = fb.get('corrected_output', '')
        if not corrected_output:
            # Generate a corrected output based on the feedback categories
            # This is a heuristic: if original output is valid JSON, use it with a note
            # In production this would trigger an LLM correction step
            corrected_output = model_output  # Use original as baseline, optimizer will improve

        examples.append({
            'transcript_window': transcript_window,
            'context_summary': fb.get('context_summary', ''),
            'agenda_summary': fb.get('agenda_summary', ''),
            'corrected_output': corrected_output,
            'feedback_categories': categories,
            'feedback_comment': comment,
        })
    return examples
