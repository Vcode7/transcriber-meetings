"""
Training Dataset Service — Read-only interface to existing pipeline data.
Builds training samples from stored meeting data WITHOUT modifying any existing tables or pipeline.
"""
import json
import uuid
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from pathlib import Path

from database import get_db_context, from_json
from sqlalchemy import text
from .training_models import TrainingStage, DatasetSample
from .training_storage_service import save_dataset

logger = logging.getLogger(__name__)


async def get_meeting_list(user_id: str) -> List[Dict[str, Any]]:
    """Return list of recordings with enough data for training, including Stage 2 edit indicators."""
    async with get_db_context() as db:
        r = await db.execute(
            text("""
                SELECT r.id, r.filename, r.duration, r.status, r.created_at, r.processed_at,
                       (r.transcript IS NOT NULL AND r.transcript != '[]') as has_transcript,
                       (r.rom_data IS NOT NULL) as has_rom_data,
                       (SELECT COUNT(*) FROM stage2_edit_history h WHERE h.recording_id = r.id AND h.user_id = :uid AND h.is_reverted = 0) as stage2_edit_count
                FROM recordings r
                WHERE r.user_id = :uid AND r.status = 'done'
                ORDER BY r.created_at DESC
                LIMIT 200
            """),
            {'uid': user_id},
        )
        rows = r.mappings().fetchall()

    meetings = []
    for row in rows:
        has_transcript = bool(row.get('has_transcript'))
        has_rom_data = bool(row.get('has_rom_data'))
        edit_count = row.get('stage2_edit_count') or 0
        stages_available = []
        if has_transcript:
            stages_available.append('stage_1')
        if has_transcript and has_rom_data:
            stages_available.append('stage_2')
            stages_available.append('stage_3')
        meetings.append({
            'id': row['id'],
            'filename': row['filename'],
            'duration': row['duration'],
            'created_at': row['created_at'],
            'processed_at': row.get('processed_at'),
            'stages_available': stages_available,
            'has_transcript': has_transcript,
            'has_rom_data': has_rom_data,
            'has_stage2_edits': edit_count > 0,
            'stage2_edit_count': edit_count,
        })
    return meetings


async def get_meeting_training_data(user_id: str, recording_id: str) -> Dict[str, Any]:
    """Read all relevant data for a meeting to build training samples."""
    async with get_db_context() as db:
        r = await db.execute(
            text('SELECT * FROM recordings WHERE id = :id AND user_id = :uid'),
            {'id': recording_id, 'uid': user_id},
        )
        rec = r.mappings().fetchone()
        if not rec:
            raise ValueError(f'Recording {recording_id} not found')
        rec = dict(rec)

        # Get MoM if exists
        r2 = await db.execute(
            text('SELECT * FROM minutes_of_meeting WHERE recording_id = :rid AND user_id = :uid LIMIT 1'),
            {'rid': recording_id, 'uid': user_id},
        )
        mom_row = r2.mappings().fetchone()
        mom = dict(mom_row) if mom_row else None

        # Get attachments
        r3 = await db.execute(
            text('SELECT * FROM recording_attachments WHERE recording_id = :rid'),
            {'rid': recording_id},
        )
        attachments = [dict(a) for a in r3.mappings().fetchall()]

        # Query stage 2 edit count
        r_edits = await db.execute(
            text('SELECT COUNT(*) FROM stage2_edit_history WHERE recording_id = :rid AND user_id = :uid AND is_reverted = 0'),
            {'rid': recording_id, 'uid': user_id},
        )
        edit_count = r_edits.scalar() or 0

    transcript = from_json(rec.get('transcript') or '[]', [])
    rom_data = from_json(rec.get('rom_data') or '{}', {})
    raw_mom = from_json(rec.get('raw_mom') or '{}', {})

    stage2_data = rom_data.get('stage2', {})
    edited_stage2_points = stage2_data.get('polished_points', [])
    # If edits occurred, stage2.polished_points contains the edited version
    original_stage2_points = stage2_data.get('original_polished_points') or rom_data.get('stage1', {}).get('discussion_points', [])

    return {
        'recording_id': recording_id,
        'filename': rec.get('filename', ''),
        'transcript': transcript,
        'rom_data': rom_data,
        'raw_mom': raw_mom,
        'mom': mom,
        'attachments': attachments,
        'agenda_summary': rec.get('agenda_summary'),
        'context_summary': rec.get('context_summary'),
        'reference_summary': rec.get('reference_summary'),
        'parsed_agenda_json': from_json(rec.get('parsed_agenda_json'), []),
        'has_stage2_edits': edit_count > 0 or bool(stage2_data.get('has_edits')),
        'stage2_edit_count': edit_count,
        'original_stage2_points': original_stage2_points,
        'edited_stage2_points': edited_stage2_points,
    }


def extract_raw_mom_points_with_llm(raw_mom_text: str, context_text: str = "") -> List[str]:
    """
    Extract raw MoM points into JSON using LLM.
    Strictly preserves original wording without paraphrasing, summaries, or intro headings.
    """
    if not raw_mom_text or not raw_mom_text.strip():
        return []

    try:
        from services.ai_provider import get_provider
        provider = get_provider()

        prompt = f"""You are a strict meeting document extraction assistant.
Your task is to extract all actual/raw Minutes of Meeting (MoM) discussion points from the input text below.

CRITICAL CONSTRAINTS:
1. You MUST return ONLY a valid JSON object with key "points" containing an array of strings:
   {{"points": ["Exact raw point 1 text", "Exact raw point 2 text", ...]}}
2. Do NOT generate or include:
   - Introduction sections
   - Agenda headings
   - Extra explanations
   - Summaries
   - Reworded content
3. Extract all actual/raw MoM points and PRESERVE THE ORIGINAL WORDING EXACTLY.
   Do NOT change, paraphrase, merge, shorten, expand, or reinterpret any point.
4. Do NOT wrap response in markdown code fences (```json). Output raw JSON only.

INPUT MANUAL MOM TEXT:
{raw_mom_text}

{"INPUT CONTEXT TEXT:" if context_text else ""}
{context_text if context_text else ""}

JSON Output:"""

        res_str = provider.generate(prompt)
        import re
        cleaned = re.sub(r'^```(json)?\s*', '', res_str.strip())
        cleaned = re.sub(r'\s*```$', '', cleaned)

        data = json.loads(cleaned)
        if isinstance(data, dict) and "points" in data and isinstance(data["points"], list):
            return [str(p).strip() for p in data["points"] if str(p).strip()]
        elif isinstance(data, list):
            return [str(p).strip() for p in data if str(p).strip()]
    except Exception as e:
        logger.warning(f"[TrainingDatasetService] LLM MoM point extraction failed: {e}. Falling back to text line parsing.")

    # Fallback to line splitting
    lines = [line.strip("- *• \t") for line in raw_mom_text.splitlines() if line.strip()]
    return [l for l in lines if len(l) > 3]


def build_stage1_samples(
    meeting_data: Dict[str, Any],
    manual_mom: Optional[str] = None,
) -> List[DatasetSample]:
    """
    Stage 1 training samples.
    Input: transcript window segments + available context
    Target: extracted discussion points (from existing rom_data or derived from manual MoM)
    """
    samples = []
    transcript = meeting_data.get('transcript', [])
    rom_data = meeting_data.get('rom_data', {})
    stage1_data = rom_data.get('stage1', {})
    discussion_points = stage1_data.get('discussion_points', [])
    recording_id = meeting_data.get('recording_id', '')

    if not transcript:
        return samples

    # Window the transcript into 2-minute chunks (matching Stage 1 defaults)
    window_seconds = 120.0
    windows = []
    current_window = []
    current_start = transcript[0].get('start', 0.0) if transcript else 0.0
    for seg in transcript:
        seg_start = seg.get('start', 0.0)
        if current_window and (seg_start - current_start) >= window_seconds:
            windows.append(current_window)
            current_window = [seg]
            current_start = seg_start
        else:
            current_window.append(seg)
    if current_window:
        windows.append(current_window)

    # Match each window to its discussion points output
    dp_by_window = {}
    for dp in discussion_points:
        wi = dp.get('window_index')
        if wi is not None:
            dp_by_window.setdefault(int(wi), []).append(dp)

    context_summary = meeting_data.get('context_summary', '') or ''
    agenda_summary = meeting_data.get('agenda_summary', '') or ''

    for i, window in enumerate(windows):
        # Build transcript text for this window
        transcript_text = '\n'.join(
            f"[{seg.get('speaker_label', 'Speaker')}, {seg.get('start', 0):.1f}s]: {seg.get('text', '')}"
            for seg in window
        )
        window_dps = dp_by_window.get(i, [])

        # Fallback if no window_index match was found
        if not window_dps and discussion_points:
            pts_per_win = max(1, len(discussion_points) // len(windows))
            window_dps = discussion_points[i * pts_per_win : (i + 1) * pts_per_win]
            if not window_dps and i == 0:
                window_dps = discussion_points

        if window_dps:
            target_json = json.dumps(window_dps, ensure_ascii=False)
        elif manual_mom:
            target_json = manual_mom
        else:
            target_json = json.dumps([{"discussion_point": transcript_text[:300]}], ensure_ascii=False)

        sample = DatasetSample(
            sample_id=str(uuid.uuid4()),
            stage=TrainingStage.STAGE_1,
            source_meeting_id=recording_id,
            inputs={
                'transcript_window': transcript_text,
                'context_summary': context_summary[:2000] if context_summary else '',
                'agenda_summary': agenda_summary[:1000] if agenda_summary else '',
                'window_index': i,
                'window_start': window[0].get('start', 0.0) if window else 0.0,
                'window_end': window[-1].get('end', 0.0) if window else 0.0,
            },
            target=target_json,
            metadata={'source_meeting_id': recording_id, 'window_index': i},
        )
        samples.append(sample)

    return samples


def build_stage2_samples(
    meeting_data: Dict[str, Any],
    manual_mom: Optional[str] = None,
    use_edited_stage2: bool = False,
) -> List[DatasetSample]:
    """
    Stage 2 training samples.
    Input: discussion points + context
    Target: enhanced discussion points (from stage2 output, edited stage2 version, or manual MoM)
    """
    samples = []
    rom_data = meeting_data.get('rom_data', {})
    recording_id = meeting_data.get('recording_id', '')

    stage1_dps = rom_data.get('stage1', {}).get('discussion_points', [])
    
    # Target selection: Use edited stage2 points if use_edited_stage2 is True, otherwise original
    stage2_enhanced = []
    if use_edited_stage2:
        stage2_enhanced = meeting_data.get('edited_stage2_points', []) or rom_data.get('stage2', {}).get('polished_points', [])
    if not stage2_enhanced:
        stage2_enhanced = rom_data.get('stage2', {}).get('enhanced_points', []) or rom_data.get('stage2', {}).get('polished_points', [])

    # If stage1_dps is empty but transcript exists, attempt to extract from transcript
    if not stage1_dps:
        transcript = meeting_data.get('transcript', [])
        if transcript:
            stage1_dps = [{"discussion_point": seg.get('text', ''), "window_index": 0} for seg in transcript[:10]]

    if not stage1_dps:
        return samples

    context_summary = meeting_data.get('context_summary', '') or ''
    reference_summary = meeting_data.get('reference_summary', '') or ''
    target_version_label = "edited_stage2" if (use_edited_stage2 and meeting_data.get('has_stage2_edits')) else "original_stage2"

    for i, raw_dp in enumerate(stage1_dps):
        enhanced_dp = stage2_enhanced[i] if i < len(stage2_enhanced) else None
        if enhanced_dp:
            target_str = json.dumps(enhanced_dp, ensure_ascii=False) if isinstance(enhanced_dp, (dict, list)) else str(enhanced_dp)
        elif manual_mom:
            target_str = manual_mom
        else:
            target_str = json.dumps(raw_dp, ensure_ascii=False)

        sample = DatasetSample(
            sample_id=str(uuid.uuid4()),
            stage=TrainingStage.STAGE_2,
            source_meeting_id=recording_id,
            inputs={
                'discussion_point': json.dumps(raw_dp, ensure_ascii=False),
                'context_summary': context_summary[:2000] if context_summary else '',
                'reference_summary': reference_summary[:2000] if reference_summary else '',
                'point_index': i,
            },
            target=target_str,
            metadata={
                'source_meeting_id': recording_id,
                'point_index': i,
                'target_version': target_version_label,
            },
        )
        samples.append(sample)

    return samples


def build_stage3_samples(
    meeting_data: Dict[str, Any],
    manual_mom: Optional[str] = None,
    use_edited_stage2: bool = False,
) -> List[DatasetSample]:
    """
    Stage 3 training samples.
    Input: enhanced discussion points + agenda + context
    Target: agenda-matched MoM sections
    """
    samples = []
    rom_data = meeting_data.get('rom_data', {})
    recording_id = meeting_data.get('recording_id', '')

    if use_edited_stage2:
        stage2_enhanced = meeting_data.get('edited_stage2_points', []) or rom_data.get('stage2', {}).get('polished_points', [])
    else:
        stage2_enhanced = rom_data.get('stage2', {}).get('enhanced_points', []) or rom_data.get('stage2', {}).get('polished_points', []) or rom_data.get('stage1', {}).get('discussion_points', [])

    stage3_agendas = rom_data.get('stage3', {}).get('agendas', [])
    agenda_summary = meeting_data.get('agenda_summary', '') or ''
    parsed_agenda = meeting_data.get('parsed_agenda_json', []) or []
    context_summary = meeting_data.get('context_summary', '') or ''

    if not stage3_agendas and parsed_agenda:
        stage3_agendas = parsed_agenda
    elif not stage3_agendas:
        stage3_agendas = [{'agenda_title': 'General Discussion', 'discussion_points': stage2_enhanced}]

    for i, agenda_item in enumerate(stage3_agendas):
        agenda_title = agenda_item.get('agenda_title') or agenda_item.get('title') or f'Agenda {i+1}'
        assigned_points = agenda_item.get('discussion_points') or agenda_item.get('points') or stage2_enhanced

        if assigned_points:
            target_str = json.dumps(assigned_points, ensure_ascii=False)
        elif manual_mom:
            target_str = manual_mom
        else:
            target_str = json.dumps([{"title": agenda_title}], ensure_ascii=False)

        sample = DatasetSample(
            sample_id=str(uuid.uuid4()),
            stage=TrainingStage.STAGE_3,
            source_meeting_id=recording_id,
            inputs={
                'enhanced_discussion_points': json.dumps(stage2_enhanced, ensure_ascii=False)[:4000],
                'agenda_summary': agenda_summary[:1000] if agenda_summary else '',
                'context_summary': context_summary[:1500] if context_summary else '',
                'agenda_item': agenda_title,
                'agenda_index': i,
            },
            target=target_str,
            metadata={
                'source_meeting_id': recording_id,
                'agenda_index': i,
                'agenda_title': agenda_title,
            },
        )
        samples.append(sample)

    return samples


async def build_dataset(
    user_id: str,
    stages: List[TrainingStage],
    source_type: str,
    meeting_id: Optional[str] = None,
    manual_mom: Optional[str] = None,
    use_edited_stage2: bool = False,
    transcript_text: Optional[str] = None,
    context_text: Optional[str] = None,
    agenda_text: Optional[str] = None,
) -> Dict[str, Any]:
    """Build and persist a training dataset."""
    dataset_id = str(uuid.uuid4())
    all_samples = []
    samples_by_stage = {}

    if source_type == 'meeting' and meeting_id:
        meeting_data = await get_meeting_training_data(user_id, meeting_id)
        for stage in stages:
            s_enum = TrainingStage(stage) if isinstance(stage, str) else stage
            if s_enum == TrainingStage.STAGE_1:
                s = build_stage1_samples(meeting_data, manual_mom)
            elif s_enum == TrainingStage.STAGE_2:
                s = build_stage2_samples(meeting_data, manual_mom, use_edited_stage2=use_edited_stage2)
            elif s_enum == TrainingStage.STAGE_3:
                s = build_stage3_samples(meeting_data, manual_mom, use_edited_stage2=use_edited_stage2)
            else:
                s = []
            samples_by_stage[s_enum.value] = len(s)
            all_samples.extend(s)
    elif source_type == 'upload':
        meeting_data = {
            'recording_id': 'upload',
            'filename': 'uploaded_data',
            'transcript': _parse_transcript_text(transcript_text or ''),
            'rom_data': {},
            'raw_mom': {},
            'mom': None,
            'attachments': [],
            'agenda_summary': agenda_text or '',
            'context_summary': context_text or '',
            'reference_summary': '',
            'parsed_agenda_json': [],
        }
        for stage in stages:
            s_enum = TrainingStage(stage) if isinstance(stage, str) else stage
            if s_enum == TrainingStage.STAGE_1:
                s = build_stage1_samples(meeting_data, manual_mom)
            elif s_enum == TrainingStage.STAGE_2:
                s = build_stage2_samples(meeting_data, manual_mom, use_edited_stage2=False)
            elif s_enum == TrainingStage.STAGE_3:
                s = build_stage3_samples(meeting_data, manual_mom, use_edited_stage2=False)
            else:
                s = []
            samples_by_stage[s_enum.value] = len(s)
            all_samples.extend(s)

    stage_values = [
        s.value if hasattr(s, 'value') else TrainingStage(s).value
        for s in stages
    ]

    dataset = {
        'dataset_id': dataset_id,
        'user_id': user_id,
        'stages': stage_values,
        'total_samples': len(all_samples),
        'samples_by_stage': samples_by_stage,
        'samples': [s.model_dump() for s in all_samples],
        'created_at': datetime.now(timezone.utc).isoformat(),
        'source_type': source_type,
        'meeting_id': meeting_id,
        'manual_mom': manual_mom,
        'use_edited_stage2': use_edited_stage2,
    }
    save_dataset(dataset_id, dataset)
    return dataset


def _parse_transcript_text(text: str) -> List[Dict]:
    """Convert plain text into pseudo-transcript segments."""
    if not text.strip():
        return []
    lines = text.strip().split('\n')
    segments = []
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        segments.append({
            'speaker_label': 'Speaker',
            'start': float(i * 10),
            'end': float(i * 10 + 9),
            'text': line,
        })
    return segments


async def parse_uploaded_file_content(file_content: bytes, filename: str) -> str:
    """Extract text from uploaded file using existing doc_extractor with OCR support."""
    import tempfile
    import os
    from services.doc_extractor import extract_text_from_file

    ext = os.path.splitext(filename.lower())[1] or '.txt'
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(file_content)
        tmp_path = tmp.name

    try:
        extracted = extract_text_from_file(tmp_path, filename)
        return extracted or ""
    except Exception as e:
        logger.warning(f"[TrainingDatasetService] Document extraction failed for '{filename}': {e}")
        try:
            return file_content.decode('utf-8', errors='replace')
        except Exception:
            return ''
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

