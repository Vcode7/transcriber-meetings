"""
Training & Optimization API Router.
All endpoints under /api/training/.
Completely isolated from existing transcription and MoM pipeline.
"""
from __future__ import annotations
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import List, Optional
from pydantic import BaseModel

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy import text

from database import get_db, get_db_context, from_json, to_json, dt_to_str
from routers.auth import get_current_user
from services.training.training_models import TrainingStage, TrainingMethod, JobStatus
from services.training.training_schemas import (
    BuildDatasetRequest,
    StartTrainingJobRequest,
    ValidateRequest,
    ValidateResponse,
    StageTrainingConfig,
)
from services.training.training_model_service import get_ollama_models, validate_model_for_method
from services.training.training_dataset_service import (
    get_meeting_list,
    get_meeting_training_data,
    build_dataset,
    parse_uploaded_file_content,
)
from services.training.training_service import (
    start_training_job,
    cancel_job,
    get_job,
    list_jobs,
    get_job_eval_results,
)
from services.training.training_storage_service import (
    list_artifacts,
    activate_artifact,
    get_active_artifact,
    list_datasets,
    load_dataset,
)
from services.training.training_schemas import ActivateRequest

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/training", tags=["training"])


def _validate_user(current_user) -> str:
    if isinstance(current_user, dict):
        uid = current_user.get('id') or current_user.get('user_id') or current_user.get('sub')
    elif hasattr(current_user, 'id'):
        uid = current_user.id
    else:
        uid = None
    if not uid or not isinstance(uid, str):
        raise HTTPException(status_code=401, detail='Invalid authentication')
    return uid.strip()


# ── Ollama Models ─────────────────────────────────────────────────────────────

@router.get('/models')
async def list_ollama_models(current_user: dict = Depends(get_current_user)):
    """Return the list of locally available Ollama models."""
    _validate_user(current_user)
    models = await get_ollama_models()
    return {'models': models, 'count': len(models)}


# ── Validation ────────────────────────────────────────────────────────────────

@router.post('/validate')
async def validate_config(
    req: ValidateRequest,
    current_user: dict = Depends(get_current_user),
):
    """Validate a model+method+stage combination before starting training."""
    _validate_user(current_user)
    result = validate_model_for_method(req.model_name, req.method, req.hf_model_path)
    return ValidateResponse(**result)


# ── Meetings (read-only from existing data) ───────────────────────────────────

@router.get('/meetings')
async def list_meetings_for_training(current_user: dict = Depends(get_current_user)):
    """List recordings available for training dataset creation."""
    user_id = _validate_user(current_user)
    meetings = await get_meeting_list(user_id)
    return {'meetings': meetings, 'count': len(meetings)}


@router.get('/meetings/{recording_id}/data')
async def get_meeting_data(
    recording_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Get training-relevant data for a specific meeting."""
    user_id = _validate_user(current_user)
    try:
        data = await get_meeting_training_data(user_id, recording_id)
        transcript = data.get('transcript', [])
        rom_data = data.get('rom_data', {})
        return {
            'recording_id': recording_id,
            'filename': data.get('filename', ''),
            'transcript_segments': len(transcript),
            'has_stage1': bool(rom_data.get('stage1', {}).get('discussion_points')),
            'has_stage2': bool(rom_data.get('stage2', {}).get('enhanced_points') or rom_data.get('stage2', {}).get('polished_points')),
            'has_stage3': bool(rom_data.get('stage3', {}).get('agendas')),
            'has_mom': bool(data.get('mom')),
            'has_agenda': bool(data.get('parsed_agenda_json')) or bool(rom_data.get('stage3', {}).get('agendas')) or bool(data.get('agenda_summary')),
            'stage1_points': len(rom_data.get('stage1', {}).get('discussion_points', [])),
            'stage2_points': len(rom_data.get('stage2', {}).get('polished_points', []) or rom_data.get('stage2', {}).get('enhanced_points', [])),
            'stage3_agendas': len(rom_data.get('stage3', {}).get('agendas', [])),
            'has_stage2_edits': data.get('has_stage2_edits', False),
            'stage2_edit_count': data.get('stage2_edit_count', 0),
            'original_stage2_points': data.get('original_stage2_points', []),
            'edited_stage2_points': data.get('edited_stage2_points', []),
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ── File & Point Extraction ───────────────────────────────────────────────────

@router.post('/extract-file')
async def extract_file_endpoint(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    """Extract plain text from uploaded PDF, DOC/DOCX, image (OCR), or text file."""
    _validate_user(current_user)
    content = await file.read()
    filename = file.filename or 'upload.txt'
    extracted_text = await parse_uploaded_file_content(content, filename)
    return {
        'filename': filename,
        'extracted_text': extracted_text,
        'char_count': len(extracted_text),
    }


class ExtractPointsRequest(BaseModel):
    raw_mom_text: str
    context_text: Optional[str] = None


@router.post('/extract-points')
async def extract_points_endpoint(
    req: ExtractPointsRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Pass manual MoM and context to LLM.
    Returns JSON containing only extracted points preserving original wording exactly.
    """
    _validate_user(current_user)
    from services.training.training_dataset_service import extract_raw_mom_points_with_llm
    points = extract_raw_mom_points_with_llm(req.raw_mom_text, req.context_text or "")
    return {
        'points': points,
        'count': len(points),
    }


# ── Datasets ──────────────────────────────────────────────────────────────────

@router.get('/datasets')
async def list_training_datasets(current_user: dict = Depends(get_current_user)):
    """List all built datasets for this user."""
    user_id = _validate_user(current_user)
    datasets = list_datasets(user_id)
    return {'datasets': [{
        'dataset_id': d.get('dataset_id'),
        'stages': d.get('stages', []),
        'total_samples': d.get('total_samples', 0),
        'samples_by_stage': d.get('samples_by_stage', {}),
        'created_at': d.get('created_at'),
        'source_type': d.get('source_type'),
        'meeting_id': d.get('meeting_id'),
        'use_edited_stage2': d.get('use_edited_stage2', False),
    } for d in datasets]}


@router.post('/datasets/build')
async def build_training_dataset(
    req: BuildDatasetRequest,
    current_user: dict = Depends(get_current_user),
):
    """Build a training dataset from an existing meeting or uploaded data."""
    user_id = _validate_user(current_user)
    try:
        dataset = await build_dataset(
            user_id=user_id,
            stages=req.stages,
            source_type=req.source_type,
            meeting_id=req.meeting_id,
            manual_mom=req.manual_mom,
            use_edited_stage2=req.use_edited_stage2,
            transcript_text=req.transcript_text,
            context_text=req.context_text,
            agenda_text=req.agenda_text,
        )
        return {
            'dataset_id': dataset['dataset_id'],
            'total_samples': dataset['total_samples'],
            'samples_by_stage': dataset['samples_by_stage'],
            'stages': dataset['stages'],
            'created_at': dataset['created_at'],
            'samples_preview': dataset['samples'][:9],
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post('/datasets/upload')
async def upload_training_files(
    stages: str = Form(...),  # JSON array of stage strings
    manual_mom: Optional[str] = Form(None),
    transcript_file: Optional[UploadFile] = File(None),
    context_file: Optional[UploadFile] = File(None),
    agenda_file: Optional[UploadFile] = File(None),
    mom_file: Optional[UploadFile] = File(None),
    current_user: dict = Depends(get_current_user),
):
    """Upload training files and build a dataset from them."""
    user_id = _validate_user(current_user)
    try:
        stage_list = [TrainingStage(s) for s in json.loads(stages)]
    except Exception:
        raise HTTPException(status_code=422, detail='Invalid stages format')

    transcript_text = None
    context_text = None
    agenda_text = None
    mom_text = manual_mom

    if transcript_file:
        content = await transcript_file.read()
        transcript_text = await parse_uploaded_file_content(content, transcript_file.filename or 'transcript.txt')

    if context_file:
        content = await context_file.read()
        context_text = await parse_uploaded_file_content(content, context_file.filename or 'context.txt')

    if agenda_file:
        content = await agenda_file.read()
        agenda_text = await parse_uploaded_file_content(content, agenda_file.filename or 'agenda.txt')

    if mom_file and not mom_text:
        content = await mom_file.read()
        mom_text = await parse_uploaded_file_content(content, mom_file.filename or 'mom.txt')

    try:
        dataset = await build_dataset(
            user_id=user_id,
            stages=stage_list,
            source_type='upload',
            manual_mom=mom_text,
            transcript_text=transcript_text,
            context_text=context_text,
            agenda_text=agenda_text,
        )
        return {
            'dataset_id': dataset['dataset_id'],
            'total_samples': dataset['total_samples'],
            'samples_by_stage': dataset['samples_by_stage'],
            'stages': dataset['stages'],
            'created_at': dataset['created_at'],
            'samples_preview': dataset['samples'][:9],
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get('/datasets/{dataset_id}')
async def get_dataset(dataset_id: str, current_user: dict = Depends(get_current_user)):
    """Get full dataset including all samples."""
    user_id = _validate_user(current_user)
    data = load_dataset(dataset_id)
    if not data:
        raise HTTPException(status_code=404, detail='Dataset not found')
    if data.get('user_id') != user_id:
        raise HTTPException(status_code=403, detail='Forbidden')
    return data


# ── Training Jobs ─────────────────────────────────────────────────────────────

@router.get('/jobs')
async def list_training_jobs(current_user: dict = Depends(get_current_user)):
    """List all training jobs for the current user."""
    user_id = _validate_user(current_user)
    jobs = list_jobs(user_id)
    return {'jobs': jobs, 'count': len(jobs)}


@router.post('/jobs')
async def create_training_job(
    req: StartTrainingJobRequest,
    current_user: dict = Depends(get_current_user),
):
    """Start a new training job."""
    user_id = _validate_user(current_user)
    if not req.stage_configs:
        raise HTTPException(status_code=422, detail='At least one stage config is required')

    # Pre-validate all configs
    for cfg in req.stage_configs:
        validation = validate_model_for_method(cfg.model_name, cfg.method, cfg.hf_model_path)
        if not validation['valid']:
            raise HTTPException(
                status_code=422,
                detail=f'{cfg.stage.value} {cfg.method.value} validation failed: {validation["reason"]}'
            )

    job_id = start_training_job(user_id, req)
    return {'job_id': job_id, 'status': 'pending', 'message': 'Training job started in background'}


@router.get('/jobs/{job_id}')
async def get_training_job(job_id: str, current_user: dict = Depends(get_current_user)):
    """Get the status and logs of a training job."""
    user_id = _validate_user(current_user)
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail='Job not found')
    if job.get('user_id') != user_id:
        raise HTTPException(status_code=403, detail='Forbidden')
    return job


@router.post('/jobs/{job_id}/cancel')
async def cancel_training_job(job_id: str, current_user: dict = Depends(get_current_user)):
    """Request cancellation of a running training job."""
    user_id = _validate_user(current_user)
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail='Job not found')
    if job.get('user_id') != user_id:
        raise HTTPException(status_code=403, detail='Forbidden')
    success = cancel_job(job_id)
    return {'job_id': job_id, 'cancelled': success}


@router.get('/jobs/{job_id}/results')
async def get_job_results(
    job_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Get evaluation results for a completed training job."""
    user_id = _validate_user(current_user)
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail='Job not found')
    if job.get('user_id') != user_id:
        raise HTTPException(status_code=403, detail='Forbidden')
    results = get_job_eval_results(job_id)
    return {'job_id': job_id, 'eval_results': results or {}}


@router.post('/jobs/{job_id}/activate')
async def activate_training_result(
    job_id: str,
    req: ActivateRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Activate a trained artifact for a specific stage.
    This marks the artifact as active in metadata ONLY.
    It does NOT replace or modify the existing active model/prompts used by the pipeline.
    The pipeline must be explicitly updated to use activated artifacts.
    """
    user_id = _validate_user(current_user)
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail='Job not found')
    if job.get('user_id') != user_id:
        raise HTTPException(status_code=403, detail='Forbidden')

    # Find the artifact from this job for the requested stage
    artifacts = job.get('artifacts', [])
    target_artifact = None
    for a in artifacts:
        if a.get('stage') == req.stage.value and a.get('artifact_id') == req.artifact_id:
            target_artifact = a
            break

    if not target_artifact:
        raise HTTPException(status_code=404, detail=f'Artifact {req.artifact_id} not found in job {job_id}')

    method = TrainingMethod(target_artifact.get('method'))
    success = activate_artifact(req.artifact_id, req.stage, method)
    if not success:
        raise HTTPException(status_code=404, detail='Artifact metadata not found on disk')

    return {
        'activated': True,
        'artifact_id': req.artifact_id,
        'stage': req.stage.value,
        'method': method.value,
        'note': 'Artifact marked as active. The existing pipeline continues to use its current configuration until manually updated.',
    }


# ── Artifacts ─────────────────────────────────────────────────────────────────

@router.get('/artifacts')
async def list_training_artifacts(
    stage: Optional[str] = None,
    method: Optional[str] = None,
    current_user: dict = Depends(get_current_user),
):
    """List all stored training artifacts."""
    _validate_user(current_user)
    stage_enum = TrainingStage(stage) if stage else None
    method_enum = TrainingMethod(method) if method else None
    artifacts = list_artifacts(stage_enum, method_enum)
    return {'artifacts': [a.model_dump() for a in artifacts], 'count': len(artifacts)}


# ── History ───────────────────────────────────────────────────────────────────

@router.get('/history')
async def get_training_history(current_user: dict = Depends(get_current_user)):
    """Get full training history for the current user (all jobs + artifacts)."""
    user_id = _validate_user(current_user)
    jobs = list_jobs(user_id)
    artifacts = list_artifacts()

    # Build enriched history
    history = []
    for job in jobs:
        entry = {
            'job_id': job.get('job_id'),
            'created_at': job.get('created_at'),
            'status': job.get('status'),
            'description': job.get('description'),
            'stages': [cfg.get('stage') for cfg in job.get('stage_configs', [])],
            'methods': [cfg.get('method') for cfg in job.get('stage_configs', [])],
            'models': list(set(cfg.get('model_name') for cfg in job.get('stage_configs', []))),
            'artifacts': job.get('artifacts', []),
            'eval_results': job.get('eval_results', {}),
            'error': job.get('error'),
        }
        history.append(entry)
    return {
        'history': history,
        'artifacts': [a.model_dump() for a in artifacts],
        'total_jobs': len(jobs),
        'total_artifacts': len(artifacts),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 1 Training Workflow (new endpoints, additive only)
# ═══════════════════════════════════════════════════════════════════════════════

from services.training.stage1_training_service import (
    split_transcript_into_windows,
    start_stage1_optimization,
    get_run_status,
    get_all_variants,
    get_variant,
    run_variant_on_window,
    save_feedback,
    convert_feedback_to_training_examples,
    get_history as get_stage1_history,
    load_settings as load_stage1_settings,
    save_settings as save_stage1_settings,
    evaluate_stage1_output,
)


class Stage1WindowRequest(BaseModel):
    meeting_id: str
    window_size_seconds: float = 120.0
    overlap_seconds: float = 30.0


class Stage1OptimizeRequest(BaseModel):
    meeting_id: str
    windows: list  # List of window dicts with selected/role flags
    settings: dict = {}
    parent_variant_id: Optional[str] = 'default'
    feedback_examples: Optional[list] = None


class Stage1ValidateRequest(BaseModel):
    variant_id: str
    meeting_id: str
    window_id: Optional[str] = None
    transcript_text: str
    context_summary: Optional[str] = ''
    agenda_summary: Optional[str] = ''


class Stage1FeedbackRequest(BaseModel):
    meeting_id: str
    window_id: str
    variant_id: str
    transcript_window: str
    model_output: str
    corrected_output: Optional[str] = None
    categories: list
    comment: Optional[str] = ''
    context_summary: Optional[str] = ''
    agenda_summary: Optional[str] = ''


class Stage1FeedbackRetrainRequest(BaseModel):
    feedback: dict
    parent_variant_id: str
    meeting_id: str
    windows: list
    settings: dict = {}


class Stage1SettingsRequest(BaseModel):
    settings: dict


@router.post('/stage1/windows')
async def stage1_prepare_windows(
    req: Stage1WindowRequest,
    current_user: dict = Depends(get_current_user),
):
    """Split a meeting transcript into configurable windows for Stage 1 training."""
    user_id = _validate_user(current_user)
    try:
        data = await get_meeting_training_data(user_id, req.meeting_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    transcript = data.get('transcript', [])
    if not transcript:
        raise HTTPException(status_code=422, detail='Meeting has no transcript data')

    context_summary = data.get('context_summary', '') or ''
    agenda_summary = data.get('agenda_summary', '') or ''

    windows = split_transcript_into_windows(
        transcript,
        window_size_seconds=req.window_size_seconds,
        overlap_seconds=req.overlap_seconds,
    )

    # Attach meeting context to each window for use during optimization
    for w in windows:
        w['context_summary'] = context_summary[:2000]
        w['agenda_summary'] = agenda_summary[:1000]
        w['meeting_id'] = req.meeting_id

    return {
        'meeting_id': req.meeting_id,
        'filename': data.get('filename', ''),
        'windows': windows,
        'total_windows': len(windows),
        'window_size_seconds': req.window_size_seconds,
        'overlap_seconds': req.overlap_seconds,
    }


@router.post('/stage1/optimize')
async def stage1_start_optimization(
    req: Stage1OptimizeRequest,
    current_user: dict = Depends(get_current_user),
):
    """Start a Stage 1 DSPy optimization run. Returns a run_id to poll."""
    user_id = _validate_user(current_user)
    if not req.windows:
        raise HTTPException(status_code=422, detail='No windows provided')

    run_id = str(uuid.uuid4())
    settings = req.settings or load_stage1_settings()

    import asyncio
    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        None,
        lambda: start_stage1_optimization(
            run_id=run_id,
            user_id=user_id,
            meeting_id=req.meeting_id,
            windows=req.windows,
            settings=settings,
            parent_variant_id=req.parent_variant_id,
            feedback_examples=req.feedback_examples,
        )
    )

    return {'run_id': run_id, 'status': 'starting'}


@router.get('/stage1/runs/{run_id}')
async def stage1_get_run_status(
    run_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Poll the status of a Stage 1 optimization run."""
    _validate_user(current_user)
    status = get_run_status(run_id)
    if status is None:
        raise HTTPException(status_code=404, detail='Run not found')
    return status


@router.post('/stage1/validate')
async def stage1_run_validation(
    req: Stage1ValidateRequest,
    current_user: dict = Depends(get_current_user),
):
    """Run a Stage 1 variant on a transcript window for interactive validation."""
    _validate_user(current_user)
    try:
        result = run_variant_on_window(
            variant_id=req.variant_id,
            transcript_text=req.transcript_text,
            context_summary=req.context_summary or '',
            agenda_summary=req.agenda_summary or '',
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/stage1/feedback')
async def stage1_submit_feedback(
    req: Stage1FeedbackRequest,
    current_user: dict = Depends(get_current_user),
):
    """Submit user feedback on a Stage 1 variant output."""
    user_id = _validate_user(current_user)
    feedback_id = save_feedback({
        'user_id': user_id,
        'meeting_id': req.meeting_id,
        'window_id': req.window_id,
        'variant_id': req.variant_id,
        'transcript_window': req.transcript_window,
        'model_output': req.model_output,
        'corrected_output': req.corrected_output,
        'categories': req.categories,
        'comment': req.comment,
        'context_summary': req.context_summary or '',
        'agenda_summary': req.agenda_summary or '',
    })
    return {'feedback_id': feedback_id, 'saved': True}


@router.post('/stage1/retrain')
async def stage1_retrain_from_feedback(
    req: Stage1FeedbackRetrainRequest,
    current_user: dict = Depends(get_current_user),
):
    """Submit feedback and immediately start a retraining run. Returns run_id."""
    user_id = _validate_user(current_user)

    # Save the feedback
    feedback_id = save_feedback({
        'user_id': user_id,
        **req.feedback,
    })

    # Convert feedback to training examples
    feedback_examples = convert_feedback_to_training_examples([req.feedback])

    settings = req.settings or load_stage1_settings()
    run_id = str(uuid.uuid4())

    import asyncio
    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        None,
        lambda: start_stage1_optimization(
            run_id=run_id,
            user_id=user_id,
            meeting_id=req.meeting_id,
            windows=req.windows,
            settings=settings,
            parent_variant_id=req.parent_variant_id,
            feedback_examples=feedback_examples,
        )
    )

    return {'run_id': run_id, 'feedback_id': feedback_id, 'status': 'starting'}


@router.get('/stage1/variants')
async def stage1_list_variants(current_user: dict = Depends(get_current_user)):
    """List all Stage 1 variants including the default."""
    _validate_user(current_user)
    return {'variants': get_all_variants(), 'count': len(get_all_variants())}


@router.get('/stage1/variants/{variant_id}')
async def stage1_get_variant(
    variant_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Get details for a specific Stage 1 variant."""
    _validate_user(current_user)
    v = get_variant(variant_id)
    if not v:
        raise HTTPException(status_code=404, detail=f'Variant {variant_id} not found')
    return v


@router.get('/stage1/history')
async def stage1_get_history(current_user: dict = Depends(get_current_user)):
    """Get Stage 1 training run history."""
    _validate_user(current_user)
    return {'history': get_stage1_history(), 'count': len(get_stage1_history())}


@router.get('/stage1/settings')
async def stage1_get_settings(current_user: dict = Depends(get_current_user)):
    """Load persisted Stage 1 training settings."""
    _validate_user(current_user)
    return load_stage1_settings()


@router.post('/stage1/settings')
async def stage1_save_settings(
    req: Stage1SettingsRequest,
    current_user: dict = Depends(get_current_user),
):
    """Persist Stage 1 training settings."""
    _validate_user(current_user)
    save_stage1_settings(req.settings)
    return {'saved': True, 'settings': req.settings}


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 2 Training Workflow (new endpoints, additive only)
# ═══════════════════════════════════════════════════════════════════════════════

from services.training.stage2_training_service import (
    start_stage2_optimization,
    get_run_status as get_stage2_run_status,
    get_all_variants as get_all_stage2_variants,
    get_variant as get_stage2_variant,
    get_active_variant as get_active_stage2_variant,
    set_active_variant as set_active_stage2_variant,
    validate_hf_model_path as validate_stage2_hf_model_path,
    run_variant_on_group,
    save_feedback as save_stage2_feedback,
    convert_feedback_to_training_examples as convert_stage2_feedback,
    get_history as get_stage2_history,
    load_settings as load_stage2_settings,
    save_settings as save_stage2_settings,
    get_stage1_points_for_meeting,
    get_reference_points,
    preview_groups,
    retrieve_context_for_meeting,
    group_stage1_points,
    load_example_points as load_stage2_example_points,
    save_example_points as save_stage2_example_points,
)


class Stage2GetStage1PointsResponse(BaseModel):
    meeting_id: str
    stage1_points: list
    stage1_point_count: int
    has_stage2: bool
    has_stage2_edits: bool
    reference_source: str
    reference_points: list


class Stage2PreviewGroupsRequest(BaseModel):
    meeting_id: str
    points_per_group: int = 5
    context_retrieval: bool = True


class Stage2OptimizeRequest(BaseModel):
    meeting_id: str
    groups: list          # preview group dicts (with points + context)
    reference_points: list
    settings: dict = {}
    parent_variant_id: Optional[str] = 'default'
    feedback_examples: Optional[list] = None
    example_points: Optional[List[str]] = None   # style-guide-only writing examples


class Stage2ValidateRequest(BaseModel):
    variant_id: str
    meeting_id: str
    group_index: Optional[int] = 0
    stage1_points: list
    global_context: Optional[str] = ''
    meeting_context: Optional[str] = ''
    reference_points: Optional[list] = None
    example_points: Optional[List[str]] = None   # style-guide-only writing examples


class Stage2FeedbackRequest(BaseModel):
    meeting_id: str
    group_index: int
    variant_id: str
    stage1_points_input: str      # JSON string of Stage 1 points
    model_output: str
    corrected_output: Optional[str] = None
    categories: list
    comment: Optional[str] = ''
    global_context: Optional[str] = ''
    meeting_context: Optional[str] = ''


class Stage2FeedbackRetrainRequest(BaseModel):
    feedback: dict
    parent_variant_id: str
    meeting_id: str
    groups: list
    reference_points: list
    settings: dict = {}


class Stage2SettingsRequest(BaseModel):
    settings: dict


class Stage2ValidateModelRequest(BaseModel):
    hf_model_path: str
    method: str = 'lora'


class Stage2ExamplePointsRequest(BaseModel):
    points: List[str]


@router.post('/stage2/validate-model')
async def stage2_validate_model(
    req: Stage2ValidateModelRequest,
    current_user: dict = Depends(get_current_user),
):
    """Validate a local Hugging Face model directory for Stage 2 LoRA/QLoRA training."""
    _validate_user(current_user)
    return validate_stage2_hf_model_path(req.hf_model_path, method=req.method)


@router.get('/stage2/example-points')
async def stage2_get_example_points(current_user: dict = Depends(get_current_user)):
    """Load persisted Stage 2 style-guide example points."""
    _validate_user(current_user)
    return {'points': load_stage2_example_points(), 'count': len(load_stage2_example_points())}


@router.post('/stage2/example-points')
async def stage2_save_example_points(
    req: Stage2ExamplePointsRequest,
    current_user: dict = Depends(get_current_user),
):
    """Persist Stage 2 style-guide example points.

    These points are ONLY used as writing style references.
    They are never treated as factual context or meeting information.
    """
    _validate_user(current_user)
    save_stage2_example_points(req.points)
    return {'saved': True, 'count': len(req.points)}


@router.get('/stage2/meetings/{meeting_id}/stage1-points')
async def stage2_get_stage1_points(
    meeting_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Get Stage 1 discussion points for a meeting, plus reference source info."""
    user_id = _validate_user(current_user)
    try:
        data = await get_meeting_training_data(user_id, meeting_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    stage1_points = get_stage1_points_for_meeting(data)
    reference_points, reference_source = get_reference_points(data)
    rom_data = data.get('rom_data', {})

    return {
        'meeting_id': meeting_id,
        'stage1_points': stage1_points,
        'stage1_point_count': len(stage1_points),
        'has_stage2': bool(rom_data.get('stage2', {}).get('polished_points') or
                           rom_data.get('stage2', {}).get('enhanced_points')),
        'has_stage2_edits': data.get('has_stage2_edits', False),
        'reference_source': reference_source,
        'reference_points': reference_points,
    }


@router.post('/stage2/preview-groups')
async def stage2_preview_groups(
    req: Stage2PreviewGroupsRequest,
    current_user: dict = Depends(get_current_user),
):
    """Group Stage 1 points and attach retrieved context — for preview before training."""
    user_id = _validate_user(current_user)
    try:
        data = await get_meeting_training_data(user_id, req.meeting_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    stage1_points = get_stage1_points_for_meeting(data)
    if not stage1_points:
        raise HTTPException(status_code=422, detail='Meeting has no Stage 1 discussion points')

    global_context, meeting_context = retrieve_context_for_meeting(data)
    groups = preview_groups(
        stage1_points,
        points_per_group=req.points_per_group,
        global_context=global_context,
        meeting_context=meeting_context,
        context_retrieval=req.context_retrieval,
    )

    return {
        'meeting_id': req.meeting_id,
        'total_points': len(stage1_points),
        'total_groups': len(groups),
        'points_per_group': req.points_per_group,
        'context_retrieval': req.context_retrieval,
        'groups': groups,
    }


@router.post('/stage2/extract-mom')
async def stage2_extract_mom(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    """Extract discussion points from an uploaded MoM document (PDF/DOCX/TXT/image)."""
    _validate_user(current_user)
    content = await file.read()
    filename = file.filename or 'mom.txt'
    try:
        extracted_text = await parse_uploaded_file_content(content, filename)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'File extraction failed: {e}')

    from services.training.training_dataset_service import extract_raw_mom_points_with_llm
    points = extract_raw_mom_points_with_llm(extracted_text, '')
    return {
        'filename': filename,
        'extracted_text': extracted_text[:500],   # preview only
        'points': points,
        'point_count': len(points),
    }


@router.post('/stage2/optimize')
async def stage2_start_optimization(
    req: Stage2OptimizeRequest,
    current_user: dict = Depends(get_current_user),
):
    """Start a Stage 2 DSPy optimization run. Returns a run_id to poll."""
    user_id = _validate_user(current_user)
    if not req.groups:
        raise HTTPException(status_code=422, detail='No groups provided')
    if not req.reference_points:
        raise HTTPException(status_code=422, detail='No reference points provided — upload a MoM or ensure Stage 2 output exists')

    run_id = str(uuid.uuid4())
    settings = req.settings or load_stage2_settings()

    import asyncio
    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        None,
        lambda: start_stage2_optimization(
            run_id=run_id,
            user_id=user_id,
            meeting_id=req.meeting_id,
            groups=req.groups,
            reference_points=req.reference_points,
            settings=settings,
            parent_variant_id=req.parent_variant_id or 'default',
            feedback_examples=req.feedback_examples,
            example_points=req.example_points or [],
        )
    )

    return {'run_id': run_id, 'status': 'starting'}


@router.get('/stage2/runs/{run_id}')
async def stage2_get_run_status(
    run_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Poll the status of a Stage 2 optimization run."""
    _validate_user(current_user)
    status = get_stage2_run_status(run_id)
    if status is None:
        raise HTTPException(status_code=404, detail='Run not found')
    return status


@router.post('/stage2/validate')
async def stage2_run_validation(
    req: Stage2ValidateRequest,
    current_user: dict = Depends(get_current_user),
):
    """Run a Stage 2 variant on a group of Stage 1 points for interactive validation."""
    _validate_user(current_user)
    try:
        result = run_variant_on_group(
            variant_id=req.variant_id,
            stage1_points=req.stage1_points,
            global_context=req.global_context or '',
            meeting_context=req.meeting_context or '',
            reference_points=req.reference_points or [],
            example_points=req.example_points or [],
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/stage2/feedback')
async def stage2_submit_feedback(
    req: Stage2FeedbackRequest,
    current_user: dict = Depends(get_current_user),
):
    """Submit user feedback on a Stage 2 variant output."""
    user_id = _validate_user(current_user)
    feedback_id = save_stage2_feedback({
        'user_id': user_id,
        'meeting_id': req.meeting_id,
        'group_index': req.group_index,
        'variant_id': req.variant_id,
        'stage1_points_input': req.stage1_points_input,
        'model_output': req.model_output,
        'corrected_output': req.corrected_output,
        'categories': req.categories,
        'comment': req.comment,
        'global_context': req.global_context or '',
        'meeting_context': req.meeting_context or '',
    })
    return {'feedback_id': feedback_id, 'saved': True}


@router.post('/stage2/retrain')
async def stage2_retrain_from_feedback(
    req: Stage2FeedbackRetrainRequest,
    current_user: dict = Depends(get_current_user),
):
    """Submit feedback and immediately start a Stage 2 retraining run. Returns run_id."""
    user_id = _validate_user(current_user)

    # Save the feedback
    feedback_id = save_stage2_feedback({'user_id': user_id, **req.feedback})

    # Convert feedback to training examples
    feedback_examples = convert_stage2_feedback([req.feedback])

    settings = req.settings or load_stage2_settings()
    run_id = str(uuid.uuid4())

    import asyncio
    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        None,
        lambda: start_stage2_optimization(
            run_id=run_id,
            user_id=user_id,
            meeting_id=req.meeting_id,
            groups=req.groups,
            reference_points=req.reference_points,
            settings=settings,
            parent_variant_id=req.parent_variant_id,
            feedback_examples=feedback_examples,
        )
    )

    return {'run_id': run_id, 'feedback_id': feedback_id, 'status': 'starting'}


@router.get('/stage2/variants')
async def stage2_list_variants(current_user: dict = Depends(get_current_user)):
    """List all Stage 2 variants including the default baseline."""
    _validate_user(current_user)
    variants = get_all_stage2_variants()
    return {'variants': variants, 'count': len(variants)}


@router.get('/stage2/variants/active')
async def stage2_get_active_variant(current_user: dict = Depends(get_current_user)):
    """Get the currently active Stage 2 variant."""
    _validate_user(current_user)
    active = get_active_stage2_variant()
    return {'active_variant': active}


@router.post('/stage2/variants/{variant_id}/activate')
async def stage2_activate_variant(
    variant_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Activate a Stage 2 variant (DSPy or LoRA/QLoRA) for ROM pipeline inference."""
    _validate_user(current_user)
    success = set_active_stage2_variant(variant_id)
    if not success:
        raise HTTPException(status_code=404, detail=f'Stage 2 variant {variant_id} not found')
    return {'activated': True, 'variant_id': variant_id}


@router.get('/stage2/variants/{variant_id}')
async def stage2_get_variant(
    variant_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Get details for a specific Stage 2 variant."""
    _validate_user(current_user)
    v = get_stage2_variant(variant_id)
    if not v:
        raise HTTPException(status_code=404, detail=f'Stage 2 variant {variant_id} not found')
    return v


@router.get('/stage2/history')
async def stage2_get_history(current_user: dict = Depends(get_current_user)):
    """Get Stage 2 training run history."""
    _validate_user(current_user)
    history = get_stage2_history()
    return {'history': history, 'count': len(history)}


@router.get('/stage2/settings')
async def stage2_get_settings(current_user: dict = Depends(get_current_user)):
    """Load persisted Stage 2 training settings."""
    _validate_user(current_user)
    return load_stage2_settings()


@router.post('/stage2/settings')
async def stage2_save_settings(
    req: Stage2SettingsRequest,
    current_user: dict = Depends(get_current_user),
):
    """Persist Stage 2 training settings."""
    _validate_user(current_user)
    save_stage2_settings(req.settings)
    return {'saved': True, 'settings': req.settings}


# ─── Stage 3: Point -> Agenda Assignment Training ──────────────────────────────

from services.training.stage3_training_service import (
    start_stage3_optimization,
    get_run_status as get_stage3_run_status,
    get_all_variants as get_all_stage3_variants,
    get_variant as get_stage3_variant,
    save_feedback as save_stage3_feedback,
    convert_feedback_to_training_examples as convert_stage3_feedback,
    get_history as get_stage3_history,
    load_settings as load_stage3_settings,
    save_settings as save_stage3_settings,
    prepare_stage3_batches,
    run_stage3_validation,
)


class Stage3PreviewBatchesRequest(BaseModel):
    meeting_id: str
    batch_size: int = 15


class Stage3OptimizeRequest(BaseModel):
    meeting_id: str
    batches: List[dict]
    agendas: List[dict]
    settings: Optional[dict] = None
    parent_variant_id: str = 'default'


class Stage3ValidateRequest(BaseModel):
    variant_id: str = 'default'
    meeting_id: str
    batch_index: int = 0
    batch_points: List[dict]
    agendas: List[dict]


class Stage3FeedbackRequest(BaseModel):
    meeting_id: str
    point_id: str
    point_text: str = ""
    original_agenda_id: str = ""
    correct_agenda_id: str
    reason: Optional[str] = "Manual user correction"


class Stage3RetrainRequest(BaseModel):
    feedback: dict
    parent_variant_id: str = 'default'
    meeting_id: str
    batches: List[dict]
    agendas: List[dict]
    settings: Optional[dict] = None


class Stage3SettingsRequest(BaseModel):
    settings: dict


@router.get('/stage3/meetings/{meeting_id}/data')
async def stage3_get_meeting_data(
    meeting_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Load Stage 2 discussion points and Agendas for Stage 3 training."""
    user_id = _validate_user(current_user)
    data = await get_meeting_training_data(user_id, meeting_id)
    rom_data = data.get('rom_data', {})

    # Extract Stage 2 points
    stage2_points = (
        rom_data.get('stage2', {}).get('polished_points', []) or
        rom_data.get('stage2', {}).get('enhanced_points', []) or
        []
    )

    # Extract Agendas from stage3 or parsed_agenda_json
    agendas = rom_data.get('stage3', {}).get('agendas', [])
    if not agendas:
        parsed_ag = data.get('parsed_agenda_json', [])
        if isinstance(parsed_ag, list) and parsed_ag:
            for idx, a in enumerate(parsed_ag, 1):
                if isinstance(a, dict):
                    agendas.append({
                        'agenda_id': a.get('agenda_id') or f"A{idx}",
                        'title': a.get('title') or a.get('item') or f"Agenda {idx}",
                        'description': a.get('description') or '',
                        'keywords': a.get('keywords', []),
                    })
                elif isinstance(a, str) and a.strip():
                    agendas.append({
                        'agenda_id': f"A{idx}",
                        'title': a.strip(),
                        'description': '',
                        'keywords': [],
                    })

    # Ground truth mappings if Stage 3 was previously run
    ground_truth = {}
    for a in rom_data.get('stage3', {}).get('final_rom', {}).get('agendas', []) or rom_data.get('stage3', {}).get('agendas', []):
        aid = a.get('agenda_id')
        for p in a.get('points', []):
            pid = p.get('id') if isinstance(p, dict) else None
            if pid and aid:
                ground_truth[pid] = aid

    return {
        'meeting_id': meeting_id,
        'filename': data.get('filename', ''),
        'stage2_points': stage2_points,
        'agendas': agendas,
        'has_stage2': bool(stage2_points),
        'has_agendas': bool(agendas),
        'ground_truth_mappings': ground_truth,
    }


class Stage3GenerateAgendasRequest(BaseModel):
    meeting_id: str
    agenda_text: str
    previous_mom_texts: Optional[List[str]] = None
    use_global_context: bool = True
    global_context_top_k: int = 5
    meeting_context_top_k: int = 5


@router.post('/stage3/extract-file')
async def stage3_extract_file(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    """Extract plain text from an uploaded Agenda or previous MoM file."""
    _validate_user(current_user)
    content = await file.read()
    filename = file.filename or 'document.txt'
    extracted_text = await parse_uploaded_file_content(content, filename)
    return {
        'filename': filename,
        'extracted_text': extracted_text,
        'char_count': len(extracted_text),
    }


@router.post('/stage3/generate-agendas')
async def stage3_generate_agendas(
    req: Stage3GenerateAgendasRequest,
    current_user: dict = Depends(get_current_user),
):
    """Generate and enrich agendas using global context and previous MoMs (Stage 3 Step 1)."""
    user_id = _validate_user(current_user)

    if not req.agenda_text.strip():
        raise HTTPException(status_code=400, detail='Agenda text cannot be empty.')

    # Retrieve transcript
    async with get_db_context() as db:
        r = await db.execute(
            text('SELECT transcript, rom_data FROM recordings WHERE id = :id AND user_id = :uid'),
            {'id': req.meeting_id, 'uid': user_id},
        )
        row = r.mappings().fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f'Meeting {req.meeting_id} not found')
        transcript = from_json(row.get('transcript') or '[]', [])
        rom_data = from_json(row.get('rom_data') or '{}', {})

    from services.rom_service import rom_service
    agendas_res = rom_service.generate_agendas_only(
        agenda_text=req.agenda_text,
        recording_id=req.meeting_id,
        user_id=user_id,
        meeting_context_top_k=req.meeting_context_top_k if req.use_global_context else 0,
        global_context_top_k=req.global_context_top_k if req.use_global_context else 0,
        transcript=transcript,
        previous_mom_texts=req.previous_mom_texts or [],
    )

    agendas = agendas_res.get('agendas', [])
    expanded = agendas_res.get('expanded_agendas', [])

    # Persist into ROM data & recordings
    async with get_db_context() as db:
        if 'stage3' not in rom_data:
            rom_data['stage3'] = {}
        rom_data['stage3']['agendas'] = agendas
        rom_data['stage3']['expanded_agendas'] = expanded
        rom_data['stage3']['status'] = 'agendas_ready'
        await db.execute(
            text('UPDATE recordings SET rom_data = :rom_data, agenda_summary = :agenda_summary WHERE id = :id AND user_id = :uid'),
            {
                'id': req.meeting_id,
                'uid': user_id,
                'rom_data': to_json(rom_data),
                'agenda_summary': req.agenda_text[:1000],
            },
        )

    return {
        'meeting_id': req.meeting_id,
        'agendas': agendas,
        'expanded_agendas': expanded,
        'count': len(agendas),
    }


@router.post('/stage3/upload-agenda')
async def stage3_upload_agenda(
    meeting_id: str = Form(...),
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    """Upload and extract an agenda file for a meeting that lacks agendas."""
    user_id = _validate_user(current_user)
    content = await file.read()
    filename = file.filename or 'agenda.txt'
    extracted_text = await parse_uploaded_file_content(content, filename)

    if not extracted_text.strip():
        raise HTTPException(status_code=400, detail='Failed to extract text from the agenda file.')

    from services.rom_service import rom_service
    agendas_res = rom_service.generate_agendas_only(
        agenda_text=extracted_text,
        user_id=user_id,
        recording_id=meeting_id,
    )

    agendas = agendas_res.get('agendas', [])

    # Persist into ROM data & recordings table
    async with get_db_context() as db:
        r = await db.execute(
            text('SELECT rom_data FROM recordings WHERE id = :id AND user_id = :uid'),
            {'id': meeting_id, 'uid': user_id},
        )
        row = r.mappings().fetchone()
        if row:
            rom_data = from_json(row.get('rom_data') or '{}', {})
            if 'stage3' not in rom_data:
                rom_data['stage3'] = {}
            rom_data['stage3']['agendas'] = agendas
            rom_data['stage3']['expanded_agendas'] = agendas_res.get('expanded_agendas', [])
            await db.execute(
                text('UPDATE recordings SET rom_data = :rom_data, agenda_summary = :agenda_summary WHERE id = :id AND user_id = :uid'),
                {
                    'id': meeting_id,
                    'uid': user_id,
                    'rom_data': to_json(rom_data),
                    'agenda_summary': extracted_text[:1000],
                },
            )

    return {
        'meeting_id': meeting_id,
        'agendas': agendas,
        'agenda_text': extracted_text,
        'count': len(agendas),
    }


@router.post('/stage3/preview-batches')
async def stage3_preview_batches(
    req: Stage3PreviewBatchesRequest,
    current_user: dict = Depends(get_current_user),
):
    """Compute cosine similarity and group Stage 2 points into batches of 10-20."""
    user_id = _validate_user(current_user)
    data = await get_meeting_training_data(user_id, req.meeting_id)
    rom_data = data.get('rom_data', {})

    stage2_points = (
        rom_data.get('stage2', {}).get('polished_points', []) or
        rom_data.get('stage2', {}).get('enhanced_points', []) or
        []
    )
    agendas = rom_data.get('stage3', {}).get('agendas', [])

    if not stage2_points:
        raise HTTPException(status_code=400, detail='No Stage 2 points found for this meeting.')
    if not agendas:
        raise HTTPException(status_code=400, detail='No Agendas found for this meeting. Please upload an agenda file.')

    batches = prepare_stage3_batches(stage2_points, agendas, batch_size=req.batch_size)
    return {
        'meeting_id': req.meeting_id,
        'batch_size': req.batch_size,
        'total_points': len(stage2_points),
        'total_batches': len(batches),
        'agendas': agendas,
        'batches': batches,
    }


@router.post('/stage3/optimize')
async def stage3_start_optimization(
    req: Stage3OptimizeRequest,
    current_user: dict = Depends(get_current_user),
):
    """Start Stage 3 DSPy optimization for Point -> Agenda assignment."""
    _validate_user(current_user)
    settings = req.settings or load_stage3_settings()

    res = start_stage3_optimization(
        meeting_id=req.meeting_id,
        batches=req.batches,
        agendas=req.agendas,
        settings=settings,
        parent_variant_id=req.parent_variant_id,
    )
    return res


@router.get('/stage3/runs/{run_id}')
async def stage3_get_run_status_endpoint(
    run_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Get active or historical run status for Stage 3 optimization."""
    _validate_user(current_user)
    return get_stage3_run_status(run_id)


@router.post('/stage3/validate')
async def stage3_validate(
    req: Stage3ValidateRequest,
    current_user: dict = Depends(get_current_user),
):
    """Run validation on a single batch of points using a Stage 3 variant."""
    _validate_user(current_user)
    res = run_stage3_validation(
        variant_id=req.variant_id,
        meeting_id=req.meeting_id,
        batch_index=req.batch_index,
        batch_points=req.batch_points,
        agendas=req.agendas,
    )
    return res


@router.post('/stage3/feedback')
async def stage3_submit_feedback(
    req: Stage3FeedbackRequest,
    current_user: dict = Depends(get_current_user),
):
    """Save user manual assignment correction / move as training feedback."""
    user_id = _validate_user(current_user)
    fb_id = save_stage3_feedback({
        'user_id': user_id,
        'meeting_id': req.meeting_id,
        'point_id': req.point_id,
        'point_text': req.point_text,
        'original_agenda_id': req.original_agenda_id,
        'correct_agenda_id': req.correct_agenda_id,
        'reason': req.reason,
    })
    return {'feedback_id': fb_id, 'saved': True}


@router.post('/stage3/retrain')
async def stage3_retrain(
    req: Stage3RetrainRequest,
    current_user: dict = Depends(get_current_user),
):
    """Save feedback and trigger immediate Stage 3 retraining."""
    user_id = _validate_user(current_user)
    feedback_id = save_stage3_feedback({'user_id': user_id, **req.feedback})
    settings = req.settings or load_stage3_settings()

    res = start_stage3_optimization(
        meeting_id=req.meeting_id,
        batches=req.batches,
        agendas=req.agendas,
        settings=settings,
        parent_variant_id=req.parent_variant_id,
    )
    return {'run_id': res['run_id'], 'feedback_id': feedback_id, 'status': 'started'}


@router.get('/stage3/variants')
async def stage3_list_variants(current_user: dict = Depends(get_current_user)):
    """List all Stage 3 variants including the default baseline."""
    _validate_user(current_user)
    variants = get_all_stage3_variants()
    return {'variants': variants, 'count': len(variants)}


@router.get('/stage3/variants/{variant_id}')
async def stage3_get_variant_endpoint(
    variant_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Get details for a specific Stage 3 variant."""
    _validate_user(current_user)
    v = get_stage3_variant(variant_id)
    if not v:
        raise HTTPException(status_code=404, detail=f'Stage 3 variant {variant_id} not found')
    return v


@router.get('/stage3/history')
async def stage3_get_history_endpoint(current_user: dict = Depends(get_current_user)):
    """Get Stage 3 training run history."""
    _validate_user(current_user)
    history = get_stage3_history()
    return {'history': history, 'count': len(history)}


@router.get('/stage3/settings')
async def stage3_get_settings_endpoint(current_user: dict = Depends(get_current_user)):
    """Load persisted Stage 3 training settings."""
    _validate_user(current_user)
    return load_stage3_settings()


@router.post('/stage3/settings')
async def stage3_save_settings_endpoint(
    req: Stage3SettingsRequest,
    current_user: dict = Depends(get_current_user),
):
    """Persist Stage 3 training settings."""
    _validate_user(current_user)
    save_stage3_settings(req.settings)
    return {'saved': True, 'settings': req.settings}
