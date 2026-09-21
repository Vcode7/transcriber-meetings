"""
Training Service — Central orchestrator for all training jobs.
Manages background threads, job state, cancellation.
All training runs in background threads to avoid blocking the FastAPI event loop
and the existing transcription/MoM pipeline.
"""
import json
import uuid
import logging
import threading
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone

from .training_models import TrainingStage, TrainingMethod, JobStatus, DatasetSample, TrainingJobProgress
from .training_schemas import StartTrainingJobRequest, StageTrainingConfig
from .training_storage_service import load_dataset

logger = logging.getLogger(__name__)

# ── In-memory job registry ───────────────────────────────────────────────────
# Persisted to DB; this is the runtime state.
_jobs: Dict[str, dict] = {}
_jobs_lock = threading.Lock()
_cancel_events: Dict[str, threading.Event] = {}


def _get_job(job_id: str) -> Optional[dict]:
    with _jobs_lock:
        return dict(_jobs.get(job_id, {})) if job_id in _jobs else None


def _update_job(job_id: str, **kwargs):
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(kwargs)


def _append_log(job_id: str, msg: str):
    ts = datetime.now(timezone.utc).strftime('%H:%M:%S')
    line = f'[{ts}] {msg}'
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].setdefault('logs', []).append(line)
            logger.info(f'[Training Job {job_id[:8]}] {msg}')


def start_training_job(
    user_id: str,
    request: StartTrainingJobRequest,
) -> str:
    """Start a new training job. Returns job_id."""
    job_id = str(uuid.uuid4())
    cancel_event = threading.Event()

    job = {
        'job_id': job_id,
        'user_id': user_id,
        'description': request.description or f'Training job {job_id[:8]}',
        'status': JobStatus.PENDING.value,
        'stage_configs': [c.model_dump() for c in request.stage_configs],
        'dataset_id': request.dataset_id,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'started_at': None,
        'completed_at': None,
        'error': None,
        'logs': [],
        'progress': None,
        'artifacts': [],
        'eval_results': {},
    }

    with _jobs_lock:
        _jobs[job_id] = job
        _cancel_events[job_id] = cancel_event

    # Start background thread
    thread = threading.Thread(
        target=_run_job_thread,
        args=(job_id, user_id, request, cancel_event),
        daemon=True,
        name=f'training-job-{job_id[:8]}',
    )
    thread.start()
    logger.info(f'[Training Service] Started job {job_id} in background thread')
    return job_id


def cancel_job(job_id: str) -> bool:
    """Signal a running job to stop. Returns True if the signal was sent."""
    event = _cancel_events.get(job_id)
    if not event:
        return False
    event.set()
    _update_job(job_id, status=JobStatus.CANCELLED.value)
    _append_log(job_id, 'Cancellation requested by user.')
    return True


def get_job(job_id: str) -> Optional[dict]:
    return _get_job(job_id)


def list_jobs(user_id: str) -> List[dict]:
    with _jobs_lock:
        jobs = [dict(j) for j in _jobs.values() if j.get('user_id') == user_id]
    return sorted(jobs, key=lambda x: x.get('created_at', ''), reverse=True)


def get_job_eval_results(job_id: str) -> Optional[dict]:
    job = _get_job(job_id)
    if not job:
        return None
    return job.get('eval_results', {})


# ── Background job thread ────────────────────────────────────────────────────

def _run_job_thread(
    job_id: str,
    user_id: str,
    request: StartTrainingJobRequest,
    cancel_event: threading.Event,
):
    """Runs in a background thread. Orchestrates all stage training configs."""
    _update_job(job_id,
        status=JobStatus.BUILDING_DATASET.value,
        started_at=datetime.now(timezone.utc).isoformat()
    )
    _append_log(job_id, f'Job started. Loading dataset {request.dataset_id}...')

    try:
        # Load dataset
        dataset = load_dataset(request.dataset_id)
        if not dataset:
            raise ValueError(f'Dataset {request.dataset_id} not found on disk')

        all_samples_raw = dataset.get('samples', [])
        samples_by_stage: Dict[str, List[DatasetSample]] = {}
        for s_raw in all_samples_raw:
            try:
                s = DatasetSample(**s_raw)
                stage_key = s.stage.value
                if stage_key not in samples_by_stage:
                    samples_by_stage[stage_key] = []
                samples_by_stage[stage_key].append(s)
            except Exception as e:
                logger.warning(f'[Training Job {job_id[:8]}] Could not parse sample: {e}')

        _append_log(job_id, f'Dataset loaded: {len(all_samples_raw)} total samples across {len(samples_by_stage)} stages.')

        artifacts = []
        eval_results = {}

        for cfg_dict in request.stage_configs:
            if cancel_event.is_set():
                _append_log(job_id, 'Cancelled before starting next stage.')
                break

            if isinstance(cfg_dict, StageTrainingConfig):
                cfg = cfg_dict
            elif isinstance(cfg_dict, dict):
                cfg = StageTrainingConfig(**cfg_dict)
            else:
                cfg = StageTrainingConfig.model_validate(cfg_dict)

            stage = cfg.stage
            method = cfg.method
            stage_samples = samples_by_stage.get(stage.value, [])

            _append_log(job_id, f'Starting {method.value} training for {stage.value} with {len(stage_samples)} samples...')
            _update_job(job_id, status=JobStatus.TRAINING.value)

            def make_progress_callback(jid: str):
                def cb(p: TrainingJobProgress):
                    _update_job(jid, progress=p.model_dump())
                    _append_log(jid, p.message)
                return cb

            progress_cb = make_progress_callback(job_id)

            try:
                if method == TrainingMethod.DSPY:
                    from .dspy_training_service import run_dspy_optimization
                    artifact = run_dspy_optimization(stage, cfg, stage_samples, progress_cb, cancel_event)
                elif method == TrainingMethod.LORA:
                    from .lora_training_service import run_lora_training
                    artifact = run_lora_training(stage, cfg, stage_samples, progress_cb, cancel_event)
                elif method == TrainingMethod.QLORA:
                    from .qlora_training_service import run_qlora_training
                    artifact = run_qlora_training(stage, cfg, stage_samples, progress_cb, cancel_event)
                else:
                    raise ValueError(f'Unknown training method: {method}')

                artifacts.append(artifact)
                _append_log(job_id, f'{stage.value} {method.value} training complete. Artifact: {artifact.get("artifact_id", "?")}'
                )

                # Run evaluation
                _update_job(job_id, status=JobStatus.EVALUATING.value)
                _append_log(job_id, f'Evaluating {stage.value}...')
                try:
                    from .training_evaluation_service import build_evaluation_response
                    eval_resp = build_evaluation_response(
                        job_id=job_id,
                        stage=stage,
                        method=method,
                        artifact_id=artifact.get('artifact_id', ''),
                        samples=stage_samples[:20],
                    )
                    eval_results[stage.value] = eval_resp
                    _append_log(
                        job_id,
                        f'{stage.value} score: {eval_resp.get("score_before", 0)}% -> {eval_resp.get("score_after", 0)}%'
                    )
                except Exception as eval_err:
                    logger.warning(f'[Training Job] Evaluation error: {eval_err}')
                    eval_results[stage.value] = {'error': str(eval_err)}

            except InterruptedError:
                _append_log(job_id, f'{stage.value} training interrupted by cancellation.')
                break
            except Exception as stage_err:
                _append_log(job_id, f'ERROR in {stage.value}: {stage_err}')
                _update_job(job_id, error=str(stage_err), status=JobStatus.ERROR.value)
                raise

        if not cancel_event.is_set():
            _update_job(
                job_id,
                status=JobStatus.DONE.value,
                completed_at=datetime.now(timezone.utc).isoformat(),
                artifacts=artifacts,
                eval_results=eval_results,
            )
            _append_log(job_id, f'All stages complete. {len(artifacts)} artifact(s) saved.')
        else:
            _update_job(
                job_id,
                status=JobStatus.CANCELLED.value,
                completed_at=datetime.now(timezone.utc).isoformat(),
                artifacts=artifacts,
                eval_results=eval_results,
            )
            _append_log(job_id, 'Job was cancelled.')

    except Exception as e:
        logger.error(f'[Training Job {job_id[:8]}] Fatal error: {e}', exc_info=True)
        _update_job(
            job_id,
            status=JobStatus.ERROR.value,
            error=str(e),
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
        _append_log(job_id, f'FATAL ERROR: {e}')
    finally:
        with _jobs_lock:
            _cancel_events.pop(job_id, None)
