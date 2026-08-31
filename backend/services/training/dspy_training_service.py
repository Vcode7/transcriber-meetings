"""
DSPy Prompt Optimization Service.
Runs DSPy BootstrapFewShot or MIPROv2 to optimize stage-specific prompts.
DSPy is optional — if not installed, this service returns a clear error.
"""
import json
import logging
import uuid
import os
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime, timezone
from pathlib import Path

from .training_models import TrainingStage, TrainingMethod, DatasetSample, TrainingJobProgress, TrainingArtifactMetadata
from .training_storage_service import create_artifact_dir, save_artifact_metadata, get_stage_method_dir
from .training_schemas import StageTrainingConfig

logger = logging.getLogger(__name__)

DSPY_AVAILABLE = False
try:
    import dspy
    DSPY_AVAILABLE = True
    logger.info('[DSPy Training] dspy-ai library available.')
except ImportError:
    logger.warning('[DSPy Training] dspy-ai not installed. DSPy optimization disabled. Install with: pip install dspy-ai')


# ── DSPy Signatures (one per stage) ──────────────────────────────────────────

if DSPY_AVAILABLE:
    class Stage1Signature(dspy.Signature):
        """Extract structured discussion points from a meeting transcript window.
        
        Given a transcript window and available context, identify all distinct discussion
        points with their timeline, speakers, key terms, dates, numbers, and action items.
        Return a JSON array of discussion point objects.
        """
        transcript_window: str = dspy.InputField(desc='The transcript text for this time window with speaker labels')
        context_summary: str = dspy.InputField(desc='Meeting context summary (may be empty)')
        agenda_summary: str = dspy.InputField(desc='Agenda summary (may be empty)')
        discussion_points_json: str = dspy.OutputField(desc='JSON array of extracted discussion point objects')

    class Stage2Signature(dspy.Signature):
        """Enhance a discussion point using context and reference documents.
        
        Given a raw discussion point and retrieved context, enrich the point with
        additional details, better terminology, and more complete information.
        Return the enhanced discussion point as JSON.
        """
        discussion_point: str = dspy.InputField(desc='Raw discussion point JSON')
        context_summary: str = dspy.InputField(desc='Meeting context summary')
        reference_summary: str = dspy.InputField(desc='Reference document summary')
        enhanced_point_json: str = dspy.OutputField(desc='Enhanced discussion point JSON')

    class Stage3Signature(dspy.Signature):
        """Assign discussion points to agenda items and build agenda-wise MoM sections.
        
        Given enhanced discussion points and an agenda item, assign the most relevant
        discussion points and generate the agenda-wise record of meeting content.
        Return a JSON array of assigned discussion points.
        """
        enhanced_discussion_points: str = dspy.InputField(desc='JSON array of enhanced discussion points')
        agenda_item: str = dspy.InputField(desc='Current agenda item title')
        agenda_summary: str = dspy.InputField(desc='Full agenda context')
        context_summary: str = dspy.InputField(desc='Meeting context summary')
        assigned_points_json: str = dspy.OutputField(desc='JSON array of discussion points assigned to this agenda item')


def _get_signature_for_stage(stage: TrainingStage):
    if not DSPY_AVAILABLE:
        return None
    return {TrainingStage.STAGE_1: Stage1Signature, TrainingStage.STAGE_2: Stage2Signature, TrainingStage.STAGE_3: Stage3Signature}.get(stage)


def _sample_to_dspy_example(sample: DatasetSample, stage: TrainingStage):
    """Convert a DatasetSample to a dspy.Example."""
    if not DSPY_AVAILABLE:
        return None
    inputs = sample.inputs
    if stage == TrainingStage.STAGE_1:
        return dspy.Example(
            transcript_window=inputs.get('transcript_window', ''),
            context_summary=inputs.get('context_summary', ''),
            agenda_summary=inputs.get('agenda_summary', ''),
            discussion_points_json=sample.target,
        ).with_inputs('transcript_window', 'context_summary', 'agenda_summary')
    elif stage == TrainingStage.STAGE_2:
        return dspy.Example(
            discussion_point=inputs.get('discussion_point', ''),
            context_summary=inputs.get('context_summary', ''),
            reference_summary=inputs.get('reference_summary', ''),
            enhanced_point_json=sample.target,
        ).with_inputs('discussion_point', 'context_summary', 'reference_summary')
    elif stage == TrainingStage.STAGE_3:
        return dspy.Example(
            enhanced_discussion_points=inputs.get('enhanced_discussion_points', ''),
            agenda_item=inputs.get('agenda_item', ''),
            agenda_summary=inputs.get('agenda_summary', ''),
            context_summary=inputs.get('context_summary', ''),
            assigned_points_json=sample.target,
        ).with_inputs('enhanced_discussion_points', 'agenda_item', 'agenda_summary', 'context_summary')
    return None


def _create_dspy_lm(model_name: str, ollama_url: str):
    """
    Creates a per-job DSPy Language Model connection without calling global dspy.settings.configure().
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
            lm = dspy.LM(model_str, api_base=ollama_url, max_tokens=1024, temperature=0.0)
            logger.info(f'[DSPy] Created per-job LM: dspy.LM("{model_str}", api_base="{ollama_url}")')
            return lm
        except Exception as e:
            errors.append(f'dspy.LM("{model_str}"): {e}')

    # 2. Fallback: try "ollama/<model_name>" format with dspy.LM
    if hasattr(dspy, 'LM'):
        try:
            raw_model = model_name.replace('ollama_chat/', '').replace('ollama/', '')
            lm = dspy.LM(f"ollama/{raw_model}", api_base=ollama_url, max_tokens=1024, temperature=0.0)
            logger.info(f'[DSPy] Fallback created per-job LM: dspy.LM("ollama/{raw_model}", api_base="{ollama_url}")')
            return lm
        except Exception as e:
            errors.append(f'dspy.LM("ollama/{raw_model}"): {e}')

    # 3. Fallback: dspy.Ollama provider
    if hasattr(dspy, 'Ollama'):
        try:
            raw_model = model_name.replace('ollama_chat/', '').replace('ollama/', '')
            lm = dspy.Ollama(model=raw_model, base_url=ollama_url, max_tokens=1024, temperature=0.0)
            logger.info(f'[DSPy] Fallback created per-job LM via dspy.Ollama(model="{raw_model}", base_url="{ollama_url}")')
            return lm
        except Exception as e:
            errors.append(f'dspy.Ollama: {e}')

    # 4. Fallback: dspy.OllamaLocal provider
    if hasattr(dspy, 'OllamaLocal'):
        try:
            raw_model = model_name.replace('ollama_chat/', '').replace('ollama/', '')
            lm = dspy.OllamaLocal(model=raw_model, base_url=ollama_url, max_tokens=1024, temperature=0.0)
            logger.info(f'[DSPy] Fallback created per-job LM via dspy.OllamaLocal(model="{raw_model}", base_url="{ollama_url}")')
            return lm
        except Exception as e:
            errors.append(f'dspy.OllamaLocal: {e}')

    raise RuntimeError(
        f"Could not configure DSPy LM for Ollama model '{model_name}'. "
        f"Attempted providers: {'; '.join(errors) if errors else 'No compatible provider found in dspy module'}"
    )


def run_dspy_optimization(
    stage: TrainingStage,
    config: StageTrainingConfig,
    samples: List[DatasetSample],
    progress_callback: Optional[Callable[[TrainingJobProgress], None]] = None,
    cancel_event=None,
) -> Dict[str, Any]:
    """
    Run DSPy optimization for a single stage.
    Returns artifact metadata dict.
    Runs synchronously (call from a background thread).
    """
    if not DSPY_AVAILABLE:
        raise RuntimeError(
            'dspy-ai is not installed. Install it with: pip install dspy-ai\n'
            'Then restart the backend server.'
        )

    if not samples:
        raise ValueError(f'No training samples available for {stage.value}')

    def _progress(msg: str, pct: float, step: int = 0, total: int = 0):
        if progress_callback:
            progress_callback(TrainingJobProgress(step=step, total_steps=total, percent=pct, message=msg))
        logger.info(f'[DSPy {stage.value}] {msg}')

    # Explicit CUDA / GPU inspection log
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            _progress(f'[CUDA Check] PyTorch CUDA GPU Active: {gpu_name} ({vram:.2f} GB VRAM) | Ollama GPU acceleration enabled.', 3.0)
        else:
            _progress('[CUDA Check] PyTorch CUDA not detected locally, using Ollama server GPU acceleration.', 3.0)
    except Exception as e:
        _progress(f'[CUDA Check] GPU inspection: {e}', 3.0)

    _progress(f'Configuring DSPy Ollama connection to model {config.model_name}...', 5.0)

    ollama_url = 'http://localhost:11434'
    try:
        from config import settings
        ollama_url = getattr(settings, 'OLLAMA_SERVER_URL', ollama_url)
    except Exception:
        pass

    try:
        lm = _create_dspy_lm(config.model_name, ollama_url)
    except Exception as e:
        raise RuntimeError(f'Failed to configure DSPy with Ollama model {config.model_name}: {e}')

    _progress('Preparing training examples...', 10.0)

    examples = []
    for s in samples:
        ex = _sample_to_dspy_example(s, stage)
        if ex is not None:
            examples.append(ex)

    if not examples:
        raise ValueError(f'Could not build any DSPy examples for stage {stage.value}')

    # Split train/eval
    split = max(1, int(len(examples) * (1 - config.eval_split)))
    train_examples = examples[:split]
    eval_examples = examples[split:] or examples[:1]

    _progress(f'Built {len(train_examples)} train + {len(eval_examples)} eval examples.', 15.0)

    if cancel_event and cancel_event.is_set():
        raise InterruptedError('Training cancelled by user')

    SignatureClass = _get_signature_for_stage(stage)
    if SignatureClass is None:
        raise ValueError(f'No DSPy signature defined for stage {stage.value}')

    with dspy.context(lm=lm):
        predictor = dspy.Predict(SignatureClass)

        # Simple JSON-match metric
        def json_match_metric(example, pred, trace=None):
            try:
                output_field = list(SignatureClass.output_fields.keys())[0]
                pred_text = getattr(pred, output_field, '') or ''
                expected = example.get(output_field, '') or ''
                # Normalize: check if predicted JSON can be parsed
                json.loads(pred_text)
                return 1.0 if pred_text.strip() and expected.strip() else 0.0
            except Exception:
                return 0.0

        _progress(f'Running {config.dspy_optimizer} optimizer...', 20.0)

        compiled_program = None
        try:
            if config.dspy_optimizer == 'MIPROv2' and hasattr(dspy, 'MIPROv2'):
                optimizer = dspy.MIPROv2(
                    metric=json_match_metric,
                    auto='light',
                    verbose=False,
                    num_threads=1,
                )
                compiled_program = optimizer.compile(
                    predictor,
                    trainset=train_examples,
                    max_bootstrapped_demos=config.dspy_max_bootstrapped_demos,
                    max_labeled_demos=config.dspy_max_labeled_demos,
                    num_trials=config.dspy_num_trials,
                )
            else:
                # BootstrapFewShot (default, works without API calls)
                optimizer = dspy.BootstrapFewShot(
                    metric=json_match_metric,
                    max_bootstrapped_demos=config.dspy_max_bootstrapped_demos,
                    max_labeled_demos=config.dspy_max_labeled_demos,
                )
                compiled_program = optimizer.compile(predictor, trainset=train_examples)
        except Exception as e:
            raise RuntimeError(f'DSPy optimization failed: {e}')

        if cancel_event and cancel_event.is_set():
            raise InterruptedError('Training cancelled by user')

        _progress('Optimization complete. Evaluating on held-out samples...', 80.0)

        # Evaluate
        score = 0.0
        if eval_examples:
            try:
                correct = 0
                output_field = list(SignatureClass.output_fields.keys())[0]
                for ex in eval_examples:
                    try:
                        pred = compiled_program(**{k: ex.get(k, '') for k in SignatureClass.input_fields})
                        pred_text = getattr(pred, output_field, '')
                        json.loads(pred_text)
                        correct += 1
                    except Exception:
                        pass
                score = correct / len(eval_examples)
            except Exception as e:
                logger.warning(f'[DSPy Training] Evaluation error: {e}')

    _progress('Saving optimized program to disk...', 90.0)

    # Save artifact
    artifact_id, artifact_dir = create_artifact_dir(stage, TrainingMethod.DSPY)
    program_path = artifact_dir / 'optimized_program.json'

    # Save the compiled DSPy program state
    try:
        compiled_program.save(str(program_path))
    except Exception:
        # Fallback: save as JSON manually
        program_data = {}
        try:
            for pred_name, pred in compiled_program.named_predictors():
                demos = getattr(pred, 'demos', [])
                program_data[pred_name] = {
                    'instructions': getattr(pred, 'extended_signature', {}).get('instructions', ''),
                    'demos': [d.toDict() if hasattr(d, 'toDict') else {} for d in demos],
                }
        except Exception:
            pass
        with open(program_path, 'w', encoding='utf-8') as f:
            json.dump(program_data, f, indent=2, default=str)

    metadata = TrainingArtifactMetadata(
        artifact_id=artifact_id,
        stage=stage,
        method=TrainingMethod.DSPY,
        base_model=config.model_name,
        dataset_ids=[],
        training_date=datetime.now(timezone.utc).isoformat(),
        config=config.model_dump(),
        status='done',
        eval_metrics={
            'json_parse_success_rate': round(score, 4),
            'eval_samples': len(eval_examples),
            'train_samples': len(train_examples),
        },
        artifact_path=str(program_path),
        version=1,
        is_active=False,
    )
    save_artifact_metadata(metadata)

    _progress('DSPy optimization complete!', 100.0)
    return metadata.model_dump()
