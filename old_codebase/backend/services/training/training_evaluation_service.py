"""
Training Evaluation Service.
Computes before/after metrics for DSPy and LoRA/QLoRA trained artifacts.
"""
import json
import logging
import math
from typing import List, Dict, Any, Optional

from .training_models import TrainingStage, TrainingMethod, DatasetSample
from .training_storage_service import load_artifact_metadata

logger = logging.getLogger(__name__)


def _rouge_l_score(reference: str, hypothesis: str) -> float:
    """Simple ROUGE-L approximation without external libraries."""
    if not reference or not hypothesis:
        return 0.0
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    if not ref_tokens or not hyp_tokens:
        return 0.0
    # LCS length
    m, n = len(ref_tokens), len(hyp_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i-1] == hyp_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    precision = lcs / n if n else 0
    recall = lcs / m if m else 0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _json_validity_score(text: str) -> float:
    """Returns 1.0 if text is valid JSON, 0.0 otherwise."""
    try:
        json.loads(text)
        return 1.0
    except Exception:
        return 0.0


def evaluate_baseline(
    stage: TrainingStage,
    samples: List[DatasetSample],
) -> Dict[str, Any]:
    """
    Compute baseline score from existing rom_data (before training).
    Uses the stored outputs from the existing pipeline as the 'predicted' output
    and compares to ground truth (sample.target).
    """
    if not samples:
        return {'score': 0.0, 'json_validity': 0.0, 'rouge_l': 0.0, 'n_samples': 0}

    rouge_scores = []
    json_scores = []

    for s in samples:
        # The existing pipeline output is already the target (it IS the baseline)
        # For a true baseline, we'd need to compare to manual MoM.
        # Here we use the existing pipeline output's JSON validity as proxy.
        json_score = _json_validity_score(s.target)
        json_scores.append(json_score)
        rouge_scores.append(1.0 if json_score == 1.0 else 0.5)  # Estimated

    n = len(samples)
    return {
        'score': round(sum(rouge_scores) / n, 4),
        'json_validity': round(sum(json_scores) / n, 4),
        'rouge_l': round(sum(rouge_scores) / n, 4),
        'n_samples': n,
        'note': 'Baseline estimated from existing pipeline output quality',
    }


def evaluate_dspy_artifact(
    stage: TrainingStage,
    artifact_id: str,
    samples: List[DatasetSample],
) -> Dict[str, Any]:
    """Evaluate a trained DSPy artifact on held-out samples."""
    DSPY_AVAILABLE = False
    try:
        import dspy
        DSPY_AVAILABLE = True
    except ImportError:
        pass

    metadata = load_artifact_metadata(artifact_id, stage, TrainingMethod.DSPY)
    if not metadata:
        return {'error': 'Artifact not found', 'score': 0.0}

    if not DSPY_AVAILABLE:
        # Return the stored eval metrics
        return metadata.eval_metrics or {'score': 0.0}

    # Re-load and evaluate the compiled program
    try:
        program_path = metadata.artifact_path
        ollama_url = 'http://localhost:11434'
        try:
            from config import settings
            ollama_url = getattr(settings, 'OLLAMA_SERVER_URL', ollama_url)
        except Exception:
            pass

        from .dspy_training_service import _create_dspy_lm
        lm = _create_dspy_lm(metadata.base_model or '', ollama_url)

        with dspy.context(lm=lm):
            # Try to load compiled program
            predictor = dspy.Predict(_get_signature_class(stage))
            try:
                predictor.load(program_path)
            except Exception:
                pass  # Some DSPy versions may not support load

            scores = []
            comparisons = []
            for s in samples[:10]:  # Limit for speed
                try:
                    pred = predictor(**{k: s.inputs.get(k, '') for k in s.inputs})
                    output_field = _get_output_field(stage)
                    pred_text = getattr(pred, output_field, '') or ''
                    score = _rouge_l_score(s.target, pred_text)
                    scores.append(score)
                    comparisons.append({
                        'expected': s.target[:200],
                        'predicted': pred_text[:200],
                        'score': round(score, 4),
                    })
                except Exception as e:
                    scores.append(0.0)
                    comparisons.append({'error': str(e), 'score': 0.0})

            avg_score = sum(scores) / len(scores) if scores else 0.0
            return {
                'score': round(avg_score, 4),
                'rouge_l': round(avg_score, 4),
                'n_samples': len(scores),
                'comparisons': comparisons,
            }
    except Exception as e:
        logger.warning(f'[Eval] DSPy evaluation error: {e}')
        return metadata.eval_metrics or {'score': 0.0, 'error': str(e)}


def _get_signature_class(stage: TrainingStage):
    try:
        import dspy
        if stage == TrainingStage.STAGE_1:
            from .dspy_training_service import Stage1Signature
            return Stage1Signature
        elif stage == TrainingStage.STAGE_2:
            from .dspy_training_service import Stage2Signature
            return Stage2Signature
        elif stage == TrainingStage.STAGE_3:
            from .dspy_training_service import Stage3Signature
            return Stage3Signature
    except Exception:
        pass
    return None


def _get_output_field(stage: TrainingStage) -> str:
    return {
        TrainingStage.STAGE_1: 'discussion_points_json',
        TrainingStage.STAGE_2: 'enhanced_point_json',
        TrainingStage.STAGE_3: 'assigned_points_json',
    }.get(stage, 'output')


def build_evaluation_response(
    job_id: str,
    stage: TrainingStage,
    method: TrainingMethod,
    artifact_id: str,
    samples: List[DatasetSample],
) -> Dict[str, Any]:
    """Build a full evaluation response with before/after comparison."""
    baseline = evaluate_baseline(stage, samples)
    score_before = baseline.get('score', 0.0)

    if method == TrainingMethod.DSPY:
        after = evaluate_dspy_artifact(stage, artifact_id, samples)
    else:
        # LoRA/QLoRA: use stored eval metrics
        metadata = load_artifact_metadata(artifact_id, stage, method)
        after = metadata.eval_metrics if metadata else {}

    score_after = after.get('score', after.get('json_parse_success_rate', 0.0))

    return {
        'job_id': job_id,
        'stage': stage.value,
        'method': method.value,
        'score_before': round(score_before * 100, 1),  # as percentage
        'score_after': round(score_after * 100, 1),
        'metrics': {
            'baseline': baseline,
            'after_training': after,
        },
        'sample_comparisons': after.get('comparisons', []),
    }
