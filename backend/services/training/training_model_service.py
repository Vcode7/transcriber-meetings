import logging
import httpx
from typing import List, Dict, Any
from config import settings
from .training_models import TrainingMethod

logger = logging.getLogger(__name__)

async def get_ollama_models() -> List[Dict[str, Any]]:
    """Fetch list of locally installed Ollama models."""
    try:
        ollama_url = getattr(settings, 'OLLAMA_SERVER_URL', 'http://localhost:11434')
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{ollama_url}/api/tags")
            resp.raise_for_status()
            data = resp.json()
            models = data.get('models', [])
            return [
                {
                    'name': m.get('name', ''),
                    'size': m.get('size', 0),
                    'modified_at': m.get('modified_at', ''),
                    'details': m.get('details', {}),
                }
                for m in models
            ]
    except Exception as e:
        logger.warning(f'[Training Model Service] Could not fetch Ollama models: {e}')
        return []

def validate_model_for_method(model_name: str, method: TrainingMethod, hf_model_path: str = None) -> Dict[str, Any]:
    """
    Validates whether a model+method combination is supported.
    
    - DSPy: valid for any Ollama model (uses it as inference backend only)
    - LoRA/QLoRA: requires a HuggingFace-format model directory (NOT a GGUF/Ollama model)
      All models from Ollama are GGUF-quantized and cannot be fine-tuned with PEFT.
      A valid HF model path must be provided via hf_model_path.
    """
    if method == TrainingMethod.DSPY:
        if not model_name:
            return {'valid': False, 'reason': 'No model selected.', 'warnings': []}
        return {
            'valid': True,
            'reason': f'DSPy prompt optimization works with any Ollama model. {model_name} is compatible.',
            'warnings': []
        }
    
    # LoRA or QLoRA
    method_name = 'QLoRA' if method == TrainingMethod.QLORA else 'LoRA'
    
    if hf_model_path:
        import os
        if os.path.isdir(hf_model_path):
            # Check for HF model files
            has_config = os.path.exists(os.path.join(hf_model_path, 'config.json'))
            if has_config:
                return {
                    'valid': True,
                    'reason': f'{method_name} fine-tuning is possible with the HuggingFace model at {hf_model_path}.',
                    'warnings': ['Ensure the model architecture supports PEFT LoRA (e.g., LLaMA, Mistral, Qwen).'],
                }
            else:
                return {
                    'valid': False,
                    'reason': f'The directory {hf_model_path} does not appear to be a valid HuggingFace model (missing config.json).',
                    'warnings': [],
                }
        else:
            return {
                'valid': False,
                'reason': f'HuggingFace model path not found: {hf_model_path}',
                'warnings': [],
            }
    
    # No HF path provided — Ollama model selected
    return {
        'valid': False,
        'reason': (
            f'{method_name} fine-tuning cannot be performed on Ollama models. '
            f'Ollama serves quantized GGUF models which do not support PEFT adapter training. '
            f'To use {method_name}: (1) Download the base HuggingFace model in non-quantized format, '
            f'(2) Provide the local directory path in the HF Model Path field, '
            f'(3) Ensure you have sufficient VRAM (typically 12GB+ for LoRA on 7B models, 8GB+ for QLoRA). '
            f'Alternatively, use DSPy Prompt Optimization — it works with any Ollama model.'
        ),
        'warnings': [],
    }
