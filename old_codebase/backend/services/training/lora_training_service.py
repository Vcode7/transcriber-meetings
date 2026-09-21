"""
LoRA Fine-tuning Service.

IMPORTANT: LoRA fine-tuning requires a HuggingFace-format model directory.
Ollama models are GGUF-quantized and CANNOT be fine-tuned with PEFT.
This service validates the model before starting and returns a clear error
if the model/environment combination is unsupported.

The existing active model is NEVER replaced. All adapters are saved separately.
"""
import json
import logging
import uuid
from typing import List, Optional, Callable, Dict, Any
from datetime import datetime, timezone
from pathlib import Path

from .training_models import TrainingStage, TrainingMethod, DatasetSample, TrainingJobProgress, TrainingArtifactMetadata
from .training_storage_service import create_artifact_dir, save_artifact_metadata
from .training_schemas import StageTrainingConfig
from .training_model_service import validate_model_for_method

logger = logging.getLogger(__name__)


def run_lora_training(
    stage: TrainingStage,
    config: StageTrainingConfig,
    samples: List[DatasetSample],
    progress_callback: Optional[Callable[[TrainingJobProgress], None]] = None,
    cancel_event=None,
) -> Dict[str, Any]:
    """
    Run LoRA fine-tuning for a single stage.
    Returns artifact metadata dict.
    
    Requirements:
    - config.hf_model_path must point to a valid HuggingFace model directory
    - PEFT library must be installed
    - Sufficient GPU/CPU memory
    """
    def _progress(msg: str, pct: float, step: int = 0, total: int = 0):
        if progress_callback:
            progress_callback(TrainingJobProgress(step=step, total_steps=total, percent=pct, message=msg))
        logger.info(f'[LoRA {stage.value}] {msg}')

    # Validate model compatibility first
    validation = validate_model_for_method(
        config.model_name,
        TrainingMethod.LORA,
        config.hf_model_path,
    )
    if not validation['valid']:
        raise ValueError(f'LoRA validation failed: {validation["reason"]}')

    _progress('Validating environment...', 2.0)

    # CUDA GPU Log Verification
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            _progress(f'[CUDA Check] PyTorch CUDA GPU Active: {gpu_name} ({vram:.2f} GB VRAM) | Device: cuda:0', 3.0)
        else:
            _progress('[CUDA Check] WARNING: PyTorch CUDA not detected. Training will fall back to CPU.', 3.0)
    except Exception as e:
        _progress(f'[CUDA Check] GPU inspection error: {e}', 3.0)

    # Check PEFT availability
    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError:
        raise RuntimeError(
            'PEFT library not installed. Install with: pip install peft\\n'
            'Also required: transformers, accelerate, torch'
        )

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
    except ImportError:
        raise RuntimeError('transformers library not available')

    import torch

    if not samples:
        raise ValueError(f'No training samples for {stage.value}')

    hf_path = config.hf_model_path
    _progress(f'Loading model from {hf_path}...', 5.0)

    try:
        tokenizer = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            hf_path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map='auto' if torch.cuda.is_available() else 'cpu',
        )
    except Exception as e:
        raise RuntimeError(f'Failed to load HuggingFace model from {hf_path}: {e}')

    _progress('Applying LoRA configuration...', 15.0)

    target_modules = config.lora_target_modules or None
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=target_modules,
        lora_dropout=config.lora_dropout,
        bias='none',
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    _progress('Preparing dataset...', 20.0)

    # Tokenize samples
    texts = []
    for s in samples:
        inputs_str = json.dumps(s.inputs, ensure_ascii=False)
        text = f'INPUT:\\n{inputs_str}\\n\\nOUTPUT:\\n{s.target}'
        texts.append(text)

    split = max(1, int(len(texts) * (1 - config.eval_split)))
    train_texts = texts[:split]
    eval_texts = texts[split:] or texts[:1]

    def tokenize(text_list):
        return tokenizer(
            text_list,
            truncation=True,
            padding='max_length',
            max_length=config.max_seq_length,
            return_tensors='pt',
        )

    try:
        from datasets import Dataset
        train_ds = Dataset.from_dict({'text': train_texts})
        eval_ds = Dataset.from_dict({'text': eval_texts})

        def tokenize_fn(examples):
            enc = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=config.max_seq_length)
            enc['labels'] = enc['input_ids'].copy()
            return enc

        train_ds = train_ds.map(tokenize_fn, batched=True)
        eval_ds = eval_ds.map(tokenize_fn, batched=True)
    except ImportError:
        raise RuntimeError('datasets library not installed. Install with: pip install datasets')

    artifact_id, artifact_dir = create_artifact_dir(stage, TrainingMethod.LORA)

    _progress('Starting LoRA training...', 25.0)

    training_args = TrainingArguments(
        output_dir=str(artifact_dir),
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        fp16=torch.cuda.is_available(),
        logging_steps=5,
        save_strategy='no',
        eval_strategy='no',
        report_to='none',
    )

    class ProgressCallback:
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and progress_callback:
                step = state.global_step
                total = state.max_steps
                pct = 25.0 + 65.0 * (step / max(total, 1))
                loss = logs.get('loss', 0.0)
                progress_callback(TrainingJobProgress(
                    step=step, total_steps=total, epoch=int(state.epoch or 0),
                    total_epochs=config.num_train_epochs, loss=loss, percent=pct,
                    message=f'Training step {step}/{total}, loss={loss:.4f}'
                ))
            if cancel_event and cancel_event.is_set():
                control.should_training_stop = True

    try:
        from transformers import TrainerCallback
        class CancelCallback(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                if cancel_event and cancel_event.is_set():
                    control.should_training_stop = True
                    return control

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            callbacks=[CancelCallback()],
        )
        trainer.train()
    except Exception as e:
        raise RuntimeError(f'LoRA training failed: {e}')

    if cancel_event and cancel_event.is_set():
        raise InterruptedError('Training cancelled by user')

    _progress('Saving LoRA adapter...', 90.0)

    adapter_path = artifact_dir / 'adapter'
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))

    metadata = TrainingArtifactMetadata(
        artifact_id=artifact_id,
        stage=stage,
        method=TrainingMethod.LORA,
        base_model=hf_path,
        dataset_ids=[],
        training_date=datetime.now(timezone.utc).isoformat(),
        config=config.model_dump(),
        status='done',
        eval_metrics={'train_samples': len(train_texts), 'eval_samples': len(eval_texts)},
        artifact_path=str(adapter_path),
        version=1,
        is_active=False,
    )
    save_artifact_metadata(metadata)
    _progress('LoRA training complete!', 100.0)
    return metadata.model_dump()
