"""
QLoRA Fine-tuning Service (4-bit quantized LoRA).
Same requirements and limitations as LoRA — requires HuggingFace-format model.
Ollama/GGUF models are NOT supported.
"""
import json
import logging
from typing import List, Optional, Callable, Dict, Any
from datetime import datetime, timezone

from .training_models import TrainingStage, TrainingMethod, DatasetSample, TrainingJobProgress, TrainingArtifactMetadata
from .training_storage_service import create_artifact_dir, save_artifact_metadata
from .training_schemas import StageTrainingConfig
from .training_model_service import validate_model_for_method

logger = logging.getLogger(__name__)


def run_qlora_training(
    stage: TrainingStage,
    config: StageTrainingConfig,
    samples: List[DatasetSample],
    progress_callback: Optional[Callable[[TrainingJobProgress], None]] = None,
    cancel_event=None,
) -> Dict[str, Any]:
    """
    Run QLoRA (4-bit quantized LoRA) fine-tuning.
    Requires bitsandbytes + peft + transformers with GPU support.
    """
    def _progress(msg: str, pct: float, step: int = 0, total: int = 0):
        if progress_callback:
            progress_callback(TrainingJobProgress(step=step, total_steps=total, percent=pct, message=msg))
        logger.info(f'[QLoRA {stage.value}] {msg}')

    validation = validate_model_for_method(config.model_name, TrainingMethod.QLORA, config.hf_model_path)
    if not validation['valid']:
        raise ValueError(f'QLoRA validation failed: {validation["reason"]}')

    _progress('Validating QLoRA environment...', 2.0)

    # CUDA GPU Log Verification
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            _progress(f'[CUDA Check] PyTorch CUDA GPU Active: {gpu_name} ({vram:.2f} GB VRAM) | QLoRA BitsAndBytes 4-bit CUDA accelerated.', 3.0)
        else:
            _progress('[CUDA Check] ERROR: PyTorch CUDA not detected. QLoRA requires a CUDA-enabled GPU.', 3.0)
    except Exception as e:
        _progress(f'[CUDA Check] GPU inspection error: {e}', 3.0)

    try:
        import bitsandbytes as bnb
    except ImportError:
        raise RuntimeError('bitsandbytes not installed. Install with: pip install bitsandbytes')

    try:
        from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    except ImportError:
        raise RuntimeError('PEFT library not installed. Install with: pip install peft')

    try:
        from transformers import (
            AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig, TrainerCallback
        )
    except ImportError:
        raise RuntimeError('transformers library not available')

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            'QLoRA requires CUDA GPU. No GPU detected. '
            'Consider using DSPy Prompt Optimization (works on CPU) or LoRA (works on CPU but is much slower).'
        )

    hf_path = config.hf_model_path
    _progress(f'Loading model with 4-bit quantization from {hf_path}...', 5.0)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            hf_path,
            quantization_config=bnb_config,
            device_map='auto',
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
    except Exception as e:
        raise RuntimeError(f'Failed to load model for QLoRA: {e}')

    _progress('Applying QLoRA configuration...', 15.0)

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules or None,
        lora_dropout=config.lora_dropout,
        bias='none',
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    _progress('Preparing dataset...', 20.0)

    texts = []
    for s in samples:
        text = f'INPUT:\\n{json.dumps(s.inputs, ensure_ascii=False)}\\n\\nOUTPUT:\\n{s.target}'
        texts.append(text)

    split = max(1, int(len(texts) * (1 - config.eval_split)))
    train_texts = texts[:split]
    eval_texts = texts[split:] or texts[:1]

    try:
        from datasets import Dataset
        def tokenize_fn(examples):
            enc = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=config.max_seq_length)
            enc['labels'] = enc['input_ids'].copy()
            return enc
        train_ds = Dataset.from_dict({'text': train_texts}).map(tokenize_fn, batched=True)
        eval_ds = Dataset.from_dict({'text': eval_texts}).map(tokenize_fn, batched=True)
    except ImportError:
        raise RuntimeError('datasets library not installed')

    artifact_id, artifact_dir = create_artifact_dir(stage, TrainingMethod.QLORA)

    _progress('Starting QLoRA training...', 25.0)

    training_args = TrainingArguments(
        output_dir=str(artifact_dir),
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        bf16=True,
        logging_steps=5,
        save_strategy='no',
        eval_strategy='no',
        report_to='none',
        optim='paged_adamw_8bit',
    )

    class CancelCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if cancel_event and cancel_event.is_set():
                control.should_training_stop = True
            if progress_callback:
                step = state.global_step
                total = state.max_steps
                pct = 25.0 + 65.0 * (step / max(total, 1))
                progress_callback(TrainingJobProgress(
                    step=step, total_steps=total, percent=pct,
                    message=f'QLoRA step {step}/{total}'
                ))
            return control

    try:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            callbacks=[CancelCallback()],
        )
        trainer.train()
    except Exception as e:
        raise RuntimeError(f'QLoRA training failed: {e}')

    if cancel_event and cancel_event.is_set():
        raise InterruptedError('Training cancelled by user')

    _progress('Saving QLoRA adapter...', 90.0)

    adapter_path = artifact_dir / 'adapter'
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))

    metadata = TrainingArtifactMetadata(
        artifact_id=artifact_id,
        stage=stage,
        method=TrainingMethod.QLORA,
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
    _progress('QLoRA training complete!', 100.0)
    return metadata.model_dump()
