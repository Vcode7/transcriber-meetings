"""
ROM Training Service
Orchestrates LoRA/QLoRA fine-tuning for Short ROM generation.
"""
import json
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from pathlib import Path
import sys

logger = logging.getLogger(__name__)


def _resolve_checkpoints_dir() -> str:
    """
    Resolve checkpoint storage directory:
    - In frozen PyInstaller .exe: Application/backend/checkpoints/rom_training
    - In normal development: backend/checkpoints/rom_training
    """
    if getattr(sys, "frozen", False):
        base = Path(sys.executable).parent
    else:
        base = Path(__file__).resolve().parent.parent.parent
    chk_dir = base / "checkpoints" / "rom_training"
    chk_dir.mkdir(parents=True, exist_ok=True)
    return str(chk_dir)


CHECKPOINTS_DIR = _resolve_checkpoints_dir()


def _get_adapter_path(variant_id: str) -> str:
    return os.path.join(CHECKPOINTS_DIR, variant_id, "adapter")


async def create_variant_record(
    user_id: str,
    name: str,
    description: Optional[str],
    base_model: str,
    training_type: str,
    parent_variant_id: Optional[str],
    config: Dict[str, Any],
    dataset_size: int,
    test_size: int,
    db,
) -> str:
    """Create a variant row in DB with status=pending, return its ID."""
    from sqlalchemy import text
    from database import to_json

    variant_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    adapter_path = _get_adapter_path(variant_id)

    await db.execute(
        text("""
            INSERT INTO rom_training_variants
            (id, user_id, name, description, base_model, training_type, parent_variant_id,
             adapter_path, status, config, dataset_size, test_size, test_results,
             created_at)
            VALUES (:id, :uid, :name, :desc, :bm, :tt, :pvid, :ap, 'pending', :cfg,
                    :ds, :ts, '[]', :now)
        """),
        {
            "id": variant_id,
            "uid": user_id,
            "name": name,
            "desc": description,
            "bm": base_model,
            "tt": training_type,
            "pvid": parent_variant_id,
            "ap": adapter_path,
            "cfg": to_json(config),
            "ds": dataset_size,
            "ts": test_size,
            "now": now,
        },
    )
    await db.commit()
    return variant_id


async def start_training(
    variant_id: str,
    user_id: str,
    base_model: str,
    training_type: str,
    parent_variant_id: Optional[str],
    lora_config: Dict[str, Any],
    train_data: List[Dict[str, Any]],
    test_data: List[Dict[str, Any]],
    db_url: str,
) -> None:
    """Start training in a background thread."""
    thread = threading.Thread(
        target=_run_training_thread,
        args=(variant_id, user_id, base_model, training_type, parent_variant_id,
               lora_config, train_data, test_data, db_url),
        daemon=True,
    )
    thread.start()
    logger.info(f"[RomTraining] Started training thread for variant {variant_id}")


def _run_training_thread(
    variant_id: str,
    user_id: str,
    base_model: str,
    training_type: str,
    parent_variant_id: Optional[str],
    lora_config: Dict[str, Any],
    train_data: List[Dict[str, Any]],
    test_data: List[Dict[str, Any]],
    db_url: str,
) -> None:
    """Background thread: runs LoRA/QLoRA fine-tuning."""
    import asyncio
    import aiosqlite

    async def _update_status(status: str, error: Optional[str] = None,
                              test_results=None, accuracy: Optional[str] = None):
        now = datetime.now(timezone.utc).isoformat()
        from sqlalchemy import text
        from database import to_json, AsyncSessionLocal
        async with AsyncSessionLocal() as db:
            updates = {"status": status, "id": variant_id}
            set_parts = ["status = :status"]
            if error is not None:
                updates["error"] = error
                set_parts.append("error = :error")
            if test_results is not None:
                updates["test_results"] = to_json(test_results)
                set_parts.append("test_results = :test_results")
            if accuracy is not None:
                updates["accuracy"] = accuracy
                set_parts.append("accuracy = :accuracy")
            if status in ("done", "error"):
                updates["completed_at"] = now
                set_parts.append("completed_at = :completed_at")
            await db.execute(
                text(f"UPDATE rom_training_variants SET {', '.join(set_parts)} WHERE id = :id"),
                updates,
            )
            await db.commit()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(_update_status("training"))
        logger.info(f"[RomTraining] variant={variant_id} starting LoRA fine-tune")

        adapter_path = _get_adapter_path(variant_id)
        os.makedirs(adapter_path, exist_ok=True)

        _run_lora_training(
            variant_id=variant_id,
            base_model=base_model,
            training_type=training_type,
            parent_variant_id=parent_variant_id,
            lora_config=lora_config,
            train_data=train_data,
            adapter_path=adapter_path,
        )

        # Free training gradients & optimizer from VRAM before test inference
        try:
            from services.rom_training.rom_training_inference import clean_vram
            clean_vram()
        except Exception:
            pass

        # Run test inference
        logger.info(f"[RomTraining] variant={variant_id} running test inference")
        try:
            test_results = _run_test_inference(
                variant_id=variant_id,
                adapter_path=adapter_path,
                base_model=base_model,
                test_data=test_data,
            )
            accuracy = _compute_accuracy(test_results)
            loop.run_until_complete(_update_status("done", test_results=test_results, accuracy=accuracy))
            logger.info(f"[RomTraining] variant={variant_id} training complete. accuracy={accuracy}")
        except Exception as test_err:
            logger.error(f"[RomTraining] variant={variant_id} test inference warning: {test_err}", exc_info=True)
            # Training succeeded and adapter was saved; record partial warning instead of failing the variant
            loop.run_until_complete(_update_status("done", error=f"Training succeeded and adapter saved, but test inference had an issue: {test_err}"))

    except Exception as e:
        logger.error(f"[RomTraining] variant={variant_id} training failed: {e}", exc_info=True)
        loop.run_until_complete(_update_status("error", error=str(e)))
    finally:
        loop.close()


def _build_training_examples(
    train_data: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """
    Build instruction-tuning examples from training data pairs.
    Input: Long ROM points for an agenda
    Target: Manually written MoM points for the same agenda
    """
    from services.prompt_service import get_prompt_sync

    examples = []
    # Use the same short ROM prompt template to build the input
    try:
        prompt_template = get_prompt_sync("rom_version_short")
    except Exception:
        prompt_template = (
            "Agenda: {agenda_title}\n\nDiscussion Points:\n{points_json}"
            "\n\nCondense the above into key MoM points.{rules_section}{mandatory_section}"
        )

    for item in train_data:
        agenda_title = item.get("agenda_title", "General")
        long_rom_points = item.get("long_rom_points", [])
        manual_mom_points = item.get("manual_mom_points", [])

        if not long_rom_points or not manual_mom_points:
            continue

        lines = []
        for i, pt in enumerate(long_rom_points):
            if isinstance(pt, dict):
                text_content = (
                    pt.get("polished_text")
                    or pt.get("text")
                    or pt.get("discussion_point")
                    or pt.get("point")
                    or ""
                ).strip()
            else:
                text_content = str(pt).strip()
            lines.append(f"[Point {i+1}] {text_content}")
        points_json = "\n\n".join(lines)

        try:
            input_prompt = prompt_template.format(
                agenda_title=agenda_title,
                points_json=points_json,
                rules_section="",
                mandatory_section="",
            )
        except Exception:
            input_prompt = f"Agenda: {agenda_title}\n\nPoints:\n{points_json}"

        # Build target points (JSON array strictly matching prompt output contract)
        target_pts = []
        for pt in manual_mom_points:
            if isinstance(pt, str):
                t = pt.strip()
                if t:
                    target_pts.append(t)
            elif isinstance(pt, dict):
                t = (pt.get("polished_text") or pt.get("text") or "").strip()
                if t:
                    target_pts.append(t)

        if not target_pts:
            continue

        target_text = json.dumps(target_pts, ensure_ascii=False, indent=2)
        examples.append({"input": input_prompt, "output": target_text})

    return examples


def _run_lora_training(
    variant_id: str,
    base_model: str,
    training_type: str,
    parent_variant_id: Optional[str],
    lora_config: Dict[str, Any],
    train_data: List[Dict[str, Any]],
    adapter_path: str,
) -> None:
    """Run LoRA/QLoRA fine-tuning using HuggingFace PEFT + TRL."""
    try:
        import torch
        from transformers import (
            AutoModelForCausalLM, AutoTokenizer, TrainingArguments,
            BitsAndBytesConfig, DataCollatorForSeq2Seq,
        )
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
        try:
            from trl import SFTTrainer, SFTConfig
            has_sft_config = True
        except ImportError:
            from trl import SFTTrainer
            has_sft_config = False
        from datasets import Dataset
    except ImportError as e:
        raise RuntimeError(
            f"Training dependencies not installed: {e}. "
            "Install: pip install transformers peft trl datasets bitsandbytes"
        )

    examples = _build_training_examples(train_data)
    if not examples:
        raise ValueError("No valid training examples could be built from the provided data.")

    logger.info(f"[RomTraining] variant={variant_id}: {len(examples)} training examples")

    method = lora_config.get("method", "qlora")
    use_cuda = torch.cuda.is_available()
    use_4bit = use_cuda and lora_config.get("use_4bit", True) and method == "qlora"

    resolved_model = resolve_base_model_path(base_model)
    is_local_dir = os.path.isdir(resolved_model)

    if not is_local_dir:
        from config import settings as app_settings
        if getattr(app_settings, "OFFLINE_MODE", False) or ":" in resolved_model or "\\" in resolved_model or "/" in resolved_model:
            raise FileNotFoundError(
                f"Base model directory not found at '{resolved_model}'. "
                "Please verify that the local base model folder ('nlp_engine') exists on this machine."
            )

    tokenizer_kwargs = {"trust_remote_code": True}
    if is_local_dir:
        tokenizer_kwargs["local_files_only"] = True

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(resolved_model, **tokenizer_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    from services.rom_training.rom_training_inference import _get_offload_dir
    offload_dir = _get_offload_dir()

    use_cuda = torch.cuda.is_available()
    has_bf16 = use_cuda and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()

    raw_dtype_str = str(lora_config.get("bnb_4bit_compute_dtype", "")).lower()
    if raw_dtype_str in ("bfloat16", "bf16"):
        if has_bf16:
            compute_dtype = torch.bfloat16
        else:
            logger.warning("[RomTraining] bfloat16 requested but current CUDA device does not support native bf16 (Ampere+ required). Falling back to float16.")
            compute_dtype = torch.float16
    elif raw_dtype_str == "float32":
        compute_dtype = torch.float32
    elif raw_dtype_str in ("float16", "fp16"):
        compute_dtype = torch.float16
    else:
        # Auto-detect optimal compute dtype: bfloat16 if GPU natively supports it (matches base model), else float16
        compute_dtype = torch.bfloat16 if has_bf16 else (torch.float16 if use_cuda else torch.float32)

    model_dtype = compute_dtype if use_cuda else torch.float32

    # Load base model
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=False,
        )
        model = AutoModelForCausalLM.from_pretrained(
            resolved_model,
            quantization_config=bnb_config,
            torch_dtype=model_dtype,
            device_map="auto",
            offload_folder=offload_dir,
            trust_remote_code=True,
            local_files_only=is_local_dir,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            resolved_model,
            torch_dtype=model_dtype,
            device_map="auto" if use_cuda else None,
            offload_folder=offload_dir if use_cuda else None,
            trust_remote_code=True,
            local_files_only=is_local_dir,
        )

    # If upgrade training, load parent adapter directly in trainable mode
    is_upgrade = False
    if training_type == "upgrade" and parent_variant_id:
        parent_adapter = resolve_adapter_path(_get_adapter_path(parent_variant_id), parent_variant_id)
        if os.path.exists(parent_adapter):
            logger.info(f"[RomTraining] Loading parent adapter from {parent_adapter} for upgrade fine-tuning")
            model = PeftModel.from_pretrained(model, parent_adapter, is_trainable=True)
            is_upgrade = True

    # Configure LoRA if new training
    if not is_upgrade:
        target_modules = [m.strip() for m in lora_config.get("target_modules", "q_proj,v_proj").split(",") if m.strip()]
        peft_config = LoraConfig(
            r=lora_config.get("r", 16),
            lora_alpha=lora_config.get("lora_alpha", 32),
            lora_dropout=lora_config.get("lora_dropout", 0.05),
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        peft_config = None

    # Determine mixed precision based on verified device capabilities
    train_bf16 = has_bf16 and compute_dtype == torch.bfloat16
    train_fp16 = use_cuda and not train_bf16

    # ── Critical Mixed-Precision & GradScaler Sanitization ──────────
    # PyTorch GradScaler Rule (when train_fp16 is True):
    # - Trainable parameters (LoRA adapter weights) MUST be torch.float32 master weights.
    #   If any trainable tensor is bfloat16, GradScaler crashes with:
    #   "torch._amp_foreach_non_finite_check_and_unscale not implemented for bfloat16".
    # - All non-trainable weights that were loaded in bfloat16 from base model must be float16.
    # - In native BF16 mode (train_bf16 is True): GradScaler is completely disabled by Trainer,
    #   and trainable parameters align to torch.bfloat16.
    if train_fp16:
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.dtype != torch.float32:
                    param.data = param.data.to(torch.float32)
            else:
                if param.dtype == torch.bfloat16:
                    param.data = param.data.to(torch.float16)
    elif train_bf16:
        for name, param in model.named_parameters():
            if param.requires_grad and param.dtype != torch.bfloat16:
                param.data = param.data.to(torch.bfloat16)

    # Build HF Dataset using model's chat template
    def format_example(ex):
        messages = [
            {"role": "user", "content": ex["input"]},
            {"role": "assistant", "content": ex["output"]},
        ]
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False)
        except Exception:
            text = f"<|im_start|>user\n{ex['input']}<|im_end|>\n<|im_start|>assistant\n{ex['output']}<|im_end|>"
        return {"text": text}

    raw_dataset = [format_example(ex) for ex in examples]
    hf_dataset = Dataset.from_list(raw_dataset)

    # Dynamic batch size & gradient accumulation
    num_samples = len(examples)
    configured_batch = lora_config.get("per_device_train_batch_size", 2)
    effective_batch = max(1, min(configured_batch, num_samples))
    configured_accum = max(1, lora_config.get("gradient_accumulation_steps", 4))
    if num_samples <= effective_batch:
        grad_accum = 1
    else:
        grad_accum = max(1, min(configured_accum, num_samples // effective_batch))

    # Training arguments
    max_seq_len = lora_config.get("max_seq_length", 2048)

    if has_sft_config:
        import inspect
        sft_cfg_params = inspect.signature(SFTConfig.__init__).parameters
        sft_cfg_kwargs = {
            "output_dir": adapter_path,
            "num_train_epochs": lora_config.get("num_epochs", 3),
            "per_device_train_batch_size": effective_batch,
            "gradient_accumulation_steps": grad_accum,
            "learning_rate": lora_config.get("learning_rate", 2e-4),
            "logging_steps": 1,
            "save_steps": 50,
            "save_total_limit": 2,
            "fp16": train_fp16,
            "bf16": train_bf16,
            "report_to": "none",
            "remove_unused_columns": False,
            "warmup_ratio": 0.03,
            "lr_scheduler_type": "cosine",
            "packing": False,
        }
        if "dataset_text_field" in sft_cfg_params:
            sft_cfg_kwargs["dataset_text_field"] = "text"
        if "max_length" in sft_cfg_params:
            sft_cfg_kwargs["max_length"] = max_seq_len
        elif "max_seq_length" in sft_cfg_params:
            sft_cfg_kwargs["max_seq_length"] = max_seq_len

        training_args = SFTConfig(**sft_cfg_kwargs)

        sft_init_params = inspect.signature(SFTTrainer.__init__).parameters
        trainer_kwargs = {
            "model": model,
            "train_dataset": hf_dataset,
            "args": training_args,
        }
        if "processing_class" in sft_init_params:
            trainer_kwargs["processing_class"] = tokenizer
        elif "tokenizer" in sft_init_params:
            trainer_kwargs["tokenizer"] = tokenizer
        if peft_config is not None:
            trainer_kwargs["peft_config"] = peft_config

        trainer = SFTTrainer(**trainer_kwargs)
    else:
        training_args = TrainingArguments(
            output_dir=adapter_path,
            num_train_epochs=lora_config.get("num_epochs", 3),
            per_device_train_batch_size=effective_batch,
            gradient_accumulation_steps=grad_accum,
            learning_rate=lora_config.get("learning_rate", 2e-4),
            logging_steps=1,
            save_steps=50,
            save_total_limit=2,
            fp16=train_fp16,
            bf16=train_bf16,
            report_to="none",
            remove_unused_columns=False,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
        )

        import inspect
        sft_init_params = inspect.signature(SFTTrainer.__init__).parameters
        trainer_kwargs = {
            "model": model,
            "train_dataset": hf_dataset,
            "args": training_args,
        }
        if peft_config is not None:
            trainer_kwargs["peft_config"] = peft_config
        if "dataset_text_field" in sft_init_params:
            trainer_kwargs["dataset_text_field"] = "text"
        if "max_seq_length" in sft_init_params:
            trainer_kwargs["max_seq_length"] = max_seq_len
        if "processing_class" in sft_init_params:
            trainer_kwargs["processing_class"] = tokenizer
        elif "tokenizer" in sft_init_params:
            trainer_kwargs["tokenizer"] = tokenizer

        trainer = SFTTrainer(**trainer_kwargs)

    trainer.train()
    try:
        trainer.save_model(adapter_path)
    except Exception as e:
        logger.warning(f"[RomTraining] trainer.save_model warning ({e}), falling back to model.save_pretrained()")
        model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    logger.info(f"[RomTraining] Adapter saved to {adapter_path}")

    # Save variant training metadata for future inference quantization detection
    meta = {
        "variant_id": variant_id,
        "base_model": base_model,
        "training_type": training_type,
        "method": method,
        "use_4bit": use_4bit,
        "lora_config": lora_config,
    }
    try:
        with open(os.path.join(adapter_path, "variant_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    except Exception as me:
        logger.warning(f"[RomTraining] Could not write variant_meta.json: {me}")

    # Immediately release training resources from VRAM
    try:
        del trainer
        del model
        del tokenizer
        from services.rom_training.rom_training_inference import clean_vram
        clean_vram()
    except Exception:
        pass


def _run_test_inference(
    variant_id: str,
    adapter_path: str,
    base_model: str,
    test_data: List[Dict[str, Any]],
    disable_thinking: bool = True,
) -> List[Dict[str, Any]]:
    """
    Sequentially runs test inference adhering strictly to memory isolation:
    1. Phase 1: Load base model -> generate all 'before' outputs -> unload completely from VRAM.
    2. Phase 2: Load variant model -> generate all 'after' outputs -> unload completely from VRAM.
    3. Return combined results with ground truth.
    Never runs concurrently and never keeps both in VRAM.
    """
    from services.rom_training.rom_training_inference import run_sequential_test_inference

    examples = _build_training_examples(test_data)
    if not examples or not test_data:
        return []

    return run_sequential_test_inference(
        test_data=test_data,
        variant_id=variant_id,
        adapter_path=adapter_path,
        base_model=base_model,
        examples=examples,
        max_tokens=2048,
        disable_thinking=disable_thinking,
    )


def _compute_accuracy(test_results: List[Dict[str, Any]]) -> str:
    """
    Simple accuracy metric: average word-overlap improvement of 'after' vs 'before'
    relative to 'manual' target.
    """
    if not test_results:
        return "N/A"

    import re

    def word_overlap(a: str, b: str) -> float:
        a_words = set(re.sub(r'[^a-z0-9 ]', '', a.lower()).split())
        b_words = set(re.sub(r'[^a-z0-9 ]', '', b.lower()).split())
        if not a_words or not b_words:
            return 0.0
        return len(a_words & b_words) / max(len(a_words), len(b_words))

    before_scores = []
    after_scores = []
    for r in test_results:
        manual = r.get("manual", "")
        before = r.get("before", "")
        after = r.get("after", "")
        before_scores.append(word_overlap(before, manual))
        after_scores.append(word_overlap(after, manual))

    avg_before = sum(before_scores) / len(before_scores)
    avg_after = sum(after_scores) / len(after_scores)
    improvement = avg_after - avg_before

    return (
        f"Avg word overlap - Before: {avg_before:.1%}, After: {avg_after:.1%}, "
        f"Improvement: {improvement:+.1%} ({len(test_results)} test samples)"
    )


# ── Variant CRUD & Test helpers ──────────────────────────────────────────────

def resolve_base_model_path(base_model: Optional[str]) -> str:
    """
    Robustly resolve the physical directory of the base LLM model.
    Handles:
    - Normalizing path (stripping whitespace, quotes, trailing slashes)
    - If path exists as is and contains config.json/tokenizer/weights, return it
    - If path does NOT exist (e.g. from developer's machine or broken drive letter):
      1. ModelLoader.get_model_path("nlp_engine")
      2. Candidate search across standard locations (Application/runtime/nlp_engine, etc.)
    - Returns resolved directory string if found, otherwise normalized original string
    """
    if not base_model:
        from services.model_loader import ModelLoader
        detected = ModelLoader.get_model_path("nlp_engine")
        return str(detected) if detected else ""

    cleaned = str(base_model).strip().strip('"').strip("'")
    if not cleaned:
        return ""

    def _is_valid_model_dir(p: str) -> bool:
        if not p or not os.path.isdir(p):
            return False
        has_config = os.path.exists(os.path.join(p, "config.json"))
        has_tok = (
            os.path.exists(os.path.join(p, "tokenizer.json")) or
            os.path.exists(os.path.join(p, "tokenizer_config.json")) or
            os.path.exists(os.path.join(p, "vocab.json"))
        )
        has_weights = (
            os.path.exists(os.path.join(p, "model.safetensors")) or
            os.path.exists(os.path.join(p, "model.safetensors.index.json")) or
            any(f.startswith("model-") and f.endswith(".safetensors") for f in os.listdir(p) if os.path.isfile(os.path.join(p, f))) or
            os.path.exists(os.path.join(p, "pytorch_model.bin"))
        )
        return (has_config and has_tok) or (has_config and has_weights)

    if _is_valid_model_dir(cleaned):
        return os.path.normpath(cleaned)

    # If it is a generic name like "nlp_engine" or an invalid path, query ModelLoader
    from services.model_loader import ModelLoader
    detected = ModelLoader.get_model_path("nlp_engine")
    if detected and _is_valid_model_dir(str(detected)):
        return os.path.normpath(str(detected))

    # Candidate locations search
    from config import settings
    base_dir = Path(settings.BASE_DIR)
    search_dirs = [
        base_dir.parent / "Application" / "runtime" / "nlp_engine",
        base_dir.parent / "Application" / "runtime" / "nlp-engine",
        base_dir.parent / "Application" / "runtime" / "models" / "nlp_engine",
        base_dir.parent / "runtime" / "nlp_engine",
        base_dir / "runtime" / "nlp_engine",
        Path.cwd() / "Application" / "runtime" / "nlp_engine",
        Path.cwd() / "runtime" / "nlp_engine",
    ]
    if getattr(sys, "frozen", False):
        exe_parent = Path(sys.executable).parent
        search_dirs.extend([
            exe_parent / "runtime" / "nlp_engine",
            exe_parent / "runtime" / "nlp-engine",
            exe_parent.parent / "runtime" / "nlp_engine",
            exe_parent.parent / "runtime" / "nlp-engine",
            exe_parent.parent / "runtime" / "models" / "nlp_engine",
        ])

    for cand in search_dirs:
        if _is_valid_model_dir(str(cand)):
            return os.path.normpath(str(cand))

    return os.path.normpath(cleaned)


def resolve_adapter_path(adapter_path: Optional[str], variant_id: Optional[str] = None) -> str:
    """
    Robustly resolve the physical directory of a trained LoRA adapter.
    Handles:
    - Normal paths matching the current filesystem
    - Frozen PyInstaller .exe execution (Application/backend/checkpoints/rom_training)
    - Older .exe builds (Application/backend/_internal/checkpoints/rom_training)
    - Application/runtime/checkpoints/rom_training
    - Moving the application directory to a new drive letter or machine
    """
    def _is_valid_adapter(p: str) -> bool:
        if not p or not os.path.isdir(p):
            return False
        has_cfg = os.path.exists(os.path.join(p, "adapter_config.json"))
        has_w = (
            os.path.exists(os.path.join(p, "adapter_model.safetensors")) or
            os.path.exists(os.path.join(p, "adapter_model.bin")) or
            os.path.exists(os.path.join(p, "model.safetensors"))
        )
        return has_cfg or has_w

    if adapter_path and _is_valid_adapter(adapter_path):
        return os.path.normpath(adapter_path)

    # Candidate variant IDs
    candidates_vid = []
    if variant_id:
        candidates_vid.append(str(variant_id).strip())
    if adapter_path:
        parts = adapter_path.replace("\\", "/").rstrip("/").split("/")
        if len(parts) >= 2 and parts[-1].lower() == "adapter":
            candidates_vid.append(parts[-2])
        elif parts:
            candidates_vid.append(parts[-1])

    # Search locations
    search_roots = [
        Path(CHECKPOINTS_DIR),
    ]
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).parent  # Application/backend
        search_roots.extend([
            exe_dir / "checkpoints" / "rom_training",
            exe_dir / "_internal" / "checkpoints" / "rom_training",
            exe_dir.parent / "runtime" / "checkpoints" / "rom_training",
            exe_dir.parent / "checkpoints" / "rom_training",
        ])
    else:
        dev_base = Path(__file__).resolve().parent.parent.parent
        search_roots.extend([
            dev_base / "checkpoints" / "rom_training",
            dev_base.parent / "Application" / "backend" / "checkpoints" / "rom_training",
            dev_base.parent / "Application" / "backend" / "_internal" / "checkpoints" / "rom_training",
        ])

    for vid in candidates_vid:
        for root in search_roots:
            candidate = root / vid / "adapter"
            if _is_valid_adapter(str(candidate)):
                return str(candidate)
            candidate_parent = root / vid
            if _is_valid_adapter(str(candidate_parent)):
                return str(candidate_parent)

    return os.path.normpath(adapter_path) if adapter_path else ""


def check_checkpoint_exists(adapter_path: str, variant_id: Optional[str] = None) -> bool:
    """Check if the adapter directory exists and contains PEFT adapter files."""
    if not adapter_path and not variant_id:
        return False
    resolved = resolve_adapter_path(adapter_path, variant_id)
    if not resolved or not os.path.isdir(resolved):
        return False
    has_config = os.path.exists(os.path.join(resolved, "adapter_config.json"))
    has_weights = (
        os.path.exists(os.path.join(resolved, "adapter_model.safetensors")) or
        os.path.exists(os.path.join(resolved, "adapter_model.bin")) or
        os.path.exists(os.path.join(resolved, "model.safetensors"))
    )
    return has_config or has_weights


async def list_variants(user_id: str, db) -> List[Dict[str, Any]]:
    from sqlalchemy import text
    from database import from_json
    r = await db.execute(
        text("""
            SELECT id, name, description, base_model, training_type, parent_variant_id,
                   adapter_path, status, config, dataset_size, test_size, test_results,
                   accuracy, error, created_at, completed_at
            FROM rom_training_variants
            WHERE user_id = :uid
            ORDER BY created_at DESC
        """),
        {"uid": user_id},
    )
    rows = r.fetchall()
    variants = []
    for row in rows:
        vid = row[0]
        raw_adapter_path = row[6] or ""
        resolved_path = resolve_adapter_path(raw_adapter_path, vid)
        variants.append({
            "id": vid, "name": row[1], "description": row[2],
            "base_model": row[3], "training_type": row[4], "parent_variant_id": row[5],
            "adapter_path": resolved_path, "status": row[7],
            "config": from_json(row[8], default={}),
            "dataset_size": row[9], "test_size": row[10],
            "test_results": from_json(row[11], default=[]),
            "accuracy": row[12], "error": row[13],
            "created_at": row[14], "completed_at": row[15],
            "has_checkpoint": check_checkpoint_exists(resolved_path, vid),
        })
    return variants


async def get_variant(variant_id: str, user_id: str, db) -> Optional[Dict[str, Any]]:
    from sqlalchemy import text
    from database import from_json
    r = await db.execute(
        text("""
            SELECT id, name, description, base_model, training_type, parent_variant_id,
                   adapter_path, status, config, dataset_size, test_size, test_results,
                   accuracy, error, created_at, completed_at
            FROM rom_training_variants
            WHERE id = :id AND user_id = :uid
        """),
        {"id": variant_id, "uid": user_id},
    )
    row = r.fetchone()
    if not row:
        return None
    raw_adapter_path = row[6] or ""
    resolved_path = resolve_adapter_path(raw_adapter_path, variant_id)
    return {
        "id": row[0], "name": row[1], "description": row[2],
        "base_model": row[3], "training_type": row[4], "parent_variant_id": row[5],
        "adapter_path": resolved_path, "status": row[7],
        "config": from_json(row[8], default={}),
        "dataset_size": row[9], "test_size": row[10],
        "test_results": from_json(row[11], default=[]),
        "accuracy": row[12], "error": row[13],
        "created_at": row[14], "completed_at": row[15],
        "has_checkpoint": check_checkpoint_exists(resolved_path, variant_id),
    }


async def rerun_variant_test(
    variant_id: str,
    user_id: str,
    db,
    test_size: Optional[int] = None,
    sample_ids: Optional[List[str]] = None,
    disable_thinking: bool = True,
) -> Dict[str, Any]:
    """
    Rerun test inference for an existing trained variant using saved checkpoint adapter.
    Uses local HuggingFace base model via PyTorch without adapter for 'BEFORE',
    and variant's adapter for 'AFTER'.
    Supports specific sample_ids selection or random test_size sampling.
    """
    import asyncio
    from sqlalchemy import text
    from database import to_json
    from services.rom_training import rom_training_dataset_service as ds

    variant = await get_variant(variant_id, user_id, db)
    if not variant:
        raise ValueError(f"Variant '{variant_id}' not found.")

    adapter_path = variant.get("adapter_path", "")
    if not check_checkpoint_exists(adapter_path):
        raise ValueError(f"Checkpoint adapter files not found at '{adapter_path}'. Cannot rerun test.")

    all_data = await ds.list_training_data(user_id, db)
    if not all_data:
        raise ValueError("No training data pairs found in database to evaluate against.")

    if sample_ids and len(sample_ids) > 0:
        sample_ids_set = {str(s).strip() for s in sample_ids}
        test_data = [d for d in all_data if str(d.get("id")).strip() in sample_ids_set]
        if not test_data:
            raise ValueError("None of the selected sample IDs were found in the dataset.")
    elif test_size is not None and test_size > 0:
        import random
        shuffled = list(all_data)
        random.shuffle(shuffled)
        test_data = shuffled[:max(1, min(test_size, len(all_data)))]
    else:
        # Match previously evaluated samples if available, otherwise sample from DB
        target_count = variant.get("test_size") or 5
        target_count = max(1, min(target_count, len(all_data)))

        prev_results = variant.get("test_results") or []
        prev_keys = {(r.get("recording_title"), r.get("agenda_title")) for r in prev_results if r.get("agenda_title")}

        matched_data = [d for d in all_data if (d.get("recording_title"), d.get("agenda_title")) in prev_keys]
        if len(matched_data) >= target_count:
            test_data = matched_data[:target_count]
        else:
            remaining = [d for d in all_data if d not in matched_data]
            import random
            random.shuffle(remaining)
            test_data = (matched_data + remaining)[:target_count]

    logger.info(
        f"[RomTraining] Rerunning test inference for variant {variant_id} "
        f"with {len(test_data)} samples (disable_thinking={disable_thinking})"
    )

    resolved_base_model = resolve_base_model_path(variant.get("base_model"))

    # Run in executor thread to prevent blocking FastAPI async loop
    loop = asyncio.get_running_loop()
    test_results = await loop.run_in_executor(
        None,
        lambda: _run_test_inference(
            variant_id=variant_id,
            adapter_path=adapter_path,
            base_model=resolved_base_model,
            test_data=test_data,
            disable_thinking=disable_thinking,
        )
    )

    accuracy = _compute_accuracy(test_results)

    # Update database record (mark as done, clear any previous test error)
    await db.execute(
        text("""
            UPDATE rom_training_variants
            SET test_results = :tr, accuracy = :acc, test_size = :ts, status = 'done', error = NULL
            WHERE id = :id AND user_id = :uid
        """),
        {
            "tr": to_json(test_results),
            "acc": accuracy,
            "ts": len(test_data),
            "id": variant_id,
            "uid": user_id,
        },
    )
    await db.commit()
    logger.info(f"[RomTraining] Rerun test complete for variant {variant_id}. Accuracy: {accuracy}")

    return {
        "variant_id": variant_id,
        "test_results": test_results,
        "accuracy": accuracy,
        "test_size": len(test_data),
        "has_checkpoint": True,
        "status": "done",
    }


async def delete_variant(variant_id: str, user_id: str, db) -> bool:
    from sqlalchemy import text
    r = await db.execute(
        text("DELETE FROM rom_training_variants WHERE id = :id AND user_id = :uid"),
        {"id": variant_id, "uid": user_id},
    )
    await db.commit()
    # Also try to clean up the adapter files
    adapter_path = resolve_adapter_path(_get_adapter_path(variant_id), variant_id)
    variant_folder = os.path.dirname(adapter_path) if adapter_path.endswith("adapter") else adapter_path
    target_to_remove = variant_folder if os.path.exists(variant_folder) else adapter_path
    if os.path.exists(target_to_remove):
        try:
            import shutil
            shutil.rmtree(target_to_remove)
        except Exception:
            pass
    return r.rowcount > 0
