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

logger = logging.getLogger(__name__)

CHECKPOINTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints", "rom_training")
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)


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

        # Run test inference
        logger.info(f"[RomTraining] variant={variant_id} running test inference")
        test_results = _run_test_inference(
            variant_id=variant_id,
            adapter_path=adapter_path,
            base_model=base_model,
            test_data=test_data,
        )

        accuracy = _compute_accuracy(test_results)
        loop.run_until_complete(_update_status("done", test_results=test_results, accuracy=accuracy))
        logger.info(f"[RomTraining] variant={variant_id} training complete. accuracy={accuracy}")

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

        # Build input (same format as Short ROM generation uses)
        lines = []
        for i, pt in enumerate(long_rom_points):
            if isinstance(pt, dict):
                text_content = (pt.get("polished_text") or pt.get("text") or "").strip()
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

        # Build target (manual MoM points as numbered list)
        target_lines = []
        for pt in manual_mom_points:
            if isinstance(pt, str):
                target_lines.append(f"- {pt}")
            elif isinstance(pt, dict):
                target_lines.append(f"- {pt.get('text', str(pt))}")
        target_text = "\n".join(target_lines)

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
        from trl import SFTTrainer
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
    use_4bit = lora_config.get("use_4bit", True) and method == "qlora"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load base model
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=getattr(torch, lora_config.get("bnb_4bit_compute_dtype", "float16")),
            bnb_4bit_use_double_quant=False,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

    # If upgrade training, load parent adapter first
    if training_type == "upgrade" and parent_variant_id:
        parent_adapter = _get_adapter_path(parent_variant_id)
        if os.path.exists(parent_adapter):
            logger.info(f"[RomTraining] Loading parent adapter from {parent_adapter}")
            model = PeftModel.from_pretrained(model, parent_adapter)
            model = model.merge_and_unload()  # merge so we can add new adapter

    # Configure LoRA
    target_modules = [m.strip() for m in lora_config.get("target_modules", "q_proj,v_proj").split(",") if m.strip()]
    peft_config = LoraConfig(
        r=lora_config.get("r", 16),
        lora_alpha=lora_config.get("lora_alpha", 32),
        lora_dropout=lora_config.get("lora_dropout", 0.05),
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Build HF Dataset
    def format_example(ex):
        text = f"<|user|>\n{ex['input']}<|end|>\n<|assistant|>\n{ex['output']}<|end|>"
        return {"text": text}

    raw_dataset = [format_example(ex) for ex in examples]
    hf_dataset = Dataset.from_list(raw_dataset)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=adapter_path,
        num_train_epochs=lora_config.get("num_epochs", 3),
        per_device_train_batch_size=lora_config.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=lora_config.get("gradient_accumulation_steps", 4),
        learning_rate=lora_config.get("learning_rate", 2e-4),
        logging_steps=1,
        save_steps=50,
        save_total_limit=2,
        fp16=not use_4bit,
        bf16=False,
        report_to="none",
        remove_unused_columns=False,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=hf_dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=lora_config.get("max_seq_length", 2048),
        tokenizer=tokenizer,
        args=training_args,
        packing=False,
    )

    trainer.train()
    trainer.save_model(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    logger.info(f"[RomTraining] Adapter saved to {adapter_path}")


def _run_test_inference(
    variant_id: str,
    adapter_path: str,
    base_model: str,
    test_data: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    For each test sample, run inference with:
    - base model (before training)
    - trained model (after training)
    - return alongside the manual MoM as ground truth
    """
    from services.ai_provider import get_provider
    from services.rom_training.rom_training_inference import infer_short_rom_with_variant
    from services.rom_training.rom_training_dataset_service import _extract_mom_regex

    examples = _build_training_examples(test_data)
    provider = get_provider()
    results = []

    for item, example in zip(test_data, examples[:len(test_data)]):
        input_prompt = example["input"]
        manual_output = example["output"]

        # Before: base model
        try:
            if hasattr(provider, 'query'):
                before_output = provider.query(input_prompt, max_tokens=2048, temperature=0.2)
            else:
                before_output = provider._infer(input_prompt, max_new_tokens=2048)
        except Exception as e:
            before_output = f"[Error: {e}]"

        # After: trained variant
        try:
            after_output = infer_short_rom_with_variant(
                prompt=input_prompt,
                variant_id=variant_id,
                adapter_path=adapter_path,
                base_model=base_model,
                max_tokens=2048,
            )
        except Exception as e:
            after_output = f"[Error: {e}]"

        results.append({
            "agenda_title": item.get("agenda_title", ""),
            "recording_title": item.get("recording_title", ""),
            "before": before_output,
            "after": after_output,
            "manual": manual_output,
        })

    return results


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
        f"Avg word overlap — Before: {avg_before:.1%}, After: {avg_after:.1%}, "
        f"Improvement: {improvement:+.1%} ({len(test_results)} test samples)"
    )


# ── Variant CRUD helpers ──────────────────────────────────────────────────────

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
    return [{
        "id": row[0], "name": row[1], "description": row[2],
        "base_model": row[3], "training_type": row[4], "parent_variant_id": row[5],
        "adapter_path": row[6], "status": row[7],
        "config": from_json(row[8], default={}),
        "dataset_size": row[9], "test_size": row[10],
        "test_results": from_json(row[11], default=[]),
        "accuracy": row[12], "error": row[13],
        "created_at": row[14], "completed_at": row[15],
    } for row in rows]


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
    return {
        "id": row[0], "name": row[1], "description": row[2],
        "base_model": row[3], "training_type": row[4], "parent_variant_id": row[5],
        "adapter_path": row[6], "status": row[7],
        "config": from_json(row[8], default={}),
        "dataset_size": row[9], "test_size": row[10],
        "test_results": from_json(row[11], default=[]),
        "accuracy": row[12], "error": row[13],
        "created_at": row[14], "completed_at": row[15],
    }


async def delete_variant(variant_id: str, user_id: str, db) -> bool:
    from sqlalchemy import text
    r = await db.execute(
        text("DELETE FROM rom_training_variants WHERE id = :id AND user_id = :uid"),
        {"id": variant_id, "uid": user_id},
    )
    await db.commit()
    # Also try to clean up the adapter files
    adapter_path = _get_adapter_path(variant_id)
    if os.path.exists(adapter_path):
        try:
            import shutil
            shutil.rmtree(adapter_path)
        except Exception:
            pass
    return r.rowcount > 0
