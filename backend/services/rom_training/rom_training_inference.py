"""
ROM Training Inference Service
Handles loading LoRA/QLoRA adapters and running inference for Short ROM generation.
"""
import logging
import threading
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# ── In-memory adapter cache (variant_id → loaded model+tokenizer) ──────────
_adapter_cache: Dict[str, Any] = {}
_cache_lock = threading.Lock()


def load_variant(
    variant_id: str,
    adapter_path: str,
    base_model: str,
) -> bool:
    """
    Load a LoRA/QLoRA adapter into the cache.
    Returns True on success, False on failure.
    """
    with _cache_lock:
        if variant_id in _adapter_cache:
            return True  # already loaded

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel

        logger.info(f"[RomInference] Loading variant {variant_id} from adapter {adapter_path}")

        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Detect 4-bit quantization from adapter config
        import json
        import os
        adapter_config_path = os.path.join(adapter_path, "adapter_config.json")
        use_4bit = False
        if os.path.exists(adapter_config_path):
            with open(adapter_config_path) as f:
                ac = json.load(f)
            # Check if QLoRA (quantization config present in base model)
            use_4bit = ac.get("quantization_config", {}).get("load_in_4bit", False)

        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=False,
            )
            base = AutoModelForCausalLM.from_pretrained(
                base_model,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            base = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )

        model = PeftModel.from_pretrained(base, adapter_path)
        model.eval()

        with _cache_lock:
            _adapter_cache[variant_id] = {
                "model": model,
                "tokenizer": tokenizer,
                "base_model": base_model,
                "adapter_path": adapter_path,
            }
        logger.info(f"[RomInference] Variant {variant_id} loaded successfully.")
        return True

    except Exception as e:
        logger.error(f"[RomInference] Failed to load variant {variant_id}: {e}")
        return False


def unload_variant(variant_id: str) -> None:
    """Evict a variant from the cache and free GPU memory."""
    with _cache_lock:
        entry = _adapter_cache.pop(variant_id, None)
    if entry:
        try:
            import torch
            del entry["model"]
            del entry["tokenizer"]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"[RomInference] Variant {variant_id} unloaded.")
        except Exception as e:
            logger.warning(f"[RomInference] Error unloading variant {variant_id}: {e}")


def infer_short_rom_with_variant(
    prompt: str,
    variant_id: str,
    adapter_path: str,
    base_model: str,
    max_tokens: int = 4096,
    use_adapter: bool = True,
) -> str:
    """
    Run inference using a loaded LoRA/QLoRA adapter (or base model if use_adapter=False).
    Auto-loads the adapter if not already in cache.
    Returns the generated text string.
    """
    with _cache_lock:
        cached = _adapter_cache.get(variant_id)

    if cached is None:
        success = load_variant(variant_id, adapter_path, base_model)
        if not success:
            raise RuntimeError(f"Failed to load variant {variant_id} for inference.")
        with _cache_lock:
            cached = _adapter_cache[variant_id]

    try:
        import torch
        model = cached["model"]
        tokenizer = cached["tokenizer"]

        messages = [{"role": "user", "content": prompt}]
        text_input = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(text_input, return_tensors="pt")
        target_device = getattr(model, "device", None) or ("cuda" if torch.cuda.is_available() else "cpu")
        inputs = {k: v.to(target_device) for k, v in inputs.items()}

        pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

        with torch.no_grad():
            if not use_adapter and hasattr(model, "disable_adapter"):
                with model.disable_adapter():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=False,
                        pad_token_id=pad_token_id,
                    )
            else:
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=pad_token_id,
                )

        # Decode only the newly generated tokens
        new_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
        result = tokenizer.decode(new_ids, skip_special_tokens=True)
        return result

    except Exception as e:
        logger.error(f"[RomInference] Inference failed for variant {variant_id} (use_adapter={use_adapter}): {e}")
        raise


def infer_short_rom_base(
    prompt: str,
    base_model: str,
    variant_id: Optional[str] = None,
    adapter_path: Optional[str] = None,
    max_tokens: int = 4096,
) -> str:
    """
    Run inference directly on the local HuggingFace base model without adapter.
    If variant_id and adapter_path are provided, leverages the loaded PeftModel
    via disable_adapter() to prevent loading duplicate models into VRAM.
    """
    if variant_id and adapter_path:
        return infer_short_rom_with_variant(
            prompt=prompt,
            variant_id=variant_id,
            adapter_path=adapter_path,
            base_model=base_model,
            max_tokens=max_tokens,
            use_adapter=False,
        )

    # Standalone base model inference fallback
    cache_key = f"_base_{base_model}"
    with _cache_lock:
        cached = _adapter_cache.get(cache_key)

    if cached is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        logger.info(f"[RomInference] Loading standalone base model {base_model}...")
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        base.eval()
        cached = {
            "model": base,
            "tokenizer": tokenizer,
            "base_model": base_model,
            "adapter_path": "",
        }
        with _cache_lock:
            _adapter_cache[cache_key] = cached

    try:
        import torch
        model = cached["model"]
        tokenizer = cached["tokenizer"]
        messages = [{"role": "user", "content": prompt}]
        text_input = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(text_input, return_tensors="pt")
        target_device = getattr(model, "device", None) or ("cuda" if torch.cuda.is_available() else "cpu")
        inputs = {k: v.to(target_device) for k, v in inputs.items()}

        pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
            )
        new_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
        return tokenizer.decode(new_ids, skip_special_tokens=True)
    except Exception as e:
        logger.error(f"[RomInference] Base model inference failed: {e}")
        raise
