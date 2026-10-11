"""
ROM Training Inference Service
Handles loading LoRA/QLoRA adapters and running inference for Short ROM generation.
Guarantees strict sequential lifecycle:
- Loads base model, generates baseline outputs, unloads completely to free VRAM.
- Loads variant adapter model, generates fine-tuned outputs, unloads completely to free VRAM.
- Never runs base and variant concurrently to avoid GPU Out-Of-Memory (OOM).
"""
import gc
import json
import logging
import os
import tempfile
import threading
from typing import Optional, Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

# ── In-memory adapter cache (variant_id → loaded model+tokenizer) ──────────
_adapter_cache: Dict[str, Any] = {}
_cache_lock = threading.Lock()


def clean_vram() -> None:
    """Aggressively release all unused memory from GPU and RAM."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            vram_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            vram_max_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            logger.info(f"[RomInference] VRAM cleaned. Current allocated: {vram_mb:.1f} MB (peak: {vram_max_mb:.1f} MB)")
    except Exception as e:
        logger.warning(f"[RomInference] Error cleaning VRAM: {e}")


def _get_offload_dir() -> str:
    """Provide a dedicated offload directory to prevent accelerate offload errors."""
    offload_dir = os.path.join(tempfile.gettempdir(), "transcriber_model_offload")
    os.makedirs(offload_dir, exist_ok=True)
    return offload_dir


def _should_use_4bit(adapter_path: Optional[str] = None, explicit_4bit: Optional[bool] = None) -> bool:
    """
    Determine if 4-bit NF4 quantization should be used.
    Enables 4-bit if:
    1. CUDA is available (4-bit is GPU-only).
    2. Explicitly requested.
    3. Variant metadata indicates QLoRA or use_4bit.
    4. Running on CUDA with <= 12GB VRAM (e.g. RTX 4050 6GB) to prevent OOM on 3B+ models.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return False
    except Exception:
        return False

    if explicit_4bit is not None:
        return explicit_4bit

    if adapter_path:
        meta_file = os.path.join(adapter_path, "variant_meta.json")
        if os.path.exists(meta_file):
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    if "use_4bit" in meta:
                        return bool(meta["use_4bit"])
                    if meta.get("method") == "qlora":
                        return True
            except Exception:
                pass

        cfg_file = os.path.join(adapter_path, "adapter_config.json")
        if os.path.exists(cfg_file):
            try:
                with open(cfg_file, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    if cfg.get("quantization_config", {}).get("load_in_4bit", False):
                        return True
            except Exception:
                pass

    # Hardware check: if GPU VRAM <= 12GB and bitsandbytes is available, use 4-bit
    try:
        import torch
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            if vram_gb <= 12.0:
                try:
                    import bitsandbytes
                    return True
                except ImportError:
                    pass
    except Exception:
        pass

    return False


def _generate_text(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 2048,
    disable_thinking: bool = True,
) -> str:
    """Helper to run text generation using chat template format with optional thinking suppression."""
    import torch
    from services.ai_provider import strip_thinking

    messages = [{"role": "user", "content": prompt}]
    try:
        text_input = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=not disable_thinking,
        )
    except Exception:
        try:
            text_input = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            text_input = prompt

    inputs = tokenizer(text_input, return_tensors="pt", truncation=True, max_length=4096)
    target_device = getattr(model, "device", None) or ("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {k: v.to(target_device) for k, v in inputs.items()}
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    eos_token_id = tokenizer.eos_token_id

    gen_kwargs = {
        "max_new_tokens": max_tokens,
        "do_sample": False,
        "pad_token_id": pad_token_id,
    }
    if eos_token_id is not None:
        gen_kwargs["eos_token_id"] = eos_token_id

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            **gen_kwargs,
        )

    new_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    raw_text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    if disable_thinking:
        raw_text = strip_thinking(raw_text)

    return raw_text


def _load_base_model_raw(base_model: str, use_4bit: Optional[bool] = None) -> Tuple[Any, Any]:
    """Load standalone base model directly without caching."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from services.rom_training.rom_training_service import resolve_base_model_path

    resolved_model = resolve_base_model_path(base_model)
    is_local_dir = os.path.isdir(resolved_model)

    if not is_local_dir:
        from config import settings as app_settings
        if getattr(app_settings, "OFFLINE_MODE", False) or ":" in resolved_model or "\\" in resolved_model or "/" in resolved_model:
            raise FileNotFoundError(
                f"Base model directory not found at '{resolved_model}'. "
                "Please verify that the local base model folder ('nlp_engine') exists on this machine."
            )

    logger.info(f"[RomInference] Loading base model '{resolved_model}' (use_4bit={use_4bit})...")

    tokenizer_kwargs = {"trust_remote_code": True}
    if is_local_dir:
        tokenizer_kwargs["local_files_only"] = True

    tokenizer = AutoTokenizer.from_pretrained(resolved_model, **tokenizer_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_cuda = torch.cuda.is_available()
    should_4bit = use_cuda and _should_use_4bit(explicit_4bit=use_4bit)
    offload_dir = _get_offload_dir()

    if should_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
        )
        base = AutoModelForCausalLM.from_pretrained(
            resolved_model,
            quantization_config=bnb_config,
            torch_dtype=torch.float16 if use_cuda else torch.float32,
            device_map="auto",
            offload_folder=offload_dir,
            trust_remote_code=True,
            local_files_only=is_local_dir,
        )
    else:
        base = AutoModelForCausalLM.from_pretrained(
            resolved_model,
            torch_dtype=torch.float16 if use_cuda else torch.float32,
            device_map="auto" if use_cuda else None,
            offload_folder=offload_dir if use_cuda else None,
            trust_remote_code=True,
            local_files_only=is_local_dir,
        )
    base.eval()
    return base, tokenizer


def _load_variant_model_raw(adapter_path: str, base_model: str, use_4bit: Optional[bool] = None) -> Tuple[Any, Any]:
    """Load base model + LoRA/QLoRA adapter directly without caching."""
    from peft import PeftModel
    from services.rom_training.rom_training_service import resolve_adapter_path, resolve_base_model_path

    adapter_path = resolve_adapter_path(adapter_path)
    resolved_base_model = resolve_base_model_path(base_model)
    logger.info(f"[RomInference] Loading variant adapter from '{adapter_path}' onto '{resolved_base_model}'...")
    should_4bit = _should_use_4bit(adapter_path=adapter_path, explicit_4bit=use_4bit)
    base, tokenizer = _load_base_model_raw(resolved_base_model, use_4bit=should_4bit)
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return model, tokenizer


def run_sequential_test_inference(
    test_data: List[Dict[str, Any]],
    variant_id: str,
    adapter_path: str,
    base_model: str,
    examples: List[Dict[str, str]],
    max_tokens: int = 2048,
    disable_thinking: bool = True,
) -> List[Dict[str, Any]]:
    """
    Sequentially runs test inference adhering strictly to memory isolation:
    1. Phase 1: Load BASE model -> generate all 'before' outputs -> UNLOAD completely from VRAM.
    2. Phase 2: Load VARIANT adapter -> generate all 'after' outputs -> UNLOAD completely from VRAM.
    Ensures VRAM is cleanly cleared between steps and never tries to load concurrently.
    """
    from services.rom_training.rom_training_service import resolve_base_model_path, resolve_adapter_path
    base_model = resolve_base_model_path(base_model)
    adapter_path = resolve_adapter_path(adapter_path, variant_id)

    logger.info(
        f"[RomInference] Starting sequential test inference for {len(test_data)} samples "
        f"(variant {variant_id}, disable_thinking={disable_thinking})"
    )

    # Step 0: Ensure memory is completely scrubbed
    unload_variant(variant_id)
    clean_vram()

    before_outputs: List[str] = []
    after_outputs: List[str] = []

    # Phase 1: BASE model
    logger.info(f"[RomInference] === Phase 1: Loading Base Model ({base_model}) ===")
    base_model_obj = None
    base_tok = None
    try:
        base_model_obj, base_tok = _load_base_model_raw(base_model)
        for i, ex in enumerate(examples[:len(test_data)]):
            logger.info(f"[RomInference] Base model generating sample {i + 1}/{len(test_data)}")
            try:
                out = _generate_text(
                    base_model_obj, base_tok, ex["input"],
                    max_tokens=max_tokens, disable_thinking=disable_thinking,
                )
                before_outputs.append(out)
            except Exception as e:
                logger.error(f"[RomInference] Base model generation failed for sample {i + 1}: {e}")
                before_outputs.append(f"[Base generation error: {e}]")
    except Exception as e:
        logger.error(f"[RomInference] Failed to load base model for baseline inference: {e}", exc_info=True)
        before_outputs = [f"[Base model load error: {e}]"] * len(test_data)
    finally:
        # Guarantee base model is completely freed before variant is touched
        if base_model_obj is not None:
            del base_model_obj
        if base_tok is not None:
            del base_tok
        clean_vram()
        logger.info("[RomInference] === Phase 1 Complete: Base model fully unloaded from VRAM ===")

    # Phase 2: VARIANT model
    logger.info(f"[RomInference] === Phase 2: Loading Variant Adapter ({variant_id}) ===")
    variant_model_obj = None
    variant_tok = None
    try:
        variant_model_obj, variant_tok = _load_variant_model_raw(adapter_path, base_model)
        for i, ex in enumerate(examples[:len(test_data)]):
            logger.info(f"[RomInference] Variant model generating sample {i + 1}/{len(test_data)}")
            try:
                out = _generate_text(
                    variant_model_obj, variant_tok, ex["input"],
                    max_tokens=max_tokens, disable_thinking=disable_thinking,
                )
                after_outputs.append(out)
            except Exception as e:
                logger.error(f"[RomInference] Variant generation failed for sample {i + 1}: {e}")
                after_outputs.append(f"[Variant generation error: {e}]")
    except Exception as e:
        logger.error(f"[RomInference] Failed to load variant model for fine-tuned inference: {e}", exc_info=True)
        after_outputs = [f"[Variant load error: {e}]"] * len(test_data)
    finally:
        # Guarantee variant model is completely freed
        if variant_model_obj is not None:
            del variant_model_obj
        if variant_tok is not None:
            del variant_tok
        clean_vram()
        logger.info("[RomInference] === Phase 2 Complete: Variant model fully unloaded from VRAM ===")

    # Combine results
    results = []
    for item, ex, before_out, after_out in zip(test_data, examples[:len(test_data)], before_outputs, after_outputs):
        results.append({
            "agenda_title": item.get("agenda_title", ""),
            "recording_title": item.get("recording_title", ""),
            "before": before_out,
            "after": after_out,
            "manual": ex.get("output", ""),
        })

    logger.info(f"[RomInference] Completed sequential test inference for {len(results)} samples.")
    return results


def load_variant(
    variant_id: str,
    adapter_path: str,
    base_model: str,
    use_4bit: Optional[bool] = None,
) -> bool:
    """
    Load a LoRA/QLoRA adapter into the cache.
    Returns True on success, False on failure.
    """
    with _cache_lock:
        if variant_id in _adapter_cache:
            return True  # already loaded

    try:
        # Ensure previous models are cleared to guarantee VRAM availability
        clean_vram()
        model, tokenizer = _load_variant_model_raw(adapter_path, base_model, use_4bit=use_4bit)

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
        logger.error(f"[RomInference] Failed to load variant {variant_id}: {e}", exc_info=True)
        clean_vram()
        return False


def unload_variant(variant_id: str) -> None:
    """Evict a variant from the cache and free GPU memory."""
    with _cache_lock:
        entry = _adapter_cache.pop(variant_id, None)
    if entry:
        try:
            del entry["model"]
            del entry["tokenizer"]
            entry.clear()
        except Exception:
            pass
    clean_vram()
    logger.info(f"[RomInference] Variant {variant_id} unloaded.")


def infer_short_rom_with_variant(
    prompt: str,
    variant_id: str,
    adapter_path: str,
    base_model: str,
    max_tokens: int = 4096,
    use_adapter: bool = True,
    disable_thinking: bool = True,
) -> str:
    """
    Run inference using a loaded LoRA/QLoRA adapter.
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
            cached = _adapter_cache.get(variant_id)

    if cached is None:
        raise RuntimeError(f"Variant {variant_id} could not be retrieved from cache.")

    try:
        model = cached["model"]
        tokenizer = cached["tokenizer"]
        return _generate_text(
            model, tokenizer, prompt,
            max_tokens=max_tokens, disable_thinking=disable_thinking,
        )
    except Exception as e:
        logger.error(f"[RomInference] Inference failed for variant {variant_id}: {e}")
        raise


def infer_short_rom_base(
    prompt: str,
    base_model: str,
    variant_id: Optional[str] = None,
    adapter_path: Optional[str] = None,
    max_tokens: int = 4096,
    disable_thinking: bool = True,
) -> str:
    """
    Run inference directly on the local HuggingFace base model without adapter.
    Loads the base model cleanly and releases memory when done.
    """
    logger.info(f"[RomInference] Running standalone base model inference on {base_model}...")
    base = None
    tokenizer = None
    try:
        clean_vram()
        base, tokenizer = _load_base_model_raw(base_model)
        result = _generate_text(
            base, tokenizer, prompt,
            max_tokens=max_tokens, disable_thinking=disable_thinking,
        )
        return result
    finally:
        if base is not None:
            del base
        if tokenizer is not None:
            del tokenizer
        clean_vram()
