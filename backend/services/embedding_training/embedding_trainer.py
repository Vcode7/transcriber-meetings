"""
embedding_trainer.py — Background domain adaptation training engine for local embedding models.

Implements unsupervised contrastive domain adaptation (SimCSE) with in-batch negatives
and domain terminology context learning. Operates completely offline.
Runs in a background thread to prevent blocking FastAPI.

Training setup
--------------
* SimCSE needs two *different* stochastic views of every passage. Decoder-style
  embedding models (Qwen3-Embedding) ship with no nn.Dropout modules and
  attention_dropout=0.0, so dropout is enabled explicitly (attention dropout +
  hidden-state dropout on every transformer layer) for the duration of training.
* Trainable weights are kept in float32 ("master weights"); forward/backward run
  under autocast (bf16, or fp16 + GradScaler). Optimizer updates of ~lr=2e-5 are
  below bf16/fp16 resolution, so training half-precision weights directly would
  silently discard most updates.
* Only the top ``trainable_layers`` transformer layers (+ final norm) are trained;
  token embeddings and lower layers stay frozen in half precision. This keeps
  fp32 AdamW within consumer-GPU VRAM (0.6B model, 8 layers ≈ 3 GB).
* After saving, the staged model is reloaded through the same loader the RAG
  pipeline uses and compared with the base model on corpus passages, proving that
  the variant loads and actually produces different embeddings.
"""
from __future__ import annotations

import gc
import json
import logging
import os
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from config import settings

logger = logging.getLogger(__name__)

# Active jobs registry
_active_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = threading.Lock()
_cancel_events: Dict[str, threading.Event] = {}

# Staging directory for completed trained models pending final variant save
STAGING_DIR = Path(settings.RUNTIME_DIR) / "embedding_training_staging"
STAGING_DIR.mkdir(parents=True, exist_ok=True)

CONTRASTIVE_DROPOUT = 0.1
VERIFICATION_PROBES = 16


def get_job_status(job_id: str) -> Optional[Dict[str, Any]]:
    """Return status and progress for a training job."""
    with _jobs_lock:
        job = _active_jobs.get(job_id)
        if not job:
            return None
        return dict(job)


def cancel_training_job(job_id: str) -> bool:
    """Request cancellation of an active training job."""
    with _jobs_lock:
        event = _cancel_events.get(job_id)
        if event:
            event.set()
            if job_id in _active_jobs:
                _active_jobs[job_id]["status"] = "cancelling"
                _append_log(job_id, "Cancellation requested by user.")
            return True
        return False


def _append_log(job_id: str, message: str) -> None:
    now_str = datetime.now().strftime("%H:%M:%S")
    entry = f"[{now_str}] {message}"
    logger.info(f"[EmbeddingTrainer:{job_id[:8]}] {message}")
    with _jobs_lock:
        if job_id in _active_jobs:
            _active_jobs[job_id].setdefault("logs", []).append(entry)


def _mean_pooling(model_output, attention_mask):
    """Mean pooling over token embeddings taking attention mask into account."""
    import torch
    token_embeddings = model_output[0]  # First element contains hidden state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(token_embeddings.dtype)
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def _transformer_layers(model):
    """Return the ModuleList of transformer blocks (Qwen/Llama, BERT, GPT-2 layouts), or None."""
    import torch
    for path in ("layers", "model.layers", "encoder.layer", "transformer.h", "h"):
        obj = model
        for attr in path.split("."):
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if isinstance(obj, torch.nn.ModuleList) and len(obj) > 0:
            return obj
    return None


def _final_norm(model):
    for attr in ("norm", "final_layernorm", "ln_f"):
        module = getattr(model, attr, None)
        if module is not None and any(True for _ in module.parameters()):
            return module
    return None


def _select_trainable_modules(model, trainable_layers: int):
    """
    Freeze the model, then unfreeze the top ``trainable_layers`` transformer layers
    and the final norm (0 = all layers). Token embeddings always stay frozen.
    Returns (trainable_modules, description).
    """
    for p in model.parameters():
        p.requires_grad_(False)

    layers = _transformer_layers(model)
    if layers is None:
        # Unknown architecture: train everything except the input embeddings.
        embeddings = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
        embedding_params = {id(p) for p in embeddings.parameters()} if embeddings is not None else set()
        for p in model.parameters():
            if id(p) not in embedding_params:
                p.requires_grad_(True)
        return [model], "all parameters except token embeddings (unknown layer layout)"

    n_layers = len(layers)
    n_train = n_layers if trainable_layers <= 0 else min(trainable_layers, n_layers)
    modules = list(layers[n_layers - n_train:])
    norm = _final_norm(model)
    if norm is not None:
        modules.append(norm)
    for module in modules:
        for p in module.parameters():
            p.requires_grad_(True)
    return modules, f"top {n_train}/{n_layers} transformer layers + final norm"


def _enable_contrastive_dropout(model, p: float) -> List[Any]:
    """
    Make the forward pass stochastic so the two SimCSE views of a passage differ.
    Returns forward-hook handles; call _disable_contrastive_dropout() to restore.
    """
    import torch
    import torch.nn.functional as F

    dropout_modules = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = p
            dropout_modules += 1
        # Qwen/Llama attention modules read this float attribute at forward time
        if isinstance(getattr(module, "attention_dropout", None), float):
            module._orig_attention_dropout = module.attention_dropout
            module.attention_dropout = p

    handles = []
    if dropout_modules == 0:
        # No dropout layers at all (decoder embedding models): add hidden-state dropout
        layers = _transformer_layers(model)

        def _hidden_dropout(module, inputs, output):
            if not module.training:
                return output
            if isinstance(output, tuple):
                return (F.dropout(output[0], p=p, training=True),) + tuple(output[1:])
            if isinstance(output, torch.Tensor):
                return F.dropout(output, p=p, training=True)
            return output

        for layer in (layers or []):
            handles.append(layer.register_forward_hook(_hidden_dropout))
    return handles


def _disable_contrastive_dropout(model, handles: List[Any]) -> None:
    for handle in handles:
        handle.remove()
    for module in model.modules():
        if hasattr(module, "_orig_attention_dropout"):
            module.attention_dropout = module._orig_attention_dropout
            del module._orig_attention_dropout


def _verify_variant(
    job_id: str,
    base_model_path: str,
    staged_model_dir: Path,
    chunks: List[str],
) -> Dict[str, Any]:
    """
    Reload base and trained models through the RAG embedding loader and compare their
    embeddings of corpus passages. Proves the saved variant loads exactly like it will
    in production and that training actually changed the embedding space.
    """
    from services.text_embedding_service import TextEmbeddingService

    if not chunks:
        return {"status": "skipped", "reason": "empty corpus"}
    step = max(1, len(chunks) // VERIFICATION_PROBES)
    probes = [c for c in chunks[::step] if c and c.strip()][:VERIFICATION_PROBES]

    vectors = {}
    for label, path in (("base", base_model_path), ("variant", str(staged_model_dir))):
        svc = TextEmbeddingService(model_name=f"{label}-verification", model_dir=path)
        try:
            vectors[label] = svc.encode_batch(probes, batch_size=8)
            dim = svc.embedding_dim()
        finally:
            svc.unload()

    base_vecs, variant_vecs = vectors["base"], vectors["variant"]
    cosines = np.sum(base_vecs * variant_vecs, axis=1)
    mean_cos = float(np.mean(cosines))

    # Spread of the variant's space over the corpus (lower mean pairwise similarity =
    # passages are better separated, the goal of the contrastive objective).
    def _mean_pairwise(v: np.ndarray) -> float:
        if len(v) < 2:
            return 1.0
        sims = v @ v.T
        n = len(v)
        return float((sims.sum() - np.trace(sims)) / (n * (n - 1)))

    result = {
        "status": "ok",
        "variant_loads": True,
        "probe_passages": len(probes),
        "embedding_dim": int(dim),
        "mean_cosine_to_base": round(mean_cos, 6),
        "min_cosine_to_base": round(float(np.min(cosines)), 6),
        "embedding_shift": round(1.0 - mean_cos, 6),
        "differs_from_base": bool(mean_cos < 0.99999),
        "base_mean_pairwise_similarity": round(_mean_pairwise(base_vecs), 4),
        "variant_mean_pairwise_similarity": round(_mean_pairwise(variant_vecs), 4),
    }
    _append_log(
        job_id,
        f"Verification: variant reloads via the RAG loader (dim={dim}); mean cosine to base "
        f"{result['mean_cosine_to_base']:.4f} (shift {result['embedding_shift']:.4f}); "
        f"corpus pairwise similarity {result['base_mean_pairwise_similarity']} → "
        f"{result['variant_mean_pairwise_similarity']}."
    )
    if not result["differs_from_base"]:
        _append_log(job_id, "WARNING: the trained variant produces (near-)identical embeddings to the base model.")
    return result


def _run_training_thread(
    job_id: str,
    base_model_name: str,
    base_model_path: str,
    variant_name: str,
    chunks: List[str],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    temperature: float,
    trainable_layers: int,
    cancel_event: threading.Event,
) -> None:
    """Worker function executed in background thread."""
    start_time = time.time()
    _append_log(job_id, f"Initializing domain adaptation for variant '{variant_name}'")
    _append_log(job_id, f"Base model: {base_model_name} | Corpus: {len(chunks)} chunks | Epochs: {epochs}")

    torch = None
    model = None
    optimizer = None
    try:
        import torch
        import torch.nn.functional as F
        from transformers import AutoTokenizer, AutoModel

        has_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if has_cuda else "cpu")
        _append_log(job_id, f"Target device: {str(device).upper()} (CUDA: {has_cuda})")

        # Set offline flags
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        # 1. Load Tokenizer & Model
        _append_log(job_id, f"Loading base model from {base_model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Unload any existing embedder from RAM/VRAM to avoid memory contention
        try:
            from services.text_embedding_service import unload_text_embedder
            unload_text_embedder()
        except Exception:
            pass

        # Frozen weights are stored in half precision on GPU; trainable weights are
        # upcast to float32 below. Compute runs under autocast.
        if has_cuda:
            amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            amp_dtype = None
        storage_dtype = amp_dtype or torch.float32
        model = AutoModel.from_pretrained(
            base_model_path,
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=storage_dtype,
        )
        model.to(device)

        trainable_modules, trainable_desc = _select_trainable_modules(model, trainable_layers)
        for module in trainable_modules:
            module.float()
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        n_trainable = sum(p.numel() for p in trainable_params)
        n_total = sum(p.numel() for p in model.parameters())
        model.train()

        precision_desc = (
            f"fp32 master weights + {str(amp_dtype).replace('torch.', '')} autocast"
            if amp_dtype is not None else "fp32"
        )
        _append_log(
            job_id,
            f"Base model loaded. Training {trainable_desc}: {n_trainable / 1e6:.1f}M of "
            f"{n_total / 1e6:.1f}M parameters ({precision_desc}).",
        )

        # SimCSE: two dropout-noised views of every passage form the positive pair.
        dropout_handles = _enable_contrastive_dropout(model, CONTRASTIVE_DROPOUT)

        # 2. Setup Optimizer & Data Batches
        optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=0.01)

        # Build batches from chunks
        # Shuffle chunks per epoch
        total_chunks = len(chunks)
        effective_batch_size = max(2, min(batch_size, total_chunks))
        steps_per_epoch = max(1, total_chunks // effective_batch_size)
        total_steps = steps_per_epoch * epochs

        with _jobs_lock:
            _active_jobs[job_id]["status"] = "training"
            _active_jobs[job_id]["total_steps"] = total_steps
            _active_jobs[job_id]["total_epochs"] = epochs

        _append_log(job_id, f"Starting training loop: {total_steps} total steps across {epochs} epochs.")

        global_step = 0
        loss_history = []
        recent_losses = []

        scaler = torch.amp.GradScaler("cuda") if amp_dtype == torch.float16 else None

        def _contrastive_loss(batch_texts: List[str]):
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                out1 = model(**inputs)
                out2 = model(**inputs)
                emb1 = _mean_pooling(out1, inputs["attention_mask"])
                emb2 = _mean_pooling(out2, inputs["attention_mask"])
            # Similarity + loss in float32 for numerical stability
            emb1 = F.normalize(emb1.float(), p=2, dim=1)
            emb2 = F.normalize(emb2.float(), p=2, dim=1)
            sim_matrix = torch.mm(emb1, emb2.t()) / temperature
            labels = torch.arange(emb1.size(0), device=device)
            return F.cross_entropy(sim_matrix, labels)

        for epoch in range(1, epochs + 1):
            if cancel_event.is_set():
                break

            # Shuffle chunks
            shuffled_indices = np.random.permutation(total_chunks)
            shuffled_chunks = [chunks[i] for i in shuffled_indices]

            epoch_loss = 0.0
            epoch_steps = 0

            for step_idx in range(steps_per_epoch):
                if cancel_event.is_set():
                    break

                batch_texts = shuffled_chunks[step_idx * effective_batch_size : (step_idx + 1) * effective_batch_size]
                if len(batch_texts) < 2:
                    continue

                optimizer.zero_grad(set_to_none=True)
                loss = _contrastive_loss(batch_texts)

                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                    optimizer.step()

                loss_val = float(loss.item())
                epoch_loss += loss_val
                epoch_steps += 1
                global_step += 1
                recent_losses.append(loss_val)

                # Update progress
                elapsed = time.time() - start_time
                steps_remaining = total_steps - global_step
                avg_step_time = elapsed / max(1, global_step)
                eta = int(steps_remaining * avg_step_time)
                progress_pct = round((global_step / max(1, total_steps)) * 100, 1)

                loss_point = {"step": global_step, "epoch": epoch, "loss": round(loss_val, 4)}
                loss_history.append(loss_point)

                with _jobs_lock:
                    _active_jobs[job_id]["current_step"] = global_step
                    _active_jobs[job_id]["current_epoch"] = epoch
                    _active_jobs[job_id]["current_loss"] = round(loss_val, 4)
                    _active_jobs[job_id]["progress_percent"] = progress_pct
                    _active_jobs[job_id]["elapsed_seconds"] = int(elapsed)
                    _active_jobs[job_id]["eta_seconds"] = eta
                    _active_jobs[job_id]["loss_history"] = loss_history[-60:]  # Keep last 60 for chart

                if global_step % max(1, steps_per_epoch // 2) == 0 or global_step == total_steps:
                    avg_recent = sum(recent_losses[-10:]) / max(1, len(recent_losses[-10:]))
                    _append_log(
                        job_id,
                        f"Epoch {epoch}/{epochs} | Step {global_step}/{total_steps} ({progress_pct}%) | "
                        f"Loss: {loss_val:.4f} (avg: {avg_recent:.4f}) | ETA: {eta}s"
                    )

            if cancel_event.is_set():
                break

        # Check if cancelled
        if cancel_event.is_set():
            _append_log(job_id, "Training was cancelled by user. Cleaning up...")
            with _jobs_lock:
                _active_jobs[job_id]["status"] = "cancelled"
            return

        if global_step == 0:
            raise RuntimeError("No training steps were executed (corpus too small for the batch size).")

        # 3. Save to Staging Directory
        _disable_contrastive_dropout(model, dropout_handles)
        model.eval()
        staging_model_dir = STAGING_DIR / f"staged_{job_id}"
        staging_model_dir.mkdir(parents=True, exist_ok=True)
        _append_log(job_id, f"Domain adaptation complete. Saving model weights to staging: {staging_model_dir}")

        # Trained layers are saved in float32 so small learned updates are not rounded
        # away; frozen layers keep the base model's precision.
        model.save_pretrained(str(staging_model_dir))

        # The tokenizer is not trained: copy the base model's tokenizer files and
        # sentence-transformers configs verbatim (re-serializing the tokenizer adds
        # files and trips a spurious transformers regex warning on reload).
        weight_suffixes = (".safetensors", ".bin", ".pt", ".pth", ".index.json")
        for src in Path(base_model_path).iterdir():
            dst = staging_model_dir / src.name
            if dst.exists() or src.name == "config.json" or src.name.endswith(weight_suffixes):
                continue
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)

        final_loss = round(sum(recent_losses[-20:]) / max(1, len(recent_losses[-20:])), 4) if recent_losses else 0.0
        initial_loss = round(sum(recent_losses[:10]) / max(1, len(recent_losses[:10])), 4) if recent_losses else 0.0
        duration = round(time.time() - start_time, 1)
        _append_log(job_id, f"Model successfully saved to staging. Loss {initial_loss:.4f} → {final_loss:.4f} (Elapsed: {duration}s)")

        # Free training memory before verification loads two models
        del optimizer
        optimizer = None
        model = None
        gc.collect()
        if has_cuda:
            torch.cuda.empty_cache()

        # 4. Verify the saved variant through the production loader
        with _jobs_lock:
            _active_jobs[job_id]["status"] = "verifying"
        _append_log(job_id, "Verifying the saved variant (reloading base + variant through the RAG embedding loader)...")
        try:
            verification = _verify_variant(job_id, base_model_path, staging_model_dir, chunks)
        except Exception as ver_err:
            # A variant that cannot be loaded must never be saved/activated.
            raise RuntimeError(f"Trained variant failed verification (could not be reloaded): {ver_err}") from ver_err

        training_info = {
            "base_model": base_model_name,
            "epochs": epochs,
            "steps": global_step,
            "batch_size": effective_batch_size,
            "learning_rate": learning_rate,
            "temperature": temperature,
            "trainable": trainable_desc,
            "trainable_params": n_trainable,
            "precision": precision_desc,
            "contrastive_dropout": CONTRASTIVE_DROPOUT,
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "duration_seconds": duration,
        }
        with open(staging_model_dir / "training_report.json", "w", encoding="utf-8") as f:
            json.dump({"training": training_info, "verification": verification}, f, indent=2)

        with _jobs_lock:
            _active_jobs[job_id]["status"] = "completed"
            _active_jobs[job_id]["progress_percent"] = 100.0
            _active_jobs[job_id]["final_loss"] = final_loss
            _active_jobs[job_id]["initial_loss"] = initial_loss
            _active_jobs[job_id]["duration_seconds"] = duration
            _active_jobs[job_id]["staged_model_path"] = str(staging_model_dir)
            _active_jobs[job_id]["verification"] = verification

    except Exception as exc:
        logger.exception(f"[EmbeddingTrainer] Training error in job {job_id}: {exc}")
        _append_log(job_id, f"ERROR: Training failed: {exc}")
        with _jobs_lock:
            _active_jobs[job_id]["status"] = "failed"
            _active_jobs[job_id]["error"] = str(exc)

    finally:
        # Free VRAM and memory
        model = None
        optimizer = None
        gc.collect()
        try:
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        with _jobs_lock:
            _cancel_events.pop(job_id, None)
        _append_log(job_id, "Memory garbage collected and VRAM released.")


def start_domain_training(
    base_model_name: str,
    base_model_path: str,
    variant_name: str,
    chunks: List[str],
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    temperature: float = 0.05,
    trainable_layers: int = 8,
) -> str:
    """
    Launch an unsupervised domain adaptation training job in a background thread.
    Returns job_id.
    """
    import uuid
    job_id = str(uuid.uuid4())
    cancel_event = threading.Event()

    job_record = {
        "job_id": job_id,
        "variant_name": variant_name,
        "base_model_name": base_model_name,
        "base_model_path": base_model_path,
        "status": "initializing",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_chunks": len(chunks),
        "epochs": epochs,
        "current_epoch": 0,
        "total_epochs": epochs,
        "current_step": 0,
        "total_steps": 0,
        "current_loss": None,
        "initial_loss": None,
        "final_loss": None,
        "progress_percent": 0.0,
        "elapsed_seconds": 0,
        "eta_seconds": 0,
        "loss_history": [],
        "logs": [],
        "error": None,
        "staged_model_path": None,
        "trainable_layers": trainable_layers,
        "verification": None,
    }

    with _jobs_lock:
        _active_jobs[job_id] = job_record
        _cancel_events[job_id] = cancel_event

    thread = threading.Thread(
        target=_run_training_thread,
        args=(
            job_id,
            base_model_name,
            base_model_path,
            variant_name,
            chunks,
            epochs,
            batch_size,
            learning_rate,
            temperature,
            trainable_layers,
            cancel_event,
        ),
        daemon=True,
    )
    thread.start()

    return job_id
