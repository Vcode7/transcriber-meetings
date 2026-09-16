"""
embedding_trainer.py — Background domain adaptation training engine for local embedding models.

Implements unsupervised contrastive domain adaptation (SimCSE) with in-batch negatives
and domain terminology context learning. Operates completely offline.
Runs in a background thread to prevent blocking FastAPI.
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
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


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
    cancel_event: threading.Event,
) -> None:
    """Worker function executed in background thread."""
    start_time = time.time()
    _append_log(job_id, f"Initializing domain adaptation for variant '{variant_name}'")
    _append_log(job_id, f"Base model: {base_model_name} | Corpus: {len(chunks)} chunks | Epochs: {epochs}")

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

        # Load weights in half-precision on GPU, float32 on CPU
        torch_dtype = torch.bfloat16 if (has_cuda and torch.cuda.is_bf16_supported()) else (torch.float16 if has_cuda else torch.float32)
        model = AutoModel.from_pretrained(
            base_model_path,
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
        model.to(device)
        model.train()

        _append_log(job_id, "Base model loaded into memory. Configuring SimCSE contrastive optimizer...")

        # Enable dropout for contrastive self-supervision if model supports it
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.15

        # 2. Setup Optimizer & Data Batches
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

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

        scaler = torch.amp.GradScaler('cuda') if (has_cuda and torch_dtype == torch.float16) else None

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

                # Tokenize batch twice (SimCSE self-contrastive pass with dropout)
                inputs1 = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                ).to(device)

                inputs2 = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                ).to(device)

                optimizer.zero_grad()

                if scaler:
                    with torch.amp.autocast('cuda', dtype=torch.float16):
                        out1 = model(**inputs1)
                        out2 = model(**inputs2)
                        emb1 = _mean_pooling(out1, inputs1["attention_mask"])
                        emb2 = _mean_pooling(out2, inputs2["attention_mask"])
                        # L2 normalize
                        emb1 = F.normalize(emb1, p=2, dim=1)
                        emb2 = F.normalize(emb2, p=2, dim=1)

                        # Cosine similarity matrix (B, B)
                        sim_matrix = torch.mm(emb1, emb2.t()) / temperature
                        labels = torch.arange(emb1.size(0), device=device)
                        loss = F.cross_entropy(sim_matrix, labels)

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    out1 = model(**inputs1)
                    out2 = model(**inputs2)
                    emb1 = _mean_pooling(out1, inputs1["attention_mask"])
                    emb2 = _mean_pooling(out2, inputs2["attention_mask"])
                    emb1 = F.normalize(emb1, p=2, dim=1)
                    emb2 = F.normalize(emb2, p=2, dim=1)

                    sim_matrix = torch.mm(emb1, emb2.t()) / temperature
                    labels = torch.arange(emb1.size(0), device=device)
                    loss = F.cross_entropy(sim_matrix, labels)

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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

        # 3. Save to Staging Directory
        staging_model_dir = STAGING_DIR / f"staged_{job_id}"
        staging_model_dir.mkdir(parents=True, exist_ok=True)
        _append_log(job_id, f"Domain adaptation complete. Saving model weights to staging: {staging_model_dir}")

        model.eval()
        model.save_pretrained(str(staging_model_dir))
        tokenizer.save_pretrained(str(staging_model_dir))

        # Copy sentence_transformers configs from base model if they exist
        base_path_obj = Path(base_model_path)
        for extra_item in ("modules.json", "config_sentence_transformers.json", "1_Pooling"):
            src = base_path_obj / extra_item
            dst = staging_model_dir / extra_item
            if src.exists():
                if src.is_dir():
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dst)

        final_loss = round(sum(recent_losses[-20:]) / max(1, len(recent_losses[-20:])), 4) if recent_losses else 0.0
        duration = round(time.time() - start_time, 1)
        _append_log(job_id, f"Model successfully saved to staging. Final average loss: {final_loss:.4f} (Elapsed: {duration}s)")

        with _jobs_lock:
            _active_jobs[job_id]["status"] = "completed"
            _active_jobs[job_id]["progress_percent"] = 100.0
            _active_jobs[job_id]["final_loss"] = final_loss
            _active_jobs[job_id]["duration_seconds"] = duration
            _active_jobs[job_id]["staged_model_path"] = str(staging_model_dir)

    except Exception as exc:
        logger.exception(f"[EmbeddingTrainer] Training error in job {job_id}: {exc}")
        _append_log(job_id, f"ERROR: Training failed: {exc}")
        with _jobs_lock:
            _active_jobs[job_id]["status"] = "failed"
            _active_jobs[job_id]["error"] = str(exc)

    finally:
        # Free VRAM and memory
        try:
            del model
            del optimizer
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
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
        "final_loss": None,
        "progress_percent": 0.0,
        "elapsed_seconds": 0,
        "eta_seconds": 0,
        "loss_history": [],
        "logs": [],
        "error": None,
        "staged_model_path": None,
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
            cancel_event,
        ),
        daemon=True,
    )
    thread.start()

    return job_id
