"""
variant_manager.py — Lifecycle management, persistence, selection, and deletion of embedding model variants.

Ensures the base model is never overwritten and isolates each variant separately.
Allows activating a variant for the offline RAG pipeline.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import text

from config import settings, BASE_DIR, RUNTIME_DIR
from database import get_db_context, from_json, to_json

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "Qwen3-Embedding-0.6B"

_VARIANT_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_variant_name(variant_name: str) -> str:
    """
    Turn a user-supplied variant name into a safe single directory name.
    Spaces become hyphens; anything outside [A-Za-z0-9._-] is dropped, so the
    name can never escape the embeddings directory (no separators, no '..').
    """
    clean = _VARIANT_NAME_RE.sub("", (variant_name or "").strip().replace(" ", "-"))
    clean = clean.strip(".-_")
    if not clean:
        raise ValueError("Variant name must contain letters or digits.")
    if len(clean) > 100:
        raise ValueError("Variant name is too long (max 100 characters).")
    return clean


def _resolve_staged_path(staged_model_path: str) -> Path:
    """Only accept staged model directories produced by the trainer (inside its staging dir)."""
    from services.embedding_training.embedding_trainer import STAGING_DIR

    staging_root = STAGING_DIR.resolve()
    staged = Path(staged_model_path).resolve()
    if staged == staging_root or staging_root not in staged.parents:
        raise ValueError("Staged model path is not a trained model produced by the embedding trainer.")
    if not staged.is_dir():
        raise FileNotFoundError(f"Staged model files not found at {staged_model_path}")
    return staged


def _read_variant_meta(model_dir: Path) -> Dict[str, Any]:
    try:
        with open(model_dir / "variant_meta.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _get_embeddings_base_dir() -> Path:
    """Return directory where embedding models and variants reside."""
    dev_path = BASE_DIR.parent / "Application" / "runtime" / "embeddings"
    if dev_path.is_dir() and not getattr(os.sys, "frozen", False):
        return dev_path
    runtime_path = RUNTIME_DIR / "embeddings"
    runtime_path.mkdir(parents=True, exist_ok=True)
    return runtime_path


def discover_base_models() -> List[Dict[str, Any]]:
    """Scan local directories for available base embedding models."""
    base_dir = _get_embeddings_base_dir()
    models: List[Dict[str, Any]] = []

    if not base_dir.exists():
        return models

    for item in base_dir.iterdir():
        if not item.is_dir():
            continue
        # Skip staging or hidden folders
        if item.name.startswith((".", "_")) or item.name == "variants":
            continue

        config_path = item / "config.json"
        safetensors = list(item.glob("*.safetensors")) or list(item.glob("*.bin"))

        if config_path.exists() or safetensors:
            # Check if it has variant_meta.json — if so, it is a trained variant, not base
            if (item / "variant_meta.json").exists():
                continue

            dim = 1024
            arch = "Transformer"
            if config_path.exists():
                try:
                    with open(config_path, "r", encoding="utf-8") as f:
                        cfg = json.load(f)
                        dim = cfg.get("hidden_size", cfg.get("dim", 1024))
                        archs = cfg.get("architectures", [])
                        arch = archs[0] if archs else cfg.get("model_type", "Transformer")
                except Exception:
                    pass

            # Calculate total size in MB
            total_bytes = sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
            size_mb = round(total_bytes / (1024 * 1024), 1)

            models.append({
                "id": item.name,
                "name": item.name,
                "path": str(item.resolve()),
                "dimension": dim,
                "architecture": arch,
                "size_mb": size_mb,
                "is_default": item.name == "Qwen3-Embedding-0.6B" or item.name == settings.EMBEDDING_MODEL,
                "status": "ready",
            })

    return models


async def save_trained_variant(
    user_id: str,
    variant_name: str,
    base_model_name: str,
    staged_model_path: str,
    document_count: int,
    document_names: List[str],
    chunk_count: int,
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Save a trained model variant from staging to the persistent embeddings directory,
    ensuring base models are protected and variant metadata is recorded.
    """
    from services.text_embedding_service import validate_embedding_model_dir

    # Sanitize variant name (it becomes a directory name and the active model name)
    clean_name = sanitize_variant_name(variant_name)

    # Guard: prevent overwriting base model
    base_models = discover_base_models()
    base_names = [m["name"].lower() for m in base_models]
    if clean_name.lower() in base_names:
        raise ValueError(f"Cannot overwrite base model '{clean_name}'. Please specify a distinct variant name.")

    embeddings_root = _get_embeddings_base_dir()
    target_dir = embeddings_root / clean_name

    # If target dir already exists, raise error or handle
    if target_dir.exists():
        raise ValueError(f"A variant or model named '{clean_name}' already exists.")

    staged_path = _resolve_staged_path(staged_model_path)
    staged_error = validate_embedding_model_dir(staged_path)
    if staged_error:
        raise FileNotFoundError(f"Staged model is incomplete: {staged_error}")

    # Training report written by the trainer (loss + post-training verification)
    training_report: Dict[str, Any] = {}
    try:
        with open(staged_path / "training_report.json", "r", encoding="utf-8") as f:
            training_report = json.load(f)
    except Exception:
        pass
    metrics = {**(metrics or {})}
    if training_report.get("verification"):
        metrics["verification"] = training_report["verification"]
    if training_report.get("training"):
        metrics.setdefault("training", training_report["training"])

    # Move the staged model into a hidden partial directory first (an instant rename when
    # staging and embeddings share a drive; copy+delete otherwise), write the metadata,
    # then rename it into place. A failure (e.g. disk full) never leaves a half-written
    # variant directory that could be mistaken for a loadable model.
    partial_dir = embeddings_root / f".{clean_name}.partial-{uuid.uuid4().hex[:8]}"
    try:
        shutil.move(str(staged_path), str(partial_dir))
    except Exception:
        shutil.rmtree(partial_dir, ignore_errors=True)
        raise

    # Calculate total size
    total_bytes = sum(f.stat().st_size for f in partial_dir.rglob("*") if f.is_file())

    embedding_dim = None
    try:
        with open(partial_dir / "config.json", "r", encoding="utf-8") as f:
            embedding_dim = json.load(f).get("hidden_size")
    except Exception:
        pass

    # Write variant_meta.json
    meta = {
        "variant_name": clean_name,
        "base_model": base_model_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "document_count": document_count,
        "document_names": document_names,
        "chunk_count": chunk_count,
        "metrics": metrics,
        "size_bytes": total_bytes,
        "embedding_dim": embedding_dim,
    }

    variant_id = str(uuid.uuid4())
    now_iso = meta["created_at"]

    renamed = False
    try:
        with open(partial_dir / "variant_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        partial_dir.rename(target_dir)
        renamed = True

        # Insert into SQLite embedding_variants
        async with get_db_context() as db:
            await db.execute(
                text("""
                    INSERT INTO embedding_variants (
                        id, user_id, name, base_model, created_at,
                        document_count, document_names, chunk_count,
                        status, model_path, size_bytes, metrics, is_active
                    ) VALUES (
                        :id, :user_id, :name, :base_model, :created_at,
                        :document_count, :document_names, :chunk_count,
                        :status, :model_path, :size_bytes, :metrics, :is_active
                    )
                """),
                {
                    "id": variant_id,
                    "user_id": user_id,
                    "name": clean_name,
                    "base_model": base_model_name,
                    "created_at": now_iso,
                    "document_count": document_count,
                    "document_names": json.dumps(document_names),
                    "chunk_count": chunk_count,
                    "status": "completed",
                    "model_path": str(target_dir.resolve()),
                    "size_bytes": total_bytes,
                    "metrics": json.dumps(metrics),
                    "is_active": 0,
                },
            )
            await db.commit()
    except Exception:
        # Roll back: return the model to staging so saving can be retried without retraining.
        current = target_dir if renamed else partial_dir
        try:
            (current / "variant_meta.json").unlink(missing_ok=True)
            shutil.move(str(current), str(staged_path))
        except Exception:
            shutil.rmtree(current, ignore_errors=True)
        raise

    logger.info(f"[VariantManager] Saved new embedding variant '{clean_name}' to {target_dir}")

    return {
        "id": variant_id,
        "name": clean_name,
        "base_model": base_model_name,
        "created_at": now_iso,
        "document_count": document_count,
        "chunk_count": chunk_count,
        "status": "completed",
        "model_path": str(target_dir.resolve()),
        "size_bytes": total_bytes,
        "metrics": metrics,
        "is_active": False,
    }


async def list_variants(user_id: str) -> List[Dict[str, Any]]:
    """List all saved embedding model variants."""
    variants = []
    async with get_db_context() as db:
        res = await db.execute(
            text("""
                SELECT id, name, base_model, created_at, document_count,
                       document_names, chunk_count, status, model_path,
                       size_bytes, metrics, is_active
                FROM embedding_variants
                ORDER BY created_at DESC
            """)
        )
        rows = res.fetchall()

        # "Active" means: the model the running application actually embeds with.
        # (The is_active column is only a persisted mirror and may lag behind.)
        active_model_name = settings.EMBEDDING_MODEL

        for r in rows:
            v_name = r[1]
            is_currently_active = v_name == active_model_name
            variants.append({
                "id": r[0],
                "name": v_name,
                "base_model": r[2],
                "created_at": r[3],
                "document_count": r[4],
                "document_names": from_json(r[5], []),
                "chunk_count": r[6],
                "status": r[7],
                "model_path": r[8],
                "size_bytes": r[9],
                "size_mb": round(r[9] / (1024 * 1024), 1) if r[9] else 0.0,
                "metrics": from_json(r[10], {}),
                "is_active": is_currently_active,
                "exists_on_disk": Path(r[8]).is_dir() if r[8] else False,
            })

    return variants


async def _persist_active_embedding_model(db, user_id: str, model_name: str) -> None:
    """Persist the active model in user_settings and mirror it in embedding_variants.is_active."""
    await db.execute(
        text("""
            UPDATE user_settings
            SET embedding_model = :model_name, updated_at = :now
            WHERE user_id = :uid
        """),
        {"model_name": model_name, "uid": user_id, "now": datetime.now(timezone.utc).isoformat()},
    )
    await db.execute(
        text("UPDATE embedding_variants SET is_active = CASE WHEN name = :name THEN 1 ELSE 0 END"),
        {"name": model_name},
    )


async def sync_variant_active_flags(model_name: str) -> None:
    """Mirror the active embedding model into embedding_variants.is_active."""
    async with get_db_context() as db:
        await db.execute(
            text("UPDATE embedding_variants SET is_active = CASE WHEN name = :name THEN 1 ELSE 0 END"),
            {"name": model_name},
        )
        await db.commit()


async def restore_active_embedding_model() -> str:
    """
    Startup: re-apply the persisted embedding model selection (base model or trained
    variant) so the selected variant survives an application restart.

    The embedding model is process-wide, so the most recently updated user_settings row
    wins. If the persisted model is missing on disk, the default base model is kept and
    a warning is logged (the persisted value is left untouched so it can be restored
    once the files are back).
    """
    from services.text_embedding_service import (
        get_active_embedding_model,
        set_active_embedding_model,
        validate_embedding_model_dir,
    )

    persisted: Optional[str] = None
    try:
        async with get_db_context() as db:
            res = await db.execute(
                text("""
                    SELECT embedding_model FROM user_settings
                    WHERE embedding_model IS NOT NULL AND embedding_model != ''
                    ORDER BY updated_at DESC
                    LIMIT 1
                """)
            )
            row = res.fetchone()
            persisted = row[0] if row else None
    except Exception as e:
        logger.warning(f"[VariantManager] Could not read persisted embedding model: {e}")

    if persisted:
        error = validate_embedding_model_dir(settings.embedding_model_dir(persisted))
        if error:
            logger.warning(
                f"[VariantManager] Persisted embedding model '{persisted}' is not available ({error}). "
                f"Keeping '{get_active_embedding_model()}'."
            )
        else:
            set_active_embedding_model(persisted, source="startup restore")

    active = get_active_embedding_model()
    try:
        await sync_variant_active_flags(active)
    except Exception as e:
        logger.warning(f"[VariantManager] Could not sync variant active flags: {e}")
    logger.info(f"[VariantManager] Active embedding model at startup: {active}")
    return active


async def select_variant(user_id: str, variant_id: str) -> Dict[str, Any]:
    """
    Select an embedding variant as the active model for every embedding-dependent
    feature (RAG retrieval, indexing, Stage 2/3 similarity, AI chat).
    Persists the choice (survives restarts) and swaps the in-memory model.

    Raises FileNotFoundError (unknown variant / files missing) and
    EmbeddingModelLockedError (ChromaDB reindex running).
    """
    from services.text_embedding_service import set_active_embedding_model

    async with get_db_context() as db:
        res = await db.execute(
            text("SELECT id, name, model_path FROM embedding_variants WHERE id = :vid"),
            {"vid": variant_id}
        )
        row = res.fetchone()
        if not row:
            raise FileNotFoundError(f"Variant with ID '{variant_id}' not found.")

        variant_name = row[1]

        # Validates the variant files and swaps the runtime model (raises before any DB change)
        set_active_embedding_model(variant_name, source=f"variant selection by {user_id}")

        await _persist_active_embedding_model(db, user_id, variant_name)
        await db.commit()

    logger.info(f"[VariantManager] Activated embedding variant '{variant_name}' for all embedding features.")

    return {
        "status": "success",
        "variant_id": variant_id,
        "active_variant_name": variant_name,
        "message": (
            f"Successfully activated variant '{variant_name}' for all embeddings. "
            "Existing ChromaDB data embedded with the previous model should be reindexed "
            "(Settings → RAG → Reindex ChromaDB with Trained Variant)."
        ),
    }


async def delete_variant(user_id: str, variant_id: str) -> Dict[str, Any]:
    """
    Delete an embedding variant from disk and SQLite registry.
    If the variant is the active model, the active model falls back to the variant's
    base model (or the default base model) *before* any files are removed.
    """
    from services.text_embedding_service import set_active_embedding_model, validate_embedding_model_dir

    async with get_db_context() as db:
        res = await db.execute(
            text("SELECT id, name, model_path, base_model FROM embedding_variants WHERE id = :vid"),
            {"vid": variant_id}
        )
        row = res.fetchone()
        if not row:
            raise FileNotFoundError(f"Variant with ID '{variant_id}' not found.")

        name = row[1]
        model_path = Path(row[2])
        base_model = row[3]

        fallback_model: Optional[str] = None
        if settings.EMBEDDING_MODEL == name:
            fallback_model = DEFAULT_EMBEDDING_MODEL
            if base_model and not validate_embedding_model_dir(settings.embedding_model_dir(base_model)):
                fallback_model = base_model
            # Raises EmbeddingModelLockedError during a reindex — nothing deleted yet.
            set_active_embedding_model(fallback_model, source=f"active variant '{name}' deleted")

        # Delete from DB; every user pointing at the deleted variant falls back too.
        await db.execute(
            text("DELETE FROM embedding_variants WHERE id = :vid"),
            {"vid": variant_id}
        )
        await db.execute(
            text("""
                UPDATE user_settings
                SET embedding_model = :fallback
                WHERE embedding_model = :name
            """),
            {"fallback": fallback_model or settings.EMBEDDING_MODEL, "name": name},
        )
        if fallback_model:
            await _persist_active_embedding_model(db, user_id, fallback_model)

        await db.commit()

    # Delete from disk last, once nothing references the variant anymore.
    embeddings_root = _get_embeddings_base_dir().resolve()
    resolved = model_path.resolve()
    if resolved.exists() and embeddings_root in resolved.parents:
        shutil.rmtree(resolved, ignore_errors=True)
    elif resolved.exists():
        logger.warning(f"[VariantManager] Not deleting '{resolved}': outside the embeddings directory.")

    logger.info(
        f"[VariantManager] Deleted embedding variant '{name}'."
        + (f" Active embedding model reset to '{fallback_model}'." if fallback_model else "")
    )
    return {
        "status": "deleted",
        "variant_id": variant_id,
        "name": name,
        "active_model": settings.EMBEDDING_MODEL,
        "fallback_model": fallback_model,
    }
