"""
variant_manager.py — Lifecycle management, persistence, selection, and deletion of embedding model variants.

Ensures the base model is never overwritten and isolates each variant separately.
Allows activating a variant for the offline RAG pipeline.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import text

from config import settings, BASE_DIR, RUNTIME_DIR
from database import get_db_context, from_json, to_json

logger = logging.getLogger(__name__)


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
    # Sanitize variant name
    clean_name = variant_name.strip().replace(" ", "-")
    if not clean_name:
        raise ValueError("Variant name cannot be empty.")

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

    staged_path = Path(staged_model_path)
    if not staged_path.exists():
        raise FileNotFoundError(f"Staged model files not found at {staged_model_path}")

    # Copy / move staged files to target directory
    shutil.copytree(staged_path, target_dir)

    # Calculate total size
    total_bytes = sum(f.stat().st_size for f in target_dir.rglob("*") if f.is_file())

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
    }
    with open(target_dir / "variant_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    variant_id = str(uuid.uuid4())
    now_iso = meta["created_at"]

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

    # Clean up staging directory
    try:
        shutil.rmtree(staged_path, ignore_errors=True)
    except Exception:
        pass

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

        # Check current active model in user_settings
        settings_res = await db.execute(
            text("SELECT embedding_model FROM user_settings WHERE user_id = :uid"),
            {"uid": user_id}
        )
        s_row = settings_res.fetchone()
        active_model_name = s_row[0] if s_row and s_row[0] else settings.EMBEDDING_MODEL

        for r in rows:
            v_name = r[1]
            is_currently_active = (v_name == active_model_name) or bool(r[11])
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
            })

    return variants


async def select_variant(user_id: str, variant_id: str) -> Dict[str, Any]:
    """
    Select an embedding variant as the active model for the RAG pipeline.
    Updates user_settings and unloads existing in-memory model.
    """
    async with get_db_context() as db:
        res = await db.execute(
            text("SELECT id, name, model_path FROM embedding_variants WHERE id = :vid"),
            {"vid": variant_id}
        )
        row = res.fetchone()
        if not row:
            raise FileNotFoundError(f"Variant with ID '{variant_id}' not found.")

        variant_name = row[1]
        model_path = Path(row[2])

        if not model_path.exists():
            raise FileNotFoundError(f"Model variant directory not found at {model_path}")

        # Update active flags in DB
        await db.execute(text("UPDATE embedding_variants SET is_active = 0"))
        await db.execute(
            text("UPDATE embedding_variants SET is_active = 1 WHERE id = :vid"),
            {"vid": variant_id}
        )

        # Update user_settings
        await db.execute(
            text("""
                UPDATE user_settings
                SET embedding_model = :model_name
                WHERE user_id = :uid
            """),
            {"model_name": variant_name, "uid": user_id}
        )
        await db.commit()

    # Update in-memory app settings and unload cached embedder
    settings.EMBEDDING_MODEL = variant_name
    try:
        from services.text_embedding_service import unload_text_embedder
        unload_text_embedder()
    except Exception as e:
        logger.warning(f"[VariantManager] unload_text_embedder error: {e}")

    logger.info(f"[VariantManager] Activated embedding variant '{variant_name}' for RAG pipeline.")

    return {
        "status": "success",
        "variant_id": variant_id,
        "active_variant_name": variant_name,
        "message": f"Successfully activated variant '{variant_name}' for RAG embeddings."
    }


async def delete_variant(user_id: str, variant_id: str) -> Dict[str, Any]:
    """
    Delete an embedding variant from disk and SQLite registry.
    If the variant was active, resets active embedding model to base Qwen3 model.
    """
    async with get_db_context() as db:
        res = await db.execute(
            text("SELECT id, name, model_path, is_active FROM embedding_variants WHERE id = :vid"),
            {"vid": variant_id}
        )
        row = res.fetchone()
        if not row:
            raise FileNotFoundError(f"Variant with ID '{variant_id}' not found.")

        name = row[1]
        model_path = Path(row[2])
        was_active = bool(row[3])

        # Delete from disk
        if model_path.exists():
            shutil.rmtree(model_path, ignore_errors=True)

        # Delete from DB
        await db.execute(
            text("DELETE FROM embedding_variants WHERE id = :vid"),
            {"vid": variant_id}
        )

        # If it was active, reset to base model
        if was_active or settings.EMBEDDING_MODEL == name:
            default_base = "Qwen3-Embedding-0.6B"
            settings.EMBEDDING_MODEL = default_base
            await db.execute(
                text("""
                    UPDATE user_settings
                    SET embedding_model = :base_name
                    WHERE user_id = :uid
                """),
                {"base_name": default_base, "uid": user_id}
            )
            try:
                from services.text_embedding_service import unload_text_embedder
                unload_text_embedder()
            except Exception:
                pass

        await db.commit()

    logger.info(f"[VariantManager] Deleted embedding variant '{name}'.")
    return {
        "status": "deleted",
        "variant_id": variant_id,
        "name": name,
    }
