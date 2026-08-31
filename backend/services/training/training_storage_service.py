import json
import uuid
import logging
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from config import RUNTIME_DIR
from .training_models import TrainingArtifactMetadata, TrainingStage, TrainingMethod

logger = logging.getLogger(__name__)

TRAINING_BASE_DIR = RUNTIME_DIR / "training"

def get_stage_method_dir(stage: TrainingStage, method: TrainingMethod) -> Path:
    return TRAINING_BASE_DIR / stage.value / method.value

def get_datasets_dir() -> Path:
    return TRAINING_BASE_DIR / "datasets"

def ensure_dirs():
    for stage in TrainingStage:
        for method in TrainingMethod:
            get_stage_method_dir(stage, method).mkdir(parents=True, exist_ok=True)
    get_datasets_dir().mkdir(parents=True, exist_ok=True)

def save_dataset(dataset_id: str, data: dict) -> Path:
    ensure_dirs()
    path = get_datasets_dir() / f"{dataset_id}.json"
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=str)
    return path

def load_dataset(dataset_id: str) -> Optional[dict]:
    path = get_datasets_dir() / f"{dataset_id}.json"
    if not path.exists():
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def list_datasets(user_id: str) -> List[dict]:
    ensure_dirs()
    datasets = []
    ds_dir = get_datasets_dir()
    for p in ds_dir.glob('*.json'):
        try:
            with open(p, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if data.get('user_id') == user_id:
                datasets.append(data)
        except Exception:
            pass
    return sorted(datasets, key=lambda x: x.get('created_at', ''), reverse=True)

def create_artifact_dir(stage: TrainingStage, method: TrainingMethod) -> tuple[str, Path]:
    artifact_id = str(uuid.uuid4())
    artifact_dir = get_stage_method_dir(stage, method) / artifact_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_id, artifact_dir

def save_artifact_metadata(metadata: TrainingArtifactMetadata) -> None:
    stage = TrainingStage(metadata.stage)
    method = TrainingMethod(metadata.method)
    artifact_dir = get_stage_method_dir(stage, method) / metadata.artifact_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    meta_path = artifact_dir / 'metadata.json'
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata.model_dump(), f, indent=2, default=str)

def load_artifact_metadata(artifact_id: str, stage: TrainingStage, method: TrainingMethod) -> Optional[TrainingArtifactMetadata]:
    artifact_dir = get_stage_method_dir(stage, method) / artifact_id
    meta_path = artifact_dir / 'metadata.json'
    if not meta_path.exists():
        return None
    with open(meta_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return TrainingArtifactMetadata(**data)

def list_artifacts(stage: Optional[TrainingStage] = None, method: Optional[TrainingMethod] = None) -> List[TrainingArtifactMetadata]:
    ensure_dirs()
    artifacts = []
    stages = [stage] if stage else list(TrainingStage)
    methods = [method] if method else list(TrainingMethod)
    for s in stages:
        for m in methods:
            d = get_stage_method_dir(s, m)
            for artifact_dir in d.iterdir():
                if artifact_dir.is_dir():
                    meta_path = artifact_dir / 'metadata.json'
                    if meta_path.exists():
                        try:
                            with open(meta_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                            artifacts.append(TrainingArtifactMetadata(**data))
                        except Exception as e:
                            logger.warning(f'[Training Storage] Failed to load artifact metadata: {e}')
    return sorted(artifacts, key=lambda x: x.training_date, reverse=True)

def deactivate_all_for_stage(stage: TrainingStage) -> None:
    for method in TrainingMethod:
        d = get_stage_method_dir(stage, method)
        for artifact_dir in d.iterdir():
            if artifact_dir.is_dir():
                meta_path = artifact_dir / 'metadata.json'
                if meta_path.exists():
                    try:
                        with open(meta_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        data['is_active'] = False
                        with open(meta_path, 'w', encoding='utf-8') as f:
                            json.dump(data, f, indent=2, default=str)
                    except Exception:
                        pass

def activate_artifact(artifact_id: str, stage: TrainingStage, method: TrainingMethod) -> bool:
    deactivate_all_for_stage(stage)
    artifact_dir = get_stage_method_dir(stage, method) / artifact_id
    meta_path = artifact_dir / 'metadata.json'
    if not meta_path.exists():
        return False
    with open(meta_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    data['is_active'] = True
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=str)
    return True

def get_active_artifact(stage: TrainingStage) -> Optional[TrainingArtifactMetadata]:
    for method in TrainingMethod:
        for artifact in list_artifacts(stage, method):
            if artifact.is_active:
                return artifact
    return None
