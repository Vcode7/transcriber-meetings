from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from datetime import datetime

class TrainingStage(str, Enum):
    STAGE_1 = "stage_1"
    STAGE_2 = "stage_2"
    STAGE_3 = "stage_3"

class TrainingMethod(str, Enum):
    DSPY = "dspy"
    LORA = "lora"
    QLORA = "qlora"

class JobStatus(str, Enum):
    PENDING = "pending"
    BUILDING_DATASET = "building_dataset"
    TRAINING = "training"
    EVALUATING = "evaluating"
    DONE = "done"
    ERROR = "error"
    CANCELLED = "cancelled"

class TrainingArtifactMetadata(BaseModel):
    artifact_id: str
    stage: TrainingStage
    method: TrainingMethod
    base_model: str
    dataset_ids: List[str]
    training_date: str
    config: Dict[str, Any]
    status: str
    eval_metrics: Optional[Dict[str, Any]] = None
    artifact_path: str
    version: int
    is_active: bool = False
    job_id: Optional[str] = None
    notes: Optional[str] = None

class DatasetSample(BaseModel):
    sample_id: str
    stage: TrainingStage
    source_meeting_id: Optional[str] = None
    inputs: Dict[str, Any]
    target: str
    metadata: Optional[Dict[str, Any]] = None

class TrainingJobProgress(BaseModel):
    step: int = 0
    total_steps: int = 0
    epoch: int = 0
    total_epochs: int = 0
    loss: Optional[float] = None
    message: str = ""
    percent: float = 0.0
