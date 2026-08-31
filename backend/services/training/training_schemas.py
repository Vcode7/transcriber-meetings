from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from .training_models import TrainingStage, TrainingMethod, JobStatus, TrainingArtifactMetadata, DatasetSample, TrainingJobProgress

class StageTrainingConfig(BaseModel):
    stage: TrainingStage
    method: TrainingMethod
    model_name: str
    # DSPy config
    dspy_optimizer: str = "BootstrapFewShot"  # or MIPROv2
    dspy_max_bootstrapped_demos: int = 3
    dspy_max_labeled_demos: int = 4
    dspy_num_trials: int = 10
    # LoRA/QLoRA config  
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_seq_length: int = 2048
    hf_model_path: Optional[str] = None  # For LoRA/QLoRA with HF model
    # Eval
    eval_split: float = 0.2
    eval_metric: str = "rouge"  # or 'similarity'

class BuildDatasetRequest(BaseModel):
    source_type: str  # 'meeting' or 'upload'
    meeting_id: Optional[str] = None
    stages: List[TrainingStage]
    manual_mom: Optional[str] = None  # ground-truth MoM text
    use_edited_stage2: bool = False  # Use edited Stage 2 version as training target if available
    # For upload source
    transcript_text: Optional[str] = None
    context_text: Optional[str] = None
    agenda_text: Optional[str] = None

class StartTrainingJobRequest(BaseModel):
    dataset_id: str
    stage_configs: List[StageTrainingConfig]
    description: Optional[str] = None

class ValidateRequest(BaseModel):
    model_name: str
    method: TrainingMethod
    stage: TrainingStage
    hf_model_path: Optional[str] = None

class ValidateResponse(BaseModel):
    valid: bool
    reason: str
    warnings: List[str] = []

class TrainingJobResponse(BaseModel):
    job_id: str
    user_id: str
    description: Optional[str]
    status: JobStatus
    stage_configs: List[StageTrainingConfig]
    dataset_id: Optional[str]
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    error: Optional[str]
    logs: List[str]
    progress: Optional[TrainingJobProgress]
    artifacts: List[TrainingArtifactMetadata] = []

class DatasetResponse(BaseModel):
    dataset_id: str
    stages: List[TrainingStage]
    total_samples: int
    samples_by_stage: Dict[str, int]
    samples: List[DatasetSample]
    created_at: str
    source_type: str
    meeting_id: Optional[str]

class ActivateRequest(BaseModel):
    artifact_id: str
    stage: TrainingStage

class EvaluationResponse(BaseModel):
    job_id: str
    stage: TrainingStage
    method: TrainingMethod
    score_before: Optional[float]
    score_after: Optional[float]
    metrics: Dict[str, Any]
    sample_comparisons: List[Dict[str, Any]]
