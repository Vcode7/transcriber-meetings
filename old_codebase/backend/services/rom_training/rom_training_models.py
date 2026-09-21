from pydantic import BaseModel, Field
from typing import Optional, List, Any, Dict
from enum import Enum

class RomTrainingType(str, Enum):
    new = "new"
    upgrade = "upgrade"

class RomTrainingMethod(str, Enum):
    lora = "lora"
    qlora = "qlora"

class RomLoRAConfig(BaseModel):
    method: RomTrainingMethod = RomTrainingMethod.qlora
    r: int = Field(default=16, ge=1, le=256)
    lora_alpha: int = Field(default=32, ge=1)
    lora_dropout: float = Field(default=0.05, ge=0.0, le=0.5)
    target_modules: str = "q_proj,v_proj"  # comma-separated
    learning_rate: float = Field(default=2e-4, gt=0)
    num_epochs: int = Field(default=3, ge=1, le=100)
    per_device_train_batch_size: int = Field(default=2, ge=1)
    gradient_accumulation_steps: int = Field(default=4, ge=1)
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    max_seq_length: int = Field(default=2048, ge=128)

class StartRomTrainingRequest(BaseModel):
    name: str
    description: Optional[str] = None
    model_path: str
    training_type: RomTrainingType = RomTrainingType.new
    parent_variant_id: Optional[str] = None
    lora_config: RomLoRAConfig = Field(default_factory=RomLoRAConfig)
    dataset_size: int = Field(default=50, ge=1)
    test_size: int = Field(default=10, ge=1)

class ExtractMomRequest(BaseModel):
    # used when text is passed directly (not file upload)
    text: Optional[str] = None

class AgendaMatchItem(BaseModel):
    mom_agenda_title: str
    mom_agenda_idx: int
    meeting_agenda_id: Optional[str] = None
    meeting_agenda_title: Optional[str] = None
    confidence: float = 0.0
    manual: bool = False  # True if user manually assigned

class SaveTrainingDataRequest(BaseModel):
    recording_id: str
    recording_title: Optional[str] = None
    pairs: List[Dict[str, Any]]  # list of {agenda_id, agenda_title, long_rom_points, manual_mom_points}
