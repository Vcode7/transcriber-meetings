from pydantic import BaseModel, Field
from datetime import datetime


class UserSettings(BaseModel):
    user_id: str
    speaker_similarity_threshold: float = 0.75
    word_conf_low: float = 0.7
    word_conf_mid: float = 0.85
    min_segment_duration: float = 1.5
    whisper_model_size: str = "medium"
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = {"arbitrary_types_allowed": True}


class UserSettingsUpdate(BaseModel):
    speaker_similarity_threshold: float | None = None
    word_conf_low: float | None = None
    word_conf_mid: float | None = None
    min_segment_duration: float | None = None
