from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class WordToken(BaseModel):
    word: str
    start: float
    end: float
    probability: float


class TranscriptSegment(BaseModel):
    speaker_label: str          # "John" or "Speaker 1"
    speaker_id: Optional[str]   # voice_profile id if matched
    start: float
    end: float
    text: str
    words: List[WordToken] = []
    confidence: float = 1.0
    is_overlap: bool = False


class RecordingCreate(BaseModel):
    user_id: str
    filename: str
    duration: float
    file_path: str


class Recording(BaseModel):
    id: Optional[str] = None
    user_id: str
    filename: str
    file_path: str
    duration: float = 0.0
    status: str = "pending"          # pending | processing | done | error
    error_message: Optional[str] = None
    transcript: List[TranscriptSegment] = []
    raw_text: Optional[str] = None
    summary: Optional[str] = None
    key_points: List[str] = []
    action_items: List[str] = []
    speakers_detected: List[str] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)
    processed_at: Optional[datetime] = None

    model_config = {"arbitrary_types_allowed": True}


class RecordingListItem(BaseModel):
    id: str
    filename: str
    duration: float
    status: str
    speakers_detected: List[str]
    created_at: datetime
    has_summary: bool


class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
