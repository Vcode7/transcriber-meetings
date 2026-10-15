"""ROM pipeline settings chosen on the Upload / Record / Tab Recording pages.

Stored as JSON in ``recordings.rom_settings`` (per meeting) and
``user_settings.rom_pipeline_defaults`` (per user, "Save as default").
The background ROM run (tasks/rom_auto_pipeline.py) and the ROM page both read them.
"""
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

OutputType = Literal["off", "mom", "rom"]


class StagedFileRef(BaseModel):
    staged_id: str
    filename: str = ""


class RomPipelineSettings(BaseModel):
    # off = no automatic ROM run (transcription behaves as before)
    # mom = Stage 1 -> Stage 2 -> MoM from Stage 2 points
    # rom = Stage 1 -> Stage 2 -> Create Agenda -> Final ROM (long)
    output_type: OutputType = "off"

    # Stage 1 (same bounds as Stage1Request)
    transcript_window_minutes: float = Field(default=2.0, ge=0.5, le=30.0)

    # Stage 2 (same bounds as Stage2Request)
    meeting_top_k: int = Field(default=5, ge=0, le=30)
    global_top_k: int = Field(default=3, ge=0, le=30)
    discussion_window_size: int = Field(default=5, ge=3, le=50)
    # Optional raw-cosine floor; relevance is decided by the store-relative cutoff (services/rom_retrieval.py)
    min_similarity_threshold: Optional[float] = Field(default=0.60, ge=0.0, le=1.0)
    process_all_together: Optional[bool] = None
    previous_meeting_mode: Literal["auto", "select", "off"] = "auto"
    previous_meeting_id: Optional[str] = None
    previous_meeting_top_k: int = Field(default=3, ge=0, le=20)

    # Final ROM (rom only): generate action points for every agenda
    include_action_points: bool = True

    # Agenda document (rom only): take agenda items from its tables / table-like blocks only,
    # ignoring headings and other text; off scans the whole document (lists and other formats)
    agenda_from_tables: bool = True

    # Meeting-specific inputs (never saved as defaults)
    context_files: List[StagedFileRef] = Field(default_factory=list)
    agenda_doc: Optional[StagedFileRef] = None
    agendas: Optional[List[Dict[str, Any]]] = None
    agenda_text: Optional[str] = None

    @property
    def auto_run(self) -> bool:
        return self.output_type in ("mom", "rom")


# Fields that "Save as default" keeps; the rest belong to one meeting only.
DEFAULT_FIELDS = (
    "output_type", "transcript_window_minutes", "meeting_top_k", "global_top_k",
    "discussion_window_size", "min_similarity_threshold", "process_all_together",
    "previous_meeting_mode", "previous_meeting_top_k", "include_action_points", "agenda_from_tables",
)


def parse_rom_settings(raw: Any) -> Optional[RomPipelineSettings]:
    """Settings from a JSON string / dict; None when missing or invalid."""
    import json
    import logging

    if not raw:
        return None
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(data, dict):
            return None
        return RomPipelineSettings.model_validate(data)
    except Exception as exc:
        logging.getLogger(__name__).warning(f"[ROMPipeline] Ignoring invalid rom_settings: {exc}")
        return None
