from pydantic import BaseModel, Field
from datetime import datetime


class UserSettings(BaseModel):
    user_id: str
    speaker_similarity_threshold: float = Field(default=0.75, ge=0.5, le=0.99)
    word_conf_low: float = Field(default=0.7, ge=0.1, le=1.0)
    word_conf_mid: float = Field(default=0.85, ge=0.1, le=1.0)
    min_segment_duration: float = Field(default=1.5, ge=0.1, le=10.0)
    whisper_model_size: str = "large-v3"
    use_ollama: bool = False
    ollama_server_url: str = "http://localhost:11434"
    ollama_port: int = Field(default=11434, ge=1, le=65535)
    ollama_model_priority: str = "gemma,qwen,llama,deepseek,mistral"
    rag_chunk_size: int = Field(default=400, ge=10, le=5000)
    rag_chunk_overlap: int = Field(default=50, ge=0, le=1000)
    rag_retrieval_k_global: int = Field(default=2, ge=0, le=50)
    rag_retrieval_k_meeting: int = Field(default=3, ge=0, le=50)
    rag_retrieval_k_transcript: int = Field(default=10, ge=0, le=50)
    rag_max_collection_context: int = Field(default=10, ge=1, le=50)
    rag_relative_score_cutoff: float = Field(default=0.01, ge=0.0, le=1.0)
    generate_mom_auto: bool = True
    embedding_model: str = "Qwen3-Embedding-0.6B"
    
    # New Ollama specific settings
    ollama_num_ctx: int = Field(default=32768, ge=512, le=131072)
    ollama_dynamic_ctx: bool = Field(default=True)
    ollama_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    ollama_top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    ollama_top_k: int = Field(default=40, ge=0)
    ollama_repeat_penalty: float = Field(default=1.15, ge=0.0)
    ollama_seed: int = Field(default=-1, ge=-1)
    ollama_stop: str = ""
    ollama_keep_alive: str = "5m"
    ollama_num_thread: int = Field(default=0, ge=0)
    ollama_num_gpu: int = Field(default=-1, ge=-1)

    # Individual task max token limits
    max_tokens_mom: int = Field(default=1500, ge=1)
    max_tokens_mom_merge: int = Field(default=3072, ge=1)
    max_tokens_raw_mom_to_mom: int = Field(default=3000, ge=1)
    max_tokens_raw_mom_extraction: int = Field(default=1024, ge=1)
    max_tokens_raw_mom_repair: int = Field(default=1024, ge=1)
    max_tokens_agenda_compress: int = Field(default=2000, ge=1)
    max_tokens_reference_compress: int = Field(default=2000, ge=1)
    max_tokens_agenda_from_summary: int = Field(default=1024, ge=1)
    max_tokens_executive_summary: int = Field(default=700, ge=1)
    max_tokens_short_summary: int = Field(default=120, ge=1)
    max_tokens_detailed_summary: int = Field(default=3000, ge=1)
    max_tokens_chunk_summary: int = Field(default=256, ge=1)
    max_tokens_key_points: int = Field(default=1028, ge=1)
    max_tokens_action_items: int = Field(default=1028, ge=1)
    max_tokens_key_decisions: int = Field(default=1028, ge=1)
    max_tokens_speaker_summary: int = Field(default=200, ge=1)
    max_tokens_speaker_key_points: int = Field(default=350, ge=1)
    max_tokens_speaker_action_items: int = Field(default=250, ge=1)
    max_tokens_collection_chat: int = Field(default=1500, ge=1)
    max_tokens_collection_compare: int = Field(default=1500, ge=1)
    max_tokens_collection_topic_growth: int = Field(default=1500, ge=1)
    max_tokens_vocab_extractor: int = Field(default=512, ge=1)

    updated_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = {"arbitrary_types_allowed": True}


class UserSettingsUpdate(BaseModel):
    speaker_similarity_threshold: float | None = Field(default=None, ge=0.5, le=0.99)
    word_conf_low: float | None = Field(default=None, ge=0.1, le=1.0)
    word_conf_mid: float | None = Field(default=None, ge=0.1, le=1.0)
    min_segment_duration: float | None = Field(default=None, ge=0.1, le=10.0)
    use_ollama: bool | None = None
    ollama_server_url: str | None = None
    ollama_port: int | None = Field(default=None, ge=1, le=65535)
    ollama_model_priority: str | None = None
    rag_chunk_size: int | None = Field(default=None, ge=10, le=5000)
    rag_chunk_overlap: int | None = Field(default=None, ge=0, le=1000)
    rag_retrieval_k_global: int | None = Field(default=None, ge=0, le=50)
    rag_retrieval_k_meeting: int | None = Field(default=None, ge=0, le=50)
    rag_retrieval_k_transcript: int | None = Field(default=None, ge=0, le=50)
    rag_max_collection_context: int | None = Field(default=None, ge=1, le=50)
    rag_relative_score_cutoff: float | None = Field(default=None, ge=0.0, le=1.0)
    generate_mom_auto: bool | None = None
    embedding_model: str | None = None
    
    # New Ollama settings
    ollama_num_ctx: int | None = Field(default=None, ge=512, le=131072)
    ollama_dynamic_ctx: bool | None = None
    ollama_temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    ollama_top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    ollama_top_k: int | None = Field(default=None, ge=0)
    ollama_repeat_penalty: float | None = Field(default=None, ge=0.0)
    ollama_seed: int | None = Field(default=None, ge=-1)
    ollama_stop: str | None = None
    ollama_keep_alive: str | None = None
    ollama_num_thread: int | None = Field(default=None, ge=0)
    ollama_num_gpu: int | None = Field(default=None, ge=-1)

    # Individual task max token limits
    max_tokens_mom: int | None = Field(default=None, ge=1)
    max_tokens_mom_merge: int | None = Field(default=None, ge=1)
    max_tokens_raw_mom_to_mom: int | None = Field(default=None, ge=1)
    max_tokens_raw_mom_extraction: int | None = Field(default=None, ge=1)
    max_tokens_raw_mom_repair: int | None = Field(default=None, ge=1)
    max_tokens_agenda_compress: int | None = Field(default=None, ge=1)
    max_tokens_reference_compress: int | None = Field(default=None, ge=1)
    max_tokens_agenda_from_summary: int | None = Field(default=None, ge=1)
    max_tokens_executive_summary: int | None = Field(default=None, ge=1)
    max_tokens_short_summary: int | None = Field(default=None, ge=1)
    max_tokens_detailed_summary: int | None = Field(default=None, ge=1)
    max_tokens_chunk_summary: int | None = Field(default=None, ge=1)
    max_tokens_key_points: int | None = Field(default=None, ge=1)
    max_tokens_action_items: int | None = Field(default=None, ge=1)
    max_tokens_key_decisions: int | None = Field(default=None, ge=1)
    max_tokens_speaker_summary: int | None = Field(default=None, ge=1)
    max_tokens_speaker_key_points: int | None = Field(default=None, ge=1)
    max_tokens_speaker_action_items: int | None = Field(default=None, ge=1)
    max_tokens_collection_chat: int | None = Field(default=None, ge=1)
    max_tokens_collection_compare: int | None = Field(default=None, ge=1)
    max_tokens_collection_topic_growth: int | None = Field(default=None, ge=1)
    max_tokens_vocab_extractor: int | None = Field(default=None, ge=1)


