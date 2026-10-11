# AI Services, Local LLM Inference & Embedding Architecture

## 1. Overview of AI Engines

VoiceSum operates completely offline by running state-of-the-art machine learning models directly on the user's workstation. The intelligence subsystem comprises four primary engines:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   VoiceSum Local AI Engines Layer                      │
├────────────────────┬───────────────────────────────────────────────────┤
│ Generative LLM     │ Qwen3-4B-Instruct (4-bit NF4 / bfloat16 via       │
│ Engine             │ Hugging Face Transformers) with Ollama fallback.  │
├────────────────────┼───────────────────────────────────────────────────┤
│ Dense Embedding    │ Qwen3-Embedding-0.6B producing 1024-dimensional   │
│ Engine             │ normalized vectors for semantic similarity.       │
├────────────────────┼───────────────────────────────────────────────────┤
│ Vector Database    │ ChromaDB running in embedded persistent mode on   │
│ Engine             │ local filesystem (runtime/vector_store).          │
├────────────────────┼───────────────────────────────────────────────────┤
│ Visual Document    │ PaddleOCR PP-OCRv6 (offline detection,            │
│ & OCR Engine       │ recognition, direction classifier).               │
└────────────────────┴───────────────────────────────────────────────────┘
```

---

## 2. Generative LLM: Local Model & Ollama Fallback

### 2.1 Dual-Engine Architecture (`services/ai_provider.py`)
VoiceSum implements a transparent multi-provider abstraction:
1. **Primary Engine — In-Process Hugging Face Model**:
   - Model: `Qwen/Qwen3-4B-Instruct`.
   - Execution: Managed by `services/model_loader.py`.
   - Quantization: 4-bit NormalFloat (NF4) via `bitsandbytes` or native `bfloat16` when 12GB+ GPU VRAM is available.
   - Context Window: Dynamically scaled up to 32,768 tokens (`ollama_num_ctx` / `max_position_embeddings`).
2. **Fallback Engine — Local Ollama Service**:
   - If user activates Ollama in Settings (`use_ollama=1`), or if VRAM is constrained, requests route to `http://localhost:11434/api/generate` or `/api/chat`.
   - Supports models: `qwen2.5:3b`, `gemma2:2b`, `llama3.2:3b`, `deepseek-r1:1.5b`.

### 2.2 VRAM & Device Memory Management (`services/device_utils.py`)
To prevent CUDA Out-of-Memory (OOM) crashes during concurrent transcription, diarization, and LLM generation:
- Models are loaded with `device_map="auto"`.
- Explicit GPU cache purging: `torch.cuda.empty_cache()` and `gc.collect()` are called before and after intensive generation batches.
- Active memory monitoring: logs allocated vs reserved GPU memory in `log_gpu_memory()`.

---

## 3. Summarization & Intelligence Capabilities (`services/llm.py`)

The LLM engine provides specialized generation endpoints:

| Generation Task | Function in `services/llm.py` | Token Budget | Primary Purpose |
|---|---|---|---|
| **Executive Summary** | `generate_executive_summary()` | 700 tokens | Concise, high-level briefing for executives. |
| **Short Summary** | `generate_short_summary()` | 120 tokens | 2-sentence thumbnail for cards and table lists. |
| **Detailed Summary** | `generate_detailed_summary()` | 3,000 tokens | Comprehensive narrative covering all topics discussed. |
| **Key Points** | `generate_key_points()` | 1,028 tokens | Bulleted list of pivotal decisions and updates. |
| **Action Items** | `generate_action_items()` | 1,028 tokens | Structured `{owner, task, deadline, status}` items. |
| **Speaker Summaries**| `generate_speaker_summaries()`| 200 tokens/spk| Individual contribution breakdown per person. |
| **Minutes of Meeting**| `generate_mom()` | 1,500 tokens | Formal standard corporate MoM layout. |

---

## 4. Text Embedding Engine (`services/text_embedding_service.py`)

- **Model:** `Qwen3-Embedding-0.6B`.
- **Location:** `Application/runtime/embeddings/Qwen3-Embedding-0.6B`.
- **Dimensionality:** 1024 float32 dimensions.
- **Normalization:** Vectors are $L_2$-normalized so dot product equals cosine similarity:
  $$\text{sim}(q, d) = \sum_{i=1}^{1024} q_i \cdot d_i$$
- **Inference Mode:**
  - Evaluated via PyTorch on CUDA or CPU.
  - Supports batch encoding of text chunks up to batch size 64.

---

## 5. Vector Store Management (`services/vector_store.py`)

ChromaDB operates entirely on-disk in `backend/runtime/vector_store/` using SQLite + HNSW indices.

### Active Collections

1. **`final_rom_{user_id}`**:
   - Stores finalized discussion points from ROMs and Manual MoMs.
   - Used as **Priority 1 & 2** retrieval source during AI Chat queries.
2. **`meeting_{recording_id}`**:
   - Stores raw transcript chunks of a single meeting.
   - Indexed with timestamp boundaries (`start_sec`, `end_sec`) and speaker labels.
3. **`meeting_context_{recording_id}`**:
   - Stores extracted text from meeting slides, PDFs, and agendas.

---

## 6. Document Extraction & OCR Engine (`services/ocr_engine.py` & `doc_extractor.py`)

VoiceSum extracts text from documents without needing Microsoft Office, Adobe Acrobat, or cloud APIs:
- **PDF Documents:** Evaluated via PyMuPDF (`fitz`) and pdfplumber.
- **Office Documents:** `.docx` via `python-docx`, `.pptx` via `python-pptx`, `.xlsx`/`.csv` via `openpyxl`/`csv`.
- **Scanned Documents & Video Frames:**
  - Powered by **PaddleOCR PP-OCRv6** (Offline).
  - Models stored locally in `Application/runtime/ocr_models/`:
    - Detection model: `ch_PP-OCRv4_det_infer`
    - Recognition model: `ch_PP-OCRv4_rec_infer`
    - Angle classifier: `ch_ppocr_mobile_v2.0_cls_infer`
  - Performs text rotation correction, contrast enhancement, and bounding-box text stitching.
