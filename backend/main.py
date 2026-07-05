"""FastAPI application entry point."""
# ── Force HuggingFace offline mode BEFORE any HF library is imported ──────────
# This must be the very first code that runs. HF hub performs connectivity checks
# at import time (not just at model-load time), so setting these env vars after
# any `import transformers / whisperx / pyannote` is already too late.
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
# ─────────────────────────────────────────────────────────────────────────────
import json
import logging
import warnings
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
import torch
import torchaudio

from license import check_license, LICENSE_EXPIRED_BODY

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning, module='pyannote')
warnings.filterwarnings('ignore', category=UserWarning, module='torchcodec')
warnings.filterwarnings('ignore', category=FutureWarning, module='librosa')

from config import settings
from database import connect_db, close_db
from services.diarization import init_diarization
from routers.auth import router as auth_router
from routers.voice import router as voice_router
from routers.audio import router as audio_router
from routers.history import router as history_router
from routers.settings_router import router as settings_router
from routers.pdf_router import router as pdf_router
from routers.mom_router import router as mom_router
from routers.prompt_router import router as prompt_router
from routers.dictionary_router import router as dictionary_router
from routers.analytics_router import router as analytics_router
from services.record import OverlapModel
from services.device_utils import DEVICE as _ML_DEVICE, log_device_info as _log_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── Overlap model — loaded once at startup, may be None ──────
_overlap_model: OverlapModel | None = None
_overlap_device: str = "cpu"


def _load_overlap_model() -> OverlapModel | None:
    """Load the Wav2Vec2-based overlap classifier. Returns None if unavailable."""
    model_path = settings.OVERLAP_MODEL_PATH
    if not model_path:
        logger.info(
            "[OverlapModel] OVERLAP_MODEL_PATH not configured — "
            "cross-talk detection disabled (set it in .env to enable)."
        )
        return None
    if not os.path.exists(model_path):
        logger.info(
            f"[OverlapModel] Model file not found at '{model_path}' — "
            "cross-talk detection disabled. Set OVERLAP_MODEL_PATH in .env to enable."
        )
        return None

    # Resolve the local Wav2Vec2 encoder directory.
    wav2vec2_dir: str | None = None
    try:
        from services.model_loader import ModelLoader
        align_path = ModelLoader.get_model_path("align_engine")
        if align_path and align_path.exists():
            wav2vec2_dir = str(align_path)
            logger.info(f"[OverlapModel] Using local Wav2Vec2 encoder from: {wav2vec2_dir}")
        else:
            logger.error(
                "[OverlapModel] align_engine/ not found in MODELS_DIR. "
                "Cross-talk detection cannot start in offline mode."
            )
            return None
    except Exception as e:
        logger.error(f"[OverlapModel] Could not resolve align_engine path ({e}).")
        return None

    try:
        m = OverlapModel(model_dir=wav2vec2_dir)
        m.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=False))
        m.eval()
        logger.info(f"[OverlapModel] Loaded from '{model_path}' ✓")
        return m
    except Exception as e:
        logger.error(f"[OverlapModel] Failed to load: {e}")
        return None


def get_overlap_model() -> OverlapModel | None:
    """Lazy-load overlap model when first required."""
    global _overlap_model, _overlap_device
    if _overlap_model is None:
        logger.info("[OverlapModel] Lazy loading overlap classifier...")
        _overlap_model = _load_overlap_model()
        if _overlap_model is not None:
            _overlap_model.to(_overlap_device)
    return _overlap_model


def unload_overlap_model():
    """Unload overlap model from memory."""
    global _overlap_model
    if _overlap_model is not None:
        logger.info("[OverlapModel] Unloading overlap classifier...")
        del _overlap_model
        _overlap_model = None
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info("[OverlapModel] Overlap classifier unloaded.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _overlap_model, _overlap_device

    # Startup
    logger.info("Starting VoiceSum API (fully offline mode)...")

    # ── License check ─────────────────────────────────────────────
    _valid, _msg = check_license()
    if not _valid:
        logger.critical(f"[License] APPLICATION LICENSE HAS EXPIRED: {_msg}")
        logger.critical("[License] All API requests will be rejected with HTTP 503.")
    else:
        logger.info("[License] License valid — application is authorized to run.")
    # ─────────────────────────────────────────────────────────────

    # Apply HF offline environment settings (belt-and-suspenders after top-of-file injection)
    from services.model_loader import setup_offline_hf_environment
    setup_offline_hf_environment()
    _log_device()           # logs "Using CUDA -- GPU: ..." or "using CPU"
    await connect_db()
    init_diarization()      # try load pyannote if HF_TOKEN set

    # Overlap model device setup (do not pre-load model to save RAM/VRAM)
    _overlap_device = _ML_DEVICE

    # Ensure upload dir exists
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

    # Warm up the Qwen3 model (now a deferred log to optimize startup/idle footprint)
    try:
        from services.ai_provider import warm_up_model
        warm_up_model()
    except Exception as e:
        logger.warning(f"[Startup] Qwen3 model warm-up skipped: {e}")

    yield

    # Shutdown
    await close_db()
    logger.info("VoiceSum API shut down.")


app = FastAPI(
    title="VoiceSum API",
    description="Voice Conversation Summarization with Speaker Identification",
    version="1.0.0",
    lifespan=lifespan,
)


# ── License Middleware ─────────────────────────────────────────
# Runs before every request. Returns HTTP 503 after the license expiry date.
# Passes /health through so the frontend can detect the expired state.
class LicenseMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        valid, _ = check_license()
        if not valid:
            # Allow /health so clients can detect the expired state
            if request.url.path in ("/health", "/"):
                return await call_next(request)
            return JSONResponse(
                status_code=503,
                content=LICENSE_EXPIRED_BODY,
            )
        return await call_next(request)


app.add_middleware(LicenseMiddleware)

# ── CORS ──────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)

# ── Routers ───────────────────────────────────────────────────
app.include_router(auth_router)
app.include_router(voice_router)
app.include_router(audio_router)
app.include_router(history_router)
app.include_router(settings_router)
app.include_router(pdf_router)
app.include_router(mom_router)
app.include_router(prompt_router)
app.include_router(dictionary_router)
app.include_router(analytics_router)

# ── Serve uploaded audio files ────────────────────────────────
if os.path.exists(settings.UPLOAD_DIR):
    app.mount("/files", StaticFiles(directory=settings.UPLOAD_DIR), name="files")


@app.get("/", tags=["health"])
async def root():
    return {"status": "ok", "service": "VoiceSum API", "version": "1.0.0"}


@app.get("/health", tags=["health"])
async def health():
    from services.diarization import is_pyannote_available
    from services.ai_provider import QwenProvider
    from license import check_license, LICENSE_EXPIRY_DATE
    license_valid, _ = check_license()
    return {
        "status": "ok",
        "license_valid": license_valid,
        "license_expiry": LICENSE_EXPIRY_DATE.isoformat(),
        "pyannote_available": is_pyannote_available(),
        "llm_ready": QwenProvider._pipeline is not None,
        "llm_model": settings.QWEN_MODEL_ID,
        "offline_mode": True,
        "overlap_model_loaded": _overlap_model is not None,
        "overlap_device": _overlap_device,
    }


# ── Cross-talk / overlap detection ───────────────────────────
# Registered under /api/detect-overlap (prefix matches frontend call)
@app.post("/api/detect-overlap", tags=["realtime"])
async def detect_overlap(file: UploadFile = File(...)):
    """
    Accept a ~1-second audio chunk (webm/opus/wav) and return
    {"overlap": 1} if cross-talk is detected, {"overlap": 0} otherwise.
    """
    import subprocess, tempfile, uuid

    model = get_overlap_model()
    if model is None:
        return {"overlap": 0, "status": "disabled"}

    audio_bytes = await file.read()
    uid = uuid.uuid4().hex[:8]

    # Browser MediaRecorder sends WebM/Opus — save with correct extension
    # so ffmpeg can auto-detect the container.
    src_path = os.path.join(settings.UPLOAD_DIR, f"_overlap_src_{uid}.webm")
    dst_path = os.path.join(settings.UPLOAD_DIR, f"_overlap_wav_{uid}.wav")

    with open(src_path, "wb") as f:
        f.write(audio_bytes)

    try:
        # Transcode WebM → 16-kHz mono PCM WAV via ffmpeg
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", src_path,
                "-ar", "16000",
                "-ac", "1",
                "-f", "wav",
                dst_path,
            ],
            capture_output=True,
            timeout=5,
        )
        if result.returncode != 0:
            logger.warning(
                f"[OverlapDetect] ffmpeg failed (rc={result.returncode}): "
                f"{result.stderr.decode(errors='replace')[-300:]}"
            )
            return {"overlap": 0}

        waveform, sr = torchaudio.load(dst_path)
    except FileNotFoundError:
        logger.warning("[OverlapDetect] ffmpeg not found — cannot decode WebM chunks.")
        return {"overlap": 0}
    except Exception as e:
        logger.warning(f"[OverlapDetect] Failed to load audio: {e}")
        return {"overlap": 0}
    finally:
        for p in (src_path, dst_path):
            try:
                os.remove(p)
            except OSError:
                pass

    # torchaudio.load already honours the -ar 16000 conversion above,
    # but resample defensively if something changed.
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

    waveform = waveform.squeeze(0)

    # Normalise
    peak = waveform.abs().max()
    if peak > 0:
        waveform = waveform / peak

    # Pad or trim to exactly 1 second (16 000 samples)
    target_len = 16000
    if waveform.shape[0] < target_len:
        waveform = torch.cat([waveform, torch.zeros(target_len - waveform.shape[0])])
    else:
        waveform = waveform[:target_len]

    waveform = waveform.unsqueeze(0).to(_overlap_device)

    with torch.no_grad():
        prob: float = model(waveform).item()
    logger.debug(f"[OverlapDetect] overlap_prob={prob:.4f}")
    return {"overlap": 1 if prob > 0.6 else 0, "probability": round(prob, 4)}


if __name__ == "__main__":
    import uvicorn
    # Pass the app object directly so Uvicorn does not attempt to import "main" at runtime (fails in PyInstaller)
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")


