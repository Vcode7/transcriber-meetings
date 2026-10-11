"""
vlm_ocr_experimental.py — EXPERIMENTAL Chandra OCR 2 inference through Ollama.

Testing only. This module is used exclusively by the OCR Settings page through the
/ocr/vlm/* endpoints in routers/ocr_router.py. It is NOT part of the production OCR
pipeline (services/ocr_engine.py, services/doc_extractor.py) and must not be imported
anywhere else.

Prompts are copied verbatim from datalab-to/chandra (chandra/prompts.py) and the image
preparation mirrors chandra/input.py and chandra/model/util.py::scale_to_fit.
Chandra is Copyright Datalab, licensed under the Apache License 2.0.
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import math
import re
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Chandra prompts (verbatim from datalab-to/chandra, Apache-2.0) ────────────
ALLOWED_TAGS = [
    "math", "br", "i", "b", "u", "del", "sup", "sub", "table", "tr", "td", "p", "th",
    "div", "pre", "h1", "h2", "h3", "h4", "h5", "ul", "ol", "li", "input", "a", "span",
    "img", "hr", "tbody", "small", "caption", "strong", "thead", "big", "code", "chem",
]
ALLOWED_ATTRIBUTES = [
    "class", "colspan", "rowspan", "display", "checked", "type", "border", "value",
    "style", "href", "alt", "align", "data-bbox", "data-label",
]

PROMPT_ENDING = f"""
Only use these tags {ALLOWED_TAGS}, and these attributes {ALLOWED_ATTRIBUTES}.

Guidelines:
* Inline math: Surround math with <math>...</math> tags. Math expressions should be rendered in KaTeX-compatible LaTeX. Use display for block math.
* Tables: Use colspan and rowspan attributes to match table structure.
* Formatting: Maintain consistent formatting with the image, including spacing, indentation, subscripts/superscripts, and special characters.
* Images: Include a description of any images in the alt attribute of an <img> tag. Do not fill out the src property. Describe in detail inside the div tag. Also convert charts to high fidelity data, and convert diagrams to mermaid.
* Forms: Mark checkboxes and radio buttons properly.
* Text: join lines together properly into paragraphs using <p>...</p> tags.  Use <br> tags for line breaks within paragraphs, but only when absolutely necessary to maintain meaning.
* Chemistry: Use <chem>...</chem> tags for chemical formulas with reactive SMILES.
* Lists: Preserve indents and proper list markers.
* Use the simplest possible HTML structure that accurately represents the content of the block.
* Make sure the text is accurate and easy for a human to read and interpret.  Reading order should be correct and natural.
""".strip()

OCR_LAYOUT_PROMPT = f"""
OCR this image to HTML, arranged as layout blocks.  Each layout block should be a div with the data-bbox attribute representing the bounding box of the block in x0 y0 x1 y1 format.  Bboxes are normalized 0-1000. The data-label attribute is the label for the block.

Use the following labels:
- Caption
- Footnote
- Equation-Block
- List-Group
- Page-Header
- Page-Footer
- Image
- Section-Header
- Table
- Text
- Complex-Block
- Code-Block
- Form
- Table-Of-Contents
- Figure
- Chemical-Block
- Diagram
- Bibliography
- Blank-Page

{PROMPT_ENDING}
""".strip()

OCR_PROMPT = f"""
OCR this image to HTML.

{PROMPT_ENDING}
""".strip()

PROMPT_MAPPING = {"ocr_layout": OCR_LAYOUT_PROMPT, "ocr": OCR_PROMPT}

# ── Chandra image preparation constants (chandra/settings.py) ─────────────────
IMAGE_DPI = 192
MIN_PDF_IMAGE_DIM = 1024
MIN_IMAGE_DIM = 1536
MAX_IMAGE_SIZE = (3072, 2048)
MIN_IMAGE_SIZE = (1792, 28)
GRID_SIZE = 28

DEFAULT_MAX_TOKENS = 8192
MAX_CONTEXT = 32768
REQUEST_TIMEOUT_S = 900

# One experimental job at a time: a VLM run can occupy the GPU for minutes.
_job_lock = asyncio.Lock()


class VlmOcrError(Exception):
    """Error with an HTTP status code suitable for the API response."""

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


class VlmOcrCancelled(Exception):
    """The client disconnected while the model was generating."""


def is_busy() -> bool:
    return _job_lock.locked()


# ── Image preparation ─────────────────────────────────────────────────────────

def scale_to_fit(img, max_size=MAX_IMAGE_SIZE, min_size=MIN_IMAGE_SIZE, grid_size=GRID_SIZE):
    """Port of chandra.model.util.scale_to_fit: pixel budget on a 28-px grid, aspect preserved."""
    from PIL import Image

    width, height = img.size
    if width <= 0 or height <= 0:
        return img
    original_ar = width / height
    current_pixels = width * height
    max_pixels = max_size[0] * max_size[1]
    min_pixels = min_size[0] * min_size[1]

    scale = 1.0
    if current_pixels > max_pixels:
        scale = (max_pixels / current_pixels) ** 0.5
    elif current_pixels < min_pixels:
        scale = (min_pixels / current_pixels) ** 0.5

    w_blocks = max(1, round((width * scale) / grid_size))
    h_blocks = max(1, round((height * scale) / grid_size))
    while (w_blocks * h_blocks * grid_size * grid_size) > max_pixels:
        if w_blocks == 1 and h_blocks == 1:
            break
        if w_blocks == 1:
            h_blocks -= 1
            continue
        if h_blocks == 1:
            w_blocks -= 1
            continue
        ar_w_loss = abs(((w_blocks - 1) / h_blocks) - original_ar)
        ar_h_loss = abs((w_blocks / (h_blocks - 1)) - original_ar)
        if ar_w_loss < ar_h_loss:
            w_blocks -= 1
        else:
            h_blocks -= 1

    new_size = (w_blocks * grid_size, h_blocks * grid_size)
    if new_size == (width, height):
        return img
    return img.resize(new_size, resample=Image.Resampling.LANCZOS)


def prepare_image(data: bytes, filename: str, page: int = 1) -> Tuple[Any, int]:
    """
    Decode an uploaded image or render one PDF page the way Chandra does.
    Returns (PIL RGB image scaled for the model, total page count).
    """
    from PIL import Image, ImageOps

    if filename.lower().endswith(".pdf"):
        import fitz

        with fitz.open(stream=data, filetype="pdf") as doc:
            page_count = len(doc)
            if page < 1 or page > page_count:
                raise VlmOcrError(400, f"Page {page} does not exist — the PDF has {page_count} page(s).")
            pdf_page = doc[page - 1]
            min_dim_pt = max(1.0, min(pdf_page.rect.width, pdf_page.rect.height))
            dpi = max(IMAGE_DPI, MIN_PDF_IMAGE_DIM / min_dim_pt * 72)
            pix = pdf_page.get_pixmap(dpi=int(dpi), colorspace=fitz.csRGB, alpha=False)
            image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    else:
        page_count = 1
        with Image.open(io.BytesIO(data)) as raw:
            try:
                raw = ImageOps.exif_transpose(raw)
            except Exception:
                pass
            image = raw.convert("RGB")
        if image.width < MIN_IMAGE_DIM or image.height < MIN_IMAGE_DIM:
            scale = MIN_IMAGE_DIM / min(image.width, image.height)
            image = image.resize(
                (int(image.width * scale), int(image.height * scale)), Image.Resampling.LANCZOS
            )
    return scale_to_fit(image), page_count


def encode_png_b64(image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def thumbnail_data_url(image, max_side: int = 1100) -> str:
    thumb = image.copy()
    thumb.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# ── Output post-processing ────────────────────────────────────────────────────

def clean_html(raw: str) -> str:
    """Strip reasoning blocks, markdown fences and document wrappers from model output."""
    text = raw or ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    if "</think>" in text:
        text = text.split("</think>")[-1]
    fenced = re.search(r"```(?:html)?\s*(.*?)```", text, flags=re.S | re.I)
    if fenced:
        text = fenced.group(1)
    else:
        text = re.sub(r"^\s*```(?:html)?\s*", "", text, flags=re.I)
    body = re.search(r"<body[^>]*>(.*?)(</body>|$)", text, flags=re.S | re.I)
    if body:
        text = body.group(1)
    text = re.sub(r"<!DOCTYPE[^>]*>|</?html[^>]*>|<head>.*?</head>", "", text, flags=re.S | re.I)
    return text.strip()


# ── Ollama calls ──────────────────────────────────────────────────────────────

def _client(timeout: float):
    import httpx
    return httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=10.0))


async def _show_capabilities(client, server_url: str, name: str) -> Optional[List[str]]:
    try:
        resp = await client.post(f"{server_url}/api/show", json={"model": name})
        if resp.status_code != 200:
            return None
        caps = resp.json().get("capabilities")
        return list(caps) if isinstance(caps, list) else None
    except Exception:
        return None


async def list_models(server_url: str) -> Dict[str, Any]:
    """Installed Ollama models with vision / thinking capability flags."""
    import httpx

    async with _client(8.0) as client:
        try:
            resp = await client.get(f"{server_url}/api/tags")
            resp.raise_for_status()
            raw_models = resp.json().get("models", []) or []
        except (httpx.HTTPError, ValueError) as exc:
            return {
                "success": False,
                "server_url": server_url,
                "models": [],
                "error": f"Could not connect to Ollama at {server_url}: {exc}",
            }

        running: set = set()
        try:
            ps = await client.get(f"{server_url}/api/ps", timeout=3.0)
            if ps.status_code == 200:
                running = {m.get("name") for m in ps.json().get("models", []) if isinstance(m, dict)}
        except Exception:
            pass

        names = [m.get("name") for m in raw_models if isinstance(m, dict) and m.get("name")]
        caps_list = await asyncio.gather(*(_show_capabilities(client, server_url, n) for n in names[:40]))
        caps_by_name = dict(zip(names, caps_list))

    models = []
    for m in raw_models:
        name = m.get("name")
        if not name:
            continue
        caps = caps_by_name.get(name)
        details = m.get("details") or {}
        models.append({
            "name": name,
            "size": m.get("size"),
            "parameter_size": details.get("parameter_size"),
            "family": details.get("family"),
            "vision": ("vision" in caps) if caps is not None else None,
            "thinking": ("thinking" in caps) if caps is not None else None,
            "running": name in running,
            "is_chandra": "chandra" in name.lower(),
        })
    models.sort(key=lambda x: (not x["is_chandra"], x["vision"] is not True, x["name"].lower()))
    return {"success": True, "server_url": server_url, "models": models}


async def run_vlm_ocr(
    server_url: str,
    model: str,
    image,
    *,
    prompt_mode: str = "ocr_layout",
    max_tokens: int = DEFAULT_MAX_TOKENS,
    custom_prompt: Optional[str] = None,
    is_disconnected: Optional[Callable[[], Awaitable[bool]]] = None,
) -> Dict[str, Any]:
    """
    Run one page through a Chandra OCR model in Ollama and return the HTML.
    Raises VlmOcrError (with an HTTP status) or VlmOcrCancelled.
    """
    import httpx

    if _job_lock.locked():
        raise VlmOcrError(409, "Another experimental VLM OCR run is in progress. Try again when it finishes.")

    async with _job_lock:
        # Free GPU memory held by the production OCR engine if it is idle.
        try:
            from services.ocr_engine import unload_ocr_engine
            unload_ocr_engine()
        except Exception:
            pass

        prompt = (custom_prompt or "").strip() or PROMPT_MAPPING.get(prompt_mode, OCR_LAYOUT_PROMPT)
        image_tokens = math.ceil(image.width / GRID_SIZE) * math.ceil(image.height / GRID_SIZE)
        num_ctx = min(MAX_CONTEXT, int(math.ceil((image_tokens + max_tokens + 1024) / 1024.0) * 1024))

        started = time.perf_counter()
        async with _client(REQUEST_TIMEOUT_S) as client:
            caps = await _show_capabilities(client, server_url, model)
            payload: Dict[str, Any] = {
                "model": model,
                "messages": [{"role": "user", "content": prompt, "images": [encode_png_b64(image)]}],
                "stream": True,
                "options": {
                    "temperature": 0.0,
                    "top_p": 0.1,
                    "num_predict": int(max_tokens),
                    "num_ctx": num_ctx,
                },
            }
            if caps and "thinking" in caps:
                payload["think"] = False

            chunks: List[str] = []
            final: Dict[str, Any] = {}
            last_check = time.monotonic()
            try:
                async with client.stream("POST", f"{server_url}/api/chat", json=payload) as resp:
                    if resp.status_code >= 400:
                        body = (await resp.aread()).decode("utf-8", "replace")
                        try:
                            detail = json.loads(body).get("error") or body
                        except ValueError:
                            detail = body
                        status = 404 if resp.status_code == 404 else 502
                        raise VlmOcrError(status, f"Ollama returned {resp.status_code}: {detail.strip()[:500]}")
                    async for line in resp.aiter_lines():
                        if not line.strip():
                            continue
                        if is_disconnected and time.monotonic() - last_check > 1.0:
                            last_check = time.monotonic()
                            if await is_disconnected():
                                raise VlmOcrCancelled()
                        chunk = json.loads(line)
                        if chunk.get("error"):
                            raise VlmOcrError(502, f"Ollama error: {chunk['error']}")
                        chunks.append((chunk.get("message") or {}).get("content", ""))
                        if chunk.get("done"):
                            final = chunk
                            break
            except httpx.ConnectError as exc:
                raise VlmOcrError(502, f"Could not connect to Ollama at {server_url}. Is Ollama running? ({exc})")
            except httpx.TimeoutException:
                raise VlmOcrError(504, f"Ollama did not finish within {REQUEST_TIMEOUT_S // 60} minutes.")
            except httpx.HTTPError as exc:
                raise VlmOcrError(502, f"Ollama request failed: {exc}")

        raw = "".join(chunks)
        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.info(
            f"[VLM-OCR experimental] model={model} image={image.width}x{image.height} "
            f"tokens={final.get('eval_count')} done_reason={final.get('done_reason')} "
            f"elapsed={elapsed_ms / 1000:.1f}s"
        )
        return {
            "html": clean_html(raw),
            "raw": raw,
            "model": model,
            "prompt_mode": "custom" if (custom_prompt or "").strip() else prompt_mode,
            "elapsed_ms": round(elapsed_ms),
            "prompt_tokens": final.get("prompt_eval_count"),
            "output_tokens": final.get("eval_count"),
            "done_reason": final.get("done_reason"),
            "truncated": final.get("done_reason") == "length",
            "num_ctx": num_ctx,
            "vision_capable": ("vision" in caps) if caps is not None else None,
        }
