"""Experimental Chandra OCR 2 via Ollama: image preparation, request shape, output cleanup, isolation."""
import io
import json
import re
from pathlib import Path

import httpx
import pytest
from PIL import Image

from services import vlm_ocr_experimental as vlm


def _png(size=(800, 600)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, "white").save(buf, "PNG")
    return buf.getvalue()


def _mock_client(handler):
    def factory(timeout):
        return httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=timeout)
    return factory


def test_scale_to_fit_uses_28px_grid_and_pixel_budget():
    out = vlm.scale_to_fit(Image.new("RGB", (5000, 4000)))
    assert out.width % 28 == 0 and out.height % 28 == 0
    assert out.width * out.height <= 3072 * 2048


def test_prepare_image_upscales_small_images_like_chandra():
    image, pages = vlm.prepare_image(_png((800, 600)), "scan.png")
    assert pages == 1
    assert min(image.size) >= 1500 and image.width % 28 == 0


def test_prepare_image_renders_requested_pdf_page():
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    for _ in range(2):
        doc.new_page(width=595, height=842)
    data = doc.tobytes()
    doc.close()

    image, pages = vlm.prepare_image(data, "doc.pdf", page=2)
    assert pages == 2 and image.height > image.width
    with pytest.raises(vlm.VlmOcrError) as exc:
        vlm.prepare_image(data, "doc.pdf", page=3)
    assert exc.value.status_code == 400


def test_clean_html_strips_thinking_fences_and_wrappers():
    raw = "<think>reasoning</think>```html\n<html><body><div data-label=\"Text\"><p>Hi</p></div></body></html>\n```"
    assert vlm.clean_html(raw) == '<div data-label="Text"><p>Hi</p></div>'
    assert vlm.clean_html("```html\n<p>cut off") == "<p>cut off"


def test_prompts_match_chandra():
    assert vlm.OCR_LAYOUT_PROMPT.startswith("OCR this image to HTML, arranged as layout blocks.")
    assert "Blank-Page" in vlm.OCR_LAYOUT_PROMPT
    assert vlm.OCR_PROMPT.startswith("OCR this image to HTML.")
    assert vlm.PROMPT_ENDING in vlm.OCR_PROMPT


@pytest.mark.asyncio
async def test_run_streams_chat_and_reports_truncation(monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"capabilities": ["completion", "vision"]})
        assert request.url.path == "/api/chat"
        seen["payload"] = json.loads(request.content)
        lines = [
            {"message": {"content": "```html\n<p>Hello"}, "done": False},
            {"message": {"content": " world</p>\n```"}, "done": False},
            {"message": {"content": ""}, "done": True, "done_reason": "length",
             "eval_count": 42, "prompt_eval_count": 900},
        ]
        return httpx.Response(200, content="\n".join(json.dumps(x) for x in lines).encode())

    monkeypatch.setattr(vlm, "_client", _mock_client(handler))
    image, _ = vlm.prepare_image(_png(), "page.png")
    result = await vlm.run_vlm_ocr("http://ollama.test", "chandra-ocr-2", image, max_tokens=1024)

    payload = seen["payload"]
    assert payload["model"] == "chandra-ocr-2" and payload["stream"] is True
    assert payload["messages"][0]["content"] == vlm.OCR_LAYOUT_PROMPT
    assert len(payload["messages"][0]["images"]) == 1
    assert payload["options"]["temperature"] == 0.0 and payload["options"]["num_predict"] == 1024
    assert "think" not in payload  # model does not report the thinking capability
    assert result["html"] == "<p>Hello world</p>"
    assert result["truncated"] is True and result["output_tokens"] == 42
    assert result["vision_capable"] is True


@pytest.mark.asyncio
async def test_run_maps_ollama_errors(monkeypatch):
    def missing_model(request):
        if request.url.path == "/api/show":
            return httpx.Response(404, json={"error": "model not found"})
        return httpx.Response(404, json={"error": "model 'x' not found"})

    monkeypatch.setattr(vlm, "_client", _mock_client(missing_model))
    image, _ = vlm.prepare_image(_png(), "page.png")
    with pytest.raises(vlm.VlmOcrError) as exc:
        await vlm.run_vlm_ocr("http://ollama.test", "x", image)
    assert exc.value.status_code == 404 and "not found" in exc.value.message

    def unreachable(request):
        raise httpx.ConnectError("refused", request=request)

    monkeypatch.setattr(vlm, "_client", _mock_client(unreachable))
    with pytest.raises(vlm.VlmOcrError) as exc:
        await vlm.run_vlm_ocr("http://ollama.test", "x", image)
    assert exc.value.status_code == 502 and "Is Ollama running" in exc.value.message


@pytest.mark.asyncio
async def test_list_models_flags_vision_and_chandra(monkeypatch):
    def handler(request):
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [
                {"name": "llama3:8b", "size": 1, "details": {"parameter_size": "8B"}},
                {"name": "fredrezones55/chandra-ocr-2:patch", "size": 2, "details": {}},
            ]})
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": [{"name": "llama3:8b"}]})
        body = json.loads(request.content)
        caps = ["completion", "vision"] if "chandra" in body["model"] else ["completion"]
        return httpx.Response(200, json={"capabilities": caps})

    monkeypatch.setattr(vlm, "_client", _mock_client(handler))
    res = await vlm.list_models("http://ollama.test")
    assert res["success"]
    first, second = res["models"]
    assert first["is_chandra"] and first["vision"] is True
    assert second["name"] == "llama3:8b" and second["vision"] is False and second["running"] is True


def test_experimental_module_is_isolated_from_production_code():
    """Only the OCR router may import the experimental VLM module."""
    backend = Path(__file__).resolve().parent.parent
    importers = set()
    for folder in ("routers", "services", "tasks", "utils"):
        for path in (backend / folder).rglob("*.py"):
            if path.name == "vlm_ocr_experimental.py":
                continue
            if re.search(r"vlm_ocr_experimental", path.read_text(encoding="utf-8", errors="ignore")):
                importers.add(path.name)
    assert importers == {"ocr_router.py"}
