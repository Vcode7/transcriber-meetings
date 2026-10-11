"""OCR Settings API: settings CRUD + validation, status, OCR test endpoint, experimental VLM endpoints."""
import io
import json
from contextlib import asynccontextmanager

import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image, ImageDraw, ImageFont
from sqlalchemy import text

from database import connect_db, get_db_context
from main import app
from routers.auth import get_current_user
from services import ocr_settings as S


def _font(size=30):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _text_png(lines=("Budget review approved", "Launch planned for March")) -> bytes:
    img = Image.new("RGB", (900, 260), "white")
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        draw.text((40, 50 + i * 80), line, fill="black", font=_font())
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


@asynccontextmanager
async def _api():
    """Client with auth bypassed; the single system-wide settings row is restored afterwards."""
    await connect_db()
    async with get_db_context() as db:
        row = (await db.execute(text(
            "SELECT config_json, updated_at, updated_by FROM ocr_settings WHERE id = 1"
        ))).mappings().fetchone()
        snapshot = dict(row) if row else None
    app.dependency_overrides[get_current_user] = lambda: {"id": "test_ocr_user", "username": "test"}
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            yield client
    finally:
        app.dependency_overrides.pop(get_current_user, None)
        async with get_db_context() as db:
            await db.execute(text("DELETE FROM ocr_settings WHERE id = 1"))
            if snapshot:
                await db.execute(
                    text("INSERT INTO ocr_settings (id, config_json, updated_at, updated_by) "
                         "VALUES (1, :config_json, :updated_at, :updated_by)"),
                    snapshot,
                )
            await db.commit()
        S.invalidate_cache()


@pytest.mark.asyncio
async def test_settings_round_trip_and_validation():
    async with _api() as client:
        res = await client.get("/ocr/settings")
        assert res.status_code == 200
        body = res.json()
        assert {"settings", "defaults", "presets", "schema", "updated_at"} <= set(body)
        assert body["defaults"]["det_limit_type"] == "min"
        assert any(p["id"] == "legacy" for p in body["presets"])

        new = {**body["defaults"], "pdf_render_dpi": 300, "deskew": True}
        res = await client.put("/ocr/settings", json=new)
        assert res.status_code == 200 and res.json()["settings"]["pdf_render_dpi"] == 300
        # The process-wide cache is refreshed, so every OCR entry point uses the new values
        assert S.get_ocr_settings().pdf_render_dpi == 300 and S.get_ocr_settings().deskew

        res = await client.put("/ocr/settings", json={**new, "det_db_thresh": 5})
        assert res.status_code == 422
        assert any("det_db_thresh" in err["loc"] for err in res.json()["detail"])

        res = await client.put("/ocr/settings", json={**new, "not_a_setting": 1})
        assert res.status_code == 422

        res = await client.put("/ocr/settings", json={**new, "binarize_block_size": 30})
        assert res.status_code == 422
        assert S.get_ocr_settings().pdf_render_dpi == 300  # rejected saves change nothing

        res = await client.post("/ocr/settings/reset")
        assert res.status_code == 200 and res.json()["settings"]["pdf_render_dpi"] == 200
        assert S.get_ocr_settings() == S.OcrSettings()


@pytest.mark.asyncio
async def test_status_never_loads_the_engine():
    from services import ocr_engine

    ocr_engine.unload_ocr_engine()
    async with _api() as client:
        res = await client.get("/ocr/status")
    assert res.status_code == 200
    body = res.json()
    assert body["models_present"] is True and body["loaded"] is False
    assert ocr_engine._OCR_ENGINE is None
    assert body["settings_fingerprint"].startswith("cfg=")


@pytest.mark.asyncio
async def test_ocr_test_endpoint_uses_unsaved_settings_without_saving_them():
    async with _api() as client:
        before = S.settings_fingerprint(S.get_ocr_settings())
        unsaved = {**S.OcrSettings().model_dump(), "det_limit_type": "max", "det_limit_side_len": 960}
        res = await client.post(
            "/ocr/test",
            files={"file": ("sample.png", _text_png(), "image/png")},
            data={"settings": json.dumps(unsaved), "show_boxes": "true"},
        )
        assert res.status_code == 200, res.text
        body = res.json()
        assert body["used_unsaved_settings"] is True and body["kind"] == "image"
        assert "Budget" in body["text"] and "March" in body["text"]
        page = body["pages"][0]
        assert page["source"] == "ocr" and page["line_count"] == 2
        assert page["preview"].startswith("data:image/jpeg;base64,")
        assert page["det_input_size"][0] <= 960
        assert S.settings_fingerprint(S.get_ocr_settings()) == before


@pytest.mark.asyncio
async def test_ocr_test_endpoint_handles_scanned_pdf():
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_image(fitz.Rect(40, 40, 555, 200), stream=_text_png())
    pdf = doc.tobytes()
    doc.close()

    async with _api() as client:
        res = await client.post(
            "/ocr/test",
            files={"file": ("scan.pdf", pdf, "application/pdf")},
            data={"max_pages": "3", "show_boxes": "false"},
        )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["kind"] == "pdf" and body["page_count"] == 1
    assert body["pages"][0]["source"] == "ocr" and body["pages"][0]["render_dpi"] == 200
    assert body["pages"][0].get("preview") is None
    assert "Budget" in body["text"]


@pytest.mark.asyncio
async def test_ocr_test_endpoint_rejects_bad_input():
    async with _api() as client:
        res = await client.post("/ocr/test", files={"file": ("notes.txt", b"hello", "text/plain")})
        assert res.status_code == 400

        res = await client.post("/ocr/test", files={"file": ("fake.png", b"not really a png", "image/png")})
        assert res.status_code == 400

        res = await client.post(
            "/ocr/test",
            files={"file": ("sample.png", _text_png(), "image/png")},
            data={"settings": json.dumps({"det_db_thresh": 9})},
        )
        assert res.status_code == 422
        assert res.json()["detail"][0]["loc"][:2] == ["settings", "det_db_thresh"]


@pytest.mark.asyncio
async def test_vlm_endpoints_delegate_to_experimental_service(monkeypatch):
    from services import vlm_ocr_experimental as vlm

    async def fake_list(server_url):
        return {"success": True, "server_url": server_url,
                "models": [{"name": "chandra-ocr-2", "vision": True, "thinking": False,
                            "running": False, "is_chandra": True}]}

    async def fake_run(server_url, model, image, **kwargs):
        assert kwargs["prompt_mode"] == "ocr" and kwargs["max_tokens"] == 2048
        return {"html": "<p>Budget</p>", "raw": "<p>Budget</p>", "model": model, "prompt_mode": "ocr",
                "elapsed_ms": 5, "truncated": False, "num_ctx": 8192, "vision_capable": True}

    monkeypatch.setattr(vlm, "list_models", fake_list)
    monkeypatch.setattr(vlm, "run_vlm_ocr", fake_run)

    async with _api() as client:
        res = await client.get("/ocr/vlm/models")
        assert res.status_code == 200 and res.json()["models"][0]["is_chandra"]

        res = await client.post(
            "/ocr/vlm/test",
            files={"file": ("page.png", _text_png(), "image/png")},
            data={"model": "chandra-ocr-2", "prompt_mode": "ocr", "max_tokens": "2048"},
        )
        assert res.status_code == 200, res.text
        body = res.json()
        assert body["html"] == "<p>Budget</p>" and body["page_count"] == 1
        assert body["source_preview"].startswith("data:image/jpeg;base64,")
        assert body["image_size"][0] % 28 == 0

        res = await client.post(
            "/ocr/vlm/test",
            files={"file": ("page.png", _text_png(), "image/png")},
            data={"model": "chandra-ocr-2", "prompt_mode": "markdown"},
        )
        assert res.status_code == 400

        res = await client.post(
            "/ocr/vlm/test",
            files={"file": ("page.png", _text_png(), "image/png")},
            data={"model": "chandra-ocr-2", "max_tokens": "100"},
        )
        assert res.status_code == 400
