"""PPT Testing endpoints end-to-end with a scripted LLM: upload → insights → compare → final → exports."""
import json
import re
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text

import routers.rom_router as rom_router
from database import connect_db, get_db_context
from main import app
from ppt_fixtures import LONG_POINTS, build_pptx
from routers.auth import get_current_user

USER_ID = "test_ppt_user"


def scripted_infer(prompt, max_tokens):
    """Deterministic stand-in for the LLM covering every PPT prompt type."""
    slide = re.search(r"^SLIDE (\d+) —", prompt, re.MULTILINE)
    if slide:
        n = int(slide.group(1))
        data = {
            4: {"project": "Project Atlas", "insights": [{
                "type": "delay", "status": "delayed", "headline": "Data migration dry-run 2 slipped",
                "summary": "Dry-run 2 moved from 15-Sep-2026 to 30-Oct-2026 because of vendor data quality issues.",
                "attention": "Clean customer master data", "dates": ["30-Oct-2026"]}]},
            5: {"project": "Project Beacon", "insights": [{
                "type": "dependency", "status": "planned", "headline": "Store submission depends on Atlas API v2",
                "summary": "Submission planned for 15-Nov-2026 needs Atlas API v2 by 01-Nov-2026.",
                "attention": "", "dates": ["15-Nov-2026"]}]},
        }.get(n, {"project": "", "insights": []})
        return json.dumps(data)
    if "DISCUSSION POINT (from the meeting transcript" in prompt:
        discussion = re.search(r'DISCUSSION POINT[^\n]*\n"(.*?)"', prompt, re.S).group(1)
        refs = re.findall(r"^(P\d+) \[Slide (\d+)", prompt, re.M)
        wanted = "4" if "migration" in discussion else "5" if "Beacon" in discussion else None
        return json.dumps({"matches": [{"id": r, "confidence": "high", "reason": "same topic"}
                                       for r, s in refs if s == wanted]})
    if "Write one meeting-prep point" in prompt:
        return json.dumps({"headline": "Combined point", "status": "delayed",
                           "summary": "Slides and discussion agree on the slip to 30-Oct-2026.", "attention": "Data clean-up"})
    if '"needs_attention"' in prompt:
        return json.dumps({"summary": "Project summary.", "needs_attention": []})
    return "Phase 3 is under way; the data migration slip needs attention."


@asynccontextmanager
async def _api(monkeypatch, tmp_path):
    await connect_db()
    rec_id = f"test_ppt_{uuid.uuid4().hex[:8]}"
    rom_data = {"stage2": {"polished_points": [{"id": lp["id"], "polished_text": lp["text"]} for lp in LONG_POINTS]}}
    async with get_db_context() as db:
        await db.execute(
            text("INSERT INTO recordings (id, user_id, filename, file_path, status, transcript, rom_data, created_at) "
                 "VALUES (:id, :uid, 'review.wav', 'x', 'done', '[]', :rom, :created)"),
            {"id": rec_id, "uid": USER_ID, "rom": json.dumps(rom_data),
             "created": datetime(2026, 10, 12, 10, 0, tzinfo=timezone.utc).isoformat()},
        )
        await db.commit()
    monkeypatch.setattr(rom_router, "_ppt_infer_fn", lambda: scripted_infer)

    def no_embeddings(texts):
        raise RuntimeError("embedder disabled in tests")
    monkeypatch.setattr("services.text_embedding_service.embed_chunks", no_embeddings)
    app.dependency_overrides[get_current_user] = lambda: {"id": USER_ID, "username": "test"}
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test", timeout=120) as client:
            yield client, rec_id
    finally:
        app.dependency_overrides.pop(get_current_user, None)
        async with get_db_context() as db:
            await db.execute(text("DELETE FROM rom_metadata WHERE recording_id = :id"), {"id": rec_id})
            await db.execute(text("DELETE FROM recordings WHERE id = :id"), {"id": rec_id})
            await db.commit()


@pytest.mark.asyncio
async def test_ppt_testing_pipeline(monkeypatch, tmp_path):
    deck = build_pptx(str(tmp_path / "atlas_review.pptx"))
    async with _api(monkeypatch, tmp_path) as (client, rec_id):
        with open(deck, "rb") as fh:
            res = await client.post(f"/rom/{rec_id}/ppt-testing/upload",
                                    files={"file": ("atlas_review.pptx", fh.read(),
                                                    "application/vnd.openxmlformats-officedocument.presentationml.presentation")})
        assert res.status_code == 200, res.text
        assert res.json()["slide_count"] == 13

        res = await client.post(f"/rom/{rec_id}/ppt-testing/generate-points", json={"meeting_date": "2026-13-40"})
        assert res.status_code == 400                                       # invalid date rejected

        # No date given → defaults to the recording date (2026-10-12)
        res = await client.post(f"/rom/{rec_id}/ppt-testing/generate-points", json={})
        assert res.status_code == 200, res.text
        body = res.json()
        assert body["meeting_date"] == "2026-10-12"
        assert body["cutoff"]["slide"] == 10
        analysis = body["analysis"]
        assert [i["slides"] for i in analysis["insights"]] == [[4], [5]]
        assert analysis["timeline"]["current_phase"][0]["label"].startswith("Phase 3")

        res = await client.get(f"/rom/{rec_id}/ppt-testing/progress")
        assert res.json() == {"running": False}

        res = await client.post(f"/rom/{rec_id}/ppt-testing/compare")
        assert res.status_code == 200, res.text
        body = res.json()
        groups = {g["stage2_point"]["id"]: g for g in body["comparison_groups"]}
        assert set(groups) == {"lp1", "lp2"}                                 # unrelated points discarded
        assert groups["lp1"]["slides"] == [4] and groups["lp2"]["slides"] == [5]
        assert body["stats"]["semantic_similarity"] is False                 # lexical fallback without embedder
        finals = body["final_points"]
        assert finals[0]["is_presentation_summary"] and len(finals) == 3
        assert all(p["slides"] for p in finals[1:])

        # Regenerating the final text keeps ids and selections
        await client.put(f"/rom/{rec_id}/ppt-testing/toggle-points", json={"toggles": {finals[1]["id"]: False}})
        res = await client.post(f"/rom/{rec_id}/ppt-testing/generate-final")
        assert res.status_code == 200
        regenerated = res.json()["final_points"]
        assert [p["id"] for p in regenerated] == [p["id"] for p in finals]
        assert regenerated[1]["enabled"] is False

        res = await client.post(f"/rom/{rec_id}/ppt-testing/export-docx",
                                json={"selected_point_ids": [p["id"] for p in regenerated]})
        assert res.status_code == 200 and res.content[:2] == b"PK"
        res = await client.get(f"/rom/{rec_id}/ppt-testing/export-insights-docx")
        assert res.status_code == 200 and res.content[:2] == b"PK"


@pytest.mark.asyncio
async def test_meeting_date_changes_what_is_relevant(monkeypatch, tmp_path):
    """A 2019 meeting date makes the 2019–2022 history slide relevant and today's slides far future."""
    deck = build_pptx(str(tmp_path / "atlas_review.pptx"))
    async with _api(monkeypatch, tmp_path) as (client, rec_id):
        with open(deck, "rb") as fh:
            await client.post(f"/rom/{rec_id}/ppt-testing/upload", files={"file": ("atlas_review.pptx", fh.read(), "application/octet-stream")})
        res = await client.post(f"/rom/{rec_id}/ppt-testing/generate-points",
                                json={"meeting_date": "2020-06-15", "focus": "narrow"})
        assert res.status_code == 200, res.text
        status = {s["number"]: s["status"] for s in res.json()["analysis"]["slides"]}
        assert status[6] == "analyzed"                      # Programme History (2019–2022)
        assert status[4] == "skipped_out_of_window"         # 2026 build status
        res = await client.post(f"/rom/{rec_id}/ppt-testing/generate-points", json={"focus": "huge"})
        assert res.status_code == 400
