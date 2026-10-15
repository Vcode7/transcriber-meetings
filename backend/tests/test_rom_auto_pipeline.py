"""
Automatic ROM pipeline (tasks/rom_auto_pipeline.py) started when transcription finishes, with the
settings chosen on the capture pages (routers/rom_pipeline_router.py). Runs through the real ROM
route handlers and DB; only the LLM/embedding-heavy service calls are scripted.

The three end-to-end cases: MoM without agenda, ROM with an agenda document, ROM with a manually
edited agenda.
"""
import copy
import json
import re
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text

import routers.rom_router as rom_router
import services.text_embedding_service as tes
import tasks.rom_auto_pipeline as auto
from config import settings as app_settings
from database import connect_db, get_db_context
from main import app
from ppt_fixtures import build_pptx
from routers.audio import _rom_settings_columns
from routers.auth import get_current_user
from services.rom_service import RomService, normalize_agendas
from test_agenda_doc_points import BowEmbedder
from test_ppt_testing_api import scripted_infer as ppt_infer

USER_ID = f"test_rom_auto_{uuid.uuid4().hex[:6]}"


def scripted_infer(prompt, max_tokens):
    """PPT Testing's scripted LLM, plus the long-ROM refinement (keeps the discussion, adds slide facts)."""
    if "Refine one discussion point" in prompt:
        discussion = re.search(r'MEETING DISCUSSION POINT[^\n]*\n"(.*?)"', prompt, re.S).group(1)
        return json.dumps({"headline": "Refined", "status": "delayed", "attention": "",
                           "summary": discussion + " The slides confirm the slip to 30-Oct-2026."})
    return ppt_infer(prompt, max_tokens)


TRANSCRIPT = [{"start": 0.0, "end": 30.0, "speaker": "SPEAKER_00",
               "text": "The Atlas data migration dry-run 2 slipped because of vendor data quality issues."},
              {"start": 30.0, "end": 60.0, "speaker": "SPEAKER_01",
               "text": "Beacon store submission is planned once the API is available."}]
S1_POINTS = [{"id": "s1-1", "point": "Atlas data migration dry-run 2 slipped", "window_index": 0},
             {"id": "s1-2", "point": "Beacon store submission planned", "window_index": 0}]
S2_POINTS = [{"id": "p1", "polished_text": "Ravi said the Atlas data migration dry-run 2 slipped because of vendor data quality issues."},
             {"id": "p2", "polished_text": "The Beacon store submission is planned once the Atlas API v2 is available."}]
DOC_AGENDAS = [{"title": "Project Atlas – Build Status", "description": "SAP integration and data migration"},
               {"title": "Project Beacon – Mobile App", "description": "Store submission"}]


class Calls(dict):
    def add(self, name, value=True):
        self.setdefault(name, []).append(value)


@pytest.fixture
def env(monkeypatch, tmp_path):
    calls = Calls()
    svc = rom_router.rom_service
    monkeypatch.setattr(app_settings, "UPLOAD_DIR", str(tmp_path))

    def extract(transcript, window, user_id, **kw):
        calls.add("stage1", window)
        return {"discussion_points": copy.deepcopy(S1_POINTS), "windows_processed": 1}

    def enhance(points, recording_id, user_id, meeting_top_k, global_top_k, window, **kw):
        calls.add("stage2", {"meeting_top_k": meeting_top_k, "global_top_k": global_top_k, "window": window, **kw})
        return copy.deepcopy(S2_POINTS)

    def agendas_only(**kw):
        calls.add("create_agenda", kw)
        agendas = normalize_agendas(kw.get("provided_agendas") or [{"title": "General Discussion"}],
                                    merge_duplicates=False)
        return {"agendas": agendas, "expanded_agendas": [], "retrieved_context_chunks": {},
                "discussion_analysis": None, "suggested_agendas": []}

    def mom(**kw):
        calls.add("mom", [p["id"] for p in kw["polished_points"]])
        return {"title": "MoM", "points_discussed": [p["polished_text"] for p in kw["polished_points"]],
                "action_items": [], "participants": []}

    def assign(self, **kw):
        calls.add("assign", kw)
        agendas = kw["agendas"]
        out = []
        for p in kw["polished_points"]:
            target = next((a for a in agendas if "Beacon" in p["polished_text"] and "Beacon" in a["title"]), agendas[0])
            out.append({"point_id": p["id"], "assigned_agenda_id": target["agenda_id"], "confidence": "high"})
        return {"batch_assignments": out, "candidate_results": [], "similarity_matrix": [],
                "similarity_agenda_ids": [a["agenda_id"] for a in agendas]}

    async def no_sync(*a, **k):
        return None

    provider = MagicMock()
    provider._infer.side_effect = lambda prompt, max_tokens, *_: scripted_infer(prompt, max_tokens)
    provider.generate_rom_agendas.return_value = {"agendas": copy.deepcopy(DOC_AGENDAS)}

    monkeypatch.setattr(svc, "extract_discussion_points", extract)
    monkeypatch.setattr(svc, "enhance_discussion_points", enhance)
    monkeypatch.setattr(svc, "generate_agendas_only", agendas_only)
    monkeypatch.setattr(svc, "generate_mom_from_enhanced_rom", mom)
    monkeypatch.setattr(RomService, "_assign_points_to_agendas", assign)
    monkeypatch.setattr(rom_router, "_ppt_infer_fn", lambda: scripted_infer)
    monkeypatch.setattr(rom_router, "_sync_stage2_embeddings", no_sync)
    monkeypatch.setattr(rom_router, "_sync_final_rom_embeddings", no_sync)
    monkeypatch.setattr(auto, "_stage2_examples", lambda: [])
    monkeypatch.setattr(auto, "_RUN_LOCK", None)          # each test has its own event loop
    monkeypatch.setattr("services.ai_provider.get_provider", lambda: provider)
    monkeypatch.setattr(tes, "get_text_embedder", lambda: BowEmbedder())
    monkeypatch.setattr(tes, "unload_text_embedder", lambda: None)
    monkeypatch.setattr(tes, "embed_chunks", lambda texts: BowEmbedder().encode_batch(texts))
    monkeypatch.setattr("services.rag_pipeline.embed_meeting_context", lambda rid, uid: calls.add("embed_context", rid) or 3)
    monkeypatch.setattr("services.rag_pipeline._mark_meeting_context_embedded", lambda rid, uid: None)
    return calls


@asynccontextmanager
async def _client():
    await connect_db()
    app.dependency_overrides[get_current_user] = lambda: {"id": USER_ID, "username": "test"}
    created = []

    async def make_recording(settings: dict, transcript=TRANSCRIPT):
        rec_id = f"test_rom_auto_{uuid.uuid4().hex[:8]}"
        cols = _rom_settings_columns(json.dumps(settings))
        async with get_db_context() as db:
            await db.execute(text(
                "INSERT INTO recordings (id, user_id, filename, file_path, status, transcript, created_at, "
                "rom_settings, rom_pipeline_status) VALUES (:id, :uid, 'meeting.wav', 'x', 'processing', :tr, :created, :rs, :ps)"),
                {"id": rec_id, "uid": USER_ID, "tr": json.dumps(transcript),
                 "created": datetime(2026, 10, 12, 10, 0, tzinfo=timezone.utc).isoformat(),
                 "rs": cols["rom_settings"], "ps": cols["rom_pipeline_status"]})
            await db.commit()
        created.append(rec_id)
        return rec_id

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test", timeout=120) as client:
            yield client, make_recording
    finally:
        app.dependency_overrides.pop(get_current_user, None)
        async with get_db_context() as db:
            for rec_id in created:
                for table, col in (("rom_metadata", "recording_id"), ("minutes_of_meeting", "recording_id"),
                                   ("recording_attachments", "recording_id"), ("recordings", "id")):
                    await db.execute(text(f"DELETE FROM {table} WHERE {col} = :id"), {"id": rec_id})
            await db.execute(text("DELETE FROM user_settings WHERE user_id = :uid"), {"uid": USER_ID})
            await db.commit()


async def _run_after_transcription(client, rec_id):
    """Transcription reaches 'done', then tasks/pipeline.py calls maybe_start."""
    async with get_db_context() as db:
        await db.execute(text("UPDATE recordings SET status = 'done' WHERE id = :id"), {"id": rec_id})
        await db.commit()
    assert await auto.maybe_start(rec_id, USER_ID)
    await auto._TASKS[rec_id]
    res = await client.get(f"/rom-pipeline/{rec_id}")
    assert res.status_code == 200, res.text
    rom = (await client.get(f"/rom/{rec_id}")).json()["rom_data"]
    return res.json(), rom


async def _mom_rows(rec_id):
    async with get_db_context() as db:
        r = await db.execute(text("SELECT COUNT(*) FROM minutes_of_meeting WHERE recording_id = :id"), {"id": rec_id})
        return r.scalar()


# ── Case 1: MoM without agenda ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_mom_without_agenda(env):
    async with _client() as (client, make_recording):
        res = await client.post("/rom-pipeline/context-files", files={"file": ("brief.txt", b"Atlas programme brief", "text/plain")})
        assert res.status_code == 200, res.text
        staged = res.json()

        rec_id = await make_recording({"output_type": "mom", "transcript_window_minutes": 3, "meeting_top_k": 7,
                                       "previous_meeting_mode": "off", "context_files": [staged]})
        assert await auto.replaces_direct_mom(rec_id)          # transcription skips its direct MoM
        before = (await client.get(f"/rom-pipeline/{rec_id}")).json()["status"]
        assert before["state"] == "waiting"

        pipeline, rom = await _run_after_transcription(client, rec_id)

        assert pipeline["status"]["state"] == "done", pipeline
        assert env["stage1"] == [3.0]
        assert env["stage2"][0]["meeting_top_k"] == 7 and env["stage2"][0]["previous_meeting_mode"] == "off"
        assert env["embed_context"] == [rec_id]                 # context attached and embedded before Stage 2
        assert env["mom"] == [["p1", "p2"]]                     # MoM from the Stage 2 points
        assert "create_agenda" not in env and "stage3" in rom and "final_rom" not in rom
        assert await _mom_rows(rec_id) == 1
        async with get_db_context() as db:
            r = await db.execute(text("SELECT filename FROM recording_attachments WHERE recording_id = :id AND type = 'context'"),
                                 {"id": rec_id})
            assert [row[0] for row in r.fetchall()] == ["brief.txt"]


# ── Case 2: ROM with an agenda document ───────────────────────────────────────

@pytest.mark.asyncio
async def test_rom_with_agenda_document(env, tmp_path):
    deck = build_pptx(str(tmp_path / "agenda.pptx"))
    async with _client() as (client, make_recording):
        with open(deck, "rb") as fh:
            res = await client.post("/rom-pipeline/agenda-document", files={"file": ("agenda.pptx", fh.read(),
                                    "application/vnd.openxmlformats-officedocument.presentationml.presentation")})
        assert res.status_code == 200, res.text
        extracted = res.json()
        assert extracted["slide_count"] == 13
        assert [a["title"] for a in extracted["agendas"]] == [a["title"] for a in DOC_AGENDAS]

        rec_id = await make_recording({
            "output_type": "rom", "agendas": extracted["agendas"], "agenda_text": extracted["agenda_text"],
            "agenda_doc": {"staged_id": extracted["staged_id"], "filename": extracted["filename"]},
        })
        pipeline, rom = await _run_after_transcription(client, rec_id)

        assert pipeline["status"]["state"] == "done", pipeline
        assert env["create_agenda"][0]["provided_agendas"] == extracted["agendas"]
        assert env["create_agenda"][0]["agenda_from_tables"] is True  # "Extract Agenda from Tables" default
        assert rom["agenda_documents"][0]["filename"] == "agenda.pptx"
        assert rom["agenda_documents"][0]["analysis"]["insights"]          # PPT analysis cached on the document
        assert rom["stage3"]["agenda_document_insights"]["total_doc_points"] == 2
        # doc points are mapping context ...
        assert env["assign"][0]["reference_doc_points"]
        # ... and refine the matched points of the long ROM
        final_points = [p for a in rom["final_rom"]["agendas"] for p in a["discussion_points"]]
        assert any(p.get("doc_refined") for p in final_points)
        assert rom["final_rom_active_version"] == "long"
        assert env["mom"] == [["p1", "p2"]]                     # generate_mom_auto is on by default
        from services.rom_staging import staged_dir
        assert staged_dir(USER_ID, extracted["staged_id"]) is None  # staging cleaned up


@pytest.mark.asyncio
async def test_agenda_document_extract_from_tables_option(env, tmp_path):
    """Agenda items come from the deck's table first; toggling the option re-reads the staged text."""
    from services.ai_provider import get_provider  # the scripted provider of the env fixture

    provider = get_provider()
    deck = build_pptx(str(tmp_path / "agenda.pptx"))
    async with _client() as (client, _make_recording):
        with open(deck, "rb") as fh:
            res = await client.post("/rom-pipeline/agenda-document", files={"file": ("agenda.pptx", fh.read(),
                                    "application/vnd.openxmlformats-officedocument.presentationml.presentation")})
        assert res.status_code == 200, res.text
        staged = res.json()
        assert (staged["extraction_mode"], staged["table_count"]) == ("tables", 1)
        args, kwargs = provider.generate_rom_agendas.call_args
        assert kwargs["source"] == "tables" and args[0].startswith("[Table (Slide")
        assert "Project Beacon" not in args[0]  # slide titles and bullets are not sent

        res = await client.post(f"/rom-pipeline/agenda-document/{staged['staged_id']}/agendas",
                                json={"from_tables": False})
        assert res.status_code == 200, res.text
        assert res.json()["extraction_mode"] == "document"
        assert [a["title"] for a in res.json()["agendas"]] == [a["title"] for a in DOC_AGENDAS]
        args, kwargs = provider.generate_rom_agendas.call_args
        assert kwargs["source"] == "document" and args[0] == staged["agenda_text"]

        res = await client.post("/rom-pipeline/agenda-document/0123456789abcdef/agendas", json={"from_tables": True})
        assert res.status_code == 404
        await client.delete(f"/rom-pipeline/staged/{staged['staged_id']}")


# ── Case 3: ROM with a manually edited agenda ─────────────────────────────────

@pytest.mark.asyncio
async def test_rom_with_manually_edited_agenda(env):
    edited = [
        {"agenda_id": "A2", "title": "Beacon launch readiness", "description": "Renamed by the user"},
        {"agenda_id": "A1", "title": "Project Atlas – Build Status", "description": ""},
        {"agenda_id": "A3", "title": "Risks and decisions", "description": "Added by hand", "presenter": "Priya"},
    ]
    async with _client() as (client, make_recording):
        rec_id = await make_recording({"output_type": "rom", "agendas": edited})
        pipeline, rom = await _run_after_transcription(client, rec_id)

        assert pipeline["status"]["state"] == "done", pipeline
        assert [(a["agenda_id"], a["title"]) for a in rom["stage3"]["agendas"]] == [
            ("A2", "Beacon launch readiness"), ("A1", "Project Atlas – Build Status"), ("A3", "Risks and decisions")]
        assert [a["agenda_id"] for a in rom["final_rom"]["agendas"]] == ["A2", "A1", "A3"]
        assert rom["stage3"]["agendas"][2]["presenter"] == "Priya"
        assert "agenda_documents" not in rom and "agenda_document_insights" not in rom["stage3"]
        assert not any(p.get("doc_refined") for a in rom["final_rom"]["agendas"] for p in a["discussion_points"])
        # "Include action points" is on by default: every agenda gets agenda-level action points.
        assert all(a.get("include_action_points") and a.get("action_points_mode") == "agenda"
                   for a in rom["final_rom"]["agendas"])


@pytest.mark.asyncio
async def test_rom_without_action_points_when_turned_off(env):
    async with _client() as (client, make_recording):
        rec_id = await make_recording({"output_type": "rom", "include_action_points": False,
                                       "agendas": [{"agenda_id": "A1", "title": "Project Atlas – Build Status"}]})
        pipeline, rom = await _run_after_transcription(client, rec_id)

        assert pipeline["status"]["state"] == "done", pipeline
        assert not any(a.get("include_action_points") or a.get("action_points_mode") for a in rom["final_rom"]["agendas"])


# ── Off, defaults, failures ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_off_keeps_todays_behaviour(env):
    async with _client() as (client, make_recording):
        rec_id = await make_recording({"output_type": "off"})
        assert not await auto.replaces_direct_mom(rec_id)
        async with get_db_context() as db:
            await db.execute(text("UPDATE recordings SET status = 'done' WHERE id = :id"), {"id": rec_id})
            await db.commit()
        assert not await auto.maybe_start(rec_id, USER_ID)
        assert (await client.get(f"/rom-pipeline/{rec_id}")).json()["status"]["state"] == "off"
        assert "stage1" not in env


@pytest.mark.asyncio
async def test_failed_stage_is_reported_and_can_be_retried(env):
    async with _client() as (client, make_recording):
        rec_id = await make_recording({"output_type": "mom"}, transcript=[])
        pipeline, _ = await _run_after_transcription(client, rec_id)
        status = pipeline["status"]
        assert status["state"] == "error" and "no transcript" in status["error"].lower()
        assert status["stage_label"].startswith("Failed: Stage 1")

        async with get_db_context() as db:
            await db.execute(text("UPDATE recordings SET transcript = :t WHERE id = :id"),
                             {"t": json.dumps(TRANSCRIPT), "id": rec_id})
            await db.commit()
        res = await client.post(f"/rom-pipeline/{rec_id}/start")
        assert res.status_code == 200, res.text
        await auto._TASKS[rec_id]
        assert (await client.get(f"/rom-pipeline/{rec_id}")).json()["status"]["state"] == "done"


@pytest.mark.asyncio
async def test_defaults_round_trip(env):
    async with _client() as (client, _):
        res = await client.get("/rom-pipeline/defaults")
        assert res.status_code == 200 and res.json()["saved"] is False
        assert res.json()["defaults"]["output_type"] == "mom"          # generate_mom_auto defaults to on
        body = {"output_type": "rom", "transcript_window_minutes": 4, "meeting_top_k": 9,
                "agendas": [{"title": "not a default"}]}
        assert (await client.put("/rom-pipeline/defaults", json=body)).status_code == 200
        saved = (await client.get("/rom-pipeline/defaults")).json()
        assert saved["saved"] is True
        assert saved["defaults"]["output_type"] == "rom" and saved["defaults"]["meeting_top_k"] == 9
        assert saved["defaults"]["include_action_points"] is True
        assert "agendas" not in saved["defaults"]


@pytest.mark.asyncio
async def test_previous_meetings_before_the_meeting_exists(env):
    async with _client() as (client, make_recording):
        rec_id = await make_recording({"output_type": "mom"})
        res = await client.get("/rom-pipeline/previous-meetings")
        assert res.status_code == 200, res.text
        assert rec_id in [m["id"] for m in res.json()["meetings"]]


@pytest.mark.asyncio
async def test_waits_while_another_meeting_is_transcribed(env, monkeypatch):
    """The 6 GB GPU cannot hold Whisper and the LLM at once: stages wait for other transcriptions."""
    import asyncio
    from tasks import pipeline as transcription_pipeline

    monkeypatch.setattr(auto, "TRANSCRIPTION_POLL_SEC", 0.05)
    other = asyncio.get_running_loop().create_future()
    monkeypatch.setitem(transcription_pipeline.active_tasks, "another-meeting", other)
    async with _client() as (client, make_recording):
        rec_id = await make_recording({"output_type": "mom"})
        async with get_db_context() as db:
            await db.execute(text("UPDATE recordings SET status = 'done' WHERE id = :id"), {"id": rec_id})
            await db.commit()
        assert await auto.maybe_start(rec_id, USER_ID)
        for _ in range(100):
            status = (await client.get(f"/rom-pipeline/{rec_id}")).json()["status"]
            if status.get("stage_label") == "Waiting for another transcription to finish":
                break
            await asyncio.sleep(0.05)
        assert status["stage_label"] == "Waiting for another transcription to finish"
        assert "stage1" not in env                              # nothing ran on the GPU yet
        other.set_result(None)                                  # the other transcription finished
        await auto._TASKS[rec_id]
        assert (await client.get(f"/rom-pipeline/{rec_id}")).json()["status"]["state"] == "done"
        assert env["stage1"] and env["mom"]
