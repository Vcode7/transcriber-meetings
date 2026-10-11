"""OCR settings: official defaults, validation, lenient loading, persistence and scoping."""
import json
import sqlite3

import pytest
from pydantic import ValidationError

from services import ocr_settings as S
from services.ocr_settings import (
    OcrSettings,
    OcrSettingsIn,
    get_ocr_settings,
    load_lenient,
    preset_settings,
    settings_schema,
    use_ocr_settings,
)

_TABLE_SQL = """
    CREATE TABLE ocr_settings (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        config_json TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        updated_by TEXT
    )
"""


def test_defaults_follow_official_ppocrv6_pipeline():
    cfg = OcrSettings()
    assert (cfg.det_limit_type, cfg.det_limit_side_len, cfg.det_max_side_limit) == ("min", 64, 4000)
    assert (cfg.det_db_thresh, cfg.det_db_box_thresh, cfg.det_db_unclip_ratio) == (0.3, 0.6, 1.5)
    assert cfg.det_score_mode == "fast" and not cfg.det_use_dilation
    assert cfg.use_textline_orientation and cfg.cls_thresh == 0.9
    assert (cfg.rec_max_width, cfg.rec_batch_size) == (3200, 6)
    # PP-OCRv6 consumes continuous-tone images: no destructive preprocessing by default
    assert cfg.binarize == "none" and not cfg.clahe and not cfg.grayscale and cfg.denoise == "none"


@pytest.mark.parametrize("field,value", [
    ("det_db_thresh", 1.5),
    ("rec_batch_size", 0),
    ("rec_batch_size", 9),
    ("pdf_render_dpi", 1000),
    ("denoise", "gaussian"),
    ("binarize_block_size", 30),
    ("det_limit_type", "long"),
])
def test_invalid_values_are_rejected(field, value):
    with pytest.raises(ValidationError):
        OcrSettings(**{field: value})


def test_cross_field_limit_check():
    with pytest.raises(ValidationError):
        OcrSettings(det_limit_type="max", det_limit_side_len=4000, det_max_side_limit=1000)
    # In min mode the side limit is a lower bound, so it may exceed the cap
    OcrSettings(det_limit_type="min", det_limit_side_len=960, det_max_side_limit=640)


def test_unknown_keys_rejected_on_save_but_ignored_on_load():
    with pytest.raises(ValidationError):
        OcrSettingsIn(bogus=1)
    assert load_lenient({"bogus": 1}) == OcrSettings()


def test_lenient_load_keeps_valid_and_drops_invalid_values():
    cfg = load_lenient({"det_db_thresh": 0.25, "rec_batch_size": 99, "pdf_render_dpi": "abc"})
    assert cfg.det_db_thresh == 0.25
    assert cfg.rec_batch_size == 6
    assert cfg.pdf_render_dpi == 200


def test_lenient_load_recovers_from_cross_field_error():
    cfg = load_lenient({
        "det_limit_type": "max", "det_limit_side_len": 4000, "det_max_side_limit": 1000, "det_db_thresh": 0.25,
    })
    assert cfg.det_db_thresh == 0.25
    assert cfg.det_limit_type == "min" or cfg.det_limit_side_len <= cfg.det_max_side_limit


def test_lenient_load_handles_non_dict():
    assert load_lenient(None) == OcrSettings()
    assert load_lenient([1, 2]) == OcrSettings()


def test_presets_are_valid_and_legacy_matches_previous_constants():
    for name in S.PRESETS:
        preset_settings(name)
    legacy = preset_settings("legacy")
    assert (legacy.det_limit_type, legacy.det_limit_side_len) == ("max", 960)
    assert (legacy.det_db_thresh, legacy.det_db_box_thresh, legacy.det_db_unclip_ratio) == (0.2, 0.45, 1.4)
    assert (legacy.rec_score_thresh, legacy.cls_thresh) == (0.4, 0.6)
    assert preset_settings("official") == OcrSettings()


def test_schema_exposes_bounds_types_and_groups():
    schema = settings_schema()
    fields = {f["name"]: f for f in schema["fields"]}
    assert (fields["det_db_thresh"]["min"], fields["det_db_thresh"]["max"]) == (0.05, 0.9)
    assert fields["denoise"]["type"] == "enum" and fields["denoise"]["options"][0]["value"] == "none"
    assert fields["grayscale"]["type"] == "bool"
    assert fields["rec_batch_size"]["type"] == "int"
    assert fields["clahe_clip_limit"]["depends_on"] == {"field": "clahe", "values": [True]}
    assert {f["group"] for f in schema["fields"]} == {g["id"] for g in schema["groups"]}
    assert "schema_version" not in fields
    assert set(fields) == set(OcrSettings.model_fields) - {"schema_version"}


def test_override_is_scoped_to_the_block():
    before = get_ocr_settings().det_db_thresh
    with use_ocr_settings(OcrSettings(det_db_thresh=0.42)):
        assert get_ocr_settings().det_db_thresh == 0.42
        with use_ocr_settings(OcrSettings(det_db_thresh=0.11)):
            assert get_ocr_settings().det_db_thresh == 0.11
        assert get_ocr_settings().det_db_thresh == 0.42
    assert get_ocr_settings().det_db_thresh == before


def test_sync_load_reads_saved_row_leniently(tmp_path, monkeypatch):
    db = tmp_path / "ocr.db"
    conn = sqlite3.connect(db)
    conn.execute(_TABLE_SQL)
    conn.execute(
        "INSERT INTO ocr_settings VALUES (1, ?, '2026-10-11T00:00:00+00:00', 'u')",
        (json.dumps({"pdf_render_dpi": 300, "det_db_thresh": "bad"}),),
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(S, "_sqlite_path", lambda: str(db))
    S.invalidate_cache()
    try:
        cfg = get_ocr_settings()
        assert cfg.pdf_render_dpi == 300
        assert cfg.det_db_thresh == 0.3
        assert S.get_settings_updated_at() == "2026-10-11T00:00:00+00:00"
    finally:
        S.invalidate_cache()


def test_missing_table_falls_back_to_defaults_without_caching(tmp_path, monkeypatch):
    db = tmp_path / "empty.db"
    sqlite3.connect(db).close()
    monkeypatch.setattr(S, "_sqlite_path", lambda: str(db))
    S.invalidate_cache()
    try:
        assert get_ocr_settings() == OcrSettings()
        assert S._cache is None
    finally:
        S.invalidate_cache()


@pytest.mark.asyncio
async def test_save_load_and_reset_round_trip(tmp_path):
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'settings.db'}")
    async with engine.begin() as conn:
        await conn.execute(text(_TABLE_SQL))
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with session_factory() as db:
            await S.save_ocr_settings(db, OcrSettingsIn(pdf_render_dpi=300), user_id="test_user")
            # Saving refreshes the in-process cache, so OCR worker threads see it immediately
            assert get_ocr_settings().pdf_render_dpi == 300

            cfg, updated_at = await S.load_ocr_settings(db)
            assert cfg.pdf_render_dpi == 300 and updated_at

            await S.save_ocr_settings(db, OcrSettings(pdf_render_dpi=250))  # upsert, single row
            rows = (await db.execute(text("SELECT COUNT(*) FROM ocr_settings"))).scalar()
            assert rows == 1
            assert (await S.load_ocr_settings(db))[0].pdf_render_dpi == 250

            await S.reset_ocr_settings(db)
            cfg, updated_at = await S.load_ocr_settings(db)
            assert cfg == OcrSettings() and updated_at is None
    finally:
        await engine.dispose()
        S.invalidate_cache()


def test_fingerprint_changes_with_settings():
    a = S.settings_fingerprint(OcrSettings())
    b = S.settings_fingerprint(OcrSettings(det_db_thresh=0.25))
    assert a != b and "det=min/64/4000" in a
