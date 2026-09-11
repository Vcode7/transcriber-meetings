import pytest
import logging
from unittest.mock import patch, MagicMock, ANY
from services.ai_provider import QwenProvider
from services.prompt_service import PROMPT_META, VALID_KEYS, get_prompt_sync


def test_new_prompt_templates_registered():
    """Verify new ROM and MoM action prompts are present in PROMPT_META and VALID_KEYS."""
    keys = {m["key"] for m in PROMPT_META}
    assert "rom_discussion_no_actions" in keys
    assert "rom_action_extraction" in keys
    assert "mom_action_regen" in keys
    assert "mom_action_dedup" in keys

    assert "rom_discussion_no_actions" in VALID_KEYS
    assert "rom_action_extraction" in VALID_KEYS
    assert "mom_action_regen" in VALID_KEYS
    assert "mom_action_dedup" in VALID_KEYS

    # Check sync prompt loading fallback
    p1 = get_prompt_sync("rom_discussion_no_actions")
    assert p1 and len(p1) > 50

    p2 = get_prompt_sync("rom_action_extraction")
    assert p2 and len(p2) > 50

    p3 = get_prompt_sync("mom_action_regen")
    assert p3 and len(p3) > 50


def test_active_settings_includes_all_rom_max_tokens():
    """Verify _get_active_settings includes all ROM task token limit keys with positive values."""
    cfg = QwenProvider._get_active_settings()
    # Check all keys exist with positive integer values
    # Note: actual values may differ from code defaults if user has customized them in settings.
    rom_token_keys = [
        "max_tokens_rom_discussion",
        "max_tokens_rom_discussion_no_actions",
        "max_tokens_rom_action_extraction",
        "max_tokens_mom_extract_actions",
        "max_tokens_stage1_json_repair",
        "max_tokens_mom_action_regen",
        "max_tokens_rom_polish",
        "max_tokens_rom_enhance_window",
        "max_tokens_rom_deduplicate",
        "max_tokens_rom_agenda",
        "max_tokens_rom_mom_expansion",
        "max_tokens_rom_agenda_assign_batch",
        "max_tokens_rom_agenda_doc_points",
    ]
    for key in rom_token_keys:
        assert key in cfg, f"Missing key: {key}"
        assert isinstance(cfg[key], (int, float)), f"{key} is not numeric: {cfg[key]}"
        assert cfg[key] > 0, f"{key} must be positive, got: {cfg[key]}"


@patch.object(QwenProvider, "_get_active_settings")
@patch.object(QwenProvider, "_infer")
def test_extract_rom_discussion_uses_custom_max_tokens(mock_infer, mock_get_settings):
    """Verify Stage 1 discussion extraction uses custom max token limit from settings."""
    mock_get_settings.return_value = {
        "max_tokens_rom_discussion": 7500,
        "max_tokens_rom_discussion_no_actions": 6500,
    }
    mock_infer.return_value = '{"discussion_points": []}'

    provider = QwenProvider()

    # Standard call (skip_action_items=False)
    provider.extract_rom_discussion_points("Window text", skip_action_items=False)
    mock_infer.assert_called_with(
        ANY,
        max_new_tokens=7500,
        task_key="rom_discussion_embedded"
    )

    # Separate action extraction call (skip_action_items=True)
    provider.extract_rom_discussion_points("Window text", skip_action_items=True)
    mock_infer.assert_called_with(
        ANY,
        max_new_tokens=6500,
        task_key="rom_discussion_no_actions"
    )


@patch.object(QwenProvider, "_get_active_settings")
@patch.object(QwenProvider, "_infer")
def test_extract_rom_action_uses_custom_max_tokens(mock_infer, mock_get_settings):
    """Verify Stage 1 action extraction uses custom max token limit from settings."""
    mock_get_settings.return_value = {
        "max_tokens_rom_action_extraction": 3500,
    }
    mock_infer.return_value = '{"action_items": []}'

    provider = QwenProvider()
    provider.extract_rom_action_points("Window text")

    mock_infer.assert_called_with(
        ANY,
        max_new_tokens=3500,
        task_key="rom_action_extraction"
    )


@patch.object(QwenProvider, "_get_active_settings")
@patch.object(QwenProvider, "_infer")
def test_extract_actions_from_enhanced_points_uses_custom_max_tokens(mock_infer, mock_get_settings):
    """Verify MoM extract actions from enhanced points uses max_tokens_mom_extract_actions from settings."""
    mock_get_settings.return_value = {
        "max_tokens_mom_extract_actions": 5000,
        "rom_action_generation_chunk_size": 10,
    }
    mock_infer.return_value = '{"action_items": [{"task": "Do thing", "owner": "Alice"}]}'

    provider = QwenProvider()
    points = [{"id": "pt-1", "discussion_point": "Alice will do thing."}]
    res = provider.extract_actions_from_enhanced_points(points)

    mock_infer.assert_called_with(
        ANY,
        max_new_tokens=5000,
        task_key="mom_extract_actions"
    )
    assert len(res) == 1
    assert res[0]["task"] == "Do thing"


@patch.object(QwenProvider, "_infer")
def test_stage1_json_error_logs_raw_llm_output(mock_infer, caplog):
    """Verify that Stage 1 JSON parse errors and repair attempts log raw LLM outputs."""
    invalid_json_response = "Here is the extraction: {discussion_points: [unterminated string..."
    repaired_invalid_response = "Still invalid json {..."

    # Return invalid output on initial call, and invalid output on repair call
    mock_infer.side_effect = [invalid_json_response, repaired_invalid_response]

    provider = QwenProvider()
    with caplog.at_level(logging.WARNING):
        res = provider.extract_rom_discussion_points("Sample window text")

    assert res.get("parse_error") is True
    # Verify raw initial LLM output was logged
    assert "--- START RAW INITIAL EXTRACTION LLM RESPONSE" in caplog.text
    assert invalid_json_response in caplog.text
    # Verify raw repair LLM output was logged
    assert "--- START RAW INVALID LLM RESPONSE ---" in caplog.text
    assert repaired_invalid_response in caplog.text


def test_try_deterministic_json_truncation_recovery():
    """Verify deterministic JSON truncation recovery discards incomplete trailing items and returns valid JSON."""
    provider = QwenProvider()

    truncated_json = """{
  "discussion_points": [
    {
      "discussion_point": "First complete point",
      "speakers": ["Speaker 1"],
      "action_items": []
    },
    {
      "discussion_point": "Second complete point",
      "speakers": ["Speaker 2"],
      "action_items": []
    },
    {
      "discussion_point": "Third point cut off mid-sentence because token limit was reached. The speaker mentioned that
"""

    recovered = provider.try_deterministic_json_truncation_recovery(truncated_json)
    assert recovered is not None
    assert isinstance(recovered, dict)
    pts = recovered.get("discussion_points")
    assert isinstance(pts, list)
    assert len(pts) == 2
    assert pts[0]["discussion_point"] == "First complete point"
    assert pts[1]["discussion_point"] == "Second complete point"


@patch.object(QwenProvider, "repair_stage1_json")
@patch.object(QwenProvider, "_infer")
def test_deterministic_recovery_skips_llm_repair(mock_infer, mock_repair):
    """Verify that if deterministic recovery succeeds, LLM repair is skipped."""
    truncated_json = """{
  "discussion_points": [
    {
      "discussion_point": "Valid complete point",
      "speakers": ["Alice"]
    },
    {
      "discussion_point": "Truncated point cut off...
"""
    mock_infer.return_value = truncated_json

    provider = QwenProvider()
    res = provider.extract_rom_discussion_points("Window transcript text")

    # Recovery should succeed deterministically
    assert res.get("parse_error") is not True
    assert len(res.get("discussion_points", [])) == 1
    assert res["discussion_points"][0]["discussion_point"] == "Valid complete point"
    # LLM repair MUST be skipped
    mock_repair.assert_not_called()


@pytest.mark.asyncio
async def test_mom_extract_actions_prompt_registered_and_customizable():
    """Verify mom_extract_actions prompt metadata, defaults, and DB persistence."""
    from services.prompt_service import (
        PROMPT_META,
        VALID_KEYS,
        get_prompt_sync,
        list_prompts,
        set_prompt,
        reset_prompt,
        get_prompt,
        _invalidate,
    )
    from database import get_db_context, connect_db

    await connect_db()

    meta = next((m for m in PROMPT_META if m["key"] == "mom_extract_actions"), None)
    assert meta is not None
    assert meta["name"] == "Generate Action Point from Enhanced Point"
    assert meta["category"] == "MoM"
    assert "mom_extract_actions" in VALID_KEYS

    default_prompt = get_prompt_sync("mom_extract_actions")
    assert default_prompt and len(default_prompt) > 100
    assert "STRICT ACTIONABILITY CRITERIA" in default_prompt
    assert "expected_outcome" in default_prompt

    # Test DB list, save, and reset flow
    async with get_db_context() as db:
        prompts = await list_prompts(db)
        target = next((p for p in prompts if p["key"] == "mom_extract_actions"), None)
        assert target is not None
        assert target["name"] == "Generate Action Point from Enhanced Point"
        assert target["default_template"] == default_prompt
        assert target["template"] != ""

        # Test save custom template
        custom_tpl = default_prompt + "\n# Custom rule for action extraction"
        await set_prompt(db, "mom_extract_actions", custom_tpl)
        assert await get_prompt(db, "mom_extract_actions") == custom_tpl
        assert get_prompt_sync("mom_extract_actions") == custom_tpl

        # Simulate process restart / clean cache where in-memory cache is invalidated
        _invalidate()
        # get_prompt_sync MUST reload custom template from SQLite DB
        assert get_prompt_sync("mom_extract_actions") == custom_tpl

        # Verify extract_actions_from_enhanced_points uses the custom prompt
        provider = QwenProvider.__new__(QwenProvider)
        provider._get_active_settings = MagicMock(return_value={"max_tokens_mom_extract_actions": 4096, "rom_action_generation_chunk_size": 10})
        captured_prompts = []
        def mock_infer_fn(prompt, max_new_tokens=4096, task_key="mom_extract_actions"):
            captured_prompts.append(prompt)
            return '{"action_items": []}'
        provider._infer = MagicMock(side_effect=mock_infer_fn)

        test_points = [{"id": "pt-test", "enhanced_text": "Alice to deploy release by Friday.", "speakers": ["Alice"]}]
        provider.extract_actions_from_enhanced_points(test_points)
        assert len(captured_prompts) == 1
        assert "# Custom rule for action extraction" in captured_prompts[0]

        prompts_after_save = await list_prompts(db)
        target_after_save = next((p for p in prompts_after_save if p["key"] == "mom_extract_actions"), None)
        assert target_after_save is not None
        assert target_after_save["is_modified"] is True
        assert target_after_save["template"] == custom_tpl

        # Test reset to default
        await reset_prompt(db, "mom_extract_actions")
        _invalidate()
        assert get_prompt_sync("mom_extract_actions") == default_prompt
        assert await get_prompt(db, "mom_extract_actions") == default_prompt

        prompts_after_reset = await list_prompts(db)
        target_after_reset = next((p for p in prompts_after_reset if p["key"] == "mom_extract_actions"), None)
        assert target_after_reset is not None
        assert target_after_reset["is_modified"] is False
        assert target_after_reset["template"] == default_prompt


def test_action_point_extraction_no_action_for_informational_points():
    """Verify informational or status-only discussion returns no action items."""
    provider = QwenProvider.__new__(QwenProvider)
    provider._get_active_settings = MagicMock(return_value={"max_tokens_mom_extract_actions": 4096, "rom_action_generation_chunk_size": 10})

    points = [
        {"id": "pt-1", "enhanced_text": "Alice gave a general status overview of server uptime in Q2.", "speakers": ["Alice"]},
        {"id": "pt-2", "enhanced_text": "Bob shared his opinion on the color scheme of the new dashboard.", "speakers": ["Bob"]},
    ]

    # LLM correctly adheres to the prompt and returns an empty list
    mock_llm_response = '{"action_items": []}'
    provider._infer = MagicMock(return_value=mock_llm_response)

    results = provider.extract_actions_from_enhanced_points(points)
    assert results == []


def test_action_point_extraction_with_expected_outcome():
    """Verify action points capture task, owner, deadline, and expected_outcome."""
    provider = QwenProvider.__new__(QwenProvider)
    provider._get_active_settings = MagicMock(return_value={"max_tokens_mom_extract_actions": 4096, "rom_action_generation_chunk_size": 10})

    points = [
        {
            "id": "pt-1",
            "enhanced_text": "Vikas committed to deploying the database backup script by 2026-10-15 to ensure automated DR snapshots.",
            "action_owner": "Vikas",
            "speakers": ["Vikas"]
        }
    ]

    mock_llm_response = """
{
  "action_items": [
    {
      "source_point_id": "pt-1",
      "task": "Deploy automated PostgreSQL database backup script with Slack alerting.",
      "owner": "Vikas",
      "deadline": "2026-10-15",
      "expected_outcome": "Automated nightly snapshots replicated to secondary S3 bucket with alerts."
    }
  ]
}
"""
    provider._infer = MagicMock(return_value=mock_llm_response)

    results = provider.extract_actions_from_enhanced_points(points)
    assert len(results) == 1
    assert results[0]["task"] == "Deploy automated PostgreSQL database backup script with Slack alerting."
    assert results[0]["owner"] == "Vikas"
    assert results[0]["deadline"] == "2026-10-15"
    assert results[0]["expected_outcome"] == "Automated nightly snapshots replicated to secondary S3 bucket with alerts."


def test_introduction_and_conclusion_single_paragraph_5_to_7_sentences():
    """Verify meeting introduction and conclusion are strictly 1 single paragraph and 5 to 7 sentences."""
    from services.rom_service import RomService
    import re

    rom_srv = RomService()

    # Test 1: Multi-paragraph text with 9 sentences is collapsed to 1 single paragraph with exactly 7 sentences
    multi_paragraph_long = """
First paragraph sentence one. First paragraph sentence two. First paragraph sentence three.

Second paragraph sentence four! Second paragraph sentence five? Second paragraph sentence six.

Third paragraph sentence seven. Third paragraph sentence eight. Third paragraph sentence nine.
"""
    enforced = rom_srv._enforce_single_paragraph_5_to_7_sentences(multi_paragraph_long)
    assert "\n" not in enforced
    assert "\r" not in enforced
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', enforced) if s.strip()]
    assert len(sentences) == 7
    assert sentences[0] == "First paragraph sentence one."
    assert sentences[6] == "Third paragraph sentence seven."

    # Test 2: Fallback in generate_mom_overview produces single paragraph with 5-7 sentences
    overview = rom_srv.generate_mom_overview(
        ["Discussion point 1", "Discussion point 2"],
        {"filename": "Test Recording", "created_at": "2026-10-01", "speakers_detected": ["Alice", "Bob"]}
    )
    intro = overview["introduction"]
    conclusion = overview["conclusion"]

    assert "\n" not in intro
    assert "\r" not in intro
    intro_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', intro) if s.strip()]
    assert 5 <= len(intro_sentences) <= 7

    assert "\n" not in conclusion
    assert "\r" not in conclusion
    conclusion_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', conclusion) if s.strip()]
    assert 5 <= len(conclusion_sentences) <= 7

    # Test 3: QwenProvider._parse_mom_json also enforces single paragraph 5-7 sentences
    provider = QwenProvider.__new__(QwenProvider)
    mom_json_raw = """
{
  "title": "Strategy Session",
  "introduction": "Intro sentence one. Intro sentence two.\\n\\nIntro sentence three. Intro sentence four. Intro sentence five. Intro sentence six. Intro sentence seven. Intro sentence eight.",
  "conclusion": "Conclusion sentence one.\\n\\nConclusion sentence two.\\n\\nConclusion sentence three. Conclusion sentence four. Conclusion sentence five.",
  "points_discussed": ["Point 1"],
  "action_items": []
}
"""
    parsed = provider._parse_mom_json(mom_json_raw, {"filename": "Test", "created_at": "2026-10-01", "speakers_detected": []})
    assert "\n" not in parsed["introduction"]
    parsed_intro_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', parsed["introduction"]) if s.strip()]
    assert len(parsed_intro_sentences) == 7

    assert "\n" not in parsed["conclusion"]
    parsed_conclusion_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', parsed["conclusion"]) if s.strip()]
    assert len(parsed_conclusion_sentences) == 5

