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
    """Verify _get_active_settings includes all ROM task token limit keys."""
    cfg = QwenProvider._get_active_settings()
    assert cfg.get("max_tokens_rom_discussion") == 4096
    assert cfg.get("max_tokens_rom_discussion_no_actions") == 4096
    assert cfg.get("max_tokens_rom_action_extraction") == 2048
    assert cfg.get("max_tokens_stage1_json_repair") == 4548
    assert cfg.get("max_tokens_mom_action_regen") == 4048
    assert cfg.get("max_tokens_rom_polish") == 4096
    assert cfg.get("max_tokens_rom_enhance_window") == 4096
    assert cfg.get("max_tokens_rom_deduplicate") == 2048
    assert cfg.get("max_tokens_rom_agenda") == 2048
    assert cfg.get("max_tokens_rom_mom_expansion") == 3000
    assert cfg.get("max_tokens_rom_agenda_assign_batch") == 4096
    assert cfg.get("max_tokens_rom_agenda_doc_points") == 1024


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
        task_key="rom_discussion"
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
