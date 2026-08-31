"""
Unit tests for Stage 2 LoRA and QLoRA Training Mode & Hugging Face validation.
"""
import pytest
import os
import json
import tempfile
import shutil
from pathlib import Path
from services.training.stage2_training_service import (
    validate_hf_model_path,
    get_all_variants,
    get_variant,
    set_active_variant,
    get_active_variant,
    DEFAULT_SETTINGS,
    load_settings,
    save_settings,
    _next_variant_label,
    run_variant_on_group,
)


@pytest.fixture
def temp_hf_dir():
    """Create a temporary mock Hugging Face model directory."""
    temp_dir = tempfile.mkdtemp(prefix="mock_hf_model_")
    p = Path(temp_dir)
    config = {
        "architectures": ["Qwen2ForCausalLM"],
        "model_type": "qwen2",
        "vocab_size": 152064,
    }
    (p / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (p / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (p / "tokenizer.json").write_text("{}", encoding="utf-8")
    (p / "model.safetensors").write_bytes(b"mock_weights")
    yield str(p)
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_validate_hf_model_path_empty():
    """Test validation with empty path."""
    res = validate_hf_model_path("", method="lora")
    assert res["valid"] is False
    assert "required for LoRA" in res["reason"]
    assert "Ollama" in res["reason"]

    res_q = validate_hf_model_path("   ", method="qlora")
    assert res_q["valid"] is False
    assert "required for QLoRA" in res_q["reason"]


def test_validate_hf_model_path_nonexistent():
    """Test validation with non-existent path."""
    res = validate_hf_model_path("Z:\\path\\that\\does\\not\\exist_99999", method="lora")
    assert res["valid"] is False
    assert "not found" in res["reason"].lower() or "does not exist" in res["reason"].lower()


def test_validate_hf_model_path_missing_config():
    """Test validation when config.json is missing."""
    temp_dir = tempfile.mkdtemp(prefix="mock_empty_dir_")
    try:
        res = validate_hf_model_path(temp_dir, method="lora")
        assert res["valid"] is False
        assert "config.json" in res["reason"]
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_validate_hf_model_path_valid(temp_hf_dir):
    """Test validation with a valid mock Hugging Face model directory."""
    res = validate_hf_model_path(temp_hf_dir, method="lora")
    assert res["valid"] is True
    assert res["details"]["model_type"] == "qwen2"
    assert res["details"]["weights_count"] >= 1
    assert res["details"]["has_tokenizer"] is True


def test_stage2_settings_lora_defaults():
    """Test that default settings include LoRA/QLoRA fields."""
    settings = load_settings()
    assert "training_mode" in settings
    assert "hf_model_path" in settings
    assert "lora_r" in settings
    assert "lora_alpha" in settings
    assert "num_train_epochs" in settings


def test_variant_labeling_lora_and_qlora():
    """Test variant naming with LoRA / QLoRA prefixes."""
    variants = [
        {"label": "DSPy Variant 001"},
        {"label": "LoRA Variant 001"},
        {"label": "QLoRA Variant 001"},
    ]
    next_lora = _next_variant_label(variants, method="lora")
    assert next_lora == "LoRA Variant 002"

    next_qlora = _next_variant_label(variants, method="qlora")
    assert next_qlora == "QLoRA Variant 002"

    next_dspy = _next_variant_label(variants, method="dspy")
    assert next_dspy == "DSPy Variant 002"


def test_active_variant_management():
    """Test setting and getting the active Stage 2 variant."""
    active = get_active_variant()
    assert active is not None
    assert "variant_id" in active

    success = set_active_variant("default")
    assert success is True
    active = get_active_variant()
    assert active["variant_id"] == "default"
    assert active.get("is_active") is True


def test_run_variant_on_group_default(monkeypatch):
    """Test running the baseline variant on a sample group with mock response."""
    sample_points = [
        {"point": "Discussed Q3 product roadmap timeline", "speakers": ["Alice"], "action_owner": ["Bob"]},
        {"point": "Agreed to finalize budget by Friday", "speakers": ["Bob"], "action_owner": ["Bob"]},
    ]
    mock_json = json.dumps({
        "points": [
            {"point": "Discussed Q3 product roadmap and agreed to finalize budget by Friday.", "speaker": ["Alice", "Bob"], "action_owner": ["Bob"]}
        ]
    })
    monkeypatch.setattr(
        "services.training.stage2_training_service._run_default_variant",
        lambda *args, **kwargs: mock_json
    )

    res = run_variant_on_group(
        variant_id="default",
        stage1_points=sample_points,
        global_context="Product team context",
        meeting_context="Sprint planning meeting",
    )
    assert res is not None
    assert "raw_output" in res
    assert "eval_scores" in res
    assert "overall" in res["eval_scores"]
    assert res["eval_scores"]["overall"] > 0
