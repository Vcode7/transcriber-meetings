import pytest
from unittest.mock import patch
from services.ai_provider import QwenProvider

def test_json_repair_success():
    """Verify that when raw MoM JSON is malformed, we run the repair prompt and parse the result."""
    provider = QwenProvider()

    malformed_raw = '{\n  "agenda_topic": "Topic A",\n  "discussion": [\n    {\n      "type": "decision",\n      "speaker": "Alice",\n      "point": "Incomplete JSON'
    repaired_raw = '''```json
{
  "agenda_topic": "Topic A",
  "agenda_speaker": "Alice",
  "discussion": [
    {
      "type": "decision",
      "speaker": "Alice",
      "point": "Repaired JSON point successfully",
      "dates": [{"value": "2026-07-20", "purpose": "deadline"}],
      "action": {
        "owner": "Alice",
        "description": "Complete repair task",
        "deadline": "2026-07-20",
        "status": "in_progress"
      }
    }
  ]
}
```'''

    with patch.object(QwenProvider, "_infer") as mock_infer:
        # First call returns malformed JSON, second call returns repaired JSON
        mock_infer.side_effect = [malformed_raw, repaired_raw]

        result = provider.extract_raw_mom_for_agenda(
            agenda_topic="Topic A",
            agenda_speaker="Alice",
            evidence="Some retrieved facts",
        )

        assert mock_infer.call_count == 2
        
        # Verify first call args contain the extraction key/prompt
        first_call_prompt = mock_infer.call_args_list[0][0][0]
        assert "RETRIEVED EVIDENCE" in first_call_prompt

        # Verify second call args contain raw_mom_repair prompt
        second_call_prompt = mock_infer.call_args_list[1][0][0]
        assert "INVALID RAW JSON TO REPAIR" in second_call_prompt
        assert malformed_raw in second_call_prompt

        # Verify output parsing succeeded using repaired JSON
        assert result["agenda_topic"] == "Topic A"
        assert result["agenda_speaker"] == "Alice"
        assert len(result["discussion"]) == 1
        disc = result["discussion"][0]
        assert disc["type"] == "decision"
        assert disc["speaker"] == "Alice"
        assert disc["point"] == "Repaired JSON point successfully"
        assert disc["action"]["owner"] == "Alice"
        assert disc["action"]["status"] == "in_progress"


def test_json_repair_failure_fallback():
    """Verify that if the repair also fails, we fall back to returning original raw string."""
    provider = QwenProvider()

    malformed_raw = '{\n  "agenda_topic": "Topic A",\n  "discussion": [\n    {\n      "type": "decision",\n      "speaker": "Alice",\n      "point": "Incomplete JSON'
    repaired_failed = 'Still invalid json response'

    with patch.object(QwenProvider, "_infer") as mock_infer:
        mock_infer.side_effect = [malformed_raw, repaired_failed]

        result = provider.extract_raw_mom_for_agenda(
            agenda_topic="Topic A",
            agenda_speaker="Alice",
            evidence="Some retrieved facts",
        )

        assert mock_infer.call_count == 2
        assert result["agenda_topic"] == "Topic A"
        assert result["agenda_speaker"] == "Alice"
        # discussion should contain the original malformed_raw text
        assert result["discussion"] == malformed_raw
