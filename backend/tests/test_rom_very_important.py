import pytest
from unittest.mock import patch, MagicMock
from services.rom_service import RomService


@patch("services.ai_provider.get_provider")
def test_short_version_replaces_summarized_marked_point_with_full_original(mock_get_provider):
    """
    When LLM attempts to summarize or shorten a Very Important point in Short version,
    the output must contain the COMPLETE ORIGINAL CONTENT intact without summarization.
    """
    mock_provider = MagicMock()
    # LLM returned a shortened/summarized version of point 1, plus point 2 condensed
    mock_provider.query.return_value = (
        '["Alice discussed Q3 financials and 20% revenue growth.", '
        '"Bob will deploy the security update."]'
    )
    mock_get_provider.return_value = mock_provider

    rom_service = RomService()
    final_rom = {
        "meeting_date": "2026-09-15",
        "agendas": [
            {
                "agenda_id": "A1",
                "title": "Financial Review",
                "discussion_points": [
                    {
                        "id": "pt-101",
                        "text": (
                            "Alice presented the detailed Q3 financial metrics, highlighting a 20% YoY "
                            "revenue increase, reduction in churn to 1.2%, and requested board approval for "
                            "the expansion budget."
                        ),
                        "polished_text": (
                            "Alice presented the detailed Q3 financial metrics, highlighting a 20% YoY "
                            "revenue increase, reduction in churn to 1.2%, and requested board approval for "
                            "the expansion budget."
                        ),
                        "speaker": "Alice",
                        "speakers": ["Alice"],
                        "action_owner": "Alice",
                        "action_items": [{"task": "Submit expansion budget for board sign-off", "owner": "Alice"}],
                        "timeline_start": 12.0,
                        "timeline_end": 45.0,
                    },
                    {
                        "id": "pt-102",
                        "text": "Bob discussed deploying the latest security patch to staging servers by Wednesday.",
                        "polished_text": "Bob discussed deploying the latest security patch to staging servers by Wednesday.",
                        "speaker": "Bob",
                        "action_owner": "Bob",
                    }
                ],
            }
        ],
    }

    # Mark pt-101 as Very Important
    important_points = [
        {
            "id": "imp-1",
            "point_id": "pt-101",
            "agenda_id": "A1",
            "text": "Alice presented the detailed Q3 financial metrics",
        }
    ]

    result = rom_service.generate_rom_version(
        final_rom=final_rom,
        version="short",
        important_points=important_points,
    )

    pts = result["agendas"][0]["discussion_points"]
    assert len(pts) >= 1

    # Find the marked point in the generated output
    marked_pt = next((p for p in pts if p.get("original_point_id") == "pt-101" or p.get("is_very_important")), None)
    assert marked_pt is not None, "Very Important point pt-101 must be included in Short output"

    # Complete original content MUST be preserved intact without summarization
    expected_full_text = (
        "Alice presented the detailed Q3 financial metrics, highlighting a 20% YoY "
        "revenue increase, reduction in churn to 1.2%, and requested board approval for "
        "the expansion budget."
    )
    assert marked_pt["text"] == expected_full_text
    assert marked_pt["polished_text"] == expected_full_text
    assert marked_pt["speaker"] == "Alice"
    assert marked_pt["action_owner"] == "Alice"
    assert len(marked_pt["action_items"]) == 1
    assert marked_pt["action_items"][0]["task"] == "Submit expansion budget for board sign-off"
    assert marked_pt["is_very_important"] is True


@patch("services.ai_provider.get_provider")
def test_medium_version_replaces_summarized_marked_point_with_full_original(mock_get_provider):
    """
    When LLM attempts to summarize a Very Important point in Medium version,
    the complete original content must be restored intact.
    """
    mock_provider = MagicMock()
    # LLM heavily compressed both points into one
    mock_provider.query.return_value = '["Dr. Kumar announced SOC2 audit completion and team celebration."]'
    mock_get_provider.return_value = mock_provider

    rom_service = RomService()
    full_original = (
        "Dr. Kumar confirmed that the SOC2 Type II compliance audit has officially concluded "
        "with zero exceptions reported across all five trust service criteria. He formally commended "
        "the infrastructure and security operations teams for their tireless efforts over the past 9 months."
    )
    final_rom = {
        "agendas": [
            {
                "agenda_id": "A2",
                "title": "Security & Compliance",
                "discussion_points": [
                    {
                        "id": "pt-soc2",
                        "text": full_original,
                        "polished_text": full_original,
                        "speaker": "Dr. Kumar",
                        "action_owner": None,
                        "is_very_important": True,  # Directly flagged on point
                    }
                ],
            }
        ],
    }

    result = rom_service.generate_rom_version(
        final_rom=final_rom,
        version="medium",
    )

    pts = result["agendas"][0]["discussion_points"]
    assert len(pts) == 1
    assert pts[0]["text"] == full_original
    assert pts[0]["polished_text"] == full_original
    assert pts[0]["speaker"] == "Dr. Kumar"
    assert pts[0]["is_very_important"] is True


@patch("services.ai_provider.get_provider")
def test_short_version_reinserts_omitted_marked_point(mock_get_provider):
    """
    If the LLM completely omits a Very Important point from its output,
    the engine must deterministically re-insert it with its complete original content fully intact.
    """
    mock_provider = MagicMock()
    # LLM completely ignored the marked point pt-critical and only generated point for pt-other
    mock_provider.query.return_value = '["Team agreed to reschedule weekly sync to Thursdays."]'
    mock_get_provider.return_value = mock_provider

    rom_service = RomService()
    critical_text = "MANDATORY ACTION: All production database credentials must be rotated before 18:00 UTC today."
    final_rom = {
        "agendas": [
            {
                "agenda_id": "A1",
                "title": "Operations",
                "discussion_points": [
                    {
                        "id": "pt-critical",
                        "text": critical_text,
                        "polished_text": critical_text,
                        "speaker": "DevOps Lead",
                        "action_owner": "DevOps Lead",
                        "action_items": [{"task": "Rotate prod credentials", "owner": "DevOps Lead"}],
                    },
                    {
                        "id": "pt-other",
                        "text": "Weekly sync meeting should be shifted to Thursdays at 10 AM.",
                        "polished_text": "Weekly sync meeting should be shifted to Thursdays at 10 AM.",
                        "speaker": "Sarah",
                        "action_owner": None,
                    },
                ],
            }
        ],
    }

    important_points = [
        {"point_id": "pt-critical", "agenda_id": "A1", "text": "production database credentials"}
    ]

    result = rom_service.generate_rom_version(
        final_rom=final_rom,
        version="short",
        important_points=important_points,
    )

    pts = result["agendas"][0]["discussion_points"]
    assert len(pts) == 2

    # Check that critical point was re-inserted fully intact
    crit = next((p for p in pts if p.get("original_point_id") == "pt-critical"), None)
    assert crit is not None
    assert crit["text"] == critical_text
    assert crit["speaker"] == "DevOps Lead"
    assert crit["action_owner"] == "DevOps Lead"
    assert crit["is_very_important"] is True


@patch("services.ai_provider.get_provider")
def test_prompt_includes_very_important_tags_and_mandatory_rules(mock_get_provider):
    """
    Verify that prompt sent to LLM explicitly tags marked points and includes
    strict mandatory section forbidding summarization.
    """
    mock_provider = MagicMock()
    mock_provider.query.return_value = '["Point 1"]'
    mock_get_provider.return_value = mock_provider

    rom_service = RomService()
    final_rom = {
        "agendas": [
            {
                "agenda_id": "A1",
                "title": "Executive Review",
                "discussion_points": [
                    {
                        "id": "p1",
                        "text": "Crucial executive directive to freeze Q4 hiring immediately.",
                        "speaker": "CEO",
                        "is_very_important": True,
                    }
                ],
            }
        ],
    }

    rom_service.generate_rom_version(final_rom=final_rom, version="short")

    call_prompt = mock_provider.query.call_args[0][0]
    # Check that points_json has the tag
    assert "*** VERY IMPORTANT - PRESERVE INTACT WITHOUT SUMMARIZATION ***" in call_prompt
    # Check that mandatory section was built with strict rules
    assert "MANDATORY VERY IMPORTANT POINTS (MUST BE INCLUDED VERBATIM WITHOUT ANY SUMMARIZATION)" in call_prompt
    assert "Crucial executive directive to freeze Q4 hiring immediately." in call_prompt
    assert "Do NOT summarize, shorten, merge, paraphrase, or omit any Very Important point." in call_prompt
