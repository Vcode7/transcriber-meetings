import json
import pytest
from unittest.mock import patch, MagicMock
from routers.rom_router import AiEditPointsRequest, ConcludePointsRequest
from services.rom_service import rom_service


def test_ai_edit_request_allows_more_than_3_points():
    """Verify point selection limit is fixed and supports more than 3 points."""
    point_ids = ["p1", "p2", "p3", "p4", "p5"]
    point_details = [{"id": pid, "text": f"Point {pid}"} for pid in point_ids]

    req = AiEditPointsRequest(
        point_ids=point_ids,
        point_details=point_details,
        prompt="Make formal",
    )
    assert len(req.point_ids) == 5
    assert len(req.point_details) == 5


def test_conclude_points_request_validation():
    """Verify ConcludePointsRequest requires at least 2 points."""
    # 2 points should succeed
    req = ConcludePointsRequest(
        point_ids=["p1", "p2"],
        point_details=[{"id": "p1", "text": "Point 1"}, {"id": "p2", "text": "Point 2"}],
    )
    assert len(req.point_ids) == 2

    # 1 point should fail Pydantic validation
    with pytest.raises(Exception):
        ConcludePointsRequest(
            point_ids=["p1"],
            point_details=[{"id": "p1", "text": "Point 1"}],
        )


def test_ai_edit_points_disables_thinking_mode():
    """Verify thinking mode is explicitly disabled (think=False) in ai_edit_points."""
    points = [
        {"id": "p1", "text": "Text 1", "speaker": "Alice"},
        {"id": "p2", "text": "Text 2", "speaker": "Bob"},
        {"id": "p3", "text": "Text 3", "speaker": "Charlie"},
        {"id": "p4", "text": "Text 4", "speaker": "David"},
    ]

    mock_provider = MagicMock()
    mock_provider.query.return_value = json.dumps({
        "updated_points": [
            {"id": "p1", "text": "Updated 1"},
            {"id": "p2", "text": "Updated 2"},
            {"id": "p3", "text": "Updated 3"},
            {"id": "p4", "text": "Updated 4"},
        ],
        "explanation": "Updated tone"
    })

    with patch("services.ai_provider.get_provider", return_value=mock_provider):
        res = rom_service.ai_edit_points(points=points, prompt="Make concise")

        # Verify think=False was passed to query
        assert mock_provider.query.called
        _, kwargs = mock_provider.query.call_args
        assert kwargs.get("think") is False
        assert kwargs.get("task_key") == "rom_ai_edit_points"

        assert len(res["updated_points"]) == 4
        assert res["updated_points"][0]["text"] == "Updated 1"


def test_ai_edit_points_strips_thinking_tags():
    """Verify <think> tags are stripped if model returns them."""
    points = [{"id": "p1", "text": "Raw point"}]
    mock_provider = MagicMock()
    mock_provider.query.return_value = (
        "<think>Let me ponder this edit...</think>"
        + json.dumps({
            "updated_points": [{"id": "p1", "text": "Polished point"}],
            "explanation": "Cleaned up"
        })
    )

    with patch("services.ai_provider.get_provider", return_value=mock_provider):
        res = rom_service.ai_edit_points(points=points, prompt="Clean")
        assert res["updated_points"][0]["text"] == "Polished point"


def test_conclude_points_combines_multiple_points_and_disables_thinking():
    """Verify conclude_points combines points into one concise conclusion with think=False."""
    points = [
        {
            "id": "p1",
            "text": "The engineering team evaluated three microservice architecture designs for the payments processor.",
            "speaker": "Sarah Lin",
            "speakers": ["Sarah Lin"],
            "technical_terms": ["microservice", "Kafka"],
            "dates": ["Oct 12"],
            "numbers": ["3 designs"],
            "references": ["RFC-402"],
            "action_items": [{"task": "Draft proposal", "assignee": "Sarah"}],
            "timeline_start": 100.0,
            "timeline_end": 150.0,
        },
        {
            "id": "p2",
            "text": "David Miller raised latency concerns with the event-driven option and recommended hybrid caching.",
            "speaker": "David Miller",
            "speakers": ["David Miller"],
            "technical_terms": ["event-driven", "Redis"],
            "dates": [],
            "numbers": ["10ms limit"],
            "references": [],
            "action_items": [],
            "timeline_start": 150.0,
            "timeline_end": 200.0,
        },
        {
            "id": "p3",
            "text": "The committee concluded to proceed with the hybrid caching model to guarantee sub-10ms latency.",
            "speaker": "Elena Rostova",
            "speakers": ["Elena Rostova"],
            "technical_terms": ["hybrid caching"],
            "dates": ["Oct 20"],
            "numbers": ["sub-10ms"],
            "references": ["Board Decision"],
            "action_items": [{"task": "Benchmark Redis cluster", "assignee": "David"}],
            "timeline_start": 200.0,
            "timeline_end": 280.0,
        }
    ]

    mock_provider = MagicMock()
    mock_provider.query.return_value = json.dumps({
        "conclusion": (
            "The committee evaluated three microservice architecture designs and addressed latency concerns raised by David Miller. "
            "They agreed to adopt a hybrid caching model utilizing Redis and Kafka to ensure sub-10ms latency for the payments processor. "
            "Sarah Lin and David Miller will prepare the design proposal and benchmark the Redis cluster ahead of the Oct 20 rollout."
        ),
        "explanation": "Synthesized architecture evaluation, latency constraints, and final decision into a 3-sentence executive conclusion."
    })

    with patch("services.ai_provider.get_provider", return_value=mock_provider):
        res = rom_service.conclude_points(points=points, agenda_context="Payment Gateway Architecture")

        # Verify thinking mode disabled
        assert mock_provider.query.called
        _, kwargs = mock_provider.query.call_args
        assert kwargs.get("think") is False
        assert kwargs.get("task_key") == "rom_conclude_points"

        # Verify replacement and removal ids
        assert res["replaced_point_id"] == "p1"
        assert res["removed_point_ids"] == ["p2", "p3"]

        concluded = res["concluded_point"]
        assert concluded["id"] == "p1"
        assert "hybrid caching" in concluded["text"]
        assert concluded["is_concluded"] is True

        # Verify metadata aggregation
        assert set(concluded["speakers"]) == {"Sarah Lin", "David Miller", "Elena Rostova"}
        assert len(concluded["action_items"]) == 2
        assert "Kafka" in concluded["technical_terms"]
        assert "Redis" in concluded["technical_terms"]
        assert "Oct 12" in concluded["dates"]
        assert "Oct 20" in concluded["dates"]
        assert concluded["timeline_start"] == 100.0
        assert concluded["timeline_end"] == 280.0


def test_conclude_points_fallback_synthesizes_without_verbatim_concatenation():
    """Verify conclude_points does not concatenate raw paragraphs when JSON parsing fails."""
    points = [
        {"id": "p1", "text": "First point detail.", "speaker": "Alice"},
        {"id": "p2", "text": "Second point detail.", "speaker": "Bob"}
    ]
    mock_provider = MagicMock()
    # Return malformed raw string with direct summary
    mock_provider.query.return_value = "The team evaluated first and second point details and reached consensus."

    with patch("services.ai_provider.get_provider", return_value=mock_provider):
        res = rom_service.conclude_points(points=points)
        assert res["concluded_point"]["text"] == "The team evaluated first and second point details and reached consensus."
        assert res["concluded_point"]["text"] != "First point detail. Second point detail."


@pytest.mark.asyncio
async def test_stage2_merge_3_of_7_points_preserves_other_4_points_intact():
    """Verify that merging 3 out of 7 points in Stage 2 replaces only the 3 points with an AI summary, keeping the other 4 unchanged."""
    from unittest.mock import AsyncMock
    from routers.rom_router import merge_stage2_points, MergePointsRequest

    original_7_points = [
        {"id": f"pt_{i}", "polished_text": f"Point {i} distinct content.", "timeline_start": float(i * 10), "timeline_end": float((i + 1) * 10), "speakers": [f"Speaker_{i}"]}
        for i in range(7)
    ]
    # Select 3 points: pt_2, pt_3, pt_4
    selected_ids = ["pt_3", "pt_2", "pt_4"]  # arbitrary selection order

    fake_data = {
        "stage2": {
            "status": "done",
            "polished_points": [dict(p) for p in original_7_points]
        }
    }

    mock_db = MagicMock()
    mock_provider = MagicMock()
    mock_provider.merge_discussion_points.return_value = {
        "polished_text": "The committee reached a unified consensus across points 2, 3, and 4 in two concise sentences.",
        "speakers": ["Speaker_2", "Speaker_3", "Speaker_4"],
        "technical_terms": ["ConsensusEngine"],
        "dates": [],
        "numbers": [],
        "references": [],
        "action_items": []
    }
    mock_provider._extract_clean_text_value.side_effect = lambda x: str(x)

    with patch("routers.rom_router._get_rom_data", new=AsyncMock(return_value=fake_data)), \
         patch("routers.rom_router._save_rom_data", new=AsyncMock()), \
         patch("routers.rom_router._record_edit_history", new=AsyncMock(return_value="chg_123")), \
         patch("services.ai_provider.get_provider", return_value=mock_provider):

        req = MergePointsRequest(point_ids=selected_ids)
        result = await merge_stage2_points(
            recording_id="rec_123",
            req=req,
            current_user={"id": "user_1"},
            db=mock_db
        )

        assert result["status"] == "success"
        merged_pt = result["merged_point"]
        assert merged_pt["polished_text"] == "The committee reached a unified consensus across points 2, 3, and 4 in two concise sentences."

        new_polished_points = fake_data["stage2"]["polished_points"]
        # Exactly 5 points remaining (original 7 minus 3 selected + 1 merged)
        assert len(new_polished_points) == 5

        # Check that the 4 unselected points (pt_0, pt_1, pt_5, pt_6) remain completely unchanged and in position
        assert new_polished_points[0]["id"] == "pt_0"
        assert new_polished_points[0]["polished_text"] == "Point 0 distinct content."
        assert new_polished_points[1]["id"] == "pt_1"
        assert new_polished_points[1]["polished_text"] == "Point 1 distinct content."

        # Merged point is at index 2 (earliest selected point index)
        assert new_polished_points[2]["id"] == merged_pt["id"]

        assert new_polished_points[3]["id"] == "pt_5"
        assert new_polished_points[3]["polished_text"] == "Point 5 distinct content."
        assert new_polished_points[4]["id"] == "pt_6"
        assert new_polished_points[4]["polished_text"] == "Point 6 distinct content."


def test_final_rom_conclude_3_of_7_points_preserves_other_4_points_intact():
    """Verify that concluding 3 out of 7 Final ROM points replaces those 3 with a 1-3 sentence summary, keeping the other 4 unchanged."""
    all_7_points = [
        {"id": f"p_{i}", "text": f"Point {i} text.", "speaker": f"Speaker_{i}"}
        for i in range(7)
    ]
    # Select 3 points: p_1, p_3, p_4
    selected_points = [all_7_points[1], all_7_points[3], all_7_points[4]]

    mock_provider = MagicMock()
    mock_provider.query.return_value = json.dumps({
        "conclusion": "Points 1, 3, and 4 were synthesized into this cohesive two-sentence summary. All key outcomes were preserved.",
        "explanation": "Summarized selected points into 2 sentences."
    })

    with patch("services.ai_provider.get_provider", return_value=mock_provider):
        res = rom_service.conclude_points(points=selected_points, agenda_context="Executive Strategy")

        assert res["replaced_point_id"] == "p_1"
        assert set(res["removed_point_ids"]) == {"p_3", "p_4"}

        concluded = res["concluded_point"]
        assert concluded["text"] == "Points 1, 3, and 4 were synthesized into this cohesive two-sentence summary. All key outcomes were preserved."

        # Simulate the frontend replacement logic
        replaced_id = res["replaced_point_id"]
        removed_ids = set(res["removed_point_ids"])
        new_points = []
        for pt in all_7_points:
            if pt["id"] == replaced_id:
                new_points.append(concluded)
            elif pt["id"] not in removed_ids:
                new_points.append(pt)

        # 5 points total (4 untouched + 1 concluded)
        assert len(new_points) == 5
        assert new_points[0]["id"] == "p_0"
        assert new_points[0]["text"] == "Point 0 text."
        assert new_points[1]["id"] == "p_1"
        assert new_points[1]["text"] == concluded["text"]
        assert new_points[2]["id"] == "p_2"
        assert new_points[2]["text"] == "Point 2 text."
        assert new_points[3]["id"] == "p_5"
        assert new_points[3]["text"] == "Point 5 text."
        assert new_points[4]["id"] == "p_6"
        assert new_points[4]["text"] == "Point 6 text."

