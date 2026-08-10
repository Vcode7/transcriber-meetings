import pytest
from unittest.mock import patch, MagicMock
from services.llm import summarize_long_points, summarize_long_discussion_point

def test_summarize_long_points_filtering_and_condensing():
    short_point = "Budget Review: Approved Q3 budget allocations."
    long_point = (
        "Project Architecture Review: The team engaged in an extremely lengthy and detailed discussion "
        "regarding the microservice architecture migration. They analyzed various database schemas, "
        "evaluated caching mechanisms including Redis and Memcached, discussed network latency impact on "
        "the frontend rendering pipeline, debated serverless deployment vs dedicated container clusters, "
        "and ultimately agreed to proceed with a phased containerized deployment strategy starting with "
        "non-critical services over the next two quarters while maintaining strict zero-downtime SLA requirements."
    )

    points = [short_point, long_point]
    threshold = 30  # 30 words threshold

    # Count words
    short_words = len(short_point.split())
    long_words = len(long_point.split())

    assert short_words <= threshold
    assert long_words > threshold

    mock_provider = MagicMock()
    mock_provider._infer.return_value = "Project Architecture Review: Phased containerized migration approved for next two quarters with zero downtime SLA."

    with patch("services.llm.get_provider", return_value=mock_provider):
        updated_points, condensed_count = summarize_long_points(points, threshold=threshold)

        # Short point should remain identical
        assert updated_points[0] == short_point
        # Long point should be condensed by LLM
        assert updated_points[1] == "Project Architecture Review: Phased containerized migration approved for next two quarters with zero downtime SLA."
        # Exactly 1 point condensed
        assert condensed_count == 1
        # LLM infer called exactly once for the long point
        assert mock_provider._infer.call_count == 1


def test_summarize_long_points_none_exceed_threshold():
    points = ["Point 1: Short summary.", "Point 2: Another short point."]
    threshold = 70

    mock_provider = MagicMock()

    with patch("services.llm.get_provider", return_value=mock_provider):
        updated_points, condensed_count = summarize_long_points(points, threshold=threshold)

        assert updated_points == points
        assert condensed_count == 0
        assert mock_provider._infer.call_count == 0
