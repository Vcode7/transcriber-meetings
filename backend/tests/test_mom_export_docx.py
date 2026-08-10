import docx
import io
import pytest
from routers.mom_router import generate_mom_docx

def test_generate_mom_docx():
    mom = {
        "title": "Quarterly Planning & Architecture Review",
        "date": "07 August 2026",
        "participants": ["Alice", "Bob", "Charlie"],
        "planned_start_time": "10:00 AM",
        "actual_start_time": "10:05 AM",
        "planned_end_time": "11:00 AM",
        "actual_end_time": "11:05 AM",
        "introduction": "The committee discussed Q3 goals, system architecture, and team resource allocation.",
        "points_discussed": [
            "Architecture Migration: Agreed to migrate background workers to containerized services.",
            "Database Indexing: Reviewed slow queries and approved composite index additions."
        ],
        "action_items": [
            {"task": "Prepare Docker deployment manifests", "owner": "Alice", "deadline": "Friday"},
            {"task": "Run database index benchmark", "owner": "Bob", "deadline": "ASAP"},
            {"task": "Update documentation portal", "owner": "Unassigned", "deadline": "Next Week"}
        ],
        "conclusion": "Meeting adjourned successfully. Next sync scheduled for next Monday."
    }

    docx_bytes = generate_mom_docx(mom)

    assert isinstance(docx_bytes, bytes)
    assert len(docx_bytes) > 0

    # Parse generated docx with python-docx
    doc = docx.Document(io.BytesIO(docx_bytes))

    # Combine all paragraph text
    full_text = "\n".join(p.text for p in doc.paragraphs)

    assert "Quarterly Planning & Architecture Review" in full_text
    assert "07 August 2026" in full_text
    assert "Alice, Bob, Charlie" in full_text
    assert "INTRODUCTION" in full_text
    assert "POINTS DISCUSSED" in full_text
    assert "Architecture Migration" in full_text
    assert "ACTION POINTS" in full_text
    assert "Alice" in full_text
    assert "Prepare Docker deployment manifests" in full_text
    assert "(Due: Friday)" in full_text
    assert "General" in full_text
    assert "Update documentation portal" in full_text
    assert "CONCLUSION" in full_text
