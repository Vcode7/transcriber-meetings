"""PPT Testing: slide segmentation, conclusion cutoff, meeting-date relevance, timeline and insight assembly."""
import json
import re

import pytest

from ppt_fixtures import MEETING_DATE, build_pptx
from services.ppt_insights import (
    FOCUS_WINDOWS,
    Slide,
    analyze_presentation,
    apply_conclusion_cutoff,
    classify_closing,
    deduplicate,
    extract_timeline,
    group_by_project,
    insight_point_text,
    parse_dates,
    parse_json_object,
    slide_relevance,
    split_into_slides,
    temporal_class,
    timeline_context,
)

WINDOW = FOCUS_WINDOWS["standard"]


@pytest.fixture(scope="module")
def deck_slides(tmp_path_factory):
    from services.doc_extractor import extract_text_from_file
    path = str(tmp_path_factory.mktemp("ppt") / "atlas_review.pptx")
    build_pptx(path)
    return split_into_slides(extract_text_from_file(path, "atlas_review.pptx"), "atlas_review.pptx")


# ── Segmentation ──────────────────────────────────────────────────────────────

def test_pptx_is_split_into_numbered_slides_with_titles(deck_slides):
    assert [s.number for s in deck_slides] == list(range(1, 14))
    assert deck_slides[2].title == "Programme Timeline" and "Phase 3" in deck_slides[2].text
    assert deck_slides[3].title.startswith("Project Atlas")


def test_pdf_pages_and_plain_documents_are_segmented():
    pages = split_into_slides("", "x.pdf", pages=["Status\nAll good", "Risks\nVendor"])
    assert [(s.number, s.title, s.kind) for s in pages] == [(1, "Status", "page"), (2, "Risks", "page")]
    doc = split_into_slides("\n\n".join(f"Paragraph {i} " + "x" * 400 for i in range(8)), "x.docx")
    assert len(doc) >= 2 and all(s.kind == "section" for s in doc)


# ── Conclusion cutoff ─────────────────────────────────────────────────────────

def test_stops_at_conclusion_and_excludes_references_and_appendix(deck_slides):
    kept, excluded, cutoff = apply_conclusion_cutoff(deck_slides)
    assert cutoff["slide"] == 10 and cutoff["kind"] == "conclusion"
    assert kept[-1].title == "Conclusion"                      # the conclusion itself is analysed
    assert [e["number"] for e in excluded] == [11, 12, 13]     # thank-you, references, appendix
    assert {e["reason"] for e in excluded} == {"after_conclusion", "back_matter_after_conclusion"}


def test_section_ending_slides_do_not_stop_processing(deck_slides):
    section_end = next(s for s in deck_slides if s.title == "End of Section 1")
    assert classify_closing(section_end) is None
    for title in ("Summary", "Section 2 wrap-up", "Key Takeaways", "Next Steps"):
        assert classify_closing(Slide(5, title, f"{title}\nPhase 3 continues into Q1 2027")) is None


@pytest.mark.parametrize("title,text,kind", [
    ("Thank You", "Thank You\nQuestions?", "closing"),
    ("Q&A", "Q&A", "closing"),
    ("Concluding Remarks", "Concluding Remarks\nWe are on track", "conclusion"),
    ("Bibliography", "Bibliography\nSmith 2020", "back_matter"),
    ("Contact", "Contact\nThanks for listening!", "closing"),     # short slide with a thank-you
])
def test_closing_slide_kinds(title, text, kind):
    assert classify_closing(Slide(9, title, text)) == kind


def test_thank_you_slide_stops_and_is_excluded():
    slides = [Slide(1, "Review", "Review"), Slide(2, "Status", "Status\nAll green"),
              Slide(3, "Thank you", "Thank you"), Slide(4, "References", "References\nA")]
    kept, excluded, cutoff = apply_conclusion_cutoff(slides)
    assert [s.number for s in kept] == [1, 2] and cutoff["kind"] == "closing"
    assert [e["number"] for e in excluded] == [3, 4]


def test_first_slide_never_triggers_the_cutoff():
    slides = [Slide(1, "Thank you for joining", "Thank you for joining"), Slide(2, "Status", "Status\nok")]
    kept, excluded, cutoff = apply_conclusion_cutoff(slides)
    assert cutoff is None and len(kept) == 2


@pytest.mark.parametrize("slide,position,kind", [
    (Slide(1, "Atlas Programme Review", "Atlas Programme Review\nSteering Committee\nOctober 2026"), 0, "title"),
    (Slide(2, "Agenda", "Agenda\nTimeline\nStatus\nRisks"), 1, "agenda"),
    (Slide(7, "End of Section 1", "End of Section 1\nNext: risks"), 6, "divider"),
    (Slide(4, "Go-live", "Go-live\nMoved to 15-Nov-2026"), 3, None),          # short but dated → real content
    (Slide(4, "Budget", "Budget\nSpend at 92%"), 3, None),                     # short but has figures
])
def test_structural_slides(slide, position, kind):
    from services.ppt_insights import is_structural
    assert is_structural(slide, position) == kind


def test_meta_insights_are_filtered():
    from services.ppt_insights import _META_INSIGHT
    assert _META_INSIGHT.search("This slide serves as a transition point")
    assert _META_INSIGHT.search("The agenda includes the Atlas build")
    assert not _META_INSIGHT.search("Dry-run 2 slipped to 30-Oct-2026")


def test_dates_must_be_grounded_in_the_slide():
    from services.ppt_insights import grounded_dates, ungrounded_dates
    slide_text = "Dry-run 2 moved from 15-Sep-2026 to 30-Oct-2026. Phase 3 ends Dec 2026."
    assert grounded_dates(["30-Oct-2026", "Dec 2026", "31-Oct-2026", "Q1 2027"], slide_text) == ["30-Oct-2026", "Dec 2026"]
    assert ungrounded_dates("due October 30, 2026 and 31-Oct-2026", [slide_text]) == ["31-Oct-2026"]


def test_conclusion_slide_keeps_at_most_one_new_item():
    slides = [
        Slide(1, "Review", "Review"),
        Slide(2, "Atlas status", "Atlas status\nData migration dry-run 2 delayed to 30-Oct-2026 due to vendor data"),
        Slide(3, "Conclusion", "Conclusion\nData migration slipped; cutover readiness is the focus for Nov 2026"),
    ]

    def infer(prompt, max_tokens):
        if "SLIDE 2 —" in prompt:
            return json.dumps({"project": "Atlas", "insights": [{
                "type": "delay", "status": "delayed", "headline": "Data migration dry-run 2 delayed",
                "summary": "Dry-run 2 delayed to 30-Oct-2026 due to vendor data issues.", "dates": ["30-Oct-2026"]}]})
        if "SLIDE 3 —" in prompt:
            return json.dumps({"project": "Atlas", "insights": [
                {"type": "delay", "status": "delayed", "headline": "Data migration delayed",
                 "summary": "Dry-run 2 delayed to 30-Oct-2026 due to vendor data issues.", "dates": []},
                {"type": "action", "status": "planned", "headline": "Cutover readiness is next month's focus",
                 "summary": "The team will concentrate on cutover readiness in Nov 2026.", "dates": ["Nov 2026"]},
                {"type": "status", "status": "ongoing", "headline": "Programme continues",
                 "summary": "Work continues in Nov 2026.", "dates": ["Nov 2026"]},
            ]})
        return "{}"

    result = analyze_presentation(slides, meeting=MEETING_DATE, infer=infer)
    conclusion_items = [i for i in result["insights"] if i["slides"] == [3]]
    assert [i["headline"] for i in conclusion_items] == ["Cutover readiness is next month's focus"]


def test_range_spanning_the_meeting_is_current():
    from services.ppt_insights import insight_temporal
    temporal, _ = insight_temporal({"dates": ["Apr 2026", "Dec 2026"]}, [], MEETING_DATE, WINDOW)
    assert temporal == "current"


# ── Dates and relevance ───────────────────────────────────────────────────────

@pytest.mark.parametrize("text,start,end", [
    ("22-Jun-2026", "2026-06-22", "2026-06-22"),
    ("June 22, 2026", "2026-06-22", "2026-06-22"),
    ("2026-06-22", "2026-06-22", "2026-06-22"),
    ("15/03/2026", "2026-03-15", "2026-03-15"),
    ("Q3 2026", "2026-07-01", "2026-09-30"),
    ("H2 2026", "2026-07-01", "2026-12-31"),
    ("Jun 2026", "2026-06-01", "2026-06-30"),
    ("Mar'27", "2027-03-01", "2027-03-31"),
    ("FY2026-27", "2026-04-01", "2027-03-31"),
    ("2025", "2025-01-01", "2025-12-31"),
])
def test_parse_dates(text, start, end):
    spans = parse_dates(text)
    assert len(spans) == 1
    assert (spans[0].start.isoformat(), spans[0].end.isoformat()) == (start, end)


def test_parse_dates_ignores_plain_numbers():
    assert parse_dates("Phase 2 kickoff with 500 users and 70% complete") == []


def test_temporal_classes_relative_to_meeting():
    cls = lambda t: temporal_class(parse_dates(t)[0], MEETING_DATE, WINDOW)
    assert cls("Oct 2026") == "current"
    assert cls("15-Nov-2026") == "upcoming"
    assert cls("05-Aug-2026") == "recent"
    assert cls("March 2019") == "past"
    assert cls("2030") == "future"


def test_far_dated_slides_are_out_of_window(deck_slides):
    by_title = {s.title: s for s in deck_slides}
    assert slide_relevance(by_title["Programme History"], MEETING_DATE, WINDOW)[0] == "out_of_window"
    assert slide_relevance(by_title["Long-term Vision 2030"], MEETING_DATE, WINDOW)[0] == "out_of_window"
    assert slide_relevance(by_title["Programme Timeline"], MEETING_DATE, WINDOW)[0] == "dated"
    assert slide_relevance(by_title["Agenda"], MEETING_DATE, WINDOW)[0] == "undated"


def test_far_dates_with_live_status_words_are_still_analysed():
    slide = Slide(4, "Legacy migration", "Legacy migration\nContract signed 2019; migration still ongoing and delayed")
    assert slide_relevance(slide, MEETING_DATE, WINDOW)[0] == "dated"


# ── Timeline and project phase ────────────────────────────────────────────────

def test_current_phase_and_upcoming_milestones(deck_slides):
    kept, _, _ = apply_conclusion_cutoff(deck_slides)
    ctx = timeline_context(extract_timeline(kept), MEETING_DATE, WINDOW)
    assert ctx["current_phase"][0]["label"].startswith("Phase 3")
    upcoming = [(u["start"], u["slide"]) for u in ctx["upcoming"]]
    assert ("2026-10-31", 8) in upcoming      # cutover plan
    assert ("2026-11-15", 5) in upcoming      # app store submission
    labels = " ".join(u["label"] for u in ctx["upcoming"])
    assert "from" not in labels.split() and "2026" not in labels   # dates stripped cleanly from labels
    assert all(r["start"] >= "2026-06" for r in ctx["recent"])


def test_partial_percent_is_ongoing_not_completed(deck_slides):
    items = extract_timeline(deck_slides)
    sap = next(i for i in items if i.label.startswith("SAP integration"))
    assert sap.status_hint == "ongoing"
    dry_run = next(i for i in items if "dry-run" in i.label)
    assert dry_run.status_hint == "delayed" and "due to vendor data quality issues" in dry_run.label


# ── Full orchestration with a scripted LLM ────────────────────────────────────

SCRIPTED = {
    3: {"project": "Atlas Programme", "insights": [
        {"type": "status", "status": "ongoing", "headline": "Programme is in Phase 3 Build & Integration",
         "summary": "Phase 3 runs Apr 2026 to Dec 2026 and is in progress; Phase 4 UAT starts Jan 2027.",
         "attention": "Confirm Phase 3 exit criteria", "dates": ["Apr 2026", "Dec 2026"]},
        {"type": "progress", "status": "completed", "headline": "Discovery phase finished",
         "summary": "Discovery completed in Jun 2025.", "attention": "", "dates": ["Jun 2025"]},   # far → dropped
    ]},
    4: {"project": "Project Atlas", "insights": [
        {"type": "delay", "status": "delayed", "headline": "Data migration dry-run 2 slipped to 30-Oct-2026",
         "summary": "Dry-run 2 moved from 15-Sep-2026 to 30-Oct-2026 because of vendor data quality issues.",
         "attention": "Vendor data clean-up before 30-Oct-2026", "dates": ["15-Sep-2026", "30-Oct-2026"]},
    ]},
    5: {"project": "Project Beacon", "insights": [
        {"type": "dependency", "status": "planned", "headline": "Store submission depends on Atlas API v2",
         "summary": "Submission planned for 15-Nov-2026 needs Atlas API v2 by 01-Nov-2026.",
         "attention": "Confirm API v2 date", "dates": ["15-Nov-2026", "01-Nov-2026"]},
    ]},
    8: {"project": "Atlas", "insights": [
        {"type": "delay", "status": "delayed", "headline": "Data migration dry-run 2 slipped to 30-Oct-2026",
         "summary": "Dry-run 2 moved from 15-Sep-2026 to 30-Oct-2026 because of vendor data quality issues.",
         "attention": "", "dates": ["30-Oct-2026"]},                                            # duplicate of slide 4
        {"type": "action", "status": "planned", "headline": "Priya to finalise cutover plan",
         "summary": "The cutover plan is due 31-Oct-2026.", "attention": "", "dates": ["31-Oct-2026"]},
    ]},
}


def scripted_infer(calls):
    def infer(prompt, max_tokens):
        calls.append(prompt)
        m = re.search(r"^(SLIDE|PAGE|SECTION) (\d+) —", prompt, re.MULTILINE)
        if m:
            return "```json\n" + json.dumps(SCRIPTED.get(int(m.group(2)), {"project": "", "insights": []})) + "\n```"
        if '"needs_attention"' in prompt:
            return json.dumps({"summary": "Atlas is in Phase 3; data migration slipped.", "needs_attention": ["Dry-run 2"]})
        return "Phase 3 is under way; the data migration slip needs attention."
    return infer


def test_analysis_end_to_end(deck_slides):
    calls = []
    result = analyze_presentation(deck_slides, meeting=MEETING_DATE, infer=scripted_infer(calls), filename="atlas.pptx")

    analysed_slides = sorted(int(re.search(r"^SLIDE (\d+) —", c, re.M).group(1)) for c in calls if re.search(r"^SLIDE (\d+) —", c, re.M))
    # Far past/future slides and everything after the conclusion are never sent to the LLM
    assert 6 not in analysed_slides and 9 not in analysed_slides
    assert not any(n >= 11 for n in analysed_slides)
    assert 10 in analysed_slides                                     # the conclusion itself is analysed
    status = {s["number"]: s["status"] for s in result["slides"]}
    assert status[6] == status[9] == "skipped_out_of_window"
    # Title, agenda and the section divider carry no insights; the divider did not stop processing
    assert status[1] == status[2] == status[7] == "skipped_structural"
    assert status[8] == "analyzed"
    assert result["cutoff"]["slide"] == 10

    # Later slides are told what earlier slides already reported, so they don't repeat it
    slide8_prompt = next(c for c in calls if "SLIDE 8 —" in c)
    assert "(slide 4) Data migration dry-run 2 slipped" in slide8_prompt
    assert "CONCLUSION" in next(c for c in calls if "SLIDE 10 —" in c)
    assert "CONCLUSION" not in slide8_prompt

    # The meeting date and window are given to the model, with relative dates
    slide4_prompt = next(c for c in calls if "SLIDE 4 —" in c)
    assert "12 Oct 2026" in slide4_prompt and "after the meeting" in slide4_prompt
    assert "Phase 3" in slide4_prompt                                # current phase passed as context

    insights = result["insights"]
    headlines = [i["headline"] for i in insights]
    assert "Discovery phase finished" not in headlines               # dated far from the meeting → dropped
    assert result["dropped_out_of_window"] == 1
    dry = [i for i in insights if "dry-run 2" in i["headline"].lower()]
    assert len(dry) == 1 and dry[0]["slides"] == [4, 8]              # de-duplicated, both slides kept
    assert all(i["slides"] and i["ref"].startswith("P") for i in insights)

    projects = {p["name"]: p for p in result["projects"]}
    assert "Project Atlas" in projects and "Project Beacon" in projects
    atlas = projects["Project Atlas"]
    assert atlas["overall_status"] == "delayed" and atlas["needs_attention"]
    assert result["overview"]
    assert insight_point_text(dry[0]).startswith("[Slide 4, 8 · Project Atlas]")


def test_llm_failures_do_not_break_the_analysis(deck_slides):
    def broken(prompt, max_tokens):
        raise RuntimeError("model offline")
    result = analyze_presentation(deck_slides, meeting=MEETING_DATE, infer=broken)
    assert result["insights"] == []
    assert any(s["status"] == "error" for s in result["slides"])


# ── Helpers ───────────────────────────────────────────────────────────────────

def test_parse_json_object_tolerates_noise():
    raw = '<think>hmm</think> Sure! ```json\n{"insights": [{"headline": "x",}],}\n```'
    assert parse_json_object(raw) == {"insights": [{"headline": "x"}]}
    assert parse_json_object("no json here") == {}


def test_deduplicate_keeps_distinct_projects_apart():
    base = {"summary": "Go-live moved to Nov 2026", "attention": "", "dates": [], "priority": 0.8,
            "headline": "Go-live moved", "status": "delayed", "type": "delay"}
    items = [{**base, "project": "Atlas", "slides": [2], "id": "a"},
             {**base, "project": "Beacon", "slides": [3], "id": "b"},
             {**base, "project": "Atlas", "slides": [5], "id": "c"}]
    out = deduplicate(items)
    assert len(out) == 2
    assert sorted(i["project"] for i in out) == ["Atlas", "Beacon"]


def test_group_by_project_merges_name_variants():
    mk = lambda proj, slide, status: {"project": proj, "slides": [slide], "priority": 0.8, "status": status,
                                      "summary": "s", "headline": "h", "attention": "", "dates": [], "id": f"{proj}{slide}"}
    groups = group_by_project([mk("Project Atlas", 4, "ongoing"), mk("Atlas", 8, "delayed"), mk("", 2, "unknown")])
    names = [g["name"] for g in groups]
    assert names == ["Project Atlas", "General"]
    assert groups[0]["slides"] == [4, 8] and groups[0]["overall_status"] == "delayed"
