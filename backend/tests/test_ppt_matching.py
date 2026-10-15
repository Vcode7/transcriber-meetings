"""PPT ↔ long point matching: candidate scoring, strict verification, one-to-one assignment, synthesis."""
import json
import re

import numpy as np

from ppt_fixtures import LONG_POINTS, MEETING_DATE
from services.ppt_matching import (
    final_point_record,
    lexical_score,
    match_points,
    project_mentioned,
    score_candidates,
)


def _ins(ref, project, headline, summary, slides, priority=0.8, status="ongoing", attention=""):
    return {"id": f"id-{ref}", "ref": ref, "project": project, "headline": headline, "summary": summary,
            "attention": attention, "dates": [], "slides": slides, "status": status, "type": "status",
            "priority": priority}


INSIGHTS = [
    _ins("P1", "Project Atlas", "Data migration dry-run 2 slipped to 30-Oct-2026",
         "Dry-run 2 moved from 15-Sep-2026 to 30-Oct-2026 because of vendor data quality issues.", [4],
         priority=1.0, status="delayed", attention="Clean customer master data"),
    _ins("P2", "Project Beacon", "Store submission depends on Atlas API v2",
         "App store submission planned for 15-Nov-2026 needs Atlas API v2 by 01-Nov-2026.", [5], status="planned"),
    _ins("P3", "Atlas Programme", "Extra QA budget approved",
         "Steering committee approved additional QA budget of $250K on 01-Oct-2026.", [8], status="completed"),
    _ins("P4", "Atlas Programme", "Vendor attrition risk",
         "Attrition of two senior vendor engineers may slow integration testing.", [8], status="at_risk"),
    _ins("P5", "Project Atlas", "Performance testing scheduled",
         "Performance testing is scheduled for 20-Nov-2026.", [4], priority=0.9, status="planned"),
]


def test_lexical_score_handles_acronyms_and_typos():
    assert lexical_score("TCEC migration on track", "T C E C migration is on track")["score"] > 0.5
    assert lexical_score("Iron Bird rig ready", "the iron brid rig is ready")["score"] > 0.4
    assert lexical_score("QA budget approved", "Parking for staff next quarter")["score"] < 0.2


def test_project_mentioned_is_typo_tolerant():
    assert project_mentioned("Project Beacon", "for the beacon mobile app the store submission")
    assert project_mentioned("Atlas Programme", "the atlas programe board")
    assert not project_mentioned("Project Beacon", "the office move to the new building")
    assert not project_mentioned("General", "anything")


def test_unrelated_long_point_gets_no_candidates():
    cands, semantic = score_candidates(INSIGHTS, LONG_POINTS)
    assert semantic is False
    assert cands[3] == []                                   # office move: nothing in the deck
    assert cands[0][0]["index"] == 0                        # dry-run discussion → dry-run insight first


def test_semantic_scores_are_blended_when_an_embedder_is_available():
    def embed(texts):
        # Toy embedding: bag of a few domain words
        vocab = ["migration", "dry", "beacon", "store", "budget", "qa", "vendor", "office", "performance"]
        return np.array([[t.lower().count(w) + 0.01 for w in vocab] for t in texts], dtype=np.float32)
    cands, semantic = score_candidates(INSIGHTS, LONG_POINTS, embed)
    assert semantic is True
    assert cands[1][0]["index"] == 1 and cands[1][0]["semantic"] is not None


def _verifier(decisions, calls):
    """Fake LLM: verification answers from `decisions` (long-point keyword → refs); synthesis echoes facts."""
    def infer(prompt, max_tokens):
        calls.append(prompt)
        if "DISCUSSION POINT (from the meeting transcript" in prompt:
            discussion = re.search(r'DISCUSSION POINT[^\n]*\n"(.*?)"', prompt, re.S).group(1)
            offered = set(re.findall(r"^(P\d+) \[", prompt, re.M))
            for key, refs in decisions.items():
                if key in discussion:
                    return json.dumps({"matches": [{"id": r, "confidence": "high", "reason": "same topic"}
                                                   for r in refs if r in offered]})
            return json.dumps({"matches": []})
        return json.dumps({"headline": "Dry-run 2 slip blocks cutover plan", "status": "delayed",
                           "summary": "Slide 4 shows dry-run 2 moved to 30-Oct-2026; the meeting tied the cutover plan to it.",
                           "attention": "Data clean-up this week"})
    return infer


def test_match_points_assigns_each_insight_once_and_keeps_slide_refs():
    calls = []
    # The LLM tries to give P1 to two long points; it must be used only once (best score wins)
    decisions = {"data migration": ["P1"], "Beacon": ["P2", "P1"], "QA budget": ["P3", "P4"]}
    result = match_points([dict(i) for i in INSIGHTS], LONG_POINTS, infer=_verifier(decisions, calls), meeting=MEETING_DATE)

    groups = result["comparison_groups"]
    used = [ref for g in groups for ref in g["insight_refs"]]
    assert len(used) == len(set(used))                              # one-to-one use of slide insights
    by_long = {g["stage2_point"]["id"]: g for g in groups}
    assert by_long["lp1"]["insight_refs"] == ["P1"] and by_long["lp1"]["slides"] == [4]
    assert by_long["lp2"]["insight_refs"] == ["P2"]
    assert sorted(by_long["lp3"]["insight_refs"]) == ["P3", "P4"] and by_long["lp3"]["slides"] == [8]
    assert "lp4" not in by_long                                     # unrelated discussion → no point
    assert [g["rank"] for g in groups] == list(range(1, len(groups) + 1))

    # Unmatched but important slide insight is surfaced as "not discussed"
    assert [u["ref"] for u in result["unmatched_priority_insights"]] == ["P5"]

    record = final_point_record(by_long["lp1"])
    assert record["slides"] == [4] and record["projects"] == ["Project Atlas"]
    assert record["text"].startswith("Dry-run 2 slip blocks cutover plan.") and "Needs attention:" in record["text"]
    # Verification prompt is strict about generic overlaps
    verify_prompt = next(c for c in calls if "DISCUSSION POINT" in c)
    assert "Sharing generic words" in verify_prompt and "[Slide 4 · Project Atlas]" in verify_prompt


def test_rejected_candidates_produce_no_points():
    result = match_points([dict(i) for i in INSIGHTS], LONG_POINTS, infer=_verifier({}, []), meeting=MEETING_DATE)
    assert result["comparison_groups"] == []
    assert result["stats"]["verified_pairs"] == 0


def test_synthesis_retries_when_a_date_is_invented():
    from services.ppt_matching import synthesize_group
    prompts = []
    replies = iter([
        json.dumps({"headline": "Dry-run slip", "status": "delayed",
                    "summary": "Dry-run 2 moved to 30-Oct-2026; data cleansing due 31-Oct-2026.", "attention": ""}),
        json.dumps({"headline": "Dry-run slip", "status": "delayed",
                    "summary": "Dry-run 2 moved to 30-Oct-2026; data cleansing must finish before it.", "attention": ""}),
    ])

    def infer(prompt, max_tokens):
        prompts.append(prompt)
        return next(replies)

    out = synthesize_group("The dry run slipped to the end of October.", [dict(INSIGHTS[0])], infer, MEETING_DATE)
    assert len(prompts) == 2 and "Do NOT mention 31-Oct-2026" in prompts[1]
    assert "31-Oct" not in out["text"] and "30-Oct-2026" in out["text"]


def test_llm_outage_falls_back_to_high_similarity_only():
    def broken(prompt, max_tokens):
        raise RuntimeError("offline")
    result = match_points([dict(i) for i in INSIGHTS], LONG_POINTS, infer=broken, meeting=MEETING_DATE)
    for g in result["comparison_groups"]:
        assert g["match_score"] >= 0.5 and g["match_type"] == "similarity_match"
        assert g["final_point"]                                   # falls back to the slide finding text



# ── Small local models: decorated ids and truncated JSON (seen with gemma-4-E4B Q3 via Ollama) ──

def test_verified_ids_decorated_by_the_model_are_accepted():
    from services.ppt_matching import parse_verified
    raw = json.dumps({"matches": [
        {"id": "P5 [Slide 5 · Proposed Ordinance 2026-0108]", "confidence": "high", "reason": "deadline"},
        {"id": "3", "confidence": "medium", "reason": "appraisal"},
        {"id": "P9", "confidence": "high", "reason": "not a candidate"},
    ]})
    assert parse_verified(raw, {"P5": 0, "P3": 1}) == [(0, "high", "deadline"), (1, "medium", "appraisal")]


def test_truncated_json_is_repaired():
    from services.ppt_insights import parse_json_object
    raw = '{"matches": [\n  {"id": "P5", "confidence": "high", "reason": "same ordinance {2026-0108}"},\n  {"id": "P3", "confidence": "medium"}'
    assert [m["id"] for m in parse_json_object(raw)["matches"]] == ["P5", "P3"]
    assert parse_json_object('{"a": 1}') == {"a": 1}
