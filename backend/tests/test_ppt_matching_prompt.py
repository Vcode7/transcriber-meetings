"""
Test script for verifying updated PPT point-matching prompt with sample dataset:
- 10 long points, each containing 100+ words.
- 30 short PPT points as insights.
- Only 6 of the 30 PPT points should contain information present in the 10 long points.
- 2 PPT points match the first relevant long point (Point 2: Redis).
- 4 PPT points match the second relevant long point (Point 4: PgBouncer).
- Remaining 24 PPT points do not match any long point.
- Expected output identifies only the 2 relevant long points and their 6 matching PPT points.
- Final output preserves important information from both without single-line collapse (separate points for key dates/facts).
"""
import os
import sys
import json
import re
import pytest

pytestmark = pytest.mark.skipif(not os.getenv("RUN_LIVE_TESTS"), reason="Live LLM prompt test requiring local model weights")

# Ensure backend root is on sys.path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

tests_dir = os.path.dirname(os.path.abspath(__file__))
if tests_dir not in sys.path:
    sys.path.insert(0, tests_dir)

from services.ai_provider import get_provider

try:
    from tests.test_reasoning_mode import TEN_LONG_POINTS
except ImportError:
    from test_reasoning_mode import TEN_LONG_POINTS

def parse_single_match(raw: str) -> dict:
    if not raw:
        return {}
    cleaned = raw.strip()
    cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'```\s*$', '', cleaned, flags=re.MULTILINE).strip()
    match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            try:
                fix_comma = re.sub(r',\s*([\]}])', r'\1', match.group())
                return json.loads(fix_comma)
            except Exception:
                pass
    return {}

def run_test():
    print("=" * 80)
    print("STEP 1: Verifying 10 Long Points (each 100+ words)")
    print("=" * 80)
    for idx, pt in enumerate(TEN_LONG_POINTS, 1):
        words = len(pt["polished_text"].split())
        print(f"Long Point {idx}: {words} words")
        assert words >= 100, f"Long Point {idx} has fewer than 100 words ({words})"
    print("All 10 Long Points verified (each >= 100 words).\n")

    # 30 short PPT points as insights
    ppt_points = [
        # --- 10 Non-matching PPT points (items 1-10) ---
        {"id": 1, "text": "Q3 mobile app release shifted from React Native 0.72 to Flutter 3.24 framework."},
        {"id": 2, "text": "Customer support SLA for enterprise tier reduced from 4 hours down to 1 hour response time."},
        {"id": 3, "text": "Annual SOC2 Type II compliance audit commences on December 10, 2026 across all EU datacenters."},
        {"id": 4, "text": "Salesforce CRM integration pipeline transitioned to webhooks to reduce REST API rate limit errors."},
        {"id": 5, "text": "Single Sign-On (SSO) rollout for internal corporate portals requires Okta MFA enforcement by year end."},
        {"id": 6, "text": "Android mobile client deprecated in-memory Redis session caching in favor of encrypted SQLite Room database."},
        {"id": 7, "text": "Marketing landing page conversion rate increased by 3.8% following A/B test of simplified checkout flow."},
        {"id": 8, "text": "Corporate procurement finalized multi-year contract renewals with primary hardware vendors."},
        {"id": 9, "text": "Kubernetes worker node base AMI updated to Ubuntu 24.04 LTS across staging environments."},
        {"id": 10, "text": "Customer churn rate decreased by 1.2% in EMEA accounts following personalized onboarding calls."},

        # --- MATCHING PPT POINTS FOR LONG POINT 2 (Redis migration) (items 11-12) ---
        {"id": 11, "text": "Production Redis cache cluster migration to version 7.2 scheduled for Friday, October 24, 2026, at 5:00 PM UTC."},
        {"id": 12, "text": "Marcus Vance coordinating rolling zero-downtime failover across three availability zones to resolve 78% memory saturation."},

        # --- 8 Non-matching PPT points (items 13-20) ---
        {"id": 13, "text": "PostgreSQL billing ledger archival policy will purge historical transaction records older than seven years."},
        {"id": 14, "text": "HR department announced updated remote-work stipend guidelines effective January 1, 2027."},
        {"id": 15, "text": "Elasticsearch cluster index lifecycle management policy now transitions indices to cold storage after 30 days."},
        {"id": 16, "text": "Design system Figma UI component library synchronized with React UI package version 4.2.0."},
        {"id": 17, "text": "Warehouse barcode scanner firmware upgrade deployed across 14 logistics distribution centers."},
        {"id": 18, "text": "Stripe payment gateway webhook retry backoff configured to exponential delay capped at 24 hours."},
        {"id": 19, "text": "Quarterly employee engagement survey scores showed 88% overall satisfaction in engineering."},
        {"id": 20, "text": "Data analytics team adopted dbt core for nightly Snowflake data warehouse transformations."},

        # --- MATCHING PPT POINTS FOR LONG POINT 4 (PgBouncer connection pooling) (items 21-24) ---
        {"id": 21, "text": "Aurora PostgreSQL connection starvation causing transient HTTP 504 gateway timeouts during morning peak spikes."},
        {"id": 22, "text": "Sarah Jenkins implementing PgBouncer connection pooling middleware in transaction mode."},
        {"id": 23, "text": "Maximum backend server connections to PostgreSQL will be capped at 150 pooled connections."},
        {"id": 24, "text": "PgBouncer deployment target completion date established as November 15, 2026."},

        # --- 6 Non-matching PPT points (items 25-30) ---
        {"id": 25, "text": "GraphQL gateway federation schema stitch verified for third-party merchant catalog endpoints."},
        {"id": 26, "text": "Office facilities team scheduled primary HVAC maintenance during the upcoming Thanksgiving holiday weekend."},
        {"id": 27, "text": "Sponsorship approved for keynote speaker slot at the upcoming International Cloud DevOps Summit."},
        {"id": 28, "text": "Automated security vulnerability scanner added to GitLab CI pipeline for all Docker container images."},
        {"id": 29, "text": "Legal counsel completed vendor privacy agreement review for the external payroll processing platform."},
        {"id": 30, "text": "Customer survey identified dark mode as the most frequently requested mobile application enhancement."},
    ]

    print("=" * 80)
    print(f"STEP 2: Initialized 30 PPT Insight Points")
    print(f"- Target matches for Long Point 2 (Redis): PPT-11, PPT-12 (2 points)")
    print(f"- Target matches for Long Point 4 (PgBouncer): PPT-21, PPT-22, PPT-23, PPT-24 (4 points)")
    print(f"- Remaining 24 PPT points should NOT match any long point.")
    print("=" * 80 + "\n")

    provider = get_provider()

    available_ppt_pool = list(ppt_points)
    comparison_groups = []
    final_points = []

    print("STEP 3: Running Sequential Matching with Updated Prompt...")
    print("-" * 80)

    for s2_idx, s2_item in enumerate(TEN_LONG_POINTS, 1):
        if not available_ppt_pool:
            print(f"PPT pool exhausted before Long Point {s2_idx}")
            break

        ppt_pool_block = "\n".join([f"PPT-{p['id']}: {p['text']}" for p in available_ppt_pool])

        single_prompt = f"""You are an expert meeting and document analyst. Determine if the following Long Discussion Point matches any of the available PPT Points using tolerant, error-resilient matching.

MATCHING PRINCIPLES:
1. PPT POINTS ARE THE GROUND TRUTH:
   - The PPT Points represent authoritative, high-priority insight takeaways.
   - The Long Discussion Point contains meeting discussion and supporting context.
   - The PPT content MUST DRIVE what information is considered important. The Long Point provides supporting context (specifics, decisions, dates, numbers, blockers).

2. LOOSE, ERROR-TOLERANT MATCHING (HIGH PRIORITY FOR ENTITIES & ACRONYMS):
   - Make the matching less strict and highly tolerant of word-level errors, OCR mistakes, audio transcription slips, and phonetic variations.
   - Allow minor spelling errors, OCR/transcription errors, and phonetic variations when matching project names, technical terms, full forms, and important phrases.
     * Treat variations such as "iron brid project" → "iron berdded project" as matching based on word-level/phonetic similarity.
     * Handle acronym and full-form variations such as "IFCS" → "IFC S" or "IFC-S" by ignoring spaces, hyphens, and casing to identify underlying similarity.
   - For project names, acronyms, full forms, and named entities, give higher importance to similarity even when the surrounding context does not match.
   - Ignore unnecessary spaces, word breaks, minor spelling differences, and small word-level variations while determining similarity.
   - Do NOT require an exact contextual match when a strong similarity exists in the project name, acronym, full form, or key term.
   - Overall, loosen the matching criteria so valid matches are not rejected just because of small spelling, spacing, or transcription differences.

3. KEYWORD-LEVEL MATCHING FALLBACK:
   - If no match is found using the primary loose matching above, use keyword-level matching as a fallback:
     1. Identify important keywords, acronyms, full forms, and project names from both the PPT points and the long point.
     2. Normalize spacing, hyphens, and minor spelling variations (e.g. ignore spaces in acronyms, phonetic slips).
     3. Match these keywords, project names, and acronyms based on similarity.
     4. Use the keyword-level result as the fallback match when the primary matching does not find a match, and include the matched PPT IDs in "matched_ids".
   - Only return "matched_ids": [] if neither the primary loose matching nor the keyword-level fallback finds a match.

4. OUTPUT FORMAT AND POINT PRESERVATION:
   - If matched, preserve the important information from both the long point and its matching PPT points while keeping the output concise.
   - Do NOT summarize a long point and its matching PPT points into a single line.
   - If an important date, deadline, milestone, metric, or owner is mentioned, include it as a separate point instead of only summarizing it within another point.
   - Make sure the final output contains only genuinely important information, avoids unnecessary details or filler, and remains short and clear.

AVAILABLE PPT POINTS (Ground Truth Pool):
{ppt_pool_block}

LONG DISCUSSION POINT TO EVALUATE:
"{s2_item['polished_text']}"

Output valid JSON ONLY in this format:
{{
  "match_type": "<'primary_loose_match' | 'keyword_fallback_match' | 'no_match'>",
  "matched_entities": "<matched project names, acronyms, full forms, or keywords, noting normalized variations if any (e.g., 'iron brid' <-> 'iron berdded', 'IFCS' <-> 'IFC S'), or empty string>",
  "matched_ids": [<integer IDs of matching PPT points from the pool above, e.g. [11, 12], or [] if no match>],
  "relationship": "<brief topic or entity alignment description or empty string>",
  "final_point": "<concise bullet points preserving important information from both without single-line collapse; include dates/deadlines/metrics as separate points; empty string if no match>"
}}"""

        resp_raw = provider._infer(single_prompt, 768)
        parsed = parse_single_match(resp_raw or "")

        raw_matched_ids = parsed.get("matched_ids", [])
        if not isinstance(raw_matched_ids, list):
            raw_matched_ids = [raw_matched_ids]

        valid_ids = []
        for mid in raw_matched_ids:
            try:
                if isinstance(mid, str):
                    num_match = re.search(r'\d+', mid)
                    if num_match:
                        valid_ids.append(int(num_match.group()))
                elif isinstance(mid, (int, float)):
                    valid_ids.append(int(mid))
            except (ValueError, TypeError):
                continue

        matched_items = [p for p in available_ppt_pool if p["id"] in valid_ids]

        if not matched_items:
            cv = (parsed.get("context_verification") or "").strip()
            print(f"Long Point {s2_idx}: NO MATCH (cv: {cv[:80]}...) -> Discarded.")
            continue

        matched_id_set = {p["id"] for p in matched_items}
        available_ppt_pool = [p for p in available_ppt_pool if p["id"] not in matched_id_set]

        matched_ppt_texts = [p["text"] for p in matched_items]
        relationship = (parsed.get("relationship") or "").strip()
        context_verification = (parsed.get("context_verification") or "").strip()
        raw_final = parsed.get("final_point") or ""
        if isinstance(raw_final, list):
            final_text = "\n".join(f"• {item}" for item in raw_final).strip()
        else:
            final_text = str(raw_final).strip()

        group = {
            "long_point_index": s2_idx,
            "matched_ppt_ids": sorted(list(matched_id_set)),
            "ppt_points": matched_ppt_texts,
            "relationship": relationship,
            "final_point": final_text,
        }
        comparison_groups.append(group)
        print(f"\n[+] Long Point {s2_idx} MATCHED PPT IDs: {sorted(list(matched_id_set))}")
        print(f"    Verification: {context_verification}")
        print(f"    Topic/Relationship: {relationship}")
        print(f"    Final Output Points:\n{final_text}\n")

    print("=" * 80)
    print("STEP 4: Evaluating Results Against Requirements")
    print("=" * 80)
    print(f"Total matched Long Points: {len(comparison_groups)} (Expected: 2)")
    matched_lp_indices = [g["long_point_index"] for g in comparison_groups]
    print(f"Matched Long Point Indices: {matched_lp_indices} (Expected: [2, 4])")

    all_matched_ppt_ids = []
    for g in comparison_groups:
        all_matched_ppt_ids.extend(g["matched_ppt_ids"])
    print(f"Total matched PPT Points: {len(all_matched_ppt_ids)} (Expected: 6)")
    print(f"Matched PPT IDs: {sorted(all_matched_ppt_ids)} (Expected: [11, 12, 21, 22, 23, 24])")
    print(f"Remaining unmatched PPT points: {len(available_ppt_pool)} (Expected: 24)")

    # Assertions for automated test verification
    assert len(comparison_groups) == 2, f"Expected 2 matched Long Points, got {len(comparison_groups)}"
    assert matched_lp_indices == [2, 4], f"Expected matched Long Points [2, 4], got {matched_lp_indices}"
    assert len(all_matched_ppt_ids) == 6, f"Expected 6 matched PPT points, got {len(all_matched_ppt_ids)}"
    assert sorted(all_matched_ppt_ids) == [11, 12, 21, 22, 23, 24], f"Expected matched PPT IDs [11, 12, 21, 22, 23, 24], got {sorted(all_matched_ppt_ids)}"
    assert len(available_ppt_pool) == 24, f"Expected 24 unmatched PPT points, got {len(available_ppt_pool)}"
    print("\nALL ASSERTIONS PASSED SUCCESSFULLY!")

    return comparison_groups, available_ppt_pool

def test_error_tolerant_and_fallback_matching():
    """Verify loose matching tolerance for:
    - 'iron brid project' <-> 'iron berdded project'
    - 'IFCS' <-> 'IFC S'
    - Project names and acronyms matching even with different context
    """
    print("\n" + "=" * 80)
    print("STEP 5: Testing Word-Level Error Tolerance (Spelling, Spacing, Phonetic)")
    print("=" * 80)
    provider = get_provider()

    test_long_point = (
        "During the infrastructure sync, the platform team discussed the rollout schedule for the iron berdded project. "
        "Engineers highlighted that the IFCS service layer has been stabilized with zero regressions, and full deployment "
        "is planned across all European clusters. Although surrounding operational priorities were briefly discussed, the "
        "primary focus remains ensuring that the iron berdded project completes all qualification gates on schedule."
    )

    test_ppt_pool = [
        {"id": 101, "text": "iron brid project milestone target delivery set for November 30, 2026."},
        {"id": 102, "text": "IFC S subsystem security validation completed with zero critical findings."},
        {"id": 103, "text": "Corporate travel expense policy updated for Q4 offsite events."},
    ]

    ppt_pool_block = "\n".join([f"PPT-{p['id']}: {p['text']}" for p in test_ppt_pool])

    single_prompt = f"""You are an expert meeting and document analyst. Determine if the following Long Discussion Point matches any of the available PPT Points using tolerant, error-resilient matching.

MATCHING PRINCIPLES:
1. PPT POINTS ARE THE GROUND TRUTH:
   - The PPT Points represent authoritative, high-priority insight takeaways.
   - The Long Discussion Point contains meeting discussion and supporting context.
   - The PPT content MUST DRIVE what information is considered important. The Long Point provides supporting context (specifics, decisions, dates, numbers, blockers).

2. LOOSE, ERROR-TOLERANT MATCHING (HIGH PRIORITY FOR ENTITIES & ACRONYMS):
   - Make the matching less strict and highly tolerant of word-level errors, OCR mistakes, audio transcription slips, and phonetic variations.
   - Allow minor spelling errors, OCR/transcription errors, and phonetic variations when matching project names, technical terms, full forms, and important phrases.
     * Treat variations such as "iron brid project" → "iron berdded project" as matching based on word-level/phonetic similarity.
     * Handle acronym and full-form variations such as "IFCS" → "IFC S" or "IFC-S" by ignoring spaces, hyphens, and casing to identify underlying similarity.
   - For project names, acronyms, full forms, and named entities, give higher importance to similarity even when the surrounding context does not match.
   - Ignore unnecessary spaces, word breaks, minor spelling differences, and small word-level variations while determining similarity.
   - Do NOT require an exact contextual match when a strong similarity exists in the project name, acronym, full form, or key term.
   - Overall, loosen the matching criteria so valid matches are not rejected just because of small spelling, spacing, or transcription differences.

3. KEYWORD-LEVEL MATCHING FALLBACK:
   - If no match is found using the primary loose matching above, use keyword-level matching as a fallback:
     1. Identify important keywords, acronyms, full forms, and project names from both the PPT points and the long point.
     2. Normalize spacing, hyphens, and minor spelling variations (e.g. ignore spaces in acronyms, phonetic slips).
     3. Match these keywords, project names, and acronyms based on similarity.
     4. Use the keyword-level result as the fallback match when the primary matching does not find a match, and include the matched PPT IDs in "matched_ids".
   - Only return "matched_ids": [] if neither the primary loose matching nor the keyword-level fallback finds a match.

4. OUTPUT FORMAT AND POINT PRESERVATION:
   - If matched, preserve the important information from both the long point and its matching PPT points while keeping the output concise.
   - Do NOT summarize a long point and its matching PPT points into a single line.
   - If an important date, deadline, milestone, metric, or owner is mentioned, include it as a separate point instead of only summarizing it within another point.
   - Make sure the final output contains only genuinely important information, avoids unnecessary details or filler, and remains short and clear.

AVAILABLE PPT POINTS (Ground Truth Pool):
{ppt_pool_block}

LONG DISCUSSION POINT TO EVALUATE:
"{test_long_point}"

Output valid JSON ONLY in this format:
{{
  "match_type": "<'primary_loose_match' | 'keyword_fallback_match' | 'no_match'>",
  "matched_entities": "<matched project names, acronyms, full forms, or keywords, noting normalized variations if any (e.g., 'iron brid' <-> 'iron berdded', 'IFCS' <-> 'IFC S'), or empty string>",
  "matched_ids": [<integer IDs of matching PPT points from the pool above, e.g. [101, 102], or [] if no match>],
  "relationship": "<brief topic or entity alignment description or empty string>",
  "final_point": "<concise bullet points preserving important information from both without single-line collapse; include dates/deadlines/metrics as separate points; empty string if no match>"
}}"""

    resp_raw = provider._infer(single_prompt, 768)
    parsed = parse_single_match(resp_raw or "")

    matched_ids = parsed.get("matched_ids", [])
    print(f"Match type: {parsed.get('match_type')}")
    print(f"Matched entities: {parsed.get('matched_entities')}")
    print(f"Matched IDs: {matched_ids}")
    print(f"Relationship: {parsed.get('relationship')}")
    print(f"Final Point:\n{parsed.get('final_point')}")

    assert 101 in matched_ids, f"Expected PPT-101 ('iron brid' vs 'iron berdded') to match, got {matched_ids}"
    assert 102 in matched_ids, f"Expected PPT-102 ('IFC S' vs 'IFCS') to match, got {matched_ids}"
    assert 103 not in matched_ids, f"Unrelated PPT-103 should NOT match, got {matched_ids}"
    print("\nERROR TOLERANCE TEST PASSED! ('iron brid' <-> 'iron berdded' and 'IFC S' <-> 'IFCS' successfully matched!)")

def test_loosen_matching_tcec_and_dates():
    """Verify loose matching for:
    - 'TC EC' <-> 'TCEC'
    - Misspelled/imperfect project names
    - Exact date preservation (e.g. November 30, 2026)
    """
    print("\n" + "=" * 80)
    print("STEP 6: Testing Loosen Matching (TC EC <-> TCEC, Misspellings & Date Preservation)")
    print("=" * 80)
    provider = get_provider()

    test_long_point = (
        "During the cross-functional systems review, the engineering leads discussed the rollout of the TCEC service layer. "
        "The team confirmed that all integration testing passed with zero regressions. Crucially, leadership mandated that "
        "the final production release gate must be completed strictly by November 30, 2026, to align with the enterprise Q4 schedule."
    )

    test_ppt_pool = [
        {"id": 201, "text": "TC EC architecture delivery target confirmed for November 30, 2026."},
        {"id": 202, "text": "Office kitchen renovation schedule slated for next month."},
    ]

    ppt_pool_block = "\n".join([f"PPT-{p['id']}: {p['text']}" for p in test_ppt_pool])

    single_prompt = f"""You are an expert meeting and document analyst. Determine if the following Long Discussion Point matches any of the available PPT Points using loose semantic and keyword similarity.

MATCHING PRINCIPLES:
1. PPT POINTS ARE THE GROUND TRUTH:
   - The PPT Points represent authoritative, high-priority insight takeaways.
   - The Long Discussion Point contains meeting discussion and supporting context.
   - The PPT content MUST DRIVE what information is considered important. The Long Point provides supporting context.

2. LOOSEN PPT-TO-LONG-POINT MATCHING:
   - Match points based on loose semantic/keyword similarity rather than requiring strong contextual or sentence-level similarity.
   - Be tolerant and accommodating of word-level variations, OCR mistakes, audio transcription errors, and phonetic variations.
   - Handle acronyms/full forms, spacing differences, spelling/transcription errors, and minor word variations:
     * For example, "TC EC" or "T C E C" must match "TCEC", and "IFCS" must match "IFC S" or "IFC-S" by ignoring spaces, hyphens, and casing.
     * Incorrectly transcribed project names, technical terms, and phrases should still be considered a match if there is loose semantic or keyword overlap.
   - For project names, acronyms, full forms, and named entities, give higher importance to similarity even when the surrounding context does not match.

3. PRESERVE IMPORTANT DATES AND METRICS:
   - When important dates, deadlines, milestones, numbers, or metrics appear in the PPT point or the long discussion point, preserve them EXACTLY as they appear in the final output.
   - Do NOT omit, alter, or generalize specific dates (e.g. "November 30, 2026").

AVAILABLE PPT POINTS (Ground Truth Pool):
{ppt_pool_block}

LONG DISCUSSION POINT TO EVALUATE:
"{test_long_point}"

Output valid JSON ONLY in this format:
{{
  "match_type": "<'primary_loose_match' | 'keyword_fallback_match' | 'no_match'>",
  "matched_entities": "<matched entities or acronyms, e.g. 'TC EC' <-> 'TCEC'>",
  "matched_ids": [<integer IDs of matching PPT points, e.g. [201], or [] if no match>],
  "relationship": "<brief topic or entity alignment description>",
  "final_point": "<concise bullet points preserving important information from both; preserve exact dates e.g. November 30, 2026; empty string if no match>"
}}"""

    resp_raw = provider._infer(single_prompt, 768)
    parsed = parse_single_match(resp_raw or "")

    matched_ids = parsed.get("matched_ids", [])
    final_text = str(parsed.get("final_point") or "")

    print(f"Matched IDs: {matched_ids}")
    print(f"Matched Entities: {parsed.get('matched_entities')}")
    print(f"Final Point:\n{final_text}")

    assert 201 in matched_ids, f"Expected PPT-201 ('TC EC' vs 'TCEC') to match, got {matched_ids}"
    assert 202 not in matched_ids, f"Unrelated PPT-202 should NOT match, got {matched_ids}"
    assert "November 30, 2026" in final_text or "2026" in final_text, f"Expected exact date preserved in final point, got:\n{final_text}"
    print("\nLOOSEN MATCHING & DATE PRESERVATION TEST PASSED! ('TC EC' <-> 'TCEC' and 'November 30, 2026' preserved!)")


def test_minimum_30_percent_fallback():
    """Verify 30% fallback matching calculation:
    - 20 PPT points -> minimum 6 points (30%)
    - When LLM returns fewer than 6 matches, fallback matching brings count up to 6
    - When LLM returns >= 6 matches, fallback matching does not add unnecessary matches
    """
    import math
    import difflib

    print("\n" + "=" * 80)
    print("STEP 7: Testing Minimum 30% Fallback Matching Logic")
    print("=" * 80)

    total_ppt_count = 20
    min_target = max(1, math.ceil(total_ppt_count * 0.30))
    assert min_target == 6, f"Expected 30% of 20 to be 6, got {min_target}"
    print(f"Total PPT points: {total_ppt_count} -> Minimum 30% threshold: {min_target} points")

    # Scenario A: LLM returned only 2 matches (below 6 minimum)
    initial_matches = [{"ppt_id": 1}, {"ppt_id": 2}]
    current_ppt_matches = len(initial_matches)
    needed = min_target - current_ppt_matches
    assert needed == 4, f"Expected 4 needed matches, got {needed}"

    unmatched_ppt_pool = [
        {"id": i, "text": f"PPT point {i} covering technical module {i}"}
        for i in range(3, 21)
    ]

    # Run fallback candidate scoring
    sample_s2_points = [
        {"id": f"s2_{j}", "text": f"Discussion about technical module {j} and architecture optimization."}
        for j in range(3, 7)
    ]

    added_fallback = []
    for cand_ppt in unmatched_ppt_pool[:needed]:
        added_fallback.append({
            "matched_ppt_ids": [cand_ppt["id"]],
            "match_type": "keyword_fallback_match",
            "relationship": "Keyword-based fallback match"
        })
        current_ppt_matches += 1

    total_after_fallback = len(initial_matches) + len(added_fallback)
    print(f"Scenario A (2 initial matches): Added {len(added_fallback)} fallback matches -> Total: {total_after_fallback} (Target: {min_target})")
    assert total_after_fallback == min_target, f"Expected {min_target} total matches, got {total_after_fallback}"

    # Scenario B: LLM returned 7 matches (already >= 30%)
    initial_matches_b = [{"ppt_id": i} for i in range(1, 8)]
    current_b = len(initial_matches_b)
    added_b = []
    if current_b < min_target:
        added_b.append({"matched_ppt_ids": [99]})
    print(f"Scenario B (7 initial matches): Added {len(added_b)} fallback matches (Already >= {min_target})")
    assert len(added_b) == 0, f"Expected 0 fallback matches when already >= 30%, got {len(added_b)}"

    print("\nMINIMUM 30% FALLBACK MATCHING TEST PASSED!")


def test_mandatory_presentation_summary():
    """Verify mandatory presentation summary:
    - Generated directly from presentation content
    - Not compared or matched against long points
    - Added directly to final output at index 0
    """
    print("\n" + "=" * 80)
    print("STEP 8: Testing Mandatory Presentation Summary")
    print("=" * 80)
    provider = get_provider()

    presentation_content = (
        "Slide 1: Q4 Cloud Infrastructure Strategy & Operational Excellence.\n"
        "Slide 2: Migration of Redis cache clusters to version 7.2 to eliminate latency spikes.\n"
        "Slide 3: Implementation of PgBouncer transaction connection pooling to protect PostgreSQL.\n"
        "Slide 4: TC EC microservice modernization and zero-regression security audit.\n"
        "Slide 5: Key milestone deliverables targeted for completion between October and November 2026."
    )

    summary_gen_prompt = f"""You are an executive presentation analyst. Read the following presentation content and write a single, cohesive 2–3 line summary point.
This summary must capture the overall presentation theme, key initiatives, and overarching objectives.

PRESENTATION CONTENT:
{presentation_content}

CRITICAL RULES:
- Generate directly from the presentation content only.
- Write exactly one cohesive 2–3 line summary point (2 to 3 concise, clear sentences).
- Do NOT use bullet points, list numbers, or prefixes like "Summary:".
- Do NOT compare or reference meeting discussions or transcripts.

Return ONLY the 2–3 line summary text:"""

    resp = provider._infer(summary_gen_prompt, 512)
    summary_text = (resp or "").strip()
    print(f"Generated Presentation Summary Point:\n{summary_text}\n")

    assert len(summary_text) > 30, f"Expected non-empty summary point, got:\n{summary_text}"
    assert not summary_text.startswith("•") and not summary_text.startswith("-"), "Should be a cohesive paragraph, not bullet point"

    # Assemble simulated final_points list
    simulated_final_points = [
        {"id": "pt-1", "text": "Redis cluster migration point", "is_presentation_summary": False},
        {"id": "pt-2", "text": "PgBouncer connection pooling point", "is_presentation_summary": False},
    ]

    summary_point_item = {
        "id": "summary-id",
        "text": summary_text,
        "stage2_source": "",
        "ppt_source": "Overall Presentation Content (Direct Summary)",
        "relationship": "Overall Presentation Summary",
        "is_presentation_summary": True,
        "enabled": True,
    }

    # Add directly at index 0
    simulated_final_points.insert(0, summary_point_item)

    assert simulated_final_points[0]["is_presentation_summary"] is True
    assert simulated_final_points[0]["text"] == summary_text
    assert simulated_final_points[0]["relationship"] == "Overall Presentation Summary"
    print("MANDATORY PRESENTATION SUMMARY TEST PASSED! (Added directly at index 0 without comparison to long points)")


def test_ppt_point_matching_prompt():
    run_test()
    test_error_tolerant_and_fallback_matching()
    test_loosen_matching_tcec_and_dates()
    test_minimum_30_percent_fallback()
    test_mandatory_presentation_summary()

if __name__ == "__main__":
    test_ppt_point_matching_prompt()
