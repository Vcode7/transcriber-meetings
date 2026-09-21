"""
Script to update RAW_MOM_EXTRACTION_PROMPT and RAW_MOM_REPAIR_PROMPT in ai_provider.py
"""
import sys

filepath = "services/ai_provider.py"
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Normalize to unix for matching
content_unix = content.replace('\r\n', '\n').replace('\r', '\n')

# ─── 1. Replace RAW_MOM_EXTRACTION_PROMPT ────────────────────────────────────

OLD_START = 'RAW_MOM_EXTRACTION_PROMPT = """\\\n'
OLD_END = '\nJSON:"""\n'

# Find exact boundaries
start_idx = content_unix.find(OLD_START)
if start_idx == -1:
    print("ERROR: Could not find RAW_MOM_EXTRACTION_PROMPT start marker")
    sys.exit(1)

# Find the closing JSON:" + triple-quote that ends THIS prompt (not the repair prompt)
end_search_start = start_idx + len(OLD_START)
end_idx = content_unix.find('\nJSON:"""', end_search_start)
if end_idx == -1:
    print("ERROR: Could not find end of RAW_MOM_EXTRACTION_PROMPT")
    sys.exit(1)
end_idx += len('\nJSON:"""')  # include the closing triple-quote

old_prompt_block = content_unix[start_idx:end_idx]
print(f"Found old prompt block ({len(old_prompt_block)} chars)")

NEW_PROMPT_BLOCK = '''RAW_MOM_EXTRACTION_PROMPT = """\\\nYou are a precise information extractor for meeting records. Your ONLY job is structured extraction \u2014 not summarization.\n\nYou will receive evidence retrieved from the transcript, presentation slides, and organizational documents for ONE agenda topic.\nThe evidence is divided into clearly labelled sections:\n  - TRANSCRIPT sections: actual spoken discussion from the meeting recording\n  - MEETING CONTEXT / AGENDA CONTEXT sections: uploaded documents (slides, specs, plans)\n  - GLOBAL CONTEXT section: organizational knowledge documents\n  - AGENDA-SPECIFIC CONTEXT: additional per-agenda reference material\n\nThe transcript text includes timeline headers in the format: [HH:MM:SS - HH:MM:SS] Speaker Name\n\nBefore extracting information, first understand the meaning and purpose of the agenda topic.\n\nThe retrieved evidence may contain unrelated information because semantic retrieval is not perfect. Do NOT assume that every retrieved sentence belongs to the current agenda.\n\nYour task is to extract ALL factual information that is directly relevant to the current agenda topic.\n\n---\nCORE EXTRACTION REQUIREMENTS:\n\n1. EXTRACT ALL MEANINGFUL POINTS \u2014 no artificial limit. Capture every:\n   - Decision made\n   - Discussion point or topic covered\n   - Question raised and answer provided\n   - Concern or risk identified\n   - Resolution or agreement reached\n   - Follow-up item or pending action\n   - Action item with owner/deadline\n   - Milestone or deadline reference\n   - Dependency noted\n   Merge duplicate ideas into professional concise points. Do NOT truncate or omit content to reduce count.\n\n2. INFORMATION-DENSE ENTRIES: Each discussion entry must be information-dense. Combine all related facts into comprehensive points instead of splitting them into multiple small points.\n\n3. DO NOT OMIT IMPORTANT FACTS: Preserve every important factual detail including:\n   - Dates, deadlines, effective dates, approval dates, extensions, and timelines.\n   - Decisions, amendments, motions, votes, and outcomes.\n   - Financial values, quantities, locations, ordinance numbers, property details, and technical information.\n   - Questions raised, responses provided, concerns discussed, and follow-up requests.\n   - Action items with owner, deadline, and status when available.\n\n4. SOURCE ATTRIBUTION \u2014 every entry MUST include:\n   - "source_type": one of "Transcript", "Meeting Context", "Agenda Context", "Global Context"\n   - "timeline": {{"start": <seconds as float>, "end": <seconds as float>}} if from transcript; null otherwise\n   - "source_reference": the timestamp range "HH:MM:SS - HH:MM:SS" if from transcript; the document filename if from context; "Global Context" if from global knowledge\n\n5. CONTEXT-ONLY INFORMATION: If information comes ONLY from Meeting Context, Agenda Context, or Global Context and was NOT explicitly discussed in the transcript, set "type" to "reference" and include a note like "(For Information)" at the start of the "point" field. Do NOT present it as if it was discussed in the meeting.\n\n6. NO SUMMARIES: Do not generate introductions, conclusions, or summaries. Perform factual extraction only.\n\n---\nReturn ONLY a valid JSON object matching this exact schema:\n\n{{\n  "agenda_topic": "{agenda_topic}",\n  "agenda_speaker": "{agenda_speaker}",\n  "discussion": [\n    {{\n      "type": "decision|action|discussion|clarification|risk|milestone|dependency|reference",\n      "speaker": "Speaker name or null if unknown",\n      "point": "Comprehensive information-dense fact or statement preserving all details (dates, numbers, decisions, questions/answers, etc.)",\n      "source_type": "Transcript|Meeting Context|Agenda Context|Global Context",\n      "timeline": {{"start": 123.4, "end": 145.6}},\n      "source_reference": "HH:MM:SS - HH:MM:SS or filename.pdf or Global Context",\n      "dates": [\n        {{"value": "date/time value", "purpose": "what this date is for"}}\n      ],\n      "action": {{\n        "owner": "person responsible or null",\n        "description": "what needs to be done or null",\n        "deadline": "deadline date or null",\n        "status": "open|in_progress|completed|null"\n      }}\n    }}\n  ]\n}}\n\nRules for discussion entries:\n- Extract ALL meaningful points \u2014 there is NO maximum limit. Completeness is the goal.\n- Group related facts together so each entry contains a complete, cohesive subset of discussion details.\n- "dates" array: include ONLY if the entry involves a specific date/deadline/milestone.\n- "action": fill ONLY if the entry is an action item; set all fields to null otherwise.\n- "type" must be one of: decision, action, discussion, clarification, risk, milestone, dependency, reference.\n- "source_type" is REQUIRED on every entry. Use "Transcript" only for entries from the TRANSCRIPT section.\n- "timeline" is REQUIRED for Transcript entries. Parse the [HH:MM:SS - HH:MM:SS] header from the evidence text. Set to null for non-transcript entries.\n- "source_reference" is REQUIRED. For Transcript: use "HH:MM:SS - HH:MM:SS". For context docs: use the filename. For global: use "Global Context".\n- Preserve all technical terms, acronyms, numbers, system names, and project names exactly.\n\nAGENDA TOPIC: {agenda_topic}\nSPEAKER: {agenda_speaker}\n\nRETRIEVED EVIDENCE:\n{evidence}\n\nJSON:"""'''

new_content_unix = content_unix[:start_idx] + NEW_PROMPT_BLOCK + content_unix[end_idx:]
print(f"Replacement done. New content length: {len(new_content_unix)}")

# ─── 2. Update RAW_MOM_REPAIR_PROMPT schema ──────────────────────────────────
# Add source_type, timeline, source_reference fields to repair prompt schema

OLD_REPAIR_SCHEMA = '''      "point": "Text statement",
      "dates": [
        {"value": "date/time", "purpose": "purpose"}
      ],
      "action": {
        "owner": "owner name or null",
        "description": "description or null",
        "deadline": "deadline or null",
        "status": "open|in_progress|completed|null"
      }'''

NEW_REPAIR_SCHEMA = '''      "point": "Text statement",
      "source_type": "Transcript|Meeting Context|Agenda Context|Global Context",
      "timeline": {"start": 123.4, "end": 145.6},
      "source_reference": "HH:MM:SS - HH:MM:SS or filename or Global Context",
      "dates": [
        {"value": "date/time", "purpose": "purpose"}
      ],
      "action": {
        "owner": "owner name or null",
        "description": "description or null",
        "deadline": "deadline or null",
        "status": "open|in_progress|completed|null"
      }'''

if OLD_REPAIR_SCHEMA in new_content_unix:
    new_content_unix = new_content_unix.replace(OLD_REPAIR_SCHEMA, NEW_REPAIR_SCHEMA, 1)
    print("SUCCESS: RAW_MOM_REPAIR_PROMPT schema updated")
else:
    print("WARNING: Could not find old repair prompt schema - skipping")

# ─── 3. Write back with CRLF ─────────────────────────────────────────────────
with open(filepath, 'w', encoding='utf-8', newline='') as f:
    # Restore CRLF line endings
    f.write(new_content_unix.replace('\n', '\r\n'))

print("File written successfully.")

# ─── 4. Verify ───────────────────────────────────────────────────────────────
with open(filepath, 'r', encoding='utf-8') as f:
    check = f.read()

checks = [
    ('MAXIMUM OF 5 DISCUSSION ENTRIES', False, '5-point limit removed'),
    ('source_type', True, 'source_type field present'),
    ('source_reference', True, 'source_reference field present'),
    ('timeline', True, 'timeline field present'),
    ('EXTRACT ALL MEANINGFUL POINTS', True, 'unlimited extraction instruction present'),
    ('For Information', True, 'context-only guidance present'),
    ('reference', True, '"reference" type added'),
]
for text, should_exist, label in checks:
    found = text in check
    if found == should_exist:
        print(f'VERIFIED: {label}')
    else:
        expected = 'present' if should_exist else 'absent'
        actual = 'present' if found else 'absent'
        print(f'FAIL: {label} - expected {expected} but is {actual}')
