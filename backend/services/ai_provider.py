# -*- coding: utf-8 -*-
"""
AI Provider - 100% offline Qwen3 4B Instruct (4-bit quantized).

Architecture:
  - Single provider: QwenProvider
  - Model is loaded ONCE at startup and kept resident in memory
  - GPU (CUDA) when available, CPU fallback otherwise
  - Hierarchical chunked summarization for long meetings
  - Dedicated prompt templates per task (no shared generic prompt)

Removed:
  - LocalSummarizationProvider (DistilBART)
  - GroqProvider (cloud Groq API)
  - AIProviderFactory with cloud routing
  - LlamaProvider (replaced by QwenProvider)
"""
from __future__ import annotations

import json
import logging
import re
import textwrap
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# Prompt Templates - one per task, purpose-built for Qwen3
# ══════════════════════════════════════════════════════════════


EXECUTIVE_SUMMARY_PROMPT = """\
You are an expert enterprise meeting analyst. Analyze the following meeting transcript and write a professional Executive Summary.

Answer each of the following questions concisely and professionally:

1. What was the main agenda of the meeting?
2. What problems or topics were discussed?
3. What work was completed or achieved?
4. What decisions were taken?
5. What conclusions were reached?
6. What are the next expected steps?

Format your response EXACTLY as follows (use these section headers):

MEETING PURPOSE:
[2-5 sentences answering question 1]

MAIN DISCUSSION POINTS:
- [point derived from question 2]
- [point derived from question 2]
- up to 5 or more (based on the transcript)

OUTCOMES:
- [outcome from questions 3 and 4]
- [outcome from questions 3 and 4]
- up to 5 or more (based on the transcript)

NEXT STEPS:
- [step from questions 5 and 6]
- [step from questions 5 and 6]
- up to 5 or more (based on the transcript)

Rules:
- Be concise and professional. Suitable for senior management.
- Do NOT repeat verbatim sentences from the transcript.
- Do NOT attribute statements to specific speakers.
- Do NOT add information not present in the transcript.
- Preserve all technical terms, acronyms, and project names exactly as spoken.

TRANSCRIPT:
{transcript}

Now write the Executive Summary:"""


SHORT_SUMMARY_PROMPT = """\
You are an expert meeting analyst. Summarize the following meeting transcript in exactly 120 words or fewer.

Write as a single professional paragraph. Do NOT use bullet points. Focus on what was discussed and decided.
Do NOT mention who said what. Do NOT repeat transcript sentences. Preserve all technical terms exactly.

TRANSCRIPT:
{transcript}

120-word summary:"""


DETAILED_SUMMARY_PROMPT = """\
You are an expert meeting analyst writing a comprehensive meeting report.

Analyze the transcript and write a detailed summary with SEPARATE SECTIONS for each major discussion topic.

For each topic section:
- Give the section a clear title (use ## Title format)
- Explain the topic and context
- Summarize the discussion details and technical observations
- State conclusions or outcomes reached for that topic

After all topic sections, add:

## Key Decisions
- [decision 1]
- [decision 2]

## Action Items
- [action item with owner if mentioned]

Rules:
- Do NOT attribute statements to specific speakers.
- Preserve all technical terms, acronyms, aircraft names, project names exactly.
- Be thorough. Reading this summary alone should give full meeting understanding.
- Do NOT repeat verbatim transcript sentences.

TRANSCRIPT:
{transcript}

Comprehensive meeting report:"""

MOM_PROMPT = """\
You are an experienced Executive Assistant responsible for producing professional, comprehensive, and factual Minutes of Meeting (MoM).

Your task is to analyze the complete meeting transcript and generate a structured Minutes of Meeting that captures ALL meaningful discussion while remaining concise and free of repetition.

Return ONLY a valid JSON object.
Do NOT wrap the response in markdown.
Do NOT include ```json.
Do NOT include explanations or any text outside the JSON.

The output MUST strictly follow this schema:

{{
  "title": "Professional meeting title",
  "introduction": "A professional overview describing the meeting purpose, agenda, participants' objectives, and overall context.",
  "points_discussed": [
    "Each item describes one meaningful discussion topic."
  ],
  "action_items": [
    {{
      "task": "Task description",
      "owner": "Speaker name or Unassigned",
      "deadline": "Deadline if mentioned, otherwise ASAP or null"
    }}
  ],
  "general_action_items": [
    "Action item without a specific owner"
  ],
  "conclusion": "Professional conclusion summarizing decisions, agreements, pending items, risks, and next steps.",
  "actual_start_time": "HH:MM if explicitly mentioned, otherwise null",
  "next_meeting_date": "Date/time if explicitly mentioned, otherwise null"
}}

Generation Guidelines

TITLE
- Generate a concise, professional meeting title.
- Reflect the primary objective of the meeting.
{agenda_section}

INTRODUCTION
- Write a well-structured executive introduction.
- Clearly explain:
  - meeting purpose
  - overall agenda
  - major subjects discussed
  - project/context
- Do not simply repeat the transcript.

POINTS DISCUSSED
- Capture EVERY important discussion topic.
- There is NO maximum or minimum number of points.
- Create one point for every meaningful topic discussed.
- Merge duplicate or repetitive discussions into a single comprehensive point.
- Ignore greetings, filler conversations, acknowledgements, interruptions, and casual chit-chat.
- Preserve important technical details, architecture discussions, implementation decisions, requirements, risks, blockers, design changes, timelines, dependencies, and stakeholder decisions.
- Each point should be:
  - self-contained
  - factually accurate
  - written as a complete sentence
  - professional
- Maintain the chronological flow whenever practical.

ACTION ITEMS
- Extract EVERY actionable task discussed.
- Include:
  - implementation tasks
  - follow-up work
  - reviews
  - testing
  - documentation
  - approvals
  - bug fixes
  - investigations
  - deployments
- Assign tasks ONLY when the assignee is an explicit person's name, organization name, or a specific role/title.
- Do NOT use vague references such as "my team," "our team," "you," "they," "everyone," "we," "someone," or similar pronouns/generic groups as assignees.
- If the transcript does not clearly identify a valid assignee, return "Unassigned".
- Preserve deadlines exactly if mentioned.
- Never invent deadlines.

GENERAL ACTION ITEMS
- Include only actionable tasks that have no identifiable owner.
- Do not duplicate items already listed in action_items.

CONCLUSION
- Write a professional executive summary.
- Include:
  - key outcomes
  - decisions made
  - agreements reached
  - unresolved issues
  - risks
  - future work
  - next steps
- Do not introduce information not present in the transcript.

EXTRACTION RULES
- Never hallucinate facts.
- Never invent attendees, owners, dates, deadlines, or decisions.
- Preserve all technical terminology exactly:
  - API names
  - model names
  - class names
  - filenames
  - project names
  - architecture
  - frameworks
  - libraries
  - database names
  - commands
  - code identifiers
  - acronyms
- If conflicting information exists, report only what was actually discussed.
- Ignore repeated statements and produce a clean consolidated MoM.
- If a field has no relevant information:
  - arrays → []
  - string/date fields → null

The final output must be valid JSON and nothing else.
{reference_section}
TRANSCRIPT:
{transcript}

JSON:
"""
MOM_MERGE_PROMPT = """\
You are an experienced Executive Assistant responsible for producing professional, comprehensive, and consolidated Minutes of Meeting (MoM).

You will be given multiple partial Minutes of Meeting (MoM) generated from different sections of the same meeting. Your task is to merge and consolidate these partial MoMs into a single, unified, and cohesive final MoM.

Return ONLY a valid JSON object matching the exact schema.
Do NOT wrap the response in markdown.
Do NOT include ```json.
Do NOT include explanations or any text outside the JSON.

The output MUST strictly follow this schema:
{{
  "title": "Consolidated professional meeting title",
  "introduction": "A unified professional overview describing the meeting purpose, agenda, participants' objectives, and overall context.",
  "points_discussed": [
    "Each item describes one meaningful discussion topic."
  ],
  "action_items": [
    {{
      "task": "Task description",
      "owner": "Speaker name or Unassigned",
      "deadline": "Deadline if mentioned, otherwise ASAP or null"
    }}
  ],
  "general_action_items": [
    "Action item without a specific owner"
  ],
  "conclusion": "Unified professional conclusion summarizing decisions, agreements, pending items, risks, and next steps."
}}

Guidelines for Consolidation:
1. TITLE: Choose the most professional and representative title across the partial MoMs.
2. INTRODUCTION: Write a unified introduction. Combine the purpose and objectives from the partial introductions into a single cohesive paragraph.
3. POINTS DISCUSSED:
   - Merge all discussion points.
   - Consolidate repetitive or overlapping discussions across different sections into a single comprehensive point.
   - Preserve all unique technical details, architecture descriptions, requirements, risks, and decisions.
   - Do not lose unique details. Enforce clean, professional sentences.
4. ACTION ITEMS & GENERAL ACTION ITEMS:
   - Combine all action items and general action items.
   - Remove duplicate action items.
   - Keep owners and deadlines intact.
5. CONCLUSION:
   - Synthesize a final unified conclusion summarizing the overall decisions, agreements, and next steps for the entire meeting.

PARTIAL MOMs TO CONSOLIDATE:
{partial_moms_json}

Now generate the consolidated final MoM JSON:"""


AGENDA_COMPRESS_PROMPT = """\
You are an expert meeting agenda parser.

Your task is to extract ONLY the actual meeting agenda items from the document.

IMPORTANT:
Do NOT immediately start extracting agenda items.

Instead, complete the task in the following stages.

STEP 1 - IDENTIFY THE AGENDA SECTION
First identify where the actual meeting agenda begins and ends.

Agenda sections commonly appear after introductory information such as:

- Organization name
- Meeting title
- Meeting date
- Meeting time
- Meeting location
- Committee members
- Staff information
- Contact details
- Public comment instructions
- Zoom/Webinar information
- Accessibility information
- General meeting instructions
- Headers
- Footers
- Page numbers

Ignore all of the above.

The agenda section usually contains discussion topics that will actually be covered during the meeting.

Examples include:

- Call to Order
- Roll Call
- Approval of Minutes
- Public Comment
- Consent Agenda
- Discussion Items
- Briefings
- Presentations
- Reports
- Proposed Ordinances
- Proposed Motions
- Project Updates
- Technical Reviews
- Design Reviews
- Other Business
- Adjournment

STEP 2 - DETECT THE AGENDA STRUCTURE
Agenda items may appear in many different formats.

Examples:

1.
2.
3.

A.
B.
C.

•
•

Roman numerals

Tables

Indented lists

Nested sections

Paragraphs

Do NOT assume a specific layout.

Understand the document structure before extracting.

If the agenda is presented in a table, identify the columns first.

Typical columns include:

- Agenda ID
- Item Number
- Agenda Title
- Topic
- Subject
- Presenter
- Presented By
- Speaker
- Lead
- Owner
- Responsible Person
- Department
- Duration

If an agenda item contains a title followed by one or more descriptive
paragraphs, they belong to the SAME agenda item.

The title and its description should be merged into a single topic.

Do not split a single agenda item into multiple topics.

STEP 3 - IDENTIFY THE PRESENTER
For every agenda item determine whether a presenter is explicitly associated with it.

Possible labels include (but are not limited to):

- Presenter
- Presented By
- Speaker
- Agenda Presenter
- Lead
- Owner
- Responsible
- Presented by
- Council Staff
- Staff
- Department Lead
- Project Lead
- Facilitator
- Chair
- Vice Chair

The presenter's name may appear:

- in the same row
- below the agenda item
- above the agenda item
- in an adjacent table column
- immediately after the topic

When extracting an agenda item, search the nearby text before and after
the topic.

If a presenter, council staff, presenter, presented by, project lead,
staff lead, briefing by or similar information is found immediately
before or after the agenda item, associate that person (it would be mention near by) with the agenda item.

If none of them is explicitly associated with the agenda item, return:

null

Do NOT guess.

Do NOT use committee members listed at the beginning of the document unless they are explicitly associated with that agenda item.


STEP 4 - EXTRACT THE AGENDA 

After identifying the agenda section:

Extract every agenda item.

Rules:

- Copy the agenda topic exactly as written whenever possible.
- Do NOT summarize.
- Do NOT rewrite.
- Do NOT shorten.
- If the title spans multiple lines, combine them into one topic.
- Do NOT include administrative information.
- Do NOT include contact information.
- Do NOT include webinar instructions.
- Do NOT include meeting metadata.
- Do NOT include organization information.
- Do NOT include committee member lists.
- Do NOT include page headers or page footers.

If no agenda section exists, return:

[]

also extract

details
- Include ALL remaining information that belongs to this agenda item but is not part of the topic.
- Examples include:
    - agenda description
    - ordinance description
    - motion description
    - proposal description
    - objective
    - supporting information
    - sponsor
    - department
    - project information
    - references
    - document numbers
    - any explanatory text directly associated with this agenda item
- Preserve the wording as closely as possible.
- Do NOT summarize.
- Do NOT remove technical details.
- Do NOT include information belonging to another agenda item.
- If there are multiple paragraphs describing the same agenda item, combine them into a single details string.

If an agenda item consists of a title followed by descriptive paragraphs, they represent ONE agenda item.


For every agenda item extract:

{{
    "topic": "...",
    "speaker": "..."
    "details": "..."
}}

Rules:

1. Copy the agenda topic exactly as written whenever possible.
2. Do not summarize.
3. Do not merge multiple agenda items.
4. If an agenda item spans multiple lines, combine them into a single topic.
5. If a presenter, owner, lead, chair, council staff, presenter, speaker or responsible person is explicitly associated with that agenda item, extract their name.
6. If no presenter is explicitly associated with the agenda item, return null.
7. Never invent a speaker.
8. Return ONLY valid JSON.
9. If no agenda items are found, return an empty JSON array [].


AGENDA DOCUMENT:
{text}

JSON:"""


AGENDA_COMPRESS_WITH_CONTEXT_PROMPT = """\
You are an expert meeting agenda parser.

Your task is to extract ONLY the actual meeting agenda items from the document.

=== SECTION 1: AGENDA DOCUMENT (PRIMARY SOURCE) ===

This is the ONLY source you may use to:
- Generate agenda titles
- Determine agenda items
- Establish agenda hierarchy and ordering
- Define the number of agenda items

IMPORTANT:
Do NOT immediately start extracting agenda items.

Instead, complete the task in the following stages.

STEP 1 - IDENTIFY THE AGENDA SECTION
First identify where the actual meeting agenda begins and ends.

Agenda sections commonly appear after introductory information such as:

- Organization name
- Meeting title
- Meeting date
- Meeting time
- Meeting location
- Committee members
- Staff information
- Contact details
- Public comment instructions
- Zoom/Webinar information
- Accessibility information
- General meeting instructions
- Headers
- Footers
- Page numbers

Ignore all of the above.

The agenda section usually contains discussion topics that will actually be covered during the meeting.

Examples include:

- Call to Order
- Roll Call
- Approval of Minutes
- Public Comment
- Consent Agenda
- Discussion Items
- Briefings
- Presentations
- Reports
- Proposed Ordinances
- Proposed Motions
- Project Updates
- Technical Reviews
- Design Reviews
- Other Business
- Adjournment

STEP 2 - DETECT THE AGENDA STRUCTURE
Agenda items may appear in many different formats.

Examples:

1.
2.
3.

A.
B.
C.

•
•

Roman numerals

Tables

Indented lists

Nested sections

Paragraphs

Do NOT assume a specific layout.

Understand the document structure before extracting.

If the agenda is presented in a table, identify the columns first.

Typical columns include:

- Agenda ID
- Item Number
- Agenda Title
- Topic
- Subject
- Presenter
- Presented By
- Speaker
- Lead
- Owner
- Responsible Person
- Department
- Duration

If an agenda item contains a title followed by one or more descriptive
paragraphs, they belong to the SAME agenda item.

The title and its description should be merged into a single topic.

Do not split a single agenda item into multiple topics.

STEP 3 - IDENTIFY THE PRESENTER
For every agenda item determine whether a presenter is explicitly associated with it.

Possible labels include (but are not limited to):

- Presenter
- Presented By
- Speaker
- Agenda Presenter
- Lead
- Owner
- Responsible
- Presented by
- Council Staff
- Staff
- Department Lead
- Project Lead
- Facilitator
- Chair
- Vice Chair

The presenter's name may appear:

- in the same row
- below the agenda item
- above the agenda item
- in an adjacent table column
- immediately after the topic

When extracting an agenda item, search the nearby text before and after
the topic.

If a presenter, council staff, presenter, presented by, project lead,
staff lead, briefing by or similar information is found immediately
before or after the agenda item, associate that person with the agenda item.

If none of them is explicitly associated with the agenda item, return:

null

Do NOT guess.

Do NOT use committee members listed at the beginning of the document unless they are explicitly associated with that agenda item.


STEP 4 - EXTRACT THE AGENDA

After identifying the agenda section:

Extract every agenda item.

Rules:

- Copy the agenda topic exactly as written whenever possible.
- Do NOT summarize.
- Do NOT rewrite.
- Do NOT shorten.
- If the title spans multiple lines, combine them into one topic.
- Do NOT include administrative information.
- Do NOT include contact information.
- Do NOT include webinar instructions.
- Do NOT include meeting metadata.
- Do NOT include organization information.
- Do NOT include committee member lists.
- Do NOT include page headers or page footers.

If no agenda section exists, return:

[]

also extract

details
- Include ALL remaining information that belongs to this agenda item but is not part of the topic.
- Examples include:
    - agenda description
    - ordinance description
    - motion description
    - proposal description
    - objective
    - supporting information
    - sponsor
    - department
    - project information
    - references
    - document numbers
    - any explanatory text directly associated with this agenda item
- Preserve the wording as closely as possible.
- Do NOT summarize.
- Do NOT remove technical details.
- Do NOT include information belonging to another agenda item.
- If there are multiple paragraphs describing the same agenda item, combine them into a single details string.

If an agenda item consists of a title followed by descriptive paragraphs, they represent ONE agenda item.


For every agenda item extract:

{{
    "topic": "...",
    "speaker": "..."
    "details": "..."
}}

Rules:

1. Copy the agenda topic exactly as written whenever possible.
2. Do not summarize.
3. Do not merge multiple agenda items.
4. If an agenda item spans multiple lines, combine them into a single topic.
5. If a presenter, owner, lead, chair, council staff, presenter, speaker or responsible person is explicitly associated with that agenda item, extract their name.
6. If no presenter is explicitly associated with the agenda item, return null.
7. Never invent a speaker.
8. Return ONLY valid JSON.
9. If no agenda items are found, return an empty JSON array [].


AGENDA DOCUMENT:
{text}


=== SECTION 2: GLOBAL CONTEXT (SUPPORTING INFORMATION ONLY) ===

The following information has been retrieved from the organization's Global Context knowledge base.

PURPOSE: Use this ONLY to improve your understanding of terminology, abbreviations, project names, department names, organization names, and technical background mentioned in the Agenda Document above.

CRITICAL RULES - READ CAREFULLY:
- Do NOT create additional agenda items from this section.
- Do NOT add topics that appear in Global Context but are absent from the Agenda Document.
- Do NOT modify the agenda structure in any way.
- Do NOT merge or combine topics based on Global Context.
- Do NOT change the number of agenda items.
- The Agenda Document in Section 1 is the SOLE authority for all agenda items.
- If the Agenda Document contains 5 items, your output must contain exactly 5 items - no more, no less.
- Global Context exists only to help you understand what the agenda items mean, not to create new ones.

GLOBAL CONTEXT:
{global_context}

JSON:"""


REFERENCE_COMPRESS_PROMPT = """\
You are a technical analyst preparing background knowledge for a meeting. Given the reference/context document below, extract the most relevant information in concise bullet points.

Focus on:
- Key facts, figures, decisions, or specifications mentioned
- Background knowledge relevant to understanding the meeting topic
- Project status, system descriptions, technical context
- Any constraints, requirements, or policies mentioned

Rules:
- Output ONLY bullet points starting with -
- Be concise. Each bullet should be one complete sentence.
- Preserve technical terms, project names, and system names exactly.
- Do NOT include irrelevant or generic content.

REFERENCE DOCUMENT:
{text}

Knowledge summary:"""


AGENDA_FROM_SUMMARY_PROMPT = """\
You are an expert meeting agenda reconstructor.

Given the meeting's transcription summary below, identify the main topics or agenda items that were discussed in the meeting.
For each topic, extract or identify the main speaker/presenter associated with that topic if it is clear from the summary. If no specific speaker is associated, use null.
First understand complete meeting then identify the agenda item.

Output ONLY a valid JSON array of objects, where each object has:
- "topic": a concise title of the agenda topic discussed.
- "speaker": the name of the speaker lead (or null if not specified/clear).

Return ONLY the JSON array. Do not include markdown formatting or wrapping.

Example:
[
  {{"topic": "Q3 Financial Review", "speaker": "Alice"}},
  {{"topic": "Product Roadmap Discussion", "speaker": null}}
]

TRANSCRIPTION SUMMARY:
{summary}

JSON:"""






RAW_MOM_EXTRACTION_PROMPT = """\
You are a precise information extractor for meeting records. Your ONLY job is structured extraction - not summarization.

You will receive evidence retrieved from the transcript, presentation slides, and organizational documents for ONE agenda topic.
The evidence is divided into clearly labelled sections:
  - TRANSCRIPT sections: actual spoken discussion from the meeting recording
  - MEETING CONTEXT / AGENDA CONTEXT sections: uploaded documents (slides, specs, plans)
  - GLOBAL CONTEXT section: organizational knowledge documents
  - AGENDA-SPECIFIC CONTEXT: additional per-agenda reference material

The transcript text includes timeline headers in the format: [HH:MM:SS - HH:MM:SS] Speaker Name
For every Transcript entry:

- Read the timeline directly from the transcript headers.
- If the discussion spans multiple consecutive transcript blocks, use the earliest start time and latest end time.
- Never invent, estimate, or modify timestamps.
- Never use timestamps from another discussion.
- The timeline must correspond exactly to the transcript used to produce the point.

The retrieved evidence may contain discussion from multiple agenda topics because semantic retrieval is not perfect.

Do NOT discard transcript discussion simply because it appears unrelated to the supplied agenda.

If a transcript section contains factual discussion, extract it faithfully with its correct speaker and timeline.

Never rewrite or modify discussion so that it appears to match the agenda.

Your task is to extract ALL factual information that is directly relevant to the current agenda topic.

---
CORE EXTRACTION REQUIREMENTS:

1. EXTRACT ALL MEANINGFUL POINTS - no artificial limit. Capture every:
   - Decision made
   - Discussion point or topic covered
   - Question raised and answer provided
   - Concern or risk identified
   - Resolution or agreement reached
   - Follow-up item or pending action
   - Action item with owner/deadline
   - Milestone or deadline reference
   - Dependency noted
   Merge duplicate ideas into professional concise points. Do NOT truncate or omit content to reduce count.

2. DISCUSSION SEGMENTATION:
Extract discussion as a sequence of individual discussion points, similar to meeting minutes.
- Create a new discussion entry whenever the conversation shifts to a new idea, question, response, decision, concern, proposal, clarification, action item, or significant fact.
- Do NOT combine multiple independent discussion topics into a single entry, even if they occur within the same transcript segment.
- Keep each entry focused on one coherent discussion point.
- If several sentences expand or explain the same idea, combine them into one entry.
- If the speaker moves to a different topic, create a new entry.
- Preserve the natural flow of the meeting by keeping discussion points in chronological order.

3. DO NOT OMIT IMPORTANT FACTS: Preserve every important factual detail including:
   - Dates, deadlines, effective dates, approval dates, extensions, and timelines.
   - Decisions, amendments, motions, votes, and outcomes.
   - Financial values, quantities, locations, ordinance numbers, property details, and technical information.
   - Questions raised, responses provided, concerns discussed, and follow-up requests.
   - Action items with owner, deadline, and status when available.

4. SOURCE ATTRIBUTION - every entry MUST include:
   - "source_type": one of "Transcript", "Meeting Context", "Agenda Context", "Global Context"
   - "timeline": {{"start": <seconds as float>, "end": <seconds as float>}} if from transcript; null otherwise
   - "source_reference": the timestamp range "HH:MM:SS - HH:MM:SS" if from transcript; the document filename if from context; "Global Context" if from global knowledge

5. CONTEXT-ONLY INFORMATION: If information comes ONLY from Meeting Context, Agenda Context, or Global Context and was NOT explicitly discussed in the transcript, set "type" to "reference" and include a note like "(For Information)" at the start of the "point" field. Do NOT present it as if it was discussed in the meeting.

6. NO SUMMARIES: Do not generate introductions, conclusions, or summaries. Perform factual extraction only.

For Transcript entries:

- Extract ONLY information explicitly stated in the transcript.
- Never infer missing facts.
- Never combine facts from different transcript sections unless they are clearly discussing the same topic.
- Never replace transcript facts with similar information from context documents.
- Every transcript entry MUST be traceable to one or more transcript lines.
- If multiple timeline blocks belong to the same discussion, merge them and include the complete timeline covering all supporting transcript blocks.
- Preserve chronological order exactly as spoken.
---
Return ONLY a valid JSON object matching this exact schema:

{{
  "agenda_topic": "{agenda_topic}",
  "agenda_speaker": "{agenda_speaker}",
  "discussion": [
    {{
      "type": "decision|action|discussion|clarification|risk|milestone|dependency|reference",
      "speaker": "Speaker name or null if unknown",
      "point": "One complete discussion point. Preserve all important details relevant to that point, including dates, numbers, decisions, questions, answers, technical terms, and outcomes. If the discussion moves to a different topic or idea, create a new discussion entry instead of extending this one.",
      "source_type": "Transcript|Meeting Context|Agenda Context|Global Context",
      "timeline": {{"start": 123.4, "end": 145.6}},
      "source_reference": "HH:MM:SS - HH:MM:SS or filename.pdf or Global Context",
      "dates": [
        {{"value": "date/time value", "purpose": "what this date is for"}}
      ],
      "action": {{
        "owner": "person responsible or null",
        "description": "what needs to be done or null",
        "deadline": "deadline date or null",
        "status": "open|in_progress|completed|null"
      }}
    }}
  ]
}}

Rules for discussion entries:
- Extract ALL meaningful points - there is NO maximum limit. Completeness is the goal.
- Group related facts together so each entry contains a complete, cohesive subset of discussion details.
- "dates" array: include ONLY if the entry involves a specific actual calendar date or deadline (e.g. "July 8, 2026", "2026-07-08"). NEVER output transcript timestamps, timeline ranges, or offsets like 00:02-05:03 or 3975.18 as dates.
- "action": fill ONLY if the entry is an action item; set all fields to null otherwise.
- "type" must be one of: decision, action, discussion, clarification, risk, milestone, dependency, reference.
- "source_type" is REQUIRED on every entry. Use "Transcript" only for entries from the TRANSCRIPT section.
- "timeline" is REQUIRED for Transcript entries. Parse the [HH:MM:SS - HH:MM:SS] header from the evidence text. Set to null for non-transcript entries.
- "source_reference" is REQUIRED. For Transcript: use "HH:MM:SS - HH:MM:SS". For context docs: use the filename. For global: use "Global Context".
- Preserve all technical terms, acronyms, numbers, system names, and project names exactly.

AGENDA TOPIC: {agenda_topic}
SPEAKER: {agenda_speaker}

RETRIEVED EVIDENCE:
{evidence}

JSON:"""


RAW_MOM_REPAIR_PROMPT = """\
You are an expert JSON repair utility. Your ONLY job is to take an invalid/malformed JSON string from a meeting tool and output a valid JSON object matching the required schema.

Schema requirements:
- Return ONLY a valid JSON object with the expected keys.
- Do NOT rewrite, change, or summarize any content.
- Do NOT add or remove discussion points.
- Do NOT hallucinate missing information.
- Keep all fields, speaker attributions, text statements, dates, and action items exactly as they appear in the original text.
- Only repair JSON syntax, escaping, commas, brackets, quotes, and schema formatting.
  
Expected Schema:
{{
  "agenda_topic": "Topic text",
  "agenda_speaker": "Presenter name or null",
  "discussion": [
    {{
      "type": "decision|action|discussion|clarification|risk|milestone|dependency|reference",
      "speaker": "Speaker name or null",
      "point": "Text statement",
      "source_type": "Transcript|Meeting Context|Agenda Context|Global Context",
      "timeline": {"start": 123.4, "end": 145.6},
      "source_reference": "HH:MM:SS - HH:MM:SS or filename or Global Context",
      "dates": [
        {{"value": "date/time", "purpose": "purpose"}}
      ],
      "action": {{
        "owner": "owner name or null",
        "description": "description or null",
        "deadline": "deadline or null",
        "status": "open|in_progress|completed|null"
      }}
    }}
  ]
}}

Return ONLY the repaired JSON.
Do NOT include any explanations, markdown code blocks, or formatting fences.

INVALID RAW JSON TO REPAIR:
{raw_json}

REPAIRED VALID JSON:"""


# ══════════════════════════════════════════════════════════════
# Raw MoM → Final MoM Conversion Prompt
# Completely independent of the transcript-based MOM_PROMPT above.
# ══════════════════════════════════════════════════════════════

RAW_MOM_TO_MOM_PROMPT = """\
You are an experienced Executive Assistant responsible for producing professional, comprehensive, and detailed Minutes of Meeting (MoM).

You will receive a structured Raw MoM - a collection of agenda items, each with extracted discussion entries including decisions, actions, risks, milestones, and clarifications.

Your task is to convert this structured data into a polished, professional, and detailed Final Minutes of Meeting. Do not create brief summaries or one-line summaries. Generate complete, detailed documentation suitable for official records.

Return ONLY a valid JSON object.
Do NOT wrap the response in markdown.
Do NOT include ```json.
Do NOT include explanations or any text outside the JSON.

The output MUST strictly follow this schema:

{{
  "title": "Professional meeting title derived from the agenda topics",
  "introduction": "A professional executive overview describing the meeting purpose, major agenda topics covered, and overall context.",
  "points_discussed": [
    "Comprehensive detailed description explaining what was discussed, why it was discussed, background context, decisions made, and relevant details."
  ],
  "action_items": [
    {{
      "task": "Detailed task description",
      "background": "Background/context explaining why the task is needed and what supporting information is relevant",
      "owner": "Responsible owner or Unassigned",
      "deadline": "Deadline if present, otherwise ASAP or null",
      "status": "Current status or expected outcome/result"
    }}
  ],
  "conclusion": "Professional conclusion summarizing the key decisions made, agreements reached, actions assigned, risks noted, and next steps.",
  "actual_start_time": null
}}

Conversion Rules:

TITLE
- Derive a concise, professional title from the meeting agenda topics.
- Reflect the primary objectives of the meeting.

INTRODUCTION
- Write a well-structured executive introduction (3-5 sentences).
- Summarize what was discussed across all agenda items.
- Do NOT copy raw entries verbatim. Synthesize them professionally.

POINTS DISCUSSED
- Expand each agenda topic into comprehensive discussion points rather than concise summaries.
- Each key point should clearly explain:
  * WHAT was discussed.
  * WHY it was discussed.
  * Important background or supporting information.
  * Decisions made.
  * Relevant dates, numbers, ordinance references, financial values, locations, and technical details.
- Write each point as a complete, highly informative, and detailed paragraph.
- Preserve ALL technical terms, project names, system names, acronyms, and numeric values exactly.
- Do NOT omit any meaningful discussion entry.
- Do NOT hallucinate or invent new content not present in the Raw MoM data.

ACTION ITEMS
- Generate detailed action items from the raw discussion/action entries.
- For each action item, include:
  * task: Clear description of the task.
  * background: Relevant background/context from the discussion.
  * owner: Responsible owner (if known, else "Unassigned").
  * deadline: Deadline or important dates (ASAP if unspecified).
  * status: Current status or expected outcome.
- Do NOT duplicate action items.
- Do NOT invent owners or deadlines not present in the data.

CONCLUSION
- Write a professional executive summary.
- Cover key outcomes, decisions made, agreements reached, unresolved items, risks, and next steps.
- Base ONLY on the provided Raw MoM data. Do NOT add information not present.

STRICT RULES
- Never hallucinate facts not present in the Raw MoM input.
- Produce detailed, professional meeting minutes suitable for official documentation rather than concise summaries.
- If a field has no relevant information: arrays → [], string fields → null.
- The output must be valid JSON and nothing else.

RAW MOM DATA:
{raw_mom_text}

JSON:"""


KEY_POINTS_PROMPT = """\
You are an expert meeting analyst. Extract the key discussion points from the following meeting transcript.

For each key point, provide:
• [Topic Name]
[One to three sentences explanation of what was discussed about this topic]

Extract all key points. Cover all major topics discussed.

Rules:
- Preserve all technical terms, acronyms, and project names exactly as spoken.
- Do NOT attribute to speakers.
- Do NOT repeat verbatim transcript sentences.

TRANSCRIPT:
{transcript}

Key discussion points:"""


ACTION_ITEMS_PROMPT = """\
You are an expert meeting analyst. Extract all action items from the following meeting transcript.

Group them into sections:

General Action Items
- [items not clearly assigned to any specific speaker]

[Speaker Name or Role if items were assigned to them]
- [their specific action items]

Rules:
- Only include a speaker section if that speaker has action items.
- Do NOT include empty sections.
- Do NOT include Owner or Deadline fields.
- Do NOT number items. Use - prefix for each item.
- If there are no action items, return: None identified.
- Preserve all technical terms and project names exactly.

TRANSCRIPT:
{transcript}

Action items (grouped by speaker):"""


KEY_DECISIONS_PROMPT = """\
You are an expert meeting analyst. Extract all concrete decisions that were made or agreed upon in the following meeting.

Return a numbered list of decisions only. Each decision should be one clear sentence.
If there are no decisions, return: None identified.

Rules:
- Only include decisions explicitly stated or agreed upon (not proposals or suggestions).
- Preserve all technical terms exactly.

TRANSCRIPT:
{transcript}

Decisions made:"""


CHUNK_SUMMARY_PROMPT = """\
Summarize the following meeting excerpt in 3-7 sentences. Focus on the main topics discussed and any decisions or outcomes.
Preserve all technical terms, acronyms, and project names exactly. Do not attribute to speakers.

EXCERPT:
{chunk}

Summary:"""


SPEAKER_SUMMARY_PROMPT = """\
You are an expert meeting analyst. Summarize only {speaker}'s contributions from the transcript below in 2-4 sentences.

Focus on what {speaker} specifically discussed, proposed, or decided. Do NOT repeat verbatim sentences.
Preserve all technical terms, acronyms, and project names exactly.

TRANSCRIPT (only {speaker}'s lines):
{transcript}

{speaker}'s summary:"""


SPEAKER_KEY_POINTS_PROMPT = """\
You are an expert meeting analyst. Extract 3-6 key points from {speaker}'s contributions in the transcript below.

Format:
• [Key point title]: [One sentence explanation]

Preserve all technical terms exactly. Only include what {speaker} actually said.

TRANSCRIPT (only {speaker}'s lines):
{transcript}

{speaker}'s key points:"""


SPEAKER_ACTION_ITEMS_PROMPT = """\
You are an expert meeting analyst. Extract any action items that {speaker} committed to or was assigned in the transcript below.

Return a numbered list. If none, return: None identified.

TRANSCRIPT (only {speaker}'s lines):
{transcript}

{speaker}'s action items:"""


# ══════════════════════════════════════════════════════════════
# Collection AI Chat Prompts - used by Collection AI feature
# ══════════════════════════════════════════════════════════════

COLLECTION_PLANNING_PROMPT = """\
You are an expert AI triage and retrieval planner for an enterprise meeting analysis system.
Analyze the user's question and recent conversation history, then decide if meeting context from vector store is needed to answer.

You MUST respond with valid JSON strictly following this schema:
```json
{{
  "context_required": true | false,
  "detail": "string"
}}
```

Rules:
1. If the question requires information from meetings, transcripts, decisions, action items, or project context:
   - Set "context_required": true
   - Set "detail": A concise, targeted retrieval plan string containing key search terms, entities, technical concepts, topics, speaker names, or time constraints needed for vector search. Do NOT answer the user's question at this stage.

2. If the question is general conversation, greeting (e.g., "Hello", "How are you"), meta-question (e.g., "What can you do?"), or basic knowledge that does NOT require meeting context:
   - Set "context_required": false
   - Set "detail": The complete, polite, and helpful final answer to the user's question. No retrieval should be performed.

{conversation_history}

USER QUESTION: {question}

JSON RESPONSE:"""

COLLECTION_CHAT_PROMPT = """\
You are an expert meeting analyst assistant. You have access to transcripts and notes from multiple meetings within a collection.

Answer the user's question using ONLY the retrieved meeting context below. Be thorough, accurate, and professional.

Rules:
- Base your answer ONLY on the provided context. Do NOT make up information.
- When referencing information from a specific meeting, cite it using the format: [Meeting: <meeting_name>]
- If timestamps are available, include them: [Meeting: <meeting_name> @ MM:SS]
- If the context does not contain enough information to answer, say so clearly.
- Preserve all technical terms, project names, and acronyms exactly.
- Format your response in clear Markdown with headers, bullets, and bold text where appropriate.

{conversation_history}

MEETING CONTEXT:
{context}

USER QUESTION: {question}

Answer:"""

COLLECTION_COMPARE_PROMPT = """\
You are an expert meeting analyst. Compare the following two meetings and generate a structured comparison report.

Generate a detailed comparison document in Markdown format with EXACTLY these sections:

## Common Discussion Topics
Topics that appear in both meetings.

## New Topics Introduced
Topics in Meeting B that were NOT discussed in Meeting A.

## Progress & Improvements
Areas where progress was made between Meeting A and Meeting B.

## Decisions Changed
Decisions that were revised or updated between the two meetings.

## Pending Items Carried Over
Action items or open issues from Meeting A that remain unresolved in Meeting B.

## Recommendations & Next Steps
Based on the trajectory across both meetings, what should happen next.

Rules:
- Be specific and reference actual discussion content from each meeting.
- Cite which meeting each point comes from: [Meeting A: <name>] or [Meeting B: <name>]
- Preserve all technical terms and project names exactly.
- If a section has no relevant content, write "None identified." - do NOT omit the section.
- Do NOT invent information not present in the meeting data.

MEETING A: {meeting_a_name} ({meeting_a_date})
{meeting_a_context}

MEETING B: {meeting_b_name} ({meeting_b_date})
{meeting_b_context}

Comparison Report:"""

COLLECTION_TOPIC_GROWTH_PROMPT = """\
You are an expert meeting analyst. Track the evolution of a specific topic across multiple meetings in chronological order.

Generate a chronological report in Markdown format with EXACTLY these sections:

## Topic Overview
Brief summary of the topic and its significance across the meetings.

## Chronological Timeline

For EACH meeting that discusses this topic, create a subsection:

### <Meeting Name> (<Date>)
- **What was discussed**: Key points about the topic in this meeting
- **Decisions made**: Any decisions related to the topic
- **Action items**: Tasks assigned related to the topic

## Progress Over Time
How the topic evolved from the earliest to the latest meeting.

## Current Status
The most recent state of this topic based on the latest meeting.

## Open Issues & Next Steps
Unresolved items and recommended next actions.

Rules:
- Only include meetings that actually discuss the topic.
- Be specific - reference actual discussion content.
- Cite meetings: [Meeting: <name>]
- Preserve all technical terms and project names exactly.
- If a section has no relevant content, write "None identified."
- Do NOT invent information not present in the meeting data.

TOPIC: {topic}

MEETING DATA (chronological order):
{meetings_context}

Topic Growth Report:"""


# ══════════════════════════════════════════════════════════════
# ROM Prompts
# ══════════════════════════════════════════════════════════════

ROM_DISCUSSION_EXTRACTION_PROMPT = """You are an expert meeting analyst. Extract structured Stage 1 discussion points from the following transcript window.

CRITICAL RULES FOR TRANSCRIPT PRESERVATION & ACTION ITEMS:

1. TRANSCRIPT FIDELITY
   - Preserve the transcript as the single source of truth.
   - Every discussion point must faithfully represent what was explicitly spoken in the transcript.
   - Capture the complete discussion naturally, including who spoke, what was said, who was addressed if explicit, and the substance of the discussion, while preserving the original meaning and intent.
   - Do not summarize, reinterpret, infer, embellish, or alter the discussion.
   - Only make minimal grammatical or formatting corrections necessary for readability without changing the meaning, emphasis, or level of detail.
   - Do not omit important information or split a single continuous discussion into multiple discussion points unless the conversation clearly changes to a different topic.
   - Merge related discussions into a single discussion point if it makes sense to merge them.
   - Do not introduce information, assumptions, conclusions, context, or terminology that was not explicitly stated in the transcript or if video context if available.

2. IMPROVE ACTION ITEM EXTRACTION:
   - Every action item in "action_items" MUST explicitly identify:
     * Who assigned the task (assigner / requester).
     * Who is responsible for completing it (assignee / action owner).
     * What needs to be done.
     * Any deadline, timeframe, or conditions mentioned in the discussion.
   - Assign tasks ONLY when the assignee is an explicit person's name, organization name, or a specific role/title  .
   - Do NOT use vague references such as "my team," "our team," "you," "they," "everyone," "we," "someone," or similar pronouns/generic groups as assignees.
   - If the transcript does not clearly identify a valid assignee, return null for the assignee field.
   - Always infer the assigner and assignee from the conversation context when they are explicitly mentioned or identify them from the transcipt .
   - Preserve the original intent and wording from the transcript, and NEVER invent or alter responsibilities.

3. DATES & TIMELINES:
   - DATES MUST BE ACTUAL CALENDAR DATES ONLY: In the "dates" array, extract ONLY actual calendar dates or explicit calendar references mentioned in the text (e.g., "July 8, 2026", "2026-07-08", "April 8th", "next Monday"). NEVER output transcript timestamps, window ranges, audio offsets, or values like 00:02-05:03 or 3975.18 as dates.
   - Do NOT generate or decide timeline start/end values. Timelines are computed automatically by the system.

4. TECHNICAL ACCURACY:
   - Preserve ALL technical terminology, abbreviations, acronyms, project names EXACTLY as spoken (e.g., LCA Mk2, AMCA, DRDO, ADA, HAL).
   - Never replace abbreviations with generic wording.
   - If you need additional context to understand a point, add it to required_information instead of guessing.

REFERENCE CONTEXT:
Use the previous_context and video_context_section strictly as supporting reference only. They may help resolve clarify ambiguous references, or provide additional context. Never allow them to override, rewrite, or alter information extracted from the current transcript window. If there is any conflict, the TRANSCRIPT WINDOW is the single source of truth.
previous_context : 
{previous_context}
video_context :
{video_context_section}

TRANSCRIPT WINDOW:
{window_text}

Extract discussion points as JSON:
```json
{{
  "discussion_points": [
    {{
      "discussion_point": "Faithful capture of who said what to whom and what was discussed exactly as spoken",
      "speakers": ["Speaker Name"],
      "decisions": ["Any decisions made"],
      "questions": ["Any questions raised"],
      "technical_terms": ["Exact abbreviations and technical terms used"],
      "dates": ["Actual calendar dates mentioned (e.g. July 8, 2026)"],
      "numbers": ["Any numbers, measurements, quantities mentioned"],
      "project_names": ["Any project names mentioned"],
      "action_items": [
        {{
            "assigner": "Person who assigned/requested the task or null",
            "assignee": "Person responsible for the task or null",
            "task": "Task exactly as requested in the transcript",
            "deadline": "Deadline/timeframe mentioned or null"
        }}
      ],
      "action_owner": "Person responsible for completing the task (assignee) or null",
      "references": ["Any documents, standards, or references mentioned"],
      "required_information": ["Additional context needed to fully understand this point"]
    }}
  ]
}}
```"""

ROM_DISCUSSION_NO_ACTION_ITEMS_PROMPT = """You are an expert meeting analyst. Extract structured Stage 1 discussion points from the following transcript window.

IMPORTANT: This call focuses ONLY on discussion points. Do NOT extract action items — leave the action_items list empty for every point. Action items are being extracted separately.

CRITICAL RULES FOR TRANSCRIPT PRESERVATION:

1. TRANSCRIPT FIDELITY
   - Preserve the transcript as the single source of truth.
   - Every discussion point must faithfully represent what was explicitly spoken in the transcript.
   - Capture the complete discussion naturally, including who spoke, what was said, who was addressed if explicit, and the substance of the discussion, while preserving the original meaning and intent.
   - Do not summarize, reinterpret, infer, embellish, or alter the discussion.
   - Only make minimal grammatical or formatting corrections necessary for readability without changing the meaning, emphasis, or level of detail.
   - Do not omit important information or split a single continuous discussion into multiple discussion points unless the conversation clearly changes to a different topic.
   - Merge related discussions into a single discussion point if it makes sense to merge them.
   - Do not introduce information, assumptions, conclusions, context, or terminology that was not explicitly stated in the transcript or if video context if available.

2. DATES & TIMELINES:
   - DATES MUST BE ACTUAL CALENDAR DATES ONLY: In the "dates" array, extract ONLY actual calendar dates or explicit calendar references mentioned in the text (e.g., "July 8, 2026", "2026-07-08", "April 8th", "next Monday"). NEVER output transcript timestamps, window ranges, audio offsets, or values like 00:02-05:03 or 3975.18 as dates.
   - Do NOT generate or decide timeline start/end values. Timelines are computed automatically by the system.

3. TECHNICAL ACCURACY:
   - Preserve ALL technical terminology, abbreviations, acronyms, project names EXACTLY as spoken (e.g., LCA Mk2, AMCA, DRDO, ADA, HAL).
   - Never replace abbreviations with generic wording.
   - If you need additional context to understand a point, add it to required_information instead of guessing.

REFERENCE CONTEXT:
Use the previous_context and video_context_section strictly as supporting reference only. They may help resolve clarify ambiguous references, or provide additional context. Never allow them to override, rewrite, or alter information extracted from the current transcript window. If there is any conflict, the TRANSCRIPT WINDOW is the single source of truth.
previous_context : 
{previous_context}
video_context :
{video_context_section}

TRANSCRIPT WINDOW:
{window_text}

Extract discussion points as JSON. Leave action_items as an empty list for every point:
```json
{{
  "discussion_points": [
    {{
      "discussion_point": "Faithful capture of who said what to whom and what was discussed exactly as spoken",
      "speakers": ["Speaker Name"],
      "decisions": ["Any decisions made"],
      "questions": ["Any questions raised"],
      "technical_terms": ["Exact abbreviations and technical terms used"],
      "dates": ["Actual calendar dates mentioned (e.g. July 8, 2026)"],
      "numbers": ["Any numbers, measurements, quantities mentioned"],
      "project_names": ["Any project names mentioned"],
      "action_items": [],
      "action_owner": null,
      "references": ["Any documents, standards, or references mentioned"],
      "required_information": ["Additional context needed to fully understand this point"]
    }}
  ]
}}
```"""

ROM_ACTION_EXTRACTION_PROMPT = """You are an expert meeting analyst specialising in action item extraction. Your ONLY task is to identify every task, commitment, assignment, or follow-up action mentioned in the transcript window below.

EXTRACTION RULES:

1. TASK DESCRIPTIONS
   - Every `task` must be complete, self-contained, and understandable without reading the transcript.
   - Include all relevant context that was actually spoken, such as what needs to be done, what it relates to, any important details, and any expected outcome if mentioned.
   - Use only information explicitly present in the transcript.
   - Do **not** invent, summarize beyond the transcript, or add assumptions.
   - If important context was spoken, include it in the task. If it was not spoken, do not create it.

2. IDENTIFY EVERY PARTICIPANT:
   - Always extract who assigned the task (assigner) and who is responsible for completing it (assignee) from the conversation context.
   - Assign tasks ONLY when the assignee is an explicit person's name, organization name, or a specific role/title.
   - Do NOT use vague references such as "my team," "our team," "you," "they," "everyone," "we," "someone," or similar pronouns/generic groups as assignees.
   - If the transcript does not clearly identify a valid assignee, return null for the assignee field. Do NOT guess.
   - Carefully follow the conversation flow and speaker turns.
   - Determine who requested or assigned the work (`assigner`) and who became responsible for completing it (`assignee`).
   - Do not assume that the speaker mentioning the task is automatically the assignee.

3. DEADLINES:
   - For every action item, actively search the entire transcript window for any explicit due date, timeframe, schedule, milestone, or completion condition related to that task.
   - Deadlines may appear before or after the task assignment.
   - Associate the correct deadline with the correct task whenever the conversation clearly indicates they belong together.
   - If multiple deadlines are mentioned, use only the one that clearly applies to the task.
   - If no deadline is explicitly stated anywhere in the transcript window, return null.
   - Never invent or infer deadlines.
   - Do NOT use transcript timestamps or audio offsets as deadlines.

4. SCOPE:
   - Extract ALL action items, commitments, and follow-up tasks visible in this window — even implicit ones where a participant clearly agrees to do something.
   - If no action items exist in this window, return an empty action_items array.

REFERENCE CONTEXT:
Use the context below only as supporting reference to identify speakers and technical terms. Never invent actions not present in the transcript.
previous_context:
{previous_context}
video_context:
{video_context_section}

TRANSCRIPT WINDOW:
{window_text}

Return ONLY a JSON object with an action_items array:
```json
{{
  "action_items": [
    {{
      "assigner": "Person who assigned/requested the task or null",
      "assignee": "Person responsible for the task or null",
      "task": "Complete self-contained task description including full context from the transcript",
      "deadline": "Deadline/timeframe mentioned or null"
    }}
  ]
}}
```"""

MOM_REGENERATE_ACTION_POINTS_PROMPT = """You are a specialist in extracting action items from meeting transcripts. Your ONLY task is to identify every task, commitment, assignment, follow-up action, or agreed responsibility mentioned in the transcript window below and return them in the required JSON format.
EXTRACTION RULES:

1. TASK DESCRIPTIONS
   - Every `task` must be complete, self-contained, and understandable without reading the transcript.
   - Include all relevant context that was actually spoken, such as what needs to be done, what it relates to, any important details, and any expected outcome if mentioned.
   - Use only information explicitly present in the transcript.
   - Do **not** invent, summarize beyond the transcript, or add assumptions.
   - If important context was spoken, include it in the task. If it was not spoken, do not create it.

2. IDENTIFY EVERY PARTICIPANT:
   - Always extract who assigned the task (assigner) and who is responsible for completing it (assignee) from the conversation context.
   - Assign tasks ONLY when the assignee is an explicit person's name, organization name, or a specific role/title  .
   - Do NOT use vague references such as "my team," "our team," "you," "they," "everyone," "we," "someone," or similar pronouns/generic groups as assignees.
   - If the transcript does not clearly identify a valid assignee, return null for the assignee field. Do NOT guess.
   - Carefully follow the conversation flow and speaker turns.
   - Determine who requested or assigned the work (`assigner`) and who became responsible for completing it (`assignee`).
   - Do not assume that the speaker mentioning the task is automatically the assignee.

3. DEADLINES:
   - For every action item, actively search the entire transcript window for any explicit due date, timeframe, schedule, milestone, or completion condition related to that task.
   - Deadlines may appear before or after the task assignment.
   - Associate the correct deadline with the correct task whenever the conversation clearly indicates they belong together.
   - If multiple deadlines are mentioned, use only the one that clearly applies to the task.
   - If no deadline is explicitly stated anywhere in the transcript window, return null.
   - Never invent or infer deadlines.
   - Do NOT use transcript timestamps or audio offsets as deadlines.

4. SCOPE:
   - Extract ALL action items, commitments, and follow-up tasks visible in this window — even implicit ones where a participant clearly agrees to do something.
   - If no action items exist in this window, return an empty action_items array.


OUTPUT REQUIREMENTS
* Do not include explanations, reasoning, markdown, or any text outside the JSON object.
* Ensure the JSON is complete and properly closed.


Return ONLY a JSON object with an action_items array exactly this schema:
```json
{{
  "action_items": [
    {{
      "assigner": "Person who assigned/requested the task or null",
      "assignee": "Person responsible for completing the task or null",
      "task": "Complete self-contained task description using only transcript information",
      "deadline": "Deadline or timeframe mentioned or null"
    }}
  ]
}}
```

## TRANSCRIPT WINDOW

{window_text}

```"""

MOM_DEDUPLICATE_ACTION_POINTS_PROMPT = """You are an expert editor specializing in action item deduplication and consolidation. You are provided with a complete list of action points extracted from a meeting.

Your task is to perform a final global deduplication pass over all action points according to these strict rules:

RULES:
1. IDENTIFY DUPLICATES:
   - Identify duplicate or highly similar action points across all items.
   - Remove exact duplicates and redundant variations.

2. MERGE COMPLEMENTARY INFORMATION:
   - Merge action points that refer to the same underlying task into a single, complete action point while preserving all unique details.
   - Preserve the correct assigner, assignee, and deadline when merging. If multiple versions contain complementary information, combine them into the most complete version using only information already present in the input items.
   - Assign tasks ONLY when the assignee is an explicit person's name, organization name, or a specific role/title  . Do NOT use vague references such as "my team," "our team," "you," "they," "everyone," "we," "someone," or similar pronouns/generic groups as assignees. If a valid assignee is not clearly identified, return null for the assignee field.

3. DO NOT OVER-MERGE:
   - Do NOT merge different tasks, even if they are related or belong to the same topic. Keep distinct tasks separate.

4. PRESERVE ORDER AND SCHEMA:
   - Maintain the original sequence of tasks where possible.
   - Use only information already present in the input list. Do not invent or add new details.

OUTPUT REQUIREMENTS:
* Do not include explanations, reasoning, markdown text, or anything outside the JSON object.
* Ensure the JSON is complete and properly closed.

Return ONLY a JSON object with an action_items array using this exact schema:
```json
{{
  "action_items": [
    {{
      "assigner": "Person who assigned/requested the task or null",
      "assignee": "Person responsible for completing the task or null",
      "task": "Complete self-contained task description",
      "deadline": "Deadline or timeframe mentioned or null"
    }}
  ]
}}
```

INPUT ACTION ITEMS:
{action_items_json}
"""

STAGE1_JSON_REPAIR_PROMPT = """You are a JSON repair assistant.

Your ONLY task is to repair the JSON provided below.

Rules:

* Preserve all extracted information exactly as it appears.
* Do not summarize, rewrite, paraphrase, improve, remove, or invent any content.
* Do not add missing discussion points or action items.
* Only fix JSON syntax and structure so that it becomes valid JSON.
* Close any missing braces, brackets, quotes, or commas if required.
* Remove markdown fences if present.
* If the JSON is truncated, complete only the required JSON syntax to make it valid. Do not generate new content that was not already present.
* Preserve the original schema exactly.
* Return only the repaired JSON object.
* Do not include explanations, reasoning, markdown, or any additional text.

Invalid JSON:

```text
{invalid_json}
```

Return only the repaired valid JSON."""

ROM_POLISH_PROMPT = """You are an expert meeting analyst. Enhance and merge the following discussion points using the retrieved context.

CRITICAL RULES & MERGING INSTRUCTIONS:
- PRESERVE TRANSCRIPT FAITHFULNESS: Do NOT summarize or reinterpret the actual spoken meaning, statements, who said what to whom, or original responsibilities.
- ACTION ITEM EXTRACTION & FORMAT: Every action item MUST explicitly identify who assigned the task (assigner), who is responsible for completing it (assignee), what needs to be done, and any deadlines/timeframes mentioned (e.g., "John assigned Sarah to prepare the budget report by Friday." or "Speaker 1 requested Speaker 2 to submit the revised proposal before next week's meeting."). Assign tasks ONLY when the assignee is an explicit person's name, organization name, or a specific role/title  . Do NOT use vague references such as "my team," "our team," "you," "they," "everyone," "we," "someone," or similar pronouns/generic groups as assignees. If a valid assignee is not clearly identified, return null for the assignee field. Never alter or invent responsibilities.
- DATES MUST BE ACTUAL CALENDAR DATES ONLY: In the "dates" array, extract/preserve ONLY actual calendar dates or explicit calendar references (e.g., "July 8, 2026", "April 8th"). NEVER output transcript timestamps, window ranges, audio offsets, or values like 00:02-05:03 or 3975.18 as dates.
- INFORMATION-DENSE SUMMARY: The enhanced point (`polished_text`) must be a clear, informative, and detailed summary of what was discussed, preserving all important facts, who said what to whom, decisions, dates, numbers, and outcomes.
- MERGE DUPLICATES & SIMILAR POINTS: Identify discussion points in the batch that refer to the same topic, decision, project, or issue. Combine/merge duplicate or highly similar points into a single enriched, comprehensive point. Do NOT output redundant points.
- ENRICH WITH CONTEXT: Use the retrieved context (if any) to add precision, technical details, background, and clarity to the merged points.
- PRESERVE ALL FACTS: Retain ALL technical terminology, project names, acronyms, action owners, decisions, action items, dates, and numbers from the original points.
- RETURN ORIGINAL IDS: For each merged point, return a list of all `original_point_ids` that were merged into it.

BATCH OF ORIGINAL DISCUSSION POINTS:
{original_points}

RETRIEVED CONTEXT:
{retrieved_context}

Return merged and enhanced discussion points as JSON:
```json
{{
  "polished_points": [
    {{
      "original_point_ids": ["id-1", "id-2"],
      "polished_text": "Clear, detailed, information-dense summary of what was discussed, capturing who said what to whom",
      "speakers": ["Speaker Name"],
      "decisions": ["Decisions"],
      "technical_terms": ["Terms"],
      "dates": ["Actual calendar dates mentioned (e.g. July 8, 2026)"],
      "numbers": ["Numbers"],
      "references": ["References"],
      "action_items": [
        {{
          "assigner": "Person who assigned/requested the task or null",
          "assignee": "Person responsible for completing the task or null",
          "task": "Task description",
          "deadline": "Deadline/timeframe mentioned or null"
        }}
      ],
      "action_owner": "Person responsible for completing the task (assignee) or null"
    }
  ]
}}
```"""

ROM_ENHANCE_WINDOW_PROMPT = """You are an expert meeting analyst. Your task is to ENHANCE a window of consecutive discussion points extracted from a meeting transcript using two independently retrieved reference sources: Meeting Context and Global Context.

WRITING STYLE:
- FORMAL THIRD-PERSON MEETING MINUTES: Rewrite every enhanced point in a formal, objective, third-person meeting minutes style. Use a balanced mix of passive and active voice, with a preference toward passive or objective phrasing where it sounds natural.
- VARIED REPORTING VERBS: Attribute contributions naturally, selecting the most contextually appropriate reporting verb or sentence structure for each speaker. Vary the phrasing across points to avoid repetition. Choose from expressions such as: introduced, explained, clarified, questioned, emphasised, agreed, disagreed, proposed, confirmed, concluded, requested, observed, highlighted, reported, noted, stated, advised, recommended, raised, outlined, acknowledged, indicated, expressed concern, sought clarification, presented, or any other verb that best reflects the speaker's intent. Do not restrict yourself to a fixed set; use whatever wording most precisely represents how the speaker contributed to the discussion.
- OBJECTIVE AND PROFESSIONAL TONE: Maintain a neutral, professional tone throughout. Avoid informal language, first or second person, and subjective interpretation.

ENHANCEMENT RULES:
- PRESERVE TRANSCRIPT FAITHFULNESS: Do NOT summarize or reinterpret the actual spoken statements, who said what, or original responsibilities. Never remove or contradict any information from the original discussion points.
- FACTUAL ACCURACY: Preserve all facts as stated. Do NOT add interpretations, assumptions, or inferences beyond what is directly supported by the transcript or retrieved context. Every statement in the enhanced text must be traceable to the original discussion or the retrieved reference context.
- ACTION ITEM EXTRACTION & FORMAT: Every action item MUST explicitly identify who assigned the task (assigner), who is responsible for completing it (assignee), what needs to be done, and any deadlines/timeframes mentioned. Assign tasks ONLY when the assignee is an explicit person's name, organization name, or a specific role/title  . Do NOT use vague references such as "my team," "our team," "you," "they," "everyone," "we," "someone," or similar pronouns/generic groups as assignees. If a valid assignee is not clearly identified, return null for the assignee field. Never alter or invent responsibilities.
- VERIFY TECHNICAL TERMINOLOGY: If the retrieved context confirms or clarifies a technical term, incorporate the clarification.
- EXPAND ABBREVIATIONS: If the context makes an abbreviation's full form clear, expand it the first time it appears in enhanced_text (e.g., "SRS (Software Requirements Specification)").
- ADD MISSING TECHNICAL DETAILS: Only add details that are DIRECTLY supported by the retrieved context. NEVER invent or hallucinate.
- IMPROVE CLARITY: Make the enhanced point clearer and more complete while preserving the original meaning.
- INFORMATION-DENSE: Write clear, professional prose that captures the full substance of what was discussed.
- DATES - CALENDAR DATES ONLY: In the "dates" array, output ONLY actual calendar dates (e.g., "July 8, 2026", "Q3 2026"). NEVER include transcript timestamps, audio offsets, or values like "3975.18" or "00:02-05:03".
- MERGE WITHOUT LOSS: When multiple discussion points in the window cover the same topic, merge them into a single coherent, well-structured enhanced point. You MUST retain every unique fact, decision, clarification, question, and outcome from all merged points — do not discard any detail during merging. Combine the content into well-structured sentences rather than keeping duplicate or near-duplicate points. Return `original_point_ids` listing all merged IDs.
- CONTEXT USAGE REPORT: For EVERY enhanced point, you MUST fill in the `context_usage_report` object honestly. If a field is false, set it to false — do not set everything to true.

DISCUSSION WINDOW (points to enhance):
{window_json}

MEETING CONTEXT (from agenda docs, presentations, design docs, previous MoM, etc.):
{meeting_context}

GLOBAL CONTEXT (from standards, SOPs, manuals, technical references, etc.):
{global_context}

Return the enhanced points as a JSON object:
```json
{{
  "enhanced_points": [
    {{
      "original_point_ids": ["id-1", "id-2"],
      "enhanced_text": "Clear, information-dense enhanced discussion point text written in formal third-person meeting minutes style.",
      "speakers": ["Speaker Name"],
      "decisions": ["Decisions made"],
      "technical_terms": ["Technical terms used"],
      "dates": ["Actual calendar dates only"],
      "numbers": ["Numbers and quantities"],
      "references": ["Document or section references"],
      "action_owner": "Person responsible or null",
      "action_items": [
        {{
          "assigner": "Person who assigned/requested the task or null",
          "assignee": "Person responsible for completing the task or null",
          "task": "Task description",
          "deadline": "Deadline/timeframe mentioned or null"
        }}
      ],
      "context_usage_report": {{
        "verified": true,
        "technical_details_added": false,
        "abbreviations_expanded": true,
        "references_added": false,
        "terminology_clarified": true,
        "no_useful_context": false,
        "meeting_context_docs": ["filename.pdf"],
        "global_context_docs": ["standard.pdf"]
      }}
    }}
  ]
}}
```"""

ROM_DEDUPLICATION_PROMPT = """You are an expert meeting analyst. Evaluate the following candidate discussion points that have high semantic similarity (>= 0.90).

DEDUPLICATION & MERGING INSTRUCTIONS:
1. Determine whether the points in this group are:
   - "duplicate": Virtually identical in topic and details. Select/produce the single best, clearest, and most comprehensive version.
   - "complementary": Refer to the same core topic with different or additional details. Merge them into a single, comprehensive, information-dense point (`polished_text`) without losing any factual details, technical terms, decisions, action items, dates, or numbers.
   - "different": Represent distinct facts, decisions, or topics despite high semantic similarity. Keep all points separate in `result_points`.

2. CRITICAL RULES:
   - DATES MUST BE ACTUAL CALENDAR DATES ONLY: Extract/preserve ONLY actual calendar dates (e.g. "July 8, 2026", "April 8th"). NEVER output transcript timestamps, window ranges, audio offsets, or values like 3975.18 as dates.
   - PRESERVE ALL FACTS: Do NOT drop any unique technical terms, project names, acronyms, decisions, or action items.

CANDIDATE DISCUSSION POINTS:
{candidate_points}

Return result as JSON:
```json
{{
  "group_classification": "duplicate|complementary|different",
  "result_points": [
    {{
      "original_point_ids": ["id-1", "id-2"],
      "polished_text": "Single comprehensive description combining all details if duplicate/complementary, or individual texts if different",
      "speakers": ["Speaker Name"],
      "decisions": ["Decisions"],
      "technical_terms": ["Terms"],
      "dates": ["Dates"],
      "numbers": ["Numbers"],
      "references": ["References"],
      "action_items": [
        {{
          "assigner": "Person who assigned/requested the task or null",
          "assignee": "Person responsible for completing the task or null",
          "task": "Task description",
          "deadline": "Deadline/timeframe mentioned or null"
        }}
      ],
      "action_owner": "Person responsible or null"
    }}
  ]
}}
```"""

ROM_AGENDA_GENERATION_PROMPT = """You are an expert meeting analyst. Generate structured agenda sections from the following input.

{agenda_input}

{context}

{previous_mom}

Generate enriched agenda sections as JSON. Each agenda must have rich semantic representation for accurate discussion point mapping:
```json
{{
  "agendas": [
    {{
      "agenda_id": "A1",
      "title": "Agenda title",
      "description": "Detailed description of what this agenda covers",
      "keywords": ["key", "words", "for", "matching"],
      "related_concepts": ["Related technical concepts"],
      "alternative_terminology": ["Alternative ways this topic might be referred to"],
      "expected_themes": ["Expected discussion themes under this agenda"]
    }}
  ]
}}
```"""

ROM_MOM_EXPANSION_PROMPT = """You are an expert meeting analyst. You have been given a list of agenda items for the CURRENT MEETING and one or more PREVIOUS MEETING MoM (Minutes of Meeting) documents.

Your task: For each agenda item, search ONLY the provided Previous Meeting MoM documents and extract any relevant information that was discussed before.

STRICT RULES:
- Use ONLY information explicitly found in the provided Previous Meeting MoM documents.
- If an agenda item was NOT discussed in any previous meeting document, set "found_in_previous_meeting" to false and leave all text fields as empty strings and all list fields as empty arrays. DO NOT invent, guess, or paraphrase anything not present.
- Extract only verifiable content: previous discussion summary, decisions made, action items assigned, pending work, follow-up items.
- Be concise and factual.

CURRENT MEETING AGENDA LIST:
{agenda_list}

PREVIOUS MEETING MoM DOCUMENTS:
{previous_mom_docs}

Return a JSON object with an "expansions" array covering ALL agenda items:
```json
{{
  "expansions": [
    {{
      "agenda_id": "A1",
      "found_in_previous_meeting": true,
      "previous_discussion": "Concise summary of what was discussed about this topic in the previous meeting",
      "previous_decisions": ["Decision 1", "Decision 2"],
      "previous_action_items": ["Action item 1"],
      "pending_work": ["Pending work item 1"],
      "follow_up_items": ["Follow-up item 1"]
    }},
    {{
      "agenda_id": "A2",
      "found_in_previous_meeting": false,
      "previous_discussion": "",
      "previous_decisions": [],
      "previous_action_items": [],
      "pending_work": [],
      "follow_up_items": []
    }}
  ]
}}
```"""

ROM_AGENDA_ASSIGN_BATCH_PROMPT = """You are an expert meeting analyst. Assign each discussion point in this batch to the most appropriate agenda from its provided candidate agendas.

ASSIGNMENT RULES:
- You MUST assign an agenda_id ONLY from the "candidate_agendas" list provided for each individual point.
- Do NOT invent new agenda names or IDs.
- Do NOT assign an agenda_id that is not present in that point's candidate_agendas list.
- If you are uncertain, choose the closest candidate and set confidence to "low".
- Write a concise 1-sentence reason for each assignment.
- Confidence levels: "high" = clearly on-topic, "medium" = likely related, "low" = uncertain but best match.

AGENDA REFERENCE (titles and descriptions for context):
{agenda_reference}

DISCUSSION POINTS BATCH:
{batch_json}

Return ONLY a JSON object. One entry per discussion point, in the same order as the batch:
```json
{{
  "assignments": [
    {{
      "point_id": "uuid-string",
      "assigned_agenda_id": "A2",
      "confidence": "high",
      "reason": "The point discusses X which directly relates to agenda A2 on Y."
    }}
  ]
}}
```"""

ROM_AGENDA_DOC_POINTS_PROMPT = """You are an expert meeting analyst. A supporting document has been uploaded for a specific agenda item in a meeting. Extract the most important factual points from the document that are directly relevant to this agenda topic.

AGENDA TITLE: {agenda_title}
AGENDA DESCRIPTION: {agenda_description}

SUPPORTING DOCUMENT CONTENT:
{document_text}

INSTRUCTIONS:
- Extract ONLY 2 to 5 factual, agenda-specific points from the document.
- Check the document for any presenter, speaker, author, owner, or 'presented by' name (e.g., 'Presenter: John Doe', 'Presented by Alice', 'Speaker: Bob', 'Owner: Charlie'). If found, return the presenter name in the "presenter" field. Otherwise set "presenter" to null.
- Each point must be a complete, standalone statement of fact, decision, milestone, figure, date, or key information shown in the document.
- Points must be directly relevant to the agenda topic. Ignore unrelated content.
- Do NOT paraphrase or summarize broadly. Extract precise, specific factual information.
- Do NOT include decorative text, slide headers, repeated boilerplate, page numbers, or company logos.
- If the document has very little relevant content, extract fewer points (even 0 is acceptable).
- Write each point as a professional, information-dense sentence suitable for meeting records.

Return ONLY a JSON object:
```json
{
  "presenter": "Speaker / Presenter name if mentioned in document, or null",
  "points": [
    "Factual point 1 extracted from the document relevant to the agenda.",
    "Factual point 2 extracted from the document relevant to the agenda."
  ]
}
```"""

# ══════════════════════════════════════════════════════════════
# Shared utilities
# ══════════════════════════════════════════════════════════════

def _get_prompt(key: str) -> str:
    """
    Return the active AI prompt template for the given key.

    Lookup order:
    1. In-process cache in services.prompt_service (populated from DB on first request).
    2. Hardcoded module-level constant string (always available - never removed).

    This function is synchronous and safe to call from worker threads.
    It NEVER hits the database directly.
    """
    try:
        from services.prompt_service import get_prompt_sync
        result = get_prompt_sync(key)
        if result and result.strip():
            return result
    except Exception:
        pass
    # Final fallback: return the module-level constant
    _CONSTANT_MAP = {
        "mom": MOM_PROMPT,
        "mom_merge": MOM_MERGE_PROMPT,
        "raw_mom_to_mom": "",
        "rom_discussion": ROM_DISCUSSION_EXTRACTION_PROMPT,
        "rom_discussion_no_actions": ROM_DISCUSSION_NO_ACTION_ITEMS_PROMPT,
        "rom_action_extraction": ROM_ACTION_EXTRACTION_PROMPT,
        "stage1_json_repair": STAGE1_JSON_REPAIR_PROMPT,
        "mom_action_regen": MOM_REGENERATE_ACTION_POINTS_PROMPT,
        "mom_action_dedup": MOM_DEDUPLICATE_ACTION_POINTS_PROMPT,
        "rom_polish": ROM_POLISH_PROMPT,
        "rom_enhance_window": ROM_ENHANCE_WINDOW_PROMPT,
        "rom_deduplicate": ROM_DEDUPLICATION_PROMPT,
        "rom_agenda": ROM_AGENDA_GENERATION_PROMPT,
        "rom_mom_expansion": ROM_MOM_EXPANSION_PROMPT,
        "rom_agenda_assign_batch": ROM_AGENDA_ASSIGN_BATCH_PROMPT,
        "rom_agenda_doc_points": ROM_AGENDA_DOC_POINTS_PROMPT,
        "raw_mom_extraction": "",
        "agenda_compress": AGENDA_COMPRESS_PROMPT,
        "agenda_compress_with_context": AGENDA_COMPRESS_WITH_CONTEXT_PROMPT,
        "reference_compress": REFERENCE_COMPRESS_PROMPT,
        "agenda_from_summary": AGENDA_FROM_SUMMARY_PROMPT,
        "executive_summary": EXECUTIVE_SUMMARY_PROMPT,
        "short_summary": SHORT_SUMMARY_PROMPT,
        "detailed_summary": DETAILED_SUMMARY_PROMPT,
        "chunk_summary": CHUNK_SUMMARY_PROMPT,
        "key_points": KEY_POINTS_PROMPT,
        "action_items": ACTION_ITEMS_PROMPT,
        "key_decisions": KEY_DECISIONS_PROMPT,
        "speaker_summary": SPEAKER_SUMMARY_PROMPT,
        "speaker_key_points": SPEAKER_KEY_POINTS_PROMPT,
        "speaker_action_items": SPEAKER_ACTION_ITEMS_PROMPT,
        "raw_mom_repair": "",
        "collection_planning": COLLECTION_PLANNING_PROMPT,
        "collection_chat": COLLECTION_CHAT_PROMPT,
        "collection_compare": COLLECTION_COMPARE_PROMPT,
        "collection_topic_growth": COLLECTION_TOPIC_GROWTH_PROMPT,
    }
    return _CONSTANT_MAP.get(key, "")


def _format_transcript(transcript: List[Dict]) -> str:
    """Convert transcript segments into readable dialogue string (Speaker: text)."""
    lines = []
    for seg in transcript:
        label = seg.get("speaker_label", "Unknown")
        text = seg.get("text", "").strip()
        if text:
            lines.append(f"{label}: {text}")
    return "\n".join(lines)


def _format_for_summary(transcript: List[Dict]) -> str:
    """Convert transcript into plain text with no speaker labels - for prose summaries."""
    parts = []
    for seg in transcript:
        text = seg.get("text", "").strip()
        if text:
            parts.append(text)
    return " ".join(parts)


def _clean_list(raw: str) -> List[str]:
    """Parse a numbered/bulleted list from LLM output."""
    lines = [l.strip() for l in raw.split("\n") if l.strip()]
    cleaned = []
    for line in lines:
        if "none identified" in line.lower():
            continue
        if line and line[0].isdigit() and ". " in line:
            cleaned.append(line.split(". ", 1)[1])
        elif line.startswith("- "):
            cleaned.append(line[2:])
        elif line.startswith("• "):
            cleaned.append(line[2:])
        else:
            cleaned.append(line)
    return [c for c in cleaned if c and len(c) > 3]


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> block or unclosed <think> tags from LLM output."""
    if not text:
        return ""
    import re
    # Remove complete <think>...</think> block
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Also handle unclosed <think> tag if it was truncated at the end
    if "<think>" in cleaned:
        parts = cleaned.split("<think>", 1)
        cleaned = parts[0]
    return cleaned.strip()



def _split_sentences(text: str) -> List[str]:
    """Simple sentence splitter."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def _chunk_text(text: str, max_words: int = 700) -> List[str]:
    """Split text into word-count-bounded chunks, breaking on sentence boundaries."""
    sentences = _split_sentences(text)
    chunks = []
    current_words = []
    current_count = 0

    for sent in sentences:
        word_count = len(sent.split())
        if current_count + word_count > max_words and current_words:
            chunks.append(" ".join(current_words))
            current_words = [sent]
            current_count = word_count
        else:
            current_words.append(sent)
            current_count += word_count

    if current_words:
        chunks.append(" ".join(current_words))

    return [c for c in chunks if len(c.split()) >= 30]


def _empty_mom(recording_meta: dict) -> dict:
    return {
        "title": recording_meta.get("filename", "Meeting Notes"),
        "date": recording_meta.get("created_at", ""),
        "duration": recording_meta.get("duration", 0),
        "planned_start_time": "",
        "actual_start_time": "",
        "participants": recording_meta.get("speakers_detected", []),
        "introduction": "Failed to generate meeting introduction.",
        "points_discussed": [],
        "action_items": [],
        "conclusion": "",
    }


# ══════════════════════════════════════════════════════════════
# Abstract base
# ══════════════════════════════════════════════════════════════

class AIProvider(ABC):
    """Abstract interface for all AI providers."""

    @abstractmethod
    def build_context_summary(self, transcript: List[Dict]) -> str: ...

    @abstractmethod
    def compress_agenda(self, text: str) -> str: ...

    @abstractmethod
    def compress_reference(self, text: str) -> str: ...

    @abstractmethod
    def generate_summary(self, transcript: List[Dict], context: Optional[str] = None) -> str: ...

    @abstractmethod
    def generate_key_points(self, transcript: List[Dict], context: Optional[str] = None) -> List[str]: ...

    @abstractmethod
    def generate_action_items(self, transcript: List[Dict], context: Optional[str] = None) -> List[str]: ...

    @abstractmethod
    def generate_key_decisions(self, transcript: List[Dict], context: Optional[str] = None) -> List[str]: ...

    @abstractmethod
    def generate_mom(self, transcript: List[Dict], recording_meta: dict, context: Optional[str] = None) -> dict: ...

    @abstractmethod
    def generate_executive_summary(self, transcript: List[Dict], context: Optional[str] = None) -> dict: ...

    @abstractmethod
    def generate_short_summary(self, transcript: List[Dict], context: Optional[str] = None) -> str: ...

    @abstractmethod
    def generate_detailed_summary(self, transcript: List[Dict], context: Optional[str] = None) -> str: ...

    @abstractmethod
    def generate_speaker_summaries(self, transcript: List[Dict]) -> Dict[str, Dict]: ...

    @abstractmethod
    def generate_agenda_from_summary(self, summary: str) -> List[Dict]: ...

    def deduplicate_action_points(self, action_items: List[Dict]) -> List[Dict]:
        return action_items


STANDARD_CTX_SIZES = (2048, 4096, 8192, 16384, 32768, 65536, 131072)


def calculate_dynamic_num_ctx(
    prompt: str,
    max_new_tokens: int,
    safety_buffer: int = 512,
    system_content: Optional[str] = None,
    tokenizer=None,
) -> tuple[int, int]:
    """
    Calculate required context size and select smallest suitable standard num_ctx value.

    Returns:
        (est_input_tokens, selected_num_ctx)
    """
    full_text = prompt
    if system_content:
        full_text = f"{system_content}\n{prompt}"

    est_input_tokens = 0
    if tokenizer:
        try:
            est_input_tokens = len(tokenizer.encode(full_text))
        except Exception:
            est_input_tokens = 0

    if est_input_tokens <= 0:
        est_input_tokens = max(1, len(full_text) // 3)

    needed = est_input_tokens + max_new_tokens + safety_buffer

    selected_num_ctx = None
    for size in STANDARD_CTX_SIZES:
        if size >= needed:
            selected_num_ctx = size
            break

    if selected_num_ctx is None:
        import math
        selected_num_ctx = 2 ** math.ceil(math.log2(needed))

    return est_input_tokens, selected_num_ctx


class QwenProvider(AIProvider):
    """
    100% offline summarization and document generation using
    Qwen/Qwen3-4B-Instruct (4-bit BitsAndBytes quantization).

    Model is loaded ONCE at startup and kept resident in memory.
    GPU (CUDA) acceleration when available; CPU fallback otherwise.
    Optimized for RTX 4050 6GB VRAM.

    Qwen3 specifics:
      - Thinking mode is disabled (enable_thinking=False) for structured
        document generation tasks - ensures clean, deterministic output
        without <think>...</think> preamble blocks.
      - Uses the standard chat template via apply_chat_template.
    """

    _model = None
    _tokenizer = None
    _pipeline = None
    _load_attempted = False
    _device = None
    _cached_ollama_port = None
    _cached_ollama_priority = None
    _cached_ollama_model = None

    @classmethod
    def clean_up_other_models(cls):
        """Ensure other AI models are fully unloaded and references are cleared to free VRAM/RAM."""
        logger.info("[QwenAI] Cleaning up other AI models from memory...")
        try:
            from services.transcription import unload_whisperx_model, unload_align_model
            unload_whisperx_model()
            unload_align_model()
        except Exception as e:
            logger.warning(f"[QwenAI] Failed to unload transcription/alignment models: {e}")

        try:
            from services.diarization import unload_diarization_pipeline
            unload_diarization_pipeline()
        except Exception as e:
            logger.warning(f"[QwenAI] Failed to unload diarization pipeline: {e}")

        try:
            from services.embedding import unload_encoder
            unload_encoder()
        except Exception as e:
            logger.warning(f"[QwenAI] Failed to unload speaker encoder: {e}")

        try:
            from main import unload_overlap_model
            unload_overlap_model()
        except Exception:
            pass

        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                logger.info(f"[QwenAI] VRAM cleaned up. Allocated memory: {torch.cuda.memory_allocated() / (1024 * 1024):.2f} MB")
        except Exception:
            pass

    @classmethod
    def _load_model_impl(cls):
        """Actual model loading code block (tokenizer, model, pipeline)."""
        import torch
        from transformers import (
            AutoTokenizer,
            AutoModelForCausalLM,
            BitsAndBytesConfig,
            pipeline,
        )
        from config import settings
        from services.model_loader import ModelLoader

        model_id = getattr(settings, "QWEN_MODEL_ID", "Qwen/Qwen3-4B")
        load_in_4bit = getattr(settings, "QWEN_LOAD_IN_4BIT", True)

        local_path = ModelLoader.get_model_path("nlp_engine")
        if local_path is not None:
            load_from = str(local_path)
            local_files_only = True
            logger.info(f"[QwenAI] Loading from local path: {load_from}")
        else:
            raise FileNotFoundError(
                f"[QwenAI] nlp_engine model not found in MODELS_DIR ({settings.MODELS_DIR}) "
                "or runtime/nlp_engine/. "
                "Copy the Qwen3-4B model folder to 'Application/runtime/nlp_engine/' and restart."
            )

        use_cuda = torch.cuda.is_available()
        cls._device = "cuda" if use_cuda else "cpu"
        logger.info(f"[QwenAI] Loading {load_from} on {cls._device} ...")

        bnb_config = None
        if load_in_4bit and use_cuda:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            logger.info("[QwenAI] 4-bit NF4 quantization enabled (RTX 4050 optimized)")
        elif use_cuda:
            logger.info("[QwenAI] Loading in float16 on CUDA (no quantization)")
        else:
            logger.info("[QwenAI] Loading on CPU (float32, slower inference)")

        hf_token = getattr(settings, "HF_TOKEN", None) or None
        cls._tokenizer = AutoTokenizer.from_pretrained(
            load_from,
            local_files_only=local_files_only,
            token=hf_token if not local_files_only else None,
        )

        model_kwargs = {
            "local_files_only": local_files_only,
            "device_map": "cuda:0" if use_cuda else None,
            "torch_dtype": torch.float16 if use_cuda else torch.float32,
        }
        if not local_files_only:
            model_kwargs["token"] = hf_token
        if bnb_config:
            model_kwargs["quantization_config"] = bnb_config

        cls._model = AutoModelForCausalLM.from_pretrained(load_from, **model_kwargs)
        cls._model.eval()

        pipe = pipeline(
            "text-generation",
            model=cls._model,
            tokenizer=cls._tokenizer,
        )
        logger.info(f"[QwenAI] Qwen3 4B Instruct ready on {cls._device} ✓")
        return pipe

    @classmethod
    def _get_pipeline(cls):
        """Return cached pipeline, loading model once if needed."""
        if cls._pipeline is not None:
            return cls._pipeline
        if cls._load_attempted:
            return None

        from services.device_utils import log_gpu_memory
        logger.info("[QwenAI] Initializing model load. Running pre-load memory cleanup...")
        cls.clean_up_other_models()
        log_gpu_memory("Pre-load QwenLLM")

        try:
            cls._pipeline = cls._load_model_impl()
            if cls._pipeline is not None:
                logger.info("[QwenAI] Model loaded successfully on first attempt.")
                log_gpu_memory("Post-load QwenLLM (Success)")
                return cls._pipeline
        except Exception as e:
            logger.warning(f"[QwenAI] First loading attempt failed: {e}. Recovering VRAM and retrying load once...", exc_info=True)

        # First load failed. Run recovery sequence and retry once.
        cls.clean_up_other_models()
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except Exception:
            pass

        log_gpu_memory("Pre-retry load QwenLLM")

        try:
            cls._pipeline = cls._load_model_impl()
            if cls._pipeline is not None:
                logger.info("[QwenAI] Model loaded successfully on retry.")
                log_gpu_memory("Post-load QwenLLM (Success on retry)")
                return cls._pipeline
        except Exception as retry_err:
            logger.error(f"[QwenAI] Loading attempt retry failed: {retry_err}. Disabling Qwen LLM.", exc_info=True)
            cls._pipeline = None

        # Mark load attempted to prevent further retry loops
        cls._load_attempted = True
        log_gpu_memory("Post-load QwenLLM (Permanent Failure)")
        return None

    @classmethod
    def unload_model(cls):
        """Unload Qwen model from VRAM and CPU memory."""
        from services.device_utils import log_gpu_memory
        log_gpu_memory("Pre-unload QwenLLM")
        logger.info("[QwenAI] Unloading Qwen3 model...")
        cls._pipeline = None
        cls._model = None
        cls._tokenizer = None
        cls._load_attempted = False
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info("[QwenAI] Qwen3 model unloaded.")
        log_gpu_memory("Post-unload QwenLLM")


    @classmethod
    def _get_active_settings(cls):
        """Retrieve all active settings (Ollama options + task limits) from DB/config."""
        from config import settings
        
        cfg = {
            "use_ollama": getattr(settings, "USE_OLLAMA", True),
            "ollama_server_url": getattr(settings, "OLLAMA_SERVER_URL", "http://localhost:11434"),
            "ollama_port": getattr(settings, "OLLAMA_PORT", 11434),
            "ollama_model_priority": getattr(settings, "OLLAMA_MODEL_PRIORITY", "gemma,qwen,llama,deepseek,mistral"),
            "ollama_num_ctx": 32768,
            "ollama_dynamic_ctx": True,
            "ollama_think": False,
            "ollama_temperature": 0.0,
            "ollama_top_p": 0.9,
            "ollama_top_k": 40,
            "ollama_repeat_penalty": 1.15,
            "ollama_seed": -1,
            "ollama_stop": "",
            "ollama_keep_alive": "5m",
            "ollama_num_thread": 0,
            "ollama_num_gpu": -1,
            "max_tokens_mom": 1500,
            "max_tokens_mom_merge": 3072,
            "max_tokens_agenda_compress": 2000,
            "max_tokens_reference_compress": 2000,
            "max_tokens_agenda_from_summary": 1024,
            "max_tokens_executive_summary": 700,
            "max_tokens_short_summary": 120,
            "max_tokens_detailed_summary": 3000,
            "max_tokens_chunk_summary": 256,
            "max_tokens_key_points": 1028,
            "max_tokens_action_items": 1028,
            "max_tokens_key_decisions": 1028,
            "max_tokens_speaker_summary": 200,
            "max_tokens_speaker_key_points": 350,
            "max_tokens_speaker_action_items": 250,
            "max_tokens_collection_chat": 1500,
            "max_tokens_collection_compare": 1500,
            "max_tokens_collection_topic_growth": 1500,
            "max_tokens_vocab_extractor": 512,
            "rom_transcript_window": 2.0,
            "rom_meeting_top_k": 5,
            "rom_global_top_k": 3,
            "rom_windows_per_batch": 5,
            "rom_parallel_window_processing": 2,
            "rom_separate_action_extraction": False,
            "whisper_batch_size": 8,
            "max_tokens_rom_discussion": 4096,
            "max_tokens_rom_discussion_no_actions": 4096,
            "max_tokens_rom_action_extraction": 2048,
            "max_tokens_stage1_json_repair": 4548,
            "max_tokens_mom_action_regen": 4048,
            "max_tokens_rom_polish": 4096,
            "max_tokens_rom_enhance_window": 4096,
            "max_tokens_rom_deduplicate": 2048,
            "max_tokens_rom_agenda": 2048,
            "max_tokens_rom_mom_expansion": 3000,
            "max_tokens_rom_agenda_assign_batch": 4096,
            "max_tokens_rom_agenda_doc_points": 1024,
            "max_tokens_raw_mom_to_mom": 3000,
            "max_tokens_raw_mom_extraction": 1024,
            "max_tokens_raw_mom_repair": 1024,
        }

        import sqlite3
        db_url = getattr(settings, "DATABASE_URL", "")
        db_path = None
        if db_url.startswith("sqlite+aiosqlite:///"):
            db_path = db_url[len("sqlite+aiosqlite:///"):]
        elif db_url.startswith("sqlite:///"):
            db_path = db_url[len("sqlite:///"):]
        elif db_url:
            db_path = db_url

        if db_path:
            try:
                conn = sqlite3.connect(db_path, timeout=5.0)
                try:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()
                    cursor.execute("SELECT * FROM user_settings ORDER BY id DESC LIMIT 1")
                    row = cursor.fetchone()
                finally:
                    conn.close()

                if row:
                    for k in cfg:
                        try:
                            val = row[k]
                            if val is not None:
                                if isinstance(cfg[k], bool):
                                    cfg[k] = bool(val)
                                else:
                                    cfg[k] = val
                        except Exception:
                            pass
            except Exception as db_err:
                logger.debug(f"[QwenAI] Failed to read settings from SQLite DB: {db_err}")

        # Normalize server_url
        server_url = cfg["ollama_server_url"]
        if not server_url.startswith("http://") and not server_url.startswith("https://"):
            server_url = f"http://{server_url}"
        cfg["ollama_server_url"] = server_url.rstrip("/")

        return cfg

    @classmethod
    def _get_ollama_settings(cls):
        """Retrieve active Ollama settings from config and SQLite database."""
        cfg = cls._get_active_settings()
        return cfg["use_ollama"], cfg["ollama_server_url"], cfg["ollama_port"], cfg["ollama_model_priority"]

    @classmethod
    def _detect_ollama_model(cls, server_url: str, priority_list_str: str) -> Optional[str]:
        """Detect the best supported Ollama model currently running or installed."""
        import urllib.request
        import json

        base_url = server_url.rstrip("/")
        priorities = [p.strip().lower() for p in priority_list_str.split(",") if p.strip()]
        if not priorities:
            return None

        def get_url_json(url):
            try:
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=2.0) as resp:
                    if resp.status == 200:
                        return json.loads(resp.read().decode('utf-8'))
            except Exception:
                pass
            return None

        def find_match(models_list):
            for priority in priorities:
                for m in models_list:
                    name = m.get("name", "").lower()
                    model_id = m.get("model", "").lower()
                    details = m.get("details", {})
                    family = (details.get("family") or "").lower()
                    families = [f.lower() for f in details.get("families") or [] if f]
                    
                    if (priority in name or 
                        priority in model_id or 
                        priority in family or 
                        any(priority in fam for fam in families)):
                        return m.get("name")
            return None

        # 1. Check running models first via /api/ps
        ps_data = get_url_json(f"{base_url}/api/ps")
        if ps_data and "models" in ps_data:
            match = find_match(ps_data["models"])
            if match:
                logger.info(f"[QwenAI] Detected running Ollama model: {match} at {base_url}")
                return match

        # 2. Check installed models via /api/tags
        tags_data = get_url_json(f"{base_url}/api/tags")
        if tags_data and "models" in tags_data:
            match = find_match(tags_data["models"])
            if match:
                logger.info(f"[QwenAI] Detected installed Ollama model: {match} at {base_url}")
                return match

        return None

    @classmethod
    def _call_ollama(cls, server_url: str, model: str, prompt: str, max_new_tokens: int) -> Optional[str]:
        """Perform chat generation call using configured Ollama server endpoint."""
        import urllib.request
        import json

        cfg = cls._get_active_settings()
        
        system_content = (
            "You are an expert enterprise meeting analyst. "
            "Follow instructions exactly. Preserve all technical terminology."
        )

        dynamic_enabled = bool(cfg.get("ollama_dynamic_ctx", True))
        manual_num_ctx = int(cfg.get("ollama_num_ctx", 32768))

        est_input_tokens, calculated_num_ctx = calculate_dynamic_num_ctx(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            safety_buffer=512,
            system_content=system_content,
            tokenizer=cls._tokenizer,
        )

        selected_num_ctx = calculated_num_ctx if dynamic_enabled else manual_num_ctx

        think_enabled = bool(cfg.get("ollama_think", False))

        # Prepare options payload with validation defaults
        options = {
            "num_predict": max_new_tokens,
            "temperature": float(cfg["ollama_temperature"]),
            "num_ctx": selected_num_ctx,
            "repeat_penalty": float(cfg["ollama_repeat_penalty"]),
            "top_p": float(cfg["ollama_top_p"]),
            "top_k": int(cfg["ollama_top_k"]),
            "think": think_enabled,
        }
        if cfg["ollama_seed"] is not None and cfg["ollama_seed"] >= 0:
            options["seed"] = int(cfg["ollama_seed"])
        if cfg["ollama_stop"]:
            stop_seqs = [s.strip() for s in cfg["ollama_stop"].split(",") if s.strip()]
            if stop_seqs:
                options["stop"] = stop_seqs
        if cfg["ollama_num_thread"] is not None and cfg["ollama_num_thread"] > 0:
            options["num_thread"] = int(cfg["ollama_num_thread"])
        if cfg["ollama_num_gpu"] is not None and cfg["ollama_num_gpu"] >= 0:
            options["num_gpu"] = int(cfg["ollama_num_gpu"])

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": system_content,
                },
                {"role": "user", "content": prompt},
            ],
            "options": options,
            "think": think_enabled,
            "stream": False
        }
        if cfg["ollama_keep_alive"] is not None:
            try:
                payload["keep_alive"] = int(cfg["ollama_keep_alive"])
            except ValueError:
                payload["keep_alive"] = str(cfg["ollama_keep_alive"])

        base_url = server_url.rstrip("/")
        url = f"{base_url}/api/chat"

        logger.info(
            f"[QwenAI] Sending Ollama Request:\n"
            f"  - Model: {model}\n"
            f"  - Server URL: {server_url}\n"
            f"  - Dynamic Context Window: {'ENABLED (ON)' if dynamic_enabled else 'DISABLED (OFF)'}\n"
            f"  - Thinking Mode (think): {'ENABLED (ON)' if think_enabled else 'DISABLED (OFF)'}\n"
            f"  - Estimated Input Tokens: {est_input_tokens}\n"
            f"  - Max Output Tokens (num_predict): {max_new_tokens}\n"
            f"  - Selected num_ctx: {selected_num_ctx} (calculated: {calculated_num_ctx}, manual setting: {manual_num_ctx})\n"
            f"  - Prompt Chars: {len(prompt)}\n"
            f"  - Options: {options}\n"
            f"  - Keep Alive: {payload.get('keep_alive', 'N/A')}"
        )
        logger.debug(f"[QwenAI] Complete Ollama Request JSON Payload: {json.dumps(payload)}")

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode('utf-8'),
                headers={"Content-Type": "application/json"}
            )
            # Timeout set to 90 seconds for long generation tasks
            with urllib.request.urlopen(req) as response:
                status_code = response.status
                raw_body = response.read().decode('utf-8')
                logger.debug(f"[QwenAI] Raw Ollama response: {raw_body}")

                if status_code == 200:
                    resp_data = json.loads(raw_body)
                    content = resp_data.get("message", {}).get("content", "")
                    
                    total_dur = resp_data.get("total_duration")
                    load_dur = resp_data.get("load_duration")
                    prompt_eval_dur = resp_data.get("prompt_eval_duration")
                    eval_dur = resp_data.get("eval_duration")
                    
                    total_dur_str = f"{total_dur / 1e9:.2f}s" if total_dur is not None else "N/A"
                    load_dur_str = f"{load_dur / 1e9:.2f}s" if load_dur is not None else "N/A"
                    prompt_eval_dur_str = f"{prompt_eval_dur / 1e9:.2f}s" if prompt_eval_dur is not None else "N/A"
                    eval_dur_str = f"{eval_dur / 1e9:.2f}s" if eval_dur is not None else "N/A"

                    done_reason = resp_data.get("done_reason", "")
                    end_reason = "Unknown"
                    if done_reason == "stop":
                        end_reason = "EOS token or stop sequence reached"
                    elif done_reason == "length":
                        end_reason = "Reached maximum tokens (num_predict limit)"

                    resp_tokens = "N/A"
                    try:
                        if cls._tokenizer:
                            resp_tokens = len(cls._tokenizer.encode(content))
                    except Exception:
                        pass

                    logger.info(
                        f"[QwenAI] Received Ollama Response:\n"
                        f"  - HTTP Status: {status_code}\n"
                        f"  - Prompt Eval Tokens: {resp_data.get('prompt_eval_count', 'N/A')}\n"
                        f"  - Output Tokens: {resp_data.get('eval_count', 'N/A')}\n"
                        f"  - Total Duration: {total_dur_str}\n"
                        f"  - Load Duration: {load_dur_str}\n"
                        f"  - Prompt Eval Duration: {prompt_eval_dur_str}\n"
                        f"  - Output Eval Duration: {eval_dur_str}\n"
                        f"  - Done: {resp_data.get('done', 'N/A')}\n"
                        f"  - Done Reason: {done_reason} ({end_reason})\n"
                        f"  - Response Chars: {len(content)}\n"
                        f"  - Response Tokens (est): {resp_tokens}"
                    )
                    return strip_thinking(content)
                else:
                    logger.error(f"[QwenAI] Ollama request failed with status {status_code}")
        except Exception as e:
            logger.error(f"[QwenAI] Ollama chat inference failed for model '{model}' at {base_url}: {e}")
        return None

    def _infer(self, prompt: str, max_new_tokens: int = 5120, task_key: Optional[str] = None) -> str:
        """Run inference using Ollama offline fallback or local Qwen3 model."""
        cls = self.__class__
        
        cfg = cls._get_active_settings()
        use_ollama = cfg["use_ollama"]
        server_url = cfg["ollama_server_url"]
        port = cfg["ollama_port"]
        priority_list = cfg["ollama_model_priority"]

        # Override max_new_tokens dynamically from config if task_key matches
        if task_key:
            db_max_tokens = cfg.get(f"max_tokens_{task_key}")
            if db_max_tokens is not None and db_max_tokens > 0:
                max_new_tokens = db_max_tokens

        if use_ollama:
            # Invalidate cache if server_url, port, or priority list settings changed
            if (getattr(cls, "_cached_ollama_server_url", None) != server_url or 
                getattr(cls, "_cached_ollama_port", None) != port or 
                getattr(cls, "_cached_ollama_priority", None) != priority_list):
                cls._cached_ollama_model = None

            # Check cached model
            model_name = getattr(cls, "_cached_ollama_model", None)
            if model_name is not None:
                logger.info(f"[QwenAI] Attempting inference using cached Ollama model '{model_name}' at {server_url}...")
                res = cls._call_ollama(server_url, model_name, prompt, max_new_tokens)
                if res is not None:
                    logger.info(f"[QwenAI] Inference successfully completed using Ollama model '{model_name}'.")
                    return res
                else:
                    logger.warning(f"[QwenAI] Inference failed using cached Ollama model '{model_name}'. Invalidating cache.")
                    cls._cached_ollama_model = None

            # Perform detection if cache is empty
            logger.info(f"[QwenAI] Checking Ollama server at {server_url}...")
            model_name = cls._detect_ollama_model(server_url, priority_list)
            if model_name is not None:
                cls._cached_ollama_server_url = server_url
                cls._cached_ollama_port = port
                cls._cached_ollama_priority = priority_list
                cls._cached_ollama_model = model_name
                
                logger.info(f"[QwenAI] Ollama model detected. Running inference with '{model_name}' at {server_url}...")
                res = cls._call_ollama(server_url, model_name, prompt, max_new_tokens)
                if res is not None:
                    logger.info(f"[QwenAI] Inference successfully completed using Ollama model '{model_name}'.")
                    return res
                else:
                    logger.warning(f"[QwenAI] Inference failed using newly detected Ollama model '{model_name}'. Invalidating cache.")
                    cls._cached_ollama_model = None
        else:
            logger.info("[QwenAI] Ollama integration disabled. Skipping Ollama check entirely.")

        # Fall back to local Qwen model inference
        logger.info("[QwenAI] Ollama offline fallback unavailable/failed or disabled. Running inference via local Qwen model...")
        pipe = self._get_pipeline()
        if pipe is None:
            logger.warning("[QwenAI] Local Qwen model not available. Returning empty result.")
            return ""

        try:
            # Use chat template for Qwen3 Instruct
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an expert enterprise meeting analyst. "
                        "Follow instructions exactly. Preserve all technical terminology."
                    ),
                },
                {"role": "user", "content": prompt},
            ]

            # Apply Qwen3 chat template with thinking disabled
            tokenizer = self._tokenizer
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,  # Qwen3: disable chain-of-thought preamble
            )

            rep_penalty = cfg.get("ollama_repeat_penalty", 1.15)
            if rep_penalty is None or rep_penalty <= 0:
                rep_penalty = 1.15

            output = pipe(
                text,
                max_new_tokens=max_new_tokens,
                do_sample=False,          # deterministic - enterprise doc quality
                temperature=1.0,          # ignored when do_sample=False
                repetition_penalty=rep_penalty,  # reduce repetitive output
                return_full_text=False,
            )

            # Extract generated text
            result = output[0]["generated_text"]
            if isinstance(result, list):
                for msg in reversed(result):
                    if msg.get("role") == "assistant":
                        return msg.get("content", "").strip()
            return str(result).strip()

        except Exception as e:
            err_msg = str(e).lower()
            if "out of memory" in err_msg or "oom" in err_msg or "cuda memory" in err_msg:
                raise e
            logger.error(f"[QwenAI] Inference failed: {e}", exc_info=True)
            return ""

    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 1.0) -> str:
        """
        Public generate method - used by vocab_extractor for AI-assisted extraction.
        Delegates to _infer.
        """
        return self._infer(prompt, max_new_tokens=max_new_tokens, task_key="vocab_extractor")

    def _hierarchical_summarize(self, text: str) -> str:
        """
        Hierarchical chunked summarization for long transcripts.
        Splits → summarizes each chunk → merges → returns combined summary text.
        """
        word_count = len(text.split())
        if word_count <= 1500:
            return text  # Short enough to use directly

        logger.info(f"[QwenAI] Long transcript ({word_count} words) - applying hierarchical chunking")
        chunks = _chunk_text(text, max_words=700)
        logger.info(f"[QwenAI] Split into {len(chunks)} chunks")

        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            logger.info(f"[QwenAI] Summarizing chunk {i+1}/{len(chunks)}")
            summary = self._infer(
                _get_prompt("chunk_summary").format(chunk=chunk),
                max_new_tokens=256,
                task_key="chunk_summary",
            )
            if summary:
                chunk_summaries.append(summary)

        merged = "\n\n".join(chunk_summaries)
        logger.info(f"[QwenAI] Merged {len(chunk_summaries)} chunk summaries ({len(merged.split())} words)")
        return merged

    def build_context_summary(self, transcript: List[Dict]) -> str:
        """
        Build the compressed context summary for a transcript.

        This is the SINGLE entry point for hierarchical summarization.
        Call this once, store the result in the DB, and pass it as the
        `context` parameter to all downstream AI methods to avoid
        redundant LLM inference.

        For transcripts <= 1500 words, returns the raw formatted dialogue
        (no inference needed). For longer transcripts, runs chunked
        summarization (N inference calls, one per ~700-word chunk).
        """
        dialogue = _format_transcript(transcript)
        if not dialogue.strip():
            return ""
        return self._hierarchical_summarize(dialogue)

    def compress_agenda(self, text: str) -> str:
        """Compress raw agenda/objectives text into a structured numbered list."""
        if not text or not text.strip():
            return ""
        words = text.split()
        if len(words) > 2500:
            chunks = _chunk_text(text, max_words=1200)
            summaries = []
            for chunk in chunks:
                out = self._infer(_get_prompt("agenda_compress").format(text=chunk), max_new_tokens=2000, task_key="agenda_compress")
                if out:
                    summaries.append(out.strip())
            text = "\n".join(summaries)
        result = self._infer(_get_prompt("agenda_compress").format(text=text), max_new_tokens=2000, task_key="agenda_compress")
        return result.strip() if result else ""

    def compress_reference(self, text: str) -> str:
        """Compress reference/context document text into a concise knowledge summary."""
        if not text or not text.strip():
            return ""
        words = text.split()
        if len(words) > 1500:
            chunks = _chunk_text(text, max_words=1200)
            summaries = []
            for chunk in chunks:
                out = self._infer(_get_prompt("reference_compress").format(text=chunk), max_new_tokens=2000, task_key="reference_compress")
                if out:
                    summaries.append(out.strip())
            text = "\n".join(summaries)
        result = self._infer(_get_prompt("reference_compress").format(text=text), max_new_tokens=2000, task_key="reference_compress")
        return result.strip() if result else ""

    # ── AIProvider interface ──────────────────────────────────

    def generate_summary(self, transcript: List[Dict], context: Optional[str] = None) -> str:
        """Generate a short meeting summary (backward compat alias for short_summary)."""
        return self.generate_short_summary(transcript, context=context)

    def generate_short_summary(self, transcript: List[Dict], context: Optional[str] = None) -> str:
        """Generate a concise ~120-word professional summary. Uses pre-computed context if supplied."""
        if context is None:
            plain = _format_for_summary(transcript)
            if not plain.strip():
                return "No transcript available to summarize."
            context = self._hierarchical_summarize(plain)
        elif not context.strip():
            return "No transcript available to summarize."
        result = self._infer(
            _get_prompt("short_summary").format(transcript=context),
            max_new_tokens=120,
            task_key="short_summary",
        )
        return result or "Unable to generate summary."

    def generate_detailed_summary(self, transcript: List[Dict], context: Optional[str] = None) -> str:
        """Generate a comprehensive sectioned meeting report. Uses pre-computed context if supplied."""
        if context is None:
            dialogue = _format_transcript(transcript)
            if not dialogue.strip():
                return "No transcript available to summarize."
            context = self._hierarchical_summarize(dialogue)
        elif not context.strip():
            return "No transcript available to summarize."
        result = self._infer(
            _get_prompt("detailed_summary").format(transcript=context),
            max_new_tokens=3000,
            task_key="detailed_summary",
        )
        return result or "Unable to generate detailed summary."

    def generate_key_points(self, transcript: List[Dict], context: Optional[str] = None) -> List[str]:
        """Extract key discussion points. Uses pre-computed context if supplied."""
        if context is None:
            dialogue = _format_transcript(transcript)
            if not dialogue.strip():
                return []
            context = self._hierarchical_summarize(dialogue)
        elif not context.strip():
            return []
        raw = self._infer(
            _get_prompt("key_points").format(transcript=context),
            max_new_tokens=1028,
            task_key="key_points",
        )
        if not raw:
            return []

        # Parse bullet-format key points
        points = []
        current_topic = None
        for line in raw.split("\n"):
            line = line.strip()
            if not line:
                continue
            if line.startswith("•"):
                current_topic = line.lstrip("• ").rstrip(":")
            elif current_topic and not line.startswith("•"):
                points.append(f"{current_topic}: {line}")
                current_topic = None
            elif not current_topic and len(line) > 10:
                points.append(line)

        return points if points else _clean_list(raw)

    def generate_action_items(self, transcript: List[Dict], context: Optional[str] = None) -> List[str]:
        """Extract action items grouped by General / Speaker sections. Uses pre-computed context if supplied."""
        if context is None:
            dialogue = _format_transcript(transcript)
            if not dialogue.strip():
                return []
            context = self._hierarchical_summarize(dialogue)
        elif not context.strip():
            return []
        raw = self._infer(
            _get_prompt("action_items").format(transcript=context),
            max_new_tokens=1028,
            task_key="action_items",
        )
        if not raw or "none identified" in raw.lower():
            return []

        # Clean/split Qwen output
        lines = []
        for line in raw.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            # Remove leading numbers/bullets
            stripped = re.sub(r'^\d+\.\s*', '', stripped)
            lines.append(stripped)
        return lines if lines else []

    def generate_key_decisions(self, transcript: List[Dict], context: Optional[str] = None) -> List[str]:
        """Extract concrete decisions made during the meeting. Uses pre-computed context if supplied."""
        if context is None:
            dialogue = _format_transcript(transcript)
            if not dialogue.strip():
                return []
            context = self._hierarchical_summarize(dialogue)
        elif not context.strip():
            return []
        raw = self._infer(
            _get_prompt("key_decisions").format(transcript=context),
            max_new_tokens=1028,
            task_key="key_decisions",
        )
        if not raw or "none identified" in raw.lower():
            return []
        return _clean_list(raw)

    def generate_executive_summary(self, transcript: List[Dict], context: Optional[str] = None) -> dict:
        """Generate structured executive summary for PDF reports. Uses pre-computed context if supplied."""
        if context is None:
            dialogue = _format_transcript(transcript)
            if not dialogue.strip():
                return {
                    "purpose": "No transcript available.",
                    "discussion_points": [], "outcomes": [], "next_steps": [],
                }
            context = self._hierarchical_summarize(dialogue)
        elif not context.strip():
            return {
                "purpose": "No transcript available.",
                "discussion_points": [], "outcomes": [], "next_steps": [],
            }
        raw = self._infer(
            _get_prompt("executive_summary").format(transcript=context),
            max_new_tokens=700,
            task_key="executive_summary",
        )

        if not raw:
            return {
                "purpose": "Unable to generate executive summary.",
                "discussion_points": [],
                "outcomes": [],
                "next_steps": [],
            }

        def _extract_section(text: str, header: str) -> List[str]:
            lines = text.split("\n")
            collecting = False
            items = []
            headers = ["MEETING PURPOSE:", "MAIN DISCUSSION POINTS:", "OUTCOMES:", "NEXT STEPS:"]
            for line in lines:
                if header.upper() in line.upper():
                    collecting = True
                    continue
                if collecting:
                    stripped = line.strip()
                    if stripped.startswith("-"):
                        items.append(stripped[1:].strip())
                    elif stripped and any(h in stripped.upper() for h in headers) and stripped != header:
                        break
            return [i for i in items if i]

        def _extract_paragraph(text: str, header: str) -> str:
            lines = text.split("\n")
            collecting = False
            parts = []
            for line in lines:
                if header.upper() in line.upper():
                    collecting = True
                    continue
                if collecting:
                    stripped = line.strip()
                    if stripped and any(h in stripped.upper() for h in
                                        ["MAIN DISCUSSION POINTS:", "OUTCOMES:", "NEXT STEPS:"]):
                        break
                    if stripped and not stripped.startswith("-"):
                        parts.append(stripped)
            return " ".join(parts).strip()

        return {
            "purpose": _extract_paragraph(raw, "MEETING PURPOSE:") or "See transcript for details.",
            "discussion_points": _extract_section(raw, "MAIN DISCUSSION POINTS:"),
            "outcomes": _extract_section(raw, "OUTCOMES:"),
            "next_steps": _extract_section(raw, "NEXT STEPS:"),
        }

    def generate_speaker_summaries(self, transcript: List[Dict]) -> Dict[str, Dict]:
        """
        Generate per-speaker summary, key points, and action items.

        Groups transcript segments by speaker_label, then runs three separate
        Qwen3 inference calls per speaker. Returns:
        {
            "SpeakerName": {
                "summary": str,
                "key_points": [str, ...],
                "action_items": [str, ...],
            },
            ...
        }
        """
        # Group segments by speaker
        speaker_segments: Dict[str, List[Dict]] = {}
        for seg in transcript:
            label = seg.get("speaker_label", "Unknown")
            if label in ("Unknown",) or seg.get("is_overlap"):
                continue
            speaker_segments.setdefault(label, []).append(seg)

        if not speaker_segments:
            logger.info("[QwenAI] No named speakers found - skipping per-speaker summaries")
            return {}

        logger.info(f"[QwenAI] Generating per-speaker summaries for: {list(speaker_segments.keys())}")
        results: Dict[str, Dict] = {}

        for speaker, segs in speaker_segments.items():
            # Build plain-text transcript for this speaker
            lines = [seg.get("text", "").strip() for seg in segs if seg.get("text", "").strip()]
            if not lines:
                continue
            speaker_text = "\n".join(lines)
            # Truncate to avoid context overflow
            if len(speaker_text.split()) > 1200:
                # Summarize speaker chunks first
                speaker_text = self._hierarchical_summarize(speaker_text)
            speaker_text = speaker_text[:3000]

            logger.info(f"[QwenAI] Summarizing speaker: {speaker} ({len(lines)} segments)")

            # Summary
            summary_raw = self._infer(
                _get_prompt("speaker_summary").format(speaker=speaker, transcript=speaker_text),
                max_new_tokens=200,
                task_key="speaker_summary",
            )

            # Key points
            kp_raw = self._infer(
                _get_prompt("speaker_key_points").format(speaker=speaker, transcript=speaker_text),
                max_new_tokens=350,
                task_key="speaker_key_points",
            )
            key_points = []
            if kp_raw:
                topic = None
                for line in kp_raw.split("\n"):
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("•"):
                        topic = line.lstrip("• ").rstrip(":")
                    elif topic and not line.startswith("•"):
                        key_points.append(f"{topic}: {line}")
                        topic = None
                    elif len(line) > 8:
                        key_points.append(line)
                if not key_points:
                    key_points = _clean_list(kp_raw)[:6]

            # Action items
            ai_raw = self._infer(
                _get_prompt("speaker_action_items").format(speaker=speaker, transcript=speaker_text),
                max_new_tokens=250,
                task_key="speaker_action_items",
            )
            action_items = [] if (not ai_raw or "none identified" in ai_raw.lower()) else _clean_list(ai_raw)

            results[speaker] = {
                "summary": summary_raw or "",
                "key_points": key_points[:6],
                "action_items": action_items[:8],
            }
            logger.info(f"[QwenAI] Speaker '{speaker}' done: {len(key_points)} kp, {len(action_items)} ai")

        return results

    # ── RAG Pipeline Methods (Raw MoM only) ──────────────────────────────────
    # These methods are ONLY used by the Raw MoM pipeline.
    # They do NOT affect the existing generate_mom() method.

    def generate_agenda_from_summary(self, summary: str) -> List[Dict]:
        """
        Generate structured agenda items from a transcription summary.
        """
        if not summary or not summary.strip():
            return []

        raw = self._infer(
            _get_prompt("agenda_from_summary").format(summary=summary.strip()),
            max_new_tokens=1024,
            task_key="agenda_from_summary",
        )
        if not raw:
            return []

        # Strip markdown fences
        raw = raw.strip()
        if raw.startswith("```json"):
            raw = raw[7:]
        elif raw.startswith("```"):
            raw = raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

        try:
            items = json.loads(raw)
            if not isinstance(items, list):
                raise ValueError("Expected JSON array")
            result = []
            for item in items:
                if isinstance(item, dict) and item.get("topic"):
                    result.append({
                        "topic": str(item["topic"]).strip(),
                        "speaker": str(item["speaker"]).strip() if item.get("speaker") else None,
                    })
            logger.info(f"[QwenAI] Generated {len(result)} agenda items from summary")
            return result
        except Exception as e:
            # Attempt regex extraction
            import re
            match = re.search(r'\[.*\]', raw, re.DOTALL)
            if match:
                try:
                    items = json.loads(match.group())
                    if isinstance(items, list):
                        return [
                            {
                                "topic": str(i.get("topic", "")).strip(),
                                "speaker": str(i.get("speaker", "")).strip() or None,
                            }
                            for i in items if isinstance(i, dict) and i.get("topic")
                        ]
                except Exception:
                    pass
            logger.warning(f"[QwenAI] Failed to parse generated agenda JSON: {e}. Raw: {raw}")
            return []

    def parse_agenda_items(self, agenda_text: str) -> List[Dict]:
        """
        Parse an agenda document into a list of {topic, speaker} dicts.

        The topic is copied VERBATIM from the agenda - no rewriting or shortening.
        The speaker is the presenter/owner if explicitly mentioned, else None.

        Returns
        -------
        List of dicts: [{"topic": str, "speaker": str|None}]
        Empty list if parsing fails or no items found.
        """
        if not agenda_text or not agenda_text.strip():
            return []

        raw = self._infer(
            _get_prompt("agenda_compress").format(text=agenda_text.strip()),
            max_new_tokens=1024,
            task_key="agenda_compress",
        )
        if not raw:
            return []

        # Strip markdown fences
        raw = raw.strip()
        if raw.startswith("```json"):
            raw = raw[7:]
        elif raw.startswith("```"):
            raw = raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

        try:
            items = json.loads(raw)
            if not isinstance(items, list):
                raise ValueError("Expected JSON array")
            result = []
            for item in items:
                if isinstance(item, dict) and item.get("topic"):
                    result.append({
                        "topic": str(item["topic"]).strip(),
                        "speaker": str(item["speaker"]).strip() if item.get("speaker") else None,
                    })
            logger.info(f"[QwenAI] Parsed {len(result)} agenda items")
            return result
        except Exception as e:
            # Attempt to extract JSON array from the raw output
            match = re.search(r'\[.*\]', raw, re.DOTALL)
            if match:
                try:
                    items = json.loads(match.group())
                    if isinstance(items, list):
                        return [
                            {
                                "topic": str(i.get("topic", "")).strip(),
                                "speaker": str(i.get("speaker", "")).strip() or None,
                            }
                            for i in items
                            if isinstance(i, dict) and i.get("topic")
                        ]
                except Exception:
                    pass
            logger.warning(f"[QwenAI] Agenda parsing failed: {e} - raw: {raw[:200]}")
            return []

    def parse_agenda_items_with_context(
        self,
        agenda_text: str,
        global_context: str,
    ) -> List[Dict]:
        """
        Parse an agenda document into a list of {topic, speaker} dicts, using
        Global Context as supplementary understanding material.

        The agenda document (Section 1) is the SOLE authority for agenda generation.
        The global context (Section 2) may only improve the model's understanding
        of terminology, project names, and organizational background - it must NOT
        be used to create, modify, add, or remove agenda items.

        Falls back to parse_agenda_items() if global_context is empty.

        Parameters
        ----------
        agenda_text     : Raw text of the uploaded agenda document.
        global_context  : Retrieved global context chunks as a formatted string.

        Returns
        -------
        List of dicts: [{"topic": str, "speaker": str|None}]
        Empty list if parsing fails or no items found.
        """
        if not agenda_text or not agenda_text.strip():
            return []

        # Fall back to standard parsing if no context is available
        if not global_context or not global_context.strip():
            return self.parse_agenda_items(agenda_text)

        raw = self._infer(
            _get_prompt("agenda_compress_with_context").format(
                text=agenda_text.strip(),
                global_context=global_context.strip(),
            ),
            max_new_tokens=1024,
            task_key="agenda_compress",
        )
        if not raw:
            return []

        # Strip markdown fences
        raw = raw.strip()
        if raw.startswith("```json"):
            raw = raw[7:]
        elif raw.startswith("```"):
            raw = raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

        try:
            items = json.loads(raw)
            if not isinstance(items, list):
                raise ValueError("Expected JSON array")
            result = []
            for item in items:
                if isinstance(item, dict) and item.get("topic"):
                    result.append({
                        "topic": str(item["topic"]).strip(),
                        "speaker": str(item["speaker"]).strip() if item.get("speaker") else None,
                    })
            logger.info(f"[QwenAI] Parsed {len(result)} agenda items (with global context)")
            return result
        except Exception as e:
            # Attempt to extract JSON array from the raw output
            match = re.search(r'\[.*\]', raw, re.DOTALL)
            if match:
                try:
                    items = json.loads(match.group())
                    if isinstance(items, list):
                        return [
                            {
                                "topic": str(i.get("topic", "")).strip(),
                                "speaker": str(i.get("speaker", "")).strip() or None,
                            }
                            for i in items
                            if isinstance(i, dict) and i.get("topic")
                        ]
                except Exception:
                    pass
            logger.warning(
                f"[QwenAI] Agenda parsing with context failed: {e} - "
                f"falling back to standard parsing"
            )
            # Final fallback: parse without context
            return self.parse_agenda_items(agenda_text)

    def extract_rom_discussion_points(
        self,
        window_text: str,
        previous_points_json: Optional[str] = None,
        video_context: str = "",
        skip_action_items: bool = False,
    ) -> Dict:
        """Extract structured discussion points from a transcript window for ROM pipeline.
        
        Args:
            window_text:           Formatted transcript window text (speaker + timestamps).
            previous_points_json:  JSON string of recent discussion points for continuity.
            video_context:         Optional OCR text from presentation slides/screen shares
                                   overlapping this window's time range. Empty string for
                                   audio-only recordings.
            skip_action_items:     When True, uses the no-actions prompt variant
                                   (ROM_DISCUSSION_NO_ACTION_ITEMS_PROMPT) which instructs
                                   the LLM to leave action_items empty. Used when
                                   separate_action_extraction is enabled so that action
                                   items are handled by a dedicated parallel call instead.
        """
        prev_context = ""
        if previous_points_json:
            prev_context = f"PREVIOUS DISCUSSION POINTS (for continuity - do not repeat, only add new points):\n{previous_points_json}"

        # Build video context section - injected only when OCR text is present
        if video_context and video_context.strip():
            video_context_section = (
                "VIDEO OCR TRANSCRIPT (Screen/Slide Frame Extraction - Use With Intelligence):\n"
                "This text was automatically extracted from video frames using OCR (Optical Character Recognition).\n"
                "OCR extraction is imperfect: it may contain recognition errors, incomplete words, fragmented\n"
                "sentences, repeated headers, or noisy text. Do NOT rely on exact wording.\n"
                "Use the meeting audio transcript as the PRIMARY source of truth. Use OCR content only as\n"
                "supplementary evidence and interpret it intelligently using the meeting context.\n"
                "\n"
                "WHEN TO CREATE AN ADDITIONAL POINT FROM OCR:\n"
                "Create an additional ROM discussion point from OCR content ONLY when it contains meaningful\n"
                "factual information that was NOT captured in the audio transcript, such as:\n"
                "  - Decisions or conclusions shown on slides\n"
                "  - Specific dates, deadlines, or milestones\n"
                "  - Action items or assignments displayed on screen\n"
                "  - Important numbers, figures, budgets, or measurements\n"
                "  - Announcements or key statements shown in presentation slides\n"
                "  - Critical information displayed on shared screens\n"
                "\n"
                "DO NOT create points from: decorative text, repeated slide headers/footers, company logos,\n"
                "page numbers, navigation menus, or content that is insignificant or redundant with spoken content.\n"
                "\n"
                "OCR-ONLY POINTS - SPEAKER LABEL:\n"
                "If you create a discussion point that originates ONLY from the OCR transcript (not from the audio),\n"
                "set the speakers field to [\"For Information\"] for that point. Do NOT add any other labels or categories.\n"
                "\n"
                "NEVER output OCR timestamps (e.g. 00:15 → 00:45) as dates in the dates array.\n"
                f"\n{video_context.strip()}"
            )
    def _parse_json_dict(self, raw: str) -> Optional[Dict]:
        """Attempt to parse raw LLM response text into a Python dict."""
        import ast as _ast
        if not raw or not raw.strip():
            return None

        raw_clean = raw.strip()
        json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', raw_clean, re.DOTALL)
        if json_match:
            raw_clean = json_match.group(1).strip()
        else:
            brace_match = re.search(r'\{.*\}', raw_clean, re.DOTALL)
            if brace_match:
                raw_clean = brace_match.group().strip()

        if "{{" in raw_clean:
            raw_clean = raw_clean.replace("{{", "{").replace("}}", "}")

        try:
            d = json.loads(raw_clean)
            if isinstance(d, dict):
                return d
        except Exception:
            pass

        try:
            d = _ast.literal_eval(raw_clean)
            if isinstance(d, dict):
                return json.loads(json.dumps(d))
        except Exception:
            pass

        return None

    def repair_stage1_json(self, raw_invalid_json: str, orig_max_tokens: int) -> Optional[Dict]:
        """Attempt to repair malformed or truncated Stage 1 JSON output using a dedicated LLM call.

        Uses STAGE1_JSON_REPAIR_PROMPT with max_new_tokens set to orig_max_tokens + 500
        to ensure sufficient budget to close open quotes/braces.

        Args:
            raw_invalid_json: Raw text response from the previous LLM extraction call.
            orig_max_tokens:  The max_new_tokens value used in the original call.

        Returns:
            Parsed dictionary if repair succeeds, or None if repair fails.
        """
        cfg = self._get_active_settings()
        repair_max_tokens = cfg.get("max_tokens_stage1_json_repair") or (orig_max_tokens + 500)
        logger.info(
            f"[QwenAI] Invoking Stage 1 LLM JSON Repair step "
            f"(orig max_tokens={orig_max_tokens} -> repair max_tokens={repair_max_tokens}, input chars={len(raw_invalid_json)})"
        )
        logger.warning(
            f"[QwenAI] Stage 1 invalid LLM response being sent to JSON Repair ({len(raw_invalid_json)} chars):\n"
            f"--- START RAW INVALID LLM RESPONSE ---\n{raw_invalid_json}\n--- END RAW INVALID LLM RESPONSE ---"
        )

        prompt = _get_prompt("stage1_json_repair").replace("{invalid_json}", raw_invalid_json)
        raw = self._infer(prompt, max_new_tokens=repair_max_tokens, task_key="stage1_json_repair")

        if not raw or not raw.strip():
            logger.warning("[QwenAI] repair_stage1_json received empty response from LLM.")
            return None

        parsed = self._parse_json_dict(raw)
        if isinstance(parsed, dict):
            logger.info("[QwenAI] Stage 1 JSON Repair succeeded! Repaired JSON successfully parsed.")
            return parsed

        logger.error(
            f"[QwenAI] Stage 1 JSON Repair failed — repaired output could not be parsed as valid JSON.\n"
            f"--- START RAW REPAIR LLM RESPONSE ({len(raw)} chars) ---\n{raw}\n--- END RAW REPAIR LLM RESPONSE ---"
        )
        return None

    def extract_rom_discussion_points(
        self,
        window_text: str,
        previous_points_json: Optional[str] = None,
        video_context: str = "",
        skip_action_items: bool = False,
    ) -> Dict:
        """Stage 1: Extract discussion points (and optionally action items) from a transcript window.

        Args:
            window_text:          Formatted transcript window text (speaker + timestamps).
            previous_points_json: JSON string of recent discussion points for context.
            video_context:        Optional OCR text from presentation slides.
            skip_action_items:    If True, uses ROM_DISCUSSION_NO_ACTION_ITEMS_PROMPT.

        Returns:
            {"discussion_points": [...]} or {"discussion_points": [], "parse_error": True} on failure.
        """
        cfg = self._get_active_settings()
        prompt_key = "rom_discussion_no_actions" if skip_action_items else "rom_discussion"
        orig_max_tokens = cfg.get(f"max_tokens_{prompt_key}") or cfg.get("max_tokens_rom_discussion") or 4048

        prev_context = ""
        if previous_points_json:
            prev_context = f"PREVIOUS DISCUSSION POINTS (for continuity context):\n{previous_points_json}"

        if video_context and video_context.strip():
            video_context_section = (
                "VIDEO OCR TRANSCRIPT (supplementary — audio transcript is primary):\n"
                "  - Key text, bullet points, headers, slide titles, diagrams from presentation video\n"
                "  - Speaker presentation slides\n"
                "  - Text from shared screens, whiteboards, or documents shown on screen\n"
                "  - Action items or assignments displayed on screen\n"
                "  - Important numbers, figures, budgets, or measurements\n"
                "  - Announcements or key statements shown in presentation slides\n"
                "  - Critical information displayed on shared screens\n"
                "\n"
                "DO NOT create points from: decorative text, repeated slide headers/footers, company logos,\n"
                "page numbers, navigation menus, or content that is insignificant or redundant with spoken content.\n"
                "\n"
                "OCR-ONLY POINTS - SPEAKER LABEL:\n"
                "If you create a discussion point that originates ONLY from the OCR transcript (not from the audio),\n"
                "set the speakers field to [\"For Information\"] for that point. Do NOT add any other labels or categories.\n"
                "\n"
                "NEVER output OCR timestamps (e.g. 00:15 → 00:45) as dates in the dates array.\n"
                f"\n{video_context.strip()}"
            )
        else:
            video_context_section = ""

        # Select prompt: use the no-actions variant when separate action extraction is active
        prompt = (
            _get_prompt(prompt_key)
            .replace("{previous_context}", prev_context)
            .replace("{video_context_section}", video_context_section)
            .replace("{window_text}", window_text)
        )
        raw = self._infer(prompt, max_new_tokens=orig_max_tokens, task_key=prompt_key)

        if not raw or not raw.strip():
            logger.warning("[QwenAI] extract_rom_discussion_points received empty response from LLM.")
            return {"discussion_points": [], "parse_error": True, "error_reason": "LLM returned empty response"}

        # Attempt normal parsing
        data = self._parse_json_dict(raw)

        # If parsing fails, invoke JSON Repair
        if data is None:
            logger.warning(
                f"[QwenAI] Stage 1 discussion extraction JSON parse failed on initial call. Prompt key: '{prompt_key}'.\n"
                f"--- START RAW INITIAL EXTRACTION LLM RESPONSE ({len(raw)} chars) ---\n{raw}\n--- END RAW INITIAL EXTRACTION LLM RESPONSE ---\n"
                f"Invoking Stage 1 JSON Repair step..."
            )
            data = self.repair_stage1_json(raw, orig_max_tokens=orig_max_tokens)

        if isinstance(data, dict):
            data.setdefault("discussion_points", [])
            return data

        logger.error(
            f"[QwenAI] extract_rom_discussion_points output is not a JSON object after repair.\n"
            f"--- START FINAL RAW LLM RESPONSE ({len(raw)} chars) ---\n{raw}\n--- END FINAL RAW LLM RESPONSE ---"
        )
        return {"discussion_points": [], "parse_error": True, "error_reason": "Output was not a JSON object"}

    def extract_rom_action_points(
        self,
        window_text: str,
        previous_points_json: Optional[str] = None,
        video_context: str = "",
    ) -> Dict:
        """Stage 1 (separate action extraction): Extract action items only from a transcript window.

        Used when rom_separate_action_extraction is enabled. Runs in parallel with
        extract_rom_discussion_points (no-actions variant) and returns only action_items.

        Args:
            window_text:          Formatted transcript window text (speaker + timestamps).
            previous_points_json: JSON string of recent discussion points for context.
            video_context:        Optional OCR text from presentation slides.

        Returns:
            {"action_items": [...]} or {"action_items": [], "parse_error": True} on failure.
        """
        cfg = self._get_active_settings()
        orig_max_tokens = cfg.get("max_tokens_rom_action_extraction") or 2048

        prev_context = ""
        if previous_points_json:
            prev_context = f"PREVIOUS DISCUSSION POINTS (for continuity context):\n{previous_points_json}"

        if video_context and video_context.strip():
            video_context_section = (
                "VIDEO OCR TRANSCRIPT (supplementary — audio transcript is primary):\n"
                f"{video_context.strip()}"
            )
        else:
            video_context_section = ""

        prompt = (
            _get_prompt("rom_action_extraction")
            .replace("{previous_context}", prev_context)
            .replace("{video_context_section}", video_context_section)
            .replace("{window_text}", window_text)
        )
        raw = self._infer(prompt, max_new_tokens=orig_max_tokens, task_key="rom_action_extraction")

        if not raw or not raw.strip():
            logger.warning("[QwenAI] extract_rom_action_points received empty response from LLM.")
            return {"action_items": [], "parse_error": True, "error_reason": "LLM returned empty response"}

        # Attempt normal parsing
        data = self._parse_json_dict(raw)

        # If parsing fails, invoke JSON Repair
        if data is None:
            logger.warning(
                f"[QwenAI] Stage 1 action extraction JSON parse failed on initial call.\n"
                f"--- START RAW INITIAL ACTION EXTRACTION LLM RESPONSE ({len(raw)} chars) ---\n{raw}\n--- END RAW INITIAL ACTION EXTRACTION LLM RESPONSE ---\n"
                f"Invoking Stage 1 JSON Repair step..."
            )
            data = self.repair_stage1_json(raw, orig_max_tokens=orig_max_tokens)

        if isinstance(data, dict):
            data.setdefault("action_items", [])
            return data

        logger.error(
            f"[QwenAI] extract_rom_action_points output is not a JSON object after repair.\n"
            f"--- START FINAL RAW LLM RESPONSE ({len(raw)} chars) ---\n{raw}\n--- END FINAL RAW LLM RESPONSE ---"
        )
        return {"action_items": [], "parse_error": True, "error_reason": "Output was not a JSON object"}

    def regenerate_mom_action_points(self, window_text: str) -> Dict:
        """Extract action items from a transcript window for MoM Action Point regeneration.

        Uses the dedicated MOM_REGENERATE_ACTION_POINTS_PROMPT.  Returns only the
        action_items list; no video context or previous-point context is needed since this
        is a standalone extraction call for MoM regeneration.

        Args:
            window_text: Formatted transcript window (speaker labels + timestamps).

        Returns:
            {"action_items": [...]} or {"action_items": [], "parse_error": True} on failure.
        """
        import ast as _ast

        prompt = _get_prompt("mom_action_regen").replace("{window_text}", window_text)
        raw = self._infer(prompt, max_new_tokens=4048, task_key="mom_action_regen")

        if not raw or not raw.strip():
            logger.warning("[QwenAI] regenerate_mom_action_points received empty response from LLM.")
            return {"action_items": [], "parse_error": True, "error_reason": "LLM returned empty response"}

        # ── Always log the raw LLM output for debugging ──────────────────────
        logger.info(
            f"[QwenAI] regenerate_mom_action_points raw LLM response ({len(raw)} chars):\n{raw}"
        )

        raw_clean = raw.strip()

        # Stage 1: strip markdown code fence  ```json … ``` or ``` … ```
        json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', raw_clean, re.DOTALL)
        if json_match:
            raw_clean = json_match.group(1).strip()
            logger.info(f"[QwenAI] regenerate_mom_action_points: stripped markdown fence, candidate ({len(raw_clean)} chars)")
        else:
            # Stage 2: extract the first {...} block from the full response
            brace_match = re.search(r'\{.*\}', raw_clean, re.DOTALL)
            if brace_match:
                raw_clean = brace_match.group().strip()
                logger.info(f"[QwenAI] regenerate_mom_action_points: extracted brace block ({len(raw_clean)} chars)")

        # Stage 2b: Sanitize double-braces if LLM echoed {{ ... }}
        if raw_clean.startswith("{{") or "{{" in raw_clean:
            raw_clean = raw_clean.replace("{{", "{").replace("}}", "}")
            logger.info(f"[QwenAI] regenerate_mom_action_points: normalized double braces {{ ... }} to {{ ... }}")

        def _try_parse(text: str):
            """Attempt JSON parse, then ast.literal_eval as fallback for single-quoted output."""
            # Attempt 1: standard json.loads
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass
            # Attempt 2: ast.literal_eval handles Python-style {'key': 'value'} dicts
            try:
                result = _ast.literal_eval(text)
                if isinstance(result, dict):
                    # Re-encode to proper JSON and back to normalise types
                    return json.loads(json.dumps(result))
            except Exception:
                pass
            # Attempt 3: brace-extract then standard parse (catches leading/trailing prose)
            brace = re.search(r'\{.*\}', text, re.DOTALL)
            if brace:
                try:
                    return json.loads(brace.group())
                except Exception:
                    pass
                try:
                    result = _ast.literal_eval(brace.group())
                    if isinstance(result, dict):
                        return json.loads(json.dumps(result))
                except Exception:
                    pass
            return None

        data = _try_parse(raw_clean)

        if data is None:
            logger.warning(
                f"[QwenAI] regenerate_mom_action_points: all JSON parse attempts failed.\n"
                f"Candidate text was:\n{raw_clean}"
            )
            return {"action_items": [], "parse_error": True, "error_reason": "All JSON parse attempts failed"}

        if isinstance(data, dict):
            data.setdefault("action_items", [])
            logger.info(f"[QwenAI] regenerate_mom_action_points: parsed OK, {len(data['action_items'])} action item(s)")
            return data

        logger.warning(f"[QwenAI] regenerate_mom_action_points output is not a JSON object. Got type: {type(data)}")
        return {"action_items": [], "parse_error": True, "error_reason": "Output was not a JSON object"}

    def deduplicate_action_points(self, action_items: List[Dict]) -> List[Dict]:
        """Perform a final global deduplication pass over all extracted action items.

        Uses MOM_DEDUPLICATE_ACTION_POINTS_PROMPT to identify duplicate or highly similar
        action points across all windows/agendas, merge complementary details, and remove
        redundant variations while preserving unique details.

        Args:
            action_items: List of raw action item dicts [{assigner, assignee, task, deadline}, ...]

        Returns:
            Deduplicated list of action item dicts.
        """
        import ast as _ast

        if not action_items or len(action_items) <= 1:
            return action_items

        items_json = json.dumps(action_items, indent=2)
        prompt = _get_prompt("mom_action_dedup").replace("{action_items_json}", items_json)
        raw = self._infer(prompt, max_new_tokens=4048, task_key="mom_action_dedup")

        if not raw or not raw.strip():
            logger.warning("[QwenAI] deduplicate_action_points received empty response from LLM.")
            return action_items

        logger.info(
            f"[QwenAI] deduplicate_action_points raw LLM response ({len(raw)} chars):\n{raw}"
        )

        raw_clean = raw.strip()

        # Stage 1: strip markdown code fence ```json … ``` or ``` … ```
        json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', raw_clean, re.DOTALL)
        if json_match:
            raw_clean = json_match.group(1).strip()
            logger.info(f"[QwenAI] deduplicate_action_points: stripped markdown fence ({len(raw_clean)} chars)")
        else:
            brace_match = re.search(r'\{.*\}', raw_clean, re.DOTALL)
            if brace_match:
                raw_clean = brace_match.group().strip()
                logger.info(f"[QwenAI] deduplicate_action_points: extracted brace block ({len(raw_clean)} chars)")

        if raw_clean.startswith("{{") or "{{" in raw_clean:
            raw_clean = raw_clean.replace("{{", "{").replace("}}", "}")

        def _try_parse(text: str):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass
            try:
                result = _ast.literal_eval(text)
                if isinstance(result, dict):
                    return json.loads(json.dumps(result))
            except Exception:
                pass
            brace = re.search(r'\{.*\}', text, re.DOTALL)
            if brace:
                try:
                    return json.loads(brace.group())
                except Exception:
                    pass
                try:
                    result = _ast.literal_eval(brace.group())
                    if isinstance(result, dict):
                        return json.loads(json.dumps(result))
                except Exception:
                    pass
            return None

        data = _try_parse(raw_clean)
        if isinstance(data, dict) and isinstance(data.get("action_items"), list):
            deduped = data["action_items"]
            logger.info(f"[QwenAI] deduplicate_action_points: parsed OK, {len(action_items)} -> {len(deduped)} item(s)")
            return deduped

        logger.warning("[QwenAI] deduplicate_action_points output invalid or parse failed. Returning original items.")
        return action_items

    def polish_rom_points(
        self,
        original_points_json: str,
        retrieved_context: str,
    ) -> Dict:
        """Stage 2: Polish and enrich extracted ROM points with retrieved RAG context for ROM pipeline."""
        prompt = _get_prompt("rom_polish").replace("{original_points}", original_points_json).replace("{retrieved_context}", retrieved_context)
        raw = self._infer(prompt, max_new_tokens=4096, task_key="rom_polish")
        
        if not raw:
            return {"polished_points": []}
        
        raw = raw.strip()
        json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', raw, re.DOTALL)
        if json_match:
            raw = json_match.group(1).strip()
        
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except Exception:
                    return {"polished_points": []}
            else:
                return {"polished_points": []}
        
        return data if isinstance(data, dict) else {"polished_points": []}
    def enhance_rom_discussion_window(
        self,
        window_json: str,
        meeting_context: str,
        global_context: str,
    ) -> Dict:
        """
        Enhance a discussion window using independently retrieved Meeting and Global Context.

        Returns a dict:
          {
            "enhanced_points": [
              {
                "original_point_ids": [...],
                "enhanced_text": "...",
                "speakers": [...],
                "action_owner": null,
                "decisions": [...],
                "technical_terms": [...],
                "dates": [...],
                "numbers": [...],
                "references": [...],
                "action_items": [...],
                "context_usage_report": {
                  "verified": bool,
                  "technical_details_added": bool,
                  "abbreviations_expanded": bool,
                  "references_added": bool,
                  "terminology_clarified": bool,
                  "no_useful_context": bool,
                  "meeting_context_docs": [...],
                  "global_context_docs": [...],
                }
              }
            ]
          }
        """
        meeting_section = (
            f"[Meeting Context]\n{meeting_context.strip()}"
            if meeting_context and meeting_context.strip()
            else "[Meeting Context]\nNo meeting context retrieved."
        )
        global_section = (
            f"[Global Context]\n{global_context.strip()}"
            if global_context and global_context.strip()
            else "[Global Context]\nNo global context retrieved."
        )

        prompt = (
            _get_prompt("rom_enhance_window")
            .replace("{window_json}", window_json)
            .replace("{meeting_context}", meeting_section)
            .replace("{global_context}", global_section)
        )

        raw = self._infer(prompt, max_new_tokens=4096, task_key="rom_enhance_window")

        if not raw:
            return {"enhanced_points": []}

        raw = raw.strip()
        json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', raw, re.DOTALL)
        if json_match:
            raw = json_match.group(1).strip()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except Exception:
                    return {"enhanced_points": []}
            else:
                return {"enhanced_points": []}

        return data if isinstance(data, dict) else {"enhanced_points": []}

    def deduplicate_rom_points(self, candidate_points_json: str) -> Dict:
        """Evaluate high-similarity points at the end of Stage 2 (duplicate/complementary/different)."""
        prompt = _get_prompt("rom_deduplicate").replace("{candidate_points}", candidate_points_json)
        raw = self._infer(prompt, max_new_tokens=2048, task_key="rom_deduplicate")

        if not raw:
            return {"group_classification": "different", "result_points": []}

        raw = raw.strip()
        json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', raw, re.DOTALL)
        if json_match:
            raw = json_match.group(1).strip()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except Exception:
                    return {"group_classification": "different", "result_points": []}
            else:
                return {"group_classification": "different", "result_points": []}

        return data if isinstance(data, dict) else {"group_classification": "different", "result_points": []}

    def generate_rom_agendas(self, agenda_text: Optional[str], context: str, previous_mom: Optional[str] = None) -> Dict:
        """Generate enriched agenda representations for ROM pipeline."""
        agenda_input = ""
        if agenda_text:
            agenda_input = f"UPLOADED AGENDA:\n{agenda_text}"
        else:
            agenda_input = "No agenda file provided. Generate appropriate agendas based on the context."
        
        context_section = f"SUPPORTING CONTEXT:\n{context}" if context else ""
        prev_mom_section = f"PREVIOUS MEETING MOM:\n{previous_mom}" if previous_mom else ""
        
        prompt = _get_prompt("rom_agenda").replace("{agenda_input}", agenda_input).replace("{context}", context_section).replace("{previous_mom}", prev_mom_section)
        raw = self._infer(prompt, max_new_tokens=2048, task_key="rom_agenda")
        
        if not raw:
            return {"agendas": []}
        
        raw = raw.strip()
        json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', raw, re.DOTALL)
        if json_match:
            raw = json_match.group(1).strip()
        
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            match_obj = re.search(r'\{.*\}', raw, re.DOTALL)
            match_arr = re.search(r'\[.*\]', raw, re.DOTALL)
            data = None
            if match_obj:
                try:
                    data = json.loads(match_obj.group())
                except Exception:
                    pass
            if not data and match_arr:
                try:
                    data = json.loads(match_arr.group())
                except Exception:
                    pass
            if not data:
                return {"agendas": []}
        
        if isinstance(data, list):
            data = {"agendas": data}
        elif isinstance(data, dict):
            if "agendas" not in data and "agenda" in data:
                data["agendas"] = data["agenda"]
        
        return data if isinstance(data, dict) else {"agendas": []}

    def expand_agendas_with_previous_mom(self, agenda_list_json: str, previous_mom_combined: str) -> Dict:
        """Phase 1: Expand each agenda item using Previous Meeting MoM documents.
        Only called when previous MoM documents have been uploaded.
        Returns per-agenda expansion with previous discussion, decisions, action items, pending work, follow-ups.
        """
        prompt = (
            _get_prompt("rom_mom_expansion")
            .replace("{agenda_list}", agenda_list_json)
            .replace("{previous_mom_docs}", previous_mom_combined)
        )
        raw = self._infer(prompt, max_new_tokens=3000, task_key="rom_mom_expansion")

        if not raw:
            return {"expansions": []}

        raw = raw.strip()
        json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', raw, re.DOTALL)
        if json_match:
            raw = json_match.group(1).strip()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except Exception:
                    return {"expansions": []}
            else:
                return {"expansions": []}

        return data if isinstance(data, dict) else {"expansions": []}

    def assign_agenda_batch(self, batch_json: str, agenda_reference_json: str) -> Dict:
        """Phase 4: Batch agenda assignment - LLM assigns each point to one of its Top-3 candidates.
        Returns assignments list with point_id, assigned_agenda_id, confidence, reason.
        """
        prompt = (
            _get_prompt("rom_agenda_assign_batch")
            .replace("{agenda_reference}", agenda_reference_json)
            .replace("{batch_json}", batch_json)
        )
        raw = self._infer(prompt, max_new_tokens=4096, task_key="rom_agenda_assign_batch")

        if not raw:
            return {"assignments": []}

        raw = raw.strip()
        json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', raw, re.DOTALL)
        if json_match:
            raw = json_match.group(1).strip()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except Exception:
                    return {"assignments": []}
            else:
                return {"assignments": []}

        return data if isinstance(data, dict) else {"assignments": []}

    def generate_agenda_document_points(self, agenda_title: str, agenda_description: str, document_text: str) -> Dict:
        """Extract 2-5 factual agenda-specific points from a supporting document uploaded for an agenda item.
        
        Args:
            agenda_title:       Title of the agenda item this document supports.
            agenda_description: Description of the agenda item for context.
            document_text:      Extracted text content of the uploaded supporting document.
        
        Returns:
            Dict with 'points' list of factual string statements.
        """
        # Truncate document to avoid token overflow (keep first ~6000 chars)
        doc_text_truncated = document_text[:6000].strip() if document_text else ""
        
        if not doc_text_truncated:
            return {"points": []}
        
        prompt = (
            _get_prompt("rom_agenda_doc_points")
            .replace("{agenda_title}", agenda_title or "Meeting Agenda")
            .replace("{agenda_description}", agenda_description or "No additional description provided.")
            .replace("{document_text}", doc_text_truncated)
        )
        
        raw = self._infer(prompt, max_new_tokens=1024, task_key="rom_agenda_doc_points")
        
        if not raw:
            return {"points": []}
        
        raw = raw.strip()
        json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', raw, re.DOTALL)
        if json_match:
            raw = json_match.group(1).strip()
        
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except Exception:
                    return {"points": []}
            else:
                return {"points": []}
        
        # Validate and clean the points list
        raw_points = data.get("points", []) if isinstance(data, dict) else []
        clean_points = [
            str(p).strip() for p in raw_points
            if p and str(p).strip() and len(str(p).strip()) > 10
        ][:5]  # Cap at 5 points
        
        presenter = data.get("presenter") if isinstance(data, dict) else None
        if isinstance(presenter, str):
            presenter = presenter.strip()
            if presenter.lower() in ("null", "none", "n/a", "unknown", "undefined", ""):
                presenter = None

        if not presenter and document_text:
            # Fallback regex search in document content for presenter/speaker/owner patterns
            m = re.search(
                r'(?:Presenter|Speaker|Presented\s+[Bb]y|Owner|Lead|Author|Prepared\s+[Bb]y)\s*[:\-]\s*([A-Za-z0-9 .,_\-]{2,60})',
                document_text[:4000],
                re.IGNORECASE
            )
            if m:
                presenter = m.group(1).strip()
        
        return {"points": clean_points, "presenter": presenter}

    def _parse_mom_json(self, raw: str, recording_meta: dict) -> Optional[dict]:

        """Utility to parse and clean MoM JSON output from Qwen3."""
        if not raw or not raw.strip():
            return None

        raw = raw.strip()
        if raw.startswith("```json"):
            raw = raw[7:]
        elif raw.startswith("```"):
            raw = raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except Exception:
                    logger.error("[QwenAI] Failed to parse MoM JSON from Qwen3 output")
                    return None
            else:
                logger.error("[QwenAI] No JSON found in Qwen3 MoM output")
                return None

        # Merge action_items + general_action_items
        ai_list = data.get("action_items", []) or []
        for g in (data.get("general_action_items", []) or []):
            if isinstance(g, str) and g.strip():
                ai_list.append({"task": g, "owner": "Unassigned", "deadline": "ASAP"})

        action_items = [
            {
                "task": (a.get("task", a) if isinstance(a, dict) else str(a)),
                "background": (a.get("background") if (isinstance(a, dict) and "background" in a) else None),
                "owner": (a.get("owner", "Unassigned") if isinstance(a, dict) else "Unassigned"),
                "deadline": (a.get("deadline", "ASAP") if isinstance(a, dict) else "ASAP"),
                "status": (a.get("status") if (isinstance(a, dict) and "status" in a) else None),
            }
            for a in ai_list
        ]

        points = data.get("points_discussed", []) or []
        if isinstance(points, list):
            points = [str(p) for p in points if p]
        elif isinstance(points, str) and points.strip():
            # LLM returned a string instead of a list - split on newlines or wrap
            lines = [ln.strip().lstrip("•-*").strip() for ln in points.splitlines() if ln.strip()]
            points = lines if lines else [points.strip()]
        else:
            points = []

        return {
            "title": data.get("title") or recording_meta.get("filename", "Meeting Notes"),
            "date": recording_meta.get("created_at", ""),
            "duration": recording_meta.get("duration", 0),
            "planned_start_time": "",
            "actual_start_time": data.get("actual_start_time") or "",
            "participants": recording_meta.get("speakers_detected", []),
            "introduction": data.get("introduction") or "",
            "points_discussed": points,
            "action_items": action_items,
            "conclusion": data.get("conclusion") or "",
        }

    def _split_context_into_sections(self, context: str, max_tokens: int = 4000) -> List[str]:
        """Split final meeting context into sections using token count, preserving line/sentence boundaries."""
        tokenizer = self._tokenizer
        lines = context.split("\n")
        sections = []
        current_lines = []
        current_tokens = 0

        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            if tokenizer:
                line_tokens = len(tokenizer.encode(line + "\n"))
            else:
                line_tokens = int(len(line_stripped) / 4) + 1
            if current_tokens + line_tokens > max_tokens and current_lines:
                sections.append("\n".join(current_lines))
                current_lines = [line]
                current_tokens = line_tokens
            else:
                current_lines.append(line)
                current_tokens += line_tokens

        if current_lines:
            sections.append("\n".join(current_lines))

        return sections

    def _generate_mom_section_wise(
        self,
        transcript: List[Dict],
        recording_meta: dict,
        context: str,
        agenda_section: str,
        reference_section: str,
    ) -> dict:
        """Sequential Section-wise MoM Generation pipeline with manual fallback support."""
        logger.info("[QwenAI] Starting Section-wise MoM Generation pipeline...")
        
        # 1. Split context by token count
        from config import settings
        sections = self._split_context_into_sections(context, max_tokens=settings.MOM_CONTEXT_TOKEN_THRESHOLD)
        logger.info(f"[QwenAI] Context split into {len(sections)} sections (threshold: {settings.MOM_CONTEXT_TOKEN_THRESHOLD}).")

        # 2. Generate partial MoM for each section
        partial_moms = []
        for idx, section in enumerate(sections):
            logger.info(f"[QwenAI] Generating partial MoM for section {idx+1}/{len(sections)}...")
            try:
                raw_sec = self._infer(
                    _get_prompt("mom").format(
                        transcript=section,
                        agenda_section=agenda_section,
                        reference_section=reference_section,
                    ),
                    max_new_tokens=1500,
                    task_key="mom",
                )
                sec_data = self._parse_mom_json(raw_sec, recording_meta)
                if sec_data:
                    partial_moms.append(sec_data)
            except Exception as sec_err:
                logger.warning(f"[QwenAI] Section {idx+1} generation failed: {sec_err}. Continuing with remaining sections...")
            finally:
                # Release temporary tensors
                import gc
                gc.collect()
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if not partial_moms:
            logger.error("[QwenAI] All partial MoMs failed to generate.")
            return _empty_mom(recording_meta)

        if len(partial_moms) == 1:
            logger.info("[QwenAI] Single partial MoM generated successfully. Skipping merge pass.")
            mom_data = partial_moms[0]
            return mom_data

        # 3. Merge partial MoMs using the LLM
        logger.info(f"[QwenAI] Merging {len(partial_moms)} partial MoMs using LLM...")
        partial_moms_json = json.dumps(partial_moms, indent=2)
        try:
            raw_merge = self._infer(
                    _get_prompt("mom_merge").format(partial_moms_json=partial_moms_json),
                    max_new_tokens=3072,
                    task_key="mom_merge",
                )
            final_data = self._parse_mom_json(raw_merge, recording_meta)
            if final_data:
                logger.info("[QwenAI] MoM consolidation complete.")
                return final_data
        except Exception as merge_err:
            logger.error(f"[QwenAI] LLM MoM merge failed: {merge_err}. Using manual consolidation fallback.", exc_info=True)

        # 4. Fallback manual merge in Python
        merged_points = []
        merged_action_items = []
        seen_tasks = set()
        introductions = []
        conclusions = []

        for p_mom in partial_moms:
            if p_mom.get("introduction"):
                introductions.append(p_mom["introduction"])
            if p_mom.get("conclusion"):
                conclusions.append(p_mom["conclusion"])
            for pt in p_mom.get("points_discussed", []):
                if pt not in merged_points:
                    merged_points.append(pt)
            for item in p_mom.get("action_items", []):
                task_key = (item.get("task", "") or "").lower().strip()
                if task_key and task_key not in seen_tasks:
                    seen_tasks.add(task_key)
                    merged_action_items.append(item)

        return {
            "title": partial_moms[0].get("title") or recording_meta.get("filename", "Meeting Notes"),
            "date": recording_meta.get("created_at", ""),
            "duration": recording_meta.get("duration", 0),
            "planned_start_time": "",
            "actual_start_time": partial_moms[0].get("actual_start_time") or "",
            "participants": recording_meta.get("speakers_detected", []),
            "introduction": " ".join(introductions)[:1000],
            "points_discussed": merged_points,
            "action_items": merged_action_items,
            "conclusion": " ".join(conclusions)[:1000],
        }

    def generate_mom(
        self,
        transcript: List[Dict],
        recording_meta: dict,
        context: Optional[str] = None,
        agenda_summary: Optional[str] = None,
        reference_summary: Optional[str] = None,
    ) -> dict:
        """Generate full enterprise-grade Minutes of Meeting with Section-wise OOM fallback."""
        if context is None:
            dialogue = _format_transcript(transcript)
            if not dialogue.strip():
                return _empty_mom(recording_meta)
            context = self._hierarchical_summarize(dialogue)
        elif not context.strip():
            return _empty_mom(recording_meta)

        # Build optional agenda / reference inject sections
        if agenda_summary and agenda_summary.strip():
            agenda_section = (
                "\nMEETING AGENDA (use this to frame the title and introduction):\n"
                + agenda_summary.strip()
            )
        else:
            agenda_section = ""

        if reference_summary and reference_summary.strip():
            reference_section = (
                "\nREFERENCE KNOWLEDGE (background context - use to enrich the MoM):\n"
                + reference_summary.strip()
                + "\n\n"
            )
        else:
            reference_section = "\n"

        # Calculate final context token length and decide generation strategy
        from config import settings
        
        use_ollama, _, _, _ = self._get_ollama_settings()
        if not use_ollama:
            self._get_pipeline()  # Ensure model/tokenizer is loaded
            
        if self._tokenizer:
            context_tokens = len(self._tokenizer.encode(context))
        else:
            # Heuristic fallback if tokenizer is unavailable
            context_tokens = int(len(context) / 4)
        logger.info(f"[QwenAI] Final context length ({context_tokens} tokens).")


        if context_tokens > settings.MOM_CONTEXT_TOKEN_THRESHOLD:
            logger.info(
                f"[QwenAI] Final context length ({context_tokens} tokens) exceeds "
                f"threshold ({settings.MOM_CONTEXT_TOKEN_THRESHOLD} tokens). "
                "Automatically switching to Section-wise MoM Generation."
            )
            return self._generate_mom_section_wise(
                transcript=transcript,
                recording_meta=recording_meta,
                context=context,
                agenda_section=agenda_section,
                reference_section=reference_section,
            )

        # Try single-pass generation first
        try:
            raw = self._infer(
                _get_prompt("mom").format(
                    transcript=context,
                    agenda_section=agenda_section,
                    reference_section=reference_section,
                ),
                max_new_tokens=1500,
                task_key="mom",
            )
            data = self._parse_mom_json(raw, recording_meta)
            if data and data.get("points_discussed"):
                return data
            raise ValueError("[QwenAI] Single-pass MoM returned empty points or invalid data")

        except Exception as e:
            err_msg = str(e).lower()
            is_oom = "out of memory" in err_msg or "oom" in err_msg or "cuda memory" in err_msg
            if is_oom:
                logger.warning(
                    f"[QwenAI] Single-pass MoM generation encountered OOM: {e}. "
                    "Triggering Section-wise MoM Generation pipeline fallback..."
                )
                return self._generate_mom_section_wise(
                    transcript=transcript,
                    recording_meta=recording_meta,
                    context=context,
                    agenda_section=agenda_section,
                    reference_section=reference_section,
                )
            else:
                logger.error(f"[QwenAI] Single-pass MoM generation failed: {e}", exc_info=True)
                return _empty_mom(recording_meta)

    def query(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.2) -> str:
        """General text generation interface for QwenProvider."""
        return self._infer(prompt, max_new_tokens=max_tokens)


# ══════════════════════════════════════════════════════════════
# Provider singleton accessor
# ══════════════════════════════════════════════════════════════

_provider_instance: Optional[QwenProvider] = None


def get_provider() -> QwenProvider:
    """Return the singleton QwenProvider instance."""
    global _provider_instance
    if _provider_instance is None:
        _provider_instance = QwenProvider()
    return _provider_instance


def warm_up_model():
    """
    Pre-loading is disabled to optimize memory lifecycle.
    Models are loaded lazily on demand and unloaded immediately after use.
    """
    logger.info("[QwenAI] Model warm-up deferred (load-on-demand enabled).")


# ══════════════════════════════════════════════════════════════
# Backward-compatibility alias
# ══════════════════════════════════════════════════════════════

# Keep LlamaProvider as an alias so any import from external code
# that still references it does not crash immediately.
LlamaProvider = QwenProvider

