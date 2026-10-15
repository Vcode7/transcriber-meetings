"""
"Extract Agenda from Tables": finding tables in extracted text (services/text_tables.py), the
tables-first agenda extraction with its whole-document fallback (rom_service.extract_agendas)
and the extraction rules added to the rom_agenda prompt (QwenProvider.generate_rom_agendas).
"""
import json

import services.ai_provider as ai_provider
from services.ai_provider import ROM_AGENDA_EXTRACTION_RULES, ROM_AGENDA_GENERATION_PROMPT, get_provider
from services.rom_service import extract_agendas
from services.text_tables import find_table_blocks, strip_layout_markup

AGENDA_TABLE = (
    "[Table (Page 1, #1): 3 rows x 3 cols]\n"
    "| Time | Topic | Presenter |\n"
    "| --- | --- | --- |\n"
    "| 10:00 | Opening remarks | Chair |\n"
    "| 10:30 | Budget review | CFO |"
)
DOCUMENT = f"Board Meeting Agenda\nVenue: Conference Room 2\n\n{AGENDA_TABLE}\n\nPlease be on time."


# ── Finding tables ────────────────────────────────────────────────────────────

def test_markdown_table_is_found_without_headings_or_other_text():
    assert find_table_blocks(DOCUMENT) == [AGENDA_TABLE]


def test_header_without_data_rows_is_not_a_table():
    assert find_table_blocks("[Table: 1 rows x 2 cols]\n| Time | Topic |\n| --- | --- |") == []


def test_spreadsheet_block_is_found_without_its_metadata():
    text = ("Source: agenda.xlsx\nSheet: Day 1\nRows: 2-3\n---------------------\n"
            "Headers: Time | Topic\n[Row 2] 10:00 | Opening\n[Row 3] 10:30 | Budget")
    assert find_table_blocks(text) == ["Headers: Time | Topic\n[Row 2] 10:00 | Opening\n[Row 3] 10:30 | Budget"]


def test_tab_and_space_aligned_lines_are_table_like():
    tabs = "Agenda\n10:00\tOpening remarks\tChair\n10:30\tBudget review\tCFO\nEnd of agenda"
    assert find_table_blocks(tabs) == ["10:00\tOpening remarks\tChair\n10:30\tBudget review\tCFO"]
    spaces = "1.  Opening remarks      10:00\n2.  Budget review        10:30"
    assert find_table_blocks(spaces) == [spaces]


def test_prose_lists_and_double_spaces_are_not_tables():
    prose = ("The meeting opened at ten.  Members reviewed the minutes.\n"
             "The budget was approved.  Hiring was deferred.\n"
             "1. Opening remarks\n2. Budget review\n- Any other business")
    assert find_table_blocks(prose) == []


def test_single_delimited_line_or_changing_column_count_is_not_a_table():
    assert find_table_blocks("Venue: Room 2\tDate: 14 Oct") == []
    assert find_table_blocks("a\tb\nc\td\te\tf\tg") == []


def test_table_inside_embedded_image_ocr_is_found():
    text = f"Slide title\n[Embedded Image OCR: {AGENDA_TABLE}]"
    assert find_table_blocks(text) == [AGENDA_TABLE]


def test_strip_layout_markup_keeps_only_the_words():
    text = "[Table: 2 rows x 2 cols]\n| A | B \\| C |\n| --- | --- |\n| 1 | 2 |\nAfter the table"
    assert strip_layout_markup(text) == "A B | C\n1 2\nAfter the table"


# ── Tables first, whole document as fallback ──────────────────────────────────

class FakeProvider:
    def __init__(self, agendas_by_source):
        self.agendas_by_source = agendas_by_source
        self.calls = []

    def generate_rom_agendas(self, agenda_text, context="", previous_mom=None, source=None):
        self.calls.append({"source": source, "text": agenda_text, "context": context})
        return {"agendas": self.agendas_by_source.get(source, [])}


TABLE_AGENDAS = [{"title": "Opening remarks", "presenter": "Chair"}, {"title": "Budget review", "presenter": "CFO"}]
DOCUMENT_AGENDAS = [{"title": "Board Meeting Agenda"}, {"title": "Opening remarks"}, {"title": "Budget review"}]


def test_tables_mode_sends_only_the_tables():
    provider = FakeProvider({"tables": TABLE_AGENDAS, "document": DOCUMENT_AGENDAS})
    result = extract_agendas(provider, DOCUMENT, context="ctx", from_tables=True)
    assert provider.calls == [{"source": "tables", "text": AGENDA_TABLE, "context": "ctx"}]
    assert [a["title"] for a in result["agendas"]] == ["Opening remarks", "Budget review"]
    assert [a["agenda_id"] for a in result["agendas"]] == ["A1", "A2"]
    assert result["agendas"][0]["presenter"] == "Chair"
    assert (result["extraction_mode"], result["table_count"]) == ("tables", 1)


def test_document_without_tables_is_scanned_whole():
    text = "Agenda\n1. Opening remarks\n2. Budget review"
    provider = FakeProvider({"document": DOCUMENT_AGENDAS[1:]})
    result = extract_agendas(provider, text, from_tables=True)
    assert provider.calls == [{"source": "document", "text": text, "context": ""}]
    assert (result["extraction_mode"], result["table_count"]) == ("document", 0)
    assert len(result["agendas"]) == 2


def test_tables_without_agenda_rows_fall_back_to_the_whole_document():
    provider = FakeProvider({"tables": [], "document": DOCUMENT_AGENDAS[1:]})
    result = extract_agendas(provider, DOCUMENT, from_tables=True)
    assert [c["source"] for c in provider.calls] == ["tables", "document"]
    assert provider.calls[1]["text"] == DOCUMENT
    assert (result["extraction_mode"], result["table_count"]) == ("document", 1)


def test_option_off_scans_the_whole_document():
    provider = FakeProvider({"tables": TABLE_AGENDAS, "document": DOCUMENT_AGENDAS})
    result = extract_agendas(provider, DOCUMENT, from_tables=False)
    assert provider.calls == [{"source": "document", "text": DOCUMENT, "context": ""}]
    assert len(result["agendas"]) == 3 and result["extraction_mode"] == "document"


# ── Extraction rules in the rom_agenda prompt ─────────────────────────────────

def _capture_prompt(monkeypatch, template, **kwargs):
    provider = get_provider()
    prompts = []

    def fake_infer(prompt, max_new_tokens=2048, task_key=None, **_):
        prompts.append(prompt)
        return json.dumps({"agendas": [{"title": "Opening remarks"}]})

    monkeypatch.setattr(ai_provider, "_get_prompt", lambda key: template)
    monkeypatch.setattr(provider, "_infer", fake_infer)
    result = provider.generate_rom_agendas(AGENDA_TABLE, context="", **kwargs)
    assert result["agendas"][0]["title"] == "Opening remarks"
    return prompts[0]


def test_prompt_carries_the_rules_of_the_source(monkeypatch):
    prompt = _capture_prompt(monkeypatch, ROM_AGENDA_GENERATION_PROMPT, source="tables")
    assert ROM_AGENDA_EXTRACTION_RULES["tables"] in prompt
    assert ROM_AGENDA_EXTRACTION_RULES["document"] not in prompt
    assert "{extraction_rules}" not in prompt
    assert prompt.index("UPLOADED AGENDA") < prompt.index("AGENDA SOURCE: TABLES ONLY")

    prompt = _capture_prompt(monkeypatch, ROM_AGENDA_GENERATION_PROMPT, source="document")
    assert ROM_AGENDA_EXTRACTION_RULES["document"] in prompt


def test_prompt_without_source_has_no_rules(monkeypatch):
    prompt = _capture_prompt(monkeypatch, ROM_AGENDA_GENERATION_PROMPT)
    assert "AGENDA SOURCE" not in prompt and "{extraction_rules}" not in prompt


def test_customized_prompt_without_placeholder_still_gets_the_rules(monkeypatch):
    template = "Extract agendas.\n\n{agenda_input}\n\n{context}\n\nReturn JSON."
    prompt = _capture_prompt(monkeypatch, template, source="tables")
    assert prompt.index(AGENDA_TABLE) < prompt.index("AGENDA SOURCE: TABLES ONLY") < prompt.index("Return JSON.")
