"""Stage 2 end to end with a scripted LLM and no models: validation checks each point against the
transcript it was written from (not the whole meeting), and the final dedupe step."""
import zlib
from unittest.mock import MagicMock

import numpy as np

import services.text_embedding_service as tes
from services.rom_service import RomService, _point_evidence, _supporting_documents


class FakeEmbedder:
    """Deterministic unit vectors per text; texts listed in ``same`` share one vector."""
    _dim = 64

    def __init__(self, same=()):
        self.same = {t: group for group in same for t in group}

    def load(self):
        pass

    def _vec(self, text):
        key = self.same.get(text, (text,))[0]
        rng = np.random.default_rng(zlib.crc32(key.encode("utf-8")))
        v = rng.normal(size=self._dim)
        return (v / np.linalg.norm(v)).astype(np.float32)

    def encode_batch(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def encode(self, texts):
        return self.encode_batch(texts if isinstance(texts, list) else [texts])


class EmptyStore:
    _meta = []

    def load_or_create(self):
        pass

    def total_vectors(self):
        return 0


STAGE1 = [
    {"id": "s1", "discussion_point": "Meena presented the hostel renovation budget of 42 lakh.",
     "speakers": ["Meena"], "timeline_start": 0.0, "timeline_end": 60.0, "numbers": ["42 lakh"],
     "technical_terms": ["PWD"], "raw_transcript_text": "[T1 | 00:00-00:30] Meena: WINDOW-A budget is 42 lakh."},
    {"id": "s2", "discussion_point": "Ravi will float the hostel renovation tender by Friday.",
     "speakers": ["Ravi"], "timeline_start": 60.0, "timeline_end": 120.0, "dates": ["Friday"],
     "raw_transcript_text": "[T2 | 00:31-01:00] Ravi: WINDOW-A I will float the tender by Friday."},
    {"id": "s3", "discussion_point": "The canteen vendor contract expires in March.",
     "speakers": ["Anita"], "timeline_start": 120.0, "timeline_end": 180.0,
     "raw_transcript_text": "[T3 | 01:01-01:30] Anita: WINDOW-B canteen contract ends in March."},
]
TRANSCRIPT = [{"speaker_label": "Meena", "text": "FULL-TRANSCRIPT-ONLY sentence", "start": 0.0, "end": 5.0}]


def _run(monkeypatch, enhanced, provider=None, embedder=None):
    provider = provider or MagicMock()
    provider.enhance_rom_discussion_window.return_value = {"enhanced_points": enhanced}
    monkeypatch.setattr("services.ai_provider.get_provider", lambda: provider)
    monkeypatch.setattr(tes, "get_text_embedder", lambda: embedder or FakeEmbedder())
    monkeypatch.setattr(tes, "unload_text_embedder", lambda: None)
    monkeypatch.setattr("services.vector_store.get_meeting_context_store", lambda rid, dim=0: EmptyStore())
    monkeypatch.setattr("services.vector_store.get_global_context_store", lambda uid, dim=0: EmptyStore())
    monkeypatch.setattr(RomService, "index_stage2_points_in_chromadb", lambda self, **kw: 0)
    points = RomService().enhance_discussion_points(
        [dict(p) for p in STAGE1], "rec-1", "test_user", discussion_window_size=5,
        parallel_window_processing=1, transcript=TRANSCRIPT, previous_meeting_mode="off",
    )
    return points, provider


def test_validation_uses_the_points_own_window_text_and_skips_carried_points(monkeypatch):
    provider = MagicMock()
    provider.validate_stage2_point.return_value = True
    enhanced = [{"original_point_ids": ["P1", "P2"],
                 "discussion_point": "Meena presented the 42 lakh budget and Ravi will float the tender by Friday."}]

    points, provider = _run(monkeypatch, enhanced, provider)

    assert [p["original_point_ids"] for p in points] == [["s1", "s2"], ["s3"]]
    assert provider.validate_stage2_point.call_count == 1          # the carried point (s3) is not validated
    point_text, evidence, supporting = provider.validate_stage2_point.call_args.args
    assert "WINDOW-A budget" in evidence and "WINDOW-A I will float" in evidence
    assert "WINDOW-B" not in evidence and "FULL-TRANSCRIPT-ONLY" not in evidence
    assert supporting == ""


def test_correction_gets_per_point_evidence_and_changes_text_only(monkeypatch):
    provider = MagicMock()
    provider.validate_stage2_point.return_value = False
    provider.correct_invalid_stage2_points.return_value = [
        {"id": "P1", "polished_text": "Meena presented the 42 lakh hostel budget.", "speakers": ["Bob"]},
    ]
    enhanced = [{"original_point_ids": ["P1"], "discussion_point": "Meena presented a 50 lakh budget."},
                {"original_point_ids": ["P2"], "discussion_point": "Ravi will float the tender by Friday."}]

    points, provider = _run(monkeypatch, enhanced, provider)

    batch_json = provider.correct_invalid_stage2_points.call_args.args[0]
    assert "WINDOW-A budget is 42 lakh" in batch_json and "FULL-TRANSCRIPT-ONLY" not in batch_json
    first = points[0]
    assert first["polished_text"] == "Meena presented the 42 lakh hostel budget." and first["stage2_corrected"]
    assert first["speakers"] == ["Meena"]
    assert points[1]["polished_text"] == "Ravi will float the tender by Friday."   # no correction returned


_BUDGET_A = "Meena presented the hostel renovation budget of 42 lakh for the PWD works."
_BUDGET_B = "The hostel renovation budget of 42 lakh for the PWD works was presented by Meena."


def test_reworded_repeat_is_deduplicated_and_merged_point_keeps_every_fact(monkeypatch):
    provider = MagicMock()
    provider.validate_stage2_point.return_value = True
    provider.deduplicate_rom_points.return_value = {
        "group_classification": "duplicate",
        "result_points": [{"polished_text": _BUDGET_A, "technical_terms": [], "numbers": []}],
    }
    enhanced = [{"original_point_ids": ["P1"], "discussion_point": _BUDGET_A},
                {"original_point_ids": ["P2"], "discussion_point": _BUDGET_B}]

    points, provider = _run(monkeypatch, enhanced, provider, FakeEmbedder(same=[(_BUDGET_A, _BUDGET_B)]))

    provider.deduplicate_rom_points.assert_called_once()
    merged = next(p for p in points if p["original_point_ids"] == ["s1", "s2"])
    assert merged["technical_terms"] == ["PWD"] and merged["numbers"] == ["42 lakh"] and merged["dates"] == ["Friday"]
    assert (merged["timeline_start"], merged["timeline_end"]) == (0.0, 120.0)
    assert merged["discussion_point"] == _BUDGET_A


def test_similar_embeddings_with_different_wording_are_not_deduplicated(monkeypatch):
    provider = MagicMock()
    provider.validate_stage2_point.return_value = True
    other = "Ravi will float the canteen tender by Friday after the vendor visit."
    enhanced = [{"original_point_ids": ["P1"], "discussion_point": _BUDGET_A},
                {"original_point_ids": ["P2"], "discussion_point": other}]

    _run(monkeypatch, enhanced, provider, FakeEmbedder(same=[(_BUDGET_A, other)]))

    provider.deduplicate_rom_points.assert_not_called()


def test_evidence_helpers():
    by_id = {p["id"]: p for p in STAGE1}
    assert _point_evidence({"original_point_ids": ["s1", "s1", "s3"]}, by_id).count("WINDOW-A budget") == 1
    assert _point_evidence({"original_point_ids": ["missing"], "owner_transcript_excerpt": "[T9] x"}, by_id) == "[T9] x"
    point = {"context_usage_report": {"context_added": True},
             "retrieved_context": {"meeting_chunks": [{"text": "Budget sheet v3"}], "global_chunks": [{"text": "PWD SOP"}]}}
    assert _supporting_documents(point) == "Budget sheet v3\n\nPWD SOP"
    assert _supporting_documents({**point, "context_usage_report": {"context_added": False}}) == ""
