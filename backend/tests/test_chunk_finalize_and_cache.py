import sys
import os
import asyncio
import unittest
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from tasks.pipeline import _offset_segment_timestamps, _run_finalize_pipeline_impl
from services.transcription import transcribe, unload_align_model, _align_model_cache


class TestOffsetSegmentTimestamps(unittest.TestCase):
    def test_offset_segment_timestamps(self):
        segment = {
            "start": 1.0,
            "end": 5.0,
            "text": "hello world",
            "words": [
                {"word": "hello", "start": 1.2, "end": 1.8, "probability": 0.95},
                {"word": "world", "start": 2.0, "end": 2.5, "probability": 0.98},
                {"word": "unaligned_word"} # missing start/end
            ]
        }
        
        offset = 10.0
        offset_seg = _offset_segment_timestamps(segment, offset)
        
        self.assertEqual(offset_seg["start"], 11.0)
        self.assertEqual(offset_seg["end"], 15.0)
        self.assertEqual(offset_seg["words"][0]["start"], 11.2)
        self.assertEqual(offset_seg["words"][0]["end"], 11.8)
        self.assertEqual(offset_seg["words"][1]["start"], 12.0)
        self.assertEqual(offset_seg["words"][1]["end"], 12.5)
        self.assertNotIn("start", offset_seg["words"][2])


class TestAlignmentModelCaching(unittest.TestCase):
    @patch.dict("sys.modules", {"whisperx": MagicMock()})
    @patch("services.audio_preprocessing.detect_speech_regions")
    @patch("services.transcription.get_whisperx_model")
    @patch("services.transcription._resolve_device")
    @patch("services.transcription._align_segments_chunked")
    def test_alignment_model_caching(self, mock_align_segments, mock_resolve_device, mock_get_model, mock_detect_speech):
        mock_resolve_device.return_value = ("cpu", "int8")
        
        # Mock WhisperX transcription model
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "segments": [{"start": 0.0, "end": 2.0, "text": "test segment"}],
            "language": "en"
        }
        mock_get_model.return_value = mock_model
        
        # Mock load_align_model returning dummy model and metadata
        mock_load_align_model = sys.modules["whisperx"].load_align_model
        dummy_model = MagicMock()
        dummy_metadata = {"lang": "en"}
        mock_load_align_model.return_value = (dummy_model, dummy_metadata)
        
        mock_align_segments.return_value = {
            "segments": [{"start": 0.0, "end": 2.0, "text": "test segment", "words": []}]
        }
        
        # Clear cache first
        unload_align_model()
        self.assertEqual(len(_align_model_cache), 0)
        
        # First call: should load model
        res1 = transcribe("dummy_path.wav", language="en")
        self.assertEqual(mock_load_align_model.call_count, 1)
        self.assertIn(("en", "cpu", "facebook/wav2vec2-base"), _align_model_cache)
        
        # Second call: should reuse cached model
        res2 = transcribe("dummy_path.wav", language="en")
        self.assertEqual(mock_load_align_model.call_count, 1) # still 1
        
        # Unload
        unload_align_model()
        self.assertEqual(len(_align_model_cache), 0)


class AsyncContextManagerMock:
    def __init__(self, mock_conn):
        self.mock_conn = mock_conn
    async def __aenter__(self):
        return self.mock_conn
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class MappingsMock:
    def __init__(self, data):
        self._data = data
    def fetchall(self):
        return self._data
    def fetchone(self):
        if self._data:
            return self._data[0]
        return None


class TestRunFinalizePipeline(unittest.IsolatedAsyncioTestCase):
    @patch("database.get_db_context")
    @patch("tasks.pipeline.get_db_context")
    @patch("tasks.pipeline.transcribe")
    @patch("tasks.pipeline.diarize")
    @patch("tasks.pipeline.identify_speakers")
    @patch("tasks.pipeline.refine_transcript_speakers_with_ecapa")
    @patch("tasks.pipeline.generate_mom")
    @patch("tasks.pipeline._emit_analytics")
    @patch("tasks.pipeline._generate_speaker_aware_chunk_summaries")
    @patch("tasks.pipeline.build_context_summary")
    @patch("services.ai_provider.get_provider")
    async def test_run_finalize_pipeline_merges_chunks(
        self, mock_get_provider, mock_build_ctx, mock_gen_chunks, mock_emit, mock_gen_mom, mock_refine, mock_identify, mock_diarize, mock_transcribe, mock_get_db_ctx, mock_db_ctx_base
    ):

        mock_db = MagicMock()
        mock_db.commit = AsyncMock()
        mock_get_db_ctx.return_value = AsyncContextManagerMock(mock_db)
        mock_db_ctx_base.return_value = AsyncContextManagerMock(mock_db)

        
        chunk_rows = [
            {
                "id": "chunk_1",
                "status": "done",
                "chunk_start_sec": 0.0,
                "transcript": '[{"start": 1.0, "end": 3.0, "text": "chunk one", "words": [{"word": "chunk", "start": 1.0, "end": 2.0}, {"word": "one", "start": 2.0, "end": 3.0}]}]',
                "aligned_result": '{"segments": [{"start": 1.0, "end": 3.0, "text": "chunk one", "words": [{"word": "chunk", "start": 1.0, "end": 2.0}, {"word": "one", "start": 2.0, "end": 3.0}]}]}',
                "raw_text": "chunk one"
            },
            {
                "id": "chunk_2",
                "status": "done",
                "chunk_start_sec": 10.0,
                "transcript": '[{"start": 0.5, "end": 2.5, "text": "chunk two", "words": [{"word": "chunk", "start": 0.5, "end": 1.5}, {"word": "two", "start": 1.5, "end": 2.5}]}]',
                "aligned_result": '{"segments": [{"start": 0.5, "end": 2.5, "text": "chunk two", "words": [{"word": "chunk", "start": 0.5, "end": 1.5}, {"word": "two", "start": 1.5, "end": 2.5}]}]}',
                "raw_text": "chunk two"
            }
        ]
        
        # SQL-specific database execution mocking
        async def mock_execute(sql, params=None):
            sql_str = str(sql)
            res = MagicMock()
            if "voice_profiles" in sql_str:
                res.mappings.return_value = MappingsMock([])
            elif "recording_chunks" in sql_str:
                res.mappings.return_value = MappingsMock(chunk_rows)
            elif "recordings" in sql_str:
                # detected language query or metadata query
                if "SELECT filename" in sql_str:
                    res.mappings.return_value = MappingsMock([{"filename": "test.wav", "created_at": "now", "duration": 20}])
                else:
                    res.fetchone.return_value = ("en",)
            elif "user_settings" in sql_str:
                res.mappings.return_value = MappingsMock([{"speaker_similarity_threshold": 0.5}])
            else:
                res.mappings.return_value = MappingsMock([])
                res.fetchone.return_value = None
            return res
            
        mock_db.execute = AsyncMock(side_effect=mock_execute)
        
        # Diarization/ID/ECAPA mocks
        mock_diarize.return_value = [{"start": 0.0, "end": 20.0, "speaker": "SPEAKER_00"}]
        mock_identify.return_value = [{"start": 0.0, "end": 20.0, "speaker": "SPEAKER_00", "speaker_label": "Speaker 1", "speaker_profile_id": None}]
        mock_refine.side_effect = lambda file_path, speaker_segments, **kwargs: speaker_segments
        mock_gen_mom.return_value = {"title": "MoM Title"}
        
        await _run_finalize_pipeline_impl(
            recording_id="rec_id",
            full_wav_path="dummy_path.wav",
            chunk_ids=["chunk_1", "chunk_2"],
            user_id="user_id",
        )
        
        # transcribe() should NOT have been called (no full audio transcription because chunks were usable)
        self.assertFalse(mock_transcribe.called)
        
        # Verify DB save transcript contains merged segments with offset timestamps
        save_call = None
        for call in mock_db.execute.call_args_list:
            query = str(call[0][0])
            if "UPDATE recordings SET" in query and "status = 'transcript_ready'" in query:
                save_call = call
                break
                
        self.assertIsNotNone(save_call)
        params = save_call[0][1]
        import json
        saved_transcript = json.loads(params["transcript"])
        
        self.assertEqual(len(saved_transcript), 2)
        self.assertEqual(saved_transcript[0]["start"], 1.0)
        self.assertEqual(saved_transcript[0]["end"], 3.0)
        self.assertEqual(saved_transcript[1]["start"], 10.5)
        self.assertEqual(saved_transcript[1]["end"], 12.5)
        self.assertEqual(saved_transcript[1]["words"][0]["start"], 10.5)
        self.assertEqual(saved_transcript[1]["words"][1]["start"], 11.5)

    @patch("database.get_db_context")
    @patch("tasks.pipeline.get_db_context")
    @patch("tasks.pipeline.transcribe")
    @patch("tasks.pipeline.diarize")
    @patch("tasks.pipeline.identify_speakers")
    @patch("tasks.pipeline.refine_transcript_speakers_with_ecapa")
    @patch("tasks.pipeline.generate_mom")
    @patch("tasks.pipeline._emit_analytics")
    @patch("tasks.pipeline._generate_speaker_aware_chunk_summaries")
    @patch("tasks.pipeline.build_context_summary")
    @patch("services.ai_provider.get_provider")
    async def test_run_finalize_pipeline_fallback_to_full_audio(
        self, mock_get_provider, mock_build_ctx, mock_gen_chunks, mock_emit, mock_gen_mom, mock_refine, mock_identify, mock_diarize, mock_transcribe, mock_get_db_ctx, mock_db_ctx_base
    ):
        mock_db = MagicMock()

        mock_db.commit = AsyncMock()
        mock_get_db_ctx.return_value = AsyncContextManagerMock(mock_db)
        mock_db_ctx_base.return_value = AsyncContextManagerMock(mock_db)

        
        # SQL-specific database execution mocking
        async def mock_execute(sql, params=None):
            sql_str = str(sql)
            res = MagicMock()
            if "recordings" in sql_str and "SELECT filename" in sql_str:
                res.mappings.return_value = MappingsMock([{"filename": "test.wav", "created_at": "now", "duration": 20}])
            elif "user_settings" in sql_str:
                res.mappings.return_value = MappingsMock([{"speaker_similarity_threshold": 0.5}])
            else:
                res.mappings.return_value = MappingsMock([])
                res.fetchone.return_value = None
            return res
            
        mock_db.execute = AsyncMock(side_effect=mock_execute)
        
        # Mock full audio transcription result with matching words list so resegmentation reconstructs text correctly
        mock_transcribe.return_value = {
            "segments": [{"start": 1.0, "end": 4.0, "text": "full audio text", "words": [
                {"word": "full", "start": 1.0, "end": 2.0},
                {"word": "audio", "start": 2.0, "end": 3.0},
                {"word": "text", "start": 3.0, "end": 4.0}
            ]}],
            "language": "en",
            "raw_text": "full audio text",
            "aligned_result": {"segments": [{"start": 1.0, "end": 4.0, "text": "full audio text", "words": [
                {"word": "full", "start": 1.0, "end": 2.0},
                {"word": "audio", "start": 2.0, "end": 3.0},
                {"word": "text", "start": 3.0, "end": 4.0}
            ]}]}
        }
        
        mock_diarize.return_value = [{"start": 0.0, "end": 20.0, "speaker": "SPEAKER_00"}]
        mock_identify.return_value = [{"start": 0.0, "end": 20.0, "speaker": "SPEAKER_00", "speaker_label": "Speaker 1", "speaker_profile_id": None}]
        mock_refine.side_effect = lambda file_path, speaker_segments, **kwargs: speaker_segments
        mock_gen_mom.return_value = {"title": "MoM Title"}
        
        await _run_finalize_pipeline_impl(
            recording_id="rec_id",
            full_wav_path="dummy_path.wav",
            chunk_ids=[],
            user_id="user_id",
        )
        
        # transcribe() SHOULD have been called (no chunks were usable)
        self.assertTrue(mock_transcribe.called)
        
        # Verify DB save transcript contains the full audio transcribed segments
        save_call = None
        for call in mock_db.execute.call_args_list:
            query = str(call[0][0])
            if "UPDATE recordings SET" in query and "status = 'transcript_ready'" in query:
                save_call = call
                break
                
        self.assertIsNotNone(save_call)
        params = save_call[0][1]
        import json
        saved_transcript = json.loads(params["transcript"])
        self.assertEqual(len(saved_transcript), 1)
        self.assertEqual(saved_transcript[0]["text"], "full audio text")


def _chunk_row(cid, start, end, segs):
    import json
    return {
        "id": cid, "status": "done", "chunk_index": int(start // 10),
        "chunk_start_sec": start, "chunk_end_sec": end,
        "transcript": json.dumps(segs), "aligned_result": json.dumps({"segments": segs}),
        "raw_text": " ".join(s["text"] for s in segs),
    }


def _seg(text, start, end):
    words = text.split()
    step = (end - start) / len(words)
    return {"start": start, "end": end, "text": text, "words": [
        {"word": w, "start": round(start + i * step, 3), "end": round(start + (i + 1) * step, 3)}
        for i, w in enumerate(words)
    ]}


class TestFinalizeTrimAndTail(unittest.IsolatedAsyncioTestCase):
    """Live recordings: trimmed chunk transcripts are remapped, and the audio after the last chunk is transcribed."""

    def setUp(self):
        import tempfile
        self._tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmpdir.cleanup()

    def _wav(self, seconds):
        import numpy as np
        import soundfile as sf
        path = os.path.join(self._tmpdir.name, f"full_{seconds}.wav")
        t = np.arange(int(16000 * seconds)) / 16000
        sf.write(path, 0.3 * np.sin(2 * np.pi * 220 * t), 16000, subtype="PCM_16")
        return path

    async def _run(self, chunk_rows, full_wav_path, transcribe_result=None, **kwargs):
        import contextlib
        import json
        mock_db = MagicMock()
        mock_db.commit = AsyncMock()

        async def mock_execute(sql, params=None):
            sql_str = str(sql)
            res = MagicMock()
            if "recording_chunks" in sql_str:
                res.mappings.return_value = MappingsMock(chunk_rows)
            elif "recordings" in sql_str and "SELECT filename" in sql_str:
                res.mappings.return_value = MappingsMock([{"filename": "t.wav", "created_at": "now", "duration": 30}])
            elif "recordings" in sql_str:
                res.fetchone.return_value = ("en",)
                res.mappings.return_value = MappingsMock([])
            else:
                res.mappings.return_value = MappingsMock([])
                res.fetchone.return_value = None
            return res

        mock_db.execute = AsyncMock(side_effect=mock_execute)
        with contextlib.ExitStack() as stack:
            p = lambda target, **kw: stack.enter_context(patch(target, **kw))
            p("database.get_db_context", return_value=AsyncContextManagerMock(mock_db))
            p("tasks.pipeline.get_db_context", return_value=AsyncContextManagerMock(mock_db))
            mock_transcribe = p("tasks.pipeline.transcribe", return_value=transcribe_result)
            p("tasks.pipeline.diarize", return_value=[{"start": 0.0, "end": 30.0, "speaker": "SPEAKER_00"}])
            p("tasks.pipeline.identify_speakers", return_value=[
                {"start": 0.0, "end": 30.0, "speaker": "SPEAKER_00", "speaker_label": "Speaker 1", "speaker_profile_id": None}])
            p("tasks.pipeline.refine_transcript_speakers_with_ecapa", side_effect=lambda file_path, speaker_segments, **kw: speaker_segments)
            p("tasks.pipeline.generate_mom", return_value={"title": "MoM"})
            p("tasks.pipeline._emit_analytics")
            mock_summaries = p("tasks.pipeline._generate_speaker_aware_chunk_summaries")
            p("tasks.pipeline.build_context_summary")
            p("services.ai_provider.get_provider")
            await _run_finalize_pipeline_impl(
                recording_id="rec_id", full_wav_path=full_wav_path,
                chunk_ids=[r["id"] for r in chunk_rows], user_id="user_id", **kwargs,
            )

        saved = next(
            c for c in mock_db.execute.call_args_list
            if "UPDATE recordings SET" in str(c[0][0]) and "status = 'transcript_ready'" in str(c[0][0])
        )
        return json.loads(saved[0][1]["transcript"]), mock_transcribe, mock_summaries

    async def test_audio_after_last_chunk_is_transcribed(self):
        full = self._wav(25)
        rows = [
            _chunk_row("c1", 0.0, 10.0, [_seg("chunk one", 1.0, 3.0)]),
            _chunk_row("c2", 10.0, 20.0, [_seg("chunk two", 0.5, 2.5)]),
        ]
        tail_seg = _seg("tail words", 1.0, 3.0)
        transcript, mock_transcribe, _ = await self._run(
            rows, full,
            transcribe_result={"segments": [tail_seg], "raw_text": "tail words",
                               "aligned_result": {"segments": [tail_seg]}, "language": "en"},
        )
        mock_transcribe.assert_called_once()
        tail_path = mock_transcribe.call_args[0][0]
        self.assertTrue(tail_path.endswith("_tail.wav"))
        self.assertFalse(os.path.exists(tail_path))  # temp tail file cleaned up
        self.assertEqual([s["text"] for s in transcript], ["chunk one", "chunk two", "tail words"])
        self.assertEqual((transcript[2]["start"], transcript[2]["end"]), (21.0, 23.0))

    async def test_no_tail_transcription_when_chunks_cover_everything(self):
        full = self._wav(20)
        rows = [
            _chunk_row("c1", 0.0, 10.0, [_seg("chunk one", 1.0, 3.0)]),
            _chunk_row("c2", 10.0, 20.0, [_seg("chunk two", 0.5, 2.5)]),
        ]
        transcript, mock_transcribe, _ = await self._run(rows, full)
        self.assertFalse(mock_transcribe.called)
        self.assertEqual(len(transcript), 2)

    async def test_trimmed_recording_remaps_chunk_transcripts(self):
        # Source 20 s, user removed 5–10 s → edited audio is 15 s.
        full = self._wav(15)
        rows = [
            _chunk_row("c1", 0.0, 10.0, [_seg("chunk one", 1.0, 3.0), _seg("removed words", 6.0, 8.0)]),
            _chunk_row("c2", 10.0, 20.0, [_seg("chunk two", 0.5, 2.5)]),
        ]
        transcript, mock_transcribe, mock_summaries = await self._run(
            rows, full, edit_keep_ranges=[[0.0, 5.0], [10.0, 20.0]],
        )
        self.assertFalse(mock_transcribe.called)  # edited coverage reaches the end → no tail
        self.assertEqual([s["text"] for s in transcript], ["chunk one", "chunk two"])
        self.assertEqual((transcript[0]["start"], transcript[0]["end"]), (1.0, 3.0))
        self.assertEqual((transcript[1]["start"], transcript[1]["end"]), (5.5, 7.5))
        # Per-chunk summaries use windows on the edited timeline too.
        summary_rows = mock_summaries.call_args[0][0]
        self.assertEqual(
            [(r["chunk_start_sec"], r["chunk_end_sec"]) for r in summary_rows],
            [(0.0, 5.0), (5.0, 15.0)],
        )


if __name__ == "__main__":
    unittest.main()
