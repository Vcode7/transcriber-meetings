"""
Tests for parallel Whisper transcription helpers.
"""
import os
import wave
import pytest
from unittest.mock import patch, MagicMock


def _make_wav(path, duration_sec, sr=16000):
    n_frames = int(duration_sec * sr)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(b"\x00\x00" * n_frames)


class TestSplitAudioIntoChunks:
    def test_single_chunk_short_audio(self, tmp_path):
        from services.transcription import _split_audio_into_chunks
        wav = str(tmp_path / "short.wav")
        _make_wav(wav, duration_sec=120.0)
        chunks = _split_audio_into_chunks(wav, chunk_minutes=10)
        assert len(chunks) == 1
        tmp_wav, start_sec = chunks[0]
        assert start_sec == 0.0
        assert os.path.exists(tmp_wav)
        for p, _ in chunks:
            os.unlink(p)

    def test_multiple_chunks_25_minutes(self, tmp_path):
        from services.transcription import _split_audio_into_chunks
        wav = str(tmp_path / "long.wav")
        _make_wav(wav, duration_sec=25 * 60)
        chunks = _split_audio_into_chunks(wav, chunk_minutes=10)
        assert len(chunks) == 3
        offsets = [c[1] for c in chunks]
        assert offsets[0] == 0.0
        assert abs(offsets[1] - 600.0) < 1.0
        assert abs(offsets[2] - 1200.0) < 1.0
        for p, _ in chunks:
            assert os.path.exists(p)
            os.unlink(p)

    def test_chunks_are_valid_wav_files(self, tmp_path):
        import soundfile as sf
        from services.transcription import _split_audio_into_chunks
        wav = str(tmp_path / "test.wav")
        _make_wav(wav, duration_sec=21 * 60, sr=16000)
        chunks = _split_audio_into_chunks(wav, chunk_minutes=10)
        assert len(chunks) == 3
        for p, _ in chunks:
            audio, file_sr = sf.read(p, dtype="float32")
            assert file_sr == 16000
            assert len(audio) > 0
            os.unlink(p)


class TestOffsetSegmentTimestamps:
    def test_basic_offset(self):
        from services.transcription import _offset_segment_timestamps_parallel
        seg = {
            "start": 5.0, "end": 10.0, "text": "hello",
            "words": [{"word": "hello", "start": 5.1, "end": 5.5, "probability": 0.9}],
        }
        out = _offset_segment_timestamps_parallel(seg, offset=600.0)
        assert abs(out["start"] - 605.0) < 0.01
        assert abs(out["end"] - 610.0) < 0.01
        assert abs(out["words"][0]["start"] - 605.1) < 0.01
        assert abs(out["words"][0]["end"] - 605.5) < 0.01

    def test_zero_offset_unchanged(self):
        from services.transcription import _offset_segment_timestamps_parallel
        seg = {"start": 3.5, "end": 7.2, "text": "test", "words": []}
        out = _offset_segment_timestamps_parallel(seg, offset=0.0)
        assert out["start"] == 3.5
        assert out["end"] == 7.2

    def test_no_words_key(self):
        from services.transcription import _offset_segment_timestamps_parallel
        seg = {"start": 1.0, "end": 2.0, "text": "hi"}
        out = _offset_segment_timestamps_parallel(seg, offset=100.0)
        assert out["start"] == 101.0
        assert out["end"] == 102.0
        assert "words" not in out


class TestConfigDefaults:
    def test_whisper_parallel_processing_default(self):
        from config import settings
        val = getattr(settings, "WHISPER_PARALLEL_PROCESSING", None)
        assert val is not None
        assert val == 1

    def test_whisper_parallel_chunk_minutes_default(self):
        from config import settings
        val = getattr(settings, "WHISPER_PARALLEL_CHUNK_MINUTES", None)
        assert val is not None
        assert val == 10
