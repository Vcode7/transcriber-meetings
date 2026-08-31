import sys
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
from dataclasses import dataclass
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from services.transcription import transcribe, unload_align_model


@dataclass
class DummyTranscriptionOptions:
    beam_size: int = 5
    best_of: int = 5
    initial_prompt: str = None
    language: str = None


class TestWhisperInitialPrompt(unittest.TestCase):
    def setUp(self):
        unload_align_model()

    @patch("soundfile.read")
    @patch("services.transcription.get_whisperx_model")
    @patch("services.transcription._resolve_device")
    @patch("services.transcription._align_segments_chunked")
    def test_whisperx_pipeline_initial_prompt_configured_on_options(
        self, mock_align_segments, mock_resolve_device, mock_get_model, mock_sf_read
    ):
        """Verify that when using WhisperX (FasterWhisperPipeline), initial_prompt is set on model.options dataclass and NOT passed to transcribe() kwargs."""
        mock_resolve_device.return_value = ("cpu", "int8")
        mock_sf_read.return_value = (np.zeros(16000, dtype=np.float32), 16000)

        # Pipeline mock with options dataclass
        mock_pipeline = MagicMock()
        mock_pipeline.options = DummyTranscriptionOptions(initial_prompt=None)
        
        def mock_transcribe(audio_path, **kwargs):
            # Assert that initial_prompt was NOT passed as a kwarg to FasterWhisperPipeline.transcribe
            if "initial_prompt" in kwargs:
                raise TypeError(f"FasterWhisperPipeline.transcribe() got an unexpected keyword argument 'initial_prompt'")
            return {
                "segments": [{"start": 0.0, "end": 2.0, "text": "test with initial prompt"}],
                "language": "en",
            }

        mock_pipeline.transcribe.side_effect = mock_transcribe
        mock_get_model.return_value = mock_pipeline
        mock_align_segments.return_value = {
            "segments": [{"start": 0.0, "end": 2.0, "text": "test with initial prompt", "words": []}]
        }

        test_prompt = "Dict: llm.\nTerms: Cuda, Torch."
        result = transcribe(
            "dummy_audio.wav",
            initial_prompt=test_prompt,
            language="en",
            user_settings={"enable_transcription_vad": False, "enable_audio_normalization": False},
        )

        # Verify initial_prompt was set on pipeline options dataclass
        self.assertEqual(mock_pipeline.options.initial_prompt, test_prompt)
        # Verify result was returned successfully without dropping or TypeError
        self.assertEqual(result["segments"][0]["text"], "test with initial prompt")

    @patch("soundfile.read")
    @patch("services.transcription.get_whisperx_model")
    @patch("services.transcription._resolve_device")
    @patch("services.transcription._align_segments_chunked")
    def test_whisperx_pipeline_clears_prompt_when_none(
        self, mock_align_segments, mock_resolve_device, mock_get_model, mock_sf_read
    ):
        """Verify that when initial_prompt is empty, model.options.initial_prompt is cleared to None."""
        mock_resolve_device.return_value = ("cpu", "int8")
        mock_sf_read.return_value = (np.zeros(16000, dtype=np.float32), 16000)

        mock_pipeline = MagicMock()
        mock_pipeline.options = DummyTranscriptionOptions(initial_prompt="old prompt")
        mock_pipeline.transcribe.return_value = {
            "segments": [{"start": 0.0, "end": 2.0, "text": "standard audio"}],
            "language": "en",
        }
        mock_get_model.return_value = mock_pipeline
        mock_align_segments.return_value = {
            "segments": [{"start": 0.0, "end": 2.0, "text": "standard audio", "words": []}]
        }

        transcribe(
            "dummy_audio.wav",
            initial_prompt="",
            language="en",
            user_settings={"enable_transcription_vad": False, "enable_audio_normalization": False},
        )

        self.assertIsNone(mock_pipeline.options.initial_prompt)

    @patch("soundfile.read")
    @patch("services.transcription.get_whisperx_model")
    @patch("services.transcription._resolve_device")
    @patch("services.transcription._align_segments_chunked")
    def test_raw_whisper_fallback_passes_kwarg(
        self, mock_align_segments, mock_resolve_device, mock_get_model, mock_sf_read
    ):
        """Verify that when model has no options attribute (e.g. raw faster-whisper WhisperModel), initial_prompt is passed via kwargs."""
        mock_resolve_device.return_value = ("cpu", "int8")
        mock_sf_read.return_value = (np.zeros(16000, dtype=np.float32), 16000)

        mock_raw_model = MagicMock(spec=["transcribe"])  # No options attribute
        mock_raw_model.transcribe.return_value = {
            "segments": [{"start": 0.0, "end": 2.0, "text": "raw model output"}],
            "language": "en",
        }
        mock_get_model.return_value = mock_raw_model
        mock_align_segments.return_value = {
            "segments": [{"start": 0.0, "end": 2.0, "text": "raw model output", "words": []}]
        }

        test_prompt = "Dict: terms."
        transcribe(
            "dummy_audio.wav",
            initial_prompt=test_prompt,
            language="en",
            user_settings={"enable_transcription_vad": False, "enable_audio_normalization": False},
        )

        # Verify transcribe kwargs received initial_prompt
        mock_raw_model.transcribe.assert_called_once()
        _, kwargs = mock_raw_model.transcribe.call_args
        self.assertEqual(kwargs.get("initial_prompt"), test_prompt)


if __name__ == "__main__":
    unittest.main()
