"""Tests for the auto-transcribe-voice pipeline."""
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from underthesea.pipeline import transcribe as transcribe_module
from underthesea.pipeline.transcribe import (
    DEFAULT_MODEL,
    DEFAULT_SAMPLE_RATE,
    auto_transcribe,
    listen,
    transcribe,
)


class TestTranscribePublicAPI(unittest.TestCase):
    def test_exports(self):
        self.assertTrue(callable(transcribe))
        self.assertTrue(callable(auto_transcribe))
        self.assertIs(listen, transcribe)
        self.assertEqual(DEFAULT_SAMPLE_RATE, 16_000)
        self.assertEqual(DEFAULT_MODEL, "openai/whisper-small")

    def test_top_level_optional_import(self):
        import underthesea
        self.assertIn("transcribe", underthesea.__all__)


class TestTranscribeWithMockedPipeline(unittest.TestCase):
    def setUp(self):
        transcribe_module._load_pipeline.cache_clear()

    def test_transcribe_from_numpy(self):
        fake_pipeline = MagicMock(return_value={"text": "  xin chào  "})
        with patch.object(transcribe_module, "_load_pipeline", return_value=fake_pipeline):
            waveform = np.zeros(DEFAULT_SAMPLE_RATE, dtype="float32")
            text = transcribe(waveform)

        self.assertEqual(text, "xin chào")
        fake_pipeline.assert_called_once()
        call_args, call_kwargs = fake_pipeline.call_args
        self.assertEqual(call_args[0]["sampling_rate"], DEFAULT_SAMPLE_RATE)
        self.assertIn("generate_kwargs", call_kwargs)
        self.assertEqual(call_kwargs["generate_kwargs"]["language"], "vi")

    def test_transcribe_empty_waveform_short_circuits(self):
        with patch.object(transcribe_module, "_load_pipeline") as loader:
            text = transcribe(np.zeros(0, dtype="float32"))
        self.assertEqual(text, "")
        loader.assert_not_called()

    def test_non_whisper_model_skips_generate_kwargs(self):
        fake_pipeline = MagicMock(return_value={"text": "hello"})
        with patch.object(transcribe_module, "_load_pipeline", return_value=fake_pipeline):
            waveform = np.zeros(100, dtype="float32")
            transcribe(waveform, model="facebook/wav2vec2-base", language="vi")
        _, call_kwargs = fake_pipeline.call_args
        self.assertNotIn("generate_kwargs", call_kwargs)

    def test_transcribe_handles_string_result(self):
        fake_pipeline = MagicMock(return_value="raw text")
        with patch.object(transcribe_module, "_load_pipeline", return_value=fake_pipeline):
            text = transcribe(np.zeros(100, dtype="float32"))
        self.assertEqual(text, "raw text")

    def test_transcribe_records_when_audio_is_none(self):
        fake_pipeline = MagicMock(return_value={"text": "recorded"})
        fake_recording = np.ones(DEFAULT_SAMPLE_RATE, dtype="float32")
        with patch.object(transcribe_module, "_record_until_silence",
                          return_value=fake_recording) as recorder, \
             patch.object(transcribe_module, "_load_pipeline", return_value=fake_pipeline):
            text = transcribe()

        recorder.assert_called_once()
        self.assertEqual(text, "recorded")


class TestAutoTranscribe(unittest.TestCase):
    def test_auto_transcribe_invokes_recorder(self):
        fake_pipeline = MagicMock(return_value={"text": "hi"})
        fake_recording = np.ones(DEFAULT_SAMPLE_RATE, dtype="float32")
        with patch.object(transcribe_module, "_record_until_silence",
                          return_value=fake_recording) as recorder, \
             patch.object(transcribe_module, "_load_pipeline", return_value=fake_pipeline):
            text = auto_transcribe()
        recorder.assert_called_once()
        self.assertEqual(text, "hi")


if __name__ == "__main__":
    unittest.main()
