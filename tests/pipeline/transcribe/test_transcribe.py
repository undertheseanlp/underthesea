"""Tests for the auto-transcribe-voice pipeline."""
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from underthesea.pipeline import transcribe as transcribe_module
from underthesea.pipeline.transcribe import (
    DEFAULT_MODEL,
    DEFAULT_SAMPLE_RATE,
    MODEL_REGISTRY,
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

    def test_default_model_is_phowhisper_large(self):
        # PhoWhisper-large is currently the highest-accuracy Vietnamese ASR.
        self.assertEqual(DEFAULT_MODEL, "vinai/PhoWhisper-large")

    def test_model_registry_exposes_phowhisper_aliases(self):
        for alias in ("tiny", "base", "small", "medium", "large"):
            self.assertIn(alias, MODEL_REGISTRY)
            self.assertTrue(MODEL_REGISTRY[alias].startswith("vinai/PhoWhisper"))

    def test_top_level_optional_import(self):
        import underthesea
        self.assertIn("transcribe", underthesea.__all__)


class TestTranscribeWithMockedPipeline(unittest.TestCase):
    def setUp(self):
        transcribe_module._load_pipeline.cache_clear()

    def _patch_pipeline(self, return_value):
        fake = MagicMock(return_value=return_value)
        return fake, patch.object(transcribe_module, "_load_pipeline", return_value=fake)

    def test_transcribe_from_numpy_uses_beam_search_and_chunking(self):
        fake, ctx = self._patch_pipeline({"text": "  xin chào việt nam  "})
        with ctx, patch.object(transcribe_module, "_normalize_vi", side_effect=lambda x: x):
            waveform = np.zeros(DEFAULT_SAMPLE_RATE, dtype="float32")
            text = transcribe(waveform)

        self.assertEqual(text, "xin chào việt nam")
        fake.assert_called_once()
        _args, call_kwargs = fake.call_args
        # Long-form chunking enabled by default
        self.assertEqual(call_kwargs["chunk_length_s"], 30.0)
        self.assertEqual(call_kwargs["stride_length_s"], 5.0)
        # Whisper-family generate_kwargs
        gk = call_kwargs["generate_kwargs"]
        self.assertEqual(gk["language"], "vi")
        self.assertEqual(gk["task"], "transcribe")
        self.assertEqual(gk["num_beams"], 5)

    def test_transcribe_empty_waveform_short_circuits(self):
        with patch.object(transcribe_module, "_load_pipeline") as loader:
            text = transcribe(np.zeros(0, dtype="float32"))
        self.assertEqual(text, "")
        loader.assert_not_called()

    def test_transcribe_empty_waveform_with_timestamps_returns_dict(self):
        with patch.object(transcribe_module, "_load_pipeline") as loader:
            result = transcribe(np.zeros(0, dtype="float32"), timestamps=True)
        self.assertEqual(result, {"text": "", "chunks": []})
        loader.assert_not_called()

    def test_non_whisper_model_skips_generate_kwargs(self):
        fake, ctx = self._patch_pipeline({"text": "hello"})
        with ctx, patch.object(transcribe_module, "_normalize_vi", side_effect=lambda x: x):
            transcribe(np.zeros(100, dtype="float32"), model="wav2vec2-vi")
        _args, call_kwargs = fake.call_args
        self.assertNotIn("generate_kwargs", call_kwargs)
        # Chunking is still applied for long-form audio
        self.assertIn("chunk_length_s", call_kwargs)

    def test_alias_resolves_to_phowhisper_id(self):
        fake, ctx = self._patch_pipeline({"text": "alo"})
        with ctx as loader, patch.object(transcribe_module, "_normalize_vi",
                                         side_effect=lambda x: x):
            transcribe(np.zeros(100, dtype="float32"), model="base")
        loader.assert_called_once_with("vinai/PhoWhisper-base")

    def test_timestamps_returns_chunks(self):
        chunks = [{"text": "xin", "timestamp": [0.0, 0.5]},
                  {"text": " chào", "timestamp": [0.5, 1.0]}]
        fake, ctx = self._patch_pipeline({"text": "xin chào", "chunks": chunks})
        with ctx, patch.object(transcribe_module, "_normalize_vi", side_effect=lambda x: x):
            result = transcribe(np.zeros(100, dtype="float32"), timestamps=True)
        self.assertEqual(result["text"], "xin chào")
        self.assertEqual(result["chunks"], chunks)
        _args, call_kwargs = fake.call_args
        self.assertTrue(call_kwargs["return_timestamps"])

    def test_transcribe_handles_string_result(self):
        fake, ctx = self._patch_pipeline("raw text")
        with ctx, patch.object(transcribe_module, "_normalize_vi", side_effect=lambda x: x):
            text = transcribe(np.zeros(100, dtype="float32"))
        self.assertEqual(text, "raw text")

    def test_transcribe_records_when_audio_is_none(self):
        fake, ctx = self._patch_pipeline({"text": "recorded"})
        fake_recording = np.ones(DEFAULT_SAMPLE_RATE, dtype="float32")
        with patch.object(transcribe_module, "_record_until_silence",
                          return_value=fake_recording) as recorder, \
             ctx, patch.object(transcribe_module, "_normalize_vi",
                               side_effect=lambda x: x):
            text = transcribe()

        recorder.assert_called_once()
        self.assertEqual(text, "recorded")

    def test_normalize_is_applied_by_default(self):
        fake, ctx = self._patch_pipeline({"text": "xin chao"})
        with ctx, patch.object(transcribe_module, "_normalize_vi",
                               return_value="XIN CHAO") as norm:
            text = transcribe(np.zeros(100, dtype="float32"))
        norm.assert_called_once_with("xin chao")
        self.assertEqual(text, "XIN CHAO")

    def test_normalize_can_be_disabled(self):
        fake, ctx = self._patch_pipeline({"text": "xin chao"})
        with ctx, patch.object(transcribe_module, "_normalize_vi") as norm:
            text = transcribe(np.zeros(100, dtype="float32"), normalize=False)
        norm.assert_not_called()
        self.assertEqual(text, "xin chao")

    def test_num_beams_one_skips_beam_kwarg(self):
        fake, ctx = self._patch_pipeline({"text": "hi"})
        with ctx, patch.object(transcribe_module, "_normalize_vi", side_effect=lambda x: x):
            transcribe(np.zeros(100, dtype="float32"), num_beams=1)
        _args, call_kwargs = fake.call_args
        self.assertNotIn("num_beams", call_kwargs.get("generate_kwargs", {}))


class TestAutoTranscribe(unittest.TestCase):
    def test_auto_transcribe_invokes_recorder(self):
        fake_pipeline = MagicMock(return_value={"text": "hi"})
        fake_recording = np.ones(DEFAULT_SAMPLE_RATE, dtype="float32")
        with patch.object(transcribe_module, "_record_until_silence",
                          return_value=fake_recording) as recorder, \
             patch.object(transcribe_module, "_load_pipeline", return_value=fake_pipeline), \
             patch.object(transcribe_module, "_normalize_vi", side_effect=lambda x: x):
            text = auto_transcribe()
        recorder.assert_called_once()
        self.assertEqual(text, "hi")


if __name__ == "__main__":
    unittest.main()
