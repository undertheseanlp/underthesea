import inspect
from unittest.mock import patch

import pytest


def test_invalid_backend_raises_value_error():
    from underthesea.pipeline.tts import text_to_speech
    with pytest.raises(ValueError, match="Unknown TTS backend 'nonexistent'"):
        text_to_speech("hello", backend="nonexistent")


def test_vieneu_missing_raises_import_error():
    import builtins
    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "vieneu":
            raise ImportError("No module named 'vieneu'")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        from underthesea.pipeline.tts import text_to_speech
        with pytest.raises(ImportError, match="pip install underthesea\\[voice-vieneu\\]"):
            text_to_speech("hello", backend="vieneu")


def test_viettts_backend_is_default():
    from underthesea.pipeline.tts import text_to_speech, tts
    sig_tts = inspect.signature(tts)
    sig_t2s = inspect.signature(text_to_speech)
    assert sig_tts.parameters["backend"].default == "viettts"
    assert sig_t2s.parameters["backend"].default == "viettts"


def test_valid_backends_constant():
    from underthesea.pipeline.tts import _VALID_BACKENDS
    assert _VALID_BACKENDS == ("viettts", "vieneu")


def test_cli_backend_option():
    from click.testing import CliRunner
    from underthesea.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["tts", "--help"])
    assert "--backend" in result.output
    assert "viettts" in result.output
    assert "vieneu" in result.output
