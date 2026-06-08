import time
from os.path import join

import numpy as np

from underthesea.file_utils import MODELS_FOLDER

# Legacy VietTTS V0.4.1 model location (used only by the "viettts" backend).
model_path = join(MODELS_FOLDER, "VIET_TTS_V0_4_1")

# Lazily-created VieNeu-TTS singleton (default backend).
_VIENEU = None


def _get_vieneu():
    """Load VieNeu-TTS v3 Turbo once and cache it.

    On CPU it runs torch-free via ONNX Runtime; on GPU it uses PyTorch — both are
    selected automatically by ``Vieneu()``.
    """
    global _VIENEU
    if _VIENEU is None:
        try:
            from vieneu import Vieneu
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "The 'vieneu' TTS backend requires the vieneu package. "
                "Install it with `pip install underthesea[speech]` (or `pip install vieneu`)."
            ) from e
        _VIENEU = Vieneu()
    return _VIENEU


def _vieneu_synthesize(text, voice=None, ref_audio=None, **kwargs):
    """VieNeu-TTS v3 Turbo → (sample_rate, int16 waveform)."""
    engine = _get_vieneu()
    wav = engine.infer(text, voice=voice, ref_audio=ref_audio, **kwargs)
    y = (np.clip(np.asarray(wav, dtype=np.float32), -1.0, 1.0) * 32767).astype(np.int16)
    return engine.sample_rate, y


def _viettts_synthesize(text):
    """Legacy VietTTS V0.4.1 (JAX/haiku) → (16000, int16 waveform).

    Imported lazily so the default vieneu backend does not pull in jax/dm-haiku.
    """
    from underthesea.pipeline.tts.viettts_ import nat_normalize_text
    from underthesea.pipeline.tts.viettts_.hifigan.mel2wave import mel2wave
    from underthesea.pipeline.tts.viettts_.nat.text2mel import text2mel

    # prevent too long text
    if len(text) > 500:
        text = text[:500]
    text = nat_normalize_text(text)
    mel = text2mel(
        text,
        join(model_path, "lexicon.txt"),
        0.2,
        join(model_path, "acoustic_latest_ckpt.pickle"),
        join(model_path, "duration_latest_ckpt.pickle"),
    )
    wave = mel2wave(mel, join(model_path, "config.json"), join(model_path, "hk_hifi.pickle"))
    return 16_000, (wave * (2**15)).astype(np.int16)


def _synthesize(text, backend="vieneu", voice=None, ref_audio=None, **kwargs):
    if backend == "vieneu":
        return _vieneu_synthesize(text, voice=voice, ref_audio=ref_audio, **kwargs)
    if backend == "viettts":
        return _viettts_synthesize(text)
    raise ValueError(f"Unknown TTS backend: {backend!r}. Use 'vieneu' or 'viettts'.")


def text_to_speech(text, backend="vieneu", voice=None, ref_audio=None, **kwargs):
    """Synthesize speech and return an ``int16`` numpy waveform.

    Args:
        text: the text to speak.
        backend: ``"vieneu"`` (default) — VieNeu-TTS v3 Turbo: 48 kHz, many
            built-in voices, instant voice cloning. ``"viettts"`` — legacy
            VietTTS V0.4.1 (16 kHz, needs the ``[voice]`` extra and the
            ``VIET_TTS_V0_4_1`` model).
        voice: built-in voice name for the vieneu backend (e.g. ``"Ngọc Linh"``).
        ref_audio: path to a 3–5 s reference clip to clone (vieneu backend).
        **kwargs: forwarded to ``Vieneu.infer`` (temperature, top_k, ...).

    Note: the vieneu backend returns 48 kHz audio (vs. 16 kHz for viettts); use
    :func:`tts` if you need the matching sample rate written to disk.
    """
    return _synthesize(text, backend=backend, voice=voice, ref_audio=ref_audio, **kwargs)[1]


def tts(text, outfile="sound.wav", play=False, backend="vieneu", voice=None, ref_audio=None, **kwargs):
    """Synthesize ``text`` to ``outfile`` and optionally play it.

    Returns ``(sample_rate, waveform)``. See :func:`text_to_speech` for backend
    options (default ``"vieneu"``).
    """
    sr, y = _synthesize(text, backend=backend, voice=voice, ref_audio=ref_audio, **kwargs)
    import soundfile as sf
    sf.write(outfile, y, sr)
    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"\n{time_str}: {text}")
    if play:
        from playsound3 import playsound
        playsound(outfile)
    return sr, y


# Alias for backward compatibility
say = tts


def think_and_tts(text, outfile="sound.wav"):
    # create two processes, one for thinking, one for tts
    import multiprocessing
    p1 = multiprocessing.Process(target=terminal_thinking)
    p2 = multiprocessing.Process(target=tts, args=(text, outfile))
    p1.start()
    p2.start()
    # while p2 is end, terminate p1
    p2.join()
    p1.terminate()
    from playsound3 import playsound
    playsound(outfile)


def terminal_thinking():
    # each 300ms, print a dot
    import time
    while True:
        print(".", end="")
        time.sleep(0.5)


if __name__ == '__main__':
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    think_and_tts("xin chào")
    think_and_tts("Có đấy. Tình yêu giống như một ly kem, nếu không ăn nó đúng lúc, nó sẽ tan chảy. Haha, anh thấy thế nào?")
