"""Auto transcribe voice (Vietnamese ASR).

Lightweight wrapper around a Hugging Face automatic-speech-recognition
pipeline that turns spoken Vietnamese into text. Supports transcribing
existing audio files and auto-recording from the microphone until the
speaker stops talking.

Usage:
    from underthesea.pipeline.transcribe import transcribe

    text = transcribe("hello.wav")
    text = transcribe()  # records from microphone and auto-stops
"""
import time
from functools import lru_cache

DEFAULT_MODEL = "openai/whisper-small"
DEFAULT_SAMPLE_RATE = 16_000


@lru_cache(maxsize=4)
def _load_pipeline(model_name: str):
    """Lazy-load the transformers ASR pipeline. Cached per model name."""
    try:
        from transformers import pipeline
    except ImportError as e:
        raise ImportError(
            "transcribe requires extra dependencies. "
            "Install with: pip install \"underthesea[voice]\" \"underthesea[deep]\""
        ) from e
    return pipeline("automatic-speech-recognition", model=model_name)


def _read_audio(path: str):
    """Read audio file as mono float32 at DEFAULT_SAMPLE_RATE."""
    try:
        import numpy as np
        import soundfile as sf
    except ImportError as e:
        raise ImportError(
            "transcribe requires soundfile. "
            "Install with: pip install \"underthesea[voice]\""
        ) from e
    data, sr = sf.read(path, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != DEFAULT_SAMPLE_RATE:
        # Lightweight linear resample to avoid pulling in scipy / librosa.
        ratio = DEFAULT_SAMPLE_RATE / sr
        new_len = int(round(len(data) * ratio))
        if new_len > 0:
            xp = np.linspace(0.0, 1.0, num=len(data), endpoint=False)
            x = np.linspace(0.0, 1.0, num=new_len, endpoint=False)
            data = np.interp(x, xp, data).astype("float32")
        sr = DEFAULT_SAMPLE_RATE
    return data, sr


def _record_until_silence(
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    silence_threshold: float = 0.01,
    silence_duration: float = 1.5,
    max_duration: float = 30.0,
):
    """Record from microphone and stop after `silence_duration` of quiet."""
    try:
        import numpy as np
        import sounddevice as sd
    except ImportError as e:
        raise ImportError(
            "Microphone recording requires sounddevice. "
            "Install with: pip install sounddevice"
        ) from e

    chunk = int(sample_rate * 0.1)  # 100 ms windows
    silent_windows_needed = int(silence_duration / 0.1)
    max_windows = int(max_duration / 0.1)

    buffer = []
    silent_count = 0
    spoke = False

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32") as stream:
        for _ in range(max_windows):
            data, _overflow = stream.read(chunk)
            window = data[:, 0]
            buffer.append(window)
            rms = float(np.sqrt(np.mean(window ** 2)))
            if rms > silence_threshold:
                spoke = True
                silent_count = 0
            elif spoke:
                silent_count += 1
                if silent_count >= silent_windows_needed:
                    break

    return np.concatenate(buffer) if buffer else np.zeros(0, dtype="float32")


def transcribe(
    audio=None,
    model: str = DEFAULT_MODEL,
    language: str = "vi",
):
    """Transcribe spoken Vietnamese to text.

    Args:
        audio: Path to an audio file, a numpy waveform (mono float32 at 16 kHz),
            or None to record from the default microphone until the speaker
            stops talking.
        model: Hugging Face ASR model name. Defaults to whisper-small.
        language: ISO language code passed to whisper-style models.

    Returns:
        The transcribed text as a string.
    """
    if audio is None:
        waveform = _record_until_silence()
        sr = DEFAULT_SAMPLE_RATE
    elif isinstance(audio, str):
        waveform, sr = _read_audio(audio)
    else:
        waveform = audio
        sr = DEFAULT_SAMPLE_RATE

    if len(waveform) == 0:
        return ""

    asr = _load_pipeline(model)

    inputs = {"array": waveform, "sampling_rate": sr}
    kwargs = {}
    # Whisper-family models accept generate_kwargs for language hints.
    if "whisper" in model.lower() and language:
        kwargs["generate_kwargs"] = {"language": language, "task": "transcribe"}

    result = asr(inputs, **kwargs)
    if isinstance(result, dict):
        return result.get("text", "").strip()
    return str(result).strip()


# Alias for symmetry with `tts` / `say`.
listen = transcribe


def auto_transcribe(outfile: str = None, model: str = DEFAULT_MODEL, language: str = "vi"):
    """Record from microphone, transcribe, and optionally save the recording."""
    waveform = _record_until_silence()
    if outfile and len(waveform) > 0:
        import soundfile as sf
        sf.write(outfile, waveform, DEFAULT_SAMPLE_RATE)
    text = transcribe(waveform, model=model, language=language)
    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"\n{time_str}: {text}")
    return text


if __name__ == "__main__":
    print(auto_transcribe())
