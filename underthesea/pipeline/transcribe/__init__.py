"""Auto transcribe voice — high-accuracy Vietnamese ASR.

Built on top of Hugging Face's `automatic-speech-recognition` pipeline,
with defaults tuned for the best Vietnamese accuracy currently available:

- Default model: ``vinai/PhoWhisper-large`` (VinAI Whisper fine-tuned on
  844 h of Vietnamese speech, SOTA WER on most public Vietnamese
  benchmarks).
- Beam search (``num_beams=5``) instead of greedy decoding.
- Automatic 30 s chunking with 5 s stride for long-form audio.
- Auto-detected GPU + fp16 when available.
- Optional Vietnamese text post-processing via
  :func:`underthesea.text_normalize`.

Usage::

    from underthesea.pipeline.transcribe import transcribe

    text = transcribe("hello.wav")        # file
    text = transcribe()                    # mic, auto-stops on silence
    text = transcribe("long.wav", timestamps=True)  # returns chunks
"""
import time
from functools import lru_cache

# PhoWhisper (VinAI) is the SOTA Vietnamese ASR family. We default to the
# largest variant; the registry below lets callers pick smaller checkpoints
# for latency-sensitive setups.
DEFAULT_MODEL = "vinai/PhoWhisper-large"
DEFAULT_SAMPLE_RATE = 16_000

MODEL_REGISTRY = {
    "tiny": "vinai/PhoWhisper-tiny",
    "base": "vinai/PhoWhisper-base",
    "small": "vinai/PhoWhisper-small",
    "medium": "vinai/PhoWhisper-medium",
    "large": "vinai/PhoWhisper-large",
    # Backup options
    "whisper-large-v3": "openai/whisper-large-v3",
    "wav2vec2-vi": "nguyenvulebinh/wav2vec2-base-vietnamese-250h",
}


def _resolve_model(name: str) -> str:
    """Map a short alias to the full Hugging Face model id."""
    if not name:
        return DEFAULT_MODEL
    return MODEL_REGISTRY.get(name, name)


def _select_device():
    """Return (device_index, torch_dtype) for the best available accelerator."""
    try:
        import torch
    except ImportError:
        return -1, None
    if torch.cuda.is_available():
        return 0, torch.float16
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", torch.float16
    return -1, None


@lru_cache(maxsize=4)
def _load_pipeline(model_name: str):
    """Lazy-load the transformers ASR pipeline. Cached per model name."""
    try:
        from transformers import pipeline
    except ImportError as e:
        raise ImportError(
            "transcribe requires extra dependencies. "
            'Install with: pip install "underthesea[voice]" "underthesea[deep]"'
        ) from e

    device, dtype = _select_device()
    kwargs = {"task": "automatic-speech-recognition", "model": model_name}
    if device != -1:
        kwargs["device"] = device
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    return pipeline(**kwargs)


def _read_audio(path: str):
    """Read audio file as mono float32 at DEFAULT_SAMPLE_RATE."""
    try:
        import numpy as np
        import soundfile as sf
    except ImportError as e:
        raise ImportError(
            "transcribe requires soundfile. "
            'Install with: pip install "underthesea[voice]"'
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


def _normalize_vi(text: str) -> str:
    """Apply Vietnamese text normalization. Falls back to raw text on failure."""
    try:
        from underthesea.pipeline.text_normalize import text_normalize
        return text_normalize(text)
    except Exception:
        return text


def transcribe(
    audio=None,
    model: str = DEFAULT_MODEL,
    language: str = "vi",
    num_beams: int = 5,
    chunk_length_s: float = 30.0,
    stride_length_s: float = 5.0,
    timestamps: bool = False,
    normalize: bool = True,
):
    """Transcribe spoken Vietnamese to text with high accuracy.

    Args:
        audio: Path to an audio file, a numpy waveform (mono float32 at 16 kHz),
            or ``None`` to record from the default microphone until the speaker
            stops talking.
        model: Hugging Face model id, or a short alias from
            :data:`MODEL_REGISTRY` (``"tiny"``, ``"base"``, ``"small"``,
            ``"medium"``, ``"large"``). Defaults to ``vinai/PhoWhisper-large``.
        language: ISO language code passed to Whisper-family models.
        num_beams: Beam-search width. Larger = more accurate, slower.
        chunk_length_s: Window length (seconds) for long-form audio.
        stride_length_s: Overlap (seconds) between adjacent chunks.
        timestamps: When ``True``, return a dict with ``text`` and
            ``chunks`` (each chunk has ``text`` and ``timestamp``).
        normalize: Apply :func:`underthesea.text_normalize` to the output.

    Returns:
        ``str`` with the transcript, or ``dict`` when ``timestamps=True``.
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
        return {"text": "", "chunks": []} if timestamps else ""

    resolved = _resolve_model(model)
    asr = _load_pipeline(resolved)

    inputs = {"array": waveform, "sampling_rate": sr}
    call_kwargs = {}

    # Long-form chunking applies to Whisper-style models. wav2vec2 pipelines
    # also accept chunk_length_s, so this is safe across backends.
    if chunk_length_s and chunk_length_s > 0:
        call_kwargs["chunk_length_s"] = chunk_length_s
        if stride_length_s and stride_length_s > 0:
            call_kwargs["stride_length_s"] = stride_length_s

    if timestamps:
        call_kwargs["return_timestamps"] = True

    # Decoding hints — beam search + language/task — only meaningful for
    # encoder-decoder Whisper-family checkpoints.
    is_whisper = "whisper" in resolved.lower()
    if is_whisper:
        generate_kwargs = {}
        if language:
            generate_kwargs["language"] = language
            generate_kwargs["task"] = "transcribe"
        if num_beams and num_beams > 1:
            generate_kwargs["num_beams"] = num_beams
        if generate_kwargs:
            call_kwargs["generate_kwargs"] = generate_kwargs

    result = asr(inputs, **call_kwargs)

    if isinstance(result, dict):
        text = (result.get("text") or "").strip()
        if normalize and text:
            text = _normalize_vi(text)
        if timestamps:
            return {"text": text, "chunks": result.get("chunks", [])}
        return text
    text = str(result).strip()
    if normalize and text:
        text = _normalize_vi(text)
    return text


# Alias for symmetry with `tts` / `say`.
listen = transcribe


def auto_transcribe(
    outfile: str = None,
    model: str = DEFAULT_MODEL,
    language: str = "vi",
    **kwargs,
):
    """Record from microphone, transcribe, and optionally save the recording."""
    waveform = _record_until_silence()
    if outfile and len(waveform) > 0:
        import soundfile as sf
        sf.write(outfile, waveform, DEFAULT_SAMPLE_RATE)
    text = transcribe(waveform, model=model, language=language, **kwargs)
    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    display = text["text"] if isinstance(text, dict) else text
    print(f"\n{time_str}: {display}")
    return text


if __name__ == "__main__":
    print(auto_transcribe())
