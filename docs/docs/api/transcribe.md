# transcribe

Auto transcribe voice — convert spoken Vietnamese audio into text with
the highest accuracy currently available.

!!! note "Requires Additional Setup"
    ```bash
    pip install "underthesea[voice]" "underthesea[deep]"
    # Optional, for live microphone input:
    pip install sounddevice
    ```

    The first call downloads the chosen Hugging Face model.

## Why this is accurate

- **Default model: `vinai/PhoWhisper-large`** — VinAI's Whisper-large
  fine-tuned on 844 hours of Vietnamese speech. State-of-the-art WER on
  most public Vietnamese benchmarks (VIVOS, Common Voice, VLSP-2020 T1,
  VLSP-2021 T1, FOSD, FLEURS, INFORE).
- **Beam search** (`num_beams=5`) instead of greedy decoding.
- **Long-form chunking** (`chunk_length_s=30`, `stride_length_s=5`) so
  arbitrarily long files are handled with overlap-and-merge.
- **GPU + fp16** auto-detected; falls back to CPU.
- **Vietnamese text normalization** applied to the output by default.

## Usage

```python
from underthesea.pipeline.transcribe import transcribe

# From an audio file (any length, any sample rate)
text = transcribe("hello.wav")

# From the default microphone — auto-stops after ~1.5 s of silence
text = transcribe()

# Long-form with timestamps
result = transcribe("podcast.mp3", timestamps=True)
print(result["text"])
for chunk in result["chunks"]:
    print(chunk["timestamp"], chunk["text"])
```

## Function Signature

```python
def transcribe(
    audio: str | np.ndarray | None = None,
    model: str = "vinai/PhoWhisper-large",
    language: str = "vi",
    num_beams: int = 5,
    chunk_length_s: float = 30.0,
    stride_length_s: float = 5.0,
    timestamps: bool = False,
    normalize: bool = True,
) -> str | dict
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `audio` | `str \| np.ndarray \| None` | `None` | Path to file, mono float32 waveform @ 16 kHz, or `None` for mic |
| `model` | `str` | `"vinai/PhoWhisper-large"` | HF model id, or short alias (see registry) |
| `language` | `str` | `"vi"` | Language hint for whisper-style models |
| `num_beams` | `int` | `5` | Beam-search width — larger = more accurate, slower |
| `chunk_length_s` | `float` | `30.0` | Window length for long-form audio |
| `stride_length_s` | `float` | `5.0` | Overlap between adjacent chunks |
| `timestamps` | `bool` | `False` | Return `{"text", "chunks"}` with timestamps |
| `normalize` | `bool` | `True` | Apply `underthesea.text_normalize` on the result |

## Model Registry

| Alias | Hugging Face ID | Notes |
|-------|-----------------|-------|
| `tiny` | `vinai/PhoWhisper-tiny` | Fastest, lowest accuracy |
| `base` | `vinai/PhoWhisper-base` | |
| `small` | `vinai/PhoWhisper-small` | |
| `medium` | `vinai/PhoWhisper-medium` | |
| `large` | `vinai/PhoWhisper-large` | **Default — best Vietnamese WER** |
| `whisper-large-v3` | `openai/whisper-large-v3` | Multilingual fallback |
| `wav2vec2-vi` | `nguyenvulebinh/wav2vec2-base-vietnamese-250h` | CTC, no LM |

You can also pass any Hugging Face ASR model id directly.

## CLI

```bash
# Transcribe a file with PhoWhisper-large
underthesea transcribe input.wav

# Pick a smaller variant for speed
underthesea transcribe input.wav -m small

# Record from mic, save WAV, transcribe with timestamps
underthesea transcribe -o rec.wav --timestamps
```

## Notes

- Stereo files are automatically downmixed to mono.
- Non-16 kHz files are resampled to 16 kHz.
- Microphone input requires `sounddevice` and a working audio backend.
- The first call downloads and loads the model — be patient.
- Credits: PhoWhisper by [VinAI Research](https://github.com/VinAIResearch/PhoWhisper).
