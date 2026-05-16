# transcribe

Auto transcribe voice — convert spoken Vietnamese audio into text.

!!! note "Requires Additional Setup"
    This function requires extra dependencies:
    ```bash
    pip install "underthesea[voice]" "underthesea[deep]"
    # Optional, for live microphone input:
    pip install sounddevice
    ```

    The first call downloads the chosen Hugging Face model
    (default: `openai/whisper-small`).

## Usage

```python
from underthesea.pipeline.transcribe import transcribe

# From an audio file
text = transcribe("hello.wav")

# From the default microphone (auto-stops after ~1.5s of silence)
text = transcribe()
```

## Function Signature

```python
def transcribe(
    audio: str | np.ndarray | None = None,
    model: str = "openai/whisper-small",
    language: str = "vi",
) -> str
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `audio` | `str \| np.ndarray \| None` | `None` | Path to audio file, a mono float32 waveform at 16 kHz, or `None` to record from the microphone |
| `model` | `str` | `"openai/whisper-small"` | Hugging Face ASR model name |
| `language` | `str` | `"vi"` | Language hint for whisper-style models |

## Returns

| Type | Description |
|------|-------------|
| `str` | The transcribed text |

## Examples

### Transcribe an existing file

```python
from underthesea.pipeline.transcribe import transcribe

text = transcribe("xin_chao.wav")
print(text)
```

### Auto record from the microphone

```python
from underthesea.pipeline.transcribe import transcribe

# Records until you stop talking, then transcribes
text = transcribe()
print(text)
```

### Save the recording while transcribing

```python
from underthesea.pipeline.transcribe import auto_transcribe

text = auto_transcribe(outfile="recording.wav")
```

### Use a different model

```python
from underthesea.pipeline.transcribe import transcribe

# Use a Vietnamese wav2vec2 model
text = transcribe("audio.wav", model="nguyenvulebinh/wav2vec2-base-vietnamese-250h")
```

### Command Line Usage

```bash
# Transcribe an audio file
underthesea transcribe input.wav

# Record from microphone and transcribe (auto-stops on silence)
underthesea transcribe

# Record, transcribe, and save the recording
underthesea transcribe -o recording.wav
```

## Notes

- Mono input is expected; stereo files are automatically downmixed.
- Non-16 kHz files are linearly resampled to 16 kHz.
- Microphone input requires `sounddevice` (and a working audio backend).
- The first call downloads and loads the model, which may take a while.
