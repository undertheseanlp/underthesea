# tts

Convert Vietnamese text to speech.

!!! note "Requires Additional Setup"
    This function requires extra dependencies and model download:
    ```bash
    pip install "underthesea[voice]"
    underthesea download-model VIET_TTS_V0_4_1
    ```

## Usage

```python
from underthesea.pipeline.tts import tts

text = "Xin chào Việt Nam"
tts(text)
# Generates sound.wav in current directory
```

## Function Signature

```python
def tts(text: str, outfile: str = "sound.wav", play: bool = False) -> tuple[int, np.ndarray]
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | | The Vietnamese text to convert to speech |
| `outfile` | `str` | `"sound.wav"` | Output audio file path |
| `play` | `bool` | `False` | Whether to play audio after generation |

## Returns

| Type | Description |
|------|-------------|
| `tuple[int, np.ndarray]` | Sample rate (16000) and audio waveform array |

## Examples

### Basic Usage

```python
from underthesea.pipeline.tts import tts

# Generate speech
tts("Xin chào Việt Nam")
# Creates sound.wav

# Custom output file
tts("Hà Nội là thủ đô của Việt Nam", outfile="hanoi.wav")

# Generate and play immediately
tts("Xin chào", play=True)
```

### Command Line Usage

```bash
underthesea tts "Xin chào Việt Nam"
# Creates sound.wav and plays it
```

### Generating Multiple Audio Files

```python
from underthesea.pipeline.tts import tts

sentences = [
    ("Xin chào", "hello.wav"),
    ("Tạm biệt", "goodbye.wav"),
    ("Cảm ơn", "thanks.wav")
]

for text, filename in sentences:
    tts(text, outfile=filename)
    print(f"Generated: {filename}")
```

### Playing Audio (with external library)

```python
from underthesea.pipeline.tts import tts
import subprocess

# Generate audio
tts("Xin chào Việt Nam")

# Play audio (macOS)
subprocess.run(["afplay", "sound.wav"])

# Play audio (Linux with aplay)
# subprocess.run(["aplay", "sound.wav"])
```

## Notes

- Uses the VietTTS model for high-quality Vietnamese speech synthesis
- Output format is WAV audio at 16kHz sample rate
- First call may take longer due to model loading
- Requires downloading the TTS model before first use
- Credits: Based on [NTT123/vietTTS](https://github.com/ntt123/vietTTS)

## Troubleshooting

### Model Not Found

If you get a model not found error:

```bash
underthesea download-model VIET_TTS_V0_4_1
```

### Audio Quality Issues

- Ensure input text is in Vietnamese
- Longer sentences produce smoother audio
- Punctuation affects prosody
