# say

Convert Vietnamese text to speech.

!!! note "Requires Additional Setup"
    This function requires extra dependencies and model download:
    ```bash
    pip install "underthesea[voice]"
    underthesea download-model VIET_TTS_V0_4_1
    ```

## Usage

```python
from underthesea.pipeline.say import say

text = "Xin chào Việt Nam"
say(text)
# Generates sound.wav in current directory
```

## Function Signature

```python
def say(text: str, output_file: str = "sound.wav") -> None
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | | The Vietnamese text to convert to speech |
| `output_file` | `str` | `"sound.wav"` | Output audio file path |

## Returns

| Type | Description |
|------|-------------|
| `None` | Writes audio to file |

## Examples

### Basic Usage

```python
from underthesea.pipeline.say import say

# Generate speech
say("Xin chào Việt Nam")
# Creates sound.wav

# Custom output file
say("Hà Nội là thủ đô của Việt Nam", output_file="hanoi.wav")
```

### Command Line Usage

```bash
underthesea say "Xin chào Việt Nam"
# Creates sound.wav

underthesea say "Hà Nội là thủ đô" -o output.wav
# Creates output.wav
```

### Generating Multiple Audio Files

```python
from underthesea.pipeline.say import say

sentences = [
    ("Xin chào", "hello.wav"),
    ("Tạm biệt", "goodbye.wav"),
    ("Cảm ơn", "thanks.wav")
]

for text, filename in sentences:
    say(text, output_file=filename)
    print(f"Generated: {filename}")
```

### Playing Audio (with external library)

```python
from underthesea.pipeline.say import say
import subprocess

# Generate audio
say("Xin chào Việt Nam")

# Play audio (macOS)
subprocess.run(["afplay", "sound.wav"])

# Play audio (Linux with aplay)
# subprocess.run(["aplay", "sound.wav"])
```

## Notes

- Uses the VietTTS model for high-quality Vietnamese speech synthesis
- Output format is WAV audio
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
