# Text-to-Speech Tutorial

Learn how to convert Vietnamese text to speech.

## Installation

Text-to-speech requires additional setup:

```bash
# Install dependencies
pip install "underthesea[voice]"

# Download the TTS model
underthesea download-model VIET_TTS_V0_4_1
```

## Basic Usage

```python
from underthesea.pipeline.say import say

text = "Xin chào Việt Nam"
say(text)
# Creates sound.wav in current directory
```

## Command Line Usage

```bash
# Basic usage
underthesea say "Xin chào Việt Nam"

# Custom output file
underthesea say "Hà Nội là thủ đô của Việt Nam" -o output.wav
```

## Custom Output File

```python
from underthesea.pipeline.say import say

# Save to specific file
say("Xin chào", output_file="hello.wav")
say("Tạm biệt", output_file="goodbye.wav")
```

## Practical Applications

### Audio Book Generator

```python
from underthesea.pipeline.say import say
from underthesea import sent_tokenize
import os

def generate_audiobook(text, output_dir="audiobook"):
    """Generate audio files for each sentence."""
    os.makedirs(output_dir, exist_ok=True)

    sentences = sent_tokenize(text)
    audio_files = []

    for i, sentence in enumerate(sentences):
        if sentence.strip():
            filename = f"{output_dir}/sentence_{i+1:03d}.wav"
            say(sentence, output_file=filename)
            audio_files.append(filename)
            print(f"Generated: {filename}")

    return audio_files

text = """
Việt Nam là một quốc gia xinh đẹp.
Thủ đô của Việt Nam là Hà Nội.
Ẩm thực Việt Nam nổi tiếng trên thế giới.
"""

files = generate_audiobook(text)
print(f"Generated {len(files)} audio files")
```

### Vocabulary Pronunciation Guide

```python
from underthesea.pipeline.say import say
import os

def create_pronunciation_guide(words, output_dir="pronunciation"):
    """Create audio files for vocabulary pronunciation."""
    os.makedirs(output_dir, exist_ok=True)

    for word in words:
        # Clean filename
        filename = word.replace(" ", "_").lower()
        filepath = f"{output_dir}/{filename}.wav"

        say(word, output_file=filepath)
        print(f"Created: {filepath}")

vocabulary = [
    "Xin chào",
    "Cảm ơn",
    "Tạm biệt",
    "Xin lỗi",
    "Không sao"
]

create_pronunciation_guide(vocabulary)
```

### Notification System

```python
from underthesea.pipeline.say import say
import subprocess
import platform

def speak_notification(message):
    """Speak a notification message."""
    output_file = "/tmp/notification.wav"
    say(message, output_file=output_file)

    # Play audio based on platform
    system = platform.system()
    if system == "Darwin":  # macOS
        subprocess.run(["afplay", output_file])
    elif system == "Linux":
        subprocess.run(["aplay", output_file])
    elif system == "Windows":
        import winsound
        winsound.PlaySound(output_file, winsound.SND_FILENAME)

# Usage
speak_notification("Bạn có một tin nhắn mới")
speak_notification("Cuộc họp sẽ bắt đầu sau 5 phút")
```

### Language Learning App

```python
from underthesea.pipeline.say import say
from underthesea import translate
import os

def create_learning_material(vietnamese_phrases, output_dir="learning"):
    """Create learning materials with audio and translations."""
    os.makedirs(output_dir, exist_ok=True)

    materials = []

    for i, phrase in enumerate(vietnamese_phrases):
        # Generate audio
        audio_file = f"{output_dir}/phrase_{i+1}.wav"
        say(phrase, output_file=audio_file)

        # Get translation
        english = translate(phrase)

        materials.append({
            'vietnamese': phrase,
            'english': english,
            'audio': audio_file
        })

        print(f"Created: {phrase} -> {english}")

    return materials

phrases = [
    "Bạn khỏe không?",
    "Tôi rất vui được gặp bạn",
    "Hôm nay thời tiết đẹp quá"
]

materials = create_learning_material(phrases)
```

### Batch Audio Generation

```python
from underthesea.pipeline.say import say
import os
from concurrent.futures import ThreadPoolExecutor

def generate_audio_batch(texts, output_dir="batch_audio"):
    """Generate audio files in batch."""
    os.makedirs(output_dir, exist_ok=True)

    def generate_one(args):
        i, text = args
        filename = f"{output_dir}/audio_{i+1:03d}.wav"
        say(text, output_file=filename)
        return filename

    # Note: For true parallel processing, you might need
    # to handle model loading carefully
    results = []
    for i, text in enumerate(texts):
        result = generate_one((i, text))
        results.append(result)
        print(f"Generated: {result}")

    return results

texts = [
    "Đây là câu thứ nhất",
    "Đây là câu thứ hai",
    "Đây là câu thứ ba"
]

files = generate_audio_batch(texts)
```

## Tips for Better Audio

1. **Punctuation**: Use proper punctuation for natural pauses
2. **Sentence Length**: Shorter sentences produce cleaner audio
3. **Text Normalization**: Normalize text before generating audio

```python
from underthesea.pipeline.say import say
from underthesea import text_normalize

def quality_say(text, output_file="sound.wav"):
    """Generate audio with text preprocessing."""
    # Normalize text first
    normalized = text_normalize(text)

    # Generate audio
    say(normalized, output_file=output_file)

quality_say("Ðây là văn bản cần chuẩn hoá")
```

## Troubleshooting

### Model Not Found

```bash
# Download the model
underthesea download-model VIET_TTS_V0_4_1
```

### No Audio Output

- Check that the output file was created
- Verify audio player is installed on your system
- Check file permissions in the output directory

### Audio Quality Issues

- Ensure input text is proper Vietnamese
- Use complete sentences with punctuation
- Avoid very long sentences

## Next Steps

- Learn about [Language Detection](../api/lang_detect.md)
- Explore [Translation](translation.md)
- See the [API Reference](../api/say.md)
