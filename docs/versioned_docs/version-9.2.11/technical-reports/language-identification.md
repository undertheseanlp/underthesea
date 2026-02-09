# Language Identification

## Overview

The language identification module in Underthesea provides automatic language detection for text input using the Radar-1 model. Given a text string, the module identifies which of 11 supported languages the text is written in.

**Model:** [undertheseanlp/radar-1](https://huggingface.co/undertheseanlp/radar-1)
**License:** Apache 2.0

## Supported Languages

| Code | Language |
|------|----------|
| vi | Vietnamese |
| en | English |
| zh | Chinese |
| ja | Japanese |
| ko | Korean |
| fr | French |
| de | German |
| es | Spanish |
| th | Thai |
| lo | Lao |
| km | Khmer |

## Architecture

### Radar-1 Model

The Radar-1 model is a text classification model designed for language detection, with a focus on Vietnamese and Southeast Asian languages.

```
Radar-1 Pipeline
├── Text Input
│   └── Raw text string
├── Feature Extraction
│   └── Character and token-level features
├── Classification
│   └── Language prediction with confidence score
└── Output
    ├── Language code (e.g., "vi", "en")
    └── Confidence score (0.0 - 1.0)
```

## Usage

### Basic Usage

```python
from underthesea import lang_detect

text = "Xin chào, tôi là người Việt Nam"
language = lang_detect(text)
print(language)  # vi
```

### Advanced API with Confidence Scores

```python
from radar import RadarLangDetector, detect

# Quick detection
lang = detect("Hello world")
print(lang)  # en

# With confidence scores
detector = RadarLangDetector.load("models/radar-1")
result = detector.predict("Xin chào Việt Nam")
print(result.lang)   # vi
print(result.score)  # 0.98
```

### Multi-language Examples

```python
from underthesea import lang_detect

# Vietnamese
lang_detect("Xin chào, tôi là người Việt Nam")  # vi

# English
lang_detect("Hello, how are you?")  # en

# Japanese
lang_detect("こんにちは世界")  # ja

# Chinese
lang_detect("你好世界")  # zh

# Korean
lang_detect("안녕하세요")  # ko

# Thai
lang_detect("สวัสดีครับ")  # th
```

## Training

```bash
python src/train.py
```

## References

1. [Radar-1 on Hugging Face](https://huggingface.co/undertheseanlp/radar-1)
2. [Underthesea GitHub Repository](https://github.com/undertheseanlp/underthesea)
