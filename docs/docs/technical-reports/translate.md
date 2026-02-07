# Machine Translation

## Overview

The machine translation module in Underthesea provides bidirectional Vietnamese-English translation using the EnviT5 transformer model from VietAI.

**Model:** [VietAI/envit5-translation](https://huggingface.co/VietAI/envit5-translation)

**Requirements:**
```bash
pip install "underthesea[deep]"
```

## Architecture

### EnviT5 Translator

```
Translation Pipeline
├── Text Input
│   └── Source language text
├── Preprocessing
│   └── Language prefix: "{lang}: {text}"
├── EnviT5 Model
│   ├── AutoTokenizer
│   └── AutoModelForSeq2SeqLM
│       └── Beam search (num_beams=5)
└── Output
    └── Translated text
```

### Model Configuration

| Parameter | Value |
|-----------|-------|
| Model | VietAI/envit5-translation |
| Architecture | T5 (Seq2Seq) |
| Beam search | num_beams=5 |
| Max length | 512 tokens |
| Languages | Vietnamese (vi), English (en) |

## Usage

### Vietnamese to English (Default)

```python
from underthesea import translate

result = translate("Hà Nội là thủ đô của Việt Nam")
# 'Hanoi is the capital of Vietnam'
```

### English to Vietnamese

```python
translate("I love Vietnamese food", source_lang='en', target_lang='vi')
# 'Tôi yêu ẩm thực Việt Nam'
```

### Document Translation

For long texts, combine with sentence tokenization:

```python
from underthesea import sent_tokenize, translate

text = "Hà Nội là thủ đô. Thành phố rất đẹp."
sentences = sent_tokenize(text)
translated = [translate(s) for s in sentences]
```

### Function Signature

```python
def translate(
    text: str,
    source_lang: str = 'vi',
    target_lang: str = 'en'
) -> str
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | str | required | Text to translate |
| `source_lang` | str | `'vi'` | Source language code |
| `target_lang` | str | `'en'` | Target language code |

## Limitations

1. Works best with well-formed sentences
2. Long texts should be split into sentences for better results
3. Only supports Vietnamese-English language pair

## References

1. [VietAI/envit5-translation](https://huggingface.co/VietAI/envit5-translation)
2. [Underthesea GitHub Repository](https://github.com/undertheseanlp/underthesea)
