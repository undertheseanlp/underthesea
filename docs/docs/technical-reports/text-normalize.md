# Text Normalization

## Overview

The text normalization module in Underthesea fixes common Vietnamese encoding and diacritic issues, converting text to Unicode NFC normalized form. It handles legacy encoding problems from older Vietnamese text systems.

## Architecture

```
Text Normalization Pipeline
├── Text Input
│   └── Raw Vietnamese text (potentially with encoding issues)
├── Word Tokenization
│   └── Split text into tokens
├── Token Normalization
│   ├── Character Normalization
│   │   ├── Đ/Ð confusion (Ðại → Đại)
│   │   ├── Old-style diacritics (hoá → hóa)
│   │   └── Incorrect vowel composition (lựơng → lượng)
│   └── Unicode NFC Normalization
└── Output
    └── Normalized text string
```

### Components

| Component | Description |
|-----------|-------------|
| `text_normalizer` | Main normalization entry point |
| `token_normalize` | Token-level normalization |
| `character_normalize` | Character-level encoding fixes |

## Common Issues Fixed

| Issue | Example | Corrected |
|-------|---------|-----------|
| Đ/Ð confusion | Ðại học | Đại học |
| Old-style diacritics | hoá học | hóa học |
| Incorrect vowel composition | lựơng | lượng |
| Mixed encoding | baỏ đảm | bảo đảm |

## Usage

### Basic Usage

```python
from underthesea import text_normalize

text = "Ðảm baỏ chất lựơng"
result = text_normalize(text)
print(result)  # "Đảm bảo chất lượng"
```

### Function Signature

```python
def text_normalize(text: str, tokenizer: str = 'underthesea') -> str
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | str | required | Input text to normalize |
| `tokenizer` | str | 'underthesea' | Tokenizer to use |

## References

1. [Unicode NFC Normalization](https://unicode.org/reports/tr15/)
2. [Underthesea GitHub Repository](https://github.com/undertheseanlp/underthesea)
