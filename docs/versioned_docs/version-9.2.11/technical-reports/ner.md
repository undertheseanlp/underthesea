# Named Entity Recognition

## Overview

The NER module in Underthesea identifies and classifies named entities in Vietnamese text, supporting both a lightweight CRF model and a deep learning Transformers model. Entities are classified into persons (PER), locations (LOC), and organizations (ORG).

## Architecture

### Dual Model Support

```
NER Pipeline
├── Text Input
│   └── Raw Vietnamese text
├── Mode Selection
│   ├── Shallow (default)
│   │   ├── word_tokenize()
│   │   ├── pos_tag()
│   │   ├── chunk()
│   │   └── CRF NER Model
│   └── Deep (deep=True)
│       └── HuggingFace Transformers
│           └── undertheseanlp/vietnamese-ner-v1.4.0a2
└── Output
    └── Entity annotations (BIO format)
```

### CRF Model (Default)

The shallow NER model builds on the full preprocessing pipeline (word tokenization → POS tagging → chunking) and applies a CRF sequence labeler for entity classification.

### Transformers Model (Deep)

The deep learning model uses HuggingFace's `AutoModelForTokenClassification` with the pretrained model `undertheseanlp/vietnamese-ner-v1.4.0a2`. It handles subword tokenization merging internally.

**Requirements:**
```bash
pip install "underthesea[deep]"
```

## Entity Types

| Tag | Entity Type | Example |
|-----|------------|---------|
| `B-PER` / `I-PER` | Person | Donald Trump |
| `B-LOC` / `I-LOC` | Location | Việt Nam, Mỹ |
| `B-ORG` / `I-ORG` | Organization | Bộ Giáo dục |
| `O` | Not an entity | — |

## Usage

### CRF Model (Default)

```python
from underthesea import ner

text = "Chưa tiết lộ lịch trình tới Việt Nam của Tổng thống Mỹ Donald Trump"
entities = ner(text)
# [('Chưa', 'R', 'O', 'O'),
#  ('tiết lộ', 'V', 'B-VP', 'O'),
#  ('lịch trình', 'V', 'B-VP', 'O'),
#  ('tới', 'E', 'B-PP', 'O'),
#  ('Việt Nam', 'Np', 'B-NP', 'B-LOC'),
#  ('của', 'E', 'B-PP', 'O'),
#  ('Tổng thống', 'N', 'B-NP', 'O'),
#  ('Mỹ', 'Np', 'B-NP', 'B-LOC'),
#  ('Donald', 'Np', 'B-NP', 'B-PER'),
#  ('Trump', 'Np', 'B-NP', 'I-PER')]
```

### Deep Learning Model

```python
entities = ner(text, deep=True)
# [{'entity': 'LOC', 'word': 'Việt Nam'},
#  {'entity': 'LOC', 'word': 'Mỹ'},
#  {'entity': 'PER', 'word': 'Donald Trump'}]
```

### Function Signature

```python
def ner(
    sentence: str,
    format: str = None,
    deep: bool = False
) -> list[tuple] | list[dict]
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sentence` | str | required | Input text |
| `format` | str | None | Output format |
| `deep` | bool | False | Use deep learning model |

## Models

| Model | Type | HuggingFace |
|-------|------|-------------|
| CRF (default) | Sequence labeling | — |
| Deep | Token classification | [undertheseanlp/vietnamese-ner-v1.4.0a2](https://huggingface.co/undertheseanlp/vietnamese-ner-v1.4.0a2) |

## References

1. [undertheseanlp/vietnamese-ner-v1.4.0a2](https://huggingface.co/undertheseanlp/vietnamese-ner-v1.4.0a2)
2. [Underthesea GitHub Repository](https://github.com/undertheseanlp/underthesea)
