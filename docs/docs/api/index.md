# API Reference

This section provides complete API documentation for all Underthesea functions.

## Core Functions

| Function | Description | Install |
|----------|-------------|---------|
| [`sent_tokenize`](sent-tokenize) | Sentence segmentation | Core |
| [`text_normalize`](text-normalize) | Text normalization | Core |
| [`word_tokenize`](word-tokenize) | Word segmentation | Core |
| [`pos_tag`](pos-tag) | Part-of-speech tagging | Core |
| [`chunk`](chunk) | Phrase chunking | Core |
| [`ner`](ner) | Named entity recognition | Core |
| [`classify`](classify) | Text classification | Core |
| [`sentiment`](sentiment) | Sentiment analysis | Core |

## Deep Learning Functions

| Function | Description | Install |
|----------|-------------|---------|
| [`dependency_parse`](dependency-parse) | Dependency parsing | `[deep]` |
| [`translate`](translate) | Vietnamese-English translation | `[deep]` |

## Additional Functions

| Function | Description | Install |
|----------|-------------|---------|
| [`lang_detect`](lang-detect) | Language detection | `[langdetect]` |
| [`tts`](tts) | Text-to-speech | `[voice]` |
| [`agent`](agent) | Conversational AI agent | `[agent]` |

## Quick Import

All main functions can be imported directly from `underthesea`:

```python
from underthesea import (
    sent_tokenize,
    text_normalize,
    word_tokenize,
    pos_tag,
    chunk,
    ner,
    classify,
    sentiment,
    dependency_parse,  # requires [deep]
    translate,         # requires [deep]
    lang_detect,       # requires [langdetect]
    agent,             # requires [agent]
)
```

## Common Parameters

Many functions share common parameters:

### `format`

Controls output format:

- `None` (default): Returns a list
- `"text"`: Returns a string with underscores joining multi-word tokens

```python
word_tokenize("Việt Nam", format=None)   # ['Việt Nam']
word_tokenize("Việt Nam", format="text") # 'Việt_Nam'
```

### `model`

Specifies which model to use:

```python
# Use default model
ner("text")

# Use specific model
ner("text", deep=True)  # Use deep learning model
classify("text", model='prompt')  # Use OpenAI model
```

### `domain`

Specifies the domain for domain-specific models:

```python
classify("text", domain='bank')
sentiment("text", domain='bank')
```
