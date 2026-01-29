# API Reference

This section provides complete API documentation for all Underthesea functions.

## Core Functions

| Function | Description | Install |
|----------|-------------|---------|
| [`sent_tokenize`](sent_tokenize.md) | Sentence segmentation | Core |
| [`text_normalize`](text_normalize.md) | Text normalization | Core |
| [`word_tokenize`](word_tokenize.md) | Word segmentation | Core |
| [`pos_tag`](pos_tag.md) | Part-of-speech tagging | Core |
| [`chunk`](chunk.md) | Phrase chunking | Core |
| [`ner`](ner.md) | Named entity recognition | Core |
| [`classify`](classify.md) | Text classification | Core |
| [`sentiment`](sentiment.md) | Sentiment analysis | Core |

## Deep Learning Functions

| Function | Description | Install |
|----------|-------------|---------|
| [`dependency_parse`](dependency_parse.md) | Dependency parsing | `[deep]` |
| [`translate`](translate.md) | Vietnamese-English translation | `[deep]` |

## Additional Functions

| Function | Description | Install |
|----------|-------------|---------|
| [`lang_detect`](lang_detect.md) | Language detection | `[langdetect]` |
| [`tts`](tts.md) | Text-to-speech | `[voice]` |
| [`agent`](agent.md) | Conversational AI agent | `[agent]` |

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
