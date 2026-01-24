# Core Concepts

This page explains the key concepts and design principles behind Underthesea.

## Pipeline Architecture

Underthesea is organized as a collection of NLP pipelines. Each pipeline handles a specific NLP task:

```
underthesea/
└── pipeline/
    ├── sent_tokenize/    # Sentence segmentation
    ├── text_normalize/   # Text normalization
    ├── word_tokenize/    # Word segmentation
    ├── pos_tag/          # Part-of-speech tagging
    ├── chunking/         # Phrase chunking
    ├── dependency_parse/ # Dependency parsing
    ├── ner/              # Named entity recognition
    ├── classification/   # Text classification
    ├── sentiment/        # Sentiment analysis
    ├── translate/        # Translation
    ├── lang_detect/      # Language detection
    └── say/              # Text-to-speech
```

## Lazy Model Loading

Underthesea uses lazy loading for models to minimize startup time and memory usage:

```python
# Model is NOT loaded yet
from underthesea import word_tokenize

# Model is loaded on first call
result = word_tokenize("Xin chào")

# Subsequent calls use cached model (fast)
result = word_tokenize("Việt Nam")
```

This means:

- **Fast imports**: Importing the library is instant
- **Memory efficient**: Only load models you actually use
- **First call overhead**: The first call may take longer due to model loading

## Model Types

Underthesea uses different types of models depending on the task:

### CRF Models (Default)

Conditional Random Fields (CRF) models are lightweight and fast:

- Word segmentation
- POS tagging
- Chunking
- Named entity recognition
- Text classification
- Sentiment analysis

### Deep Learning Models

Transformer-based models for complex tasks:

- Dependency parsing
- Deep NER (`ner(..., deep=True)`)
- Translation

Requires `pip install underthesea[deep]`

### Prompt-based Models

Uses OpenAI API for zero-shot classification:

- Text classification (`classify(..., model='prompt')`)

Requires `pip install underthesea[prompt]`

### FastText Models

For language identification:

- Language detection

Requires `pip install underthesea[langdetect]`

## Output Formats

Most functions support multiple output formats:

### List Format (Default)

```python
from underthesea import word_tokenize

word_tokenize("Xin chào Việt Nam")
# ['Xin', 'chào', 'Việt Nam']
```

### Text Format

```python
word_tokenize("Xin chào Việt Nam", format="text")
# 'Xin chào Việt_Nam'
```

Multi-word tokens are joined with underscores.

### Tuple Format

For tagging tasks (POS, NER, chunking):

```python
from underthesea import pos_tag

pos_tag("Xin chào")
# [('Xin', 'V'), ('chào', 'V')]
```

## Vietnamese NLP Challenges

### No Word Boundaries

Unlike English, Vietnamese text doesn't have spaces between words:

```
English: "I love Vietnam"     → 3 words
Vietnamese: "Tôi yêu Việt Nam" → 4 words (not 3!)
```

Underthesea's word segmentation handles this:

```python
word_tokenize("Tôi yêu Việt Nam")
# ['Tôi', 'yêu', 'Việt Nam']
```

### Diacritics and Unicode

Vietnamese uses Latin script with diacritics. Underthesea's text normalization handles common encoding issues:

```python
from underthesea import text_normalize

# Fix common Unicode issues
text_normalize("Ðảm baỏ chất lựơng")
# "Đảm bảo chất lượng"
```

### Compound Words

Many Vietnamese words consist of multiple syllables:

- "Việt Nam" (Vietnam) - 2 syllables, 1 word
- "khởi nghiệp" (start a business) - 2 syllables, 1 word
- "thành phố Hồ Chí Minh" (Ho Chi Minh City) - 5 syllables, 1 entity

## Next Steps

- Learn about [Optional Dependencies](optional_deps.md)
- Explore [Available Models](models.md)
- Read the [API Reference](../api/index.md)
