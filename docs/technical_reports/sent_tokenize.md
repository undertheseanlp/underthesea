# Sentence Tokenization Technical Report

## Overview

The sentence tokenization module in Underthesea provides Vietnamese sentence boundary detection using a custom Punkt-style algorithm. This document describes the architecture, algorithm, and the migration from NLTK dependency to a standalone implementation.

## Architecture

### Algorithm: Punkt-style Sentence Tokenizer

The implementation is inspired by the Punkt sentence boundary detection algorithm developed by Kiss and Strunk (2006).

**Reference:** Tibor Kiss and Jan Strunk. 2006. [Unsupervised Multilingual Sentence Boundary Detection](https://aclanthology.org/J06-4003/)

### Components

```
PunktSentenceTokenizer
├── Configuration
│   ├── SENT_END_CHARS: frozenset('.', '?', '!')
│   └── abbrev_types: Set[str] (abbreviation dictionary)
├── Core Methods
│   ├── sentences_from_text(): Main tokenization entry point
│   ├── _is_sentence_boundary(): Boundary detection logic
│   └── _get_preceding_word(): Abbreviation extraction
└── Data
    └── punkt_params.json (368 Vietnamese abbreviations)
```

## Algorithm Details

### Sentence Boundary Detection

The algorithm identifies sentence boundaries through a multi-step process:

1. **Punctuation Detection**: Scan for sentence-ending punctuation (`.`, `?`, `!`)
2. **Abbreviation Check**: Verify if the period follows a known abbreviation
3. **Context Analysis**: Examine following characters to confirm boundary

### Boundary Decision Rules

| Condition | Is Boundary? |
|-----------|--------------|
| Period after abbreviation (e.g., "Dr.") | No |
| Punctuation followed by uppercase letter | Yes |
| Punctuation followed by newline | Yes |
| Punctuation at end of text | Yes |
| Single letter followed by period | No (treated as abbreviation) |

### Abbreviation Handling

The tokenizer maintains a comprehensive abbreviation dictionary:

**Built-in Categories:**
- Single letters (A-Z, a-z)
- Vietnamese abbreviations (TP, TS, ThS, BS, PGS, GS, etc.)
- English abbreviations (Dr, Mr, Mrs, Prof, Inc, etc.)
- Domain-specific terms (e.g., ".hcm", ".hn", "g.m.t")

**Pre-trained Abbreviations:** 368 entries extracted from Vietnamese text corpora

## Data Format

### punkt_params.json

```json
{
  "abbrev_types": [
    ".hcm",
    ".hn",
    "tp",
    "ts",
    "ths",
    ...
  ],
  "sent_starters": []
}
```

| Field | Type | Description |
|-------|------|-------------|
| abbrev_types | List[str] | Known abbreviations that don't end sentences |
| sent_starters | List[str] | Words that typically start sentences (unused) |

## Usage

### Basic Usage

```python
from underthesea import sent_tokenize

text = "Xin chào. Tôi là sinh viên."
sentences = sent_tokenize(text)
# Output: ['Xin chào.', 'Tôi là sinh viên.']
```

### Edge Cases

```python
# Empty text
sent_tokenize("")  # Returns: []

# Single sentence without punctuation
sent_tokenize("hôm nay")  # Returns: ['hôm nay']

# Abbreviations
sent_tokenize("Ông Dr. Nguyễn đã đến.")  # Returns: ['Ông Dr. Nguyễn đã đến.']

# Multiple punctuation
sent_tokenize("Thật sao?! Tuyệt vời!")  # Returns: ['Thật sao?!', 'Tuyệt vời!']
```

## Performance

### Complexity

| Operation | Time Complexity | Space Complexity |
|-----------|-----------------|------------------|
| Tokenization | O(n) | O(n) |
| Model Loading | O(a) | O(a) |

Where:
- n = length of input text
- a = number of abbreviations (~400)

### Benchmarks

| Text Length | Time (ms) |
|-------------|-----------|
| 100 chars | <0.1 |
| 1,000 chars | ~0.5 |
| 10,000 chars | ~5 |

*Benchmarks run on Apple M1, Python 3.11*

## Limitations

1. **No Probabilistic Model**: Unlike NLTK's Punkt, this implementation uses a fixed abbreviation dictionary rather than a probabilistic model trained on text
2. **No Sentence Starters**: The `sent_starters` parameter is preserved in JSON but not used in boundary detection
3. **Limited Ellipsis Handling**: Sequences like "..." are treated as single boundary markers

## Future Improvements

- [ ] Train a Vietnamese-specific abbreviation model
- [ ] Add support for quoted speech boundary detection
- [ ] Handle numeric expressions (e.g., "1.000.000 đồng")
- [ ] Add streaming/generator API for large texts

## References

1. Kiss, T., & Strunk, J. (2006). Unsupervised Multilingual Sentence Boundary Detection. Computational Linguistics, 32(4), 485-525.
2. [NLTK Punkt Tokenizer](https://www.nltk.org/api/nltk.tokenize.punkt.html)
3. [Underthesea GitHub Repository](https://github.com/undertheseanlp/underthesea)

## Changelog

### Version 9.2.0 (PR #877)
- Removed NLTK dependency
- Implemented custom Punkt-style sentence tokenizer
- Converted model from pickle to JSON format
- Removed unused Tree/TreeSentence classes from conll.py
