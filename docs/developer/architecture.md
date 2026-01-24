# Architecture

This document describes the internal architecture of Underthesea.

## Overview

Underthesea is organized as a collection of NLP pipelines, each handling a specific task.

```
underthesea/
├── pipeline/              # Main NLP modules
│   ├── sent_tokenize/     # Sentence segmentation
│   ├── text_normalize/    # Text normalization
│   ├── word_tokenize/     # Word segmentation
│   ├── pos_tag/           # POS tagging
│   ├── chunking/          # Phrase chunking
│   ├── dependency_parse/  # Dependency parsing
│   ├── ner/               # Named entity recognition
│   ├── classification/    # Text classification
│   ├── sentiment/         # Sentiment analysis
│   ├── translate/         # Translation
│   ├── lang_detect/       # Language detection
│   └── say/               # Text-to-speech
├── models/                # Model implementations
├── datasets/              # Built-in datasets
├── corpus/                # Corpus handling
├── resources/             # Static resources
└── cli.py                 # CLI interface
```

## Pipeline Module Structure

Each pipeline module follows a consistent pattern:

```
pipeline/word_tokenize/
├── __init__.py            # Main API function
├── model.py               # Model implementation
├── feature.py             # Feature extraction
└── default_model/         # Default model files
```

### Main API (`__init__.py`)

```python
# Lazy loading pattern
_model = None

def word_tokenize(sentence, format=None):
    global _model
    if _model is None:
        _model = load_model()
    return _model.predict(sentence, format)
```

### Model Implementation

```python
class CRFModel:
    def __init__(self, model_path):
        self.model = load_crf(model_path)

    def predict(self, text):
        features = extract_features(text)
        return self.model.tag(features)
```

## Lazy Loading

Models are loaded on first use to minimize startup time:

```python
# At import time - no model loaded
from underthesea import word_tokenize

# First call - model loaded and cached
result = word_tokenize("text")

# Subsequent calls - uses cached model
result = word_tokenize("more text")
```

Benefits:
- Fast import time
- Memory efficiency (only used models loaded)
- Simple API

## Model Types

### CRF Models

Used for: word segmentation, POS tagging, chunking, NER, classification, sentiment

```python
# Uses python-crfsuite
import pycrfsuite

class CRFTagger:
    def __init__(self, model_path):
        self.tagger = pycrfsuite.Tagger()
        self.tagger.open(model_path)

    def tag(self, features):
        return self.tagger.tag(features)
```

### Deep Learning Models

Used for: dependency parsing, deep NER, translation

```python
# Uses transformers
from transformers import AutoModel, AutoTokenizer

class TransformerModel:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
```

### FastText Models

Used for: language detection

```python
import fasttext

class LangDetector:
    def __init__(self, model_path):
        self.model = fasttext.load_model(model_path)

    def detect(self, text):
        prediction = self.model.predict(text)
        return prediction[0][0].replace('__label__', '')
```

## Feature Extraction

Features are extracted for CRF models:

```python
def extract_features(sentence):
    features = []
    for i, word in enumerate(sentence):
        word_features = {
            'word': word,
            'is_upper': word.isupper(),
            'is_title': word.istitle(),
            'prev_word': sentence[i-1] if i > 0 else 'BOS',
            'next_word': sentence[i+1] if i < len(sentence)-1 else 'EOS',
        }
        features.append(word_features)
    return features
```

## Resource Management

### Model Storage

Models are stored in `~/.underthesea/models/`:

```
~/.underthesea/
├── models/
│   ├── WS_VLSP2013_CRF/
│   ├── POS_VLSP2013_CRF/
│   └── NER_VLSP2016_BERT/
└── datasets/
    ├── VNTC/
    └── UTS2017-BANK/
```

### Model Download

```python
def download_model(model_name):
    url = get_model_url(model_name)
    local_path = get_local_path(model_name)

    if not os.path.exists(local_path):
        download_file(url, local_path)
        extract_archive(local_path)

    return local_path
```

## Rust Extension

Performance-critical code uses the Rust extension:

```
extensions/underthesea_core/
├── src/
│   └── lib.rs             # Rust implementation
├── Cargo.toml             # Rust dependencies
└── pyproject.toml         # Python binding config
```

Built with maturin:

```bash
cd extensions/underthesea_core
maturin develop
```

## CLI Architecture

The CLI uses Click:

```python
# cli.py
import click

@click.group()
def cli():
    pass

@cli.command()
def list_data():
    """List available datasets."""
    for dataset in get_datasets():
        print(dataset)

@cli.command()
@click.argument('text')
def say(text):
    """Convert text to speech."""
    from underthesea.pipeline.say import say
    say(text)
```

## Optional Dependencies

Optional features are guarded:

```python
def translate(text):
    try:
        from transformers import AutoModel
    except ImportError:
        raise ImportError(
            "Translation requires deep learning dependencies. "
            "Install with: pip install 'underthesea[deep]'"
        )
    # ... translation logic
```

## Testing Architecture

```
tests/
├── pipeline/
│   ├── word_tokenize/
│   │   └── test_word_tokenize.py
│   ├── pos_tag/
│   │   └── test_pos_tag.py
│   └── ner/
│       └── test_ner.py
└── conftest.py            # Pytest fixtures
```

## Extending Underthesea

### Adding a New Pipeline

1. Create directory: `underthesea/pipeline/new_task/`
2. Implement `__init__.py` with main API
3. Add model implementation
4. Export from `underthesea/__init__.py`
5. Add tests in `tests/pipeline/new_task/`
6. Add documentation

### Adding a New Model

1. Train the model using appropriate toolkit
2. Save model files
3. Update model registry
4. Add download logic
5. Test with existing pipeline
