# Optional Dependencies

Underthesea is designed to be lightweight by default. Advanced features require optional dependencies.

## Installation Options

| Install Command | Features Enabled |
|-----------------|------------------|
| `pip install underthesea` | Core NLP features |
| `pip install underthesea[deep]` | Deep learning models |
| `pip install underthesea[prompt]` | OpenAI-based classification |
| `pip install underthesea[langdetect]` | Language detection |
| `pip install underthesea[dev]` | Development tools |

## Core Package

The base installation includes:

```bash
pip install underthesea
```

**Features:**

- `sent_tokenize` - Sentence segmentation
- `text_normalize` - Text normalization
- `word_tokenize` - Word segmentation
- `pos_tag` - POS tagging
- `chunk` - Chunking
- `ner` - Named entity recognition (CRF)
- `classify` - Text classification (CRF)
- `sentiment` - Sentiment analysis

**Dependencies:**

- scikit-learn
- python-crfsuite
- joblib
- PyYAML

## Deep Learning Package

For transformer-based models:

```bash
pip install underthesea[deep]
```

**Features:**

- `dependency_parse` - Dependency parsing
- `ner(..., deep=True)` - Deep learning NER
- `translate` - Vietnamese-English translation

**Dependencies:**

- torch
- transformers

**Example:**

```python
from underthesea import dependency_parse, translate

# Dependency parsing
dependency_parse("Tối 29/11, Việt Nam thêm 2 ca mắc Covid-19")

# Translation
translate("Hà Nội là thủ đô của Việt Nam")
# 'Hanoi is the capital of Vietnam'
```

## Prompt Package

For OpenAI-powered classification:

```bash
pip install underthesea[prompt]
export OPENAI_API_KEY=your_api_key
```

**Features:**

- `classify(..., model='prompt')` - Zero-shot classification

**Dependencies:**

- openai

**Example:**

```python
from underthesea import classify

classify("HLV ngoại đòi gần tỷ mỗi tháng dẫn dắt tuyển Việt Nam", model='prompt')
# 'Thể thao'
```

## Language Detection Package

For language identification:

```bash
pip install underthesea[langdetect]
```

**Features:**

- `lang_detect` - Language detection

**Dependencies:**

- fasttext

**Example:**

```python
from underthesea import lang_detect

lang_detect("Cựu binh Mỹ trả nhật ký nhẹ lòng")
# 'vi'

lang_detect("Hello, how are you?")
# 'en'
```

## Development Package

For contributors and developers:

```bash
pip install underthesea[dev]
```

**Features:**

- Linting with Ruff
- Testing with tox
- Build tools

**Dependencies:**

- ruff
- tox

## Combining Packages

You can install multiple optional packages:

```bash
# Deep learning + language detection
pip install underthesea[deep,langdetect]

# All optional dependencies
pip install underthesea[deep,prompt,langdetect,dev]
```

## Checking Installed Features

You can check which features are available:

```python
import underthesea

# Check version
print(underthesea.__version__)

# Try importing optional features
try:
    from underthesea import translate
    print("Deep learning features: Available")
except ImportError:
    print("Deep learning features: Not installed")

try:
    from underthesea import lang_detect
    print("Language detection: Available")
except ImportError:
    print("Language detection: Not installed")
```

## Troubleshooting

### PyTorch Installation Issues

If you have trouble installing PyTorch, install it separately first:

```bash
# CPU only
pip install torch --index-url https://download.pytorch.org/whl/cpu

# With CUDA support
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### FastText Installation Issues

On some systems, FastText may require compilation:

```bash
# Install build tools first
pip install wheel setuptools

# Then install fasttext
pip install fasttext
```

### OpenAI API Key

For prompt-based models, set your API key:

```bash
export OPENAI_API_KEY=sk-...
```

Or in Python:

```python
import os
os.environ['OPENAI_API_KEY'] = 'sk-...'
```
