# POS Tagging Technical Report

## Overview

The POS tagging module in Underthesea provides Vietnamese Part-of-Speech tagging using the TRE-1 model, a Conditional Random Field (CRF) based tagger trained on the Universal Dependencies Dataset (UDD-v0.1).

**Model:** [undertheseanlp/tre-1](https://huggingface.co/undertheseanlp/tre-1)
**License:** Apache 2.0

## Architecture

### TRE-1 Model

```
TRE-1 Pipeline
├── Text Input
│   └── Pre-tokenized Vietnamese text
├── Feature Extraction
│   ├── Current Token Features
│   │   ├── Word form
│   │   ├── Lowercase form
│   │   ├── Prefix (2-3 chars)
│   │   ├── Suffix (2-3 chars)
│   │   └── Character type checks
│   ├── Context Features
│   │   ├── Previous 1-2 tokens
│   │   └── Next 1-2 tokens
│   ├── Bigram Features
│   │   └── Adjacent token combinations
│   └── Dictionary Features
│       └── In-vocabulary checks
├── CRF Classification
│   └── python-crfsuite
└── Output
    └── UPOS tags for each token
```

### Algorithm: Conditional Random Fields (CRF)

The model uses CRF sequence labeling implemented with python-crfsuite, predicting Universal POS (UPOS) tags for each token in the input sequence.

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Algorithm | CRF (python-crfsuite) |
| L1 regularization (c1) | 1.0 |
| L2 regularization (c2) | 1e-3 |
| Max iterations | 100 |
| Training data | undertheseanlp/UDD-v0.1 |
| Tagset | Universal POS tags (UPOS) |

### Feature Templates

| Feature Type | Description |
|-------------|-------------|
| Current token | Word form, lowercase, prefix/suffix (2-3 chars), character type |
| Context | Previous and next 1-2 tokens |
| Bigram | Adjacent token combinations |
| Dictionary | In-vocabulary checks |

## Performance

Evaluated on the held-out test set from UDD-v0.1:

| Metric | Score |
|--------|-------|
| Accuracy | ~94% |
| F1 (macro) | ~90% |
| F1 (weighted) | ~94% |

## Usage

### Basic Usage

```python
from underthesea import pos_tag

text = "Tôi yêu Việt Nam"
result = pos_tag(text)
print(result)
# [('Tôi', 'PRON'), ('yêu', 'VERB'), ('Việt Nam', 'PROPN')]
```

### Inference API

```python
import requests

API_URL = "https://api-inference.huggingface.co/models/undertheseanlp/tre-1"
headers = {"Authorization": "Bearer YOUR_TOKEN"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

output = query({"inputs": "Tôi yêu Việt Nam"})
# [{"token": "Tôi", "tag": "PRON"}, {"token": "yêu", "tag": "VERB"}, ...]
```

### Local Usage

```python
import pycrfsuite
from handler import EndpointHandler

handler = EndpointHandler(path="./")
result = handler({"inputs": "Tôi yêu Việt Nam"})
```

## Limitations

1. **Pre-tokenized Input**: Requires pre-tokenized input (whitespace-separated tokens)
2. **Domain Sensitivity**: Performance may vary on out-of-domain text
3. **No Word Segmentation**: Does not handle Vietnamese word segmentation internally

## References

1. [TRE-1 on Hugging Face](https://huggingface.co/undertheseanlp/tre-1)
2. [Universal Dependencies](https://universaldependencies.org/)
3. [python-crfsuite](https://github.com/scrapinghub/python-crfsuite)
4. [Underthesea GitHub Repository](https://github.com/undertheseanlp/underthesea)

## Citation

```bibtex
@misc{tre1-pos-tagger,
  author = {undertheseanlp},
  title = {Vietnamese POS Tagger TRE-1},
  year = {2025},
  publisher = {Hugging Face},
  url = {https://huggingface.co/undertheseanlp/tre-1}
}
```
