---
slug: rust-text-classifier
title: Rust-Powered Text Classification - 273x Faster Inference
authors: [rain1024]
tags: [rust, performance, classification, nlp]
---

# Rust-Powered Text Classification: 273x Faster Inference

In underthesea v9.2.9, we've completely rewritten the text classification pipeline using our Rust-based `TextClassifier`. This delivers up to **273x faster inference** compared to the previous sklearn-based implementation.

<!-- truncate -->

## Background

Text classification in underthesea supports two domains:
- **General**: News categorization (10 categories)
- **Bank**: Banking intent classification (14 categories)

Previously, we used scikit-learn's `TfidfVectorizer` + `LinearSVC` loaded via joblib. While accurate, this approach had significant overhead.

## The Architecture Change

### Before (sklearn-based)

```
┌─────────────────────────────────────────────────────────────┐
│                        Python                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │    Input     │───▶│TfidfVectorizer───▶│  LinearSVC   │  │
│  │    Text      │    │  (sklearn)   │    │  (sklearn)   │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│                              │                   │          │
│                      joblib.load()        joblib.load()     │
└─────────────────────────────────────────────────────────────┘
```

Loading two separate pickle files, with Python-based vectorization and inference.

### After (Rust-based)

```
┌─────────────────────────────────────────────────────────────┐
│                        Python                               │
│  ┌──────────────┐    ┌──────────────────────────────────┐  │
│  │    Input     │───▶│         TextClassifier           │  │
│  │    Text      │    │  TF-IDF + LinearSVC (Rust)       │  │
│  └──────────────┘    └──────────────────────────────────┘  │
│                              │                              │
│                      single .bin file                       │
│                      underthesea-core                       │
└─────────────────────────────────────────────────────────────┘
```

Single binary model file, vectorization and inference fused in Rust.

## Code Changes

The API remains unchanged:

```python
from underthesea import classify

# General classification
classify("Việt Nam vô địch AFF Cup")
# "The thao"

# Bank domain
classify("Lãi suất tiết kiệm bao nhiêu?", domain="bank")
# ['INTEREST_RATE']
```

Internally, the implementation is much simpler:

**Before:**
```python
import joblib
from underthesea.pipeline.classification import bank

vectorizer = joblib.load("vectorizer.pkl")
classifier = joblib.load("classifier.pkl")
features = vectorizer.transform([text])
prediction = classifier.predict(features)
```

**After:**
```python
from underthesea_core import TextClassifier

classifier = TextClassifier.load("model.bin")
prediction = classifier.predict(text)
```

## Benchmark Results

Tested on the same hardware with batch inference:

| Domain | sklearn | Rust | Speedup |
|--------|---------|------|---------|
| General | 1,228 samples/sec | 66,678 samples/sec | **54x** |
| Bank | 244 samples/sec | 66,678 samples/sec | **273x** |

Single sample latency: **4ms → 0.465ms**

## Why Is It Faster?

### 1. Fused Pipeline

TF-IDF vectorization and SVM inference run in a single Rust function call, eliminating Python overhead between stages.

### 2. Optimized Sparse Operations

```rust
pub fn predict(&self, text: &str) -> String {
    // Tokenize and hash features in one pass
    let features = self.vectorizer.transform(text);

    // Sparse dot product with pre-sorted indices
    let scores = self.svm.decision_function(&features);

    self.classes[scores.argmax()].clone()
}
```

### 3. Single File Model

One `.bin` file instead of multiple pickle files:
- Faster loading
- Atomic deployment
- Smaller size (JSON-based, not pickle)

### 4. No Python GIL Contention

Rust code releases the GIL during computation, enabling true parallelism.

## Model Accuracy

The new models maintain high accuracy:

| Dataset | Categories | Accuracy |
|---------|------------|----------|
| VNTC | 10 news topics | 92.49% |
| Bank | 14 banking intents | 75.76% |

## Label Format Change

Labels now use Title case with spaces:

| Old | New |
|-----|-----|
| `the_thao` | `The thao` |
| `kinh_doanh` | `Kinh doanh` |
| `vi_tinh` | `Vi tinh` |

Bank domain labels remain uppercase: `INTEREST_RATE`, `MONEY_TRANSFER`, etc.

## Simplified Codebase

We consolidated three separate modules into one:

**Before:**
```
classification/
├── bank/
│   └── __init__.py
├── sonar_core_1/
│   └── __init__.py
├── vntc/
│   └── __init__.py
└── __init__.py
```

**After:**
```
classification/
├── __init__.py      # Everything here
└── classification_prompt.py
```

~190 lines removed, single source of truth for model URLs and loading logic.

## Try It Out

```bash
pip install underthesea==9.2.9
```

```python
from underthesea import classify

# 273x faster!
classify("Thị trường chứng khoán tăng điểm mạnh")
# "Kinh doanh"

classify.labels
# ['Chinh tri Xa hoi', 'Doi song', 'Khoa hoc', 'Kinh doanh', ...]

classify("Mở thẻ tín dụng", domain="bank")
# ['CARD']

classify.bank.labels
# ['ACCOUNT', 'CARD', 'CUSTOMER_SUPPORT', ...]
```

## Links

- [PR #935](https://github.com/undertheseanlp/underthesea/pull/935) - Classification pipeline refactor
- [Sen-1](https://github.com/undertheseanlp/sen-1) - Training code and technical report
- [underthesea-core](https://pypi.org/project/underthesea-core/) - Rust extension on PyPI
