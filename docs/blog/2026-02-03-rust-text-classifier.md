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

## The Models

### sen-classifier-general

General Vietnamese news classification model trained on VNTC dataset.

**Training Data:** [VNTC](https://github.com/duyvuleo/VNTC) (Vietnamese News Text Classification)
- 33,759 training samples
- 50,373 test samples
- 10 news categories

**Categories:**

| Label | Vietnamese | English |
|-------|------------|---------|
| Chinh tri Xa hoi | Chính trị Xã hội | Politics/Society |
| Doi song | Đời sống | Lifestyle |
| Khoa hoc | Khoa học | Science |
| Kinh doanh | Kinh doanh | Business |
| Phap luat | Pháp luật | Law |
| Suc khoe | Sức khỏe | Health |
| The gioi | Thế giới | World |
| The thao | Thể thao | Sports |
| Van hoa | Văn hóa | Culture |
| Vi tinh | Vi tính | Technology |

**Performance:**
- Accuracy: **92.49%**
- F1 (weighted): 92.40%
- Training time: 37.6s

### sen-classifier-bank

Vietnamese banking intent classification model trained on UTS2017_Bank dataset.

**Training Data:** [UTS2017_Bank](https://huggingface.co/datasets/undertheseanlp/UTS2017_Bank)
- 1,581 training samples
- 396 test samples
- 14 banking categories

**Categories:**

| Label | Description | Samples |
|-------|-------------|---------|
| CUSTOMER_SUPPORT | Customer support queries | 774 |
| TRADEMARK | Brand/trademark mentions | 697 |
| LOAN | Loan services | 73 |
| INTERNET_BANKING | Internet banking | 69 |
| CARD | Card services | 66 |
| INTEREST_RATE | Interest rates | 58 |
| PROMOTION | Promotions | 56 |
| DISCOUNT | Discounts | 40 |
| MONEY_TRANSFER | Money transfer | 37 |
| OTHER | Other queries | 70 |
| PAYMENT | Payment services | 17 |
| SAVING | Savings | 12 |
| ACCOUNT | Account services | 5 |
| SECURITY | Security | 3 |

**Performance:**
- Accuracy: **75.76%** (+3.29% vs previous sonar_core_1)
- F1 (weighted): 72.70%
- Training time: 0.13s

## Training Pipeline

Both models use a 3-stage TF-IDF + Linear SVM pipeline:

```
Input Text
    ↓
┌─────────────────────────────────────┐
│  CountVectorizer                    │
│  - max_features: 20,000             │
│  - ngram_range: (1, 2)              │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  TfidfTransformer                   │
│  - use_idf: True                    │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  LinearSVC                          │
│  - C: 1.0                           │
│  - max_iter: 2000                   │
│  - loss: squared_hinge              │
└─────────────────────────────────────┘
    ↓
Predicted Label + Confidence
```

**Key design decisions:**
- **Syllable-level tokenization**: No word segmentation for speed
- **Character n-grams (1-2)**: Captures Vietnamese morphology
- **20K vocabulary**: Balances accuracy and model size
- **Linear SVM**: Fast training, works well with sparse high-dimensional data

Training code: [sen-1/src/scripts/train_vntc.py](https://github.com/undertheseanlp/sen-1)

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
