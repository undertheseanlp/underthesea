# underthesea_core

High-performance Rust extension for underthesea, providing ML models and text processing tools with Python bindings via PyO3.

## Installation

```bash
pip install underthesea-core
```

## What's Included

| Module | Classes | Description |
|--------|---------|-------------|
| [CRF](./crf) | `CRFTrainer`, `CRFModel`, `CRFTagger`, `CRFFeaturizer` | Conditional Random Fields for sequence labeling |
| [Logistic Regression](./lr) | `LRTrainer`, `LRModel`, `LRClassifier` | Logistic regression for text classification |
| [Text Classifier](./text-classifier) | `TextClassifier`, `LinearSVC`, `Label`, `Sentence` | End-to-end TF-IDF + SVM text classification pipeline |
| [TF-IDF](./tfidf) | `TfIdfVectorizer` | TF-IDF vectorization with n-gram support |
| [Text Preprocessor](./text-preprocessor) | `TextPreprocessor` | Vietnamese text preprocessing pipeline |

## Quick Start

```python
from underthesea_core import CRFTrainer, CRFTagger, CRFModel

# Train a CRF model
trainer = CRFTrainer(loss_function="lbfgs", max_iterations=100)
model = trainer.train(X_train, y_train)
model.save("model.bin")

# Load and predict
tagger = CRFTagger()
tagger.load("model.bin")
labels = tagger.tag(features)
```

## Performance

- Built with Rust for optimal processing speed
- L-BFGS optimizer with OWL-QN for L1 regularization
- 10x faster feature lookup with flat data structure
- 1.24x faster than python-crfsuite for word segmentation
