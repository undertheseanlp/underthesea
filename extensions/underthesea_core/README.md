# Underthesea Core

[![PyPI version](https://badge.fury.io/py/underthesea-core.svg)](https://badge.fury.io/py/underthesea-core)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Underthesea Core is a powerful extension of the popular natural language processing library Underthesea, which includes a range of efficient data preprocessing tools and machine learning models for training. Built with Rust for optimal performance, Underthesea Core offers fast processing speeds and is easy to implement, with Python bindings for seamless integration into existing projects. This extension is an essential tool for developers looking to build high-performance NLP systems that deliver accurate and reliable results.

## Installation

```bash
pip install underthesea-core
```

## Version

**Current version: 2.0.0**

### What's New in 2.0.0

- L-BFGS optimizer with OWL-QN for L1 regularization
- 10x faster feature lookup with flat data structure
- 1.24x faster than python-crfsuite for word segmentation
- Loop unrolling and unsafe bounds-check elimination for performance

## Usage

### CRFTrainer

Train a CRF model with L-BFGS optimization:

```python
from underthesea_core import CRFTrainer, CRFTagger

# Prepare training data
# X: list of sequences, each sequence is a list of feature lists (one per token)
# y: list of label sequences
X_train = [
    [["word=Tôi", "is_upper=False"], ["word=yêu", "is_upper=False"], ["word=Việt", "is_upper=True"], ["word=Nam", "is_upper=True"]],
    [["word=Hà", "is_upper=True"], ["word=Nội", "is_upper=True"], ["word=đẹp", "is_upper=False"]],
]
y_train = [
    ["O", "O", "B-LOC", "I-LOC"],
    ["B-LOC", "I-LOC", "O"],
]

# Create trainer with L-BFGS optimizer
trainer = CRFTrainer(
    loss_function="lbfgs",  # L-BFGS with OWL-QN (recommended)
    l1_penalty=1.0,         # L1 regularization
    l2_penalty=0.001,       # L2 regularization
    max_iterations=100,
    verbose=1
)

# Train and get model
model = trainer.train(X_train, y_train)
print(f"Labels: {model.get_labels()}")
print(f"Features: {model.num_state_features()}")

# Save model
model.save("ner_model.bin")
```

### CRFTagger

Load a trained model and make predictions:

```python
from underthesea_core import CRFTagger, CRFModel

# Load model and create tagger
model = CRFModel.load("ner_model.bin")
tagger = CRFTagger.from_model(model)

# Or load directly
tagger = CRFTagger()
tagger.load("ner_model.bin")

# Predict labels for a sequence
features = [
    ["word=Tôi", "is_upper=False"],
    ["word=sống", "is_upper=False"],
    ["word=ở", "is_upper=False"],
    ["word=Hà", "is_upper=True"],
    ["word=Nội", "is_upper=True"],
]
labels = tagger.tag(features)
print(labels)  # ['O', 'O', 'O', 'B-LOC', 'I-LOC']

# Get labels with score
labels, score = tagger.tag_with_score(features)
print(f"Labels: {labels}, Score: {score}")

# Get marginal probabilities
marginals = tagger.marginals(features)
print(f"Marginals shape: {len(marginals)}x{len(marginals[0])}")
```

### CRFFeaturizer

Extract features from tokenized sentences:

```python
from underthesea_core import CRFFeaturizer

features = ["T[-1]", "T[0]", "T[1]"]
dictionary = set(["sinh viên"])
featurizer = CRFFeaturizer(features, dictionary)
sentences = [[["sinh", "X"], ["viên", "X"], ["đi", "X"], ["học", "X"]]]
featurizer.process(sentences)
# [[['T[-1]=BOS', 'T[0]=sinh', 'T[1]=viên'],
#   ['T[-1]=sinh', 'T[0]=viên', 'T[1]=đi'],
#   ['T[-1]=viên', 'T[0]=đi', 'T[1]=học'],
#   ['T[-1]=đi', 'T[0]=học', 'T[1]=EOS']]]
```

## API Reference

### CRFTrainer

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| loss_function | str | "lbfgs" | "lbfgs" (recommended) or "perceptron" |
| l1_penalty | float | 0.0 | L1 regularization coefficient |
| l2_penalty | float | 0.01 | L2 regularization coefficient |
| max_iterations | int | 100 | Maximum training iterations |
| learning_rate | float | 0.1 | Learning rate (perceptron only) |
| averaging | bool | True | Use averaged perceptron |
| verbose | int | 1 | Verbosity (0=quiet, 1=progress, 2=detailed) |

### CRFTagger

| Method | Description |
|--------|-------------|
| `tag(features)` | Predict labels for a sequence |
| `tag_with_score(features)` | Predict labels with sequence score |
| `marginals(features)` | Get marginal probabilities |
| `labels()` | Get all label names |
| `num_labels()` | Get number of labels |

### CRFModel

| Method | Description |
|--------|-------------|
| `save(path)` | Save model to file |
| `load(path)` | Load model from file |
| `get_labels()` | Get all label names |
| `num_state_features()` | Get number of state features |
| `num_transition_features()` | Get number of transition features |