# CRF (Conditional Random Fields)

Classes for training, loading, and running CRF sequence labeling models.

## CRFTrainer

Train a CRF model with L-BFGS or Structured Perceptron optimization.

### Usage

```python
from underthesea_core import CRFTrainer

X_train = [
    [["word=Tôi", "is_upper=False"], ["word=yêu", "is_upper=False"],
     ["word=Việt", "is_upper=True"], ["word=Nam", "is_upper=True"]],
]
y_train = [
    ["O", "O", "B-LOC", "I-LOC"],
]

trainer = CRFTrainer(
    loss_function="lbfgs",
    l1_penalty=1.0,
    l2_penalty=0.001,
    max_iterations=100,
    verbose=1,
)
model = trainer.train(X_train, y_train)
model.save("ner_model.bin")
```

### Constructor

```python
CRFTrainer(
    loss_function="lbfgs",
    l1_penalty=0.0,
    l2_penalty=0.01,
    learning_rate=0.1,
    max_iterations=100,
    averaging=True,
    verbose=1,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `loss_function` | `str` | `"lbfgs"` | `"lbfgs"` (recommended) or `"perceptron"` |
| `l1_penalty` | `float` | `0.0` | L1 regularization coefficient |
| `l2_penalty` | `float` | `0.01` | L2 regularization coefficient |
| `learning_rate` | `float` | `0.1` | Learning rate (perceptron only) |
| `max_iterations` | `int` | `100` | Maximum training iterations |
| `averaging` | `bool` | `True` | Use averaged perceptron (perceptron only) |
| `verbose` | `int` | `1` | Verbosity: 0=quiet, 1=progress, 2=detailed |

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `train(X, y)` | `CRFModel` | Train on sequences X (list of feature lists) and labels y |
| `set_l1_penalty(penalty)` | `None` | Set L1 regularization penalty |
| `set_l2_penalty(penalty)` | `None` | Set L2 regularization penalty |
| `set_max_iterations(max_iter)` | `None` | Set maximum iterations |
| `get_model()` | `CRFModel` | Get the current model |

---

## CRFModel

Stores trained CRF model weights, labels, and features. Supports save/load.

### Usage

```python
from underthesea_core import CRFModel

# Load a saved model
model = CRFModel.load("ner_model.bin")
print(model.num_labels)        # number of labels
print(model.num_attributes)    # number of attributes
print(model.get_labels())      # list of label names

# Create with predefined labels
model = CRFModel.with_labels(["O", "B-LOC", "I-LOC"])
```

### Constructor

```python
CRFModel()
```

### Static Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `load(path)` | `CRFModel` | Load model from file |
| `with_labels(labels)` | `CRFModel` | Create model with predefined labels |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `num_labels` | `int` | Number of labels |
| `num_attributes` | `int` | Number of attributes |

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `save(path)` | `None` | Save model to file (CRFsuite format) |
| `get_labels()` | `list[str]` | Get all label names |
| `num_state_features()` | `int` | Get number of state features |
| `num_transition_features()` | `int` | Get number of transition features |
| `l2_norm_squared()` | `float` | Get L2 norm squared of all weights |
| `l1_norm()` | `float` | Get L1 norm of all weights |

---

## CRFTagger

Load a trained CRF model and make predictions on sequences.

### Usage

```python
from underthesea_core import CRFTagger, CRFModel

# Load directly
tagger = CRFTagger()
tagger.load("ner_model.bin")

# Or create from model
model = CRFModel.load("ner_model.bin")
tagger = CRFTagger.from_model(model)

# Predict
features = [
    ["word=Tôi", "is_upper=False"],
    ["word=sống", "is_upper=False"],
    ["word=ở", "is_upper=False"],
    ["word=Hà", "is_upper=True"],
    ["word=Nội", "is_upper=True"],
]
labels = tagger.tag(features)
# ['O', 'O', 'O', 'B-LOC', 'I-LOC']

# Get labels with score
labels, score = tagger.tag_with_score(features)

# Get marginal probabilities
marginals = tagger.marginals(features)
```

### Constructor

```python
CRFTagger()
```

### Static Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `from_model(model)` | `CRFTagger` | Create tagger from a `CRFModel` |

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `load(path)` | `None` | Load model from file |
| `tag(features)` | `list[str]` | Predict labels for a sequence |
| `tag_with_score(features)` | `(list[str], float)` | Predict labels with sequence score |
| `marginals(features)` | `list[list[float]]` | Get marginal probabilities per position and label |
| `labels()` | `list[str]` | Get all label names |
| `num_labels()` | `int` | Get number of labels |

---

## CRFFeaturizer

Extract features from tokenized sentences for CRF models.

### Usage

```python
from underthesea_core import CRFFeaturizer

features = ["T[-1]", "T[0]", "T[1]"]
dictionary = set(["sinh viên"])
featurizer = CRFFeaturizer(features, dictionary)

sentences = [[["sinh", "X"], ["viên", "X"], ["đi", "X"], ["học", "X"]]]
result = featurizer.process(sentences)
# [[['T[-1]=BOS', 'T[0]=sinh', 'T[1]=viên'],
#   ['T[-1]=sinh', 'T[0]=viên', 'T[1]=đi'],
#   ['T[-1]=viên', 'T[0]=đi', 'T[1]=học'],
#   ['T[-1]=đi', 'T[0]=học', 'T[1]=EOS']]]
```

### Constructor

```python
CRFFeaturizer(feature_configs, dictionary)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `feature_configs` | `list[str]` | Feature template strings (e.g., `["T[-1]", "T[0]", "T[1]"]`) |
| `dictionary` | `set[str]` | Dictionary of known words/phrases |

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `process(sentences)` | `list[list[list[str]]]` | Extract features from tokenized sentences |
