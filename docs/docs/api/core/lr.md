# Logistic Regression

Classes for training and running logistic regression classifiers.

## LRTrainer

Train a logistic regression model with SGD optimization.

### Usage

```python
from underthesea_core import LRTrainer

X_train = [
    ["word=tốt", "len=3"],
    ["word=xấu", "len=3"],
    ["word=đẹp", "len=3"],
]
y_train = ["positive", "negative", "positive"]

trainer = LRTrainer(
    l2_penalty=0.01,
    learning_rate=0.1,
    max_epochs=100,
    verbose=1,
)
model = trainer.train(X_train, y_train)
model.save("lr_model.bin")
```

### Constructor

```python
LRTrainer(
    l1_penalty=0.0,
    l2_penalty=0.01,
    learning_rate=0.1,
    max_epochs=100,
    batch_size=1,
    tol=1e-4,
    verbose=1,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `l1_penalty` | `float` | `0.0` | L1 regularization (lasso) |
| `l2_penalty` | `float` | `0.01` | L2 regularization (ridge) |
| `learning_rate` | `float` | `0.1` | Learning rate for SGD |
| `max_epochs` | `int` | `100` | Maximum training epochs |
| `batch_size` | `int` | `1` | Mini-batch size (1 = pure SGD) |
| `tol` | `float` | `1e-4` | Convergence tolerance for early stopping |
| `verbose` | `int` | `1` | Verbosity: 0=quiet, 1=progress, 2=detailed |

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `train(X, y)` | `LRModel` | Train on feature lists X and labels y |
| `set_l1_penalty(penalty)` | `None` | Set L1 regularization |
| `set_l2_penalty(penalty)` | `None` | Set L2 regularization |
| `set_learning_rate(lr)` | `None` | Set learning rate |
| `set_max_epochs(epochs)` | `None` | Set maximum epochs |
| `set_batch_size(size)` | `None` | Set batch size |
| `get_model()` | `LRModel` | Get the current model |

---

## LRModel

Stores trained logistic regression model weights and class labels.

### Usage

```python
from underthesea_core import LRModel

# Load a saved model
model = LRModel.load("lr_model.bin")
print(model.num_classes)    # number of classes
print(model.num_features)   # number of features
print(model.get_classes())  # list of class labels

# Create with predefined classes
model = LRModel.with_classes(["positive", "negative"])
```

### Constructor

```python
LRModel()
```

### Static Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `load(path)` | `LRModel` | Load model from file |
| `with_classes(classes)` | `LRModel` | Create model with predefined classes |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `num_classes` | `int` | Number of classes |
| `num_features` | `int` | Number of features |

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `save(path)` | `None` | Save model to file |
| `get_classes()` | `list[str]` | Get all class labels |
| `num_weights()` | `int` | Get number of non-zero weights |
| `l2_norm_squared()` | `float` | Get L2 norm squared of all weights |
| `l1_norm()` | `float` | Get L1 norm of all weights |

---

## LRClassifier

Load a trained LR model and make predictions.

### Usage

```python
from underthesea_core import LRClassifier

# Load from file
classifier = LRClassifier.load("lr_model.bin")

# Or create from model
from underthesea_core import LRModel
model = LRModel.load("lr_model.bin")
classifier = LRClassifier.from_model(model)

# Predict
features = ["word=tốt", "len=3"]
label = classifier.predict(features)

# Predict with probability
label, prob = classifier.predict_with_prob(features)

# Get probability distribution
proba = classifier.predict_proba(features)
# [("positive", 0.85), ("negative", 0.15)]

# Get top-k predictions
top2 = classifier.predict_top_k(features, k=2)
```

### Constructor

```python
LRClassifier()
```

### Static Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `load(path)` | `LRClassifier` | Load classifier from file |
| `from_model(model)` | `LRClassifier` | Create classifier from an `LRModel` |

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `predict(features)` | `str` | Predict the most likely class |
| `predict_with_prob(features)` | `(str, float)` | Predict with probability |
| `predict_proba(features)` | `list[(str, float)]` | Get probability distribution over all classes |
| `predict_top_k(features, k)` | `list[(str, float)]` | Get top-k most likely classes with probabilities |
| `num_classes()` | `int` | Get number of classes |
| `classes()` | `list[str]` | Get all class labels |
