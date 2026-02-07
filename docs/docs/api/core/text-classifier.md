# Text Classifier

End-to-end text classification combining TF-IDF vectorization and Linear SVM, running entirely in Rust for maximum performance.

## TextClassifier

Unified TF-IDF + SVM pipeline that avoids Python-Rust boundary overhead for intermediate vectors.

### Usage

```python
from underthesea_core import TextClassifier

texts = [
    "sản phẩm rất tốt",
    "hàng đẹp giá rẻ",
    "hàng xấu quá",
    "tệ lắm không mua nữa",
]
labels = ["positive", "positive", "negative", "negative"]

clf = TextClassifier(max_features=20000, ngram_range=(1, 2), c=1.0)
clf.fit(texts, labels)

# Single prediction
label = clf.predict("sản phẩm tốt")

# Prediction with confidence
label, score = clf.predict_with_score("sản phẩm tốt")

# Batch prediction
labels = clf.predict_batch(["sản phẩm tốt", "hàng xấu"])

# Save/load
clf.save("classifier.bin")
clf = TextClassifier.load("classifier.bin")
```

### Constructor

```python
TextClassifier(
    max_features=20000,
    ngram_range=(1, 2),
    min_df=1,
    max_df=1.0,
    c=1.0,
    max_iter=1000,
    tol=0.1,
    preprocessor=None,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_features` | `int` | `20000` | Maximum vocabulary size |
| `ngram_range` | `(int, int)` | `(1, 2)` | Min and max n-gram range |
| `min_df` | `int` | `1` | Minimum document frequency |
| `max_df` | `float` | `1.0` | Maximum document frequency ratio |
| `c` | `float` | `1.0` | SVM regularization parameter |
| `max_iter` | `int` | `1000` | Maximum SVM training iterations |
| `tol` | `float` | `0.1` | Convergence tolerance |
| `preprocessor` | `TextPreprocessor` | `None` | Optional text preprocessor |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `is_fitted` | `bool` | Whether the model has been trained |
| `n_features` | `int` | Vocabulary size |
| `classes` | `list[str]` | Class labels |
| `preprocessor` | `TextPreprocessor` | The preprocessor (if set) |

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `fit(texts, labels)` | `None` | Train on texts and labels |
| `predict(text)` | `str` | Predict label for a single text |
| `predict_with_score(text)` | `(str, float)` | Predict with confidence score |
| `predict_batch(texts)` | `list[str]` | Predict labels for multiple texts |
| `predict_with_scores(texts)` | `list[(str, float)]` | Batch predict with scores |
| `predict_sentence(sentence)` | `None` | Predict and add labels to a `Sentence` object |
| `save(path)` | `None` | Save model to binary file |
| `load(path)` | `TextClassifier` | Load model from binary file (static) |

### With Preprocessor

```python
from underthesea_core import TextClassifier, TextPreprocessor

pp = TextPreprocessor()
clf = TextClassifier(preprocessor=pp)
clf.fit(texts, labels)

# Teencode is auto-expanded before prediction
clf.predict("sp ko tốt")  # "sp" → "sản phẩm", "ko" → "không"
```

---

## LinearSVC

LIBLINEAR-style linear SVM using Dual Coordinate Descent. Used internally by `TextClassifier`, but also available standalone.

### Usage

```python
from underthesea_core import LinearSVC

svm = LinearSVC()
svm.fit(features, labels, c=1.0, max_iter=1000, tol=0.1)

label = svm.predict(features_single)
labels = svm.predict_batch(features_batch)

svm.save("svm.bin")
svm = LinearSVC.load("svm.bin")
```

### Constructor

```python
LinearSVC()
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `classes` | `list[str]` | Class labels |
| `n_features` | `int` | Number of features |

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `fit(features, labels, c=1.0, max_iter=1000, tol=0.1)` | `None` | Train the SVM classifier |
| `predict(features)` | `str` | Predict label for a single instance |
| `predict_batch(batch)` | `list[str]` | Predict labels for a batch |
| `save(path)` | `None` | Save model to file |
| `load(path)` | `LinearSVC` | Load model from file (static) |

### `fit` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `features` | `list[list[float]]` | | Dense feature vectors |
| `labels` | `list[str]` | | Class labels |
| `c` | `float` | `1.0` | Regularization parameter |
| `max_iter` | `int` | `1000` | Maximum iterations |
| `tol` | `float` | `0.1` | Convergence tolerance |

---

## Label

A classification label with value and confidence score. Compatible with the underthesea API.

### Usage

```python
from underthesea_core import Label

label = Label("positive", 0.95)
print(label.value)  # "positive"
print(label.score)  # 0.95
```

### Constructor

```python
Label(value, score=1.0)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `value` | `str` | | Label text |
| `score` | `float` | `1.0` | Confidence score (clamped to 0.0-1.0) |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `value` | `str` | Label text (read/write) |
| `score` | `float` | Confidence score (read/write) |

---

## Sentence

A text sentence with associated labels. Compatible with the underthesea API.

### Usage

```python
from underthesea_core import Sentence, Label

sentence = Sentence("sản phẩm rất tốt")
sentence.add_label(Label("positive", 0.95))
print(sentence.text)    # "sản phẩm rất tốt"
print(sentence.labels)  # [positive (0.9500)]
```

### Constructor

```python
Sentence(text="", labels=None)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | `""` | Sentence text |
| `labels` | `list[Label]` | `None` | Initial labels |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `text` | `str` | Sentence text (read/write) |
| `labels` | `list[Label]` | Associated labels (read/write) |

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `add_label(label)` | `None` | Add a single label |
| `add_labels(labels)` | `None` | Add multiple labels |
