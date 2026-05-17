# TfIdfVectorizer

TF-IDF vectorization with n-gram support, L2 normalization, and sublinear TF scaling.

## Usage

```python
from underthesea_core import TfIdfVectorizer

documents = [
    "sản phẩm rất tốt",
    "hàng đẹp giá rẻ",
    "sản phẩm kém chất lượng",
]

vectorizer = TfIdfVectorizer(max_features=10000, ngram_range=(1, 2))
vectorizer.fit(documents)

# Sparse transform
sparse = vectorizer.transform("sản phẩm tốt")
# [(0, 0.577), (3, 0.577), (7, 0.577)]

# Dense transform
dense = vectorizer.transform_dense("sản phẩm tốt")
# [0.577, 0.0, 0.0, 0.577, ...]

# Feature strings for LRClassifier
features = vectorizer.transform_to_features("sản phẩm tốt")
# ["tfidf_0=0.5774", "tfidf_3=0.5774", ...]

# Fit and transform in one step
sparse_vectors = vectorizer.fit_transform(documents)

# Save/load
vectorizer.save("vectorizer.bin")
vectorizer = TfIdfVectorizer.load("vectorizer.bin")
```

## Constructor

```python
TfIdfVectorizer(
    min_df=1,
    max_df=1.0,
    max_features=0,
    sublinear_tf=False,
    lowercase=True,
    ngram_range=(1, 1),
    min_token_length=2,
    norm=True,
)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_df` | `int` | `1` | Minimum document frequency |
| `max_df` | `float` | `1.0` | Maximum document frequency ratio |
| `max_features` | `int` | `0` | Maximum vocabulary size (0 = unlimited) |
| `sublinear_tf` | `bool` | `False` | Use sublinear TF scaling (1 + log(tf)) |
| `lowercase` | `bool` | `True` | Convert text to lowercase |
| `ngram_range` | `(int, int)` | `(1, 1)` | Min and max n-gram range |
| `min_token_length` | `int` | `2` | Minimum token length |
| `norm` | `bool` | `True` | Apply L2 normalization |

## Properties

| Property | Type | Description |
|----------|------|-------------|
| `vocab_size` | `int` | Vocabulary size |
| `n_docs` | `int` | Number of documents used for fitting |

## Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `fit(documents)` | `None` | Fit the vectorizer on a list of documents |
| `transform(document)` | `list[(int, float)]` | Transform to sparse TF-IDF (feature_index, value) |
| `transform_dense(document)` | `list[float]` | Transform to dense TF-IDF vector |
| `transform_to_features(document)` | `list[str]` | Transform to feature strings for `LRClassifier` |
| `fit_transform(documents)` | `list[list[(int, float)]]` | Fit and transform in one step |
| `is_fitted()` | `bool` | Check if the vectorizer has been fitted |
| `get_feature_names()` | `list[str]` | Get vocabulary words in order |
| `get_idf()` | `list[float]` | Get IDF values for all features |
| `top_features_by_idf(n)` | `list[(str, float)]` | Get top n features by IDF value |
| `get_index(word)` | `int \| None` | Get index of a word in vocabulary |
| `get_word(index)` | `str \| None` | Get word at a given index |
| `save(path)` | `None` | Save vectorizer to file |
| `load(path)` | `TfIdfVectorizer` | Load vectorizer from file (static) |
