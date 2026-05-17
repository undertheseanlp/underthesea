# TextPreprocessor

Configurable Vietnamese text preprocessing pipeline. Serializable with the `TextClassifier` model so preprocessing config always travels with the model.

## Pipeline Steps

Applied in order:

1. Unicode NFC normalization
2. Lowercase
3. URL removal
4. Repeated character normalization (`"đẹppp"` → `"đẹpp"`)
5. Punctuation normalization (`"!!!"` → `"!"`, `"????"` → `"?"`)
6. Teencode expansion (`"ko"` → `"không"`, `"dc"` → `"được"`)
7. Negation marking (`"không tốt"` → `"không NEG_tốt"`)

## Usage

```python
from underthesea_core import TextPreprocessor

# Default Vietnamese preprocessing
pp = TextPreprocessor()
pp.transform("Sản phẩm ko đẹp lắm!!!")
# "sản phẩm không NEG_đẹp NEG_lắm!"

# Batch processing
results = pp.transform_batch(["Ko đẹp", "SP tốt lắm!!!"])
# ["không NEG_đẹp", "sản phẩm tốt lắm!"]

# Custom teencode dictionary
pp = TextPreprocessor(teencode={"ko": "không", "dc": "được"})

# Custom negation words and window
pp = TextPreprocessor(
    negation_words=["không", "chưa", "chẳng"],
    negation_window=3,
)

# Disable specific steps
pp = TextPreprocessor(lowercase=False, remove_urls=False)

# Disable teencode and negation entirely
pp = TextPreprocessor(teencode=None, negation_words=None, use_defaults=False)
```

## Constructor

```python
TextPreprocessor(
    lowercase=True,
    unicode_normalize=True,
    remove_urls=True,
    normalize_repeated_chars=True,
    normalize_punctuation=True,
    teencode=None,
    negation_words=None,
    negation_window=2,
    use_defaults=True,
)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lowercase` | `bool` | `True` | Convert text to lowercase |
| `unicode_normalize` | `bool` | `True` | Apply Unicode NFC normalization |
| `remove_urls` | `bool` | `True` | Remove URLs (http/https/www) |
| `normalize_repeated_chars` | `bool` | `True` | Reduce 3+ repeated chars to 2 |
| `normalize_punctuation` | `bool` | `True` | Reduce repeated punctuation |
| `teencode` | `dict \| None` | `None` | Custom teencode dictionary. With `use_defaults=True`, defaults to built-in Vietnamese teencode |
| `negation_words` | `list[str] \| None` | `None` | Custom negation words. With `use_defaults=True`, defaults to built-in Vietnamese negation words |
| `negation_window` | `int` | `2` | Number of words after negation word to mark with `NEG_` prefix |
| `use_defaults` | `bool` | `True` | When `True`, use Vietnamese defaults for teencode/negation if not provided. When `False`, `None` means disabled |

## Properties

| Property | Type | Description |
|----------|------|-------------|
| `teencode` | `dict \| None` | Current teencode dictionary |
| `negation_words` | `list[str] \| None` | Current negation words |
| `negation_window` | `int` | Current negation window size |

## Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `transform(text)` | `str` | Preprocess a single text string |
| `transform_batch(texts)` | `list[str]` | Preprocess a list of texts |

## Default Teencode Dictionary

| Teencode | Expansion |
|----------|-----------|
| `ko`, `k`, `hok`, `hem` | không |
| `dc`, `đc`, `dk` | được |
| `sp` | sản phẩm |
| `bt`, `bth` | bình thường |
| `ok`, `oke` | tốt |
| `tks`, `thanks`, `thank` | cảm ơn |
| `ntn` | như thế nào |
| `mn` | mọi người |
| `cx`, `cg` | cũng |
| `vs` | với |
| ... | *(30+ rules total)* |

## Default Negation Words

`không`, `chẳng`, `chả`, `chưa`, `đừng`, `ko`, `hok`, `hem`, `chăng`

## With TextClassifier

```python
from underthesea_core import TextClassifier, TextPreprocessor

pp = TextPreprocessor()
clf = TextClassifier(preprocessor=pp)
clf.fit(texts, labels)

# Preprocessor is saved together with the model
clf.save("model.bin")
clf = TextClassifier.load("model.bin")  # preprocessor is restored
```
