# lang_detect

Identify the language of text.

!!! note "Requires Language Detection"
    This function requires the langdetect dependencies:
    ```bash
    pip install underthesea[langdetect]
    ```

## Usage

```python
from underthesea import lang_detect

text = "Cựu binh Mỹ trả nhật ký nhẹ lòng khi thấy cuộc sống hòa bình tại Việt Nam"
lang = lang_detect(text)
print(lang)
# 'vi'
```

## Function Signature

```python
def lang_detect(text: str) -> str
```

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `text` | `str` | The input text to analyze |

## Returns

| Type | Description |
|------|-------------|
| `str` | ISO 639-1 language code |

## Supported Languages

The function can detect 176 languages. Common codes:

| Code | Language |
|------|----------|
| `vi` | Vietnamese |
| `en` | English |
| `zh` | Chinese |
| `ja` | Japanese |
| `ko` | Korean |
| `fr` | French |
| `de` | German |
| `es` | Spanish |
| `ru` | Russian |
| `th` | Thai |

## Examples

### Basic Usage

```python
from underthesea import lang_detect

# Vietnamese
lang_detect("Cựu binh Mỹ trả nhật ký nhẹ lòng")
# 'vi'

# English
lang_detect("Hello, how are you today?")
# 'en'

# Chinese
lang_detect("你好，今天怎么样？")
# 'zh'

# Japanese
lang_detect("こんにちは、元気ですか？")
# 'ja'
```

### Detecting Multiple Texts

```python
from underthesea import lang_detect

texts = [
    "Xin chào Việt Nam",
    "Hello World",
    "Bonjour le monde",
    "Hallo Welt"
]

for text in texts:
    lang = lang_detect(text)
    print(f"{text} -> {lang}")
# Xin chào Việt Nam -> vi
# Hello World -> en
# Bonjour le monde -> fr
# Hallo Welt -> de
```

### Filtering by Language

```python
from underthesea import lang_detect

documents = [
    "Việt Nam là một đất nước xinh đẹp",
    "This is an English sentence",
    "Hôm nay trời đẹp quá",
    "The weather is nice today"
]

# Filter Vietnamese documents
vietnamese_docs = [doc for doc in documents if lang_detect(doc) == 'vi']
print(vietnamese_docs)
# ['Việt Nam là một đất nước xinh đẹp', 'Hôm nay trời đẹp quá']
```

### Language Statistics

```python
from collections import Counter
from underthesea import lang_detect

documents = [
    "Xin chào",
    "Hello",
    "Tạm biệt",
    "Goodbye",
    "Cảm ơn",
    "Merci"
]

langs = [lang_detect(doc) for doc in documents]
distribution = Counter(langs)
print(distribution)
# Counter({'vi': 3, 'en': 2, 'fr': 1})
```

## Notes

- Uses FastText's language identification model
- Works best with longer text (at least a few words)
- Very short text may be less accurate
- First call may take longer due to model loading
