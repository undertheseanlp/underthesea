# text_normalize

Normalize Vietnamese text by fixing common encoding and diacritic issues.

## Usage

```python
from underthesea import text_normalize

text = "Ðảm baỏ chất lựơng phòng thí nghịêm hoá học"
normalized = text_normalize(text)
print(normalized)
# "Đảm bảo chất lượng phòng thí nghiệm hóa học"
```

## Function Signature

```python
def text_normalize(text: str, tokenizer: str = 'underthesea') -> str
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | | The input text to normalize |
| `tokenizer` | `str` | `'underthesea'` | The tokenizer to use |

## Returns

| Type | Description |
|------|-------------|
| `str` | The normalized text |

## Examples

### Basic Usage

```python
from underthesea import text_normalize

# Fix diacritic issues
text_normalize("Ðảm baỏ chất lựơng")
# "Đảm bảo chất lượng"
```

### Common Fixes

```python
# Fix Đ/Ð confusion
text_normalize("Ðại học")
# "Đại học"

# Fix vowel diacritics
text_normalize("hoá học")
# "hóa học"

# Fix tone marks
text_normalize("nghịêm")
# "nghiệm"
```

### Full Text Normalization

```python
text = "Ðây là một ví dụ về việc chuẩn hoá văn bản tiếng Việt"
normalized = text_normalize(text)
print(normalized)
# "Đây là một ví dụ về việc chuẩn hóa văn bản tiếng Việt"
```

## Common Issues Fixed

| Issue | Example | Fixed |
|-------|---------|-------|
| Đ/Ð confusion | `Ðại` | `Đại` |
| Old-style diacritics | `hoá` | `hóa` |
| Incorrect vowel composition | `lựơng` | `lượng` |
| Unicode normalization | Various | NFC form |

## Notes

- This function is useful for preprocessing Vietnamese text
- It handles common encoding issues from legacy systems
- The output is in Unicode NFC normalized form
