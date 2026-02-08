# word_tokenize

Segment Vietnamese text into words.

## Usage

```python
from underthesea import word_tokenize

text = "Chàng trai 9X Quảng Trị khởi nghiệp từ nấm sò"
words = word_tokenize(text)
print(words)
# ["Chàng trai", "9X", "Quảng Trị", "khởi nghiệp", "từ", "nấm", "sò"]
```

## Function Signature

```python
def word_tokenize(
    sentence: str,
    format: str = None,
    use_token_normalize: bool = True,
    fixed_words: list = None
) -> list[str] | str
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sentence` | `str` | | The input text to tokenize |
| `format` | `str` | `None` | Output format: `None` for list, `"text"` for string |
| `use_token_normalize` | `bool` | `True` | Whether to normalize tokens |
| `fixed_words` | `list` | `None` | List of words that should not be split |

## Returns

| Type | Description |
|------|-------------|
| `list[str]` | List of words (when `format=None`) |
| `str` | Space-separated string with underscores (when `format="text"`) |

## Examples

### Basic Usage

```python
from underthesea import word_tokenize

text = "Chàng trai 9X Quảng Trị khởi nghiệp từ nấm sò"
words = word_tokenize(text)
print(words)
# ["Chàng trai", "9X", "Quảng Trị", "khởi nghiệp", "từ", "nấm", "sò"]
```

### Text Format

```python
text = "Chàng trai 9X Quảng Trị khởi nghiệp từ nấm sò"
result = word_tokenize(text, format="text")
print(result)
# "Chàng_trai 9X Quảng_Trị khởi_nghiệp từ nấm sò"
```

### Fixed Words

Use `fixed_words` to ensure certain words are kept together:

```python
text = "Viện Nghiên Cứu chiến lược quốc gia về học máy"
fixed_words = ["Viện Nghiên Cứu", "học máy"]
result = word_tokenize(text, fixed_words=fixed_words, format="text")
print(result)
# "Viện_Nghiên_Cứu chiến_lược quốc_gia về học_máy"
```

### Processing Multiple Sentences

```python
sentences = [
    "Tôi yêu Việt Nam",
    "Hà Nội là thủ đô của Việt Nam"
]

for sentence in sentences:
    words = word_tokenize(sentence)
    print(words)
# ['Tôi', 'yêu', 'Việt Nam']
# ['Hà Nội', 'là', 'thủ đô', 'của', 'Việt Nam']
```

## Notes

- Vietnamese word segmentation is challenging because spaces don't always indicate word boundaries
- The function uses a CRF model trained on Vietnamese text
- Multi-syllable words are joined (e.g., "Việt Nam" is one word, not two)
- Use `fixed_words` parameter for domain-specific terminology
