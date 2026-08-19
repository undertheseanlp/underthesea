# restore_diacritics

Restore diacritics (thêm dấu) for Vietnamese text written without them.

## Usage

```python
from underthesea import restore_diacritics

text = "chung ta co the lam duoc"
restored = restore_diacritics(text)
print(restored)
# "chúng ta có thể làm được"
```

## Function Signature

```python
def restore_diacritics(text: str) -> str
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | | The input text, with or without diacritics |

## Returns

| Type | Description |
|------|-------------|
| `str` | The text with Vietnamese diacritics restored |

## Examples

### Basic Usage

```python
from underthesea import restore_diacritics

restore_diacritics("toi yeu viet nam")
# "tôi yêu việt nam"
```

### Context Awareness

Ambiguous syllables are resolved from their context using a syllable
bigram model with Viterbi decoding:

```python
restore_diacritics("chung ta co the lam duoc")
# "chúng ta có thể làm được"

restore_diacritics("Ha Noi la thu do cua Viet Nam")
# "Hà Nội là thủ đô của Việt Nam"
```

### Mixed Content

Numbers, URLs, emails and syllables that already carry diacritics are
kept unchanged:

```python
restore_diacritics("toi co 2 con meo.")
# "tôi có 2 con mèo."

restore_diacritics("email cua toi la test@gmail.com")
# "email của tôi là test@gmail.com"
```

## Notes

- The first call lazily loads a ~1.5 MB n-gram model (once per process);
  subsequent calls take well under a millisecond per sentence.
- This is the lightweight mode requested in
  [issue #766](https://github.com/undertheseanlp/underthesea/issues/766):
  dictionary + n-gram statistics, no deep learning dependencies. A
  context-aware AI mode may be added later.
