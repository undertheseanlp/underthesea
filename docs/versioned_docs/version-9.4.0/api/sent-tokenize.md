# sent_tokenize

Segment text into sentences.

## Usage

```python
from underthesea import sent_tokenize

text = 'Taylor cho biết lúc đầu cô cảm thấy ngại với cô bạn thân Amanda nhưng rồi mọi thứ trôi qua nhanh chóng. Amanda cũng thoải mái với mối quan hệ này.'

sentences = sent_tokenize(text)
print(sentences)
# [
#   "Taylor cho biết lúc đầu cô cảm thấy ngại với cô bạn thân Amanda nhưng rồi mọi thứ trôi qua nhanh chóng.",
#   "Amanda cũng thoải mái với mối quan hệ này."
# ]
```

## Function Signature

```python
def sent_tokenize(text: str) -> list[str]
```

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `text` | `str` | The input text to segment into sentences |

## Returns

| Type | Description |
|------|-------------|
| `list[str]` | A list of sentences |

## Examples

### Basic Usage

```python
from underthesea import sent_tokenize

text = "Xin chào. Tôi là sinh viên. Tôi học ở Hà Nội."
sentences = sent_tokenize(text)
print(sentences)
# ['Xin chào.', 'Tôi là sinh viên.', 'Tôi học ở Hà Nội.']
```

### Multiple Sentences

```python
text = """Việt Nam là một quốc gia. Thủ đô là Hà Nội. Thành phố lớn nhất là TP. Hồ Chí Minh."""
sentences = sent_tokenize(text)
print(len(sentences))  # 3
```

### Handling Abbreviations

The function handles common Vietnamese abbreviations:

```python
text = "TP. Hồ Chí Minh là thành phố lớn nhất Việt Nam. Dân số khoảng 9 triệu người."
sentences = sent_tokenize(text)
print(sentences)
# ['TP. Hồ Chí Minh là thành phố lớn nhất Việt Nam.', 'Dân số khoảng 9 triệu người.']
```

## Notes

- The function uses rule-based sentence boundary detection
- It handles common Vietnamese punctuation patterns
- Abbreviations like "TP." (thành phố) are handled correctly
