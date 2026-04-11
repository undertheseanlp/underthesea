# chunk

Group words into meaningful phrases (chunking/shallow parsing).

## Usage

```python
from underthesea import chunk

text = "Bác sĩ bây giờ có thể thản nhiên báo tin bệnh nhân bị ung thư?"
chunks = chunk(text)
print(chunks)
# [('Bác sĩ', 'N', 'B-NP'),
#  ('bây giờ', 'P', 'B-NP'),
#  ('có thể', 'R', 'O'),
#  ('thản nhiên', 'A', 'B-AP'),
#  ('báo', 'V', 'B-VP'),
#  ('tin', 'N', 'B-NP'),
#  ('bệnh nhân', 'N', 'B-NP'),
#  ('bị', 'V', 'B-VP'),
#  ('ung thư', 'N', 'B-NP'),
#  ('?', 'CH', 'O')]
```

## Function Signature

```python
def chunk(
    sentence: str,
    format: str = None
) -> list[tuple[str, str, str]]
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sentence` | `str` | | The input text to chunk |
| `format` | `str` | `None` | Output format (currently only `None` supported) |

## Returns

| Type | Description |
|------|-------------|
| `list[tuple[str, str, str]]` | List of (word, POS tag, chunk tag) tuples |

## Chunk Tags

| Tag | Description |
|-----|-------------|
| `B-NP` | Beginning of Noun Phrase |
| `I-NP` | Inside Noun Phrase |
| `B-VP` | Beginning of Verb Phrase |
| `I-VP` | Inside Verb Phrase |
| `B-AP` | Beginning of Adjective Phrase |
| `I-AP` | Inside Adjective Phrase |
| `B-PP` | Beginning of Prepositional Phrase |
| `I-PP` | Inside Prepositional Phrase |
| `O` | Outside any chunk |

## Examples

### Basic Usage

```python
from underthesea import chunk

text = "Bác sĩ bây giờ có thể thản nhiên báo tin bệnh nhân bị ung thư?"
chunks = chunk(text)
for word, pos, chunk_tag in chunks:
    print(f"{word:15} {pos:5} {chunk_tag}")
```

### Extracting Noun Phrases

```python
text = "Sinh viên Đại học Bách Khoa Hà Nội đạt giải nhất"
chunks = chunk(text)

# Extract noun phrases
current_np = []
noun_phrases = []

for word, pos, chunk_tag in chunks:
    if chunk_tag == 'B-NP':
        if current_np:
            noun_phrases.append(' '.join(current_np))
        current_np = [word]
    elif chunk_tag == 'I-NP':
        current_np.append(word)
    else:
        if current_np:
            noun_phrases.append(' '.join(current_np))
            current_np = []

if current_np:
    noun_phrases.append(' '.join(current_np))

print(noun_phrases)
```

### Extracting Verb Phrases

```python
text = "Tôi đang học tiếng Việt và sẽ đi du lịch"
chunks = chunk(text)

verb_phrases = [word for word, pos, tag in chunks if tag.endswith('VP')]
print(verb_phrases)
```

## Notes

- Chunking is performed on top of word segmentation and POS tagging
- It provides shallow syntactic structure without full parsing
- Useful for extracting noun phrases, verb phrases, etc.
