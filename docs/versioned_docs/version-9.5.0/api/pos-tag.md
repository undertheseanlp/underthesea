# pos_tag

Label words with their part-of-speech tags.

## Usage

```python
from underthesea import pos_tag

text = "Chợ thịt chó nổi tiếng ở Sài Gòn bị truy quét"
tagged = pos_tag(text)
print(tagged)
# [('Chợ', 'N'), ('thịt', 'N'), ('chó', 'N'), ('nổi tiếng', 'A'),
#  ('ở', 'E'), ('Sài Gòn', 'Np'), ('bị', 'V'), ('truy quét', 'V')]
```

## Function Signature

```python
def pos_tag(
    sentence: str,
    format: str = None,
    model: str = None
) -> list[tuple[str, str]]
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sentence` | `str` | | The input text to tag |
| `format` | `str` | `None` | Output format (currently only `None` supported) |
| `model` | `str` | `None` | Path to custom model |

## Returns

| Type | Description |
|------|-------------|
| `list[tuple[str, str]]` | List of (word, POS tag) tuples |

## POS Tags

| Tag | Description | Example |
|-----|-------------|---------|
| `N` | Noun | chợ, thịt, chó |
| `Np` | Proper noun | Sài Gòn, Việt Nam |
| `V` | Verb | bị, truy quét |
| `A` | Adjective | nổi tiếng, đẹp |
| `P` | Pronoun | tôi, bạn, nó |
| `R` | Adverb | rất, đang, sẽ |
| `E` | Preposition | ở, trong, trên |
| `C` | Conjunction | và, hoặc, nhưng |
| `M` | Number | một, hai, ba |
| `L` | Determiner | các, những, mọi |
| `X` | Unknown | - |
| `CH` | Punctuation | . , ? ! |

## Examples

### Basic Usage

```python
from underthesea import pos_tag

text = "Chợ thịt chó nổi tiếng ở Sài Gòn bị truy quét"
tagged = pos_tag(text)
for word, tag in tagged:
    print(f"{word}: {tag}")
# Chợ: N
# thịt: N
# chó: N
# nổi tiếng: A
# ở: E
# Sài Gòn: Np
# bị: V
# truy quét: V
```

### Filtering by POS Tag

```python
text = "Tôi yêu Việt Nam vì Việt Nam rất đẹp"
tagged = pos_tag(text)

# Get all nouns
nouns = [word for word, tag in tagged if tag in ('N', 'Np')]
print(nouns)
# ['Việt Nam', 'Việt Nam']

# Get all verbs
verbs = [word for word, tag in tagged if tag == 'V']
print(verbs)
# ['yêu']
```

### Processing Multiple Sentences

```python
sentences = [
    "Hà Nội là thủ đô của Việt Nam",
    "Thành phố Hồ Chí Minh là thành phố lớn nhất"
]

for sentence in sentences:
    tagged = pos_tag(sentence)
    print(tagged)
```

## Notes

- Word segmentation is performed automatically before POS tagging
- The model is trained on Vietnamese treebank data
- Proper nouns (Np) include names, locations, organizations
