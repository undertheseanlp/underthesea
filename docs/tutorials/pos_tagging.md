# POS Tagging Tutorial

Learn how to perform Part-of-Speech tagging on Vietnamese text.

## Introduction

Part-of-Speech (POS) tagging assigns grammatical labels (noun, verb, adjective, etc.) to each word in a sentence. This is essential for understanding the structure of text.

## Basic Usage

```python
from underthesea import pos_tag

text = "Chợ thịt chó nổi tiếng ở Sài Gòn bị truy quét"
tagged = pos_tag(text)
print(tagged)
# [('Chợ', 'N'), ('thịt', 'N'), ('chó', 'N'), ('nổi tiếng', 'A'),
#  ('ở', 'E'), ('Sài Gòn', 'Np'), ('bị', 'V'), ('truy quét', 'V')]
```

## Understanding POS Tags

| Tag | Meaning | Examples |
|-----|---------|----------|
| `N` | Common noun | chợ, thịt, chó, sách |
| `Np` | Proper noun | Việt Nam, Hà Nội, Trump |
| `V` | Verb | chạy, ăn, nói, bị |
| `A` | Adjective | đẹp, nổi tiếng, lớn |
| `P` | Pronoun | tôi, bạn, nó, họ |
| `R` | Adverb | rất, đang, sẽ, đã |
| `E` | Preposition | ở, trong, trên, dưới |
| `C` | Conjunction | và, hoặc, nhưng |
| `M` | Number | một, hai, ba, 100 |
| `L` | Determiner | các, những, mọi |
| `CH` | Punctuation | . , ? ! |

## Practical Examples

### Extracting Nouns

```python
from underthesea import pos_tag

text = "Sinh viên Đại học Bách Khoa Hà Nội đạt giải nhất cuộc thi lập trình"
tagged = pos_tag(text)

# Extract all nouns
nouns = [word for word, tag in tagged if tag in ('N', 'Np')]
print("Nouns:", nouns)
# ['Sinh viên', 'Đại học', 'Bách Khoa', 'Hà Nội', 'giải', 'cuộc thi', 'lập trình']
```

### Extracting Verbs

```python
text = "Anh ấy chạy nhanh và nhảy cao"
tagged = pos_tag(text)

verbs = [word for word, tag in tagged if tag == 'V']
print("Verbs:", verbs)
# ['chạy', 'nhảy']
```

### Finding Proper Nouns (Names, Locations)

```python
text = "Tổng thống Mỹ Donald Trump thăm Việt Nam vào tháng 11"
tagged = pos_tag(text)

proper_nouns = [word for word, tag in tagged if tag == 'Np']
print("Proper nouns:", proper_nouns)
# ['Mỹ', 'Donald', 'Trump', 'Việt Nam']
```

### Analyzing Sentence Structure

```python
from collections import Counter

text = "Hà Nội là thủ đô xinh đẹp của Việt Nam với nhiều địa điểm du lịch nổi tiếng"
tagged = pos_tag(text)

# Count POS distribution
pos_counts = Counter(tag for word, tag in tagged)
print("POS Distribution:")
for tag, count in pos_counts.most_common():
    print(f"  {tag}: {count}")
```

### Building a Simple Grammar Checker

```python
from underthesea import pos_tag

def check_sentence_structure(text):
    """Check if sentence has basic subject-verb structure."""
    tagged = pos_tag(text)

    has_noun = any(tag in ('N', 'Np', 'P') for word, tag in tagged)
    has_verb = any(tag == 'V' for word, tag in tagged)

    if has_noun and has_verb:
        return "Valid sentence structure"
    elif not has_verb:
        return "Warning: No verb found"
    else:
        return "Warning: No subject found"

sentences = [
    "Tôi yêu Việt Nam",
    "Rất đẹp",
    "Chạy nhanh"
]

for sent in sentences:
    result = check_sentence_structure(sent)
    print(f"{sent}: {result}")
```

### Filtering Content Words

```python
def get_content_words(text):
    """Extract content words (nouns, verbs, adjectives)."""
    tagged = pos_tag(text)
    content_tags = {'N', 'Np', 'V', 'A'}
    return [word for word, tag in tagged if tag in content_tags]

text = "Việt Nam là một đất nước xinh đẹp với nhiều cảnh quan tuyệt vời"
content = get_content_words(text)
print("Content words:", content)
# ['Việt Nam', 'đất nước', 'xinh đẹp', 'cảnh quan', 'tuyệt vời']
```

## Combining with Word Segmentation

POS tagging automatically performs word segmentation first:

```python
from underthesea import word_tokenize, pos_tag

text = "Tôi yêu Việt Nam"

# These produce consistent results
words = word_tokenize(text)
tagged = pos_tag(text)

print(f"Words: {words}")
print(f"Tagged: {tagged}")
# Words: ['Tôi', 'yêu', 'Việt Nam']
# Tagged: [('Tôi', 'P'), ('yêu', 'V'), ('Việt Nam', 'Np')]
```

## Next Steps

- Learn about [Named Entity Recognition](ner.md)
- Explore [Chunking](../api/chunk.md)
- See the [API Reference](../api/pos_tag.md)
