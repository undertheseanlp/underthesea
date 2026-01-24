# Word Segmentation Tutorial

Learn how to segment Vietnamese text into words using Underthesea.

## Introduction

Vietnamese text doesn't have spaces between words like English. For example:

- English: `"I love Vietnam"` → 3 words
- Vietnamese: `"Tôi yêu Việt Nam"` → 4 words (not 3!)

The phrase "Việt Nam" is a single word (country name), but it appears as two syllables separated by a space. Word segmentation identifies the correct word boundaries.

## Basic Usage

```python
from underthesea import word_tokenize

text = "Chàng trai 9X Quảng Trị khởi nghiệp từ nấm sò"
words = word_tokenize(text)
print(words)
# ["Chàng trai", "9X", "Quảng Trị", "khởi nghiệp", "từ", "nấm", "sò"]
```

Notice that:
- "Chàng trai" (young man) is identified as one word
- "Quảng Trị" (province name) is identified as one word
- "khởi nghiệp" (entrepreneurship) is identified as one word

## Output Formats

### List Format (Default)

```python
words = word_tokenize("Tôi yêu Việt Nam")
print(words)
# ['Tôi', 'yêu', 'Việt Nam']
print(type(words))
# <class 'list'>
```

### Text Format

Use `format="text"` to get a string with underscores joining multi-syllable words:

```python
result = word_tokenize("Tôi yêu Việt Nam", format="text")
print(result)
# 'Tôi yêu Việt_Nam'
```

This format is useful for:
- Training machine learning models
- Text preprocessing pipelines
- Downstream NLP tasks

## Using Fixed Words

Sometimes you need specific terms to be kept together. Use the `fixed_words` parameter:

```python
text = "Viện Nghiên Cứu chiến lược quốc gia về học máy"

# Without fixed_words
result = word_tokenize(text, format="text")
print(result)
# Might split "Viện Nghiên Cứu" or "học máy" incorrectly

# With fixed_words
fixed_words = ["Viện Nghiên Cứu", "học máy"]
result = word_tokenize(text, fixed_words=fixed_words, format="text")
print(result)
# 'Viện_Nghiên_Cứu chiến_lược quốc_gia về học_máy'
```

This is especially useful for:
- Domain-specific terminology
- Organization names
- Technical terms

## Practical Examples

### Processing a Document

```python
from underthesea import sent_tokenize, word_tokenize

document = """
Việt Nam là một quốc gia nằm ở Đông Nam Á. Thủ đô của Việt Nam là Hà Nội.
Thành phố lớn nhất là Thành phố Hồ Chí Minh.
"""

# First, segment into sentences
sentences = sent_tokenize(document)

# Then, tokenize each sentence
for sentence in sentences:
    words = word_tokenize(sentence)
    print(f"Sentence: {sentence}")
    print(f"Words: {words}")
    print(f"Word count: {len(words)}\n")
```

### Building a Word Frequency Counter

```python
from collections import Counter
from underthesea import word_tokenize

text = """
Việt Nam có nhiều địa điểm du lịch đẹp. Du lịch Việt Nam ngày càng phát triển.
Khách du lịch quốc tế đến Việt Nam ngày càng tăng.
"""

# Tokenize
words = word_tokenize(text)

# Count frequencies
word_counts = Counter(words)
print("Most common words:")
for word, count in word_counts.most_common(5):
    print(f"  {word}: {count}")
```

### Preprocessing for Machine Learning

```python
from underthesea import word_tokenize

def preprocess(text):
    """Preprocess Vietnamese text for ML."""
    # Tokenize with text format
    tokenized = word_tokenize(text, format="text")
    # Convert to lowercase
    tokenized = tokenized.lower()
    return tokenized

texts = [
    "Đội tuyển Việt Nam thắng đậm",
    "Giá vàng tăng mạnh hôm nay"
]

preprocessed = [preprocess(t) for t in texts]
print(preprocessed)
# ['đội_tuyển việt_nam thắng đậm', 'giá vàng tăng mạnh hôm_nay']
```

## Performance Tips

1. **Batch Processing**: Process texts in batches for efficiency
2. **Caching**: The model is loaded once and cached
3. **First Call**: First call is slower due to model loading

```python
# First call - slower (model loading)
word_tokenize("Hello")

# Subsequent calls - fast
for text in large_text_list:
    word_tokenize(text)
```

## Next Steps

- Learn about [POS Tagging](pos_tagging.md)
- Explore [Named Entity Recognition](ner.md)
- See the [API Reference](../api/word_tokenize.md)
