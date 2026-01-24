# Quick Start

Get started with Underthesea in 5 minutes!

## Step 1: Install Underthesea

```bash
pip install underthesea
```

## Step 2: Try Your First NLP Task

### Word Segmentation

Vietnamese text doesn't have spaces between words. Underthesea can segment text into words:

```python
from underthesea import word_tokenize

text = "Chàng trai 9X Quảng Trị khởi nghiệp từ nấm sò"
words = word_tokenize(text)
print(words)
# ['Chàng trai', '9X', 'Quảng Trị', 'khởi nghiệp', 'từ', 'nấm', 'sò']
```

### POS Tagging

Label each word with its part-of-speech:

```python
from underthesea import pos_tag

text = "Chợ thịt chó nổi tiếng ở Sài Gòn bị truy quét"
tagged = pos_tag(text)
print(tagged)
# [('Chợ', 'N'), ('thịt', 'N'), ('chó', 'N'), ('nổi tiếng', 'A'),
#  ('ở', 'E'), ('Sài Gòn', 'Np'), ('bị', 'V'), ('truy quét', 'V')]
```

### Named Entity Recognition

Identify named entities like people, locations, and organizations:

```python
from underthesea import ner

text = "Chưa tiết lộ lịch trình tới Việt Nam của Tổng thống Mỹ Donald Trump"
entities = ner(text)
for word, pos, chunk, entity in entities:
    if entity != 'O':
        print(f"{word}: {entity}")
# Việt Nam: B-LOC
# Mỹ: B-LOC
# Donald: B-PER
# Trump: I-PER
```

### Text Classification

Categorize Vietnamese text:

```python
from underthesea import classify

text = "HLV đầu tiên ở Premier League bị sa thải sau 4 vòng đấu"
category = classify(text)
print(category)
# ['The thao']
```

### Sentiment Analysis

Determine the sentiment of text:

```python
from underthesea import sentiment

text = "Sản phẩm hơi nhỏ so với tưởng tượng nhưng chất lượng tốt"
result = sentiment(text)
print(result)
# 'positive'
```

## Step 3: Explore More Features

### Translation (requires `[deep]` install)

```bash
pip install underthesea[deep]
```

```python
from underthesea import translate

text = "Hà Nội là thủ đô của Việt Nam"
english = translate(text)
print(english)
# 'Hanoi is the capital of Vietnam'
```

### Language Detection (requires `[langdetect]` install)

```bash
pip install underthesea[langdetect]
```

```python
from underthesea import lang_detect

text = "Cựu binh Mỹ trả nhật ký nhẹ lòng"
lang = lang_detect(text)
print(lang)
# 'vi'
```

## Next Steps

- Read the [Tutorials](tutorials/word_segmentation.md) for in-depth guides
- Explore the [API Reference](api/index.md) for complete documentation
- Learn about [Optional Dependencies](user_guide/optional_deps.md)
- Check out the [Concepts](user_guide/concepts.md) to understand how Underthesea works
