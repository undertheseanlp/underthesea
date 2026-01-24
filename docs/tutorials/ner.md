# Named Entity Recognition Tutorial

Learn how to identify and extract named entities from Vietnamese text.

## Introduction

Named Entity Recognition (NER) identifies and classifies named entities in text into predefined categories such as:

- **PER**: Person names
- **LOC**: Locations (countries, cities, addresses)
- **ORG**: Organizations (companies, institutions)

## Basic Usage

```python
from underthesea import ner

text = "Chưa tiết lộ lịch trình tới Việt Nam của Tổng thống Mỹ Donald Trump"
entities = ner(text)
print(entities)
```

Output:
```python
[('Chưa', 'R', 'O', 'O'),
 ('tiết lộ', 'V', 'B-VP', 'O'),
 ('lịch trình', 'V', 'B-VP', 'O'),
 ('tới', 'E', 'B-PP', 'O'),
 ('Việt Nam', 'Np', 'B-NP', 'B-LOC'),
 ('của', 'E', 'B-PP', 'O'),
 ('Tổng thống', 'N', 'B-NP', 'O'),
 ('Mỹ', 'Np', 'B-NP', 'B-LOC'),
 ('Donald', 'Np', 'B-NP', 'B-PER'),
 ('Trump', 'Np', 'B-NP', 'I-PER')]
```

## Understanding the Output

Each tuple contains:
1. **Word**: The token
2. **POS**: Part-of-speech tag
3. **Chunk**: Phrase chunk tag
4. **Entity**: Named entity tag

### Entity Tags (BIO Format)

| Tag | Meaning |
|-----|---------|
| `B-PER` | Beginning of Person |
| `I-PER` | Inside Person |
| `B-LOC` | Beginning of Location |
| `I-LOC` | Inside Location |
| `B-ORG` | Beginning of Organization |
| `I-ORG` | Inside Organization |
| `O` | Outside (not an entity) |

## Extracting Entities

### Simple Extraction

```python
from underthesea import ner

text = "Tổng thống Mỹ Donald Trump thăm Việt Nam"
entities = ner(text)

# Extract only named entities
for word, pos, chunk, entity in entities:
    if entity != 'O':
        print(f"{word}: {entity}")
# Mỹ: B-LOC
# Donald: B-PER
# Trump: I-PER
# Việt Nam: B-LOC
```

### Grouping Multi-word Entities

```python
def extract_entities(text):
    """Extract and group multi-word entities."""
    results = ner(text)

    entities = []
    current_entity = []
    current_type = None

    for word, pos, chunk, entity in results:
        if entity.startswith('B-'):
            # Save previous entity if exists
            if current_entity:
                entities.append((' '.join(current_entity), current_type))
            # Start new entity
            current_entity = [word]
            current_type = entity[2:]  # Remove 'B-' prefix
        elif entity.startswith('I-'):
            # Continue current entity
            current_entity.append(word)
        else:
            # End current entity
            if current_entity:
                entities.append((' '.join(current_entity), current_type))
                current_entity = []
                current_type = None

    # Don't forget the last entity
    if current_entity:
        entities.append((' '.join(current_entity), current_type))

    return entities

text = "Tổng thống Mỹ Donald Trump thăm Việt Nam vào tháng 11"
print(extract_entities(text))
# [('Mỹ', 'LOC'), ('Donald Trump', 'PER'), ('Việt Nam', 'LOC')]
```

### Filtering by Entity Type

```python
from underthesea import ner

text = "Ông Nguyễn Văn A từ Công ty ABC ở Hà Nội đến thăm Đà Nẵng"
entities = ner(text)

# Extract by type
persons = [w for w, p, c, e in entities if e.endswith('PER')]
locations = [w for w, p, c, e in entities if e.endswith('LOC')]
organizations = [w for w, p, c, e in entities if e.endswith('ORG')]

print(f"Persons: {persons}")
print(f"Locations: {locations}")
print(f"Organizations: {organizations}")
```

## Deep Learning Model

For better accuracy, use the deep learning model:

```bash
pip install "underthesea[deep]"
```

```python
from underthesea import ner

text = "Bộ Công Thương xóa một tổng cục, giảm nhiều đầu mối"
entities = ner(text, deep=True)
print(entities)
# [
#   {'entity': 'B-ORG', 'word': 'Bộ'},
#   {'entity': 'I-ORG', 'word': 'Công'},
#   {'entity': 'I-ORG', 'word': 'Thương'}
# ]
```

## Practical Applications

### News Article Analysis

```python
from underthesea import ner
from collections import defaultdict

article = """
Thủ tướng Phạm Minh Chính đã gặp Tổng thống Mỹ Joe Biden tại Washington.
Cuộc gặp diễn ra tại Nhà Trắng vào ngày 25 tháng 9.
Đại sứ quán Việt Nam tại Mỹ đã tổ chức buổi tiếp đón.
"""

# Analyze entities
entities_by_type = defaultdict(set)

for sentence in article.strip().split('\n'):
    if sentence:
        for word, pos, chunk, entity in ner(sentence):
            if entity != 'O':
                entity_type = entity.split('-')[1]
                entities_by_type[entity_type].add(word)

print("Entities found:")
for entity_type, words in entities_by_type.items():
    print(f"  {entity_type}: {', '.join(words)}")
```

### Building an Entity Database

```python
from underthesea import ner

def build_entity_database(documents):
    """Build a database of entities from documents."""
    database = {
        'PER': set(),
        'LOC': set(),
        'ORG': set()
    }

    for doc in documents:
        entities = ner(doc)
        for word, pos, chunk, entity in entities:
            if entity.endswith('PER'):
                database['PER'].add(word)
            elif entity.endswith('LOC'):
                database['LOC'].add(word)
            elif entity.endswith('ORG'):
                database['ORG'].add(word)

    return database

docs = [
    "Việt Nam và Mỹ tăng cường hợp tác",
    "Công ty Samsung đầu tư tại Bắc Ninh",
    "Ông Nguyễn Văn A được bổ nhiệm làm giám đốc"
]

db = build_entity_database(docs)
print(db)
```

## Next Steps

- Learn about [Text Classification](text_classification.md)
- Explore [Sentiment Analysis](sentiment_analysis.md)
- See the [API Reference](../api/ner.md)
