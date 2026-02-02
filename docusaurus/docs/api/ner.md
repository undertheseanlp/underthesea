# ner

Identify and classify named entities in text.

## Usage

```python
from underthesea import ner

text = "Chưa tiết lộ lịch trình tới Việt Nam của Tổng thống Mỹ Donald Trump"
entities = ner(text)
print(entities)
# [('Chưa', 'R', 'O', 'O'),
#  ('tiết lộ', 'V', 'B-VP', 'O'),
#  ('lịch trình', 'V', 'B-VP', 'O'),
#  ('tới', 'E', 'B-PP', 'O'),
#  ('Việt Nam', 'Np', 'B-NP', 'B-LOC'),
#  ('của', 'E', 'B-PP', 'O'),
#  ('Tổng thống', 'N', 'B-NP', 'O'),
#  ('Mỹ', 'Np', 'B-NP', 'B-LOC'),
#  ('Donald', 'Np', 'B-NP', 'B-PER'),
#  ('Trump', 'Np', 'B-NP', 'I-PER')]
```

## Function Signature

```python
def ner(
    sentence: str,
    format: str = None,
    deep: bool = False
) -> list[tuple] | list[dict]
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sentence` | `str` | | The input text |
| `format` | `str` | `None` | Output format |
| `deep` | `bool` | `False` | Use deep learning model (requires `[deep]` install) |

## Returns

### CRF Model (default)

| Type | Description |
|------|-------------|
| `list[tuple[str, str, str, str]]` | List of (word, POS, chunk, entity) tuples |

### Deep Learning Model (`deep=True`)

| Type | Description |
|------|-------------|
| `list[dict]` | List of dictionaries with `entity` and `word` keys |

## Entity Types

| Tag | Description |
|-----|-------------|
| `PER` | Person |
| `LOC` | Location |
| `ORG` | Organization |
| `O` | Not an entity |

Tags use BIO format:

- `B-XXX`: Beginning of entity
- `I-XXX`: Inside entity
- `O`: Outside (not an entity)

## Examples

### Basic Usage (CRF Model)

```python
from underthesea import ner

text = "Chưa tiết lộ lịch trình tới Việt Nam của Tổng thống Mỹ Donald Trump"
entities = ner(text)

# Extract only named entities
for word, pos, chunk, entity in entities:
    if entity != 'O':
        print(f"{word}: {entity}")
# Việt Nam: B-LOC
# Mỹ: B-LOC
# Donald: B-PER
# Trump: I-PER
```

### Deep Learning Model

!!! note "Requires Installation"
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

### Extracting Entities by Type

```python
text = "Ông Nguyễn Văn A từ Hà Nội đến công ty ABC"
entities = ner(text)

persons = []
locations = []
orgs = []

for word, pos, chunk, entity in entities:
    if entity.endswith('PER'):
        persons.append(word)
    elif entity.endswith('LOC'):
        locations.append(word)
    elif entity.endswith('ORG'):
        orgs.append(word)

print(f"Persons: {persons}")
print(f"Locations: {locations}")
print(f"Organizations: {orgs}")
```

### Combining Multi-word Entities

```python
text = "Tổng thống Mỹ Donald Trump thăm Việt Nam"
entities = ner(text)

# Combine B-/I- tags into full entities
current_entity = []
current_type = None
full_entities = []

for word, pos, chunk, entity in entities:
    if entity.startswith('B-'):
        if current_entity:
            full_entities.append((' '.join(current_entity), current_type))
        current_entity = [word]
        current_type = entity[2:]
    elif entity.startswith('I-'):
        current_entity.append(word)
    else:
        if current_entity:
            full_entities.append((' '.join(current_entity), current_type))
            current_entity = []
            current_type = None

if current_entity:
    full_entities.append((' '.join(current_entity), current_type))

print(full_entities)
# [('Donald Trump', 'PER'), ('Việt Nam', 'LOC')]
```

## Notes

- The CRF model is fast and lightweight
- The deep learning model provides better accuracy but requires more resources
- First call with `deep=True` may take longer due to model loading
