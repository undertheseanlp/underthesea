# dependency_parse

Analyze the grammatical structure and dependencies between words.

!!! note "Requires Deep Learning"
    This function requires the deep learning dependencies:
    ```bash
    pip install underthesea[deep]
    ```

## Usage

```python
from underthesea import dependency_parse

text = "Tối 29/11, Việt Nam thêm 2 ca mắc Covid-19"
result = dependency_parse(text)
print(result)
# [('Tối', 5, 'obl:tmod'),
#  ('29/11', 1, 'flat:date'),
#  (',', 1, 'punct'),
#  ('Việt Nam', 5, 'nsubj'),
#  ('thêm', 0, 'root'),
#  ('2', 7, 'nummod'),
#  ('ca', 5, 'obj'),
#  ('mắc', 7, 'nmod'),
#  ('Covid-19', 8, 'nummod')]
```

## Function Signature

```python
def dependency_parse(sentence: str) -> list[tuple[str, int, str]]
```

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `sentence` | `str` | The input text to parse |

## Returns

| Type | Description |
|------|-------------|
| `list[tuple[str, int, str]]` | List of (word, head_index, relation) tuples |

Each tuple contains:

- `word`: The word token
- `head_index`: Index of the head word (0 = root)
- `relation`: The dependency relation type

## Dependency Relations

| Relation | Description |
|----------|-------------|
| `root` | Root of the sentence |
| `nsubj` | Nominal subject |
| `obj` | Object |
| `obl` | Oblique nominal |
| `obl:tmod` | Temporal modifier |
| `amod` | Adjectival modifier |
| `nmod` | Nominal modifier |
| `nummod` | Numeric modifier |
| `punct` | Punctuation |
| `flat:date` | Flat date expression |
| `compound` | Compound word |

## Examples

### Basic Usage

```python
from underthesea import dependency_parse

text = "Tối 29/11, Việt Nam thêm 2 ca mắc Covid-19"
result = dependency_parse(text)

for i, (word, head, rel) in enumerate(result, 1):
    print(f"{i}\t{word}\t{head}\t{rel}")
# 1    Tối         5    obl:tmod
# 2    29/11       1    flat:date
# 3    ,           1    punct
# 4    Việt Nam    5    nsubj
# 5    thêm        0    root
# 6    2           7    nummod
# 7    ca          5    obj
# 8    mắc         7    nmod
# 9    Covid-19    8    nummod
```

### Finding the Root

```python
text = "Tôi yêu Việt Nam"
result = dependency_parse(text)

root = [(i, word) for i, (word, head, rel) in enumerate(result, 1) if rel == 'root']
print(f"Root: {root}")
# Root: [(2, 'yêu')]
```

### Finding Subjects and Objects

```python
text = "Sinh viên đọc sách ở thư viện"
result = dependency_parse(text)

subjects = [word for word, head, rel in result if rel == 'nsubj']
objects = [word for word, head, rel in result if rel == 'obj']

print(f"Subjects: {subjects}")
print(f"Objects: {objects}")
```

## Notes

- This function uses a transformer-based model
- First call may take longer due to model loading
- Requires significant memory for the deep learning model
