# translate

Translate text between Vietnamese and English.

!!! note "Requires Deep Learning"
    This function requires the deep learning dependencies:
    ```bash
    pip install underthesea[deep]
    ```

## Usage

```python
from underthesea import translate

# Vietnamese to English (default)
text = "Hà Nội là thủ đô của Việt Nam"
english = translate(text)
print(english)
# 'Hanoi is the capital of Vietnam'
```

## Function Signature

```python
def translate(
    text: str,
    source_lang: str = 'vi',
    target_lang: str = 'en'
) -> str
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | | The input text to translate |
| `source_lang` | `str` | `'vi'` | Source language code |
| `target_lang` | `str` | `'en'` | Target language code |

## Returns

| Type | Description |
|------|-------------|
| `str` | The translated text |

## Supported Languages

| Code | Language |
|------|----------|
| `vi` | Vietnamese |
| `en` | English |

## Examples

### Vietnamese to English

```python
from underthesea import translate

# Basic translation
translate("Hà Nội là thủ đô của Việt Nam")
# 'Hanoi is the capital of Vietnam'

translate("Ẩm thực Việt Nam nổi tiếng trên thế giới")
# 'Vietnamese cuisine is famous around the world'

translate("Xin chào, tôi là sinh viên")
# 'Hello, I am a student'
```

### English to Vietnamese

```python
from underthesea import translate

translate("I love Vietnamese food", source_lang='en', target_lang='vi')
# 'Tôi yêu ẩm thực Việt Nam'

translate("Vietnam is a beautiful country", source_lang='en', target_lang='vi')
# 'Việt Nam là một đất nước xinh đẹp'
```

### Translating Multiple Sentences

```python
from underthesea import translate

sentences = [
    "Hà Nội là thủ đô của Việt Nam",
    "Việt Nam có nhiều địa điểm du lịch đẹp",
    "Ẩm thực Việt Nam rất phong phú"
]

for sentence in sentences:
    english = translate(sentence)
    print(f"{sentence}")
    print(f"-> {english}\n")
```

### Handling Long Text

For long documents, consider splitting into sentences first:

```python
from underthesea import sent_tokenize, translate

text = """Việt Nam là một quốc gia. Thủ đô là Hà Nội.
Thành phố lớn nhất là TP. Hồ Chí Minh."""

sentences = sent_tokenize(text)
translations = [translate(s) for s in sentences]

for vi, en in zip(sentences, translations):
    print(f"VI: {vi}")
    print(f"EN: {en}\n")
```

## Notes

- Uses a transformer-based neural machine translation model
- First call may take longer due to model loading
- Works best with well-formed sentences
- Long texts should be split into sentences for better results
