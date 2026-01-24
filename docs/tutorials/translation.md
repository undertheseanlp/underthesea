# Translation Tutorial

Learn how to translate between Vietnamese and English.

## Installation

Translation requires the deep learning dependencies:

```bash
pip install underthesea[deep]
```

## Basic Usage

```python
from underthesea import translate

# Vietnamese to English (default)
text = "Hà Nội là thủ đô của Việt Nam"
english = translate(text)
print(english)
# 'Hanoi is the capital of Vietnam'
```

## Vietnamese to English

```python
from underthesea import translate

examples = [
    "Xin chào, tôi là sinh viên",
    "Việt Nam có nhiều địa điểm du lịch đẹp",
    "Ẩm thực Việt Nam nổi tiếng trên thế giới"
]

for text in examples:
    english = translate(text)
    print(f"VI: {text}")
    print(f"EN: {english}\n")
```

## English to Vietnamese

```python
from underthesea import translate

examples = [
    "I love Vietnamese food",
    "Vietnam is a beautiful country",
    "Hello, how are you today?"
]

for text in examples:
    vietnamese = translate(text, source_lang='en', target_lang='vi')
    print(f"EN: {text}")
    print(f"VI: {vietnamese}\n")
```

## Practical Applications

### Document Translation

```python
from underthesea import sent_tokenize, translate

def translate_document(document, source_lang='vi', target_lang='en'):
    """Translate a document sentence by sentence."""
    sentences = sent_tokenize(document)
    translations = []

    for sentence in sentences:
        if sentence.strip():
            translated = translate(sentence, source_lang, target_lang)
            translations.append(translated)

    return ' '.join(translations)

document = """
Việt Nam là một quốc gia nằm ở Đông Nam Á.
Thủ đô của Việt Nam là Hà Nội.
Thành phố lớn nhất là Thành phố Hồ Chí Minh.
"""

english = translate_document(document)
print(english)
```

### Bilingual Content Generator

```python
from underthesea import translate

def create_bilingual_content(texts, source_lang='vi'):
    """Create bilingual content from source texts."""
    target_lang = 'en' if source_lang == 'vi' else 'vi'

    bilingual = []
    for text in texts:
        translated = translate(text, source_lang, target_lang)
        bilingual.append({
            'original': text,
            'translated': translated,
            'source_lang': source_lang,
            'target_lang': target_lang
        })

    return bilingual

texts = [
    "Xin chào",
    "Cảm ơn bạn",
    "Tạm biệt"
]

result = create_bilingual_content(texts)
for item in result:
    print(f"{item['source_lang'].upper()}: {item['original']}")
    print(f"{item['target_lang'].upper()}: {item['translated']}\n")
```

### Translation Memory

```python
from underthesea import translate

class TranslationMemory:
    """Simple translation memory for caching translations."""

    def __init__(self):
        self.memory = {}

    def translate(self, text, source_lang='vi', target_lang='en'):
        key = (text, source_lang, target_lang)

        if key not in self.memory:
            # Translate and cache
            result = translate(text, source_lang, target_lang)
            self.memory[key] = result

        return self.memory[key]

    def get_stats(self):
        return {
            'cached_translations': len(self.memory)
        }

# Usage
tm = TranslationMemory()

# First call - translates and caches
result1 = tm.translate("Xin chào Việt Nam")

# Second call - returns from cache (faster)
result2 = tm.translate("Xin chào Việt Nam")

print(result1)  # Hello Vietnam
print(tm.get_stats())  # {'cached_translations': 1}
```

### Website Localization Helper

```python
from underthesea import translate

def localize_ui_strings(strings, target_lang='en'):
    """Translate UI strings for localization."""
    localized = {}

    for key, text in strings.items():
        localized[key] = translate(text, 'vi', target_lang)

    return localized

# Vietnamese UI strings
vi_strings = {
    'welcome': 'Chào mừng bạn đến với ứng dụng',
    'login': 'Đăng nhập',
    'logout': 'Đăng xuất',
    'settings': 'Cài đặt',
    'profile': 'Hồ sơ cá nhân'
}

# Generate English translations
en_strings = localize_ui_strings(vi_strings)

print("Localization Results:")
for key in vi_strings:
    print(f"  {key}:")
    print(f"    VI: {vi_strings[key]}")
    print(f"    EN: {en_strings[key]}")
```

### Parallel Text Corpus Builder

```python
from underthesea import translate

def build_parallel_corpus(source_texts, source_lang='vi', target_lang='en'):
    """Build a parallel text corpus for training/research."""
    corpus = []

    for text in source_texts:
        translated = translate(text, source_lang, target_lang)
        corpus.append({
            'source': text,
            'target': translated,
            'source_lang': source_lang,
            'target_lang': target_lang
        })

    return corpus

def save_corpus(corpus, filename):
    """Save parallel corpus to file."""
    with open(filename, 'w', encoding='utf-8') as f:
        for item in corpus:
            f.write(f"{item['source']}\t{item['target']}\n")

# Build corpus
texts = [
    "Việt Nam là một đất nước xinh đẹp",
    "Hà Nội là thủ đô của Việt Nam",
    "Ẩm thực Việt Nam rất phong phú"
]

corpus = build_parallel_corpus(texts)
for item in corpus:
    print(f"VI: {item['source']}")
    print(f"EN: {item['target']}\n")
```

## Tips for Better Translations

1. **Sentence-level**: Translate sentence by sentence for better quality
2. **Context**: Provide complete sentences, not fragments
3. **Preprocessing**: Clean and normalize text before translation
4. **Post-editing**: Review translations for domain-specific terms

```python
from underthesea import text_normalize, translate

def quality_translate(text, source_lang='vi', target_lang='en'):
    """Translate with preprocessing for better quality."""
    # Normalize text first
    if source_lang == 'vi':
        text = text_normalize(text)

    # Translate
    result = translate(text, source_lang, target_lang)

    return result
```

## Next Steps

- Learn about [Text-to-Speech](tts.md)
- Explore [Language Detection](../api/lang_detect.md)
- See the [API Reference](../api/translate.md)
