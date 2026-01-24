# Underthesea

<p align="center">
  <img src="img/logo.png" alt="Underthesea Logo" width="300"/>
</p>

<p align="center">
  <a href="https://pypi.python.org/pypi/underthesea">
    <img src="https://img.shields.io/pypi/v/underthesea.svg" alt="PyPI version">
  </a>
  <a href="https://pypi.python.org/pypi/underthesea">
    <img src="https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue" alt="Python versions">
  </a>
  <a href="http://undertheseanlp.com/">
    <img src="https://img.shields.io/badge/demo-live-brightgreen" alt="Live demo">
  </a>
</p>

## Open-source Vietnamese Natural Language Processing Toolkit

**Underthesea** is a suite of open source Python modules, datasets, and tutorials supporting research and development in Vietnamese Natural Language Processing.

We provide an extremely easy API to quickly apply pretrained NLP models to your Vietnamese text.

## Features

| Feature | Description | Model Type |
|---------|-------------|------------|
| [Sentence Segmentation](api/sent_tokenize.md) | Breaking text into individual sentences | CRF |
| [Text Normalization](api/text_normalize.md) | Standardizing textual data representation | Rule-based |
| [Word Segmentation](api/word_tokenize.md) | Dividing text into individual words | CRF |
| [POS Tagging](api/pos_tag.md) | Labeling words with their part-of-speech | CRF |
| [Chunking](api/chunk.md) | Grouping words into meaningful phrases | CRF |
| [Dependency Parsing](api/dependency_parse.md) | Analyzing grammatical structure | Deep Learning |
| [Named Entity Recognition](api/ner.md) | Identifying named entities | CRF / Deep Learning |
| [Text Classification](api/classify.md) | Categorizing text into predefined groups | CRF / Prompt |
| [Sentiment Analysis](api/sentiment.md) | Determining text's emotional tone | CRF |
| [Translation](api/translate.md) | Translating Vietnamese to English | Deep Learning |
| [Language Detection](api/lang_detect.md) | Identifying the language of text | FastText |
| [Text-to-Speech](api/say.md) | Converting text into spoken audio | Deep Learning |

## Quick Example

```python
from underthesea import word_tokenize, pos_tag, ner

# Word Segmentation
text = "Chàng trai 9X Quảng Trị khởi nghiệp từ nấm sò"
print(word_tokenize(text))
# ['Chàng trai', '9X', 'Quảng Trị', 'khởi nghiệp', 'từ', 'nấm', 'sò']

# POS Tagging
text = "Chợ thịt chó nổi tiếng ở Sài Gòn bị truy quét"
print(pos_tag(text))
# [('Chợ', 'N'), ('thịt', 'N'), ('chó', 'N'), ('nổi tiếng', 'A'), ...]

# Named Entity Recognition
text = "Chưa tiết lộ lịch trình tới Việt Nam của Tổng thống Mỹ Donald Trump"
print(ner(text))
# [('Việt Nam', 'Np', 'B-NP', 'B-LOC'), ('Mỹ', 'Np', 'B-NP', 'B-LOC'), ...]
```

## What's New

!!! success "New in v9.0.0"
    Vietnamese-English translation is here! Use `translate("Xin chào Việt Nam")` to translate Vietnamese text to English.

## Getting Started

<div class="grid cards" markdown>

-   :material-download:{ .lg .middle } **Installation**

    ---

    Install Underthesea with pip in seconds

    [:octicons-arrow-right-24: Installation Guide](installation.md)

-   :material-rocket-launch:{ .lg .middle } **Quick Start**

    ---

    Get started with your first NLP task

    [:octicons-arrow-right-24: Quick Start](quickstart.md)

-   :material-book-open-variant:{ .lg .middle } **Tutorials**

    ---

    Learn with step-by-step tutorials

    [:octicons-arrow-right-24: Tutorials](tutorials/word_segmentation.md)

-   :material-api:{ .lg .middle } **API Reference**

    ---

    Complete API documentation

    [:octicons-arrow-right-24: API Reference](api/index.md)

</div>

## Community

- [GitHub Repository](https://github.com/undertheseanlp/underthesea)
- [Facebook Page](https://www.facebook.com/undertheseanlp/)
- [YouTube Channel](https://www.youtube.com/channel/UC9Jv1Qg49uprg6SjkyAqs9A)
- [Google Colab Notebook](https://colab.research.google.com/drive/1gD8dSMSE_uNacW4qJ-NSnvRT85xo9ZY2)

## Support

If you found this project helpful, please consider [supporting us](https://github.com/undertheseanlp/underthesea/blob/main/docs/contribute/SUPPORT_US.md).
