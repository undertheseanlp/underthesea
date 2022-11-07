<p align="center">
  <br>
  <img src="https://raw.githubusercontent.com/undertheseanlp/underthesea/master/logo.png"/>
  <br/>
</p>

<p align="center">
  <a href="https://pypi.python.org/pypi/underthesea">
    <img src="https://img.shields.io/pypi/v/underthesea.svg">
  </a>
  <a href="https://pypi.python.org/pypi/underthesea">
    <img src="https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8%20%7C%203.9%20%7C%203.10-blue">
  </a>
  <a href="http://undertheseanlp.com/">
    <img src="https://img.shields.io/badge/demo-live-brightgreen">
  </a>
  <a href="https://underthesea.readthedocs.io/en/latest/">
    <img src="https://img.shields.io/badge/docs-live-brightgreen">
  </a>
  <a href="https://colab.research.google.com/drive/1gD8dSMSE_uNacW4qJ-NSnvRT85xo9ZY2">
    <img src="https://img.shields.io/badge/latest-ff9f01?logo=google-colab&logoColor=white">
  </a>
  <a href="https://colab.research.google.com/drive/1U6EWY7ewNUtCXGsa5uZtDEz4I5exO_fo">
    <img src="https://img.shields.io/badge/stable-ff9f01?logo=google-colab&logoColor=white">
  </a>
  <a href="https://www.facebook.com/undertheseanlp/">
    <img src="https://img.shields.io/badge/Facebook-1877F2?logo=facebook&logoColor=white">
  </a>
  <a href="https://www.youtube.com/channel/UC9Jv1Qg49uprg6SjkyAqs9A">
    <img src="https://img.shields.io/badge/YouTube-FF0000?logo=youtube&logoColor=white">
  </a>
</p>

<h3 align="center">
Open-source Vietnamese Natural Language Process Toolkit
</h3>

`Underthesea` is:

🌊 **A Vietnamese NLP toolkit.** Underthesea is a suite of open source Python modules data sets and tutorials supporting research and development in [Vietnamese Natural Language Processing](https://github.com/undertheseanlp/underthesea). We provides extremely easy API to quickly apply pretrained NLP models to your Vietnamese text, such as word segmentation, part-of-speech tagging (PoS), named entity recognition (NER), text classification and dependency parsing.

🌊 **A Pytorch library.** Underthesea is backed by one of most popular deep learning libraries, [Pytorch](https://pytorch.org/), make it easy to train your deep learning models and experiment with new approaches using Underthesea modules and classes.

🌊 **An open-source software.** Underthesea is published under the [GNU General Public License v3.0](https://github.com/undertheseanlp/underthesea/blob/master/LICENSE) license. Permissions of this strong copyleft license are conditioned on making available complete source code of licensed works and modifications, which include larger works using a licensed work, under the same license.

💝 [**Support Us!**](#-support-us)

## Installation


To install underthesea, simply:

```bash
$ pip install underthesea
✨🍰✨
```

Satisfaction, guaranteed.

## Tutorials

* [1. Sentence Segmentation](#1-sentence-segmentation)
* [2. Text Normalization](#2-text-normalization)
* [3. Word Segmentation](#3-word-segmentation)
* [4. POS Tagging](#4-pos-tagging)
* [5. Chunking](#5-chunking)
* [6. Dependency Parsing](#6-dependency-parsing)
* [7. Named Entity Recognition](#7-named-entity-recognition)
* [8. Text Classification](#8-text-classification)
* [9. Sentiment Analysis](#9-sentiment-analysis)
* [10. Vietnamese NLP Resources](#10-vietnamese-nlp-resources)

### 1. Sentence Segmentation

Usage

```python
>>> from underthesea import sent_tokenize
>>> text = 'Taylor cho biết lúc đầu cô cảm thấy ngại với cô bạn thân Amanda nhưng rồi mọi thứ trôi qua nhanh chóng. Amanda cũng thoải mái với mối quan hệ này.'

>>> sent_tokenize(text)
[
  "Taylor cho biết lúc đầu cô cảm thấy ngại với cô bạn thân Amanda nhưng rồi mọi thứ trôi qua nhanh chóng.",
  "Amanda cũng thoải mái với mối quan hệ này."
]
```

### 2. Text Normalization

Usage

```python
>>> from underthesea import text_normalize
>>> text_normalize('Ðảm baỏ chất lựơng phòng thí nghịêm hoá học')
'Đảm bảo chất lượng phòng thí nghiệm hóa học'
```

### 3. Word Segmentation

Usage

```python
>>> from underthesea import word_tokenize
>>> sentence = 'Chàng trai 9X Quảng Trị khởi nghiệp từ nấm sò'

>>> word_tokenize(sentence)
['Chàng trai', '9X', 'Quảng Trị', 'khởi nghiệp', 'từ', 'nấm', 'sò']

>>> word_tokenize(sentence, format="text")
'Chàng_trai 9X Quảng_Trị khởi_nghiệp từ nấm sò'
```

### 4. POS Tagging

Usage

```python
>>> from underthesea import pos_tag
>>> pos_tag('Chợ thịt chó nổi tiếng ở Sài Gòn bị truy quét')
[('Chợ', 'N'),
 ('thịt', 'N'),
 ('chó', 'N'),
 ('nổi tiếng', 'A'),
 ('ở', 'E'),
 ('Sài Gòn', 'Np'),
 ('bị', 'V'),
 ('truy quét', 'V')]
```


### 5. Chunking

Usage

```python
>>> from underthesea import chunk
>>> text = 'Bác sĩ bây giờ có thể thản nhiên báo tin bệnh nhân bị ung thư?'
>>> chunk(text)
[('Bác sĩ', 'N', 'B-NP'),
 ('bây giờ', 'P', 'I-NP'),
 ('có thể', 'R', 'B-VP'),
 ('thản nhiên', 'V', 'I-VP'),
 ('báo tin', 'N', 'B-NP'),
 ('bệnh nhân', 'N', 'I-NP'),
 ('bị', 'V', 'B-VP'),
 ('ung thư', 'N', 'I-VP'),
 ('?', 'CH', 'O')]
```


### 6. Dependency Parsing

Install dependencies for deep learning

```bash
$ pip install underthesea[deep]
```

Usage

```python
>>> from underthesea import dependency_parse
>>> text = 'Tối 29/11, Việt Nam thêm 2 ca mắc Covid-19'
>>> dependency_parse(text)
[('Tối', 5, 'obl:tmod'),
 ('29/11', 1, 'flat:date'),
 (',', 1, 'punct'),
 ('Việt Nam', 5, 'nsubj'),
 ('thêm', 0, 'root'),
 ('2', 7, 'nummod'),
 ('ca', 5, 'obj'),
 ('mắc', 7, 'nmod'),
 ('Covid-19', 8, 'nummod')]
```

### 7. Named Entity Recognition

Usage

```python
>>> from underthesea import ner
>>> text = 'Chưa tiết lộ lịch trình tới Việt Nam của Tổng thống Mỹ Donald Trump'
>>> ner(text)
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

With Deep Learning

```bash
$ pip install underthesea[deep]
```

```python
>>> from underthesea import ner
>>> text = "Bộ Công Thương xóa một tổng cục, giảm nhiều đầu mối"
>>> ner(text, deep=True)
[
  {'entity': 'B-ORG', 'word': 'Bộ'},
  {'entity': 'I-ORG', 'word': 'Công'},
  {'entity': 'I-ORG', 'word': 'Thương'}
]
```

### 8. Text Classification

Usage

```python
>>> from underthesea import classify

>>> classify('HLV đầu tiên ở Premier League bị sa thải sau 4 vòng đấu')
['The thao']

>>> classify('Hội đồng tư vấn kinh doanh Asean vinh danh giải thưởng quốc tế')
['Kinh doanh']

>> classify('Lãi suất từ BIDV rất ưu đãi', domain='bank')
['INTEREST_RATE']
```

### 9. Sentiment Analysis

Usage

```python
>>> from underthesea import sentiment

>>> sentiment('hàng kém chất lg,chăn đắp lên dính lông lá khắp người. thất vọng')
'negative'
>>> sentiment('Sản phẩm hơi nhỏ so với tưởng tượng nhưng chất lượng tốt, đóng gói cẩn thận.')
'positive'

>>> sentiment('Đky qua đường link ở bài viết này từ thứ 6 mà giờ chưa thấy ai lhe hết', domain='bank')
['CUSTOMER_SUPPORT#negative']
>>> sentiment('Xem lại vẫn thấy xúc động và tự hào về BIDV của mình', domain='bank')
['TRADEMARK#positive']
```

### 10. Vietnamese NLP Resources

List resources

```bash
$ underthesea list-data
| Name                | Type        | License | Year | Directory                    |
|---------------------+-------------+---------+------+------------------------------|
| UIT_ABSA_RESTAURANT | Sentiment   | Open    | 2021 | datasets/UIT_ABSA_RESTAURANT |
| UIT_ABSA_HOTEL      | Sentiment   | Open    | 2021 | datasets/UIT_ABSA_HOTEL      |
| SE_Vietnamese-UBS   | Sentiment   | Open    | 2020 | datasets/SE_Vietnamese-UBS   |
| CP_Vietnamese-UNC   | Plaintext   | Open    | 2020 | datasets/CP_Vietnamese-UNC   |
| DI_Vietnamese-UVD   | Dictionary  | Open    | 2020 | datasets/DI_Vietnamese-UVD   |
| UTS2017-BANK        | Categorized | Open    | 2017 | datasets/UTS2017-BANK        |
| VNTQ_SMALL          | Plaintext   | Open    | 2012 | datasets/LTA                 |
| VNTQ_BIG            | Plaintext   | Open    | 2012 | datasets/LTA                 |
| VNESES              | Plaintext   | Open    | 2012 | datasets/LTA                 |
| VNTC                | Categorized | Open    | 2007 | datasets/VNTC                |

$ underthesea list-data --all
```

Download resources

```bash
$ underthesea download-data VNTC
100%|██████████| 74846806/74846806 [00:09<00:00, 8243779.16B/s]
Resource VNTC is downloaded in ~/.underthesea/datasets/VNTC folder
```

### Up Coming Features

* Machine Translation
* Text to Speech
* Automatic Speech Recognition

## Contributing

Do you want to contribute with underthesea development? Great! Please read more details at [CONTRIBUTING.rst](https://github.com/undertheseanlp/underthesea/blob/master/CONTRIBUTING.rst)

## 💝 Support Us

If you found this project helpful and would like to support our work, you can just buy us a coffee ☕.

Your support is our biggest encouragement 🎁!


<img src="https://raw.githubusercontent.com/undertheseanlp/underthesea/master/img/support.png"/>
