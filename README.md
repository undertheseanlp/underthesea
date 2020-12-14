<p align="center">
  <br>
  <img src="logo.png"/>
  <br/>
</p>

<p align="center">
  <a href="https://pypi.python.org/pypi/underthesea">
    <img src="https://img.shields.io/pypi/v/underthesea.svg">
  </a>
  <a href="https://pypi.python.org/pypi/underthesea">
    <img src="https://img.shields.io/pypi/pyversions/underthesea.svg">
  </a>
  <a href="https://pypi.python.org/pypi/underthesea">
    <img src="https://img.shields.io/badge/license-GNU%20General%20Public%20License%20v3-brightgreen.svg">
  </a>
  <a href="https://travis-ci.org/undertheseanlp/underthesea">
    <img src="https://img.shields.io/travis/undertheseanlp/underthesea.svg">
  </a>
  <a href="http://underthesea.readthedocs.io/en/latest/">
    <img src="https://readthedocs.org/projects/underthesea/badge/?version=latest">
  </a>
</p>

<p align="center">
  <a href="https://www.facebook.com/undertheseanlp/">
    <img src="https://img.shields.io/badge/Facebook-1877F2?style=for-the-badge&logo=facebook&logoColor=white" height="24px">
  </a>
  <a href="https://www.youtube.com/channel/UC9Jv1Qg49uprg6SjkyAqs9A">
    <img src="https://img.shields.io/badge/YouTube-FF0000?logo=youtube&logoColor=white">
  </a>
</p>

<h3 align="center">
Open Vietnamese Natural Language Process Toolkit
</h3>

Underthesea is: 

🌊 **A Vietnamese NLP library.** Underthesea is a suite of open source Python modules data sets and tutorials supporting research and development in Vietnamese Natural Language Processing.

🌊 Underthesea provides extremely easy API to quickly download and use those pretrained models 

🌊 Underthesa is backed by one of most popular deep learning libraries, [Pytorch](https://pytorch.org/), with a seamless integration between them, allowing you to train your models with one then load it for inference with the other.

💫 **Version 1.3.0 out now!** [Underthesea meet deep learning!](https://github.com/undertheseanlp/underthesea/issues/359)


<table>
<tbody>
  <tr>
    <td>Live demo</td>
    <td>
      <a href="http://undertheseanlp.com">undertheseanlp.com</a>
    </td>
  </tr>
  <tr>
    <td>Colab notebooks</td>
    <td>
        <a href="https://colab.research.google.com/drive/1gD8dSMSE_uNacW4qJ-NSnvRT85xo9ZY2">latest</a>
        / 
        <a href="https://colab.research.google.com/drive/1U6EWY7ewNUtCXGsa5uZtDEz4I5exO_fo">stable</a>
    </td>
  </tr>
</tbody>
</table>

## Installation


To install underthesea, simply:

```bash
$ pip install underthesea
✨🍰✨
```

Satisfaction, guaranteed.

## Tutorials

* [1. Sentence Segmentation](#1-sentence-segmentation)
* [2. Word Segmentation](#2-word-segmentation)
* [3. POS Tagging](#3-pos-tagging)
* [4. Chunking](#4-chunking)
* [5. Dependency Parsing](#5-dependency-parsing)
* [6. Named Entity Recognition](#6-named-entity-recognition)
* [7. Text Classification](#7-text-classification)
* [8. Sentiment Analysis](#8-sentiment-analysis)
* [9. Vietnamese NLP Resources](#9-vietnamese-nlp-resources)

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
   
### 2. Word Segmentation

Usage

```python
>>> from underthesea import word_tokenize
>>> sentence = 'Chàng trai 9X Quảng Trị khởi nghiệp từ nấm sò'

>>> word_tokenize(sentence)
['Chàng trai', '9X', 'Quảng Trị', 'khởi nghiệp', 'từ', 'nấm', 'sò']

>>> word_tokenize(sentence, format="text")
'Chàng_trai 9X Quảng_Trị khởi_nghiệp từ nấm sò'
``` 

### 3. POS Tagging


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


### 4. Chunking


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


### 5. Dependency Parsing


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


### 6. Named Entity Recognition


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

### 7. Text Classification


Download models

```bash
$ underthesea download-model TC_GENERAL
$ underthesea download-model TC_BANK
```

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


### 8. Sentiment Analysis

Download models

```bash
$ underthesea download-model SA_GENERAL
$ underthesea download-model SA_BANK
```

Usage


```python
>>> from underthesea import sentiment

>>> sentiment('hàng kém chất lg,chăn đắp lên dính lông lá khắp người. thất vọng')
negative
>>> sentiment('Sản phẩm hơi nhỏ so với tưởng tượng nhưng chất lượng tốt, đóng gói cẩn thận.')
positive

>>> sentiment('Đky qua đường link ở bài viết này từ thứ 6 mà giờ chưa thấy ai lhe hết', domain='bank')
['CUSTOMER_SUPPORT#negative']
>>> sentiment('Xem lại vẫn thấy xúc động và tự hào về BIDV của mình', domain='bank')
['TRADEMARK#positive']
```

### 9. Vietnamese NLP Resources

List resources

```bash
$ underthesea list-data
| Name         | Type        | License   |   Year | Directory             |
|--------------+-------------+-----------+--------+-----------------------|
| UTS2017-BANK | Categorized | Open      |   2017 | datasets/UTS2017-BANK |
| VNESES       | Plaintext   | Open      |   2012 | datasets/LTA          |
| VNTQ_BIG     | Plaintext   | Open      |   2012 | datasets/LTA          |
| VNTQ_SMALL   | Plaintext   | Open      |   2012 | datasets/LTA          |
| VNTC         | Categorized | Open      |   2007 | datasets/VNTC         |

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

### Contributing

Do you want to contribute with underthesea development? Great! Please read more details at [CONTRIBUTING.rst](https://github.com/undertheseanlp/underthesea/blob/master/CONTRIBUTING.rst)
