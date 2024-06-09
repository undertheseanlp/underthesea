<p align="center">
  <br>
  <img src="https://raw.githubusercontent.com/undertheseanlp/underthesea/main/img/logo.png"/>
  <br/>
</p>

<p align="center">
  <a href="https://pypi.python.org/pypi/underthesea">
    <img src="https://img.shields.io/pypi/v/underthesea.svg">
  </a>
  <a href="https://pypi.python.org/pypi/underthesea">
    <img src="https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue">
  </a>
  <a href="http://undertheseanlp.com/">
    <img src="https://img.shields.io/badge/demo-live-brightgreen">
  </a>
  <a href="https://underthesea.readthedocs.io/en/latest/">
    <img src="https://img.shields.io/badge/docs-live-brightgreen">
  </a>
  <a href="https://colab.research.google.com/drive/1gD8dSMSE_uNacW4qJ-NSnvRT85xo9ZY2">
    <img src="https://img.shields.io/badge/colab-ff9f01?logo=google-colab&logoColor=white">
  </a>
  <a href="https://www.facebook.com/undertheseanlp/">
    <img src="https://img.shields.io/badge/Facebook-1877F2?logo=facebook&logoColor=white">
  </a>
  <a href="https://www.youtube.com/channel/UC9Jv1Qg49uprg6SjkyAqs9A">
    <img src="https://img.shields.io/badge/YouTube-FF0000?logo=youtube&logoColor=white">
  </a>
</p>

<br/>

<p align="center">
  <a href="https://github.com/undertheseanlp/underthesea/blob/main/contribute/SPONSORS.md">
    <img src="https://img.shields.io/badge/sponsors-6-red?style=social&logo=GithubSponsors">
  </a>
</p>

<h3 align="center">
Open-source Vietnamese Natural Language Process Toolkit
</h3>

`Underthesea` is:

🌊 **A Vietnamese NLP toolkit.** Underthesea is a suite of open source Python modules data sets and tutorials supporting research and development in [Vietnamese Natural Language Processing](https://github.com/undertheseanlp/underthesea). We provides extremely easy API to quickly apply pretrained NLP models to your Vietnamese text, such as word segmentation, part-of-speech tagging (PoS), named entity recognition (NER), text classification and dependency parsing.

🌊 **An open-source software.** Underthesea is published under the [GNU General Public License v3.0](https://github.com/undertheseanlp/underthesea/blob/master/LICENSE) license. Permissions of this strong copyleft license are conditioned on making available complete source code of licensed works and modifications, which include larger works using a licensed work, under the same license.

🎁 [**Support Us!**](#-support-us) Every bit of support helps us achieve our goals. Thank you so much. 💝💝💝

🎉 **Hey there!** Have you heard about **LLMs**, the **prompt-based models**? Well, guess what? Starting from Underthesea version 6.7.0, you can now dive deep with this **super-cool feature** for [text classification](https://github.com/undertheseanlp/underthesea/issues/682)! Dive in and make a splash! 💦🚀

## Installation


To install underthesea, simply:

```bash
$ pip install underthesea
✨🍰✨
```

Satisfaction, guaranteed.

## Tutorials

<details>
<summary><b><a href="">Sentence Segmentation</a></b> - Breaking text into individual sentences
<code>📜</code>
</summary>

- 📜 Usage

    ```python
    >>> from underthesea import sent_tokenize
    >>> text = 'Taylor cho biết lúc đầu cô cảm thấy ngại với cô bạn thân Amanda nhưng rồi mọi thứ trôi qua nhanh chóng. Amanda cũng thoải mái với mối quan hệ này.'

    >>> sent_tokenize(text)
    [
      "Taylor cho biết lúc đầu cô cảm thấy ngại với cô bạn thân Amanda nhưng rồi mọi thứ trôi qua nhanh chóng.",
      "Amanda cũng thoải mái với mối quan hệ này."
    ]
    ```
</details>

<details>
<summary><b><a href="">Text Normalization</a></b> - Standardizing textual data representation
<code>📜</code>
</summary>

- 📜 Usage

    ```python
    >>> from underthesea import text_normalize
    >>> text_normalize("Ðảm baỏ chất lựơng phòng thí nghịêm hoá học")
    "Đảm bảo chất lượng phòng thí nghiệm hóa học"
    ```
</details>

<details>
<summary><b><a href="">Word Segmentation</a></b> - Dividing text into individual words
<code>📜</code>
</summary>

- 📜 Usage

    ```python
    >>> from underthesea import word_tokenize
    >>> text = "Chàng trai 9X Quảng Trị khởi nghiệp từ nấm sò"
    
    >>> word_tokenize(text)
    ["Chàng trai", "9X", "Quảng Trị", "khởi nghiệp", "từ", "nấm", "sò"]
    
    >>> word_tokenize(sentence, format="text")
    "Chàng_trai 9X Quảng_Trị khởi_nghiệp từ nấm sò"
    
    >>> text = "Viện Nghiên Cứu chiến lược quốc gia về học máy"
    >>> fixed_words = ["Viện Nghiên Cứu", "học máy"]
    >>> word_tokenize(text, fixed_words=fixed_words)
    "Viện_Nghiên_Cứu chiến_lược quốc_gia về học_máy"
    ```
</details>

<details>
<summary><b><a href="">POS Tagging</a></b> - Labeling words with their part-of-speech
<code>📜</code>
</summary>

- 📜 Usage

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
</details>

<details><summary><b><a href="">Chunking</a></b> - Grouping words into meaningful phrases or units
<code>📜</code>
</summary>

- 📜 Usage

    ```python
    >>> from underthesea import chunk
    >>> text = 'Bác sĩ bây giờ có thể thản nhiên báo tin bệnh nhân bị ung thư?'
    >>> chunk(text)
    [('Bác sĩ', 'N', 'B-NP'),
     ('bây giờ', 'P', 'B-NP'),
     ('có thể', 'R', 'O'),
     ('thản nhiên', 'A', 'B-AP'),
     ('báo', 'V', 'B-VP'),
     ('tin', 'N', 'B-NP'),
     ('bệnh nhân', 'N', 'B-NP'),
     ('bị', 'V', 'B-VP'),
     ('ung thư', 'N', 'B-NP'),
     ('?', 'CH', 'O')]
    ```
</details>

<details>
<summary><b><a href="">Dependency Parsing</a></b> - Analyzing grammatical structure between words
<code>⚛️</code>
</summary>
<br/>

- ⚛️ Deep Learning Model
    
    ```bash
    $ pip install underthesea[deep]
    ```
    
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
</details>

<details>
<summary><b><a href="">Named Entity Recognition</a></b> -  Identifying named entities (e.g., names, locations)
<code>📜</code> <code>⚛️</code>
</summary>
<br/>

- 📜 Usage

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
    
- ⚛️ Deep Learning Model

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
</details>

<details>
<summary><b><a href="">Text Classification</a></b> - Categorizing text into predefined groups
<code>📜</code> <code>⚡</code>
</summary>

- 📜 Usage

    ```python
    >>> from underthesea import classify
    
    >>> classify('HLV đầu tiên ở Premier League bị sa thải sau 4 vòng đấu')
    ['The thao']
    
    >>> classify('Hội đồng tư vấn kinh doanh Asean vinh danh giải thưởng quốc tế')
    ['Kinh doanh']
    
    >> classify('Lãi suất từ BIDV rất ưu đãi', domain='bank')
    ['INTEREST_RATE']
    ```

- ⚡ Prompt-based Model

    ```bash
    $ pip install underthesea[prompt]
    $ export OPENAI_API_KEY=YOUR_KEY
    ```
    
    ```python
    >>> from underthesea import classify
    >>> text = "HLV ngoại đòi gần tỷ mỗi tháng dẫn dắt tuyển Việt Nam"
    >>> classify(text, model='prompt')
    Thể thao
    ```
</details>

<details>
<summary><b><a href="">Sentiment Analysis</a></b> - Determining text's emotional tone or sentiment
<code>📜</code>
</summary>

- 📜 Usage

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
</details>

<details>
<summary><b><a href="">Lang Detect</a></b> - Identifying the Language of Text
<code>⚛️</code>
</summary>

<br/>

Lang Detect API. Thanks to awesome work from [FastText](https://fasttext.cc/docs/en/language-identification.html)

Install extend dependencies and models

    ```bash
    $ pip install underthesea[langdetect]
    ```

Usage examples in script

    ```python
    >>> from underthesea import lang_detect
    
    >>> lang_detect("Cựu binh Mỹ trả nhật ký nhẹ lòng khi thấy cuộc sống hòa bình tại Việt Nam")
    vi
    ```
</details>

<details>
<summary><b><a href="">Say 🗣️</a></b> - Converting written text into spoken audio
<code>⚛️</code>
</summary>

<br/>

Text to Speech API. Thanks to awesome work from [NTT123/vietTTS](https://github.com/ntt123/vietTTS)

Install extend dependencies and models

    ```bash
    $ pip install underthesea[wow]
    $ underthesea download-model VIET_TTS_V0_4_1
    ```

Usage examples in script

    ```python
    >>> from underthesea.pipeline.say import say
    
    >>> say("Cựu binh Mỹ trả nhật ký nhẹ lòng khi thấy cuộc sống hòa bình tại Việt Nam")
    A new audio file named `sound.wav` will be generated.
    ```

Usage examples in command line

    ```sh
    $ underthesea say "Cựu binh Mỹ trả nhật ký nhẹ lòng khi thấy cuộc sống hòa bình tại Việt Nam"
    ```
</details>

<details>
<summary><b><a href="">Vietnamese NLP Resources</a></b></summary>

<br/>

List resources

```bash
$ underthesea list-data
| Name                      | Type        | License | Year | Directory                          |
|---------------------------+-------------+---------+------+------------------------------------|
| CP_Vietnamese_VLC_v2_2022 | Plaintext   | Open    | 2023 | datasets/CP_Vietnamese_VLC_v2_2022 |
| UIT_ABSA_RESTAURANT       | Sentiment   | Open    | 2021 | datasets/UIT_ABSA_RESTAURANT       |
| UIT_ABSA_HOTEL            | Sentiment   | Open    | 2021 | datasets/UIT_ABSA_HOTEL            |
| SE_Vietnamese-UBS         | Sentiment   | Open    | 2020 | datasets/SE_Vietnamese-UBS         |
| CP_Vietnamese-UNC         | Plaintext   | Open    | 2020 | datasets/CP_Vietnamese-UNC         |
| DI_Vietnamese-UVD         | Dictionary  | Open    | 2020 | datasets/DI_Vietnamese-UVD         |
| UTS2017-BANK              | Categorized | Open    | 2017 | datasets/UTS2017-BANK              |
| VNTQ_SMALL                | Plaintext   | Open    | 2012 | datasets/LTA                       |
| VNTQ_BIG                  | Plaintext   | Open    | 2012 | datasets/LTA                       |
| VNESES                    | Plaintext   | Open    | 2012 | datasets/LTA                       |
| VNTC                      | Categorized | Open    | 2007 | datasets/VNTC                      |

$ underthesea list-data --all
```

Download resources

```bash
$ underthesea download-data CP_Vietnamese_VLC_v2_2022
Resource CP_Vietnamese_VLC_v2_2022 is downloaded in ~/.underthesea/datasets/CP_Vietnamese_VLC_v2_2022 folder
```

</details>

### Up Coming Features

* Automatic Speech Recognition
* Machine Translation
* Chatbot Agent

## Contributing

Do you want to contribute with underthesea development? Great! Please read more details at [CONTRIBUTING.rst](https://github.com/undertheseanlp/underthesea/blob/main/contribute/CONTRIBUTING.rst)

## 💝 Support Us

If you found this project helpful and would like to support our work, you can just buy us a coffee ☕.

Your support is our biggest encouragement 🎁!

<img src="https://raw.githubusercontent.com/undertheseanlp/underthesea/main/img/support.png"/>
