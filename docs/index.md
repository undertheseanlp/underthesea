# Underthesea

<p align="center">
  <img src="img/logo.png" alt="Underthesea Logo" width="300"/>
</p>

<p align="center">
  <a href="https://pypi.python.org/pypi/underthesea">
    <img src="https://img.shields.io/pypi/v/underthesea.svg" alt="PyPI version">
  </a>
  <a href="https://pypi.python.org/pypi/underthesea">
    <img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue" alt="Python versions">
  </a>
  <a href="http://undertheseanlp.com/">
    <img src="https://img.shields.io/badge/demo-live-brightgreen" alt="Live demo">
  </a>
  <a href="https://colab.research.google.com/drive/1gD8dSMSE_uNacW4qJ-NSnvRT85xo9ZY2">
    <img src="https://img.shields.io/badge/colab-ff9f01?logo=google-colab&logoColor=white" alt="Colab">
  </a>
</p>

**Underthesea** is a suite of open source Python modules, datasets, and tutorials supporting research and development in Vietnamese Natural Language Processing.

We provide an extremely easy API to quickly apply pretrained NLP models to your Vietnamese text.

!!! success "New in v9.1.5"
    Conversational AI Agent is here! Use `agent("Xin chào")` to chat with an AI assistant specialized in Vietnamese NLP.

## Installation

To install underthesea, simply:

```bash
pip install underthesea
```

Install with extras (note: use quotes in zsh):

```bash
pip install "underthesea[deep]"       # Deep learning support
pip install "underthesea[voice]"      # Text-to-Speech support
pip install "underthesea[prompt]"     # OpenAI-based classification
pip install "underthesea[langdetect]" # Language detection
pip install "underthesea[agent]"      # Conversational AI agent
```

## Tutorials

??? note "Sentence Segmentation - Breaking text into individual sentences"
    ```python
    >>> from underthesea import sent_tokenize
    >>> text = 'Taylor cho biết lúc đầu cô cảm thấy ngại với cô bạn thân Amanda nhưng rồi mọi thứ trôi qua nhanh chóng. Amanda cũng thoải mái với mối quan hệ này.'

    >>> sent_tokenize(text)
    [
      "Taylor cho biết lúc đầu cô cảm thấy ngại với cô bạn thân Amanda nhưng rồi mọi thứ trôi qua nhanh chóng.",
      "Amanda cũng thoải mái với mối quan hệ này."
    ]
    ```

??? note "Text Normalization - Standardizing textual data representation"
    ```python
    >>> from underthesea import text_normalize
    >>> text_normalize("Ðảm baỏ chất lựơng phòng thí nghịêm hoá học")
    "Đảm bảo chất lượng phòng thí nghiệm hóa học"
    ```

??? note "Word Segmentation - Dividing text into individual words"
    ```python
    >>> from underthesea import word_tokenize
    >>> text = "Chàng trai 9X Quảng Trị khởi nghiệp từ nấm sò"

    >>> word_tokenize(text)
    ["Chàng trai", "9X", "Quảng Trị", "khởi nghiệp", "từ", "nấm", "sò"]

    >>> word_tokenize(text, format="text")
    "Chàng_trai 9X Quảng_Trị khởi_nghiệp từ nấm sò"

    >>> text = "Viện Nghiên Cứu chiến lược quốc gia về học máy"
    >>> fixed_words = ["Viện Nghiên Cứu", "học máy"]
    >>> word_tokenize(text, fixed_words=fixed_words)
    "Viện_Nghiên_Cứu chiến_lược quốc_gia về học_máy"
    ```

??? note "POS Tagging - Labeling words with their part-of-speech"
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

??? note "Chunking - Grouping words into meaningful phrases"
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

??? note "Dependency Parsing - Analyzing grammatical structure (requires `[deep]`)"
    ```bash
    pip install "underthesea[deep]"
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

??? note "Named Entity Recognition - Identifying named entities"
    **CRF Model (Default)**

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

    **Deep Learning Model (requires `[deep]`)**

    ```bash
    pip install "underthesea[deep]"
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

??? note "Text Classification - Categorizing text into predefined groups"
    **CRF Model (Default)**

    ```python
    >>> from underthesea import classify

    >>> classify('HLV đầu tiên ở Premier League bị sa thải sau 4 vòng đấu')
    ['The thao']

    >>> classify('Hội đồng tư vấn kinh doanh Asean vinh danh giải thưởng quốc tế')
    ['Kinh doanh']

    >>> classify('Lãi suất từ BIDV rất ưu đãi', domain='bank')
    ['INTEREST_RATE']
    ```

    **Prompt-based Model (requires `[prompt]`)**

    ```bash
    pip install "underthesea[prompt]"
    export OPENAI_API_KEY=YOUR_KEY
    ```

    ```python
    >>> from underthesea import classify
    >>> text = "HLV ngoại đòi gần tỷ mỗi tháng dẫn dắt tuyển Việt Nam"
    >>> classify(text, model='prompt')
    'Thể thao'
    ```

??? note "Sentiment Analysis - Determining text's emotional tone"
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

??? note "Translation - Translating Vietnamese to English (requires `[deep]`)"
    ```bash
    pip install "underthesea[deep]"
    ```

    ```python
    >>> from underthesea import translate

    >>> translate("Hà Nội là thủ đô của Việt Nam")
    'Hanoi is the capital of Vietnam'

    >>> translate("Ẩm thực Việt Nam nổi tiếng trên thế giới")
    'Vietnamese cuisine is famous around the world'

    >>> translate("I love Vietnamese food", source_lang='en', target_lang='vi')
    'Tôi yêu ẩm thực Việt Nam'
    ```

??? note "Language Detection - Identifying the language of text (requires `[langdetect]`)"
    ```bash
    pip install "underthesea[langdetect]"
    ```

    ```python
    >>> from underthesea import lang_detect

    >>> lang_detect("Cựu binh Mỹ trả nhật ký nhẹ lòng khi thấy cuộc sống hòa bình tại Việt Nam")
    'vi'
    ```

??? note "Text-to-Speech - Converting text into spoken audio (requires `[voice]`)"
    ```bash
    pip install "underthesea[voice]"
    underthesea download-model VIET_TTS_V0_4_1
    ```

    ```python
    >>> from underthesea.pipeline.tts import tts

    >>> tts("Cựu binh Mỹ trả nhật ký nhẹ lòng khi thấy cuộc sống hòa bình tại Việt Nam")
    # A new audio file named `sound.wav` will be generated.
    ```

    Command line usage:

    ```bash
    underthesea tts "Cựu binh Mỹ trả nhật ký nhẹ lòng khi thấy cuộc sống hòa bình tại Việt Nam"
    ```

??? note "Conversational AI Agent - Chat with AI for Vietnamese NLP (requires `[agent]`)"
    ```bash
    pip install "underthesea[agent]"
    export OPENAI_API_KEY=your_api_key
    # Or for Azure OpenAI:
    # export AZURE_OPENAI_API_KEY=your_key
    # export AZURE_OPENAI_ENDPOINT=https://xxx.openai.azure.com
    ```

    ```python
    >>> from underthesea import agent

    >>> agent("Xin chào!")
    'Xin chào! Tôi có thể giúp gì cho bạn?'

    >>> agent("NLP là gì?")
    'NLP (Natural Language Processing) là xử lý ngôn ngữ tự nhiên...'

    >>> agent("Cho ví dụ về word tokenization tiếng Việt")
    'Word tokenization trong tiếng Việt là quá trình...'

    # Reset conversation
    >>> agent.reset()
    ```

    Supports both OpenAI and Azure OpenAI:

    ```python
    # Use Azure OpenAI
    >>> agent("Hello", provider="azure", model="my-gpt4-deployment")
    ```

## Vietnamese NLP Resources

List available resources:

```bash
underthesea list-data
```

| Name | Type | License | Year |
|------|------|---------|------|
| CP_Vietnamese_VLC_v2_2022 | Plaintext | Open | 2023 |
| UIT_ABSA_RESTAURANT | Sentiment | Open | 2021 |
| UIT_ABSA_HOTEL | Sentiment | Open | 2021 |
| SE_Vietnamese-UBS | Sentiment | Open | 2020 |
| DI_Vietnamese-UVD | Dictionary | Open | 2020 |
| UTS2017-BANK | Categorized | Open | 2017 |
| VNTC | Categorized | Open | 2007 |

Download resources:

```bash
underthesea download-data CP_Vietnamese_VLC_v2_2022
```

## Up Coming Features

* Automatic Speech Recognition

## Community

- [GitHub Repository](https://github.com/undertheseanlp/underthesea)
- [Facebook Page](https://www.facebook.com/undertheseanlp/)
- [YouTube Channel](https://www.youtube.com/channel/UC9Jv1Qg49uprg6SjkyAqs9A)
- [Google Colab Notebook](https://colab.research.google.com/drive/1gD8dSMSE_uNacW4qJ-NSnvRT85xo9ZY2)

## Support

If you found this project helpful, please consider [supporting us](https://github.com/undertheseanlp/underthesea/blob/main/docs/contribute/SUPPORT_US.md).
