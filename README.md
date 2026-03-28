<p align="center">
  <br>
  <img src="https://raw.githubusercontent.com/undertheseanlp/underthesea/main/docs/static/img/logo.png"/>
  <br/>
</p>

<p align="center">
  <a href="https://pypi.python.org/pypi/underthesea">
    <img src="https://img.shields.io/pypi/v/underthesea.svg">
  </a>
  <a href="https://pypi.python.org/pypi/underthesea">
    <img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue">
  </a>
  <a href="http://undertheseanlp.com/">
    <img src="https://img.shields.io/badge/demo-live-brightgreen">
  </a>
  <a href="https://undertheseanlp.github.io/underthesea/">
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
  <a href="https://github.com/undertheseanlp/underthesea/blob/main/docs/contribute/SPONSORS.md">
    <img src="https://img.shields.io/badge/sponsors-6-red?style=social&logo=GithubSponsors">
  </a>
</p>

<h3 align="center">
Open-source Vietnamese Natural Language Process Toolkit
</h3>

`Underthesea` is:

🌊 **A Vietnamese NLP toolkit.** Underthesea is a suite of open source Python modules data sets and tutorials supporting research and development in [Vietnamese Natural Language Processing](https://github.com/undertheseanlp/underthesea). We provides extremely easy API to quickly apply pretrained NLP models to your Vietnamese text, such as word segmentation, part-of-speech tagging (PoS), named entity recognition (NER), text classification and dependency parsing.

🎁 [**Support Us!**](#-support-us) Every bit of support helps us achieve our goals. Thank you so much. 💝💝💝

🎉 **New in v9.3!** Agent Harness with multi-provider support (OpenAI, Azure, Anthropic, Gemini), streaming, multi-session with context management. Zero external dependencies. 🚀✨

## Installation


To install underthesea, simply:

```bash
$ pip install underthesea
✨🍰✨
```

Satisfaction, guaranteed.

Install with extras (note: use quotes in zsh):

```bash
$ pip install "underthesea[deep]"    # Deep learning support
$ pip install "underthesea[voice]"   # Text-to-Speech support
$ pip install "underthesea[agent]"   # AI agent (zero extra deps, uses raw HTTP)
```

## Tutorials

### Natural Language Processing

<details>
<summary><b><a href="">Sentence Segmentation</a></b> - Breaking text into individual sentences
</summary>

- Usage

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
<summary><b><a href="">Text Normalization</a></b> - Standardizing textual data representation and address conversion
</summary>

- Usage

    ```python
    >>> from underthesea import text_normalize
    >>> text_normalize("Ðảm baỏ chất lựơng phòng thí nghịêm hoá học")
    "Đảm bảo chất lượng phòng thí nghiệm hóa học"
    ```

- Address Conversion

    ```python
    >>> from underthesea import convert_address
    >>> result = convert_address("Phường Phúc Xá, Quận Ba Đình, Thành phố Hà Nội")
    >>> result.converted
    "Phường Hồng Hà, Thành phố Hà Nội"
    >>> result.mapping_type
    <MappingType.MERGED: 'merged'>
    ```

- Supports abbreviations

    ```python
    >>> result = convert_address("P. Phúc Xá, Q. Ba Đình, TP. Hà Nội")
    >>> result.converted
    "Phường Hồng Hà, Thành phố Hà Nội"
    ```
</details>

<details>
<summary><b><a href="">Tagging</a></b> - Word segmentation, POS tagging, chunking, dependency parsing
</summary>
<br/>

- **Word Segmentation**

    ```python
    >>> from underthesea import word_tokenize
    >>> word_tokenize("Chàng trai 9X Quảng Trị khởi nghiệp từ nấm sò")
    ["Chàng trai", "9X", "Quảng Trị", "khởi nghiệp", "từ", "nấm", "sò"]

    >>> word_tokenize("Chàng trai 9X Quảng Trị khởi nghiệp từ nấm sò", format="text")
    "Chàng_trai 9X Quảng_Trị khởi_nghiệp từ nấm sò"
    ```

- **POS Tagging**

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

- **Chunking**

    ```python
    >>> from underthesea import chunk
    >>> chunk('Bác sĩ bây giờ có thể thản nhiên báo tin bệnh nhân bị ung thư?')
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

- **Dependency Parsing**

    ```bash
    $ pip install underthesea[deep]
    ```

    ```python
    >>> from underthesea import dependency_parse
    >>> dependency_parse('Tối 29/11, Việt Nam thêm 2 ca mắc Covid-19')
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
<summary><b><a href="">Named Entity Recognition</a></b> - Identifying named entities (e.g., names, locations)
</summary>
<br/>

- Usage

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

- Deep Learning Model

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
<summary><b><a href="">Classification</a></b> - Text classification and sentiment analysis
</summary>
<br/>

- **Text Classification**

    ```python
    >>> from underthesea import classify

    >>> classify('HLV đầu tiên ở Premier League bị sa thải sau 4 vòng đấu')
    ['The thao']

    >>> classify('Hội đồng tư vấn kinh doanh Asean vinh danh giải thưởng quốc tế')
    ['Kinh doanh']

    >> classify('Lãi suất từ BIDV rất ưu đãi', domain='bank')
    ['INTEREST_RATE']
    ```

- **Sentiment Analysis**

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

- **Prompt-based Classification**

    ```bash
    $ pip install underthesea[prompt]
    $ export OPENAI_API_KEY=YOUR_KEY
    ```

    ```python
    >>> from underthesea import classify
    >>> classify("HLV ngoại đòi gần tỷ mỗi tháng dẫn dắt tuyển Việt Nam", model='prompt')
    Thể thao
    ```
</details>

<details>
<summary><b><a href="">Lang Detect</a></b> - Identifying the Language of Text
</summary>

<br/>

Lang Detect API. Powered by [FastText](https://fasttext.cc/docs/en/language-identification.html) language identification model, using pure Rust inference via `underthesea_core`.

Usage examples in script

    ```python
    >>> from underthesea import lang_detect

    >>> lang_detect("Cựu binh Mỹ trả nhật ký nhẹ lòng khi thấy cuộc sống hòa bình tại Việt Nam")
    vi
    ```
</details>

<details>
<summary><b><a href="">Translation</a></b> - Translating Vietnamese text to English
</summary>
<br/>

- Deep Learning Model

    ```bash
    $ pip install underthesea[deep]
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
</details>

### Voice

<details>
<summary><b><a href="">Text-to-Speech</a></b> - Converting written text into spoken audio
</summary>

<br/>

Text to Speech API. Thanks to awesome work from [NTT123/vietTTS](https://github.com/ntt123/vietTTS)

Install extend dependencies and models

    ```bash
    $ pip install "underthesea[voice]"
    $ underthesea download-model VIET_TTS_V0_4_1
    ```

Usage examples in script

    ```python
    >>> from underthesea.pipeline.tts import tts

    >>> tts("Cựu binh Mỹ trả nhật ký nhẹ lòng khi thấy cuộc sống hòa bình tại Việt Nam")
    A new audio file named `sound.wav` will be generated.
    ```

Usage examples in command line

    ```sh
    $ underthesea tts "Cựu binh Mỹ trả nhật ký nhẹ lòng khi thấy cuộc sống hòa bình tại Việt Nam"
    ```
</details>

### Agents

<details>
<summary><b><a href="">AI Agent</a></b> - Multi-provider, streaming, tool calling, multi-session
</summary>

<br/>

AI Agent with multi-provider support. **Zero external dependencies** - uses raw HTTP calls to LLM APIs.

Supported providers: **OpenAI**, **Azure OpenAI**, **Anthropic Claude**, **Google Gemini**

Setup (chỉ cần set env var, không cần cài thêm package)

    ```bash
    # Chọn 1 trong các provider:
    $ export OPENAI_API_KEY=sk-...
    $ export AZURE_OPENAI_API_KEY=... && export AZURE_OPENAI_ENDPOINT=https://...
    $ export ANTHROPIC_API_KEY=sk-ant-...
    $ export GOOGLE_API_KEY=...
    ```

Quick Start

    ```python
    >>> from underthesea.agent import Agent, LLM

    # Auto-detect provider từ env vars
    >>> agent = Agent(name="assistant", provider=LLM())
    >>> agent("Xin chào!")
    'Xin chào! Tôi có thể giúp gì cho bạn?'
    ```

Explicit Provider

    ```python
    >>> from underthesea.agent import Agent, OpenAI, AzureOpenAI, Anthropic, Gemini

    >>> agent = Agent(name="bot", provider=OpenAI(api_key="sk-..."))
    >>> agent = Agent(name="bot", provider=AzureOpenAI(
    ...     api_key="...", endpoint="https://my.openai.azure.com", deployment="gpt-4"
    ... ))
    >>> agent = Agent(name="bot", provider=Anthropic(api_key="sk-ant-..."))
    >>> agent = Agent(name="bot", provider=Gemini(api_key="..."))
    ```

Streaming

    ```python
    >>> for chunk in agent.stream("Giải thích AI agent là gì?"):
    ...     print(chunk, end="", flush=True)
    ```

Tool Calling (Function Calling)

    ```python
    >>> from underthesea.agent import Agent, Tool, OpenAI

    >>> def get_weather(location: str) -> dict:
    ...     """Get current weather for a location."""
    ...     return {"location": location, "temp": 25, "condition": "sunny"}

    >>> agent = Agent(
    ...     name="assistant",
    ...     provider=OpenAI(),
    ...     tools=[Tool(get_weather)],
    ...     instruction="You are a helpful Vietnamese assistant."
    ... )
    >>> agent("Thời tiết ở Hà Nội thế nào?")
    'Thời tiết ở Hà Nội hiện tại là 25°C và nắng.'
    ```

Multi-Session Agent (với context reset và structured handoff)

    ```python
    >>> from underthesea.agent import Agent, Session, AzureOpenAI

    >>> agent = Agent(name="researcher", provider=AzureOpenAI(...))
    >>> session = Session(agent, progress_file="progress.json")
    >>> session.create_task("Phân tích tài liệu", [
    ...     "Đọc và phân loại tài liệu",
    ...     "Tóm tắt từng nhóm",
    ...     "Viết báo cáo tổng hợp",
    ... ])
    >>> session.run_until_complete(max_sessions=5)
    ```

Default Tools (calculator, web search, file I/O, shell, python...)

    ```python
    >>> from underthesea.agent import Agent, default_tools, LLM

    >>> agent = Agent(name="assistant", provider=LLM(), tools=default_tools)
    >>> agent("Tính sqrt(144) + 10")  # Uses calculator tool
    ```
</details>

### Resources

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

## Contributing

Do you want to contribute with underthesea development? Great! Please read more details at [Contributing Guide](https://undertheseanlp.github.io/underthesea/docs/developer/contributing)

## 💝 Support Us

If you found this project helpful and would like to support our work, you can just buy us a coffee ☕.

Your support is our biggest encouragement 🎁!

<img src="https://raw.githubusercontent.com/undertheseanlp/underthesea/main/docs/static/img/support.png"/>
