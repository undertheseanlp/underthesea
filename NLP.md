# Vietnamese NLP

Tài liệu chi tiết về các NLP pipelines của Underthesea.

## Sentence Segmentation

Breaking text into individual sentences.

```python
>>> from underthesea import sent_tokenize
>>> text = 'Taylor cho biết lúc đầu cô cảm thấy ngại với cô bạn thân Amanda nhưng rồi mọi thứ trôi qua nhanh chóng. Amanda cũng thoải mái với mối quan hệ này.'

>>> sent_tokenize(text)
[
  "Taylor cho biết lúc đầu cô cảm thấy ngại với cô bạn thân Amanda nhưng rồi mọi thứ trôi qua nhanh chóng.",
  "Amanda cũng thoải mái với mối quan hệ này."
]
```

## Text Normalization

Standardizing textual data representation and address conversion.

```python
>>> from underthesea import text_normalize
>>> text_normalize("Ðảm baỏ chất lựơng phòng thí nghịêm hoá học")
"Đảm bảo chất lượng phòng thí nghiệm hóa học"
```

Address Conversion:

```python
>>> from underthesea import convert_address
>>> result = convert_address("Phường Phúc Xá, Quận Ba Đình, Thành phố Hà Nội")
>>> result.converted
"Phường Hồng Hà, Thành phố Hà Nội"
>>> result.mapping_type
<MappingType.MERGED: 'merged'>
```

Supports abbreviations:

```python
>>> result = convert_address("P. Phúc Xá, Q. Ba Đình, TP. Hà Nội")
>>> result.converted
"Phường Hồng Hà, Thành phố Hà Nội"
```

## Tagging

### Word Segmentation

```python
>>> from underthesea import word_tokenize
>>> word_tokenize("Chàng trai 9X Quảng Trị khởi nghiệp từ nấm sò")
["Chàng trai", "9X", "Quảng Trị", "khởi nghiệp", "từ", "nấm", "sò"]

>>> word_tokenize("Chàng trai 9X Quảng Trị khởi nghiệp từ nấm sò", format="text")
"Chàng_trai 9X Quảng_Trị khởi_nghiệp từ nấm sò"
```

### POS Tagging

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

### Chunking

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

### Dependency Parsing

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

## Named Entity Recognition

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

Deep Learning Model:

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

## Classification

### Text Classification

```python
>>> from underthesea import classify

>>> classify('HLV đầu tiên ở Premier League bị sa thải sau 4 vòng đấu')
['The thao']

>>> classify('Hội đồng tư vấn kinh doanh Asean vinh danh giải thưởng quốc tế')
['Kinh doanh']

>> classify('Lãi suất từ BIDV rất ưu đãi', domain='bank')
['INTEREST_RATE']
```

### Sentiment Analysis

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

### Prompt-based Classification

```bash
$ pip install underthesea[prompt]
$ export OPENAI_API_KEY=YOUR_KEY
```

```python
>>> from underthesea import classify
>>> classify("HLV ngoại đòi gần tỷ mỗi tháng dẫn dắt tuyển Việt Nam", model='prompt')
Thể thao
```

## Lang Detect

Powered by [FastText](https://fasttext.cc/docs/en/language-identification.html) language identification model, using pure Rust inference via `underthesea_core`.

```python
>>> from underthesea import lang_detect

>>> lang_detect("Cựu binh Mỹ trả nhật ký nhẹ lòng khi thấy cuộc sống hòa bình tại Việt Nam")
vi
```

## Translation

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

## Text-to-Speech

Thanks to awesome work from [NTT123/vietTTS](https://github.com/ntt123/vietTTS)

```bash
$ pip install "underthesea[voice]"
$ underthesea download-model VIET_TTS_V0_4_1
```

```python
>>> from underthesea.pipeline.tts import tts

>>> tts("Cựu binh Mỹ trả nhật ký nhẹ lòng khi thấy cuộc sống hòa bình tại Việt Nam")
A new audio file named `sound.wav` will be generated.
```

```sh
$ underthesea tts "Cựu binh Mỹ trả nhật ký nhẹ lòng khi thấy cuộc sống hòa bình tại Việt Nam"
```

## Resources

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

```bash
$ underthesea download-data CP_Vietnamese_VLC_v2_2022
Resource CP_Vietnamese_VLC_v2_2022 is downloaded in ~/.underthesea/datasets/CP_Vietnamese_VLC_v2_2022 folder
```
