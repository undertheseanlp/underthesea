# classify

Categorize text into predefined categories.

## Usage

```python
from underthesea import classify

text = "HLV đầu tiên ở Premier League bị sa thải sau 4 vòng đấu"
category = classify(text)
print(category)
# ['The thao']
```

## Function Signature

```python
def classify(
    X: str,
    domain: str = None,
    model: str = None
) -> list[str] | str
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `X` | `str` | | The input text to classify |
| `domain` | `str` | `None` | Domain for classification (`'bank'`) |
| `model` | `str` | `None` | Model type (`'prompt'` for OpenAI) |

## Returns

| Type | Description |
|------|-------------|
| `list[str]` | List of predicted categories |

## Available Domains

### General (Default)

News topic classification:

| Category | Description |
|----------|-------------|
| `The thao` | Sports |
| `Kinh doanh` | Business |
| `Chinh tri Xa hoi` | Politics & Society |
| `Van hoa` | Culture |
| `Khoa hoc` | Science |
| `Phap luat` | Law |
| `Suc khoe` | Health |
| `Doi song` | Lifestyle |
| `The gioi` | World |
| `Vi tinh` | Technology |

### Bank Domain

Bank-related topic classification:

```python
classify(text, domain='bank')
```

| Category | Description |
|----------|-------------|
| `INTEREST_RATE` | Interest rate related |
| `CUSTOMER_SUPPORT` | Customer service |
| `PRODUCT` | Bank products |
| `TRADEMARK` | Brand/trademark |

## Examples

### Basic Usage

```python
from underthesea import classify

# Sports
classify("HLV đầu tiên ở Premier League bị sa thải sau 4 vòng đấu")
# ['The thao']

# Business
classify("Hội đồng tư vấn kinh doanh Asean vinh danh giải thưởng quốc tế")
# ['Kinh doanh']
```

### Bank Domain

```python
from underthesea import classify

classify("Lãi suất từ BIDV rất ưu đãi", domain='bank')
# ['INTEREST_RATE']

classify("Nhân viên hỗ trợ rất nhiệt tình", domain='bank')
# ['CUSTOMER_SUPPORT']
```

### Prompt-based Model

!!! note "Requires Installation"
    ```bash
    pip install underthesea[prompt]
    export OPENAI_API_KEY=your_api_key
    ```

```python
from underthesea import classify

text = "HLV ngoại đòi gần tỷ mỗi tháng dẫn dắt tuyển Việt Nam"
result = classify(text, model='prompt')
print(result)
# 'Thể thao'
```

### Processing Multiple Documents

```python
from underthesea import classify

documents = [
    "Đội tuyển Việt Nam thắng đậm trong trận đấu",
    "Giá vàng tăng mạnh trong tuần qua",
    "Phát hiện virus mới gây bệnh ở châu Phi"
]

for doc in documents:
    category = classify(doc)
    print(f"{doc[:30]}... -> {category}")
```

## Accessing Available Labels

You can access all available category labels using the `labels` property:

```python
from underthesea import classify

# Get labels for general domain
classify.labels
# ['chinh_tri_xa_hoi', 'doi_song', 'khoa_hoc', 'kinh_doanh', 'phap_luat',
#  'suc_khoe', 'the_gioi', 'the_thao', 'van_hoa', 'vi_tinh']

# Get labels for bank domain
classify.bank.labels
# ['ACCOUNT', 'CARD', 'CUSTOMER_SUPPORT', 'DISCOUNT', 'INTEREST_RATE',
#  'INTERNET_BANKING', 'LOAN', 'MONEY_TRANSFER', 'OTHER', 'PAYMENT',
#  'PROMOTION', 'SAVING', 'SECURITY', 'TRADEMARK']
```

## Notes

- The default model is trained on Vietnamese news data
- The bank domain model is specialized for banking feedback
- Prompt-based model uses OpenAI API and requires an API key
- First call may take longer due to model loading
- Use `classify.labels` to get all available categories for the default domain
- Use `classify.bank.labels` to get all available categories for the bank domain
