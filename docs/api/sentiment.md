# sentiment

Analyze the sentiment of text.

## Usage

```python
from underthesea import sentiment

text = "Sản phẩm hơi nhỏ so với tưởng tượng nhưng chất lượng tốt"
result = sentiment(text)
print(result)
# 'positive'
```

## Function Signature

```python
def sentiment(
    X: str,
    domain: str = 'general'
) -> str | list[str]
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `X` | `str` | | The input text to analyze |
| `domain` | `str` | `'general'` | Domain for analysis (`'general'` or `'bank'`) |

## Returns

### General Domain

| Type | Description |
|------|-------------|
| `str` | Sentiment label: `'positive'`, `'negative'`, or `'neutral'` |

### Bank Domain

| Type | Description |
|------|-------------|
| `list[str]` | List of aspect-sentiment pairs (e.g., `['ASPECT#sentiment']`) |

## Examples

### Basic Usage

```python
from underthesea import sentiment

# Positive sentiment
sentiment("Sản phẩm chất lượng tốt, đóng gói cẩn thận")
# 'positive'

# Negative sentiment
sentiment("hàng kém chất lg, chăn đắp lên dính lông lá khắp người. thất vọng")
# 'negative'
```

### Bank Domain

The bank domain provides aspect-based sentiment analysis:

```python
from underthesea import sentiment

# Customer support aspect
sentiment("Đky qua đường link ở bài viết này từ thứ 6 mà giờ chưa thấy ai lhe hết", domain='bank')
# ['CUSTOMER_SUPPORT#negative']

# Trademark aspect
sentiment("Xem lại vẫn thấy xúc động và tự hào về BIDV của mình", domain='bank')
# ['TRADEMARK#positive']
```

### Bank Domain Aspects

| Aspect | Description |
|--------|-------------|
| `INTEREST_RATE` | Interest rate related |
| `CUSTOMER_SUPPORT` | Customer service quality |
| `PRODUCT` | Product/service quality |
| `TRADEMARK` | Brand perception |

### Processing Reviews

```python
from underthesea import sentiment

reviews = [
    "Dịch vụ tuyệt vời, nhân viên nhiệt tình",
    "Giao hàng chậm, đóng gói không cẩn thận",
    "Sản phẩm bình thường, không có gì đặc biệt"
]

for review in reviews:
    result = sentiment(review)
    print(f"{review[:30]}... -> {result}")
# Dịch vụ tuyệt vời, nhân viên n... -> positive
# Giao hàng chậm, đóng gói không... -> negative
# Sản phẩm bình thường, không có... -> neutral
```

### Counting Sentiment Distribution

```python
from collections import Counter
from underthesea import sentiment

reviews = [
    "Sản phẩm tốt",
    "Không hài lòng",
    "Rất thích",
    "Tệ quá",
    "Bình thường"
]

sentiments = [sentiment(r) for r in reviews]
distribution = Counter(sentiments)
print(distribution)
# Counter({'positive': 2, 'negative': 2, 'neutral': 1})
```

## Accessing Available Labels

You can access all available sentiment labels using the `labels` property:

```python
from underthesea import sentiment

# Get labels for general domain
sentiment.labels
# ['positive', 'negative']

# Get labels for bank domain
sentiment.bank.labels
# ['ACCOUNT#negative', 'CARD#negative', 'CARD#neutral', 'CARD#positive',
#  'CUSTOMER_SUPPORT#negative', 'CUSTOMER_SUPPORT#neutral', 'CUSTOMER_SUPPORT#positive',
#  'DISCOUNT#negative', 'DISCOUNT#neutral', 'DISCOUNT#positive', ...]
```

### Checking Valid Labels

```python
from underthesea import sentiment

# Check if a result is a valid label
result = sentiment("Sản phẩm rất tốt")
print(result in sentiment.labels)  # True

# Get all available aspect-sentiment combinations for bank
print(f"Bank domain has {len(sentiment.bank.labels)} labels")
```

## Notes

- The general domain model classifies into positive/negative
- The bank domain model provides aspect-based sentiment
- First call may take longer due to model loading
- Use `sentiment.labels` to get all available labels for the general domain
- Use `sentiment.bank.labels` to get all available labels for the bank domain
