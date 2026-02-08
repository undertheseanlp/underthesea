# Classification

## Overview

This report covers two classification pipelines in Underthesea: **Text Classification** and **Sentiment Analysis**. Both use `underthesea_core.TextClassifier` and support multiple domains.

---

## Text Classification

The text classification module categorizes Vietnamese text into predefined categories. It supports a general news domain (10 categories) and a bank domain (14 categories), with an optional OpenAI prompt-based model.

### Architecture

```
Text Classification Pipeline
├── Text Input
│   └── Raw Vietnamese text
├── Model Selection
│   ├── General Domain (default)
│   │   └── underthesea_core.TextClassifier
│   ├── Bank Domain (domain='bank')
│   │   └── underthesea_core.TextClassifier
│   └── Prompt Model (model='prompt')
│       └── OpenAI API
└── Output
    └── List of predicted categories
```

### Models

| Model | File | Description |
|-------|------|-------------|
| General | `sen-classifier-general-1.0.0-20260207.bin` | Vietnamese news classification |
| Bank | `sen-bank-1.0.0-20260207.bin` | Banking feedback classification |
| Prompt | OpenAI API | LLM-based classification |

### Categories

#### General Domain (10 categories)

| Category | Description |
|----------|-------------|
| The thao | Sports |
| Kinh doanh | Business |
| Chinh tri Xa hoi | Politics & Society |
| Van hoa | Culture |
| Khoa hoc | Science |
| Phap luat | Law |
| Suc khoe | Health |
| Doi song | Lifestyle |
| The gioi | World |
| Vi tinh | Technology |

#### Bank Domain (14 categories)

| Category | Description |
|----------|-------------|
| ACCOUNT | Account management |
| CARD | Card services |
| CUSTOMER_SUPPORT | Customer support |
| DISCOUNT | Discounts |
| INTEREST_RATE | Interest rates |
| INTERNET_BANKING | Internet banking |
| LOAN | Loan services |
| MONEY_TRANSFER | Money transfers |
| OTHER | Other topics |
| PAYMENT | Payments |
| PROMOTION | Promotions |
| SAVING | Savings |
| SECURITY | Security |
| TRADEMARK | Brand-related |

### Usage

```python
from underthesea import classify

text = "HLV đầu tiên ở Premier League bị sa thải sau 4 vòng đấu"
category = classify(text)
# ['The thao']

# Bank domain
classify("Lãi suất tiết kiệm quá thấp", domain='bank')
# ['INTEREST_RATE']

# Access labels
classify.labels        # General domain labels
classify.bank.labels   # Bank domain labels
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `X` | str | required | Text to classify |
| `domain` | str | None | Domain — `None` for general, `'bank'` for banking |
| `model` | str | None | Model type — `None` for default, `'prompt'` for OpenAI |

---

## Sentiment Analysis

The sentiment analysis module analyzes the sentiment of Vietnamese text. The general domain returns positive/negative/neutral classification. The bank domain provides aspect-based sentiment analysis.

### Architecture

```
Sentiment Analysis Pipeline
├── Text Input
│   └── Raw Vietnamese text
├── Model Selection
│   ├── General Domain (default)
│   │   └── underthesea_core.TextClassifier
│   │       └── 3-class: positive / negative / neutral
│   └── Bank Domain (domain='bank')
│       └── underthesea_core.TextClassifier
│           └── Aspect-based sentiment
└── Output
    ├── General: sentiment string
    └── Bank: list of aspect#sentiment pairs
```

### Models

| Model | File | Description |
|-------|------|-------------|
| General | `sen-sentiment-general-1.0.0-20260207.bin` | 3-class sentiment |
| Bank | `sen-sentiment-bank-1.0.0-20260207.bin` | Aspect-based sentiment |

### Sentiment Labels

#### General Domain

| Label | Description |
|-------|-------------|
| `positive` | Positive sentiment |
| `negative` | Negative sentiment |
| `neutral` | Neutral sentiment |

#### Bank Domain — Aspects

| Aspect | Description |
|--------|-------------|
| INTEREST_RATE | Interest rate related |
| CUSTOMER_SUPPORT | Customer service quality |
| PRODUCT | Product/service quality |
| TRADEMARK | Brand perception |

### Usage

```python
from underthesea import sentiment

text = "Sản phẩm hơi nhỏ so với tưởng tượng nhưng chất lượng tốt"
result = sentiment(text)
# 'positive'

# Bank domain
sentiment("Lãi suất quá cao, nhân viên hỗ trợ tốt", domain='bank')
# ['INTEREST_RATE#negative', 'CUSTOMER_SUPPORT#positive']

# Access labels
sentiment.labels        # General domain labels
sentiment.bank.labels   # Bank domain labels
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `X` | str | required | Text to analyze |
| `domain` | str | `'general'` | Domain — `'general'` or `'bank'` |

---

## References

1. [Underthesea GitHub Repository](https://github.com/undertheseanlp/underthesea)
