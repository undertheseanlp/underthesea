# Sentiment Analysis Tutorial

Learn how to analyze the sentiment of Vietnamese text.

## Introduction

Sentiment analysis determines the emotional tone of text:
- **Positive**: Happy, satisfied, praising
- **Negative**: Unhappy, disappointed, complaining
- **Neutral**: Factual, no strong emotion

## Basic Usage

```python
from underthesea import sentiment

text = "Sản phẩm hơi nhỏ so với tưởng tượng nhưng chất lượng tốt"
result = sentiment(text)
print(result)
# 'positive'
```

## General Sentiment Analysis

```python
from underthesea import sentiment

examples = [
    "Sản phẩm chất lượng tốt, đóng gói cẩn thận",  # positive
    "hàng kém chất lg, thất vọng",                   # negative
    "Sản phẩm bình thường, không có gì đặc biệt"    # neutral
]

for text in examples:
    result = sentiment(text)
    print(f"{result:10} | {text[:40]}...")
```

## Bank Domain Sentiment

For bank-related feedback, use `domain='bank'`:

```python
from underthesea import sentiment

examples = [
    "Lãi suất rất hấp dẫn, tôi rất hài lòng",
    "Nhân viên hỗ trợ chậm, phải chờ lâu",
    "App ngân hàng rất dễ sử dụng"
]

for text in examples:
    result = sentiment(text, domain='bank')
    print(f"{text}")
    print(f"  -> {result}\n")
```

### Bank Domain Output

The bank domain returns aspect-based sentiment:

| Aspect | Example |
|--------|---------|
| `INTEREST_RATE#positive` | "Lãi suất rất tốt" |
| `CUSTOMER_SUPPORT#negative` | "Nhân viên không nhiệt tình" |
| `PRODUCT#positive` | "Thẻ tín dụng có nhiều ưu đãi" |
| `TRADEMARK#positive` | "Tự hào là khách hàng của BIDV" |

## Practical Applications

### Product Review Analysis

```python
from underthesea import sentiment
from collections import Counter

def analyze_reviews(reviews):
    """Analyze sentiment of product reviews."""
    sentiments = [sentiment(review) for review in reviews]

    # Count distribution
    distribution = Counter(sentiments)
    total = len(sentiments)

    print("Sentiment Distribution:")
    for sent, count in distribution.most_common():
        percentage = count / total * 100
        print(f"  {sent}: {count} ({percentage:.1f}%)")

    return distribution

reviews = [
    "Sản phẩm rất tốt, đáng mua",
    "Giao hàng nhanh, đóng gói cẩn thận",
    "Hàng không giống hình, thất vọng",
    "Chất lượng tạm được",
    "Tuyệt vời, sẽ mua lại",
    "Quá tệ, không khuyên mua"
]

analyze_reviews(reviews)
```

### Customer Feedback Dashboard

```python
from underthesea import sentiment

class FeedbackAnalyzer:
    def __init__(self):
        self.feedback = []

    def add_feedback(self, text, source="unknown"):
        result = sentiment(text)
        self.feedback.append({
            'text': text,
            'sentiment': result,
            'source': source
        })

    def get_summary(self):
        if not self.feedback:
            return "No feedback yet"

        positive = sum(1 for f in self.feedback if f['sentiment'] == 'positive')
        negative = sum(1 for f in self.feedback if f['sentiment'] == 'negative')
        neutral = sum(1 for f in self.feedback if f['sentiment'] == 'neutral')

        total = len(self.feedback)
        return {
            'total': total,
            'positive': positive,
            'negative': negative,
            'neutral': neutral,
            'satisfaction_rate': positive / total * 100
        }

    def get_negative_feedback(self):
        return [f for f in self.feedback if f['sentiment'] == 'negative']

# Usage
analyzer = FeedbackAnalyzer()
analyzer.add_feedback("Dịch vụ tuyệt vời", source="email")
analyzer.add_feedback("Chờ lâu quá", source="phone")
analyzer.add_feedback("Nhân viên nhiệt tình", source="survey")

print("Summary:", analyzer.get_summary())
print("Negative:", analyzer.get_negative_feedback())
```

### Social Media Monitoring

```python
from underthesea import sentiment
from datetime import datetime

def monitor_brand_sentiment(mentions):
    """Monitor brand sentiment from social media mentions."""
    results = []

    for mention in mentions:
        result = sentiment(mention['text'])
        results.append({
            **mention,
            'sentiment': result
        })

    # Aggregate by sentiment
    positive_mentions = [r for r in results if r['sentiment'] == 'positive']
    negative_mentions = [r for r in results if r['sentiment'] == 'negative']

    print(f"Total mentions: {len(results)}")
    print(f"Positive: {len(positive_mentions)}")
    print(f"Negative: {len(negative_mentions)}")

    if negative_mentions:
        print("\nNegative mentions (need attention):")
        for m in negative_mentions:
            print(f"  - {m['text'][:50]}...")

    return results

mentions = [
    {'text': 'Sản phẩm ABC rất tốt, đáng mua', 'platform': 'facebook'},
    {'text': 'Dịch vụ của công ty XYZ quá tệ', 'platform': 'twitter'},
    {'text': 'Rất hài lòng với đơn hàng', 'platform': 'instagram'}
]

monitor_brand_sentiment(mentions)
```

### Sentiment Trend Analysis

```python
from underthesea import sentiment
from collections import defaultdict

def analyze_sentiment_trend(data_by_date):
    """Analyze sentiment trend over time."""
    trend = {}

    for date, texts in data_by_date.items():
        sentiments = [sentiment(t) for t in texts]
        positive_rate = sum(1 for s in sentiments if s == 'positive') / len(sentiments)
        trend[date] = positive_rate * 100

    print("Sentiment Trend (Positive %):")
    for date, rate in sorted(trend.items()):
        bar = '█' * int(rate / 5)
        print(f"  {date}: {bar} {rate:.1f}%")

    return trend

data = {
    '2024-01-01': ["Sản phẩm tốt", "Hài lòng", "Tạm được"],
    '2024-01-02': ["Thất vọng", "Không tốt", "Bình thường"],
    '2024-01-03': ["Tuyệt vời", "Rất thích", "Sẽ mua lại", "Tốt"]
}

analyze_sentiment_trend(data)
```

## Tips for Better Results

1. **Preprocessing**: Clean text before analysis (remove noise, fix typos)
2. **Context**: Consider using domain-specific model when available
3. **Batch Processing**: Process multiple texts for efficiency
4. **Error Handling**: Handle edge cases (empty text, very short text)

```python
def safe_sentiment(text):
    """Safely analyze sentiment with error handling."""
    if not text or len(text.strip()) < 5:
        return 'neutral'
    try:
        return sentiment(text)
    except Exception as e:
        print(f"Error analyzing: {text[:20]}... - {e}")
        return 'unknown'
```

## Next Steps

- Learn about [Translation](translation.md)
- Explore [Text-to-Speech](tts.md)
- See the [API Reference](../api/sentiment.md)
