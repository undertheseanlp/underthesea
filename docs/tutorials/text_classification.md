# Text Classification Tutorial

Learn how to classify Vietnamese text into categories.

## Introduction

Text classification assigns predefined categories to text. Underthesea provides models for:

- **General news classification**: Sports, Business, Politics, etc.
- **Bank domain classification**: Interest rates, Customer support, etc.
- **Prompt-based classification**: Using OpenAI for custom categories

## Basic Usage

```python
from underthesea import classify

text = "HLV đầu tiên ở Premier League bị sa thải sau 4 vòng đấu"
category = classify(text)
print(category)
# ['The thao']
```

## General Classification

The default model classifies Vietnamese news into categories:

```python
from underthesea import classify

examples = [
    ("Đội tuyển Việt Nam thắng đậm 3-0", "The thao"),
    ("Giá vàng tăng mạnh trong tuần qua", "Kinh doanh"),
    ("Quốc hội thông qua luật mới", "Chinh tri Xa hoi"),
    ("Phát hiện loài động vật mới", "Khoa hoc"),
]

for text, expected in examples:
    result = classify(text)
    print(f"{text[:30]}... -> {result[0]}")
```

### Available Categories

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

## Bank Domain Classification

For bank-related text, use `domain='bank'`:

```python
from underthesea import classify

examples = [
    "Lãi suất từ BIDV rất ưu đãi",
    "Nhân viên hỗ trợ rất nhiệt tình",
    "Thẻ tín dụng có nhiều ưu đãi",
    "Logo mới của ngân hàng rất đẹp"
]

for text in examples:
    result = classify(text, domain='bank')
    print(f"{text} -> {result}")
```

### Bank Domain Categories

| Category | Description |
|----------|-------------|
| `INTEREST_RATE` | Interest rate discussions |
| `CUSTOMER_SUPPORT` | Customer service feedback |
| `PRODUCT` | Product/service features |
| `TRADEMARK` | Brand perception |

## Prompt-based Classification

Use OpenAI for flexible classification:

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

## Practical Applications

### News Categorization System

```python
from underthesea import classify
from collections import defaultdict

def categorize_articles(articles):
    """Categorize a list of articles."""
    categorized = defaultdict(list)

    for article in articles:
        category = classify(article)[0]
        categorized[category].append(article)

    return dict(categorized)

articles = [
    "Đội tuyển Việt Nam chuẩn bị cho AFF Cup",
    "Chứng khoán tăng điểm phiên đầu tuần",
    "Phát hiện vaccine mới chống Covid-19",
    "Cầu thủ Quang Hải gia nhập CLB mới"
]

result = categorize_articles(articles)
for category, texts in result.items():
    print(f"\n{category}:")
    for text in texts:
        print(f"  - {text[:40]}...")
```

### Content Filtering

```python
from underthesea import classify

def filter_content(texts, allowed_categories):
    """Filter texts by allowed categories."""
    return [
        text for text in texts
        if classify(text)[0] in allowed_categories
    ]

all_articles = [
    "Kết quả bóng đá hôm nay",
    "Giá USD tăng mạnh",
    "Vụ án hình sự được xét xử",
    "Đội tuyển chuẩn bị thi đấu"
]

# Only keep sports articles
sports_only = filter_content(all_articles, ['The thao'])
print("Sports articles:", sports_only)
```

### Classification with Confidence Analysis

```python
from underthesea import classify

def classify_batch(texts):
    """Classify multiple texts with statistics."""
    results = []
    for text in texts:
        category = classify(text)[0]
        results.append({
            'text': text[:50] + '...' if len(text) > 50 else text,
            'category': category
        })
    return results

texts = [
    "Việt Nam vô địch AFF Cup sau 10 năm chờ đợi",
    "Ngân hàng Nhà nước điều chỉnh lãi suất",
    "Công nghệ AI đang thay đổi thế giới"
]

results = classify_batch(texts)
for r in results:
    print(f"{r['category']}: {r['text']}")
```

### Building a Simple Recommender

```python
from underthesea import classify

class ContentRecommender:
    def __init__(self):
        self.user_preferences = {}

    def record_read(self, user_id, article):
        """Record what user reads to learn preferences."""
        category = classify(article)[0]
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = {}
        self.user_preferences[user_id][category] = \
            self.user_preferences[user_id].get(category, 0) + 1

    def recommend(self, user_id, articles):
        """Recommend articles based on user preferences."""
        if user_id not in self.user_preferences:
            return articles  # No history, return all

        prefs = self.user_preferences[user_id]
        favorite_category = max(prefs, key=prefs.get)

        return [a for a in articles if classify(a)[0] == favorite_category]

# Usage
recommender = ContentRecommender()
recommender.record_read("user1", "Đội tuyển Việt Nam thắng lớn")
recommender.record_read("user1", "Cầu thủ ghi bàn đẹp mắt")

new_articles = [
    "Kết quả bóng đá hôm nay",
    "Giá vàng tăng cao",
    "Trận đấu kịch tính"
]

recommended = recommender.recommend("user1", new_articles)
print("Recommended:", recommended)
```

## Next Steps

- Learn about [Sentiment Analysis](sentiment_analysis.md)
- Explore [Translation](translation.md)
- See the [API Reference](../api/classify.md)
