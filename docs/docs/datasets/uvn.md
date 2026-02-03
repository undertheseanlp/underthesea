# Vietnamese News Dataset

A dataset of Vietnamese news articles collected from 6 major Vietnamese newspapers for NLP research.

## Dataset Description

- **Homepage:** https://github.com/undertheseanlp/underthesea
- **Repository:** https://huggingface.co/datasets/undertheseanlp/UVN-1
- **Point of Contact:** Vu Anh (anhv.ict91@gmail.com)

### Dataset Summary

This dataset contains 3,268 Vietnamese news articles covering various topics including politics, business, sports, entertainment, education, health, and technology. It is designed for Vietnamese NLP research tasks such as:

- Text classification (news categorization)
- Language modeling
- Text generation
- Named entity recognition
- Keyword extraction

### Languages

Vietnamese (vi)

## Dataset Structure

### Data Instance

```json
{
  "id": "article-00001",
  "source": "vnexpress",
  "url": "https://vnexpress.net/example-123.html",
  "category": "Thời sự",
  "title": "Tiêu đề bài báo",
  "content": "Nội dung bài báo...",
  "publish_date": "2026-01-28"
}
```

### Data Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier (article-XXXXX) |
| `source` | string | News source |
| `url` | string | Original article URL |
| `category` | string | News category |
| `title` | string | Article title |
| `content` | string | Full article content |
| `publish_date` | string | Publication date (yyyy-mm-dd) |

### Data Splits

| Split | Count |
|-------|------:|
| train | 3,268 |

### Sources

| Source | Website | Count |
|--------|---------|------:|
| VnExpress | vnexpress.net | 1,000 |
| Dân Trí | dantri.com.vn | 976 |
| Thanh Niên | thanhnien.vn | 579 |
| Tuổi Trẻ | tuoitre.vn | 210 |
| Tiền Phong | tienphong.vn | 240 |
| Người Lao Động | nld.com.vn | 263 |

### Categories

- Thời sự (Politics)
- Thế giới (World)
- Kinh doanh (Business)
- Giải trí (Entertainment)
- Thể thao (Sports)
- Pháp luật (Law)
- Giáo dục (Education)
- Sức khỏe (Health)
- Đời sống (Lifestyle)
- Khoa học (Science/Technology)

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("undertheseanlp/UVN-1")

# Access articles
for article in dataset["train"]:
    print(article["id"], article["title"])
    break

# Filter by source
vnexpress = dataset["train"].filter(lambda x: x["source"] == "vnexpress")

# Filter by category
sports = dataset["train"].filter(lambda x: x["category"] == "Thể thao")
```

## Considerations

### Biases

- Reflects editorial perspectives of mainstream Vietnamese news outlets
- May be biased toward urban and national news coverage

### Limitations

- Articles from 2025-2026 time period
- Uneven distribution across sources due to website structure differences

## License

CC-BY-NC-4.0. For research and educational purposes only.

## Citation

```bibtex
@misc{vietnamese_news_dataset,
  title={Vietnamese News Dataset},
  author={Underthesea NLP},
  year={2026},
  publisher={HuggingFace},
  url={https://huggingface.co/datasets/undertheseanlp/UVN-1}
}
```
