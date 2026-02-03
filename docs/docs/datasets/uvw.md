# UVW

Underthesea Vietnamese Wikipedia Dataset (2026 Edition)

A high-quality, cleaned dataset of 1.1M Vietnamese Wikipedia articles enriched with Wikidata metadata for NLP research.

## HuggingFace Dataset

**Dataset:** [undertheseanlp/UVW-2026](https://huggingface.co/datasets/undertheseanlp/UVW-2026)

## Features

| Feature | Type | Description |
|---------|------|-------------|
| id | string | Unique identifier (URL-safe title) |
| title | string | Article title |
| content | string | Cleaned article text |
| num_chars | int32 | Character count |
| num_sentences | int32 | Sentence count |
| quality_score | int32 | Quality score (1-10) |
| wikidata_id | string | Wikidata Q-identifier |
| main_category | string | Primary category from Wikidata P31 |

## Usage

### Load from HuggingFace

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("undertheseanlp/UVW-2026")

# Access the data
train = dataset["train"]
print(train[0])

# Filter high-quality articles (score >= 7)
high_quality = train.filter(lambda x: x["quality_score"] >= 7)

# Filter by category
people = train.filter(lambda x: x["main_category"] == "người")
```

## Statistics

| Metric | Value |
|--------|-------|
| Total articles | 1,118,224 |
| Train split | 894,579 (80%) |
| Validation split | 111,822 (10%) |
| Test split | 111,823 (10%) |
| Wikidata coverage | 99.4% |
| Category coverage | 97.0% |
| Unique categories | 11,549 |
| Avg. characters | 1,190 |
| Avg. sentences | 10 |

## Quality Score Distribution

| Score | Count | Percentage |
|-------|------:|----------:|
| 1 | 134 | 0.0% |
| 2 | 376 | 0.0% |
| 3 | 28,267 | 2.5% |
| 4 | 607,081 | 54.3% |
| 5 | 208,304 | 18.6% |
| 6 | 134,385 | 12.0% |
| 7 | 70,345 | 6.3% |
| 8 | 57,054 | 5.1% |
| 9 | 9,649 | 0.9% |
| 10 | 2,629 | 0.2% |

## Top Categories

| Category (Vietnamese) | Count | Percentage |
|----------------------|------:|----------:|
| đơn vị phân loại (taxon) | 618,281 | 55.3% |
| người (human) | 78,191 | 7.0% |
| xã của Pháp | 35,635 | 3.2% |
| khu định cư | 20,276 | 1.8% |
| tiểu hành tinh | 17,891 | 1.6% |
| xã của Việt Nam | 7,088 | 0.6% |
| đô thị của Ý | 6,700 | 0.6% |
| trang định hướng Wikimedia | 6,202 | 0.6% |

## Quality Scoring

Articles are scored 1-10 based on:

| Component | Weight | Criteria |
|-----------|--------|----------|
| Length | 40% | Character count (200 - 100,000 optimal) |
| Sentences | 30% | Sentence count (3 - 1,000 optimal) |
| Density | 30% | Avg sentence length (80-150 chars optimal) |
| Wikidata bonus | +0.5 | Has wikidata_id |
| Category bonus | +0.5 | Has main_category |
| Markup penalty | -1 to -3 | Remaining Wikipedia markup |

## Data Processing Pipeline

1. **Download** - Vietnamese Wikipedia XML dump from Wikimedia
2. **Extract** - Parse XML and extract article content
3. **Clean** - Remove Wikipedia markup (templates, refs, links, tables)
4. **Normalize** - Apply Unicode NFC normalization
5. **Score** - Calculate quality metrics for each article
6. **Enrich** - Add Wikidata IDs and categories via Wikidata API
7. **Filter** - Remove special pages, redirects, disambiguation, short articles
8. **Split** - Create train/validation/test splits (80/10/10, seed=42)

## Removed Content

- Wikipedia templates (`{{...}}`)
- References and citations (`<ref>...</ref>`)
- HTML tags and comments
- Category links (`[[Thể loại:...]]`)
- File/image links (`[[Tập tin:...]]`, `[[File:...]]`)
- Interwiki links
- Tables (`{| ... |}`)
- Infoboxes and navigation templates

## Sample Articles

| Title | Category | Quality | Wikidata |
|-------|----------|---------|----------|
| Việt Nam | quốc gia có chủ quyền | 9 | Q881 |
| Hà Nội | thủ đô | 9 | Q1858 |
| Nguyễn Du | người | 8 | Q332972 |
| Sông Mê Kông | sông | 8 | Q3056359 |
| Phở | món ăn | 7 | Q217666 |

## Citation

```bibtex
@dataset{uvw2026,
  title = {UVW 2026: Underthesea Vietnamese Wikipedia Dataset},
  author = {Underthesea NLP},
  year = {2026},
  publisher = {Hugging Face},
  url = {https://huggingface.co/datasets/undertheseanlp/UVW-2026},
  note = {Vietnamese Wikipedia articles enriched with Wikidata metadata}
}
```

## References

- [HuggingFace Dataset](https://huggingface.co/datasets/undertheseanlp/UVW-2026)
- [GitHub Repository](https://github.com/undertheseanlp/UVW-2026)
- [GitHub Issue #896](https://github.com/undertheseanlp/underthesea/issues/896)
- [Vietnamese Wikipedia](https://vi.wikipedia.org)
- [Wikidata](https://www.wikidata.org)
