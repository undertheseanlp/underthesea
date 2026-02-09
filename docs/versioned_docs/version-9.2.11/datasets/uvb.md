# UVB

Underthesea Vietnamese Books Dataset (2026 Edition)

A collection of 447 Vietnamese books with full text content and Goodreads metadata for NLP research.

## HuggingFace Dataset

**Dataset:** [undertheseanlp/UVB-v0.1](https://huggingface.co/datasets/undertheseanlp/UVB-v0.1)

## Features

| Feature | Type | Description |
|---------|------|-------------|
| id | string | Unique identifier (e.g., vn_000001) |
| title | string | Book title |
| author | string | Author name |
| content | string | Full text content of the book |
| genres | list[string] | Book genres from Goodreads |
| first_publish | string | First publication year |
| goodreads_id | string | Goodreads book ID |
| goodreads_url | string | Goodreads URL |
| goodreads_rating | float | Goodreads rating (1-5) |
| goodreads_num_ratings | int | Number of ratings |

## Usage

### Load from HuggingFace

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("undertheseanlp/UVB-v0.1")

# Access the data
for item in dataset["train"]:
    print(f"Title: {item['title']}")
    print(f"Author: {item['author']}")
    print(f"Content: {item['content'][:200]}...")
    print(f"Genres: {item['genres']}")
    print(f"First publish: {item['first_publish']}")
    break

# Filter by genre
fiction = dataset["train"].filter(lambda x: "Fiction" in (x.get("genres") or []))
non_fiction = dataset["train"].filter(lambda x: "Non Fiction" in (x.get("genres") or []))
```

## Statistics

| Metric | Value |
|--------|-------|
| Total books | 447 |
| Books with genres | 230 (51.5%) |
| Books with publication year | 421 (94.2%) |
| Total size | ~209 MB |

## Top Genres

| Genre | Count |
|-------|-------|
| Non Fiction | 76 |
| Fiction | 62 |
| Romance | 37 |
| Classics | 30 |
| Novels | 27 |
| Philosophy | 25 |
| Self Help | 25 |
| Literature | 24 |
| History | 22 |
| Childrens | 20 |

## Publication Year Distribution

| Period | Count |
|--------|-------|
| Before 1900 | 6 |
| 1900-1950 | 12 |
| 1951-1980 | 38 |
| 1981-2000 | 82 |
| 2001-2010 | 134 |
| 2011+ | 149 |

## Source Data

- **Vietnamese books:** [tmnam20/Vietnamese-Book-Corpus](https://huggingface.co/datasets/tmnam20/Vietnamese-Book-Corpus)
- **Goodreads metadata:** [BrightData/Goodreads-Books](https://huggingface.co/datasets/BrightData/Goodreads-Books)

## Processing Scripts

Scripts included in the dataset repository:

- `scripts/map_goodreads.py` - Map Vietnamese books to Goodreads entries using fuzzy matching
- `scripts/add_genres.py` - Fetch genres from Goodreads pages
- `scripts/add_publish_date.py` - Fetch first publication year from Goodreads

## Sample Books

### Fiction

| Title | Author | Year | Rating |
|-------|--------|------|--------|
| THE COMPLETE SHERLOCK HOLMES | Arthur Conan Doyle | 1983 | 4.01 |
| LĨNH NAM CHÍCH QUÁI | Trần Thế Pháp | 1492 | 3.80 |
| 1Q84 | Haruki Murakami | 2009 | 4.10 |
| David Copperfield | Charles Dickens | 2009 | 4.17 |
| SỐNG MÒN | Nam Cao | 2008 | 4.23 |

### Non Fiction

| Title | Author | Year | Rating |
|-------|--------|------|--------|
| THE TIBETAN BOOK OF LIVING AND DYING | Sogyal Rinpoche | 1992 | 4.21 |
| VIỆT NAM PHONG TỤC | Phan Kế Bính | 1972 | 4.09 |
| TỰ HỌC MỘT NHU CẦU CỦA THỜI ĐẠI | Nguyễn Hiến Lê | 2007 | 4.27 |
| THE INFORMATION | James Gleick | 2011 | 4.03 |
| SỬ KÝ TƯ MÃ THIÊN | Tư Mã Thiên | - | 4.21 |

## Citation

```bibtex
@misc{uvb_dataset,
  title={UVB: Underthesea Vietnamese Books Dataset},
  author={Underthesea NLP},
  year={2026},
  publisher={HuggingFace},
  url={https://huggingface.co/datasets/undertheseanlp/UVB-v0.1}
}
```

## References

- [HuggingFace Dataset](https://huggingface.co/datasets/undertheseanlp/UVB-v0.1)
- [GitHub Issue #720](https://github.com/undertheseanlp/underthesea/issues/720)
- [Goodreads](https://www.goodreads.com/)
