# UTS_VLC_2026

Vietnamese Legal Corpus (2026 Edition)

Vietnam's system of legal documents - Updated with laws passed through 2022-2025 (15th National Assembly).

## HuggingFace Dataset

**Dataset:** [undertheseanlp/UTS_VLC](https://huggingface.co/datasets/undertheseanlp/UTS_VLC)

## Splits

| Split | Documents | Description |
|-------|-----------|-------------|
| 2021  | 110       | Original corpus with full text |
| 2023  | 208       | Expanded corpus with processed text |
| 2026  | 318       | Latest corpus with 2022-2025 laws |

## Features

| Feature | Type | Description |
|---------|------|-------------|
| id | string | Document identifier (slug) |
| filename | string | Source filename |
| title | string | Vietnamese title |
| type | string | Document type: "code", "law", or "constitution" |
| content | string | Full text content |
| content_length | int32 | Character count |

## Usage

### Load from HuggingFace

```python
from datasets import load_dataset

# Load all splits
ds = load_dataset("undertheseanlp/UTS_VLC")

# Load specific split
ds_2026 = load_dataset("undertheseanlp/UTS_VLC", split="2026")

# Filter by type
codes = ds_2026.filter(lambda x: x["type"] == "code")
laws = ds_2026.filter(lambda x: x["type"] == "law")

# Search by title
land_law = ds_2026.filter(lambda x: "Đất đai" in x["title"])
```

### Load from Underthesea

```python
from underthesea import download_data

# Download the corpus
download_data("CP_Vietnamese_VLC_2026")
```

## Statistics

| Metric | 2021 | 2023 | 2026 |
|--------|------|------|------|
| Documents | 110 | 208 | 318 |
| Total Characters | ~14M | ~21M | ~24M |
| Codes | 6 | 6 | 6 |
| Laws | 104 | 202 | 312 |

## Document Types

- **Constitution (Hiến pháp)**: 1 document (2013 Constitution)
- **Codes (Bộ luật)**: 6 major codes
  - Civil Code 2015 (Bộ luật Dân sự)
  - Civil Procedure Code 2015 (Bộ luật Tố tụng Dân sự)
  - Criminal Code 2015 (Bộ luật Hình sự)
  - Criminal Procedure Code 2015 (Bộ luật Tố tụng Hình sự)
  - Maritime Code 2015 (Bộ luật Hàng hải)
  - Labor Code 2019 (Bộ luật Lao động)
- **Laws (Luật)**: 300+ laws covering various domains

## Recent Laws (2022-2025)

The 2026 split includes laws passed by the 15th National Assembly:

- **2022**: 8 laws (Inspection, Anti-Money Laundering, IP, Insurance, Cinema, etc.)
- **2023**: 12 laws (E-Transactions, Consumer Protection, Bidding, Telecoms, Housing, etc.)
- **2024**: 24 laws (Land, Credit Institutions, Roads, Data, Electricity, Social Insurance, etc.)
- **2025**: 50+ laws including landmark legislation:
  - **AI Law** (Luật Trí tuệ nhân tạo) - First AI law in Vietnam
  - **E-commerce Law** (Luật Thương mại điện tử)
  - **Digital Transformation Law** (Luật Chuyển đổi số)
  - **Personal Data Protection Law** (Luật Bảo vệ dữ liệu cá nhân)

## References

- [HuggingFace Dataset](https://huggingface.co/datasets/undertheseanlp/UTS_VLC)
- [Vietnam's System of Legal Documents](https://vietnamlawmagazine.vn/vietnams-system-of-legal-documents-5017.html)
- [Thu Vien Phap Luat](https://thuvienphapluat.vn/) - Official Vietnamese legal document database
- [National Assembly of Vietnam](https://quochoi.vn/)
