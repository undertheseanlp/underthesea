# UUD-v0.1

Universal Dependency Dataset for Vietnamese

Vietnamese Universal Dependency dataset following [Universal Dependencies](https://universaldependencies.org/) annotation guidelines. Machine-generated using Underthesea NLP toolkit.

## HuggingFace Dataset

**Dataset:** [undertheseanlp/UDD-v0.1](https://huggingface.co/datasets/undertheseanlp/UDD-v0.1)

## Summary

| Metric | Value |
|--------|-------|
| Sentences | 3,000 |
| Tokens | 64,814 |
| Avg sentence length | 21.60 |
| Max sentence length | 65 |
| Avg tree depth | 6.77 |
| Max tree depth | 21 |
| Source | Vietnamese Legal Corpus (UTS_VLC) |
| Validation | 0 errors (passes all UD checks) |

## Features

| Field | Type | Description |
|-------|------|-------------|
| `sent_id` | string | Sentence identifier |
| `text` | string | Original sentence text |
| `tokens` | list[string] | Tokenized words |
| `lemmas` | list[string] | Lemmatized forms |
| `upos` | list[string] | Universal POS tags |
| `xpos` | list[string] | Language-specific POS tags |
| `feats` | list[string] | Morphological features |
| `head` | list[string] | Head token indices |
| `deprel` | list[string] | Dependency relations |
| `deps` | list[string] | Enhanced dependencies |
| `misc` | list[string] | Miscellaneous annotations |

## Usage

### Load from HuggingFace

```python
from datasets import load_dataset

dataset = load_dataset("undertheseanlp/UDD-v0.1")
print(dataset["train"][0])
```

### Clone and Run Scripts

```bash
# Clone the dataset repository
git clone https://huggingface.co/datasets/undertheseanlp/UDD-v0.1
cd UDD-v0.1

# Install dependencies with uv
uv sync

# Fetch sentences from UTS_VLC
uv run python scripts/fetch_data.py

# Convert to UD format
uv run python scripts/convert_to_ud.py

# Run statistics
uv run python scripts/statistics.py

# Upload to HuggingFace
uv run python scripts/upload_to_hf.py
```

## UPOS Distribution

| Tag | Count | Percent |
|-----|------:|--------:|
| NOUN | 21,599 | 33.32% |
| VERB | 15,793 | 24.37% |
| PUNCT | 6,391 | 9.86% |
| ADP | 6,309 | 9.73% |
| CCONJ | 2,942 | 4.54% |
| AUX | 2,665 | 4.11% |
| ADV | 2,518 | 3.88% |
| ADJ | 2,254 | 3.48% |
| NUM | 1,444 | 2.23% |
| DET | 1,350 | 2.08% |
| PRON | 1,128 | 1.74% |
| PROPN | 318 | 0.49% |

## Top Dependency Relations

| Relation | Count | Percent |
|----------|------:|--------:|
| obj | 6,448 | 9.95% |
| punct | 6,391 | 9.86% |
| nmod | 5,870 | 9.06% |
| case | 5,853 | 9.03% |
| conj | 4,920 | 7.59% |
| compound | 3,314 | 5.11% |
| root | 3,000 | 4.63% |
| acl:subj | 2,889 | 4.46% |
| nsubj | 2,869 | 4.43% |
| nmod:poss | 1,656 | 2.56% |

## Root UPOS Distribution

| UPOS | Count | Percent |
|------|------:|--------:|
| VERB | 2,220 | 74.00% |
| NOUN | 639 | 21.30% |
| ADJ | 63 | 2.10% |
| ADP | 41 | 1.37% |
| AUX | 17 | 0.57% |
| PROPN | 14 | 0.47% |

## Scripts

The dataset repository includes scripts for data processing:

| Script | Description |
|--------|-------------|
| `scripts/fetch_data.py` | Fetch sentences from UTS_VLC corpus |
| `scripts/convert_to_ud.py` | Convert to UD format with syntax fixes |
| `scripts/statistics.py` | Compute dataset statistics |
| `scripts/upload_to_hf.py` | Upload to HuggingFace Hub |

## Related Datasets

Other Vietnamese dependency treebanks include: [UD_Vietnamese-VTB](https://universaldependencies.org/treebanks/vi_vtb/index.html) - the official Vietnamese treebank in Universal Dependencies, converted from VietTreebank constituent treebank created by VLSP project (UD v1.4+); [VnDT](https://github.com/datquocnguyen/VnDT) - the first Vietnamese dependency treebank with 10,200 sentences automatically converted from VietTreebank and manually edited (2013, revised 2016); [BKTreebank](https://aclanthology.org/L18-1341/) - a dependency treebank with 6,900 sentences featuring custom POS tagset and dependency relations designed specifically for Vietnamese linguistic characteristics (LREC 2018); and [VLSP shared task data](https://vlsp.org.vn/vlsp2020/eval/udp) - training and test data from VLSP dependency parsing shared tasks with 8,152 sentences following Universal Dependencies v2 annotation scheme (2019-2020), where top models achieved 76.27% LAS and 84.65% UAS using PhoBERT+ELMO/Biaffine architecture.

| Dataset | Sentences | Tokens | Domain | Annotation | Format | Available |
|---------|----------:|-------:|--------|------------|--------|-----------|
| **UUD-v0.1** | 3,000 | 64,814 | Legal | Machine-generated | CoNLL-U | [HuggingFace](https://huggingface.co/datasets/undertheseanlp/UDD-v0.1) |
| UD_Vietnamese-VTB | 3,323 | 58,069 | News (Tuoi Tre) | Manual | CoNLL-U | [UD](https://universaldependencies.org/treebanks/vi_vtb/), [GitHub](https://github.com/UniversalDependencies/UD_Vietnamese-VTB), [HuggingFace](https://huggingface.co/datasets/commul/universal_dependencies/viewer/vi_vtb?row=0) |
| VnDT | 10,200 | ~170K | News (Tuoi Tre) | Semi-automatic | CoNLL | [GitHub](https://github.com/datquocnguyen/VnDT) |
| BKTreebank | 6,900 | ~115K | Mixed | Manual | CoNLL | [ACL](https://aclanthology.org/L18-1341/) |
| VLSP 2020 | 8,152 | ~140K | Mixed | Manual | CoNLL-U | [VLSP](https://vlsp.org.vn/vlsp2020/eval/udp) |

## References

- [HuggingFace Dataset](https://huggingface.co/datasets/undertheseanlp/UDD-v0.1)
- [Universal Dependencies](https://universaldependencies.org/)
- [UD Vietnamese Guidelines](https://universaldependencies.org/vi/)
- [UTS_VLC Dataset](uts-vlc.md) - Source corpus
- [VLSP - Vietnamese Language and Speech Processing](https://vlsp.org.vn/)
