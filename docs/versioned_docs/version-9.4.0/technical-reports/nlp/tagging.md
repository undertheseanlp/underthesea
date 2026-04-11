# Tagging

## Overview

This report covers four sequence labeling pipelines in Underthesea: **Word Tokenization**, **POS Tagging**, **Chunking**, and **Dependency Parsing**. These pipelines form the core syntactic analysis chain for Vietnamese text.

```
Vietnamese Text
  → Word Tokenization (CRF)
    → POS Tagging (CRF)
      → Chunking (CRF)
      → Dependency Parsing (Biaffine Neural Parser)
```

---

## Word Tokenization

The word tokenization module performs Vietnamese word segmentation using a Conditional Random Field (CRF) model. Vietnamese word segmentation is challenging because spaces don't always indicate word boundaries — multi-syllable words like "Việt Nam" or "khởi nghiệp" are written with spaces between syllables.

**Author:** Vu Anh
**Model:** CRF trained on VLSP2013 dataset (checkpoint: `20230727`)
**Integrated since:** underthesea v6.6.0

### Architecture

```
Word Tokenization Pipeline
├── Text Input
│   └── Raw Vietnamese text
├── Regex Tokenization
│   └── Split by whitespace and punctuation
├── Feature Extraction
│   ├── Unigram features: T[-2], T[-1], T[0], T[1], T[2]
│   ├── Bigram features: T[-2,-1], T[-1,0], T[0,1], T[1,2]
│   ├── Lowercase features
│   ├── Case features (isTitle, isDigit)
│   └── Dictionary features (is_in_dict)
├── CRF Model (FastCRFSequenceTagger)
│   └── BIO sequence labeling
└── Output
    └── List of segmented words
```

### Feature Engineering

| Feature Type | Features |
|-------------|----------|
| Unigram | `T[-2]`, `T[-1]`, `T[0]`, `T[1]`, `T[2]` |
| Bigram | `T[-2,-1]`, `T[-1,0]`, `T[0,1]`, `T[1,2]`, `T[-2,0]`, `T[-1,1]`, `T[0,2]` |
| Lowercase Unigram | `T[-2].lower`, `T[-1].lower`, `T[0].lower`, `T[1].lower`, `T[2].lower` |
| Lowercase Bigram | `T[-2,-1].lower`, `T[-1,0].lower`, `T[0,1].lower`, `T[1,2].lower` |
| Is Digit | `T[-1].isdigit`, `T[0].isdigit`, `T[1].isdigit` |
| Is Title | `T[-2].istitle`, `T[-1].istitle`, `T[0].istitle`, `T[1].istitle`, `T[2].istitle`, `T[0,1].istitle`, `T[0,2].istitle` |
| Is in Dictionary | `T[-2].is_in_dict`, `T[-1].is_in_dict`, `T[0].is_in_dict`, `T[1].is_in_dict`, `T[2].is_in_dict`, and bigram/trigram dictionary lookups |

### Performance

| Dataset | Model | F1 Score |
|:--------|:------|--------:|
| UTS_WTK (1.0.0) | CRF | 0.977 |
| VLSP2013_WTK | CRF | 0.973 |

### Usage

```python
from underthesea import word_tokenize

text = "Chàng trai 9X Quảng Trị khởi nghiệp từ nấm sò"
words = word_tokenize(text)
# ["Chàng trai", "9X", "Quảng Trị", "khởi nghiệp", "từ", "nấm", "sò"]

# Text format output
word_tokenize(text, format="text")
# "Chàng_trai 9X Quảng_Trị khởi_nghiệp từ nấm sò"

# Fixed words
word_tokenize("Sinh viên đại học Bách Khoa", fixed_words=["đại học Bách Khoa"])
# ["Sinh viên", "đại học Bách Khoa"]
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sentence` | str | required | Text to tokenize |
| `format` | str | None | Output format — `None` for list, `"text"` for underscore-joined string |
| `use_token_normalize` | bool | True | Whether to normalize tokens |
| `fixed_words` | list | None | Words that should not be split |

---

## POS Tagging

The POS tagging module provides Vietnamese Part-of-Speech tagging using the TRE-1 model, a CRF-based tagger trained on the Universal Dependencies Dataset (UDD-v0.1).

**Model:** [undertheseanlp/tre-1](https://huggingface.co/undertheseanlp/tre-1)
**License:** Apache 2.0

### Architecture

```
TRE-1 Pipeline
├── Text Input
│   └── Pre-tokenized Vietnamese text
├── Feature Extraction
│   ├── Current Token Features
│   │   ├── Word form, lowercase form
│   │   ├── Prefix/suffix (2-3 chars)
│   │   └── Character type checks
│   ├── Context Features (previous/next 1-2 tokens)
│   ├── Bigram Features
│   └── Dictionary Features
├── CRF Classification (python-crfsuite)
└── Output
    └── UPOS tags for each token
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Algorithm | CRF (python-crfsuite) |
| L1 regularization (c1) | 1.0 |
| L2 regularization (c2) | 1e-3 |
| Max iterations | 100 |
| Training data | undertheseanlp/UDD-v0.1 |
| Tagset | Universal POS tags (UPOS) |

### POS Tag Set

| Tag | Description | Example |
|-----|-------------|---------|
| `N` | Noun | chợ, thịt, chó |
| `Np` | Proper noun | Sài Gòn, Việt Nam |
| `V` | Verb | bị, truy quét |
| `A` | Adjective | nổi tiếng, đẹp |
| `P` | Pronoun | tôi, bạn, nó |
| `R` | Adverb | rất, đang, sẽ |
| `E` | Preposition | ở, trong, trên |
| `C` | Conjunction | và, hoặc, nhưng |
| `M` | Number | một, hai, ba |
| `L` | Determiner | các, những, mọi |
| `X` | Unknown | — |
| `CH` | Punctuation | . , ? ! |

### Performance

| Metric | Score |
|--------|-------|
| Accuracy | ~94% |
| F1 (macro) | ~90% |
| F1 (weighted) | ~94% |

### Usage

```python
from underthesea import pos_tag

text = "Chợ thịt chó nổi tiếng ở Sài Gòn bị truy quét"
tagged = pos_tag(text)
# [('Chợ', 'N'), ('thịt', 'N'), ('chó', 'N'), ('nổi tiếng', 'A'),
#  ('ở', 'E'), ('Sài Gòn', 'Np'), ('bị', 'V'), ('truy quét', 'V')]
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sentence` | str | required | Text to tag |
| `format` | str | None | Output format |
| `model` | str | None | Path to custom model |

---

## Chunking

The chunking module performs shallow parsing for Vietnamese text, grouping words into meaningful phrases such as noun phrases (NP), verb phrases (VP), adjective phrases (AP), and prepositional phrases (PP). Chunking is built on top of word segmentation and POS tagging.

### Architecture

```
Chunking Pipeline
├── Text Input
│   └── Raw Vietnamese text
├── Word Tokenization
│   └── word_tokenize()
├── POS Tagging
│   └── pos_tag()
├── Feature Extraction
│   └── Token + POS features
├── CRF Model
│   └── BIO chunk labeling
└── Output
    └── List of (word, POS tag, chunk tag) tuples
```

### Chunk Tags

The module uses BIO (Begin-Inside-Outside) tagging format:

| Tag | Description | Example |
|-----|-------------|---------|
| `B-NP` | Beginning of Noun Phrase | Bác sĩ, bệnh nhân |
| `I-NP` | Inside Noun Phrase | (continuation) |
| `B-VP` | Beginning of Verb Phrase | báo, bị |
| `I-VP` | Inside Verb Phrase | (continuation) |
| `B-AP` | Beginning of Adjective Phrase | thản nhiên |
| `I-AP` | Inside Adjective Phrase | (continuation) |
| `B-PP` | Beginning of Prepositional Phrase | ở, trong |
| `I-PP` | Inside Prepositional Phrase | (continuation) |
| `O` | Outside any chunk | — |

### Usage

```python
from underthesea import chunk

text = "Bác sĩ bây giờ có thể thản nhiên báo tin bệnh nhân bị ung thư?"
result = chunk(text)
# [('Bác sĩ', 'N', 'B-NP'),
#  ('bây giờ', 'P', 'B-NP'),
#  ('thản nhiên', 'A', 'B-AP'),
#  ('báo', 'V', 'B-VP'),
#  ('bệnh nhân', 'N', 'B-NP'),
#  ('bị', 'V', 'B-VP'),
#  ('ung thư', 'N', 'B-NP')]
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sentence` | str | required | Text to chunk |
| `format` | str | None | Output format |

---

## Dependency Parsing

The dependency parsing module provides Vietnamese dependency parsing using a Biaffine Neural Dependency Parser based on the architecture proposed by Dozat and Manning (2017).

### Architecture

```
DependencyParser
├── Embeddings
│   ├── word_embed: nn.Embedding
│   ├── feat_embed: CharLSTM | BertEmbedding | nn.Embedding
│   └── embed_dropout: IndependentDropout
├── Encoder
│   ├── lstm: BiLSTM (3-layer bidirectional)
│   └── lstm_dropout: SharedDropout
├── MLP Layers
│   ├── mlp_arc_d/h: MLP (arc head/dependent)
│   └── mlp_rel_d/h: MLP (relation head/dependent)
└── Biaffine Attention
    ├── arc_attn: Biaffine (arc scoring)
    └── rel_attn: Biaffine (relation scoring)
```

### Default Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| n_embed | 50 | Word embedding dimension |
| n_feat_embed | 100 | Feature embedding dimension |
| n_char_embed | 50 | Character embedding dimension |
| n_lstm_hidden | 400 | BiLSTM hidden size |
| n_lstm_layers | 3 | Number of BiLSTM layers |
| n_mlp_arc | 500 | Arc MLP output size |
| n_mlp_rel | 100 | Relation MLP output size |
| embed_dropout | 0.33 | Embedding dropout |
| lstm_dropout | 0.33 | LSTM dropout |
| mlp_dropout | 0.33 | MLP dropout |

### Performance

| Model | UAS | LAS | UCM | LCM |
|-------|-----|-----|-----|-----|
| MaltParser (baseline) | 75.41% | 66.11% | - | - |
| Biaffine Attention (v1) | 87.28% | 72.63% | 30.67% | 6.98% |
| **vi-dp-v1a1 (current)** | **87.10%** | **80.00%** | - | - |

| Metric | Description |
|--------|-------------|
| **UAS** | Unlabeled Attachment Score — % tokens with correct head |
| **LAS** | Labeled Attachment Score — % tokens with correct head AND relation |
| **UCM** | Unlabeled Complete Match — % sentences with ALL heads correct |
| **LCM** | Labeled Complete Match — % sentences with ALL heads and labels correct |

### Dependency Relations

| Relation | Description | Example |
|----------|-------------|---------|
| root | Root of sentence | Main verb |
| nsubj | Nominal subject | Tôi → ăn |
| obj | Direct object | cơm ← ăn |
| cop | Copula | là → noun |
| compound | Compound | Việt Nam ← sinh viên |
| nmod | Nominal modifier | của relations |
| amod | Adjectival modifier | đẹp → noun |
| advmod | Adverbial modifier | rất → adj |
| punct | Punctuation | . , ! ? |

### Usage

```python
from underthesea import dependency_parse

result = dependency_parse("Tôi là sinh viên Việt Nam")
# [('Tôi', 3, 'nsubj'), ('là', 3, 'cop'), ('sinh viên', 0, 'root'), ('Việt Nam', 3, 'compound')]
```

### Visualization

```python
from underthesea.pipeline.dependency_parse import render, display

svg = render("Tôi yêu Việt Nam")
display("Tôi yêu Việt Nam")  # In Jupyter notebook
```

### Training

```python
from underthesea.datasets.vlsp2020_dp import VLSP2020_DP_SAMPLE
from underthesea.models.dependency_parser import DependencyParser
from underthesea.modules.embeddings import FieldEmbeddings, CharacterEmbeddings
from underthesea.trainers.dependency_parser_trainer import DependencyParserTrainer

corpus = VLSP2020_DP_SAMPLE()
embeddings = [FieldEmbeddings(), CharacterEmbeddings()]
parser = DependencyParser(embeddings=embeddings, init_pre_train=True)

trainer = DependencyParserTrainer(parser, corpus)
trainer.train(base_path="path/to/save/model", max_epochs=100, lr=2e-3, mu=0.9, batch_size=5000)
```

---

## References

1. Lafferty, J., McCallum, A., & Pereira, F. (2001). Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data. ICML.
2. Dozat, T., & Manning, C. D. (2017). Deep Biaffine Attention for Neural Dependency Parsing. ICLR 2017.
3. VLSP 2020 Shared Task: Vietnamese Dependency Parsing
4. [undertheseanlp/tre-1](https://huggingface.co/undertheseanlp/tre-1)
5. [Universal Dependencies](https://universaldependencies.org/)
6. [Underthesea GitHub Repository](https://github.com/undertheseanlp/underthesea)

## Changelog

### Version 9.1.3 (PR #871)
- Added PyTorch v2.0+ support for dependency parsing
- Fixed deprecated API usage
- Added training CI test (train-dep)

### Version 6.6.0
- Integrated CRF word tokenization model (checkpoint: `20230727`)
