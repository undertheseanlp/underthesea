# Dependency Parsing Technical Report

## Overview

The dependency parsing module in Underthesea provides Vietnamese dependency parsing using a Biaffine Neural Dependency Parser. This document describes the architecture, training process, and recent updates for PyTorch v2 compatibility.

## Architecture

### Model: Biaffine Dependency Parser

The implementation is based on the Deep Biaffine Attention architecture proposed by Dozat and Manning (2017).

**Reference:** Timothy Dozat and Christopher D. Manning. 2017. [Deep Biaffine Attention for Neural Dependency Parsing](https://openreview.net/forum?id=Hk95PK9le)

### Components

```
DependencyParser
├── Embeddings
│   ├── word_embed: nn.Embedding (word embeddings)
│   ├── feat_embed: CharLSTM | BertEmbedding | nn.Embedding (feature embeddings)
│   └── embed_dropout: IndependentDropout
├── Encoder
│   ├── lstm: BiLSTM (3-layer bidirectional LSTM)
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

## File Structure

```
underthesea/
├── pipeline/dependency_parse/
│   ├── __init__.py          # Main API: dependency_parse()
│   └── visualize.py         # Visualization utilities
├── models/
│   └── dependency_parser.py # DependencyParser model
├── modules/
│   ├── base.py              # BiLSTM, MLP, Biaffine, CharLSTM
│   ├── bert.py              # BERT embeddings
│   └── embeddings.py        # FieldEmbeddings, CharacterEmbeddings
├── trainers/
│   └── dependency_parser_trainer.py
├── transforms/
│   └── conll.py             # CoNLL format handling
└── utils/
    ├── sp_field.py          # Field, SubwordField
    ├── sp_vocab.py          # Vocabulary
    ├── sp_data.py           # Dataset
    ├── sp_metric.py         # AttachmentMetric (UAS, LAS)
    └── sp_alg.py            # Eisner, MST algorithms
```

## Performance

### Benchmark Results on VLSP2020-DP

| Model | UAS | LAS | UCM | LCM |
|-------|-----|-----|-----|-----|
| MaltParser (baseline) | 75.41% | 66.11% | - | - |
| Biaffine Attention (v1) | 87.28% | 72.63% | 30.67% | 6.98% |
| **vi-dp-v1a1 (current)** | **87.10%** | **80.00%** | - | - |

### Metrics Description

| Metric | Description |
|--------|-------------|
| **UAS** (Unlabeled Attachment Score) | Percentage of tokens with correct head |
| **LAS** (Labeled Attachment Score) | Percentage of tokens with correct head AND relation label |
| **UCM** (Unlabeled Complete Match) | Percentage of sentences with ALL heads correct |
| **LCM** (Labeled Complete Match) | Percentage of sentences with ALL heads and labels correct |

### Training History

**MaltParser Baseline (2020)**
```
Metric     | Precision |    Recall |  F1 Score
-----------+-----------+-----------+-----------
UAS        |     75.41 |     75.41 |     75.41
LAS        |     66.11 |     66.11 |     66.11
CLAS       |     62.70 |     62.17 |     62.43
```

**Biaffine Attention v1 (240 epochs, 2020)**
```
2020-11-29 23:05:58 Epoch 240 saved
dev:   - UCM: 30.67% LCM:  6.98% UAS: 87.28% LAS: 72.63%
test:  - UCM: 30.67% LCM:  6.98% UAS: 87.28% LAS: 72.63%
Training time: 33m 46s, 5.96s/epoch
```

### Comparison with Other Methods

| Method | Year | UAS | LAS | Notes |
|--------|------|-----|-----|-------|
| MST Parser (N.L. Minh et al.) | 2008 | - | - | 450 sentences corpus |
| MaltParser (N.T. Luong et al.) | 2013 | - | - | Vietnamese treebank |
| MaltParser (baseline) | 2020 | 75.41% | 66.11% | VLSP2020-DP |
| Biaffine Attention | 2020 | 87.28% | 72.63% | VLSP2020-DP |
| NLP@UIT (VLSP2019 winner) | 2019 | - | - | Ensemble model |

## Pretrained Models

| Model | Description | UAS | LAS | Source |
|-------|-------------|-----|-----|--------|
| vi-dp-v1a1 | Default Vietnamese dependency parser | 87.10% | 80.00% | Trained on VLSP2020-DP |

Models are automatically downloaded from the Underthesea resources repository.

## Training

### Dataset

- **VLSP2020-DP**: Vietnamese dependency parsing dataset from VLSP 2020 shared task
- **VLSP2020_DP_SAMPLE**: Small sample dataset for testing (auto-downloadable)

### Training Script

```python
from underthesea.datasets.vlsp2020_dp import VLSP2020_DP_SAMPLE
from underthesea.models.dependency_parser import DependencyParser
from underthesea.modules.embeddings import FieldEmbeddings, CharacterEmbeddings
from underthesea.trainers.dependency_parser_trainer import DependencyParserTrainer

# Load corpus
corpus = VLSP2020_DP_SAMPLE()

# Initialize parser
embeddings = [FieldEmbeddings(), CharacterEmbeddings()]
parser = DependencyParser(embeddings=embeddings, init_pre_train=True)

# Train
trainer = DependencyParserTrainer(parser, corpus)
trainer.train(
    base_path="path/to/save/model",
    max_epochs=100,
    lr=2e-3,
    mu=0.9,
    batch_size=5000
)
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| lr | 2e-3 | Learning rate |
| mu | 0.9 | Adam beta1 |
| nu | 0.9 | Adam beta2 |
| epsilon | 1e-12 | Adam epsilon |
| clip | 5.0 | Gradient clipping |
| decay | 0.75 | LR decay rate |
| decay_steps | 5000 | LR decay steps |
| patience | 100 | Early stopping patience |
| batch_size | 5000 | Batch size (tokens) |
| buckets | 1000 | Number of length buckets |

## Usage

### Basic Usage

```python
from underthesea import dependency_parse

result = dependency_parse("Tôi là sinh viên Việt Nam")
# Output: [('Tôi', 3, 'nsubj'), ('là', 3, 'cop'), ('sinh viên', 0, 'root'), ('Việt Nam', 3, 'compound')]
```

### Output Format

Each tuple contains:
- **word**: The token
- **head**: Index of the head token (0 = root)
- **relation**: Dependency relation label

### Visualization

```python
from underthesea.pipeline.dependency_parse import render, display

# Render as SVG
svg = render("Tôi yêu Việt Nam")

# Display in notebook
display("Tôi yêu Việt Nam")
```

## PyTorch v2 Compatibility (GH-706)

### Issue

The dependency parsing module was incompatible with PyTorch v2.0+ due to:
1. Deprecated `weights_only` parameter in `torch.load`
2. Deprecated `apply_permutation` function
3. Deprecated non-tuple sequence indexing

### Fixes Applied

#### 1. torch.load with weights_only=False

PyTorch 2.0+ requires explicit `weights_only` parameter. Since models contain pickled Python objects (transforms, vocabularies), we use `weights_only=False`:

```python
# Before
state = torch.load(path)

# After
state = torch.load(path, map_location='cpu', weights_only=False)
```

**Files modified:**
- `underthesea/models/dependency_parser.py`
- `underthesea/models/model.py`
- `underthesea/modules/nn.py`

#### 2. Replace deprecated apply_permutation

```python
# Before
from torch.nn.modules.rnn import apply_permutation
h = apply_permutation(hx[0], permutation)

# After
h = hx[0].index_select(0, permutation)
```

**File modified:** `underthesea/modules/base.py`

#### 3. Fix non-tuple sequence indexing

```python
# Before
out_tensor[i][[slice(0, i) for i in tensor.size()]] = tensor

# After
out_tensor[i][tuple(slice(0, s) for s in tensor.size())] = tensor
```

**File modified:** `underthesea/utils/sp_fn.py`

#### 4. Additional Fixes

- Added `self.bert` attribute storage in `DependencyParser.__init__` for training compatibility
- Fixed `n_rels` parameter in `_init_model_with_state_dict` (was incorrectly using `n_feats`)
- Fixed device import in `DependencyParserTrainer`
- Added first-epoch model saving to ensure training always produces a model

### Testing

CI test added to verify training works with PyTorch v2:

```bash
# Run training test
tox -e train-dep

# Or directly
python -m unittest tests.pipeline.dependency_parse.test_train
```

## Dependency Relations

Common Vietnamese dependency relations:

| Relation | Description | Example |
|----------|-------------|---------|
| root | Root of sentence | Main verb |
| nsubj | Nominal subject | Tôi (I) → ăn (eat) |
| obj | Direct object | cơm (rice) ← ăn (eat) |
| cop | Copula | là (is) → noun |
| compound | Compound | Việt Nam ← sinh viên |
| nmod | Nominal modifier | của (of) relations |
| amod | Adjectival modifier | đẹp (beautiful) → noun |
| advmod | Adverbial modifier | rất (very) → adj |
| punct | Punctuation | . , ! ? |

## References

1. Dozat, T., & Manning, C. D. (2017). Deep Biaffine Attention for Neural Dependency Parsing. ICLR 2017.
2. VLSP 2020 Shared Task: Vietnamese Dependency Parsing
3. [Underthesea GitHub Repository](https://github.com/undertheseanlp/underthesea)

## Changelog

### Version 9.1.3 (PR #871)
- Added PyTorch v2.0+ support
- Fixed deprecated API usage
- Added training CI test (train-dep)
- Re-enabled dependency_parse tests in CI
